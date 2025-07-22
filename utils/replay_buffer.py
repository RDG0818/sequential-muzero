# replay_buffer.py
import jax
import jax.numpy as jnp
import chex
from flax import struct
from typing import NamedTuple, Tuple

class Transition(NamedTuple):
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    policy_target: chex.Array
    value_target: chex.Array

class Sample(NamedTuple):
    obs: chex.Array             # (B, N, D_obs)
    actions: chex.Array         # (B, U, N)
    rewards: chex.Array         # (B, U)
    dones: chex.Array           # (B, U)
    policy_targets: chex.Array  # (B, U+1, N, A)
    value_targets: chex.Array   # (B, U+1)
    mask: chex.Array

@struct.dataclass
class ReplayBuffer:
    """A JAX-native uniform-sampling replay buffer."""
    data: Transition
    priorities: chex.Array
    position: int
    capacity: int = struct.field(pytree_node=False)
    alpha: float = struct.field(pytree_node=False)

    @classmethod
    def create(cls, capacity: int, alpha: float, sample_transition: Transition):
        """
        Creates an empty replay buffer.

        Args:
            capacity: The total number of individual transitions to store.
            sample_transition: A sample Transition object (including a batch dimension) 
                               to infer shapes and dtypes.
        """        
        storage_sample = jax.tree_util.tree_map(lambda x: x[0], sample_transition)
        
        buffer_data = jax.tree_util.tree_map(
            lambda x: jnp.zeros((capacity, *x.shape), dtype=x.dtype),
            storage_sample
        )
        priorities = jnp.zeros(capacity, dtype=jnp.float32)
        return cls(data=buffer_data, priorities=priorities, position=0, capacity=capacity, alpha=alpha)

    def add(self, transitions: Transition):
        """Adds a batch of transitions to the buffer."""
        batch_size = jax.tree_util.tree_leaves(transitions)[0].shape[0]
        
        indices = (self.position + jnp.arange(batch_size)) % self.capacity

        max_priority = jnp.max(self.priorities)
        initial_priority = jnp.full((batch_size,), jnp.where(max_priority > 0, max_priority, 1.0))

        def update_leaf(buffer_leaf, transition_leaf):
            return buffer_leaf.at[indices].set(transition_leaf)

        new_data = jax.tree_util.tree_map(update_leaf, self.data, transitions)
        new_priorities = self.priorities.at[indices].set(initial_priority)
        new_position = (self.position + batch_size) % self.capacity
        
        return self.replace(data=new_data, priorities=new_priorities, position=new_position)

    def sample(self, key: chex.PRNGKey, batch_size: int, unroll_steps: int, n_step: int, beta: float) -> Tuple[Sample, chex.Array, chex.Array]:
        """Samples a batch of unrolled sequences."""
        num_valid_transitions = jnp.sum(self.priorities > 0)
        safe_num_valid_transitions = jnp.maximum(num_valid_transitions, 1)
        powered_priorities = jnp.power(self.priorities, self.alpha)
        total_priority = jnp.sum(powered_priorities)
        safe_total_priority = jnp.maximum(total_priority, 1e-6)
        probs = powered_priorities / safe_total_priority

        start_indices = jax.random.choice(key, self.capacity, shape=(batch_size,), p=probs)

        sampled_probs = probs[start_indices]
        weights = jnp.power(safe_num_valid_transitions * sampled_probs, -beta)

        max_weight = jnp.max(weights)
        safe_max_weight = jnp.maximum(max_weight, 1e-6)
        weights = weights / safe_max_weight

        sequence_length = unroll_steps + n_step
        sequence_indices = jnp.arange(sequence_length)
        indices = (start_indices[:, None] + sequence_indices[None, :]) % self.capacity

        def gather_sequence(buffer_leaf):
            return jnp.take(buffer_leaf, indices, axis=0)
        
        sequence_data = jax.tree_util.tree_map(gather_sequence, self.data)

        unrolled_done = jax.lax.dynamic_slice_in_dim(sequence_data.done, 0, unroll_steps, axis=1)
        loss_mask = jnp.cumsum(jnp.cumsum(unrolled_done, axis=1), axis=1) <= 1

        sample = Sample(
            obs=self.data.obs[start_indices],
            actions=jax.lax.dynamic_slice_in_dim(sequence_data.action, 0, unroll_steps, axis=1),
            dones=unrolled_done,
            rewards=sequence_data.reward,
            policy_targets=sequence_data.policy_target,
            value_targets=sequence_data.value_target,
            mask=loss_mask
        )

        return sample, weights, start_indices

    def update_priorities(self, indices: chex.Array, priorities: chex.Array):
        """Updates the priorities for the given indices."""
        return self.replace(priorities=self.priorities.at[indices].set(priorities))
