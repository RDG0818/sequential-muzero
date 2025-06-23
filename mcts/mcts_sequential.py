import jax
import jax.numpy as jnp
import mctx
from typing import NamedTuple
from model.model import FlaxMAMuZeroNet
import utils
from utils import DiscreteSupport
import functools

class MCTSPlanOutput(NamedTuple):
    """Output of MCTSPlanner.plan()."""
    joint_action:   jnp.ndarray  # shape (N,), chosen action per agent
    policy_targets: jnp.ndarray  # shape (N, A), visit-count targets per agent
    root_value:     float        # scalar root value estimate

class MCTSSequentialPlanner:
    def __init__(
        self,
        model: FlaxMAMuZeroNet,
        num_simulations: int,
        max_depth_gumbel_search: int,
        num_gumbel_samples: int,
        mode: str = 'independent',
        num_joint_samples: int = 16
    ):
        # Validate planner mode
        if mode not in ['independent', 'sequential', 'joint']:
            raise ValueError(f"Invalid planner mode: {mode}")

        self.model = model
        self.num_agents = model.num_agents        # N agents
        self.action_space_size = model.action_space_size  # A actions
        self.num_simulations = num_simulations   # MCTS rollout count
        self.max_depth_gumbel_search = max_depth_gumbel_search
        self.num_gumbel_samples = num_gumbel_samples
        self.mode = mode
        self.value_support = DiscreteSupport(min=-model.value_support_size, max=model.value_support_size)
        self.reward_support = DiscreteSupport(min=-model.reward_support_size, max=model.reward_support_size)

        def recurrent_fn(params, rng_key, action, embedding):
            """
            Batched one-step rollout for a single agent in MuZero search.
            """
            latent, idx, factor = embedding

            current_joint_action = factor.at[idx].set(action.squeeze(axis=-1))

            # Dynamics + prediction
            next_latent, reward_logits, multi_logits, value_logits = self.model.apply(
                {'params': params}, latent, current_joint_action,
                method=self.model.recurrent_inference
            )
            value= utils.support_to_scalar(value_logits, self.value_support)
            reward = utils.support_to_scalar(reward_logits, self.reward_support)

            prior = multi_logits[:, idx, :]

            new_embed = (next_latent, idx, current_joint_action)
            out = mctx.RecurrentFnOutput(
                reward=reward,
                discount= jnp.full_like(reward, 0.99),
                prior_logits= prior,
                value=value
            )
            return out, new_embed

        self._recurrent_fn = jax.jit(recurrent_fn)

        self.plan_jit = jax.jit(self._plan_loop, static_argnums=())

    def plan(self, params, rng_key, observation):
        return self.plan_jit(params, rng_key, observation)

    def _plan_loop(self, params, rng_key, observation):
        root_latent, _, root_logits, root_value_logits = self.model.apply(
            {'params': params}, observation
        )  

        root_value = utils.support_to_scalar(root_value_logits, self.value_support)

        # Prepare per-agent inputs for scan
        keys = jax.random.split(rng_key, self.num_agents)  # (N,2)
        idxs = jnp.arange(self.num_agents, dtype=jnp.int32)     # (N,)

        initial_joint_action = jnp.argmax(root_logits, axis=-1)

        carry = (root_logits, root_value.reshape(-1), root_latent, initial_joint_action)

        def agent_step(carry, inputs):
            logits_b, value_b, latent_b, joint_action_so_far = carry
            key, agent = inputs
            # Extract agent's (1,A) prior
            p_slice = jax.lax.dynamic_slice(
                logits_b,
                start_indices=(0, agent, 0),
                slice_sizes=(1, 1, self.action_space_size)
            )  # shape (1,1,A)
            p = p_slice.squeeze(1)
            v = value_b                              
            emb = (latent_b, jnp.array([agent], jnp.int32), joint_action_so_far)

            out = mctx.gumbel_muzero_policy(
                params=params, rng_key=key, root=mctx.RootFnOutput(prior_logits=p, value=v, embedding=emb),
                recurrent_fn=self._recurrent_fn, num_simulations=self.num_simulations, 
                max_depth=self.max_depth_gumbel_search, max_num_considered_actions=self.num_gumbel_samples, 
                qtransform=functools.partial(mctx.qtransform_completed_by_mix_value, use_mixed_value=True)
            )
            
            updated_joint_action = joint_action_so_far.at[agent].set(out.action.squeeze())
            next_carry = (logits_b, value_b, latent_b, updated_joint_action)

            return next_carry, (out.action.squeeze(0), out.action_weights.squeeze(0))

        _, results = jax.lax.scan(agent_step, carry, (keys, idxs)) # Pretty much a for loop with a carry value
        actions, weights = results  

        return MCTSPlanOutput(
            joint_action=   actions,
            policy_targets= weights,
            root_value=     root_value.squeeze().astype(float)
        )
