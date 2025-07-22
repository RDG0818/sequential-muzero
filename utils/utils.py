# utils.py
import jax
import jax.numpy as jnp
import chex
from typing import NamedTuple
from functools import partial

@partial(jax.jit, static_argnames=['n_steps'])
def n_step_returns_fn(
    rewards: chex.Array, 
    mcts_values: chex.Array, 
    n_steps: int, 
    discount_gamma: float
) -> chex.Array:
    """
    Computes N-step returns with a shrinking horizon for the end of the sequence.

    Args:
        rewards: A sequence of rewards. Shape: (T,).
        mcts_values: The sequence of MCTS values. Shape: (T+1,). (V_0 to V_T)
    
    Returns:
        A sequence of targets for states s_0 to s_{T-1}. Shape: (T,).
    """
    T = rewards.shape[0]
    targets = jnp.zeros_like(rewards)

    def loop_body(t, current_targets):
        horizon = jnp.minimum(n_steps, T - t)

        bootstrap_val = mcts_values[t + horizon]
        
        g = bootstrap_val
        def inner_loop(k, inner_g):
            reward_idx = t + k
            return rewards[reward_idx] + discount_gamma * inner_g
        
        g = jax.lax.fori_loop(0, horizon, lambda k, val: inner_loop(horizon - 1 - k, val), g)

        return current_targets.at[t].set(g)

    targets = jnp.concatenate([jax.lax.fori_loop(0, T, loop_body, targets), mcts_values[-1:]], axis=-1) # add final 0-step target
    return targets

class DiscreteSupport(NamedTuple):
    """A class to represent the discrete support for categorical distributions."""
    min: int
    max: int

    @property
    def size(self) -> int:
        return self.max - self.min + 1

def _h(x: jnp.ndarray, epsilon: float = 1e-3) -> jnp.ndarray:
    """
    MuZero scaling function to reduce the scale of rewards and values.
    Reference: Appendix A in https://arxiv.org/pdf/1805.11593.pdf
    """
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1 + epsilon * x)

def _h_inv(x: jnp.ndarray, epsilon: float = 1e-3) -> jnp.ndarray:
    """
    Inverse of the MuZero scaling function.
    """
    sign = jnp.sign(x)
    sqrt_term = jnp.sqrt(1 + 4 * epsilon * (jnp.abs(x) + 1 + epsilon))
    numerator = sqrt_term - 1
    denominator = 2 * epsilon
    squared_term = (numerator / denominator) ** 2
    return sign * (squared_term - 1)

def scalar_to_support(scalar: jnp.ndarray, support: DiscreteSupport) -> jnp.ndarray:
    """
    A JIT-compatible function to transform a scalar value into a categorical distribution.
    """
    scaled_scalar = _h(scalar)
    clipped_scalar = jnp.clip(scaled_scalar, support.min, support.max)

    floor = jnp.floor(clipped_scalar).astype(jnp.int32)
    ceil = jnp.ceil(clipped_scalar).astype(jnp.int32)
    prob = clipped_scalar - floor

    floor_indices = (floor - support.min).astype(jnp.int32)
    ceil_indices = (ceil - support.min).astype(jnp.int32)
    
    floor_one_hot = jax.nn.one_hot(floor_indices, num_classes=support.size)
    ceil_one_hot = jax.nn.one_hot(ceil_indices, num_classes=support.size)

    distribution = floor_one_hot * (1 - prob)[..., None] + ceil_one_hot * prob[..., None]
    
    return distribution


def support_to_scalar(distribution: jnp.ndarray, support: DiscreteSupport, use_logits: bool = True) -> jnp.ndarray:
    """
    Transforms a categorical distribution back to a scalar value.
    This is the equivalent of the `_inv_phi` and inverse transform functions.

    Args:
        distribution: A probability distribution (or logits) over the support.
        support: A DiscreteSupport object defining the range of the distribution.
        use_logits: If True, a softmax will be applied to the distribution first.

    Returns:
        The expected scalar value(s).
    """
    if use_logits:
        distribution = jax.nn.softmax(distribution, axis=-1)
    
    support_range = jnp.arange(support.min, support.max + 1, dtype=jnp.float32)

    scalar = jnp.sum(distribution * jnp.broadcast_to(support_range, distribution.shape), axis=-1)
    
    return _h_inv(scalar)


def categorical_cross_entropy_loss(prediction_logits: jnp.ndarray, target_distribution: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the cross-entropy loss between predicted logits and a target distribution.
    This is the standard loss function for the value and reward heads when using categorical representation.
    Equivalent to `optax.softmax_cross_entropy`.
    
    Args:
        prediction_logits: The raw output from the model's prediction head.
        target_distribution: The target probability distribution created by `scalar_to_support`.

    Returns:
        The loss for each item in the batch.
    """
    return -(jax.nn.log_softmax(prediction_logits, axis=-1) * target_distribution).sum(-1)

