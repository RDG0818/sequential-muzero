# utils.py
import jax
import jax.numpy as jnp
from typing import NamedTuple

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
    Transforms a scalar value into a categorical representation (a probability distribution).
    This is the equivalent of the `_phi` function.

    Args:
        scalar: The scalar value(s) to transform. Can be a single value or a batch.
        support: A DiscreteSupport object defining the range of the distribution.

    Returns:
        A probability distribution over the support.
    """
    # 1. Scale the scalar value using the h-transform.
    scaled_scalar = _h(scalar)

    # 2. Clip the scaled value to be within the support range.
    clipped_scalar = jnp.clip(scaled_scalar, support.min, support.max)
    
    # 3. Find the integer buckets it falls between.
    floor = jnp.floor(clipped_scalar).astype(jnp.int32)
    ceil = jnp.ceil(clipped_scalar).astype(jnp.int32)

    # 4. Calculate the probability weight for the 'ceil' bucket.
    # The weight for the 'floor' bucket is (1 - prob).
    prob = clipped_scalar - floor

    # 5. Create the one-hot like distribution.
    # The output shape will be (*scalar.shape, support.size).
    output_shape = (*scalar.shape, support.size)
    output = jnp.zeros(output_shape)

    # Distribute the probability mass to the floor and ceil buckets.
    floor_indices = (floor - support.min).astype(jnp.int32)
    ceil_indices = (ceil - support.min).astype(jnp.int32)

    # Using .at[...].add(...) is safer than .set(...) if floor == ceil
    output = output.at[..., floor_indices].add(1 - prob)
    output = output.at[..., ceil_indices].add(prob)

    return output


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
    
    # 1. Create the support range tensor.
    support_range = jnp.arange(support.min, support.max + 1, dtype=jnp.float32)
    
    # 2. Calculate the expected value by multiplying the distribution by the support range.
    # We broadcast support_range to match the distribution's shape.
    scalar = jnp.sum(distribution * jnp.broadcast_to(support_range, distribution.shape), axis=-1)
    
    # 3. Inverse scale the value using the h-inverse transform.
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

