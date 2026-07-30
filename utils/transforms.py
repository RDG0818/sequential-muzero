# utils/transforms.py
import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple, Tuple


def muzero_scale(x: jnp.ndarray, epsilon: float = 1e-3) -> jnp.ndarray:
    """
    MuZero value scaling function — reduces the scale of large rewards/values
    to improve numerical stability during training.

    Reference: Appendix A, https://arxiv.org/pdf/1805.11593.pdf
    """
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1 + epsilon * x)


def muzero_scale_inv(x: jnp.ndarray, epsilon: float = 1e-3) -> jnp.ndarray:
    """Inverse of muzero_scale — converts scaled values back to original scale."""
    sign = jnp.sign(x)
    sqrt_term = jnp.sqrt(1 + 4 * epsilon * (jnp.abs(x) + 1 + epsilon))
    numerator = sqrt_term - 1
    denominator = 2 * epsilon
    squared_term = (numerator / denominator) ** 2
    return sign * (squared_term - 1)


def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """DreamerV3 value/reward transform — alternative to muzero_scale, same
    role, different curve. Hafner et al., Nature 2025 (arXiv:2301.04104)."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse of symlog."""
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


class DiscreteSupport(NamedTuple):
    """Discrete support for categorical value/reward distributions."""
    min: int
    max: int
    scale_fn: Callable = muzero_scale
    inv_scale_fn: Callable = muzero_scale_inv

    @property
    def size(self) -> int:
        return self.max - self.min + 1


_VALUE_TRANSFORMS = {
    "hyperbolic": (muzero_scale, muzero_scale_inv),
    "symlog": (symlog, symexp),
}


def get_value_transform_fns(name: str) -> Tuple[Callable, Callable]:
    """(scale_fn, inv_scale_fn) for a DiscreteSupport by name: "hyperbolic"
    (MuZero paper, Pohlen et al. 2018, default) or "symlog" (see symlog()).
    Opt-in via ModelConfig.value_transform."""
    try:
        return _VALUE_TRANSFORMS[name]
    except KeyError:
        raise ValueError(
            f"Unknown value_transform {name!r}; expected one of "
            f"{list(_VALUE_TRANSFORMS)}"
        ) from None


def scalar_to_support(scalar: jnp.ndarray, support: DiscreteSupport) -> jnp.ndarray:
    """Two-hot encodes a scalar over support: support.scale_fn, then linear
    interpolation between the two nearest atoms. Shape (*scalar.shape, support.size)."""
    scaled_scalar = support.scale_fn(scalar)
    clipped_scalar = jnp.clip(scaled_scalar, support.min, support.max)

    floor = jnp.floor(clipped_scalar).astype(jnp.int32)
    ceil = jnp.ceil(clipped_scalar).astype(jnp.int32)
    prob = clipped_scalar - floor

    floor_indices = (floor - support.min).astype(jnp.int32)
    ceil_indices = (ceil - support.min).astype(jnp.int32)

    floor_one_hot = jax.nn.one_hot(floor_indices, num_classes=support.size)
    ceil_one_hot = jax.nn.one_hot(ceil_indices, num_classes=support.size)

    return floor_one_hot * (1 - prob)[..., None] + ceil_one_hot * prob[..., None]


def support_to_scalar(distribution: jnp.ndarray, support: DiscreteSupport) -> jnp.ndarray:
    """Decodes logits (or probabilities) over support to a scalar: softmax,
    expected value over the support atoms, then support.inv_scale_fn."""
    probs = jax.nn.softmax(distribution, axis=-1)
    support_range = jnp.arange(support.min, support.max + 1, dtype=jnp.float32)
    scalar = jnp.sum(probs * jnp.broadcast_to(support_range, probs.shape), axis=-1)
    return support.inv_scale_fn(scalar)
