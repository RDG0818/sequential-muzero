"""
Unit tests for utils/transforms.py.

Run with:
    conda run -n mazero pytest tests/test_transforms.py -v
"""

import pytest
import jax
import jax.numpy as jnp

from utils.transforms import (
    DiscreteSupport,
    muzero_scale,
    muzero_scale_inv,
    symlog,
    symexp,
    get_value_transform_fns,
    scalar_to_support,
    support_to_scalar,
)


def test_symlog_symexp_are_inverses():
    x = jnp.array([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
    assert jnp.allclose(symexp(symlog(x)), x, atol=1e-3)


def test_symlog_zero_is_zero():
    assert float(symlog(jnp.array(0.0))) == 0.0


def test_symlog_matches_closed_form():
    # symlog(e - 1) = sign(e-1) * ln((e-1) + 1) = ln(e) = 1
    x = jnp.array(jnp.e - 1)
    assert jnp.allclose(symlog(x), 1.0, atol=1e-5)


def test_get_value_transform_fns_hyperbolic():
    scale_fn, inv_scale_fn = get_value_transform_fns("hyperbolic")
    assert scale_fn is muzero_scale
    assert inv_scale_fn is muzero_scale_inv


def test_get_value_transform_fns_symlog():
    scale_fn, inv_scale_fn = get_value_transform_fns("symlog")
    assert scale_fn is symlog
    assert inv_scale_fn is symexp


def test_get_value_transform_fns_unknown_name_raises():
    with pytest.raises(ValueError):
        get_value_transform_fns("nonexistent")


def test_discrete_support_defaults_to_hyperbolic_scale():
    support = DiscreteSupport(min=-5, max=5)
    assert support.scale_fn is muzero_scale
    assert support.inv_scale_fn is muzero_scale_inv


def test_scalar_to_support_default_matches_pre_refactor_hyperbolic_path():
    """Regression guard: byte-for-byte match against the old hardcoded
    muzero_scale call path this refactor replaces."""
    support = DiscreteSupport(min=-5, max=5)
    x = jnp.array([0.3, -2.0, 4.9])

    dist = scalar_to_support(x, support)

    scaled = jnp.clip(muzero_scale(x), support.min, support.max)
    floor = jnp.floor(scaled).astype(jnp.int32)
    ceil = jnp.ceil(scaled).astype(jnp.int32)
    prob = scaled - floor
    floor_oh = jax.nn.one_hot((floor - support.min).astype(jnp.int32), support.size)
    ceil_oh = jax.nn.one_hot((ceil - support.min).astype(jnp.int32), support.size)
    expected = floor_oh * (1 - prob)[..., None] + ceil_oh * prob[..., None]

    assert jnp.allclose(dist, expected)


def test_scalar_to_support_round_trip_with_symlog_transform():
    support = DiscreteSupport(min=-5, max=5, scale_fn=symlog, inv_scale_fn=symexp)
    x = jnp.array([0.0, 1.5, -3.0, 4.0])

    dist = scalar_to_support(x, support)
    recovered = support_to_scalar(dist, support)

    assert jnp.allclose(recovered, x, atol=0.05)
