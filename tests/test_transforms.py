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
