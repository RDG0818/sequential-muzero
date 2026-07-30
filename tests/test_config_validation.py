"""Tests for train/muzero.py's _build_config validation.

Run with:
    conda run -n mazero pytest tests/test_config_validation.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from omegaconf import OmegaConf

from train.muzero import _build_config

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_cfg(overrides=None):
    """Builds a DictConfig shaped like the real Hydra-composed config by
    loading the actual default YAMLs (model/mcts/train), then merging in
    test-specific overrides."""
    model = OmegaConf.load(os.path.join(_ROOT, "configs/model/default.yaml"))
    mcts = OmegaConf.load(os.path.join(_ROOT, "configs/mcts/default.yaml"))
    train = OmegaConf.load(os.path.join(_ROOT, "configs/train/default.yaml"))
    cfg = OmegaConf.create({"model": model, "mcts": mcts, "train": train})
    if overrides:
        cfg = OmegaConf.merge(cfg, overrides)
    return cfg


def test_build_config_accepts_defaults():
    """The shipped default config (unimix_ratio=0.0, value_transform=hyperbolic)
    must not raise."""
    cfg = _load_cfg()
    config = _build_config(cfg)
    assert config.model.unimix_ratio == 0.0
    assert config.model.value_transform == "hyperbolic"


def test_build_config_rejects_unimix_ratio_above_one():
    """unimix_ratio=1.5 is a plausible typo for 0.01 and must fail fast in
    the driver process, not silently produce NaN loss deep into training."""
    cfg = _load_cfg({"model": {"unimix_ratio": 1.5}})
    with pytest.raises(ValueError, match="unimix_ratio"):
        _build_config(cfg)


def test_build_config_rejects_negative_unimix_ratio():
    cfg = _load_cfg({"model": {"unimix_ratio": -0.5}})
    with pytest.raises(ValueError, match="unimix_ratio"):
        _build_config(cfg)


def test_build_config_accepts_unimix_ratio_boundary_zero():
    cfg = _load_cfg({"model": {"unimix_ratio": 0.0}})
    _build_config(cfg)  # must not raise


def test_build_config_rejects_unimix_ratio_equal_one():
    """Upper bound is exclusive: at ratio=1.0 the network prediction term is
    fully zeroed out, degenerating unimix_cross_entropy into a constant
    uniform-distribution loss."""
    cfg = _load_cfg({"model": {"unimix_ratio": 1.0}})
    with pytest.raises(ValueError, match="unimix_ratio"):
        _build_config(cfg)


def test_build_config_rejects_unknown_value_transform():
    """A typo'd value_transform must raise in the driver process (fast, cheap
    failure) rather than inside a Ray actor's __init__ (actor-death traceback)."""
    cfg = _load_cfg({"model": {"value_transform": "not_a_real_transform"}})
    with pytest.raises(ValueError, match="value_transform"):
        _build_config(cfg)


def test_build_config_accepts_symlog_value_transform():
    cfg = _load_cfg({"model": {"value_transform": "symlog"}})
    config = _build_config(cfg)
    assert config.model.value_transform == "symlog"
