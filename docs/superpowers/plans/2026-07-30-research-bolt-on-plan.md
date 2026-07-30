# Research Bolt-On (symlog transform + unimix smoothing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two DreamerV3-derived, config-flagged, off-by-default options to the value/reward categorical heads — a symlog/symexp transform as an alternative to the existing hyperbolic (MuZero-paper) scale, and unimix categorical smoothing on the value/reward cross-entropy losses — so they can be A/B'd against the current baseline in the upcoming training run.

**Architecture:** `DiscreteSupport` (a `NamedTuple` in `utils/transforms.py`) gains two `Callable` fields, `scale_fn`/`inv_scale_fn`, defaulting to the existing `muzero_scale`/`muzero_scale_inv` so all current call sites are unaffected unless a config explicitly opts into `value_transform: symlog`. `scalar_to_support`/`support_to_scalar` read the transform off the `support` object instead of hardcoding `muzero_scale`. Separately, `actors/loss.py` gains a `unimix_cross_entropy` helper that is byte-for-byte `optax.softmax_cross_entropy` at `unimix_ratio=0.0` and only diverges when a config explicitly sets a nonzero ratio. Both knobs live on `ModelConfig`, default to the current behavior, and require zero changes at any of this file's existing call sites beyond where the two `DiscreteSupport` instances are constructed.

**Tech Stack:** JAX/Flax, `optax`, `pytest`. No new dependencies.

## Global Constraints

- Every existing test must stay green — this is a refactor-plus-additive-feature change, not a rewrite. `conda run -n mazero pytest tests/ -v` is the standing regression gate.
- Default behavior (config omits the two new fields, or sets `value_transform: hyperbolic`, `unimix_ratio: 0.0`) must be numerically identical to pre-change behavior — verified explicitly, not assumed.
- No GPU is available in this working session — every test in this plan must pass on CPU (`JAX_PLATFORMS=cpu` or no GPU present), matching how `tests/` already runs.
- Cite DreamerV3 (Hafner et al., *Nature* 2025) in docstrings/comments for the new transform and smoothing functions, per this repo's existing "WHY-only, cite the paper" comment convention (see `CLAUDE.md`).
- `DiscreteSupport` is only ever constructed with keyword arguments across the whole repo (verified: `actors/learner_actor.py:68,72`, `mcts/mcts_joint_osla.py:510,514`) — adding new fields with defaults at the end is safe everywhere, no positional-argument breakage possible.
- `value_support`/`reward_support` are always closure-captured or `functools.partial`-bound before crossing a `jax.jit`/`jax.vmap` boundary (verified: `actors/loss.py:325` `jax.jit(train_step)` closes over them; `mcts/mcts_joint_osla.py:553` `jax.vmap(plan_single, ...)` receives them via `functools.partial`, not as a vmapped positional arg) — so `DiscreteSupport` holding plain Python `Callable`s never gets flattened as a JAX pytree leaf. Do not pass a `DiscreteSupport` instance directly as a `jax.jit`/`jax.vmap`-traced argument; if that ever changes, this assumption needs re-verification.

---

### Task 1: `symlog`/`symexp` transform + `get_value_transform_fns` selector

**Files:**
- Modify: `utils/transforms.py`
- Test: `tests/test_transforms.py` (new file — `utils/transforms.py` currently has no dedicated test file)

**Interfaces:**
- Produces: `symlog(x: jnp.ndarray) -> jnp.ndarray`, `symexp(x: jnp.ndarray) -> jnp.ndarray`, `get_value_transform_fns(name: str) -> Tuple[Callable, Callable]` (raises `ValueError` on unknown `name`; valid names: `"hyperbolic"`, `"symlog"`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_transforms.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n mazero pytest tests/test_transforms.py -v`
Expected: `ImportError`/collection failure — `symlog`, `symexp`, `get_value_transform_fns` don't exist in `utils/transforms.py` yet.

- [ ] **Step 3: Implement `symlog`/`symexp`/`get_value_transform_fns`**

In `utils/transforms.py`, add after `muzero_scale_inv` (before `scalar_to_support`):

```python
def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """
    DreamerV3 value/reward transform — alternative to muzero_scale.

    symlog(x) = sign(x) * ln(|x| + 1). Same role as muzero_scale (compress
    large magnitudes before feeding a categorical support), different curve.
    Reference: Hafner et al., "Mastering Diverse Domains through World
    Models," Nature 2025 (arXiv:2301.04104).
    """
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse of symlog."""
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


_VALUE_TRANSFORMS = {
    "hyperbolic": (muzero_scale, muzero_scale_inv),
    "symlog": (symlog, symexp),
}


def get_value_transform_fns(name: str):
    """
    Selects the (scale_fn, inv_scale_fn) pair for a DiscreteSupport by name.

    "hyperbolic" is the original MuZero paper's transform (Pohlen et al.
    2018, the repo default). "symlog" is DreamerV3's transform (see symlog
    docstring) — same role, different curve, opt-in via
    ModelConfig.value_transform.
    """
    try:
        return _VALUE_TRANSFORMS[name]
    except KeyError:
        raise ValueError(
            f"Unknown value_transform {name!r}; expected one of "
            f"{list(_VALUE_TRANSFORMS)}"
        ) from None
```

Also add `from typing import Callable, Tuple` to the top imports (alongside the existing `from typing import NamedTuple`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n mazero pytest tests/test_transforms.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/transforms.py tests/test_transforms.py
git commit -m "feat: add symlog/symexp transform and get_value_transform_fns selector

DreamerV3-derived alternative to the existing hyperbolic (MuZero paper)
value/reward scale. Not wired into DiscreteSupport yet — that's next."
```

---

### Task 2: Wire `scale_fn`/`inv_scale_fn` into `DiscreteSupport`

**Files:**
- Modify: `utils/transforms.py`
- Test: `tests/test_transforms.py`

**Interfaces:**
- Consumes: `symlog`, `symexp`, `muzero_scale`, `muzero_scale_inv` from Task 1.
- Produces: `DiscreteSupport(min: int, max: int, scale_fn: Callable = muzero_scale, inv_scale_fn: Callable = muzero_scale_inv)`. `scalar_to_support`/`support_to_scalar` signatures unchanged — they now read `support.scale_fn`/`support.inv_scale_fn` instead of the module-level `muzero_scale`/`muzero_scale_inv` directly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transforms.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n mazero pytest tests/test_transforms.py -v`
Expected: `test_discrete_support_defaults_to_hyperbolic_scale` and the symlog round-trip test FAIL (`DiscreteSupport` has no `scale_fn`/`inv_scale_fn` fields yet — `TypeError: unexpected keyword argument`). The byte-for-byte regression test passes trivially right now (it's checking current behavior before the refactor) — that's expected; it must still pass after Step 3.

- [ ] **Step 3: Implement the wiring**

In `utils/transforms.py`, change the `DiscreteSupport` class (move it below `symexp` so the defaults can reference `muzero_scale`/`muzero_scale_inv`, which are already defined above it):

```python
class DiscreteSupport(NamedTuple):
    """Discrete support for categorical value/reward distributions."""
    min: int
    max: int
    scale_fn: Callable = muzero_scale
    inv_scale_fn: Callable = muzero_scale_inv

    @property
    def size(self) -> int:
        return self.max - self.min + 1
```

Remove the old `DiscreteSupport` definition from the top of the file (it currently sits above `muzero_scale` at lines 7-14 — delete it from there since it's been moved).

Update `scalar_to_support` to use `support.scale_fn` instead of `muzero_scale`:

```python
def scalar_to_support(scalar: jnp.ndarray, support: DiscreteSupport) -> jnp.ndarray:
    """
    Encodes a scalar value into a two-hot categorical distribution over the support.

    Applies support.scale_fn first, then distributes probability mass between
    the two nearest support atoms via linear interpolation.

    Args:
        scalar: Scalar values to encode. Any shape.
        support: DiscreteSupport defining the range and scale transform.

    Returns:
        Categorical distribution. Shape: (*scalar.shape, support.size)
    """
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
```

Update `support_to_scalar` to use `support.inv_scale_fn` instead of `muzero_scale_inv`:

```python
def support_to_scalar(distribution: jnp.ndarray, support: DiscreteSupport) -> jnp.ndarray:
    """
    Decodes a categorical distribution (or logits) back to a scalar value.

    Applies softmax to convert logits to probabilities, computes the expected
    value over the support atoms, then inverts via support.inv_scale_fn.

    Args:
        distribution: Logits or probabilities over the support.
                      Shape: (*batch_shape, support.size)
        support: DiscreteSupport defining the range and scale transform.

    Returns:
        Scalar values. Shape: (*batch_shape,)
    """
    probs = jax.nn.softmax(distribution, axis=-1)
    support_range = jnp.arange(support.min, support.max + 1, dtype=jnp.float32)
    scalar = jnp.sum(probs * jnp.broadcast_to(support_range, probs.shape), axis=-1)
    return support.inv_scale_fn(scalar)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n mazero pytest tests/test_transforms.py -v`
Expected: all 9 tests PASS.

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `conda run -n mazero pytest tests/ -v`
Expected: all tests PASS (no regressions from reordering `DiscreteSupport` or changing `scalar_to_support`/`support_to_scalar` internals — every existing caller still constructs `DiscreteSupport(min=..., max=...)` and gets the same default `muzero_scale`/`muzero_scale_inv` behavior).

- [ ] **Step 6: Commit**

```bash
git add utils/transforms.py tests/test_transforms.py
git commit -m "refactor: make DiscreteSupport carry its own scale/inv_scale fns

scalar_to_support/support_to_scalar now read the transform off the support
object instead of hardcoding muzero_scale, so a support can opt into
symlog/symexp. Default behavior unchanged and regression-tested byte-for-byte
against the pre-refactor hardcoded path."
```

---

### Task 3: `ModelConfig.value_transform` config flag, wired end-to-end

**Files:**
- Modify: `config.py`
- Modify: `mcts/mcts_joint_osla.py:27,510-517`
- Modify: `actors/learner_actor.py:37,68-75`
- Modify: `configs/model/default.yaml`
- Test: `tests/test_mcts.py`

**Interfaces:**
- Consumes: `get_value_transform_fns` from Task 1, `DiscreteSupport(scale_fn=, inv_scale_fn=)` from Task 2.
- Produces: `ModelConfig.value_transform: str = "hyperbolic"`. `MCTSJointOSLAPlanner.value_support`/`.reward_support` and `LearnerActor`'s local `value_support`/`reward_support` now carry the config-selected transform.

- [ ] **Step 1: Write the failing test**

In `tests/test_mcts.py`, add near the other `MCTSJointOSLAPlanner` tests (after the `model_and_params` fixture, before `class TestMCTSJointOSLAPlanner:`):

```python
def test_planner_uses_configured_value_transform(model_and_params, test_config):
    import dataclasses
    from mcts.mcts_joint_osla import MCTSJointOSLAPlanner
    from utils.transforms import symlog, symexp, muzero_scale, muzero_scale_inv

    net, _ = model_and_params

    default_planner = MCTSJointOSLAPlanner(model=net, config=test_config)
    assert default_planner.value_support.scale_fn is muzero_scale
    assert default_planner.reward_support.inv_scale_fn is muzero_scale_inv

    symlog_config = dataclasses.replace(
        test_config, model=dataclasses.replace(test_config.model, value_transform="symlog")
    )
    symlog_planner = MCTSJointOSLAPlanner(model=net, config=symlog_config)
    assert symlog_planner.value_support.scale_fn is symlog
    assert symlog_planner.reward_support.inv_scale_fn is symexp
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n mazero pytest tests/test_mcts.py::test_planner_uses_configured_value_transform -v`
Expected: FAIL — `ModelConfig.__init__() got an unexpected keyword argument 'value_transform'` (raised by `dataclasses.replace`).

- [ ] **Step 3: Add the config field**

In `config.py`, add to `ModelConfig` (after `pred_out: int`, the last existing field):

```python
    pred_out: int
    value_transform: str = "hyperbolic"  # "hyperbolic" (MuZero paper) or "symlog" (DreamerV3)
```

- [ ] **Step 4: Wire `MCTSJointOSLAPlanner`**

In `mcts/mcts_joint_osla.py`, change the import at line 27:

```python
from utils.transforms import DiscreteSupport, get_value_transform_fns
```

And change the `__init__` body at lines 510-517:

```python
        scale_fn, inv_scale_fn = get_value_transform_fns(config.model.value_transform)
        self.value_support = DiscreteSupport(
            min=-config.model.value_support_size,
            max=config.model.value_support_size,
            scale_fn=scale_fn,
            inv_scale_fn=inv_scale_fn,
        )
        self.reward_support = DiscreteSupport(
            min=-config.model.reward_support_size,
            max=config.model.reward_support_size,
            scale_fn=scale_fn,
            inv_scale_fn=inv_scale_fn,
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `conda run -n mazero pytest tests/test_mcts.py::test_planner_uses_configured_value_transform -v`
Expected: PASS.

- [ ] **Step 6: Mirror the same wiring in `LearnerActor`**

`LearnerActor` is a `@ray.remote` class with no existing unit test coverage for its `__init__` anywhere in this repo (only the pure functions it calls, in `actors/loss.py`, are unit-tested) — this step mirrors Step 4's already-tested pattern exactly, so no new test is added here; verify by inspection that it matches Step 4.

In `actors/learner_actor.py`, change the import at line 37:

```python
        from utils.transforms import DiscreteSupport, get_value_transform_fns
```

And change lines 68-75:

```python
        scale_fn, inv_scale_fn = get_value_transform_fns(config.model.value_transform)
        value_support = DiscreteSupport(
            min=-config.model.value_support_size,
            max=config.model.value_support_size,
            scale_fn=scale_fn,
            inv_scale_fn=inv_scale_fn,
        )
        reward_support = DiscreteSupport(
            min=-config.model.reward_support_size,
            max=config.model.reward_support_size,
            scale_fn=scale_fn,
            inv_scale_fn=inv_scale_fn,
        )
```

- [ ] **Step 7: Add the config knob to the YAML with a citing comment**

In `configs/model/default.yaml`, append:

```yaml
# "hyperbolic" (MuZero paper, Pohlen et al. 2018) or "symlog" (DreamerV3,
# Hafner et al., Nature 2025) — see utils/transforms.py. Same role, different
# curve; symlog is opt-in, not the validated default.
value_transform: hyperbolic
```

- [ ] **Step 8: Run the full suite**

Run: `conda run -n mazero pytest tests/ -v`
Expected: all tests PASS.

- [ ] **Step 9: Commit**

```bash
git add config.py mcts/mcts_joint_osla.py actors/learner_actor.py configs/model/default.yaml tests/test_mcts.py
git commit -m "feat: wire ModelConfig.value_transform into planner and learner

Both DiscreteSupport construction sites now select scale_fn/inv_scale_fn via
get_value_transform_fns(config.model.value_transform). Default 'hyperbolic'
preserves current behavior exactly; 'symlog' is opt-in for the upcoming
training-run ablation."
```

---

### Task 4: `ModelConfig.unimix_ratio` + `unimix_cross_entropy` in the value/reward losses

**Files:**
- Modify: `config.py`
- Modify: `actors/loss.py`
- Modify: `configs/model/default.yaml`
- Test: `tests/test_learner.py`

**Interfaces:**
- Produces: `ModelConfig.unimix_ratio: float = 0.0`. `unimix_cross_entropy(logits, target_probs, unimix_ratio: float) -> Array` in `actors/loss.py`, same call signature shape as `optax.softmax_cross_entropy(logits, target_probs)` plus the ratio.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_learner.py`:

```python
def test_unimix_cross_entropy_zero_ratio_matches_optax_exactly():
    """unimix_ratio=0.0 must be byte-for-byte optax.softmax_cross_entropy —
    this is the 'disabled' default and must carry zero behavior change."""
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    import optax
    from actors.loss import unimix_cross_entropy

    rng = jax.random.PRNGKey(0)
    logits = jax.random.normal(rng, (4, 10))
    target = jax.nn.one_hot(jnp.array([2, 5, 0, 9]), 10)

    result = unimix_cross_entropy(logits, target, 0.0)
    expected = optax.softmax_cross_entropy(logits, target)

    assert jnp.allclose(result, expected)


def test_unimix_cross_entropy_smooths_overconfident_predictions():
    """With a nonzero ratio, a near-one-hot prediction that exactly matches
    the target should incur *higher* loss than at ratio=0 — the uniform
    floor prevents the loss from ever reaching zero, which is the point
    (DreamerV3's stability argument)."""
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    from actors.loss import unimix_cross_entropy

    confident_logits = jnp.array([[20.0, -20.0, -20.0]])  # ~one-hot at class 0
    target = jax.nn.one_hot(jnp.array([0]), 3)

    loss_disabled = unimix_cross_entropy(confident_logits, target, 0.0)
    loss_smoothed = unimix_cross_entropy(confident_logits, target, 0.01)

    assert float(loss_smoothed[0]) > float(loss_disabled[0])


def test_unimix_cross_entropy_matches_manual_smoothing_formula():
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    from actors.loss import unimix_cross_entropy

    logits = jnp.array([[1.0, 2.0, 3.0]])
    target = jnp.array([[0.0, 0.0, 1.0]])
    ratio = 0.1

    result = unimix_cross_entropy(logits, target, ratio)

    probs = jax.nn.softmax(logits, axis=-1)
    smoothed = (1.0 - ratio) * probs + ratio / 3
    expected = -jnp.sum(target * jnp.log(smoothed + 1e-8), axis=-1)

    assert jnp.allclose(result, expected, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n mazero pytest tests/test_learner.py -k unimix -v`
Expected: FAIL — `ImportError: cannot import name 'unimix_cross_entropy' from 'actors.loss'`.

- [ ] **Step 3: Implement `unimix_cross_entropy`**

In `actors/loss.py`, add after `_awpo_weight` (before `make_train_step`):

```python
def unimix_cross_entropy(logits: '_jax.Array', target_probs: '_jax.Array', unimix_ratio: float) -> '_jax.Array':
    """Cross-entropy against target_probs, with unimix_ratio uniform mass
    mixed into the predicted distribution before taking the log.

    unimix_ratio=0.0 is exactly optax.softmax_cross_entropy — the smoothing
    only engages when a config opts in. Reference: DreamerV3's "1% unimix
    for all categoricals" (Hafner et al., Nature 2025); applied here to the
    value/reward categorical heads only, not the policy head (this repo
    already applies Dirichlet noise to the policy at the MCTS root, which
    covers similar ground on exploration — unimix's distinct contribution
    here is stabilizing the value/reward heads, not policy exploration).
    """
    import optax
    if unimix_ratio == 0.0:
        return optax.softmax_cross_entropy(logits, target_probs)
    probs = _jax.nn.softmax(logits, axis=-1)
    num_classes = probs.shape[-1]
    smoothed = (1.0 - unimix_ratio) * probs + unimix_ratio / num_classes
    return -_jax.numpy.sum(target_probs * _jax.numpy.log(smoothed + 1e-8), axis=-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n mazero pytest tests/test_learner.py -k unimix -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Add the config field**

In `config.py`, add to `ModelConfig` (after `value_transform` from Task 3):

```python
    value_transform: str = "hyperbolic"  # "hyperbolic" (MuZero paper) or "symlog" (DreamerV3)
    unimix_ratio: float = 0.0  # DreamerV3 unimix smoothing on value/reward CE; 0.0 = disabled
```

- [ ] **Step 6: Wire `unimix_cross_entropy` into `make_train_step`**

In `actors/loss.py`, inside `make_train_step` (after `awpo_alpha = float(config.train.awpo_alpha)` at line 95), add:

```python
    unimix_ratio = float(config.model.unimix_ratio)  # 0.0 = disabled
```

Replace the `v0_loss` computation (currently `optax.softmax_cross_entropy(init_out.value_logits, value_target_dist[:, 0])`) with:

```python
            v0_loss = unimix_cross_entropy(
                init_out.value_logits, value_target_dist[:, 0], unimix_ratio
            )
```

In **both** `scan_step` definitions (the `awpo_alpha > 0.0` branch and the `else` branch — value/reward loss computation is identical in both), replace:

```python
                    ri_loss = optax.softmax_cross_entropy(out.reward_logits, ri_dist)
                    vi_loss = optax.softmax_cross_entropy(out.value_logits, vi_dist)
```

with:

```python
                    ri_loss = unimix_cross_entropy(out.reward_logits, ri_dist, unimix_ratio)
                    vi_loss = unimix_cross_entropy(out.value_logits, vi_dist, unimix_ratio)
```

(This touches the two `scan_step` closures separately — one in the `if awpo_alpha > 0.0:` block, one in the `else:` block. Both need the same two-line change. Policy loss lines — `ce_p0`, `pi_loss` in the non-AWPO branch — are untouched; unimix is scoped to value/reward only per the docstring above.)

- [ ] **Step 7: Add the config knob to the YAML**

In `configs/model/default.yaml`, append after the `value_transform` line added in Task 3:

```yaml
# DreamerV3 unimix smoothing on the value/reward categorical losses
# (Hafner et al., Nature 2025) — see actors/loss.py:unimix_cross_entropy.
# 0.0 = disabled (default); try ~0.01 for the ablation run.
unimix_ratio: 0.0
```

- [ ] **Step 8: Run the full suite**

Run: `conda run -n mazero pytest tests/ -v`
Expected: all tests PASS — with `unimix_ratio` defaulting to `0.0` everywhere, `make_train_step`'s value/reward losses take the exact `optax.softmax_cross_entropy` code path (Step 3's `if unimix_ratio == 0.0` branch), so no existing test's numerics can move.

- [ ] **Step 9: Commit**

```bash
git add config.py actors/loss.py configs/model/default.yaml tests/test_learner.py
git commit -m "feat: add unimix_ratio config flag, wire into value/reward CE losses

DreamerV3-derived categorical smoothing (Hafner et al., Nature 2025),
scoped to the value/reward heads (policy exploration already covered by
existing root Dirichlet noise). Default 0.0 = disabled, byte-for-byte
optax.softmax_cross_entropy at that setting."
```

---

### Task 5: Full regression pass + documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run the full test suite one more time from a clean state**

Run: `conda run -n mazero pytest tests/ -v`
Expected: all tests PASS, no failures or errors, no warnings about deprecated APIs newly introduced by this plan.

- [ ] **Step 2: Document the two new config fields**

In `CLAUDE.md`, find the "Notable config fields added since original docs" section and add two lines after the existing `TrainConfig.awpo_alpha` line:

```markdown
- `ModelConfig.value_transform: str` — "hyperbolic" (MuZero paper default) or "symlog" (DreamerV3, Hafner et al. Nature 2025); see `utils/transforms.py`
- `ModelConfig.unimix_ratio: float` — DreamerV3 unimix smoothing on value/reward categorical CE; 0.0 = disabled (default)
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document value_transform and unimix_ratio config fields"
```
