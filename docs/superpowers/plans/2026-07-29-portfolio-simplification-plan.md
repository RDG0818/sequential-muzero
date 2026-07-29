# Portfolio Simplification Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `sequential-muzero` read as a dead-simple, well-abstracted portfolio piece: no dead planner code, sparse WHY-only comments, and no oversized files — without changing training behavior beyond one explicit, documented default-planner switch.

**Architecture:** Three sequential passes, each its own test-green checkpoint on a new branch `simplify/portfolio-pass` (off `updates`): (1) delete the two non-default MCTS planners and everything downstream of them, fix a config-inheritance detail and a stale `eval.py` bug this uncovers, and clean up tracked `__pycache__`; (2) trim comments/docstrings repo-wide to WHY-only, citing paper equation numbers instead of re-deriving math in prose; (3) split the two largest files along already-tested seams.

**Tech Stack:** Python 3.10, JAX/Flax, Hydra/OmegaConf configs, pytest.

## Global Constraints

- New branch: `simplify/portfolio-pass`, created off `updates` at the start of Task 1.
- Full test suite (`pytest tests/ -v`, currently 141 tests) must pass at the end of every task in this plan. No task ends with red tests.
- No behavior changes beyond the single one flagged in Task 1 (default planner `independent` → `joint`). Everything else in this plan is comments, imports, dead-code removal, or pure file moves.
- Do not rewrite git history. Every task ends in its own commit.
- CLAUDE.md must stay accurate to the code after every task that changes something CLAUDE.md documents — update it in the same task/commit, not deferred.
- Spec: `docs/superpowers/specs/2026-07-29-portfolio-simplification-design.md`.

---

## Task 1: Remove `MCTSIndependentPlanner` and `MCTSJointPlanner` (dead planners, `mctx`, `communicate()`)

**Files:**
- Delete: `mcts/mcts_independent.py`, `mcts/mcts_joint.py`
- Modify: `mcts/__init__.py`, `requirements.txt`, `model/model.py`, `config.py`, `configs/config.yaml`, `configs/mcts/default.yaml`, `actors/data_actor.py:78,97-107`, `actors/reanalyze_actor.py:72,82-92`, `eval.py:43,87-88`, `tests/test_mcts.py`, `CLAUDE.md`
- Test: `tests/test_mcts.py` (existing file, edited in place)

**Interfaces:**
- Produces: `MCTSConfig` (in `config.py`) no longer has `independent_argmax` or `use_root_communication` fields. `mcts/__init__.py` exports only `MCTSPlanner`, `MCTSPlanOutput`, `MCTSJointOSLAPlanner`. `FlaxMAMuZeroNet` no longer has a `communicate()` method or `communication_net` attribute. Every `planner_map` dict repo-wide has exactly two keys: `"joint"` → `MCTSJointOSLAPlanner`.

Context before you start: `MCTSJointOSLAPlanner` (`mcts/mcts_joint_osla.py`) is the only planner used in the SMAX training target and just passed a full correctness audit. `MCTSIndependentPlanner` and `MCTSJointPlanner` (`joint_legacy`) are unused ablation-only code paths. Grep confirms the full blast radius is exactly the files listed above — nothing else in the repo imports these two classes, `mctx`, or `model.communicate`.

**Important discovery to preserve:** `configs/mcts/default.yaml` is NOT just an "independent planner preset" — `configs/mcts/joint.yaml` and `configs/mcts/smax.yaml` both start with `defaults: - default`, i.e. they inherit their base hyperparameters from `default.yaml` via Hydra's config-group inheritance. **Do not delete `configs/mcts/default.yaml`** — doing so breaks `joint.yaml`/`smax.yaml`. Instead, edit its contents so it becomes a joint-planner-mode base preset.

- [ ] **Step 1: Delete the two dead planner files**

```bash
git checkout -b simplify/portfolio-pass updates
git rm mcts/mcts_independent.py mcts/mcts_joint.py
```

- [ ] **Step 2: Fix `mcts/__init__.py`**

Current content:
```python
from mcts.base import MCTSPlanner, MCTSPlanOutput
from mcts.mcts_independent import MCTSIndependentPlanner
from mcts.mcts_joint import MCTSJointPlanner
from mcts.mcts_joint_osla import MCTSJointOSLAPlanner
```

New content:
```python
from mcts.base import MCTSPlanner, MCTSPlanOutput
from mcts.mcts_joint_osla import MCTSJointOSLAPlanner
```

- [ ] **Step 3: Remove `mctx` from `requirements.txt`**

Delete the line `mctx` (it was only imported by the two deleted files; `mcts_joint_osla.py` uses `mctx.RecurrentFnOutput` — check this before deleting).

Run: `grep -rn "import mctx\|from mctx" --include="*.py" .`
Expected: only `mcts/mcts_joint_osla.py` remains as a hit (it uses `mctx.RecurrentFnOutput` as a plain NamedTuple-like return type, not the search algorithm) — **so `mctx` must stay in `requirements.txt`.** Do not remove it. (This corrects the original design assumption — verify with the grep above before touching `requirements.txt`, and skip this file if the grep shows `mcts_joint_osla.py` still depends on it.)

- [ ] **Step 4: Remove `model.communicate()` and its dedicated attention module from `model/model.py`**

In `setup()` (around line 194-213), current content:
```python
    def setup(self):
        attention_module = None
        if self.config.attention_type == "transformer":
            attention_module = TransformerAttentionEncoder(
                num_layers=self.config.attention_layers,
                num_heads=self.config.attention_heads,
                hidden_size=self.config.hidden_state_size,
                dropout_rate=self.config.dropout_rate,
            )
            # Separate communication module for pre-search root attention.
            # Distinct params from dynamics attention — serves a different role
            # (agents sharing intent at the root vs. coordinating during transitions).
            self.communication_net = TransformerAttentionEncoder(
                num_layers=self.config.attention_layers,
                num_heads=self.config.attention_heads,
                hidden_size=self.config.hidden_state_size,
                dropout_rate=self.config.dropout_rate,
            )
        else:
            self.communication_net = None
        self.representation_net = RepresentationNetwork(
```

New content:
```python
    def setup(self):
        attention_module = None
        if self.config.attention_type == "transformer":
            attention_module = TransformerAttentionEncoder(
                num_layers=self.config.attention_layers,
                num_heads=self.config.attention_heads,
                hidden_size=self.config.hidden_state_size,
                dropout_rate=self.config.dropout_rate,
            )
        self.representation_net = RepresentationNetwork(
```

In `__call__`'s dummy-init block (around line 261-266), remove the `communicate` line:
```python
        if self.is_initializing():
            dummy_actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)
            self.dynamics_net(hidden_states, dummy_actions, deterministic=True)
            self.project_online(hidden_states)
            self.project_target(hidden_states)
            self.communicate(hidden_states)
```
becomes:
```python
        if self.is_initializing():
            dummy_actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)
            self.dynamics_net(hidden_states, dummy_actions, deterministic=True)
            self.project_online(hidden_states)
            self.project_target(hidden_states)
```

Delete the entire `communicate` method (around line 319-335):
```python
    def communicate(self, hidden_states: chex.Array) -> chex.Array:
        """
        Cross-agent attention pass over root latent states.

        Called by MCTSIndependentPlanner before each agent's independent search
        so that each agent's root embedding reflects the other agents' current
        policy state. No-op when attention_type != "transformer".

        Args:
            hidden_states: Root latent states. Shape: (B, N, D)

        Returns:
            Communication-augmented latent states. Shape: (B, N, D)
        """
        if self.communication_net is None:
            return hidden_states
        return self.communication_net(hidden_states, deterministic=True)
```

Also update the class docstring (around line 177-190) to drop the now-removed method from its listed public API — remove `predict` line's mention of `MCTSIndependentPlanner` reference (change to a planner-agnostic description) since that planner no longer exists:
```python
      predict             — prediction head only (used by MCTSIndependentPlanner for
                            prior policies of non-searching agents during each search step)
```
becomes:
```python
      predict             — prediction head only, no dynamics
```

- [ ] **Step 5: Remove dead fields from `MCTSConfig` in `config.py`**

Current (lines 30-44):
```python
@dataclass(frozen=True)
class MCTSConfig:
    """Hyperparameters for the MCTS planner."""
    planner_mode: str
    num_simulations: int
    max_depth_gumbel_search: int
    num_gumbel_samples: int
    dirichlet_alpha: float
    dirichlet_fraction: float
    independent_argmax: bool
    use_root_communication: bool
    mcts_rho: float = 0.25    # OS(λ): top quantile fraction to DISCARD (1-rho is the fraction kept)
    mcts_lambda: float = 0.8  # OS(λ): depth discount weight (lambda^depth)
    pb_c_base: float = 19652.0   # UCB: log-visit exploration scaling (MuZero/MAZero standard)
    pb_c_init: float = 1.25     # UCB: base exploration constant (MuZero/MAZero standard)
    value_delta_lb: float = 0.01  # UCB: min-max Q-normalization floor, prevents divide-by-~0
```

New:
```python
@dataclass(frozen=True)
class MCTSConfig:
    """Hyperparameters for the MCTS planner."""
    planner_mode: str
    num_simulations: int
    max_depth_gumbel_search: int
    num_gumbel_samples: int
    dirichlet_alpha: float
    dirichlet_fraction: float
    mcts_rho: float = 0.25    # OS(λ): top quantile fraction to DISCARD (1-rho is the fraction kept)
    mcts_lambda: float = 0.8  # OS(λ): depth discount weight (lambda^depth)
    pb_c_base: float = 19652.0   # UCB: log-visit exploration scaling (MuZero/MAZero standard)
    pb_c_init: float = 1.25     # UCB: base exploration constant (MuZero/MAZero standard)
    value_delta_lb: float = 0.01  # UCB: min-max Q-normalization floor, prevents divide-by-~0
```

- [ ] **Step 6: Fix config defaults and presets**

`configs/config.yaml` — change the mcts group default:
```yaml
defaults:
  - model: default
  - mcts: default
  - train: default
  - _self_
```
→ no change to this file's structure needed — `mcts: default` still resolves to `configs/mcts/default.yaml`, which Step below repurposes to be joint-mode. (Do **not** change this to `mcts: joint` — `configs/mcts/joint.yaml` is a *lighter* preset (50 sims vs 100), not a synonym for "default planner". Switching the root default's *planner algorithm* is done inside `default.yaml` itself, not by pointing at a different file.)

`configs/mcts/default.yaml` — current:
```yaml
planner_mode: independent  # "independent" or "joint"
num_simulations: 100
max_depth_gumbel_search: 10
num_gumbel_samples: 10
dirichlet_alpha: 0.3
dirichlet_fraction: 0.25
independent_argmax: true
use_root_communication: false
mcts_rho: 0.25
mcts_lambda: 0.8
pb_c_base: 19652.0
pb_c_init: 1.25
value_delta_lb: 0.01
```
New:
```yaml
planner_mode: joint
num_simulations: 100
max_depth_gumbel_search: 10
num_gumbel_samples: 10
dirichlet_alpha: 0.3
dirichlet_fraction: 0.25
mcts_rho: 0.25
mcts_lambda: 0.8
pb_c_base: 19652.0
pb_c_init: 1.25
value_delta_lb: 0.01
```

No changes needed to `configs/mcts/joint.yaml` or `configs/mcts/smax.yaml` — they already set `planner_mode: joint` explicitly and don't reference the two removed fields.

- [ ] **Step 7: Fix `actors/data_actor.py`**

Line 78:
```python
        from mcts import MCTSIndependentPlanner, MCTSJointPlanner, MCTSJointOSLAPlanner
```
→
```python
        from mcts import MCTSJointOSLAPlanner
```

Lines 97-107:
```python
        model = FlaxMAMuZeroNet(config.model, action_size)
        planner_map = {
            "independent": MCTSIndependentPlanner,
            "joint": MCTSJointOSLAPlanner,        # OS(λ) planner replaces mctx-based joint
            "joint_legacy": MCTSJointPlanner,     # keep for ablations
        }
        if config.mcts.planner_mode not in planner_map:
            raise ValueError(
                f"Unknown planner_mode '{config.mcts.planner_mode}'. "
                f"Choose from: {list(planner_map)}"
            )
        planner = planner_map[config.mcts.planner_mode](model=model, config=config)
```
→
```python
        model = FlaxMAMuZeroNet(config.model, action_size)
        planner_map = {"joint": MCTSJointOSLAPlanner}
        if config.mcts.planner_mode not in planner_map:
            raise ValueError(
                f"Unknown planner_mode '{config.mcts.planner_mode}'. "
                f"Choose from: {list(planner_map)}"
            )
        planner = planner_map[config.mcts.planner_mode](model=model, config=config)
```

- [ ] **Step 8: Fix `actors/reanalyze_actor.py`** (same pattern as Step 7)

Line 72:
```python
        from mcts import MCTSIndependentPlanner, MCTSJointPlanner, MCTSJointOSLAPlanner
```
→
```python
        from mcts import MCTSJointOSLAPlanner
```

Lines 82-92:
```python
        model = FlaxMAMuZeroNet(config.model, action_size)
        planner_map = {
            "independent": MCTSIndependentPlanner,
            "joint": MCTSJointOSLAPlanner,
            "joint_legacy": MCTSJointPlanner,
        }
        if config.mcts.planner_mode not in planner_map:
            raise ValueError(
                f"Unknown planner_mode '{config.mcts.planner_mode}'. "
                f"Choose from: {list(planner_map.keys())}"
            )
        planner = planner_map[config.mcts.planner_mode](model=model, config=config)
```
→
```python
        model = FlaxMAMuZeroNet(config.model, action_size)
        planner_map = {"joint": MCTSJointOSLAPlanner}
        if config.mcts.planner_mode not in planner_map:
            raise ValueError(
                f"Unknown planner_mode '{config.mcts.planner_mode}'. "
                f"Choose from: {list(planner_map.keys())}"
            )
        planner = planner_map[config.mcts.planner_mode](model=model, config=config)
```

**Note:** `tests/test_mcts.py::test_reanalyze_actor_uses_osla_planner` asserts the exact string `'"joint": MCTSJointOSLAPlanner'` appears in `actors/reanalyze_actor.py`'s source — the replacement above preserves that exact substring, so this test keeps passing unmodified.

- [ ] **Step 9: Fix `eval.py` (this also fixes a pre-existing bug)**

Line 43:
```python
    from mcts import MCTSIndependentPlanner, MCTSJointPlanner
```
→
```python
    from mcts import MCTSJointOSLAPlanner
```

Lines 87-88 — **note this is a real pre-existing bug**: `eval.py` currently maps `"joint"` to the deleted `MCTSJointPlanner` (mctx-based legacy), while `data_actor.py`/`reanalyze_actor.py` both map `"joint"` to `MCTSJointOSLAPlanner`. `eval.py` was never updated when the OSLA planner became the real "joint" implementation. Fix it to match:
```python
    planner_map = {"independent": MCTSIndependentPlanner, "joint": MCTSJointPlanner}
    planner = planner_map[config.mcts.planner_mode](model=model, config=config)
```
→
```python
    planner_map = {"joint": MCTSJointOSLAPlanner}
    planner = planner_map[config.mcts.planner_mode](model=model, config=config)
```

- [ ] **Step 10: Update `tests/test_mcts.py`**

10a. Module docstring (lines 1-14) — current:
```python
"""
Unit tests for mcts/mcts_independent.py and mcts/mcts_joint.py.

Run with:
    conda run -n mazero pytest tests/test_mcts.py -v

Tests are split into:
  - Shared output contract tests (shapes, validity, determinism) for both planners.
  - Independent-planner-specific tests (argmax vs sample mode).
  - Joint-planner-specific tests (helper functions: logit factorization, marginalization).

Module-scoped fixtures compile JAX/JIT once for the whole file to keep
the total run time reasonable.
"""
```
New:
```python
"""
Unit tests for mcts/mcts_joint_osla.py (the OS(λ) joint MCTS planner) and
its helper functions.

Run with:
    conda run -n mazero pytest tests/test_mcts.py -v

Module-scoped fixtures compile JAX/JIT once for the whole file to keep
the total run time reasonable.
"""
```

10b. Import line 24:
```python
from mcts import MCTSPlanOutput, MCTSIndependentPlanner, MCTSJointPlanner
```
→
```python
from mcts import MCTSPlanOutput
```

10c. `test_config` fixture's `mcts=MCTSConfig(...)` block (lines 96-107):
```python
        mcts=MCTSConfig(
            planner_mode="independent",
            num_simulations=8,           # minimum viable for gumbel (>= num_gumbel_samples)
            max_depth_gumbel_search=3,
            num_gumbel_samples=4,
            dirichlet_alpha=0.3,
            dirichlet_fraction=0.25,
            independent_argmax=True,
            use_root_communication=False,
            mcts_rho=0.75,
            mcts_lambda=0.8,
        ),
```
→
```python
        mcts=MCTSConfig(
            planner_mode="joint",
            num_simulations=8,           # minimum viable for gumbel (>= num_gumbel_samples)
            max_depth_gumbel_search=3,
            num_gumbel_samples=4,
            dirichlet_alpha=0.3,
            dirichlet_fraction=0.25,
            mcts_rho=0.75,
            mcts_lambda=0.8,
        ),
```

10d. Delete the `independent_plan_fn` and `joint_plan_fn` fixtures entirely (lines 121-134):
```python
@pytest.fixture(scope="module")
def independent_plan_fn(model_and_params, test_config):
    """JIT-compiled plan function for the independent planner (compiled once)."""
    net, _ = model_and_params
    planner = MCTSIndependentPlanner(model=net, config=test_config)
    return jax.jit(planner.plan), planner


@pytest.fixture(scope="module")
def joint_plan_fn(model_and_params, test_config):
    """JIT-compiled plan function for the joint planner (compiled once)."""
    net, _ = model_and_params
    planner = MCTSJointPlanner(model=net, config=test_config)
    return jax.jit(planner.plan), planner
```

10e. Delete the now-redundant `osla_config` fixture (lines 137-155) — it becomes byte-identical to `test_config.mcts` once Step 10c switches `test_config`'s `planner_mode` to `"joint"` and drops the two removed fields. Change `osla_plan_fn` (lines 158-164) to depend on `test_config` directly instead:
```python
@pytest.fixture(scope="module")
def osla_config(test_config):
    """MCTSConfig with joint planner + OS(λ)."""
    return ExperimentConfig(
        train=test_config.train,
        model=test_config.model,
        mcts=MCTSConfig(
            planner_mode="joint",
            num_simulations=8,
            max_depth_gumbel_search=3,
            num_gumbel_samples=4,
            dirichlet_alpha=0.3,
            dirichlet_fraction=0.25,
            independent_argmax=True,
            use_root_communication=False,
            mcts_rho=0.75,
            mcts_lambda=0.8,
        ),
    )


@pytest.fixture(scope="module")
def osla_plan_fn(model_and_params, osla_config):
    """JIT-compiled plan function for MCTSJointOSLAPlanner."""
    from mcts.mcts_joint_osla import MCTSJointOSLAPlanner
    net, _ = model_and_params
    planner = MCTSJointOSLAPlanner(model=net, config=osla_config)
    return jax.jit(planner.plan), planner
```
→
```python
@pytest.fixture(scope="module")
def osla_plan_fn(model_and_params, test_config):
    """JIT-compiled plan function for MCTSJointOSLAPlanner."""
    from mcts.mcts_joint_osla import MCTSJointOSLAPlanner
    net, _ = model_and_params
    planner = MCTSJointOSLAPlanner(model=net, config=test_config)
    return jax.jit(planner.plan), planner
```

10f. Delete the entire `TestMCTSIndependentPlanner` class (the `# ─── Independent planner tests ──` banner + class, lines 179-286) and the entire `TestMCTSJointPlanner` class (the `# ─── Joint planner tests ──` banner + class, lines 288-441) in one contiguous deletion — everything between the `osla_plan_fn` fixture (ends ~line 164) and the `# ─── MCTSConfig field tests ───` banner (line 444) except the `params`/`obs` fixtures (lines 167-176), which must be kept (used throughout the rest of the file). Concretely: keep lines 167-176 (`params`/`obs` fixtures), delete everything from the `# ─── Independent planner tests ──` banner through the end of `TestMCTSJointPlanner` (through line 441).

10g. `test_mcts_config_has_osla_fields` (lines 446-461) — remove the two dead kwargs:
```python
def test_mcts_config_has_osla_fields():
    from config import MCTSConfig
    cfg = MCTSConfig(
        planner_mode="joint",
        num_simulations=8,
        max_depth_gumbel_search=3,
        num_gumbel_samples=4,
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
        independent_argmax=True,
        use_root_communication=False,
        mcts_rho=0.75,
        mcts_lambda=0.8,
    )
    assert cfg.mcts_rho == 0.75
    assert cfg.mcts_lambda == 0.8
```
→
```python
def test_mcts_config_has_osla_fields():
    from config import MCTSConfig
    cfg = MCTSConfig(
        planner_mode="joint",
        num_simulations=8,
        max_depth_gumbel_search=3,
        num_gumbel_samples=4,
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
        mcts_rho=0.75,
        mcts_lambda=0.8,
    )
    assert cfg.mcts_rho == 0.75
    assert cfg.mcts_lambda == 0.8
```

- [ ] **Step 11: Update CLAUDE.md**

11a. In the `mcts/` block of **Package Layout**, delete these two lines:
```
  mcts_independent.py     # MCTSIndependentPlanner; uses model.communicate() before each search
  mcts_joint.py            # MCTSJointPlanner (legacy mctx-based; used as joint_legacy)
```
Leave `base.py` and `mcts_joint_osla.py` lines as-is (Task 9 will edit the latter further).

11b. In the `model/` block of **Package Layout**, change:
```
    model.py                # FlaxMAMuZeroNet and sub-networks; communicate() for root attention
```
→
```
    model.py                # FlaxMAMuZeroNet and sub-networks
```

11c. In the `configs/mcts/` block of **Package Layout**, change:
```
    default.yaml          # MCTS hyperparameters (independent planner)
```
→
```
    default.yaml          # MCTS hyperparameters (base defaults; joint planner)
```

11d. In **Architecture → MCTS planners**, delete these two bullets:
```
- `MCTSIndependentPlanner`: one `gumbel_muzero_policy` search per agent via `jax.lax.scan`; other agents fixed to prior argmax during each agent's search. Optionally calls `model.communicate()` on root latents before each search when `use_root_communication=True`. Uses mctx.
- `MCTSJointPlanner` (`joint_legacy`): single search over `A^N` joint space with mctx. Mean backup. Kept for ablations.
```
And change the remaining bullet:
```
- `MCTSJointOSLAPlanner` (`joint`, default for SMAX): custom JAX MCTS (not mctx). Per-node OS(λ) backup — each node tracks per-simulation values/depths; UCB selection uses OS(λ)-estimated Q-values (top (1-rho) quantile weighted by λ^depth). Vmapped over B environments; `jax.lax.fori_loop` over simulations. Matches MAZero algorithm exactly.
```
→
```
- `MCTSJointOSLAPlanner` (`joint`, the only planner, default everywhere): custom JAX MCTS (not mctx's search — `mctx.RecurrentFnOutput` is reused as a plain return-type container). Per-node OS(λ) backup — each node tracks per-simulation values/depths; UCB selection uses OS(λ)-estimated Q-values (top (1-rho) quantile weighted by λ^depth). Vmapped over B environments; `jax.lax.fori_loop` over simulations. Matches MAZero algorithm exactly.
```

11e. In **World model** bullets, delete the `communicate()` bullet entirely:
```
  - `communicate()`: separate `TransformerAttentionEncoder` (distinct params from dynamics attention) for pre-search cross-agent attention in `MCTSIndependentPlanner`. No-op when `attention_type != "transformer"`. Controlled by `use_root_communication` in `MCTSConfig`.
```

11f. In **Notable config fields added since original docs**, delete:
```
- `MCTSConfig.use_root_communication: bool` — enables `model.communicate()` before independent search
```

11g. In the **MAZero Reference Implementation** section, delete this bullet (it documents a feature this task removes, so it no longer applies):
```
**No `communicate()` in MAZero**: The separate pre-search root attention pass is our addition.
```

11h. Update **Current Training Target** if it references `mcts=joint` as if switching away from a different default — check the line:
```
python train/muzero.py train=smax_3m model=smax mcts=joint
```
This still works fine (it explicitly selects the lighter 50-sim `joint.yaml` preset over the new default's 100-sim `default.yaml`), so **no change needed** here — just confirm while editing nearby text that nothing implies `independent` is still an option.

- [ ] **Step 12: Run the full test suite**

```bash
pytest tests/ -v
```
Expected: all tests pass (previously 141; expect a small reduction from deleting `TestMCTSIndependentPlanner`'s 12 tests and `TestMCTSJointPlanner`'s 17 tests, ~112 remaining). Investigate and fix any failure before proceeding — do not skip or xfail.

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: remove MCTSIndependentPlanner and MCTSJointPlanner (dead ablation code)

MCTSJointOSLAPlanner is the sole planner used anywhere in this repo's
training configs and just passed a full correctness audit. Removes the
two unused mctx-based planners, model.communicate() (their only
consumer), and the now-dead independent_argmax/use_root_communication
config fields.

Behavior change: the root config default (configs/mcts/default.yaml)
switches planner_mode from "independent" to "joint" — anyone running
`python train/muzero.py` with no mcts= override now gets the OS(λ)
planner instead of the never-audited independent planner.

Also fixes eval.py, which had gone stale and still mapped "joint" to
the legacy mctx-based MCTSJointPlanner instead of MCTSJointOSLAPlanner
(data_actor.py and reanalyze_actor.py already had this right).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Git hygiene — stop tracking `__pycache__`/`.pyc`

**Files:**
- Modify: `.gitignore`
- Delete (from git index only, not disk): all tracked `__pycache__/` and `*.pyc` files

**Interfaces:** None — this task has no code interface, it only affects what git tracks.

- [ ] **Step 1: Confirm the current tracked count**

```bash
git ls-files | grep -c "__pycache__\|\.pyc$"
```
Expected: a nonzero count (the design spec estimated ~57).

- [ ] **Step 2: Update `.gitignore`**

Current content:
```
wandb
.worktrees/
```
New content:
```
wandb
.worktrees/
__pycache__/
*.pyc
```

- [ ] **Step 3: Untrack (but don't delete on disk) all currently-tracked pycache files**

```bash
git rm -r --cached --ignore-unmatch $(git ls-files | grep "__pycache__\|\.pyc$")
```

- [ ] **Step 4: Verify nothing else broke**

```bash
git status
pytest tests/ -v
```
Expected: `git status` shows only the intended deletions-from-index plus the `.gitignore` change (compiled files remain on disk, just untracked — pytest will happily regenerate/read them). All tests still pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore: stop tracking __pycache__/.pyc files

A portfolio repo shouldn't have compiled bytecode checked into git.
Files stay on disk (still gitignored going forward), just untracked.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Trim comments in `mcts/mcts_joint_osla.py` (Pass 2, worked example)

**Files:**
- Modify: `mcts/mcts_joint_osla.py`

**Interfaces:** None — comment-only changes, no signature or behavior changes.

**The rule** (from the spec, applies to every remaining Pass 2 task too): delete comments that restate what the code already says. Keep only WHY — non-obvious rationale, invariants, workarounds, or algorithm/paper references. Single line by default; 2-4 line paragraphs allowed only where the reasoning genuinely needs it. Where a comment explains OS(λ)/AWPO/UCB math, cite the paper's equation number (arXiv 2405.11778) instead of re-deriving it in prose.

- [ ] **Step 1: Compress `compute_osla_value_jax`'s docstring**

Current (lines 24-30):
```python
    """JAX-native OS(λ), matching MAZero's `SubTreeValueSet`: the top
    (1-rho) quantile is selected SEPARATELY within each depth bucket, then
    buckets are combined with lambda^depth weighting. Uses a single global
    sort plus a per-bucket cumulative rank (rather than one argsort per
    depth bucket) to stay O(max_sims log max_sims) instead of
    O(max_depth * max_sims log max_sims).
    """
```
New:
```python
    """OS(λ) value estimate, paper Eq. 5-7 (top-(1-rho) quantile per depth
    bucket, weighted by lambda^depth), matching MAZero's `SubTreeValueSet`.
    Uses one global sort + per-bucket cumulative rank instead of one argsort
    per depth bucket: O(max_sims log max_sims) instead of O(max_depth *
    max_sims log max_sims).
    """
```

- [ ] **Step 2: Compress the tie-break comment inside `_best_ucb`**

Current (lines 219-226):
```python
        # Tie-break toward the highest-prior child: when a node has just
        # been expanded (visit_count==1), parent_visits=0 makes pb_c=0 for
        # every child, and unvisited children also score value_score=0, so
        # the whole UCB vector is [0, ..., 0]. Without this epsilon,
        # argmax would deterministically pick index 0 regardless of prior,
        # defeating prior-guided exploration at every node's first
        # internal descent. 1e-6 is small enough to only break exact ties,
        # not perturb genuine UCB comparisons.
```
New:
```python
        # At a freshly-expanded node (parent_visits=0), UCB is all-zero for
        # every child, so bare argmax always picks index 0 regardless of
        # prior. The 1e-6*prior tie-break fixes that without perturbing
        # genuine (non-degenerate) UCB comparisons.
```

- [ ] **Step 3: Compress the backup-direction comment block above `backup_step`**

Current (lines 322-327):
```python
    # ── 4. Backup ─────────────────────────────────────────────────────────────
    # Walk from leaf (leaf_depth) up to root (depth 0), updating visit_counts and value_sum.
    # V at each node = reward_into_that_node + gamma * V_below.
    # We go from leaf upward: at step k=0, update leaf (depth=leaf_depth) with V=leaf_value.
    # At step k=1, update parent (depth=leaf_depth-1) with V = reward[leaf] + gamma * leaf_value.
    # etc.
```
New:
```python
    # ── 4. Backup ─────────────────────────────────────────────────────────────
    # Leaf-to-root walk: V(node) = reward_into_node + gamma * V(child). Step k
    # updates the node at depth (leaf_depth - k), starting at the leaf (k=0).
```

- [ ] **Step 4: Compress the qmin/qmax rationale comment inside `backup_step`**

Current (lines 349-354):
```python
        # Running min/max Q-baseline-diff, for non-root nodes only (matches
        # MAZero: `if (i != 0) minmax_stat.insert(...)`). Uses this
        # simulation's own backed-up value V as the node's Q signal rather
        # than recomputing the full OS(λ) estimate here, which would need
        # an extra O(num_simulations) argsort per backup step — the running
        # bound only needs to stay roughly representative, not exact.
```
New:
```python
        # Running min/max Q-baseline-diff over non-root nodes only (matches
        # MAZero's `if (i != 0) minmax_stat.insert(...)`). Uses this sim's own
        # backed-up V as the Q signal instead of a full OS(λ) recompute here —
        # the running bound only needs to stay roughly representative.
```

- [ ] **Step 5: Compress the root cumulative-reward comment above `accum_step`**

Current (lines 371-372):
```python
    # ── 5. Record OS(λ) data ───────────────────────────────────────────────────
    # Root backup value: cumulative discounted rewards from root to leaf, plus gamma^depth * V_leaf
```
New (keep — already one line, no change needed): leave as-is.

- [ ] **Step 6: Leave all shape-annotation comments untouched**

Comments like `# [max_sims] float32` or `# (B, N)` are type/shape annotations, not explanatory prose — they stay exactly as-is throughout this file (and every other file in this plan). Do not remove or alter these.

- [ ] **Step 7: Run tests**

```bash
pytest tests/test_mcts.py -v
```
Expected: all pass, unchanged from Task 1's post-fix count (comment-only edit).

- [ ] **Step 8: Commit**

```bash
git add mcts/mcts_joint_osla.py
git commit -m "$(cat <<'EOF'
docs: trim mcts_joint_osla.py comments to WHY-only, cite paper equations

Comment-only change. Cites arXiv 2405.11778 Eq. 5-7 for OS(lambda)
instead of re-deriving the math in prose; compresses multi-paragraph
rationale comments to their essential point.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Trim comments/docstrings in `actors/learner_actor.py`

**Files:**
- Modify: `actors/learner_actor.py`

**Interfaces:** None — comment/docstring-only changes.

- [ ] **Step 1: Delete two trivial docstrings that just restate the function name**

Current (lines 28-30, 33-35):
```python
def _scale_grad_half_fwd(x):
    """Forward pass: identity."""
    return x, ()


def _scale_grad_half_bwd(_, g):
    """Backward pass: halve the gradient."""
    return (_jax.tree_util.tree_map(lambda gi: gi * 0.5, g),)
```
New:
```python
def _scale_grad_half_fwd(x):
    return x, ()


def _scale_grad_half_bwd(_, g):
    return (_jax.tree_util.tree_map(lambda gi: gi * 0.5, g),)
```
(`scale_grad_half`'s own docstring above these, lines 20-24, already explains the mechanism and stays as-is — it's genuine WHY, not restated WHAT.)

- [ ] **Step 2: Compress the multi-step consistency comment block**

Current (lines 264-274):
```python
            # ---- Multi-step SPR consistency ----
            # For each k in 1..consistency_horizon and each valid start position t,
            # compare project_online(h_t) against project_target(h_{t+k}).
            # k=1 reproduces the original single-step consistency loss.
            # k>1 adds longer-range targets, improving latent prediction accuracy
            # over multiple dynamics steps (SPR, Schwarzer et al. 2021).
            #
            # XLA CSE: project_online(h_t) is only computed once regardless of how
            # many k values reference h_t, so cost stays O(U+1) projections.
            #
            # all_hiddens[i] = h_i,  shape (U+1, B, N, D)
```
New:
```python
            # ---- Multi-step SPR consistency (Schwarzer et al. 2021) ----
            # For each k in 1..consistency_horizon and start t, compare
            # project_online(h_t) vs project_target(h_{t+k}); k=1 is the
            # original single-step loss. XLA CSE means project_online(h_t)
            # is computed once regardless of how many k reference it.
            # all_hiddens[i] = h_i,  shape (U+1, B, N, D)
```

- [ ] **Step 3: Consolidate the repeated "current V_net, not stale MCTS value" explanation**

This idea is currently explained three times: in `_awpo_weight`'s docstring (lines 42-46, keep as the single source of truth — it's already concise and correct), and again in prose at lines 107-111 and 116-120. Current (lines 107-120):
```python
            # AWPO: weight root policy loss by action-level AWAC weights.
            # Uses current V_net as baseline + batch normalization of advantages
            # so the signal is non-trivial even when Q ≈ V ≈ 0 (MAZero formulation).
            # When awpo_alpha=0 (disabled), this branch is eliminated at JIT
            # trace time and reduces to plain mean cross-entropy.
            ce_p0 = optax.softmax_cross_entropy(
                init_out.policy_logits, batch.policy_target[:, 0]
            ).mean(axis=-1)  # (B,)
            if awpo_alpha > 0.0 and q_data is not None:
                # Action-level AWPO (MAZero/AWAC formulation): weight each sampled
                # root action by exp((Q_k - V_net - mean) / (std + eps) / alpha).
                # V_net is the CURRENT network prediction (not stale stored value).
                # Batch normalization ensures non-trivial gradient signal even when
                # Q ≈ V ≈ 0 — critical for cold-start on sparse rewards like SMAX.
```
New:
```python
            # AWPO root policy loss (paper Eq. 10, 14); no-op mean CE when
            # awpo_alpha=0 (branch eliminated at JIT trace time).
            ce_p0 = optax.softmax_cross_entropy(
                init_out.policy_logits, batch.policy_target[:, 0]
            ).mean(axis=-1)  # (B,)
            if awpo_alpha > 0.0 and q_data is not None:
                # Action-level AWPO: weight each sampled root action by
                # exp((Q_k - V_net) / alpha), batch-normalized (_awpo_weight).
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_learner.py -v
```
Expected: all pass (comment-only edit).

- [ ] **Step 5: Commit**

```bash
git add actors/learner_actor.py
git commit -m "$(cat <<'EOF'
docs: trim learner_actor.py comments to WHY-only

Comment-only change. Drops two trivial docstrings that just restated
their function name, compresses the SPR consistency comment, and
consolidates three repetitions of the same AWPO baseline explanation
down to one (in _awpo_weight's docstring, already correct).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Trim comments in `utils/replay_buffer.py`

**Files:**
- Modify: `utils/replay_buffer.py`

**Interfaces:** None — comment-only changes.

- [ ] **Step 1: Delete pure banner comments that just restate the method/section name above**

These 7 blocks add nothing beyond what the function signature or class docstring already says — delete each 3-line banner entirely, leaving the code that follows untouched:

- Lines 93-96: `# C++ backend detection...` / `# Falls back silently...` banner above the import-detection block — delete the banner, but keep a single-line version of the "falls back silently" fact if it's not already stated elsewhere: replace with one line `# Falls back to the pure-Python implementation if the C++ extension isn't built.`
- Lines 107-109: `# ReplayBuffer — thin shim that delegates to C++ when available.` banner — delete (this belongs in the class's own docstring; check `class ReplayBuffer:` has this fact already documented, and if not, add it there instead of as a floating comment).
- Lines 173-175: `# add()` banner — delete.
- Lines 207-209: `# sample()` banner — delete.
- Lines 246-248: `# sample_for_reanalysis()` banner — delete.
- Lines 262-264: `# update_targets()` banner — delete.
- Lines 277-279: `# update_priorities()` banner — delete.
- Lines 291-293: `# get_stats()` banner — delete.
- Lines 319-321: `# Episode processing (pure Python / NumPy — unchanged)` banner — delete.

Line numbers above may drift slightly as earlier deletions in this same task shift later ones — re-run `grep -n "^\s*#" utils/replay_buffer.py` before each deletion if the surrounding content doesn't match what's shown here.

- [ ] **Step 2: Keep and lightly shorten the genuine WHY comments**

Lines 54-56 (keep, already fine, no edit needed):
```python
    # Per-step Q-data for action-level AWPO at all U+1 positions.
    # Not in the JAX pytree — stored in ReplayBufferActor sidecar, injected at sample time.
    # None when collected by a non-OSLA planner.
```

Lines 376-377 (keep, already fine, no edit needed):
```python
        # Broadcast scalar value/reward targets to per-agent arrays.
        # np.broadcast_to returns a read-only view; .copy() makes it writable.
```

Lines 386-387 (keep, already fine, no edit needed):
```python
        # Collect Q-data for all U+1 positions in this window.
        # Require ALL positions to have Q-data; partial windows are treated as no Q-data.
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_replay_buffer.py -v
```
Expected: all pass (comment-only edit).

- [ ] **Step 4: Commit**

```bash
git add utils/replay_buffer.py
git commit -m "$(cat <<'EOF'
docs: remove restating banner comments from replay_buffer.py

Comment-only change. Deletes 7 section banners that just repeated the
method name directly below them (add(), sample(), etc.) — the
signature already says that. Genuine WHY comments (Q-data sidecar,
broadcast_to copy-on-write gotcha) are untouched.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Trim docstrings in `utils/obs_norm.py` and `utils/profiler.py`

**Files:**
- Modify: `utils/obs_norm.py`, `utils/profiler.py`

**Interfaces:** None — docstring-only changes.

- [ ] **Step 1: Compress `utils/obs_norm.py`'s module docstring**

Current (lines 2-21):
```python
"""
Online EMA normalizer for observations.

Maintains a running mean and variance over the per-feature observation
statistics using exponential moving averages.  Updated on each training batch
(over all (B * N, obs_size) observations); read-only during MCTS data
collection.

The normalizer state is serializable (plain numpy dicts) so it can be synced
to DataActors and ReanalyzeActors alongside model parameters via get_params().

Usage in LearnerActor._train_step():
    self.obs_norm.update(batch.observation)       # update running stats
    batch = replace(batch, observation=self.obs_norm.normalize(batch.observation))
    # then device_put and train_step as usual

Usage in DataActor.run_episode():
    obs_to_plan = self.obs_norm.normalize(np.array(observations))
    plan_output = self.plan_fn(self.params, plan_key, jnp.array(obs_to_plan))
"""
```
New:
```python
"""EMA per-feature observation normalizer. State is plain-numpy serializable
so it syncs to DataActors/ReanalyzeActors alongside model params via
get_params(); read-only during MCTS data collection, updated during training.
"""
```

Also remove the two banner comments inside the class (they restate the method group's purpose, which the docstrings right below each method already cover):

Lines 48-50:
```python
    # ------------------------------------------------------------------
    # Update / normalize
    # ------------------------------------------------------------------
```
→ delete entirely.

Lines 86-88:
```python
    # ------------------------------------------------------------------
    # Serialization (for syncing to DataActors alongside model params)
    # ------------------------------------------------------------------
```
→ delete entirely.

Keep the cold-start comment (lines 64-65) exactly as-is — genuine WHY:
```python
            # Cold start: use batch stats directly so the first normalization
            # is already sensible instead of dividing by 1.0 everywhere.
```

- [ ] **Step 2: Compress `utils/profiler.py`'s module docstring**

Current (lines 1-24):
```python
"""Lightweight wall-clock profiler for periodic performance logging.

Each actor creates one Profiler instance. Operations are timed with the
`time()` context manager; `step()` is called once per logical unit of work
(training step, episode, reanalyze batch). Stats are logged and reset every
`log_interval` steps.

JAX note: JAX dispatches GPU kernels asynchronously. To measure actual GPU
compute time (not just dispatch time), call `jax.block_until_ready(result)`
inside the `time()` block before it exits.

Example::

    profiler = Profiler("LearnerActor", log_interval=100)

    with profiler.time("sample_wait"):
        batch = ray.get(prefetch_future)

    with profiler.time("train_step"):
        params, metrics = train_step(...)
        jax.block_until_ready(params)   # blocks until GPU kernel finishes

    profiler.step()                     # logs every log_interval calls
"""
```
New:
```python
"""Wall-clock profiler: time() context manager per named operation, step()
logs and resets every log_interval calls. JAX note: call
jax.block_until_ready(result) inside time() to measure actual compute time,
not just async dispatch time.
"""
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/ -v -k "obs_norm or profiler"
```
If no tests match that filter, instead run the full suite (these two modules may not have dedicated test files):
```bash
pytest tests/ -v
```
Expected: all pass (docstring-only edit).

- [ ] **Step 4: Commit**

```bash
git add utils/obs_norm.py utils/profiler.py
git commit -m "$(cat <<'EOF'
docs: trim obs_norm.py and profiler.py module docstrings

Docstring-only change. Cuts usage examples that duplicated actual call
sites in learner_actor.py/data_actor.py down to a 1-3 line summary;
keeps the non-obvious JAX async-dispatch gotcha in profiler.py.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Quick-check sweep of remaining files (expect minimal changes)

**Files:**
- Review only, fix if needed: `mcts/base.py`, `model/attention.py`, `model/layers.py`, `model/model.py`, `actors/data_actor.py`, `actors/reanalyze_actor.py`, `actors/replay_buffer_actor.py`, `training/loop.py`, `utils/transforms.py`, `utils/logging_utils.py`, `config.py`, `envs/smax_env_wrapper.py`, `envs/mpe_env_wrapper.py`, `envs/__init__.py`

**Interfaces:** None expected — these files were already surveyed and found to already follow the WHY-only rule (comment density is 0-25 lines each, all reviewed and found to be genuine rationale/gotcha comments already, e.g. Ray's `CUDA_VISIBLE_DEVICES` quirk in `data_actor.py`, JaxMARL auto-reset/win-detection in `smax_env_wrapper.py`). This task is a confirmation pass, not a rewrite.

- [ ] **Step 1: Fix one confirmed stale comment in `envs/mpe_env_wrapper.py`**

Line 1 currently reads:
```python
# utils/mpe_env_wrapper.py
```
This is a leftover from before the file moved from `utils/` to `envs/` — wrong path. Fix to:
```python
# envs/mpe_env_wrapper.py
```

- [ ] **Step 2: Spot-check the rest for anything that restates code**

For each file in this task's list, run:
```bash
grep -n "^\s*#" <file>
```
and read every comment line found. Only edit a line if it purely restates what the adjacent code/signature already says (e.g. `# increment counter` above `counter += 1`). Based on the pre-plan survey, expect **zero further edits** — every comment found in these files during planning was genuine WHY (GPU memory allocation ordering in `data_actor.py`, thread-pool thrashing rationale, `communicate()`-adjacent comments already removed in Task 1 from `model/model.py`, etc.). If you find something that should change, apply the same rule as Tasks 3-6 (WHY-only, cite paper equations for algorithm comments) and note what you changed and why in the commit message.

- [ ] **Step 3: Run tests**

```bash
pytest tests/ -v
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
docs: fix stale file-path comment in mpe_env_wrapper.py

Quick-check sweep of the remaining low-comment-density files
(mcts/base.py, model/attention.py, model/layers.py, actors/data_actor.py,
actors/reanalyze_actor.py, actors/replay_buffer_actor.py, training/loop.py,
utils/transforms.py, utils/logging_utils.py, config.py, envs/*) found
they already follow the WHY-only comment rule — no further changes
needed beyond this one leftover path comment from before the file
moved out of utils/.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Split `mcts/mcts_joint_osla.py` — extract pure helper functions into `mcts/osla_math.py`

**Files:**
- Create: `mcts/osla_math.py`
- Modify: `mcts/mcts_joint_osla.py`, `tests/test_mcts.py`

**Interfaces:**
- Produces: `mcts/osla_math.py` exports `compute_osla_value_jax`, `compute_osla_value`, `compute_ucb_scores`, `_sample_k_actions`, `_logits_to_joint_logits`, `_joint_policy_to_marginal` — all pure, stateless functions with no JAX closures over mutable tree state. `mcts/mcts_joint_osla.py` imports these from `mcts.osla_math` and keeps `OSLATree`, `SimCarry`, `SelectCarry`, `_run_single_sim`, `_osla_plan_single`, `MCTSJointOSLAPlanner`.

**Important — this deviates from the original 3-way split floated during brainstorming** (which proposed separate `osla_selection.py` and `osla_backup.py` files). Investigation during plan-writing found that the selection logic (`_best_ucb`, `_select_cond`, `_select_body`) and backup logic (`backup_step`, `accum_step`) inside `_run_single_sim` are **nested closures**, not standalone top-level functions — they capture `tree`, `rng`, `leaf_depth`, `path_nodes`, `path_rewards`, `gamma` etc. from `_run_single_sim`'s own scope. Splitting those out would require refactoring them into explicit-parameter top-level functions, a real (if mechanical) rewrite of a file that just passed a rigorous correctness audit — too much correctness risk for a comments/organization pass. The safe, already-tested split boundary is the set of functions that are already standalone and already unit-tested in isolation (`TestComputeOslaValue`, `TestOSLAHelpers`, `TestUCBLogVisitScaling`, `TestSampleKActions` in `tests/test_mcts.py` all test these exact functions with no tree/carry state) — that's what this task does instead.

- [ ] **Step 1: Create `mcts/osla_math.py` with the six standalone functions**

Move these functions out of `mcts/mcts_joint_osla.py` verbatim (post-Task-3 comment trims already applied — use the current file content, not the pre-Task-3 version):
- `compute_osla_value_jax` (current lines ~16-47)
- `compute_osla_value` (current lines ~50-63)
- `compute_ucb_scores` (current lines ~118-146)
- `_sample_k_actions` (current lines ~151-161)
- `_logits_to_joint_logits` (current lines ~398-408)
- `_joint_policy_to_marginal` (current lines ~411-423)

```python
# mcts/osla_math.py
"""Pure OS(λ) value/UCB/policy helper functions for MCTSJointOSLAPlanner —
no JAX closures over tree state, each independently unit-tested."""

import chex
import jax
import jax.numpy as jnp


def compute_osla_value_jax(
    sim_values: chex.Array,
    sim_depths: chex.Array,
    n_visits: chex.Array,
    rho: float,
    lam: float,
    max_depth: int,
) -> chex.Array:
    """OS(λ) value estimate, paper Eq. 5-7 (top-(1-rho) quantile per depth
    bucket, weighted by lambda^depth), matching MAZero's `SubTreeValueSet`.
    Uses one global sort + per-bucket cumulative rank instead of one argsort
    per depth bucket: O(max_sims log max_sims) instead of O(max_depth *
    max_sims log max_sims).
    """
    max_sims = sim_values.shape[0]
    valid_mask = jnp.arange(max_sims) < n_visits

    order = jnp.argsort(jnp.where(valid_mask, sim_values, -jnp.inf))[::-1]
    v_s = sim_values[order]
    d_s = sim_depths[order]
    m_s = valid_mask[order]

    onehot = jax.nn.one_hot(d_s, max_depth) * m_s[:, None]
    rank_in_bucket = (jnp.cumsum(onehot, axis=0) - 1.0)[jnp.arange(max_sims), d_s]
    counts = onehot.sum(axis=0)
    size_lim = jnp.maximum(1, jnp.ceil(counts * (1.0 - rho))).astype(jnp.int32)

    include = m_s & (rank_in_bucket < size_lim[d_s])
    weight = jnp.where(include, lam ** d_s.astype(jnp.float32), 0.0)

    return (v_s * weight).sum() / (weight.sum() + 1e-8)


def compute_osla_value(
    sim_depths: chex.Array,
    sim_values: chex.Array,
    rho: float = 0.25,
    lam: float = 0.8,
    max_depth: int = 20,
) -> chex.Array:
    """Non-padded convenience wrapper over `compute_osla_value_jax` for
    callers where every entry in `sim_values`/`sim_depths` is valid.
    """
    return compute_osla_value_jax(
        sim_values, sim_depths, jnp.array(sim_values.shape[0]), rho, lam, max_depth
    )


def compute_ucb_scores(
    child_q_baseline_diff: chex.Array,
    child_visits: chex.Array,
    prior_probs: chex.Array,
    parent_visits: chex.Array,
    qmin: chex.Array,
    qmax: chex.Array,
    pb_c_base: float,
    pb_c_init: float,
    value_delta_lb: float,
) -> chex.Array:
    """PUCT with MuZero/MAZero log-visit pb_c scaling and min-max Q
    normalization (matches `CTree::ucb_score` in MAZero's cnode.cpp).
    """
    pb_c = jnp.log((parent_visits + pb_c_base + 1.0) / pb_c_base) + pb_c_init
    pb_c = pb_c * jnp.sqrt(parent_visits) / (1.0 + child_visits)
    prior_score = pb_c * prior_probs

    has_stats = qmax > qmin
    delta = jnp.maximum(value_delta_lb, qmax - qmin)
    normalized = jnp.where(has_stats, (child_q_baseline_diff - qmin) / delta, child_q_baseline_diff)
    value_score = jnp.where(child_visits > 0, jnp.clip(normalized, 0.0, 1.0), 0.0)

    return prior_score + value_score


def _sample_k_actions(
    rng: chex.Array,
    logits: chex.Array,
    K: int,
    A_N: int,
) -> tuple[chex.Array, chex.Array]:
    """Sample K joint actions proportional to prior; return (actions [K] int32, probs [K] float32)."""
    probs = jax.nn.softmax(logits)
    replace = K >= A_N
    actions = jax.random.choice(rng, A_N, shape=(K,), replace=replace, p=probs)
    return actions, probs[actions]


def _logits_to_joint_logits(
    per_agent_logits: chex.Array,
    N: int,
) -> chex.Array:
    """Per-agent logits (N, A) → joint logits (A_N,) under independence assumption."""
    A = per_agent_logits.shape[-1]
    log_probs = jax.nn.log_softmax(per_agent_logits, axis=-1)
    joint = log_probs[0]
    for i in range(1, N):
        joint = (joint[:, None] + log_probs[i][None, :]).reshape(-1)
    return joint


def _joint_policy_to_marginal(
    joint_policy: chex.Array,
    N: int,
    joint_action_shape: tuple,
) -> chex.Array:
    """Joint policy (B, A_N) → per-agent marginals (B, N, A)."""
    B = joint_policy.shape[0]
    reshaped = joint_policy.reshape(B, *joint_action_shape)
    marginals = []
    for i in range(N):
        other_axes = tuple(j + 1 for j in range(N) if j != i)
        marginals.append(jnp.sum(reshaped, axis=other_axes))
    return jnp.stack(marginals, axis=1)
```

- [ ] **Step 2: Delete the six moved functions from `mcts/mcts_joint_osla.py` and import them instead**

Remove their definitions (the same line ranges as Step 1). Add near the top of the file, after the existing imports:
```python
from mcts.osla_math import (
    compute_osla_value_jax,
    compute_osla_value,
    compute_ucb_scores,
    _sample_k_actions,
    _logits_to_joint_logits,
    _joint_policy_to_marginal,
)
```
Every call site within `mcts_joint_osla.py` (`_best_ucb`, `_run_single_sim`, `_osla_plan_single`, `recurrent_fn_batched`) keeps working unchanged since the imported names are identical.

- [ ] **Step 3: Update `tests/test_mcts.py` imports**

Grep for every import site that needs splitting:
```bash
grep -n "from mcts.mcts_joint_osla import" tests/test_mcts.py
```
For each matched line, split it into two import lines: one `from mcts.osla_math import ...` for any of `compute_osla_value`, `compute_osla_value_jax`, `compute_ucb_scores`, `_sample_k_actions` that appear on that line, and one `from mcts.mcts_joint_osla import ...` for anything else on that line (`_run_single_sim`, `OSLATree`, `SimCarry`, `MCTSJointOSLAPlanner`). Concretely, based on the pre-split file, these are the lines to fix (re-verify line numbers against the live file, since Task 1 already shifted some of them):

- `TestComputeOslaValue` class: 4 occurrences of `from mcts.mcts_joint_osla import compute_osla_value` → `from mcts.osla_math import compute_osla_value`
- `TestOSLAHelpers`: `from mcts.mcts_joint_osla import compute_ucb_scores` (×2) → `from mcts.osla_math import compute_ucb_scores`; `from mcts.mcts_joint_osla import OSLATree, SimCarry, SelectCarry` stays pointed at `mcts.mcts_joint_osla` unchanged (check whether `SelectCarry` is actually still imported here or if this was already just `OSLATree, SimCarry` — verify against live file, `SelectCarry` may not be part of the public test surface)
- `TestUCBLogVisitScaling` (4 methods): `from mcts.mcts_joint_osla import compute_ucb_scores` → `from mcts.osla_math import compute_ucb_scores`
- `TestUCBZeroScoreTiebreakEndToEnd`: `from mcts.mcts_joint_osla import _run_single_sim, OSLATree, SimCarry` — stays unchanged, these three did not move
- `TestOSLAPerDepthQuantile` (2 methods): `from mcts.mcts_joint_osla import compute_osla_value_jax` → `from mcts.osla_math import compute_osla_value_jax`
- `TestSampleKActions` (4 methods): `from mcts.mcts_joint_osla import _sample_k_actions` → `from mcts.osla_math import _sample_k_actions`
- `TestRunSingleSimBackup.test_root_backup_value_single_step`: `from mcts.mcts_joint_osla import _run_single_sim, OSLATree, SimCarry, _sample_k_actions` → split into `from mcts.mcts_joint_osla import _run_single_sim, OSLATree, SimCarry` and `from mcts.osla_math import _sample_k_actions`
- `TestComputeOslaValueJax.test_matches_python_version`: `from mcts.mcts_joint_osla import compute_osla_value, compute_osla_value_jax` → `from mcts.osla_math import compute_osla_value, compute_osla_value_jax`
- `TestComputeOslaValueJax.test_partial_fill`: `from mcts.mcts_joint_osla import compute_osla_value_jax` → `from mcts.osla_math import compute_osla_value_jax`
- `TestPerNodeOsla.test_oslatree_has_node_sim_fields`: `from mcts.mcts_joint_osla import OSLATree` stays unchanged
- `TestRootRoundRobin.test_first_k_simulations_visit_each_root_child_once`: `from mcts.mcts_joint_osla import _run_single_sim, OSLATree, SimCarry, _sample_k_actions` → split the same way as the `TestRunSingleSimBackup` case above

After editing, confirm no import of `_sample_k_actions`, `compute_osla_value`, `compute_osla_value_jax`, or `compute_ucb_scores` still points at `mcts.mcts_joint_osla`:
```bash
grep -n "from mcts.mcts_joint_osla import" tests/test_mcts.py | grep -E "compute_osla_value|compute_ucb_scores|_sample_k_actions"
```
Expected: no output.

- [ ] **Step 4: Run the full test suite**

```bash
pytest tests/ -v
```
Expected: all pass, same count as before this task (pure move, no logic change).

- [ ] **Step 5: Commit**

```bash
git add mcts/osla_math.py mcts/mcts_joint_osla.py tests/test_mcts.py
git commit -m "$(cat <<'EOF'
refactor: extract pure OS(λ) helper functions into mcts/osla_math.py

Moves compute_osla_value(_jax), compute_ucb_scores, _sample_k_actions,
_logits_to_joint_logits, and _joint_policy_to_marginal out of
mcts_joint_osla.py — these are the functions already tested in
isolation (no tree/carry state), so this is the split boundary that's
already proven safe by the existing test suite. Pure move + import
fixup, no logic changes.

Kept _run_single_sim/_osla_plan_single/MCTSJointOSLAPlanner together in
mcts_joint_osla.py rather than the originally-floated 3-way split: its
selection/backup phases are nested closures over shared mutable tree
state, not standalone functions, and refactoring them apart would be a
real rewrite of a file that just passed a correctness audit — not
worth the risk for an organization pass.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Split `actors/learner_actor.py` — extract pure loss/train-step code into `actors/loss.py`

**Files:**
- Create: `actors/loss.py`
- Modify: `actors/learner_actor.py`, `tests/test_learner.py`

**Interfaces:**
- Produces: `actors/loss.py` exports `scale_grad_half`, `_awpo_weight`, `make_train_step(model, optimizer, value_support, reward_support, config)`. `actors/learner_actor.py` imports `make_train_step` from `actors.loss` and keeps only the `LearnerActor` Ray class. `actors/loss.py` has no `ray` import and no Ray-actor code — it's importable/testable standalone.

- [ ] **Step 1: Create `actors/loss.py`**

Move these (from the current, post-Task-4-comment-trim version of `actors/learner_actor.py`) verbatim:
- The `# ─── Gradient scaling utilities ──` section: `scale_grad_half`, `_scale_grad_half_fwd`, `_scale_grad_half_bwd`, and the `scale_grad_half.defvjp(...)` call (current lines ~15-38)
- `_awpo_weight` (current lines ~41-55)
- `make_train_step` (current lines ~58-346)

```python
# actors/loss.py
"""Pure-JAX loss/train-step logic for the learner — no Ray import, so this
module is importable and unit-testable without spinning up a Ray actor."""

import jax as _jax

from config import ExperimentConfig


@_jax.custom_vjp
def scale_grad_half(x):
    """Identity in forward pass; halves gradients in backward pass.

    Used to prevent dynamics-network updates from dominating representation
    gradients over long unrolls (MuZero paper §E, MAZero Appendix).
    """
    return x


def _scale_grad_half_fwd(x):
    return x, ()


def _scale_grad_half_bwd(_, g):
    return (_jax.tree_util.tree_map(lambda gi: gi * 0.5, g),)


scale_grad_half.defvjp(_scale_grad_half_fwd, _scale_grad_half_bwd)


def _awpo_weight(q: '_jax.Array', v_baseline: '_jax.Array', alpha: float) -> '_jax.Array':
    """AWAC-style exponential advantage weight, batch-normalized.

    `v_baseline` is stop-gradiented: the AWPO weight must act as a fixed
    multiplier on the policy loss, not a term the value head can shrink or
    grow to inflate its own weighting.
    """
    v_baseline = _jax.lax.stop_gradient(v_baseline)
    if v_baseline.ndim < q.ndim:
        v_baseline = v_baseline[..., None]
    adv_raw = q - v_baseline
    adv_mean = adv_raw.mean()
    adv_std = adv_raw.std()
    adv_norm = (adv_raw - adv_mean) / (adv_std + 1e-8)
    return _jax.numpy.exp(_jax.numpy.clip(adv_norm / alpha, -5.0, 5.0))


def make_train_step(model, optimizer, value_support, reward_support, config: ExperimentConfig):
    """
    Returns a JIT-compiled training step function.

    Captures model, optimizer, supports, and config in a closure so JIT only
    traces once — no static_argnames needed.

    JAX is imported lazily here; this function is only ever called from
    LearnerActor.__init__ after JAX has already been imported in that process.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from utils.transforms import scalar_to_support, support_to_scalar

    U = config.train.unroll_steps
    value_scale = config.train.value_scale
    consistency_scale = config.train.consistency_scale
    # Clamp horizon to U so range(U+1-k) is always ≥ 1.
    consistency_horizon = min(int(config.train.consistency_horizon), U)
    awpo_alpha = float(config.train.awpo_alpha)  # 0.0 = disabled
    K = config.mcts.num_gumbel_samples   # static: K sampled joint actions per MCTS node
    N = config.train.num_agents          # static: number of agents

    def train_step(params, opt_state, batch, weights, rng_key, ema_params, q_data=None):
        # Pre-compute categorical support targets outside loss_fn so they
        # are constants w.r.t. the gradient — zero gradient flows through them.
        value_target_dist = scalar_to_support(
            batch.value_target.mean(axis=2), value_support
        )   # (B, U+1, Sv)
        reward_target_dist = scalar_to_support(
            batch.reward_target.mean(axis=2), reward_support
        )   # (B, U, Sr)

        def loss_fn(p):
            rng_init, _, rng_unroll = jax.random.split(rng_key, 3)
            unroll_keys = jax.random.split(rng_unroll, U)  # (U, 2)

            # ---- Step 0: initial inference ----
            init_out = model.apply(
                {"params": p}, batch.observation, rngs={"dropout": rng_init}
            )
            hidden = init_out.hidden_state  # (B, N, D)

            # Centralized value → (B,)
            v0_loss = optax.softmax_cross_entropy(
                init_out.value_logits, value_target_dist[:, 0]
            )

            # AWPO root policy loss (paper Eq. 10, 14); no-op mean CE when
            # awpo_alpha=0 (branch eliminated at JIT trace time).
            ce_p0 = optax.softmax_cross_entropy(
                init_out.policy_logits, batch.policy_target[:, 0]
            ).mean(axis=-1)  # (B,)
            if awpo_alpha > 0.0 and q_data is not None:
                # Action-level AWPO: weight each sampled root action by
                # exp((Q_k - V_net) / alpha), batch-normalized (_awpo_weight).
                q_valid       = q_data["all_child_valid"][:, 0]      # (B,) bool — root position
                q_k           = q_data["all_child_q"][:, 0]          # (B, K)
                visits_k      = q_data["all_child_visits"][:, 0]     # (B, K)
                child_actions = q_data["all_child_actions"][:, 0]    # (B, K, N)

                # Current network value prediction as AWAC baseline (not stale MCTS value)
                v_net = support_to_scalar(init_out.value_logits, value_support)  # (B,)
                awpo_w_k = _awpo_weight(q_k, v_net, awpo_alpha)  # (B, K)

                # Per-agent log-probs for each sampled joint action:
                # log_probs[b, n, a] → gather with child_actions[b, k, n]
                log_probs = jax.nn.log_softmax(init_out.policy_logits, axis=-1)  # (B, N, A)
                B_, K_ = q_k.shape
                N_, A_ = log_probs.shape[1], log_probs.shape[2]
                # Expand log_probs to (B, K, N, A) and gather at child_actions
                log_probs_exp = jnp.broadcast_to(
                    log_probs[:, None, :, :], (B_, K_, N_, A_)
                )
                gathered = jnp.take_along_axis(
                    log_probs_exp,
                    child_actions[:, :, :, None],  # (B, K, N, 1)
                    axis=-1,
                ).squeeze(-1)  # (B, K, N)
                joint_log_prob_k = gathered.sum(axis=-1)  # (B, K)

                # Visit-count-weighted AWPO loss
                visit_weights = visits_k / (visits_k.sum(axis=-1, keepdims=True) + 1e-8)
                action_awpo_loss = -(visit_weights * awpo_w_k * joint_log_prob_k).sum(axis=-1)  # (B,)

                # Fall back to plain CE for items where Q-data was not stored
                p0_loss = jnp.where(q_valid, action_awpo_loss, ce_p0)  # (B,)
            elif awpo_alpha > 0.0:
                # No Q-data available (e.g. non-OSLA planner): state-level fallback
                v_mcts = batch.value_target[:, 0].mean(axis=-1)  # (B,)
                v_net = support_to_scalar(init_out.value_logits, value_support)  # (B,)
                awpo_w = _awpo_weight(v_mcts, v_net, awpo_alpha)
                p0_loss = awpo_w * ce_p0  # (B,)
            else:
                p0_loss = ce_p0  # (B,)

            # ---- Steps 1..U: unroll via scan ----
            # Consistency is computed outside the scan so multi-step pairs
            # (h_t, h_{t+k}) for k>1 can reuse the same hidden states.
            if awpo_alpha > 0.0:
                # Build per-step Q-data for the scan (positions 1..U).
                # When q_data is None (non-OSLA planner), pass zeros with all-invalid mask
                # so scan_step always has the same input structure.
                if q_data is not None:
                    step_q_acts  = jnp.moveaxis(q_data["all_child_actions"][:, 1:], 1, 0)  # (U, B, K, N)
                    step_q_q     = jnp.moveaxis(q_data["all_child_q"][:, 1:],       1, 0)  # (U, B, K)
                    step_q_vis   = jnp.moveaxis(q_data["all_child_visits"][:, 1:],  1, 0)  # (U, B, K)
                    step_q_valid = jnp.moveaxis(q_data["all_child_valid"][:, 1:],   1, 0)  # (U, B)
                else:
                    B_ = batch.observation.shape[0]
                    step_q_acts  = jnp.zeros((U, B_, K, N), jnp.int32)
                    step_q_q     = jnp.zeros((U, B_, K),    jnp.float32)
                    step_q_vis   = jnp.zeros((U, B_, K),    jnp.float32)
                    step_q_valid = jnp.zeros((U, B_),       jnp.bool_)

                xs = (
                    jnp.moveaxis(batch.actions, 1, 0),
                    jnp.moveaxis(reward_target_dist, 1, 0),
                    jnp.moveaxis(batch.policy_target[:, 1:], 1, 0),
                    jnp.moveaxis(value_target_dist[:, 1:], 1, 0),
                    unroll_keys,
                    step_q_acts,
                    step_q_q,
                    step_q_vis,
                    step_q_valid,
                )

                def scan_step(hidden, inputs):
                    ai, ri_dist, pi_target, vi_dist, step_key, qd_acts, qd_q, qd_vis, qd_valid = inputs
                    hidden = scale_grad_half(hidden)
                    out = model.apply(
                        {"params": p}, hidden, ai,
                        method=model.recurrent_inference,
                        rngs={"dropout": step_key},
                    )
                    next_hidden = out.hidden_state
                    ri_loss = optax.softmax_cross_entropy(out.reward_logits, ri_dist)
                    vi_loss = optax.softmax_cross_entropy(out.value_logits, vi_dist)

                    # AWPO at this unroll step (mirrors root step computation)
                    ce_loss = optax.softmax_cross_entropy(
                        out.policy_logits, pi_target
                    ).mean(axis=-1)  # (B,) fallback
                    v_net_step = support_to_scalar(out.value_logits, value_support)  # (B,)
                    awpo_w = _awpo_weight(qd_q, v_net_step, awpo_alpha)  # (B, K)
                    B_ = qd_q.shape[0]
                    K_ = qd_q.shape[1]
                    N_ = out.policy_logits.shape[1]
                    A_ = out.policy_logits.shape[2]
                    log_probs = jax.nn.log_softmax(out.policy_logits, axis=-1)      # (B, N, A)
                    lp_exp = jnp.broadcast_to(log_probs[:, None, :, :], (B_, K_, N_, A_))
                    gathered = jnp.take_along_axis(
                        lp_exp, qd_acts[:, :, :, None], axis=-1
                    ).squeeze(-1)                                                    # (B, K, N)
                    jlp_k = gathered.sum(axis=-1)                                   # (B, K)
                    vis_w = qd_vis / (qd_vis.sum(axis=-1, keepdims=True) + 1e-8)   # (B, K)
                    awpo_loss = -(vis_w * awpo_w * jlp_k).sum(axis=-1)             # (B,)
                    pi_loss = jnp.where(qd_valid, awpo_loss, ce_loss)              # (B,)

                    return next_hidden, (ri_loss, pi_loss, vi_loss, next_hidden)

            else:
                xs = (
                    jnp.moveaxis(batch.actions, 1, 0),
                    jnp.moveaxis(reward_target_dist, 1, 0),
                    jnp.moveaxis(batch.policy_target[:, 1:], 1, 0),
                    jnp.moveaxis(value_target_dist[:, 1:], 1, 0),
                    unroll_keys,
                )

                def scan_step(hidden, inputs):
                    ai, ri_dist, pi_target, vi_dist, step_key = inputs
                    hidden = scale_grad_half(hidden)  # half-gradient on hidden states (MuZero paper §E)
                    out = model.apply(
                        {"params": p}, hidden, ai,
                        method=model.recurrent_inference,
                        rngs={"dropout": step_key},
                    )
                    next_hidden = out.hidden_state
                    ri_loss = optax.softmax_cross_entropy(out.reward_logits, ri_dist)
                    pi_loss = optax.softmax_cross_entropy(
                        out.policy_logits, pi_target
                    ).mean(axis=-1)
                    vi_loss = optax.softmax_cross_entropy(out.value_logits, vi_dist)
                    # next_hidden returned as output so the caller can collect all
                    # hidden states for multi-step consistency.
                    return next_hidden, (ri_loss, pi_loss, vi_loss, next_hidden)

            # Transpose to step-major for scan: (B, U, ...) → (U, B, ...)
            _, (ri_losses, pi_losses, vi_losses, scan_hiddens) = jax.lax.scan(
                scan_step, hidden, xs
            )
            # ri_losses, pi_losses, vi_losses: (U, B)
            # scan_hiddens: (U, B, N, D) — h_1 through h_U

            reward_loss = ri_losses.mean(axis=0)
            policy_loss = (p0_loss + pi_losses.sum(axis=0)) / (U + 1)
            value_loss  = (v0_loss + vi_losses.sum(axis=0)) / (U + 1)

            # ---- Multi-step SPR consistency (Schwarzer et al. 2021) ----
            # For each k in 1..consistency_horizon and start t, compare
            # project_online(h_t) vs project_target(h_{t+k}); k=1 is the
            # original single-step loss. XLA CSE means project_online(h_t)
            # is computed once regardless of how many k reference it.
            # all_hiddens[i] = h_i,  shape (U+1, B, N, D)
            all_hiddens = jnp.concatenate([hidden[jnp.newaxis], scan_hiddens], axis=0)
            cons_pairs = []
            for k in range(1, consistency_horizon + 1):
                for t in range(U + 1 - k):
                    h_t  = all_hiddens[t]        # (B, N, D)
                    h_tk = all_hiddens[t + k]    # (B, N, D)
                    online = model.apply(
                        {"params": p}, h_t, method=model.project_online
                    )
                    target = model.apply(
                        {"params": ema_params}, h_tk, method=model.project_target
                    )
                    B_, N_, D_ = online.shape
                    # epsilon=1e-8 prevents 0/0 NaN with near-zero projection norms
                    # (common early in training and with dead-agent zeroed obs).
                    sim = optax.cosine_similarity(
                        online.reshape(B_ * N_, D_),
                        target.reshape(B_ * N_, D_),
                        epsilon=1e-8,
                    ).reshape(B_, N_).mean(axis=-1)  # (B,)
                    cons_pairs.append(-sim)
            consistency_loss = jnp.stack(cons_pairs).mean(axis=0)  # (B,)

            loss = (
                reward_loss
                + policy_loss
                + value_loss * value_scale
                + consistency_loss * consistency_scale
            )
            total_loss = (loss * weights).mean()

            td_error = jnp.abs(
                support_to_scalar(init_out.value_logits, value_support)
                - batch.value_target[:, 0].mean(axis=1)
            )

            policy_probs = jax.nn.softmax(init_out.policy_logits, axis=-1)  # (B, N, A)
            policy_entropy = -jnp.sum(
                policy_probs * jnp.log(policy_probs + 1e-8), axis=-1
            ).mean()  # scalar; uniform over 9 actions = log(9) ≈ 2.197

            metric_scalars = jnp.stack([
                total_loss,
                reward_loss.mean(),
                policy_loss.mean(),
                value_loss.mean(),
                consistency_loss.mean(),
                policy_entropy,
            ])
            return total_loss, (metric_scalars, td_error)

        (_, (metric_scalars, td_error)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        grad_norm = optax.global_norm(grads)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_priorities = td_error + 1e-6

        # Pack all scalars that need D2H transfer into one contiguous array so
        # the host pays for a single PCIe DMA transaction instead of one per scalar.
        # Layout: [total, reward, policy, value, consistency, grad_norm, priorities...]
        transfer_buf = jnp.concatenate([
            metric_scalars,
            grad_norm[jnp.newaxis],
            new_priorities,
        ])

        return new_params, new_opt_state, transfer_buf, new_priorities

    return jax.jit(train_step)
```

**Note for the implementer:** the body above already has Task 4's comment trims applied (compressed AWPO and SPR-consistency comments). If Task 4 hasn't landed yet when you execute this task, run Task 4 first — the tasks in this plan are meant to execute in order.

- [ ] **Step 2: Delete the moved code from `actors/learner_actor.py` and import it instead**

Remove `scale_grad_half`, `_scale_grad_half_fwd`, `_scale_grad_half_bwd`, the `defvjp` call, `_awpo_weight`, and `make_train_step` from `actors/learner_actor.py`. Replace the top-of-file imports:
```python
import dataclasses
import os
import time
import numpy as np

import ray
import jax as _jax

from config import ExperimentConfig
from utils.logging_utils import logger
from utils.obs_norm import ObsRunningNorm
from utils.profiler import Profiler
```
→
```python
import dataclasses
import os
import time
import numpy as np

import ray

from actors.loss import make_train_step
from config import ExperimentConfig
from utils.logging_utils import logger
from utils.obs_norm import ObsRunningNorm
from utils.profiler import Profiler
```
(The `jax as _jax` import is no longer needed in this file — `LearnerActor.__init__` already does its own lazy `import jax` inside the method body.) The rest of `LearnerActor` (the `@ray.remote(num_gpus=1)` class, `__init__`, `_prefetch_batch`, `_train_step`, `run_training_loop`, `train`, `_save_checkpoint`, `get_params`, `get_train_step_count`) stays exactly as-is — it already calls `make_train_step(...)` as a plain function call, which now resolves via the import instead of local definition.

- [ ] **Step 3: Update `tests/test_learner.py` imports**

```python
from actors.learner_actor import scale_grad_half
```
(2 occurrences) →
```python
from actors.loss import scale_grad_half
```

```python
from actors.learner_actor import _awpo_weight
```
(2 occurrences) →
```python
from actors.loss import _awpo_weight
```

- [ ] **Step 4: Run the full test suite**

```bash
pytest tests/ -v
```
Expected: all pass, same count as before (pure move, no logic change).

- [ ] **Step 5: Commit**

```bash
git add actors/loss.py actors/learner_actor.py tests/test_learner.py
git commit -m "$(cat <<'EOF'
refactor: extract make_train_step/scale_grad_half/_awpo_weight into actors/loss.py

Separates the pure-JAX loss/train-step math (no Ray import, independently
testable) from LearnerActor's Ray-actor plumbing (GPU init, checkpointing,
param serving). Pure move + import fixup, no logic changes.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes (for whoever executes this plan)

- Task 1 Step 3 corrects an assumption from the design spec: `mctx` stays in `requirements.txt` because `mcts_joint_osla.py` uses `mctx.RecurrentFnOutput` as a return-type container, even though it doesn't use mctx's search algorithm. Verify this with the grep in that step before touching `requirements.txt` — if a later change removes that usage too, revisit.
- Task 1 Step 6 corrects a second design-spec assumption: `configs/mcts/default.yaml` cannot be deleted (it's inherited by `joint.yaml`/`smax.yaml` via Hydra's `defaults: - default`), only repointed to `planner_mode: joint`.
- Task 8 deviates from the original 3-way file split floated during brainstorming, for the closure-coupling reason explained in that task. This was a deliberate call made during planning after reading the actual file, not an oversight — flagged explicitly in that task and its commit message.
