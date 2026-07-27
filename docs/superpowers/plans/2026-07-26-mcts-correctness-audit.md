# MCTS Correctness Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix confirmed correctness bugs in `mcts/mcts_joint_osla.py` and `actors/learner_actor.py` by porting the relevant formulas from MAZero's reference implementation (`../MAZero/core/mcts/ctree/ctree_sampled/lib/cnode.cpp`), and add regression tests confirming two jaxzero-diagnosed bug classes are (or now are) absent here.

**Architecture:** This is a correctness-fix plan, not a redesign — it touches `mcts/mcts_joint_osla.py`'s UCB selection and OS(λ) value estimation formulas, and `actors/learner_actor.py`'s AWPO loss. Each task follows: reproduce the discrepancy in a small deterministic test → port the correct formula from MAZero's C++ → verify the test passes → run the full suite.

**Tech Stack:** JAX, Flax, chex, pytest. No new dependencies.

## Global Constraints

- All fixes must keep `mcts_joint_osla.py`'s functions traceable under `jax.jit`/`jax.vmap`/`jax.lax.fori_loop` — no Python-level control flow depending on traced values, no dynamic-shape arrays.
- No JAX imports at module top level outside actor files (existing project rule; `mcts_joint_osla.py` already imports JAX at module level, which is fine — it's not a Ray actor file).
- Existing test suite (`pytest tests/ -v`) must stay green after every task.
- MAZero reference values to match: `pb_c_base=19652`, `pb_c_init=1.25`, `tree_value_stat_delta_lb=0.01` (from `../MAZero/core/config.py:78-80`), `mcts_rho=0.25`, `mcts_lambda=0.8` (from `../MAZero/train_smac.sh:9-10` — already matched in `configs/mcts/joint.yaml`/`smax.yaml`, no change needed there).

---

## Background: Confirmed Divergences From MAZero

Read directly from `../MAZero/core/mcts/ctree/ctree_sampled/lib/cnode.cpp`, `cnode.h`, and `../MAZero/core/mcts/ctree/common_lib/utils.cpp`:

1. **UCB formula.** MAZero: `pb_c = log((N_parent + pb_c_base + 1) / pb_c_base) + pb_c_init`, scaled by `sqrt(N_parent) / (1 + N_child)`; the Q-term is `get_qsa(child) - parent.pred_value`, normalized into `[0, 1]` by a running min/max over every Q-baseline-diff seen anywhere in the tree so far (`tools::CMinMaxStats`), clipped to `[0, 1]`. Our `compute_ucb_scores` uses a **fixed** `c_puct=1.25` (no log-visit scaling) and **raw, unnormalized** `child_q` added directly to the prior term. Since our value support spans `[-5, 5]`, the Q-term can be 1-2 orders of magnitude larger or smaller than the prior term depending on where training is at, making the exploration/exploitation balance essentially arbitrary rather than tuned. This is the highest-confidence, highest-impact bug found.
2. **OS(λ) quantile computed per-depth-bucket, not pooled.** MAZero's `tools::SubTreeValueSet` (in `utils.h`/`utils.cpp`) keeps the top-`(1-rho)` fraction **separately for each depth bucket** (`big[depth]`/`small[depth]` multisets, `size_lim = ceil(count[depth] * (1-rho))`), then sums `lambda^depth * (top values at that depth)` across buckets. Our `compute_osla_value`/`compute_osla_value_jax` sort **all** `(value, depth)` pairs together and take the global top-`(1-rho)` fraction, only applying `lambda^depth` weighting after the fact. This mixes sims of different depths (which have different bias/variance — depth-0 sims are raw network value predictions, deeper sims are backed-up through more real tree search) before ranking them against each other, which is not what "OS(λ)" — the algorithm this repo is named after — actually specifies.
3. **Root forces one visit to each of the K sampled children before UCB engages.** MAZero: `if (node->is_root && node->visit_count <= node->num_children) child_index = node->visit_count - 1;` — the first K simulations round-robin through the K sampled root actions unconditionally. Ours picks via UCB from simulation 0, which (with only `num_gumbel_samples=5` and unvisited children scoring `child_q=0`) usually has a similar practical effect but isn't guaranteed, and diverges from the paper's implementation.
4. **Sampled-action importance correction (`beta_hat`/`beta`) is out of scope for this plan.** MAZero samples the K child actions **with replacement** and corrects each child's prior by its empirical duplicate frequency (`prior = pred_prob * beta_hat / beta`). Our `_sample_k_actions` samples **without replacement** when `K < A_N`, which cannot produce duplicates and therefore cannot use this correction as-is — porting it requires a sampling-scheme change (with-replacement + de-duplication into unique children), which is a design decision, not a drop-in fix. Left as a follow-up item (see Out of Scope).

Also confirmed via direct code inspection of this repo (independent of MAZero, matching jaxzero's postmortem bug classes):

5. **AWPO value baseline is not stop-gradiented** (`actors/learner_actor.py:110`, `:144`, `:201`) — `v_net`/`v_net_step` feed into the exponential advantage weight without `jax.lax.stop_gradient`, so gradients from the policy loss leak into the value head through the weighting term. This is the exact bug jaxzero's postmortem fixed (`docs/superpowers/plans/2026-05-05-fix-loss-plateau.md` in `../jaxzero/`, commit `a1e9ae3`). **Confirmed present, needs fixing.**
6. **AWPO std-normalization** — jaxzero's postmortem also found and fixed a missing `std` term in the advantage normalization. Checked here: `actors/learner_actor.py` already divides by `(adv_std + 1e-8)` in all three AWPO branches (lines 115-116, 147-148, 203-205). **Confirmed already correct** — add a regression test so it's asserted, not just observed.
7. **Out-of-episode mask reuse** — jaxzero's postmortem found `game.make_target` reusing the terminal step's mask for windows that unroll past episode end. Checked here: `utils/replay_buffer.py:process_episode` only creates windows via `range(ep_len - unroll_steps)`, so every window's positions are always `< ep_len` by construction — there is no padding/reuse path at all. **Confirmed not applicable** — add a boundary regression test.

---

### Task 1: Fix AWPO value→policy gradient leak

**Files:**
- Modify: `actors/learner_actor.py` (add helper near top of file, replace 3 call sites)
- Test: `tests/test_learner.py`

**Interfaces:**
- Produces: `_awpo_weight(q: jnp.ndarray, v_baseline: jnp.ndarray, alpha: float) -> jnp.ndarray` — module-level helper in `actors/learner_actor.py`. `q` and `v_baseline` broadcast together (`v_baseline` gets a trailing axis added if it has fewer dims than `q`); returns same shape as `q` after broadcasting.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_learner.py`:

```python
def test_awpo_weight_stops_gradient_into_baseline():
    """AWPO weight must not backprop into the value baseline. If it did, the
    value head could shrink or grow its own prediction purely to inflate the
    policy-loss weighting it produces, leaking policy-loss gradient into the
    value head instead of acting as a fixed advantage-weighted multiplier."""
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    from actors.learner_actor import _awpo_weight

    q = jnp.array([1.0, 0.0])  # (B,) fixed targets, independent of the param under test

    def weight_sum(v_scale):
        v_baseline = jnp.array([0.5, 0.5]) * v_scale  # depends on v_scale
        return _awpo_weight(q, v_baseline, alpha=1.0).sum()

    grad = jax.grad(weight_sum)(1.0)
    assert grad == 0.0, f"gradient into value baseline should be zero, got {grad}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n mazero pytest tests/test_learner.py::test_awpo_weight_stops_gradient_into_baseline -v
```

Expected: FAIL — `ImportError`/`ModuleNotFoundError`, `_awpo_weight` does not exist yet.

- [ ] **Step 3: Add `_awpo_weight` helper to `actors/learner_actor.py`**

Add near the top of the file, after the imports and before `make_train_step` (or wherever module-level helpers like `scale_grad_half` currently live):

```python
def _awpo_weight(q: jnp.ndarray, v_baseline: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """AWAC-style exponential advantage weight, batch-normalized.

    `v_baseline` is stop-gradiented: the AWPO weight must act as a fixed
    multiplier on the policy loss, not a term the value head can shrink or
    grow to inflate its own weighting.
    """
    v_baseline = jax.lax.stop_gradient(v_baseline)
    if v_baseline.ndim < q.ndim:
        v_baseline = v_baseline[..., None]
    adv_raw = q - v_baseline
    adv_mean = adv_raw.mean()
    adv_std = adv_raw.std()
    adv_norm = (adv_raw - adv_mean) / (adv_std + 1e-8)
    return jnp.exp(jnp.clip(adv_norm / alpha, -5.0, 5.0))
```

- [ ] **Step 4: Replace the root Q-data branch (around `actors/learner_actor.py:109-117`)**

Replace:

```python
                # Current network value prediction as AWAC baseline (not stale MCTS value)
                v_net = support_to_scalar(init_out.value_logits, value_support)  # (B,)
                action_adv_raw = q_k - v_net[:, None]  # (B, K)
                # Batch-normalize so exp weights have non-trivial variance regardless
                # of absolute Q/V scale (matches MAZero advantage normalization)
                adv_mean = action_adv_raw.mean()
                adv_std = action_adv_raw.std()
                action_adv_norm = (action_adv_raw - adv_mean) / (adv_std + 1e-8)
                awpo_w_k = jnp.exp(jnp.clip(action_adv_norm / awpo_alpha, -5.0, 5.0))  # (B, K)
```

With:

```python
                # Current network value prediction as AWAC baseline (not stale MCTS value)
                v_net = support_to_scalar(init_out.value_logits, value_support)  # (B,)
                awpo_w_k = _awpo_weight(q_k, v_net, awpo_alpha)  # (B, K)
```

- [ ] **Step 5: Replace the root fallback branch (around `actors/learner_actor.py:142-149`)**

Replace:

```python
                v_mcts = batch.value_target[:, 0].mean(axis=-1)  # (B,)
                v_net = support_to_scalar(init_out.value_logits, value_support)  # (B,)
                advantage = v_mcts - v_net
                adv_mean = advantage.mean()
                adv_std = advantage.std()
                advantage_norm = (advantage - adv_mean) / (adv_std + 1e-8)
                awpo_w = jnp.exp(jnp.clip(advantage_norm / awpo_alpha, -5.0, 5.0))
```

With:

```python
                v_mcts = batch.value_target[:, 0].mean(axis=-1)  # (B,)
                v_net = support_to_scalar(init_out.value_logits, value_support)  # (B,)
                awpo_w = _awpo_weight(v_mcts, v_net, awpo_alpha)
```

- [ ] **Step 6: Replace the per-unroll-step branch inside `scan_step` (around `actors/learner_actor.py:201-206`)**

Replace:

```python
                    v_net_step = support_to_scalar(out.value_logits, value_support)  # (B,)
                    adv_raw  = qd_q - v_net_step[:, None]                            # (B, K)
                    adv_mean = adv_raw.mean()
                    adv_std  = adv_raw.std()
                    adv_norm = (adv_raw - adv_mean) / (adv_std + 1e-8)              # (B, K)
                    awpo_w   = jnp.exp(jnp.clip(adv_norm / awpo_alpha, -5.0, 5.0)) # (B, K)
```

With:

```python
                    v_net_step = support_to_scalar(out.value_logits, value_support)  # (B,)
                    awpo_w = _awpo_weight(qd_q, v_net_step, awpo_alpha)  # (B, K)
```

- [ ] **Step 7: Run the new test and the full test file**

```bash
conda run -n mazero pytest tests/test_learner.py -v
```

Expected: all pass, including `test_awpo_weight_stops_gradient_into_baseline`.

- [ ] **Step 8: Commit**

```bash
git add actors/learner_actor.py tests/test_learner.py
git commit -m "fix: stop-gradient AWPO value baseline to prevent policy-loss leakage into value head"
```

---

### Task 2: Regression test confirming AWPO std-normalization is scale-invariant

**Files:**
- Test: `tests/test_learner.py`

- [ ] **Step 1: Write the test (expected to pass immediately)**

Add to `tests/test_learner.py`:

```python
def test_awpo_weight_is_scale_invariant():
    """Std-normalization must make the weight distribution insensitive to the
    absolute scale of Q/V. This is the exact bug jaxzero's postmortem found
    and fixed (missing std term made near-uniform Q collapse to near-uniform
    weights regardless of relative structure) — this test confirms
    sequential-muzero's `_awpo_weight` already normalizes correctly."""
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    from actors.learner_actor import _awpo_weight

    v_baseline = jnp.zeros(1)
    small_q = jnp.array([[0.1, 0.2, 0.1, 0.2]])   # std ~ 0.05
    large_q = jnp.array([[1.0, 2.0, 1.0, 2.0]])   # std ~ 0.5, same relative shape

    w_small = _awpo_weight(small_q, v_baseline, alpha=3.0)
    w_large = _awpo_weight(large_q, v_baseline, alpha=3.0)

    assert jnp.allclose(w_small, w_large, atol=1e-4), (
        f"std-normalized weights should be scale-invariant: {w_small} vs {w_large}"
    )
```

- [ ] **Step 2: Run it**

```bash
conda run -n mazero pytest tests/test_learner.py::test_awpo_weight_is_scale_invariant -v
```

Expected: PASS immediately (this documents existing-correct behavior via the `_awpo_weight` helper introduced in Task 1; no implementation change needed here).

- [ ] **Step 3: Commit**

```bash
git add tests/test_learner.py
git commit -m "test: confirm AWPO advantage weighting is scale-invariant (std-normalization already correct)"
```

---

### Task 3: Regression test confirming `process_episode` never reads past episode end

**Files:**
- Test: `tests/test_replay_buffer.py`

- [ ] **Step 1: Write the test (expected to pass immediately)**

Add to `tests/test_replay_buffer.py`:

```python
def test_process_episode_never_indexes_past_episode_end():
    """process_episode's sliding window must never read past the real episode
    end and silently reuse the terminal step's data for phantom future steps
    — the bug jaxzero's postmortem found in its own `make_target`
    (`docs/superpowers/plans/2026-05-05-fix-loss-plateau.md` in ../jaxzero/).
    Confirms this repo's process_episode avoids the bug class by
    construction: it only emits windows fully inside the episode, and
    returns none at all for episodes too short to fit one.
    """
    from utils.replay_buffer import Transition, Episode, process_episode
    import numpy as np

    N, obs_size, A = 2, 4, 3
    unroll_steps, n_step, discount = 5, 3, 0.99

    def make_transition(step: int) -> Transition:
        return Transition(
            observation=np.full((N, obs_size), step, dtype=np.float32),
            action=np.zeros(N, dtype=np.int32),
            reward=1.0,
            done=False,
            policy_target=np.ones((N, A), dtype=np.float32) / A,
            value_target=0.5,
            agent_order=np.arange(N),
        )

    # Episode shorter than unroll_steps: must produce zero items, never a
    # padded/truncated one.
    short_episode = Episode()
    for t in range(3):
        short_episode.add_step(make_transition(t))
    assert process_episode(short_episode, unroll_steps, n_step, discount, N) == []

    # Episode longer than unroll_steps: every returned item's window must be
    # fully inside [0, ep_len).
    ep_len = 9
    long_episode = Episode()
    for t in range(ep_len):
        long_episode.add_step(make_transition(t))

    items = process_episode(long_episode, unroll_steps, n_step, discount, N)
    assert len(items) == ep_len - unroll_steps
    for i in range(len(items)):
        assert i + unroll_steps < ep_len, (
            f"item {i}'s window end {i + unroll_steps} reaches/exceeds ep_len {ep_len}"
        )
```

- [ ] **Step 2: Run it**

```bash
conda run -n mazero pytest tests/test_replay_buffer.py::test_process_episode_never_indexes_past_episode_end -v
```

Expected: PASS immediately (documents existing-correct behavior).

- [ ] **Step 3: Commit**

```bash
git add tests/test_replay_buffer.py
git commit -m "test: confirm process_episode never windows past episode end (jaxzero mask-reuse bug class not applicable)"
```

---

### Task 4: Fix OS(λ) to compute the top-quantile per depth bucket, not pooled

**Files:**
- Modify: `mcts/mcts_joint_osla.py:16-74` (`compute_osla_value`, `compute_osla_value_jax`)
- Modify: `mcts/mcts_joint_osla.py` (all call sites of both functions — add `max_depth` argument)
- Test: `tests/test_mcts.py`

**Interfaces:**
- Produces: `compute_osla_value_jax(sim_values, sim_depths, n_visits, rho, lam, max_depth) -> chex.Array` — same as before but with a new required `max_depth: int` (static) argument giving the number of depth buckets to scan (`0..max_depth-1`).
- Produces: `compute_osla_value(sim_depths, sim_values, rho, lam, max_depth) -> chex.Array` — same signature change.
- Consumes by: `_best_ucb` (inside `_run_single_sim`, calls `compute_osla_value_jax` per child), `_osla_plan_single` (calls `compute_osla_value` for the root value and `compute_osla_value_jax` for root child Q-values).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mcts.py` (near the existing `compute_osla_value` tests, e.g. after `TestOSLAHelpers` or alongside it):

```python
class TestOSLAPerDepthQuantile:

    def test_top_quantile_selected_within_depth_bucket_not_pooled(self):
        """Two sims at depth 0 with small values, one sim at depth 3 with a
        huge value. Pooled-across-depths ranking (the old behavior) would
        let the depth-3 outlier dominate the global top-quantile regardless
        of its own bucket size. Per-depth-bucket ranking (MAZero's
        SubTreeValueSet) keeps the depth-3 sim's bucket separate: with only
        one sim at depth 3, size_lim=1, so it's fully included in its own
        bucket either way — but a depth-0 value that's LARGER than any
        depth-0 peer must still be selected as "top" for depth 0, even
        if it's smaller than the depth-3 outlier. This test uses 4 sims
        at depth 0 (values 1, 2, 3, 4) with rho=0.75 (keep top 25% = top 1
        of 4) and 1 sim at depth 5 (value 100). Expected: only the top
        depth-0 value (4) and the depth-5 value (100) are selected.
        weighted_sum = lam**0 * 4 + lam**5 * 100
        tot_weight   = lam**0 * 1 + lam**5 * 1
        """
        from mcts.mcts_joint_osla import compute_osla_value_jax

        lam = 0.8
        sim_values = jnp.array([1.0, 2.0, 3.0, 4.0, 100.0, 0.0, 0.0, 0.0])
        sim_depths = jnp.array([0,   0,   0,   0,   5,     0,   0,   0], dtype=jnp.int32)
        n_visits = jnp.array(5)  # first 5 entries are real, rest is padding

        result = compute_osla_value_jax(
            sim_values, sim_depths, n_visits, rho=0.75, lam=lam, max_depth=8
        )
        expected_weighted_sum = (lam ** 0) * 4.0 + (lam ** 5) * 100.0
        expected_tot_weight = (lam ** 0) * 1.0 + (lam ** 5) * 1.0
        expected = expected_weighted_sum / expected_tot_weight
        assert jnp.allclose(result, expected, atol=1e-4), (
            f"expected {expected:.4f}, got {float(result):.4f}"
        )

    def test_matches_old_pooled_behavior_when_all_sims_same_depth(self):
        """When every sim is at the same depth, per-depth-bucket ranking and
        pooled ranking are equivalent (there's only one bucket) — sanity
        check that the rewrite doesn't change single-depth behavior."""
        from mcts.mcts_joint_osla import compute_osla_value_jax

        sim_values = jnp.array([1.0, 2.0, 3.0, 4.0, 0.0])
        sim_depths = jnp.zeros(5, dtype=jnp.int32)
        n_visits = jnp.array(4)

        result = compute_osla_value_jax(
            sim_values, sim_depths, n_visits, rho=0.75, lam=0.8, max_depth=4
        )
        # rho=0.75 -> keep top ceil(4*0.25)=1 -> only value 4.0
        assert jnp.allclose(result, 4.0, atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n mazero pytest tests/test_mcts.py::TestOSLAPerDepthQuantile -v
```

Expected: FAIL — `compute_osla_value_jax()` doesn't accept a `max_depth` keyword yet (`TypeError`), or (if called positionally without it) produces the old pooled result, which for the first test is `weighted_sum = 4*lam**0 + 100*lam**5` only if pooling happens to also select exactly `{4.0, 100.0}` as top-2-of-5 — pooled top `ceil(5*0.25)=2` of `[1,2,3,4,100]` is `{100, 4}`, same selection here by coincidence, but weighted differently only if depths differ... Actually re-verify: with `n_visits=5` pooled top-2 is also `{100, 4}` — so this specific example's numeric answer might coincidentally match between old and new code. If Step 2 does not fail, tighten the test with values where pooled and per-bucket selection diverge (e.g. add more depth-0 entries so the pooled quantile size differs from the per-bucket quantile size) before proceeding — see Step 3 note.

- [ ] **Step 3: Rewrite `compute_osla_value_jax` and `compute_osla_value` in `mcts/mcts_joint_osla.py`**

Replace both functions (lines 16-74) with:

```python
def compute_osla_value_jax(
    sim_values: chex.Array,   # [max_sims] float32 — padded with 0 after n_visits
    sim_depths: chex.Array,   # [max_sims] int32 — node-to-leaf depth per sim
    n_visits: chex.Array,     # scalar int32 — number of valid entries
    rho: float,
    lam: float,
    max_depth: int,           # static — number of depth buckets to scan (0..max_depth-1)
) -> chex.Array:
    """JAX-native OS(λ), matching MAZero's `SubTreeValueSet`: the top
    (1-rho) quantile is selected SEPARATELY within each depth bucket, then
    buckets are combined with lambda^depth weighting. Pooling all depths
    together before ranking (the previous behavior) lets one depth's
    naturally larger/smaller value scale dominate which sims count as "top",
    instead of comparing sims fairly within their own depth bucket.
    """
    max_sims = sim_values.shape[0]
    valid_mask = jnp.arange(max_sims) < n_visits

    def bucket_contribution(d):
        depth_mask = valid_mask & (sim_depths == d)
        count_d = depth_mask.sum()
        size_lim = jnp.maximum(
            1, jnp.ceil(count_d.astype(jnp.float32) * (1.0 - rho)).astype(jnp.int32)
        )
        masked_vals = jnp.where(depth_mask, sim_values, -jnp.inf)
        order = jnp.argsort(masked_vals)[::-1]
        ranked_mask = depth_mask[order]
        rank = jnp.arange(max_sims)
        include = (rank < size_lim) & ranked_mask
        sorted_vals = sim_values[order]
        weight = lam ** jnp.float32(d)
        has_any = count_d > 0
        bucket_sum = jnp.where(has_any, jnp.where(include, sorted_vals, 0.0).sum() * weight, 0.0)
        bucket_count = jnp.where(has_any, include.sum().astype(jnp.float32) * weight, 0.0)
        return bucket_sum, bucket_count

    bucket_sums, bucket_counts = jax.vmap(bucket_contribution)(jnp.arange(max_depth))
    return bucket_sums.sum() / (bucket_counts.sum() + 1e-8)


def compute_osla_value(
    sim_depths: chex.Array,    # [K] int32 — simulation leaf depths
    sim_values: chex.Array,    # [K] float32 — backup values from root
    rho: float = 0.25,
    lam: float = 0.8,
    max_depth: int = 20,
) -> chex.Array:
    """Non-padded convenience wrapper over `compute_osla_value_jax` for
    callers (e.g. the root value in `_osla_plan_single`) where every entry
    in `sim_values`/`sim_depths` is valid — no `n_visits` padding needed.
    """
    return compute_osla_value_jax(
        sim_values, sim_depths, jnp.array(sim_values.shape[0]), rho, lam, max_depth
    )
```

Note: `compute_osla_value`'s default `rho` changed from `0.75` to `0.25` to match the value actually used everywhere in this codebase's configs (`configs/mcts/joint.yaml`, `configs/mcts/smax.yaml`, and MAZero's `train_smac.sh`) — the old `0.75` default was dead/never-used-in-practice and confusing next to the docstring's own "rho=0.75 keeps top 25%" framing (rho is the *discard* fraction, not the *keep* fraction — the docstring in the original code had this backwards relative to how `configs/mcts/*.yaml` comment it: `"rho=0.25 # keep top 75% of sims"`).

- [ ] **Step 4: Update call sites to pass `max_depth`**

In `_run_single_sim` (the `_best_ucb` closure), find:

```python
        child_osla_v = jax.vmap(
            lambda v, d, n: compute_osla_value_jax(v, d, n, rho, lam)
        )(
```

Replace with (two occurrences in this file — one inside `_best_ucb`, one in `_osla_plan_single`'s root-child-Q computation):

```python
        child_osla_v = jax.vmap(
            lambda v, d, n: compute_osla_value_jax(v, d, n, rho, lam, max_depth + 2)
        )(
```

`max_depth + 2` covers every depth value that can actually occur: `leaf_depth = sc.depth + 1` where `sc.depth` is bounded by the selection `while_loop`'s `sc.depth < max_depth` condition, so `leaf_depth` can reach `max_depth + 1`, and backup records depths `0..leaf_depth` inclusive — i.e. up to `max_depth + 1`, so bucket indices `0..max_depth+1` (that's `max_depth + 2` buckets) must all be covered.

In `_osla_plan_single`, find:

```python
    osla_root_value = compute_osla_value(
        final_carry.sim_depths, final_carry.sim_values, rho=rho, lam=lam
    )
```

Replace with:

```python
    osla_root_value = compute_osla_value(
        final_carry.sim_depths, final_carry.sim_values, rho=rho, lam=lam, max_depth=max_depth + 2
    )
```

- [ ] **Step 5: Run the new tests, verify they pass**

```bash
conda run -n mazero pytest tests/test_mcts.py::TestOSLAPerDepthQuantile -v
```

Expected: PASS. If the sanity example in Step 1 needs tightening because pooled vs. per-bucket coincidentally agree, adjust the test's `sim_values` so the two orderings diverge (e.g. increase the number of depth-0 entries so its `size_lim` differs enough from the pooled `size_lim` that the *set* of selected depth-0 values changes) and re-verify by hand before locking in the assertion.

- [ ] **Step 6: Run the full MCTS test file to check for regressions**

```bash
conda run -n mazero pytest tests/test_mcts.py -v --tb=short
```

Expected: all existing tests pass (existing `compute_osla_value` tests at lines ~471-507 will need their call sites updated to pass `max_depth` — add it as a keyword argument, e.g. `max_depth=8`, matching whatever `K`/array size they use).

- [ ] **Step 7: Commit**

```bash
git add mcts/mcts_joint_osla.py tests/test_mcts.py
git commit -m "fix: compute OS(lambda) top-quantile per depth bucket, matching MAZero's SubTreeValueSet"
```

---

### Task 5: Fix UCB formula — log-visit exploration scaling + min-max Q normalization

**Files:**
- Modify: `config.py` (`MCTSConfig`)
- Modify: `configs/mcts/default.yaml`
- Modify: `mcts/mcts_joint_osla.py` (`compute_ucb_scores`, `OSLATree`, `SimCarry`, `_run_single_sim`, `_osla_plan_single`, `MCTSJointOSLAPlanner.__init__`/`_plan_loop`)
- Test: `tests/test_mcts.py`

**Interfaces:**
- Produces: `compute_ucb_scores(child_q_baseline_diff, child_visits, prior_probs, parent_visits, qmin, qmax, pb_c_base, pb_c_init, value_delta_lb) -> chex.Array` — replaces the old `(child_q, child_visits, prior_probs, parent_visits, c_puct)` signature.
- Produces: `OSLATree.pred_value: chex.Array  # [max_nodes] float32` — new field, the raw network value prediction at each node (distinct from its OS(λ)-refined `value()`).
- Produces: `SimCarry.qmin: chex.Array  # [] float32`, `SimCarry.qmax: chex.Array  # [] float32` — running min/max of Q-baseline-diffs seen anywhere in this single-environment tree so far.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mcts.py`, replacing/extending `TestOSLAHelpers`:

```python
class TestUCBLogVisitScaling:

    def test_pb_c_grows_with_parent_visits(self):
        """The prior-exploration term must grow with log(parent_visits), not
        stay fixed — matches MuZero/MAZero's pb_c formula, not a constant
        c_puct."""
        from mcts.mcts_joint_osla import compute_ucb_scores

        child_visits = jnp.zeros(3)
        prior_probs = jnp.ones(3) / 3
        qmin = jnp.array(0.0)
        qmax = jnp.array(0.0)  # no stats yet (qmax <= qmin)

        ucb_low_n = compute_ucb_scores(
            jnp.zeros(3), child_visits, prior_probs, parent_visits=jnp.array(1.0),
            qmin=qmin, qmax=qmax, pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01,
        )
        ucb_high_n = compute_ucb_scores(
            jnp.zeros(3), child_visits, prior_probs, parent_visits=jnp.array(1000.0),
            qmin=qmin, qmax=qmax, pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01,
        )
        # All children still unvisited in both cases, so the difference is
        # purely the pb_c(N_parent) prior-exploration scaling.
        assert jnp.all(ucb_high_n > ucb_low_n), (
            "prior-exploration term should grow with parent visit count"
        )

    def test_value_term_normalized_into_unit_range(self):
        """With running qmin/qmax stats present, the Q-baseline-diff term
        must be clipped into [0, 1] regardless of its raw scale."""
        from mcts.mcts_joint_osla import compute_ucb_scores

        # Raw Q-baseline diffs of huge magnitude (as if value support were
        # [-5, 5] and Q-values routinely differ by several units) must not
        # swamp the prior term once normalized.
        child_q_diff = jnp.array([-5.0, 0.0, 5.0])
        child_visits = jnp.array([1.0, 1.0, 1.0])
        prior_probs = jnp.ones(3) / 3
        ucb = compute_ucb_scores(
            child_q_diff, child_visits, prior_probs, parent_visits=jnp.array(3.0),
            qmin=jnp.array(-5.0), qmax=jnp.array(5.0),
            pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01,
        )
        # value_score for each child should be (diff - qmin) / (qmax - qmin),
        # i.e. 0.0, 0.5, 1.0 respectively, plus an identical prior term.
        value_scores = ucb - ucb[1] + 0.5  # back out value_score using the middle child as anchor
        assert jnp.allclose(value_scores, jnp.array([0.0, 0.5, 1.0]), atol=1e-3)

    def test_unvisited_children_get_zero_value_score(self):
        """Matches MAZero: `if (child->visit_count == 0) value_score = 0`."""
        from mcts.mcts_joint_osla import compute_ucb_scores

        child_q_diff = jnp.array([100.0, 100.0])  # would be huge if not masked
        child_visits = jnp.array([0.0, 0.0])
        prior_probs = jnp.array([0.5, 0.5])
        ucb = compute_ucb_scores(
            child_q_diff, child_visits, prior_probs, parent_visits=jnp.array(1.0),
            qmin=jnp.array(-1.0), qmax=jnp.array(1.0),
            pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01,
        )
        pb_c = jnp.log((1.0 + 19652.0 + 1.0) / 19652.0) + 1.25
        pb_c = pb_c * jnp.sqrt(1.0) / (1.0 + 0.0)
        expected_prior_score = pb_c * 0.5
        assert jnp.allclose(ucb, expected_prior_score, atol=1e-3), (
            "unvisited children should score purely on the prior term (value_score=0)"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n mazero pytest tests/test_mcts.py::TestUCBLogVisitScaling -v
```

Expected: FAIL — `compute_ucb_scores()` doesn't accept `qmin`/`qmax`/`pb_c_base`/`pb_c_init`/`value_delta_lb` yet (`TypeError`).

- [ ] **Step 3: Add `pb_c_base`, `pb_c_init`, `value_delta_lb` to `MCTSConfig`**

In `config.py`, in the `MCTSConfig` dataclass (after `mcts_lambda`):

```python
    mcts_rho: float = 0.75    # OS(λ): top quantile fraction to keep (1-rho is discarded)
    mcts_lambda: float = 0.8  # OS(λ): depth discount weight (lambda^depth)
    pb_c_base: float = 19652.0   # UCB: log-visit exploration scaling (MuZero/MAZero standard)
    pb_c_init: float = 1.25     # UCB: base exploration constant (MuZero/MAZero standard)
    value_delta_lb: float = 0.01  # UCB: min-max Q-normalization floor, prevents divide-by-~0
```

In `configs/mcts/default.yaml`, append:

```yaml
pb_c_base: 19652.0
pb_c_init: 1.25
value_delta_lb: 0.01
```

- [ ] **Step 4: Rewrite `compute_ucb_scores` in `mcts/mcts_joint_osla.py`**

Replace:

```python
def compute_ucb_scores(
    child_q:       chex.Array,  # [K] float32 — mean Q-value per child
    child_visits:  chex.Array,  # [K] float32 — visit counts (0 = unvisited)
    prior_probs:   chex.Array,  # [K] float32 — prior probabilities
    parent_visits: chex.Array,  # [] float32 — total visits at parent
    c_puct: float = 1.25,
) -> chex.Array:
    """PUCT formula: Q(a) + c_puct * P(a) * sqrt(N_parent + 1) / (1 + N(a))."""
    exploration = c_puct * prior_probs * jnp.sqrt(parent_visits + 1.0) / (1.0 + child_visits)
    return child_q + exploration
```

With:

```python
def compute_ucb_scores(
    child_q_baseline_diff: chex.Array,  # [K] float32 — get_qsa(child) - parent.pred_value
    child_visits:  chex.Array,  # [K] float32 — visit counts (0 = unvisited)
    prior_probs:   chex.Array,  # [K] float32 — prior probabilities
    parent_visits: chex.Array,  # [] float32 — total visits at parent BEFORE this simulation
    qmin: chex.Array,           # [] float32 — running min Q-baseline-diff seen in this tree
    qmax: chex.Array,           # [] float32 — running max Q-baseline-diff seen in this tree
    pb_c_base: float,
    pb_c_init: float,
    value_delta_lb: float,
) -> chex.Array:
    """PUCT with MuZero/MAZero-style log-visit exploration scaling and
    min-max Q normalization (matches `CTree::ucb_score` in MAZero's
    cnode.cpp): the prior-exploration term grows with log(parent_visits)
    instead of a fixed constant, and the Q-value term is normalized into
    [0, 1] by the running min/max seen anywhere in the tree so far, so it's
    comparable in scale to the prior term regardless of the value support's
    absolute range.
    """
    pb_c = jnp.log((parent_visits + pb_c_base + 1.0) / pb_c_base) + pb_c_init
    pb_c = pb_c * jnp.sqrt(parent_visits) / (1.0 + child_visits)
    prior_score = pb_c * prior_probs

    has_stats = qmax > qmin
    delta = jnp.maximum(value_delta_lb, qmax - qmin)
    normalized = jnp.where(has_stats, (child_q_baseline_diff - qmin) / delta, child_q_baseline_diff)
    value_score = jnp.where(child_visits > 0, jnp.clip(normalized, 0.0, 1.0), 0.0)

    return prior_score + value_score
```

- [ ] **Step 5: Run `TestUCBLogVisitScaling` tests, verify they pass**

```bash
conda run -n mazero pytest tests/test_mcts.py::TestUCBLogVisitScaling -v
```

Expected: PASS.

- [ ] **Step 6: Update `tests/test_mcts.py`'s existing `compute_ucb_scores` tests to the new signature**

The existing `TestOSLAHelpers.test_compute_ucb_prefers_unvisited` and `test_compute_ucb_shape` (lines ~514-533) call the old 5-arg signature. Update them:

```python
    def test_compute_ucb_prefers_unvisited(self):
        """Unvisited children (visit_count=0) should have higher UCB than visited ones."""
        from mcts.mcts_joint_osla import compute_ucb_scores
        child_visits = jnp.array([5.0, 0.0, 3.0, 0.0])
        child_q_diff = jnp.array([0.5, 0.0, 0.3, 0.0])
        prior_probs = jnp.array([0.25, 0.25, 0.25, 0.25])
        parent_visits = jnp.array(8.0)
        ucb = compute_ucb_scores(
            child_q_diff, child_visits, prior_probs, parent_visits,
            qmin=jnp.array(0.0), qmax=jnp.array(1.0),
            pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01,
        )
        assert ucb[1] > ucb[0]
        assert ucb[3] > ucb[2]

    def test_compute_ucb_shape(self):
        from mcts.mcts_joint_osla import compute_ucb_scores
        K = 10
        ucb = compute_ucb_scores(
            jnp.zeros(K), jnp.zeros(K), jnp.ones(K) / K, jnp.array(1.0),
            qmin=jnp.array(0.0), qmax=jnp.array(1.0),
            pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01,
        )
        assert ucb.shape == (K,)
```

- [ ] **Step 7: Add `pred_value` to `OSLATree`, `qmin`/`qmax` to `SimCarry`**

In `mcts/mcts_joint_osla.py`, in the `OSLATree` dataclass, add a field after `reward`:

```python
    reward:           chex.Array  # [max_nodes] float32 — reward to enter this node
    pred_value:       chex.Array  # [max_nodes] float32 — raw network value prediction at expansion
```

In `SimCarry`, add two fields after `sim_values`:

```python
    sim_values: chex.Array   # [num_simulations] float32
    qmin:       chex.Array   # [] float32 — running min Q-baseline-diff seen in this tree
    qmax:       chex.Array   # [] float32 — running max Q-baseline-diff seen in this tree
```

- [ ] **Step 8: Populate `pred_value` at root init and at expansion in `_run_single_sim`**

In `_osla_plan_single`, where `tree = OSLATree(...)` is constructed, compute the root's raw predicted value and add the field:

```python
    root_pred_value = utils.support_to_scalar(init_out.value_logits, value_support)
    ...
    tree = OSLATree(
        visit_counts=jnp.zeros(max_nodes, jnp.int32).at[0].set(1),
        value_sum=jnp.zeros(max_nodes, jnp.float32),
        reward=jnp.zeros(max_nodes, jnp.float32),
        pred_value=jnp.zeros(max_nodes, jnp.float32).at[0].set(root_pred_value),
        embedding=jnp.zeros((max_nodes, N, D), jnp.float32).at[0].set(root_embedding),
        ...
    )
```

And where `init_carry = SimCarry(...)` is constructed:

```python
    init_carry = SimCarry(
        tree=tree,
        next_free=jnp.array(1, jnp.int32),
        rng=rng,
        sim_depths=jnp.zeros(num_simulations, jnp.int32),
        sim_values=jnp.zeros(num_simulations, jnp.float32),
        qmin=jnp.array(1e9, jnp.float32),
        qmax=jnp.array(-1e9, jnp.float32),
    )
```

(`qmin > qmax` initially is the "no stats yet" sentinel state that `compute_ucb_scores`'s `has_stats = qmax > qmin` check treats as false.)

In `_run_single_sim`, where the new node is written into the tree (the `tree = tree.replace(...)` block that sets `reward`, `depth`, `parent`, etc.), add:

```python
    tree = tree.replace(
        embedding=tree.embedding.at[new_node_idx].set(new_embedding),
        reward=tree.reward.at[new_node_idx].set(leaf_reward),
        pred_value=tree.pred_value.at[new_node_idx].set(leaf_value),
        depth=tree.depth.at[new_node_idx].set(leaf_depth),
        ...
    )
```

- [ ] **Step 9: Update `_best_ucb` to use the new UCB signature and thread `qmin`/`qmax`**

Replace the body of `_best_ucb` inside `_run_single_sim`:

```python
    def _best_ucb(node_idx):
        child_node_idxs = tree.child_node_idx[node_idx]   # [K]
        safe_idxs = jnp.maximum(child_node_idxs, 0)
        child_visits = jnp.where(
            child_node_idxs >= 0,
            tree.visit_counts[safe_idxs].astype(jnp.float32),
            0.0,
        )
        child_osla_v = jax.vmap(
            lambda v, d, n: compute_osla_value_jax(v, d, n, rho, lam, max_depth + 2)
        )(
            tree.node_sim_values[safe_idxs],
            tree.node_sim_depths[safe_idxs],
            jnp.where(child_node_idxs >= 0, tree.visit_counts[safe_idxs],
                      jnp.zeros(K, jnp.int32)),
        )  # [K]
        child_qsa = jnp.where(
            child_node_idxs >= 0,
            tree.reward[safe_idxs] + gamma * child_osla_v,
            0.0,
        )  # [K]
        parent_pred_value = tree.pred_value[node_idx]
        child_q_baseline_diff = child_qsa - parent_pred_value
        prior_probs = tree.child_prior_prob[node_idx]
        parent_visits = jnp.maximum(tree.visit_counts[node_idx].astype(jnp.float32) - 1.0, 0.0)
        ucb = compute_ucb_scores(
            child_q_baseline_diff, child_visits, prior_probs, parent_visits,
            carry.qmin, carry.qmax, pb_c_base, pb_c_init, value_delta_lb,
        )
        ucb_choice = jnp.argmax(ucb).astype(jnp.int32)

        is_root = node_idx == 0
        root_visits = tree.visit_counts[0]
        round_robin_active = is_root & (root_visits <= K)
        round_robin_choice = (root_visits - 1).astype(jnp.int32)
        return jnp.where(round_robin_active, round_robin_choice, ucb_choice)
```

(The `round_robin_active`/`round_robin_choice` lines implement Task 6 below — included here since they live in the same function and this plan writes `_best_ucb` once. Skip them for now if executing tasks strictly in order; Task 6 will add exactly these two lines if this step only ports the UCB normalization.)

`_best_ucb` and `_run_single_sim` now need `pb_c_base`, `pb_c_init`, `value_delta_lb` as parameters. Update `_run_single_sim`'s signature:

```python
def _run_single_sim(
    carry: SimCarry,
    sim_idx: chex.Array,
    params,
    recurrent_fn,
    K: int,
    A_N: int,
    max_depth: int,
    gamma: float,
    c_puct: float = 1.25,
    rho: float = 0.75,
    lam: float = 0.8,
    pb_c_base: float = 19652.0,
    pb_c_init: float = 1.25,
    value_delta_lb: float = 0.01,
) -> SimCarry:
```

(`c_puct` stays as an unused-but-harmless parameter for call-site compatibility, or remove it and update the one call site in `_osla_plan_single`'s `sim_step` — prefer removing it since it's dead once `compute_ucb_scores` no longer takes `c_puct`.)

- [ ] **Step 10: Update `backup_step` to maintain `qmin`/`qmax`**

Replace the `backup_step` function and its `fori_loop` call:

```python
    def backup_step(k, bcarry):
        btree, V, qmin, qmax = bcarry
        i = leaf_depth - k
        valid = k <= leaf_depth
        node_i = jnp.where(valid, path_nodes[i], 0)

        write_slot = btree.visit_counts[node_i]

        new_vc = btree.visit_counts.at[node_i].add(jnp.where(valid, 1, 0))
        new_vs = btree.value_sum.at[node_i].add(jnp.where(valid, V, 0.0))

        new_nsv = btree.node_sim_values.at[node_i, write_slot].set(jnp.where(valid, V, 0.0))
        new_nsd = btree.node_sim_depths.at[node_i, write_slot].set(jnp.where(valid, k, 0))

        btree = btree.replace(
            visit_counts=new_vc, value_sum=new_vs,
            node_sim_values=new_nsv, node_sim_depths=new_nsd,
        )

        # Running min/max Q-baseline-diff, for non-root nodes only (matches
        # MAZero: `if (i != 0) minmax_stat.insert(...)`). Uses this
        # simulation's own backed-up value V as the node's Q signal rather
        # than recomputing the full OS(λ) estimate here, which would need
        # an extra O(num_simulations) argsort per backup step — the running
        # bound only needs to stay roughly representative, not exact.
        is_root = i == 0
        parent_node_i = jnp.where(valid & ~is_root, path_nodes[jnp.maximum(i - 1, 0)], 0)
        qsa_diff = path_rewards[jnp.maximum(i, 0)] + gamma * V - btree.pred_value[parent_node_i]
        update_stat = valid & ~is_root
        new_qmin = jnp.where(update_stat, jnp.minimum(qmin, qsa_diff), qmin)
        new_qmax = jnp.where(update_stat, jnp.maximum(qmax, qsa_diff), qmax)

        r_i = path_rewards[jnp.maximum(i, 0)]
        V_new = r_i + gamma * V
        V = jnp.where(valid & (k < leaf_depth), V_new, V)
        return btree, V, new_qmin, new_qmax

    tree, _, final_qmin, final_qmax = jax.lax.fori_loop(
        0, max_depth + 1, backup_step, (tree, leaf_value, carry.qmin, carry.qmax)
    )
```

And update `_run_single_sim`'s return to thread the new stats through:

```python
    return SimCarry(
        tree=tree,
        next_free=carry.next_free + 1,
        rng=rng,
        sim_depths=sim_depths,
        sim_values=sim_values,
        qmin=final_qmin,
        qmax=final_qmax,
    )
```

- [ ] **Step 11: Update `_osla_plan_single`, `MCTSJointOSLAPlanner._plan_loop` to pass the new config values through**

In `_osla_plan_single`'s signature, add `pb_c_base: float`, `pb_c_init: float`, `value_delta_lb: float` parameters, and pass them into the `sim_step`/`functools.partial(_run_single_sim, ...)` call alongside the existing `rho`, `lam`, `gamma`.

In `MCTSJointOSLAPlanner.__init__`, add:

```python
        self.pb_c_base = config.mcts.pb_c_base
        self.pb_c_init = config.mcts.pb_c_init
        self.value_delta_lb = config.mcts.value_delta_lb
```

In `_plan_loop`'s `functools.partial(_osla_plan_single, ...)` call, add:

```python
            pb_c_base=self.pb_c_base,
            pb_c_init=self.pb_c_init,
            value_delta_lb=self.value_delta_lb,
```

- [ ] **Step 12: Run the full MCTS test suite**

```bash
conda run -n mazero pytest tests/test_mcts.py -v --tb=short
```

Expected: all tests pass. Fix any remaining call sites the interpreter flags with `TypeError`/`missing argument` — the `TestOSLAPlanner`/`TestRunSingleSimBackup` fixtures in `tests/test_mcts.py` build `OSLATree`/`SimCarry` instances directly and will need `pred_value`/`qmin`/`qmax` added to match the new dataclass fields (mirror the pattern already used for `node_sim_values`/`node_sim_depths` in `test_dataclasses_are_pytrees` and `test_root_backup_value_single_step`).

- [ ] **Step 13: Commit**

```bash
git add config.py configs/mcts/default.yaml mcts/mcts_joint_osla.py tests/test_mcts.py
git commit -m "fix: UCB uses log-visit pb_c scaling and min-max Q normalization, matching MAZero's ucb_score"
```

---

### Task 6: Root round-robin visits each sampled child once before UCB engages

**Files:**
- Modify: `mcts/mcts_joint_osla.py` (`_best_ucb`, already partially covered by Task 5 Step 9 — this task is the verification + isolated test if Task 5 was executed without the round-robin lines)
- Test: `tests/test_mcts.py`

**Interfaces:**
- No new public interfaces — this is a selection-order change internal to `_run_single_sim`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mcts.py`:

```python
class TestRootRoundRobin:

    def test_first_k_simulations_visit_each_root_child_once(self):
        """MAZero: `if (node->is_root && node->visit_count <= node->num_children)
        child_index = node->visit_count - 1;` — the first K simulations must
        round-robin through the K sampled root actions before UCB selection
        kicks in, regardless of prior/value differences between them."""
        from mcts.mcts_joint_osla import _run_single_sim, OSLATree, SimCarry, _sample_k_actions
        import mctx

        K, A_N, N, D, max_depth, gamma = 4, 25, 2, 8, 3, 0.99

        # Deliberately skewed priors so UCB alone (without round-robin)
        # would pick child 0 every time.
        rng = jax.random.PRNGKey(0)
        skewed_logits = jnp.array([10.0, 0.0, 0.0, 0.0] + [0.0] * (A_N - 4))
        child_actions, child_probs = _sample_k_actions(rng, skewed_logits, K, A_N)

        max_nodes = K + 2
        tree = OSLATree(
            visit_counts=jnp.array([1] + [0] * (max_nodes - 1), jnp.int32),
            value_sum=jnp.zeros(max_nodes),
            reward=jnp.zeros(max_nodes),
            pred_value=jnp.zeros(max_nodes),
            embedding=jnp.zeros((max_nodes, N, D)),
            depth=jnp.zeros(max_nodes, jnp.int32),
            parent=jnp.full(max_nodes, -1, jnp.int32),
            child_actions=jnp.zeros((max_nodes, K), jnp.int32).at[0].set(child_actions),
            child_node_idx=jnp.full((max_nodes, K), -1, jnp.int32),
            child_prior_prob=jnp.zeros((max_nodes, K)).at[0].set(child_probs),
            node_sim_values=jnp.zeros((max_nodes, K + 1)),
            node_sim_depths=jnp.zeros((max_nodes, K + 1), jnp.int32),
        )

        def fake_recurrent_fn(params, rng, flat_action, embedding):
            B = flat_action.shape[0]
            return (
                mctx.RecurrentFnOutput(
                    reward=jnp.zeros((B,)),
                    discount=jnp.ones((B,)),
                    prior_logits=jnp.zeros((B, A_N)),
                    value=jnp.zeros((B,)),
                ),
                embedding,
            )

        carry = SimCarry(
            tree=tree, next_free=jnp.array(1, jnp.int32), rng=jax.random.PRNGKey(1),
            sim_depths=jnp.zeros(K, jnp.int32), sim_values=jnp.zeros(K, jnp.float32),
            qmin=jnp.array(1e9), qmax=jnp.array(-1e9),
        )

        visited_children = []
        for sim_idx in range(K):
            carry = _run_single_sim(
                carry, jnp.array(sim_idx), None, fake_recurrent_fn,
                K, A_N, max_depth, gamma,
                pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01, rho=0.25, lam=0.8,
            )
            # child_node_idx[0] tracks which of the K children have been
            # expanded so far; the newly-expanded one this round is the one
            # that just went from -1 to >= 0.
            expanded = jnp.where(carry.tree.child_node_idx[0] >= 0)[0]
            visited_children.append(int(expanded[-1]) if len(expanded) > len(visited_children) else None)

        distinct_children_visited = {c for c in visited_children if c is not None}
        assert len(distinct_children_visited) == K, (
            f"expected all {K} root children visited within the first {K} sims, "
            f"got {visited_children}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n mazero pytest tests/test_mcts.py::TestRootRoundRobin -v
```

Expected: FAIL — without round-robin, the skewed prior on child 0 makes UCB pick child 0 for most/all of the first K simulations, so `distinct_children_visited` has fewer than `K` entries.

- [ ] **Step 3: Add round-robin to `_best_ucb`**

If Task 5 Step 9 was executed with the round-robin lines included, this step is already done — just verify. Otherwise, in `_best_ucb` (inside `_run_single_sim`), after computing `ucb_choice = jnp.argmax(ucb).astype(jnp.int32)`, add:

```python
        is_root = node_idx == 0
        root_visits = tree.visit_counts[0]
        round_robin_active = is_root & (root_visits <= K)
        round_robin_choice = (root_visits - 1).astype(jnp.int32)
        return jnp.where(round_robin_active, round_robin_choice, ucb_choice)
```

(replacing the bare `return jnp.argmax(ucb).astype(jnp.int32)`).

- [ ] **Step 4: Run test, verify it passes**

```bash
conda run -n mazero pytest tests/test_mcts.py::TestRootRoundRobin -v
```

Expected: PASS.

- [ ] **Step 5: Run the full test suite**

```bash
conda run -n mazero pytest tests/ -v --tb=short
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add mcts/mcts_joint_osla.py tests/test_mcts.py
git commit -m "fix: root round-robins through K sampled children before UCB engages, matching MAZero"
```

---

## Out of Scope (Follow-Up, Not This Plan)

- **Sampled-action importance correction** (`beta_hat`/`beta`, divergence #4 above) — requires switching `_sample_k_actions` from without-replacement to with-replacement sampling plus de-duplication into unique children with empirical frequency weights. This changes the shape/semantics of `OSLATree.child_actions`/`child_prior_prob` (currently always exactly `K` distinct children; would become "however many unique actions were drawn, ≤ K"), which touches the fixed-shape assumptions this whole file is built on. Needs its own design pass, not a drop-in fix.
- **Exact MAZero-style min-max removal** (multiset `insert`/`remove` keyed on each node's live Q-estimate) — this plan uses a monotonic running min/max instead (matches the standard MuZero paper's reference `MinMaxStats`, which is also monotonic-only; MAZero's C++ multiset-with-removal is a refinement beyond the paper spec). Exact parity would require a bounded streaming top-k/removal structure inside a `jax.lax.fori_loop`, which is a meaningfully larger, riskier change for uncertain marginal benefit — revisit only if Phase 3 (real SMAX 3m training run, per the roadmap spec) shows search quality is still off after this plan's fixes.
- **Throughput/perf impact of these changes** — Task 4/5 add per-simulation work (extra `vmap` over depth buckets in `compute_osla_value_jax`, extra tree field reads). Quantify in the roadmap's Phase 2 (throughput check), not here.
- **Unvisited-child value_score normalization** — this repo's `compute_ucb_scores` gives unvisited children a hard `value_score=0`, skipping `minmax_stat.normalize()`/clip entirely (standard MuZero pseudocode). MAZero's `cnode.cpp` also zeroes the pre-normalization score for unvisited children, but still runs it through `normalize()`/clip — with a `[-5,5]` value support and `qmin<0` (the common case), this makes MAZero score unvisited children *higher* than this repo does, biasing this repo's search slightly against exploring genuinely-new children relative to the reference. Found in the whole-branch review after Task 6; left as-is (defensible standard-MuZero behavior, changing it now risks new bugs without a dedicated task) — revisit only if search quality issues trace back to under-exploration.
