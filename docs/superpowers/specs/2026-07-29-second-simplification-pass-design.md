# Second Simplification Pass — Design

## Goal

Follow-up to the first portfolio simplification pass (`2026-07-29-portfolio-simplification-design.md`,
merged as `simplify/portfolio-pass`). That pass removed dead MCTS planner classes and an off-paper
attention pass. This pass goes further: cut unused features, redundant abstractions, and an
over-engineered custom C++/CUDA subsystem that isn't earning its complexity for a portfolio piece.

Full audit (see conversation) surfaced candidates ranked by impact/confidence. This spec captures
everything approved for action.

## Non-Goals

- No algorithmic changes to the OS(λ) MCTS backup, AWPO, or SPR consistency loss.
- No new features.
- `model/model.py`'s `DynamicsNetwork` attention stays untouched (per the first pass's Non-Goals —
  still core to the paper, still not part of this cut list).

## Scope

### 1. Dead code / duplicate removal (no tradeoffs)

- Delete `agent_order` end-to-end: `MCTSPlanOutput` (`mcts/base.py`), `Transition`/`Episode`/
  `process_episode` (`utils/replay_buffer.py`), `ReplayItem` pytree flatten/unflatten, the C++
  struct + bindings (superseded anyway by Section 3), `sample()`/`sample_for_reanalysis()`. It's
  produced as `jnp.arange(N)` and never read past a `_` discard in `reanalyze_actor.py`.
- Replace `mctx.RecurrentFnOutput` usage in `mcts/mcts_joint_osla.py` with a local NamedTuple.
  Drop `mctx` from `requirements.txt`.
- Untrack `build/` and the committed `.so`; add both to `.gitignore`. (Moot once Section 3 deletes
  `csrc/`/`setup.py`'s build step entirely, but do this first regardless.)
- Delete `configs/mcts/smax.yaml` — byte-for-byte duplicate of `configs/mcts/default.yaml`.
- Collapse `planner_mode` config field + the `planner_map` dict duplicated in `data_actor.py`,
  `reanalyze_actor.py`, `eval.py` — import `MCTSJointOSLAPlanner` directly, no dispatch table.
  Fold the `MCTSPlanner` ABC into the one concrete class; delete the abstract-but-unused
  `_recurrent_fn` (raises `NotImplementedError`, never called through the base) and the dead
  `self._recurrent_fn_jit = None` assignment.
- Delete now-unreachable branches left over from the already-deleted non-OSLA planners: the
  `has_q_data`/`None` guards and defaults in `utils/replay_buffer.py` and
  `actors/replay_buffer_actor.py`, and in `actors/loss.py` the `q_data is not None` guard, the
  `elif awpo_alpha > 0.0` "non-OSLA planner" fallback branch, and the zeros branch. The single
  surviving planner always populates these fields, so the alternate branches never execute.
- Fix: MCTS search calls `model.apply(..., rngs={"dropout": ...})` without `deterministic=True` in
  `mcts_joint_osla.py` — dropout is currently active during tree search (`dropout_rate=0.1`
  default). Add `deterministic=True` to both call sites.
- Fix: `mcts/base.py`'s `MCTSPlanOutput.root_value` is typed `float` but holds an array; correct
  the annotation.

### 2. SMAX-only (drop MPE + baselines)

CLAUDE.md already states SMAX 3m is the primary target; MPE and the IPPO/MAPPO baselines have no
test coverage and can't run against SMAX. Commit fully:

- Delete `envs/mpe_env_wrapper.py`, `baselines/` (`networks.py`, `ippo.py`, `mappo.py`),
  `train/ippo.py`, `train/mappo.py`, `configs/baseline/` (`ippo.yaml`, `mappo.yaml`).
- `envs/__init__.py`: remove the `"MPE_"`-prefix routing branch; `make_env_wrapper`/
  `make_vec_env_wrapper` go straight to the SMAX wrappers (still parameterized by env name/agent
  count/max steps, just one backend now).
- `configs/train/default.yaml` and `configs/config.yaml`: default env becomes SMAX 3m. Since
  `smax_3m.yaml` would then be the only meaningful train preset, fold its contents into
  `default.yaml` and delete the split file (one preset, not two that must stay in sync).
- `eval.py`: remove the `is_smac` branch, single SMAX-only path.
- CLAUDE.md: remove the MPE/IPPO/MAPPO sections from Package Layout, Architecture, and Commands;
  remove the env-routing description.

### 3. Replay buffer → cpprb

Replace the custom C++/CUDA extension with the `cpprb` library (pip-installable, C++-backed PER).
The existing pure-Python fallback in `utils/replay_buffer.py` already targets a similar API shape,
which keeps this migration bounded.

- Delete `csrc/` (all of `sum_tree.h`, `pinned_alloc.h`, `replay_buffer.h/.cpp`, `bindings.cpp`),
  `CMakeLists.txt`, the `build_ext` step in `setup.py` (or `setup.py` entirely if nothing else in
  it is needed), the committed `.so`, and both the C++ bindings layer and the pure-Python fallback
  code paths in `utils/replay_buffer.py`.
- `ReplayBufferActor` wraps `cpprb.PrioritizedReplayBuffer` directly. Preserve the same external
  behavior: alpha/beta priority annealing, `add`/`sample`/`update_priorities` call signatures used
  by `data_actor.py`, `learner_actor.py`, `reanalyze_actor.py` stay stable so those callers need
  minimal changes.
- Reanalyze's targeted "overwrite position 0 of a stored sequence" update: implement as a small
  adapter over cpprb's indexed access — this is new glue code, not a rewrite of reanalyze's logic.
- Accept the loss of the CUDA pinned-memory DMA optimization (`jax.device_put()` reverts to the
  normal pageable-memory path). This is a known, explicitly accepted tradeoff — simplicity over
  that specific throughput optimization.
- `requirements.txt`: add `cpprb`, remove `pybind11`.
- CLAUDE.md: remove the "Build C++ replay buffer extension" setup step, update the `ReplayBufferActor`
  description to describe the cpprb wrapper instead of the lock-free/pinned-memory design.
- `tests/test_replay_buffer.py`: rewrite around cpprb's actual behavior (add/sample/priority-update
  correctness, stratified sampling) instead of the current C++-vs-Python dual-path tests.
- Delete `benchmarks/replay_buffer_benchmark.py` — it compared the custom C++ backend against the
  Python fallback, both of which are gone; the comparison is no longer meaningful.

### 4. Cut dead feature flags

Both flags are fully wired but `false`/disabled in every shipped config — built, never turned on.

- Delete `run_training_loop_sync` (`training/loop.py`) and `LearnerActor.train()` (its only
  caller). Delete `TrainConfig.sync`. The async loop (`run_training_loop`) becomes the only path.
- Delete `utils/obs_norm.py` (`ObsRunningNorm`), `ModelConfig.use_obs_normalization`, and the
  `norm_state` half of the param-sync protocol across `learner_actor.py`, `data_actor.py`,
  `reanalyze_actor.py` (the `get_params()` contract simplifies back to just `{"params": ...}`).
  Remove the two obs-norm checkpoint fields from save/restore.
- CLAUDE.md: remove the "Observation Normalization" section, remove `sync`/
  `run_training_loop_sync` references, update the "Param sync protocol" description.
- `tests/`: delete or update any test that exercises these two flags directly (e.g. sync-loop
  tests, obs-norm tests).

### 5. Misc smaller cleanups

- Promote never-varied MCTS-math constants — `pb_c_base`, `pb_c_init`, `value_delta_lb`,
  `dirichlet_alpha`, `dirichlet_fraction` — from `MCTSConfig` fields to module-level constants in
  `mcts/osla_math.py`/`mcts/mcts_joint_osla.py`. Keep `mcts_rho`, `mcts_lambda`, `num_simulations`
  as config fields — these are the paper hyperparameters a user might actually sweep.
- Rename Gumbel-search leftovers (mctx's Gumbel search itself was removed in the first pass, these
  names are now misnomers): `num_gumbel_samples` → `num_sampled_actions`,
  `max_depth_gumbel_search` → `max_search_depth`. Update `MCTSConfig`, all config YAML, all call
  sites, CLAUDE.md.
- `eval.py` currently reconstructs the full optax warmup-cosine schedule solely to shape a
  checkpoint-restore target, duplicating `learner_actor.py`'s schedule construction. Simplify to a
  bare `opt_state` restore target, or extract a shared schedule factory both files call.

## Branch / Verification Policy

- New branch `simplify/second-pass` off `updates`, via isolated worktree (same pattern as the
  first pass).
- Full test suite must pass after each section's commit(s). Five checkpoints (one per section
  above), not squashed together.
- Section 3 (replay buffer) is the highest-risk section — verify end-to-end (an actual short
  training run, not just unit tests) before considering it done, since it changes a load-bearing
  runtime dependency.
- CLAUDE.md updated incrementally alongside each section.
- No git history rewriting, per existing repo policy.

## Success Criteria

- All five sections land as separate, test-green commits/commit-groups on `simplify/second-pass`.
- Zero references remain anywhere (code, config, docs) to: `agent_order`, `mctx`, `planner_mode`,
  `MPE_`/MPE env wrapper, `baselines/`, IPPO/MAPPO, the custom `csrc/` C++ replay buffer, `sync`/
  `run_training_loop_sync`, `use_obs_normalization`/`ObsRunningNorm`/`norm_state`,
  `num_gumbel_samples`/`max_depth_gumbel_search`.
- `python train/muzero.py train=smax_3m model=smax mcts=joint` (or its post-fold equivalent, once
  `smax_3m.yaml` merges into `default.yaml`) runs a short training loop successfully with the new
  cpprb-backed replay buffer.
- CLAUDE.md accurately reflects the post-pass repo — no stale references to anything removed above.
