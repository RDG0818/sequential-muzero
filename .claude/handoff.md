# Handoff: Second Portfolio Simplification Pass

## Goal

Cut unnecessary complexity and unused features from `sequential-muzero` (JAX/Flax
multi-agent MuZero, SMAX 3m target) for portfolio purposes — this is a follow-up
to an earlier "first simplification pass" (already merged prior to this session).
User's framing: "go through and see if there is anything that can be cut that is
adding unneeded complexities across the codebase."

## Current state

**Done and merged.** Branch `simplify/second-pass` was merged into `updates` at
commit `2baa87b`. The worktree at `.worktrees/second-pass` and the branch itself
have both been deleted (cleanup already ran). `updates` is the current branch in
the main repo checkout, HEAD is `2baa87b`, working tree clean except two
pre-existing untracked items unrelated to this work (`.claude/` dir, a stray
`2405.11778v1.pdf`).

Full test suite passes: `conda run -n mazero pytest tests/ -v` → **105 passed**.

LOC: real source (excluding `docs/superpowers/`, `build/`, `.so`, stray `.pdf`)
went from **9124 → 5997 lines** (~34% cut), plus ~270k lines of committed
C++/CUDA build artifacts (`build/`, the `.so`) removed from the tree.

9 plan tasks landed as individual commits on top of the prior session's plan
commit (`d611e5b`), then a final whole-branch review (opus) found 5 more issues,
2 of which were fixed by a follow-up subagent and 2 by the controller directly,
plus a scoped re-review confirming the fix. Full commit list, oldest→newest:

```
e82fad6 refactor: replace custom C++ replay buffer with cpprb
1cc1a3f docs: update CLAUDE.md and README.md for cpprb replay buffer swap
394694d refactor: drop mctx dependency, use a local RecurrentFnOutput NamedTuple
8a84dae refactor: fold MCTSPlanner ABC into MCTSJointOSLAPlanner, drop planner_mode
e359a04 refactor: drop MPE env wrapper and IPPO/MAPPO baselines, SMAX-only
1188b82 refactor: delete the synchronous training loop
47b2011 refactor: delete observation normalization
5f8fa7b refactor: promote never-varied MCTS-math constants out of config
83a759d refactor: rename num_gumbel_samples/max_depth_gumbel_search
0b68406 refactor: extract shared make_optimizer, dedup eval.py vs learner_actor.py
800b86a docs: fix stale AWPO description, dangling MPE/planner_mode/Gumbel refs
2baa87b refactor: remove dead all_child_valid Q-data mask and orphaned locals
```

Key functional facts about the resulting code, for anyone picking this up:

- `utils/replay_buffer.py` — `ReplayBuffer` is now `cpprb.PrioritizedReplayBuffer`-backed
  only, no C++ extension, no Python fallback branch. `ReplayItem`/`Transition` no
  longer carry `agent_order` or `all_child_valid` — all Q-data fields are required
  (no `=None` defaults) and unconditionally populated by the sole planner.
- `mcts/mcts_joint_osla.py` — `MCTSJointOSLAPlanner` is a standalone class (no
  ABC base), owns all its own config extraction, defines its own
  `RecurrentFnOutput` NamedTuple (no `mctx` import anywhere in the repo). Dropout
  fix applied: `deterministic=True` on the `recurrent_inference` call site only
  (the root `__call__` inference has no dropout and no such kwarg).
- `mcts/osla_math.py` — `PB_C_BASE`, `PB_C_INIT`, `VALUE_DELTA_LB`,
  `DIRICHLET_ALPHA`, `DIRICHLET_FRACTION` are now module constants, not
  `MCTSConfig` fields (they were never varied in any shipped YAML).
  `MCTSConfig` still keeps `mcts_rho`, `mcts_lambda`, `num_simulations`,
  `num_sampled_actions` (renamed from `num_gumbel_samples`), `max_search_depth`
  (renamed from `max_depth_gumbel_search`) as real tunables.
- `actors/loss.py` — `make_optimizer(config)` factory (shared by
  `learner_actor.py` and `eval.py`, fixes a latent warmup-clamp inconsistency).
  AWPO in `make_train_step` is action-level only now — no more dead
  `jnp.where(q_valid, ...)` CE-fallback branch when `awpo_alpha > 0`; the
  `awpo_alpha == 0.0` plain-CE branch is untouched and still the correct path
  when AWPO is disabled.
- Repo is SMAX-only: `envs/mpe_env_wrapper.py`, `baselines/`, `train/ippo.py`,
  `train/mappo.py`, `configs/baseline/` are gone. `configs/train/smax_3m.yaml`'s
  contents were folded into `configs/train/default.yaml`, which is now the sole
  train preset. Primary command is now `python train/muzero.py model=smax mcts=joint`.
- `TrainConfig.sync` / `run_training_loop_sync` / `LearnerActor.train()` are gone
  (async loop was the only path ever used).
- `ModelConfig.use_obs_normalization` / `utils/obs_norm.py` / the `norm_state`
  half of `get_params()` are gone — `get_params()` returns just `{"params": ...}`.
- `CLAUDE.md` and `README.md` were updated incrementally per task, plus a final
  pass fixing a stale AWPO description (was documenting the deleted state-level
  formula) and a few dangling MPE/planner_mode/Gumbel-naming references.

## Files in flight

All changes are already committed and merged — nothing is uncommitted. For
reference, files touched across the whole pass (see commit list above for which
commit touched what):

- `CLAUDE.md`, `README.md` — updated incrementally, then a final consistency pass
- `config.py` — removed `planner_mode`, obs-norm, sync, and 5 MCTS-constant fields; renamed 2 fields
- `configs/mcts/default.yaml`, `configs/mcts/joint.yaml` — same field removals/renames
- `configs/mcts/smax.yaml` — deleted (byte-for-byte duplicate)
- `configs/model/default.yaml`, `configs/model/smax.yaml` — obs-norm field removed / stale ref fixed
- `configs/train/default.yaml` — folded `smax_3m.yaml` in, removed `sync: false`
- `configs/train/smax_3m.yaml` — deleted (folded into `default.yaml`)
- `configs/baseline/ippo.yaml`, `configs/baseline/mappo.yaml` — deleted
- `mcts/base.py` — `MCTSPlanner` ABC removed, `MCTSPlanOutput` lost `agent_order`, gained required Q-fields
- `mcts/mcts_joint_osla.py` — absorbed ABC logic, local `RecurrentFnOutput`, dropout fix, constant promotion
- `mcts/osla_math.py` — gained 5 new module constants
- `mcts/__init__.py` — no longer exports `MCTSPlanner`
- `utils/replay_buffer.py` — full rewrite around cpprb, `agent_order`/`all_child_valid` removed
- `utils/obs_norm.py` — deleted
- `actors/data_actor.py`, `actors/reanalyze_actor.py` — planner_map collapsed, norm_state removed
- `actors/replay_buffer_actor.py` — dead branches removed, `_q_valid` sidecar removed
- `actors/loss.py` — gained `make_optimizer`, AWPO dead-branch cleanup, dead `K`/`N` locals removed
- `actors/learner_actor.py` — sync method removed, obs-norm removed, uses shared `make_optimizer`
- `eval.py` — `is_smac` branching removed, uses shared `make_optimizer`
- `envs/__init__.py` — MPE routing removed, SMAX-only
- `envs/mpe_env_wrapper.py` — deleted
- `envs/smax_env_wrapper.py` — stale MPE-comparison docstrings fixed
- `train/muzero.py` — unconditional `run_training_loop` call
- `train/ippo.py`, `train/mappo.py` — deleted
- `training/loop.py`, `training/__init__.py` — sync loop removed
- `baselines/` (whole dir) — deleted
- `csrc/`, `CMakeLists.txt`, `setup.py`, the compiled `.so`, `build/` (87 files) — deleted
- `benchmarks/replay_buffer_benchmark.py` — deleted
- `requirements.txt` — removed `mctx`, `pybind11`; added `cpprb`
- `tests/test_mcts.py`, `tests/test_model.py`, `tests/test_replay_buffer.py` — updated for all of the above

## What changed

(No uncommitted changes remain — everything above is already committed and merged. See commit list in "Current state" for the exact sequence.)

## Failed attempts

- **Running the plan's mandated GPU training smoke test** (`train/muzero.py
  train.num_episodes=300 ...`) — could not run in this sandbox. `nvidia-smi` is
  not present and `jax.devices()` only shows `CpuDevice`; `LearnerActor` is
  hard-decorated `@ray.remote(num_gpus=1)`, so Ray can never schedule it here
  and `DataActor`s hang forever waiting on `get_params()`. This is an
  environment limitation, not a code bug — confirmed by checking for GPU
  hardware directly, not by trial and error on the training script.
  **Workaround used instead:** wrote a standalone CPU-only script
  (`/tmp/.../scratchpad/smoke_cpprb.py`, not committed, scratch only) that
  drives the real MCTS planner + real SMAX env + `process_episode()` +
  the real cpprb-backed `ReplayBufferActor` end-to-end (add/sample/
  update_priorities/sample_for_reanalysis/update_root_q) — this covers
  everything except the GPU-only `LearnerActor` training step itself, and it
  passed. **If you have GPU access, the actual smoke test
  (`python train/muzero.py train.num_episodes=300 train.warmup_episodes=20
  train.checkpoint_dir=/tmp/smoke_ckpt`) should still be run once** since the
  full learner-in-the-loop path was never exercised in this session.
- **Trusting the design plan's stated rationale for keeping `all_child_valid`**
  — the plan (written earlier this session, before implementation) explicitly
  chose to keep this Q-data validity mask on the stated grounds that it "masks
  cold ring-buffer slots." This was flagged in the plan's own Self-Review Notes
  as a conservative choice made under uncertainty. The final whole-branch
  reviewer (opus) empirically disproved the rationale: the ring-buffer write
  pointer and cpprb's sample() always move in lockstep from 0, so a "cold slot"
  can never be sampled — the mask was always `True`, and removing it was safe.
  Fixed in commit `2baa87b`. Lesson: a "keep it, we're not sure" note in a plan
  should be revisited with fresh empirical verification at final review time,
  not treated as permanently settled.

## Known issues / blockers

None outstanding in the code. One thing worth flagging to whoever trains next:

- **The full GPU training loop was never actually run end-to-end this
  session** (see "Failed attempts" above) — only a CPU-only substitute that
  covers the replay-buffer/MCTS/env pipeline but not `LearnerActor`'s
  `_train_step`. Recommend running the real smoke test
  (`python train/muzero.py train.num_episodes=300 train.warmup_episodes=20
  train.checkpoint_dir=/tmp/smoke_ckpt`) on the actual WSL2/GPU machine before
  trusting this branch for a real training run, even though all 105 unit tests
  pass on CPU.
- A stray worktree at `.worktrees/simplify` (branch `refactor/simplify`,
  commit `3efdf4a`) exists in the repo and was left untouched this session —
  it predates this work and wasn't investigated; not part of this handoff's
  scope but worth asking the user about if it's stale.

## Next steps

1. **Run the real GPU smoke test** on the target WSL2 machine:
   `conda run -n mazero python train/muzero.py train.num_episodes=300 train.warmup_episodes=20 train.checkpoint_dir=/tmp/smoke_ckpt`
   — confirm it completes without error and produces a checkpoint. This is the
   one verification step this session couldn't do (no GPU in this sandbox).
2. If that passes, consider whether to `git push` the merged `updates` branch
   to `origin/updates` (not done this session — merge was local only, per the
   user's "Merge to updates locally" choice).
3. Ask the user about the stray `.worktrees/simplify` (`refactor/simplify`
   branch) — unclear if it's abandoned work or something still in progress.
4. If more simplification passes are wanted, `CLAUDE.md`'s "Future Improvements"
   section still lists several `[medium]`/`[research]` items (recurrent
   dynamics, data augmentation, sequential MCTS, factored policy targets) that
   were explicitly out of scope for both simplification passes — those are
   feature ideas, not cleanup, so a different kind of session.
