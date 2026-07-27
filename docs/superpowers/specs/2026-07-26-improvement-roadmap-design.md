# Improvement Roadmap: sequential-muzero as a Portfolio Piece

## Goal

Turn this repo into a presentable portfolio/resume piece targeting **ML infra / platform engineering** roles, without doing a from-scratch rewrite. Timeline is open-ended (no hard deadline).

## Context / Why Not a Rewrite

Two prior full-rewrite paths were evaluated and rejected:

- **MAZero** (`../MAZero/`) — the original paper codebase (PyTorch + Cython `ctree_sampled`). Confirmed working on real SMAC by the user. Not adopted wholesale (old stack, Cython build step, sync-only, not this project's own work) but now treated as a **cherry-pick source**, not just a read-only oracle — goal is "an improvement on MAZero," so porting correct pieces directly (algorithm formulas, specific fixes) is in scope.
- **jaxzero** (`../jaxzero/`) — a prior full JAX-from-scratch rewrite attempt. Went deep (sync + async Ray training, ctree-driven sampled MCTS, reanalyze, TDD-fixed 3 real bugs: stale terminal-mask reuse in `make_target`, missing std-normalization in AWPO advantage weighting, value→policy gradient leakage through the AWPO baseline). Despite the fixes, its own async training log (`output.txt`) shows the learner GPU-starved (`gpu_frac≈0.2%`) with a single reanalyze actor consuming ~42s per 10 learner steps, and loss essentially flat. Conclusion: a third from-scratch attempt would re-risk the same correctness and throughput failure modes rather than avoid them. jaxzero is a **cherry-pick source** for its diagnosed bugs and design docs (`docs/superpowers/plans/2026-05-05-fix-loss-plateau.md` etc.), not a base to build on.

Given this, and that `sequential-muzero`'s core logic is only ~4850 lines (actors+mcts+model+training+utils+envs), **incremental correctness audit + cleanup** on the existing repo is more time-efficient and lower-risk than rewrite #3.

## Known Risk

The user has explicitly stated low confidence in the correctness of `mcts/mcts_joint_osla.py` (611 lines, custom vmapped JAX implementation of MAZero's OS(λ) backup, the single most-churned file in the repo — 11 commits). The repo has never trained well on SMAX/SMAC. This is the highest-risk, highest-value component to verify, since it invalidates all downstream policy/value targets if wrong.

## Approach: Parallel Tracks (Track 1 + Track 2), Then Sequential Phases

Two isolated git worktrees running independently, merged into `updates` when each is ready. Chosen over correctness-first-only sequencing because Track 1 is low-risk/mostly-already-done, and over infra-first because packaging a system with unverified core correctness first is a weaker story.

### Track 1 — Readability (narrow scope, continue existing work)

- **Location:** existing worktree `.worktrees/simplify`, branch `refactor/simplify`.
- **Scope:** finish what's already started — helper extraction, function splitting, named constants. Specifically continues the pattern already in this branch's 5 commits (`_MetricsWindow` in `loop.py`; `_plan_step`/`_step_environment`/`_process_and_ship_episodes` split in `DataActor`; loss-helper extraction + named EMA factory in `learner_actor.py`).
- **Explicitly NOT in scope for this track:** dead code removal, config consolidation, `.gitignore`/pycache cleanup, dependency changes, directory restructuring. These are deferred to a later "broader organization" phase (see below) because they depend on Track 2's findings — e.g. don't remove `mctx` / `mcts_independent.py` until the audit confirms `MCTSJointOSLAPlanner` is the sole correct path forward.
- **Constraint:** pure refactor. Existing test suite must stay green throughout. No behavior changes.

### Track 2 — MCTS / Training-Loss Correctness Audit

- **Location:** new worktree, branch `audit/mcts-correctness`.
- **Scope:**
  1. Build a small deterministic test scenario (fixed rewards/values, hand-computable expected visit counts and Q-backups) and numerically compare `mcts_joint_osla.py`'s OS(λ) backup against MAZero's `core/mcts/tree_search/mcts_sampled.py` + `ctree_sampled` (the confirmed-working reference).
  2. Check `sequential-muzero` for the same 3 bug classes jaxzero's postmortem diagnosed and fixed elsewhere:
     - Out-of-episode step masks reusing the terminal-step mask instead of zeroing (in `process_episode` / `utils/replay_buffer.py`).
     - Missing std-normalization in the AWPO advantage weighting (in `actors/learner_actor.py`).
     - Value→policy gradient leakage through the AWPO value baseline (missing `stop_gradient`).
  3. Where `mcts_joint_osla.py` or the AWPO loss diverges from MAZero's reference formulas, port the correct version over directly (MAZero is now an approved cherry-pick source, not just an oracle).
  4. Each confirmed bug ships as: failing test that reproduces the discrepancy → fix → passing test, added to `tests/test_mcts.py` / `tests/test_learner.py`. Mirrors the TDD process jaxzero's own fix-plateau plan used successfully.
- **Isolation from Track 1:** touches `mcts/mcts_joint_osla.py` and the loss math inside `actors/learner_actor.py`; Track 1 touches structural/naming aspects of `data_actor.py` and `loop.py`. Expected low conflict; whichever branch merges second rebases onto the other.

## After Both Tracks Merge (Sequential Phases)

These happen one at a time, each gated on the prior phase's outcome — not parallelized, since each depends on what the audit actually found:

1. **Broader organization pass** — dead code removal (informed by audit: is `mcts_independent.py`/`mctx` still needed?), config sprawl consolidation, fix `.gitignore` (57 tracked `__pycache__` files currently — no `*.pyc`/`build/` exclusion), dependency cleanup, module docstrings, directory restructuring if warranted.
2. **Throughput check** — verify sequential-muzero doesn't hit jaxzero's GPU-starvation pattern (single reanalyze actor dominating learner throughput). Already has C++ replay buffer + pinned memory that jaxzero lacked, but unverified under the (possibly now-fixed) correct MCTS.
3. **Real SMAX 3m training run** — see where win-rate actually lands post-fixes. Resolves the earlier open question ("does this need real results, or is correct+documented enough") — was left as "unsure yet" pending audit outcome.
4. **Docker packaging.**
5. **AWS distributed training** — Infrastructure-as-Code (Terraform/CDK) + Ray cluster launch config, validated with one small/cheap proof run (not a full training job). Chosen over "actually run full training" (too costly to require) and "design-only, no AWS spend" (too weak a proof for an infra-focused resume piece).

## Git History Policy

Keep existing history as-is (200+ commits, some informally named like "gpu again" — normal for iterative research work). New commits from this point forward should be clean and descriptive. No history rewriting/squashing planned.

## Success Criteria

- Track 1: existing test suite green, `refactor/simplify` branch's structural cleanups merged, no behavior change.
- Track 2: at least the 3 jaxzero-diagnosed bug classes checked (fixed if present, documented as not-applicable if absent), OS(λ) backup numerically validated against MAZero on a deterministic test case, new regression tests committed.
- Overall: decision made (not deferred further) on whether SMAX 3m needs to hit a real win-rate bar for the portfolio story, based on Phase 3's actual training run.

## Out of Scope (For Now)

- Specific win-rate target number — decided after the post-fix training run.
- AWS run scale/cost ceiling beyond "small proof run."
- Full broader-organization item list — finalized after Track 2 lands, not pre-committed here.
