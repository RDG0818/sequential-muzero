# Portfolio Simplification Pass — Design

## Goal

Turn `sequential-muzero` into a "dead simple, well abstracted" portfolio piece: sparse comments,
no dead code, no oversized files. Supersedes Track 1's narrow scope (`refactor/simplify` branch)
with a wider combined pass, per the roadmap spec's "broader organization" phase
(`2026-07-26-improvement-roadmap-design.md`).

Framing: this repo should read as a faster, cleaner, more "ML-eng-friendly" version of MAZero
(`../MAZero/`, the paper's reference implementation) — not a from-scratch reinvention. Where a
component maps to something in the paper (arXiv 2405.11778) or MAZero's code, comments should
say so concisely (equation/section number) instead of re-deriving the math in prose.

## Non-Goals

- No behavior changes beyond what's explicitly listed in Pass 1.
- No new features, no algorithmic changes, no performance tuning.
- No changes to `model/model.py`'s `DynamicsNetwork` attention — confirmed core to the paper
  (Eq. 4's communication function `e_θ`, Fig. 2) and to MAZero's actual code
  (`config/smac/attention.py` + `config/smac/model.py:120`, 3-layer/8-head). Untouched by this pass.

## Branch / Verification Policy

- New branch `simplify/portfolio-pass` off `updates`. Existing `refactor/simplify` branch/worktree
  (6 commits: helper extraction in `data_actor.py`/`learner_actor.py`/`loop.py`) is superseded —
  left alone, not merged, not deleted without separate confirmation.
- Full test suite (`pytest tests/ -v`) must pass after each pass's commit(s). Three checkpoints:
  Pass 1, Pass 2, Pass 3 — not squashed together.
  Only Pass 1 has a behavior-visible change (default planner switches `default`→`joint`); called
  out explicitly in its commit message. Passes 2 and 3 are behavior-neutral.
- CLAUDE.md updated incrementally alongside each pass so it never goes stale mid-effort.
- No git history rewriting, per existing roadmap policy.

## Pass 1 — Dead Code + Config

**Remove** (blast radius confirmed via grep):
- `mcts/mcts_independent.py`, `mcts/mcts_joint.py` (joint_legacy)
- `mctx` dependency (`requirements.txt`) — only consumer was the two files above
- `model.communicate()` method + its dedicated `TransformerAttentionEncoder` instance in
  `model/model.py` — this repo's own addition (not in the paper or MAZero), a separate
  pre-search root-attention pass only ever consumed by `MCTSIndependentPlanner`. Distinct from
  the DynamicsNetwork attention, which stays (see Non-Goals).
- `MCTSConfig.use_root_communication` field (`config.py`)
- `configs/mcts/default.yaml` (the independent-planner preset)

**Change:**
- `configs/config.yaml`: mcts group default `default` → `joint`

**Update references in:** `eval.py`, `actors/data_actor.py`, `actors/reanalyze_actor.py`,
`mcts/__init__.py`, `tests/test_mcts.py`, CLAUDE.md (remove independent/joint_legacy/communicate()/
use_root_communication docs, mctx from stack table).

**Git hygiene:** add `__pycache__/` and `*.pyc` to `.gitignore`; `git rm --cached` the ~57
currently-tracked compiled files.

## Pass 2 — Comments / Docstrings

Rule: no comments restating WHAT code does (names/types already say that). Keep only WHY —
non-obvious rationale, invariants, workarounds, or algorithm/paper references.

- Single line by default; short paragraphs (2-4 lines) allowed only where the reasoning genuinely
  needs it (e.g. the JAX/Ray GPU-allocation constraint).
- Algorithm comments (OS(λ), AWPO, UCB) cite the paper's equation number instead of re-deriving
  the math, e.g. `# OS(λ): top-(1-ρ) quantile weighted by λ^depth, paper Eq. 5-8` rather than a
  paragraph restating the derivation.
- Module docstrings: 1-2 lines, no restating CLAUDE.md's architecture prose.
- Function/class docstrings: kept only where the signature doesn't already convey it (non-obvious
  return shape, side effects, or a paper/MAZero reference).
- Applies repo-wide: `mcts/`, `model/`, `actors/`, `training/`, `utils/`, `envs/`, `config.py`.

## Pass 3 — Structural Splits

Only where a file is still hard to hold in your head after Passes 1-2. Two candidates identified
now (line counts as of `updates` tip):

1. **`mcts/mcts_joint_osla.py`** (672 lines) → split by concern:
   - `mcts/osla_selection.py` — UCB scoring/selection (`compute_ucb_scores`, `_sample_k_actions`,
     `_best_ucb`, `_select_cond`/`_select_body`)
   - `mcts/osla_backup.py` — OS(λ) value backup (`compute_osla_value_jax`, `compute_osla_value`,
     `backup_step`, `accum_step`, `OSLATree`/`SimCarry`/`SelectCarry`)
   - `mcts/mcts_joint_osla.py` — orchestration only: `_osla_plan_single`, `sim_step`,
     `recurrent_fn_batched`, `MCTSJointOSLAPlanner`, policy-marginal helpers
2. **`actors/learner_actor.py`** (655 lines) → split infra from math:
   - `actors/loss.py` — pure JAX, no Ray import: `make_train_step`, `scale_grad_half` + its VJP,
     `_awpo_weight`. Independently importable/testable without spinning up Ray.
   - `actors/learner_actor.py` — `LearnerActor` Ray class only, imports `make_train_step`.

Constraint: pure move + import fixup, no logic changes. If a proposed split turns out not to
reduce cognitive load once attempted (circular imports, awkward shared state), skip it and note
why rather than forcing it.

## Success Criteria

- All three passes land as separate, test-green commits/commit-groups on `simplify/portfolio-pass`.
- Zero references remain to `mcts_independent`, `mcts_joint` (legacy), `mctx`, `communicate()`,
  or `use_root_communication` anywhere in code, config, or docs.
- No comment in the touched files restates what adjacent code already says.
- `mcts_joint_osla.py` and `learner_actor.py` are split as designed, or the spec is amended inline
  with the reason a given split was skipped.
- CLAUDE.md accurately reflects the post-pass repo (no stale references to removed code).
