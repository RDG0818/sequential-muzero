# Portfolio Direction: Week-Scoped Plan

## Context

`sequential-muzero` started as an attempt at a novel research idea (sequential
per-agent MCTS, agents conditioning on prior agents' committed actions). That
idea never panned out and was objectively worse than the original MAZero
(joint) formulation, so it was never built out — `MCTSJointOSLAPlanner` (the
joint planner) is the sole planner in the repo today, and "Sequential MCTS" is
listed only as an unbuilt `[medium]` idea in CLAUDE.md's Future Improvements.

Two full simplification passes (Track 2 MCTS audit, portfolio simplification
pass, second simplification pass) are done and merged into `updates`. LOC is
down ~34% from the pre-cleanup baseline, `mctx` and the custom C++ replay
buffer are gone (replaced by `cpprb`), and the README already frames the repo
correctly: a faster JAX/Flax reimplementation of published research (MAZero),
not a claim of research novelty.

**This spec covers what to do with the repo's last ~week of active work**, to
turn it into a finished portfolio piece for ML infra/platform engineering
roles.

## Goals

- Keep the repo simple and readable — this is the #1 constraint on every
  decision below.
- Ship a **meaningful, honest engineering contribution**: the ~300× async
  throughput story (JaxMARL + Ray actor-learner + prioritized replay), backed
  by real, freshly-measured numbers and a real training run's results — not
  just architecture description.
- Do the "standard infra stuff" that ML infra/platform roles expect to see:
  a GPU-enabled Dockerfile, and an AWS IaC sketch showing how this Ray
  actor-learner architecture would scale across machines.
- If a genuinely cheap, well-cited research addition surfaces, include it —
  but don't force one in for its own sake, and don't let it gate the rest.

## Non-goals

- Reviving the "sequential MCTS" research idea — confirmed dead, worse than
  joint, not worth resurrecting for a portfolio piece.
- Any full world-model architecture rewrite (transformer-tokenized dynamics,
  entity-transformer generalist models, etc.) — real 2024–2025 MARL research
  (MAGENTA, "Decentralized Transformers with Centralized Aggregation," MATWM)
  moved in this direction, but all of it is a rewrite, not a drop-in, and is
  out of scope for a week.
- CI/lint/pre-commit setup — explicitly deprioritized by the user; not
  important for this portfolio piece relative to everything else on the list.
- Actually deploying to AWS or spending real cloud budget — the IaC track is
  a "here's how you'd deploy this" sketch/doc, never applied.
- Reproducibility benchmarking against the original MAZero paper's exact
  numbers — too much compute for the available budget.

## Constraints

- **~1 week** of active work remaining before the user calls this done.
- **A few days of borrowed RTX 5080** access for the one real training run —
  scarce, don't waste it on unvalidated code.
- Otherwise: the target machine described in CLAUDE.md (RTX 3060 Ti, 8GB) for
  everything else, including this session's own work (no local GPU in this
  sandbox — every GPU-dependent step in this plan needs to run on the user's
  actual hardware, not in-session).

## Phases

### Phase 0 — Verify performance and readability before building on top

**Why this comes first:** the second simplification pass replaced the custom
C++/CUDA replay buffer (CUDA-pinned-memory DMA) with `cpprb`'s plain pageable
`jax.device_put()` path. The README's "~300× throughput vs MAZero" table was
never re-measured after that swap — it's a real risk that the flagship
benchmark numbers on the front page of the portfolio piece are stale. Nothing
else in this plan (training run, bolt-on ablation, Docker) should build on
numbers that haven't been re-confirmed.

- Run the existing `utils/profiler.py` output on a real GPU box (the user's
  3060 Ti or the borrowed 5080) for a short training session, and compare
  against the README's current throughput table (episodes/sec,
  transitions/sec, train step time, buffer sample wait).
- If numbers moved meaningfully (e.g. GPU transfer got slower post-cpprb),
  update the README table and decide whether any GPU-side fix (see the
  "candidate: mixed precision" note below) is warranted — but only in
  response to what the profile actually shows, not speculatively.
- Targeted readability check: not a third full sweep (two were already done)
  — just confirm nothing crept back in since the second pass (dead code,
  stale comments, anything the cpprb swap left half-finished).
- **Candidate optimization, gated on the profile:** if the GPU train step is
  the actual bottleneck, bf16 mixed precision on the large matmuls is the
  standard, cheap, easy-to-explain win (small `dtype` change, no custom
  kernels). If the bottleneck is still CPU/Ray-side (the more likely case
  given the architecture — episode collection and Ray scheduling overhead
  were always the dominant cost), the right lever is one of CLAUDE.md's
  existing `[easy]` TODOs (bump `batch_size`, tune DataActor count), not a
  GPU kernel technique. **Flash attention was considered and rejected**: the
  attention encoder's sequence length is the agent count (3–20), far too
  small for flash attention's memory-bandwidth-bound sweet spot; it would add
  real complexity (custom pallas kernels or an external dependency) against
  a bottleneck that doesn't exist at this scale.

### Phase 1 — Research bolt-on (optional, small, config-flagged)

Two DreamerV3 tricks (Hafner et al., *Nature* 2025) fit the existing
categorical value/reward machinery almost for free. Both generic MBRL
robustness techniques, not MARL-specific — framed honestly as "adopted from
DreamerV3," not oversold as a MARL breakthrough:

1. **Symlog/symexp value+reward transform** — swap `muzero_scale`/
   `muzero_scale_inv` (`utils/transforms.py`, the original MuZero paper's
   Pohlen et al. 2018 hyperbolic h-transform) for DreamerV3's
   `symlog(x) = sign(x)·ln(|x|+1)` / `symexp`. Same call sites, same
   `scalar_to_support`/`support_to_scalar` categorical-support machinery —
   untouched. Gate behind a config choice
   (`ModelConfig.value_transform: hyperbolic|symlog`, default `hyperbolic`
   to preserve current behavior) so it's a trivial A/B, not a blind replace.
2. **Unimix categorical smoothing** — mix ~1% uniform probability into
   policy/value/reward categorical distributions before the loss. Honest
   caveat: this repo already does Dirichlet noise at the MCTS root, which
   covers similar ground on policy exploration — unimix's more distinct
   contribution here is stabilizing the value/reward categorical heads, not
   exploration. Gate behind a config flag
   (`ModelConfig.unimix_ratio: float`, default `0.0` = disabled).

Both are unit-testable in isolation (pure functions in `utils/transforms.py`
and wherever the categorical loss is computed) without needing a GPU or a
full training run — cheap to verify correctness before spending any borrowed
5080 time on them.

**Not pursuing:** no MARL-specific architectural change. Everything found in
recent literature search (entity-transformer generalist models, decentralized
transformer world models with centralized aggregation, transformer world
models for MARL) is a full world-model rewrite — explicitly out of scope
(see Non-goals).

### Phase 2 — Real training run

- Baseline run first (current shipped config, `value_transform: hyperbolic`,
  `unimix_ratio: 0.0`) on the borrowed RTX 5080 — this locks in a real result
  even if Phase 1's bolt-on underperforms or the borrowed-GPU window is
  shorter than hoped.
- If 5080 time remains after the baseline finishes: one bolt-on variant run
  (symlog + unimix enabled) for a head-to-head comparison plot.
- Output: real learning-curve / win-rate plots for the README, replacing or
  augmenting the current throughput-only table with evidence the system
  actually learns SMAX 3m. wandb is already wired (`wandb_mode: online` in
  `configs/train/default.yaml`) — use it for the run(s) and link/embed the
  resulting charts.

### Phase 3 — GPU-enabled Dockerfile

- CUDA base image, conda/pip install of `requirements.txt`, entrypoint
  running `train.py`.
- Cannot be verified end-to-end in this session (no GPU in this sandbox,
  same constraint as everything GPU-dependent) — needs a real run on the
  user's machine or the borrowed 5080 to confirm the image actually launches
  training. Flag this explicitly rather than claiming it works untested.

### Phase 4 — AWS IaC sketch (not deployed)

- Terraform module provisioning the actual shape of this repo's own
  architecture: one GPU instance (g5/g4dn family) for `LearnerActor`, one
  multi-core CPU instance (or N smaller instances) for the `DataActor`s /
  `ReplayBufferActor` / `ReanalyzeActor`, plus a bootstrap script installing
  dependencies and starting the Ray processes.
- Paired with a short "how you'd deploy this" doc in the README or a
  `docs/deployment.md`.
- Deliberately skipping Ray's own autoscaler YAML layer to keep this lean —
  note it as a "future extension" in the doc, not build it.
- Never actually applied/deployed — this is a demonstration of IaC
  competence, not a live deployment, per the constraint on cloud spend.

## Explicitly deferred

- CI (GitHub Actions running pytest/lint on push).
- Pre-commit hooks, `ruff`/`mypy` type checking.
- Both deferred at the user's direction — not worth the time relative to the
  phases above for this portfolio piece.

## Risks / open questions

- **Borrowed 5080 window may be shorter than "a few days."** Phase 2's
  ordering (baseline before bolt-on) is specifically designed so a truncated
  window still produces a usable result.
- **Docker and IaC phases can't be verified in this sandbox** — both need to
  be tried on real hardware (the user's 3060 Ti box or the 5080) before being
  claimed as working in the README.
- **Phase 0's profiling might reveal a real regression** from the cpprb swap
  that needs a fix beyond what this spec anticipates (e.g. pinned-memory
  `jax.device_put` reintroduced some other way) — if so, that fix should be
  scoped and sequenced ahead of Phase 2's training run, since the run's
  quality depends on it.
