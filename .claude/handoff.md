# Handoff

## Goal

Turn `sequential-muzero` (JAX/Flax reimplementation of MAZero, multi-agent MuZero with OS(λ) MCTS for cooperative MARL) into a portfolio piece for ML infra/platform engineering roles. The original "sequential MCTS" research premise failed (never built, was worse than the joint planner when tried) and was abandoned — portfolio value is now the engineering/throughput story plus a real training result, not a novel research claim. Full scope lives in `docs/superpowers/specs/2026-07-30-portfolio-direction-design.md`: Phase 0 perf/readability check, Phase 1 DreamerV3-derived research bolt-on, Phase 2 real training run on a borrowed GPU, Phase 3 GPU Dockerfile, Phase 4 AWS IaC sketch (never deployed). User has about a week of active work budgeted; CI/lint explicitly deferred.

## Current state

All 5 phases of the spec are built. 125/125 tests green (`pytest tests/ -v`). `updates` branch is pushed to `origin/updates`, clean working tree, HEAD at `22ccf2c`.

- **Phase 1 (research bolt-on)**: `ModelConfig.value_transform` (`"hyperbolic"`|`"symlog"`) and `ModelConfig.unimix_ratio` in `config.py`, both config-flagged and default-off (byte-identical to prior behavior at defaults). Validated fast at driver startup in both `train/muzero.py:_build_config` and `eval.py:_build_config` (`get_value_transform_fns` raises on unknown name, `unimix_ratio` range-checked to `[0.0, 1.0)`). `utils/transforms.py` has `symlog`/`symexp`/`get_value_transform_fns`, docstrings trimmed to single-paragraph WHY statements.
- **Phase 3 (Docker)**: `Dockerfile` (repo root) + `.dockerignore`. `nvidia/cuda:12.4.1-runtime-ubuntu22.04` base, two-step pip install (`requirements.txt` then `jax[cuda12]` upgrade — matches the exact install quirk hit on real hardware). Build-validated locally in a prior session (pip install completes, same jaxmarl/jax version warning the user saw on bare metal); GPU runtime (`docker run --gpus all`) still untested anywhere — no GPU in this sandbox, user hasn't run it on the training box yet either.
- **Phase 4 (AWS IaC)**: `infra/aws/main.tf` + `infra/aws/bootstrap.sh.tpl` + `docs/deployment.md`. Deliberately a single-instance sketch (`g5.2xlarge`, whole Ray cluster on one box), not the GPU-node/CPU-node split the spec originally assumed — `terraform validate`/`fmt` clean. **Never `terraform apply`'d against a real AWS account** — that's the immediate next step, see Next Steps.
- **Phase 2 (training run)**: real SMAX 3m run on a borrowed RTX 5070 Ti (16GB, i9-14900K, `num_actors=4`). First run plateaued (flat loss/return, episode ~14k-20k) under the paper's constant LR — root cause confirmed by reading `actors/loss.py:23-29`, the `warmup_cosine_decay_schedule`'s `end_value == peak_value` when `end_lr_factor=1.0`, so the "schedule" was mechanically flat. Fixed via `configs/train/default.yaml`: `end_lr_factor: 0.1` (commit `34f333d`). Eval at that plateaued checkpoint (step 250,000): mean return 1.44 ± 0.62, **win rate 54.0% (54/100)**, logged in README. User is currently re-running training with the LR fix in place (resumes from the step-250,000 checkpoint; since the resumed step counter already exceeds the schedule's `decay_steps`, the fix applies as an immediate LR drop to 5e-5, not a gradual anneal) — as of session end this run was still in progress and returns were still climbing on live observation, contradicting a first guess (from stale log data) that it had already plateaued for good. **Do not assume the plateau diagnosis is final** — trust what the user reports from the live run over the earlier log snapshot.

Two real pre-existing bugs were found and fixed via actual usage this session (not code review, the user hit both by running the commands):
- `configs/config.yaml` never declared `eval_episodes` as a config key, so `python eval.py eval_episodes=200` failed under Hydra's strict-key mode (needed `+eval_episodes=200`). Fixed by declaring `eval_episodes: 100` in `configs/config.yaml`; `eval.py` now reads `cfg.eval_episodes` directly.
- `eval.py`'s docstring showed `train.num_simulations=100` — wrong config group, `num_simulations` lives under `mcts=`, so that override was silently ignored. Fixed to `mcts.num_simulations=100`.
- `eval.py` had its own duplicate `_build_config` with no range validation on `value_transform`/`unimix_ratio` (train path had it, eval path didn't — a previously "known parked gap" from an earlier review). Mirrored the same validation.
- `.gitignore` only covered `checkpoints/`, not `checkpoints_symlog/` (the ablation run's output dir). Fixed with a `checkpoints_*/` pattern.

## Files in flight

All committed and pushed, nothing uncommitted. This session's commits on top of the already-merged research-bolt-on/portfolio work:
- `1c3f590` — trimmed verbose docstrings in `utils/transforms.py`
- `78d628d` — added `Dockerfile` + `.dockerignore`, README Docker section + install fix
- `2176b24` — added `infra/aws/main.tf`, `infra/aws/bootstrap.sh.tpl`, `docs/deployment.md`
- `990c29d` — fixed `eval_episodes` Hydra strict-key bug, added clear error on checkpoint/config shape mismatch in `eval.py`
- `34f333d` — `configs/train/default.yaml`: `end_lr_factor` 1.0 → 0.1, to break the observed LR plateau
- `ad07f96` — fixed `eval.py` docstring typo (`train.num_simulations` → `mcts.num_simulations`), added validation to `eval.py`'s duplicate `_build_config`, `.gitignore` fix
- `adf4986` — README Results section with baseline eval numbers (blank row for the LR-fix rerun)
- `0365aac` — trimmed README to portfolio-facing content only (removed stale MAZero comparison table, removed Implementation Highlights section, cut em dashes/heavy bold)
- `22ccf2c` — user's own further manual README trim (not authored by me this session — don't revert; current README is the source of truth for its content/tone)

## What changed

See commit list above. In short: shipped Phases 3 and 4 of the spec (Docker, AWS sketch), ran the real Phase 2 training + eval, diagnosed and fixed a genuine LR-schedule bug found from the training curve, found and fixed 4 small pre-existing bugs in `eval.py`/config surfaced by the user actually running the commands, and rewrote the README twice (once by me per explicit "de-AI, remove implementation details, remove stale table" instructions, once more by the user directly).

## Failed attempts

- **Assuming the Phase 4 IaC sketch could split GPU-node/CPU-node** (per the original spec text): disproven by Phase 0's own profiling log — `DataActor` and `ReanalyzeActor` both print `Using GPU: cuda:0`, they call `model.recurrent_inference` during MCTS on GPU too, not just the learner. A CPU-only worker node would silently break those actors. Caught this before building (not after), asked the user, built a single-instance sketch instead and documented the real limitation in `docs/deployment.md` as a future extension requiring a CPU-JAX-backend code change first.
- **`num_actors=6` on the borrowed RTX 5070 Ti (16GB)**: OOM'd during the real Phase 2 baseline run (worked fine during the shorter Phase 0 profiling run at the same actor count — likely accumulates over a long run). Every actor preallocates GPU memory via JAX per-process, so actor count is VRAM-bound on this box, not CPU-core-bound like the 3060 Ti box `CLAUDE.md` describes. User settled on `num_actors=4`.
- **First diagnosis of the loss/return plateau as terminal**: read a training log window (episode ~14k-20k, step 180k-250k) showing flat loss/oscillating return and concluded the run had converged and needed the LR fix to make further progress. The user later reported the *next* continuation (still on the old constant-LR config, run before the fix was even written) was still climbing gradually on live observation — meaning the "plateau" in that specific window may have just been noise, not true convergence. I updated my stance rather than defending the original read. **Net effect: the LR fix is still a real, justified change (the math confirms the schedule was genuinely flat, not a schedule bug misdiagnosis), but its necessity for unblocking further improvement is not proven — the old constant-LR schedule may have continued improving on its own given enough episodes.**

## Known issues / blockers

- **Phase 2 LR-fix rerun result unknown.** User was actively running it as of session end; no log or eval output for it has been shared yet. The README's Results table has a blank row waiting on this (`step _______, LR decay fix`).
- **Docker GPU runtime never verified on real hardware.** Build was validated in a prior session (pip install completes) but `docker run --gpus all` has never actually been run, here or on the user's box.
- **AWS Terraform sketch never applied.** `terraform validate`/`fmt` clean, but the actual `apply` → verify → `destroy` cycle has not been run. This was the explicit next step the user asked for going into this session's final stretch.
- **MAZero comparison benchmark is stale/removed.** The old README table (MAZero vs this repo throughput, pre-cpprb-swap numbers) was deliberately removed at the user's request. They plan to set up the original MAZero repo on the same borrowed 5070 Ti box later and re-run a real comparison themselves — not scheduled, no commands prepared for it yet.
- **Ablation run (symlog/unimix) not yet run.** User agreed to run it but it hadn't started as of session end. Command: `python train.py model=smax mcts=joint train.wandb_mode=online model.value_transform=symlog model.unimix_ratio=0.01 train.checkpoint_dir=checkpoints_symlog`.

## Next steps

1. **Get the LR-fix rerun result from the user** (log + `python eval.py model=smax mcts=joint eval_episodes=200` output) and fill in the blank row of README's Results table.
2. **Docker GPU smoke test on the user's box**: `docker build -t sequential-muzero:gpu .` then `docker run --gpus all sequential-muzero:gpu model=smax mcts=joint` — confirm it actually trains, not just that the image builds.
3. **AWS apply/verify/destroy cycle** — user wants to do this for real but briefly/cheaply. Runbook already given to the user in-conversation (not yet written to a file): `terraform init`, `terraform apply -var="key_name=..." -var="ssh_cidr=$(curl -s ifconfig.me)/32"` from `infra/aws/`, wait ~5-10 min for the bootstrap script's git clone + docker build, verify via `curl -I http://<ip>:8265` (Ray dashboard) and `ssh ... "docker ps"`, then `terraform destroy` immediately to stop billing (~$0.30-0.40 for a 15-20 min test on `g5.2xlarge`).
4. Once the ablation run (symlog/unimix) is done, decide whether it's worth a second row in the Results table or a separate note.
5. Deferred to a future revisit (not this week): entity-attention/variable-action-space architecture work needed for true multi-scenario generalist training (confirmed via `envs/smax_env_wrapper.py` that action/obs sizes vary per SMAX scenario, so this needs a real architecture change, not a config flag), an episode replay visualizer, additional SMAX scenarios (8m, 2s3z, 3s5z), and the real MAZero comparison benchmark.
