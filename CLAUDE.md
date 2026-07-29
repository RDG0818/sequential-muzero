# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Target System

- **GPU**: NVIDIA GeForce RTX 3060 Ti, 8 GiB VRAM
- **CPU**: 12 cores
- **RAM**: 23 GiB
- **OS**: WSL2 (Ubuntu on Windows 11), accessed via SSH → Windows Terminal → WSL
- **CUDA driver**: 591.86 (WSL2 passthrough; JAX sees `CudaDevice(id=0)` correctly)

## Current Training Target

Primary scenario: **SMAX 3m** (3 allies vs 3 scripted marines, JaxMARL HeuristicEnemySMAX).

```bash
python train/muzero.py train=smax_3m model=smax mcts=joint
```

Key config group files:
- `configs/train/smax_3m.yaml` — episode/buffer/actor counts; sets `awpo_alpha=1.0`, `n_step=5`
- `configs/model/smax.yaml` — 2-layer [128,128] nets, 3-layer 8-head attention, value support [-5,5]
- `configs/mcts/joint.yaml` — MCTSJointOSLAPlanner, 50 sims, rho=0.25, K=5
- `configs/mcts/smax.yaml` — same as joint but 100 sims (closer to paper; noisier policy targets at 50)

Resource layout on the 12-core machine:
- 3 DataActors × 1 CPU (`@ray.remote(num_cpus=1)`, `OMP_NUM_THREADS=2`) = 3 CPUs claimed by Ray
- 1 ReanalyzeActor × 2 CPUs (`@ray.remote(num_cpus=2)`, `OMP_NUM_THREADS=2`) = 2 CPUs
- ~7 CPUs left for OS / LearnerActor dispatch / ReplayBufferActor

Note: DataActor is `num_cpus=1` in Ray scheduling but uses 2 XLA threads (`OMP_NUM_THREADS=2`).

## Commands

```bash
# Setup
conda create -n mazero python=3.10.18 && conda activate mazero
pip install -r requirements.txt
wandb login  # set wandb_mode to "online" in configs/train/default.yaml to enable

# Build C++ replay buffer extension (run once after cloning, or after modifying csrc/)
pip install pybind11
python setup.py build_ext --inplace

# Run training
python train/muzero.py                                  # default config (MPE simple_spread)
python train/muzero.py train=smax_3m model=smax mcts=joint  # SMAX 3m (primary target)
python train/muzero.py mcts=joint                       # lighter 50-sim preset (default is 100-sim mcts/default.yaml; planner is joint either way)
python train/muzero.py train.num_episodes=50000         # override single value
python train/muzero.py train.batch_size=128 mcts.num_simulations=50

# Run baselines
python train/ippo.py
python train/mappo.py

# Evaluate a checkpoint
python eval.py                                   # latest checkpoint, default config
python eval.py eval_episodes=200                 # more episodes
python eval.py train.checkpoint_dir=checkpoints  # explicit checkpoint dir

# Run tests
pytest tests/ -v

# Benchmark C++ vs Python replay buffer
python benchmarks/replay_buffer_benchmark.py
```

Hyperparameters live in `configs/`. Edit the relevant YAML or pass overrides on the CLI — no code changes needed.

## Critical JAX + Ray Constraint

JAX eagerly allocates the entire GPU, and Ray spawns isolated processes. To avoid SEGFAULTs and driver crashes:
- **All JAX imports must be inside Ray actor methods/`__init__`**, never at module top-level in actor files.
- **No JAX objects in global scope** of any Ray-managed process.
- The replay buffer converts everything to NumPy before storage to prevent DeviceArrays crossing process boundaries.

Violating these rules causes silent failures or segfaults that are difficult to debug.

## Package Layout

```
csrc/
  replay_buffer/
    sum_tree.h            # lock-free atomic sum tree (header-only)
    pinned_alloc.h        # cudaMallocHost / malloc fallback (header-only)
    replay_buffer.h/.cpp  # ReplayBuffer C++ class
    bindings.cpp          # pybind11 module (_replay_buffer_cpp)
CMakeLists.txt            # builds _replay_buffer_cpp.*.so; CUDA optional
setup.py                  # python setup.py build_ext --inplace
benchmarks/
  replay_buffer_benchmark.py  # C++ vs Python backend perf comparison
train/
  muzero.py               # entry point (@hydra.main, builds ExperimentConfig, launches Ray actors)
  ippo.py                 # entry point for IPPO baseline (pure JAX, no Ray)
  mappo.py                # entry point for MAPPO baseline (pure JAX, no Ray)
eval.py                   # standalone eval: loads checkpoint, runs N MCTS episodes, logs return
conftest.py               # pytest: adds project root to sys.path
config.py                 # dataclasses only: ModelConfig, MCTSConfig, TrainConfig, ExperimentConfig
configs/
  config.yaml             # root defaults (model: default, mcts: default, train: default)
  model/
    default.yaml          # model architecture hyperparameters
    smax.yaml             # SMAX overrides: deeper nets, 3-layer 8-head attn, tight value support
  mcts/
    default.yaml          # MCTS hyperparameters (base defaults; joint planner)
    joint.yaml            # joint planner preset (50 sims, rho=0.25, K=5)
    smax.yaml             # SMAX MCTS preset (100 sims; paper value)
  train/
    default.yaml          # training hyperparameters (MPE defaults)
    smax_3m.yaml          # SMAX 3m overrides: lr=1e-4, n_step=5, awpo_alpha=1.0, buf=500k
  baseline/
    ippo.yaml             # IPPO hyperparameters
    mappo.yaml            # MAPPO hyperparameters
actors/
  learner_actor.py        # LearnerActor (GPU): training loop, param sync, checkpointing
  loss.py                 # make_train_step factory, scale_grad_half, _awpo_weight — pure JAX, no Ray
  data_actor.py           # DataActor (CPU, num_cpus=1)
  reanalyze_actor.py      # ReanalyzeActor (CPU, num_cpus=2): re-runs MCTS to freshen targets
  replay_buffer_actor.py  # ReplayBufferActor (wraps ReplayBuffer)
baselines/
  networks.py             # ActorCritic + CentralizedCritic Flax modules (shared params)
  ippo.py                 # IPPO: GAE + PPO clip, decentralized actor + decentralized critic
  mappo.py                # MAPPO: GAE + PPO clip, decentralized actor + centralized critic
training/
  loop.py                 # run_warmup(), run_training_loop(), run_training_loop_sync()
model/
  model.py                # FlaxMAMuZeroNet and sub-networks
  attention.py            # TransformerAttentionEncoder
  layers.py               # MLP
mcts/
  base.py                 # MCTSPlanner base class, MCTSPlanOutput
  osla_math.py            # pure OS(λ) helpers: compute_osla_value(_jax), compute_ucb_scores, _sample_k_actions, _logits_to_joint_logits, _joint_policy_to_marginal
  mcts_joint_osla.py      # MCTSJointOSLAPlanner — custom JAX MCTS with OS(λ) per-node backup
envs/
  __init__.py             # make_env_wrapper / make_vec_env_wrapper factory functions
  mpe_env_wrapper.py      # MPEEnvWrapper + VecMPEEnvWrapper (JaxMARL MPE)
  smax_env_wrapper.py     # SMAXEnvWrapper + VecSMAXEnvWrapper (JaxMARL HeuristicEnemySMAX)
utils/
  obs_norm.py             # ObsRunningNorm — EMA per-feature observation normalizer
  transforms.py           # DiscreteSupport, scalar_to_support, support_to_scalar, muzero_scale/inv
  replay_buffer.py        # ReplayBuffer, ReplayItem, Episode, Transition, process_episode
  logging_utils.py        # logger singleton
  profiler.py             # Profiler class — per-operation timing for all actors
tests/
  test_model.py
  test_mcts.py
  test_learner.py         # tests for scale_grad_half (forward identity, backward halves grad)
  test_replay_buffer.py   # comprehensive C++ backend and Python fallback tests
```

## Architecture

**Training system** (`train/muzero.py` + `actors/` + `training/`): Asynchronous actor-learner pattern using Ray.
- `LearnerActor` (GPU): runs a self-driving training loop (`run_training_loop(N)`) that executes N steps internally before returning to the main loop. Eliminates Ray round-trip overhead between steps. Prefetches the next batch from the buffer while the GPU trains. EMA update is JIT-compiled to avoid per-leaf kernel launches. `make_train_step` returns a single packed `transfer_buf = jnp.concatenate([metric_scalars, grad_norm, priorities])` so all GPU→CPU data crosses PCIe in one DMA transaction; `_train_step` slices it after `np.array(transfer_buf)`.
- `DataActor` (CPU, N instances, `@ray.remote(num_cpus=1)`): runs MCTS episodes, ships `ReplayItem`s to the buffer. Syncs params asynchronously (fires `get_params.remote()` at end of episode, resolves at start of next) so the ~300ms transfer overlaps with MCTS compute. Applies observation normalization on CPU before sending to JAX if `use_obs_normalization=True`.
- `ReplayBufferActor`: wraps `ReplayBuffer` (prioritized experience replay). Uses the C++ backend (`_replay_buffer_cpp`) when built: lock-free ring buffer + sum tree via `std::atomic`, CUDA pinned output buffers so `jax.device_put()` DMA's directly without a pageable copy. Falls back to a pure-Python/cpprb implementation if the `.so` is not present. Build with `python setup.py build_ext --inplace`.
- `ReanalyzeActor` (CPU, optional, `@ray.remote(num_cpus=2)`): continuously re-runs MCTS on stored observations with the latest params to generate fresher policy/value targets. Updates only position 0 (root) of each stored sequence. Controlled by `num_reanalyze_actors` and `reanalyze_batch_size` in `TrainConfig`.
- `training/loop.py`: `run_warmup()` fills the buffer; `run_training_loop()` drives learner, data actors, and reanalyze actors asynchronously via `ray.wait`.
- `utils/profiler.py`: `Profiler` class used in all actors. Reports mean time per named operation every `debug_interval` steps. JAX note: always call `jax.block_until_ready()` inside a `profiler.time()` block to measure actual GPU/CPU compute, not just async dispatch time.

**Param sync protocol**: `get_params()` returns `{"params": ..., "norm_state": ...}` where `norm_state` is `None` when obs normalization is disabled, or a plain numpy dict `{mean, var, initialized}` that actors apply manually (avoiding a JAX import in the normalization path).

**World model** (`model/`):
- `FlaxMAMuZeroNet`: top-level Flax module with four sub-networks.
  - `RepresentationNetwork`: per-agent observation → latent state `(B, N, D)`.
  - `DynamicsNetwork`: latent + joint action → next latent + reward logits. Optionally prepends `TransformerAttentionEncoder` for inter-agent communication during transitions.
  - `PredictionNetwork`: latent → per-agent policy logits `(B, N, A)` + centralized value logits.
  - `ProjectionNetwork`: BYOL-style EMA consistency head. Online branch applies projection + prediction MLP; target branch applies projection only using a slowly-moving EMA copy of the params (`ema_decay=0.999`, updated in `LearnerActor._train_step()` asynchronously on GPU).
- Reward and value are **categorical distributions** over a discrete support (tz-transform). Use `utils.transforms.scalar_to_support` / `support_to_scalar` to convert. The scaling functions are `muzero_scale` / `muzero_scale_inv` (in `utils/transforms.py`).
- `model.__call__` = initial inference; `model.recurrent_inference` = dynamics unroll (used inside MCTS simulations); `model.predict` = prediction head only, no dynamics.

**Training loss** (`actors/loss.py`):
- `scale_grad_half`: custom VJP — identity forward, 0.5× backward. Applied to hidden states between unroll steps to prevent dynamics gradients from dominating representation gradients (MuZero paper §E, MAZero Appendix).
- **Multi-step SPR consistency**: for each k in `1..consistency_horizon` and each valid start position t, compare `project_online(h_t)` against `project_target(h_{t+k})` using cosine similarity. `consistency_horizon=1` reproduces original single-step loss. Higher values improve latent prediction accuracy but cost O(horizon²) projections (XLA CSE mitigates by reusing `project_online(h_t)` across k values).
- **AWPO** (Advantage-Weighted Policy Optimization): when `awpo_alpha > 0`, policy loss at step 0 is weighted by `exp(clip((V_mcts - V_net) / alpha, -5, 5))`, normalized by batch mean. Pushes the policy toward actions the MCTS found better than the current value estimate. Disabled by default (`awpo_alpha=0.0`); set to 1.0 for SMAX.

**MCTS planners** (`mcts/`):
- `MCTSPlanner` (base): common config, `DiscreteSupport` objects, Dirichlet noise. Public entry point: `planner.plan(params, rng_key, obs)`.
- `MCTSJointOSLAPlanner` (`joint`, the only planner, default everywhere): custom JAX MCTS (not mctx's search — `mctx.RecurrentFnOutput` is reused as a plain return-type container). Per-node OS(λ) backup — each node tracks per-simulation values/depths; UCB selection uses OS(λ)-estimated Q-values (top (1-rho) quantile weighted by λ^depth). Vmapped over B environments; `jax.lax.fori_loop` over simulations. Matches MAZero algorithm exactly.

**Data flow**: `observation (B,N,obs_dim)` → [obs normalization] → representation → latent `(B,N,D)` → MCTS (calls `recurrent_inference` inside simulations) → `MCTSPlanOutput` → `Transition` → `Episode` → `process_episode` (n-step returns) → `ReplayItem` → `ReplayBuffer`.

**Config** (`config.py` + `configs/`): Hydra composes YAML files into a `DictConfig`, which `_build_config()` in `train/muzero.py` converts to typed dataclasses (`ModelConfig`, `MCTSConfig`, `TrainConfig`, `ExperimentConfig`). The dataclass instance is passed explicitly to every Ray actor constructor — there is no global singleton. All three sub-configs are `frozen=True`.

Notable config fields added since original docs:
- `ModelConfig.use_obs_normalization: bool` — enables `ObsRunningNorm` in the learner; default `false`
- `TrainConfig.consistency_horizon: int` — SPR multi-step horizon (1 = original, default in default.yaml)
- `TrainConfig.awpo_alpha: float` — AWPO temperature; 0.0 = disabled (default), 1.0 for SMAX

## Environment Wrappers

`envs/__init__.py` exposes factory functions:
- `make_env_wrapper(env_name, num_agents, max_steps)` — single-env wrapper
- `make_vec_env_wrapper(env_name, num_agents, max_steps, num_envs)` — vectorized wrapper

Routing: env names starting with `"MPE_"` → MPE wrappers; anything else (e.g. `"3m"`, `"2s3z"`) → SMAX wrappers.

`envs/smax_env_wrapper.py`: fully implemented (not a stub). `SMAXEnvWrapper` and `VecSMAXEnvWrapper` wrap `HeuristicEnemySMAX`. Key subtleties:
- Team reward = one ally's reward (not sum), to avoid overcounting by N.
- Episode termination uses `done["__all__"]`, not per-agent done (dead agent ≠ episode over).
- Win detection: `done & (reward > 0.5)` (JaxMARL adds `won_battle_bonus=1.0` to terminal reward; max per-step HP damage is ~0.13, so >0.5 is a reliable win signal).
- `VecSMAXEnvWrapper.step()` returns `(obs, states, rewards, dones, won)` — 5-tuple, unlike MPE.

Any new environment wrapper must expose:
- `reset(rng_key) → (observation, state)` where observation shape is `(1, N, obs_dim)` for single-env or `(B, N, obs_dim)` for vec
- `step(rng_key, state, actions) → (next_obs, next_state, reward, done[, won])` 
- `observation_shape: Tuple[int, ...]`, `observation_size: int`, `action_space_size: int`

## Observation Normalization

`utils/obs_norm.py` — `ObsRunningNorm` class:
- EMA per-feature normalizer updated each training batch over all `(B*N, obs_size)` observations.
- State is plain numpy (`{mean, var, initialized}`) so it can be serialized into `get_params()` and applied on CPU in DataActor/ReanalyzeActor without triggering JAX.
- Enabled by `use_obs_normalization: true` in model config (default `false`).
- Checkpoint save/restore includes `obs_norm_mean` and `obs_norm_var` when enabled.

## Baselines

IPPO and MAPPO are on-policy baselines implemented in pure JAX (no Ray). Key design choices:
- **Parameter sharing**: all agents share one network; agent one-hot ID is appended to obs
- **Vectorized rollout**: `jax.lax.scan` over T timesteps across B parallel envs simultaneously
- **No replay buffer**: on-policy — data collected each iteration is used once then discarded
- **Team reward**: summed reward across all agents (cooperative setting)
- **MAPPO centralized critic**: global state = concatenation of all agent observations `(B, N*obs_size)`

JaxMARL does **not** ship IPPO/MAPPO implementations — only environment wrappers and utilities. The `CTRolloutManager` in `jaxmarl.wrappers.baselines` provides centralized training utilities but is not used here (we implement rollout collection directly via scan).

## MAZero Reference Implementation (`../MAZero/`)

The paper codebase this repo is partially based on. Key facts for understanding where our design choices came from and where they diverge.

**Paper**: "Efficient Multi-agent Reinforcement Learning by Planning" (Liu et al., ICLR 2024).
**Repo**: `github.com/liuqh16/MAZero` (local mirror at `../MAZero/`).

### Stack Differences

| | MAZero (reference) | This repo |
|---|---|---|
| Framework | PyTorch + Cython MCTS tree (`cytree`) | JAX/Flax, pure-JAX MCTS |
| SMAC backend | Original SMAC (needs StarCraft II binary) | JaxMARL HeuristicEnemySMAX |
| MCTS tree | C++/Cython `ctree_sampled` (not vmappable) | Custom JAX (`mcts_joint_osla.py`), vmap over batch |
| Batch size | 256 | 2048 |
| Reanalyze actors | 20–30 (the throughput bottleneck) | 1 |
| Consistency target | Fresh re-encode of obs_{t+k} via representation net | EMA params applied to h_{t+k} from dynamics path |

### Paper Hyperparameters (from `train_smac.sh`)

```
num_simulations: 100    # N; our smax.yaml uses 100, joint.yaml uses 50
sampled_action_times: 10  # K (joint actions sampled per node); our num_gumbel_samples=5 for SMAX
mcts_rho: 0.25          # keep top 75% (1 - rho) of simulations — matches our config
mcts_lambda: 0.8        # depth discount — matches our config
batch_size: 256         # we use 2048
lr: 5e-4                # we use 1e-4 for SMAX
discount: 0.99
td_steps: 5             # our n_step=5 for SMAX
PG_type: sharp          # AWAC weights × visit counts (we simplified to awpo_alpha)
awac_lambda: 2          # their AWPO temperature; our awpo_alpha=1.0 for SMAX
adv_clip: 3.0           # we clip advantage to [-5, 5] in exp() before normalizing
```

### Architecture Differences

**DynamicsNetwork** (paper is more complex): MAZero's dynamics applies a 3-layer 8-head `AttentionEncoder` to `concat(h, a)`, then concatenates the attention output back: `MLP([h, a, attn(concat(h, a))])`. Our `DynamicsNetwork` optionally prepends attention to `concat(h, a)` before the MLP — closer but not identical (we skip the residual concat).

**Projection target** (different consistency formulation): MAZero computes `sg(P1(rep(obs_{t+k})))` — a fresh re-encoding of the actual next observation, stop-gradient. We use `EMA_params(P1(h_{t+k}))` where `h_{t+k}` comes from the dynamics unroll. The paper's formulation is empirically stronger when ground-truth observations are available at every step (no partial observability during training batch construction). Ours avoids storing `k` future observations per replay item.

**Policy targets** (different formulation): MAZero stores K sampled joint actions per node and their visit counts, then computes a sampled policy loss over those K actions (AWAC-weighted). Our `MCTSJointOSLAPlanner` returns per-agent marginal visit counts (summed over other agents), which are then used as cross-entropy targets for the decentralized policy head. The marginal extraction discards joint coordination information.

**Gradient scaling**: Both use 0.5× gradient on hidden states between unroll steps. MAZero: `hidden_state.register_hook(lambda grad: grad * 0.5)`. Ours: `scale_grad_half` custom VJP.

**Reanalyze flow**: MAZero's reanalyze workers preprocess full batches (re-encode observations, re-run MCTS, compute importance ratios) and push results to a `batch_storage` queue that the trainer consumes. This is the throughput bottleneck — hence 20–30 reanalyze actors on an A100. Our `ReanalyzeActor` is simpler: sample root observations, re-run MCTS, update only position-0 policy/value targets in-place.

### What We Preserved from the Paper

- OS(λ) backup: `rho`, `lambda`, quantile-weighted Q-estimation, top-(1-rho) filtering
- Network architecture topology: representation → (B,N,D) → dynamics with joint action → centralized value, decentralized policy, centralized reward
- Support sizes: `(-5, 5)` for SMAX value/reward
- LayerNorm on representation input
- EMA target encoder for consistency (though different target construction)
- Dirichlet noise at root, temperature on sampling

## Key TODOs in the Codebase

- Environment wrapper abstract base class (`envs/base.py`)
- Unit tests for `train.py`

## Future Improvements

Items marked **[easy]** are straightforward; **[medium]** require more design work; **[research]** require design decisions.

### Utils

- **[done] C++ replay buffer** — lock-free ring buffer + sum tree, CUDA pinned buffers, stratified PER sampling, Vitter's Algorithm R for uniform reanalysis sampling. Pure-Python/cpprb fallback if `.so` not built.
- **[done] Observation normalization** — `utils/obs_norm.py`: EMA per-feature normalizer, synced to actors via `get_params()`.

### Model

- **[done] Multi-step consistency loss (SPR)** — `consistency_horizon` param controls k-step lookahead; k=1 by default, SMAX uses 1 (can try 2-3).

- **[medium] Recurrent dynamics (GRU/LSTM)** — replace or augment the MLP in `DynamicsNetwork` with a recurrent cell to handle partial observability. Requires carrying hidden state through MCTS simulations, which means storing it in the `embedding` field of `MCTSJointOSLAPlanner`'s `OSLATree` (`mcts/mcts_joint_osla.py`) alongside the existing per-node latent.

- **[medium] Data augmentation for consistency** — apply random observation noise or masking before the representation network for the online branch only. The target branch sees clean observations. Shown to improve consistency loss quality in EfficientZero.

- **[research] Value Equivalence Model (VEM)** — only propagate gradients through model transitions in directions relevant to the policy/value, not all directions. Reduces model capacity wasted on irrelevant state dimensions (Grimm et al. 2020).

- **[research] Stochastic MuZero** — explicitly model transition uncertainty with a latent variable in `DynamicsNetwork`. Useful for stochastic environments where a deterministic model is systematically wrong.

### MCTS

- **[medium] Sequential MCTS (proper implementation)** — agents search in order, each conditioning on the committed actions of prior agents. The key design question is what to put in `coordination_info`: either the prior agents' selected actions (concatenated into the latent) or a communication vector from the prior agents' MCTS trees.

- **[medium] Temperature annealing** — `num_gumbel_samples` (K, the number of joint actions sampled per node in `MCTSJointOSLAPlanner`) acts like temperature. Anneal it down over training (high early for exploration, low late for exploitation). Currently fixed for the whole run.

- **[medium] Factored policy targets** — for `MCTSJointOSLAPlanner`, the current marginal extraction (summing over other agents' axes) discards coordination information. An alternative: keep the full joint policy as the target and train with a factored policy head that explicitly models correlations.

- **[research] Count-based / intrinsic exploration** — add an exploration bonus to the MCTS root value based on visitation counts or a learned density model. Helps in sparse-reward cooperative tasks where the agents need to discover coordinated behaviors.

### Training / Infrastructure

- **[easy] Increase `batch_size`** — larger batches amortize Ray scheduling overhead (~100ms/call) by doing more GPU work per step. Try 512 or 1024.
- **[done] Single D2H transfer** — `make_train_step` packs metrics + priorities into one contiguous JAX array; `_train_step` does one `np.array()` call and slices. Eliminated ~18ms/step of PCIe per-scalar overhead.
- **[done] C++ replay buffer with pinned memory** — see Utils above.
- **[done] Thread affinity for CPU actors** — each DataActor/ReanalyzeActor sets `OMP_NUM_THREADS=2` + `--xla_cpu_multi_thread_eigen=false` before JAX import. Also `OPENBLAS_NUM_THREADS=2` and `MKL_NUM_THREADS=2`.
- **[done] More DataActors (3)** — `num_actors=3` fills the buffer fast enough for `batch_size=2048`; learner no longer data-starved.
- **[done] AWPO** — advantage-weighted policy loss at the root; stabilizes policy learning on SMAX sparse rewards.
- **[note] Candidate libraries** — `rlax` could replace `utils/transforms.py` entirely (`transform_to_2hot`/`from_2hot` = `scalar_to_support`/`support_to_scalar`; `signed_hyperbolic`/`signed_parabolic` = `muzero_scale`/`inv`) and simplify baseline GAE/PPO. `flashbax` is suitable for a future pure-JAX training path (keeps buffer on GPU). `jaxtyping` + `beartype` for runtime shape checking during development.
