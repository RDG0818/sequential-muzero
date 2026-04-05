# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Commands

```bash
# Setup
conda create -n mazero python=3.10.18 && conda activate mazero
pip install -r requirements.txt
wandb login  # set wandb_mode to "online" in configs/train/default.yaml to enable

# Run training (async Ray, CPU MCTS)
python train.py                                  # default config
python train.py mcts=joint                       # switch to joint planner
python train.py train.num_episodes=50000         # override single value
python train.py train.batch_size=128 mcts.num_simulations=50

# Run training (pure-JAX, GPU MCTS, Flashbax buffer)
python train_jax.py                              # default config
python train_jax.py train=jax                    # JAX preset (32 envs, batch=512)
python train_jax.py train.num_envs=64            # override env count

# Evaluate a checkpoint
python eval.py                                   # latest checkpoint, default config
python eval.py eval_episodes=200                 # more episodes
python eval.py train.checkpoint_dir=checkpoints  # explicit checkpoint dir

# Run tests
pytest tests/ -v
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
train.py                  # entry point — async Ray actor-learner (CPU MCTS)
train_jax.py              # entry point — pure-JAX synchronous (GPU MCTS + Flashbax buffer)
eval.py                   # standalone eval: loads checkpoint, runs N MCTS episodes, logs return
train_ippo.py             # entry point for IPPO baseline (pure JAX, no Ray)
train_mappo.py            # entry point for MAPPO baseline (pure JAX, no Ray)
config.py                 # dataclasses only: ModelConfig, MCTSConfig, TrainConfig, ExperimentConfig
configs/
  config.yaml             # root defaults (model: default, mcts: default, train: default)
  model/default.yaml      # model architecture hyperparameters
  mcts/default.yaml       # MCTS hyperparameters (independent planner)
  mcts/joint.yaml         # joint planner preset (inherits default, sets planner_mode: joint)
  train/default.yaml      # training hyperparameters
  baseline/
    ippo.yaml             # IPPO hyperparameters
    mappo.yaml            # MAPPO hyperparameters
actors/
  learner_actor.py        # LearnerActor (GPU), make_train_step factory
  data_actor.py           # DataActor (CPU)
  reanalyze_actor.py      # ReanalyzeActor (CPU): re-runs MCTS with latest params to freshen targets
  replay_buffer_actor.py  # ReplayBufferActor (wraps ReplayBuffer)
baselines/
  networks.py             # ActorCritic + CentralizedCritic Flax modules (shared params)
  ippo.py                 # IPPO: GAE + PPO clip, decentralized actor + decentralized critic
  mappo.py                # MAPPO: GAE + PPO clip, decentralized actor + centralized critic
training/
  loop.py                 # run_warmup(), run_training_loop(), run_training_loop_sync()
  loop_jax.py             # run_training_loop_jax(): pure-JAX GPU loop with Flashbax
model/
  model.py                # FlaxMAMuZeroNet and sub-networks
  attention.py            # TransformerAttentionEncoder
  layers.py               # MLP
mcts/
  base.py                 # MCTSPlanner base class, MCTSPlanOutput
  mcts_independent.py     # MCTSIndependentPlanner
  mcts_joint.py           # MCTSJointPlanner
envs/
  mpe_env_wrapper.py      # MPEEnvWrapper + VecMPEEnvWrapper (JaxMARL MPE)
  smax_env_wrapper.py     # stub for future SMAC wrapper
utils/
  transforms.py           # DiscreteSupport, scalar_to_support, support_to_scalar, muzero_scale/inv
  replay_buffer.py        # ReplayBuffer, ReplayItem, Episode, Transition, process_episode
  logging_utils.py        # logger singleton
tests/
  test_model.py
  test_mcts.py
```

## Architecture

**Training system** (`train.py` + `actors/` + `training/`): Asynchronous actor-learner pattern using Ray.
- `LearnerActor` (GPU): pulls batches from `ReplayBufferActor`, runs JIT-compiled training step, serves updated params.
- `DataActor` (CPU, N instances): runs MCTS episodes, ships `ReplayItem`s to the buffer. Syncs params every `param_update_interval` episodes.
- `ReplayBufferActor`: wraps `ReplayBuffer` (prioritized experience replay) backed by pre-allocated NumPy arrays.
- `ReanalyzeActor` (CPU, optional): continuously re-runs MCTS on stored observations with the latest params to generate fresher policy/value targets. Updates only position 0 (root) of each stored sequence. Controlled by `num_reanalyze_actors` and `reanalyze_batch_size` in `TrainConfig`.
- `training/loop.py`: `run_warmup()` fills the buffer; `run_training_loop()` drives learner, data actors, and reanalyze actors asynchronously via `ray.wait`.

**World model** (`model/`):
- `FlaxMAMuZeroNet`: top-level Flax module with four sub-networks.
  - `RepresentationNetwork`: per-agent observation → latent state `(B, N, D)`.
  - `DynamicsNetwork`: latent + joint action → next latent + reward logits. Optionally prepends `TransformerAttentionEncoder` for inter-agent communication.
  - `PredictionNetwork`: latent → per-agent policy logits `(B, N, A)` + centralized value logits.
  - `ProjectionNetwork`: BYOL-style EMA consistency head. Online branch applies projection + prediction MLP; target branch applies projection only using a slowly-moving EMA copy of the params (`ema_decay=0.999`, updated in `LearnerActor.train()` outside the JIT boundary).
- Reward and value are **categorical distributions** over a discrete support (tz-transform). Use `utils.transforms.scalar_to_support` / `support_to_scalar` to convert. The scaling functions are `muzero_scale` / `muzero_scale_inv` (in `utils/transforms.py`).
- `model.__call__` = initial inference; `model.recurrent_inference` = dynamics unroll (used inside MCTS simulations); `model.predict` = prediction head only (used by `MCTSIndependentPlanner` for non-searching agent priors).

**MCTS planners** (`mcts/`):
- `MCTSPlanner` (base): common config, `DiscreteSupport` objects, Dirichlet noise. Public entry point: `planner.plan(params, rng_key, obs)`.
- `MCTSIndependentPlanner`: one `gumbel_muzero_policy` search per agent via `jax.lax.scan`; other agents fixed to prior argmax during each agent's search.
- `MCTSJointPlanner`: single search over the combinatorial joint action space `A^N` with independence-factored logits; marginal policy targets extracted by summing over other agents' axes.
- All planners use `mctx.gumbel_muzero_policy` from DeepMind's MCTX library.

**Data flow**: `observation (B,N,obs_dim)` → representation → latent `(B,N,D)` → MCTS (calls `recurrent_inference` inside simulations) → `MCTSPlanOutput` → `Transition` → `Episode` → `process_episode` (n-step returns) → `ReplayItem` → `ReplayBuffer`.

**Config** (`config.py` + `configs/`): Hydra composes YAML files into a `DictConfig`, which `_build_config()` in `train.py` converts to typed dataclasses (`ModelConfig`, `MCTSConfig`, `TrainConfig`, `ExperimentConfig`). The dataclass instance is passed explicitly to every Ray actor constructor — there is no global singleton. All three sub-configs are `frozen=True`.

## Key TODOs in the Codebase

- Environment wrapper abstract base class (`envs/base.py`)
- SMAC/jaxMARL environment wrapper (`envs/smax_env_wrapper.py` is a stub)
- Unit tests for `utils/` and `train.py`

## Environment Wrappers

`envs/mpe_env_wrapper.py` wraps JaxMARL MPE environments. Any new environment wrapper must expose:
- `reset(rng_key) → (observation, state)` where observation shape is `(1, N, obs_dim)`
- `step(rng_key, state, actions) → (next_obs, next_state, reward, done)`
- `observation_shape: Tuple[int, ...]`, `observation_size: int`, `action_space_size: int`

The IPPO/MAPPO baselines interact with JaxMARL environments directly (not through `MPEEnvWrapper`) using `jaxmarl.make()` and `jaxmarl.wrappers.baselines.LogWrapper` for episode statistics.

## Baselines

IPPO and MAPPO are on-policy baselines implemented in pure JAX (no Ray). Key design choices:
- **Parameter sharing**: all agents share one network; agent one-hot ID is appended to obs
- **Vectorized rollout**: `jax.lax.scan` over T timesteps across B parallel envs simultaneously
- **No replay buffer**: on-policy — data collected each iteration is used once then discarded
- **Team reward**: summed reward across all agents (cooperative setting)
- **MAPPO centralized critic**: global state = concatenation of all agent observations `(B, N*obs_size)`

JaxMARL does **not** ship IPPO/MAPPO implementations — only environment wrappers and utilities. The `CTRolloutManager` in `jaxmarl.wrappers.baselines` provides centralized training utilities but is not used here (we implement rollout collection directly via scan).

## Future Improvements

Collected from the refactor session. Items marked **[easy]** are straightforward implementation tasks; **[research]** require design decisions.

### Utils

- **[easy] Flashbax for the replay buffer** — [instadeep/flashbax](https://github.com/instadeep/flashbax) is a JAX-native PER buffer. Replace `ReplayBuffer` with `flashbax.make_prioritised_flat_buffer`. Eliminates the CPU→GPU batch transfer on every training step, which is the main bottleneck in the current data pipeline. Also provides a proper segment tree for O(log n) priority updates vs. the current O(n) recomputation.

### Model

- **[easy] Observation normalization** — add a running mean/variance normalization layer at the top of `RepresentationNetwork`. Stabilizes training when observation scales vary across environments. Can be implemented as a `flax.linen.Module` that updates running stats via `self.variable('stats', ...)`.

- **[medium] Recurrent dynamics (GRU/LSTM)** — replace or augment the MLP in `DynamicsNetwork` with a recurrent cell to handle partial observability. Requires carrying hidden state through MCTS simulations, which means passing it in the `embedding` field of `mctx.RootFnOutput`.

- **[medium] Multi-step consistency loss (SPR)** — instead of computing consistency only between step t and t+1, compute it over k unroll steps. SPR (Self-Predictive Representations, Schwarzer et al. 2021) shows this significantly improves sample efficiency. The unroll already exists; add consistency targets at each step.

- **[medium] Data augmentation for consistency** — apply random observation noise or masking before the representation network for the online branch only. The target branch sees clean observations. Shown to improve consistency loss quality in EfficientZero.

- **[research] Value Equivalence Model (VEM)** — only propagate gradients through model transitions in directions relevant to the policy/value, not all directions. Reduces model capacity wasted on irrelevant state dimensions (Grimm et al. 2020).

- **[research] Stochastic MuZero** — explicitly model transition uncertainty with a latent variable in `DynamicsNetwork`. Useful for stochastic environments where a deterministic model is systematically wrong.

### MCTS

- **[medium] Sequential MCTS (proper implementation)** — agents search in order, each conditioning on the committed actions of prior agents. The key design question is what to put in `coordination_info`: either the prior agents' selected actions (concatenated into the latent) or a communication vector from the prior agents' MCTS trees.

- **[medium] Temperature annealing** — Gumbel MuZero's `max_num_considered_actions` acts like temperature. Anneal it down over training (high early for exploration, low late for exploitation). Currently fixed at `num_gumbel_samples`.

- **[medium] Factored policy targets** — for `MCTSJointPlanner`, the current marginal extraction (summing over other agents' axes) discards coordination information. An alternative: keep the full joint policy as the target and train with a factored policy head that explicitly models correlations.

- **[research] Communication before commitment** — before each MCTS search, agents exchange a communication vector (e.g., their current policy embedding). This vector is concatenated to the latent state, allowing the dynamics model to condition on what other agents intend to do. Bridges the gap between independent and sequential MCTS.

- **[research] Count-based / intrinsic exploration** — add an exploration bonus to the MCTS root value based on visitation counts or a learned density model. Helps in sparse-reward cooperative tasks where the agents need to discover coordinated behaviors.

### Training / Infrastructure

- **[major] Synchronous training loop** — replace the `ray.wait()`-based async loop with a sequential `ray.get()` loop. Simpler to reason about; prerequisite for on-policy MuZero variants. Treat as an architecture change, not a small cleanup.
