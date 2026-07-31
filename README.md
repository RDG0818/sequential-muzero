# MAZero (JAX)

A faster JAX/Flax reimplementation of [MAZero](https://openreview.net/pdf?id=CpnKq3UJwp) — multi-agent MuZero with OS(λ) MCTS backup for cooperative MARL. Developed in collaboration with Mississippi State University, Rutgers University, and the Army Research Lab (TBAM project).

Contact: rdg291@msstate.edu

## Performance

30-minute timed runs vs original MAZero on SMAC 3m (batch=256, 50 sims, 10 sampled actions):

| Metric | MAZero | This repo (1 actor) | This repo (6 actors) |
|---|---|---|---|
| Episodes collected | ~2,800 | 7,300 | **18,100** |
| Transitions/sec | ~3.3 | ~330 | **~990** |
| Train step time | ~1,560ms | 14ms | **21ms** |
| Buffer sample wait | ~4,350ms | 0.7ms | **1.5ms** |

~300× throughput gain from three sources:
1. **JaxMARL environments** — SMAX runs on JAX+JIT vs Python/StarCraft II bindings
2. **Async actor-learner pipeline** — self-driving learner loop, async param sync, prioritized replay buffer
3. **Parallelism** — vectorized environments across multiple CPU actors

## Results

SMAX 3m (3 allies vs 3 scripted marines, JaxMARL `HeuristicEnemySMAX`), trained on a borrowed RTX 5070 Ti. Evaluated with `python eval.py model=smax mcts=joint eval_episodes=200`:

| Checkpoint | Mean return | Win rate | wandb |
|---|---|---|---|
| step 250,000 (constant LR — [paper default](configs/train/default.yaml), plateaued ep~14k-20k) | 1.44 ± 0.62 | 54.0% (54/100) | [run](https://wandb.ai/ryangoodwin0818-mississippi-state-university/myzero1/runs/53o5s8iw) |
| step _______ (LR decay fix, `end_lr_factor=0.1`, commit `34f333d`) | _______ | _______ | _______ |

## Implementation Highlights

**Async actor-learner** (Ray): `LearnerActor` runs N training steps per Ray call to amortize ~100ms scheduling overhead. Parameter syncs are fired at episode end and resolved at episode start, overlapping the ~300ms transfer with MCTS compute. All GPU→CPU metrics pack into a single `jnp.concatenate` for one DMA transaction.

**Prioritized replay buffer** (`utils/replay_buffer.py`): backed by `cpprb.PrioritizedReplayBuffer` — stratified PER sampling with alpha/beta annealing, uniform (without-replacement) sampling for reanalysis.

**OS(λ) MCTS** (`mcts/mcts_joint_osla.py`): custom JAX implementation of the MAZero planner. Per-node OS(λ) backup — each node tracks per-simulation values and depths; UCB selection uses quantile-weighted Q-estimates. Vmapped over batch; `jax.lax.fori_loop` over simulations. No mctx dependency anywhere in this repo.

**JAX + Ray constraint**: JAX eagerly allocates the entire GPU; Ray spawns isolated processes. All JAX imports must be inside actor `__init__` / methods — never at module top-level. The replay buffer converts everything to NumPy before storage to prevent DeviceArrays crossing process boundaries. Violating this causes SEGFAULTs.

## Installation

```bash
conda create -n mazero python=3.10.18 && conda activate mazero
pip install -r requirements.txt
pip install --upgrade "jax[cuda12]"  # requirements.txt installs CPU jax; this swaps in the GPU build
```

### Docker (GPU)

```bash
docker build -t sequential-muzero:gpu .
docker run --gpus all sequential-muzero:gpu model=smax mcts=joint
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host. Default `CMD` runs the SMAX 3m preset; override by appending Hydra args as shown above.

## Usage

```bash
# SMAX 3m (default)
python train.py model=smax mcts=joint

# Override hyperparameters
python train.py train.batch_size=512 mcts.num_simulations=100

# Evaluate (model=/mcts= must match what the checkpoint was trained with)
python eval.py model=smax mcts=joint eval_episodes=200

# Tests
pytest tests/ -v
```

All hyperparameters live in `configs/`. No code changes needed for overrides.

## Relevant Papers

- [MuZero](https://arxiv.org/pdf/1911.08265)
- [Gumbel MuZero](https://openreview.net/pdf?id=bERaNdoegnO)
- [EfficientZero](https://arxiv.org/pdf/2111.00210)
- [MAZero](https://openreview.net/pdf?id=CpnKq3UJwp)

## License

[MIT License](LICENSE)
