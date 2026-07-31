# MAZero (JAX)

A JAX/Flax reimplementation of [MAZero](https://openreview.net/pdf?id=CpnKq3UJwp), multi-agent MuZero with OS(λ) MCTS backup for cooperative MARL. Developed in collaboration with Mississippi State University, Rutgers University, and the Army Research Lab (TBAM project).

Contact: rdg291@msstate.edu

## Benchmark

Throughput comparison against the original MAZero (PyTorch, Cython MCTS, SMAC) is planned but not yet re-measured on current hardware.

## Results

SMAX 3m, 3 allies vs 3 scripted marines (JaxMARL `HeuristicEnemySMAX`), trained on a borrowed RTX 5070 Ti. Evaluated with `python eval.py model=smax mcts=joint eval_episodes=200`.

The first run used a constant learning rate (the paper's default) and plateaued around episode 14k to 20k. Dropped `end_lr_factor` to 0.1 (commit `34f333d`) to fix it.

| Checkpoint | Mean return | Win rate | wandb |
|---|---|---|---|
| step 250,000, constant LR | 1.44 ± 0.62 | 54.0% (54/100) | [run](https://wandb.ai/ryangoodwin0818-mississippi-state-university/myzero1/runs/53o5s8iw) |
| step _______, LR decay fix | _______ | _______ | _______ |

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

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host. Default `CMD` runs the SMAX 3m preset, override by appending Hydra args as shown above.

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
