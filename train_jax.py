"""
Pure-JAX synchronous training for Multi-Agent MuZero.

Runs entirely in a single process on GPU — no Ray, no CPU actors.
MCTS, env steps, trajectory processing, and training are all JIT-compiled
and GPU-resident via Flashbax.

Usage:
  python train_jax.py                              # default config
  python train_jax.py train=jax                    # JAX-optimised preset (32 envs, larger batch)
  python train_jax.py mcts=joint                   # joint planner
  python train_jax.py train.num_envs=64            # override env count
  python train_jax.py train.num_episodes=50000
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf

from config import ExperimentConfig, ModelConfig, MCTSConfig, TrainConfig
from utils.logging_utils import logger
from training import run_training_loop_jax


def _build_config(cfg: DictConfig) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True)),
        mcts=MCTSConfig(**OmegaConf.to_container(cfg.mcts, resolve=True)),
        train=TrainConfig(**OmegaConf.to_container(cfg.train, resolve=True)),
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # JAX is imported directly in this process — no Ray, no subprocess isolation needed.
    import jax
    import jax.numpy as jnp
    from envs import MPEEnvWrapper

    config = _build_config(cfg)

    if config.train.wandb_mode != "disabled":
        import wandb
        wandb.init(project=config.train.project_name, config=OmegaConf.to_container(cfg))

    # Get env metadata inline (safe since JAX is owned by this process).
    _env = MPEEnvWrapper(
        config.train.env_name,
        config.train.num_agents,
        config.train.max_episode_steps,
    )
    obs_size   = _env.observation_size
    action_size = _env.action_space_size
    del _env

    logger.info(
        f"Env: {config.train.env_name} | obs_size={obs_size} | action_size={action_size}\n"
        f"Envs: {config.train.num_envs} | Episodes: {config.train.num_episodes} | "
        f"Batch: {config.train.batch_size} | Devices: {jax.devices()}"
    )

    run_training_loop_jax(config, obs_size, action_size)

    if config.train.wandb_mode != "disabled":
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
