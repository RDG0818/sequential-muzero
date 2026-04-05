# config.py

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the MuZero model architecture."""
    hidden_state_size: int = 128
    value_support_size: int = 20
    reward_support_size: int = 5
    fc_representation_layers: Tuple[int, ...] = (128,)
    fc_dynamic_layers: Tuple[int, ...] = (128,)
    fc_reward_layers: Tuple[int, ...] = (32,)
    fc_value_layers: Tuple[int, ...] = (32,)
    fc_policy_layers: Tuple[int, ...] = (32,)
    attention_type: str = "transformer"  # "transformer" or "none"
    attention_layers: int = 1
    attention_heads: int = 1
    dropout_rate: float = 0.1
    proj_hid: int = 256
    proj_out: int = 256
    pred_hid: int = 64
    pred_out: int = 256


@dataclass(frozen=True)
class MCTSConfig:
    """Hyperparameters for the MCTS planner."""
    planner_mode: str = "independent"  # "independent" or "joint"
    num_simulations: int = 50
    max_depth_gumbel_search: int = 10
    num_gumbel_samples: int = 10
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    independent_argmax: bool = True


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for the training process."""
    env_name: str = "MPE_simple_spread_v3"
    num_agents: int = 3
    num_episodes: int = 100000
    warmup_episodes: int = 1000
    log_interval: int = 100
    num_actors: int = 3
    max_episode_steps: int = 25
    replay_buffer_size: int = 100000
    replay_buffer_alpha: float = 0.6
    replay_buffer_beta_start: float = 0.4
    replay_buffer_beta_frames: float = 100000
    batch_size: int = 256
    learning_rate: float = 3e-4
    param_update_interval: int = 20
    end_lr_factor: float = 0.1
    lr_warmup_steps: int = 5000
    value_scale: float = 0.25
    consistency_scale: float = 1.0
    gradient_clip_norm: float = 5.0
    unroll_steps: int = 5
    n_step: int = 10
    discount_gamma: float = 0.99
    wandb_mode: str = "disabled"  # "online" or "disabled"
    project_name: str = "myzero1"
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    num_envs: int = 16           # total parallel envs for JAX training (train_jax.py)
    num_envs_per_actor: int = 4  # number of parallel environments per DataActor
    sync: bool = False           # use synchronous training loop (simpler but ~2-3x slower)
    ema_decay: float = 0.999     # EMA decay for target encoder (BYOL-style consistency loss)
    num_reanalyze_actors: int = 1
    reanalyze_batch_size: int = 256
    debug: bool = False
    debug_interval: int = 100  # emit detailed debug logs every N learner training steps


@dataclass(frozen=True)
class ExperimentConfig:
    """Root configuration that composes all sub-configs."""
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)