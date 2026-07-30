# config.py

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the MuZero model architecture."""
    hidden_state_size: int
    value_support_size: int
    reward_support_size: int
    fc_representation_layers: Tuple[int, ...]
    fc_dynamic_layers: Tuple[int, ...]
    fc_reward_layers: Tuple[int, ...]
    fc_value_layers: Tuple[int, ...]
    fc_policy_layers: Tuple[int, ...]
    attention_type: str
    attention_layers: int
    attention_heads: int
    dropout_rate: float
    proj_hid: int
    proj_out: int
    pred_hid: int
    pred_out: int


@dataclass(frozen=True)
class MCTSConfig:
    """Hyperparameters for the MCTS planner."""
    num_simulations: int
    max_search_depth: int
    num_sampled_actions: int
    mcts_rho: float = 0.25    # OS(λ): top quantile fraction to DISCARD (1-rho is the fraction kept)
    mcts_lambda: float = 0.8  # OS(λ): depth discount weight (lambda^depth)


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for the training process."""
    env_name: str
    num_agents: int
    num_episodes: int
    warmup_episodes: int
    log_interval: int
    num_actors: int
    max_episode_steps: int
    replay_buffer_size: int
    replay_buffer_alpha: float
    replay_buffer_beta_start: float
    replay_buffer_beta_frames: float
    batch_size: int
    learning_rate: float
    param_update_interval: int
    end_lr_factor: float
    lr_warmup_steps: int
    value_scale: float
    consistency_scale: float
    consistency_horizon: int
    gradient_clip_norm: float
    unroll_steps: int
    n_step: int
    discount_gamma: float
    wandb_mode: str
    project_name: str
    checkpoint_dir: str
    checkpoint_interval: int
    num_envs_per_actor: int
    ema_decay: float
    num_reanalyze_actors: int
    reanalyze_batch_size: int
    debug: bool
    debug_interval: int
    awpo_alpha: float = 0.0  # AWPO temperature; 0.0 = disabled (plain BC loss)


@dataclass(frozen=True)
class ExperimentConfig:
    """Root configuration that composes all sub-configs."""
    train: TrainConfig
    model: ModelConfig
    mcts: MCTSConfig
