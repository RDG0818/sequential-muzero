# config.py

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the MuZero model architecture."""
    hidden_state_size: int = 128
    value_support_size: int = 300
    reward_support_size: int = 300
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

@dataclass
class MCTSConfig:
    """Hyperparameters for the MCTS planner."""
    planner_mode: str = "sequential"  # "independent", "sequential", "joint"
    num_simulations: int = 100
    max_depth_gumbel_search: int = 10
    num_gumbel_samples: int = 10
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    independent_argmax: bool = True
    policy_eta: float = 0.8 # The larger this is, the more the model relies on the policy network

@dataclass
class TrainConfig:
    """Hyperparameters for the training process."""
    env_name: str = "MPE_simple_spread_v3"
    num_agents: int = 3
    num_episodes: int = 100000
    warmup_episodes: int = 1000
    log_interval: int = 100
    num_actors: int = 4
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
    coordination_scale: float = 1.0
    stability_loss_scale: float = 1.0
    gradient_clip_norm: float = 5.0
    unroll_steps: int = 5
    n_step : int = 10
    discount_gamma: float = 0.99
    wandb_mode: str = "disabled" # online or disabled
    project_name: str = "myzero1"

@dataclass
class ExperimentConfig:
    """Root configuration that composes all other configs."""
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)

CONFIG = ExperimentConfig()