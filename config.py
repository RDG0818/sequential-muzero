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
    attention_layers: int = 3
    attention_heads: int = 4
    dropout_rate: float = 0.1  

@dataclass
class MCTSConfig:
    """Hyperparameters for the MCTS planner."""
    planner_mode: str = "joint"  # "independent" or "joint"
    num_simulations: int = 50
    num_joint_samples: int = 16
    max_depth_gumbel_search: int = 10
    num_gumbel_samples: int = 10

@dataclass
class TrainConfig:
    """Hyperparameters for the training process."""
    env_name: str = "MPE_simple_spread_v3"
    num_agents: int = 3
    num_episodes: int = 500000
    warmup_episodes: int = 1000
    log_interval: int = 100
    num_actors: int = 6
    max_episode_steps: int = 25
    replay_buffer_size: int = 20000
    batch_size: int = 256
    learning_rate: float = 1e-4
    param_update_interval: int = 10
    end_lr_factor: float = 0.1
    lr_warmup_steps: int = 5000
    value_loss_coefficient: float = 0.5
    gradient_clip_norm: float = 5.0
    unroll_steps: int = 5
    discount_gamma: float = 0.99

@dataclass
class ExperimentConfig:
    """Root configuration that composes all other configs."""
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)

CONFIG = ExperimentConfig()