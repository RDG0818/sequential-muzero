# mcts/base.py

from abc import ABC, abstractmethod
from typing import NamedTuple

import chex

from model import FlaxMAMuZeroNet
from config import ExperimentConfig
from utils.transforms import DiscreteSupport


class MCTSPlanOutput(NamedTuple):
    """Output of any MCTS planner for a single planning step.

    Shapes (B=batch/envs, N=agents, A=actions, K=num_gumbel_samples):
        joint_action:       (B, N)    — chosen action index per agent
        policy_targets:     (B, N, A) — MCTS-improved policy targets for training
        root_value:         (B,)      — estimated value of the root state
        root_child_actions: (B, K, N) — K sampled joint actions as per-agent indices (None for non-OSLA)
        root_child_q:       (B, K)    — Q-value for each root child (None for non-OSLA)
        root_child_visits:  (B, K)    — visit count for each root child (None for non-OSLA)
    """
    joint_action:       chex.Array
    policy_targets:     chex.Array
    root_value:         chex.Array
    root_child_actions: chex.Array = None
    root_child_q:       chex.Array = None
    root_child_visits:  chex.Array = None


class MCTSPlanner(ABC):
    """
    Abstract base class for all MCTS planner variants.

    Subclasses must implement `_recurrent_fn` and `_plan_loop`.
    JIT compilation is handled externally (e.g. in DataActor) — planners
    do not JIT their own `plan()` method.
    """

    def __init__(self, model: FlaxMAMuZeroNet, config: ExperimentConfig):
        self.model = model
        self.num_agents = config.train.num_agents
        self.action_space_size = model.action_space_size

        self.num_simulations = config.mcts.num_simulations
        self.max_depth_gumbel_search = config.mcts.max_depth_gumbel_search
        self.num_gumbel_samples = config.mcts.num_gumbel_samples
        self.discount_gamma = config.train.discount_gamma

        self.value_support = DiscreteSupport(
            min=-config.model.value_support_size,
            max=config.model.value_support_size,
        )
        self.reward_support = DiscreteSupport(
            min=-config.model.reward_support_size,
            max=config.model.reward_support_size,
        )

        self.dirichlet_alpha = config.mcts.dirichlet_alpha
        self.dirichlet_fraction = config.mcts.dirichlet_fraction

    @abstractmethod
    def _recurrent_fn(
        self, params, rng_key: chex.Array, action: chex.Array, embedding
    ):
        """Single-step batched rollout used inside the MCTS simulations."""
        pass

    @abstractmethod
    def _plan_loop(
        self, params, rng_key: chex.Array, observation: chex.Array
    ) -> MCTSPlanOutput:
        """Full planning logic for one environment step."""
        pass

    def plan(
        self, params, rng_key: chex.Array, observation: chex.Array
    ) -> MCTSPlanOutput:
        """
        Public entry point. JIT compilation is the caller's responsibility
        (DataActor wraps this with jax.jit at construction time).
        """
        return self._plan_loop(params, rng_key, observation)
