from abc import ABC, abstractmethod
from typing import NamedTuple
import jax.numpy as jnp
import chex

from model.model import FlaxMAMuZeroNet
from config import ExperimentConfig
from utils.utils import DiscreteSupport

class MCTSPlanOutput(NamedTuple):
    """
    A container for the output of any MCTS planner.

    Attributes:
        joint_action: A vector of shape (N,) containing the chosen action for each agent.
        policy_targets: The MCTS policy targets for each agent. Shape (N, A).
        root_value: A scalar representing the estimated value of the root state.
    """
    joint_action:   jnp.ndarray
    policy_targets: jnp.ndarray
    root_value:     float
    delta_magnitude: chex.Array
    coord_state_norm: chex.Array

class MCTSPlanner(ABC):
    """
    Abstract Base Class for all MCTS planner variations.

    This class defines a standard structure for all MCTS planners, ensuring they
    implement a recurrent function for rollouts and a main planning loop. It also
    establishes a consistent pattern for JIT compilation.

    Concrete subclasses are expected to:
    1. Implement the `_recurrent_fn` method with the specific simulation logic.
    2. Implement the `_plan_loop` method with the main planning algorithm.
    3. In their `__init__` method, create JIT-compiled versions of these
       methods and assign them to `self._recurrent_fn_jit` and `self.plan_jit`.
    """
    def __init__(self, model: FlaxMAMuZeroNet, config: ExperimentConfig):
        """Initializes common attributes for all planners."""
        self.model = model
        self.config = config
        self.num_agents = config.train.num_agents
        self.action_space_size = model.action_space_size
        
        # Common MCTS/MuZero parameters
        self.num_simulations = config.mcts.num_simulations
        self.max_depth_gumbel_search = config.mcts.max_depth_gumbel_search
        self.num_gumbel_samples = config.mcts.num_gumbel_samples
        
        # Support objects for converting logits to scalar values       
        self.value_support = DiscreteSupport(min=-config.model.value_support_size, max=config.model.value_support_size)
        self.reward_support = DiscreteSupport(min=-config.model.reward_support_size, max=config.model.reward_support_size)
        
        # Subclasses must define these in their own __init__
        self.plan_jit = None
        self._recurrent_fn_jit = None

    @abstractmethod
    def _recurrent_fn(self, params, rng_key, action, embedding):
        """
        Defines a single-step, batched rollout within the MCTS search.

        This abstract method must be implemented by subclasses to define the
        specifics of the environment dynamics according to the model.
        """
        pass

    @abstractmethod
    def _plan_loop(self, params, rng_key, observation: jnp.ndarray) -> MCTSPlanOutput:
        """
        The main planning logic that orchestrates the MCTS search.

        This abstract method must be implemented by subclasses. It will contain
        the core algorithm for running the search (e.g., looping over agents
        independently or performing a joint search).
        """
        pass
    
    def plan(self, params, rng_key, observation: jnp.ndarray) -> MCTSPlanOutput:
        """
        Public-facing method to execute the JIT-compiled planning process.

        This method should not be overridden. It ensures that the JIT-compiled
        version of the planning loop is always called.
        """
        if self.plan_jit is None:
            raise NotImplementedError(
                "Subclasses of MCTSPlanner must define `self.plan_jit` in their `__init__` method. "
                "This is typically done with `self.plan_jit = jax.jit(self._plan_loop)`."
            )
        return self.plan_jit(params, rng_key, observation)