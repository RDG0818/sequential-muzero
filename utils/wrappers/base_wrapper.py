import jax
import jax.numpy as jnp
import chex
from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple, Any, Dict

class JaxMarlWrapper(ABC):
    """
    An abstract base class for stateless, JAX-native environment wrappers.

    This ABC defines the core interface required for a wrapper to be compatible
    with a fully JAX-based, vmapped training pipeline. It ensures that all
    environment interactions are pure functions that can be JIT-compiled.
    """

    @property
    @abstractmethod
    def num_agents(self) -> int:
        """The number of agents in the environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """The size of the discrete action space for each agent."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]:
        """The shape of a single agent's observation."""
        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey) -> Tuple[chex.Array, Any]:
        """
        Resets the environment in a JIT-compiled, stateless manner.

        Args:
            key: A JAX random key.

        Returns:
            A tuple of (stacked_observations, environment_state).
        """
        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: jax.random.PRNGKey, state: Any, actions: chex.Array) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict]:
        """
        Steps the environment in a JIT-compiled, stateless manner.

        Args:
            key: A JAX random key.
            state: The current environment state.
            actions: A JAX array of actions for each agent.

        Returns:
            A tuple of (next_stacked_observations, next_state, team_reward, done_flag, info_dict).
        """
        raise NotImplementedError
