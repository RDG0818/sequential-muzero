import jax
import jax.numpy as jnp
import jaxmarl
from functools import partial
from typing import Tuple, Any, Dict
from utils.wrappers.base_wrapper import JaxMarlWrapper

class MPEWrapper(JaxMarlWrapper):
    """
    A stateless, JAX-native wrapper for the MPE environments from JaxMARL.
    
    This wrapper implements the JaxMarlWrapper ABC, providing a JIT-compiled,
    pure-function interface. It handles the dictionary-to-array conversions
    required by MPE environments.
    """
    def __init__(self, env_name: str, num_agents: int):
        """
        Initializes the MPE environment wrapper.

        Args:
            env_name (str): The name of the JaxMARL MPE environment 
                            (e.g., 'MPE_simple_spread_v3').
            num_agents (int): The number of agents in the environment.
        """
        self._env = jaxmarl.make(env_name, num_agents=num_agents)
        
        self._agents = self._env.agents

        self._num_agents = self._env.num_agents
        self._action_space_size = self._env.action_space(self._agents[0]).n
        self._observation_shape = self._env.observation_space(self._agents[0]).shape

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def action_space_size(self) -> int:
        return self._action_space_size
        
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._observation_shape

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Any]:
        """Resets the environment and stacks the initial observations."""
        obs_dict, state = self._env.reset(key)
        obs = jnp.stack([obs_dict[agent] for agent in self._agents])
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: jax.random.PRNGKey, state: Any, actions: jnp.ndarray) -> Tuple[jnp.ndarray, Any, jnp.ndarray, jnp.ndarray, Dict]:
        """
        Steps the environment, converting actions to a dict and converting
        the resulting observations, rewards, and dones back to arrays.
        """
        action_dict = {agent: actions[i] for i, agent in enumerate(self._agents)}
        
        obs_dict, next_state, reward_dict, done_dict, info = self._env.step(key, state, action_dict)
        
        next_obs = jnp.stack([obs_dict[agent] for agent in self._agents])
        
        team_reward = jnp.sum(jnp.array([reward_dict[agent] for agent in self._agents]))
        
        done = done_dict["__all__"]
        
        return next_obs, next_state, team_reward, done, info
