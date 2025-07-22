import jax
import jax.numpy as jnp
import jaxmarl
from jaxmarl.environments.smax import map_name_to_scenario
from functools import partial
from typing import Tuple, Any, Dict
from utils.wrappers.base_wrapper import JaxMarlWrapper

class SMAXWrapper(JaxMarlWrapper):
    """
    A stateless, JAX-native wrapper for the StarCraft Multi-Agent Challenge (SMAX)
    environments from JaxMARL.
    
    This wrapper implements the JaxMarlWrapper ABC, providing a JIT-compiled,
    pure-function interface. It handles the dictionary-to-array conversions
    required by SMAX and correctly distinguishes between allied and enemy units.
    """
    def __init__(self, scenario_name: str, **kwargs: Any):
        """
        Initializes the SMAX environment wrapper.

        Args:
            scenario_name (str): The name of the SMAX scenario (e.g., '5m_vs_6m').
            **kwargs: Additional keyword arguments to pass to jaxmarl.make().
                      This is useful for configuring the SMAX environment, for example:
                      use_self_play_reward=False, walls_cause_death=True, etc.
        """
        scenario = map_name_to_scenario(scenario_name)
        
        self._env = jaxmarl.make("HeuristicEnemySMAX", scenario=scenario, **kwargs)
        
        self._agents = [agent for agent in self._env.agents if agent.startswith("ally")]
        
        self._num_agents = len(self._agents)
        self._action_space_size = self._env.action_space(self._agents[0]).n
        self._observation_shape = self._env.observation_space(self._agents[0]).shape

    @property
    def num_agents(self) -> int:
        """The number of allied agents in the environment."""
        return self._num_agents

    @property
    def action_space_size(self) -> int:
        """The size of the discrete action space for each allied agent."""
        return self._action_space_size
        
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """The shape of a single allied agent's observation."""
        return self._observation_shape

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Any]:
        """Resets the environment and stacks the initial observations for allied agents."""
        obs_dict, state = self._env.reset(key)
        
        obs = jnp.stack([obs_dict[agent] for agent in self._agents])
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: jax.random.PRNGKey, state: Any, actions: jnp.ndarray) -> Tuple[jnp.ndarray, Any, jnp.ndarray, jnp.ndarray, Dict]:
        """
        Steps the environment with actions for the allied agents.
        
        This method converts the array of actions into the dictionary format required
        by SMAX and then converts the resulting observations, rewards, and dones
        back to arrays. It also extracts the available actions for each agent.
        """
        action_dict = {agent: actions[i] for i, agent in enumerate(self._agents)}

        obs_dict, next_state, reward_dict, done_dict, info = self._env.step(key, state, action_dict)
        
        next_obs = jnp.stack([obs_dict[agent] for agent in self._agents])
        team_reward = jnp.sum(jnp.array([reward_dict[agent] for agent in self._agents]))
        
        done = done_dict["__all__"]
        
        avail_actions_dict = self._env.get_avail_actions(state)
        avail_actions_stacked = jnp.stack([avail_actions_dict[agent] for agent in self._agents])
        
        info["avail_actions"] = avail_actions_stacked.astype(jnp.bool_)
        
        return next_obs, next_state, team_reward, done, info
