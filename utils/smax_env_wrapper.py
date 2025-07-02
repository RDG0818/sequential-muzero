import jax
import jax.numpy as jnp
from jaxmarl.wrappers.baselines import JaxMARLWrapper
from jaxmarl.environments.smax import HeuristicEnemySMAX
from functools import partial

class SMAXMuZeroWrapper(JaxMARLWrapper):
    """
    Wrapper for SMAX that adapts its output for a MuZero-style algorithm.

    - Extracts local agent observations from the dictionary format.
    - Stacks them into a single array: (num_agents, obs_dim).
    - Discards the "world_state" as it's not directly used by the
      MuZero agent's representation network.
    """
    def __init__(self, env: HeuristicEnemySMAX):
        super().__init__(env)

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        # The base environment returns obs as a dictionary
        obs_dict, env_state = self._env.reset(key)
        # We convert it to a simple array for our agent
        obs_arr = self._dict_to_arr(obs_dict)
        return obs_arr, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        # The base environment returns obs as a dictionary
        obs_dict, env_state, reward, done, info = self._env.step(key, state, action)
        # We convert it to a simple array for our agent
        obs_arr = self._dict_to_arr(obs_dict)
        return obs_arr, env_state, reward, done, info

    def _dict_to_arr(self, obs_dict: dict) -> jnp.ndarray:
        """Flattens the observation dictionary into a single array."""
        # Note: obs_dict['agent_0'], obs_dict['agent_1'], etc. are the local observations
        obs_stacked = jnp.stack([obs for agent, obs in obs_dict.items() if agent.startswith('agent_')], axis=0)
        return obs_stacked