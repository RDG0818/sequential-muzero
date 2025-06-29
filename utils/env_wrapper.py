# jaxmarl_env_wrapper.py
import jax
import jaxmarl
import numpy as np
import jax.numpy as jnp
from typing import List, Dict, Tuple, Any

class EnvWrapper:
    """
    A stateless wrapper for JaxMARL environments to provide a NumPy-based interface
    compatible with the multi-agent MuZero implementation. This wrapper is responsible for
    key generation and converting JaxMARL's dictionary-based outputs into stacked
    NumPy arrays.

    Note: Environment state is managed externally by the caller, making this class stateless.
    """
    def __init__(self, env_name: str, num_agents: int, max_steps: int, random_seed: int = 0):
        """
        Initializes the environment wrapper.

        Args:
            env_name (str): The name of the JaxMARL environment to load (e.g., 'MPE_simple_spread_v3').
            num_agents (int): The number of agents in the environment.
            max_steps (int): The maximum number of steps per episode.
            random_seed (int): The seed for the pseudo-random number generator.
        """
        self.env = jaxmarl.make(env_name, num_agents=num_agents, max_steps=max_steps)
        self.num_agents: int = num_agents
        self.agents: List[str] = self.env.agents
        self.observation_size: int = self.env.observation_space(self.agents[0]).shape[0]
        self.observation_space: Tuple = (self.num_agents, self.observation_size)
        self.action_space_size: int = self.env.action_space(self.agents[0]).n
        self.key: jax.random.PRNGKey = jax.random.PRNGKey(random_seed)

    def reset(self) -> Tuple[np.ndarray, Any]:
        """
        Resets the environment to an initial state.

        Returns:
            Tuple[np.ndarray, Any]: A tuple containing:
                - observations (np.ndarray): The initial stacked observations for all agents.
                  Shape: (1, num_agents, observation_size)
                - state (Any): The initial global state of the environment.
        """
        self.key, subkey = jax.random.split(self.key)
        obs_dict, state = self.env.reset(subkey)
        
        return self._stack_dict(obs_dict), state

    def step(self, state: Any, actions: np.ndarray) -> Tuple[np.ndarray, Any, float, bool]:
        """
        Executes a step in the environment for all agents using the provided state.

        Args:
            state (Any): The current global state of the environment.
            actions (np.ndarray): The joint action for all agents.
                                  Shape: (num_agents,)

        Returns:
            Tuple[np.ndarray, Any, float, bool]: A tuple containing:
                - next_observations (np.ndarray): Stacked observations for all agents.
                  Shape: (1, num_agents, observation_size)
                - next_state (Any): The subsequent global state of the environment.
                - team_reward (float): The summed reward for the entire team.
                - episode_done (bool): A single boolean indicating if the episode has ended.
        """
        self.key, subkey = jax.random.split(self.key)
        action_dict = {agent: action.item() for agent, action in zip(self.agents, actions)}
        next_obs, next_state, reward, done, info = self.env.step(subkey, state, action_dict)

        next_obs = {k: np.array(v) for k, v in next_obs.items()}
        reward = {k: np.array(v) for k, v in reward.items()}
        done = {k: np.array(v) for k, v in done.items()}
        return self._stack_dict(next_obs), next_state, sum(reward.values()), all(done.values())

    def _stack_dict(self, data_dict: Dict[str, jnp.ndarray]) -> np.ndarray:
        """
        Converts a dictionary of agent data (obs, rewards, dones) into a stacked NumPy array.

        Args:
            data_dict (Dict[str, jnp.ndarray]): A dictionary mapping agent IDs to their data.

        Returns:
            np.ndarray: The stacked data array, with a new leading batch dimension.
                        Shape: (1, num_agents, *data_shape)
        """
        data_list = [np.asarray(data_dict[agent], dtype=np.float32) for agent in self.agents]
        return np.stack(data_list, axis=0)[np.newaxis, ...]

    