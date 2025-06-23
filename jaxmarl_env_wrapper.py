# jaxmarl_env_wrapper.py
import jax
import jaxmarl
import numpy as np

class JaxMARLEnvWrapper:
    """
    A wrapper for JaxMARL environments to make them compatible with the existing MCTS implementation.
    """
    def __init__(self, env_name, num_agents, max_steps, random_seed=0):
        self.env = jaxmarl.make(env_name, num_agents=num_agents, max_steps=max_steps)
        self.num_agents = num_agents
        self.agents = self.env.agents
        self.key = jax.random.PRNGKey(random_seed)
        self.observation_size = self.env.observation_space(self.agents[0]).shape[0]
        self.action_space_size = self.env.action_space(self.agents[0]).n
        self.state = None

    def reset(self):
        self.key, subkey = jax.random.split(self.key)
        obs, state = self.env.reset(subkey)
        obs_dict = {k: np.array(v) for k, v in obs.items()}
        self.state = state
        return self._stack_obs(obs_dict)

    def step(self, actions: np.ndarray):
        self.key, subkey = jax.random.split(self.key)
        action_dict = {agent: action.item() for agent, action in zip(self.agents, actions)}
        next_obs, next_state, reward, done, info = self.env.step(subkey, self.state, action_dict)
        self.state = next_state

        next_obs = {k: np.array(v) for k, v in next_obs.items()}
        reward = {k: np.array(v) for k, v in reward.items()}
        done = {k: np.array(v) for k, v in done.items()}
        return self._stack_obs(next_obs), sum(reward.values()), all(done.values())
    
    def _stack_obs(self, obs_dict):
        obs_list = [np.asarray(obs_dict[agent], dtype=np.float32) for agent in self.agents]
        return np.stack(obs_list, axis=0)[np.newaxis, ...]