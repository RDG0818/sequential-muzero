# envs/mpe_env_wrapper.py
import jax
import jaxmarl
import numpy as np
import jax.numpy as jnp
from typing import Any, Dict, List, Tuple


class MPEEnvWrapper:
    """
    A stateless wrapper for JaxMARL MPE environments.

    Converts JaxMARL's dict-based outputs to stacked NumPy arrays.
    RNG keys are passed in by the caller — this class holds no random state.
    """

    def __init__(self, env_name: str, num_agents: int, max_steps: int):
        self.env = jaxmarl.make(env_name, num_agents=num_agents)
        self.num_agents: int = num_agents
        self.agents: List[str] = self.env.agents
        self.observation_size: int = self.env.observation_space(self.agents[0]).shape[0]
        self.observation_shape: Tuple[int, ...] = (self.observation_size,)
        self.action_space_size: int = self.env.action_space(self.agents[0]).n

    def reset(self, rng_key: jax.Array) -> Tuple[np.ndarray, Any]:
        """
        Resets the environment.

        Args:
            rng_key: JAX PRNG key for the reset.

        Returns:
            observations: Shape (1, num_agents, observation_size)
            state: Initial environment state.
        """
        obs_dict, state = self.env.reset(rng_key)
        return self._stack_dict(obs_dict), state

    def step(
        self, rng_key: jax.Array, state: Any, actions: np.ndarray
    ) -> Tuple[np.ndarray, Any, float, bool]:
        """
        Steps the environment.

        Args:
            rng_key: JAX PRNG key for the step.
            state: Current environment state.
            actions: Joint actions for all agents. Shape: (num_agents,)

        Returns:
            next_observations: Shape (1, num_agents, observation_size)
            next_state: Subsequent environment state.
            team_reward: Summed reward across all agents.
            episode_done: True if all agents are done.
        """
        action_dict = {agent: int(action) for agent, action in zip(self.agents, actions)}
        next_obs, next_state, reward, done, _ = self.env.step(rng_key, state, action_dict)

        next_obs = {k: np.array(v) for k, v in next_obs.items()}
        reward = {k: np.array(v) for k, v in reward.items()}
        done = {k: np.array(v) for k, v in done.items()}
        return self._stack_dict(next_obs), next_state, float(sum(reward.values())), bool(all(done.values()))

    def _stack_dict(self, data_dict: Dict[str, jnp.ndarray]) -> np.ndarray:
        """Stacks per-agent data into shape (1, num_agents, *data_shape)."""
        data_list = [np.asarray(data_dict[agent], dtype=np.float32) for agent in self.agents]
        return np.stack(data_list, axis=0)[np.newaxis, ...]


class VecMPEEnvWrapper:
    """
    Vectorized wrapper for JaxMARL MPE environments.

    Runs `num_envs` environments in parallel via jax.vmap. All outputs are
    JAX arrays with a leading batch dimension of size `num_envs`.
    """

    def __init__(self, env_name: str, num_agents: int, max_steps: int, num_envs: int):
        self.env = jaxmarl.make(env_name, num_agents=num_agents)
        self.num_envs: int = num_envs
        self.num_agents: int = num_agents
        self.agents: List[str] = self.env.agents
        self.observation_size: int = self.env.observation_space(self.agents[0]).shape[0]
        self.observation_shape: Tuple[int, ...] = (self.observation_size,)
        self.action_space_size: int = self.env.action_space(self.agents[0]).n

        self._reset_fn = jax.vmap(self.env.reset)
        self._step_fn = jax.vmap(self.env.step)

    def reset(self, rng_keys: jax.Array) -> Tuple[jnp.ndarray, Any]:
        """
        Resets num_envs environments.

        Args:
            rng_keys: Shape (num_envs, 2)

        Returns:
            observations: Shape (num_envs, num_agents, observation_size)
            states: Batched environment states (pytree with leading num_envs axis).
        """
        obs_dict, states = self._reset_fn(rng_keys)
        return self._stack_obs(obs_dict), states

    def step(
        self, rng_keys: jax.Array, states: Any, actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Any, jnp.ndarray, jnp.ndarray]:
        """
        Steps num_envs environments.

        Args:
            rng_keys: Shape (num_envs, 2)
            states: Batched environment states.
            actions: Shape (num_envs, num_agents) — integer action indices.

        Returns:
            next_observations: Shape (num_envs, num_agents, observation_size)
            next_states: Batched environment states.
            team_rewards: Shape (num_envs,) — summed reward across agents.
            episode_dones: Shape (num_envs,) — True when all agents are done.
        """
        action_dicts = {
            agent: jnp.asarray(actions[:, i], dtype=jnp.int32)
            for i, agent in enumerate(self.agents)
        }
        next_obs_dict, next_states, rewards_dict, dones_dict, _ = self._step_fn(
            rng_keys, states, action_dicts
        )
        next_obs = self._stack_obs(next_obs_dict)
        rewards = jnp.stack([rewards_dict[a] for a in self.agents], axis=-1).sum(axis=-1)
        dones = jnp.stack([dones_dict[a] for a in self.agents], axis=-1).all(axis=-1)
        return next_obs, next_states, rewards, dones

    def _stack_obs(self, obs_dict: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Stacks per-agent observations into shape (num_envs, num_agents, obs_size)."""
        return jnp.stack(
            [obs_dict[agent].astype(jnp.float32) for agent in self.agents], axis=1
        )
