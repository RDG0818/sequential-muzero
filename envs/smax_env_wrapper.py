import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
from jaxmarl.environments.smax import map_name_to_scenario


class SMAXEnvWrapper:
    """Single-environment SMAX wrapper. Allies vs scripted enemies (HeuristicEnemySMAX).

    Interface:
      reset(rng_key)            -> (obs: np.ndarray shape (1, N, obs_size), state)
      step(rng_key, state, acts) -> (obs, state, team_reward: float, done: bool)
      observation_size: int
      action_space_size: int

    env_name: SMAX scenario string, e.g. "3m", "2s3z", "8m".
    num_agents: must match the scenario's ally count (used for validation only).
    max_steps: used by DataActor loop guard; does not override the scenario horizon.
    """

    def __init__(self, env_name: str, num_agents: int, max_steps: int):
        scenario = map_name_to_scenario(env_name)
        self.env = jaxmarl.make("HeuristicEnemySMAX", scenario=scenario)
        # Sort agents by index so stacking order is deterministic.
        self.agents = sorted(self.env.agents, key=lambda a: int(a.split("_")[1]))
        self.num_agents = len(self.agents)
        self.observation_size = self.env.observation_space(self.agents[0]).shape[0]
        self.observation_shape = (self.observation_size,)
        self.action_space_size = self.env.action_space(self.agents[0]).n

    def reset(self, rng_key):
        obs_dict, state = self.env.reset(rng_key)
        return self._stack(obs_dict), state

    def step(self, rng_key, state, actions):
        # actions: (N,) int array, one per ally in sorted order.
        action_dict = {a: int(actions[i]) for i, a in enumerate(self.agents)}
        next_obs_dict, next_state, reward, done, _ = self.env.step(rng_key, state, action_dict)
        next_obs_dict = {k: np.array(v) for k, v in next_obs_dict.items()}
        # All allies receive the same team reward — take one; do NOT sum (would overcount by N).
        team_reward = float(np.array(reward[self.agents[0]]))
        # Use "__all__" for episode termination; individual done=True means agent is dead,
        # not that the episode is over.
        episode_done = bool(np.array(done["__all__"]))
        return self._stack(next_obs_dict), next_state, team_reward, episode_done

    def _stack(self, obs_dict):
        return np.stack(
            [np.asarray(obs_dict[a], dtype=np.float32) for a in self.agents], axis=0
        )[np.newaxis, ...]  # (1, N, obs_size)


class VecSMAXEnvWrapper:
    """Vectorized SMAX wrapper using jax.vmap over num_envs parallel environments.

    Interface:
      reset(rng_keys)                   -> (obs: (B, N, obs_size), states)
      step(rng_keys, states, actions)   -> (obs, states, rewards: (B,), dones: (B,))
    """

    def __init__(self, env_name: str, num_agents: int, max_steps: int, num_envs: int):
        scenario = map_name_to_scenario(env_name)
        self.env = jaxmarl.make("HeuristicEnemySMAX", scenario=scenario)
        self.num_envs = num_envs
        self.agents = sorted(self.env.agents, key=lambda a: int(a.split("_")[1]))
        self.num_agents = len(self.agents)
        self.observation_size = self.env.observation_space(self.agents[0]).shape[0]
        self.observation_shape = (self.observation_size,)
        self.action_space_size = self.env.action_space(self.agents[0]).n
        self._reset_fn = jax.vmap(self.env.reset)
        self._step_fn = jax.vmap(self.env.step)

    def reset(self, rng_keys):
        # rng_keys: (B, 2)
        obs_dict, states = self._reset_fn(rng_keys)
        return self._stack(obs_dict), states  # (B, N, obs_size)

    def step(self, rng_keys, states, actions):
        # actions: (B, N) int array.
        action_dict = {
            a: jnp.asarray(actions[:, i], dtype=jnp.int32)
            for i, a in enumerate(self.agents)
        }
        next_obs_dict, next_states, rewards_dict, dones_dict, info = self._step_fn(
            rng_keys, states, action_dict
        )
        # All allies share the same team reward — take one ally's reward, not the sum.
        rewards = rewards_dict[self.agents[0]]  # (B,)
        dones = dones_dict["__all__"]           # (B,)
        # JaxMARL auto-resets after done=True, so next_states is already the reset
        # state (time=0, all alive) — we cannot read unit_alive to detect wins.
        # Instead: won_battle_bonus=1.0 is added to the terminal step reward, while
        # max per-step HP damage is ~0.13, so reward > 0.5 at done is a reliable win.
        won = dones & (rewards > 0.5)
        return self._stack(next_obs_dict), next_states, rewards, dones, won

    def _stack(self, obs_dict):
        return jnp.stack(
            [obs_dict[a].astype(jnp.float32) for a in self.agents], axis=1
        )  # (B, N, obs_size)
