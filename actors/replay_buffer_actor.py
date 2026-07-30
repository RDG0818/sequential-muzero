import numpy as np
from typing import List

import ray

from config import ExperimentConfig


@ray.remote
class ReplayBufferActor:
    """Manages prioritized experience replay in a dedicated process."""

    def __init__(self, obs_size: int, action_size: int, config: ExperimentConfig):
        from utils.replay_buffer import ReplayBuffer

        self.buffer = ReplayBuffer(
            capacity=config.train.replay_buffer_size,
            observation_shape=(obs_size,),
            action_space_size=action_size,
            num_agents=config.train.num_agents,
            unroll_steps=config.train.unroll_steps,
            alpha=config.train.replay_buffer_alpha,
            beta_start=config.train.replay_buffer_beta_start,
            beta_frames=config.train.replay_buffer_beta_frames,
        )

        # Sidecar arrays for per-step Q-data at ALL U+1 unroll positions.
        # Indexed by ring-buffer position (add_counter % capacity).
        # Used by the learner for action-level AWPO at every unroll step.
        cap = config.train.replay_buffer_size
        K   = config.mcts.num_gumbel_samples
        N   = config.train.num_agents
        U   = config.train.unroll_steps
        self._q_actions = np.zeros((cap, U + 1, K, N), dtype=np.int32)
        self._q_values  = np.zeros((cap, U + 1, K),    dtype=np.float32)
        self._q_visits  = np.zeros((cap, U + 1, K),    dtype=np.float32)
        self._q_valid   = np.zeros((cap, U + 1),       dtype=bool)
        self._capacity  = cap
        self._add_counter = 0

    def add(self, items: list, priorities: List[float]):
        for item, priority in zip(items, priorities):
            slot = self._add_counter % self._capacity
            self.buffer.add(item, priority)
            self._q_actions[slot] = item.all_child_actions  # (U+1, K, N)
            self._q_values[slot]  = item.all_child_q        # (U+1, K)
            self._q_visits[slot]  = item.all_child_visits   # (U+1, K)
            self._q_valid[slot]   = item.all_child_valid    # (U+1,) bool
            self._add_counter += 1

    def sample(self, batch_size: int):
        result = self.buffer.sample(batch_size)
        if result[0] is None:
            return None, None, None, None
        batch, weights, indices = result
        # Attach sidecar Q-data for the sampled indices.
        # Items that were added before the sidecar was active have q_valid=False.
        q_data = {
            "all_child_actions": self._q_actions[indices],  # (B, U+1, K, N)
            "all_child_q":       self._q_values[indices],   # (B, U+1, K)
            "all_child_visits":  self._q_visits[indices],   # (B, U+1, K)
            "all_child_valid":   self._q_valid[indices],    # (B, U+1) bool
        }
        return batch, weights, indices, q_data

    def sample_for_reanalysis(self, batch_size: int):
        return self.buffer.sample_for_reanalysis(batch_size)

    def update_targets(self, indices, policy_targets, root_values):
        self.buffer.update_targets(indices, policy_targets, root_values)

    def update_root_q(self, indices, child_actions, child_q, child_visits):
        """Update position-0 Q-data sidecar for reanalyzed items."""
        self._q_actions[indices, 0] = child_actions  # (B, K, N)
        self._q_values[indices, 0]  = child_q        # (B, K)
        self._q_visits[indices, 0]  = child_visits   # (B, K)
        self._q_valid[indices, 0]   = True

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        self.buffer.update_priorities(indices, priorities)

    def get_size(self) -> int:
        return len(self.buffer)

    def get_stats(self) -> dict:
        return self.buffer.get_stats()
