# utils/replay_buffer.py
"""Prioritized experience replay, backed by cpprb.PrioritizedReplayBuffer."""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from jax import tree_util

from cpprb import PrioritizedReplayBuffer as _PRB


@dataclass
class Transition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    policy_target: np.ndarray
    value_target: float
    # Per-root-child Q-data for action-level AWPO.
    root_child_actions: np.ndarray  # (K, N) int32
    root_child_q: np.ndarray        # (K,) float32
    root_child_visits: np.ndarray   # (K,) float32


@dataclass
class Episode:
    trajectory: List[Transition] = field(default_factory=list)
    episode_return: float = 0.0

    def add_step(self, transition: Transition):
        self.trajectory.append(transition)
        self.episode_return += transition.reward


@dataclass
class ReplayItem:
    """
    A single, self-contained training sample for the MuZero model.

    Shapes (using shorthand):
        B = batch_size, U = unroll_steps, N = num_agents, A = action_space_size,
        K = num_sampled_actions
    """
    observation: np.ndarray    # (B, N, obs_size)
    actions: np.ndarray        # (B, U, N)
    policy_target: np.ndarray  # (B, U+1, N, A)
    value_target: np.ndarray   # (B, U+1, N)
    reward_target: np.ndarray  # (B, U, N)
    # Per-step Q-data for action-level AWPO at all U+1 positions.
    # Not part of the JAX pytree below — stored in ReplayBufferActor's
    # sidecar arrays, injected alongside the sampled batch at sample time.
    all_child_actions: np.ndarray  # (U+1, K, N) int32
    all_child_q: np.ndarray  # (U+1, K) float32
    all_child_visits: np.ndarray  # (U+1, K) float32


def flatten_replay_item(item: ReplayItem):
    children = (
        item.observation,
        item.actions,
        item.policy_target,
        item.value_target,
        item.reward_target,
    )
    return children, None


def unflatten_replay_item(static_data, children):
    # Q-data fields aren't part of this pytree (see ReplayItem docstring) —
    # explicitly None here since nothing reads them off a tree_map'd instance.
    return ReplayItem(
        observation=children[0],
        actions=children[1],
        policy_target=children[2],
        value_target=children[3],
        reward_target=children[4],
        all_child_actions=None,
        all_child_q=None,
        all_child_visits=None,
    )


tree_util.register_pytree_node(
    ReplayItem,
    flatten_replay_item,
    unflatten_replay_item,
)


class ReplayBuffer:
    """Prioritized experience replay, backed by cpprb.PrioritizedReplayBuffer."""

    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple,
        action_space_size: int,
        num_agents: int,
        unroll_steps: int,
        alpha: float,
        beta_start: float,
        beta_frames: float,
    ):
        self.capacity     = capacity
        self.alpha        = alpha
        self.beta_start   = beta_start
        self.beta_frames  = beta_frames
        self.frame_count  = 0

        self.observations    = np.zeros((capacity, num_agents, *observation_shape), dtype=np.float32)
        self.actions         = np.zeros((capacity, unroll_steps, num_agents),       dtype=np.int32)
        self.policy_targets  = np.zeros((capacity, unroll_steps + 1, num_agents, action_space_size), dtype=np.float32)
        self.value_targets   = np.zeros((capacity, unroll_steps + 1, num_agents),   dtype=np.float32)
        self.reward_targets  = np.zeros((capacity, unroll_steps, num_agents),        dtype=np.float32)
        self._priorities_log = np.zeros(capacity, dtype=np.float32)

        self._ptree = _PRB(
            capacity,
            env_dict={"_": {"shape": 1, "dtype": np.float32}},
            alpha=alpha,
        )

        self.pointer = 0
        self.size    = 0

    def add(self, item: ReplayItem, priority: float):
        priority = float(priority) if priority > 0 else (
            self._priorities_log[:self.size].max() if self.size > 0 else 1.0
        )
        idx = self.pointer
        self.observations[idx]    = item.observation
        self.actions[idx]         = item.actions
        self.policy_targets[idx]  = item.policy_target
        self.value_targets[idx]   = item.value_target
        self.reward_targets[idx]  = item.reward_target
        self._priorities_log[idx] = priority

        self._ptree.add(**{"_": np.zeros((1, 1), dtype=np.float32)},
                        priorities=np.array([priority], dtype=np.float32))

        self.pointer = (self.pointer + 1) % self.capacity
        self.size    = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[ReplayItem, np.ndarray, np.ndarray]:
        if self.size == 0:
            return None, None, None

        beta = min(1.0, self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames)
        self.frame_count += 1

        s       = self._ptree.sample(batch_size, beta=beta)
        indices = s["indexes"].astype(np.int64)
        weights = s["weights"].astype(np.float32)

        batch = ReplayItem(
            observation   = self.observations[indices],
            actions       = self.actions[indices],
            policy_target = self.policy_targets[indices],
            value_target  = self.value_targets[indices],
            reward_target = self.reward_targets[indices],
            all_child_actions=None,
            all_child_q=None,
            all_child_visits=None,
        )
        return batch, weights, indices

    def sample_for_reanalysis(self, batch_size: int):
        if self.size == 0:
            return None, None
        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False)
        return indices, self.observations[indices].copy()

    def update_targets(self, indices: np.ndarray, policy_targets: np.ndarray, root_values: np.ndarray):
        self.policy_targets[indices, 0] = policy_targets
        self.value_targets[indices, 0]  = root_values[:, None]

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        self._priorities_log[indices] = priorities
        self._ptree.update_priorities(indices, priorities)

    def get_stats(self) -> dict:
        if self.size == 0:
            return {"size": 0, "capacity": self.capacity, "fill_pct": 0.0}
        active = self._priorities_log[:self.size]
        beta   = min(1.0, self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames)
        return {
            "size":          self.size,
            "capacity":      self.capacity,
            "fill_pct":      100.0 * self.size / self.capacity,
            "priority_min":  float(active.min()),
            "priority_max":  float(active.max()),
            "priority_mean": float(active.mean()),
            "priority_std":  float(active.std()),
            "beta":          beta,
        }

    def __len__(self):
        return self.size


def process_episode(
    episode: "Episode",
    unroll_steps: int,
    n_step: int,
    discount_gamma: float,
    num_agents: int,
) -> list:
    """
    Converts a completed episode into ReplayItems via a sliding window.

    For each valid start position, computes n-step bootstrapped value targets
    and packs `unroll_steps` transitions into a single ReplayItem.

    Args:
        episode: Completed episode containing a trajectory of Transitions.
        unroll_steps: Number of steps per training sample (U).
        n_step: Lookahead horizon for bootstrapped value targets.
        discount_gamma: Discount factor γ.
        num_agents: Number of agents N (for broadcasting scalar targets).

    Returns:
        List of ReplayItems, one per valid start index.
        Empty list if the episode is too short to produce any samples.
    """
    trajectory = episode.trajectory
    ep_len = len(trajectory)

    if ep_len <= unroll_steps:
        return []

    # Extract arrays once to avoid repeated attribute lookups in the loop.
    observations = np.stack([t.observation for t in trajectory])      # (T, N, obs_size)
    actions = np.stack([t.action for t in trajectory])                # (T, N)
    policy_targets = np.stack([t.policy_target for t in trajectory])  # (T, N, A)
    rewards = np.array([t.reward for t in trajectory], dtype=np.float32)      # (T,)
    mcts_values = np.array([t.value_target for t in trajectory], dtype=np.float32)  # (T,)

    # Pre-compute discount coefficients [γ^0, γ^1, ..., γ^(n-1)] for np.dot.
    discount_vec = discount_gamma ** np.arange(n_step, dtype=np.float32)

    replay_items = []
    for start in range(ep_len - unroll_steps):
        # Compute n-step bootstrapped value target for each of the U+1 positions.
        value_targets = np.empty(unroll_steps + 1, dtype=np.float32)
        for i in range(unroll_steps + 1):
            t = start + i
            window = rewards[t : t + n_step]
            value_targets[i] = np.dot(window, discount_vec[: len(window)])
            bootstrap_idx = t + n_step
            if bootstrap_idx < ep_len:
                value_targets[i] += mcts_values[bootstrap_idx] * (discount_gamma ** n_step)

        # Broadcast scalar value/reward targets to per-agent arrays.
        # np.broadcast_to returns a read-only view; .copy() makes it writable.
        value_target_per_agent = np.broadcast_to(
            value_targets[:, None], (unroll_steps + 1, num_agents)
        ).copy().astype(np.float32)  # (U+1, N)

        reward_target_per_agent = np.broadcast_to(
            rewards[start : start + unroll_steps, None], (unroll_steps, num_agents)
        ).copy().astype(np.float32)  # (U, N)

        # Every Transition carries Q-data (the sole planner always produces it).
        all_child_actions = np.stack(
            [trajectory[start + i].root_child_actions for i in range(unroll_steps + 1)]
        )  # (U+1, K, N)
        all_child_q = np.stack(
            [trajectory[start + i].root_child_q for i in range(unroll_steps + 1)]
        )  # (U+1, K)
        all_child_visits = np.stack(
            [trajectory[start + i].root_child_visits for i in range(unroll_steps + 1)]
        )  # (U+1, K)

        replay_items.append(
            ReplayItem(
                observation=observations[start],                                    # (N, obs_size)
                actions=actions[start : start + unroll_steps],                     # (U, N)
                policy_target=policy_targets[start : start + unroll_steps + 1],   # (U+1, N, A)
                value_target=value_target_per_agent,                               # (U+1, N)
                reward_target=reward_target_per_agent,                             # (U, N)
                all_child_actions=all_child_actions,
                all_child_q=all_child_q,
                all_child_visits=all_child_visits,
            )
        )

    return replay_items
