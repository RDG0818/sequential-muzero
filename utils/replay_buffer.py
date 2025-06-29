# replay_buffer.py
from collections import deque
import random
import numpy as np 
from dataclasses import dataclass, field
from typing import List, Tuple
from jax import tree_util

@dataclass
class Transition:
    """
    Holds all the data for a single step (or transition) in an environment.
    """
    observation: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    policy_target: np.ndarray
    value_target: float

@dataclass
class Episode:
    """
    A container for a full episode's trajectory and metadata.
    """
    trajectory: List[Transition] = field(default_factory=list)
    episode_return: float = 0.0

    def add_step(self, transition: Transition):
        """A clean method to add a step to the episode's trajectory."""
        self.trajectory.append(transition)
        self.episode_return += transition.reward

@dataclass
class ReplayItem:
    """
    A single, self-contained training sample for the MuZero model.
    """

    observation: np.ndarray
    actions: np.ndarray  # Shape: (unroll_steps, action_space_size)
    target_observation: np.ndarray
    policy_target: np.ndarray    # Shape: (unroll_steps + 1, action_space_size)
    value_target: np.ndarray     # Shape: (unroll_steps + 1,)
    reward_target: np.ndarray    # Shape: (unroll_steps,)

def flatten_replay_item(item: ReplayItem):
    """
    Defines how to flatten the ReplayItem.
    Returns a tuple of the dynamic children and a tuple of the static data.
    """
    children = (
        item.observation,
        item.actions,
        item.target_observation,
        item.policy_target,
        item.value_target,
        item.reward_target,
    )
    # No static data needed for this class, so we return None.
    static_data = None
    return children, static_data

def unflatten_replay_item(static_data, children):
    """
    Defines how to unflatten the ReplayItem from its children.
    """
    # The order must match the order in flatten_replay_item
    return ReplayItem(
        observation=children[0],
        actions=children[1],
        target_observation=children[2],
        policy_target=children[3],
        value_target=children[4],
        reward_target=children[5],
    )

tree_util.register_pytree_node(
    ReplayItem,
    flatten_replay_item,
    unflatten_replay_item
)

class ReplayBuffer:
    """
    A replay buffer with prioritized experience replay.
    """
    def __init__(self, capacity: int, observation_space: Tuple, action_space_size: int, num_agents: int, unroll_steps: int,
                 alpha: float, beta_start: float, beta_frames: int):
        """
        Initializes the ReplayBuffer.
        Args:
            capacity: The maximum number of items to store in the buffer.
            observation_space: The shape of a single observation.
            num_agents: The number of agents
            unroll_steps: The number of steps to unroll for each training sample.
            alpha: The exponent for calculating priorities. 0 means uniform sampling.
            beta_start: The initial value of beta for importance sampling.
            beta_frames: The number of frames over which to anneal beta to 1.0.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_count = 0

        self.observations = np.zeros((capacity, num_agents, *observation_space), dtype=np.float32)
        self.actions = np.zeros((capacity, unroll_steps, num_agents), dtype=np.int32)
        self.target_observations = np.zeros((capacity, *observation_space), dtype=np.float32)
        self.policy_targets = np.zeros((capacity, unroll_steps + 1, num_agents, action_space_size), dtype=np.float32)
        self.value_targets = np.zeros((capacity, unroll_steps + 1, num_agents), dtype=np.float32)
        self.reward_targets = np.zeros((capacity, unroll_steps, num_agents), dtype=np.float32)

        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pointer = 0
        self.size = 0

    def add(self, item: ReplayItem, priority: float):
        """
        Adds a new ReplayItem to the buffer.
        Args:
            item: The ReplayItem to add.
            priority: The initial priority for the item.
        """
        self.observations[self.pointer] = item.observation
        self.actions[self.pointer] = item.actions
        self.target_observations[self.pointer] = item.target_observation
        self.policy_targets[self.pointer] = item.policy_target
        self.value_targets[self.pointer] = item.value_target
        self.reward_targets[self.pointer] = item.reward_target
        self.priorities[self.pointer] = priority

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_beta(self) -> float:
        """Calculates the current value of beta."""
        beta = self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames
        self.frame_count += 1
        return min(1.0, beta)

    def sample(self, batch_size: int) -> Tuple[ReplayItem, np.ndarray, np.ndarray]:
        """
        Samples a batch of ReplayItems from the buffer using prioritized sampling.
        Args:
            batch_size: The number of items to sample.
        Returns:
            A tuple containing:
                - A ReplayItem containing the batched data.
                - The importance sampling weights for the batch.
                - The indices of the sampled items.
        """
        if self.size == 0:
            return None, None, None

        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        beta = self._get_beta()
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        # B = batch_size
        # U = unroll_steps
        # O = observation_shape 
        # A = action_space_size
        # N = number of agents
        # value = dimension of the value (typically 1)
        # reward = dimension of the reward (typically 1)

        batch = ReplayItem(
            observation=self.observations[indices], # Shape: (B, *O)
            actions=self.actions[indices], # Shape: (B, U, A)
            target_observation=self.target_observations[indices], # Shape: (B, *O)
            policy_target=self.policy_targets[indices], # Shape: (B, U+1, A)
            value_target=self.value_targets[indices], # Shape (B, U+1, N, value)
            reward_target=self.reward_targets[indices] # Shape (B, U, N, reward)
        )

        return batch, weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Updates the priorities of the sampled items.
        Args:
            indices: The indices of the items to update.
            priorities: The new priorities for the items.
        """
        self.priorities[indices] = priorities

    def __len__(self):
        return self.size