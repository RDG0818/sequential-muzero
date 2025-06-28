# replay_buffer.py
from collections import deque
import random
import numpy as np 
from dataclasses import dataclass, field
from typing import List
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
    actions: np.ndarray  # Shape: (unroll_steps, num_actions)
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
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add_batch(self, items: list[ReplayItem]):
        self.buffer.extend(items)
    
    def __len__(self):
        return len(self.buffer)
    
    def sample(self, batch_size: int) -> ReplayItem:
        batch = random.sample(self.buffer, batch_size)

        return ReplayItem(
            observation=np.stack([item.observation for item in batch], axis=0),
            actions=np.stack([item.actions for item in batch], axis=0),
            target_observation=np.stack([item.target_observation for item in batch], axis=0),
            policy_target=np.stack([item.policy_target for item in batch], axis=0),
            value_target=np.stack([item.value_target for item in batch], axis=0),
            reward_target=np.stack([item.reward_target for item in batch], axis=0)
        )