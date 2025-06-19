# replay_buffer.py
from collections import deque, namedtuple
import random
import numpy as np 

ReplayItem = namedtuple('ReplayItem', [
    'observation',   # (1, N, obs_dim)
    'actions',       # (U, N)  sequence of joint actions
    'policy_target', # (U+1, N, A)
    'value_target',  # (U+1, value_support_size)
    'reward_target'  # (U, reward_support_size)
])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, item: ReplayItem):
        self.buffer.append(item)
    
    def __len__(self):
        return len(self.buffer)
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        
        # We now use numpy for batching, making this buffer framework-agnostic.
        return ReplayItem(
            observation=np.concatenate([item.observation for item in batch], axis=0),
            actions=np.stack([item.actions for item in batch], axis=0),
            policy_target=np.stack([item.policy_target for item in batch], axis=0),
            value_target=np.stack([item.value_target for item in batch], axis=0),
            reward_target=np.stack([item.reward_target for item in batch], axis=0)
        )