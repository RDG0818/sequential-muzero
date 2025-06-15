from collections import deque, namedtuple
import random
import jax.numpy as jnp

ReplayItem = namedtuple('ReplayItem', [
    'observation',   # (1, N, obs_dim)
    'actions',       # (U, N)  sequence of joint actions
    'policy_target', # (U+1, N, A)
    'value_target',  # (U+1, value_support_size)
    'reward_target'  # (U, reward_support_size)
])

class ReplayBuffer:
    def __init__(self, capacity): self.buffer = deque(maxlen=capacity)
    def add(self, item: ReplayItem): self.buffer.append(item)
    def __len__(self): return len(self.buffer)
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return ReplayItem(
            observation=jnp.concatenate([item.observation for item in batch], axis=0), # (B, 1, N, obs)
            actions=jnp.stack([item.actions for item in batch], axis=0), # (B, U, N)
            policy_target=jnp.stack([item.policy_target for item in batch], axis=0), # (B, U+1, N, A)
            value_target=jnp.stack([item.value_target for item in batch], axis=0), # (B, U+1, value_support_size)
            reward_target=jnp.stack([item.reward_target for item in batch], axis=0) # (B, U, reward_support_size)
        )