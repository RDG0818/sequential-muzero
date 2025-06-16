# train.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import jax
import jax.numpy as jnp
import optax
import numpy as np
from collections import deque
import time
from mpe2 import simple_spread_v3
from flax_model import FlaxMAMuZeroNet
from mcts import MCTSPlanner
from replay_buffer import ReplayItem, ReplayBuffer
import utils
from utils import DiscreteSupport

HYPERPARAMS = { "planner_mode": "independent", 
               "num_episodes": 50000, 
               "warmup_episodes": 1000, 
               "log_interval": 1, 
               "num_agents": 3, 
               "max_episode_steps": 100, 
               "num_simulations": 30, 
               "num_joint_samples": 16, 
               "max_depth_gumbel_search": 10,
               "num_gumbel_samples": 10,
               "replay_buffer_size": 10000, 
               "batch_size": 256, 
               "learning_rate": 1e-4, 
               "unroll_steps": 5, 
               "discount_gamma": 0.99,
               "value_support_size": 300,
               "reward_support_size": 300,
               "hidden_state_size": 128, 
               "fc_representation_layers": (128,), 
               "fc_dynamic_layers": (128,), 
               "fc_reward_layers": (32,), 
               "fc_value_layers": (32,), 
               "fc_policy_layers": (32,)}


class MPEEnvWrapper:
    def __init__(self, num_agents, max_steps):
        self.env = simple_spread_v3.parallel_env(N=num_agents, max_cycles=max_steps, continuous_actions=False)
        self.num_agents = num_agents
        self.agents = self.env.possible_agents
        self.observation_size = self.env.observation_space(self.agents[0]).shape[0]
        self.action_space_size = self.env.action_space(self.agents[0]).n

    def reset(self, rng_key):
        obs_dict, _ = self.env.reset(seed=int(jax.random.randint(rng_key, (), 0, 1e9)))
        return self._stack_obs(obs_dict)

    def step(self, joint_action: np.ndarray):
        action_dict = {agent: action.item() for agent, action in zip(self.agents, joint_action)}
        next_obs_dict, reward_dict, done_dict, _, _ = self.env.step(action_dict)
        return self._stack_obs(next_obs_dict), sum(reward_dict.values()), all(done_dict.values())

    def _stack_obs(self, obs_dict):
        obs_list = [np.asarray(obs_dict[agent], dtype=np.float32) for agent in self.agents]
        return np.stack(obs_list, axis=0)[np.newaxis, ...]


def train_step(model, optimizer, params, opt_state, batch):
    def loss_fn(p):
        total_loss = 0.0
        # MuZero-style unrolled loss calculation
        hidden, _, p0_logits, v0_logits = model.apply(
            {'params': p}, batch.observation
        )

        policy_loss = optax.softmax_cross_entropy(p0_logits, batch.policy_target[:, 0, :, :]).mean()
        
        value_loss = optax.softmax_cross_entropy(v0_logits, batch.value_target[:, 0, :]).mean()

        total_loss += policy_loss + value_loss

        # Unroll dynamics for U steps
        for i in range(HYPERPARAMS["unroll_steps"]):
            ai = batch.actions[:, i, :] # (B, N)

            hidden, ri_logits, pi_logits, vi_logits = model.apply(
                {'params': p},
                hidden,     # previous latent
                ai,         # this step’s joint action
                method=model.recurrent_inference
            )

            reward_loss = optax.softmax_cross_entropy(ri_logits, batch.reward_target[:, i, :]).mean()

            p_loss = optax.softmax_cross_entropy(pi_logits, batch.policy_target[:, i+1, :, :]).mean()

            v_loss = optax.softmax_cross_entropy(vi_logits, batch.value_target[:, i+1, :]).mean()
            
            total_loss += reward_loss + p_loss + v_loss
        
        return total_loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss
jitted_train_step = jax.jit(train_step, static_argnames=['model', 'optimizer'])


def process_episode(episode_history: list, unroll_steps: int, discount_gamma: float, value_support: DiscreteSupport, reward_support: DiscreteSupport) -> 'ReplayItem':
    """
    Processes a completed episode to create a ReplayItem for the buffer.
    """
    observations = [step["observation"] for step in episode_history]
    actions = [step["actions"] for step in episode_history]
    policy_targets = [step["policy_target"] for step in episode_history]
    rewards = [step["reward"] for step in episode_history]

    discounted_returns = []
    current_return = 0.0
    for r in reversed(rewards):
        current_return = r + discount_gamma * current_return
        discounted_returns.append(current_return)
    discounted_returns.reverse()

    def pad_or_clip(sequence: list, length: int, pad_value):
        """Pads or clips a list to a target length."""
        if len(sequence) >= length:
            return sequence[:length]

        padding = [pad_value] * (length - len(sequence))
        return sequence + padding
    
    actions_padded = pad_or_clip(actions, unroll_steps, pad_value=actions[-1])
    policy_targets_padded = pad_or_clip(policy_targets, unroll_steps + 1, pad_value=policy_targets[-1])
    rewards_padded = pad_or_clip(rewards, unroll_steps, pad_value=0.0)
    returns_padded = pad_or_clip(discounted_returns, unroll_steps + 1, pad_value=0.0)

    actions_arr = np.stack(actions_padded, axis=0)
    policy_targets_arr = np.stack(policy_targets_padded, axis=0)
    rewards_arr = np.array(rewards_padded, dtype=np.float32)
    returns_arr = np.array(returns_padded, dtype=np.float32)

    value_target_dist = utils.scalar_to_support(jnp.asarray(returns_arr), value_support)
    reward_target_dist = utils.scalar_to_support(jnp.asarray(rewards_arr), reward_support)

    return ReplayItem(
        observation=observations[0],
        actions=actions_arr,
        policy_target=policy_targets_arr,
        value_target=np.asarray(value_target_dist),
        reward_target=np.asarray(reward_target_dist)
    )


def train():
    print(f"Starting training with mode: {HYPERPARAMS['planner_mode']}")
    rng_key = jax.random.PRNGKey(int(time.time()))
    rng_key, plan_key = jax.random.split(rng_key)

    env = MPEEnvWrapper(HYPERPARAMS["num_agents"], HYPERPARAMS["max_episode_steps"])
    model = FlaxMAMuZeroNet(
        num_agents=env.num_agents, action_space_size=env.action_space_size, value_support_size=HYPERPARAMS["value_support_size"], reward_support_size=HYPERPARAMS["reward_support_size"],
        hidden_state_size=HYPERPARAMS["hidden_state_size"], fc_representation_layers=HYPERPARAMS["fc_representation_layers"],
        fc_dynamic_layers=HYPERPARAMS["fc_dynamic_layers"], fc_reward_layers=HYPERPARAMS["fc_reward_layers"],
        fc_value_layers=HYPERPARAMS["fc_value_layers"], fc_policy_layers=HYPERPARAMS["fc_policy_layers"]
    )
    
    dummy_obs = jnp.ones((1, env.num_agents, env.observation_size))
    params = model.init(rng_key, dummy_obs)['params'] # Initializing params 
    
    lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=HYPERPARAMS["learning_rate"], warmup_steps=HYPERPARAMS["warmup_episodes"],
                                                     decay_steps=HYPERPARAMS["num_episodes"] - HYPERPARAMS["warmup_episodes"], end_value=HYPERPARAMS["learning_rate"]/10)

    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(learning_rate=lr_schedule))
    opt_state = optimizer.init(params)
    
    planner = MCTSPlanner(
        model=model, num_simulations=HYPERPARAMS["num_simulations"],
        mode=HYPERPARAMS["planner_mode"], num_joint_samples=HYPERPARAMS["num_joint_samples"], 
        max_depth_gumbel_search=HYPERPARAMS["max_depth_gumbel_search"], num_gumbel_samples=HYPERPARAMS["num_gumbel_samples"]
    )

    plan_key, subkey = jax.random.split(plan_key)
    dummy_obs = env.reset(plan_key)
    planner.plan(params, plan_key, dummy_obs) # Initializing planner
    plan_fn = planner.plan
    
    replay_buffer = ReplayBuffer(HYPERPARAMS["replay_buffer_size"])
    episode_returns = deque(maxlen=HYPERPARAMS['log_interval'])

    for episode in range(1, HYPERPARAMS["num_episodes"] + 1):
        rng_key, episode_key = jax.random.split(rng_key)
        observation = env.reset(episode_key)
        episode_history, episode_return = [], 0.0

        for _ in range(HYPERPARAMS["max_episode_steps"]):
            plan_key, subkey = jax.random.split(plan_key)
            plan_output = plan_fn(params, subkey, observation)

            action_np = np.asarray(plan_output.joint_action)
            
            episode_history.append({"observation": observation, "actions": action_np, "policy_target": np.asarray(plan_output.policy_targets)})
            observation, reward, done = env.step(action_np)
            episode_return += reward
            episode_history[-1]['reward'] = reward
            if done: break
        
        episode_returns.append(episode_return)

        replay_item = process_episode(
            episode_history,
            HYPERPARAMS["unroll_steps"],
            HYPERPARAMS["discount_gamma"],
            DiscreteSupport(min=-HYPERPARAMS["value_support_size"], max=HYPERPARAMS["value_support_size"]),
            DiscreteSupport(min=-HYPERPARAMS["reward_support_size"], max=HYPERPARAMS["reward_support_size"])
        )
        replay_buffer.add(replay_item)

        if len(replay_buffer) >= HYPERPARAMS["batch_size"]:
            batch = replay_buffer.sample(HYPERPARAMS["batch_size"])
            params, opt_state, loss = jitted_train_step(model, optimizer, params, opt_state, batch)

        if episode % HYPERPARAMS["log_interval"] == 0:
            avg_return = np.mean(np.array(episode_returns))
            print(f"Episode {episode} | Avg Return: {avg_return:.2f} | Loss: {loss.item() if 'loss' in locals() else 'N/A'}")

if __name__ == "__main__":
    train()