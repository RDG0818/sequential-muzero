# train.py

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

HYPERPARAMS = { "planner_mode": "independent", 
               "num_episodes": 5000, 
               "log_interval": 1, 
               "num_agents": 3, 
               "max_episode_steps": 100, 
               "num_simulations": 30, 
               "num_joint_samples": 16, 
               "replay_buffer_size": 10000, 
               "batch_size": 32, 
               "learning_rate": 1e-4, 
               "unroll_steps": 5, 
               "discount_gamma": 0.99, 
               "hidden_state_size": 128, 
               "fc_representation_layers": [128], 
               "fc_dynamic_layers": [128], 
               "fc_reward_layers": [32], 
               "fc_value_layers": [32], 
               "fc_policy_layers": [32]}

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

def train():
    print(f"Starting training with mode: {HYPERPARAMS['planner_mode']}")
    rng_key = jax.random.PRNGKey(int(time.time()))
    rng_key, plan_key = jax.random.split(rng_key)

    env = MPEEnvWrapper(HYPERPARAMS["num_agents"], HYPERPARAMS["max_episode_steps"])
    model = FlaxMAMuZeroNet(
        num_agents=env.num_agents, action_space_size=env.action_space_size,
        hidden_state_size=HYPERPARAMS["hidden_state_size"], fc_representation_layers=HYPERPARAMS["fc_representation_layers"],
        fc_dynamic_layers=HYPERPARAMS["fc_dynamic_layers"], fc_reward_layers=HYPERPARAMS["fc_reward_layers"],
        fc_value_layers=HYPERPARAMS["fc_value_layers"], fc_policy_layers=HYPERPARAMS["fc_policy_layers"]
    )
    dummy_obs = jnp.ones((1, env.num_agents, env.observation_size))
    params = model.init(rng_key, dummy_obs)['params']
    
    optimizer = optax.adam(learning_rate=HYPERPARAMS["learning_rate"])
    opt_state = optimizer.init(params)
    
    planner = MCTSPlanner(
        model=model, num_simulations=HYPERPARAMS["num_simulations"],
        mode=HYPERPARAMS["planner_mode"], num_joint_samples=HYPERPARAMS["num_joint_samples"]
    )

    plan_key, subkey = jax.random.split(plan_key)
    dummy_obs = env.reset(plan_key)
    planner.plan(params, plan_key, dummy_obs)
    plan_fn = planner.plan
    
    replay_buffer = ReplayBuffer(HYPERPARAMS["replay_buffer_size"])
    episode_returns = deque(maxlen=HYPERPARAMS['log_interval'])
    
    @jax.jit
    def train_step(params, opt_state, batch):
        def loss_fn(p):
            total_loss = 0.0
            # MuZero-style unrolled loss calculation
            hidden, _, p0_logits, v0 = model.apply(
                {'params': p}, batch.observation
            )

            pi0 = batch.policy_target[:, 0, :, :]  # (B, N, A)
            policy_loss = optax.softmax_cross_entropy(
                p0_logits, pi0
            ).mean()
            
            z0 = batch.value_target[:, 0].reshape(-1, 1)  # (B, 1)
            value_loss = optax.l2_loss(v0, z0).mean()

            total_loss += policy_loss + value_loss

            # Unroll dynamics for U steps
            for i in range(HYPERPARAMS["unroll_steps"]):
                ai = batch.actions[:, i, :] # (B, N)

                hidden, ri, pi_logits, vi = model.apply(
                    {'params': p},
                    hidden,     # previous latent
                    ai,         # this step’s joint action
                    method=model.recurrent_inference
                )

                r_i_target = batch.reward_target[:, i].reshape(-1, 1)  # (B,1)
                reward_loss = optax.l2_loss(ri, r_i_target).mean()

                pi_target = batch.policy_target[:, i+1, :, :]  # (B, N, A)
                p_loss = optax.softmax_cross_entropy(pi_logits, pi_target).mean()

                z_target = batch.value_target[:, i+1].reshape(-1, 1)  # (B,1)
                v_loss = optax.l2_loss(vi, z_target).mean()
                
                total_loss += reward_loss + p_loss + v_loss
            
            return total_loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

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
        
        # after accumulating episode_history and computing discounted values per step:
        U = HYPERPARAMS["unroll_steps"]
        T = len(episode_history)

        # 1) collect the arrays in chronological order
        obs_seq   = [e["observation"] for e in episode_history]   # list length T of (1,N,obs)
        act_seq   = [e["actions"]     for e in episode_history]   # list length T of (N,)
        pi_seq    = [e["policy_target"] for e in episode_history] # list length T of (N,A)
        rew_seq   = [e["reward"]      for e in episode_history]   # list length T of scalars
        val_seq   = []  # we’ll build the bootstrap values next

        # 2) compute the bootstrapped returns z_t for t=0…T−1
        ret = 0.0
        for r in reversed(rew_seq):
            ret = r + HYPERPARAMS["discount_gamma"] * ret
            val_seq.append(ret)
        val_seq = list(reversed(val_seq))  # now length T

        # 3) pad or trim each to length U or U+1
        def pad_or_clip(lst, length, pad_val):
            if len(lst) >= length:
                return lst[:length]
            return lst + [pad_val] * (length - len(lst))

        obs_pad   = pad_or_clip(obs_seq,   1,   obs_seq[-1])     # just need root obs
        act_pad   = pad_or_clip(act_seq,   U,   act_seq[-1])
        pi_pad    = pad_or_clip(pi_seq,    U+1, pi_seq[-1])
        rew_pad   = pad_or_clip(rew_seq,   U,   0.0)
        val_pad   = pad_or_clip(val_seq,   U+1, 0.0)

        # 4) build arrays
        obs_arr   = obs_pad[0]                     # (1, N, obs_dim)
        acts_arr  = np.stack(act_pad,  axis=0)     # (U, N)
        pis_arr   = np.stack(pi_pad,   axis=0)     # (U+1, N, A)
        rews_arr  = np.array(rew_pad)              # (U,)
        vals_arr  = np.array(val_pad)              # (U+1,)

        # 5) finally add to buffer
        replay_buffer.add(ReplayItem(
            observation=obs_arr,
            actions=acts_arr,
            policy_target=pis_arr,
            value_target=vals_arr,
            reward_target=rews_arr
        ))

        if len(replay_buffer) >= HYPERPARAMS["batch_size"]:
            batch = replay_buffer.sample(HYPERPARAMS["batch_size"])
            params, opt_state, loss = train_step(params, opt_state, batch)

        if episode % HYPERPARAMS["log_interval"] == 0:
            avg_return = np.mean(np.array(episode_returns))
            print(f"Episode {episode} | Avg Return: {avg_return:.2f} | Loss: {loss.item() if 'loss' in locals() else 'N/A'}")

if __name__ == "__main__":
    train()