# main.py
import ray
import time
import numpy as np
from collections import deque
import multiprocessing as mp
import os
from replay_buffer import ReplayItem

from config import HYPERPARAMS

@ray.remote
class ReplayBufferActor:
    """Actor for the replay buffer. Safe from JAX issues."""
    def __init__(self):
        from replay_buffer import ReplayBuffer
        self.replay_buffer = ReplayBuffer(HYPERPARAMS["replay_buffer_size"])
    
    def add(self, item):
        self.replay_buffer.add(item)
    
    def get_size(self):
        return len(self.replay_buffer)
    
    def sample(self, batch_size):
        return self.replay_buffer.sample(batch_size)


@ray.remote(num_gpus=1)
class LearnerActor:
    """Learner actor that lives on the GPU."""
    def __init__(self, replay_buffer_actor):
        import jax
        import optax
        from flax_model import FlaxMAMuZeroNet
        from jaxmarl_env_wrapper import JaxMARLEnvWrapper
        print(f"(Learner pid={os.getpid()}) Initializing on GPU...")

        self.replay_buffer = replay_buffer_actor
        self.rng_key = jax.random.PRNGKey(42)
        env = JaxMARLEnvWrapper("MPE_simple_spread_v3", HYPERPARAMS["num_agents"], HYPERPARAMS["max_episode_steps"])
        model_kwargs = {k: v for k, v in HYPERPARAMS.items() if 'support' in k or 'hidden' in k or 'fc' in k or 'space' in k}
        model_kwargs['action_space_size'] = env.action_space_size
        self.model = FlaxMAMuZeroNet(num_agents=HYPERPARAMS["num_agents"], **model_kwargs)
        
        self.rng_key, init_key = jax.random.split(self.rng_key)
        dummy_obs = jax.numpy.ones((1, HYPERPARAMS["num_agents"], env.observation_size))
        self.params = self.model.init(init_key, dummy_obs)['params']

        lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=HYPERPARAMS["learning_rate"], warmup_steps=5000, decay_steps=HYPERPARAMS["num_episodes"] - 5000, end_value=HYPERPARAMS["learning_rate"]/10)
        self.optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(learning_rate=lr_schedule))
        self.opt_state = self.optimizer.init(self.params)
        self.jitted_train_step = jax.jit(self.train_step, static_argnames=['model', 'optimizer'])
        self.train_step_count = 0
        print(f"(Learner pid={os.getpid()}) Setup complete.")

    def train_step(self, model, optimizer, params, opt_state, batch, rng_key):
        import jax
        import optax
        loss_rng, new_train_rng = jax.random.split(rng_key)
        def loss_fn(p):
            total_loss = 0.0
            dropout_rng, initial_rng, unroll_rng = jax.random.split(loss_rng, 3)
            unroll_keys = jax.random.split(unroll_rng, HYPERPARAMS["unroll_steps"])
            # MuZero-style unrolled loss calculation
            hidden, _, p0_logits, v0_logits = model.apply(
                {'params': p}, batch.observation,
                rngs={'dropout': initial_rng}
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
                    method=model.recurrent_inference,
                    rngs={'dropout': unroll_keys[i]}
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

    def train(self):
        import jax

        self.rng_key, train_key = jax.random.split(self.rng_key)

        numpy_batch = ray.get(self.replay_buffer.sample.remote(HYPERPARAMS["batch_size"]))
        
        jax_batch = jax.tree_util.tree_map(lambda x: jax.device_put(x), numpy_batch)

        self.params, self.opt_state, loss = self.jitted_train_step(self.model, self.optimizer, self.params, self.opt_state, jax_batch, train_key)
        self.train_step_count += 1
        return loss.item()
        
    def get_params(self): return self.params
    def get_train_step_count(self): return self.train_step_count


@ray.remote(num_cpus=1)
class DataActor:
    """Data generating actor that initializes all components within its own process on the CPU."""
    def __init__(self, actor_id, learner_actor, replay_buffer_actor):
        # CRITICAL: Set environment variables BEFORE any other imports, especially JAX
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['JAX_PLATFORMS'] = 'cpu' # This is the key fix
        
        import jax
        from mcts.mcts_independent import MCTSPlanner
        from mcts.mcts_joint import MCTSJointPlanner
        from mcts.mcts_sequential import MCTSSequentialPlanner
        from utils import DiscreteSupport
        from jaxmarl_env_wrapper import JaxMARLEnvWrapper
        from flax_model import FlaxMAMuZeroNet
        
        print(f"(DataActor pid={os.getpid()}) Initializing on CPU...")
        self.actor_id = actor_id
        self.learner = learner_actor
        self.replay_buffer = replay_buffer_actor
        
        self.rng_key = jax.random.PRNGKey(int(time.time()) + actor_id)
        self.env_wrapper = JaxMARLEnvWrapper("MPE_simple_spread_v3", HYPERPARAMS["num_agents"], HYPERPARAMS["max_episode_steps"])
        model_kwargs = {k: v for k, v in HYPERPARAMS.items() if 'support' in k or 'hidden' in k or 'fc' in k or 'space' in k}
        model_kwargs['action_space_size'] = self.env_wrapper.action_space_size
        model = FlaxMAMuZeroNet(num_agents=HYPERPARAMS["num_agents"], **model_kwargs)
        
        if HYPERPARAMS["planner_mode"] == "independent": planner = MCTSPlanner(model=model, num_simulations=HYPERPARAMS["num_simulations"], max_depth_gumbel_search=HYPERPARAMS["max_depth_gumbel_search"], num_gumbel_samples=HYPERPARAMS["num_gumbel_samples"])
        elif HYPERPARAMS["planner_mode"] == "joint": planner = MCTSJointPlanner(model=model, num_simulations=HYPERPARAMS["num_simulations"], max_depth_gumbel_search=HYPERPARAMS["max_depth_gumbel_search"], num_gumbel_samples=HYPERPARAMS["num_gumbel_samples"])
        elif HYPERPARAMS["planner_mode"] == "sequential": planner = MCTSSequentialPlanner(model=model, num_simulations=HYPERPARAMS["num_simulations"], max_depth_gumbel_search=HYPERPARAMS["max_depth_gumbel_search"], num_gumbel_samples=HYPERPARAMS["num_gumbel_samples"])
        else: raise ValueError(f"Invalid planner mode: {HYPERPARAMS['planner_mode']}") 
        self.plan_fn = jax.jit(planner.plan)

        self.value_support = DiscreteSupport(min=-HYPERPARAMS["value_support_size"], max=HYPERPARAMS["value_support_size"])
        self.reward_support = DiscreteSupport(min=-HYPERPARAMS["reward_support_size"], max=HYPERPARAMS["reward_support_size"])
        print(f"(DataActor pid={os.getpid()}) Setup complete.")

    def process_episode(self, episode_history: list, unroll_steps: int, discount_gamma: float, value_support, reward_support):
        import utils
        from utils import DiscreteSupport
        import jax.numpy as jnp
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

    def run_episode(self):
        import jax # Needed for jax.random
        params = ray.get(self.learner.get_params.remote())
        
        self.rng_key, episode_key, plan_key = jax.random.split(self.rng_key, 3)
        observation = self.env_wrapper.reset()
        episode_history, episode_return = [], 0.0

        for _ in range(HYPERPARAMS["max_episode_steps"]):
            plan_output = self.plan_fn(params, plan_key, observation)
            action_np = np.asarray(plan_output.joint_action)
            episode_history.append({"observation": observation, "actions": action_np, "policy_target": np.asarray(plan_output.policy_targets)})
            observation, reward, done = self.env_wrapper.step(action_np)
            episode_return += reward
            episode_history[-1]['reward'] = reward
            if done: break
        
        replay_item = self.process_episode(episode_history, HYPERPARAMS["unroll_steps"], HYPERPARAMS["discount_gamma"], self.value_support, self.reward_support)
        self.replay_buffer.add.remote(replay_item)
        return episode_return


def main():
    ray.init(ignore_reinit_error=True)
    print(f"Ray cluster started. Available resources: {ray.available_resources()}")
    
    replay_buffer = ReplayBufferActor.remote()
    learner = LearnerActor.remote(replay_buffer)
    actors = [DataActor.remote(i, learner, replay_buffer) for i in range(HYPERPARAMS["num_actors"])]
    
    print("Waiting for actors to initialize and run one episode...")
    # This ensures the __init__ methods have completed before we proceed.
    ray.get([actor.run_episode.remote() for actor in actors]) # Bug Fix 1: Correctly iterate over 'actors'

    print("\nWarmup phase...")
    # Use a dictionary to map running tasks (ObjectRefs) to the actor that started them
    actor_tasks = {actor.run_episode.remote(): actor for actor in actors}
    while ray.get(replay_buffer.get_size.remote()) < HYPERPARAMS["warmup_episodes"]:
        # Wait for any single actor to finish its episode
        done_refs, _ = ray.wait(list(actor_tasks.keys()), num_returns=1)
        done_ref = done_refs[0]

        # Get the actor that just finished
        finished_actor = actor_tasks.pop(done_ref)
        
        # Start a new episode on the same actor and add it back to our task dict
        actor_tasks[finished_actor.run_episode.remote()] = finished_actor

        print(f"  Buffer size: {ray.get(replay_buffer.get_size.remote())}/{HYPERPARAMS['warmup_episodes']}", end="\r")

    print("\nWarmup complete. Starting main training loop.")
    
    episodes_processed = ray.get(replay_buffer.get_size.remote())
    returns = deque(maxlen=HYPERPARAMS['log_interval'])
    losses = deque(maxlen=HYPERPARAMS['log_interval'])
    start_time = time.time()
    
    # Start the first training step
    learner_task = learner.train.remote()

    while episodes_processed < HYPERPARAMS["num_episodes"]:
        # Wait for any single actor to finish its episode
        done_actor_refs, _ = ray.wait(list(actor_tasks.keys()), num_returns=1)
        done_actor_ref = done_actor_refs[0]
        
        # Get the result from the finished actor task
        ep_return = ray.get(done_actor_ref)
        returns.append(ep_return)
        episodes_processed += 1
        
        # Restart the actor that just finished
        finished_actor = actor_tasks.pop(done_actor_ref)
        actor_tasks[finished_actor.run_episode.remote()] = finished_actor

        # Check if the learner is done with its training step
        done_learner_refs, _ = ray.wait([learner_task], timeout=0)
        if done_learner_refs:
            losses.append(ray.get(done_learner_refs[0]))
            # Start the next training step
            learner_task = learner.train.remote()

        if episodes_processed % HYPERPARAMS['log_interval'] == 0 and returns:
            avg_return = np.mean(returns)
            avg_loss = np.mean(losses) if losses else 0.0

            print(f"Episodes: {episodes_processed} | Avg Return: {avg_return:.2f} | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.2f}s")
            start_time = time.time() 
    
    print("Training finished.")
    ray.shutdown()
    
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass
    main()