# main.py
import ray
import time
import numpy as np
from collections import deque
import multiprocessing as mp
import os

from config import HYPERPARAMS

# No JAX imports in the driver script

@ray.remote
class ReplayBufferActor:
    """Actor for the replay buffer. Safe from JAX issues."""
    def __init__(self):
        # This part is fine
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
        # All imports are local to this actor's process, after it's on the GPU.
        import jax
        import optax
        from flax_model import FlaxMAMuZeroNet
        from train import train_step, MPEEnvWrapper
        print(f"(Learner pid={os.getpid()}) Initializing on GPU...")

        self.replay_buffer = replay_buffer_actor
        self.rng_key = jax.random.PRNGKey(42)
        env = MPEEnvWrapper(HYPERPARAMS["num_agents"], HYPERPARAMS["max_episode_steps"])
        model_kwargs = {k: v for k, v in HYPERPARAMS.items() if 'support' in k or 'hidden' in k or 'fc' in k or 'space' in k}
        model_kwargs['action_space_size'] = env.action_space_size
        self.model = FlaxMAMuZeroNet(num_agents=HYPERPARAMS["num_agents"], **model_kwargs)
        
        self.rng_key, init_key = jax.random.split(self.rng_key)
        dummy_obs = jax.numpy.ones((1, HYPERPARAMS["num_agents"], env.observation_size))
        self.params = self.model.init(init_key, dummy_obs)['params']

        lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=HYPERPARAMS["learning_rate"], warmup_steps=5000, decay_steps=HYPERPARAMS["num_episodes"] - 5000, end_value=HYPERPARAMS["learning_rate"]/10)
        self.optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(learning_rate=lr_schedule))
        self.opt_state = self.optimizer.init(self.params)
        self.jitted_train_step = jax.jit(train_step, static_argnames=['model', 'optimizer'])
        self.train_step_count = 0
        print(f"(Learner pid={os.getpid()}) Setup complete.")

    def train(self):
        import jax
        from replay_buffer import ReplayItem # needed for type casting

        # 1. Get a batch of NumPy arrays from the replay buffer
        numpy_batch = ray.get(self.replay_buffer.sample.remote(HYPERPARAMS["batch_size"]))
        
        # 2. THE FIX: Convert the NumPy batch to a JAX batch before training
        jax_batch = jax.tree_util.tree_map(lambda x: jax.device_put(x), numpy_batch)
        
        # 3. Train on the JAX batch
        self.params, self.opt_state, loss = self.jitted_train_step(self.model, self.optimizer, self.params, self.opt_state, jax_batch)
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
        
        # All imports are local to this new process
        import jax
        from mcts import MCTSPlanner
        from utils import DiscreteSupport
        from train import MPEEnvWrapper, process_episode
        from flax_model import FlaxMAMuZeroNet
        
        print(f"(DataActor pid={os.getpid()}) Initializing on CPU...")
        self.actor_id = actor_id
        self.learner = learner_actor
        self.replay_buffer = replay_buffer_actor
        
        self.rng_key = jax.random.PRNGKey(int(time.time()) + actor_id)
        self.env_wrapper = MPEEnvWrapper(HYPERPARAMS["num_agents"], HYPERPARAMS["max_episode_steps"])
        model_kwargs = {k: v for k, v in HYPERPARAMS.items() if 'support' in k or 'hidden' in k or 'fc' in k or 'space' in k}
        model_kwargs['action_space_size'] = self.env_wrapper.action_space_size
        model = FlaxMAMuZeroNet(num_agents=HYPERPARAMS["num_agents"], **model_kwargs)
        
        planner = MCTSPlanner(model=model, num_simulations=HYPERPARAMS["num_simulations"], mode=HYPERPARAMS["planner_mode"], num_joint_samples=HYPERPARAMS["num_joint_samples"], max_depth_gumbel_search=HYPERPARAMS["max_depth_gumbel_search"], num_gumbel_samples=HYPERPARAMS["num_gumbel_samples"])
        self.plan_fn = jax.jit(planner.plan)

        self.value_support = DiscreteSupport(min=-HYPERPARAMS["value_support_size"], max=HYPERPARAMS["value_support_size"])
        self.reward_support = DiscreteSupport(min=-HYPERPARAMS["reward_support_size"], max=HYPERPARAMS["reward_support_size"])
        self.process_episode = process_episode
        print(f"(DataActor pid={os.getpid()}) Setup complete.")

    def run_episode(self):
        import jax # Needed for jax.random
        params = ray.get(self.learner.get_params.remote())
        
        self.rng_key, episode_key, plan_key = jax.random.split(self.rng_key, 3)
        observation = self.env_wrapper.reset(episode_key)
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