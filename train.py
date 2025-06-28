# main.py
import ray
import time
import numpy as np
from collections import deque
import multiprocessing as mp
import os
import logging
from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from utils.replay_buffer import Episode, Transition, ReplayItem

from config import CONFIG
import wandb

@ray.remote
class ReplayBufferActor:
    def __init__(self):
        from utils.replay_buffer import ReplayBuffer
        self.replay_buffer = ReplayBuffer(CONFIG.train.replay_buffer_size)

    def add_batch(self, items: list["ReplayItem"]):
        self.replay_buffer.add_batch(items)

    def get_size(self):
        return len(self.replay_buffer)
    
    def sample(self, batch_size):
        return self.replay_buffer.sample(batch_size)


@ray.remote(num_gpus=1)
class LearnerActor:
    def __init__(self, replay_buffer_actor):
        import jax
        import optax
        from utils.utils import DiscreteSupport
        from model.model import FlaxMAMuZeroNet
        from utils.env_wrapper import EnvWrapper
        print(f"(Learner pid={os.getpid()}) Initializing on GPU...")
        
        # Basic setup
        self.replay_buffer = replay_buffer_actor
        N, lr = CONFIG.train.num_agents, CONFIG.train.learning_rate
        self.rng_key = jax.random.PRNGKey(42)
        self.train_step_count = 0
        self.value_support = DiscreteSupport(min=-CONFIG.model.value_support_size, max=CONFIG.model.value_support_size)
        self.reward_support = DiscreteSupport(min=-CONFIG.model.reward_support_size, max=CONFIG.model.reward_support_size)

        # env and model setup
        env = EnvWrapper(CONFIG.train.env_name, N, CONFIG.train.max_episode_steps)
        self.model = FlaxMAMuZeroNet(CONFIG.model, env.action_space_size)

        # Initialize model parameters
        self.rng_key, init_key = jax.random.split(self.rng_key)
        dummy_obs = jax.numpy.ones((1, N, env.observation_size))
        self.params = self.model.init(init_key, dummy_obs)['params']

        # Optimizer setup
        self.lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=lr, warmup_steps=CONFIG.train.lr_warmup_steps,
            decay_steps=CONFIG.train.num_episodes - CONFIG.train.lr_warmup_steps, end_value=lr * CONFIG.train.end_lr_factor)
        self.optimizer = optax.chain(optax.clip_by_global_norm(CONFIG.train.gradient_clip_norm), optax.adamw(learning_rate=self.lr_schedule))
        self.opt_state = self.optimizer.init(self.params)

        self.jitted_train_step = jax.jit(self.train_step, static_argnames=['model', 'optimizer', 'value_support', 'reward_support'])

        print(f"(Learner pid={os.getpid()}) Setup complete.")

    @staticmethod
    def train_step(model, optimizer, params, opt_state, batch, rng_key, value_support, reward_support):
        import jax
        import jax.numpy as jnp
        import optax
        from utils import utils

        U = CONFIG.train.unroll_steps

        value_target_dist = utils.scalar_to_support(batch.value_target.mean(axis=2), value_support)
        reward_target_dist = utils.scalar_to_support(batch.reward_target.mean(axis=2), reward_support)
        
        def loss_fn(p):
            reward_loss, policy_loss, value_loss = 0.0, 0.0, 0.0

            dropout_rng, initial_rng, unroll_rng = jax.random.split(rng_key, 3)
            unroll_keys = jax.random.split(unroll_rng, U)

            reshaped_obs = batch.observation[:, 0]

            model_output = model.apply(
                {'params': p}, reshaped_obs,
                rngs={'dropout': initial_rng}
            )

            hidden, _, p0_logits, v0_logits = model_output.hidden_state, model_output.reward_logits, model_output.policy_logits, model_output.value_logits

            policy_loss += optax.softmax_cross_entropy(p0_logits, batch.policy_target[:, 0]).mean()
            value_loss += utils.categorical_cross_entropy_loss(v0_logits, value_target_dist[:, 0]).mean()

            # Unrolled Loss Calculations
            for i in range(U):
                ai = batch.actions[:, i] # (B, N)

                model_output = model.apply(
                    {'params': p},
                    hidden,     # previous latent
                    ai,         # this step’s joint action
                    method=model.recurrent_inference,
                    rngs={'dropout': unroll_keys[i]}
                )
                hidden, ri_logits, pi_logits, vi_logits = model_output.hidden_state, model_output.reward_logits, model_output.policy_logits, model_output.value_logits

                reward_loss += utils.categorical_cross_entropy_loss(ri_logits, reward_target_dist[:, i]).mean()
                policy_loss += optax.softmax_cross_entropy(pi_logits, batch.policy_target[:, i+1]).mean()
                value_loss += utils.categorical_cross_entropy_loss(vi_logits, value_target_dist[:, i+1]).mean()

                # TODO: Add the SimSiam/Consistency loss here 
            
            policy_loss /= (U + 1)
            value_loss /= (U + 1)
            reward_loss /= (U + 1e-8)

            total_loss = reward_loss + policy_loss + value_loss * CONFIG.train.value_loss_coefficient

            metrics = {
                "total_loss": total_loss,
                "reward_loss": reward_loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss
            }

            return total_loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        metrics['grad_norm'] = optax.global_norm(grads)
        
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, metrics

    def train(self):
        import jax

        # Sample a batch from the replay buffer
        self.rng_key, train_key = jax.random.split(self.rng_key)
        numpy_batch = ray.get(self.replay_buffer.sample.remote(CONFIG.train.batch_size))
        jax_batch = jax.tree_util.tree_map(lambda x: jax.device_put(x), numpy_batch)

        # Perform a training step
        self.params, self.opt_state, metrics = self.jitted_train_step(self.model, self.optimizer, self.params, self.opt_state, jax_batch, train_key, 
                                                                      self.value_support, self.reward_support)
        self.train_step_count += 1

        # Convert metrics to standard Python types for logging
        metrics = {k: v.item() for k, v in metrics.items()}
        metrics['learning_rate'] = self.lr_schedule(self.train_step_count).item()

        return metrics
        
    def get_params(self): return self.params
    def get_train_step_count(self): return self.train_step_count


@ray.remote(num_cpus=1)
class DataActor:
    def __init__(self, actor_id, learner_actor, replay_buffer_actor):
        # Force CPU usage for this actor
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['JAX_PLATFORMS'] = 'cpu'
        
        import jax
        from mcts.mcts_independent import MCTSPlanner
        from mcts.mcts_joint import MCTSJointPlanner
        from mcts.mcts_sequential import MCTSSequentialPlanner
        from utils.env_wrapper import EnvWrapper
        from model.model import FlaxMAMuZeroNet
        
        print(f"(DataActor pid={os.getpid()}) Initializing on CPU...")

        # Basic setup
        self.actor_id = actor_id
        self.learner = learner_actor
        self.replay_buffer = replay_buffer_actor

        self.params = ray.get(self.learner.get_params.remote())
        self.episodes_since_update = 0
        
        # Env and model setup
        self.rng_key = jax.random.PRNGKey(int(time.time()) + actor_id)
        self.env_wrapper = EnvWrapper(CONFIG.train.env_name, CONFIG.train.num_agents, CONFIG.train.max_episode_steps)
        model = FlaxMAMuZeroNet(CONFIG.model, self.env_wrapper.action_space_size)
        
        # MCTS planner setup
        planner_classes = {"independent": MCTSPlanner, "joint": MCTSJointPlanner,"sequential": MCTSSequentialPlanner}
        if CONFIG.mcts.planner_mode not in planner_classes: raise ValueError(f"Invalid planner mode: {CONFIG.mcts.planner_mode}")
        planner = planner_classes[CONFIG.mcts.planner_mode](model=model, config=CONFIG)
        self.plan_fn = jax.jit(planner.plan)

        print(f"(DataActor pid={os.getpid()}) Setup complete.")

    def process_episode(
        self,
        episode: "Episode",
        unroll_steps: int,
        n_step: int,  # How many steps into the future to bootstrap the value from
        discount_gamma: float
    ) -> List["ReplayItem"]:
        """
        Processes a completed episode to create a list of valid, fixed-length training samples.
        """
        from utils.replay_buffer import ReplayItem
        replay_items = []

        for start_index in range(len(episode.trajectory)):
            unroll_window = episode.trajectory[start_index : start_index + unroll_steps]
            full_target_window = episode.trajectory[start_index : start_index + unroll_steps + 1]

            if len(full_target_window) < unroll_steps + 1:
                break 

            initial_observation = unroll_window[0].observation
            actions = np.array([t.action for t in unroll_window])
            policy_targets = np.array([t.policy_target for t in full_target_window])
            rewards = [t.reward for t in unroll_window]
            reward_targets = np.array([np.full((CONFIG.train.num_agents,), r) for r in rewards], dtype=np.float32)
            target_obs = unroll_window[-1].observation
            
            value_targets = []
            for i in range(unroll_steps + 1):
                target_step_index = start_index + i
                n_step_reward_window = episode.trajectory[target_step_index : target_step_index + n_step]
                
                n_step_reward_sum = sum(
                    t.reward * (discount_gamma ** j) for j, t in enumerate(n_step_reward_window)
                )

                bootstrap_index = target_step_index + n_step
                if bootstrap_index < len(episode.trajectory):
                    bootstrap_value = (episode.trajectory[bootstrap_index].value_target * (discount_gamma ** n_step))
                else:
                    bootstrap_value = 0.0
                
                final_value_target = n_step_reward_sum + bootstrap_value
                decentralized_target = np.full((CONFIG.train.num_agents,), final_value_target)
                value_targets.append(decentralized_target)
            
            value_targets = np.array(value_targets, dtype=np.float32)

            replay_items.append(
                ReplayItem(
                observation=initial_observation,
                actions=actions,
                target_observation=target_obs,
                policy_target=policy_targets,
                value_target=value_targets, 
                reward_target=reward_targets
            )
        )

        return replay_items

    def run_episode(self):
        """ Runs a single episode, collects data, and updates the replay buffer."""
        import jax
        from utils.replay_buffer import Episode, Transition
        
        observation, state = self.env_wrapper.reset()
        episode = Episode()

        for _ in range(CONFIG.train.max_episode_steps):
            self.rng_key, plan_key = jax.random.split(self.rng_key, 2)
            plan_output = self.plan_fn(self.params, plan_key, observation)
            action_np = np.asarray(plan_output.joint_action)
            next_observation, next_state, reward, done = self.env_wrapper.step(state, action_np)
            episode.add_step(Transition(observation, action_np, reward, done, np.asarray(plan_output.policy_targets), plan_output.root_value))
            observation = next_observation
            state = next_state
            if done: break

        replay_items = self.process_episode(
            episode,
            CONFIG.train.unroll_steps,
            CONFIG.train.n_step,
            CONFIG.train.discount_gamma, 
        )

        if replay_items: self.replay_buffer.add_batch.remote(replay_items)

        self.episodes_since_update += 1
        if self.episodes_since_update >= CONFIG.train.param_update_interval:
            self.params = ray.get(self.learner.get_params.remote())
            self.episodes_since_update = 0
        
        return episode.episode_return


def main():
    # Initialization
    ray.init(ignore_reinit_error=True)
    wandb.init(project="toy_mazero", config=CONFIG)
    
    print(f"Ray cluster started. Available resources: {ray.available_resources()}")
    
    replay_buffer = ReplayBufferActor.remote()
    learner = LearnerActor.remote(replay_buffer)
    actors = [DataActor.remote(i, learner, replay_buffer) for i in range(CONFIG.train.num_actors)]

    print("Waiting for actors to initialize and run one episode...")
    # This ensures the __init__ methods have completed before we proceed.
    ray.get([actor.run_episode.remote() for actor in actors]) 

    # Warmup Phase
    print("\nWarmup phase...")

    actor_tasks = {actor.run_episode.remote(): actor for actor in actors}
    while ray.get(replay_buffer.get_size.remote()) < CONFIG.train.warmup_episodes:
        done_refs, _ = ray.wait(list(actor_tasks.keys()), num_returns=1)
        done_ref = done_refs[0]
        finished_actor = actor_tasks.pop(done_ref)
        
        actor_tasks[finished_actor.run_episode.remote()] = finished_actor

        print(f"  Buffer size: {ray.get(replay_buffer.get_size.remote())}/{CONFIG.train.warmup_episodes}", end="\r")

    # Main Training Loop
    print("\nWarmup complete. Starting main training loop.")
    episodes_processed = 0
    returns = deque(maxlen=CONFIG.train.log_interval)
    losses = deque(maxlen=CONFIG.train.log_interval)
    start_time = time.time()
    
    # Start the first training step
    learner_task = learner.train.remote()

    while episodes_processed < CONFIG.train.num_episodes:
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
            loss_dict = ray.get(done_learner_refs[0])
            losses.append(loss_dict['total_loss'])
            wandb.log(loss_dict, step=episodes_processed) 
            # Start the next training step
            learner_task = learner.train.remote()

        # Periodic logging
        if episodes_processed % CONFIG.train.log_interval == 0 and returns:
            avg_return = np.mean(returns)
            avg_loss = np.mean(losses) if losses else 0.0
            
            wandb.log({
                "avg_return": avg_return, 
                "avg_loss": avg_loss,
                "episodes": episodes_processed
            }, step=episodes_processed)

            print(f"Episodes: {episodes_processed} | Avg Return: {avg_return:.2f} | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.2f}s")
            start_time = time.time() 
    
    print("Training finished.")
    wandb.finish()
    ray.shutdown()
    
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass
    main()