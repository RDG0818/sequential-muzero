# main.py
import ray
import time
import numpy as np
from collections import deque
import multiprocessing as mp
import os
from typing import List, Tuple, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from utils.replay_buffer import Episode, ReplayItem
    from utils.utils import DiscreteSupport
from utils.logging_utils import logger
from config import CONFIG
import wandb

@ray.remote
class ReplayBufferActor:
    """
    A Ray actor for managing the prioritized replay buffer.
    """
    def __init__(self, obs_shape: Tuple, action_space_size: int):
        """
        Initializes the ReplayBufferActor.
        Args:
            config: The configuration object containing necessary parameters.
        """
        from utils.replay_buffer import ReplayBuffer
        self.replay_buffer = ReplayBuffer(
            capacity=CONFIG.train.replay_buffer_size,
            observation_space=obs_shape,
            action_space_size=action_space_size,
            num_agents=CONFIG.train.num_agents,
            unroll_steps=CONFIG.train.unroll_steps,
            alpha=CONFIG.train.replay_buffer_alpha,
            beta_start=CONFIG.train.replay_buffer_beta_start,
            beta_frames=CONFIG.train.replay_buffer_beta_frames
        )

    def add(self, items: List["ReplayItem"], priorities: List[float]):
        """
        Adds a batch of items to the replay buffer with given priorities.
        Args:
            items: A list of ReplayItems to add.
            priorities: A list of priorities corresponding to the items.
        """
        for item, priority in zip(items, priorities):
            self.replay_buffer.add(item, priority)

    def get_size(self) -> int: 
        return len(self.replay_buffer)

    def sample(self, batch_size: int) -> Tuple["ReplayItem", np.ndarray, np.ndarray]:
        """
        Samples a batch from the replay buffer.
        Args:
            batch_size: The size of the batch to sample.
        Returns:
            A tuple containing the batched ReplayItem, importance sampling weights, and indices.
        """
        return self.replay_buffer.sample(batch_size)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Updates the priorities of items in the replay buffer.
        Args:
            indices: The indices of the items to update.
            priorities: The new priorities for the items.
        """
        self.replay_buffer.update_priorities(indices, priorities)


@ray.remote(num_gpus=1)
class LearnerActor:
    """
    The main learner actor responsible for training the MuZero model.

    This actor pulls batches of experience from a remote replay buffer,
    computes gradients, and updates the model parameters. It also periodically
    sends the updated parameters back to the actors collecting data.

    Attributes:
        params (Dict): The current parameters of the Flax model.
        train_step_count (int): The number of training steps performed.
    """
    def __init__(self, replay_buffer_actor):
        """
        Initializes the LearnerActor, setting up the model, optimizer, and state.
        
        Args:
            replay_buffer_actor (ray.actor.ActorHandle): A Ray actor handle to the replay buffer.
        """
        os.environ["CUDA_VISIBLE_DEVICES"]="0"

        import jax
        import optax
        from utils.utils import DiscreteSupport
        from model.model import FlaxMAMuZeroNet
        from utils.mpe_env_wrapper import MPEEnvWrapper
        logger.info(f"(Learner pid={os.getpid()}) Initializing on GPU...")
        
        # Basic setup
        self.replay_buffer = replay_buffer_actor
        N, lr = CONFIG.train.num_agents, CONFIG.train.learning_rate
        self.rng_key = jax.random.PRNGKey(42)
        self.train_step_count = 0
        self.value_support = DiscreteSupport(min=-CONFIG.model.value_support_size, max=CONFIG.model.value_support_size)
        self.reward_support = DiscreteSupport(min=-CONFIG.model.reward_support_size, max=CONFIG.model.reward_support_size)

        # env and model setup
        env = MPEEnvWrapper(CONFIG.train.env_name, N, CONFIG.train.max_episode_steps)
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

        self.jitted_train_step = jax.jit(self._train_step, static_argnames=['model', 'optimizer', 'value_support', 'reward_support'])

        logger.info(f"(Learner pid={os.getpid()}) Setup complete.")

    @staticmethod
    def _train_step(model, optimizer, params, opt_state, batch, weights, rng_key, value_support: "DiscreteSupport", reward_support: "DiscreteSupport"):
        import jax
        import jax.numpy as jnp
        import optax
        from utils import utils

        U = CONFIG.train.unroll_steps

        value_target_dist = utils.scalar_to_support(batch.value_target.mean(axis=2), value_support)
        reward_target_dist = utils.scalar_to_support(batch.reward_target.mean(axis=2), reward_support)
        
        def loss_fn(p):
            reward_loss, policy_loss, value_loss, consistency_loss = 0.0, 0.0, 0.0, 0.0

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
                online_proj = model.apply({'params': p}, hidden, True, method=model.project)
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

                target_proj = model.apply({'params': p}, hidden, False, method=model.project)
                target_proj = jax.lax.stop_gradient(target_proj)

                B, N, D = online_proj.shape
                online_proj_flat = online_proj.reshape(B * N, D)
                target_proj_flat = target_proj.reshape(B * N, D)

                sim = optax.cosine_similarity(online_proj_flat, target_proj_flat)
                consistency_loss -= sim.mean()

            td_error = jnp.abs(utils.support_to_scalar(v0_logits, value_support) - batch.value_target[:, 0].mean(axis=1))

            reward_loss /= U
            policy_loss /= (U + 1)
            value_loss /= (U + 1)
            consistency_loss /= U

            loss = reward_loss + policy_loss + value_loss * CONFIG.train.value_scale + consistency_loss * CONFIG.train.consistency_scale
            total_loss = (loss * weights).mean()

            metrics = {
                "total_loss": total_loss,
                "reward_loss": reward_loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "consistency_loss": consistency_loss
            }

            return total_loss, (metrics, td_error)

        (loss, (metrics, td_error)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        metrics['grad_norm'] = optax.global_norm(grads)
        
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_priorities = td_error + 1e-6

        return new_params, new_opt_state, metrics, new_priorities

    def train(self) -> Dict[str, float]:
        """
        Samples a batch, performs a training step, and updates the replay buffer priorities.

        Returns:
            A dictionary of metrics from the training step, converted to standard Python types.
        """
        import jax

        # Sample a batch from the replay buffer
        self.rng_key, train_key = jax.random.split(self.rng_key)
        numpy_batch, numpy_weights, numpy_indices = ray.get(self.replay_buffer.sample.remote(CONFIG.train.batch_size))
        jax_batch = jax.tree_util.tree_map(lambda x: jax.device_put(x), numpy_batch)
        jax_weights = jax.device_put(numpy_weights)

        # Perform a training step
        self.params, self.opt_state, metrics, new_priorities = self.jitted_train_step(self.model, self.optimizer, self.params, self.opt_state, jax_batch, 
                                                                                      jax_weights, train_key, self.value_support, self.reward_support)
        self.train_step_count += 1

        numpy_priorities = np.array(new_priorities)
        self.replay_buffer.update_priorities.remote(numpy_indices, numpy_priorities)

        # Convert metrics to standard Python types for logging
        metrics = {k: v.item() for k, v in metrics.items()}
        metrics['learning_rate'] = self.lr_schedule(self.train_step_count).item()

        return metrics
        
    def get_params(self) -> Dict: return self.params
    def get_train_step_count(self) -> int: return self.train_step_count


@ray.remote(num_cpus=1)
class DataActor:
    """
    A Ray remote actor responsible for generating game experience.

    This actor runs in a separate process on a CPU. It continuously plays episodes
    of the game, processes the trajectories into training samples, and sends
    them to a central replay buffer. It periodically updates its model parameters
    from the main learner.
    """
    def __init__(self, actor_id, learner_actor, replay_buffer_actor):
        # Force CPU usage for this actor
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['JAX_PLATFORMS'] = 'cpu'
        
        import jax
        import chex
        from mcts.mcts_independent import MCTSIndependentPlanner
        from mcts.mcts_joint import MCTSJointPlanner
        from mcts.mcts_sequential import MCTSSequentialPlanner
        from utils.mpe_env_wrapper import MPEEnvWrapper
        from model.model import FlaxMAMuZeroNet
        
        logger.info("Initializing on CPU...")

        # Basic setup
        self.actor_id: int = actor_id
        self.learner = learner_actor
        self.replay_buffer = replay_buffer_actor

        self.params: chex.Array = ray.get(self.learner.get_params.remote())
        self.episodes_since_update: int = 0
        
        # Env and model setup
        self.rng_key = jax.random.PRNGKey(int(time.time()) + actor_id)
        self.env_wrapper = MPEEnvWrapper(CONFIG.train.env_name, CONFIG.train.num_agents, CONFIG.train.max_episode_steps)
        model = FlaxMAMuZeroNet(CONFIG.model, self.env_wrapper.action_space_size)
        
        # MCTS planner setup
        planner_classes = {"independent": MCTSIndependentPlanner, "joint": MCTSJointPlanner,"sequential": MCTSSequentialPlanner}
        if CONFIG.mcts.planner_mode not in planner_classes: raise ValueError(f"Invalid planner mode: {CONFIG.mcts.planner_mode}")
        planner = planner_classes[CONFIG.mcts.planner_mode](model=model, config=CONFIG)
        self.plan_fn = jax.jit(planner.plan)

        logger.info("Setup complete.")

    def process_episode(
        self,
        episode: "Episode",
        unroll_steps: int,
        n_step: int,  
        discount_gamma: float
    ) -> List["ReplayItem"]:
        """
        Processes a completed episode using a sliding window to create training samples.

        This version includes minor optimizations by vectorizing the data extraction
        before entering the main loop.

        Args:
            episode: The completed episode object.
            unroll_steps: The number of steps to unroll for each sample.
            n_step: The number of future steps to use for value bootstrapping.
            discount_gamma: The discount factor for future rewards.

        Returns:
            A list of fully-formed ReplayItem objects ready for training.
        """
        from utils.replay_buffer import ReplayItem
        import numpy as np

        replay_items = []
        trajectory = episode.trajectory
        ep_len = len(trajectory)

        observations = np.stack([t.observation for t in trajectory])
        actions = np.stack([t.action for t in trajectory])
        policy_targets = np.stack([t.policy_target for t in trajectory])
        rewards = np.array([t.reward for t in trajectory], dtype=np.float32)
        mcts_values = np.array([t.value_target for t in trajectory], dtype=np.float32)

        for start_index in range(ep_len - unroll_steps):
            unroll_slice = slice(start_index, start_index + unroll_steps)
            full_slice = slice(start_index, start_index + unroll_steps + 1)

            value_scalars = []
            for i in range(unroll_steps + 1):
                target_step_index = start_index + i
                n_step_reward_window = rewards[target_step_index : target_step_index + n_step]
                
                n_step_reward_sum = sum(
                    reward * (discount_gamma ** j) for j, reward in enumerate(n_step_reward_window)
                )

                bootstrap_index = target_step_index + n_step
                if bootstrap_index < ep_len:
                    bootstrap_value = mcts_values[bootstrap_index] * (discount_gamma ** n_step)
                else:
                    bootstrap_value = 0.0
                
                final_value_scalar = n_step_reward_sum + bootstrap_value
                value_scalars.append(final_value_scalar)

            decentralized_value_targets = np.array([np.full((CONFIG.train.num_agents,), v) for v in value_scalars], dtype=np.float32)
            decentralized_reward_targets = np.array([np.full((CONFIG.train.num_agents,), r) for r in rewards[unroll_slice]], dtype=np.float32)

            replay_items.append(
                ReplayItem(
                    observation=observations[start_index],
                    actions=actions[unroll_slice],
                    target_observation=observations[start_index + unroll_steps],
                    policy_target=policy_targets[full_slice],
                    value_target=decentralized_value_targets,
                    reward_target=decentralized_reward_targets
                )
            )
        return replay_items
    
    def run_episode(self) -> float:
        """
        Runs a single episode, processes it, and sends the data to the replay buffer.

        This method also handles periodically updating its model parameters from the
        central learner.

        Returns:
            The total undiscounted return of the completed episode.
        """
        import jax
        from utils.replay_buffer import Episode, Transition
        
        observation, state = self.env_wrapper.reset()
        episode = Episode()

        episode_metrics = {
        "total_reward": 0.0,
        "num_steps": 0,
        "delta_magnitude": 0.0,
        "coord_state_norm": 0.0
         }   

        for _ in range(CONFIG.train.max_episode_steps):
            self.rng_key, plan_key = jax.random.split(self.rng_key, 2)
            plan_output = self.plan_fn(self.params, plan_key, observation)
            action_np = np.array(plan_output.joint_action)
            next_observation, next_state, reward, done = self.env_wrapper.step(state, action_np)
            episode.add_step(Transition(observation, action_np, reward, done, np.array(plan_output.policy_targets), plan_output.root_value))
            observation = next_observation
            state = next_state

            episode_metrics["total_reward"] += reward
            episode_metrics["num_steps"] += 1
            episode_metrics["delta_magnitude"] += plan_output.delta_magnitude
            episode_metrics["coord_state_norm"] += plan_output.coord_state_norm

            if done: break

        replay_items = self.process_episode(
            episode,
            CONFIG.train.unroll_steps,
            CONFIG.train.n_step,
            CONFIG.train.discount_gamma, 
        )


        if replay_items: 
            priorities = [1.0] * len(replay_items)
            self.replay_buffer.add.remote(replay_items, priorities)

        self.episodes_since_update += 1
        if self.episodes_since_update >= CONFIG.train.param_update_interval:
            self.params = ray.get(self.learner.get_params.remote())
            self.episodes_since_update = 0
        
        num_steps = episode_metrics.pop("num_steps")
        for k in episode_metrics:
            episode_metrics[k] /= num_steps

        return episode.episode_return, episode_metrics


def main():
    # Initialization
    ray.init(ignore_reinit_error=True)
    if CONFIG.train.wandb_mode != "disabled": wandb.init(project=CONFIG.train.project_name, config=CONFIG)
    
    logger.info(f"Ray cluster started. Available resources: {ray.available_resources()}")

    from utils.mpe_env_wrapper import MPEEnvWrapper
    temp_env = MPEEnvWrapper(CONFIG.train.env_name, CONFIG.train.num_agents, CONFIG.train.max_episode_steps)
    obs_shape = temp_env.observation_space
    action_size = temp_env.action_space_size
    del temp_env
    
    replay_buffer = ReplayBufferActor.remote(obs_shape, action_size)
    learner = LearnerActor.remote(replay_buffer)
    actors = [DataActor.remote(i, learner, replay_buffer) for i in range(CONFIG.train.num_actors)]

    logger.info("Waiting for actors to initialize and run one episode...")
    ray.get([actor.run_episode.remote() for actor in actors]) 

    # Warmup Phase
    logger.info("\nWarmup phase...")

    start_time = time.time()

    actor_tasks = {actor.run_episode.remote(): actor for actor in actors}
    while ray.get(replay_buffer.get_size.remote()) < CONFIG.train.warmup_episodes:
        done_refs, _ = ray.wait(list(actor_tasks.keys()), num_returns=1)
        done_ref = done_refs[0]
        finished_actor = actor_tasks.pop(done_ref)
        
        actor_tasks[finished_actor.run_episode.remote()] = finished_actor

        print(f"  Buffer size: {ray.get(replay_buffer.get_size.remote())}/{CONFIG.train.warmup_episodes}", end="\r")

    # Main Training Loop
    end_time = time.time()
    logger.info(f"\nWarmup complete.\n Time Taken: {end_time - start_time:.2f} seconds.\n Starting main training loop.")
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
        ep_return, episode_metrics = ray.get(done_actor_ref)
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
            if CONFIG.train.wandb_mode != "disabled":
                wandb.log(loss_dict, step=episodes_processed) 
            # Start the next training step
            learner_task = learner.train.remote()

        # Periodic logging
        if episodes_processed % CONFIG.train.log_interval == 0 and returns:
            avg_return = np.mean(returns)
            avg_loss = np.mean(losses) if losses else 0.0
            if CONFIG.train.wandb_mode != "disabled":
                wandb.log({
                    "avg_return": avg_return, 
                    "avg_loss": avg_loss,
                    "episodes": episodes_processed
                }, step=episodes_processed)
                wandb.log(episode_metrics)
            logger.info(f"Episodes: {episodes_processed} | Avg Return: {avg_return:.2f} | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.2f}s")
            start_time = time.time() 
    
    logger.info("Training finished.")
    if CONFIG.train.wandb_mode != "disabled":
        wandb.finish()
    ray.shutdown()
    
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
        logger.debug("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass
    main()