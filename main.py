import jax
import jax.numpy as jnp
import numpy as np
import optax
import chex
from typing import Tuple, Any, Callable, Dict, NamedTuple
from model.model import FlaxMAMuZeroNet
from mcts.mcts_independent import MCTSIndependentPlanner
from mcts.mcts_joint import MCTSJointPlanner
from mcts.mcts_sequential import MCTSSequentialPlanner
from utils.wrappers.base_wrapper import JaxMarlWrapper
from utils.wrappers.mpe_wrapper import MPEWrapper
from utils.wrappers.smax_wrapper import SMAXWrapper
from utils.replay_buffer import Transition, ReplayBuffer
from utils.logging_utils import logger
from utils.utils import DiscreteSupport, support_to_scalar, scalar_to_support, categorical_cross_entropy_loss, n_step_returns_fn
from functools import partial
from collections import deque
import wandb
from config import CONFIG
import time
import mctx

class RunnerState(NamedTuple): # Data structure to hold all dynamic variables during training
    params: Dict
    #target_params: Dict
    opt_state: optax.OptState
    key: chex.PRNGKey
    env_state: Any
    obs: chex.Array
    replay_buffer: ReplayBuffer
    episode_returns: chex.Array
    episode_lengths: chex.Array
    delta_magnitudes: chex.Array
    coord_state_norms: chex.Array

def env_setup(key, num_envs: int, num_agents: int, env_name: str) -> Tuple[JaxMarlWrapper, Callable, Callable, chex.Array, Any]:
    """Initializes the parallel MPE environments."""
    if "MPE" in env_name:
        env = MPEWrapper(
            env_name=env_name, 
            num_agents=num_agents
        )
    else:
        env = SMAXWrapper(
            scenario_name=env_name
        )
    
    vmapped_reset = jax.jit(jax.vmap(env.reset))
    vmapped_step = jax.jit(jax.vmap(env.step))
    
    logger.info(f"Environment '{env_name}' initialized for {num_envs} parallel instances.")
    
    reset_keys = jax.random.split(key, num_envs)
    obs_batch, state_batch = vmapped_reset(reset_keys)
    # obs_batch shape: (num_envs, num_agents, obs_dim)
    
    logger.debug(f"Initial observation batch shape: {obs_batch.shape}")
    logger.debug(f"Environments reset on device: {obs_batch.device}")
    
    return env, vmapped_reset, vmapped_step, obs_batch, state_batch

def model_setup(key, env: JaxMarlWrapper, obs_batch: chex.Array) -> Tuple[FlaxMAMuZeroNet, Dict, optax.OptState, Callable]:
    """Initializes the model, optimizer, and learning rate schedule."""
    start_time = time.time()
    
    model = FlaxMAMuZeroNet(CONFIG.model, env.action_space_size)
    dummy_obs = obs_batch[0][jnp.newaxis, ...] # Shape: (1, num_agents, obs_dim)
    logger.debug(f"Dummy Obs Shape for model init: {dummy_obs.shape}")
    
    params = model.init(key, dummy_obs)['params']
    logger.info("Model initialized.")

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=CONFIG.train.learning_rate,
        warmup_steps=CONFIG.train.lr_warmup_steps,
        decay_steps=CONFIG.train.train_steps - CONFIG.train.lr_warmup_steps,
        end_value=CONFIG.train.learning_rate * CONFIG.train.end_lr_factor
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(CONFIG.train.gradient_clip_norm),
        optax.adamw(learning_rate=lr_schedule)
    )
    opt_state = optimizer.init(params)
    logger.info("Optimizer initialized.")
    
    end_time = time.time()
    logger.info(f"Model & Optimizer Initialization Time: {(end_time - start_time):.2f} seconds.")
    
    return model, params, optimizer, opt_state, lr_schedule

def planner_setup(model: FlaxMAMuZeroNet) -> Callable:
    """Initializes the MCTS planner and vmaps it for parallel execution."""
    planner_classes = {
        "independent": MCTSIndependentPlanner, 
        "joint": MCTSJointPlanner,
        "sequential": MCTSSequentialPlanner
    }
    planner_mode = CONFIG.mcts.planner_mode
    if planner_mode not in planner_classes: 
        raise ValueError(f"Invalid planner mode: {planner_mode}")
        
    planner = planner_classes[planner_mode](model=model, config=CONFIG)

    def single_env_plan(params, key, obs):
        obs_batched = jnp.expand_dims(obs, axis=0)
        return planner.plan(params, key, obs_batched) # Shape: (1, num_agents, obs_dim)
    
    vmapped_plan = jax.jit(jax.vmap(single_env_plan, in_axes=(None, 0, 0)))
    logger.info(f"MCTS planner '{planner_mode}' initialized.")
    return vmapped_plan

def action_setup(key, plan: Callable, params: Dict, obs: chex.Array, num_envs: int, env: JaxMarlWrapper) -> Tuple[chex.Array, chex.Array]: # Debug function
    logger.debug("Performing one parallel planning step to generate actions...")
    start_time = time.time()
    plan_keys = jax.random.split(key, num_envs)

    plan_output = plan(params, plan_keys, obs)
    action = plan_output.joint_action # Shape: (num_envs, num_agents)
    policy_targets = plan_output.policy_targets # Shape: (num_envs, num_agents, action_space_size)

    end_time = time.time()

    logger.debug(f"Shape of batched actions: {action.shape}")
    logger.debug(f"Shape of batched policy targets: {policy_targets.shape}")
    logger.debug(f"Device of batched actions: {action.device}")

    chex.assert_shape(action, (num_envs, env.num_agents))
    chex.assert_type(action, jnp.int32)

    chex.assert_shape(policy_targets, (num_envs, env.num_agents, env.action_space_size))
    chex.assert_type(policy_targets, jnp.float32)
    
    logger.info(f"Action Selection Time: {(end_time - start_time):.2f} seconds.")
    logger.info("Action selection successful.")
    return action, policy_targets

def step_setup(key, step_fn: Callable, num_envs: int, state: Any, action: chex.Array) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Any]: # Debug function
    logger.info("Performing one parallel environment step...")
    start_time = time.time()

    step_keys = jax.random.split(key, num_envs)

    next_obs, next_state, reward, done, info = step_fn(step_keys, state, action)
    # next_obs shape: (num_envs, num_agents, obs_dim)
    # reward shape: (num_envs,)
    # done shape: (num_envs,)

    end_time = time.time()

    logger.debug(f"Shape of next observation batch: {next_obs.shape}")
    logger.debug(f"Shape of reward batch: {reward.shape}")
    logger.debug(f"Shape of done batch: {done.shape}")
    
    chex.assert_shape(reward, (num_envs,))
    chex.assert_type(reward, jnp.float32)
    chex.assert_shape(done, (num_envs,))
    chex.assert_type(done, jnp.bool_)

    logger.info(f"Environment Step Time: {(end_time - start_time):.2f} seconds.")
    logger.info("Environment step successful.")
    return next_obs, next_state, reward, done, info

def replay_buffer_setup(obs: chex.Array, alpha: int, env: JaxMarlWrapper) -> ReplayBuffer:
    dummy_policy_target = jnp.zeros((CONFIG.train.num_envs, env.num_agents, env.action_space_size))
    dummy_value_target = jnp.zeros((CONFIG.train.num_envs,))
    sample_transition = Transition(
        obs=obs,
        action=jnp.zeros((CONFIG.train.num_envs, env.num_agents), dtype=jnp.int32),
        reward=jnp.zeros((CONFIG.train.num_envs,)),
        done=jnp.zeros((CONFIG.train.num_envs,), dtype=jnp.bool_),
        policy_target=dummy_policy_target,
        value_target=dummy_value_target
    )
    replay_buffer = ReplayBuffer.create(CONFIG.train.replay_buffer_size, alpha, sample_transition)
    logger.info(f"Replay Buffer initialized with capacity: {replay_buffer.capacity}")
    logger.info(f"Replay buffer initialized with alpha: {alpha}")
    return replay_buffer

def _rollout_step(runner_state: RunnerState, _: Any, plan: Callable, vmapped_step: Callable, vmapped_reset: Callable, env: Any):
    """
    Represents one step of interaction for all parallel environments.
    This function is designed to be used with jax.lax.scan.
    """
    key, plan_key, step_key, reset_key = jax.random.split(runner_state.key, 4)
    plan_keys = jax.random.split(plan_key, CONFIG.train.num_envs) # TODO: pass in num_envs as static argument
    
    plan_output = plan(runner_state.params, plan_keys, runner_state.obs)
    
    step_keys = jax.random.split(step_key, CONFIG.train.num_envs)
    next_obs, next_state, reward, done, info = vmapped_step(
        step_keys, runner_state.env_state, plan_output.joint_action
    )

    new_episode_returns = runner_state.episode_returns + reward
    new_episode_lengths = runner_state.episode_lengths + 1
    new_delta_magnitudes = runner_state.delta_magnitudes + plan_output.delta_magnitude
    new_coord_state_norms = runner_state.coord_state_norms + plan_output.coord_state_norm

    logged_metrics = {
        "done_mask": done,
        "episode_returns": new_episode_returns,
        "episode_lengths": new_episode_lengths,
        "delta_magnitudes": new_delta_magnitudes,
        "coord_state_norms": new_coord_state_norms,
        "won": info.get("won", jnp.zeros_like(done, dtype=jnp.bool_))
    }

    transition = Transition(
        obs=runner_state.obs,
        action=plan_output.joint_action,
        reward=reward,
        done=done,
        policy_target=plan_output.policy_targets,
        value_target=plan_output.root_value,
    )
    
    reset_keys = jax.random.split(reset_key, CONFIG.train.num_envs)

    new_obs_from_reset, new_state_from_reset = vmapped_reset(reset_keys)
    
    # 'where' selects between the next state and a reset state based on 'done'
    obs_to_reset = jnp.where(done[:, None, None], new_obs_from_reset, next_obs)
    state_to_reset = jax.tree_util.tree_map(
        lambda reset_leaf, next_leaf: jnp.where(
            jnp.expand_dims(done, axis=tuple(range(1, next_leaf.ndim))), 
            reset_leaf, 
            next_leaf
        ),
        new_state_from_reset,
        next_state
    )

    next_runner_state = runner_state._replace(
        key=key, obs=obs_to_reset, env_state=state_to_reset,
        replay_buffer=runner_state.replay_buffer.add(transition),
        episode_returns=jnp.where(done, 0.0, new_episode_returns),
        episode_lengths=jnp.where(done, 0, new_episode_lengths),
        delta_magnitudes=jnp.where(done, 0.0, new_delta_magnitudes),
        coord_state_norms=jnp.where(done, 0.0, new_coord_state_norms)
    )
    
    return next_runner_state, logged_metrics

def _update_step(runner_state: RunnerState, model: FlaxMAMuZeroNet, optimizer: optax.GradientTransformation, vmapped_n_step_returns: Callable):
    key, sample_key, train_key = jax.random.split(runner_state.key, 3)

    batch, weights, indices = runner_state.replay_buffer.sample(
        sample_key,
        CONFIG.train.batch_size, # TODO: pass these as arguments
        CONFIG.train.unroll_steps,
        CONFIG.train.n_step,
        CONFIG.train.replay_buffer_beta_start
    )

    n_step_value = vmapped_n_step_returns(batch.rewards, batch.value_targets, CONFIG.train.n_step, CONFIG.train.discount_gamma)

    def loss_fn(params, n_step_value, batch, weights, key):
        value_support = DiscreteSupport(-CONFIG.model.value_support_size, CONFIG.model.value_support_size)
        reward_support = DiscreteSupport(-CONFIG.model.reward_support_size, CONFIG.model.reward_support_size)

        value_target_for_loss = jax.lax.dynamic_slice_in_dim(n_step_value, 0, CONFIG.train.unroll_steps + 1, axis=1)
        reward_for_loss = jax.lax.dynamic_slice_in_dim(batch.rewards, 0, CONFIG.train.unroll_steps, axis=1)
        policy_targets_for_loss = jax.lax.dynamic_slice_in_dim(batch.policy_targets, 1, CONFIG.train.unroll_steps, axis=1)

        value_target_dist = jax.vmap(scalar_to_support, in_axes=(0, None))(value_target_for_loss, value_support)
        reward_target_dist = jax.vmap(scalar_to_support, in_axes=(0, None))(reward_for_loss, reward_support)

        _, initial_rng, unroll_rng = jax.random.split(key, 3)

        output = model.apply(
            {'params': params}, batch.obs, 
            rngs={'dropout': initial_rng}
        )
        hidden, _, p0_logits, v0_logits = output.hidden_state, output.reward_logits, output.policy_logits, output.value_logits

        policy_loss = optax.softmax_cross_entropy(p0_logits, batch.policy_targets[:, 0]).mean()
        value_loss = categorical_cross_entropy_loss(v0_logits, value_target_dist[:, 0]).mean()

        v0_scalar = support_to_scalar(v0_logits, value_support)
        td_error = jnp.abs(v0_scalar - batch.value_targets[:, 0])

        def unroll_step_fn(carry, step_data):
            hidden_state, key = carry
            step_key, key = jax.random.split(key)
            action, reward_target, policy_target, value_target, mask = step_data
            
            online_proj = model.apply({'params': params}, hidden_state, method=model.project)

            output = model.apply(
                {'params': params}, hidden_state, action,
                method=model.recurrent_inference,
                rngs={'dropout': step_key}
            )
            next_hidden, r_logits, p_logits, v_logits = output.hidden_state, output.reward_logits, output.policy_logits, output.value_logits

            target_proj = model.apply({'params': params}, next_hidden, method=model.project)
            target_proj = jax.lax.stop_gradient(target_proj)

            B, N, D = online_proj.shape
            sim = optax.cosine_similarity(online_proj.reshape(B * N, D), target_proj.reshape(B * N, D))
            c_loss = -sim.mean()

            r_loss = categorical_cross_entropy_loss(r_logits, reward_target)
            p_loss = optax.softmax_cross_entropy(p_logits, policy_target).mean(axis=-1)
            v_loss = categorical_cross_entropy_loss(v_logits, value_target)

            step_losses = {
                "reward": r_loss * mask,
                "policy": p_loss * mask,
                "value": v_loss * mask,
                "consistency": c_loss * mask
            }

            return (next_hidden, key), step_losses
            
        scan_data_tuple = (
            batch.actions,
            reward_target_dist,
            policy_targets_for_loss,
            value_target_dist[:, 1:],
            batch.mask
        )

        scan_data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), scan_data_tuple)

        initial_carry = (hidden, unroll_rng)
        _, unrolled_losses = jax.lax.scan(unroll_step_fn, initial_carry, scan_data, length=CONFIG.train.unroll_steps)

        unrolled_losses = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), unrolled_losses)

        reward_loss = unrolled_losses["reward"].sum(axis=1)
        policy_loss += unrolled_losses["policy"].sum(axis=1)
        value_loss += unrolled_losses["value"].sum(axis=1)
        consistency_loss = unrolled_losses["consistency"].sum(axis=1)
        
        num_valid_steps = batch.mask.sum(axis=1)
        safe_num_valid_steps = jnp.maximum(num_valid_steps, 1)

        reward_loss = (reward_loss / safe_num_valid_steps)
        policy_loss = (policy_loss / (safe_num_valid_steps + 1)) # +1 for the initial step
        value_loss = (value_loss / (safe_num_valid_steps + 1))
        consistency_loss = (consistency_loss / safe_num_valid_steps)

        total_loss = (
            reward_loss + 
            policy_loss + 
            value_loss * CONFIG.train.value_scale +
            consistency_loss * CONFIG.train.consistency_scale
        )
        
        final_loss = (total_loss * weights).mean()

        metrics = {
            "total_loss": final_loss,
            "reward_loss": reward_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "consistency_loss": consistency_loss
        }
        metrics['debug_n_step_target_mean'] = jnp.mean(n_step_value)
        metrics['debug_n_step_target_std'] = jnp.std(n_step_value)
        metrics['debug_policy_target_mean'] = jnp.mean(batch.policy_targets)
        metrics['debug_predicted_value_mean'] = jnp.mean(v0_scalar)
        metrics['debug_predicted_value_std'] = jnp.std(v0_scalar)
        return final_loss, (metrics, td_error)
    
    (loss, (metrics, td_error)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        runner_state.params, n_step_value, batch, weights, train_key
    )

    updates, new_opt_state = optimizer.update(grads, runner_state.opt_state, runner_state.params)
    new_params = optax.apply_updates(runner_state.params, updates)
    metrics['grad_norm'] = optax.global_norm(grads)

    new_replay_buffer = runner_state.replay_buffer.update_priorities(indices, td_error)

    # new_target_params = jax.tree_util.tree_map(
    #     lambda p, tp: p * CONFIG.train.tau + tp * (1 - CONFIG.train.tau),
    #     new_params,
    #     runner_state.target_params
    # )

    # new_priorities = td_error + 1e-6 
    # new_replay_buffer = runner_state.replay_buffer.update_priorities(indices, new_priorities)

    next_runner_state = runner_state._replace(
        key=key,
        params=new_params,
        #target_params=new_target_params,
        opt_state=new_opt_state,
        replay_buffer=new_replay_buffer
    )

    return next_runner_state, metrics

def main():
    if CONFIG.train.wandb_mode != "disabled": wandb.init(project=CONFIG.train.project_name, config=CONFIG)

    logger.info(f"Available devices for JAX: {jax.devices()}")
    logger.info(f"JAX default backend: {jax.default_backend()}")

    # PRNG Key and Environment Setup
    key = jax.random.PRNGKey(CONFIG.train.seed)
    logger.info(f"Using seed: {CONFIG.train.seed}")
    logger.info("Initializing Environment")

    key, env_key = jax.random.split(key)
    num_envs, num_agents, env_name = CONFIG.train.num_envs, CONFIG.train.num_agents, CONFIG.train.env_name
    env, reset_fn, step_fn, obs, state = env_setup(env_key, num_envs, num_agents, env_name)

    # Model and Optimizer Setup
    key, model_key = jax.random.split(key)
    logger.info("Initializing Model and Optimizer")
    
    model, params, optimizer, opt_state, lr_schedule = model_setup(model_key, env, obs)

    # MCTS Planner Setup
    logger.info("Initializing Planner")
    plan = planner_setup(model)

    # Replay Buffer Initialization
    replay_buffer = replay_buffer_setup(obs, CONFIG.train.replay_buffer_alpha, env)

    # Core Functions
    vmapped_n_step_returns = jax.vmap(n_step_returns_fn, in_axes=(0, 0, None, None))
    rollout_step_fn = partial(_rollout_step, plan=plan, vmapped_step=step_fn, vmapped_reset=reset_fn, env=env)
    rollout_fn = jax.jit(lambda state: jax.lax.scan(rollout_step_fn, state, None, length=CONFIG.train.rollout_length+1))
    update_fn = jax.jit(partial(_update_step, model=model, optimizer=optimizer, vmapped_n_step_returns=vmapped_n_step_returns))

    updates_per_step = CONFIG.train.update_per_step 
    logger.info(f"Performing {updates_per_step} gradient updates per data collection step.")

    # Main Training Loop
    logger.info("Starting main training loop...")
    key, runner_key = jax.random.split(key)
    runner_state = RunnerState(
        params=params,
        #target_params=params,
        opt_state=opt_state,
        key=runner_key,
        env_state=state,
        obs=obs,
        replay_buffer=replay_buffer,
        episode_returns=jnp.zeros(num_envs),
        episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
        delta_magnitudes=jnp.zeros(num_envs),
        coord_state_norms=jnp.zeros(num_envs)
    )
    
    interval_returns, interval_lengths, interval_deltas, interval_coords, interval_losses, interval_wins = [], [], [], [], [], []

    for train_step in range(1, CONFIG.train.train_steps+1):
        start_time = time.time()
        runner_state, rollout_metrics = rollout_fn(runner_state)
        
        jax.block_until_ready(rollout_metrics)
        all_update_metrics = []
        for _ in range(updates_per_step):
            runner_state, update_metrics = update_fn(runner_state)
            all_update_metrics.append(update_metrics)

        update_metrics = jax.tree_util.tree_map(lambda *x: jnp.mean(jnp.array(x)), *all_update_metrics)

        done_masks = rollout_metrics["done_mask"]
        won_metrics = rollout_metrics["won"]

        for i in range(done_masks.shape[0]):
            done_mask = done_masks[i]
            if np.any(done_mask):
                interval_returns.extend(np.array(rollout_metrics["episode_returns"][i][done_mask]))
                interval_lengths.extend(np.array(rollout_metrics["episode_lengths"][i][done_mask]))
                
                deltas = rollout_metrics["delta_magnitudes"][i]
                lengths = rollout_metrics["episode_lengths"][i]
                avg_deltas = np.where(lengths > 0, deltas / lengths, 0.0)
                interval_deltas.extend(avg_deltas[done_mask])

                coords = rollout_metrics["coord_state_norms"][i]
                avg_coords = np.where(lengths > 0, coords / lengths, 0.0)
                interval_coords.extend(avg_coords[done_mask])

                if "MPE" not in env_name:
                    interval_wins.extend(np.array(won_metrics[i][done_mask]))

        interval_losses.append(update_metrics["total_loss"])

        if train_step % CONFIG.train.log_interval == 0:
            end_time = time.time()
            total_transitions = CONFIG.train.log_interval * CONFIG.train.rollout_length * num_envs
            tps = total_transitions / (end_time - start_time)
            
            # Calculate average metrics
            avg_return = np.mean(interval_returns) if interval_returns else 0.0
            avg_length = np.mean(interval_lengths) if interval_lengths else 0.0
            avg_loss = np.mean(interval_losses) if interval_losses else 0.0
            avg_delta = np.mean(interval_deltas) if interval_deltas else 0.0
            avg_coord = np.mean(interval_coords) if interval_coords else 0.0
            win_rate = np.mean(interval_wins) if interval_wins else 0.0

            logger.info(f"Step: {train_step} | Win rate: {win_rate:.2f}% | Env Steps: {train_step * CONFIG.train.rollout_length * num_envs} | TPS: {tps:,.2f} | Avg Return: {avg_return:.2f} | Avg Loss: {avg_loss:.2f}")
            
            if CONFIG.train.wandb_mode != "disabled":
                data_metrics = {
                    "train_step": train_step, "total_env_steps": train_step * CONFIG.train.rollout_length * num_envs,
                    "transitions_per_sec": tps, "avg_episode_return": avg_return,
                    "avg_episode_length": avg_length, "win_rate": win_rate,
                    "delta_magnitudes": jnp.mean(rollout_metrics["delta_magnitudes"]) if len(rollout_metrics["delta_magnitudes"]) > 0 else 0.0,
                    "coord_state_norms": jnp.mean(rollout_metrics["coord_state_norms"]) if len(rollout_metrics["coord_state_norms"]) > 0 else 0.0
                }    
                wandb.log(update_metrics | data_metrics)
            interval_returns, interval_lengths, interval_deltas, interval_coords, interval_losses, interval_wins = [], [], [], [], [], []

if __name__ == "__main__":
    main()