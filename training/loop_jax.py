"""
Pure-JAX synchronous training loop for Multi-Agent MuZero.

Replaces the Ray async actor-learner system with a single-process, GPU-resident
pipeline:
  1. collect_episode()    — JIT-compiled; MCTS + env steps on GPU via jax.lax.scan
  2. process_trajectory() — JIT-compiled; sliding window + n-step returns on GPU
  3. Flashbax buffer      — JAX-native prioritised item buffer, lives on GPU
  4. train_step()         — JIT-compiled; unchanged from actors/learner_actor.py

All hot-path operations never leave the GPU. The Python loop manages the Flashbax
buffer state, checkpointing, and logging with negligible overhead.
"""

import time
import numpy as np
from collections import deque
from typing import NamedTuple
from pathlib import Path

import jax
import jax.numpy as jnp

from config import ExperimentConfig
from utils.logging_utils import logger


class Trajectory(NamedTuple):
    """Per-step output stacked by jax.lax.scan. Each field has leading (T, B) dims."""
    obs: jax.Array             # (T, B, N, obs_size)
    actions: jax.Array         # (T, B, N)
    policy_targets: jax.Array  # (T, B, N, A)
    root_values: jax.Array     # (T, B)
    agent_orders: jax.Array    # (T, B, N)
    rewards: jax.Array         # (T, B)
    dones: jax.Array           # (T, B)
    active: jax.Array          # (T, B) — True while episode was still running at that step


def make_collect_episode(plan_fn, env, num_envs: int, max_episode_steps: int):
    """Returns a JIT-compiled episode collection function.

    The returned function resets all envs, then runs a jax.lax.scan over
    max_episode_steps steps. MCTS planning and env transitions run entirely
    on GPU.

    Args:
        plan_fn:            JIT-compiled planner.plan function.
        env:                VecMPEEnvWrapper instance.
        num_envs:           Number of parallel environments.
        max_episode_steps:  Length of each episode scan (fixed).

    Returns:
        collect_episode(params, rng) -> (rng, Trajectory)
    """

    @jax.jit
    def collect_episode(params, rng: jax.Array):
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(jax.random.split(reset_key, num_envs))
        active = jnp.ones(num_envs, dtype=bool)

        def step(carry, _):
            obs, env_state, rng, active = carry
            rng, plan_key, step_key = jax.random.split(rng, 3)
            plan_output = plan_fn(params, plan_key, obs)
            next_obs, next_state, rewards, dones = env.step(
                jax.random.split(step_key, num_envs), env_state, plan_output.joint_action
            )
            # agent_order is (N,) — broadcast to (B, N) for uniform per-env storage
            agent_order_b = jnp.broadcast_to(
                plan_output.agent_order[None], (num_envs, plan_output.agent_order.shape[0])
            )
            out = Trajectory(
                obs=obs,
                actions=plan_output.joint_action,
                policy_targets=plan_output.policy_targets,
                root_values=plan_output.root_value,
                agent_orders=agent_order_b,
                rewards=rewards,
                dones=dones,
                active=active,              # snapshot before this step's done update
            )
            return (next_obs, next_state, rng, active & ~dones), out

        (_, _, rng, _), traj = jax.lax.scan(
            step, (obs, env_state, rng, active), None, max_episode_steps
        )
        return rng, traj  # each field: (T, B, ...)

    return collect_episode


def make_process_trajectory(
    num_agents: int,
    action_space_size: int,
    unroll_steps: int,
    n_step: int,
    discount_gamma: float,
):
    """Returns a JIT-compiled trajectory → flat replay items function.

    For a (T, B, ...) trajectory, produces (K*B, ...) flat items where K = T - unroll_steps.
    Each item is a sliding window of length unroll_steps with precomputed n-step value targets.

    Returns:
        process_trajectory(traj: Trajectory) -> dict of (K*B, ...) JAX arrays
    """
    U = unroll_steps
    n = n_step
    gamma = discount_gamma
    discount_vec = gamma ** jnp.arange(n)

    @jax.jit
    def process_trajectory(traj: Trajectory) -> dict:
        T, B = traj.active.shape
        K = T - U  # number of valid start positions per env

        # Zero out rewards for steps after episode termination.
        rewards_masked = traj.rewards * traj.active   # (T, B)

        # Pad arrays so windowed access near the end stays in-bounds.
        rewards_pad = jnp.concatenate([rewards_masked, jnp.zeros((n, B))], axis=0)
        values_pad  = jnp.concatenate([traj.root_values, jnp.zeros((n + 1, B))], axis=0)

        def nstep_at(t):
            """n-step bootstrapped value target at position t, for all B envs."""
            r_window  = jax.lax.dynamic_slice_in_dim(rewards_pad, t, n, axis=0)  # (n, B)
            bootstrap = values_pad[t + n]                                          # (B,)
            return jnp.einsum('n,nb->b', discount_vec, r_window) + gamma ** n * bootstrap

        # value_targets[t, b] = n-step return starting at step t for env b
        value_targets = jax.vmap(nstep_at)(jnp.arange(T))  # (T, B)

        def make_item(s):
            """Build one ReplayItem dict for start position s (all B envs in parallel)."""
            obs_s = traj.obs[s]                                                      # (B, N, obs_size)

            act_s = jax.lax.dynamic_slice_in_dim(traj.actions, s, U, axis=0)        # (U, B, N)
            act_s = jnp.moveaxis(act_s, 0, 1)                                       # (B, U, N)

            pt_s  = jax.lax.dynamic_slice_in_dim(traj.policy_targets, s, U + 1, axis=0)  # (U+1, B, N, A)
            pt_s  = jnp.moveaxis(pt_s, 0, 1)                                             # (B, U+1, N, A)

            vt_s  = jax.lax.dynamic_slice_in_dim(value_targets, s, U + 1, axis=0)  # (U+1, B)
            vt_s  = jnp.moveaxis(vt_s, 0, 1)                                       # (B, U+1)
            vt_s  = jnp.broadcast_to(vt_s[:, :, None], (B, U + 1, num_agents))    # (B, U+1, N)

            rt_s  = jax.lax.dynamic_slice_in_dim(rewards_masked, s, U, axis=0)     # (U, B)
            rt_s  = jnp.moveaxis(rt_s, 0, 1)                                       # (B, U)
            rt_s  = jnp.broadcast_to(rt_s[:, :, None], (B, U, num_agents))         # (B, U, N)

            ao_s  = traj.agent_orders[s]                                            # (B, N)

            return {
                'observation':   obs_s,   # (B, N, obs_size)
                'actions':       act_s,   # (B, U, N)
                'policy_target': pt_s,    # (B, U+1, N, A)
                'value_target':  vt_s,    # (B, U+1, N)
                'reward_target': rt_s,    # (B, U, N)
                'agent_order':   ao_s,    # (B, N)
            }

        # items: dict of (K, B, ...) arrays — vmap over start positions
        items = jax.vmap(make_item)(jnp.arange(K))

        # Flatten (K, B, ...) → (K*B, ...) for the flat item buffer
        return jax.tree_util.tree_map(
            lambda x: x.reshape((K * B, *x.shape[2:])), items
        )

    return process_trajectory


def run_training_loop_jax(config: ExperimentConfig, obs_size: int, action_size: int):
    """
    Pure-JAX synchronous training loop. No Ray, no CPU actors.

    Runs entirely in a single process. JAX JIT-compiles the hot path;
    the Python loop orchestrates the buffer, checkpointing, and logging.
    """
    import optax
    import flashbax as fbx
    import orbax.checkpoint as ocp

    from model import FlaxMAMuZeroNet
    from mcts import MCTSIndependentPlanner, MCTSJointPlanner
    from envs import VecMPEEnvWrapper
    from utils.transforms import DiscreteSupport
    from utils.replay_buffer import ReplayItem
    from actors.learner_actor import make_train_step

    T = config.train.max_episode_steps
    B = config.train.num_envs
    U = config.train.unroll_steps
    N = config.train.num_agents
    K = T - U

    # ---- Model + optimizer ----
    model = FlaxMAMuZeroNet(config.model, action_size)
    rng = jax.random.PRNGKey(0)
    rng, init_key = jax.random.split(rng)
    params = model.init(init_key, jnp.ones((1, N, obs_size)))["params"]
    ema_params = params

    # decay_steps in optax = total steps; cosine portion = decay_steps - warmup_steps.
    # Must satisfy decay_steps > warmup_steps to avoid a negative cosine phase.
    decay_steps = max(config.train.num_episodes - config.train.lr_warmup_steps,
                      config.train.lr_warmup_steps + 1)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.train.learning_rate,
        warmup_steps=config.train.lr_warmup_steps,
        decay_steps=decay_steps,
        end_value=config.train.learning_rate * config.train.end_lr_factor,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.train.gradient_clip_norm),
        optax.adamw(learning_rate=lr_schedule),
    )
    opt_state = optimizer.init(params)

    # ---- Planner + train step ----
    planner_map = {"independent": MCTSIndependentPlanner, "joint": MCTSJointPlanner}
    if config.mcts.planner_mode not in planner_map:
        raise ValueError(f"Unknown planner_mode '{config.mcts.planner_mode}'")
    planner = planner_map[config.mcts.planner_mode](model=model, config=config)
    plan_fn = jax.jit(planner.plan)

    value_support  = DiscreteSupport(-config.model.value_support_size,  config.model.value_support_size)
    reward_support = DiscreteSupport(-config.model.reward_support_size, config.model.reward_support_size)
    train_step = make_train_step(model, optimizer, value_support, reward_support, config)

    # ---- Environment ----
    env = VecMPEEnvWrapper(config.train.env_name, N, T, B)

    # ---- Flashbax prioritised item buffer (GPU-resident) ----
    sample_item = {
        'observation':   jnp.zeros((N, obs_size)),
        'actions':       jnp.zeros((U, N), dtype=jnp.int32),
        'policy_target': jnp.zeros((U + 1, N, action_size)),
        'value_target':  jnp.zeros((U + 1, N)),
        'reward_target': jnp.zeros((U, N)),
        'agent_order':   jnp.zeros(N, dtype=jnp.int32),
    }
    buffer = fbx.make_prioritised_item_buffer(
        max_length=config.train.replay_buffer_size,
        min_length=config.train.batch_size,
        sample_batch_size=config.train.batch_size,
        add_batches=True,
        priority_exponent=config.train.replay_buffer_alpha,
        device="gpu",
    )
    buffer_state = buffer.init(sample_item)

    # ---- JIT-compiled data pipeline ----
    collect_episode = make_collect_episode(plan_fn, env, B, T)
    process_traj    = make_process_trajectory(N, action_size, U, config.train.n_step, config.train.discount_gamma)

    # ---- Checkpointing ----
    ckpt_dir = Path(config.train.checkpoint_dir).absolute()
    ckpt_manager = ocp.CheckpointManager(
        ckpt_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
    )
    train_step_count = 0
    latest = ckpt_manager.latest_step()
    if latest is not None:
        target = {"params": params, "opt_state": opt_state, "ema_params": ema_params, "step": np.array(0)}
        restored = ckpt_manager.restore(latest, args=ocp.args.StandardRestore(target))
        params          = restored["params"]
        opt_state       = restored["opt_state"]
        ema_params      = restored.get("ema_params", params)
        train_step_count = int(restored["step"])
        logger.info(f"Restored checkpoint from step {train_step_count}.")
    else:
        logger.info("No checkpoint found — starting fresh.")

    # ---- Training loop ----
    beta_start = config.train.replay_buffer_beta_start
    beta_frames = config.train.replay_buffer_beta_frames
    returns      = deque(maxlen=config.train.log_interval)
    train_losses = deque(maxlen=config.train.log_interval)
    interval_start = time.monotonic()
    metrics = {}

    logger.info(f"Starting pure-JAX training loop: {B} envs, {T} steps/episode.")

    for ep in range(config.train.num_episodes):
        # 1. Collect episode — MCTS + env on GPU
        rng, traj = collect_episode(params, rng)

        # 2. Episode return for logging (single scalar pull to CPU)
        ep_return = float(jnp.mean((traj.rewards * traj.active).sum(axis=0)))
        returns.append(ep_return)

        # 3. Process trajectory → flat replay items (on GPU)
        items = process_traj(traj)

        # 4. Add to buffer (Flashbax assigns max priority to new items by default)
        buffer_state = buffer.add(buffer_state, items)

        # 5. Train if buffer is warm
        if buffer.can_sample(buffer_state) and ep >= config.train.warmup_episodes:
            rng, sample_key = jax.random.split(rng)
            sample = buffer.sample(buffer_state, sample_key)

            # IS weights from sampling probabilities
            buf_size = jnp.where(
                buffer_state.is_full,
                config.train.replay_buffer_size,
                buffer_state.current_index,
            )
            beta = min(1.0, beta_start + train_step_count * (1.0 - beta_start) / beta_frames)
            weights = (buf_size * sample.probabilities) ** (-beta)
            weights = weights / weights.max()

            # Wrap sampled dict → ReplayItem (registered JAX pytree, train_step expects it)
            batch = ReplayItem(
                observation   = sample.experience['observation'],
                actions       = sample.experience['actions'],
                policy_target = sample.experience['policy_target'],
                value_target  = sample.experience['value_target'],
                reward_target = sample.experience['reward_target'],
                agent_order   = sample.experience['agent_order'],
            )

            rng, train_key = jax.random.split(rng)
            params, opt_state, metrics, new_priorities = train_step(
                params, opt_state, batch, weights, train_key, ema_params
            )
            train_step_count += 1

            # EMA update (outside JIT, same as LearnerActor)
            decay = config.train.ema_decay
            ema_params = jax.tree_util.tree_map(
                lambda e, p: decay * e + (1.0 - decay) * p, ema_params, params
            )

            # Update priorities
            buffer_state = buffer.set_priorities(
                buffer_state, sample.indices, new_priorities,
                priority_exponent=config.train.replay_buffer_alpha,
            )

            train_losses.append(float(metrics["total_loss"]))

            if config.train.debug and train_step_count % config.train.debug_interval == 0:
                logger.info(
                    f"step={train_step_count} | "
                    f"total={float(metrics['total_loss']):.4f} "
                    f"reward={float(metrics['reward_loss']):.4f} "
                    f"policy={float(metrics['policy_loss']):.4f} "
                    f"value={float(metrics['value_loss']):.4f} | "
                    f"grad_norm={float(metrics['grad_norm']):.3f}"
                )

        # 6. Checkpoint
        if (train_step_count > 0
                and train_step_count % config.train.checkpoint_interval == 0
                and ep >= config.train.warmup_episodes):
            state = {
                "params": params, "opt_state": opt_state,
                "ema_params": ema_params, "step": np.array(train_step_count),
            }
            ckpt_manager.save(train_step_count, args=ocp.args.StandardSave(state))
            ckpt_manager.wait_until_finished()
            logger.info(f"Saved checkpoint at step {train_step_count}.")

        # 7. Log every log_interval episodes
        prev = ep - 1
        if ep // config.train.log_interval > prev // config.train.log_interval and returns:
            avg_return = float(np.mean(returns))
            avg_loss   = float(np.mean(train_losses)) if train_losses else float("nan")
            elapsed    = time.monotonic() - interval_start
            eps_per_sec = config.train.log_interval / max(elapsed, 1e-6)
            logger.info(
                f"Episodes: {ep:6d} | "
                f"Avg Return: {avg_return:8.3f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"eps/s: {eps_per_sec:.2f} | "
                f"train_steps: {train_step_count}"
            )
            if config.train.wandb_mode != "disabled":
                import wandb
                log_dict = {"avg_return": avg_return, "avg_loss": avg_loss,
                            "episodes": ep, "eps_per_sec": eps_per_sec}
                if metrics:
                    log_dict.update({k: float(v) for k, v in metrics.items()})
                wandb.log(log_dict, step=ep)
            interval_start = time.monotonic()

    logger.info("Training complete.")
