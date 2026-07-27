import dataclasses
import os
import time
import numpy as np

import ray
import jax as _jax

from config import ExperimentConfig
from utils.logging_utils import logger
from utils.obs_norm import ObsRunningNorm
from utils.profiler import Profiler


# ─── Gradient scaling utilities ────────────────────────────────────────────────


@_jax.custom_vjp
def scale_grad_half(x):
    """Identity in forward pass; halves gradients in backward pass.

    Used to prevent dynamics-network updates from dominating representation
    gradients over long unrolls (MuZero paper §E, MAZero Appendix).
    """
    return x


def _scale_grad_half_fwd(x):
    """Forward pass: identity."""
    return x, ()


def _scale_grad_half_bwd(_, g):
    """Backward pass: halve the gradient."""
    return (_jax.tree_util.tree_map(lambda gi: gi * 0.5, g),)


scale_grad_half.defvjp(_scale_grad_half_fwd, _scale_grad_half_bwd)


def _awpo_weight(q: '_jax.Array', v_baseline: '_jax.Array', alpha: float) -> '_jax.Array':
    """AWAC-style exponential advantage weight, batch-normalized.

    `v_baseline` is stop-gradiented: the AWPO weight must act as a fixed
    multiplier on the policy loss, not a term the value head can shrink or
    grow to inflate its own weighting.
    """
    v_baseline = _jax.lax.stop_gradient(v_baseline)
    if v_baseline.ndim < q.ndim:
        v_baseline = v_baseline[..., None]
    adv_raw = q - v_baseline
    adv_mean = adv_raw.mean()
    adv_std = adv_raw.std()
    adv_norm = (adv_raw - adv_mean) / (adv_std + 1e-8)
    return _jax.numpy.exp(_jax.numpy.clip(adv_norm / alpha, -5.0, 5.0))


def make_train_step(model, optimizer, value_support, reward_support, config: ExperimentConfig):
    """
    Returns a JIT-compiled training step function.

    Captures model, optimizer, supports, and config in a closure so JIT only
    traces once — no static_argnames needed.

    JAX is imported lazily here; this function is only ever called from
    LearnerActor.__init__ after JAX has already been imported in that process.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from utils.transforms import scalar_to_support, support_to_scalar

    U = config.train.unroll_steps
    value_scale = config.train.value_scale
    consistency_scale = config.train.consistency_scale
    # Clamp horizon to U so range(U+1-k) is always ≥ 1.
    consistency_horizon = min(int(config.train.consistency_horizon), U)
    awpo_alpha = float(config.train.awpo_alpha)  # 0.0 = disabled
    K = config.mcts.num_gumbel_samples   # static: K sampled joint actions per MCTS node
    N = config.train.num_agents          # static: number of agents

    def train_step(params, opt_state, batch, weights, rng_key, ema_params, q_data=None):
        # Pre-compute categorical support targets outside loss_fn so they
        # are constants w.r.t. the gradient — zero gradient flows through them.
        value_target_dist = scalar_to_support(
            batch.value_target.mean(axis=2), value_support
        )   # (B, U+1, Sv)
        reward_target_dist = scalar_to_support(
            batch.reward_target.mean(axis=2), reward_support
        )   # (B, U, Sr)

        def loss_fn(p):
            rng_init, _, rng_unroll = jax.random.split(rng_key, 3)
            unroll_keys = jax.random.split(rng_unroll, U)  # (U, 2)

            # ---- Step 0: initial inference ----
            init_out = model.apply(
                {"params": p}, batch.observation, rngs={"dropout": rng_init}
            )
            hidden = init_out.hidden_state  # (B, N, D)

            # Centralized value → (B,)
            v0_loss = optax.softmax_cross_entropy(
                init_out.value_logits, value_target_dist[:, 0]
            )

            # AWPO: weight root policy loss by action-level AWAC weights.
            # Uses current V_net as baseline + batch normalization of advantages
            # so the signal is non-trivial even when Q ≈ V ≈ 0 (MAZero formulation).
            # When awpo_alpha=0 (disabled), this branch is eliminated at JIT
            # trace time and reduces to plain mean cross-entropy.
            ce_p0 = optax.softmax_cross_entropy(
                init_out.policy_logits, batch.policy_target[:, 0]
            ).mean(axis=-1)  # (B,)
            if awpo_alpha > 0.0 and q_data is not None:
                # Action-level AWPO (MAZero/AWAC formulation): weight each sampled
                # root action by exp((Q_k - V_net - mean) / (std + eps) / alpha).
                # V_net is the CURRENT network prediction (not stale stored value).
                # Batch normalization ensures non-trivial gradient signal even when
                # Q ≈ V ≈ 0 — critical for cold-start on sparse rewards like SMAX.
                q_valid       = q_data["all_child_valid"][:, 0]      # (B,) bool — root position
                q_k           = q_data["all_child_q"][:, 0]          # (B, K)
                visits_k      = q_data["all_child_visits"][:, 0]     # (B, K)
                child_actions = q_data["all_child_actions"][:, 0]    # (B, K, N)

                # Current network value prediction as AWAC baseline (not stale MCTS value)
                v_net = support_to_scalar(init_out.value_logits, value_support)  # (B,)
                awpo_w_k = _awpo_weight(q_k, v_net, awpo_alpha)  # (B, K)

                # Per-agent log-probs for each sampled joint action:
                # log_probs[b, n, a] → gather with child_actions[b, k, n]
                log_probs = jax.nn.log_softmax(init_out.policy_logits, axis=-1)  # (B, N, A)
                B_, K_ = q_k.shape
                N_, A_ = log_probs.shape[1], log_probs.shape[2]
                # Expand log_probs to (B, K, N, A) and gather at child_actions
                log_probs_exp = jnp.broadcast_to(
                    log_probs[:, None, :, :], (B_, K_, N_, A_)
                )
                gathered = jnp.take_along_axis(
                    log_probs_exp,
                    child_actions[:, :, :, None],  # (B, K, N, 1)
                    axis=-1,
                ).squeeze(-1)  # (B, K, N)
                joint_log_prob_k = gathered.sum(axis=-1)  # (B, K)

                # Visit-count-weighted AWPO loss
                visit_weights = visits_k / (visits_k.sum(axis=-1, keepdims=True) + 1e-8)
                action_awpo_loss = -(visit_weights * awpo_w_k * joint_log_prob_k).sum(axis=-1)  # (B,)

                # Fall back to plain CE for items where Q-data was not stored
                p0_loss = jnp.where(q_valid, action_awpo_loss, ce_p0)  # (B,)
            elif awpo_alpha > 0.0:
                # No Q-data available (e.g. non-OSLA planner): state-level fallback
                v_mcts = batch.value_target[:, 0].mean(axis=-1)  # (B,)
                v_net = support_to_scalar(init_out.value_logits, value_support)  # (B,)
                awpo_w = _awpo_weight(v_mcts, v_net, awpo_alpha)
                p0_loss = awpo_w * ce_p0  # (B,)
            else:
                p0_loss = ce_p0  # (B,)

            # ---- Steps 1..U: unroll via scan ----
            # Consistency is computed outside the scan so multi-step pairs
            # (h_t, h_{t+k}) for k>1 can reuse the same hidden states.
            if awpo_alpha > 0.0:
                # Build per-step Q-data for the scan (positions 1..U).
                # When q_data is None (non-OSLA planner), pass zeros with all-invalid mask
                # so scan_step always has the same input structure.
                if q_data is not None:
                    step_q_acts  = jnp.moveaxis(q_data["all_child_actions"][:, 1:], 1, 0)  # (U, B, K, N)
                    step_q_q     = jnp.moveaxis(q_data["all_child_q"][:, 1:],       1, 0)  # (U, B, K)
                    step_q_vis   = jnp.moveaxis(q_data["all_child_visits"][:, 1:],  1, 0)  # (U, B, K)
                    step_q_valid = jnp.moveaxis(q_data["all_child_valid"][:, 1:],   1, 0)  # (U, B)
                else:
                    B_ = batch.observation.shape[0]
                    step_q_acts  = jnp.zeros((U, B_, K, N), jnp.int32)
                    step_q_q     = jnp.zeros((U, B_, K),    jnp.float32)
                    step_q_vis   = jnp.zeros((U, B_, K),    jnp.float32)
                    step_q_valid = jnp.zeros((U, B_),       jnp.bool_)

                xs = (
                    jnp.moveaxis(batch.actions, 1, 0),
                    jnp.moveaxis(reward_target_dist, 1, 0),
                    jnp.moveaxis(batch.policy_target[:, 1:], 1, 0),
                    jnp.moveaxis(value_target_dist[:, 1:], 1, 0),
                    unroll_keys,
                    step_q_acts,
                    step_q_q,
                    step_q_vis,
                    step_q_valid,
                )

                def scan_step(hidden, inputs):
                    ai, ri_dist, pi_target, vi_dist, step_key, qd_acts, qd_q, qd_vis, qd_valid = inputs
                    hidden = scale_grad_half(hidden)
                    out = model.apply(
                        {"params": p}, hidden, ai,
                        method=model.recurrent_inference,
                        rngs={"dropout": step_key},
                    )
                    next_hidden = out.hidden_state
                    ri_loss = optax.softmax_cross_entropy(out.reward_logits, ri_dist)
                    vi_loss = optax.softmax_cross_entropy(out.value_logits, vi_dist)

                    # AWPO at this unroll step (mirrors root step computation)
                    ce_loss = optax.softmax_cross_entropy(
                        out.policy_logits, pi_target
                    ).mean(axis=-1)  # (B,) fallback
                    v_net_step = support_to_scalar(out.value_logits, value_support)  # (B,)
                    awpo_w = _awpo_weight(qd_q, v_net_step, awpo_alpha)  # (B, K)
                    B_ = qd_q.shape[0]
                    K_ = qd_q.shape[1]
                    N_ = out.policy_logits.shape[1]
                    A_ = out.policy_logits.shape[2]
                    log_probs = jax.nn.log_softmax(out.policy_logits, axis=-1)      # (B, N, A)
                    lp_exp = jnp.broadcast_to(log_probs[:, None, :, :], (B_, K_, N_, A_))
                    gathered = jnp.take_along_axis(
                        lp_exp, qd_acts[:, :, :, None], axis=-1
                    ).squeeze(-1)                                                    # (B, K, N)
                    jlp_k = gathered.sum(axis=-1)                                   # (B, K)
                    vis_w = qd_vis / (qd_vis.sum(axis=-1, keepdims=True) + 1e-8)   # (B, K)
                    awpo_loss = -(vis_w * awpo_w * jlp_k).sum(axis=-1)             # (B,)
                    pi_loss = jnp.where(qd_valid, awpo_loss, ce_loss)              # (B,)

                    return next_hidden, (ri_loss, pi_loss, vi_loss, next_hidden)

            else:
                xs = (
                    jnp.moveaxis(batch.actions, 1, 0),
                    jnp.moveaxis(reward_target_dist, 1, 0),
                    jnp.moveaxis(batch.policy_target[:, 1:], 1, 0),
                    jnp.moveaxis(value_target_dist[:, 1:], 1, 0),
                    unroll_keys,
                )

                def scan_step(hidden, inputs):
                    ai, ri_dist, pi_target, vi_dist, step_key = inputs
                    hidden = scale_grad_half(hidden)  # half-gradient on hidden states (MuZero paper §E)
                    out = model.apply(
                        {"params": p}, hidden, ai,
                        method=model.recurrent_inference,
                        rngs={"dropout": step_key},
                    )
                    next_hidden = out.hidden_state
                    ri_loss = optax.softmax_cross_entropy(out.reward_logits, ri_dist)
                    pi_loss = optax.softmax_cross_entropy(
                        out.policy_logits, pi_target
                    ).mean(axis=-1)
                    vi_loss = optax.softmax_cross_entropy(out.value_logits, vi_dist)
                    # next_hidden returned as output so the caller can collect all
                    # hidden states for multi-step consistency.
                    return next_hidden, (ri_loss, pi_loss, vi_loss, next_hidden)

            # Transpose to step-major for scan: (B, U, ...) → (U, B, ...)
            _, (ri_losses, pi_losses, vi_losses, scan_hiddens) = jax.lax.scan(
                scan_step, hidden, xs
            )
            # ri_losses, pi_losses, vi_losses: (U, B)
            # scan_hiddens: (U, B, N, D) — h_1 through h_U

            reward_loss = ri_losses.mean(axis=0)
            policy_loss = (p0_loss + pi_losses.sum(axis=0)) / (U + 1)
            value_loss  = (v0_loss + vi_losses.sum(axis=0)) / (U + 1)

            # ---- Multi-step SPR consistency ----
            # For each k in 1..consistency_horizon and each valid start position t,
            # compare project_online(h_t) against project_target(h_{t+k}).
            # k=1 reproduces the original single-step consistency loss.
            # k>1 adds longer-range targets, improving latent prediction accuracy
            # over multiple dynamics steps (SPR, Schwarzer et al. 2021).
            #
            # XLA CSE: project_online(h_t) is only computed once regardless of how
            # many k values reference h_t, so cost stays O(U+1) projections.
            #
            # all_hiddens[i] = h_i,  shape (U+1, B, N, D)
            all_hiddens = jnp.concatenate([hidden[jnp.newaxis], scan_hiddens], axis=0)
            cons_pairs = []
            for k in range(1, consistency_horizon + 1):
                for t in range(U + 1 - k):
                    h_t  = all_hiddens[t]        # (B, N, D)
                    h_tk = all_hiddens[t + k]    # (B, N, D)
                    online = model.apply(
                        {"params": p}, h_t, method=model.project_online
                    )
                    target = model.apply(
                        {"params": ema_params}, h_tk, method=model.project_target
                    )
                    B_, N_, D_ = online.shape
                    # epsilon=1e-8 prevents 0/0 NaN with near-zero projection norms
                    # (common early in training and with dead-agent zeroed obs).
                    sim = optax.cosine_similarity(
                        online.reshape(B_ * N_, D_),
                        target.reshape(B_ * N_, D_),
                        epsilon=1e-8,
                    ).reshape(B_, N_).mean(axis=-1)  # (B,)
                    cons_pairs.append(-sim)
            consistency_loss = jnp.stack(cons_pairs).mean(axis=0)  # (B,)

            loss = (
                reward_loss
                + policy_loss
                + value_loss * value_scale
                + consistency_loss * consistency_scale
            )
            total_loss = (loss * weights).mean()

            td_error = jnp.abs(
                support_to_scalar(init_out.value_logits, value_support)
                - batch.value_target[:, 0].mean(axis=1)
            )

            policy_probs = jax.nn.softmax(init_out.policy_logits, axis=-1)  # (B, N, A)
            policy_entropy = -jnp.sum(
                policy_probs * jnp.log(policy_probs + 1e-8), axis=-1
            ).mean()  # scalar; uniform over 9 actions = log(9) ≈ 2.197

            metric_scalars = jnp.stack([
                total_loss,
                reward_loss.mean(),
                policy_loss.mean(),
                value_loss.mean(),
                consistency_loss.mean(),
                policy_entropy,
            ])
            return total_loss, (metric_scalars, td_error)

        (_, (metric_scalars, td_error)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        grad_norm = optax.global_norm(grads)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_priorities = td_error + 1e-6

        # Pack all scalars that need D2H transfer into one contiguous array so
        # the host pays for a single PCIe DMA transaction instead of one per scalar.
        # Layout: [total, reward, policy, value, consistency, grad_norm, priorities...]
        transfer_buf = jnp.concatenate([
            metric_scalars,
            grad_norm[jnp.newaxis],
            new_priorities,
        ])

        return new_params, new_opt_state, transfer_buf, new_priorities

    return jax.jit(train_step)


@ray.remote(num_gpus=1)
class LearnerActor:
    """
    Trains the MuZero model on GPU.

    Pulls batches from the replay buffer, runs a JIT-compiled training step,
    and serves updated parameters to DataActors on request.
    """

    def __init__(self, obs_size: int, action_size: int, replay_buffer_actor, config: ExperimentConfig):
        # Don't preallocate the full MEM_FRACTION at init — data actors share
        # the same GPU. With preallocation, this process would claim 5.6 GB on
        # an 8 GB card before any data actor has started, leaving no room for them.
        # With it off, JAX allocates on demand and MEM_FRACTION acts as an upper cap.
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.70"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["GLOG_minloglevel"] = "2"

        import jax
        import jax.numpy as jnp
        import optax
        import orbax.checkpoint as ocp
        from pathlib import Path
        from utils.transforms import DiscreteSupport
        from model import FlaxMAMuZeroNet

        self.config = config
        try:
            devices = jax.devices()
            gpu_devices = [d for d in devices if "gpu" in str(d).lower() or "cuda" in str(d).lower()]
            if gpu_devices:
                logger.info(f"(Learner pid={os.getpid()}) JAX devices: {devices} — using {gpu_devices[0]}")
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        logger.info(f"(Learner) GPU memory at init (used/free/total MiB): {result.stdout.strip()}")
                except Exception as e:
                    logger.warning(f"(Learner) Could not query GPU memory: {e}")
            else:
                logger.warning(f"(Learner pid={os.getpid()}) No GPU found! Training will be slow. Devices: {devices}")
        except Exception as e:
            logger.error(f"(Learner) JAX device init failed: {e}")
            raise
        logger.info(f"(Learner pid={os.getpid()}) Initializing on GPU...")

        self.replay_buffer = replay_buffer_actor
        self.train_step_count = 0
        self.rng_key = jax.random.PRNGKey(0)

        value_support = DiscreteSupport(
            min=-config.model.value_support_size,
            max=config.model.value_support_size,
        )
        reward_support = DiscreteSupport(
            min=-config.model.reward_support_size,
            max=config.model.reward_support_size,
        )

        model = FlaxMAMuZeroNet(config.model, action_size)
        dummy_obs = jnp.ones((1, config.train.num_agents, obs_size))
        self.rng_key, init_key = jax.random.split(self.rng_key)
        self.params = model.init(init_key, dummy_obs)["params"]

        self._use_obs_norm = config.model.use_obs_normalization
        self.obs_norm = ObsRunningNorm(obs_size) if self._use_obs_norm else None

        lr = config.train.learning_rate
        # Clamp warmup so short diagnostic runs (num_episodes=300) don't produce
        # a negative decay_steps and crash cosine_decay_schedule.
        _warmup = min(config.train.lr_warmup_steps, config.train.num_episodes // 2)
        _decay  = max(1, config.train.num_episodes - _warmup)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=_warmup,
            decay_steps=_decay,
            end_value=lr * config.train.end_lr_factor,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.train.gradient_clip_norm),
            optax.adamw(learning_rate=lr_schedule),
        )
        self.opt_state = optimizer.init(self.params)
        self.lr_schedule = lr_schedule

        self.train_step = make_train_step(
            model, optimizer, value_support, reward_support, config
        )

        # JIT the EMA update. Without JIT, tree_map dispatches one GPU kernel
        # per parameter leaf (dozens of small launches). JIT fuses them into one.
        decay = config.train.ema_decay
        self.ema_update = jax.jit(lambda ema, params: jax.tree_util.tree_map(
            lambda e, p: decay * e + (1.0 - decay) * p, ema, params
        ))

        # Checkpointing — restore latest checkpoint if one exists.
        ckpt_dir = Path(config.train.checkpoint_dir).absolute()
        self.ckpt_manager = ocp.CheckpointManager(
            ckpt_dir,
            options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
        )
        # EMA parameters for the target encoder (BYOL-style consistency loss).
        # Initialized to match the online params; updated after every training step.
        self.ema_params = self.params

        latest = self.ckpt_manager.latest_step()
        if latest is not None:
            target = {
                "params": self.params,
                "opt_state": self.opt_state,
                "ema_params": self.ema_params,
                "step": np.array(0),
            }
            restored = self.ckpt_manager.restore(
                latest, args=ocp.args.StandardRestore(target)
            )
            self.params = restored["params"]
            self.opt_state = restored["opt_state"]
            self.ema_params = restored.get("ema_params", self.params)
            self.train_step_count = int(restored["step"])
            if self._use_obs_norm and "obs_norm_mean" in restored:
                self.obs_norm = ObsRunningNorm.from_state(
                    {"mean": restored["obs_norm_mean"], "var": restored["obs_norm_var"]},
                    obs_size,
                )
            logger.info(
                f"(Learner pid={os.getpid()}) Restored checkpoint from step {self.train_step_count}."
            )
        else:
            logger.info(f"(Learner pid={os.getpid()}) No checkpoint found — starting fresh.")

        # Kick off the first prefetch so train() has a batch ready immediately.
        self._prefetch_future = self.replay_buffer.sample.remote(config.train.batch_size)

        # Cache lr at step 0; refreshed every lr_log_interval steps to avoid
        # per-step JAX dispatch overhead from calling the schedule function.
        self._cached_lr: float = float(lr_schedule(0))
        self._lr_log_interval = max(1, config.train.debug_interval)

        self.profiler = Profiler("learner", log_interval=config.train.debug_interval)

        logger.info(f"(Learner pid={os.getpid()}) Setup complete.")

    def _prefetch_batch(self):
        """Fires a non-blocking sample request and stores the future."""
        self._prefetch_future = self.replay_buffer.sample.remote(
            self.config.train.batch_size
        )

    def _train_step(self):
        """Runs one training step. Returns metrics dict or None if buffer empty."""
        import jax

        with self.profiler.time("sample_wait"):
            batch, weights, indices, q_data = ray.get(self._prefetch_future)
        if batch is None:
            self._prefetch_batch()
            return None

        # Fire the next buffer sample immediately so it overlaps with GPU compute.
        self._prefetch_batch()

        # Observation normalization: update running stats then normalize the batch.
        # Done on CPU (numpy) before device_put so the GPU only sees clean inputs.
        if self._use_obs_norm:
            self.obs_norm.update(batch.observation)
            batch = dataclasses.replace(
                batch, observation=self.obs_norm.normalize(batch.observation)
            )

        # Dispatch H2D transfer immediately after getting the batch (JAX async —
        # returns a future-like DeviceArray; actual DMA runs in background).
        # This overlaps the 3ms PCIe transfer with rng_split + any other CPU work,
        # so the GPU sees the batch ready by the time train_step is dispatched.
        jax_batch = jax.tree_util.tree_map(jax.device_put, batch)
        jax_weights = jax.device_put(np.array(weights, dtype=np.float32))

        with self.profiler.time("rng_split"):
            self.rng_key, train_key = jax.random.split(self.rng_key)

        with self.profiler.time("device_put"):
            # Ensure the async H2D transfer dispatched above has completed before
            # train_step consumes the arrays.
            jax.block_until_ready((jax_batch, jax_weights))

        with self.profiler.time("train_step"):
            # Convert q_data to JAX arrays for device; q_valid mask gates AWPO.
            jax_q_data = {
                k: jax.device_put(np.asarray(v)) for k, v in q_data.items()
            } if q_data is not None else None
            self.params, self.opt_state, transfer_buf, new_priorities = self.train_step(
                self.params, self.opt_state, jax_batch, jax_weights, train_key,
                self.ema_params, jax_q_data,
            )
            # Block on params and the transfer buffer (which contains metrics +
            # priorities). new_priorities is a slice of transfer_buf so blocking
            # on transfer_buf covers it too — one wait for all GPU outputs.
            jax.block_until_ready((self.params, transfer_buf))
        self.train_step_count += 1

        # Dispatch EMA update without blocking — it runs asynchronously on GPU while
        # the CPU does D2H transfer and bookkeeping. JAX will implicitly wait for
        # ema_params when it's consumed by the next train_step call.
        with self.profiler.time("ema_update"):
            self.ema_params = self.ema_update(self.ema_params, self.params)

        with self.profiler.time("d2h_transfer"):
            # Single PCIe DMA: transfer_buf = [total, reward, policy, value,
            # consistency, grad_norm, priority_0, ..., priority_{B-1}]
            buf_np = np.array(transfer_buf)
        # EMA runs async on GPU; by now (after d2h_transfer ~1.7ms) it is likely
        # done, but JAX will sync it implicitly when next train_step uses ema_params.

        N_METRICS = 7  # total, reward, policy, value, consistency, policy_entropy, grad_norm
        priorities_np = buf_np[N_METRICS:]
        self.replay_buffer.update_priorities.remote(indices, priorities_np)

        if self.train_step_count % self.config.train.checkpoint_interval == 0:
            self._save_checkpoint()

        METRIC_KEYS = ["total_loss", "reward_loss", "policy_loss", "value_loss",
                       "consistency_loss", "policy_entropy", "grad_norm"]
        metrics = dict(zip(METRIC_KEYS, buf_np[:N_METRICS].tolist()))

        # Refresh cached lr every lr_log_interval steps; avoids per-step JAX dispatch.
        if self.train_step_count % self._lr_log_interval == 0:
            self._cached_lr = float(self.lr_schedule(self.train_step_count))
        metrics["learning_rate"] = self._cached_lr

        total_loss = metrics["total_loss"]
        grad_norm = metrics["grad_norm"]
        if not np.isfinite(total_loss):
            # Log each component so we can identify the source.
            bad = [k for k in METRIC_KEYS if not np.isfinite(metrics[k])]
            logger.warning(
                f"(Learner) step={self.train_step_count} non-finite total_loss={total_loss:.4f} "
                f"| bad components: {bad} "
                f"| reward={metrics['reward_loss']:.4f} policy={metrics['policy_loss']:.4f} "
                f"value={metrics['value_loss']:.4f} consistency={metrics['consistency_loss']:.4f}"
            )
        if not np.isfinite(grad_norm):
            logger.warning(f"(Learner) step={self.train_step_count} non-finite grad_norm={grad_norm:.4f}")

        debug = self.config.train.debug
        debug_interval = self.config.train.debug_interval
        if debug and self.train_step_count % debug_interval == 0:
            logger.info(
                f"(Learner) step={self.train_step_count} | "
                f"total={total_loss:.4f} "
                f"reward={metrics['reward_loss']:.4f} "
                f"policy={metrics['policy_loss']:.4f} "
                f"value={metrics['value_loss']:.4f} "
                f"consistency={metrics['consistency_loss']:.4f} "
                f"entropy={metrics['policy_entropy']:.3f} | "
                f"grad_norm={grad_norm:.3f}"
            )

        return metrics

    def run_training_loop(self, num_steps: int):
        """Runs a tight internal training loop for num_steps steps.

        Called once from the main loop per log interval instead of once per
        step. Eliminates Ray round-trip overhead between training steps —
        the learner stays on GPU continuously rather than waiting for the
        main process to re-dispatch it after each step.
        """
        metrics = None
        for _ in range(num_steps):
            result = self._train_step()
            if result is not None:
                metrics = result
        return metrics  # last non-None metrics, or None if buffer was empty

    # Keep a single-step entry point for the sync training loop.
    def train(self):
        return self._train_step()

    def _save_checkpoint(self):
        import orbax.checkpoint as ocp
        state = {
            "params": self.params,
            "opt_state": self.opt_state,
            "ema_params": self.ema_params,
            "step": np.array(self.train_step_count),
        }
        if self._use_obs_norm:
            norm_s = self.obs_norm.state()
            state["obs_norm_mean"] = norm_s["mean"]
            state["obs_norm_var"] = norm_s["var"]
        self.ckpt_manager.save(self.train_step_count, args=ocp.args.StandardSave(state))
        self.ckpt_manager.wait_until_finished()
        logger.info(f"(Learner) Saved checkpoint at step {self.train_step_count}.")

    def get_params(self):
        norm_state = self.obs_norm.state() if self._use_obs_norm else None
        return {"params": self.params, "norm_state": norm_state}

    def get_train_step_count(self) -> int:
        return self.train_step_count
