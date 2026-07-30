# actors/loss.py
"""Pure-JAX loss/train-step logic for the learner — no Ray import, so this
module is importable and unit-testable without spinning up a Ray actor."""

import jax as _jax

from config import ExperimentConfig


def make_optimizer(config: ExperimentConfig):
    """Builds the optimizer and its LR schedule from TrainConfig.

    Shared by LearnerActor and eval.py so both construct an identically
    shaped opt_state to restore checkpoints against.
    """
    import optax

    lr = config.train.learning_rate
    # Clamp warmup so short diagnostic runs (num_episodes=300) don't produce
    # a negative decay_steps and crash cosine_decay_schedule.
    warmup = min(config.train.lr_warmup_steps, config.train.num_episodes // 2)
    decay = max(1, config.train.num_episodes - warmup)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=decay,
        end_value=lr * config.train.end_lr_factor,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.train.gradient_clip_norm),
        optax.adamw(learning_rate=lr_schedule),
    )
    return optimizer, lr_schedule


@_jax.custom_vjp
def scale_grad_half(x):
    """Identity in forward pass; halves gradients in backward pass.

    Used to prevent dynamics-network updates from dominating representation
    gradients over long unrolls (MuZero paper §E, MAZero Appendix).
    """
    return x


def _scale_grad_half_fwd(x):
    return x, ()


def _scale_grad_half_bwd(_, g):
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


def unimix_cross_entropy(logits: '_jax.Array', target_probs: '_jax.Array', unimix_ratio: float) -> '_jax.Array':
    """Cross-entropy against target_probs, with unimix_ratio uniform mass
    mixed into the predicted distribution before taking the log.

    unimix_ratio=0.0 is exactly optax.softmax_cross_entropy — the smoothing
    only engages when a config opts in. Reference: DreamerV3's 1% unimix for
    categoricals (Hafner et al., Nature 2025) — adapted here to the
    value/reward heads, which the paper's own critic does not smooth
    (DreamerV3 applies unimix to the stochastic-latent and actor
    categoricals; its critic uses symlog twohot without unimix). Applied
    here to the value/reward categorical heads only, not the policy head
    (this repo already applies Dirichlet noise to the policy at the MCTS
    root, which covers similar ground on exploration — unimix's distinct
    contribution here is stabilizing the value/reward heads, not policy
    exploration).

    Note: this smooths the loss target's effective comparison distribution
    only. `support_to_scalar`'s decode is an unconditional softmax with no
    smoothing, so a nonzero unimix_ratio introduces a small systematic bias
    between the decoded value/reward estimate and the loss's actual
    minimizer — worth accounting for when comparing metrics across ablation
    arms with different unimix_ratio values.
    """
    import optax
    if unimix_ratio == 0.0:
        return optax.softmax_cross_entropy(logits, target_probs)
    probs = _jax.nn.softmax(logits, axis=-1)
    num_classes = probs.shape[-1]
    smoothed = (1.0 - unimix_ratio) * probs + unimix_ratio / num_classes
    return -_jax.numpy.sum(target_probs * _jax.numpy.log(smoothed + 1e-8), axis=-1)


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
    unimix_ratio = float(config.model.unimix_ratio)  # 0.0 = disabled

    def train_step(params, opt_state, batch, weights, rng_key, ema_params, q_data):
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
            v0_loss = unimix_cross_entropy(
                init_out.value_logits, value_target_dist[:, 0], unimix_ratio
            )

            # AWPO root policy loss (paper Eq. 10, 14); branch eliminated at
            # JIT trace time based on awpo_alpha.
            if awpo_alpha > 0.0:
                # Action-level AWPO: weight each sampled root action by
                # exp((Q_k - V_net) / alpha), batch-normalized (_awpo_weight).
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
                p0_loss = -(visit_weights * awpo_w_k * joint_log_prob_k).sum(axis=-1)  # (B,)
            else:
                ce_p0 = optax.softmax_cross_entropy(
                    init_out.policy_logits, batch.policy_target[:, 0]
                ).mean(axis=-1)  # (B,)
                p0_loss = ce_p0  # (B,)

            # ---- Steps 1..U: unroll via scan ----
            # Consistency is computed outside the scan so multi-step pairs
            # (h_t, h_{t+k}) for k>1 can reuse the same hidden states.
            if awpo_alpha > 0.0:
                # Build per-step Q-data for the scan (positions 1..U).
                step_q_acts  = jnp.moveaxis(q_data["all_child_actions"][:, 1:], 1, 0)  # (U, B, K, N)
                step_q_q     = jnp.moveaxis(q_data["all_child_q"][:, 1:],       1, 0)  # (U, B, K)
                step_q_vis   = jnp.moveaxis(q_data["all_child_visits"][:, 1:],  1, 0)  # (U, B, K)

                xs = (
                    jnp.moveaxis(batch.actions, 1, 0),
                    jnp.moveaxis(reward_target_dist, 1, 0),
                    jnp.moveaxis(batch.policy_target[:, 1:], 1, 0),
                    jnp.moveaxis(value_target_dist[:, 1:], 1, 0),
                    unroll_keys,
                    step_q_acts,
                    step_q_q,
                    step_q_vis,
                )

                def scan_step(hidden, inputs):
                    ai, ri_dist, pi_target, vi_dist, step_key, qd_acts, qd_q, qd_vis = inputs
                    hidden = scale_grad_half(hidden)
                    out = model.apply(
                        {"params": p}, hidden, ai,
                        method=model.recurrent_inference,
                        rngs={"dropout": step_key},
                    )
                    next_hidden = out.hidden_state
                    ri_loss = unimix_cross_entropy(out.reward_logits, ri_dist, unimix_ratio)
                    vi_loss = unimix_cross_entropy(out.value_logits, vi_dist, unimix_ratio)

                    # AWPO at this unroll step (mirrors root step computation)
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
                    pi_loss = -(vis_w * awpo_w * jlp_k).sum(axis=-1)               # (B,)

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
                    ri_loss = unimix_cross_entropy(out.reward_logits, ri_dist, unimix_ratio)
                    pi_loss = optax.softmax_cross_entropy(
                        out.policy_logits, pi_target
                    ).mean(axis=-1)
                    vi_loss = unimix_cross_entropy(out.value_logits, vi_dist, unimix_ratio)
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

            # ---- Multi-step SPR consistency (Schwarzer et al. 2021) ----
            # For each k in 1..consistency_horizon and start t, compare
            # project_online(h_t) vs project_target(h_{t+k}); k=1 is the
            # original single-step loss. XLA CSE means project_online(h_t)
            # is computed once regardless of how many k reference it.
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
