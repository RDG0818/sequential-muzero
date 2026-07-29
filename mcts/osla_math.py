# mcts/osla_math.py
"""Pure OS(λ) value/UCB/policy helper functions for MCTSJointOSLAPlanner —
no JAX closures over tree state, each independently unit-tested."""

import chex
import jax
import jax.numpy as jnp


def compute_osla_value_jax(
    sim_values: chex.Array,
    sim_depths: chex.Array,
    n_visits: chex.Array,
    rho: float,
    lam: float,
    max_depth: int,
) -> chex.Array:
    """OS(λ) value estimate, paper Eq. 5-7 (top-(1-rho) quantile per depth
    bucket, weighted by lambda^depth), matching MAZero's `SubTreeValueSet`.
    Uses one global sort + per-bucket cumulative rank instead of one argsort
    per depth bucket: O(max_sims log max_sims) instead of O(max_depth *
    max_sims log max_sims).
    """
    max_sims = sim_values.shape[0]
    valid_mask = jnp.arange(max_sims) < n_visits

    order = jnp.argsort(jnp.where(valid_mask, sim_values, -jnp.inf))[::-1]
    v_s = sim_values[order]
    d_s = sim_depths[order]
    m_s = valid_mask[order]

    onehot = jax.nn.one_hot(d_s, max_depth) * m_s[:, None]
    rank_in_bucket = (jnp.cumsum(onehot, axis=0) - 1.0)[jnp.arange(max_sims), d_s]
    counts = onehot.sum(axis=0)
    size_lim = jnp.maximum(1, jnp.ceil(counts * (1.0 - rho))).astype(jnp.int32)

    include = m_s & (rank_in_bucket < size_lim[d_s])
    weight = jnp.where(include, lam ** d_s.astype(jnp.float32), 0.0)

    return (v_s * weight).sum() / (weight.sum() + 1e-8)


def compute_osla_value(
    sim_depths: chex.Array,
    sim_values: chex.Array,
    rho: float = 0.25,
    lam: float = 0.8,
    max_depth: int = 20,
) -> chex.Array:
    """Non-padded convenience wrapper over `compute_osla_value_jax` for
    callers where every entry in `sim_values`/`sim_depths` is valid.
    """
    return compute_osla_value_jax(
        sim_values, sim_depths, jnp.array(sim_values.shape[0]), rho, lam, max_depth
    )


def compute_ucb_scores(
    child_q_baseline_diff: chex.Array,
    child_visits: chex.Array,
    prior_probs: chex.Array,
    parent_visits: chex.Array,
    qmin: chex.Array,
    qmax: chex.Array,
    pb_c_base: float,
    pb_c_init: float,
    value_delta_lb: float,
) -> chex.Array:
    """PUCT with MuZero/MAZero log-visit pb_c scaling and min-max Q
    normalization (matches `CTree::ucb_score` in MAZero's cnode.cpp).
    """
    pb_c = jnp.log((parent_visits + pb_c_base + 1.0) / pb_c_base) + pb_c_init
    pb_c = pb_c * jnp.sqrt(parent_visits) / (1.0 + child_visits)
    prior_score = pb_c * prior_probs

    has_stats = qmax > qmin
    delta = jnp.maximum(value_delta_lb, qmax - qmin)
    normalized = jnp.where(has_stats, (child_q_baseline_diff - qmin) / delta, child_q_baseline_diff)
    value_score = jnp.where(child_visits > 0, jnp.clip(normalized, 0.0, 1.0), 0.0)

    return prior_score + value_score


def _sample_k_actions(
    rng: chex.Array,
    logits: chex.Array,
    K: int,
    A_N: int,
) -> tuple[chex.Array, chex.Array]:
    """Sample K joint actions proportional to prior; return (actions [K] int32, probs [K] float32)."""
    probs = jax.nn.softmax(logits)
    replace = K >= A_N
    actions = jax.random.choice(rng, A_N, shape=(K,), replace=replace, p=probs)
    return actions, probs[actions]


def _logits_to_joint_logits(
    per_agent_logits: chex.Array,
    N: int,
) -> chex.Array:
    """Per-agent logits (N, A) → joint logits (A_N,) under independence assumption."""
    A = per_agent_logits.shape[-1]
    log_probs = jax.nn.log_softmax(per_agent_logits, axis=-1)
    joint = log_probs[0]
    for i in range(1, N):
        joint = (joint[:, None] + log_probs[i][None, :]).reshape(-1)
    return joint


def _joint_policy_to_marginal(
    joint_policy: chex.Array,
    N: int,
    joint_action_shape: tuple,
) -> chex.Array:
    """Joint policy (B, A_N) → per-agent marginals (B, N, A)."""
    B = joint_policy.shape[0]
    reshaped = joint_policy.reshape(B, *joint_action_shape)
    marginals = []
    for i in range(N):
        other_axes = tuple(j + 1 for j in range(N) if j != i)
        marginals.append(jnp.sum(reshaped, axis=other_axes))
    return jnp.stack(marginals, axis=1)
