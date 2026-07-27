# mcts/mcts_joint_osla.py

import functools

import chex
import jax
import jax.numpy as jnp
import mctx

from model import FlaxMAMuZeroNet
import utils.transforms as utils
from config import ExperimentConfig
from mcts.base import MCTSPlanner, MCTSPlanOutput


def compute_osla_value_jax(
    sim_values: chex.Array,   # [max_sims] float32 — padded with 0 after n_visits
    sim_depths: chex.Array,   # [max_sims] int32 — node-to-leaf depth per sim
    n_visits: chex.Array,     # scalar int32 — number of valid entries
    rho: float,
    lam: float,
    max_depth: int,           # static — number of depth buckets to scan (0..max_depth-1)
) -> chex.Array:
    """JAX-native OS(λ), matching MAZero's `SubTreeValueSet`: the top
    (1-rho) quantile is selected SEPARATELY within each depth bucket, then
    buckets are combined with lambda^depth weighting. Pooling all depths
    together before ranking (the previous behavior) lets one depth's
    naturally larger/smaller value scale dominate which sims count as "top",
    instead of comparing sims fairly within their own depth bucket.
    """
    max_sims = sim_values.shape[0]
    valid_mask = jnp.arange(max_sims) < n_visits

    def bucket_contribution(d):
        depth_mask = valid_mask & (sim_depths == d)
        count_d = depth_mask.sum()
        size_lim = jnp.maximum(
            1, jnp.ceil(count_d.astype(jnp.float32) * (1.0 - rho)).astype(jnp.int32)
        )
        masked_vals = jnp.where(depth_mask, sim_values, -jnp.inf)
        order = jnp.argsort(masked_vals)[::-1]
        ranked_mask = depth_mask[order]
        rank = jnp.arange(max_sims)
        include = (rank < size_lim) & ranked_mask
        sorted_vals = sim_values[order]
        weight = lam ** jnp.float32(d)
        has_any = count_d > 0
        bucket_sum = jnp.where(has_any, jnp.where(include, sorted_vals, 0.0).sum() * weight, 0.0)
        bucket_count = jnp.where(has_any, include.sum().astype(jnp.float32) * weight, 0.0)
        return bucket_sum, bucket_count

    bucket_sums, bucket_counts = jax.vmap(bucket_contribution)(jnp.arange(max_depth))
    return bucket_sums.sum() / (bucket_counts.sum() + 1e-8)


def compute_osla_value(
    sim_depths: chex.Array,    # [K] int32 — simulation leaf depths
    sim_values: chex.Array,    # [K] float32 — backup values from root
    rho: float = 0.25,
    lam: float = 0.8,          # depth decay weight
    max_depth: int = 20,
) -> chex.Array:
    """Non-padded convenience wrapper over `compute_osla_value_jax` for
    callers (e.g. the root value in `_osla_plan_single`) where every entry
    in `sim_values`/`sim_depths` is valid — no `n_visits` padding needed.
    """
    return compute_osla_value_jax(
        sim_values, sim_depths, jnp.array(sim_values.shape[0]), rho, lam, max_depth
    )


# ─── Tree data structure ──────────────────────────────────────────────────────

@chex.dataclass
class OSLATree:
    """Static-shaped search tree for one environment instance (no batch dim).

    max_nodes = num_simulations + 1 (root + 1 new leaf per simulation).
    K = num_gumbel_samples (sampled joint actions per node).
    """
    visit_counts:     chex.Array  # [max_nodes] int32
    value_sum:        chex.Array  # [max_nodes] float32
    reward:           chex.Array  # [max_nodes] float32 — reward to enter this node
    pred_value:       chex.Array  # [max_nodes] float32 — raw network value prediction at expansion
    embedding:        chex.Array  # [max_nodes, N, D] float32
    depth:            chex.Array  # [max_nodes] int32
    parent:           chex.Array  # [max_nodes] int32 (-1 = root)
    child_actions:    chex.Array  # [max_nodes, K] int32 — flat joint action indices
    child_node_idx:   chex.Array  # [max_nodes, K] int32 (-1 = not yet expanded)
    child_prior_prob: chex.Array  # [max_nodes, K] float32
    # Per-node simulation data for OS(λ) UCB selection
    node_sim_values:  chex.Array  # [max_nodes, num_sims+1] float32
    node_sim_depths:  chex.Array  # [max_nodes, num_sims+1] int32


@chex.dataclass
class SimCarry:
    """Outer fori_loop state across all simulations."""
    tree:       OSLATree
    next_free:  chex.Array   # [] int32 — next available node index
    rng:        chex.Array   # PRNGKey
    sim_depths: chex.Array   # [num_simulations] int32
    sim_values: chex.Array   # [num_simulations] float32
    qmin:       chex.Array   # [] float32 — running min Q-baseline-diff seen in this tree
    qmax:       chex.Array   # [] float32 — running max Q-baseline-diff seen in this tree


@chex.dataclass
class SelectCarry:
    """Inner while_loop state during tree selection."""
    node_idx:     chex.Array  # [] int32 — current node
    depth:        chex.Array  # [] int32
    path_nodes:   chex.Array  # [max_depth+1] int32
    path_k:       chex.Array  # [max_depth+1] int32
    path_rewards: chex.Array  # [max_depth+1] float32 — reward[path_nodes[i]]
    best_k:       chex.Array  # [] int32 — UCB-best child of current node
    child_idx:    chex.Array  # [] int32 — child_node_idx[node, best_k]
    # osla_node_values removed: OS(λ) is computed lazily per-node in _best_ucb
    # over only K children rather than all max_nodes, cutting O(max_nodes) → O(K)


# ─── UCB helper ───────────────────────────────────────────────────────────────

def compute_ucb_scores(
    child_q_baseline_diff: chex.Array,  # [K] float32 — get_qsa(child) - parent.pred_value
    child_visits:  chex.Array,  # [K] float32 — visit counts (0 = unvisited)
    prior_probs:   chex.Array,  # [K] float32 — prior probabilities
    parent_visits: chex.Array,  # [] float32 — total visits at parent BEFORE this simulation
    qmin: chex.Array,           # [] float32 — running min Q-baseline-diff seen in this tree
    qmax: chex.Array,           # [] float32 — running max Q-baseline-diff seen in this tree
    pb_c_base: float,
    pb_c_init: float,
    value_delta_lb: float,
) -> chex.Array:
    """PUCT with MuZero/MAZero-style log-visit exploration scaling and
    min-max Q normalization (matches `CTree::ucb_score` in MAZero's
    cnode.cpp): the prior-exploration term grows with log(parent_visits)
    instead of a fixed constant, and the Q-value term is normalized into
    [0, 1] by the running min/max seen anywhere in the tree so far, so it's
    comparable in scale to the prior term regardless of the value support's
    absolute range.
    """
    pb_c = jnp.log((parent_visits + pb_c_base + 1.0) / pb_c_base) + pb_c_init
    pb_c = pb_c * jnp.sqrt(parent_visits) / (1.0 + child_visits)
    prior_score = pb_c * prior_probs

    has_stats = qmax > qmin
    delta = jnp.maximum(value_delta_lb, qmax - qmin)
    normalized = jnp.where(has_stats, (child_q_baseline_diff - qmin) / delta, child_q_baseline_diff)
    value_score = jnp.where(child_visits > 0, jnp.clip(normalized, 0.0, 1.0), 0.0)

    return prior_score + value_score


# ─── Action sampling helper ───────────────────────────────────────────────────

def _sample_k_actions(
    rng: chex.Array,
    logits: chex.Array,   # [A_N] joint action logits
    K: int,               # number of actions to sample (static)
    A_N: int,             # joint action space size (static)
) -> tuple[chex.Array, chex.Array]:
    """Sample K joint actions proportional to prior; return (actions [K] int32, probs [K] float32)."""
    probs = jax.nn.softmax(logits)
    replace = K >= A_N
    actions = jax.random.choice(rng, A_N, shape=(K,), replace=replace, p=probs)
    return actions, probs[actions]


# ─── Single simulation step ───────────────────────────────────────────────────

def _run_single_sim(
    carry: SimCarry,
    sim_idx: chex.Array,          # [] int32
    params,
    recurrent_fn,                 # (params, rng, flat_action[1], embedding[1,N,D]) -> (RecurrentFnOutput, next_embedding[1,N,D])
    K: int,                       # static
    A_N: int,                     # static
    max_depth: int,               # static
    gamma: float,
    rho: float = 0.75,
    lam: float = 0.8,
    pb_c_base: float = 19652.0,
    pb_c_init: float = 1.25,
    value_delta_lb: float = 0.01,
) -> SimCarry:
    """Run one MCTS simulation: selection → expansion → backup → record (depth, value)."""
    tree = carry.tree
    rng, expand_rng = jax.random.split(carry.rng)

    # ── 1. Helper: best UCB child index for a given node ─────────────────────
    # Compute OS(λ) lazily over only K children rather than all max_nodes.
    # The eager vmap(max_nodes) ran 102 argsort(101) per simulation; this
    # runs K=5 argsort(num_sims+1) only at the nodes actually visited.
    def _best_ucb(node_idx):
        child_node_idxs = tree.child_node_idx[node_idx]   # [K]
        safe_idxs = jnp.maximum(child_node_idxs, 0)
        child_visits = jnp.where(
            child_node_idxs >= 0,
            tree.visit_counts[safe_idxs].astype(jnp.float32),
            0.0,
        )
        # OS(λ) over each child's accumulated simulation history [K, num_sims+1]
        child_osla_v = jax.vmap(
            lambda v, d, n: compute_osla_value_jax(v, d, n, rho, lam, max_depth + 2)
        )(
            tree.node_sim_values[safe_idxs],   # [K, num_sims+1]
            tree.node_sim_depths[safe_idxs],   # [K, num_sims+1]
            jnp.where(child_node_idxs >= 0, tree.visit_counts[safe_idxs],
                      jnp.zeros(K, jnp.int32)),
        )  # [K]
        child_qsa = jnp.where(
            child_node_idxs >= 0,
            tree.reward[safe_idxs] + gamma * child_osla_v,
            0.0,
        )  # [K]
        parent_pred_value = tree.pred_value[node_idx]
        child_q_baseline_diff = child_qsa - parent_pred_value
        prior_probs = tree.child_prior_prob[node_idx]
        parent_visits = jnp.maximum(tree.visit_counts[node_idx].astype(jnp.float32) - 1.0, 0.0)
        ucb = compute_ucb_scores(
            child_q_baseline_diff, child_visits, prior_probs, parent_visits,
            carry.qmin, carry.qmax, pb_c_base, pb_c_init, value_delta_lb,
        )
        ucb_choice = jnp.argmax(ucb).astype(jnp.int32)
        return ucb_choice

    # ── 2. Selection via while_loop ───────────────────────────────────────────
    init_bk = _best_ucb(jnp.array(0, jnp.int32))
    init_cidx = tree.child_node_idx[0, init_bk]

    init_sc = SelectCarry(
        node_idx=jnp.array(0, jnp.int32),
        depth=jnp.array(0, jnp.int32),
        path_nodes=jnp.full(max_depth + 1, 0, jnp.int32),
        path_k=jnp.full(max_depth + 1, 0, jnp.int32),
        path_rewards=jnp.zeros(max_depth + 1, jnp.float32),
        best_k=init_bk,
        child_idx=init_cidx,
    )

    def _select_cond(sc: SelectCarry) -> bool:
        return (sc.depth < max_depth) & (sc.child_idx >= 0)

    def _select_body(sc: SelectCarry) -> SelectCarry:
        path_nodes = sc.path_nodes.at[sc.depth].set(sc.node_idx)
        path_k = sc.path_k.at[sc.depth].set(sc.best_k)
        child_reward = tree.reward[sc.child_idx]
        path_rewards = sc.path_rewards.at[sc.depth + 1].set(child_reward)
        new_bk = _best_ucb(sc.child_idx)
        new_cidx = tree.child_node_idx[sc.child_idx, new_bk]
        return SelectCarry(
            node_idx=sc.child_idx,
            depth=sc.depth + 1,
            path_nodes=path_nodes,
            path_k=path_k,
            path_rewards=path_rewards,
            best_k=new_bk,
            child_idx=new_cidx,
        )

    sc = jax.lax.while_loop(_select_cond, _select_body, init_sc)

    # After selection: sc.node_idx is the deepest expanded node (the parent of the new leaf).
    # sc.best_k is which child to expand. sc.depth is depth of sc.node_idx.
    parent_node = sc.node_idx
    expand_k = sc.best_k
    leaf_depth = sc.depth + 1  # new leaf will be at this depth

    # Record parent in path at sc.depth
    path_nodes = sc.path_nodes.at[sc.depth].set(parent_node)
    path_k = sc.path_k.at[sc.depth].set(expand_k)

    # ── 3. Expansion ─────────────────────────────────────────────────────────
    parent_embedding = tree.embedding[parent_node]          # [N, D]
    flat_action = tree.child_actions[parent_node, expand_k] # [] int32

    rec_out, new_embedding = recurrent_fn(
        params, expand_rng,
        flat_action[None],           # [1] — batched
        parent_embedding[None],      # [1, N, D] — batched
    )
    new_embedding = new_embedding[0]     # [N, D]
    leaf_reward = rec_out.reward[0]      # scalar
    leaf_value = rec_out.value[0]        # scalar
    leaf_prior_logits = rec_out.prior_logits[0]  # [A_N]

    # Sample K children for the new node
    sample_rng, _ = jax.random.split(expand_rng)
    new_child_actions, new_child_probs = _sample_k_actions(sample_rng, leaf_prior_logits, K, A_N)

    new_node_idx = carry.next_free

    # Write new node into tree
    tree = tree.replace(
        embedding=tree.embedding.at[new_node_idx].set(new_embedding),
        reward=tree.reward.at[new_node_idx].set(leaf_reward),
        pred_value=tree.pred_value.at[new_node_idx].set(leaf_value),
        depth=tree.depth.at[new_node_idx].set(leaf_depth),
        parent=tree.parent.at[new_node_idx].set(parent_node),
        child_actions=tree.child_actions.at[new_node_idx].set(new_child_actions),
        child_node_idx=tree.child_node_idx.at[new_node_idx].set(
            jnp.full(K, -1, jnp.int32)
        ),
        child_prior_prob=tree.child_prior_prob.at[new_node_idx].set(new_child_probs),
    )
    # Link parent → new node
    tree = tree.replace(
        child_node_idx=tree.child_node_idx.at[parent_node, expand_k].set(new_node_idx),
    )

    # Include new leaf in path
    path_nodes = path_nodes.at[leaf_depth].set(new_node_idx)
    path_rewards = sc.path_rewards.at[leaf_depth].set(leaf_reward)

    # ── 4. Backup ─────────────────────────────────────────────────────────────
    # Walk from leaf (leaf_depth) up to root (depth 0), updating visit_counts and value_sum.
    # V at each node = reward_into_that_node + gamma * V_below.
    # We go from leaf upward: at step k=0, update leaf (depth=leaf_depth) with V=leaf_value.
    # At step k=1, update parent (depth=leaf_depth-1) with V = reward[leaf] + gamma * leaf_value.
    # etc.
    def backup_step(k, bcarry):
        btree, V, qmin, qmax = bcarry
        i = leaf_depth - k   # depth index: leaf_depth, leaf_depth-1, ..., 0
        valid = k <= leaf_depth
        node_i = jnp.where(valid, path_nodes[i], 0)  # safe index (0 when invalid)

        # Use pre-increment visit count as the slot index for this simulation
        write_slot = btree.visit_counts[node_i]  # scalar int32

        new_vc = btree.visit_counts.at[node_i].add(jnp.where(valid, 1, 0))
        new_vs = btree.value_sum.at[node_i].add(jnp.where(valid, V, 0.0))

        # k = node-to-leaf depth (0 at leaf, increases toward root) — matches MAZero convention
        new_nsv = btree.node_sim_values.at[node_i, write_slot].set(jnp.where(valid, V, 0.0))
        new_nsd = btree.node_sim_depths.at[node_i, write_slot].set(jnp.where(valid, k, 0))

        btree = btree.replace(
            visit_counts=new_vc, value_sum=new_vs,
            node_sim_values=new_nsv, node_sim_depths=new_nsd,
        )

        # Running min/max Q-baseline-diff, for non-root nodes only (matches
        # MAZero: `if (i != 0) minmax_stat.insert(...)`). Uses this
        # simulation's own backed-up value V as the node's Q signal rather
        # than recomputing the full OS(λ) estimate here, which would need
        # an extra O(num_simulations) argsort per backup step — the running
        # bound only needs to stay roughly representative, not exact.
        is_root = i == 0
        parent_node_i = jnp.where(valid & ~is_root, path_nodes[jnp.maximum(i - 1, 0)], 0)
        qsa_diff = path_rewards[jnp.maximum(i, 0)] + gamma * V - btree.pred_value[parent_node_i]
        update_stat = valid & ~is_root
        new_qmin = jnp.where(update_stat, jnp.minimum(qmin, qsa_diff), qmin)
        new_qmax = jnp.where(update_stat, jnp.maximum(qmax, qsa_diff), qmax)

        r_i = path_rewards[jnp.maximum(i, 0)]
        V_new = r_i + gamma * V
        V = jnp.where(valid & (k < leaf_depth), V_new, V)
        return btree, V, new_qmin, new_qmax

    tree, _, final_qmin, final_qmax = jax.lax.fori_loop(
        0, max_depth + 1, backup_step, (tree, leaf_value, carry.qmin, carry.qmax)
    )

    # ── 5. Record OS(λ) data ───────────────────────────────────────────────────
    # Root backup value: cumulative discounted rewards from root to leaf, plus gamma^depth * V_leaf
    def accum_step(k, acc):
        # k=0 → d=1 (first child reward), k=leaf_depth-1 → d=leaf_depth
        d = k + 1
        valid = d <= leaf_depth
        return acc + jnp.where(valid, (gamma ** k) * path_rewards[d], 0.0)

    root_cum_reward = jax.lax.fori_loop(0, max_depth, accum_step, 0.0)
    root_backup_value = root_cum_reward + (gamma ** leaf_depth) * leaf_value

    sim_depths = carry.sim_depths.at[sim_idx].set(leaf_depth)
    sim_values = carry.sim_values.at[sim_idx].set(root_backup_value)

    return SimCarry(
        tree=tree,
        next_free=carry.next_free + 1,
        rng=rng,
        sim_depths=sim_depths,
        sim_values=sim_values,
        qmin=final_qmin,
        qmax=final_qmax,
    )


# ─── Joint logits helpers ─────────────────────────────────────────────────────

def _logits_to_joint_logits(
    per_agent_logits: chex.Array,  # [N, A] — per-agent action logits
    N: int,
) -> chex.Array:
    """Per-agent logits (N, A) → joint logits (A_N,) under independence assumption."""
    A = per_agent_logits.shape[-1]
    log_probs = jax.nn.log_softmax(per_agent_logits, axis=-1)  # (N, A)
    joint = log_probs[0]                                         # (A,)
    for i in range(1, N):
        joint = (joint[:, None] + log_probs[i][None, :]).reshape(-1)
    return joint  # (A_N,)


def _joint_policy_to_marginal(
    joint_policy: chex.Array,     # (B, A_N)
    N: int,
    joint_action_shape: tuple,    # (A, A, ...) x N  — static tuple
) -> chex.Array:
    """Joint policy (B, A_N) → per-agent marginals (B, N, A)."""
    B = joint_policy.shape[0]
    reshaped = joint_policy.reshape(B, *joint_action_shape)
    marginals = []
    for i in range(N):
        other_axes = tuple(j + 1 for j in range(N) if j != i)
        marginals.append(jnp.sum(reshaped, axis=other_axes))  # (B, A)
    return jnp.stack(marginals, axis=1)  # (B, N, A)


# ─── Full planning loop (single environment, no batch) ───────────────────────

def _osla_plan_single(
    params,
    rng_key: chex.Array,
    observation: chex.Array,          # [N, obs_size]
    model,
    num_simulations: int,             # static
    K: int,                           # static (num_gumbel_samples)
    A_N: int,                         # static (action_space_size^num_agents)
    max_depth: int,                   # static
    gamma: float,
    rho: float,
    lam: float,
    pb_c_base: float,
    pb_c_init: float,
    value_delta_lb: float,
    dirichlet_alpha: float,
    dirichlet_fraction: float,
    joint_action_shape: tuple,        # static
    value_support,
    reward_support,
) -> MCTSPlanOutput:
    """Core planning for one environment. vmapped over batch in MCTSJointOSLAPlanner."""
    init_key, dir_key, rng = jax.random.split(rng_key, 3)
    max_nodes = num_simulations + 1
    N = observation.shape[0]

    # ── Root inference ────────────────────────────────────────────────────────
    obs_batched = observation[None]  # [1, N, obs_size]
    init_out = model.apply(
        {"params": params}, obs_batched, rngs={"dropout": init_key}
    )
    root_embedding = init_out.hidden_state[0]     # [N, D]
    root_pred_value = utils.support_to_scalar(init_out.value_logits, value_support)[0]  # scalar
    root_joint_logits = _logits_to_joint_logits(init_out.policy_logits[0], N)

    # Dirichlet noise on root prior
    dir_noise = jax.random.dirichlet(dir_key, alpha=jnp.full(A_N, dirichlet_alpha))
    root_probs = jax.nn.softmax(root_joint_logits)
    noisy_probs = (1 - dirichlet_fraction) * root_probs + dirichlet_fraction * dir_noise
    noisy_root_logits = jnp.log(noisy_probs + 1e-30)

    # Sample K children for root
    sample_key, rng = jax.random.split(rng)
    root_child_actions, root_child_probs = _sample_k_actions(sample_key, noisy_root_logits, K, A_N)

    # ── Initialize tree ───────────────────────────────────────────────────────
    D = root_embedding.shape[-1]
    tree = OSLATree(
        visit_counts=jnp.zeros(max_nodes, jnp.int32).at[0].set(1),  # root starts with 1 visit
        value_sum=jnp.zeros(max_nodes, jnp.float32),
        reward=jnp.zeros(max_nodes, jnp.float32),
        pred_value=jnp.zeros(max_nodes, jnp.float32).at[0].set(root_pred_value),
        embedding=jnp.zeros((max_nodes, N, D), jnp.float32).at[0].set(root_embedding),
        depth=jnp.zeros(max_nodes, jnp.int32),
        parent=jnp.full(max_nodes, -1, jnp.int32),
        child_actions=jnp.zeros((max_nodes, K), jnp.int32).at[0].set(root_child_actions),
        child_node_idx=jnp.full((max_nodes, K), -1, jnp.int32),
        child_prior_prob=jnp.zeros((max_nodes, K), jnp.float32).at[0].set(root_child_probs),
        node_sim_values=jnp.zeros((max_nodes, num_simulations + 1), jnp.float32),
        node_sim_depths=jnp.zeros((max_nodes, num_simulations + 1), jnp.int32),
    )

    init_carry = SimCarry(
        tree=tree,
        next_free=jnp.array(1, jnp.int32),
        rng=rng,
        sim_depths=jnp.zeros(num_simulations, jnp.int32),
        sim_values=jnp.zeros(num_simulations, jnp.float32),
        qmin=jnp.array(1e9, jnp.float32),
        qmax=jnp.array(-1e9, jnp.float32),
    )

    # ── Recurrent fn (batched interface required by _run_single_sim) ──────────
    def recurrent_fn_batched(params, rng_key, flat_action_batch, embedding_batch):
        # flat_action_batch: [1]; embedding_batch: [1, N, D]
        per_agent = jnp.stack(
            jnp.unravel_index(flat_action_batch, joint_action_shape), axis=-1
        )  # [1, N]
        out = model.apply(
            {"params": params},
            embedding_batch,
            per_agent,
            method=model.recurrent_inference,
            rngs={"dropout": rng_key},
        )
        value = utils.support_to_scalar(out.value_logits, value_support)
        reward = utils.support_to_scalar(out.reward_logits, reward_support)
        # Convert per-agent logits [1, N, A] → joint logits [1, A_N]
        joint_logits = jax.vmap(
            lambda lg: _logits_to_joint_logits(lg, N)
        )(out.policy_logits)
        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=jnp.full_like(reward, gamma),
                prior_logits=joint_logits,
                value=value,
            ),
            out.hidden_state,
        )

    # ── Run simulations ───────────────────────────────────────────────────────
    def sim_step(sim_idx, carry: SimCarry) -> SimCarry:
        return _run_single_sim(
            carry, sim_idx, params, recurrent_fn_batched,
            K, A_N, max_depth, gamma, rho, lam,
            pb_c_base, pb_c_init, value_delta_lb,
        )

    final_carry = jax.lax.fori_loop(0, num_simulations, sim_step, init_carry)

    # ── OS(λ) root value ──────────────────────────────────────────────────────
    osla_root_value = compute_osla_value(
        final_carry.sim_depths, final_carry.sim_values, rho=rho, lam=lam, max_depth=max_depth + 2
    )

    # ── Policy target from root child visit counts ────────────────────────────
    root_children = final_carry.tree.child_node_idx[0]  # [K]
    safe_root_children = jnp.maximum(root_children, 0)
    root_child_visits = jnp.where(
        root_children >= 0,
        final_carry.tree.visit_counts[safe_root_children].astype(jnp.float32),
        0.0,
    )  # [K]

    # ── Per-child Q-values for action-level AWPO ──────────────────────────────
    # Q_k = reward_into_child + gamma * OS(λ)_value(child)
    # Used in the learner to compute per-action advantages instead of state-level.
    child_osla_v = jax.vmap(
        lambda v, d, n: compute_osla_value_jax(v, d, n, rho, lam, max_depth + 2)
    )(
        final_carry.tree.node_sim_values[safe_root_children],  # [K, num_sims+1]
        final_carry.tree.node_sim_depths[safe_root_children],  # [K, num_sims+1]
        jnp.where(
            root_children >= 0,
            final_carry.tree.visit_counts[safe_root_children],
            jnp.zeros(K, jnp.int32),
        ),
    )  # [K]
    child_q = jnp.where(
        root_children >= 0,
        final_carry.tree.reward[safe_root_children] + gamma * child_osla_v,
        0.0,
    )  # [K] — Q-value for each sampled root child (0 if unvisited)

    # Decode flat joint actions → per-agent indices: (K,) → (K, N)
    root_child_actions_per_agent = jnp.stack(
        jnp.unravel_index(root_child_actions, joint_action_shape), axis=-1
    )  # (K, N)

    # Build sparse joint distribution over A_N actions
    joint_visits = jnp.zeros(A_N).at[root_child_actions].add(root_child_visits)
    joint_policy = joint_visits / (joint_visits.sum() + 1e-8)  # [A_N]

    # Marginalize to per-agent targets (shape (1, A_N) for _joint_policy_to_marginal)
    marginal_policy = _joint_policy_to_marginal(
        joint_policy[None], N, joint_action_shape
    )  # (1, N, A)

    # Best action = most-visited root child
    best_k = jnp.argmax(root_child_visits)
    best_flat_action = root_child_actions[best_k]
    best_action = jnp.stack(
        jnp.unravel_index(best_flat_action, joint_action_shape), axis=-1
    )  # (N,)

    return MCTSPlanOutput(
        joint_action=best_action[None],                              # (1, N)
        policy_targets=marginal_policy,                              # (1, N, A)
        root_value=osla_root_value[None],                           # (1,)
        agent_order=jnp.arange(N),
        root_child_actions=root_child_actions_per_agent[None],      # (1, K, N)
        root_child_q=child_q[None],                                  # (1, K)
        root_child_visits=root_child_visits[None],                   # (1, K)
    )


# ─── Planner class ────────────────────────────────────────────────────────────

class MCTSJointOSLAPlanner(MCTSPlanner):
    """
    Joint MCTS with OS(λ) backup.

    Uses a custom JAX MCTS loop (not mctx) with PUCT selection,
    K sampled joint actions per node, and OS(λ) value aggregation.
    Replaces MCTSJointPlanner when planner_mode="joint".
    """

    def __init__(self, model: FlaxMAMuZeroNet, config: ExperimentConfig):
        super().__init__(model, config)
        self.joint_action_shape: tuple = (self.action_space_size,) * self.num_agents
        self.A_N = self.action_space_size ** self.num_agents
        self.mcts_rho = config.mcts.mcts_rho
        self.mcts_lambda = config.mcts.mcts_lambda
        self.pb_c_base = config.mcts.pb_c_base
        self.pb_c_init = config.mcts.pb_c_init
        self.value_delta_lb = config.mcts.value_delta_lb
        # Override: don't JIT recurrent_fn standalone (it's called inside _osla_plan_single)
        self._recurrent_fn_jit = None

    def _recurrent_fn(self, params, rng_key, action, embedding):
        raise NotImplementedError("MCTSJointOSLAPlanner uses recurrent_fn internally in _osla_plan_single.")

    def _plan_loop(
        self, params, rng_key: chex.Array, observation: chex.Array
    ) -> MCTSPlanOutput:
        """Vmapped single-env planning across batch."""
        B = observation.shape[0]
        rng_keys = jax.random.split(rng_key, B)

        plan_single = functools.partial(
            _osla_plan_single,
            model=self.model,
            num_simulations=self.num_simulations,
            K=self.num_gumbel_samples,
            A_N=self.A_N,
            max_depth=self.max_depth_gumbel_search,
            gamma=self.discount_gamma,
            rho=self.mcts_rho,
            lam=self.mcts_lambda,
            pb_c_base=self.pb_c_base,
            pb_c_init=self.pb_c_init,
            value_delta_lb=self.value_delta_lb,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_fraction=self.dirichlet_fraction,
            joint_action_shape=self.joint_action_shape,
            value_support=self.value_support,
            reward_support=self.reward_support,
        )

        # vmap over (rng_keys, observation) — params are shared (in_axes=None)
        results = jax.vmap(plan_single, in_axes=(None, 0, 0))(
            params, rng_keys, observation
        )

        # vmap produces leading B dim; _osla_plan_single adds an extra 1 dim — squeeze it
        return MCTSPlanOutput(
            joint_action=results.joint_action.squeeze(1),              # (B, N)
            policy_targets=results.policy_targets.squeeze(1),          # (B, N, A)
            root_value=results.root_value.squeeze(1),                  # (B,)
            agent_order=results.agent_order[0],                         # (N,) — same for all
            root_child_actions=results.root_child_actions.squeeze(1),  # (B, K, N)
            root_child_q=results.root_child_q.squeeze(1),              # (B, K)
            root_child_visits=results.root_child_visits.squeeze(1),    # (B, K)
        )
