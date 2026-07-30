"""
Unit tests for mcts/mcts_joint_osla.py (the OS(λ) joint MCTS planner) and
its helper functions.

Run with:
    conda run -n mazero pytest tests/test_mcts.py -v

Module-scoped fixtures compile JAX/JIT once for the whole file to keep
the total run time reasonable.
"""

import dataclasses
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from config import ExperimentConfig, ModelConfig, MCTSConfig, TrainConfig
from model import FlaxMAMuZeroNet
from mcts import MCTSPlanOutput

# ─── Test constants ────────────────────────────────────────────────────────────

N   = 3     # num agents
OBS = 18    # observation dim
A   = 5     # action space size
D   = 32    # hidden state size
S   = 10    # support size


# ─── Module-scoped fixtures (built once, shared across all tests) ──────────────

@pytest.fixture(scope="module")
def test_config():
    """Minimal config designed for fast test execution."""
    return ExperimentConfig(
        train=TrainConfig(
            env_name="MPE_simple_spread_v3",
            num_agents=N,
            num_episodes=100,
            warmup_episodes=10,
            log_interval=10,
            num_actors=1,
            max_episode_steps=25,
            replay_buffer_size=1000,
            replay_buffer_alpha=0.6,
            replay_buffer_beta_start=0.4,
            replay_buffer_beta_frames=1000,
            batch_size=16,
            learning_rate=1e-3,
            param_update_interval=1,
            end_lr_factor=0.1,
            lr_warmup_steps=100,
            value_scale=0.25,
            consistency_scale=2.0,
            consistency_horizon=1,
            gradient_clip_norm=5.0,
            unroll_steps=5,
            n_step=10,
            discount_gamma=0.99,
            wandb_mode="disabled",
            project_name="test",
            checkpoint_dir="checkpoints",
            checkpoint_interval=100,
            num_envs_per_actor=1,
            sync=True,
            ema_decay=0.999,
            num_reanalyze_actors=0,
            reanalyze_batch_size=32,
            debug=False,
            debug_interval=100,
        ),
        model=ModelConfig(
            hidden_state_size=D,
            value_support_size=S,
            reward_support_size=S,
            fc_representation_layers=(D,),
            fc_dynamic_layers=(D,),
            fc_reward_layers=(16,),
            fc_value_layers=(16,),
            fc_policy_layers=(16,),
            attention_type="none",   # skip transformer for speed
            attention_layers=1,
            attention_heads=1,
            dropout_rate=0.0,        # deterministic for reproducibility
            proj_hid=16,
            proj_out=16,
            pred_hid=16,
            pred_out=16,
            use_obs_normalization=False,
        ),
        mcts=MCTSConfig(
            planner_mode="joint",
            num_simulations=8,           # minimum viable for gumbel (>= num_gumbel_samples)
            max_depth_gumbel_search=3,
            num_gumbel_samples=4,
            dirichlet_alpha=0.3,
            dirichlet_fraction=0.25,
            mcts_rho=0.75,
            mcts_lambda=0.8,
        ),
    )


@pytest.fixture(scope="module")
def model_and_params(test_config):
    """Initialize the world model once for the whole module."""
    net = FlaxMAMuZeroNet(test_config.model, A)
    rng = jax.random.PRNGKey(0)
    dummy_obs = jnp.ones((1, N, OBS))
    params = net.init(rng, dummy_obs)["params"]
    return net, params


@pytest.fixture(scope="module")
def osla_plan_fn(model_and_params, test_config):
    """JIT-compiled plan function for MCTSJointOSLAPlanner."""
    from mcts.mcts_joint_osla import MCTSJointOSLAPlanner
    net, _ = model_and_params
    planner = MCTSJointOSLAPlanner(model=net, config=test_config)
    return jax.jit(planner.plan), planner


@pytest.fixture
def params(model_and_params):
    _, p = model_and_params
    return p


@pytest.fixture
def obs():
    """Single observation as used during actor rollouts (B=1)."""
    return jnp.ones((1, N, OBS))


# ─── MCTSConfig field tests ───────────────────────────────────────────────────

def test_mcts_config_has_osla_fields():
    from config import MCTSConfig
    cfg = MCTSConfig(
        planner_mode="joint",
        num_simulations=8,
        max_depth_gumbel_search=3,
        num_gumbel_samples=4,
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
        mcts_rho=0.75,
        mcts_lambda=0.8,
    )
    assert cfg.mcts_rho == 0.75
    assert cfg.mcts_lambda == 0.8


# ─── OS(λ) value aggregation tests ───────────────────────────────────────────

class TestComputeOslaValue:
    """Tests for the OS(λ) value aggregation function."""

    def test_all_same_depth_rho_zero_equals_mean(self):
        """When all sims reach same depth, OS(λ) with rho=0.0 (discard nothing) = plain mean.

        NOTE: prior to Task 4, this test used rho=1.0 and relied on a
        `rho >= 1.0 -> keep all` special case that was never part of MAZero's
        `SubTreeValueSet` (verified against `../MAZero/core/mcts/ctree/common_lib/utils.cpp`,
        which always computes `size_lim = max(1, ceil(count[depth] * (1 - rho)))`
        with no special-casing of rho). That special case embodied exactly the
        "rho as keep-fraction" confusion the Task 4 brief calls out (rho is the
        *discard* fraction: rho=0.0 discards nothing => keeps everything; the
        old test's premise "rho=1.0 keeps all" had this backwards). Updated to
        rho=0.0, which is the fraction that actually keeps every sim under the
        MAZero-faithful formula.
        """
        from mcts.osla_math import compute_osla_value
        depths = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
        values = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
        # rho=0.0: keep all 4, mean = 0.25
        v = compute_osla_value(depths, values, rho=0.0, lam=0.8)
        assert jnp.allclose(v, 0.25, atol=1e-4)

    def test_top_rho_amplifies_rare_wins(self):
        """rho=0.75 keeps top 25% (1 out of 4), result = the win value (1.0)."""
        from mcts.osla_math import compute_osla_value
        depths = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
        values = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
        # rho=0.75: keep top 25% = top 1 = value 1.0; weight = lambda^1 = 0.8
        v = compute_osla_value(depths, values, rho=0.75, lam=0.8)
        assert jnp.allclose(v, 1.0, atol=1e-4)

    def test_depth_weighting_discounts_deeper_sims(self):
        """Deeper sims get less weight (lam^depth); verifies weighted mean is computed correctly."""
        from mcts.osla_math import compute_osla_value
        depths = jnp.array([1, 5], dtype=jnp.int32)
        values = jnp.array([1.0, 0.5], dtype=jnp.float32)
        # rho=1.0: keep both. weights = [0.8^1, 0.8^5] = [0.8, 0.32768]
        # weighted mean = (0.8*1.0 + 0.32768*0.5) / (0.8 + 0.32768)
        w1, w2 = 0.8 ** 1, 0.8 ** 5
        expected = (w1 * 1.0 + w2 * 0.5) / (w1 + w2)
        v = compute_osla_value(depths, values, rho=1.0, lam=0.8)
        assert jnp.allclose(v, expected, atol=1e-4)

    def test_jit_compatible(self):
        """Must be JIT-compilable."""
        from mcts.osla_math import compute_osla_value
        fn = jax.jit(compute_osla_value, static_argnames=("rho", "lam"))
        depths = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        values = jnp.array([0.1, 0.5, 0.2, 0.9], dtype=jnp.float32)
        v = fn(depths, values, rho=0.75, lam=0.8)
        assert v.shape == ()
        assert jnp.isfinite(v)


# ─── OSLATree dataclass and UCB helper tests ──────────────────────────────────

class TestOSLAHelpers:

    def test_compute_ucb_prefers_unvisited(self):
        """Unvisited children (visit_count=0) should have higher UCB than visited ones."""
        from mcts.osla_math import compute_ucb_scores
        child_visits = jnp.array([5.0, 0.0, 3.0, 0.0])
        child_q_diff = jnp.array([0.5, 0.0, 0.3, 0.0])
        prior_probs = jnp.array([0.25, 0.25, 0.25, 0.25])
        parent_visits = jnp.array(8.0)
        ucb = compute_ucb_scores(
            child_q_diff, child_visits, prior_probs, parent_visits,
            qmin=jnp.array(0.0), qmax=jnp.array(1.0),
            pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01,
        )
        assert ucb[1] > ucb[0]
        assert ucb[3] > ucb[2]

    def test_compute_ucb_shape(self):
        from mcts.osla_math import compute_ucb_scores
        K = 10
        ucb = compute_ucb_scores(
            jnp.zeros(K), jnp.zeros(K), jnp.ones(K) / K, jnp.array(1.0),
            qmin=jnp.array(0.0), qmax=jnp.array(1.0),
            pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01,
        )
        assert ucb.shape == (K,)

    def test_dataclasses_are_pytrees(self):
        """OSLATree, SimCarry, SelectCarry must be registered as JAX pytrees."""
        from mcts.mcts_joint_osla import OSLATree, SimCarry, SelectCarry
        # chex.dataclass auto-registers as pytree; verify leaves/treedef work
        max_sims_slots = 9  # dummy size for test
        tree = OSLATree(
            visit_counts=jnp.zeros(5, jnp.int32),
            value_sum=jnp.zeros(5),
            reward=jnp.zeros(5),
            pred_value=jnp.zeros(5),
            embedding=jnp.zeros((5, 3, 32)),
            depth=jnp.zeros(5, jnp.int32),
            parent=jnp.full(5, -1, jnp.int32),
            child_actions=jnp.zeros((5, 4), jnp.int32),
            child_node_idx=jnp.full((5, 4), -1, jnp.int32),
            child_prior_prob=jnp.zeros((5, 4)),
            node_sim_values=jnp.zeros((5, max_sims_slots)),
            node_sim_depths=jnp.zeros((5, max_sims_slots), jnp.int32),
        )
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        assert len(leaves) == 12  # 9 original + 2 node_sim fields + 1 pred_value
        tree2 = treedef.unflatten(leaves)
        assert jnp.array_equal(tree2.visit_counts, tree.visit_counts)


class TestUCBLogVisitScaling:

    def test_pb_c_grows_with_parent_visits(self):
        """The prior-exploration term must grow with log(parent_visits), not
        stay fixed — matches MuZero/MAZero's pb_c formula, not a constant
        c_puct."""
        from mcts.osla_math import compute_ucb_scores

        child_visits = jnp.zeros(3)
        prior_probs = jnp.ones(3) / 3
        qmin = jnp.array(0.0)
        qmax = jnp.array(0.0)  # no stats yet (qmax <= qmin)

        ucb_low_n = compute_ucb_scores(
            jnp.zeros(3), child_visits, prior_probs, parent_visits=jnp.array(1.0),
            qmin=qmin, qmax=qmax, pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01,
        )
        ucb_high_n = compute_ucb_scores(
            jnp.zeros(3), child_visits, prior_probs, parent_visits=jnp.array(1000.0),
            qmin=qmin, qmax=qmax, pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01,
        )
        # All children still unvisited in both cases, so the difference is
        # purely the pb_c(N_parent) prior-exploration scaling.
        assert jnp.all(ucb_high_n > ucb_low_n), (
            "prior-exploration term should grow with parent visit count"
        )

    def test_value_term_normalized_into_unit_range(self):
        """With running qmin/qmax stats present, the Q-baseline-diff term
        must be clipped into [0, 1] regardless of its raw scale."""
        from mcts.osla_math import compute_ucb_scores

        # Raw Q-baseline diffs of huge magnitude (as if value support were
        # [-5, 5] and Q-values routinely differ by several units) must not
        # swamp the prior term once normalized.
        child_q_diff = jnp.array([-5.0, 0.0, 5.0])
        child_visits = jnp.array([1.0, 1.0, 1.0])
        prior_probs = jnp.ones(3) / 3
        ucb = compute_ucb_scores(
            child_q_diff, child_visits, prior_probs, parent_visits=jnp.array(3.0),
            qmin=jnp.array(-5.0), qmax=jnp.array(5.0),
            pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01,
        )
        # value_score for each child should be (diff - qmin) / (qmax - qmin),
        # i.e. 0.0, 0.5, 1.0 respectively, plus an identical prior term.
        value_scores = ucb - ucb[1] + 0.5  # back out value_score using the middle child as anchor
        assert jnp.allclose(value_scores, jnp.array([0.0, 0.5, 1.0]), atol=1e-3)

    def test_unvisited_children_get_zero_value_score(self):
        """This repo scores unvisited children with a hard value_score=0,
        skipping normalization entirely for them. MAZero's cnode.cpp also sets
        the pre-normalization value_score=0 for unvisited children, but then
        still runs it through minmax_stat.normalize() and clips to [0,1] — so
        MAZero's actual unvisited-child score is normalize(0), not a hard 0.
        This is a deliberate, standard-MuZero-pseudocode divergence from
        MAZero, not a bug: when qmin < 0 (true for this repo's [-5,5] value
        support), normalize(0) > 0, so MAZero scores unvisited children
        slightly higher than this repo does. Left as-is per code review
        (2026-07 MCTS correctness audit whole-branch review) — revisit only
        if search quality issues trace back to under-exploration of new
        children.
        """
        from mcts.osla_math import compute_ucb_scores

        child_q_diff = jnp.array([100.0, 100.0])  # would be huge if not masked
        child_visits = jnp.array([0.0, 0.0])
        prior_probs = jnp.array([0.5, 0.5])
        ucb = compute_ucb_scores(
            child_q_diff, child_visits, prior_probs, parent_visits=jnp.array(1.0),
            qmin=jnp.array(-1.0), qmax=jnp.array(1.0),
            pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01,
        )
        pb_c = jnp.log((1.0 + 19652.0 + 1.0) / 19652.0) + 1.25
        pb_c = pb_c * jnp.sqrt(1.0) / (1.0 + 0.0)
        expected_prior_score = pb_c * 0.5
        assert jnp.allclose(ucb, expected_prior_score, atol=1e-3), (
            "unvisited children should score purely on the prior term (value_score=0)"
        )

    def test_all_zero_ucb_when_parent_visits_zero_and_children_unvisited(self):
        """Documents the exact degenerate input _best_ucb hits at every
        node's first internal descent: parent_visits=0 (a just-expanded
        node, visit_count==1) makes pb_c*sqrt(parent_visits)=0 for every
        child, and all children being unvisited makes value_score=0 too —
        so compute_ucb_scores itself really does return an all-zero
        vector here. The caller (_best_ucb) is responsible for not letting
        bare jnp.argmax on this vector silently ignore the prior."""
        from mcts.osla_math import compute_ucb_scores

        child_visits = jnp.zeros(4)
        prior_probs = jnp.array([0.05, 0.05, 0.05, 0.85])
        ucb = compute_ucb_scores(
            jnp.zeros(4), child_visits, prior_probs, parent_visits=jnp.array(0.0),
            qmin=jnp.array(1e9), qmax=jnp.array(-1e9),
            pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01,
        )
        assert jnp.allclose(ucb, jnp.zeros(4)), (
            f"expected all-zero UCB vector for parent_visits=0, got {ucb}"
        )
        # Bare argmax on an all-zero vector always returns index 0,
        # regardless of prior — this is the bug. _best_ucb's tie-break
        # (jnp.argmax(ucb + 1e-6 * prior_probs)) must instead resolve to
        # the highest-prior child (index 3 here).
        bare_argmax_choice = int(jnp.argmax(ucb))
        assert bare_argmax_choice == 0, (
            "sanity check: bare argmax on all-zero UCB is index 0 regardless of prior "
            "(this is the behavior being fixed, not the desired one)"
        )
        tiebreak_choice = int(jnp.argmax(ucb + 1e-6 * prior_probs))
        assert tiebreak_choice == 3, (
            f"tie-break should resolve to the highest-prior child (index 3), got {tiebreak_choice}"
        )


class TestUCBZeroScoreTiebreakEndToEnd:
    """End-to-end regression test for the _best_ucb tie-break fix, built via
    a hand-constructed OSLATree/SimCarry and a real _run_single_sim call
    (following TestRootRoundRobin's pattern), rather than testing
    compute_ucb_scores in isolation. Confirms the fix actually changes
    which child gets expanded inside the real selection/expansion path,
    not just that the standalone UCB formula produces zeros."""

    def test_internal_node_expansion_respects_prior_not_index_zero(self):
        """Construct a 2-level tree: root (node 0) has one already-expanded
        child (node 1, visit_count=1) whose own K children are all
        unvisited. Root's prior overwhelmingly favors descending into node
        1 (parent_visits large there, so root's own UCB is non-degenerate
        and deterministically picks that child). But once selection
        reaches node 1, parent_visits = visit_count(node1) - 1 = 0, making
        compute_ucb_scores return an all-zero vector for node 1's
        children — exactly the degenerate case from
        test_all_zero_ucb_when_parent_visits_zero_and_children_unvisited.
        Node 1's own children have a skewed prior favoring index 3; the
        tie-break must expand index 3, not always index 0."""
        from mcts.mcts_joint_osla import _run_single_sim, OSLATree, SimCarry, RecurrentFnOutput

        K, A_N, N, D, max_depth, gamma = 4, 25, 2, 8, 3, 0.99
        max_nodes = K + 2

        def fake_recurrent_fn(params, rng, flat_action, embedding):
            B = flat_action.shape[0]
            return (
                RecurrentFnOutput(
                    reward=jnp.zeros((B,)),
                    discount=jnp.ones((B,)),
                    prior_logits=jnp.zeros((B, A_N)),
                    value=jnp.zeros((B,)),
                ),
                embedding,
            )

        visit_counts = jnp.zeros(max_nodes, jnp.int32).at[0].set(50).at[1].set(1)
        child_node_idx = jnp.full((max_nodes, K), -1, jnp.int32).at[0, 0].set(1)
        child_prior_prob = jnp.zeros((max_nodes, K))
        # Root: position 0 (-> node 1) has overwhelming prior so root's own
        # (non-degenerate, parent_visits=49) UCB deterministically descends
        # into node 1, regardless of the tie-break epsilon.
        child_prior_prob = child_prior_prob.at[0].set(jnp.array([0.999, 0.0003, 0.0003, 0.0004]))
        # Node 1: all children unvisited, skewed prior favoring index 3.
        child_prior_prob = child_prior_prob.at[1].set(jnp.array([0.05, 0.05, 0.05, 0.85]))
        child_actions = jnp.zeros((max_nodes, K), jnp.int32)
        child_actions = child_actions.at[0].set(jnp.arange(K))
        child_actions = child_actions.at[1].set(jnp.arange(K, 2 * K))

        tree = OSLATree(
            visit_counts=visit_counts,
            value_sum=jnp.zeros(max_nodes),
            reward=jnp.zeros(max_nodes),
            pred_value=jnp.zeros(max_nodes),
            embedding=jnp.zeros((max_nodes, N, D)),
            depth=jnp.zeros(max_nodes, jnp.int32).at[1].set(1),
            parent=jnp.full(max_nodes, -1, jnp.int32).at[1].set(0),
            child_actions=child_actions,
            child_node_idx=child_node_idx,
            child_prior_prob=child_prior_prob,
            node_sim_values=jnp.zeros((max_nodes, K + 1)),
            node_sim_depths=jnp.zeros((max_nodes, K + 1), jnp.int32),
        )

        carry = SimCarry(
            tree=tree, next_free=jnp.array(2, jnp.int32), rng=jax.random.PRNGKey(3),
            sim_depths=jnp.zeros(1, jnp.int32), sim_values=jnp.zeros(1, jnp.float32),
            qmin=jnp.array(1e9), qmax=jnp.array(-1e9),
        )

        result = _run_single_sim(
            carry, jnp.array(0), None, fake_recurrent_fn,
            K, A_N, max_depth, gamma,
            pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01, rho=0.25, lam=0.8,
        )

        # Whichever position of node 1's children got expanded should be
        # the highest-prior one (index 3), not index 0.
        expanded_positions = jnp.where(result.tree.child_node_idx[1] >= 0)[0]
        assert len(expanded_positions) == 1, (
            f"expected exactly one of node 1's children to be expanded, "
            f"got {result.tree.child_node_idx[1]}"
        )
        assert int(expanded_positions[0]) == 3, (
            f"expected the highest-prior child (index 3) to be expanded, "
            f"got index {int(expanded_positions[0])} "
            f"(child_node_idx[1] = {result.tree.child_node_idx[1]})"
        )


class TestOSLAPerDepthQuantile:

    def test_top_quantile_selected_within_depth_bucket_not_pooled(self):
        """Two sims at depth 0 with small values, one sim at depth 3 with a
        huge value. Pooled-across-depths ranking (the old behavior) would
        let the depth-3 outlier dominate the global top-quantile regardless
        of its own bucket size. Per-depth-bucket ranking (MAZero's
        SubTreeValueSet) keeps the depth-3 sim's bucket separate: with only
        one sim at depth 3, size_lim=1, so it's fully included in its own
        bucket either way — but a depth-0 value that's LARGER than any
        depth-0 peer must still be selected as "top" for depth 0, even
        if it's smaller than the depth-3 outlier. This test uses 4 sims
        at depth 0 (values 1, 2, 3, 4) with rho=0.75 (keep top 25% = top 1
        of 4) and 1 sim at depth 5 (value 100). Expected: only the top
        depth-0 value (4) and the depth-5 value (100) are selected.
        weighted_sum = lam**0 * 4 + lam**5 * 100
        tot_weight   = lam**0 * 1 + lam**5 * 1
        """
        from mcts.osla_math import compute_osla_value_jax

        lam = 0.8
        sim_values = jnp.array([1.0, 2.0, 3.0, 4.0, 100.0, 0.0, 0.0, 0.0])
        sim_depths = jnp.array([0,   0,   0,   0,   5,     0,   0,   0], dtype=jnp.int32)
        n_visits = jnp.array(5)  # first 5 entries are real, rest is padding

        result = compute_osla_value_jax(
            sim_values, sim_depths, n_visits, rho=0.75, lam=lam, max_depth=8
        )
        expected_weighted_sum = (lam ** 0) * 4.0 + (lam ** 5) * 100.0
        expected_tot_weight = (lam ** 0) * 1.0 + (lam ** 5) * 1.0
        expected = expected_weighted_sum / expected_tot_weight
        assert jnp.allclose(result, expected, atol=1e-4), (
            f"expected {expected:.4f}, got {float(result):.4f}"
        )

    def test_matches_old_pooled_behavior_when_all_sims_same_depth(self):
        """When every sim is at the same depth, per-depth-bucket ranking and
        pooled ranking are equivalent (there's only one bucket) — sanity
        check that the rewrite doesn't change single-depth behavior."""
        from mcts.osla_math import compute_osla_value_jax

        sim_values = jnp.array([1.0, 2.0, 3.0, 4.0, 0.0])
        sim_depths = jnp.zeros(5, dtype=jnp.int32)
        n_visits = jnp.array(4)

        result = compute_osla_value_jax(
            sim_values, sim_depths, n_visits, rho=0.75, lam=0.8, max_depth=4
        )
        # rho=0.75 -> keep top ceil(4*0.25)=1 -> only value 4.0
        assert jnp.allclose(result, 4.0, atol=1e-4)


# ─── _sample_k_actions tests ──────────────────────────────────────────────────

class TestSampleKActions:

    def test_output_shapes(self):
        from mcts.osla_math import _sample_k_actions
        rng = jax.random.PRNGKey(0)
        actions, probs = _sample_k_actions(rng, jnp.zeros(729), K=10, A_N=729)
        assert actions.shape == (10,)
        assert probs.shape == (10,)

    def test_actions_in_range(self):
        from mcts.osla_math import _sample_k_actions
        rng = jax.random.PRNGKey(1)
        actions, probs = _sample_k_actions(rng, jnp.zeros(729), K=10, A_N=729)
        assert jnp.all(actions >= 0) and jnp.all(actions < 729)

    def test_probs_are_subset_of_softmax(self):
        """Returned probs must equal softmax(logits)[actions]."""
        from mcts.osla_math import _sample_k_actions
        logits = jax.random.normal(jax.random.PRNGKey(2), (25,))
        actions, probs = _sample_k_actions(jax.random.PRNGKey(3), logits, K=5, A_N=25)
        expected_probs = jax.nn.softmax(logits)[actions]
        assert jnp.allclose(probs, expected_probs, atol=1e-6)

    def test_k_ge_an_uses_replacement(self):
        """K >= A_N must not crash (uses replacement)."""
        from mcts.osla_math import _sample_k_actions
        actions, probs = _sample_k_actions(jax.random.PRNGKey(0), jnp.zeros(3), K=10, A_N=3)
        assert actions.shape == (10,)


# ─── _run_single_sim backup correctness tests ────────────────────────────────

class TestRunSingleSimBackup:
    """Verify _run_single_sim produces correct backup values."""

    def _make_fake_recurrent_fn(self, fixed_reward: float, fixed_value: float, A_N: int, N: int, D: int):
        """Returns a recurrent_fn that always outputs fixed reward and value."""
        from mcts.mcts_joint_osla import RecurrentFnOutput
        def fake_recurrent_fn(params, rng, flat_action, embedding):
            B = flat_action.shape[0]
            return (
                RecurrentFnOutput(
                    reward=jnp.full((B,), fixed_reward),
                    discount=jnp.ones((B,)),
                    prior_logits=jnp.zeros((B, A_N)),
                    value=jnp.full((B,), fixed_value),
                ),
                embedding,  # pass through embedding unchanged
            )
        return fake_recurrent_fn

    def test_root_backup_value_single_step(self):
        """After 1 sim reaching depth 1: root_backup_value = r + gamma * V."""
        from mcts.mcts_joint_osla import _run_single_sim, OSLATree, SimCarry
        from mcts.osla_math import _sample_k_actions

        K, A_N, N, D, max_depth, gamma = 4, 25, 2, 8, 3, 0.99
        r, v = 0.5, 1.0  # fixed reward and value from the fake model

        # Build a minimal root node with K sampled children
        rng = jax.random.PRNGKey(0)
        root_logits = jnp.zeros(A_N)
        child_actions, child_probs = _sample_k_actions(rng, root_logits, K, A_N)

        max_nodes = 5
        tree = OSLATree(
            visit_counts=jnp.array([1] + [0] * (max_nodes - 1), jnp.int32),
            value_sum=jnp.zeros(max_nodes),
            reward=jnp.zeros(max_nodes),
            pred_value=jnp.zeros(max_nodes),
            embedding=jnp.zeros((max_nodes, N, D)),
            depth=jnp.zeros(max_nodes, jnp.int32),
            parent=jnp.full(max_nodes, -1, jnp.int32),
            child_actions=jnp.zeros((max_nodes, K), jnp.int32).at[0].set(child_actions),
            child_node_idx=jnp.full((max_nodes, K), -1, jnp.int32),
            child_prior_prob=jnp.zeros((max_nodes, K)).at[0].set(child_probs),
            node_sim_values=jnp.zeros((max_nodes, 9)),   # small for test
            node_sim_depths=jnp.zeros((max_nodes, 9), jnp.int32),
        )
        carry = SimCarry(
            tree=tree,
            next_free=jnp.array(1, jnp.int32),
            rng=jax.random.PRNGKey(1),
            sim_depths=jnp.zeros(1, jnp.int32),
            sim_values=jnp.zeros(1, jnp.float32),
            qmin=jnp.array(1e9, jnp.float32),
            qmax=jnp.array(-1e9, jnp.float32),
        )

        fake_rf = self._make_fake_recurrent_fn(r, v, A_N, N, D)
        result = _run_single_sim(carry, jnp.array(0), None, fake_rf, K, A_N, max_depth, gamma, rho=0.25, lam=0.8)

        # Root backup value = r + gamma * v
        expected_root_val = r + gamma * v
        assert jnp.allclose(result.sim_values[0], expected_root_val, atol=1e-4), \
            f"Expected {expected_root_val:.4f}, got {result.sim_values[0]:.4f}"

        # Root visit count should be 2 (was 1, got 1 from backup)
        assert result.tree.visit_counts[0] == 2

        # New node (index 1) should have visit count 1
        assert result.tree.visit_counts[1] == 1


# ─── MCTSJointOSLAPlanner end-to-end tests ───────────────────────────────────

class TestMCTSJointOSLAPlanner:

    def test_returns_plan_output(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert isinstance(out, MCTSPlanOutput)

    def test_joint_action_shape(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert out.joint_action.shape == (1, N)

    def test_policy_targets_shape(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert out.policy_targets.shape == (1, N, A)

    def test_actions_in_valid_range(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert jnp.all(out.joint_action >= 0)
        assert jnp.all(out.joint_action < A)

    def test_policy_targets_sum_to_one(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        sums = jnp.sum(out.policy_targets, axis=-1)
        assert jnp.allclose(sums, jnp.ones((1, N)), atol=1e-4)

    def test_root_value_shape(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert out.root_value.shape == (1,)

    def test_root_value_finite(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert jnp.isfinite(out.root_value).all()

    def test_deterministic_with_same_key(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        out1 = plan_fn(params, jax.random.PRNGKey(7), obs)
        out2 = plan_fn(params, jax.random.PRNGKey(7), obs)
        assert jnp.array_equal(out1.joint_action, out2.joint_action)

    def test_stochastic_with_different_keys(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        results = [plan_fn(params, jax.random.PRNGKey(i), obs).joint_action for i in range(10)]
        all_same = all(jnp.array_equal(results[0], r) for r in results[1:])
        assert not all_same

def test_reanalyze_actor_uses_osla_planner():
    """ReanalyzeActor's planner_map must map 'joint' to MCTSJointOSLAPlanner."""
    import pathlib
    src = (pathlib.Path(__file__).parent.parent / "actors" / "reanalyze_actor.py").read_text()
    assert '"joint": MCTSJointOSLAPlanner' in src, (
        "ReanalyzeActor planner_map must map 'joint' to MCTSJointOSLAPlanner"
    )


class TestComputeOslaValueJax:

    def test_matches_python_version(self):
        """JAX-native compute_osla_value_jax should match Python compute_osla_value."""
        from mcts.osla_math import compute_osla_value, compute_osla_value_jax
        rng = np.random.default_rng(0)
        K = 10
        depths = np.array([1, 2, 3, 1, 2, 4, 1, 2, 3, 4], dtype=np.int32)
        values = rng.uniform(0, 2, K).astype(np.float32)
        py_result = float(compute_osla_value(
            jnp.array(depths), jnp.array(values), rho=0.25, lam=0.8, max_depth=8
        ))
        jax_result = float(compute_osla_value_jax(
            jnp.array(values), jnp.array(depths), jnp.array(K, jnp.int32), 0.25, 0.8, max_depth=8
        ))
        assert abs(py_result - jax_result) < 1e-4, f"py={py_result} jax={jax_result}"

    def test_partial_fill(self):
        """compute_osla_value_jax with n_visits < max_sims ignores padding zeros.

        NOTE: prior to Task 4, this test's expectation was computed by pooling
        all 3 valid sims together and taking a single global top-2 (rho=0.25 ->
        floor(0.75*3)=2), which happened to select both depth-1 sims and drop
        the depth-2 sim. Under the corrected per-depth-bucket ranking (matching
        MAZero's SubTreeValueSet), each depth bucket is filtered independently:
        depth=1 has 2 sims (size_lim=max(1,ceil(2*0.75))=2 -> both kept),
        depth=2 has 1 sim (size_lim=max(1,ceil(1*0.75))=1 -> kept). So all 3
        valid sims are now included, each in its own depth bucket.
        """
        from mcts.osla_math import compute_osla_value_jax
        values = jnp.array([1.0, 0.5, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        depths = jnp.array([1, 2, 1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
        n_visits = jnp.array(3, jnp.int32)
        result = float(compute_osla_value_jax(values, depths, n_visits, 0.25, 0.8, max_depth=4))
        # depth=1 bucket: values [1.0, 0.8], both kept, weight = 0.8**1 each
        # depth=2 bucket: value [0.5], kept, weight = 0.8**2
        w1, w2 = 0.8 ** 1, 0.8 ** 2
        weighted_sum = w1 * (1.0 + 0.8) + w2 * 0.5
        tot_weight = w1 * 2 + w2 * 1
        expected = weighted_sum / tot_weight
        assert abs(result - expected) < 1e-4, f"expected {expected}, got {result}"


class TestPerNodeOsla:

    def test_oslatree_has_node_sim_fields(self):
        """OSLATree must have node_sim_values and node_sim_depths fields."""
        from mcts.mcts_joint_osla import OSLATree
        field_names = {f.name for f in dataclasses.fields(OSLATree)}
        assert "node_sim_values" in field_names, "OSLATree missing node_sim_values"
        assert "node_sim_depths" in field_names, "OSLATree missing node_sim_depths"

    def test_plan_output_shape_unchanged(self, osla_plan_fn, params, obs):
        """Plan output shapes must be unchanged after adding per-node tracking."""
        plan_fn, _ = osla_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert out.joint_action.shape == (1, N)
        assert out.policy_targets.shape == (1, N, A)
        assert out.root_value.shape == (1,)
        assert jnp.isfinite(out.root_value).all()


class TestRootRoundRobin:

    def test_first_k_simulations_visit_each_root_child_once(self):
        """MAZero: `if (node->is_root && node->visit_count <= node->num_children)
        child_index = node->visit_count - 1;` — the first K simulations must
        round-robin through the K sampled root actions before UCB selection
        kicks in, regardless of prior/value differences between them."""
        from mcts.mcts_joint_osla import _run_single_sim, OSLATree, SimCarry, RecurrentFnOutput
        from mcts.osla_math import _sample_k_actions

        K, A_N, N, D, max_depth, gamma = 4, 25, 2, 8, 3, 0.99

        # Deliberately skewed priors so UCB alone (without round-robin)
        # would pick child 0 every time.
        rng = jax.random.PRNGKey(0)
        skewed_logits = jnp.array([10.0, 0.0, 0.0, 0.0] + [0.0] * (A_N - 4))
        child_actions, child_probs = _sample_k_actions(rng, skewed_logits, K, A_N)

        max_nodes = K + 2
        tree = OSLATree(
            visit_counts=jnp.array([1] + [0] * (max_nodes - 1), jnp.int32),
            value_sum=jnp.zeros(max_nodes),
            reward=jnp.zeros(max_nodes),
            pred_value=jnp.zeros(max_nodes),
            embedding=jnp.zeros((max_nodes, N, D)),
            depth=jnp.zeros(max_nodes, jnp.int32),
            parent=jnp.full(max_nodes, -1, jnp.int32),
            child_actions=jnp.zeros((max_nodes, K), jnp.int32).at[0].set(child_actions),
            child_node_idx=jnp.full((max_nodes, K), -1, jnp.int32),
            child_prior_prob=jnp.zeros((max_nodes, K)).at[0].set(child_probs),
            node_sim_values=jnp.zeros((max_nodes, K + 1)),
            node_sim_depths=jnp.zeros((max_nodes, K + 1), jnp.int32),
        )

        def fake_recurrent_fn(params, rng, flat_action, embedding):
            B = flat_action.shape[0]
            return (
                RecurrentFnOutput(
                    reward=jnp.zeros((B,)),
                    discount=jnp.ones((B,)),
                    prior_logits=jnp.zeros((B, A_N)),
                    value=jnp.zeros((B,)),
                ),
                embedding,
            )

        carry = SimCarry(
            tree=tree, next_free=jnp.array(1, jnp.int32), rng=jax.random.PRNGKey(1),
            sim_depths=jnp.zeros(K, jnp.int32), sim_values=jnp.zeros(K, jnp.float32),
            qmin=jnp.array(1e9), qmax=jnp.array(-1e9),
        )

        visited_children = []
        for sim_idx in range(K):
            carry = _run_single_sim(
                carry, jnp.array(sim_idx), None, fake_recurrent_fn,
                K, A_N, max_depth, gamma,
                pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01, rho=0.25, lam=0.8,
            )
            # child_node_idx[0] tracks which of the K children have been
            # expanded so far; the newly-expanded one this round is the one
            # that just went from -1 to >= 0.
            expanded = jnp.where(carry.tree.child_node_idx[0] >= 0)[0]
            visited_children.append(int(expanded[-1]) if len(expanded) > len(visited_children) else None)

        distinct_children_visited = {c for c in visited_children if c is not None}
        assert len(distinct_children_visited) == K, (
            f"expected all {K} root children visited within the first {K} sims, "
            f"got {visited_children}"
        )
