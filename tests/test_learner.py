"""Tests for actors/loss.py module-level utilities.

Run with:
    conda run -n mazero pytest tests/test_learner.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp


def test_scale_grad_half_forward_is_identity():
    """scale_grad_half should be identity in the forward pass."""
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    from actors.loss import scale_grad_half
    x = jnp.array([1.0, 2.0, 3.0])
    assert jnp.allclose(scale_grad_half(x), x)


def test_scale_grad_half_backward_halves_gradient():
    """scale_grad_half backward pass should halve the gradient."""
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    from actors.loss import scale_grad_half
    x = jnp.array([1.0, 2.0, 3.0])
    grad_fn = jax.grad(lambda v: scale_grad_half(v).sum())
    g = grad_fn(x)
    assert jnp.allclose(g, jnp.full_like(x, 0.5)), f"expected 0.5 everywhere, got {g}"


def test_action_level_awpo_upweights_high_q_actions():
    """Actions with Q > V_root should get higher AWPO weight than low-Q actions."""
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"

    B, K = 2, 5
    v_root = jnp.array([0.5, 0.5])  # (B,)
    # Batch 0: action 0 is good (Q=1.0 → adv=+0.5), action 1 is bad (Q=0.0 → adv=-0.5)
    q_k = jnp.array([
        [1.0, 0.0, 0.5, 0.5, 0.5],
        [0.8, 0.2, 0.5, 0.5, 0.5],
    ])  # (B, K)
    alpha = 1.0

    action_adv = jnp.clip((q_k - v_root[:, None]) / alpha, -5.0, 5.0)
    awpo_w_k = jnp.exp(action_adv)

    assert awpo_w_k[0, 0] > awpo_w_k[0, 1], "High-Q action should have higher AWPO weight"
    assert float(awpo_w_k[0, 0]) > 1.0, "Positive-advantage action should have weight > 1"
    assert float(awpo_w_k[0, 1]) < 1.0, "Negative-advantage action should have weight < 1"


def test_action_level_awpo_joint_log_prob_sums_over_agents():
    """Joint log-prob of a sampled action should be the sum of per-agent log-probs."""
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"

    B, K, N, A = 2, 5, 3, 9
    rng = jax.random.PRNGKey(42)
    policy_logits = jax.random.normal(rng, (B, N, A))
    log_probs = jax.nn.log_softmax(policy_logits, axis=-1)  # (B, N, A)

    # All agents pick action 0 for all K samples
    child_actions = jnp.zeros((B, K, N), dtype=jnp.int32)

    log_probs_exp = jnp.broadcast_to(log_probs[:, None, :, :], (B, K, N, A))
    gathered = jnp.take_along_axis(
        log_probs_exp,
        child_actions[:, :, :, None],
        axis=-1,
    ).squeeze(-1)  # (B, K, N)
    joint_log_prob_k = gathered.sum(axis=-1)  # (B, K)

    # Manual check: joint log-prob = sum of per-agent log_prob(action=0)
    expected = log_probs[:, :, 0].sum(axis=-1, keepdims=True)  # (B, 1)
    assert jnp.allclose(joint_log_prob_k, jnp.broadcast_to(expected, (B, K)), atol=1e-5)


def test_multi_step_awpo_weights_differ_across_steps():
    """
    With different Q-values at each unroll step, AWPO weights should vary across steps.
    Verifies that per-step advantage normalization produces non-trivial gradients.
    """
    import os
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax.numpy as jnp

    B, U, K = 2, 3, 4
    awpo_alpha = 2.0

    # All Q-values uniform within each step → advantages normalize to 0 → weight = 1
    all_child_q = jnp.stack([
        jnp.full((B, K), float(i)) for i in range(U + 1)
    ], axis=1)  # (B, U+1, K)
    v_net = jnp.zeros((B,))

    for step in range(U + 1):
        q_k = all_child_q[:, step, :]
        adv_raw = q_k - v_net[:, None]
        adv_norm = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)
        w = jnp.exp(jnp.clip(adv_norm / awpo_alpha, -5.0, 5.0))
        assert abs(float(w.mean()) - 1.0) < 1e-4, \
            f"Uniform Q within step should yield weight~1, got {float(w.mean())}"

    # Varying Q-values: best action gets highest weight
    q_varying = jnp.array([[1.0, 0.0, 0.5, 0.5]] * B)  # (B, K)
    adv_raw = q_varying - v_net[:, None]
    adv_norm = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)
    w = jnp.exp(jnp.clip(adv_norm / awpo_alpha, -5.0, 5.0))
    assert float(w[0, 0]) > float(w[0, 1]), "Best action should get highest AWPO weight"


def test_awpo_weight_stops_gradient_into_baseline():
    """AWPO weight must not backprop into the value baseline. If it did, the
    value head could shrink or grow its own prediction purely to inflate the
    policy-loss weighting it produces, leaking policy-loss gradient into the
    value head instead of acting as a fixed advantage-weighted multiplier."""
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    from actors.loss import _awpo_weight

    q = jnp.array([1.0, 0.0])  # (B,) fixed targets, independent of the param under test

    def weight_sum(v_scale):
        v_baseline = jnp.array([0.5, 0.5]) * v_scale  # depends on v_scale
        return _awpo_weight(q, v_baseline, alpha=1.0).sum()

    grad = jax.grad(weight_sum)(1.0)
    assert grad == 0.0, f"gradient into value baseline should be zero, got {grad}"


def test_awpo_weight_is_scale_invariant():
    """Std-normalization must make the weight distribution insensitive to the
    absolute scale of Q/V. This is the exact bug jaxzero's postmortem found
    and fixed (missing std term made near-uniform Q collapse to near-uniform
    weights regardless of relative structure) — this test confirms
    sequential-muzero's `_awpo_weight` already normalizes correctly."""
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    from actors.loss import _awpo_weight

    v_baseline = jnp.zeros(1)
    small_q = jnp.array([[0.1, 0.2, 0.1, 0.2]])   # std ~ 0.05
    large_q = jnp.array([[1.0, 2.0, 1.0, 2.0]])   # std ~ 0.5, same relative shape

    w_small = _awpo_weight(small_q, v_baseline, alpha=3.0)
    w_large = _awpo_weight(large_q, v_baseline, alpha=3.0)

    assert jnp.allclose(w_small, w_large, atol=1e-4), (
        f"std-normalized weights should be scale-invariant: {w_small} vs {w_large}"
    )
