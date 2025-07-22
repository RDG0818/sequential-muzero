import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import jax.numpy as jnp
import numpy as np
from utils.utils import n_step_returns_fn

def test_n_step_targets_standard_case():
    """
    Tests the N-step return calculation for a standard case where the
    sequence length is greater than n_steps.
    """
    n_steps = 3
    discount_gamma = 0.9
    
    # T = 5, rewards are R_1 to R_5
    rewards = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # T+1 = 6, mcts_values are V_0 to V_5
    mcts_values = jnp.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

    # --- Manually calculated expected values ---
    # G_4 (h=1): R_5 + g*V_5 = 5.0 + 0.9*10.0 = 14.0
    # G_3 (h=2): R_4 + g*R_5 + g^2*V_5 = 4.0 + 0.9*5.0 + 0.81*10.0 = 4.0 + 4.5 + 8.1 = 16.6
    # G_2 (h=3): R_3 + g*R_4 + g^2*R_5 + g^3*V_5 = 3.0 + 0.9*4.0 + 0.81*5.0 + 0.729*10.0 = 3.0 + 3.6 + 4.05 + 7.29 = 17.94
    # G_1 (h=3): R_2 + g*R_3 + g^2*R_4 + g^3*V_4 = 2.0 + 0.9*3.0 + 0.81*4.0 + 0.729*10.0 = 2.0 + 2.7 + 3.24 + 7.29 = 15.23
    # G_0 (h=3): R_1 + g*R_2 + g^2*R_3 + g^3*V_3 = 1.0 + 0.9*2.0 + 0.81*3.0 + 0.729*10.0 = 1.0 + 1.8 + 2.43 + 7.29 = 12.52
    expected_targets = np.array([12.52, 15.23, 17.94, 16.6, 14.0, 10.0])

    # --- Run the function ---
    actual_targets = n_step_returns_fn(rewards, mcts_values, n_steps, discount_gamma)

    # --- Assert correctness ---
    assert actual_targets.shape == expected_targets.shape
    np.testing.assert_allclose(actual_targets, expected_targets, rtol=1e-5)
    print("\nTest case 1 (Standard) passed!")


def test_n_step_targets_short_sequence():
    """
    Tests the N-step return calculation for a case where the sequence
    length is shorter than n_steps, forcing the shrinking horizon for all steps.
    """
    n_steps = 5  # n_steps > T
    discount_gamma = 0.9

    # T = 2, rewards are R_1, R_2
    rewards = jnp.array([1.0, 2.0])
    # T+1 = 3, mcts_values are V_0, V_1, V_2
    mcts_values = jnp.array([10.0, 10.0, 10.0])

    # --- Manually calculated expected values ---
    # G_1 (h=1): R_2 + g*V_2 = 2.0 + 0.9*10.0 = 11.0
    # G_0 (h=2): R_1 + g*R_2 + g^2*V_2 = 1.0 + 0.9*2.0 + 0.81*10.0 = 1.0 + 1.8 + 8.1 = 10.9
    expected_targets = np.array([10.9, 11.0, 10.0])

    # --- Run the function ---
    actual_targets = n_step_returns_fn(rewards, mcts_values, n_steps, discount_gamma)

    # --- Assert correctness ---
    assert actual_targets.shape == expected_targets.shape
    np.testing.assert_allclose(actual_targets, expected_targets, rtol=1e-5)
    print("Test case 2 (Short Sequence) passed!")