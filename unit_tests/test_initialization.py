import sys
import os
import jax
import jax.numpy as jnp
import optax
import pytest
import chex
from typing import Callable, Any, Dict
from functools import partial

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import (
    env_setup,
    model_setup,
    planner_setup,
    action_setup,
    step_setup,
    replay_buffer_setup,
    RunnerState,
    _rollout_step,
    _update_step
)
from utils.utils import n_step_returns_fn
from config import CONFIG
from utils.wrappers.mpe_wrapper import MPEWrapper
from model.model import FlaxMAMuZeroNet
from utils.replay_buffer import ReplayBuffer

NUM_ENVS = CONFIG.train.num_envs
NUM_AGENTS = 3
ENV_NAME = "MPE_simple_spread_v3"
ALPHA = CONFIG.train.replay_buffer_alpha
BETA = CONFIG.train.replay_buffer_beta_start
TAU = CONFIG.train.tau


@pytest.fixture(scope="module")
def setup_data():
    """
    A pytest fixture to run expensive initialization once for all tests.
    This avoids re-initializing the environment and model for each test function.
    """
    key = jax.random.PRNGKey(0)
    key, env_key, model_key = jax.random.split(key, 3)
    
    env, vmapped_reset, vmapped_step, obs_batch, state_batch = env_setup(env_key, NUM_ENVS, NUM_AGENTS, ENV_NAME)
    model, params, optimizer, opt_state, _ = model_setup(model_key, env, obs_batch)
    vmapped_plan = planner_setup(model)
    replay_buffer = replay_buffer_setup(obs_batch, ALPHA, env)
    
    return {
        "env": env,
        "vmapped_reset": vmapped_reset,
        "vmapped_step": vmapped_step,
        "obs_batch": obs_batch,
        "state_batch": state_batch,
        "model": model,
        "params": params,
        "optimizer": optimizer,
        "opt_state": opt_state,
        "vmapped_plan": vmapped_plan,
        "replay_buffer": replay_buffer
    }

def test_env_setup():
    """Tests the environment initialization function."""
    key = jax.random.PRNGKey(42)
    env, vmapped_reset, vmapped_step, obs_batch, state_batch = env_setup(key, NUM_ENVS, NUM_AGENTS, ENV_NAME)
    
    assert isinstance(env, MPEWrapper)
    assert callable(vmapped_reset)
    assert callable(vmapped_step)
    assert state_batch is not None

    chex.assert_shape(obs_batch, (NUM_ENVS, NUM_AGENTS, env.observation_shape[0]))
    chex.assert_type(obs_batch, jnp.float32)

def test_model_setup(setup_data):
    """Tests the model and optimizer initialization function."""
    key = jax.random.PRNGKey(42)
    env = setup_data["env"]
    obs_batch = setup_data["obs_batch"]
    
    model, params, optimizer, opt_state, lr_schedule = model_setup(key, env, obs_batch)
    
    assert isinstance(model, FlaxMAMuZeroNet)
    assert isinstance(params, Dict)
    assert isinstance(optimizer, optax.GradientTransformation)
    assert isinstance(opt_state, tuple)
    assert callable(lr_schedule)

def test_planner_setup(setup_data):
    """Tests the MCTS planner initialization function."""
    model = setup_data["model"]
    vmapped_plan = planner_setup(model)
    
    assert callable(vmapped_plan)

def test_action_setup(setup_data):
    """Tests the debug action selection function."""
    key = jax.random.PRNGKey(42)
    plan = setup_data["vmapped_plan"]
    params = setup_data["params"]
    obs = setup_data["obs_batch"]
    env = setup_data["env"]

    action, policy_targets = action_setup(key, plan, params, obs, NUM_ENVS, env)
    
    chex.assert_shape(action, (NUM_ENVS, NUM_AGENTS))
    chex.assert_shape(policy_targets, (NUM_ENVS, NUM_AGENTS, env.action_space_size))

def test_step_setup(setup_data):
    """Tests the debug environment step function."""
    key = jax.random.PRNGKey(42)
    key, action_key = jax.random.split(key)
    
    action, _ = action_setup(action_key, setup_data["vmapped_plan"], setup_data["params"], setup_data["obs_batch"], NUM_ENVS, setup_data["env"])

    step_fn = setup_data["vmapped_step"]
    state = setup_data["state_batch"]
    
    next_obs, _, reward, done, _ = step_setup(key, step_fn, NUM_ENVS, state, action)
    
    chex.assert_shape(next_obs, (NUM_ENVS, NUM_AGENTS, setup_data["env"].observation_shape[0]))
    chex.assert_type(next_obs, jnp.float32)
    chex.assert_shape(reward, (NUM_ENVS,))
    chex.assert_shape(done, (NUM_ENVS,))

def test_replay_buffer_setup(setup_data):
    """Tests the JAX-native replay buffer initialization robustly."""
    obs = setup_data["obs_batch"]
    env = setup_data["env"]
    
    replay_buffer = replay_buffer_setup(obs, ALPHA, env)

    assert isinstance(replay_buffer, ReplayBuffer)
    assert replay_buffer.capacity == CONFIG.train.replay_buffer_size
    assert replay_buffer.alpha == CONFIG.train.replay_buffer_alpha
    assert replay_buffer.position == 0
    assert jnp.all(replay_buffer.priorities == 0)

    assert replay_buffer.data.obs.shape == (CONFIG.train.replay_buffer_size, *obs.shape[1:])
    assert replay_buffer.data.action.shape == (CONFIG.train.replay_buffer_size, env.num_agents)
    assert replay_buffer.data.reward.shape == (CONFIG.train.replay_buffer_size,)
    assert replay_buffer.data.done.shape == (CONFIG.train.replay_buffer_size,)
    assert replay_buffer.data.policy_target.shape == (CONFIG.train.replay_buffer_size, env.num_agents, env.action_space_size)
    assert replay_buffer.data.value_target.shape == (CONFIG.train.replay_buffer_size,)
    
    assert replay_buffer.data.obs.dtype == obs.dtype
    assert replay_buffer.data.action.dtype == jnp.int32
    assert replay_buffer.data.done.dtype == jnp.bool_


def test_rollout_step(setup_data):
    """Tests a single step of the data collection rollout."""
    key = jax.random.PRNGKey(123)
    
    initial_runner_state = RunnerState(
        params=setup_data["params"],
        #target_params=setup_data["params"],
        opt_state=setup_data["opt_state"],
        key=key,
        env_state=setup_data["state_batch"],
        obs=setup_data["obs_batch"],
        replay_buffer=setup_data["replay_buffer"],
        episode_returns=jnp.zeros(NUM_ENVS),
        episode_lengths=jnp.zeros(NUM_ENVS, dtype=jnp.int32),
        delta_magnitudes=jnp.zeros(NUM_ENVS),
        coord_state_norms=jnp.zeros(NUM_ENVS)
    )

    initial_buffer = initial_runner_state.replay_buffer
    initial_position = initial_buffer.position
    initial_max_priority = jnp.max(initial_buffer.priorities)
    expected_new_priority = jnp.maximum(initial_max_priority, 1.0)

    rollout_step_fn = partial(
        _rollout_step, 
        plan=setup_data["vmapped_plan"], 
        vmapped_step=setup_data["vmapped_step"], 
        vmapped_reset=setup_data["vmapped_reset"],
        env=setup_data["env"]
    )
    
    next_runner_state, logged_metrics = rollout_step_fn(initial_runner_state, None)

    assert not jnp.array_equal(initial_runner_state.key, next_runner_state.key)

    next_buffer = next_runner_state.replay_buffer

    expected_pos = (initial_position + NUM_ENVS) % initial_buffer.capacity
    assert next_buffer.position == expected_pos

    added_indices = (initial_position + jnp.arange(NUM_ENVS)) % initial_buffer.capacity
    newly_set_priorities = next_buffer.priorities[added_indices]
    assert jnp.all(newly_set_priorities == expected_new_priority)

    chex.assert_shape(next_runner_state.obs, (NUM_ENVS, NUM_AGENTS, setup_data["env"].observation_shape[0]))

def test_update_step(setup_data):
    """Tests a single training update step, ensuring parameters and priorities change."""
    key = jax.random.PRNGKey(456)

    rollout_fn = partial(
        _rollout_step, 
        plan=setup_data["vmapped_plan"], 
        vmapped_step=setup_data["vmapped_step"], 
        vmapped_reset=setup_data["vmapped_reset"],
        env=setup_data["env"]
    )
    
    runner_state = RunnerState(
        params=setup_data["params"],
        #target_params=setup_data["params"],
        opt_state=setup_data["opt_state"],
        key=key,
        env_state=setup_data["state_batch"],
        obs=setup_data["obs_batch"],
        replay_buffer=setup_data["replay_buffer"],
        episode_returns=jnp.zeros(NUM_ENVS),
        episode_lengths=jnp.zeros(NUM_ENVS, dtype=jnp.int32),
        delta_magnitudes=jnp.zeros(NUM_ENVS),
        coord_state_norms=jnp.zeros(NUM_ENVS)
    )

    for _ in range(CONFIG.train.unroll_steps + 1):
        runner_state, _ = rollout_fn(runner_state, None)

    vmapped_n_step_returns = jax.vmap(n_step_returns_fn, in_axes=(0, 0, None, None))
    update_fn = partial(
        _update_step, 
        model=setup_data["model"], 
        optimizer=setup_data["optimizer"],
        vmapped_n_step_returns=vmapped_n_step_returns
    )

    jitted_update_fn = jax.jit(update_fn)

    initial_priorities = runner_state.replay_buffer.priorities
    initial_key = runner_state.key

    state_after_first_update, _ = jitted_update_fn(runner_state)
    state_after_second_update, metrics2 = jitted_update_fn(state_after_first_update)

    initial_params_flat, _ = jax.tree_util.tree_flatten(runner_state.params)
    final_params_flat, _ = jax.tree_util.tree_flatten(state_after_second_update.params)

    # Check that the online model parameters have been updated after the second step
    updated = False
    for p1, p2 in zip(initial_params_flat, final_params_flat):
        if not jnp.allclose(p1, p2):
            updated = True
            break
    assert updated, "Online model parameters were not updated after two steps. Gradients may be zero or LR is stuck at 0."

    initial_opt_flat, _ = jax.tree_util.tree_flatten(runner_state.opt_state)
    final_opt_flat, _ = jax.tree_util.tree_flatten(state_after_second_update.opt_state)
    assert not all(jnp.allclose(s1, s2) for s1, s2 in zip(initial_opt_flat, final_opt_flat)), "Optimizer state was not updated."

    assert jnp.isfinite(metrics2["total_loss"]), f"Total loss is not finite: {metrics2['total_loss']}"

    final_target_params_flat, _ = jax.tree_util.tree_flatten(state_after_second_update.target_params)
    expected_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * CONFIG.train.tau + tp * (1 - CONFIG.train.tau),
        state_after_first_update.params,
        runner_state.target_params
    )
    expected_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * CONFIG.train.tau + tp * (1 - CONFIG.train.tau),
        state_after_second_update.params,
        expected_target_params
    )
    expected_target_params_flat, _ = jax.tree_util.tree_flatten(expected_target_params)
    
    for expected, actual in zip(expected_target_params_flat, final_target_params_flat):
        assert jnp.allclose(expected, actual, atol=1e-6), "Target network parameters were not updated correctly."

    # 5. Check that replay buffer priorities have changed
    final_priorities = state_after_second_update.replay_buffer.priorities
    assert not jnp.allclose(initial_priorities, final_priorities), "Replay buffer priorities were not updated."

    # 6. Check that the PRNG key has been updated
    assert not jnp.array_equal(initial_key, state_after_second_update.key), "PRNG key was not updated."

