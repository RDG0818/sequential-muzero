import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax
import jax.numpy as jnp
import pytest
from utils.wrappers.mpe_wrapper import MPEWrapper

ENV_NAME = 'MPE_simple_v3'
NUM_AGENTS = 1

@pytest.fixture
def mpe_wrapper():
    """Pytest fixture to create an instance of the MPEEnvWrapper."""
    return MPEWrapper(env_name=ENV_NAME, num_agents=NUM_AGENTS)

def test_mpe_wrapper_init(mpe_wrapper: MPEWrapper):
    """Tests if the wrapper initializes with the correct properties."""
    assert mpe_wrapper.num_agents == NUM_AGENTS
    assert mpe_wrapper.action_space_size > 0
    assert mpe_wrapper.observation_shape is not None

def test_mpe_wrapper_reset(mpe_wrapper: MPEWrapper):
    """Tests the JIT-compiled reset function."""
    key = jax.random.PRNGKey(0)
    obs, state = mpe_wrapper.reset(key)

    expected_obs_shape = (NUM_AGENTS, *mpe_wrapper.observation_shape)
    assert isinstance(obs, jnp.ndarray)
    assert obs.shape == expected_obs_shape

    assert state is not None

def test_mpe_wrapper_step(mpe_wrapper: MPEWrapper):
    """Tests the JIT-compiled step function."""
    key = jax.random.PRNGKey(0)
    _, state = mpe_wrapper.reset(key)

    actions = jnp.zeros((NUM_AGENTS,), dtype=jnp.int32)
    
    step_key, _ = jax.random.split(key)
    next_obs, next_state, reward, done, info = mpe_wrapper.step(step_key, state, actions)

    expected_obs_shape = (NUM_AGENTS, *mpe_wrapper.observation_shape)
    assert isinstance(next_obs, jnp.ndarray)
    assert next_obs.shape == expected_obs_shape
    assert next_state is not None
    
    assert isinstance(reward, jnp.ndarray)
    assert reward.shape == () 
    assert isinstance(done, jnp.ndarray)
    assert done.shape == ()

    key, step_key = jax.random.split(key)
    next_obs, next_state, reward, done, info = mpe_wrapper.step(step_key, state, actions)
    assert next_obs is not None

