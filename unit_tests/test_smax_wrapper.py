import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax
import jax.numpy as jnp
import pytest
import chex
from utils.wrappers.smax_wrapper import SMAXWrapper

SCENARIO_NAME = '5m_vs_6m'
# In the '5m_vs_6m' scenario, there are 5 allied agents we control.
NUM_ALLIES = 5 

@pytest.fixture
def smax_wrapper():
    """Pytest fixture to create an instance of the SMAXWrapper."""
    return SMAXWrapper(scenario_name=SCENARIO_NAME)

def test_smax_wrapper_init(smax_wrapper: SMAXWrapper):
    """Tests if the SMAX wrapper initializes with the correct properties."""
    assert smax_wrapper.num_agents == NUM_ALLIES
    assert smax_wrapper.action_space_size > 0
    assert smax_wrapper.observation_shape is not None

def test_smax_wrapper_reset(smax_wrapper: SMAXWrapper):
    """Tests the JIT-compiled reset function for SMAX."""
    key = jax.random.PRNGKey(0)
    obs, state = smax_wrapper.reset(key)

    expected_obs_shape = (NUM_ALLIES, *smax_wrapper.observation_shape)
    assert isinstance(obs, jnp.ndarray)
    assert obs.shape == expected_obs_shape
    assert state is not None

def test_smax_wrapper_step(smax_wrapper: SMAXWrapper):
    """
    Tests the JIT-compiled step function for SMAX, including the
    retrieval of available actions.
    """
    key = jax.random.PRNGKey(0)
    key, reset_key, step_key = jax.random.split(key, 3)
    _, state = smax_wrapper.reset(reset_key)

    actions = jnp.zeros((NUM_ALLIES,), dtype=jnp.int32)
    
    next_obs, next_state, reward, done, info = smax_wrapper.step(step_key, state, actions)

    expected_obs_shape = (NUM_ALLIES, *smax_wrapper.observation_shape)
    assert isinstance(next_obs, jnp.ndarray)
    assert next_obs.shape == expected_obs_shape
    assert next_state is not None
    
    assert isinstance(reward, jnp.ndarray)
    assert reward.shape == () 
    
    assert isinstance(done, jnp.ndarray)
    assert done.shape == ()

    assert "avail_actions" in info
    avail_actions = info["avail_actions"]
    
    expected_avail_actions_shape = (NUM_ALLIES, smax_wrapper.action_space_size)
    assert isinstance(avail_actions, jnp.ndarray)
    assert avail_actions.shape == expected_avail_actions_shape
    assert avail_actions.dtype == jnp.bool_

