# tests/test_model.py

import pytest
import jax
import jax.numpy as jnp
import chex
import sys
import os
from dataclasses import dataclass

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from model.model import FlaxMAMuZeroNet

@dataclass
class MockModelConfig:
    policy_type: str = "transformer"
    attention_layers: int = 2
    attention_heads: int = 4
    hidden_state_size: int = 64
    dropout_rate: float = 0.1
    fc_representation_layers: tuple = (128,)
    reward_support_size: int = 20
    value_support_size: int = 20
    fc_dynamic_layers: tuple = (128,)
    fc_reward_layers: tuple = (64,)
    decoder_blocks: int = 2
    policy_heads: int = 4
    fc_value_layers: tuple = (128, 64)
    fc_policy_layers: tuple = (128, 64)
    proj_hid: int = 128
    proj_out: int = 128
    pred_hid: int = 64
    pred_out: int = 128

@pytest.fixture(params=["transformer", "standard"])
def setup_muzero_net(request):
    """Pytest fixture to set up the full MuZero network and dummy data."""
    policy_type = request.param
    config = MockModelConfig(policy_type=policy_type)
    
    batch_size = 4
    num_agents = 5
    obs_dim = 10
    action_space_size = 7
    key = jax.random.PRNGKey(0)

    model = FlaxMAMuZeroNet(config=config, action_space_size=action_space_size)

    observations = jnp.ones((batch_size, num_agents, obs_dim))
    hidden_states = jnp.ones((batch_size, num_agents, config.hidden_state_size))
    actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)

    return {
        "model": model,
        "key": key,
        "observations": observations,
        "hidden_states": hidden_states,
        "actions": actions,
        "config": config
    }

def test_initial_inference(setup_muzero_net):
    """Tests the `__call__` method (initial inference)."""
    d = setup_muzero_net
    model, key, obs = d["model"], d["key"], d["observations"]

    params_key, dropout_key, inference_key = jax.random.split(key, 3)

    params = model.init(
        {'params': params_key, 'dropout': dropout_key}, obs, inference_key
    )['params']

    output = model.apply({'params': params}, obs, inference_key)

    chex.assert_shape(output.policy_logits, (4, 5, model.action_space_size))
    chex.assert_shape(output.value_logits, (4, d["config"].value_support_size * 2 + 1))
    print(f"\nPytest (Initial Inference, policy={d['config'].policy_type}): Shape test successful!")

def test_recurrent_inference(setup_muzero_net):
    """Tests the `recurrent_inference` method."""
    d = setup_muzero_net
    model, key, states, actions = d["model"], d["key"], d["hidden_states"], d["actions"]

    params_key, dropout_key, inference_key = jax.random.split(key, 3)
    
    params = model.init(
        {'params': params_key, 'dropout': dropout_key}, d["observations"], inference_key
    )['params']

    output = model.apply({'params': params}, states, actions, inference_key, method=model.recurrent_inference)

    chex.assert_shape(output.policy_logits, (4, 5, model.action_space_size))
    chex.assert_shape(output.value_logits, (4, d["config"].value_support_size * 2 + 1))
    print(f"Pytest (Recurrent Inference, policy={d['config'].policy_type}): Shape test successful!")

def test_train_prediction(setup_muzero_net):
    """Tests the `train_prediction` method."""
    d = setup_muzero_net
    model, key, states, actions = d["model"], d["key"], d["hidden_states"], d["actions"]

    params_key, dropout_key = jax.random.split(key, 2)
    
    # Initialize using the main entry point
    params = model.init(
        {'params': params_key, 'dropout': dropout_key}, d["observations"], key
    )['params']

    policy_logits, value_logits = model.apply(
        {'params': params}, states, actions, rngs={'dropout': dropout_key}, method=model.train_prediction
    )

    chex.assert_shape(policy_logits, (4, 5, model.action_space_size))
    chex.assert_shape(value_logits, (4, d["config"].value_support_size * 2 + 1))
    print(f"Pytest (Train Prediction, policy={d['config'].policy_type}): Shape test successful!")
