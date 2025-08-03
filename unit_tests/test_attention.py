# tests/test_attention.py

import pytest
import jax
import jax.numpy as jnp
import chex
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from model.attention import TransformerAttentionEncoder
from model.policy_attention import DecoderBlock, AutoregressivePredictionNetwork


# ===================================================================
#        Tests for the Standalone DecoderBlock
# ===================================================================

@pytest.fixture
def setup_decoder_block_data():
    """Pytest fixture to set up data specifically for DecoderBlock tests."""
    batch_size = 4
    num_agents = 5
    hidden_size = 64
    key = jax.random.PRNGKey(0)
    
    decoder_block = DecoderBlock(num_heads=4, hidden_size=hidden_size)
    
    # Dummy data
    x = jnp.ones((batch_size, num_agents, hidden_size))
    context = jnp.ones((batch_size, num_agents, hidden_size))
    
    return {
        "decoder_block": decoder_block,
        "key": key,
        "x": x,
        "context": context,
        "batch_size": batch_size,
        "num_agents": num_agents,
        "hidden_size": hidden_size
    }

def test_decoder_block_shape(setup_decoder_block_data):
    """Tests the output shape of a single DecoderBlock."""
    d = setup_decoder_block_data
    params = d["decoder_block"].init(d["key"], d["x"], d["context"], deterministic=True)['params']
    output = d["decoder_block"].apply({'params': params}, d["x"], d["context"], deterministic=True)
    chex.assert_shape(output, (d["batch_size"], d["num_agents"], d["hidden_size"]))
    print("\nPytest (DecoderBlock): Forward pass shape test successful!")

def test_decoder_block_causal_masking(setup_decoder_block_data):
    """Tests the causal masking within a single DecoderBlock."""
    d = setup_decoder_block_data
    params = d["decoder_block"].init(d["key"], d["x"], d["context"], deterministic=True)['params']
    
    output1 = d["decoder_block"].apply({'params': params}, d["x"], d["context"], deterministic=True)
    modified_x = d["x"].at[:, -1, :].set(0.0)
    output2 = d["decoder_block"].apply({'params': params}, modified_x, d["context"], deterministic=True)
    
    assert jnp.allclose(output1[:, 0, :], output2[:, 0, :])
    print("Pytest (DecoderBlock): Causal masking test successful!")

def test_decoder_block_padding_masking(setup_decoder_block_data):
    """
    Tests that the DecoderBlock correctly ignores padded positions in the input.
    """
    d = setup_decoder_block_data
    params = d["decoder_block"].init(d["key"], d["x"], d["context"], deterministic=True)['params']

    # Create a padding mask where the last two elements are masked out (False)
    padding_mask = jnp.array([[True, True, True, False, False]])
    padding_mask = jnp.repeat(padding_mask, d["batch_size"], axis=0)

    # --- First pass with the mask ---
    output1 = d["decoder_block"].apply(
        {'params': params}, d["x"], d["context"], padding_mask=padding_mask, deterministic=True
    )

    # --- Second pass with modified data in the padded positions ---
    modified_x = d["x"].at[:, -1, :].set(jnp.ones(d["hidden_size"]) * 5.0) # Change the last element
    output2 = d["decoder_block"].apply(
        {'params': params}, modified_x, d["context"], padding_mask=padding_mask, deterministic=True
    )

    # The output for the valid positions (e.g., the first element) should be IDENTICAL
    # because the change we made was in a padded position and should be ignored.
    assert jnp.allclose(output1[:, 0, :], output2[:, 0, :])
    print("Pytest (DecoderBlock): Padding masking test successful!")


# ===================================================================
#        Tests for the Full AutoregressivePredictionNetwork
# ===================================================================

@pytest.fixture
def setup_full_policy_net_data():
    """Pytest fixture to set up the full autoregressive network and its dependencies."""
    batch_size = 4
    num_agents = 5
    hidden_size = 64
    action_space_size = 10
    value_support_size = 20
    key = jax.random.PRNGKey(42)

    # 1. Instantiate the encoder that will be injected
    encoder = TransformerAttentionEncoder(
        num_layers=2,
        num_heads=4,
        hidden_size=hidden_size,
        action_space_size=action_space_size, # Required by its signature
        dropout_rate=0.1
    )

    # 2. Instantiate the main network, passing the encoder in
    policy_net = AutoregressivePredictionNetwork(
        encoder=encoder,
        action_space_size=action_space_size,
        value_support_size=value_support_size,
        num_decoder_blocks=2,
        num_heads=4,
        hidden_state_size=hidden_size,
        dropout_rate=0.1,
        fc_value_layers=(128, 64)
    )
    
    # 3. Dummy data for the test
    hidden_states = jnp.ones((batch_size, num_agents, hidden_size))
    actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)
    
    return {
        "policy_net": policy_net,
        "key": key,
        "hidden_states": hidden_states,
        "actions": actions,
        "batch_size": batch_size,
        "num_agents": num_agents,
        "action_space_size": action_space_size,
        "value_output_size": value_support_size * 2 + 1,
        "hidden_size": hidden_size
    }


def test_train_forward_shapes(setup_full_policy_net_data):
    """
    Tests the `train_forward` method to verify the teacher-forcing path.
    """
    d = setup_full_policy_net_data
    policy_net = d["policy_net"]
    
    # Initialize parameters, providing RNGs for 'params' and 'dropout'
    params_key, dropout_key = jax.random.split(d["key"])
    params = policy_net.init(
        {'params': params_key, 'dropout': dropout_key}, 
        d["hidden_states"], 
        d["actions"], 
        train=True,
    )['params']
    
    # Run the training forward pass, providing an RNG for the 'dropout' stream
    policy_logits, value_logits = policy_net.apply(
        {'params': params}, 
        d["hidden_states"], 
        d["actions"], 
        train=True,
        rngs={'dropout': dropout_key}
    )
    
    # Assert output shapes
    chex.assert_shape(policy_logits, (d["batch_size"], d["num_agents"], policy_net.action_space_size))
    chex.assert_shape(value_logits, (d["batch_size"], policy_net.value_support_size * 2 + 1))
    print("\nPytest (Full Network): `train_forward` method test successful!")


def test_generate_function_shape(setup_full_policy_net_data):
    """
    Tests that the final `generate` method produces a sequence of actions 
    of the correct shape.
    """
    d = setup_full_policy_net_data
    policy_net = d["policy_net"]
    key = d["key"]
    hidden_states = d["hidden_states"]

    # Initialize parameters
    params = policy_net.init(key, hidden_states, d["actions"], train=False)['params']
    
    # Run the generation process
    policy_logits, value_logits = policy_net.apply(
        {'params': params},
        hidden_states,
        key=key,
        deterministic=True,
        method=policy_net.generate
    )

    # Assert that the output shape is correct
    chex.assert_shape(policy_logits, (d["batch_size"], d["num_agents"], policy_net.action_space_size))
    chex.assert_shape(value_logits, (d["batch_size"], policy_net.value_support_size * 2 + 1))
    print("\nPytest (Full Network): Generate function shape test successful!")