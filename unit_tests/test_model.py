# unit_tests/test_model.py

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import chex
from flax.core import freeze

# Adjust the import paths based on your project structure
from model.model import FlaxMAMuZeroNet, MuZeroOutput, RepresentationNetwork, DynamicsNetwork, PredictionNetwork
from model.attention import TransformerAttentionEncoder, BaseAttention
from config import ModelConfig

class TestModel(unittest.TestCase):
    """Unit tests for the MuZero model components."""

    def setUp(self):
        """Set up common parameters and dummy data for all tests."""
        print("\n" + "="*50)
        print("Setting up common test parameters...")
        self.key = jax.random.PRNGKey(42)

        # --- Define model dimensions ---
        self.batch_size = 8
        self.num_agents = 3
        self.obs_dim = 10
        self.action_space_size = 5
        self.hidden_state_size = 64
        self.value_support_size = 10
        self.reward_support_size = 10
        self.unroll_steps = 5

        # --- Create dummy data ---
        self.dummy_obs = jnp.zeros((self.batch_size, self.num_agents, self.obs_dim))
        self.dummy_hidden_states = jnp.zeros((self.batch_size, self.num_agents, self.hidden_state_size))
        self.dummy_actions = jnp.zeros((self.batch_size, self.num_agents), dtype=jnp.int32)
        
        # --- Create model configurations ---
        self.config_no_attention = ModelConfig(
            hidden_state_size=self.hidden_state_size,
            value_support_size=self.value_support_size,
            reward_support_size=self.reward_support_size,
            attention_type="none" # Explicitly no attention
        )
        self.config_with_attention = ModelConfig(
            hidden_state_size=self.hidden_state_size,
            value_support_size=self.value_support_size,
            reward_support_size=self.reward_support_size,
            attention_type="transformer", # Use transformer attention
            attention_layers=2,
            attention_heads=2,
            dropout_rate=0.1
        )
        print("Setup complete.")
        print("="*50)

    def test_representation_network(self):
        """Tests the RepresentationNetwork's output shape."""
        print("\n--- Testing RepresentationNetwork ---")
        model = RepresentationNetwork(
            hidden_state_size=self.hidden_state_size,
            fc_layers=(128,)
        )
        # The network expects flattened observations per agent
        flat_obs = self.dummy_obs.reshape(self.batch_size * self.num_agents, -1)
        params = model.init(self.key, flat_obs)['params']
        output = model.apply({'params': params}, flat_obs)
        
        print(f"Input shape (flat obs): {flat_obs.shape}")
        print(f"Output shape (latent state): {output.shape}")
        
        chex.assert_shape(output, (self.batch_size * self.num_agents, self.hidden_state_size))
        print("✅ RepresentationNetwork shape test passed.")

    def test_dynamics_network_configs(self):
        """Tests the DynamicsNetwork with and without an attention module."""
        print("\n--- Testing DynamicsNetwork Configurations ---")

        # --- Case 1: No Attention (No changes needed) ---
        with self.subTest("Without Attention"):
            # ... (this sub-test remains the same)
            pass

        # --- Case 2: With Attention (Requires RNG handling) ---
        with self.subTest("With Transformer Attention"):
            print("\n  Sub-test: With Transformer Attention")
            # Create a key for dropout operations
            init_key, dropout_key = jax.random.split(self.key)
            
            attention_module = TransformerAttentionEncoder(
                num_layers=2, num_heads=2, hidden_size=self.hidden_state_size, dropout_rate=0.1
            )
            model = DynamicsNetwork(
                hidden_state_size=self.hidden_state_size,
                action_space_size=self.action_space_size,
                reward_support_size=self.reward_support_size,
                fc_dynamic_layers=(64,),
                fc_reward_layers=(32,),
                attention_module=attention_module
            )
            
            try:
                # Initialize params, providing a key for the 'dropout' RNG stream
                params = model.init(
                    {'params': init_key, 'dropout': dropout_key}, 
                    self.dummy_hidden_states, 
                    self.dummy_actions
                )['params']

                # Apply the model, providing the 'dropout' RNG stream
                next_hidden, reward_logits = model.apply(
                    {'params': params},
                    self.dummy_hidden_states,
                    self.dummy_actions,
                    rngs={'dropout': dropout_key} # <-- FIX: Provide RNGs
                )

                print(f"  Input shapes: hidden={self.dummy_hidden_states.shape}, actions={self.dummy_actions.shape}")
                print(f"  Output shapes: next_hidden={next_hidden.shape}, reward_logits={reward_logits.shape}")

                chex.assert_shape(next_hidden, (self.batch_size, self.num_agents, self.hidden_state_size))
                chex.assert_shape(reward_logits, (self.batch_size, self.reward_support_size * 2 + 1))
                print("  ✅ DynamicsNetwork (With Attention) shape test passed.")

            except Exception as e:
                print(f"  ❌ Test Failed: {e.__class__.__name__}")
                print(f"     Error: {e}")
                self.fail(f"DynamicsNetwork test failed with exception: {e}")

    def test_full_model_inference(self):
        """Tests the full model's initial and recurrent inference calls."""
        print("\n--- Testing Full Model Inference ---")
        
        for name, config in [("No Attention", self.config_no_attention), ("With Attention", self.config_with_attention)]:
            with self.subTest(name):
                print(f"\n  Sub-test: {name}")
                model = FlaxMAMuZeroNet(config=config, action_space_size=self.action_space_size)
                
                # We need separate keys for init, initial call, and recurrent call
                init_key, dropout_key_initial, dropout_key_recurrent = jax.random.split(self.key, 3)
                
                try:
                    # Initialize with dropout RNG stream if attention is on
                    rngs_for_init = {'params': init_key}
                    if config.attention_type != "none":
                        rngs_for_init['dropout'] = dropout_key_initial

                    params = model.init(rngs_for_init, self.dummy_obs)['params']

                    # --- Test Initial Inference (__call__) ---
                    print("    Testing initial inference...")
                    rngs_for_apply = {'dropout': dropout_key_initial} if config.attention_type != "none" else None
                    output = model.apply({'params': params}, self.dummy_obs, rngs=rngs_for_apply)
                    
                    self.assertIsInstance(output, MuZeroOutput)
                    print(f"    Initial output type: {type(output)}")
                    chex.assert_shape(output.policy_logits, (self.batch_size, self.num_agents, self.action_space_size))
                    print("    ✅ Initial inference shape test passed.")

                    # --- Test Recurrent Inference ---
                    print("    Testing recurrent inference...")
                    rngs_for_recurrent = {'dropout': dropout_key_recurrent} if config.attention_type != "none" else None
                    recurrent_output = model.apply(
                        {'params': params},
                        self.dummy_hidden_states,
                        self.dummy_actions,
                        method=model.recurrent_inference,
                        rngs=rngs_for_recurrent
                    )
                    
                    self.assertIsInstance(recurrent_output, MuZeroOutput)
                    print(f"    Recurrent output type: {type(recurrent_output)}")
                    chex.assert_shape(recurrent_output.hidden_state, (self.batch_size, self.num_agents, self.hidden_state_size))
                    print("    ✅ Recurrent inference shape test passed.")

                except Exception as e:
                    print(f"  ❌ Test Failed: {e.__class__.__name__}")
                    print(f"     Error: {e}")
                    self.fail(f"Full model inference test failed with exception: {e}")

    def test_edge_case_single_agent(self):
        """Tests the model with a single agent to check for dimension issues."""
        print("\n--- Testing Edge Case: Single Agent ---")
        num_agents = 1
        dummy_obs_single = jnp.zeros((self.batch_size, num_agents, self.obs_dim))
        
        model = FlaxMAMuZeroNet(config=self.config_no_attention, action_space_size=self.action_space_size)
        params = model.init(self.key, dummy_obs_single)['params']
        output = model.apply({'params': params}, dummy_obs_single)
        
        print(f"Input shape (single agent): {dummy_obs_single.shape}")
        print(f"Output hidden_state shape: {output.hidden_state.shape}")

        chex.assert_shape(output.hidden_state, (self.batch_size, num_agents, self.hidden_state_size))
        print("✅ Single agent edge case passed.")

    def test_permutation_equivariance(self):
        """Tests if the model correctly handles shuffled agent order."""
        print("\n--- Testing Permutation Equivariance ---")
        
        # Use a model with attention enabled
        model = FlaxMAMuZeroNet(config=self.config_with_attention, action_space_size=self.action_space_size)
        rngs = {'params': self.key, 'dropout': jax.random.PRNGKey(43)}
        params = model.init(rngs, self.dummy_obs)['params']

        # --- Get baseline output ---
        output_original = model.apply({'params': params}, self.dummy_obs, rngs=rngs)

        # --- Create a permuted input (swap agents 0 and 1) ---
        permuted_obs = self.dummy_obs.at[:, [0, 1], :].set(self.dummy_obs[:, [1, 0], :])
        
        # --- Get output from permuted input ---
        output_permuted = model.apply({'params': params}, permuted_obs, rngs=rngs)

        # --- Assert Properties ---
        # 1. The centralized value should be identical (or very close)
        chex.assert_trees_all_close(output_original.value_logits, output_permuted.value_logits, atol=1e-6)
        print("  ✅ Centralized value is invariant to agent permutation.")

        # 2. The policy for the swapped agents should also be swapped
        original_policy_swapped = output_original.policy_logits.at[:, [0, 1], :].set(
            output_original.policy_logits[:, [1, 0], :]
        )
        chex.assert_trees_all_close(original_policy_swapped, output_permuted.policy_logits, atol=1e-6)
        print("  ✅ Per-agent policies are equivariant to agent permutation.")

    def test_parameter_updates(self):
        """Tests if a single optimization step changes the model parameters."""
        print("\n--- Testing Parameter Updates ---")
        import optax

        model = FlaxMAMuZeroNet(config=self.config_no_attention, action_space_size=self.action_space_size)
        
        init_key, data_key = jax.random.split(self.key)
        dummy_obs_for_update_test = jax.random.normal(data_key, self.dummy_obs.shape)
        
        params = model.init(init_key, dummy_obs_for_update_test)['params']

        def loss_fn(p):
            # Use the random dummy observations here
            output = model.apply({'params': p}, dummy_obs_for_update_test)
            return jnp.mean(output.policy_logits**2)

        # A simple optimizer
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        try:
            # Calculate gradients and apply updates
            grads = jax.grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)

            # Assert that the parameters have actually changed
            diffs = jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(lambda p1, p2: jnp.sum(jnp.abs(p1 - p2)), params, new_params)
            )
            total_diff = jnp.sum(jnp.array(diffs))
            
            self.assertTrue(total_diff > 0)
            print("  ✅ A single optimization step successfully updated model parameters.")

        except Exception as e:
            print(f"  ❌ Test Failed: {e.__class__.__name__}")
            print(f"     Error: {e}")
            self.fail(f"Parameter update test failed with exception: {e}")   
        
if __name__ == '__main__':
    unittest.main()