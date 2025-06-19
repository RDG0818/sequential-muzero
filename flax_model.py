# model.py

import flax.linen as fnn
import jax
import jax.numpy as jnp
from attention import MLP, AttentionEncoder
from typing import Tuple


class RepresentationNetwork(fnn.Module):
    """Encodes a local observation into a latent state for a single agent."""
    hidden_state_size: int
    fc_layers: Tuple[int, ...]

    @fnn.compact
    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        x = fnn.LayerNorm()(observation)
        x = MLP(layer_sizes=self.fc_layers, output_size=self.hidden_state_size)(x)
        return x


class DynamicsNetwork(fnn.Module):
    """
    Predicts the next latent state and the joint reward using attention.
    """
    num_agents: int
    hidden_state_size: int
    action_space_size: int
    reward_support_size: int
    fc_dynamic_layers: Tuple[int, ...]
    fc_reward_layers: Tuple[int, ...]

    # Attention-specific hyperparameters
    attention_layers: int = 3
    attention_heads: int = 4

    @fnn.compact
    def __call__(self, hidden_states: jnp.ndarray, actions_onehot: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch_size, num_agents, _ = hidden_states.shape
        previous_states = hidden_states

        attn_input = jnp.concatenate([hidden_states, actions_onehot], axis=-1)
        
        attn_projection = fnn.Dense(features=self.hidden_state_size)(attn_input)
        attn_projection = fnn.relu(attn_projection)
        
        agent_context = AttentionEncoder(
            num_layers=self.attention_layers,
            num_heads=self.attention_heads,
            hidden_size=self.hidden_state_size
        )(attn_projection)

        dynamic_input_with_context = jnp.concatenate([hidden_states, actions_onehot, agent_context], axis=-1)
        flat_dynamic_input = dynamic_input_with_context.reshape(batch_size * num_agents, -1)
        
        dynamic_net = MLP(layer_sizes=self.fc_dynamic_layers, output_size=self.hidden_state_size)
        next_latent_states = dynamic_net(flat_dynamic_input).reshape(batch_size, num_agents, -1)
        next_latent_states += previous_states # Residual connection

        reward_input = jnp.concatenate([next_latent_states, actions_onehot], axis=-1)
        flat_reward_input = reward_input.reshape(batch_size, -1)

        reward_output_size = self.reward_support_size * 2 + 1
        reward_net = MLP(layer_sizes=self.fc_reward_layers, output_size=reward_output_size)
        reward_logits = reward_net(flat_reward_input)
        
        return next_latent_states, reward_logits


class PredictionNetwork(fnn.Module):
    """Predicts the policy for each agent and the centralized value."""
    num_agents: int
    hidden_state_size: int
    action_space_size: int
    value_support_size: int
    fc_value_layers: Tuple[int, ...]
    fc_policy_layers: Tuple[int, ...]

    @fnn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch_size, num_agents, _ = hidden_states.shape

        flat_hidden_states = hidden_states.reshape(batch_size, -1)
        value_output_size = self.value_support_size * 2 + 1
        value_net = MLP(layer_sizes=self.fc_value_layers, output_size=value_output_size)
        value_logits = value_net(flat_hidden_states)

        flat_agent_states = hidden_states.reshape(batch_size * num_agents, -1)
        policy_net = MLP(layer_sizes=self.fc_policy_layers, output_size=self.action_space_size)
        policy_logits = policy_net(flat_agent_states).reshape(batch_size, num_agents, -1)
        
        return policy_logits, value_logits


class FlaxMAMuZeroNet(fnn.Module):
    """A pure Flax/JAX implementation of the simplified MuZero-style world model."""
    num_agents: int
    action_space_size: int
    hidden_state_size: int
    value_support_size: int
    reward_support_size: int
    fc_representation_layers: Tuple[int, ...]
    fc_dynamic_layers: Tuple[int, ...]
    fc_reward_layers: Tuple[int, ...]
    fc_value_layers: Tuple[int, ...]
    fc_policy_layers: Tuple[int, ...]

    def setup(self):
        """Create the sub-networks. This is called once by Flax."""
        self.representation_net = RepresentationNetwork(
            hidden_state_size=self.hidden_state_size,
            fc_layers=self.fc_representation_layers
        )
        self.dynamics_net = DynamicsNetwork(
            num_agents=self.num_agents,
            hidden_state_size=self.hidden_state_size,
            action_space_size=self.action_space_size,
            reward_support_size=self.reward_support_size,
            fc_dynamic_layers=self.fc_dynamic_layers,
            fc_reward_layers=self.fc_reward_layers
        )
        self.prediction_net = PredictionNetwork(
            num_agents=self.num_agents,
            hidden_state_size=self.hidden_state_size,
            action_space_size=self.action_space_size,
            value_support_size=self.value_support_size,
            fc_value_layers=self.fc_value_layers,
            fc_policy_layers=self.fc_policy_layers
        )

    def __call__(self, observations: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Performs the initial inference from a batch of observations."""
        batch_size = observations.shape[0]
        
        flat_obs = observations.reshape(batch_size * self.num_agents, -1)
        hidden_states = self.representation_net(flat_obs).reshape(batch_size, self.num_agents, -1)
        
        policy_logits, value = self.prediction_net(hidden_states)
        reward = jnp.zeros_like(value)

        if self.is_mutable_collection('params'): # Initialize params for dynamics
            dummy_actions = jnp.zeros((batch_size, self.num_agents), dtype=jnp.int32)
            self.recurrent_inference(hidden_states, dummy_actions)

        return hidden_states, reward, policy_logits, value

    def recurrent_inference(self, hidden_states: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Performs one step of dynamics and prediction from a latent state."""
        # Convert action indices to one-hot format using JAX
        actions_onehot = jax.nn.one_hot(actions, num_classes=self.action_space_size)
        
        next_hidden_states, reward = self.dynamics_net(hidden_states, actions_onehot)
        next_policy_logits, next_value = self.prediction_net(next_hidden_states)
        
        return next_hidden_states, reward, next_policy_logits, next_value
    
    def predict(self, hidden_states: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Runs the prediction network from a latent state."""
        policy_logits, value = self.prediction_net(hidden_states)
        return policy_logits, value