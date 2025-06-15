# model.py

import flax.linen as fnn
import jax
import jax.numpy as jnp
from typing import List, Tuple

class MLP(fnn.Module):
    layer_sizes: List[int]
    output_size: int

    @fnn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for size in self.layer_sizes:
            x = fnn.Dense(features=size)(x)
            x = fnn.LayerNorm()(x)
            x = fnn.relu(x)
        x = fnn.Dense(features=self.output_size)(x)
        return x


class RepresentationNetwork(fnn.Module):
    """Encodes a local observation into a latent state for a single agent."""
    hidden_state_size: int
    fc_layers: List[int]

    @fnn.compact
    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        x = fnn.LayerNorm()(observation)
        x = MLP(layer_sizes=self.fc_layers, output_size=self.hidden_state_size)(x)
        return x


class DynamicsNetwork(fnn.Module):
    """Predicts the next latent state and the joint reward."""
    num_agents: int
    hidden_state_size: int
    action_space_size: int
    fc_dynamic_layers: List[int]
    fc_reward_layers: List[int]
    
    @fnn.compact
    def __call__(self, hidden_states: jnp.ndarray, actions_onehot: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch_size, num_agents, _ = hidden_states.shape
        previous_states = hidden_states

        dynamic_input = jnp.concatenate([hidden_states, actions_onehot], axis=-1)
        flat_dynamic_input = dynamic_input.reshape(batch_size * num_agents, -1)
        
        dynamic_net = MLP(layer_sizes=self.fc_dynamic_layers, output_size=self.hidden_state_size)
        next_latent_states = dynamic_net(flat_dynamic_input).reshape(batch_size, num_agents, -1)
        next_latent_states += previous_states

        reward_input = jnp.concatenate([next_latent_states, actions_onehot], axis=-1)
        flat_reward_input = reward_input.reshape(batch_size, -1)

        reward_net = MLP(layer_sizes=self.fc_reward_layers, output_size=1)
        reward = reward_net(flat_reward_input)
        
        return next_latent_states, reward

class PredictionNetwork(fnn.Module):
    """Predicts the policy for each agent and the centralized value."""
    num_agents: int
    hidden_state_size: int
    action_space_size: int
    fc_value_layers: List[int]
    fc_policy_layers: List[int]

    @fnn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch_size, num_agents, _ = hidden_states.shape

        flat_hidden_states = hidden_states.reshape(batch_size, -1)
        value_net = MLP(layer_sizes=self.fc_value_layers, output_size=1)
        value = value_net(flat_hidden_states)

        flat_agent_states = hidden_states.reshape(batch_size * num_agents, -1)
        policy_net = MLP(layer_sizes=self.fc_policy_layers, output_size=self.action_space_size)
        policy_logits = policy_net(flat_agent_states).reshape(batch_size, num_agents, -1)
        
        return policy_logits, value


class FlaxMAMuZeroNet(fnn.Module):
    """A pure Flax/JAX implementation of the simplified MuZero-style world model."""
    num_agents: int
    action_space_size: int
    hidden_state_size: int
    fc_representation_layers: List[int]
    fc_dynamic_layers: List[int]
    fc_reward_layers: List[int]
    fc_value_layers: List[int]
    fc_policy_layers: List[int]

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
            fc_dynamic_layers=self.fc_dynamic_layers,
            fc_reward_layers=self.fc_reward_layers
        )
        self.prediction_net = PredictionNetwork(
            num_agents=self.num_agents,
            hidden_state_size=self.hidden_state_size,
            action_space_size=self.action_space_size,
            fc_value_layers=self.fc_value_layers,
            fc_policy_layers=self.fc_policy_layers
        )

    def __call__(self, observations: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Performs the initial inference from a batch of observations."""
        batch_size = observations.shape[0]
        
        # Reshape for per-agent processing
        flat_obs = observations.reshape(batch_size * self.num_agents, -1)
        hidden_states = self.representation_net(flat_obs).reshape(batch_size, self.num_agents, -1)
        
        policy_logits, value = self.prediction_net(hidden_states)
        
        # Initial reward is always zero
        reward = jnp.zeros_like(value)

        if self.is_mutable_collection('params'):
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