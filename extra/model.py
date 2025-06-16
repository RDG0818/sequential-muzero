import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def mlp(input_size: int, layer_sizes: List[int], output_size: int):
    """Creates a Multi-Layer Perceptron."""
    layers = []
    in_size = input_size
    for out_size in layer_sizes:
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.LayerNorm(out_size))
        layers.append(nn.ReLU())
        in_size = out_size
    layers.append(nn.Linear(in_size, output_size))
    return nn.Sequential(*layers)


class RepresentationNetwork(nn.Module):
    """Encodes a local observation into a latent state for a single agent."""
    def __init__(self, observation_size: int, hidden_state_size: int, fc_layers: List[int]):
        super().__init__()
        self.feature_norm = nn.LayerNorm(observation_size)
        self.net = mlp(observation_size, fc_layers, hidden_state_size)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized_obs = self.feature_norm(observation)
        return self.net(normalized_obs)


class DynamicsNetwork(nn.Module):
    """
    Predicts the next latent state and the joint reward.
    This is the core of the learned world model.
    """
    def __init__(self, num_agents: int, hidden_state_size: int, action_space_size: int, 
                 fc_dynamic_layers: List[int], fc_reward_layers: List[int]):
        super().__init__()
        # Input to dynamics model for one agent: current state + action
        dynamic_input_size = hidden_state_size + action_space_size
        self.dynamic_net = mlp(dynamic_input_size, fc_dynamic_layers, hidden_state_size)

        # Input to reward model: all agents' states and actions concatenated
        reward_input_size = num_agents * (hidden_state_size + action_space_size)
        self.reward_net = mlp(reward_input_size, fc_reward_layers, 1) # Predicts a single scalar reward

    def forward(self, hidden_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: [B, N, H], actions: [B, N, A] (one-hot)
        batch_size, num_agents, _ = hidden_states.shape
        
        # Keep original state for residual connection
        previous_states = hidden_states

        # --- Predict Next State ---
        # Reshape for batch processing: [B*N, H+A]
        dynamic_input = torch.cat([hidden_states, actions], dim=-1)
        flat_dynamic_input = dynamic_input.view(batch_size * num_agents, -1)
        
        # Get next latent state and add residual connection
        next_latent_states = self.dynamic_net(flat_dynamic_input).view(batch_size, num_agents, -1)
        next_latent_states += previous_states

        # --- Predict Reward ---
        # Use the *predicted* next state for reward prediction
        reward_input = torch.cat([next_latent_states, actions], dim=-1)
        flat_reward_input = reward_input.view(batch_size, -1)
        reward = self.reward_net(flat_reward_input)
        
        return next_latent_states, reward


class PredictionNetwork(nn.Module):
    """
    Predicts the policy for each agent and the centralized value.
    This is the CTDE prediction head.
    """
    def __init__(self, num_agents: int, hidden_state_size: int, action_space_size: int,
                 fc_value_layers: List[int], fc_policy_layers: List[int]):
        super().__init__()
        # --- Policy Heads (Actors) ---
        # One MLP applied to each agent's state individually.
        # This is parameter sharing.
        self.policy_net = mlp(hidden_state_size, fc_policy_layers, action_space_size)

        # --- Value Head (Centralized Critic) ---
        # Takes the concatenated states of all agents.
        value_input_size = num_agents * hidden_state_size
        self.value_net = mlp(value_input_size, fc_value_layers, 1) # Predicts a single scalar value

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: [B, N, H]
        batch_size, num_agents, _ = hidden_states.shape

        # Value prediction from the global state
        flat_hidden_states = hidden_states.view(batch_size, -1)
        value = self.value_net(flat_hidden_states)

        # Policy prediction for each agent from their local state
        # Reshape for batch processing: [B*N, H]
        flat_agent_states = hidden_states.view(batch_size * num_agents, -1)
        policy_logits = self.policy_net(flat_agent_states).view(batch_size, num_agents, -1)
        
        return policy_logits, value

# -----------------------------------------------------------------------------

class MAMuZeroToyNet(nn.Module):
    """
    A simplified, synchronous Multi-Agent MuZero-style network.
    This integrates all the components for a complete world model.
    """
    def __init__(self, num_agents: int, observation_size: int, action_space_size: int, 
                 hidden_state_size: int, fc_representation_layers: List[int], 
                 fc_dynamic_layers: List[int], fc_reward_layers: List[int], 
                 fc_value_layers: List[int], fc_policy_layers: List[int]):
        super().__init__()
        self.num_agents = num_agents
        self.action_space_size = action_space_size

        # Instantiating the network components with explicit arguments
        self.representation = RepresentationNetwork(
            observation_size, hidden_state_size, fc_layers=fc_representation_layers
        )
        self.dynamics = DynamicsNetwork(
            num_agents, hidden_state_size, action_space_size, 
            fc_dynamic_layers=fc_dynamic_layers, fc_reward_layers=fc_reward_layers
        )
        self.prediction = PredictionNetwork(
            num_agents, hidden_state_size, action_space_size,
            fc_value_layers=fc_value_layers, fc_policy_layers=fc_policy_layers
        )

    def initial_inference(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        From a batch of observations, compute the initial latent state, value, and policy.
        observations: [B, N, ObsSize]
        """
        # Encode observations into latent states
        # Reshape for batch processing: [B*N, ObsSize] -> [B, N, H]
        batch_size = observations.shape[0]
        flat_obs = observations.view(batch_size * self.num_agents, -1)
        hidden_states = self.representation(flat_obs).view(batch_size, self.num_agents, -1)
        
        # Predict policy and value from the initial state
        policy_logits, value = self.prediction(hidden_states)
        
        # Initial reward is always zero
        reward = torch.zeros_like(value)

        return hidden_states, reward, policy_logits, value

    def recurrent_inference(self, hidden_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        From a batch of latent states and actions, predict the next state, reward, value, and policy.
        hidden_states: [B, N, H]
        actions: [B, N] (indices)
        """
        # Convert action indices to one-hot vectors
        action_onehot = F.one_hot(actions, self.action_space_size).float()
        
        # Use the dynamics model to predict the next state and reward
        next_hidden_states, reward = self.dynamics(hidden_states, action_onehot)
        
        # Predict policy and value from the *next* state
        next_policy_logits, next_value = self.prediction(next_hidden_states)
        
        return next_hidden_states, reward, next_policy_logits, next_value