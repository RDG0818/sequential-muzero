# model/model.py

import flax.linen as nn
import jax
import jax.numpy as jnp
import chex
from model.layers import MLP
from model.attention import TransformerAttentionEncoder
from typing import NamedTuple
from config import ModelConfig


class MuZeroOutput(NamedTuple):
    """Output container for both initial and recurrent inference.

    Shape conventions (B=batch, N=agents, D=hidden, A=actions, S=support size):
        hidden_state:   (B, N, D)  — per-agent latent states
        reward_logits:  (B, S)     — centralized joint reward distribution;
                                     zeros on initial inference (no action taken yet)
        policy_logits:  (B, N, A)  — per-agent action distributions
        value_logits:   (B, S)     — centralized state value distribution

    Reward and value are centralized (shared across agents), while policy is
    decentralized (one head per agent). This reflects the CTDE design.
    """
    hidden_state: chex.Array
    reward_logits: chex.Array
    policy_logits: chex.Array
    value_logits: chex.Array


class RepresentationNetwork(nn.Module):
    """Encodes a local observation into a latent state for a single agent."""
    hidden_state_size: int
    fc_layers: tuple[int, ...]

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """
        Args:
            observation: Local observation for an agent. Shape: (B, obs_dim)

        Returns:
            Encoded latent state. Shape: (B, hidden_state_size)
        """
        x = nn.LayerNorm()(observation)
        x = MLP(layer_sizes=self.fc_layers, output_size=self.hidden_state_size)(x)
        return x


class DynamicsNetwork(nn.Module):
    """Predicts the next latent state and the centralized joint reward."""
    hidden_state_size: int
    action_space_size: int
    reward_support_size: int
    fc_dynamic_layers: tuple[int, ...]
    fc_reward_layers: tuple[int, ...]
    # Optional transformer for inter-agent communication during dynamics.
    attention_module: TransformerAttentionEncoder | None = None

    @nn.compact
    def __call__(
        self,
        hidden_states: chex.Array,
        actions: chex.Array,
        deterministic: bool = False,
    ) -> tuple[chex.Array, chex.Array]:
        """
        Args:
            hidden_states: Latent states. Shape: (B, N, D_hidden)
            actions: Joint actions taken. Shape: (B, N)
            deterministic: If True, disables dropout in the attention module.

        Returns:
            next_latent_states: Shape: (B, N, D_hidden)
            reward_logits:      Shape: (B, reward_support_size * 2 + 1)
        """
        batch_size, num_agents, _ = hidden_states.shape
        actions_onehot = jax.nn.one_hot(actions, num_classes=self.action_space_size)
        chex.assert_shape(actions_onehot, (batch_size, num_agents, None))

        ha_concat = jnp.concatenate([hidden_states, actions_onehot], axis=-1)  # (B, N, D+A)

        if self.attention_module is not None:
            attn_out = self.attention_module(ha_concat, deterministic=deterministic)  # (B, N, D)
            dynamic_input = jnp.concatenate(
                [hidden_states, actions_onehot, attn_out], axis=-1
            )  # (B, N, 2D+A)
        else:
            dynamic_input = ha_concat  # (B, N, D+A)

        flat_dynamic_input = dynamic_input.reshape(batch_size * num_agents, -1)
        dynamic_net = MLP(layer_sizes=self.fc_dynamic_layers, output_size=self.hidden_state_size)
        next_latent_states = dynamic_net(flat_dynamic_input).reshape(batch_size, num_agents, -1)

        # Plain residual without LayerNorm — matches MAZero's state += hidden_state.
        next_latent_states = next_latent_states + hidden_states

        # Centralized reward: all agent (state, action) pairs → single joint reward.
        reward_input = jnp.concatenate([next_latent_states, actions_onehot], axis=-1)
        flat_reward_input = reward_input.reshape(batch_size, -1)
        reward_output_size = self.reward_support_size * 2 + 1
        reward_net = MLP(layer_sizes=self.fc_reward_layers, output_size=reward_output_size)
        reward_logits = reward_net(flat_reward_input)

        return next_latent_states, reward_logits


class PredictionNetwork(nn.Module):
    """Predicts the per-agent policy and the centralized value from a latent state."""
    action_space_size: int
    value_support_size: int
    fc_value_layers: tuple[int, ...]
    fc_policy_layers: tuple[int, ...]

    @nn.compact
    def __call__(self, hidden_states: chex.Array) -> tuple[chex.Array, chex.Array]:
        """
        Args:
            hidden_states: Latent states. Shape: (B, N, D_hidden)

        Returns:
            policy_logits: Shape: (B, N, A)
            value_logits:  Shape: (B, value_support_size * 2 + 1)
        """
        batch_size, num_agents, _ = hidden_states.shape

        # Value is centralized: concatenate all agent states before the MLP.
        flat_hidden_states = hidden_states.reshape(batch_size, -1)
        value_output_size = self.value_support_size * 2 + 1
        value_net = MLP(layer_sizes=self.fc_value_layers, output_size=value_output_size)
        value_logits = value_net(flat_hidden_states)

        # Policy is decentralized: each agent gets its own head, run in parallel via reshape.
        flat_agent_states = hidden_states.reshape(batch_size * num_agents, -1)
        policy_net = MLP(layer_sizes=self.fc_policy_layers, output_size=self.action_space_size)
        policy_logits = policy_net(flat_agent_states).reshape(batch_size, num_agents, -1)

        return policy_logits, value_logits


class ProjectionNetwork(nn.Module):
    """
    Self-supervised consistency head, inspired by SimSiam.

    Contains two sub-networks:
    1. Projection MLP: maps a hidden state to a fixed-size projection space.
    2. Prediction MLP: applied only on the online branch to predict the target projection.

    See project_online / project_target on FlaxMAMuZeroNet for how these are used.
    """
    projection_hidden_dim: int
    projection_output_dim: int
    prediction_hidden_dim: int
    prediction_output_dim: int

    def setup(self):
        self.projection_mlp = MLP(
            layer_sizes=[self.projection_hidden_dim],
            output_size=self.projection_output_dim,
        )
        self.projection_norm = nn.LayerNorm()
        self.prediction_mlp = MLP(
            layer_sizes=[self.prediction_hidden_dim],
            output_size=self.prediction_output_dim,
        )

    def __call__(self, x: chex.Array) -> chex.Array:
        """Projects a hidden state into the projection space."""
        return self.projection_norm(self.projection_mlp(x))

    def predict(self, proj: chex.Array) -> chex.Array:
        """Applies the prediction head to a projection (online branch only)."""
        return self.prediction_mlp(proj)


class FlaxMAMuZeroNet(nn.Module):
    """
    Multi-agent MuZero world model.

    Exposes three callable paths:
      __call__            — initial inference (observation → latent + predictions)
      recurrent_inference — dynamics unroll (latent + action → next latent + predictions)
      predict             — prediction head only, no dynamics

    And two projection methods for the EMA consistency loss:
      project_online  — projection + prediction head (current/online latent)
      project_target  — projection only, no head (target latent, applied with EMA params)
    """
    config: ModelConfig
    action_space_size: int

    def setup(self):
        attention_module = None
        if self.config.attention_type == "transformer":
            attention_module = TransformerAttentionEncoder(
                num_layers=self.config.attention_layers,
                num_heads=self.config.attention_heads,
                hidden_size=self.config.hidden_state_size,
                dropout_rate=self.config.dropout_rate,
            )
        self.representation_net = RepresentationNetwork(
            hidden_state_size=self.config.hidden_state_size,
            fc_layers=self.config.fc_representation_layers,
        )
        self.dynamics_net = DynamicsNetwork(
            hidden_state_size=self.config.hidden_state_size,
            action_space_size=self.action_space_size,
            reward_support_size=self.config.reward_support_size,
            fc_dynamic_layers=self.config.fc_dynamic_layers,
            fc_reward_layers=self.config.fc_reward_layers,
            attention_module=attention_module,
        )
        self.prediction_net = PredictionNetwork(
            action_space_size=self.action_space_size,
            value_support_size=self.config.value_support_size,
            fc_value_layers=self.config.fc_value_layers,
            fc_policy_layers=self.config.fc_policy_layers,
        )
        self.projection_net = ProjectionNetwork(
            projection_hidden_dim=self.config.proj_hid,
            projection_output_dim=self.config.proj_out,
            prediction_hidden_dim=self.config.pred_hid,
            prediction_output_dim=self.config.pred_out,
        )

    def __call__(self, observations: chex.Array) -> MuZeroOutput:
        """
        Initial inference: encodes observations and produces predictions.
        reward_logits is a zeros placeholder since no action has been taken yet.

        Args:
            observations: Batch of observations. Shape: (B, N, obs_dim)

        Returns:
            MuZeroOutput with hidden_state, reward_logits (zeros), policy_logits, value_logits.
        """
        batch_size, num_agents, _ = observations.shape

        flat_obs = observations.reshape(batch_size * num_agents, -1)
        hidden_states = self.representation_net(flat_obs).reshape(batch_size, num_agents, -1)

        policy_logits, value_logits = self.prediction_net(hidden_states)
        reward_logits = jnp.zeros((batch_size, self.config.reward_support_size * 2 + 1))

        # During model.init(), only __call__ is traced, so subnetworks used exclusively
        # by other methods (dynamics, projection) need a dummy forward pass here to
        # ensure their parameters are included in the returned param tree.
        if self.is_initializing():
            dummy_actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)
            self.dynamics_net(hidden_states, dummy_actions, deterministic=True)
            self.project_online(hidden_states)
            self.project_target(hidden_states)

        return MuZeroOutput(
            hidden_state=hidden_states,
            reward_logits=reward_logits,
            policy_logits=policy_logits,
            value_logits=value_logits,
        )

    def recurrent_inference(
        self,
        hidden_states: chex.Array,
        actions: chex.Array,
        deterministic: bool = False,
    ) -> MuZeroOutput:
        """
        Recurrent inference: rolls the world model forward one step.

        Args:
            hidden_states: Current latent states. Shape: (B, N, D_hidden)
            actions: Joint actions taken. Shape: (B, N)
            deterministic: If True, disables dropout (use during eval/MCTS).

        Returns:
            MuZeroOutput with the next hidden_state, reward_logits, policy_logits, value_logits.
        """
        next_hidden_states, reward_logits = self.dynamics_net(
            hidden_states, actions, deterministic=deterministic
        )
        policy_logits, value_logits = self.prediction_net(next_hidden_states)

        return MuZeroOutput(
            hidden_state=next_hidden_states,
            reward_logits=reward_logits,
            policy_logits=policy_logits,
            value_logits=value_logits,
        )

    def predict(self, hidden_states: chex.Array) -> tuple[chex.Array, chex.Array]:
        """
        Runs only the prediction head (no dynamics).

        Args:
            hidden_states: Latent states. Shape: (B, N, D_hidden)

        Returns:
            policy_logits: Shape: (B, N, A)
            value_logits:  Shape: (B, S)
        """
        return self.prediction_net(hidden_states)

    def project_online(self, hidden_state: chex.Array) -> chex.Array:
        """
        Online projection branch (projection + prediction head).
        Applied to the current latent state in the SimSiam consistency loss.
        """
        proj = self.projection_net(hidden_state)
        return self.projection_net.predict(proj)

    def project_target(self, hidden_state: chex.Array) -> chex.Array:
        """
        Target projection branch (projection only, no prediction head).
        Applied to the next latent state using EMA parameters — called as
        model.apply({"params": ema_params}, ...) so no stop_gradient is needed.
        """
        return self.projection_net(hidden_state)
