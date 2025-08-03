# model/attention.py

import flax.linen as fnn
import jax
import jax.numpy as jnp
import chex
from typing import Sequence, Tuple, Optional

class MLP(fnn.Module):
    """A simple Multi-Layer Perceptron."""
    layer_sizes: Sequence[int]
    output_size: int

    @fnn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """
        Forward pass for the MLP.

        Args:
            x (chex.Array): The input array.

        Returns:
            chex.Array: The output array after passing through the MLP.
        """
        for size in self.layer_sizes:
            x = fnn.Dense(features=size)(x)
            x = fnn.LayerNorm()(x)
            x = fnn.relu(x)
        x = fnn.Dense(features=self.output_size)(x)
        return x
    

class BaseAttention(fnn.Module):
    """
    The base class for all attention mechanisms.
    Defines the interface that attention modules must follow.
    """
    hidden_size: int

    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        """
        All attention modules must implement this call signature.

        Args:
            x: The input tensor. Shape: (batch, num_agents, features)
            deterministic: A flag to control stochastic layers like dropout.

        Returns:
            An output tensor of shape (batch, num_agents, hidden_size).
        """
        raise NotImplementedError("Subclasses must implement the __call__ method.")


class TransformerEncoderLayer(fnn.Module):
    """A single layer of a Transformer encoder."""
    num_heads: int
    hidden_size: int
    dropout_rate: float

    @fnn.compact
    def __call__(self, x: chex.Array, *, deterministic: bool) -> chex.Array:
        """
        Forward pass for the Transformer encoder layer.

        Args:
            x (chex.Array): Input array. Shape: (batch, num_agents, hidden_size)

        Returns:
            chex.Array: Output array of the same shape as input.
        """
        # --- Self-attention block ---
        y = fnn.LayerNorm()(x)
        y = fnn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic
        )(y, y)
        x = x + fnn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)

        # --- Feed-forward block ---
        y = fnn.LayerNorm()(x)
        y = MLP(layer_sizes=(self.hidden_size * 2,), output_size=self.hidden_size)(y)
        x = x + fnn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        return x


class TransformerAttentionEncoder(BaseAttention):
    """An attention-based encoder to model agent interactions."""
    num_layers: int
    num_heads: int
    action_space_size: int
    dropout_rate: float = 0.1  

    @fnn.compact
    def __call__(self, hidden_states: chex.Array, actions: Optional[chex.Array] = None, *, deterministic: bool = False) -> chex.Array:
        """
        Forward pass for the attention encoder.

        Args:
            x (chex.Array): Input array. Shape: (batch, num_agents, features)

        Returns:
            chex.Array: Output array. Shape: (batch, num_agents, hidden_size)
        """
        batch_size, num_agents, _ = hidden_states.shape

        x = fnn.Dense(
            features=self.hidden_size, name="state_projector"
        )(hidden_states)

        if actions is not None:
            actions_onehot = jax.nn.one_hot(actions, num_classes=self.action_space_size)
            
            action_projection = fnn.Dense(
                features=self.hidden_size, name="action_projector"
            )(actions_onehot)
            
            x = x + action_projection

        agent_ids = jnp.arange(num_agents)
        agent_id_embeddings = fnn.Embed(num_embeddings=num_agents, features=self.hidden_size)(agent_ids)
        x = x + agent_id_embeddings[jnp.newaxis, ...]
        x = fnn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate  
            )(x, deterministic=deterministic) 
        return x
        