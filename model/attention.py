# model/attention.py

import flax.linen as fnn
import jax
import jax.numpy as jnp
import chex
from typing import Sequence, Tuple


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
    

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> chex.Array:
    """
    Generates a sinusoidal positional encoding matrix.

    Args:
        seq_len (int): The length of the sequence (e.g., number of agents).
        d_model (int): The dimensionality of the model/embedding.

    Returns:
        chex.Array: A (1, seq_len, d_model) positional encoding matrix.
    """
    position = jnp.arange(seq_len)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
    pos_enc = jnp.zeros((seq_len, d_model))
    pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(position * div_term))
    return pos_enc[jnp.newaxis, ...]  # Add batch dimension


class TransformerEncoderLayer(fnn.Module):
    """A single layer of a Transformer encoder."""
    num_heads: int
    hidden_size: int

    @fnn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
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
            qkv_features=self.hidden_size
        )(y, y)
        x = x + y

        # --- Feed-forward block ---
        y = fnn.LayerNorm()(x)
        y = MLP(layer_sizes=(self.hidden_size * 2,), output_size=self.hidden_size)(y)
        x = x + y
        return x


class AttentionEncoder(fnn.Module):
    """An attention-based encoder to model agent interactions."""
    num_layers: int
    num_heads: int
    hidden_size: int

    @fnn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """
        Forward pass for the attention encoder.

        Args:
            x (chex.Array): Input array. Shape: (batch, num_agents, features)

        Returns:
            chex.Array: Output array. Shape: (batch, num_agents, hidden_size)
        """
        chex.assert_rank(x, 3)  # (batch, num_agents, features)

        x = fnn.Dense(features=self.hidden_size)(x)

        # --- Add positional encodings for agent identity ---
        pos_encoding = sinusoidal_positional_encoding(seq_len=x.shape[1], d_model=x.shape[2])
        x = x + pos_encoding

        # --- Transformer layers ---
        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(
                num_heads=self.num_heads,
                hidden_size=self.hidden_size
            )(x)
        return x