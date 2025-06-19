# model.py

import flax.linen as fnn
import jax
import jax.numpy as jnp
from typing import Tuple, Sequence

class MLP(fnn.Module):
    layer_sizes: Sequence[int]
    output_size: int

    @fnn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.bfloat16)
        for size in self.layer_sizes:
            x = fnn.Dense(features=size, dtype=jnp.bfloat16)(x)
            x = fnn.LayerNorm()(x)
            x = fnn.relu(x)
        x = fnn.Dense(features=self.output_size, dtype=jnp.float32)(x)
        return x


def sinusoidal_positional_encoding(seq_len: int, d_model: int):
    """Generates a sinusoidal positional encoding matrix."""
    position = jnp.arange(seq_len)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
    pos_enc = jnp.zeros((seq_len, d_model))
    pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(position * div_term))
    return pos_enc[jnp.newaxis, ...] # Add batch dimension

class TransformerEncoderLayer(fnn.Module):
    """A single layer of a Transformer encoder."""
    num_heads: int
    hidden_size: int
    dropout_rate: float = 0.0

    @fnn.compact
    def __call__(self, x, *, deterministic):
        # Self-attention block
        y = fnn.LayerNorm()(x)
        y = fnn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            deterministic=deterministic,
            dropout_rate=self.dropout_rate
        )(y, y)
        x = x + fnn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)

        # Feed-forward block
        y = fnn.LayerNorm()(x)
        y = MLP(layer_sizes=(self.hidden_size * 2,), output_size=self.hidden_size)(y)
        x = x + fnn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        return x

class AttentionEncoder(fnn.Module):
    """An attention-based encoder to model agent interactions."""
    num_layers: int
    num_heads: int
    hidden_size: int
    dropout_rate: float = 0.1

    @fnn.compact
    def __call__(self, x, *, deterministic=False):
        # x shape: (batch, num_agents, features)
        assert x.ndim == 3, f"Input must be 3-dimensional (batch, seq, features), but got {x.ndim}"
        
        # Add positional encodings to give agents a sense of order/identity
        pos_encoding = sinusoidal_positional_encoding(seq_len=x.shape[1], d_model=x.shape[2])
        x = x + pos_encoding

        # Transformer layers
        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate
            )(x, deterministic=deterministic)
        return x

