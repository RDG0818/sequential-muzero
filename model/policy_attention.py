import flax.linen as fnn
import jax
import jax.numpy as jnp
import chex
from model.attention import MLP, TransformerAttentionEncoder
from typing import Tuple, Optional

class CausalSelfAttention(fnn.Module):
    """
    A self-attention mechanism with a causal mask to ensure autoregressive
    behavior. It prevents positions from attending to subsequent positions.
    """
    num_heads: int
    hidden_size: int
    dropout_rate: float = 0.0

    @fnn.compact
    def __call__(self, x: chex.Array, *, padding_mask: Optional[chex.Array] = None, deterministic: bool) -> chex.Array:
        """
        Forward pass for the causal self-attention layer.

        Args:
            x (chex.Array): Input array of shape (batch, sequence_len, hidden_size).
            deterministic (bool): A flag to control dropout.

        Returns:
            chex.Array: Output array of the same shape as input.
        """
        mask = fnn.make_causal_mask(x[:, :, 0])

        if padding_mask is not None:
            mask = fnn.combine_masks(mask, padding_mask[:, jnp.newaxis, jnp.newaxis, :])

        attention_output = fnn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            name="CausalMultiHeadAttention"
        )(inputs_q=x, inputs_k=x, inputs_v=x, mask=mask)

        return attention_output


class DecoderBlock(fnn.Module):
    """
    A single Transformer Decoder Block.

    This block performs three main operations:
    1. Masked Self-Attention: On the target sequence (e.g., action embeddings).
    2. Cross-Attention: Between the target sequence and the context from the encoder.
    3. Feed-Forward Network: A final MLP to process the combined information.
    """
    num_heads: int
    hidden_size: int
    dropout_rate: float = 0.0

    @fnn.compact
    def __call__(self, x: chex.Array, context: chex.Array, *, padding_mask: Optional[chex.Array] = None, deterministic: bool) -> chex.Array:
        """
        Forward pass for the decoder block.

        Args:
            x (chex.Array): The target sequence (e.g., embeddings of previously
                          generated actions). Shape: (batch, target_seq_len, hidden_size).
            context (chex.Array): The context from the encoder (from agent observations).
                                 Shape: (batch, source_seq_len, hidden_size).
            deterministic (bool): A flag for controlling dropout.

        Returns:
            chex.Array: The output of the decoder block, with the same shape as x.
        """
        # 1. Masked Self-Attention (on the target sequence `x`)
        # This allows each position in `x` to attend to previous positions in `x`.
        attn_output = CausalSelfAttention(
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            dropout_rate=self.dropout_rate
        )(x, padding_mask=padding_mask, deterministic=deterministic)
        x = fnn.LayerNorm()(x + attn_output)

        # 2. Cross-Attention (between `x` and `context`)
        # This allows each position in the target sequence `x` to attend to all
        # positions in the encoder's output `context`.
        cross_attn_output = fnn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            name="CrossMultiHeadAttention"
        )(inputs_q=x, inputs_k=context, inputs_v=context)
        x = fnn.LayerNorm()(x + cross_attn_output)

        # 3. Feed-Forward Network
        ffn_output = MLP(
            layer_sizes=(self.hidden_size * 4,), # Standard practice to expand dimension
            output_size=self.hidden_size
        )(x)
        
        # Final residual connection and layer norm
        x = fnn.LayerNorm()(x + ffn_output)

        return x


class AutoregressivePredictionNetwork(fnn.Module):
    """
    Predicts policy and value using an autoregressive decoder for the policy.
    The value prediction is centralized and non-autoregressive.
    """
    encoder: TransformerAttentionEncoder
    action_space_size: int
    value_support_size: int
    num_decoder_blocks: int
    num_heads: int
    hidden_state_size: int
    fc_value_layers: Tuple[int, ...]
    dropout_rate: float = 0.0

    def setup(self):
        """Define the sub-modules used in this network."""
        value_output_size = self.value_support_size * 2 + 1
        self.value_head = MLP(
            layer_sizes=self.fc_value_layers, 
            output_size=value_output_size,
            name="value_head"
        )
        
        self.decoder_blocks = [
            DecoderBlock(
                num_heads=self.num_heads,
                hidden_size=self.hidden_state_size,
                dropout_rate=self.dropout_rate
            ) for _ in range(self.num_decoder_blocks)
        ]

        self.policy_head = fnn.Dense(
            features=self.action_space_size, 
            name="policy_head"
        )

        self.action_embedder = fnn.Embed(
            num_embeddings=self.action_space_size, 
            features=self.hidden_state_size,
            name="action_embedder"
        )

        self.start_token = self.param(
            'start_token',
            fnn.initializers.lecun_normal(),
            (1, 1, self.hidden_state_size)
        )

    @fnn.compact
    def __call__(self, hidden_states: chex.Array, actions: chex.Array, *, train: bool) -> Tuple[chex.Array, chex.Array]:
        """
        Performs the forward pass for training using teacher forcing.
        """
        batch_size, num_agents, _ = hidden_states.shape

        context = self.encoder(hidden_states, actions=None, deterministic=not train)
        flat_context = context.reshape(batch_size, -1)
        value_logits = self.value_head(flat_context)
        
        action_embeddings = self.action_embedder(actions)
        
        start_tokens = jnp.tile(self.start_token, (batch_size, 1, 1))
        
        decoder_input = jnp.concatenate([start_tokens, action_embeddings[:, :-1, :]], axis=1)

        y = decoder_input
        for block in self.decoder_blocks:
            y = block(y, context, deterministic=not train)
        
        policy_logits = self.policy_head(y)

        return policy_logits, value_logits

    
    @fnn.compact
    def generate(self, hidden_states: chex.Array, *, key: chex.PRNGKey, deterministic: bool = True) -> chex.Array:
        """
        Generates a full sequence of actions autoregressively using masking.
        """
        batch_size, num_agents, _ = hidden_states.shape

        # 1. Get context from the encoder
        context = self.encoder(hidden_states, actions=None, deterministic=deterministic)
        flat_context = context.reshape(batch_size, -1)
        value_logits = self.value_head(flat_context)

        # 2. Prepare the initial state for the scan loop
        all_action_embeddings = jnp.zeros(
            (batch_size, num_agents + 1, self.hidden_state_size)
        )
        
        all_action_embeddings = all_action_embeddings.at[:, 0:1, :].set(self.start_token)

        initial_carry = (all_action_embeddings, 1, key) # Start writing at index 1

        def scan_fn(carry, _):
            current_container, index, key = carry

            key, sample_key = jax.random.split(key)
            
            padding_mask = jnp.arange(num_agents + 1) < index
            padding_mask = jnp.tile(padding_mask[jnp.newaxis, :], (batch_size, 1))
            
            y = current_container
            for block in self.decoder_blocks:
                y = block(y, context, padding_mask=padding_mask, deterministic=True)

            last_valid_output = y[:, index - 1, :]

            next_action_logits = self.policy_head(last_valid_output)

            next_action = jax.lax.cond(
                deterministic,
                lambda: jnp.argmax(next_action_logits, axis=-1),
                lambda: jax.random.categorical(sample_key, next_action_logits)
            )

            next_action_embedded = self.action_embedder(next_action)

            updated_container = jax.lax.dynamic_update_slice(
                current_container,
                next_action_embedded[:, jnp.newaxis, :],
                (0, index, 0)
            )

            new_carry = (updated_container, index + 1, key)
            return new_carry, next_action_logits

        # 4. Run the scan loop
        final_carry, collected_logits = jax.lax.scan(
            scan_fn, initial_carry, None, length=num_agents
        )

        policy_logits = collected_logits.transpose(1, 0, 2)
        
        return policy_logits, value_logits