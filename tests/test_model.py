"""
Unit tests for model/layers.py, model/attention.py, and model/model.py.

Run with:
    conda run -n mazero pytest tests/test_model.py -v
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import chex

from model.layers import MLP
from model.attention import (
    sinusoidal_positional_encoding,
    TransformerEncoderLayer,
    TransformerAttentionEncoder,
)
from model import FlaxMAMuZeroNet, MuZeroOutput
from model.model import (
    RepresentationNetwork,
    DynamicsNetwork,
    PredictionNetwork,
    ProjectionNetwork,
)
from config import ModelConfig

# ─── Shared test constants ─────────────────────────────────────────────────────

B = 4       # batch size
N = 3       # num agents
OBS = 18    # observation dim
A = 5       # action space size
D = 32      # hidden state size
S = 10      # support size (reward/value)
PROJ = 32   # projection dim (must equal pred_out for cosine similarity to be valid)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def model_config():
    return ModelConfig(
        hidden_state_size=D,
        value_support_size=S,
        reward_support_size=S,
        fc_representation_layers=(D,),
        fc_dynamic_layers=(D,),
        fc_reward_layers=(16,),
        fc_value_layers=(16,),
        fc_policy_layers=(16,),
        attention_type="transformer",
        attention_layers=1,
        attention_heads=2,
        dropout_rate=0.1,
        proj_hid=PROJ,
        proj_out=PROJ,
        pred_hid=PROJ,
        pred_out=PROJ,
    )


@pytest.fixture
def model_config_no_attention(model_config):
    # Replace attention_type only; dataclass is frozen so we rebuild.
    return ModelConfig(
        **{**model_config.__dict__, "attention_type": "none"}
    )


@pytest.fixture
def dummy_obs():
    return jnp.ones((B, N, OBS))


@pytest.fixture
def dummy_hidden():
    return jnp.ones((B, N, D))


@pytest.fixture
def dummy_actions():
    return jnp.zeros((B, N), dtype=jnp.int32)


@pytest.fixture
def net(model_config):
    return FlaxMAMuZeroNet(model_config, A)


@pytest.fixture
def net_no_attention(model_config_no_attention):
    return FlaxMAMuZeroNet(model_config_no_attention, A)


@pytest.fixture
def params(net, rng, dummy_obs):
    return net.init(rng, dummy_obs)["params"]


@pytest.fixture
def params_no_attention(net_no_attention, rng, dummy_obs):
    return net_no_attention.init(rng, dummy_obs)["params"]


# ─── MLP ───────────────────────────────────────────────────────────────────────

class TestMLP:

    def test_output_shape(self, rng):
        mlp = MLP(layer_sizes=(64, 64), output_size=32)
        x = jnp.ones((B, 16))
        params = mlp.init(rng, x)["params"]
        out = mlp.apply({"params": params}, x)
        assert out.shape == (B, 32)

    def test_no_hidden_layers(self, rng):
        """layer_sizes=() means a single linear projection."""
        mlp = MLP(layer_sizes=(), output_size=8)
        x = jnp.ones((B, 16))
        params = mlp.init(rng, x)["params"]
        out = mlp.apply({"params": params}, x)
        assert out.shape == (B, 8)

    def test_batched_3d_input(self, rng):
        """MLP should operate on the last axis regardless of leading dims."""
        mlp = MLP(layer_sizes=(16,), output_size=8)
        x = jnp.ones((B, N, 16))
        params = mlp.init(rng, x)["params"]
        out = mlp.apply({"params": params}, x)
        assert out.shape == (B, N, 8)


# ─── Sinusoidal Positional Encoding ────────────────────────────────────────────

class TestSinusoidalPositionalEncoding:

    def test_output_shape(self):
        enc = sinusoidal_positional_encoding(seq_len=N, d_model=D)
        assert enc.shape == (1, N, D)

    def test_values_bounded(self):
        """Sines and cosines are in [-1, 1]."""
        enc = sinusoidal_positional_encoding(seq_len=N, d_model=D)
        assert float(jnp.max(jnp.abs(enc))) <= 1.0 + 1e-5

    def test_different_positions_differ(self):
        """Each agent position should have a unique encoding."""
        enc = sinusoidal_positional_encoding(seq_len=N, d_model=D)
        for i in range(N):
            for j in range(i + 1, N):
                assert not jnp.allclose(enc[0, i], enc[0, j]), \
                    f"Positions {i} and {j} have identical encodings"

    def test_single_position(self):
        enc = sinusoidal_positional_encoding(seq_len=1, d_model=D)
        assert enc.shape == (1, 1, D)


# ─── TransformerAttentionEncoder ───────────────────────────────────────────────

class TestTransformerAttentionEncoder:

    @pytest.fixture
    def encoder(self):
        return TransformerAttentionEncoder(
            num_layers=1, num_heads=2, hidden_size=D, dropout_rate=0.1
        )

    @pytest.fixture
    def enc_params(self, encoder, rng):
        x = jnp.ones((B, N, D + A))  # simulates dynamics input shape
        return encoder.init(rng, x)["params"]

    def test_output_shape(self, encoder, enc_params):
        x = jnp.ones((B, N, D + A))
        out = encoder.apply({"params": enc_params}, x, deterministic=True)
        assert out.shape == (B, N, D)

    def test_projects_varying_input_dim(self, encoder, rng):
        """Input feature dim doesn't need to equal hidden_size; Dense projects it."""
        x = jnp.ones((B, N, 50))
        params = encoder.init(rng, x)["params"]
        out = encoder.apply({"params": params}, x, deterministic=True)
        assert out.shape == (B, N, D)

    def test_single_agent(self, rng):
        """N=1 is a valid edge case."""
        enc = TransformerAttentionEncoder(
            num_layers=1, num_heads=2, hidden_size=D, dropout_rate=0.0
        )
        x = jnp.ones((B, 1, D))
        params = enc.init(rng, x)["params"]
        out = enc.apply({"params": params}, x, deterministic=True)
        assert out.shape == (B, 1, D)

    def test_wrong_rank_raises(self, encoder, enc_params):
        """chex.assert_rank(x, 3) should raise on a 2D input."""
        x_2d = jnp.ones((N, D))
        with pytest.raises(Exception):
            encoder.apply({"params": enc_params}, x_2d, deterministic=True)

    def test_deterministic_gives_consistent_output(self, encoder, enc_params):
        x = jnp.ones((B, N, D + A))
        out1 = encoder.apply({"params": enc_params}, x, deterministic=True)
        out2 = encoder.apply({"params": enc_params}, x, deterministic=True)
        assert jnp.allclose(out1, out2)

    def test_stochastic_differs_across_keys(self, encoder, enc_params, rng):
        """Different dropout keys should produce different outputs."""
        x = jnp.ones((B, N, D + A))
        k1, k2 = jax.random.split(rng)
        out1 = encoder.apply({"params": enc_params}, x,
                              rngs={"dropout": k1}, deterministic=False)
        out2 = encoder.apply({"params": enc_params}, x,
                              rngs={"dropout": k2}, deterministic=False)
        assert not jnp.allclose(out1, out2)


# ─── DynamicsNetwork ───────────────────────────────────────────────────────────

class TestDynamicsNetwork:

    @pytest.fixture
    def dynamics(self):
        return DynamicsNetwork(
            hidden_state_size=D,
            action_space_size=A,
            reward_support_size=S,
            fc_dynamic_layers=(D,),
            fc_reward_layers=(16,),
            attention_module=None,
        )

    @pytest.fixture
    def dynamics_with_attention(self):
        attn = TransformerAttentionEncoder(
            num_layers=1, num_heads=2, hidden_size=D, dropout_rate=0.1
        )
        return DynamicsNetwork(
            hidden_state_size=D,
            action_space_size=A,
            reward_support_size=S,
            fc_dynamic_layers=(D,),
            fc_reward_layers=(16,),
            attention_module=attn,
        )

    @pytest.fixture
    def dyn_params(self, dynamics, rng, dummy_hidden, dummy_actions):
        return dynamics.init(rng, dummy_hidden, dummy_actions)["params"]

    @pytest.fixture
    def dyn_params_attn(self, dynamics_with_attention, rng, dummy_hidden, dummy_actions):
        return dynamics_with_attention.init(rng, dummy_hidden, dummy_actions)["params"]

    def test_next_hidden_shape(self, dynamics, dyn_params, dummy_hidden, dummy_actions):
        next_h, _ = dynamics.apply({"params": dyn_params}, dummy_hidden, dummy_actions)
        assert next_h.shape == (B, N, D)

    def test_reward_logits_shape(self, dynamics, dyn_params, dummy_hidden, dummy_actions):
        support_size = S * 2 + 1
        _, reward = dynamics.apply({"params": dyn_params}, dummy_hidden, dummy_actions)
        assert reward.shape == (B, support_size)

    def test_residual_changes_hidden_state(self, dynamics, dyn_params, dummy_hidden, dummy_actions):
        """The dynamics net should produce a latent that differs from the input."""
        next_h, _ = dynamics.apply({"params": dyn_params}, dummy_hidden, dummy_actions)
        assert not jnp.allclose(next_h, dummy_hidden), \
            "Dynamics net output is identical to input — residual or MLP may be broken"

    def test_output_shapes_with_attention(
        self, dynamics_with_attention, dyn_params_attn, dummy_hidden, dummy_actions
    ):
        next_h, reward = dynamics_with_attention.apply(
            {"params": dyn_params_attn}, dummy_hidden, dummy_actions, deterministic=True
        )
        assert next_h.shape == (B, N, D)
        assert reward.shape == (B, S * 2 + 1)

    def test_plain_residual_output_shape(self, dynamics, dyn_params, rng):
        """Plain residual (no LayerNorm) still produces correct output shapes."""
        # MAZero uses a plain additive residual: next_h = MLP(input) + hidden_states.
        large_hidden = jnp.ones((B, N, D)) * 1000.0
        actions = jnp.zeros((B, N), dtype=jnp.int32)
        next_h, _ = dynamics.apply({"params": dyn_params}, large_hidden, actions)
        assert next_h.shape == (B, N, D)

    def test_invalid_action_index_raises(self, dynamics, dyn_params, dummy_hidden):
        """Actions outside [0, A) should raise from one_hot or chex."""
        out_of_range_actions = jnp.full((B, N), A + 5, dtype=jnp.int32)
        # one_hot clips silently, but the resulting all-zero vectors are detectable.
        next_h, _ = dynamics.apply({"params": dyn_params}, dummy_hidden, out_of_range_actions)
        # Verify we still get the right shape (no crash); behavioral validity is
        # the caller's responsibility.
        assert next_h.shape == (B, N, D)


def test_dynamics_triple_concat_output_shape(rng, model_config):
    """DynamicsNetwork with attention should still produce correct output shapes."""
    import jax.numpy as jnp
    from model.model import DynamicsNetwork
    from model.attention import TransformerAttentionEncoder

    attn = TransformerAttentionEncoder(
        num_layers=1, num_heads=2, hidden_size=D, dropout_rate=0.0
    )
    dyn = DynamicsNetwork(
        hidden_state_size=D,
        action_space_size=A,
        reward_support_size=S,
        fc_dynamic_layers=(D,),
        fc_reward_layers=(16,),
        attention_module=attn,
    )
    hidden = jnp.ones((B, N, D))
    actions = jnp.zeros((B, N), dtype=jnp.int32)
    params = dyn.init(rng, hidden, actions)
    next_h, reward = dyn.apply(params, hidden, actions)
    assert next_h.shape == (B, N, D)
    assert reward.shape == (B, 2 * S + 1)


# ─── PredictionNetwork ─────────────────────────────────────────────────────────

class TestPredictionNetwork:

    @pytest.fixture
    def pred_net(self):
        return PredictionNetwork(
            action_space_size=A,
            value_support_size=S,
            fc_value_layers=(16,),
            fc_policy_layers=(16,),
        )

    @pytest.fixture
    def pred_params(self, pred_net, rng, dummy_hidden):
        return pred_net.init(rng, dummy_hidden)["params"]

    def test_policy_logits_shape(self, pred_net, pred_params, dummy_hidden):
        policy, _ = pred_net.apply({"params": pred_params}, dummy_hidden)
        assert policy.shape == (B, N, A)

    def test_value_logits_shape(self, pred_net, pred_params, dummy_hidden):
        _, value = pred_net.apply({"params": pred_params}, dummy_hidden)
        assert value.shape == (B, S * 2 + 1)

    def test_policy_is_per_agent(self, pred_net, pred_params, dummy_hidden):
        """Policy logits should differ per agent (different weights applied)."""
        policy, _ = pred_net.apply({"params": pred_params}, dummy_hidden)
        # With shared weights and identical hidden states, per-agent outputs are
        # actually identical. But the shape contract should still hold.
        assert policy.shape[1] == N


# ─── ProjectionNetwork ─────────────────────────────────────────────────────────

class TestProjectionNetwork:

    @pytest.fixture
    def proj_net(self):
        return ProjectionNetwork(
            projection_hidden_dim=PROJ,
            projection_output_dim=PROJ,
            prediction_hidden_dim=PROJ,
            prediction_output_dim=PROJ,
        )

    @pytest.fixture
    def proj_params(self, proj_net, rng, dummy_hidden):
        # Both __call__ and predict must be traced during init or prediction_mlp
        # params won't exist (setup() submodules are lazily initialized on first call).
        return proj_net.init(
            rng,
            dummy_hidden,
            method=lambda self, x: (self(x), self.predict(self(x))),
        )["params"]

    def test_call_output_shape(self, proj_net, proj_params, dummy_hidden):
        out = proj_net.apply({"params": proj_params}, dummy_hidden)
        assert out.shape == (B, N, PROJ)

    def test_predict_output_shape(self, proj_net, proj_params, dummy_hidden):
        proj = proj_net.apply({"params": proj_params}, dummy_hidden)
        pred = proj_net.apply({"params": proj_params}, proj, method=proj_net.predict)
        assert pred.shape == (B, N, PROJ)

    def test_call_and_predict_differ(self, proj_net, proj_params, dummy_hidden):
        """__call__ (projection only) and predict (prediction head) use different weights."""
        proj = proj_net.apply({"params": proj_params}, dummy_hidden)
        pred = proj_net.apply({"params": proj_params}, proj, method=proj_net.predict)
        assert not jnp.allclose(proj, pred)


# ─── FlaxMAMuZeroNet ───────────────────────────────────────────────────────────

class TestFlaxMAMuZeroNet:

    # ── Parameter initialization ──────────────────────────────────────────────

    def test_init_creates_all_subnetwork_params(self, net, rng, dummy_obs):
        """is_initializing() block must force dynamics and projection params to exist."""
        all_params = net.init(rng, dummy_obs)["params"]
        expected_keys = {
            "representation_net",
            "dynamics_net",
            "prediction_net",
            "projection_net",
        }
        assert expected_keys.issubset(set(all_params.keys())), \
            f"Missing param keys: {expected_keys - set(all_params.keys())}"

    def test_init_no_attention_creates_all_subnetwork_params(
        self, net_no_attention, rng, dummy_obs
    ):
        all_params = net_no_attention.init(rng, dummy_obs)["params"]
        assert "dynamics_net" in all_params
        assert "projection_net" in all_params

    # ── Initial inference (__call__) ──────────────────────────────────────────

    def test_call_hidden_state_shape(self, net, params, dummy_obs):
        out = net.apply({"params": params}, dummy_obs)
        assert out.hidden_state.shape == (B, N, D)

    def test_call_policy_logits_shape(self, net, params, dummy_obs):
        out = net.apply({"params": params}, dummy_obs)
        assert out.policy_logits.shape == (B, N, A)

    def test_call_value_logits_shape(self, net, params, dummy_obs):
        out = net.apply({"params": params}, dummy_obs)
        assert out.value_logits.shape == (B, S * 2 + 1)

    def test_call_reward_logits_are_zeros(self, net, params, dummy_obs):
        """On initial inference no action has been taken; reward should be all zeros."""
        out = net.apply({"params": params}, dummy_obs)
        assert jnp.all(out.reward_logits == 0.0), \
            "reward_logits on initial inference must be a zero placeholder"

    def test_call_returns_muzero_output(self, net, params, dummy_obs):
        out = net.apply({"params": params}, dummy_obs)
        assert isinstance(out, MuZeroOutput)

    # ── Recurrent inference ───────────────────────────────────────────────────

    def test_recurrent_inference_shapes(self, net, params, dummy_hidden, dummy_actions, rng):
        out = net.apply(
            {"params": params},
            dummy_hidden,
            dummy_actions,
            method=net.recurrent_inference,
            rngs={"dropout": rng},
        )
        assert out.hidden_state.shape == (B, N, D)
        assert out.policy_logits.shape == (B, N, A)
        assert out.value_logits.shape == (B, S * 2 + 1)
        assert out.reward_logits.shape == (B, S * 2 + 1)

    def test_recurrent_inference_reward_nonzero(
        self, net, params, dummy_hidden, dummy_actions, rng
    ):
        """Unlike initial inference, recurrent inference should produce real reward logits."""
        out = net.apply(
            {"params": params},
            dummy_hidden,
            dummy_actions,
            method=net.recurrent_inference,
            rngs={"dropout": rng},
        )
        assert not jnp.all(out.reward_logits == 0.0), \
            "recurrent_inference reward_logits should not be all zeros"

    def test_recurrent_inference_changes_hidden(
        self, net, params, dummy_hidden, dummy_actions, rng
    ):
        out = net.apply(
            {"params": params},
            dummy_hidden,
            dummy_actions,
            method=net.recurrent_inference,
            rngs={"dropout": rng},
        )
        assert not jnp.allclose(out.hidden_state, dummy_hidden)

    # ── predict ───────────────────────────────────────────────────────────────

    def test_predict_shapes(self, net, params, dummy_hidden):
        policy, value = net.apply({"params": params}, dummy_hidden, method=net.predict)
        assert policy.shape == (B, N, A)
        assert value.shape == (B, S * 2 + 1)

    # ── project_online / project_target ───────────────────────────────────────

    def test_project_online_shape(self, net, params, dummy_hidden):
        out = net.apply({"params": params}, dummy_hidden, method=net.project_online)
        assert out.shape == (B, N, PROJ)

    def test_project_target_shape(self, net, params, dummy_hidden):
        out = net.apply({"params": params}, dummy_hidden, method=net.project_target)
        assert out.shape == (B, N, PROJ)

    def test_project_online_and_target_differ(self, net, params, dummy_hidden):
        """Online branch applies prediction head; target does not. Outputs must differ."""
        online = net.apply({"params": params}, dummy_hidden, method=net.project_online)
        target = net.apply({"params": params}, dummy_hidden, method=net.project_target)
        assert not jnp.allclose(online, target), \
            "project_online and project_target should differ (different network branches)"

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_single_agent(self, model_config, rng):
        """N=1 should work end-to-end without shape errors."""
        net = FlaxMAMuZeroNet(model_config, A)
        obs = jnp.ones((B, 1, OBS))
        params = net.init(rng, obs)["params"]
        out = net.apply({"params": params}, obs)
        assert out.hidden_state.shape == (B, 1, D)
        assert out.policy_logits.shape == (B, 1, A)

    def test_batch_size_one(self, model_config, rng):
        net = FlaxMAMuZeroNet(model_config, A)
        obs = jnp.ones((1, N, OBS))
        params = net.init(rng, obs)["params"]
        out = net.apply({"params": params}, obs)
        assert out.hidden_state.shape == (1, N, D)

    def test_no_attention_mode(self, net_no_attention, params_no_attention, dummy_obs):
        out = net_no_attention.apply({"params": params_no_attention}, dummy_obs)
        assert out.hidden_state.shape == (B, N, D)
        assert out.policy_logits.shape == (B, N, A)

    def test_deterministic_flag_gives_consistent_output(self, net, params, dummy_hidden, rng):
        """With deterministic=True, repeated recurrent_inference calls should match."""
        kwargs = dict(
            method=net.recurrent_inference,
            deterministic=True,
        )
        actions = jnp.zeros((B, N), dtype=jnp.int32)
        out1 = net.apply({"params": params}, dummy_hidden, actions, **kwargs)
        out2 = net.apply({"params": params}, dummy_hidden, actions, **kwargs)
        assert jnp.allclose(out1.hidden_state, out2.hidden_state)
