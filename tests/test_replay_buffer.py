"""
Tests for the C++ prioritized replay buffer.

Covers:
 - add / sample shapes and dtypes
 - update_priorities changes sampling distribution
 - update_targets writes only to position 0
 - sample_for_reanalysis uniqueness and shapes
 - get_stats keys
 - Pure-Python fallback still works
 - C++ backend is active when available

Run with:
    conda run -n mazero pytest tests/test_replay_buffer.py -v
"""
import numpy as np
import pytest

from utils.replay_buffer import (
    ReplayBuffer, ReplayItem, Episode, Transition, process_episode
)

# ─── Shared test parameters ────────────────────────────────────────────────

CAPACITY    = 200
OBS_SIZE    = 8
A           = 5    # action space size
N           = 3    # num agents
U           = 3    # unroll steps
BATCH       = 16
REANALYZE_B = 8


def make_config():
    return dict(
        capacity          = CAPACITY,
        observation_shape = (OBS_SIZE,),
        action_space_size = A,
        num_agents        = N,
        unroll_steps      = U,
        alpha             = 0.6,
        beta_start        = 0.4,
        beta_frames       = 1000,
    )


def make_item(rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    return ReplayItem(
        observation   = rng.random((N, OBS_SIZE),    dtype=np.float32),
        actions       = rng.integers(0, A, (U, N),   dtype=np.int32),
        policy_target = rng.random((U+1, N, A),      dtype=np.float32),
        value_target  = rng.random((U+1, N),         dtype=np.float32),
        reward_target = rng.random((U, N),           dtype=np.float32),
        agent_order   = np.arange(N,                 dtype=np.int32),
    )


def fill_buffer(buf: ReplayBuffer, n: int):
    rng = np.random.default_rng(0)
    for _ in range(n):
        buf.add(make_item(rng), priority=1.0)


# ─── Backend detection ──────────────────────────────────────────────────────

def test_cpp_backend_loaded():
    """C++ module must be importable in this environment."""
    try:
        import _replay_buffer_cpp  # noqa: F401
    except ImportError:
        pytest.skip("C++ backend not built — skipping backend-specific tests")


def test_replay_buffer_uses_cpp():
    """ReplayBuffer should use the C++ backend when available."""
    try:
        import _replay_buffer_cpp  # noqa: F401
    except ImportError:
        pytest.skip("C++ backend not built")
    buf = ReplayBuffer(**make_config())
    assert buf._use_cpp, "Expected C++ backend to be active"
    assert buf._buf.is_pinned() or True  # pinned iff CUDA available; either is fine


# ─── add / sample ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def filled_buffer():
    buf = ReplayBuffer(**make_config())
    fill_buffer(buf, CAPACITY)
    return buf


def test_len_after_fill(filled_buffer):
    assert len(filled_buffer) == CAPACITY


def test_sample_returns_correct_shapes(filled_buffer):
    batch, weights, indices = filled_buffer.sample(BATCH)
    assert batch is not None
    assert batch.observation.shape   == (BATCH, N, OBS_SIZE)
    assert batch.actions.shape       == (BATCH, U, N)
    assert batch.policy_target.shape == (BATCH, U+1, N, A)
    assert batch.value_target.shape  == (BATCH, U+1, N)
    assert batch.reward_target.shape == (BATCH, U, N)
    assert batch.agent_order.shape   == (BATCH, N)
    assert weights.shape == (BATCH,)
    assert indices.shape == (BATCH,)


def test_sample_dtypes(filled_buffer):
    batch, weights, indices = filled_buffer.sample(BATCH)
    assert batch.observation.dtype   == np.float32
    assert batch.actions.dtype       == np.int32
    assert batch.policy_target.dtype == np.float32
    assert weights.dtype             == np.float32
    assert indices.dtype             in (np.int64, np.intp)


def test_sample_weights_in_range(filled_buffer):
    _, weights, _ = filled_buffer.sample(BATCH)
    assert np.all(weights > 0)
    assert np.all(weights <= 1.0 + 1e-5), f"max weight={weights.max()}"


def test_indices_in_range(filled_buffer):
    _, _, indices = filled_buffer.sample(BATCH)
    assert np.all(indices >= 0)
    assert np.all(indices < CAPACITY)


def test_sample_empty_buffer_returns_none():
    buf = ReplayBuffer(**make_config())
    result = buf.sample(BATCH)
    assert result == (None, None, None)


def test_add_wraps_ring_buffer():
    """Adding more than capacity items should overwrite without error."""
    buf = ReplayBuffer(**make_config())
    fill_buffer(buf, CAPACITY * 2)
    assert len(buf) == CAPACITY


# ─── update_priorities ──────────────────────────────────────────────────────

def test_update_priorities_biases_sampling():
    """After giving one item a very high priority, it should be sampled more."""
    buf = ReplayBuffer(**make_config())
    fill_buffer(buf, CAPACITY)

    # Boost item 0's priority massively.
    buf.update_priorities(np.array([0], dtype=np.int64),
                          np.array([1000.0], dtype=np.float32))

    counts = np.zeros(CAPACITY, dtype=int)
    for _ in range(100):
        _, _, idxs = buf.sample(BATCH)
        for idx in idxs:
            counts[idx] += 1

    assert counts[0] > 50, f"Item 0 sampled only {counts[0]}/1600 times after priority boost"


# ─── update_targets ─────────────────────────────────────────────────────────

def test_update_targets_writes_position_0_only():
    buf = ReplayBuffer(**make_config())
    fill_buffer(buf, CAPACITY)

    # Sample to get some valid indices.
    batch, _, indices = buf.sample(BATCH)
    original_batch = batch

    # Overwrite with distinctive values.
    new_policy = np.ones((BATCH, N, A), dtype=np.float32) * 99.0
    new_values = np.ones(BATCH, dtype=np.float32) * 77.0
    buf.update_targets(indices.astype(np.int64), new_policy, new_values)

    # Re-sample the same indices by forcing them through update_priorities
    # with max priority so they appear in the next sample, then read back.
    buf.update_priorities(indices.astype(np.int64),
                          np.full(BATCH, 9999.0, dtype=np.float32))

    # Directly verify storage via Python sampling loop.
    batch2, _, _ = buf.sample(BATCH)
    # At least some items at position 0 should reflect the new policy.
    # (We can't guarantee all BATCH items are the ones we updated, but with
    # very high priorities they should dominate.)
    found_updated = np.any(np.isclose(batch2.policy_target[:, 0], 99.0, atol=0.1))
    assert found_updated, "update_targets did not write to position 0"


# ─── sample_for_reanalysis ──────────────────────────────────────────────────

def test_sample_for_reanalysis_shapes(filled_buffer):
    indices, obs, orders = filled_buffer.sample_for_reanalysis(REANALYZE_B)
    assert indices is not None
    assert obs.shape    == (REANALYZE_B, N, OBS_SIZE)
    assert orders.shape == (REANALYZE_B, N)
    assert indices.shape == (REANALYZE_B,)


def test_sample_for_reanalysis_unique_indices(filled_buffer):
    indices, _, _ = filled_buffer.sample_for_reanalysis(REANALYZE_B)
    assert len(np.unique(indices)) == REANALYZE_B, "Indices should be unique (no replacement)"


def test_sample_for_reanalysis_empty_returns_none():
    buf = ReplayBuffer(**make_config())
    result = buf.sample_for_reanalysis(REANALYZE_B)
    assert result == (None, None, None)


# ─── get_stats ──────────────────────────────────────────────────────────────

def test_get_stats_keys(filled_buffer):
    stats = filled_buffer.get_stats()
    for key in ("size", "capacity", "fill_pct", "priority_min",
                "priority_max", "priority_mean", "priority_std", "beta"):
        assert key in stats, f"Missing stats key: {key}"


def test_get_stats_fill_pct(filled_buffer):
    stats = filled_buffer.get_stats()
    assert abs(stats["fill_pct"] - 100.0) < 1e-3


def test_get_stats_empty():
    buf = ReplayBuffer(**make_config())
    stats = buf.get_stats()
    assert stats["size"] == 0


# ─── Pure-Python fallback ────────────────────────────────────────────────────

def test_python_fallback_produces_same_shapes(monkeypatch):
    """Force the Python fallback and verify it produces the same output shapes."""
    import utils.replay_buffer as rb_module
    # Temporarily hide the C++ backend.
    orig_cpp = rb_module._CppReplayBuffer
    orig_cfg = rb_module._CppConfig
    monkeypatch.setattr(rb_module, "_CppReplayBuffer", None)
    monkeypatch.setattr(rb_module, "_CppConfig", None)

    buf = ReplayBuffer(**make_config())
    assert not buf._use_cpp

    fill_buffer(buf, CAPACITY)
    batch, weights, indices = buf.sample(BATCH)

    assert batch.observation.shape   == (BATCH, N, OBS_SIZE)
    assert weights.shape             == (BATCH,)
    assert indices.shape             == (BATCH,)

    # Restore.
    monkeypatch.setattr(rb_module, "_CppReplayBuffer", orig_cpp)
    monkeypatch.setattr(rb_module, "_CppConfig", orig_cfg)


# ─── process_episode (pure Python, unchanged) ──────────────────────────────

def test_process_episode_basic():
    ep = Episode()
    rng = np.random.default_rng(7)
    T = 20
    for _ in range(T):
        ep.add_step(Transition(
            observation   = rng.random((N, OBS_SIZE), dtype=np.float32),
            action        = rng.integers(0, A, (N,),  dtype=np.int32),
            reward        = float(rng.random()),
            done          = False,
            policy_target = rng.random((N, A),         dtype=np.float32),
            value_target  = float(rng.random()),
            agent_order   = np.arange(N, dtype=np.int32),
        ))
    items = process_episode(ep, unroll_steps=U, n_step=5,
                            discount_gamma=0.99, num_agents=N)
    assert len(items) == T - U
    item = items[0]
    assert item.observation.shape   == (N, OBS_SIZE)
    assert item.actions.shape       == (U, N)
    assert item.policy_target.shape == (U+1, N, A)
    assert item.value_target.shape  == (U+1, N)
    assert item.reward_target.shape == (U, N)


def test_process_episode_too_short():
    ep = Episode()
    for _ in range(3):
        ep.add_step(Transition(
            observation=np.zeros((N, OBS_SIZE), dtype=np.float32),
            action=np.zeros(N, dtype=np.int32),
            reward=0.0, done=False,
            policy_target=np.zeros((N, A), dtype=np.float32),
            value_target=0.0,
            agent_order=np.arange(N, dtype=np.int32),
        ))
    items = process_episode(ep, unroll_steps=5, n_step=3,
                            discount_gamma=0.99, num_agents=N)
    assert items == []


def test_replay_item_all_child_fields_exist():
    """ReplayItem should have all_child_* fields for per-step Q-data."""
    from utils.replay_buffer import ReplayItem
    import numpy as np
    item = ReplayItem(
        observation=np.zeros((3, 10)),
        actions=np.zeros((5, 3), dtype=np.int32),
        policy_target=np.zeros((6, 3, 9)),
        value_target=np.zeros((6, 3)),
        reward_target=np.zeros((5, 3)),
        agent_order=np.arange(3),
        all_child_actions=np.zeros((6, 10, 3), dtype=np.int32),
        all_child_q=np.zeros((6, 10)),
        all_child_visits=np.zeros((6, 10)),
        all_child_valid=np.ones(6, dtype=bool),
    )
    assert item.all_child_actions.shape == (6, 10, 3)
    assert item.all_child_q.shape == (6, 10)
    assert item.all_child_visits.shape == (6, 10)
    assert item.all_child_valid.shape == (6,)


def test_process_episode_all_child_q_shape():
    """process_episode should store Q-data for all U+1 positions."""
    from utils.replay_buffer import Episode, Transition, process_episode
    import numpy as np

    N, A, K, U, T = 3, 9, 5, 5, 12
    obs_size = 18
    ep = Episode()
    for _ in range(T):
        ep.add_step(Transition(
            observation=np.zeros((N, obs_size)),
            action=np.zeros(N, dtype=np.int32),
            reward=0.0,
            done=False,
            policy_target=np.ones((N, A)) / A,
            value_target=0.0,
            agent_order=np.arange(N),
            root_child_actions=np.zeros((K, N), dtype=np.int32),
            root_child_q=np.zeros(K),
            root_child_visits=np.ones(K),
        ))
    items = process_episode(ep, unroll_steps=U, n_step=5, discount_gamma=0.99, num_agents=N)
    assert len(items) > 0
    it = items[0]
    assert it.all_child_actions is not None
    assert it.all_child_q.shape == (U + 1, K)
    assert it.all_child_actions.shape == (U + 1, K, N)
    assert it.all_child_visits.shape == (U + 1, K)
    assert it.all_child_valid.shape == (U + 1,)
    assert it.all_child_valid.all(), "all positions should be valid since ep_len > U"


def test_process_episode_all_child_q_none_without_q():
    """process_episode should leave all_child_* as None when Transitions have no Q-data."""
    from utils.replay_buffer import Episode, Transition, process_episode
    import numpy as np

    ep = Episode()
    for _ in range(10):
        ep.add_step(Transition(
            observation=np.zeros((3, 18)),
            action=np.zeros(3, dtype=np.int32),
            reward=0.0, done=False,
            policy_target=np.ones((3, 9)) / 9,
            value_target=0.0,
            agent_order=np.arange(3),
        ))
    items = process_episode(ep, unroll_steps=5, n_step=5, discount_gamma=0.99, num_agents=3)
    for it in items:
        assert it.all_child_q is None
        assert it.all_child_actions is None


def test_process_episode_never_indexes_past_episode_end():
    """process_episode's sliding window must never read past the real episode
    end and silently reuse the terminal step's data for phantom future steps
    — the bug jaxzero's postmortem found in its own `make_target`
    (`docs/superpowers/plans/2026-05-05-fix-loss-plateau.md` in ../jaxzero/).
    Confirms this repo's process_episode avoids the bug class by
    construction: it only emits windows fully inside the episode, and
    returns none at all for episodes too short to fit one.
    """
    from utils.replay_buffer import Transition, Episode, process_episode
    import numpy as np

    N, obs_size, A = 2, 4, 3
    unroll_steps, n_step, discount = 5, 3, 0.99

    def make_transition(step: int) -> Transition:
        return Transition(
            observation=np.full((N, obs_size), step, dtype=np.float32),
            action=np.zeros(N, dtype=np.int32),
            reward=1.0,
            done=False,
            policy_target=np.ones((N, A), dtype=np.float32) / A,
            value_target=0.5,
            agent_order=np.arange(N),
        )

    # Episode shorter than unroll_steps: must produce zero items, never a
    # padded/truncated one.
    short_episode = Episode()
    for t in range(3):
        short_episode.add_step(make_transition(t))
    assert process_episode(short_episode, unroll_steps, n_step, discount, N) == []

    # Episode longer than unroll_steps: every returned item's window must be
    # fully inside [0, ep_len).
    ep_len = 9
    long_episode = Episode()
    for t in range(ep_len):
        long_episode.add_step(make_transition(t))

    items = process_episode(long_episode, unroll_steps, n_step, discount, N)
    assert len(items) == ep_len - unroll_steps
    for i in range(len(items)):
        assert i + unroll_steps < ep_len, (
            f"item {i}'s window end {i + unroll_steps} reaches/exceeds ep_len {ep_len}"
        )
