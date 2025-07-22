import sys
import os
import jax
import jax.numpy as jnp
import pytest
import chex
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.replay_buffer import ReplayBuffer, Transition, Sample
from config import CONFIG

CAPACITY = 100
NUM_ENVS = 4  
NUM_ENVS_SAMPLE = 1 # Use 1 for simplicity in testing sequential data
OBS_SHAPE = (10,)
NUM_AGENTS = 2
ACTION_SPACE_SIZE = 5
UNROLL_STEPS = 5
ALPHA = CONFIG.train.replay_buffer_alpha
BETA = CONFIG.train.replay_buffer_beta_start


@pytest.fixture
def sample_transition_batch():
    """Creates a sample batch of transitions (as if from NUM_ENVS parallel environments)."""
    return Transition(
        obs=jnp.zeros((NUM_ENVS, NUM_AGENTS, *OBS_SHAPE)),
        action=jnp.zeros((NUM_ENVS, NUM_AGENTS), dtype=jnp.int32),
        reward=jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
        done=jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
        policy_target=jnp.zeros((NUM_ENVS, NUM_AGENTS, ACTION_SPACE_SIZE)),
        value_target=jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
    )

@pytest.fixture
def filled_buffer():
    """Creates a replay buffer and fills it with predictable, sequential data for unroll tests."""
    sample_transition = Transition(
        obs=jnp.zeros((NUM_ENVS_SAMPLE, NUM_AGENTS, *OBS_SHAPE)),
        action=jnp.zeros((NUM_ENVS_SAMPLE, NUM_AGENTS), dtype=jnp.int32),
        reward=jnp.zeros((NUM_ENVS_SAMPLE,), dtype=jnp.float32),
        done=jnp.zeros((NUM_ENVS_SAMPLE,), dtype=jnp.bool_),
        policy_target=jnp.zeros((NUM_ENVS_SAMPLE, NUM_AGENTS, ACTION_SPACE_SIZE)),
        value_target=jnp.zeros((NUM_ENVS_SAMPLE,), dtype=jnp.float32),
    )
    buffer = ReplayBuffer.create(CAPACITY, ALPHA, sample_transition)
    for i in range(CAPACITY):
        agent_ids = jnp.arange(NUM_AGENTS).reshape(1, -1, 1)
        obs_data = (jnp.ones_like(sample_transition.obs) * i) + (agent_ids * 1000)
        transition = jax.tree_util.tree_map(lambda x: (jnp.ones_like(x) * i).astype(x.dtype), sample_transition)
        transition = transition._replace(
            obs=obs_data.astype(transition.obs.dtype),
            action=transition.action.astype(sample_transition.action.dtype),
            reward=transition.reward.astype(sample_transition.reward.dtype),
            done=jnp.zeros_like(transition.done), # Ensure 'done' is always False
            policy_target=transition.policy_target.astype(sample_transition.policy_target.dtype),
            value_target=transition.value_target.astype(sample_transition.value_target.dtype)
        )
        buffer = buffer.add(transition)
        buffer = buffer.update_priorities(jnp.array([i]), jnp.array([i + 1.0]))
    return buffer

def test_create_replay_buffer(sample_transition_batch):
    """Tests the creation of the replay buffer."""
    buffer = ReplayBuffer.create(CAPACITY, ALPHA, sample_transition_batch)
    assert buffer.capacity == CAPACITY
    assert buffer.position == 0
    chex.assert_shape(buffer.data.obs, (CAPACITY, NUM_AGENTS, *OBS_SHAPE))
    chex.assert_shape(buffer.data.action, (CAPACITY, NUM_AGENTS))
    chex.assert_shape(buffer.data.reward, (CAPACITY,))
    assert buffer.data.done.dtype == jnp.bool_
    assert jnp.all(buffer.priorities == 0)

def test_add_sets_max_priority(sample_transition_batch):
    """Tests that adding new data sets its priority to the current maximum."""
    buffer = ReplayBuffer.create(CAPACITY, ALPHA, sample_transition_batch)

    buffer = buffer.add(sample_transition_batch)
    assert buffer.position == NUM_ENVS
    assert jnp.all(buffer.priorities[0:NUM_ENVS] == 1.0)
    assert jnp.all(buffer.priorities[NUM_ENVS:] == 0.0)

    buffer = buffer.update_priorities(jnp.array([0]), jnp.array([5.0]))
    assert buffer.priorities[0] == 5.0

    buffer = buffer.add(sample_transition_batch)
    assert buffer.position == 2 * NUM_ENVS
    assert jnp.all(buffer.priorities[NUM_ENVS : 2*NUM_ENVS] == 5.0)

def test_add_wraps_around(sample_transition_batch):
    """Tests that the buffer data and position wrap around correctly."""
    local_capacity = NUM_ENVS * 2
    buffer = ReplayBuffer.create(local_capacity, ALPHA, sample_transition_batch)

    # Batches with identifiable data
    add_batch_1_int = jax.tree_util.tree_map(lambda x: x + 1, sample_transition_batch)
    add_batch_1 = add_batch_1_int._replace(done=(add_batch_1_int.done > 0))
    
    add_batch_2_int = jax.tree_util.tree_map(lambda x: x + 2, sample_transition_batch)
    add_batch_2 = add_batch_2_int._replace(done=(add_batch_2_int.done > 0))

    add_batch_3_int = jax.tree_util.tree_map(lambda x: x + 3, sample_transition_batch)
    add_batch_3 = add_batch_3_int._replace(done=(add_batch_3_int.done > 0))

    buffer = buffer.add(add_batch_1) # Fills 0-3, pos=4. 
    buffer = buffer.update_priorities(jnp.array([0]), jnp.array([2.0])) # Max priority is now 2.0
    buffer = buffer.add(add_batch_2) # Fills 4-7, pos=0. 
    buffer = buffer.update_priorities(jnp.array([4]), jnp.array([3.0])) # Max priority is now 3.0
    buffer = buffer.add(add_batch_3) # Fills 0-3, pos=4. 
    
    assert buffer.position == NUM_ENVS
    
    chex.assert_trees_all_close(jax.tree_util.tree_map(lambda x: x[0:NUM_ENVS], buffer.data), add_batch_3)
    chex.assert_trees_all_close(jax.tree_util.tree_map(lambda x: x[NUM_ENVS:local_capacity], buffer.data), add_batch_2)
    
    # Check priorities based on wrapping and max_priority logic
    assert jnp.allclose(buffer.priorities[0:NUM_ENVS], jnp.full(NUM_ENVS, 3.0))
    expected_priorities = jnp.array([3.0, 2.0, 2.0, 2.0])
    chex.assert_trees_all_close(buffer.priorities[NUM_ENVS:local_capacity], expected_priorities)

def test_sample_unrolled_shape(filled_buffer: ReplayBuffer):
    """Tests that the sampled unrolled sequence has the correct shapes."""
    key = jax.random.PRNGKey(0)
    batch_size = 32
    sample, weights, indices = filled_buffer.sample(key, batch_size, UNROLL_STEPS, BETA)
    assert isinstance(sample, Sample)
    chex.assert_shape(sample.obs, (batch_size, NUM_AGENTS, *OBS_SHAPE))
    chex.assert_shape(sample.actions, (batch_size, UNROLL_STEPS, NUM_AGENTS))
    chex.assert_shape(sample.rewards, (batch_size, UNROLL_STEPS))
    chex.assert_shape(sample.dones, (batch_size, UNROLL_STEPS))
    chex.assert_shape(sample.policy_targets, (batch_size, UNROLL_STEPS + 1, NUM_AGENTS, ACTION_SPACE_SIZE))
    chex.assert_shape(sample.value_targets, (batch_size, UNROLL_STEPS + 1))
    chex.assert_shape(sample.mask, (batch_size, UNROLL_STEPS))
    chex.assert_shape(weights, (batch_size,))
    chex.assert_shape(indices, (batch_size,))

def test_sample_unrolled_content(filled_buffer: ReplayBuffer):
    """
    Tests that the content of a sampled sequence and its importance
    sampling weight are correct.
    """
    key = jax.random.PRNGKey(0)
    batch_size = 1
    
    sample, weights, indices = filled_buffer.sample(key, batch_size, UNROLL_STEPS, BETA)
    start_index = indices[0]
    
    expected_rewards = jnp.arange(start_index, start_index + UNROLL_STEPS, dtype=jnp.float32) % CAPACITY
    chex.assert_trees_all_close(sample.rewards[0], expected_rewards)

    expected_values = jnp.arange(start_index, start_index + UNROLL_STEPS + 1, dtype=jnp.float32) % CAPACITY
    chex.assert_trees_all_close(sample.value_targets[0], expected_values)

    assert jnp.isclose(weights[0], 1.0)

def test_sample_unrolled_wraps_around(filled_buffer: ReplayBuffer):
    """Tests that unrolling correctly wraps around the end of the buffer."""
    key = jax.random.PRNGKey(0)
    batch_size = 1
    
    wrap_index = CAPACITY - 2
    priorities = jnp.full(CAPACITY, 1e-6).at[wrap_index].set(1e6)
    wrap_buffer = filled_buffer.replace(priorities=priorities)
    
    sample, _, indices = wrap_buffer.sample(key, batch_size, UNROLL_STEPS, BETA)
    
    assert indices[0] == wrap_index

    expected_rewards = jnp.array([98, 99, 0, 1, 2], dtype=jnp.float32)
    chex.assert_trees_all_close(sample.rewards[0], expected_rewards)

def test_prioritized_sampling(filled_buffer: ReplayBuffer):
    """Tests that sampling is biased towards high-priority items."""
    key = jax.random.PRNGKey(0)
    batch_size = 1000
    _, _, indices = filled_buffer.sample(key, batch_size, UNROLL_STEPS, BETA)
    
    assert jnp.mean(indices) > 55

def test_update_priorities(filled_buffer: ReplayBuffer):
    """Tests the priority update mechanism."""
    indices_to_update = jnp.array([5, 10, 15])
    new_priorities = jnp.array([1000.0, 1000.0, 1000.0])
    
    original_priority_6 = filled_buffer.priorities[6]
    updated_buffer = filled_buffer.update_priorities(indices_to_update, new_priorities)
    
    assert updated_buffer.priorities[5] == 1000.0
    assert updated_buffer.priorities[10] == 1000.0
    assert updated_buffer.priorities[15] == 1000.0
    assert updated_buffer.priorities[6] == original_priority_6

def test_sample_masking_on_done(filled_buffer: ReplayBuffer):
    """Tests that the loss mask correctly stops after a 'done' flag."""
    key = jax.random.PRNGKey(42)
    batch_size = 1

    done_index = 50
    buffer_with_done = filled_buffer.replace(
        data=filled_buffer.data._replace(
            done=filled_buffer.data.done.at[done_index].set(True)
        )
    )

    start_index = done_index - 2 # Sample from index 48
    priorities = jnp.full(CAPACITY, 1e-6).at[start_index].set(1e6)
    forced_sample_buffer = buffer_with_done.replace(priorities=priorities)

    sample, _, indices = forced_sample_buffer.sample(key, batch_size, UNROLL_STEPS, BETA)

    assert indices[0] == start_index

    expected_mask = jnp.array([True, True, True, False, False], dtype=jnp.bool_)
    chex.assert_trees_all_close(sample.mask[0], expected_mask)

def test_multi_agent_data_integrity(filled_buffer: ReplayBuffer):
    """Tests that data for different agents is not mixed during sampling."""
    key = jax.random.PRNGKey(123)
    batch_size = 1

    start_index = 20
    priorities = jnp.full(CAPACITY, 1e-6).at[start_index].set(1e6)
    forced_sample_buffer = filled_buffer.replace(priorities=priorities)

    sample, _, indices = forced_sample_buffer.sample(key, batch_size, UNROLL_STEPS, BETA)

    assert indices[0] == start_index

    obs_agent0 = sample.obs[0, 0] 
    obs_agent1 = sample.obs[0, 1] 

    expected_obs_agent0 = jnp.full(OBS_SHAPE, 20.0)
    expected_obs_agent1 = jnp.full(OBS_SHAPE, 20.0 + 1000.0)

    chex.assert_trees_all_close(obs_agent0, expected_obs_agent0)
    chex.assert_trees_all_close(obs_agent1, expected_obs_agent1)