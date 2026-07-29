# Second Simplification Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut unused features, redundant abstractions, and the custom C++/CUDA replay buffer subsystem out of `sequential-muzero`, following the approved design spec (`docs/superpowers/specs/2026-07-29-second-simplification-pass-design.md`).

**Architecture:** Nine sequential tasks. Task 1 is the highest-risk task (replaces the custom C++ replay buffer with `cpprb` and simultaneously removes the dead `agent_order` payload and dead Q-data `None`-guards, since all three changes collide on the same dataclasses and the same test file). Tasks 2-4 finish Section 1/2 of the design (mctx removal, planner-indirection collapse, MPE/baselines removal). Tasks 5-6 delete two never-enabled feature flags. Tasks 7-9 are small mechanical cleanups (config-constant promotion, a naming fix, one dedup).

**Tech Stack:** JAX/Flax, Ray, Hydra/OmegaConf, `cpprb` (replacing the custom `csrc/` C++ extension), pytest.

## Global Constraints

- Full test suite (`conda run -n mazero pytest tests/ -v`) must pass after every task's commit(s).
- No behavior changes beyond what's explicitly listed in each task. The only intentional runtime-behavior change in this plan is Task 1's replay-buffer backend swap (C++ → cpprb) — call this out explicitly in that task's commit message.
- CLAUDE.md and README.md are updated in the same task that changes the behavior they describe — never left stale for a later task to fix.
- No git history rewriting.
- Work happens on branch `simplify/second-pass`, in an isolated worktree (see `superpowers:using-git-worktrees`).
- `model/model.py`'s `DynamicsNetwork` attention is out of scope — do not touch it in any task.

---

### Task 1: Replay buffer overhaul — cpprb migration + agent_order removal + Q-data required fields

**Files:**
- Modify: `mcts/base.py`
- Modify: `mcts/mcts_joint_osla.py`
- Modify: `actors/data_actor.py`
- Modify: `actors/reanalyze_actor.py`
- Modify: `actors/replay_buffer_actor.py`
- Modify: `actors/loss.py`
- Rewrite: `utils/replay_buffer.py`
- Rewrite: `tests/test_replay_buffer.py`
- Modify: `tests/test_mcts.py` (delete one test)
- Delete: `csrc/replay_buffer/bindings.cpp`, `csrc/replay_buffer/pinned_alloc.h`, `csrc/replay_buffer/replay_buffer.cpp`, `csrc/replay_buffer/replay_buffer.h`, `csrc/replay_buffer/sum_tree.h`, `CMakeLists.txt`, `setup.py`, `_replay_buffer_cpp.cpython-310-x86_64-linux-gnu.so`, `build/` (entire tracked directory, 87 files)
- Delete: `benchmarks/replay_buffer_benchmark.py`
- Modify: `requirements.txt`
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:**
- Produces: `utils.replay_buffer.ReplayBuffer` — same public method signatures as before (`add`, `sample`, `sample_for_reanalysis`, `update_targets`, `update_priorities`, `get_stats`, `__len__`) EXCEPT `sample_for_reanalysis` now returns a 2-tuple `(indices, observations)` instead of a 3-tuple (agent_order dropped).
- Produces: `utils.replay_buffer.Transition` — `root_child_actions`, `root_child_q`, `root_child_visits` are now required (no default), `agent_order` field removed.
- Produces: `utils.replay_buffer.ReplayItem` — `all_child_actions`, `all_child_q`, `all_child_visits`, `all_child_valid` are now required (no default), `agent_order` field removed.
- Produces: `mcts.base.MCTSPlanOutput` — `agent_order` field removed, `root_value` retyped from `float` to `chex.Array`.
- Consumes downstream by: Task 3 (folds `MCTSPlanner` ABC away — do not do that here, only remove `agent_order` from the NamedTuple).

- [ ] **Step 1: Rewrite `mcts/base.py`** — remove the `agent_order` field from `MCTSPlanOutput` and fix its `root_value` type annotation (it holds an array, not a `float`). Do NOT touch the `MCTSPlanner` ABC in this task (Task 3 handles that).

```python
# mcts/base.py

from abc import ABC, abstractmethod
from typing import NamedTuple

import chex

from model import FlaxMAMuZeroNet
from config import ExperimentConfig
from utils.transforms import DiscreteSupport


class MCTSPlanOutput(NamedTuple):
    """Output of any MCTS planner for a single planning step.

    Shapes (B=batch/envs, N=agents, A=actions, K=num_gumbel_samples):
        joint_action:       (B, N)    — chosen action index per agent
        policy_targets:     (B, N, A) — MCTS-improved policy targets for training
        root_value:         (B,)      — estimated value of the root state
        root_child_actions: (B, K, N) — K sampled joint actions as per-agent indices (None for non-OSLA)
        root_child_q:       (B, K)    — Q-value for each root child (None for non-OSLA)
        root_child_visits:  (B, K)    — visit count for each root child (None for non-OSLA)
    """
    joint_action:       chex.Array
    policy_targets:     chex.Array
    root_value:         chex.Array
    root_child_actions: chex.Array = None
    root_child_q:       chex.Array = None
    root_child_visits:  chex.Array = None


class MCTSPlanner(ABC):
    """
    Abstract base class for all MCTS planner variants.

    Subclasses must implement `_recurrent_fn` and `_plan_loop`.
    JIT compilation is handled externally (e.g. in DataActor) — planners
    do not JIT their own `plan()` method.
    """

    def __init__(self, model: FlaxMAMuZeroNet, config: ExperimentConfig):
        self.model = model
        self.num_agents = config.train.num_agents
        self.action_space_size = model.action_space_size

        self.num_simulations = config.mcts.num_simulations
        self.max_depth_gumbel_search = config.mcts.max_depth_gumbel_search
        self.num_gumbel_samples = config.mcts.num_gumbel_samples
        self.discount_gamma = config.train.discount_gamma

        self.value_support = DiscreteSupport(
            min=-config.model.value_support_size,
            max=config.model.value_support_size,
        )
        self.reward_support = DiscreteSupport(
            min=-config.model.reward_support_size,
            max=config.model.reward_support_size,
        )

        self.dirichlet_alpha = config.mcts.dirichlet_alpha
        self.dirichlet_fraction = config.mcts.dirichlet_fraction

    @abstractmethod
    def _recurrent_fn(
        self, params, rng_key: chex.Array, action: chex.Array, embedding
    ):
        """Single-step batched rollout used inside the MCTS simulations."""
        pass

    @abstractmethod
    def _plan_loop(
        self, params, rng_key: chex.Array, observation: chex.Array
    ) -> MCTSPlanOutput:
        """Full planning logic for one environment step."""
        pass

    def plan(
        self, params, rng_key: chex.Array, observation: chex.Array
    ) -> MCTSPlanOutput:
        """
        Public entry point. JIT compilation is the caller's responsibility
        (DataActor wraps this with jax.jit at construction time).
        """
        return self._plan_loop(params, rng_key, observation)
```

(This is the same file as before, minus the `agent_order` field and the `root_value` type fix — the ABC body is untouched, copied verbatim.)

- [ ] **Step 2: Edit `mcts/mcts_joint_osla.py`** — remove the two lines that produce `agent_order`:

In `_osla_plan_single`'s return statement, delete this line:
```python
        agent_order=jnp.arange(N),
```

In `MCTSJointOSLAPlanner._plan_loop`'s return statement, delete this line:
```python
            agent_order=results.agent_order[0],                         # (N,) — same for all
```

- [ ] **Step 3: Rewrite `utils/replay_buffer.py` in full** — cpprb-only backend, `agent_order` removed, Q-data fields required:

```python
# utils/replay_buffer.py
"""Prioritized experience replay, backed by cpprb.PrioritizedReplayBuffer."""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from jax import tree_util

from cpprb import PrioritizedReplayBuffer as _PRB


@dataclass
class Transition:
    """
    Holds all the data for a single step (or transition) in an environment.
    """
    observation: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    policy_target: np.ndarray
    value_target: float
    # Per-root-child Q-data for action-level AWPO.
    root_child_actions: np.ndarray  # (K, N) int32
    root_child_q: np.ndarray        # (K,) float32
    root_child_visits: np.ndarray   # (K,) float32


@dataclass
class Episode:
    """
    A container for a full episode's trajectory and metadata.
    """
    trajectory: List[Transition] = field(default_factory=list)
    episode_return: float = 0.0

    def add_step(self, transition: Transition):
        self.trajectory.append(transition)
        self.episode_return += transition.reward


@dataclass
class ReplayItem:
    """
    A single, self-contained training sample for the MuZero model.

    Shapes (using shorthand):
        B = batch_size, U = unroll_steps, N = num_agents, A = action_space_size,
        K = num_gumbel_samples
    """
    observation: np.ndarray    # (B, N, obs_size)
    actions: np.ndarray        # (B, U, N)
    policy_target: np.ndarray  # (B, U+1, N, A)
    value_target: np.ndarray   # (B, U+1, N)
    reward_target: np.ndarray  # (B, U, N)
    # Per-step Q-data for action-level AWPO at all U+1 positions.
    # Not part of the JAX pytree below — stored in ReplayBufferActor's
    # sidecar arrays, injected alongside the sampled batch at sample time.
    all_child_actions: np.ndarray  # (U+1, K, N) int32
    all_child_q: np.ndarray        # (U+1, K) float32
    all_child_visits: np.ndarray   # (U+1, K) float32
    all_child_valid: np.ndarray    # (U+1,) bool


def flatten_replay_item(item: ReplayItem):
    children = (
        item.observation,
        item.actions,
        item.policy_target,
        item.value_target,
        item.reward_target,
    )
    return children, None


def unflatten_replay_item(static_data, children):
    # Q-data fields aren't part of this pytree (see ReplayItem docstring) —
    # explicitly None here since nothing reads them off a tree_map'd instance.
    return ReplayItem(
        observation=children[0],
        actions=children[1],
        policy_target=children[2],
        value_target=children[3],
        reward_target=children[4],
        all_child_actions=None,
        all_child_q=None,
        all_child_visits=None,
        all_child_valid=None,
    )


tree_util.register_pytree_node(
    ReplayItem,
    flatten_replay_item,
    unflatten_replay_item,
)


class ReplayBuffer:
    """Prioritized experience replay, backed by cpprb.PrioritizedReplayBuffer."""

    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple,
        action_space_size: int,
        num_agents: int,
        unroll_steps: int,
        alpha: float,
        beta_start: float,
        beta_frames: float,
    ):
        self.capacity     = capacity
        self.alpha        = alpha
        self.beta_start   = beta_start
        self.beta_frames  = beta_frames
        self.frame_count  = 0

        self.observations    = np.zeros((capacity, num_agents, *observation_shape), dtype=np.float32)
        self.actions         = np.zeros((capacity, unroll_steps, num_agents),       dtype=np.int32)
        self.policy_targets  = np.zeros((capacity, unroll_steps + 1, num_agents, action_space_size), dtype=np.float32)
        self.value_targets   = np.zeros((capacity, unroll_steps + 1, num_agents),   dtype=np.float32)
        self.reward_targets  = np.zeros((capacity, unroll_steps, num_agents),        dtype=np.float32)
        self._priorities_log = np.zeros(capacity, dtype=np.float32)

        self._ptree = _PRB(
            capacity,
            env_dict={"_": {"shape": 1, "dtype": np.float32}},
            alpha=alpha,
        )

        self.pointer = 0
        self.size    = 0

    def add(self, item: ReplayItem, priority: float):
        priority = float(priority) if priority > 0 else (
            self._priorities_log[:self.size].max() if self.size > 0 else 1.0
        )
        idx = self.pointer
        self.observations[idx]    = item.observation
        self.actions[idx]         = item.actions
        self.policy_targets[idx]  = item.policy_target
        self.value_targets[idx]   = item.value_target
        self.reward_targets[idx]  = item.reward_target
        self._priorities_log[idx] = priority

        self._ptree.add(**{"_": np.zeros((1, 1), dtype=np.float32)},
                        priorities=np.array([priority], dtype=np.float32))

        self.pointer = (self.pointer + 1) % self.capacity
        self.size    = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[ReplayItem, np.ndarray, np.ndarray]:
        if self.size == 0:
            return None, None, None

        beta = min(1.0, self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames)
        self.frame_count += 1

        s       = self._ptree.sample(batch_size, beta=beta)
        indices = s["indexes"].astype(np.int64)
        weights = s["weights"].astype(np.float32)

        batch = ReplayItem(
            observation   = self.observations[indices],
            actions       = self.actions[indices],
            policy_target = self.policy_targets[indices],
            value_target  = self.value_targets[indices],
            reward_target = self.reward_targets[indices],
            all_child_actions=None,
            all_child_q=None,
            all_child_visits=None,
            all_child_valid=None,
        )
        return batch, weights, indices

    def sample_for_reanalysis(self, batch_size: int):
        if self.size == 0:
            return None, None
        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False)
        return indices, self.observations[indices].copy()

    def update_targets(self, indices: np.ndarray, policy_targets: np.ndarray, root_values: np.ndarray):
        self.policy_targets[indices, 0] = policy_targets
        self.value_targets[indices, 0]  = root_values[:, None]

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        self._priorities_log[indices] = priorities
        self._ptree.update_priorities(indices, priorities)

    def get_stats(self) -> dict:
        if self.size == 0:
            return {"size": 0, "capacity": self.capacity, "fill_pct": 0.0}
        active = self._priorities_log[:self.size]
        beta   = min(1.0, self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames)
        return {
            "size":          self.size,
            "capacity":      self.capacity,
            "fill_pct":      100.0 * self.size / self.capacity,
            "priority_min":  float(active.min()),
            "priority_max":  float(active.max()),
            "priority_mean": float(active.mean()),
            "priority_std":  float(active.std()),
            "beta":          beta,
        }

    def __len__(self):
        return self.size


def process_episode(
    episode: "Episode",
    unroll_steps: int,
    n_step: int,
    discount_gamma: float,
    num_agents: int,
) -> list:
    """
    Converts a completed episode into ReplayItems via a sliding window.

    For each valid start position, computes n-step bootstrapped value targets
    and packs `unroll_steps` transitions into a single ReplayItem.

    Args:
        episode: Completed episode containing a trajectory of Transitions.
        unroll_steps: Number of steps per training sample (U).
        n_step: Lookahead horizon for bootstrapped value targets.
        discount_gamma: Discount factor γ.
        num_agents: Number of agents N (for broadcasting scalar targets).

    Returns:
        List of ReplayItems, one per valid start index.
        Empty list if the episode is too short to produce any samples.
    """
    trajectory = episode.trajectory
    ep_len = len(trajectory)

    if ep_len <= unroll_steps:
        return []

    # Extract arrays once to avoid repeated attribute lookups in the loop.
    observations = np.stack([t.observation for t in trajectory])      # (T, N, obs_size)
    actions = np.stack([t.action for t in trajectory])                # (T, N)
    policy_targets = np.stack([t.policy_target for t in trajectory])  # (T, N, A)
    rewards = np.array([t.reward for t in trajectory], dtype=np.float32)      # (T,)
    mcts_values = np.array([t.value_target for t in trajectory], dtype=np.float32)  # (T,)

    # Pre-compute discount coefficients [γ^0, γ^1, ..., γ^(n-1)] for np.dot.
    discount_vec = discount_gamma ** np.arange(n_step, dtype=np.float32)

    replay_items = []
    for start in range(ep_len - unroll_steps):
        # Compute n-step bootstrapped value target for each of the U+1 positions.
        value_targets = np.empty(unroll_steps + 1, dtype=np.float32)
        for i in range(unroll_steps + 1):
            t = start + i
            window = rewards[t : t + n_step]
            value_targets[i] = np.dot(window, discount_vec[: len(window)])
            bootstrap_idx = t + n_step
            if bootstrap_idx < ep_len:
                value_targets[i] += mcts_values[bootstrap_idx] * (discount_gamma ** n_step)

        # Broadcast scalar value/reward targets to per-agent arrays.
        # np.broadcast_to returns a read-only view; .copy() makes it writable.
        value_target_per_agent = np.broadcast_to(
            value_targets[:, None], (unroll_steps + 1, num_agents)
        ).copy().astype(np.float32)  # (U+1, N)

        reward_target_per_agent = np.broadcast_to(
            rewards[start : start + unroll_steps, None], (unroll_steps, num_agents)
        ).copy().astype(np.float32)  # (U, N)

        # Every Transition carries Q-data (the sole planner always produces it).
        all_child_actions = np.stack(
            [trajectory[start + i].root_child_actions for i in range(unroll_steps + 1)]
        )  # (U+1, K, N)
        all_child_q = np.stack(
            [trajectory[start + i].root_child_q for i in range(unroll_steps + 1)]
        )  # (U+1, K)
        all_child_visits = np.stack(
            [trajectory[start + i].root_child_visits for i in range(unroll_steps + 1)]
        )  # (U+1, K)
        all_child_valid = np.ones(unroll_steps + 1, dtype=bool)

        replay_items.append(
            ReplayItem(
                observation=observations[start],                                    # (N, obs_size)
                actions=actions[start : start + unroll_steps],                     # (U, N)
                policy_target=policy_targets[start : start + unroll_steps + 1],   # (U+1, N, A)
                value_target=value_target_per_agent,                               # (U+1, N)
                reward_target=reward_target_per_agent,                             # (U, N)
                all_child_actions=all_child_actions,
                all_child_q=all_child_q,
                all_child_visits=all_child_visits,
                all_child_valid=all_child_valid,
            )
        )

    return replay_items
```

- [ ] **Step 4: Edit `actors/replay_buffer_actor.py`** — remove the dead `else` branch in `add()` (item's Q-data is always present now) and drop the `agent_order` 3rd return value from `sample_for_reanalysis`:

Replace the `add` method:
```python
    def add(self, items: list, priorities: List[float]):
        for item, priority in zip(items, priorities):
            slot = self._add_counter % self._capacity
            self.buffer.add(item, priority)
            self._q_actions[slot] = item.all_child_actions  # (U+1, K, N)
            self._q_values[slot]  = item.all_child_q        # (U+1, K)
            self._q_visits[slot]  = item.all_child_visits   # (U+1, K)
            self._q_valid[slot]   = item.all_child_valid    # (U+1,) bool
            self._add_counter += 1
```

`sample_for_reanalysis` stays a pure passthrough (`return self.buffer.sample_for_reanalysis(batch_size)`), unchanged — it will now return whatever the rewritten `ReplayBuffer.sample_for_reanalysis` returns (a 2-tuple).

- [ ] **Step 5: Edit `actors/data_actor.py`** — remove the dead `is not None else None` ternaries around Q-data conversion (the sole planner always populates these), and remove the `agent_order=...` line from the `Transition(...)` construction.

Replace this block in `run_episode`:
```python
            with self.profiler.time("device_get"):
                # plan_output.joint_action:   (B, N)
                # plan_output.policy_targets: (B, N, A)
                # plan_output.root_value:     (B,)
                actions_np = np.array(plan_output.joint_action)
                root_values_np = np.array(plan_output.root_value)
                policy_targets_np = np.array(plan_output.policy_targets)
                # Q-data for action-level AWPO
                root_child_actions_np = np.array(plan_output.root_child_actions)  # (B, K, N)
                root_child_q_np = np.array(plan_output.root_child_q)              # (B, K)
                root_child_visits_np = np.array(plan_output.root_child_visits)    # (B, K)
```

And replace the `Transition(...)` construction inside the `for i in range(B):` loop:
```python
                episodes[i].add_step(
                    Transition(
                        observation=np.array(observations[i]),
                        action=actions_np[i],
                        reward=float(rewards_np[i]),
                        done=bool(dones_np[i]),
                        policy_target=policy_targets_np[i],
                        value_target=float(root_values_np[i]),
                        root_child_actions=root_child_actions_np[i],
                        root_child_q=root_child_q_np[i],
                        root_child_visits=root_child_visits_np[i],
                    )
                )
```

- [ ] **Step 6: Edit `actors/reanalyze_actor.py`** — the sole planner always returns Q-data, so remove the `if plan_output.root_child_q is not None:` guard, and update the `sample_for_reanalysis` unpacking to a 2-tuple.

Replace:
```python
        with self.profiler.time("sample_wait"):
            indices, observations, _ = ray.get(
                self.replay_buffer.sample_for_reanalysis.remote(
                    self.config.train.reanalyze_batch_size
                )
            )
```
with:
```python
        with self.profiler.time("sample_wait"):
            indices, observations = ray.get(
                self.replay_buffer.sample_for_reanalysis.remote(
                    self.config.train.reanalyze_batch_size
                )
            )
```

Replace:
```python
        with self.profiler.time("buffer_update"):
            self.replay_buffer.update_targets.remote(
                indices,
                np.array(plan_output.policy_targets),
                np.array(plan_output.root_value),
            )
            if plan_output.root_child_q is not None:
                self.replay_buffer.update_root_q.remote(
                    indices,
                    np.array(plan_output.root_child_actions),  # (B, K, N)
                    np.array(plan_output.root_child_q),         # (B, K)
                    np.array(plan_output.root_child_visits),    # (B, K)
                )
```
with:
```python
        with self.profiler.time("buffer_update"):
            self.replay_buffer.update_targets.remote(
                indices,
                np.array(plan_output.policy_targets),
                np.array(plan_output.root_value),
            )
            self.replay_buffer.update_root_q.remote(
                indices,
                np.array(plan_output.root_child_actions),  # (B, K, N)
                np.array(plan_output.root_child_q),         # (B, K)
                np.array(plan_output.root_child_visits),    # (B, K)
            )
```

- [ ] **Step 7: Edit `actors/loss.py`** — `q_data` is always a real dict by the time `train_step` runs (the caller returns early on an empty buffer before ever building it). Collapse the two dead branches:

Change the function signature from `def train_step(params, opt_state, batch, weights, rng_key, ema_params, q_data=None):` to:
```python
    def train_step(params, opt_state, batch, weights, rng_key, ema_params, q_data):
```

Replace the root-policy-loss block:
```python
            if awpo_alpha > 0.0 and q_data is not None:
                # Action-level AWPO: weight each sampled root action by
                # exp((Q_k - V_net) / alpha), batch-normalized (_awpo_weight).
                q_valid       = q_data["all_child_valid"][:, 0]      # (B,) bool — root position
                q_k           = q_data["all_child_q"][:, 0]          # (B, K)
                visits_k      = q_data["all_child_visits"][:, 0]     # (B, K)
                child_actions = q_data["all_child_actions"][:, 0]    # (B, K, N)

                # Current network value prediction as AWAC baseline (not stale MCTS value)
                v_net = support_to_scalar(init_out.value_logits, value_support)  # (B,)
                awpo_w_k = _awpo_weight(q_k, v_net, awpo_alpha)  # (B, K)

                # Per-agent log-probs for each sampled joint action:
                # log_probs[b, n, a] → gather with child_actions[b, k, n]
                log_probs = jax.nn.log_softmax(init_out.policy_logits, axis=-1)  # (B, N, A)
                B_, K_ = q_k.shape
                N_, A_ = log_probs.shape[1], log_probs.shape[2]
                # Expand log_probs to (B, K, N, A) and gather at child_actions
                log_probs_exp = jnp.broadcast_to(
                    log_probs[:, None, :, :], (B_, K_, N_, A_)
                )
                gathered = jnp.take_along_axis(
                    log_probs_exp,
                    child_actions[:, :, :, None],  # (B, K, N, 1)
                    axis=-1,
                ).squeeze(-1)  # (B, K, N)
                joint_log_prob_k = gathered.sum(axis=-1)  # (B, K)

                # Visit-count-weighted AWPO loss
                visit_weights = visits_k / (visits_k.sum(axis=-1, keepdims=True) + 1e-8)
                action_awpo_loss = -(visit_weights * awpo_w_k * joint_log_prob_k).sum(axis=-1)  # (B,)

                # Fall back to plain CE for items where Q-data was not stored
                p0_loss = jnp.where(q_valid, action_awpo_loss, ce_p0)  # (B,)
            elif awpo_alpha > 0.0:
                # No Q-data available (e.g. non-OSLA planner): state-level fallback
                v_mcts = batch.value_target[:, 0].mean(axis=-1)  # (B,)
                v_net = support_to_scalar(init_out.value_logits, value_support)  # (B,)
                awpo_w = _awpo_weight(v_mcts, v_net, awpo_alpha)
                p0_loss = awpo_w * ce_p0  # (B,)
            else:
                p0_loss = ce_p0  # (B,)
```
with:
```python
            if awpo_alpha > 0.0:
                # Action-level AWPO: weight each sampled root action by
                # exp((Q_k - V_net) / alpha), batch-normalized (_awpo_weight).
                q_valid       = q_data["all_child_valid"][:, 0]      # (B,) bool — root position
                q_k           = q_data["all_child_q"][:, 0]          # (B, K)
                visits_k      = q_data["all_child_visits"][:, 0]     # (B, K)
                child_actions = q_data["all_child_actions"][:, 0]    # (B, K, N)

                # Current network value prediction as AWAC baseline (not stale MCTS value)
                v_net = support_to_scalar(init_out.value_logits, value_support)  # (B,)
                awpo_w_k = _awpo_weight(q_k, v_net, awpo_alpha)  # (B, K)

                # Per-agent log-probs for each sampled joint action:
                # log_probs[b, n, a] → gather with child_actions[b, k, n]
                log_probs = jax.nn.log_softmax(init_out.policy_logits, axis=-1)  # (B, N, A)
                B_, K_ = q_k.shape
                N_, A_ = log_probs.shape[1], log_probs.shape[2]
                # Expand log_probs to (B, K, N, A) and gather at child_actions
                log_probs_exp = jnp.broadcast_to(
                    log_probs[:, None, :, :], (B_, K_, N_, A_)
                )
                gathered = jnp.take_along_axis(
                    log_probs_exp,
                    child_actions[:, :, :, None],  # (B, K, N, 1)
                    axis=-1,
                ).squeeze(-1)  # (B, K, N)
                joint_log_prob_k = gathered.sum(axis=-1)  # (B, K)

                # Visit-count-weighted AWPO loss
                visit_weights = visits_k / (visits_k.sum(axis=-1, keepdims=True) + 1e-8)
                action_awpo_loss = -(visit_weights * awpo_w_k * joint_log_prob_k).sum(axis=-1)  # (B,)

                # q_valid masks cold ring-buffer slots that predate this item's
                # Q-data being written (not a "missing planner" case — the
                # sole planner always produces Q-data).
                p0_loss = jnp.where(q_valid, action_awpo_loss, ce_p0)  # (B,)
            else:
                p0_loss = ce_p0  # (B,)
```

Then, inside the `if awpo_alpha > 0.0:` block that builds the unroll-step `xs` (further down in the same function), remove the dead "no Q-data" `else` branch. Replace:
```python
            if awpo_alpha > 0.0:
                # Build per-step Q-data for the scan (positions 1..U).
                # When q_data is None (non-OSLA planner), pass zeros with all-invalid mask
                # so scan_step always has the same input structure.
                if q_data is not None:
                    step_q_acts  = jnp.moveaxis(q_data["all_child_actions"][:, 1:], 1, 0)  # (U, B, K, N)
                    step_q_q     = jnp.moveaxis(q_data["all_child_q"][:, 1:],       1, 0)  # (U, B, K)
                    step_q_vis   = jnp.moveaxis(q_data["all_child_visits"][:, 1:],  1, 0)  # (U, B, K)
                    step_q_valid = jnp.moveaxis(q_data["all_child_valid"][:, 1:],   1, 0)  # (U, B)
                else:
                    B_ = batch.observation.shape[0]
                    step_q_acts  = jnp.zeros((U, B_, K, N), jnp.int32)
                    step_q_q     = jnp.zeros((U, B_, K),    jnp.float32)
                    step_q_vis   = jnp.zeros((U, B_, K),    jnp.float32)
                    step_q_valid = jnp.zeros((U, B_),       jnp.bool_)

                xs = (
```
with:
```python
            if awpo_alpha > 0.0:
                # Build per-step Q-data for the scan (positions 1..U).
                step_q_acts  = jnp.moveaxis(q_data["all_child_actions"][:, 1:], 1, 0)  # (U, B, K, N)
                step_q_q     = jnp.moveaxis(q_data["all_child_q"][:, 1:],       1, 0)  # (U, B, K)
                step_q_vis   = jnp.moveaxis(q_data["all_child_visits"][:, 1:],  1, 0)  # (U, B, K)
                step_q_valid = jnp.moveaxis(q_data["all_child_valid"][:, 1:],   1, 0)  # (U, B)

                xs = (
```
(the rest of that block, `scan_step` and the `else:`/plain-CE branch below it, is unchanged).

Finally, in `actors/learner_actor.py`'s `_train_step()`, `q_data` is always a real dict by the time it reaches `train_step` — remove the ternary:
```python
            jax_q_data = {
                k: jax.device_put(np.asarray(v)) for k, v in q_data.items()
            } if q_data is not None else None
            self.params, self.opt_state, transfer_buf, new_priorities = self.train_step(
                self.params, self.opt_state, jax_batch, jax_weights, train_key,
                self.ema_params, jax_q_data,
            )
```
becomes:
```python
            jax_q_data = {
                k: jax.device_put(np.asarray(v)) for k, v in q_data.items()
            }
            self.params, self.opt_state, transfer_buf, new_priorities = self.train_step(
                self.params, self.opt_state, jax_batch, jax_weights, train_key,
                self.ema_params, jax_q_data,
            )
```

- [ ] **Step 8: Delete the C++ backend and its build tooling.**

```bash
git rm -r csrc/ CMakeLists.txt setup.py _replay_buffer_cpp.cpython-310-x86_64-linux-gnu.so build/
```

- [ ] **Step 9: Delete `benchmarks/replay_buffer_benchmark.py`** (it compared the now-gone C++ backend against the pure-Python fallback — no longer meaningful).

```bash
git rm benchmarks/replay_buffer_benchmark.py
```

- [ ] **Step 10: Edit `requirements.txt`** — remove `pybind11` (no longer building a C++ extension). `cpprb` is already listed.

- [ ] **Step 11: Delete `tests/test_mcts.py`'s `test_agent_order_sequential`** — `MCTSPlanOutput` no longer has an `agent_order` field. Find and remove the test method:
```python
    def test_agent_order_sequential(self, osla_plan_fn, params, obs):
```
(remove the whole method, including its `assert jnp.array_equal(out.agent_order, jnp.arange(N))` body — read the file around that line to capture the full method before deleting it).

- [ ] **Step 12: Rewrite `tests/test_replay_buffer.py` in full**, dropping the C++-backend and pure-Python-fallback dual-path tests (there's only one path now), dropping `agent_order` everywhere, and making Q-data fields required in every `Transition`/`ReplayItem` construction:

```python
"""
Tests for the cpprb-backed prioritized replay buffer.

Covers:
 - add / sample shapes and dtypes
 - update_priorities changes sampling distribution
 - update_targets writes only to position 0
 - sample_for_reanalysis uniqueness and shapes
 - get_stats keys
 - process_episode (pure Python, backend-independent)

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
K           = 5    # sampled joint actions per node
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
        all_child_actions = rng.integers(0, A, (U+1, K, N), dtype=np.int32),
        all_child_q        = rng.random((U+1, K), dtype=np.float32),
        all_child_visits   = rng.random((U+1, K), dtype=np.float32),
        all_child_valid    = np.ones(U+1, dtype=bool),
    )


def fill_buffer(buf: ReplayBuffer, n: int):
    rng = np.random.default_rng(0)
    for _ in range(n):
        buf.add(make_item(rng), priority=1.0)


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
    indices, obs = filled_buffer.sample_for_reanalysis(REANALYZE_B)
    assert indices is not None
    assert obs.shape    == (REANALYZE_B, N, OBS_SIZE)
    assert indices.shape == (REANALYZE_B,)


def test_sample_for_reanalysis_unique_indices(filled_buffer):
    indices, _ = filled_buffer.sample_for_reanalysis(REANALYZE_B)
    assert len(np.unique(indices)) == REANALYZE_B, "Indices should be unique (no replacement)"


def test_sample_for_reanalysis_empty_returns_none():
    buf = ReplayBuffer(**make_config())
    result = buf.sample_for_reanalysis(REANALYZE_B)
    assert result == (None, None)


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


# ─── process_episode (pure Python, backend-independent) ────────────────────

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
            root_child_actions = rng.integers(0, A, (K, N), dtype=np.int32),
            root_child_q       = rng.random(K, dtype=np.float32),
            root_child_visits  = rng.random(K, dtype=np.float32),
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
    assert item.all_child_q.shape   == (U+1, K)


def test_process_episode_too_short():
    ep = Episode()
    for _ in range(3):
        ep.add_step(Transition(
            observation=np.zeros((N, OBS_SIZE), dtype=np.float32),
            action=np.zeros(N, dtype=np.int32),
            reward=0.0, done=False,
            policy_target=np.zeros((N, A), dtype=np.float32),
            value_target=0.0,
            root_child_actions=np.zeros((K, N), dtype=np.int32),
            root_child_q=np.zeros(K),
            root_child_visits=np.ones(K),
        ))
    items = process_episode(ep, unroll_steps=5, n_step=3,
                            discount_gamma=0.99, num_agents=N)
    assert items == []


def test_replay_item_all_child_fields_exist():
    """ReplayItem should have all_child_* fields for per-step Q-data."""
    item = ReplayItem(
        observation=np.zeros((3, 10)),
        actions=np.zeros((5, 3), dtype=np.int32),
        policy_target=np.zeros((6, 3, 9)),
        value_target=np.zeros((6, 3)),
        reward_target=np.zeros((5, 3)),
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
    N_, A_, K_, U_, T_ = 3, 9, 5, 5, 12
    obs_size = 18
    ep = Episode()
    for _ in range(T_):
        ep.add_step(Transition(
            observation=np.zeros((N_, obs_size)),
            action=np.zeros(N_, dtype=np.int32),
            reward=0.0,
            done=False,
            policy_target=np.ones((N_, A_)) / A_,
            value_target=0.0,
            root_child_actions=np.zeros((K_, N_), dtype=np.int32),
            root_child_q=np.zeros(K_),
            root_child_visits=np.ones(K_),
        ))
    items = process_episode(ep, unroll_steps=U_, n_step=5, discount_gamma=0.99, num_agents=N_)
    assert len(items) > 0
    it = items[0]
    assert it.all_child_q.shape == (U_ + 1, K_)
    assert it.all_child_actions.shape == (U_ + 1, K_, N_)
    assert it.all_child_visits.shape == (U_ + 1, K_)
    assert it.all_child_valid.shape == (U_ + 1,)
    assert it.all_child_valid.all(), "all positions should be valid since ep_len > U"


def test_process_episode_never_indexes_past_episode_end():
    """process_episode's sliding window must never read past the real episode
    end and silently reuse the terminal step's data for phantom future steps
    — the bug jaxzero's postmortem found in its own `make_target`
    (`docs/superpowers/plans/2026-05-05-fix-loss-plateau.md` in ../jaxzero/).
    Confirms this repo's process_episode avoids the bug class by
    construction: it only emits windows fully inside the episode, and
    returns none at all for episodes too short to fit one.
    """
    N_, obs_size, A_, K_ = 2, 4, 3, 4
    unroll_steps, n_step, discount = 5, 3, 0.99

    def make_transition(step: int) -> Transition:
        return Transition(
            observation=np.full((N_, obs_size), step, dtype=np.float32),
            action=np.zeros(N_, dtype=np.int32),
            reward=1.0,
            done=False,
            policy_target=np.ones((N_, A_), dtype=np.float32) / A_,
            value_target=0.5,
            root_child_actions=np.zeros((K_, N_), dtype=np.int32),
            root_child_q=np.zeros(K_),
            root_child_visits=np.ones(K_),
        )

    # Episode shorter than unroll_steps: must produce zero items, never a
    # padded/truncated one.
    short_episode = Episode()
    for t in range(3):
        short_episode.add_step(make_transition(t))
    assert process_episode(short_episode, unroll_steps, n_step, discount, N_) == []

    # Episode longer than unroll_steps: every returned item's window must be
    # fully inside [0, ep_len).
    ep_len = 9
    long_episode = Episode()
    for t in range(ep_len):
        long_episode.add_step(make_transition(t))

    items = process_episode(long_episode, unroll_steps, n_step, discount, N_)
    assert len(items) == ep_len - unroll_steps
    for i in range(len(items)):
        assert i + unroll_steps < ep_len, (
            f"item {i}'s window end {i + unroll_steps} reaches/exceeds ep_len {ep_len}"
        )
```

(Deleted relative to the original: `test_cpp_backend_loaded`, `test_replay_buffer_uses_cpp` — no more C++ backend to detect; `test_python_fallback_produces_same_shapes` — there's no longer a "fallback," `test_sample_returns_correct_shapes` already covers the single path; `test_process_episode_all_child_q_none_without_q` — the "no Q-data" code path it tested no longer exists.)

- [ ] **Step 13: Install `cpprb` and rebuild the test environment if needed.**

```bash
conda run -n mazero python -c "import cpprb" && echo "cpprb OK"
```
(`cpprb` is already installed in the `mazero` conda env at the time this plan was written — this step is a sanity check, not expected to require any install action.)

- [ ] **Step 14: Run the full test suite.**

```bash
conda run -n mazero pytest tests/ -v
```
Expected: all tests pass. `tests/test_replay_buffer.py` and `tests/test_mcts.py` are the ones affected by this task.

- [ ] **Step 15: Update `CLAUDE.md`.**
  - In "Package Layout", under `csrc/` — delete the entire `csrc/` block (all 6 lines: the directory header and its 5 file bullets), delete the `CMakeLists.txt` and `setup.py` bullets, delete the `benchmarks/` block.
  - Under `utils/`, the `replay_buffer.py` bullet's description changes from "ReplayBuffer, ReplayItem, Episode, Transition, process_episode" (unchanged — signatures didn't change) but drop `agent_order` from any inline shape comment if present.
  - In "Architecture" → "Training system" bullet for `ReplayBufferActor`, replace:
    > `ReplayBufferActor`: wraps `ReplayBuffer` (prioritized experience replay). Uses the C++ backend (`_replay_buffer_cpp`) when built: lock-free ring buffer + sum tree via `std::atomic`, CUDA pinned output buffers so `jax.device_put()` DMA's directly without a pageable copy. Falls back to a pure-Python/cpprb implementation if the `.so` is not present. Build with `python setup.py build_ext --inplace`.

    with:
    > `ReplayBufferActor`: wraps `ReplayBuffer` (prioritized experience replay), backed by `cpprb.PrioritizedReplayBuffer`. `jax.device_put()` uses the normal pageable-memory path (the prior C++ backend's CUDA-pinned DMA optimization was removed along with the custom extension — see git history for the swap).
  - In "Commands", remove the "Build C++ replay buffer extension" block:
    ```bash
    # Build C++ replay buffer extension (run once after cloning, or after modifying csrc/)
    pip install pybind11
    python setup.py build_ext --inplace
    ```
    and remove:
    ```bash
    # Benchmark C++ vs Python replay buffer
    python benchmarks/replay_buffer_benchmark.py
    ```
  - Under "Future Improvements" → "Utils", the "[done] C++ replay buffer" bullet — replace with: "**[done→removed 2026-07-29]** the custom C++/CUDA replay buffer was replaced by `cpprb` in the second simplification pass — see `docs/superpowers/specs/2026-07-29-second-simplification-pass-design.md`."

- [ ] **Step 16: Update `README.md`.**
  - In "Performance" → item 2 of the "~300× throughput gain" list, change "Async actor-learner pipeline — self-driving learner loop, async param sync, C++ replay buffer" to "Async actor-learner pipeline — self-driving learner loop, async param sync, prioritized replay buffer".
  - Replace the "**C++ replay buffer** (`csrc/`)" paragraph in "Implementation Highlights":
    > **C++ replay buffer** (`csrc/`): lock-free ring buffer + sum tree via `std::atomic`. Output buffers use `cudaMallocHost` pinned memory so `jax.device_put()` DMA's directly without a pageable copy (~50–200µs saved per step). Stratified PER sampling and Vitter's Algorithm R for uniform reanalysis. Falls back to pure-Python/cpprb if the `.so` is not built.

    with:
    > **Prioritized replay buffer** (`utils/replay_buffer.py`): backed by `cpprb.PrioritizedReplayBuffer` — stratified PER sampling with alpha/beta annealing, uniform (without-replacement) sampling for reanalysis.
  - In "Installation", remove:
    ```bash
    # Build C++ replay buffer (optional but recommended)
    pip install pybind11
    python setup.py build_ext --inplace
    ```

- [ ] **Step 17: Commit.**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: replace custom C++ replay buffer with cpprb

Deletes csrc/, CMakeLists.txt, setup.py's build step, and the committed
.so/build/ artifacts (87 tracked files) in favor of cpprb's
PrioritizedReplayBuffer. Loses the CUDA pinned-memory DMA optimization
in exchange for dropping an entire C++/CUDA toolchain from the repo.

Also removes the dead agent_order payload (produced, never read, but
threaded through ~15 call sites) and the dead None-guard branches left
over from the two MCTS planners deleted in the first simplification
pass -- the sole remaining planner always populates Q-data, so those
branches never executed.
EOF
)"
```

- [ ] **Step 18 (controller, not delegated to a subagent): smoke-test the new backend end-to-end.** After this task's commit lands, run a short real training loop to confirm the cpprb-backed buffer works under Ray (not just unit tests):
```bash
conda run -n mazero python train/muzero.py train.num_episodes=300 train.warmup_episodes=20 train.checkpoint_dir=/tmp/smoke_ckpt
```
Confirm it runs to completion without error and produces log lines showing episodes processed and training steps. This step is performed directly by the controller (it needs the real GPU/Ray environment), not dispatched as an SDD task.

---

### Task 2: Replace `mctx.RecurrentFnOutput` with a local NamedTuple; drop the `mctx` dependency

**Files:**
- Modify: `mcts/mcts_joint_osla.py`
- Modify: `tests/test_mcts.py`
- Modify: `requirements.txt`
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:**
- Produces: `mcts.mcts_joint_osla.RecurrentFnOutput` — a local NamedTuple with fields `reward`, `discount`, `prior_logits`, `value`, replacing `mctx.RecurrentFnOutput` as the return-type container from `recurrent_fn_batched`.

- [ ] **Step 1: Edit `mcts/mcts_joint_osla.py`** — remove `import mctx`, add a local NamedTuple, and use it in place of `mctx.RecurrentFnOutput`.

Remove this import line:
```python
import mctx
```

Add (near the top, after the other imports, before the `OSLATree` dataclass):
```python
class RecurrentFnOutput(NamedTuple):
    """Local replacement for mctx.RecurrentFnOutput — a plain container for
    the dynamics-network rollout output, no dependency on mctx's search."""
    reward: chex.Array
    discount: chex.Array
    prior_logits: chex.Array
    value: chex.Array
```
This requires adding `from typing import NamedTuple` to the file's imports.

In `recurrent_fn_batched` (inside `_osla_plan_single`), replace:
```python
        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=jnp.full_like(reward, gamma),
                prior_logits=joint_logits,
                value=value,
            ),
            out.hidden_state,
        )
```
with:
```python
        return (
            RecurrentFnOutput(
                reward=reward,
                discount=jnp.full_like(reward, gamma),
                prior_logits=joint_logits,
                value=value,
            ),
            out.hidden_state,
        )
```

Update the class docstring line that currently reads:
```python
    Uses a custom JAX MCTS loop (not mctx) with PUCT selection,
```
to:
```python
    Uses a custom JAX MCTS loop (own RecurrentFnOutput NamedTuple, no mctx
    dependency) with PUCT selection,
```

- [ ] **Step 2: Edit `tests/test_mcts.py`** — three test call sites build a fake `recurrent_fn` using `mctx.RecurrentFnOutput`; switch them to the new local type.

At each of the three locations (near lines 401, 563, 769 as of this writing — search for `import mctx` to find all three), replace:
```python
        import mctx
```
with:
```python
        from mcts.mcts_joint_osla import RecurrentFnOutput
```
and replace each corresponding:
```python
                mctx.RecurrentFnOutput(
```
with:
```python
                RecurrentFnOutput(
```
(3 occurrences total, one per test).

- [ ] **Step 3: Edit `requirements.txt`** — remove the `mctx` line.

- [ ] **Step 4: Run the full test suite.**
```bash
conda run -n mazero pytest tests/ -v
```

- [ ] **Step 5: Update `CLAUDE.md`.**
  - In "MCTS planners" → the `MCTSJointOSLAPlanner` bullet, replace:
    > `MCTSJointOSLAPlanner` (`joint`, the only planner, default everywhere): custom JAX MCTS (not mctx's search — `mctx.RecurrentFnOutput` is reused as a plain return-type container).

    with:
    > `MCTSJointOSLAPlanner` (the only planner, default everywhere): custom JAX MCTS with its own `RecurrentFnOutput` NamedTuple as a plain return-type container — no `mctx` dependency at all.

- [ ] **Step 6: Update `README.md`** — the "OS(λ) MCTS" paragraph in "Implementation Highlights" already says "No mctx dependency for this planner"; since mctx is now removed from the whole repo (not just unused by this planner), tighten it to: "No mctx dependency anywhere in this repo."

- [ ] **Step 7: Commit.**
```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: drop mctx dependency, use a local RecurrentFnOutput NamedTuple

mctx was only ever used as a 4-field container type, never for its
search algorithm. Removing the import and dependency entirely.
EOF
)"
```

---

### Task 3: Fold `MCTSPlanner` ABC into `MCTSJointOSLAPlanner`; collapse `planner_mode`; delete duplicate SMAX MCTS config; fix MCTS-search dropout

**Files:**
- Modify: `mcts/base.py`
- Modify: `mcts/mcts_joint_osla.py`
- Modify: `mcts/__init__.py`
- Modify: `config.py`
- Modify: `configs/mcts/default.yaml`, `configs/mcts/joint.yaml`
- Delete: `configs/mcts/smax.yaml`
- Modify: `actors/data_actor.py`, `actors/reanalyze_actor.py`, `eval.py`
- Modify: `tests/test_mcts.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Produces: `mcts.mcts_joint_osla.MCTSJointOSLAPlanner` — no longer inherits from an ABC; owns all config extraction directly in `__init__`. Public interface (`plan(params, rng_key, observation)`) unchanged.
- Consumes: nothing from Task 1/2 changes except the already-updated `MCTSPlanOutput` and `RecurrentFnOutput`.

- [ ] **Step 1: Rewrite `mcts/base.py`** — down to just the `MCTSPlanOutput` NamedTuple (the ABC and its config-extraction logic move into `MCTSJointOSLAPlanner`):

```python
# mcts/base.py

from typing import NamedTuple

import chex


class MCTSPlanOutput(NamedTuple):
    """Output of MCTSJointOSLAPlanner for a single planning step.

    Shapes (B=batch/envs, N=agents, A=actions, K=num_gumbel_samples):
        joint_action:       (B, N)    — chosen action index per agent
        policy_targets:     (B, N, A) — MCTS-improved policy targets for training
        root_value:         (B,)      — estimated value of the root state
        root_child_actions: (B, K, N) — K sampled joint actions as per-agent indices
        root_child_q:       (B, K)    — Q-value for each root child
        root_child_visits:  (B, K)    — visit count for each root child
    """
    joint_action:       chex.Array
    policy_targets:     chex.Array
    root_value:         chex.Array
    root_child_actions: chex.Array = None
    root_child_q:       chex.Array = None
    root_child_visits:  chex.Array = None
```

- [ ] **Step 2: Edit `mcts/mcts_joint_osla.py`.**

Change the import line:
```python
from mcts.base import MCTSPlanner, MCTSPlanOutput
```
to:
```python
from mcts.base import MCTSPlanOutput
```
and add:
```python
from utils.transforms import DiscreteSupport
```

Replace the `MCTSJointOSLAPlanner` class definition (from `class MCTSJointOSLAPlanner(MCTSPlanner):` through the end of `__init__` and the dead `_recurrent_fn` method) with:

```python
class MCTSJointOSLAPlanner:
    """
    Joint MCTS with OS(λ) backup.

    Uses a custom JAX MCTS loop (own RecurrentFnOutput NamedTuple, no mctx
    dependency) with PUCT selection, K sampled joint actions per node, and
    OS(λ) value aggregation. The only planner in this codebase.
    """

    def __init__(self, model: FlaxMAMuZeroNet, config: ExperimentConfig):
        self.model = model
        self.num_agents = config.train.num_agents
        self.action_space_size = model.action_space_size

        self.num_simulations = config.mcts.num_simulations
        self.max_depth_gumbel_search = config.mcts.max_depth_gumbel_search
        self.num_gumbel_samples = config.mcts.num_gumbel_samples
        self.discount_gamma = config.train.discount_gamma

        self.value_support = DiscreteSupport(
            min=-config.model.value_support_size,
            max=config.model.value_support_size,
        )
        self.reward_support = DiscreteSupport(
            min=-config.model.reward_support_size,
            max=config.model.reward_support_size,
        )

        self.dirichlet_alpha = config.mcts.dirichlet_alpha
        self.dirichlet_fraction = config.mcts.dirichlet_fraction

        self.joint_action_shape: tuple = (self.action_space_size,) * self.num_agents
        self.A_N = self.action_space_size ** self.num_agents
        self.mcts_rho = config.mcts.mcts_rho
        self.mcts_lambda = config.mcts.mcts_lambda
        self.pb_c_base = config.mcts.pb_c_base
        self.pb_c_init = config.mcts.pb_c_init
        self.value_delta_lb = config.mcts.value_delta_lb

    def plan(
        self, params, rng_key: chex.Array, observation: chex.Array
    ) -> MCTSPlanOutput:
        """Public entry point. JIT compilation is the caller's responsibility
        (DataActor wraps this with jax.jit at construction time)."""
        return self._plan_loop(params, rng_key, observation)

    def _plan_loop(
        self, params, rng_key: chex.Array, observation: chex.Array
    ) -> MCTSPlanOutput:
        """Vmapped single-env planning across batch."""
        B = observation.shape[0]
        rng_keys = jax.random.split(rng_key, B)

        plan_single = functools.partial(
            _osla_plan_single,
            model=self.model,
            num_simulations=self.num_simulations,
            K=self.num_gumbel_samples,
            A_N=self.A_N,
            max_depth=self.max_depth_gumbel_search,
            gamma=self.discount_gamma,
            rho=self.mcts_rho,
            lam=self.mcts_lambda,
            pb_c_base=self.pb_c_base,
            pb_c_init=self.pb_c_init,
            value_delta_lb=self.value_delta_lb,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_fraction=self.dirichlet_fraction,
            joint_action_shape=self.joint_action_shape,
            value_support=self.value_support,
            reward_support=self.reward_support,
        )

        # vmap over (rng_keys, observation) — params are shared (in_axes=None)
        results = jax.vmap(plan_single, in_axes=(None, 0, 0))(
            params, rng_keys, observation
        )

        # vmap produces leading B dim; _osla_plan_single adds an extra 1 dim — squeeze it
        return MCTSPlanOutput(
            joint_action=results.joint_action.squeeze(1),              # (B, N)
            policy_targets=results.policy_targets.squeeze(1),          # (B, N, A)
            root_value=results.root_value.squeeze(1),                  # (B,)
            root_child_actions=results.root_child_actions.squeeze(1),  # (B, K, N)
            root_child_q=results.root_child_q.squeeze(1),              # (B, K)
            root_child_visits=results.root_child_visits.squeeze(1),    # (B, K)
        )
```

(This drops the dead `self._recurrent_fn_jit = None` line and the dead `_recurrent_fn` method that only raised `NotImplementedError`.)

- [ ] **Step 3: Add MCTS-search determinism fix (dropout was active during tree search).** In `_osla_plan_single`, the root-inference call:
```python
    init_out = model.apply(
        {"params": params}, obs_batched, rngs={"dropout": init_key}
    )
```
becomes:
```python
    init_out = model.apply(
        {"params": params}, obs_batched, rngs={"dropout": init_key}, deterministic=True
    )
```

And in `recurrent_fn_batched`:
```python
        out = model.apply(
            {"params": params},
            embedding_batch,
            per_agent,
            method=model.recurrent_inference,
            rngs={"dropout": rng_key},
        )
```
becomes:
```python
        out = model.apply(
            {"params": params},
            embedding_batch,
            per_agent,
            method=model.recurrent_inference,
            rngs={"dropout": rng_key},
            deterministic=True,
        )
```

- [ ] **Step 4: Edit `mcts/__init__.py`.**
```python
from mcts.base import MCTSPlanOutput
from mcts.mcts_joint_osla import MCTSJointOSLAPlanner
```

- [ ] **Step 5: Edit `config.py`** — remove `planner_mode: str` from `MCTSConfig`.

- [ ] **Step 6: Edit `configs/mcts/default.yaml`** — remove the `planner_mode: joint` line.

- [ ] **Step 7: Edit `configs/mcts/joint.yaml`** — remove the `planner_mode: joint` line.

- [ ] **Step 8: Delete `configs/mcts/smax.yaml`** — every field it sets (`planner_mode: joint`, `num_simulations: 100`, `num_gumbel_samples: 10`, `mcts_rho: 0.25`) is already the value inherited from `default.yaml`; it's a byte-for-byte no-op preset.
```bash
git rm configs/mcts/smax.yaml
```

- [ ] **Step 9: Edit `actors/data_actor.py`** — replace the planner dispatch table with a direct import.

Change:
```python
        from model import FlaxMAMuZeroNet
        from mcts import MCTSJointOSLAPlanner
        from envs import make_vec_env_wrapper
```
(import line already correct — no change needed here.)

Replace:
```python
        model = FlaxMAMuZeroNet(config.model, action_size)
        planner_map = {"joint": MCTSJointOSLAPlanner}
        if config.mcts.planner_mode not in planner_map:
            raise ValueError(
                f"Unknown planner_mode '{config.mcts.planner_mode}'. "
                f"Choose from: {list(planner_map)}"
            )
        planner = planner_map[config.mcts.planner_mode](model=model, config=config)
```
with:
```python
        model = FlaxMAMuZeroNet(config.model, action_size)
        planner = MCTSJointOSLAPlanner(model=model, config=config)
```

- [ ] **Step 10: Edit `actors/reanalyze_actor.py`** — same pattern:
```python
        model = FlaxMAMuZeroNet(config.model, action_size)
        planner_map = {"joint": MCTSJointOSLAPlanner}
        if config.mcts.planner_mode not in planner_map:
            raise ValueError(
                f"Unknown planner_mode '{config.mcts.planner_mode}'. "
                f"Choose from: {list(planner_map.keys())}"
            )
        planner = planner_map[config.mcts.planner_mode](model=model, config=config)
```
becomes:
```python
        model = FlaxMAMuZeroNet(config.model, action_size)
        planner = MCTSJointOSLAPlanner(model=model, config=config)
```

- [ ] **Step 11: Edit `eval.py`** — same pattern:
```python
    planner_map = {"joint": MCTSJointOSLAPlanner}
    planner = planner_map[config.mcts.planner_mode](model=model, config=config)
```
becomes:
```python
    planner = MCTSJointOSLAPlanner(model=model, config=config)
```

- [ ] **Step 12: Edit `tests/test_mcts.py`** — remove `planner_mode="joint",` from both `MCTSConfig(...)` constructions (in the `test_config` fixture and in `test_mcts_config_has_osla_fields`).

- [ ] **Step 13: Run the full test suite.**
```bash
conda run -n mazero pytest tests/ -v
```

- [ ] **Step 14: Update `CLAUDE.md`.**
  - In "Package Layout", change the `mcts/base.py` bullet from "`# MCTSPlanner base class, MCTSPlanOutput`" to "`# MCTSPlanOutput NamedTuple (planner's return type)`".
  - Delete `configs/mcts/smax.yaml` from "Package Layout"'s config listing, and delete the "Key config group files" bullet `configs/mcts/smax.yaml — same as joint but 100 sims (closer to paper; noisier policy targets at 50)`.
  - In "MCTS planners" section, remove the `MCTSPlanner (base): ...` bullet entirely (the ABC no longer exists) and update the `MCTSJointOSLAPlanner` bullet to drop "Selected via `planner_mode="joint"`" language, replacing with: "the only planner — `MCTSJointOSLAPlanner` owns all of its own config extraction directly (no separate base class)."

- [ ] **Step 15: Commit.**
```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: fold MCTSPlanner ABC into MCTSJointOSLAPlanner, drop planner_mode

There's only ever been one planner since the first simplification pass
deleted the other two. The ABC, the planner_mode config field, and the
planner_map dispatch dict duplicated in 3 files were all one-implementation
indirection. Also deletes configs/mcts/smax.yaml (a byte-for-byte
duplicate of default.yaml) and turns on deterministic=True during MCTS
search so dropout doesn't perturb tree search.
EOF
)"
```

---

### Task 4: Drop MPE + IPPO/MAPPO baselines; commit the repo to SMAX-only

**Files:**
- Delete: `envs/mpe_env_wrapper.py`, `baselines/__init__.py`, `baselines/networks.py`, `baselines/ippo.py`, `baselines/mappo.py`, `train/ippo.py`, `train/mappo.py`, `configs/baseline/ippo.yaml`, `configs/baseline/mappo.yaml`, `configs/train/smax_3m.yaml`
- Modify: `envs/__init__.py`
- Modify: `configs/train/default.yaml`
- Modify: `eval.py`
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:**
- Produces: `envs.make_env_wrapper` / `envs.make_vec_env_wrapper` — same signatures, SMAX-only body.
- Produces: `configs/train/default.yaml` — becomes the SMAX 3m preset (was previously the MPE preset merged with `smax_3m.yaml`'s overrides).

- [ ] **Step 1: Delete MPE and baseline files.**
```bash
git rm envs/mpe_env_wrapper.py
git rm -r baselines/
git rm train/ippo.py train/mappo.py
git rm -r configs/baseline/
```

- [ ] **Step 2: Rewrite `envs/__init__.py`** to drop MPE routing:
```python
from envs.smax_env_wrapper import SMAXEnvWrapper, VecSMAXEnvWrapper


def make_env_wrapper(env_name: str, num_agents: int, max_steps: int):
    """Instantiate a single-env SMAX wrapper for the given scenario string
    (e.g. "3m", "2s3z", "8m")."""
    return SMAXEnvWrapper(env_name, num_agents, max_steps)


def make_vec_env_wrapper(env_name: str, num_agents: int, max_steps: int, num_envs: int):
    """Instantiate a vectorized SMAX wrapper for the given scenario string."""
    return VecSMAXEnvWrapper(env_name, num_agents, max_steps, num_envs)
```

- [ ] **Step 3: Rewrite `configs/train/default.yaml`** — fold `smax_3m.yaml`'s overrides directly in, since it's now the only train preset:
```yaml
# SMAX 3m scenario: 3 allies vs 3 scripted marine enemies.
# Usage: python train/muzero.py model=smax mcts=joint
#
# Hyperparameters aligned with MAZero paper (SMAC settings).
#
# Other scenarios: change env_name/num_agents/max_episode_steps, e.g.
#   8m:    num_agents=8, max_episode_steps=120
#   2s3z:  num_agents=5, max_episode_steps=150
#   3s5z:  num_agents=8, max_episode_steps=150

env_name: 3m
num_agents: 3
num_episodes: 100000
warmup_episodes: 50
log_interval: 100
num_actors: 3
max_episode_steps: 100  # JaxMARL HeuristicEnemySMAX 3m terminates at 100 steps
replay_buffer_size: 500000  # 100k overwrites good episodes too fast (~30min to full refresh)
replay_buffer_alpha: 0.6
replay_buffer_beta_start: 0.4
replay_buffer_beta_frames: 100000
batch_size: 256
learning_rate: 5e-4      # paper uses 5e-4 constant
param_update_interval: 100
end_lr_factor: 1.0       # no decay (1.0 = constant LR)
lr_warmup_steps: 0       # no warmup; paper uses constant LR
value_scale: 0.25
consistency_scale: 1.0
consistency_horizon: 1   # SPR multi-step horizon (1 = original k=1 only)
gradient_clip_norm: 5.0
unroll_steps: 5
n_step: 5                # paper uses 5; 10 causes NaN on sparse SMAX rewards
discount_gamma: 0.99
wandb_mode: disabled  # "online" or "disabled"
project_name: myzero1
checkpoint_dir: checkpoints
checkpoint_interval: 2500
num_envs_per_actor: 4  # 3 actors x 4 envs = 12 parallel envs
sync: false            # use synchronous training loop (simpler but ~2-3x slower)
ema_decay: 0.999       # EMA decay for target encoder (BYOL-style consistency loss)
num_reanalyze_actors: 1  # more starves data actors on a 12-core machine
reanalyze_batch_size: 64
debug: false
debug_interval: 100  # emit detailed debug logs every N learner training steps
awpo_alpha: 2.0      # AWPO advantage temperature (MAZero awac_lambda=2)
```
Then delete the now-folded-in file:
```bash
git rm configs/train/smax_3m.yaml
```
(`configs/config.yaml` needs no change — it already selects `train: default`, which now resolves to this SMAX config.)

- [ ] **Step 4: Edit `eval.py`** — remove the `is_smac` branch (the env is always SMAX now).

Replace:
```python
    is_smac = not config.train.env_name.startswith("MPE_")

    # Load checkpoint.
```
with:
```python
    # Load checkpoint.
```

Replace:
```python
    returns = []
    wins = [] if is_smac else None
    episodes_done = 0
```
with:
```python
    returns = []
    wins = []
    episodes_done = 0
```

Replace:
```python
            episode_returns += rewards_np * active
            if is_smac:
                won_np = np.array(step_result[4])
                # won_episode is only meaningful when the episode ends
                episode_won |= (won_np & dones_np & active)
```
with:
```python
            episode_returns += rewards_np * active
            won_np = np.array(step_result[4])
            # won_episode is only meaningful when the episode ends
            episode_won |= (won_np & dones_np & active)
```

Replace:
```python
        returns.extend(episode_returns.tolist())
        if is_smac:
            wins.extend(episode_won.tolist())
        episodes_done += B
```
with:
```python
        returns.extend(episode_returns.tolist())
        wins.extend(episode_won.tolist())
        episodes_done += B
```

Replace the reporting block:
```python
    if wins is not None:
        win_rate = np.mean(wins)
        msg += f"Win rate:         {win_rate:.1%}  ({int(np.sum(wins))}/{len(wins)})\n"
    msg += f"Elapsed:          {elapsed:.1f}s"
```
with:
```python
    win_rate = np.mean(wins)
    msg += f"Win rate:         {win_rate:.1%}  ({int(np.sum(wins))}/{len(wins)})\n"
    msg += f"Elapsed:          {elapsed:.1f}s"
```

Update the module docstring line "For SMAC environments also reports win rate." to "Also reports win rate." (it's unconditional now), and update the return-type comment on `_run_eval`: "Returns (returns, wins, ckpt_step) where wins is None for non-SMAC envs." → "Returns (returns, wins, ckpt_step)."

- [ ] **Step 5: Run the full test suite.**
```bash
conda run -n mazero pytest tests/ -v
```
(No test file references MPE/baselines directly, so this should be unaffected — this step confirms that.)

- [ ] **Step 6: Update `CLAUDE.md`.**
  - Delete the "Baselines" (`## Baselines`) section entirely.
  - In "Package Layout", delete: `envs/mpe_env_wrapper.py` bullet, `baselines/` block (all 3 file bullets), `train/ippo.py`/`train/mappo.py` bullets, `configs/baseline/` block, and the `configs/train/smax_3m.yaml` bullet (folded into `default.yaml` now).
  - In "Environment Wrappers", replace:
    > Routing: env names starting with `"MPE_"` → MPE wrappers; anything else (e.g. `"3m"`, `"2s3z"`) → SMAX wrappers.

    with:
    > SMAX-only: `env_name` is a JaxMARL SMAX scenario string (e.g. `"3m"`, `"2s3z"`, `"8m"`).
  - In "Current Training Target" → "Commands" fragment, update the primary command from:
    ```
    python train/muzero.py train=smax_3m model=smax mcts=joint
    ```
    to:
    ```
    python train/muzero.py model=smax mcts=joint
    ```
    (both here and in the top-level "## Commands" section — update every occurrence of `train=smax_3m model=smax mcts=joint` in the file).
  - In "## Commands", remove:
    ```bash
    # Run baselines
    python train/ippo.py
    python train/mappo.py
    ```
    and change:
    ```bash
    python train/muzero.py                                  # default config (MPE simple_spread)
    ```
    to:
    ```bash
    python train/muzero.py                                  # default config (SMAX 3m)
    ```
  - In "Target System" / "Current Training Target" intro sentence, no change needed (already says SMAX 3m is primary) — but the "Key config group files" bullet for `configs/train/smax_3m.yaml` should be deleted (folded into `configs/train/default.yaml`); update the `configs/train/default.yaml` bullet description accordingly.

- [ ] **Step 7: Update `README.md`.**
  - Delete the "## Results" section entirely (the MPE vs. MAPPO table — MAPPO no longer exists in this repo).
  - In "## Usage", replace:
    ```bash
    # MPE simple_spread (default)
    python train/muzero.py

    # SMAX 3m
    python train/muzero.py train=smax_3m model=smax mcts=joint

    # Override hyperparameters
    python train/muzero.py train.batch_size=512 mcts.num_simulations=100

    # Baselines
    python train/ippo.py
    python train/mappo.py
    ```
    with:
    ```bash
    # SMAX 3m (default)
    python train/muzero.py model=smax mcts=joint

    # Override hyperparameters
    python train/muzero.py train.batch_size=512 mcts.num_simulations=100
    ```

- [ ] **Step 8: Commit.**
```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: drop MPE env wrapper and IPPO/MAPPO baselines, SMAX-only

Neither had test coverage, and both are hardcoded to MPE (can't run
against SMAX). CLAUDE.md already stated SMAX 3m as the primary target;
this commits the repo to that fully. Folds configs/train/smax_3m.yaml
into configs/train/default.yaml since it's now the only train preset.
EOF
)"
```

---

### Task 5: Delete the synchronous training loop

**Files:**
- Modify: `training/loop.py`, `training/__init__.py`
- Modify: `actors/learner_actor.py`
- Modify: `config.py`
- Modify: `train/muzero.py`
- Modify: `configs/train/default.yaml`
- Modify: `tests/test_mcts.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Removes: `training.loop.run_training_loop_sync`, `LearnerActor.train()`, `TrainConfig.sync`.

- [ ] **Step 1: Edit `training/loop.py`** — delete the entire `run_training_loop_sync` function (from `def run_training_loop_sync(` through its final `logger.info("Training complete (sync).")` line).

- [ ] **Step 2: Edit `training/__init__.py`.**
```python
from training.loop import run_warmup, run_training_loop
```

- [ ] **Step 3: Edit `actors/learner_actor.py`** — delete the `train()` method and its comment:
```python
    # Keep a single-step entry point for the sync training loop.
    def train(self):
        return self._train_step()

```

- [ ] **Step 4: Edit `config.py`** — remove `sync: bool` from `TrainConfig`.

- [ ] **Step 5: Edit `train/muzero.py`.**

Change:
```python
from training import run_warmup, run_training_loop, run_training_loop_sync
```
to:
```python
from training import run_warmup, run_training_loop
```

Replace:
```python
    actor_tasks = run_warmup(data_actors, replay_buffer, config)
    if config.train.sync:
        run_training_loop_sync(learner, data_actors, replay_buffer, actor_tasks, config)
    else:
        run_training_loop(learner, data_actors, replay_buffer, actor_tasks, config, reanalyze_actors=reanalyze_actors)
```
with:
```python
    actor_tasks = run_warmup(data_actors, replay_buffer, config)
    run_training_loop(learner, data_actors, replay_buffer, actor_tasks, config, reanalyze_actors=reanalyze_actors)
```

- [ ] **Step 6: Edit `configs/train/default.yaml`** — remove the `sync: false            # use synchronous training loop (simpler but ~2-3x slower)` line.

- [ ] **Step 7: Edit `tests/test_mcts.py`** — remove `sync=True,` from the `TrainConfig(...)` construction in the `test_config` fixture.

- [ ] **Step 8: Run the full test suite.**
```bash
conda run -n mazero pytest tests/ -v
```

- [ ] **Step 9: Update `CLAUDE.md`** — in "Package Layout", change the `training/loop.py` bullet from:
> `training/loop.py               # run_warmup(), run_training_loop(), run_training_loop_sync()`

to:
> `training/loop.py               # run_warmup(), run_training_loop()`

- [ ] **Step 10: Commit.**
```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: delete the synchronous training loop

TrainConfig.sync is false in every shipped config; the async loop is
what's actually used and tuned. Removes run_training_loop_sync (~78
LOC) and LearnerActor.train(), its only caller.
EOF
)"
```

---

### Task 6: Delete observation normalization

**Files:**
- Delete: `utils/obs_norm.py`
- Modify: `config.py`
- Modify: `actors/learner_actor.py`
- Modify: `actors/data_actor.py`, `actors/reanalyze_actor.py`
- Modify: `configs/model/default.yaml`
- Modify: `tests/test_model.py`, `tests/test_mcts.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Removes: `ModelConfig.use_obs_normalization`, `utils.obs_norm.ObsRunningNorm`.
- Changes: `LearnerActor.get_params()` return shape from `{"params": ..., "norm_state": ...}` to `{"params": ...}` — this is the param-sync protocol; `DataActor`/`ReanalyzeActor` callers must be updated in lockstep in this same task.

- [ ] **Step 1: Delete `utils/obs_norm.py`.**
```bash
git rm utils/obs_norm.py
```

- [ ] **Step 2: Edit `config.py`** — remove `use_obs_normalization: bool` from `ModelConfig`.

- [ ] **Step 3: Edit `actors/learner_actor.py`.**

Remove the import:
```python
from utils.obs_norm import ObsRunningNorm
```

Remove:
```python
        self._use_obs_norm = config.model.use_obs_normalization
        self.obs_norm = ObsRunningNorm(obs_size) if self._use_obs_norm else None

```

Replace the checkpoint-restore block:
```python
            self.params = restored["params"]
            self.opt_state = restored["opt_state"]
            self.ema_params = restored.get("ema_params", self.params)
            self.train_step_count = int(restored["step"])
            if self._use_obs_norm and "obs_norm_mean" in restored:
                self.obs_norm = ObsRunningNorm.from_state(
                    {"mean": restored["obs_norm_mean"], "var": restored["obs_norm_var"]},
                    obs_size,
                )
            logger.info(
```
with:
```python
            self.params = restored["params"]
            self.opt_state = restored["opt_state"]
            self.ema_params = restored.get("ema_params", self.params)
            self.train_step_count = int(restored["step"])
            logger.info(
```

Replace the observation-normalization block in `_train_step()`:
```python
        # Observation normalization: update running stats then normalize the batch.
        # Done on CPU (numpy) before device_put so the GPU only sees clean inputs.
        if self._use_obs_norm:
            self.obs_norm.update(batch.observation)
            batch = dataclasses.replace(
                batch, observation=self.obs_norm.normalize(batch.observation)
            )

        # Dispatch H2D transfer immediately after getting the batch (JAX async —
```
with:
```python
        # Dispatch H2D transfer immediately after getting the batch (JAX async —
```
(Check whether `dataclasses` and `dataclasses.replace` are still used elsewhere in the file after this removal — if this was the only use, remove the now-unused `import dataclasses` at the top of the file too.)

Replace `_save_checkpoint()`:
```python
    def _save_checkpoint(self):
        import orbax.checkpoint as ocp
        state = {
            "params": self.params,
            "opt_state": self.opt_state,
            "ema_params": self.ema_params,
            "step": np.array(self.train_step_count),
        }
        if self._use_obs_norm:
            norm_s = self.obs_norm.state()
            state["obs_norm_mean"] = norm_s["mean"]
            state["obs_norm_var"] = norm_s["var"]
        self.ckpt_manager.save(self.train_step_count, args=ocp.args.StandardSave(state))
        self.ckpt_manager.wait_until_finished()
        logger.info(f"(Learner) Saved checkpoint at step {self.train_step_count}.")
```
with:
```python
    def _save_checkpoint(self):
        import orbax.checkpoint as ocp
        state = {
            "params": self.params,
            "opt_state": self.opt_state,
            "ema_params": self.ema_params,
            "step": np.array(self.train_step_count),
        }
        self.ckpt_manager.save(self.train_step_count, args=ocp.args.StandardSave(state))
        self.ckpt_manager.wait_until_finished()
        logger.info(f"(Learner) Saved checkpoint at step {self.train_step_count}.")
```

Replace `get_params()`:
```python
    def get_params(self):
        norm_state = self.obs_norm.state() if self._use_obs_norm else None
        return {"params": self.params, "norm_state": norm_state}
```
with:
```python
    def get_params(self):
        return {"params": self.params}
```

- [ ] **Step 4: Edit `actors/data_actor.py`.**

Replace:
```python
        result = ray.get(learner_actor.get_params.remote())
        self.params = result["params"]
        self.norm_state = result["norm_state"]  # None when use_obs_normalization=false
        self._param_future = None  # in-flight async param fetch, if any
```
with:
```python
        result = ray.get(learner_actor.get_params.remote())
        self.params = result["params"]
        self._param_future = None  # in-flight async param fetch, if any
```

Replace the normalization branch in `run_episode()`:
```python
            with self.profiler.time("plan"):
                # Normalize observations if running norm is enabled.
                # Applied on CPU (numpy) before JIT boundary so normalization
                # doesn't affect the JAX trace or add a device round-trip.
                if self.norm_state is not None:
                    obs_np = np.array(observations)
                    obs_norm = (obs_np - self.norm_state["mean"]) / np.sqrt(
                        self.norm_state["var"] + 1e-5
                    )
                    plan_obs = jnp.array(obs_norm)
                else:
                    plan_obs = observations
                # block_until_ready ensures we measure actual MCTS compute, not
```
with:
```python
            with self.profiler.time("plan"):
                plan_obs = observations
                # block_until_ready ensures we measure actual MCTS compute, not
```

Replace the param-sync resolve block:
```python
            if ready:
                result = ray.get(self._param_future)
                self.params = result["params"]
                self.norm_state = result["norm_state"]
                self._param_future = None
                self.episodes_since_update = 0
```
with:
```python
            if ready:
                result = ray.get(self._param_future)
                self.params = result["params"]
                self._param_future = None
                self.episodes_since_update = 0
```

- [ ] **Step 5: Edit `actors/reanalyze_actor.py`.**

Replace:
```python
        result = ray.get(learner_actor.get_params.remote())
        self.params = result["params"]
        self.norm_state = result["norm_state"]
        self._param_future = None
```
with:
```python
        result = ray.get(learner_actor.get_params.remote())
        self.params = result["params"]
        self._param_future = None
```

Replace the param-sync resolve block:
```python
                if ready:
                    result = ray.get(self._param_future)
                    self.params = result["params"]
                    self.norm_state = result["norm_state"]
                    self._param_future = None
```
with:
```python
                if ready:
                    result = ray.get(self._param_future)
                    self.params = result["params"]
                    self._param_future = None
```

Replace the observation-normalization block in `run_reanalyze()`:
```python
        with self.profiler.time("mcts_plan"):
            self.rng_key, plan_key = jax.random.split(self.rng_key)
            obs_arr = np.array(observations)
            if self.norm_state is not None:
                obs_arr = (obs_arr - self.norm_state["mean"]) / np.sqrt(
                    self.norm_state["var"] + 1e-5
                )
            plan_output = self.plan_fn(self.params, plan_key, jnp.array(obs_arr))
            jax.block_until_ready(plan_output.policy_targets)
```
with:
```python
        with self.profiler.time("mcts_plan"):
            self.rng_key, plan_key = jax.random.split(self.rng_key)
            plan_output = self.plan_fn(self.params, plan_key, jnp.array(observations))
            jax.block_until_ready(plan_output.policy_targets)
```

- [ ] **Step 6: Edit `configs/model/default.yaml`** — remove the `use_obs_normalization: false` line. (`configs/model/smax.yaml` doesn't set this field, so no change needed there.)

- [ ] **Step 7: Edit `tests/test_model.py`** — remove `use_obs_normalization=False,` from the `model_config` fixture's `ModelConfig(...)` construction.

- [ ] **Step 8: Edit `tests/test_mcts.py`** — remove `use_obs_normalization=False,` from the `test_config` fixture's `ModelConfig(...)` construction.

- [ ] **Step 9: Run the full test suite.**
```bash
conda run -n mazero pytest tests/ -v
```

- [ ] **Step 10: Update `CLAUDE.md`.**
  - Delete the "## Observation Normalization" section entirely.
  - In "Param sync protocol" paragraph, replace:
    > **Param sync protocol**: `get_params()` returns `{"params": ..., "norm_state": ...}` where `norm_state` is `None` when obs normalization is disabled, or a plain numpy dict `{mean, var, initialized}` that actors apply manually (avoiding a JAX import in the normalization path).

    with:
    > **Param sync protocol**: `get_params()` returns `{"params": ...}`.
  - In "Notable config fields added since original docs", remove the bullet: "`ModelConfig.use_obs_normalization: bool` — enables `ObsRunningNorm` in the learner; default `false`".
  - In "Package Layout", remove the `utils/obs_norm.py` bullet.
  - Under "Future Improvements" → "Utils", remove the "[done] Observation normalization" bullet (feature no longer exists).

- [ ] **Step 11: Commit.**
```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: delete observation normalization

use_obs_normalization is false in every shipped config. Removes
utils/obs_norm.py and the norm_state half of the param-sync protocol
across learner/data/reanalyze actors and checkpoint save/restore.
EOF
)"
```

---

### Task 7: Promote never-varied MCTS-math constants out of config

**Files:**
- Modify: `config.py`
- Modify: `configs/mcts/default.yaml`
- Modify: `mcts/osla_math.py`
- Modify: `mcts/mcts_joint_osla.py`
- Modify: `tests/test_mcts.py`

**Interfaces:**
- Produces: module-level constants `PB_C_BASE`, `PB_C_INIT`, `VALUE_DELTA_LB`, `DIRICHLET_ALPHA`, `DIRICHLET_FRACTION` in `mcts/osla_math.py`.
- Removes: `MCTSConfig.pb_c_base`, `MCTSConfig.pb_c_init`, `MCTSConfig.value_delta_lb`, `MCTSConfig.dirichlet_alpha`, `MCTSConfig.dirichlet_fraction`. `MCTSConfig.mcts_rho`, `MCTSConfig.mcts_lambda`, `MCTSConfig.num_simulations` stay as config fields (paper hyperparameters a user might actually sweep).
- Changes: `_run_single_sim` and `_osla_plan_single` in `mcts/mcts_joint_osla.py` drop the `pb_c_base`, `pb_c_init`, `value_delta_lb`, `dirichlet_alpha`, `dirichlet_fraction` parameters — they read the module constants directly instead. `compute_ucb_scores` in `mcts/osla_math.py` is UNCHANGED (still a pure function taking these as explicit arguments — only the config-to-caller plumbing is removed, not the function itself).

- [ ] **Step 1: Edit `config.py`** — remove these 5 fields from `MCTSConfig`:
```python
    pb_c_base: float = 19652.0   # UCB: log-visit exploration scaling (MuZero/MAZero standard)
    pb_c_init: float = 1.25     # UCB: base exploration constant (MuZero/MAZero standard)
    value_delta_lb: float = 0.01  # UCB: min-max Q-normalization floor, prevents divide-by-~0
```
and (further up in the dataclass) `dirichlet_alpha: float` and `dirichlet_fraction: float`.

The resulting `MCTSConfig` is:
```python
@dataclass(frozen=True)
class MCTSConfig:
    """Hyperparameters for the MCTS planner."""
    num_simulations: int
    max_depth_gumbel_search: int
    num_gumbel_samples: int
    mcts_rho: float = 0.25    # OS(λ): top quantile fraction to DISCARD (1-rho is the fraction kept)
    mcts_lambda: float = 0.8  # OS(λ): depth discount weight (lambda^depth)
```

- [ ] **Step 2: Edit `configs/mcts/default.yaml`** — remove the 5 corresponding lines (`dirichlet_alpha`, `dirichlet_fraction`, `pb_c_base`, `pb_c_init`, `value_delta_lb`). Resulting file:
```yaml
num_simulations: 100
max_depth_gumbel_search: 10
num_gumbel_samples: 10
mcts_rho: 0.25
mcts_lambda: 0.8
```

- [ ] **Step 3: Edit `mcts/osla_math.py`** — add module constants near the top (after the imports, before `compute_osla_value_jax`):
```python
# MuZero/MAZero standard values — never varied across any shipped config
# preset, so these are constants rather than config-tunable fields.
PB_C_BASE = 19652.0
PB_C_INIT = 1.25
VALUE_DELTA_LB = 0.01
DIRICHLET_ALPHA = 0.3
DIRICHLET_FRACTION = 0.25
```
(`compute_ucb_scores`'s signature and body are unchanged — it still takes `pb_c_base`, `pb_c_init`, `value_delta_lb` as explicit parameters; only its callers change what they pass.)

- [ ] **Step 4: Edit `mcts/mcts_joint_osla.py`.**

Update the import:
```python
from mcts.osla_math import (
    compute_osla_value_jax,
    compute_osla_value,
    compute_ucb_scores,
    _sample_k_actions,
    _logits_to_joint_logits,
    _joint_policy_to_marginal,
)
```
to:
```python
from mcts.osla_math import (
    compute_osla_value_jax,
    compute_osla_value,
    compute_ucb_scores,
    _sample_k_actions,
    _logits_to_joint_logits,
    _joint_policy_to_marginal,
    PB_C_BASE,
    PB_C_INIT,
    VALUE_DELTA_LB,
    DIRICHLET_ALPHA,
    DIRICHLET_FRACTION,
)
```

In `_run_single_sim`'s signature, remove the 3 trailing keyword parameters:
```python
def _run_single_sim(
    carry: SimCarry,
    sim_idx: chex.Array,          # [] int32
    params,
    recurrent_fn,                 # (params, rng, flat_action[1], embedding[1,N,D]) -> (RecurrentFnOutput, next_embedding[1,N,D])
    K: int,                       # static
    A_N: int,                     # static
    max_depth: int,               # static
    gamma: float,
    rho: float = 0.25,
    lam: float = 0.8,
    pb_c_base: float = 19652.0,
    pb_c_init: float = 1.25,
    value_delta_lb: float = 0.01,
) -> SimCarry:
```
becomes:
```python
def _run_single_sim(
    carry: SimCarry,
    sim_idx: chex.Array,          # [] int32
    params,
    recurrent_fn,                 # (params, rng, flat_action[1], embedding[1,N,D]) -> (RecurrentFnOutput, next_embedding[1,N,D])
    K: int,                       # static
    A_N: int,                     # static
    max_depth: int,               # static
    gamma: float,
    rho: float = 0.25,
    lam: float = 0.8,
) -> SimCarry:
```

Inside `_best_ucb`, the `compute_ucb_scores` call:
```python
        ucb = compute_ucb_scores(
            child_q_baseline_diff, child_visits, prior_probs, parent_visits,
            carry.qmin, carry.qmax, pb_c_base, pb_c_init, value_delta_lb,
        )
```
becomes:
```python
        ucb = compute_ucb_scores(
            child_q_baseline_diff, child_visits, prior_probs, parent_visits,
            carry.qmin, carry.qmax, PB_C_BASE, PB_C_INIT, VALUE_DELTA_LB,
        )
```

In `_osla_plan_single`'s signature, remove the `pb_c_base`, `pb_c_init`, `value_delta_lb`, `dirichlet_alpha`, `dirichlet_fraction` parameters:
```python
def _osla_plan_single(
    params,
    rng_key: chex.Array,
    observation: chex.Array,          # [N, obs_size]
    model,
    num_simulations: int,             # static
    K: int,                           # static (num_gumbel_samples)
    A_N: int,                         # static (action_space_size^num_agents)
    max_depth: int,                   # static
    gamma: float,
    rho: float,
    lam: float,
    pb_c_base: float,
    pb_c_init: float,
    value_delta_lb: float,
    dirichlet_alpha: float,
    dirichlet_fraction: float,
    joint_action_shape: tuple,        # static
    value_support,
    reward_support,
) -> MCTSPlanOutput:
```
becomes:
```python
def _osla_plan_single(
    params,
    rng_key: chex.Array,
    observation: chex.Array,          # [N, obs_size]
    model,
    num_simulations: int,             # static
    K: int,                           # static (num_gumbel_samples)
    A_N: int,                         # static (action_space_size^num_agents)
    max_depth: int,                   # static
    gamma: float,
    rho: float,
    lam: float,
    joint_action_shape: tuple,        # static
    value_support,
    reward_support,
) -> MCTSPlanOutput:
```

Inside its body, the Dirichlet-noise construction:
```python
    dir_noise = jax.random.dirichlet(dir_key, alpha=jnp.full(A_N, dirichlet_alpha))
    root_probs = jax.nn.softmax(root_joint_logits)
    noisy_probs = (1 - dirichlet_fraction) * root_probs + dirichlet_fraction * dir_noise
```
becomes:
```python
    dir_noise = jax.random.dirichlet(dir_key, alpha=jnp.full(A_N, DIRICHLET_ALPHA))
    root_probs = jax.nn.softmax(root_joint_logits)
    noisy_probs = (1 - DIRICHLET_FRACTION) * root_probs + DIRICHLET_FRACTION * dir_noise
```

Its internal `sim_step` closure:
```python
    def sim_step(sim_idx, carry: SimCarry) -> SimCarry:
        return _run_single_sim(
            carry, sim_idx, params, recurrent_fn_batched,
            K, A_N, max_depth, gamma, rho, lam,
            pb_c_base, pb_c_init, value_delta_lb,
        )
```
becomes:
```python
    def sim_step(sim_idx, carry: SimCarry) -> SimCarry:
        return _run_single_sim(
            carry, sim_idx, params, recurrent_fn_batched,
            K, A_N, max_depth, gamma, rho, lam,
        )
```

In `MCTSJointOSLAPlanner.__init__`, remove:
```python
        self.dirichlet_alpha = config.mcts.dirichlet_alpha
        self.dirichlet_fraction = config.mcts.dirichlet_fraction
```
and remove:
```python
        self.pb_c_base = config.mcts.pb_c_base
        self.pb_c_init = config.mcts.pb_c_init
        self.value_delta_lb = config.mcts.value_delta_lb
```

In `_plan_loop`'s `functools.partial(_osla_plan_single, ...)` call, remove the corresponding 5 kwargs:
```python
        plan_single = functools.partial(
            _osla_plan_single,
            model=self.model,
            num_simulations=self.num_simulations,
            K=self.num_gumbel_samples,
            A_N=self.A_N,
            max_depth=self.max_depth_gumbel_search,
            gamma=self.discount_gamma,
            rho=self.mcts_rho,
            lam=self.mcts_lambda,
            pb_c_base=self.pb_c_base,
            pb_c_init=self.pb_c_init,
            value_delta_lb=self.value_delta_lb,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_fraction=self.dirichlet_fraction,
            joint_action_shape=self.joint_action_shape,
            value_support=self.value_support,
            reward_support=self.reward_support,
        )
```
becomes:
```python
        plan_single = functools.partial(
            _osla_plan_single,
            model=self.model,
            num_simulations=self.num_simulations,
            K=self.num_gumbel_samples,
            A_N=self.A_N,
            max_depth=self.max_depth_gumbel_search,
            gamma=self.discount_gamma,
            rho=self.mcts_rho,
            lam=self.mcts_lambda,
            joint_action_shape=self.joint_action_shape,
            value_support=self.value_support,
            reward_support=self.reward_support,
        )
```

- [ ] **Step 5: Edit `tests/test_mcts.py`.**
  - Remove `dirichlet_alpha=0.3,` and `dirichlet_fraction=0.25,` from both `MCTSConfig(...)` constructions (the `test_config` fixture and `test_mcts_config_has_osla_fields`).
  - At the two `_run_single_sim(...)` call sites that pass `pb_c_base=19652.0, pb_c_init=1.25, value_delta_lb=0.01, rho=0.25, lam=0.8,` (search for that exact substring — it appears twice, identically), replace it with `rho=0.25, lam=0.8,` in both places. (Calls to `compute_ucb_scores(...)` in the same file are UNCHANGED — that function's signature didn't change.)

- [ ] **Step 6: Run the full test suite.**
```bash
conda run -n mazero pytest tests/ -v
```

- [ ] **Step 7: Commit.**
```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: promote never-varied MCTS-math constants out of config

pb_c_base, pb_c_init, value_delta_lb, dirichlet_alpha, and
dirichlet_fraction were config fields but identical across every
shipped preset. Moves them to module constants in mcts/osla_math.py;
mcts_rho, mcts_lambda, and num_simulations stay config-tunable since
those are the paper hyperparameters someone might actually sweep.
EOF
)"
```

---

### Task 8: Rename Gumbel-search leftovers

**Files:**
- Modify: `config.py`
- Modify: `configs/mcts/default.yaml`, `configs/mcts/joint.yaml`
- Modify: `mcts/mcts_joint_osla.py`
- Modify: `mcts/base.py`
- Modify: `actors/loss.py`
- Modify: `actors/replay_buffer_actor.py`
- Modify: `tests/test_mcts.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Renames: `MCTSConfig.num_gumbel_samples` → `MCTSConfig.num_sampled_actions`; `MCTSConfig.max_depth_gumbel_search` → `MCTSConfig.max_search_depth`.

mctx's Gumbel search was removed in Task 2 (this repo never used mctx's search algorithm to begin with — only its container type). `num_gumbel_samples` really means "K sampled joint actions per node"; `max_depth_gumbel_search` really means "max search depth." Renaming to match.

- [ ] **Step 1: Edit `config.py`** — rename the two fields in `MCTSConfig`:
```python
@dataclass(frozen=True)
class MCTSConfig:
    """Hyperparameters for the MCTS planner."""
    num_simulations: int
    max_search_depth: int
    num_sampled_actions: int
    mcts_rho: float = 0.25    # OS(λ): top quantile fraction to DISCARD (1-rho is the fraction kept)
    mcts_lambda: float = 0.8  # OS(λ): depth discount weight (lambda^depth)
```

- [ ] **Step 2: Edit `configs/mcts/default.yaml`** — rename the two keys:
```yaml
num_simulations: 100
max_search_depth: 10
num_sampled_actions: 10
mcts_rho: 0.25
mcts_lambda: 0.8
```

- [ ] **Step 3: Edit `configs/mcts/joint.yaml`** — rename `num_gumbel_samples` → `num_sampled_actions`:
```yaml
defaults:
  - default

num_simulations: 50     # MAZero uses 50 for SMAC; halves MCTS cost vs default 100
mcts_rho: 0.25          # MAZero paper value: keep top 75% of sims
num_sampled_actions: 5  # MAZero uses 5 sampled joint actions per node (default 10)
```

- [ ] **Step 4: Edit `mcts/mcts_joint_osla.py`.**

In `MCTSJointOSLAPlanner.__init__`, rename:
```python
        self.max_depth_gumbel_search = config.mcts.max_depth_gumbel_search
        self.num_gumbel_samples = config.mcts.num_gumbel_samples
```
to:
```python
        self.max_search_depth = config.mcts.max_search_depth
        self.num_sampled_actions = config.mcts.num_sampled_actions
```

In `_plan_loop`'s `functools.partial` call, rename the corresponding kwargs:
```python
            K=self.num_gumbel_samples,
            A_N=self.A_N,
            max_depth=self.max_depth_gumbel_search,
```
to:
```python
            K=self.num_sampled_actions,
            A_N=self.A_N,
            max_depth=self.max_search_depth,
```

Update the comment in `_osla_plan_single`'s signature: `K: int,                           # static (num_gumbel_samples)` → `K: int,                           # static (num_sampled_actions)`.

- [ ] **Step 5: Edit `mcts/base.py`** — update the `MCTSPlanOutput` docstring's shape-key line: `Shapes (B=batch/envs, N=agents, A=actions, K=num_gumbel_samples):` → `Shapes (B=batch/envs, N=agents, A=actions, K=num_sampled_actions):`.

- [ ] **Step 6: Edit `actors/loss.py`** — rename the comment and config access:
```python
    K = config.mcts.num_gumbel_samples   # static: K sampled joint actions per MCTS node
```
to:
```python
    K = config.mcts.num_sampled_actions  # static: K sampled joint actions per MCTS node
```

- [ ] **Step 7: Edit `actors/replay_buffer_actor.py`** — rename:
```python
        K   = config.mcts.num_gumbel_samples
```
to:
```python
        K   = config.mcts.num_sampled_actions
```

- [ ] **Step 8: Edit `tests/test_mcts.py`** — rename `max_depth_gumbel_search=3,` → `max_search_depth=3,` and `num_gumbel_samples=4,` → `num_sampled_actions=4,` in both `MCTSConfig(...)` constructions (`test_config` fixture and `test_mcts_config_has_osla_fields`). Also update `ReplayItem`/`Transition` shape docstrings in `utils/replay_buffer.py` if they mention `K = num_gumbel_samples` (already updated to `num_sampled_actions` if Task 1 landed after this — since Task 1 runs BEFORE this task in execution order, its docstring currently reads `K = num_gumbel_samples`; update it here too):
  - In `utils/replay_buffer.py`'s `ReplayItem` docstring: `K = num_gumbel_samples` → `K = num_sampled_actions`.

- [ ] **Step 9: Run the full test suite.**
```bash
conda run -n mazero pytest tests/ -v
```

- [ ] **Step 10: Update `CLAUDE.md`** — in the "MAZero Reference Implementation" → "Paper Hyperparameters" table, the comment line:
```
sampled_action_times: 10  # K (joint actions sampled per node); our num_gumbel_samples=5 for SMAX
```
becomes:
```
sampled_action_times: 10  # K (joint actions sampled per node); our num_sampled_actions=5 for SMAX
```
Also check the "Temperature annealing" bullet under "Future Improvements" → "MCTS", which reads "`num_gumbel_samples` (K, the number of joint actions sampled per node in `MCTSJointOSLAPlanner`)" — update to "`num_sampled_actions` (K, ...)".

- [ ] **Step 11: Commit.**
```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: rename num_gumbel_samples/max_depth_gumbel_search

Both names are Gumbel-MuZero leftovers from before this planner had
its own custom JAX MCTS loop (mctx's Gumbel search was never used and
was fully removed in an earlier commit). Renamed to what they actually
mean: num_sampled_actions (K joint actions sampled per node) and
max_search_depth.
EOF
)"
```

---

### Task 9: Dedup the optax schedule/optimizer construction between `eval.py` and `learner_actor.py`

**Files:**
- Modify: `actors/loss.py`
- Modify: `actors/learner_actor.py`
- Modify: `eval.py`

**Interfaces:**
- Produces: `actors.loss.make_optimizer(config: ExperimentConfig) -> (optax.GradientTransformation, optax.Schedule)`.
- Consumes: `LearnerActor.__init__` and `eval.py`'s `_run_eval` both call this instead of duplicating the construction.

`eval.py`'s copy of this logic didn't clamp `warmup_steps`/`decay_steps` the way `learner_actor.py`'s does (`learner_actor.py` guards against short diagnostic runs producing a negative `decay_steps`); sharing one factory fixes that latent inconsistency as a side effect.

- [ ] **Step 1: Add `make_optimizer` to `actors/loss.py`** (after the imports, before `scale_grad_half`):
```python
def make_optimizer(config: ExperimentConfig):
    """Builds the optimizer and its LR schedule from TrainConfig.

    Shared by LearnerActor and eval.py so both construct an identically
    shaped opt_state to restore checkpoints against.
    """
    import optax

    lr = config.train.learning_rate
    # Clamp warmup so short diagnostic runs (num_episodes=300) don't produce
    # a negative decay_steps and crash cosine_decay_schedule.
    warmup = min(config.train.lr_warmup_steps, config.train.num_episodes // 2)
    decay = max(1, config.train.num_episodes - warmup)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=decay,
        end_value=lr * config.train.end_lr_factor,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.train.gradient_clip_norm),
        optax.adamw(learning_rate=lr_schedule),
    )
    return optimizer, lr_schedule
```

- [ ] **Step 2: Edit `actors/learner_actor.py`** — replace the inline construction with a call to the shared factory.

Change the import line:
```python
from actors.loss import make_train_step
```
to:
```python
from actors.loss import make_train_step, make_optimizer
```

Replace:
```python
        lr = config.train.learning_rate
        # Clamp warmup so short diagnostic runs (num_episodes=300) don't produce
        # a negative decay_steps and crash cosine_decay_schedule.
        _warmup = min(config.train.lr_warmup_steps, config.train.num_episodes // 2)
        _decay  = max(1, config.train.num_episodes - _warmup)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=_warmup,
            decay_steps=_decay,
            end_value=lr * config.train.end_lr_factor,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.train.gradient_clip_norm),
            optax.adamw(learning_rate=lr_schedule),
        )
        self.opt_state = optimizer.init(self.params)
        self.lr_schedule = lr_schedule
```
with:
```python
        optimizer, lr_schedule = make_optimizer(config)
        self.opt_state = optimizer.init(self.params)
        self.lr_schedule = lr_schedule
```
(The `import optax` at the top of `__init__` stays — `optimizer.init` still needs the `optax` chain object returned by the factory, and other parts of the file may still reference `optax` directly; check with a grep for `optax\.` in the file before removing the import, keep it if any remain.)

- [ ] **Step 3: Edit `eval.py`** — replace the duplicated construction the same way.

Change the import section (add):
```python
    from actors.loss import make_optimizer
```

Replace:
```python
    import optax
    lr = config.train.learning_rate
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=config.train.lr_warmup_steps,
        decay_steps=config.train.num_episodes - config.train.lr_warmup_steps,
        end_value=lr * config.train.end_lr_factor,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.train.gradient_clip_norm),
        optax.adamw(learning_rate=lr_schedule),
    )
    opt_state = optimizer.init(params)
```
with:
```python
    optimizer, _ = make_optimizer(config)
    opt_state = optimizer.init(params)
```

- [ ] **Step 4: Run the full test suite.**
```bash
conda run -n mazero pytest tests/ -v
```

- [ ] **Step 5: Commit.**
```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: extract shared make_optimizer, dedup eval.py vs learner_actor.py

eval.py's copy of the optax schedule/optimizer construction lacked
learner_actor.py's warmup-clamping guard against short diagnostic runs.
One factory in actors/loss.py now backs both, fixing that inconsistency
as a side effect of removing the duplication.
EOF
)"
```

---

## Self-Review Notes

- **cpprb already installed in the target conda env** (`mazero`) at the time this plan was written, despite the audit's note that it "isn't even installed in this env" — that referred to the sandbox environment used for research, not the user's real dev environment. Task 1 Step 13 is a sanity check, not expected to require any install action; if the mazero env changes before this plan executes, `pip install cpprb` (already a `requirements.txt` line) restores it.
- **Task 1 is intentionally one large task**, not split along the original design spec's "Section 1 dead-code" vs. "Section 3 replay buffer" boundary. `agent_order` removal, the Q-data required-fields cleanup, and the cpprb migration all collide on the same two files (`utils/replay_buffer.py`, `tests/test_replay_buffer.py`) and the same dataclasses. Splitting them would force either wasted edits to the doomed C++ backend (to strip `agent_order` from bindings that are deleted moments later) or a broken-test intermediate state between sub-tasks. One task, reviewed as a whole, avoids both.
- **`all_child_valid` is deliberately NOT removed** in Task 1, even though this plan's own dead-branch removal made it provably always-`True` in the current code paths (every `Transition` now always carries Q-data, so `process_episode` always sets it to all-`True`, so `ReplayBufferActor`'s sidecar never sees a `False` from a real add). The audit that produced the design spec explicitly called this mechanism "not dead — it legitimately masks cold ring-buffer slots," and removing it entirely would be a deeper redesign of the Q-data sidecar's indexing invariants than this pass signed up for. The `jnp.where(q_valid, ...)` masking in `actors/loss.py` stays as a defensive no-op; if a future pass wants to remove it, that's a separate, narrower decision with its own review.
- **Task ordering matters**: Tasks 3, 7, and 8 all touch `mcts/mcts_joint_osla.py`'s `MCTSJointOSLAPlanner.__init__`/`_plan_loop` and must run in this exact order — Task 3 folds the ABC into the concrete class first; Task 7 and Task 8's diffs are written against that already-folded structure, not the original ABC-based one. Do not reorder these three tasks.
- **Task 5 (sync-loop deletion) must run after Task 4** (MPE/baselines removal) — Task 4 rewrites `configs/train/default.yaml` in full (folding in `smax_3m.yaml`), and Task 5's one-line removal from that file is written against Task 4's post-fold version, not the original MPE-oriented file.
