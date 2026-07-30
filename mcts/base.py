# mcts/base.py

from typing import NamedTuple

import chex


class MCTSPlanOutput(NamedTuple):
    """Output of MCTSJointOSLAPlanner for a single planning step.

    Shapes (B=batch/envs, N=agents, A=actions, K=num_sampled_actions):
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
