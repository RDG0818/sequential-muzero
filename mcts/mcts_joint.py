# mcts_joint.py
import jax
import jax.numpy as jnp
import mctx
import functools
from typing import Tuple
from model.model import FlaxMAMuZeroNet
import utils.utils as utils
from config import ExperimentConfig 
from mcts.base import MCTSPlanner, MCTSPlanOutput

class MCTSJointPlanner(MCTSPlanner):
    """
    An MCTS planner that searches over the combinatorial joint action space.
    It uses Gumbel MuZero policy to sample a subset of joint actions to consider.
    """
    def __init__(
        self,
        model: FlaxMAMuZeroNet,
        config: ExperimentConfig
        
    ):
        super().__init__(model, config)

        def recurrent_fn(params, rng_key, action, embedding):
            """
            Batched one-step rollout for a joint action in MuZero search.
            `action` is a scalar index for the joint action.
            `embedding` is the current latent state.
            """
            latent = embedding

            # Decode the joint action index into a per-agent action array
            joint_action_tuple = jnp.unravel_index(action, self.joint_action_shape)
            # Batch size is implicitly 1 during planning
            joint_action = jnp.array(joint_action_tuple).T.reshape(latent.shape[0], self.num_agents)

            # Dynamics + prediction
            model_output = self.model.apply(
                {'params': params}, latent, joint_action,
                method=self.model.recurrent_inference,
                rngs={'dropout': rng_key}
            )
            next_latent = model_output.hidden_state
            reward_logits = model_output.reward_logits
            multi_logits = model_output.policy_logits
            value_logits = model_output.value_logits
            value = utils.support_to_scalar(value_logits, self.value_support)
            reward = utils.support_to_scalar(reward_logits, self.reward_support)

            # Convert per-agent logits (B, N, A) to joint logits (B, A^N)
            joint_logits = self._logits_to_joint_logits(multi_logits)

            out = mctx.RecurrentFnOutput(
                reward=reward,
                discount=jnp.full_like(reward, 0.99),
                prior_logits=joint_logits,
                value=value
            )
            return out, next_latent

        self._recurrent_fn = recurrent_fn
        self.plan_jit = jax.jit(self._plan_loop)

    def _logits_to_joint_logits(self, logits: jnp.ndarray) -> jnp.ndarray:
        """Converts per-agent logits (B, N, A) to joint action logits (B, A^N)."""
        # Assumes independence: log P(j1,...,jN) = sum(log P(ji))
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        
        # Start with the first agent's log_probs
        joint_log_probs = log_probs[:, 0, :]
        # Iteratively combine with other agents
        for i in range(1, self.num_agents):
            next_log_probs = log_probs[:, i, :]
            joint_log_probs = joint_log_probs[:, :, None] + next_log_probs[:, None, :]
            joint_log_probs = joint_log_probs.reshape(logits.shape[0], -1)
            
        return joint_log_probs

    def _joint_policy_to_marginal(self, joint_policy: jnp.ndarray) -> jnp.ndarray:
        """Converts a joint policy (B, A^N) to marginal policies (B, N, A)."""
        batch_size = joint_policy.shape[0]
        # Reshape to (B, A, A, ..., A)
        reshaped_policy = joint_policy.reshape(batch_size, *self.joint_action_shape)
        
        marginals = []
        for i in range(self.num_agents):
            # To get marginal for agent i, sum over all other agent axes
            axes_to_sum = tuple(j for j in range(1, self.num_agents + 1) if j != i + 1)
            marginal = jnp.sum(reshaped_policy, axis=axes_to_sum)
            marginals.append(marginal)
            
        return jnp.stack(marginals, axis=1)

    def plan(self, params, rng_key, observation):
        return self.plan_jit(params, rng_key, observation)

    def _plan_loop(self, params, rng_key, observation):
        init_key, gumbel_key = jax.random.split(rng_key)
        # 1. Initial inference from the model
        model_output = self.model.apply(
            {'params': params}, observation,
            rngs={'dropout': init_key}
        )
        root_latent = model_output.hidden_state
        root_logits_per_agent = model_output.policy_logits
        root_value_logits = model_output.value_logits

        # 2. Convert per-agent logits to joint action logits
        root_joint_logits = self._logits_to_joint_logits(root_logits_per_agent)
        root_value = utils.support_to_scalar(root_value_logits, self.value_support)
        
        # 3. Define the MCTS root and run the search
        root = mctx.RootFnOutput(
            prior_logits=root_joint_logits, 
            value=root_value,
            embedding=root_latent
        )
        
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=gumbel_key,
            root=root,
            recurrent_fn=self._recurrent_fn,
            num_simulations=self.num_simulations,
            max_depth=self.max_depth_gumbel_search,
            max_num_considered_actions=self.num_gumbel_samples,
            qtransform=functools.partial(mctx.qtransform_completed_by_mix_value, use_mixed_value=True)
        )
        
        # 4. Process the MCTS output
        # Action is a scalar index for the chosen joint action
        chosen_joint_action_index = policy_output.action
        joint_action_tuple = jnp.unravel_index(chosen_joint_action_index, self.joint_action_shape)
        final_joint_action = jnp.array(joint_action_tuple).squeeze(axis=-1)

        # Action weights are the visit counts for the joint actions
        joint_policy_target = policy_output.action_weights
        
        # Convert joint policy target back to per-agent marginals for the loss function
        marginal_policy_targets = self._joint_policy_to_marginal(joint_policy_target[None, :]).squeeze(0)

        return MCTSPlanOutput(
            joint_action=final_joint_action,
            policy_targets=marginal_policy_targets,
            root_value=root_value.squeeze().astype(float)
        )