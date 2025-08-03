# mcts_joint.py
import jax
import jax.numpy as jnp
import mctx
import functools
from typing import Tuple
import chex
from model.model import FlaxMAMuZeroNet
import utils.utils as utils
from config import ExperimentConfig
from mcts.base import MCTSPlanner, MCTSPlanOutput


class MCTSJointPlanner(MCTSPlanner):
    """
    An MCTS planner that searches over the combinatorial joint action space.

    This planner uses a Gumbel MuZero policy to sample a subset of joint actions,
    making the search in the large joint action space more tractable.
    """

    def __init__(self, model: FlaxMAMuZeroNet, config: ExperimentConfig):
        """Initializes the joint MCTS planner."""
        super().__init__(model, config)
        
        self.joint_action_shape: Tuple[int, ...] = (self.action_space_size,) * self.num_agents #(A, A, ..., A) for N agents.

        self._recurrent_fn_jit = jax.jit(self._recurrent_fn)
        self.plan_jit = jax.jit(self._plan_loop)

    def _recurrent_fn(self, params, rng_key, action: chex.Array, embedding: chex.Array) -> Tuple[mctx.RecurrentFnOutput, chex.Array]:
        """
        Defines a single-step, batched rollout within the MCTS search.

        This function is called repeatedly by the `mctx` search algorithm.

        Args:
            params: The parameters of the MuZero model.
            rng_key: A JAX random key for stochastic operations.
            action: A scalar index representing the joint action to take.
            embedding: The current latent state of the environment.

        Returns:
            A tuple containing the `mctx.RecurrentFnOutput` and the next latent state.
        """
        latent = embedding

        # Decode the joint action index into a per-agent action array
        joint_action_tuple = jnp.unravel_index(action, self.joint_action_shape)
        # The batch size is implicitly 1 during planning
        joint_action = jnp.array(joint_action_tuple).T.reshape(
            latent.shape[0], self.num_agents
        )

        # Use the dynamics model to predict the next state, reward, and policy
        model_output = self.model.apply(
            {"params": params},
            latent,
            joint_action,
            method=self.model.recurrent_inference,
            rngs={"dropout": rng_key},
        )
        next_latent = model_output.hidden_state
        reward_logits = model_output.reward_logits
        multi_logits = model_output.policy_logits  # Per-agent policy logits
        value_logits = model_output.value_logits

        # Convert scalar supports to scalar values
        value = utils.support_to_scalar(value_logits, self.value_support)
        reward = utils.support_to_scalar(reward_logits, self.reward_support)

        # Convert per-agent policy logits to joint action logits for the search
        joint_logits = self._logits_to_joint_logits(multi_logits)

        # The output required by mctx
        output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=jnp.full_like(reward, self.config.train.discount_gamma),
            prior_logits=joint_logits,
            value=value,
        )
        return output, next_latent

    def _plan_loop(self, params, rng_key, observation) -> MCTSPlanOutput:
        """
        The main planning logic that orchestrates the joint MCTS search.

        Args:
            params: The parameters of the MuZero model.
            rng_key: A JAX random key for the search.
            observation: The current environment observation.

        Returns:
            An `MCTSPlanOutput` object containing the chosen action, policy targets, and root value.
        """
        init_key, gumbel_key = jax.random.split(rng_key)
        
        model_output = self.model.apply(
            {"params": params}, observation, rngs={"dropout": init_key}
        )
        root_latent = model_output.hidden_state
        root_logits_per_agent = model_output.policy_logits
        root_value_logits = model_output.value_logits

        # Convert per-agent logits to joint action logits and get root value
        root_joint_logits = self._logits_to_joint_logits(root_logits_per_agent)
        root_value = utils.support_to_scalar(root_value_logits, self.value_support)

        root = mctx.RootFnOutput(
            prior_logits=root_joint_logits, value=root_value, embedding=root_latent
        )

        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=gumbel_key,
            root=root,
            recurrent_fn=self._recurrent_fn_jit,
            num_simulations=self.num_simulations,
            max_depth=self.max_depth_gumbel_search,
            max_num_considered_actions=self.num_gumbel_samples,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value, use_mixed_value=True
            ),
        )

        chosen_joint_action_index = policy_output.action
        joint_action_tuple = jnp.unravel_index(
            chosen_joint_action_index, self.joint_action_shape
        )
        final_joint_action = jnp.array(joint_action_tuple).squeeze(axis=-1)

        joint_policy_target = policy_output.action_weights

        marginal_policy_targets = self._joint_policy_to_marginal(
            joint_policy_target[None, :]
        ).squeeze(0)

        return MCTSPlanOutput(
            joint_action=final_joint_action,
            policy_targets=marginal_policy_targets,
            root_value=root_value.squeeze().astype(float),
            agent_order=jnp.arange(self.num_agents)
        )

    def _logits_to_joint_logits(self, logits: jnp.ndarray) -> jnp.ndarray:
        """
        Converts per-agent logits (B, N, A) to joint action logits (B, A^N).

        This assumes independence between agent policies: log P(j1, ..., jN) = sum(log P(ji)).
        """
        # Take the log-softmax to get log probabilities
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        # Start with the log probabilities of the first agent
        joint_log_probs = log_probs[:, 0, :]
        
        # Iteratively combine with the log probabilities of other agents
        for i in range(1, self.num_agents):
            next_log_probs = log_probs[:, i, :]
            # This operation is equivalent to summing the log probabilities for all combinations
            joint_log_probs = joint_log_probs[:, :, None] + next_log_probs[:, None, :]
            joint_log_probs = joint_log_probs.reshape(logits.shape[0], -1)

        return joint_log_probs

    def _joint_policy_to_marginal(self, joint_policy: jnp.ndarray) -> jnp.ndarray:
        """
        Converts a joint policy (B, A^N) to marginal policies (B, N, A).
        """
        batch_size = joint_policy.shape[0]
        # Reshape to (B, A, A, ..., A) to work with individual agent dimensions
        reshaped_policy = joint_policy.reshape(batch_size, *self.joint_action_shape)

        marginals = []
        for i in range(self.num_agents):
            # To get the marginal for agent i, sum over all other agent axes
            axes_to_sum = tuple(j for j in range(1, self.num_agents + 1) if j != i + 1)
            marginal = jnp.sum(reshaped_policy, axis=axes_to_sum)
            marginals.append(marginal)

        return jnp.stack(marginals, axis=1)