# mcts/mcts_joint.py
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
        self.plan_jit = jax.jit(self._plan_loop, static_argnames=['train_mode'])

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

    def _plan_loop(self, params, rng_key, observation, train_mode: bool) -> MCTSPlanOutput:
        """
        The main planning logic that orchestrates the joint MCTS search.

        Args:
            params: The parameters of the MuZero model.
            rng_key: A JAX random key for the search.
            observation: The current environment observation.

        Returns:
            An `MCTSPlanOutput` object containing the chosen action, policy targets, and root value.
        """
        init_key, gumbel_key, noise_key, explore_key = jax.random.split(rng_key, 4)
        
        # 1. Initial inference from the model
        model_output = self.model.apply(
            {"params": params}, observation, rngs={"dropout": init_key}
        )
        root_latent = model_output.hidden_state
        root_logits_per_agent = model_output.policy_logits
        root_value_logits = model_output.value_logits

        def add_noise_to_per_agent_logits(logits_per_agent):
            """Applies Dirichlet noise to each agent's logit vector."""
            # Logits shape is (B, N, A)
            probs = jax.nn.softmax(logits_per_agent, axis=-1)
            
            # We want noise of shape (B, N, A)
            batch_shape = logits_per_agent.shape[:-1]  # -> (B, N)
            num_actions = logits_per_agent.shape[-1]
            
            alpha_1d = jnp.full(shape=(num_actions,), fill_value=self.config.mcts.dirichlet_alpha)
            noise = jax.random.dirichlet(noise_key, alpha_1d, shape=batch_shape)
            
            # Mix probabilities
            noisy_probs = (1 - self.config.mcts.dirichlet_epsilon) * probs + self.config.mcts.dirichlet_epsilon * noise
            
            # Convert back to logits
            return jnp.log(noisy_probs + 1e-6)

        final_root_logits_per_agent = jax.lax.cond(
        train_mode,
        add_noise_to_per_agent_logits,
        lambda logits: logits,  # If not training, do nothing
        root_logits_per_agent
         )

        # 2. Convert per-agent logits to joint action logits and get root value
        root_joint_logits = self._logits_to_joint_logits(final_root_logits_per_agent)
        root_value = utils.support_to_scalar(root_value_logits, self.value_support)


        # 3. Define the MCTS root and run the Gumbel MuZero search
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

        # 4. Process the MCTS output
        # The action is a scalar index for the chosen joint action
        chosen_joint_action_index = policy_output.action
        joint_action_tuple = jnp.unravel_index(
            chosen_joint_action_index, self.joint_action_shape
        )
        final_joint_action = jnp.array(joint_action_tuple).squeeze(axis=-1)

        # The action weights are the visit counts for the joint actions
        joint_policy_target = policy_output.action_weights

        # Convert the joint policy target back to per-agent marginals for the loss function
        marginal_policy_targets = self._joint_policy_to_marginal(
            joint_policy_target[None, :]
        ).squeeze(0)

        random_action = jax.random.randint(
            explore_key,
            shape=(self.num_agents,),
            minval=0,
            maxval=self.action_space_size
        )
        
        # Decide whether to take the random action based on epsilon
        should_explore = jax.random.uniform(explore_key) < self.config.mcts.epsilon
        
        # Only explore during training mode
        explore_condition = jnp.logical_and(train_mode, should_explore)
        
        # Select the final action to be executed in the environment
        executed_action = jax.lax.cond(
            explore_condition,
            lambda _: random_action,
            lambda _: final_joint_action,
            operand=None
        )

        search_tree = policy_output.search_tree

        improved_root_value = search_tree.node_values[:, 0]

        return MCTSPlanOutput(
            joint_action=executed_action,
            policy_targets=marginal_policy_targets,
            root_value=improved_root_value.squeeze().astype(float),
            delta_magnitude=0,
            coord_state_norm=0
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