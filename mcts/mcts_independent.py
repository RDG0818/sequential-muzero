import jax
import jax.numpy as jnp
import chex
import mctx
from typing import Tuple
from model.model import FlaxMAMuZeroNet
import utils.utils as utils
import functools
from config import ExperimentConfig
from mcts.base import MCTSPlanner, MCTSPlanOutput

class MCTSIndependentPlanner(MCTSPlanner):
    """
    Performs independent Monte Carlo Tree Search (MCTS) for each agent.

    This planner implements the MCTSPlanner abstract base class. It runs a
    separate Gumbel MuZero search for each agent, assuming other agents will
    play by sampling from their predicted policy.
    """
    def __init__(self, model: FlaxMAMuZeroNet, config: ExperimentConfig):
        super().__init__(model, config)

        self.independent_argmax = config.mcts.independent_argmax
        self._recurrent_fn_jit = jax.jit(self._recurrent_fn)
        self.plan_jit = jax.jit(self._plan_loop)
    
    def _recurrent_fn(self, params, rng_key, action: chex.Array, embedding: Tuple[chex.Array, chex.Array]) -> Tuple[mctx.RecurrentFnOutput, Tuple]:
            """
            Implementation of the recurrent function for independent planning.

            During a search for a given agent, this function simulates a step by
            using the action from the search for the current agent and sampling
            actions for all other agents from the model's predicted policy.
            """
            latent, agent_idx = embedding # latent: (B,N,D), agent_idx: (B,)
            batch_size = latent.shape[0]
            batch_indices = jnp.arange(batch_size)

            # Get prior policies for all agents from the current latent state
            prior_logits, _ = self.model.apply({'params': params}, latent, method=self.model.predict)
            if self.independent_argmax:
                actions = jnp.argmax(prior_logits, axis=-1) # (B,N)
            else:
                rng_key, sample_key = jax.random.split(rng_key, 2)
                actions = jax.random.categorical(sample_key, prior_logits) # (B,N)

            joint_action = actions.at[batch_indices, agent_idx].set(action) # (B,N)

            model_output = self.model.apply(
                {'params': params}, latent, joint_action,
                method=self.model.recurrent_inference,
                rngs={'dropout': rng_key}
            )

            next_latent,  policy_logits, value_logits = model_output.hidden_state, model_output.policy_logits # (B,N,D) and (B,N,A)
            value_logits, reward_logits = model_output.value_logits, model_output.reward_logits
            value, reward = utils.support_to_scalar(value_logits, self.value_support), utils.support_to_scalar(reward_logits, self.reward_support) # (B,)

            prior = policy_logits[batch_indices, agent_idx]  # (B,A)

            new_embed = (next_latent, agent_idx)
            out = mctx.RecurrentFnOutput(
                reward=reward,
                discount= jnp.full_like(reward, 0.99),
                prior_logits= prior,
                value=value
            )
            return out, new_embed

    def _plan_loop(self, params, rng_key, observation: chex.Array):
        """
        Implementation of the main planning loop for independent MCTS.
        Uses jax.lax.scan to iterate the search over each agent.
        """
        init_key, rng_key = jax.random.split(rng_key, 2)
        model_output = self.model.apply(
            {'params': params}, observation,
            rngs={'dropout': init_key}
        )
        
        root_latent, root_logits = model_output.hidden_state, model_output.policy_logits # (1,N,D) and (1,N,A)
        root_value = utils.support_to_scalar(model_output.value_logits, self.value_support) #(1,)

        # Prepare per-agent inputs for scan
        keys = jax.random.split(rng_key, self.num_agents)  # (N,2)
        idxs = jnp.arange(self.num_agents, dtype=jnp.int32)     # (N,)

        carry = (root_logits, root_value.reshape(-1), root_latent) # Tuple of (1,N,A), (1,), and (1,N,D)

        def agent_step(carry: Tuple, inputs: Tuple) -> Tuple[Tuple, Tuple]:
            logits_b, value_b, latent_b = carry # (1,N,A), (1,), and (1,N,D)
            key, agent = inputs # (N,2) and scalar
            p_slice = jax.lax.dynamic_slice(logits_b, start_indices=(0, agent, 0), slice_sizes=(1, 1, self.action_space_size))  # shape (1,1,A)
            p = p_slice.squeeze(1) # (1,A)                     
            emb = (latent_b, jnp.array([agent], jnp.int32)) # (1,N,D), (1,)

            out = mctx.gumbel_muzero_policy(
                params=params, rng_key=key, root=mctx.RootFnOutput(prior_logits=p, value=value_b, embedding=emb),
                recurrent_fn=self._recurrent_fn_jit, num_simulations=self.num_simulations, 
                max_depth=self.max_depth_gumbel_search, max_num_considered_actions=self.num_gumbel_samples, 
                qtransform=functools.partial(mctx.qtransform_completed_by_mix_value, use_mixed_value=True)
            )

            return carry, (out.action.squeeze(0), out.action_weights.squeeze(0))

        _, results = jax.lax.scan(agent_step, carry, (keys, idxs)) 
        actions, weights = results  

        return MCTSPlanOutput(
            joint_action=   actions,
            policy_targets= weights,
            root_value=     root_value.squeeze().astype(float)
        )
