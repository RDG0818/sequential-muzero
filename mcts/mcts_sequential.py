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

class MCTSSequentialPlanner(MCTSPlanner):
    """
    Performs sequential Monte Carlo Tree Search (MCTS) for each agent,
    allowing agents to condition their plans on the refined plans of others.
    """
    def __init__(self, model: FlaxMAMuZeroNet, config: ExperimentConfig):
        super().__init__(model, config)

        self.independent_argmax = config.mcts.independent_argmax
        self.policy_eta = config.mcts.policy_eta
        self._recurrent_fn_jit = jax.jit(self._recurrent_fn)
        self.plan_jit = jax.jit(self._plan_loop)
    
    def _recurrent_fn(self, params, rng_key, action: chex.Array, embedding: Tuple[chex.Array, chex.Array, chex.Array]) -> Tuple[mctx.RecurrentFnOutput, Tuple]:
        """
        Implementation of the recurrent function for sequential planning.
        """
        latent, agent_idx, coord_state = embedding # latent: (B,N,D), agent_idx: (B,)
        planning_agent_idx = agent_idx[0] # scalar
        batch_size = latent.shape[0]
        batch_indices = jnp.arange(batch_size)

        # Get prior policies for all agents from the current latent state
        prior_logits, _ = self.model.apply({'params': params}, latent, method=self.model.predict)
        deltas = self.model.apply({'params': params}, latent, coord_state, prior_logits, method=self.model.adapt)
        adapted_logits = prior_logits + deltas

        planned_actions = jnp.argmax(adapted_logits, axis=-1)

        agent_indices = jnp.arange(self.num_agents) # (N,)
        mask = agent_indices < planning_agent_idx

        if self.independent_argmax:
            unplanned_actions = jnp.argmax(prior_logits, axis=-1) # (B,N)
        else:
            rng_key, sample_key = jax.random.split(rng_key, 2)
            unplanned_actions = jax.random.categorical(sample_key, prior_logits) # (B,N)
        
        actions = jnp.where(mask, planned_actions, unplanned_actions)
        joint_action = actions.at[batch_indices, agent_idx].set(action) # (B,N)

        model_output = self.model.apply(
            {'params': params}, latent, joint_action,
            method=self.model.recurrent_inference,
            rngs={'dropout': rng_key}
        )

        next_latent,  policy_logits = model_output.hidden_state, model_output.policy_logits # (B,N,D) and (B,N,A)
        value_logits, reward_logits = model_output.value_logits, model_output.reward_logits
        value, reward = utils.support_to_scalar(value_logits, self.value_support), utils.support_to_scalar(reward_logits, self.reward_support) # (B,)

        prior = policy_logits[batch_indices, agent_idx]  # (B,A)

        new_embed = (next_latent, agent_idx, coord_state)
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
        init_key, rng_key, perm_key = jax.random.split(rng_key, 3)
        model_output = self.model.apply(
            {'params': params}, observation,
            rngs={'dropout': init_key}
        )
        
        root_latent, root_logits = model_output.hidden_state, model_output.policy_logits # (1,N,D) and (1,N,A)
        root_value = utils.support_to_scalar(model_output.value_logits, self.value_support) #(1,)

        # Prepare per-agent inputs for scan
        keys = jax.random.split(rng_key, self.num_agents)  # (N,2)
        agent_order = jax.random.permutation(perm_key, jnp.arange(self.num_agents, dtype=jnp.int32))     # (N,)

        coord_state = jnp.zeros((1, self.config.model.hidden_state_size))

        deltas = self.model.apply({'params': params}, root_latent, coord_state, root_logits, method=self.model.adapt)
        delta_magnitude = jnp.mean(jnp.abs(deltas))

        carry = (root_logits, root_value.reshape(-1), root_latent, coord_state) # Tuple of (1,N,A), (1,), (1,N,D)

        def agent_step(carry: Tuple, inputs: Tuple) -> Tuple[Tuple, Tuple]:
            logits_b, value_b, latent_b, coord_s = carry # (1,N,A), (1,), (1,N,D)

            key, agent = inputs # (N,2) and scalar
            p_slice = jax.lax.dynamic_slice(logits_b, start_indices=(0, agent, 0), slice_sizes=(1, 1, self.action_space_size))  # shape (1,1,A)
            p = p_slice.squeeze(1) # (1,A)                     
            emb = (latent_b, jnp.array([agent], jnp.int32), coord_s) # (1,N,D), (1,)

            out = mctx.gumbel_muzero_policy(
                params=params, rng_key=key, root=mctx.RootFnOutput(prior_logits=p, value=value_b, embedding=emb),
                recurrent_fn=self._recurrent_fn_jit, num_simulations=self.num_simulations, 
                max_depth=self.max_depth_gumbel_search, max_num_considered_actions=self.num_gumbel_samples, 
                qtransform=functools.partial(mctx.qtransform_completed_by_mix_value, use_mixed_value=True)
            )
            
            weights = out.action_weights.squeeze(0)
            summary = out.search_tree.summary()
            search_value = summary.value.squeeze()
            root_q_values = summary.qvalues.squeeze()
            plan_summary = jnp.concatenate([weights, jnp.atleast_1d(search_value), root_q_values])

            next_coord_s, _ = self.model.apply({'params': params}, coord_s, plan_summary, method=self.model.coordinate)

            updated_carry = (logits_b, value_b, latent_b, next_coord_s)

            return updated_carry, (out.action.squeeze(0), weights, root_q_values, search_value)

        final_carry, results = jax.lax.scan(agent_step, carry, (keys, agent_order)) 
        actions, weights, q_values, search_values = results  

        final_actions = jnp.empty_like(actions).at[agent_order].set(actions)
        final_weights = jnp.empty_like(weights).at[agent_order].set(weights)
        final_mcts_value = jnp.mean(search_values)
        final_search_values = jnp.empty_like(search_values).at[agent_order].set(search_values)  

        final_coord_state = final_carry[-1]
        coord_state_norm = jnp.linalg.norm(final_coord_state)

        return MCTSPlanOutput(
            joint_action=   final_actions,
            policy_targets= final_weights,
            root_value=     final_mcts_value.squeeze().astype(float),
            agent_order=agent_order,
            per_agent_mcts_values=final_search_values,
            root_q_values=q_values,
            delta_magnitude=delta_magnitude,
            coord_state_norm=coord_state_norm
        )
