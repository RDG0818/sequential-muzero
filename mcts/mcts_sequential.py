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
        self._recurrent_fn_jit = jax.jit(self._recurrent_fn)
        self.plan_jit = jax.jit(self._plan_loop)
    
    def _recurrent_fn(self, params, rng_key, action: chex.Array, embedding: Tuple[chex.Array, chex.Array]) -> Tuple[mctx.RecurrentFnOutput, Tuple]:
            """
            Implementation of the recurrent function for sequential planning.

            During a search for a given agent, this function simulates a step by
            using the action from the search for the current agent and sampling
            actions for all other agents from the model's predicted policy.
            """
            latent, agent_idx, = embedding # latent: (B,N,D), agent_idx: (B,)
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

            next_latent,  policy_logits = model_output.hidden_state, model_output.policy_logits # (B,N,D) and (B,N,A)
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
        init_key, rng_key, perm_key = jax.random.split(rng_key, 3)
        model_output = self.model.apply(
            {'params': params}, observation,
            rngs={'dropout': init_key}
        )
        
        root_latent, root_logits = model_output.hidden_state, model_output.policy_logits # (1,N,D) and (1,N,A)
        root_value = utils.support_to_scalar(model_output.value_logits, self.value_support).reshape(-1) #(1,)

        # Prepare per-agent inputs for scan
        keys = jax.random.split(rng_key, self.num_agents)  # (N,2)
        agent_order = jax.random.permutation(perm_key, jnp.arange(self.num_agents, dtype=jnp.int32))     # (N,)

        coord_state = jnp.zeros((1, self.config.model.hidden_state_size))

        carry = coord_state

        def agent_step(carry_coord_state: Tuple, inputs: Tuple) -> Tuple[Tuple, Tuple]:
            key, agent = inputs # (N,2) and scalar
            p = root_logits[:, agent, :]        
            mcts_key, noisy_logits = self.add_dirichlet_noise(key, p)            
            emb = (root_latent, jnp.array([agent], jnp.int32)) # (1,N,D), (1,)

            out = mctx.gumbel_muzero_policy(
                params=params, rng_key=mcts_key, root=mctx.RootFnOutput(prior_logits=noisy_logits, value=root_value, embedding=emb),
                recurrent_fn=self._recurrent_fn_jit, num_simulations=self.num_simulations, 
                max_depth=self.max_depth_gumbel_search, max_num_considered_actions=self.num_gumbel_samples, 
                qtransform=functools.partial(mctx.qtransform_completed_by_mix_value, use_mixed_value=True)
            )
            
            mcts_policy = out.action_weights.squeeze(0)
        
            # The GRU updates the coordination state using the policy from the agent's plan.
            next_coord_state, _ = self.model.apply({'params': params}, carry_coord_state, mcts_policy, method=self.model.coordinate)
            search_value = out.search_tree.summary().value.squeeze()
            root_q_values = out.search_tree.summary().qvalues.squeeze(0)
            
            return next_coord_state, (out.action.squeeze(0), mcts_policy, root_q_values, search_value)

        final_coord_state, results = jax.lax.scan(agent_step, carry, (keys, agent_order)) 
        actions, weights, q_values, search_values = results  

        final_actions = jnp.empty_like(actions).at[agent_order].set(actions)
        final_weights = jnp.empty_like(weights).at[agent_order].set(weights)
        final_mcts_value = jnp.mean(search_values)
        final_search_values = jnp.empty_like(search_values).at[agent_order].set(search_values)  

        coord_state_norm = jnp.linalg.norm(final_coord_state)

        return MCTSPlanOutput(
            joint_action=   final_actions,
            policy_targets= final_weights,
            root_value=     final_mcts_value.squeeze().astype(float),
            agent_order=agent_order,
            per_agent_mcts_values=final_search_values,
            root_q_values=q_values,
            coord_state_norm=coord_state_norm
        )
