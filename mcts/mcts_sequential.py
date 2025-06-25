import jax
import jax.numpy as jnp
import mctx
from typing import NamedTuple
from model.model import FlaxMAMuZeroNet
import utils.utils as utils
from utils.utils import DiscreteSupport
import functools
from config import ExperimentConfig

class MCTSPlanOutput(NamedTuple):
    """Output of MCTSPlanner.plan()."""
    joint_action:   jnp.ndarray  # shape (N,), chosen action per agent
    policy_targets: jnp.ndarray  # shape (N, A), visit-count targets per agent
    root_value:     float        # scalar root value estimate

class MCTSSequentialPlanner:
    def __init__(
        self,
        model: FlaxMAMuZeroNet,
        config: ExperimentConfig
    ):

        self.model = model
        self.num_agents = config.train.num_agents        # N agents
        self.action_space_size = model.action_space_size  # A actions
        self.num_simulations = config.mcts.num_simulations   # MCTS rollout count
        self.max_depth_gumbel_search = config.mcts.max_depth_gumbel_search
        self.num_gumbel_samples = config.mcts.num_gumbel_samples
        self.value_support = DiscreteSupport(min=-config.model.value_support_size, max=config.model.value_support_size)
        self.reward_support = DiscreteSupport(min=-config.model.reward_support_size, max=config.model.reward_support_size)

        def recurrent_fn(params, rng_key, action, embedding):
            """
            Batched one-step rollout for a single agent in MuZero search.
            """
            latent, idx = embedding

            # Get prior policies for all agents from the current latent state
            prior_logits, _ = self.model.apply({'params': params}, latent, method=self.model.predict)
            greedy_actions = jnp.argmax(prior_logits, axis=-1)  # Shape: (B, N)
            
            # Build joint-actions: B x N, set agent idx to action from MCTS search
            def fill(actions_row, current_agent_action, agent_index):
                return actions_row.at[agent_index].set(current_agent_action)
            
            joint = jax.vmap(fill)(greedy_actions, action, idx) # (B, N)

            # Dynamics + prediction
            model_output = self.model.apply(
                {'params': params}, latent, joint,
                method=self.model.recurrent_inference,
                rngs={'dropout': rng_key}
            )
            next_latent = model_output.hidden_state
            reward_logits = model_output.reward_logits
            multi_logits = model_output.policy_logits
            value_logits = model_output.value_logits
            value= utils.support_to_scalar(value_logits, self.value_support)
            reward = utils.support_to_scalar(reward_logits, self.reward_support)

            def pick(l_row, i): return l_row[i]
            prior = jax.vmap(pick)(multi_logits, idx)  # (B, A)

            new_embed = (next_latent, idx)
            out = mctx.RecurrentFnOutput(
                reward=reward,
                discount= jnp.full_like(reward, 0.99),
                prior_logits= prior,
                value=value
            )
            return out, new_embed

        self._recurrent_fn = jax.jit(recurrent_fn)

        self.plan_jit = jax.jit(self._plan_loop, static_argnums=())

    def plan(self, params, rng_key, observation):
        return self.plan_jit(params, rng_key, observation)

    def _plan_loop(self, params, rng_key, observation):
        init_key, rng_key = jax.random.split(rng_key, 2)
        model_output = self.model.apply(
            {'params': params}, observation,
            rngs={'dropout': init_key}
        )
        
        root_latent = model_output.hidden_state
        root_logits = model_output.policy_logits
        root_value_logits = model_output.value_logits

        root_value = utils.support_to_scalar(root_value_logits, self.value_support)

        # Prepare per-agent inputs for scan
        keys = jax.random.split(rng_key, self.num_agents)  # (N,2)
        idxs = jnp.arange(self.num_agents, dtype=jnp.int32)     # (N,)

        carry = (root_logits, root_value.reshape(-1), root_latent)

        def agent_step(carry, inputs):
            logits_b, value_b, latent_b = carry
            key, agent = inputs
            # Extract agent's (1,A) prior
            p_slice = jax.lax.dynamic_slice(
                logits_b,
                start_indices=(0, agent, 0),
                slice_sizes=(1, 1, self.action_space_size)
            )  # shape (1,1,A)
            p = p_slice.squeeze(1)
            v = value_b                              
            emb = (latent_b, jnp.array([agent], jnp.int32))

            out = mctx.gumbel_muzero_policy(
                params=params, rng_key=key, root=mctx.RootFnOutput(prior_logits=p, value=v, embedding=emb),
                recurrent_fn=self._recurrent_fn, num_simulations=self.num_simulations, 
                max_depth=self.max_depth_gumbel_search, max_num_considered_actions=self.num_gumbel_samples, 
                qtransform=functools.partial(mctx.qtransform_completed_by_mix_value, use_mixed_value=True)
            )

            return carry, (out.action.squeeze(0), out.action_weights.squeeze(0))

        _, results = jax.lax.scan(agent_step, carry, (keys, idxs)) # Pretty much a for loop with a carry value
        actions, weights = results  

        return MCTSPlanOutput(
            joint_action=   actions,
            policy_targets= weights,
            root_value=     root_value.squeeze().astype(float)
        )
