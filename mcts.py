import jax
import jax.numpy as jnp
import mctx
from typing import NamedTuple

from flax_model import FlaxMAMuZeroNet

class MCTSPlanOutput(NamedTuple):
    """Output of MCTSPlanner.plan()."""
    joint_action:   jnp.ndarray  # shape (N,), chosen action per agent
    policy_targets: jnp.ndarray  # shape (N, A), visit-count targets per agent
    root_value:     float        # scalar root value estimate

class MCTSPlanner:
    def __init__(
        self,
        model: FlaxMAMuZeroNet,
        num_simulations: int,
        mode: str = 'independent',
        num_joint_samples: int = 16
    ):
        # Validate planner mode
        if mode not in ['independent', 'sequential', 'joint']:
            raise ValueError(f"Invalid planner mode: {mode}")

        self.model = model
        self.num_agents = model.num_agents        # N agents
        self.action_space_size = model.action_space_size  # A actions
        self.num_simulations = num_simulations   # MCTS rollout count
        self.mode = mode

        # JIT-compiled recurrent inference for search
        def recurrent_fn(params, rng_key, action, embedding):
            """
            Batched one-step rollout for a single agent in MuZero search.

            - params: model parameters PyTree
            - rng_key: random key (unused deterministically)
            - action:  jnp.ndarray (B,), chosen action for each batch
            - embedding: tuple(latent, idx)
                latent: jnp.ndarray (B, N, H)
                idx:    jnp.ndarray (B,)
            Returns (step_output, new_embedding)
            """
            latent, idx = embedding
            # Build joint-actions: B x N, set agent idx to action
            def fill(a, i):
                row = jnp.zeros((self.num_agents,), jnp.int32)
                return row.at[i].set(a)
            joint = jax.vmap(fill)(action, idx)  # (B, N)

            # Dynamics + prediction
            next_latent, reward, multi_logits, value = self.model.apply(
                {'params': params}, latent, joint,
                method=self.model.recurrent_inference
            )
            # reward: (B,1), multi_logits: (B,N,A), value: (B,1)

            # Extract this agent's logits
            def pick(l_row, i): return l_row[i]
            prior = jax.vmap(pick)(multi_logits, idx)  # (B, A)

            # Package output
            new_embed = (next_latent, idx)
            out = mctx.RecurrentFnOutput(
                reward=   reward.squeeze(-1),
                discount= jnp.full_like(reward.squeeze(-1), 0.99),
                prior_logits= prior,
                value=    value.squeeze(-1)
            )
            return out, new_embed

        self._recurrent_fn = jax.jit(recurrent_fn)

        self.plan_jit = jax.jit(self._plan_loop, static_argnums=())

    def plan(self, params, rng_key, observation):
        """
        Wrapper for jit-compiled planning.

        Args:
          params: model weights
          rng_key: PRNG key
          observation: jnp.ndarray (1, N, obs_dim)
        Returns MCTSPlanOutput
        """
        return self.plan_jit(params, rng_key, observation)

    def _plan_loop(self, params, rng_key, observation):
        """
        Core planning loop: initial inference + per-agent MCTS via scan.
        """
        # Initial inference (batch=1)
        root_latent, _, root_logits, root_value = self.model.apply(
            {'params': params}, observation
        )  # shapes: (1,N,H), (1,N,A), (1,1)

        # Prepare per-agent inputs for scan
        keys = jax.random.split(rng_key, self.num_agents)  # (N,2)
        idxs = jnp.arange(self.num_agents, dtype=jnp.int32)     # (N,)

        # Carry: (prior_logits, root_value, root_latent)
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

            out = mctx.muzero_policy(
                params, key,
                mctx.RootFnOutput(prior_logits=p, value=v, embedding=emb),
                self._recurrent_fn, self.num_simulations
            )
            # outputs: action (1,), weights (1,A)
            return carry, (out.action.squeeze(0), out.action_weights.squeeze(0))

        # Run scan over agents: results=(actions, weights)
        _, results = jax.lax.scan(agent_step, carry, (keys, idxs))
        actions, weights = results  # each shape (N,) and (N,A)

        return MCTSPlanOutput(
            joint_action=   actions,
            policy_targets= weights,
            root_value=     root_value.squeeze().astype(float)
        )
