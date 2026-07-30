import os
import numpy as np

import ray

from config import ExperimentConfig
from utils.logging_utils import logger
from utils.profiler import Profiler


@ray.remote(num_cpus=2)
class ReanalyzeActor:
    """
    Re-runs MCTS on stored observations using the latest model params to
    produce fresher policy/value targets.

    Only position 0 (the root observation) of each stored ReplayItem sequence
    can be updated — intermediate unroll states are not stored in the buffer.
    Params are synced from LearnerActor on every call to run_reanalyze().
    """

    def __init__(
        self,
        actor_id: int,
        obs_size: int,
        action_size: int,
        learner_actor,
        replay_buffer_actor,
        config: ExperimentConfig,
    ):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # Same as DataActor: don't preallocate GPU memory so multiple JAX processes
        # can share the single GPU without OOM at init time.
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        # Match DataActor: limit XLA/BLAS threads to avoid thrashing with other actors.
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("XLA_FLAGS",
            (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false").strip()
        )

        import jax
        try:
            devices = jax.devices()
            logger.info(f"(ReanalyzeActor {actor_id} pid={os.getpid()}) JAX devices: {devices}")
            gpu_devices = [d for d in devices if "gpu" in str(d).lower() or "cuda" in str(d).lower()]
            if gpu_devices:
                logger.info(f"(ReanalyzeActor {actor_id}) Using GPU: {gpu_devices[0]}")
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        logger.info(f"(ReanalyzeActor {actor_id}) GPU memory (used/free/total MiB): {result.stdout.strip()}")
                except Exception as e:
                    logger.warning(f"(ReanalyzeActor {actor_id}) Could not query GPU memory: {e}")
            else:
                logger.warning(
                    f"(ReanalyzeActor {actor_id}) No GPU found — falling back to CPU. "
                    f"MCTS will be slow. Devices: {devices}"
                )
        except Exception as e:
            logger.error(f"(ReanalyzeActor {actor_id}) JAX device init failed: {e}")
            raise
        from model import FlaxMAMuZeroNet
        from mcts import MCTSJointOSLAPlanner

        self.actor_id = actor_id
        self.config = config
        self.learner = learner_actor
        self.replay_buffer = replay_buffer_actor

        self.rng_key = jax.random.PRNGKey(actor_id * 2718 + 42)

        model = FlaxMAMuZeroNet(config.model, action_size)
        planner = MCTSJointOSLAPlanner(model=model, config=config)
        self.plan_fn = jax.jit(planner.plan)
        result = ray.get(learner_actor.get_params.remote())
        self.params = result["params"]
        self._param_future = None
        self.profiler = Profiler(f"reanalyze[{actor_id}]", log_interval=config.train.debug_interval)
        logger.info(f"(ReanalyzeActor {actor_id} pid={os.getpid()}) Setup complete.")

    def run_reanalyze(self):
        """
        Samples a batch of stored observations, re-runs MCTS with the latest
        params, and writes back fresh policy/value targets at position 0.

        Skips reanalysis during early training (warmup phase) so the model
        doesn't corrupt n-step bootstrapped targets with bad MCTS estimates.
        The warmup threshold is 2× warmup_episodes learner steps.
        """
        import jax
        import jax.numpy as jnp

        with self.profiler.time("sample_wait"):
            indices, observations = ray.get(
                self.replay_buffer.sample_for_reanalysis.remote(
                    self.config.train.reanalyze_batch_size
                )
            )
        if indices is None:
            return

        # Resolve previous async param fetch (if ready), then fire the next one.
        with self.profiler.time("param_sync"):
            if self._param_future is not None:
                ready, _ = ray.wait([self._param_future], timeout=0)
                if ready:
                    result = ray.get(self._param_future)
                    self.params = result["params"]
                    self._param_future = None
            if self._param_future is None:
                self._param_future = self.learner.get_params.remote()

        with self.profiler.time("mcts_plan"):
            self.rng_key, plan_key = jax.random.split(self.rng_key)
            plan_output = self.plan_fn(self.params, plan_key, jnp.array(observations))
            jax.block_until_ready(plan_output.policy_targets)

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


