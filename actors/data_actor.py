import os
import time
import numpy as np

import ray

from config import ExperimentConfig
from utils.logging_utils import logger
from utils.profiler import Profiler


@ray.remote(num_cpus=1)
class DataActor:
    """
    Generates experience on CPU via MCTS planning.

    Runs episodes, converts them to ReplayItems, and ships them to
    the ReplayBufferActor. Periodically syncs parameters from LearnerActor.
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
        # Ray sets CUDA_VISIBLE_DEVICES="" for CPU workers, causing the JAX CUDA
        # plugin to crash on cuInit(0). Pop the restriction so the GPU is visible.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # Don't preallocate GPU memory at JAX init — the learner and multiple data
        # actors share one GPU. With preallocation on, each process reserves a fixed
        # fraction (default 75%) at startup, causing OOM before any computation runs.
        # With it off, JAX allocates on demand up to the physical limit.
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        # Limit XLA/BLAS thread pool per actor. Without this, each JAX process
        # tries to use all CPU cores. With N actors all doing this, they thrash
        # each other (N × all_cores threads on all_cores physical cores).
        # 2 threads per actor matches @ray.remote(num_cpus=2) above.
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("XLA_FLAGS",
            (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false").strip()
        )

        import jax
        try:
            devices = jax.devices()
            logger.info(f"(DataActor {actor_id} pid={os.getpid()}) JAX devices: {devices}")
            gpu_devices = [d for d in devices if "gpu" in str(d).lower() or "cuda" in str(d).lower()]
            if gpu_devices:
                logger.info(f"(DataActor {actor_id}) Using GPU: {gpu_devices[0]}")
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        logger.info(f"(DataActor {actor_id}) GPU memory (used/free/total MiB): {result.stdout.strip()}")
                except Exception as e:
                    logger.warning(f"(DataActor {actor_id}) Could not query GPU memory: {e}")
            else:
                logger.warning(
                    f"(DataActor {actor_id}) No GPU found — falling back to CPU. "
                    f"MCTS will be slow. Devices: {devices}"
                )
        except Exception as e:
            logger.error(f"(DataActor {actor_id}) JAX device init failed: {e}")
            raise
        from model import FlaxMAMuZeroNet
        from mcts import MCTSJointOSLAPlanner
        from envs import make_vec_env_wrapper

        self.actor_id = actor_id
        self.config = config
        self.learner = learner_actor
        self.replay_buffer = replay_buffer_actor
        self.episodes_since_update = 0

        self.rng_key = jax.random.PRNGKey(actor_id * 1337 + 7)

        self.env = make_vec_env_wrapper(
            config.train.env_name,
            config.train.num_agents,
            config.train.max_episode_steps,
            config.train.num_envs_per_actor,
        )

        model = FlaxMAMuZeroNet(config.model, action_size)
        planner_map = {"joint": MCTSJointOSLAPlanner}
        if config.mcts.planner_mode not in planner_map:
            raise ValueError(
                f"Unknown planner_mode '{config.mcts.planner_mode}'. "
                f"Choose from: {list(planner_map)}"
            )
        planner = planner_map[config.mcts.planner_mode](model=model, config=config)

        # Single JIT boundary: DataActor owns compilation of the plan function.
        self.plan_fn = jax.jit(planner.plan)
        result = ray.get(learner_actor.get_params.remote())
        self.params = result["params"]
        self.norm_state = result["norm_state"]  # None when use_obs_normalization=false
        self._param_future = None  # in-flight async param fetch, if any
        self.profiler = Profiler(f"data_actor[{actor_id}]", log_interval=config.train.debug_interval)
        logger.info(f"(DataActor {actor_id} pid={os.getpid()}) Setup complete.")

    def run_episode(self) -> float:
        """
        Runs num_envs_per_actor episodes in parallel, processes them into
        ReplayItems, and ships them to the buffer.
        Returns the mean undiscounted episode return across all envs.
        """
        import jax
        import jax.numpy as jnp
        from utils.replay_buffer import Episode, Transition, process_episode

        B = self.config.train.num_envs_per_actor
        debug = self.config.train.debug
        ep_start = time.monotonic()

        # Reset all B envs simultaneously.
        self.rng_key, *reset_keys_list = jax.random.split(self.rng_key, B + 1)
        reset_keys = jnp.stack(reset_keys_list)  # (B, 2)
        observations, states = self.env.reset(reset_keys)
        # observations: (B, N, obs_size)

        episodes = [Episode() for _ in range(B)]
        active = [True] * B
        episode_wins = [False] * B

        for _ in range(self.config.train.max_episode_steps):
            self.rng_key, plan_key, step_key = jax.random.split(self.rng_key, 3)

            with self.profiler.time("plan"):
                # Normalize observations if running norm is enabled.
                # Applied on CPU (numpy) before JIT boundary so normalization
                # doesn't affect the JAX trace or add a device round-trip.
                if self.norm_state is not None:
                    obs_np = np.array(observations)
                    obs_norm = (obs_np - self.norm_state["mean"]) / np.sqrt(
                        self.norm_state["var"] + 1e-5
                    )
                    plan_obs = jnp.array(obs_norm)
                else:
                    plan_obs = observations
                # block_until_ready ensures we measure actual MCTS compute, not
                # just async dispatch time (JAX on CPU is also async).
                plan_output = self.plan_fn(self.params, plan_key, plan_obs)
                jax.block_until_ready(plan_output.policy_targets)

            with self.profiler.time("device_get"):
                # plan_output.joint_action:   (B, N)
                # plan_output.policy_targets: (B, N, A)
                # plan_output.root_value:     (B,)
                actions_np = np.array(plan_output.joint_action)
                root_values_np = np.array(plan_output.root_value)
                policy_targets_np = np.array(plan_output.policy_targets)
                # Q-data for action-level AWPO (None for non-OSLA planners)
                root_child_actions_np = (
                    np.array(plan_output.root_child_actions)
                    if plan_output.root_child_actions is not None else None
                )  # (B, K, N) or None
                root_child_q_np = (
                    np.array(plan_output.root_child_q)
                    if plan_output.root_child_q is not None else None
                )  # (B, K) or None
                root_child_visits_np = (
                    np.array(plan_output.root_child_visits)
                    if plan_output.root_child_visits is not None else None
                )  # (B, K) or None

            if debug and (np.isnan(policy_targets_np).any() or np.isnan(actions_np).any()):
                logger.warning(f"(DataActor {self.actor_id}) NaN detected in plan output")

            with self.profiler.time("env_step"):
                step_keys = jax.random.split(step_key, B)
                next_obs, next_states, rewards, dones, *extra = self.env.step(
                    step_keys, states, actions_np
                )
                rewards_np = np.array(rewards)
                dones_np = np.array(dones)
                won_np = np.array(extra[0]) if extra else np.zeros(B, dtype=bool)

            for i in range(B):
                if not active[i]:
                    continue
                episodes[i].add_step(
                    Transition(
                        observation=np.array(observations[i]),
                        action=actions_np[i],
                        reward=float(rewards_np[i]),
                        done=bool(dones_np[i]),
                        policy_target=policy_targets_np[i],
                        value_target=float(root_values_np[i]),
                        agent_order=np.array(plan_output.agent_order),
                        root_child_actions=(
                            root_child_actions_np[i] if root_child_actions_np is not None else None
                        ),
                        root_child_q=(
                            root_child_q_np[i] if root_child_q_np is not None else None
                        ),
                        root_child_visits=(
                            root_child_visits_np[i] if root_child_visits_np is not None else None
                        ),
                    )
                )
                if won_np[i]:
                    episode_wins[i] = True
                if dones_np[i]:
                    active[i] = False

            observations = next_obs
            states = next_states

            if not any(active):
                break

        # Process all B episodes and ship to replay buffer.
        with self.profiler.time("process_episode"):
            all_items = []
            for ep in episodes:
                all_items.extend(process_episode(
                    ep,
                    self.config.train.unroll_steps,
                    self.config.train.n_step,
                    self.config.train.discount_gamma,
                    self.config.train.num_agents,
                ))

        with self.profiler.time("buffer_add"):
            if all_items:
                self.replay_buffer.add.remote(all_items, [1.0] * len(all_items))

        self.episodes_since_update += B

        # Param sync: resolve any in-flight fetch from the previous episode, then
        # immediately fire the next one so it runs during the next episode's MCTS.
        with self.profiler.time("param_sync"):
            if self._param_future is not None:
                ready, _ = ray.wait([self._param_future], timeout=0)
                if ready:
                    result = ray.get(self._param_future)
                    self.params = result["params"]
                    self.norm_state = result["norm_state"]
                    self._param_future = None
                    self.episodes_since_update = 0

            if (self._param_future is None
                    and self.episodes_since_update >= self.config.train.param_update_interval):
                self._param_future = self.learner.get_params.remote()

        win_rate = float(np.mean(episode_wins))
        mean_return = float(np.mean([ep.episode_return for ep in episodes]))
        return mean_return, win_rate
