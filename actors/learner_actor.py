import os
import time
import numpy as np

import ray

from actors.loss import make_train_step, make_optimizer
from config import ExperimentConfig
from utils.logging_utils import logger
from utils.profiler import Profiler


@ray.remote(num_gpus=1)
class LearnerActor:
    """
    Trains the MuZero model on GPU.

    Pulls batches from the replay buffer, runs a JIT-compiled training step,
    and serves updated parameters to DataActors on request.
    """

    def __init__(self, obs_size: int, action_size: int, replay_buffer_actor, config: ExperimentConfig):
        # Don't preallocate the full MEM_FRACTION at init — data actors share
        # the same GPU. With preallocation, this process would claim 5.6 GB on
        # an 8 GB card before any data actor has started, leaving no room for them.
        # With it off, JAX allocates on demand and MEM_FRACTION acts as an upper cap.
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.70"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["GLOG_minloglevel"] = "2"

        import jax
        import jax.numpy as jnp
        import optax
        import orbax.checkpoint as ocp
        from pathlib import Path
        from utils.transforms import DiscreteSupport
        from model import FlaxMAMuZeroNet

        self.config = config
        try:
            devices = jax.devices()
            gpu_devices = [d for d in devices if "gpu" in str(d).lower() or "cuda" in str(d).lower()]
            if gpu_devices:
                logger.info(f"(Learner pid={os.getpid()}) JAX devices: {devices} — using {gpu_devices[0]}")
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        logger.info(f"(Learner) GPU memory at init (used/free/total MiB): {result.stdout.strip()}")
                except Exception as e:
                    logger.warning(f"(Learner) Could not query GPU memory: {e}")
            else:
                logger.warning(f"(Learner pid={os.getpid()}) No GPU found! Training will be slow. Devices: {devices}")
        except Exception as e:
            logger.error(f"(Learner) JAX device init failed: {e}")
            raise
        logger.info(f"(Learner pid={os.getpid()}) Initializing on GPU...")

        self.replay_buffer = replay_buffer_actor
        self.train_step_count = 0
        self.rng_key = jax.random.PRNGKey(0)

        value_support = DiscreteSupport(
            min=-config.model.value_support_size,
            max=config.model.value_support_size,
        )
        reward_support = DiscreteSupport(
            min=-config.model.reward_support_size,
            max=config.model.reward_support_size,
        )

        model = FlaxMAMuZeroNet(config.model, action_size)
        dummy_obs = jnp.ones((1, config.train.num_agents, obs_size))
        self.rng_key, init_key = jax.random.split(self.rng_key)
        self.params = model.init(init_key, dummy_obs)["params"]

        optimizer, lr_schedule = make_optimizer(config)
        self.opt_state = optimizer.init(self.params)
        self.lr_schedule = lr_schedule

        self.train_step = make_train_step(
            model, optimizer, value_support, reward_support, config
        )

        # JIT the EMA update. Without JIT, tree_map dispatches one GPU kernel
        # per parameter leaf (dozens of small launches). JIT fuses them into one.
        decay = config.train.ema_decay
        self.ema_update = jax.jit(lambda ema, params: jax.tree_util.tree_map(
            lambda e, p: decay * e + (1.0 - decay) * p, ema, params
        ))

        # Checkpointing — restore latest checkpoint if one exists.
        ckpt_dir = Path(config.train.checkpoint_dir).absolute()
        self.ckpt_manager = ocp.CheckpointManager(
            ckpt_dir,
            options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
        )
        # EMA parameters for the target encoder (BYOL-style consistency loss).
        # Initialized to match the online params; updated after every training step.
        self.ema_params = self.params

        latest = self.ckpt_manager.latest_step()
        if latest is not None:
            target = {
                "params": self.params,
                "opt_state": self.opt_state,
                "ema_params": self.ema_params,
                "step": np.array(0),
            }
            restored = self.ckpt_manager.restore(
                latest, args=ocp.args.StandardRestore(target)
            )
            self.params = restored["params"]
            self.opt_state = restored["opt_state"]
            self.ema_params = restored.get("ema_params", self.params)
            self.train_step_count = int(restored["step"])
            logger.info(
                f"(Learner pid={os.getpid()}) Restored checkpoint from step {self.train_step_count}."
            )
        else:
            logger.info(f"(Learner pid={os.getpid()}) No checkpoint found — starting fresh.")

        # Kick off the first prefetch so train() has a batch ready immediately.
        self._prefetch_future = self.replay_buffer.sample.remote(config.train.batch_size)

        # Cache lr at step 0; refreshed every lr_log_interval steps to avoid
        # per-step JAX dispatch overhead from calling the schedule function.
        self._cached_lr: float = float(lr_schedule(0))
        self._lr_log_interval = max(1, config.train.debug_interval)

        self.profiler = Profiler("learner", log_interval=config.train.debug_interval)

        logger.info(f"(Learner pid={os.getpid()}) Setup complete.")

    def _prefetch_batch(self):
        """Fires a non-blocking sample request and stores the future."""
        self._prefetch_future = self.replay_buffer.sample.remote(
            self.config.train.batch_size
        )

    def _train_step(self):
        """Runs one training step. Returns metrics dict or None if buffer empty."""
        import jax

        with self.profiler.time("sample_wait"):
            batch, weights, indices, q_data = ray.get(self._prefetch_future)
        if batch is None:
            self._prefetch_batch()
            return None

        # Fire the next buffer sample immediately so it overlaps with GPU compute.
        self._prefetch_batch()

        # Dispatch H2D transfer immediately after getting the batch (JAX async —
        # returns a future-like DeviceArray; actual DMA runs in background).
        # This overlaps the 3ms PCIe transfer with rng_split + any other CPU work,
        # so the GPU sees the batch ready by the time train_step is dispatched.
        jax_batch = jax.tree_util.tree_map(jax.device_put, batch)
        jax_weights = jax.device_put(np.array(weights, dtype=np.float32))

        with self.profiler.time("rng_split"):
            self.rng_key, train_key = jax.random.split(self.rng_key)

        with self.profiler.time("device_put"):
            # Ensure the async H2D transfer dispatched above has completed before
            # train_step consumes the arrays.
            jax.block_until_ready((jax_batch, jax_weights))

        with self.profiler.time("train_step"):
            # Convert q_data to JAX arrays for device; q_valid mask gates AWPO.
            jax_q_data = {
                k: jax.device_put(np.asarray(v)) for k, v in q_data.items()
            }
            self.params, self.opt_state, transfer_buf, new_priorities = self.train_step(
                self.params, self.opt_state, jax_batch, jax_weights, train_key,
                self.ema_params, jax_q_data,
            )
            # Block on params and the transfer buffer (which contains metrics +
            # priorities). new_priorities is a slice of transfer_buf so blocking
            # on transfer_buf covers it too — one wait for all GPU outputs.
            jax.block_until_ready((self.params, transfer_buf))
        self.train_step_count += 1

        # Dispatch EMA update without blocking — it runs asynchronously on GPU while
        # the CPU does D2H transfer and bookkeeping. JAX will implicitly wait for
        # ema_params when it's consumed by the next train_step call.
        with self.profiler.time("ema_update"):
            self.ema_params = self.ema_update(self.ema_params, self.params)

        with self.profiler.time("d2h_transfer"):
            # Single PCIe DMA: transfer_buf = [total, reward, policy, value,
            # consistency, grad_norm, priority_0, ..., priority_{B-1}]
            buf_np = np.array(transfer_buf)
        # EMA runs async on GPU; by now (after d2h_transfer ~1.7ms) it is likely
        # done, but JAX will sync it implicitly when next train_step uses ema_params.

        N_METRICS = 7  # total, reward, policy, value, consistency, policy_entropy, grad_norm
        priorities_np = buf_np[N_METRICS:]
        self.replay_buffer.update_priorities.remote(indices, priorities_np)

        if self.train_step_count % self.config.train.checkpoint_interval == 0:
            self._save_checkpoint()

        METRIC_KEYS = ["total_loss", "reward_loss", "policy_loss", "value_loss",
                       "consistency_loss", "policy_entropy", "grad_norm"]
        metrics = dict(zip(METRIC_KEYS, buf_np[:N_METRICS].tolist()))

        # Refresh cached lr every lr_log_interval steps; avoids per-step JAX dispatch.
        if self.train_step_count % self._lr_log_interval == 0:
            self._cached_lr = float(self.lr_schedule(self.train_step_count))
        metrics["learning_rate"] = self._cached_lr

        total_loss = metrics["total_loss"]
        grad_norm = metrics["grad_norm"]
        if not np.isfinite(total_loss):
            # Log each component so we can identify the source.
            bad = [k for k in METRIC_KEYS if not np.isfinite(metrics[k])]
            logger.warning(
                f"(Learner) step={self.train_step_count} non-finite total_loss={total_loss:.4f} "
                f"| bad components: {bad} "
                f"| reward={metrics['reward_loss']:.4f} policy={metrics['policy_loss']:.4f} "
                f"value={metrics['value_loss']:.4f} consistency={metrics['consistency_loss']:.4f}"
            )
        if not np.isfinite(grad_norm):
            logger.warning(f"(Learner) step={self.train_step_count} non-finite grad_norm={grad_norm:.4f}")

        debug = self.config.train.debug
        debug_interval = self.config.train.debug_interval
        if debug and self.train_step_count % debug_interval == 0:
            logger.info(
                f"(Learner) step={self.train_step_count} | "
                f"total={total_loss:.4f} "
                f"reward={metrics['reward_loss']:.4f} "
                f"policy={metrics['policy_loss']:.4f} "
                f"value={metrics['value_loss']:.4f} "
                f"consistency={metrics['consistency_loss']:.4f} "
                f"entropy={metrics['policy_entropy']:.3f} | "
                f"grad_norm={grad_norm:.3f}"
            )

        return metrics

    def run_training_loop(self, num_steps: int):
        """Runs a tight internal training loop for num_steps steps.

        Called once from the main loop per log interval instead of once per
        step. Eliminates Ray round-trip overhead between training steps —
        the learner stays on GPU continuously rather than waiting for the
        main process to re-dispatch it after each step.
        """
        metrics = None
        for _ in range(num_steps):
            result = self._train_step()
            if result is not None:
                metrics = result
        return metrics  # last non-None metrics, or None if buffer was empty

    def _save_checkpoint(self):
        import orbax.checkpoint as ocp
        state = {
            "params": self.params,
            "opt_state": self.opt_state,
            "ema_params": self.ema_params,
            "step": np.array(self.train_step_count),
        }
        self.ckpt_manager.save(self.train_step_count, args=ocp.args.StandardSave(state))
        self.ckpt_manager.wait_until_finished()
        logger.info(f"(Learner) Saved checkpoint at step {self.train_step_count}.")

    def get_params(self):
        return {"params": self.params}

    def get_train_step_count(self) -> int:
        return self.train_step_count
