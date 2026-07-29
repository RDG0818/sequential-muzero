"""Wall-clock profiler: time() context manager per named operation, step()
logs and resets every log_interval calls. JAX note: call
jax.block_until_ready(result) inside time() to measure actual compute time,
not just async dispatch time.
"""

import time
from collections import defaultdict
from contextlib import contextmanager

from utils.logging_utils import logger


class Profiler:
    def __init__(self, name: str, log_interval: int = 100):
        self.name = name
        self.log_interval = log_interval
        self._step_count = 0
        self._totals: dict = defaultdict(float)
        self._counts: dict = defaultdict(int)

    @contextmanager
    def time(self, key: str):
        t0 = time.monotonic()
        yield
        self._totals[key] += time.monotonic() - t0
        self._counts[key] += 1

    def step(self):
        """Increment step counter; log and reset at log_interval."""
        self._step_count += 1
        if self._step_count % self.log_interval == 0:
            self._log()
            self._reset()

    def _log(self):
        if not self._totals:
            return
        parts = []
        for key in sorted(self._totals):
            n = self._counts[key]
            mean_ms = self._totals[key] / n * 1000
            parts.append(f"{key}={mean_ms:.1f}ms")
        logger.info(f"[profile:{self.name} @ step {self._step_count}]  " + "  |  ".join(parts))

    def _reset(self):
        self._totals.clear()
        self._counts.clear()
