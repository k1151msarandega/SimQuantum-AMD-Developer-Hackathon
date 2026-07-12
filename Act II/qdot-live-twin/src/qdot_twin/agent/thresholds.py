"""Thread-safe, mutable triage thresholds.

Exists as its own tiny module so triage.py (the fast, hot-path decision
maker) and llm_supervisor.py (the slow, background threshold-tuner) can
share state without importing each other. triage.decide() reads a
snapshot on every call; llm_supervisor writes a new snapshot roughly once
every SUPERVISOR_INTERVAL_S seconds. Reads/writes are cheap tuple
copies under a lock -- this is not a hot-path bottleneck.
"""
import threading

# Same defaults as triage.py's module-level constants, so a decide() call
# with thresholds=None (untouched by this file) and a decide() call with
# a freshly-constructed TriageThresholds() behave identically.
DEFAULT_CHEAP_QUEUE_DEPTH = 10
DEFAULT_SKIP_QUEUE_DEPTH = 50
DEFAULT_STALE_THRESHOLD_S = 0.05


class TriageThresholds:
    def __init__(
        self,
        cheap_queue_depth: int = DEFAULT_CHEAP_QUEUE_DEPTH,
        skip_queue_depth: int = DEFAULT_SKIP_QUEUE_DEPTH,
        stale_threshold_s: float = DEFAULT_STALE_THRESHOLD_S,
    ):
        self._lock = threading.Lock()
        self._cheap_queue_depth = cheap_queue_depth
        self._skip_queue_depth = skip_queue_depth
        self._stale_threshold_s = stale_threshold_s

    def snapshot(self) -> tuple[int, int, float]:
        with self._lock:
            return self._cheap_queue_depth, self._skip_queue_depth, self._stale_threshold_s

    def update(self, cheap_queue_depth: int, skip_queue_depth: int, stale_threshold_s: float) -> None:
        with self._lock:
            self._cheap_queue_depth = cheap_queue_depth
            self._skip_queue_depth = skip_queue_depth
            self._stale_threshold_s = stale_threshold_s
