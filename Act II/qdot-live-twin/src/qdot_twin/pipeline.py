"""Wires stream -> twin -> staleness -> drift -> triage into three runnable modes.

Modes:
  "serial"              -- FULL tier only, one frame at a time, CPU (step 2's baseline)
  "batched"             -- FULL tier only, micro-batched on GPU, no triage
  "batched_triage"      -- micro-batched on GPU, triage agent picks FULL/CHEAP/SKIP
                            per micro-batch using real queue depth, real staleness,
                            and a real drift signal (perception/ood.py), with static
                            rule-based thresholds (agent/triage.py)
  "batched_triage_llm"  -- same as batched_triage, except the three thresholds
                            (agent/thresholds.py) are tuned in the background by an
                            LLM supervisor (agent/llm_supervisor.py) reasoning over
                            recent backlog/staleness/drift trends. The LLM runs on
                            its own slow cadence in a background thread and never
                            sits in this hot loop -- see llm_supervisor.py's
                            module docstring for why. Requires FIREWORKS_API_KEY.

MICRO-BATCHING DESIGN NOTE (read before changing this file):
stream() is a blocking, real-time-paced generator running in this same
thread -- there is no separate producer thread accumulating frames while
we process. This is a SYNCHRONOUS APPROXIMATION of a producer/consumer
system, not a true concurrent one, and that's worth being upfront about
rather than letting it pass as something it isn't. The approximation:
frames are read one at a time as they arrive; after each arrival, if
FLUSH_INTERVAL_S of wall-clock time has elapsed since the last flush,
whatever is currently buffered gets flushed together. This is a real,
legitimate micro-batching policy (a max-latency flush trigger, used in
real systems) and produces the right qualitative dynamic: batch size grows
naturally under load (many frames arrive within one flush window) and
shrinks to ~1 when the stream is slow (each frame already exceeds the
flush interval on its own) -- exactly the condition GPU batching was shown
in step 3 to handle well, without being forced.

SKIP semantics: dropping the buffered frames entirely (no compute spent),
not deferring them to the next flush. Deferring would let the queue grow
forever once it crossed the SKIP threshold, since nothing would ever
shrink it -- a real deadlock risk. Dropping actually relieves backlog,
matching the triage agent's own "shed load entirely" language.

tier_counts (returned alongside the log from _run_batched when
use_triage=True) tracks how many times each Tier was actually chosen --
added after noticing batched and batched_triage produced near-identical
numbers, which raised the question of whether the agent was ever choosing
anything other than FULL under the current flush interval and queue
thresholds. Better to check directly than assume.
"""
import time
from typing import Literal

import numpy as np

from qdot_twin.agent.triage import Tier, decide
from qdot_twin.perception.ood import RollingOODDetector
from qdot_twin.stream.generator import stream
from qdot_twin.twin.batch_estimator import estimate_batch
from qdot_twin.twin.serial_estimator import CHEAP_N_MEMBERS, estimate
from qdot_twin.twin.staleness import StalenessLog

FLUSH_INTERVAL_S = 0.02       # micro-batch window for "batched"/"batched_triage"
RECENT_DRIFT_WINDOW = 20      # frames considered "recent" for the drift signal


def _stack(frames) -> np.ndarray:
    return np.stack([f.data for f in frames]).astype(np.float32)


def _run_serial(config_path: str) -> StalenessLog:
    log = StalenessLog()
    for frame in stream(config_path):
        estimate(frame.data)  # FULL tier, one frame at a time, CPU
        now = time.time()
        log.record(frame_index=frame.frame_index, t=now, lag=now - frame.emitted_at)
    return log


def _run_batched(config_path: str, use_triage: bool, llm_supervised: bool = False):
    log = StalenessLog()
    buffer: list = []
    last_flush_time = time.time()
    last_full_update_time = time.time()

    ood = RollingOODDetector()
    recent_drift_flags: list[bool] = []
    tier_counts = {"FULL": 0, "CHEAP": 0, "SKIP": 0}
    max_queue_depth_seen = 0

    thresholds = None
    history = None
    supervisor = None
    if llm_supervised:
        from qdot_twin.agent.thresholds import TriageThresholds
        from qdot_twin.agent.llm_supervisor import LLMSupervisor, RollingHistory
        thresholds = TriageThresholds()
        history = RollingHistory()
        supervisor = LLMSupervisor(thresholds, history)
        supervisor.start()

    try:
        return _run_batched_loop(
            config_path, use_triage, log, buffer, last_flush_time,
            last_full_update_time, ood, recent_drift_flags, tier_counts,
            max_queue_depth_seen, thresholds, history, supervisor,
        )
    finally:
        if supervisor is not None:
            supervisor.stop()


def _run_batched_loop(
    config_path, use_triage, log, buffer, last_flush_time, last_full_update_time,
    ood, recent_drift_flags, tier_counts, max_queue_depth_seen, thresholds, history, supervisor,
):
    for frame in stream(config_path):
        buffer.append(frame)

        anomalous = ood.update_and_check(frame.data)
        recent_drift_flags.append(anomalous)
        if len(recent_drift_flags) > RECENT_DRIFT_WINDOW:
            recent_drift_flags.pop(0)

        now = time.time()
        if now - last_flush_time < FLUSH_INTERVAL_S:
            continue  # not time to flush yet -- keep accumulating

        last_flush_time = now
        if not buffer:
            continue

        max_queue_depth_seen = max(max_queue_depth_seen, len(buffer))

        if use_triage:
            queue_depth = len(buffer)
            time_since_full = now - last_full_update_time
            recent_drift_activity = any(recent_drift_flags)
            tier = decide(queue_depth, time_since_full, recent_drift_activity, thresholds=thresholds)
            if history is not None:
                history.record(queue_depth, time_since_full, recent_drift_activity, tier.name)
        else:
            tier = Tier.FULL

        tier_counts[tier.name] += 1

        if tier is Tier.FULL:
            estimate_batch(_stack(buffer), device="cuda")
            last_full_update_time = time.time()
        elif tier is Tier.CHEAP:
            estimate_batch(_stack(buffer), device="cuda", n_members=CHEAP_N_MEMBERS)
        # Tier.SKIP: no compute spent -- see module docstring on why this
        # drops the buffer rather than deferring it.

        completion_time = time.time()
        for f in buffer:
            log.record(frame_index=f.frame_index, t=completion_time,
                       lag=completion_time - f.emitted_at)
        buffer = []

    # Flush whatever's left when the stream ends.
    if buffer:
        estimate_batch(_stack(buffer), device="cuda")
        completion_time = time.time()
        for f in buffer:
            log.record(frame_index=f.frame_index, t=completion_time,
                       lag=completion_time - f.emitted_at)

    supervisor_events = supervisor.events if supervisor is not None else None
    return log, tier_counts, max_queue_depth_seen, supervisor_events


def run(mode: Literal["serial", "batched", "batched_triage", "batched_triage_llm"], config_path: str):
    if mode == "serial":
        return _run_serial(config_path)
    elif mode == "batched":
        log, tier_counts, max_q, _ = _run_batched(config_path, use_triage=False)
        return log
    elif mode == "batched_triage":
        log, tier_counts, max_q, _ = _run_batched(config_path, use_triage=True)
        print(f"  tier decisions: {tier_counts}  (max queue depth seen: {max_q})")
        return log
    elif mode == "batched_triage_llm":
        log, tier_counts, max_q, events = _run_batched(config_path, use_triage=True, llm_supervised=True)
        print(f"  tier decisions: {tier_counts}  (max queue depth seen: {max_q})")
        if events:
            print(f"  LLM supervisor made {len(events)} threshold updates:")
            for ev in events:
                print(f"    {ev.old} -> {ev.new}  ({ev.reasoning})")
        else:
            print("  LLM supervisor never fired (run too short for its interval, or no window yet)")
        return log
    else:
        raise ValueError(f"unknown mode: {mode!r}")


def run_detailed(mode: Literal["serial", "batched", "batched_triage", "batched_triage_llm"], config_path: str):
    """Like run(), but also returns tier_counts / max_queue_depth / LLM-supervisor
    events where applicable. run() stays the simple single-log interface that
    run_full_demo.py and other existing scripts already depend on; this exists
    for app.py's dashboard, which needs the extra detail to display.

    Returns (log, tier_counts_or_None, max_queue_depth_or_None, events_or_None).
    """
    if mode == "serial":
        return _run_serial(config_path), None, None, None
    elif mode == "batched":
        log, tier_counts, max_q, events = _run_batched(config_path, use_triage=False)
        return log, None, max_q, None
    elif mode == "batched_triage":
        log, tier_counts, max_q, events = _run_batched(config_path, use_triage=True)
        return log, tier_counts, max_q, events
    elif mode == "batched_triage_llm":
        log, tier_counts, max_q, events = _run_batched(config_path, use_triage=True, llm_supervised=True)
        return log, tier_counts, max_q, events
    else:
        raise ValueError(f"unknown mode: {mode!r}")
