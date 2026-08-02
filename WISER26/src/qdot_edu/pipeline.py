"""Wires stream -> twin -> staleness -> drift -> triage into three runnable modes.

Modes:
  "serial"              -- FULL tier only, one frame at a time, CPU (baseline)
  "batched"             -- FULL tier only, micro-batched, no triage
  "batched_triage"      -- micro-batched, triage agent picks FULL/CHEAP/SKIP
                            per micro-batch using real queue depth, real staleness,
                            and a real drift signal, with static rule-based
                            thresholds (agent/triage.py)
  "batched_triage_llm"  -- same as batched_triage, thresholds tuned by an LLM
                            supervisor. NOT YET PORTED for WISER26 -- see
                            docs/PORTING_NOTES.md. Raises ImportError until
                            agent/llm_supervisor.py exists here.

PORT NOTE: original pipeline.py hardcoded device="cuda" at every
estimate_batch() call site. This version threads a `device` parameter
(default "cpu") through instead. Everything else is unchanged; none of it
depended on the GPU. See docs/PORTING_NOTES.md.
"""
import time
from typing import Literal

import numpy as np

from qdot_edu.agent.triage import Tier, decide
from qdot_edu.perception.ood import RollingOODDetector
from qdot_edu.stream.generator import stream
from qdot_edu.twin.batch_estimator import estimate_batch
from qdot_edu.twin.serial_estimator import CHEAP_N_MEMBERS, estimate
from qdot_edu.twin.staleness import StalenessLog

FLUSH_INTERVAL_S = 0.02       # micro-batch window for "batched"/"batched_triage"
RECENT_DRIFT_WINDOW = 20      # frames considered "recent" for the drift signal
DEFAULT_DEVICE = "cpu"        # WISER26 CPU port -- was hardcoded "cuda"


def _stack(frames) -> np.ndarray:
    return np.stack([f.data for f in frames]).astype(np.float32)


def _run_serial(config_path: str) -> StalenessLog:
    log = StalenessLog()
    for frame in stream(config_path):
        estimate(frame.data)
        now = time.time()
        log.record(frame_index=frame.frame_index, t=now, lag=now - frame.emitted_at, tier="FULL")
    return log


def _run_batched(config_path: str, use_triage: bool, llm_supervised: bool = False, device: str = DEFAULT_DEVICE):
    log = StalenessLog()
    buffer: list = []
    last_flush_time = time.time()
    last_full_update_time = time.time()

    ood = RollingOODDetector()
    recent_drift_flags: list[bool] = []
    tier_counts = {"FULL": 0, "CHEAP": 0, "SKIP": 0}
    tier_compute_s = {"FULL": 0.0, "CHEAP": 0.0, "SKIP": 0.0}
    max_queue_depth_seen = 0

    thresholds = None
    history = None
    supervisor = None
    if llm_supervised:
        from qdot_edu.agent.thresholds import TriageThresholds
        from qdot_edu.agent.llm_supervisor import LLMSupervisor, RollingHistory
        thresholds = TriageThresholds()
        history = RollingHistory()
        supervisor = LLMSupervisor(thresholds, history)
        supervisor.start()

    try:
        return _run_batched_loop(
            config_path, use_triage, log, buffer, last_flush_time,
            last_full_update_time, ood, recent_drift_flags, tier_counts,
            max_queue_depth_seen, thresholds, history, supervisor,
            tier_compute_s, device,
        )
    finally:
        if supervisor is not None:
            supervisor.stop()


def _run_batched_loop(
    config_path, use_triage, log, buffer, last_flush_time, last_full_update_time,
    ood, recent_drift_flags, tier_counts, max_queue_depth_seen, thresholds, history, supervisor,
    tier_compute_s, device,
):
    for frame in stream(config_path):
        buffer.append(frame)

        anomalous = ood.update_and_check(frame.data)
        recent_drift_flags.append(anomalous)
        if len(recent_drift_flags) > RECENT_DRIFT_WINDOW:
            recent_drift_flags.pop(0)

        now = time.time()
        if now - last_flush_time < FLUSH_INTERVAL_S:
            continue

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

        t_compute_start = time.perf_counter()
        if tier is Tier.FULL:
            estimate_batch(_stack(buffer), device=device)
            last_full_update_time = time.time()
        elif tier is Tier.CHEAP:
            estimate_batch(_stack(buffer), device=device, n_members=CHEAP_N_MEMBERS)
        tier_compute_s[tier.name] += time.perf_counter() - t_compute_start

        completion_time = time.time()
        for f in buffer:
            log.record(frame_index=f.frame_index, t=completion_time,
                       lag=completion_time - f.emitted_at, tier=tier.name)
        buffer = []

    if buffer:
        t_compute_start = time.perf_counter()
        estimate_batch(_stack(buffer), device=device)
        tier_compute_s["FULL"] += time.perf_counter() - t_compute_start
        completion_time = time.time()
        for f in buffer:
            log.record(frame_index=f.frame_index, t=completion_time,
                       lag=completion_time - f.emitted_at, tier="FULL")

    supervisor_events = supervisor.events if supervisor is not None else None
    return log, tier_counts, max_queue_depth_seen, supervisor_events, tier_compute_s


def _run_serial_live(config_path: str, yield_every: int):
    """Generator twin of _run_serial() that yields partial StalenessLog
    snapshots as frames complete, instead of blocking until the whole run
    finishes. Serial has no tiers/queue/compute-time breakdown, so those
    fields are always None here -- app.py's live panel treats that as
    "not applicable to this mode," not an error.
    """
    log = StalenessLog()
    count = 0
    last_vx, last_vy = 0.0, 0.0
    for frame in stream(config_path):
        estimate(frame.data)
        now = time.time()
        log.record(frame_index=frame.frame_index, t=now, lag=now - frame.emitted_at, tier="FULL")
        last_vx, last_vy = frame.vx, frame.vy
        count += 1
        if count % yield_every == 0:
            yield {
                "done": False, "df": log.to_dataframe(), "tier_counts": None,
                "max_q": None, "tier_compute_s": None, "events": None,
                "frames_seen": frame.frame_index + 1, "vx": last_vx, "vy": last_vy,
            }
    yield {
        "done": True, "df": log.to_dataframe(), "tier_counts": None,
        "max_q": None, "tier_compute_s": None, "events": None, "log": log,
        "vx": last_vx, "vy": last_vy,
    }


def run_live(mode: Literal["serial", "batched", "batched_triage", "batched_triage_llm"],
             config_path: str, yield_every: int = 3, device: str = DEFAULT_DEVICE):
    """Generator variant of run_detailed() for live-updating UIs.

    Yields a dict of partial state every `yield_every` completed
    flushes/frames while the run is still in progress ("done": False),
    then a final dict ("done": True) carrying the same fields
    run_detailed() returns, plus the completed StalenessLog under "log".
    """
    if mode == "serial":
        yield from _run_serial_live(config_path, yield_every)
        return
    if mode not in ("batched", "batched_triage", "batched_triage_llm"):
        raise ValueError(f"unknown mode: {mode!r}")

    use_triage = mode in ("batched_triage", "batched_triage_llm")
    llm_supervised = mode == "batched_triage_llm"

    log = StalenessLog()
    buffer: list = []
    last_flush_time = time.time()
    last_full_update_time = time.time()
    ood = RollingOODDetector()
    recent_drift_flags: list[bool] = []
    tier_counts = {"FULL": 0, "CHEAP": 0, "SKIP": 0}
    tier_compute_s = {"FULL": 0.0, "CHEAP": 0.0, "SKIP": 0.0}
    max_queue_depth_seen = 0
    last_vx, last_vy = 0.0, 0.0

    thresholds = history = supervisor = None
    if llm_supervised:
        from qdot_edu.agent.thresholds import TriageThresholds
        from qdot_edu.agent.llm_supervisor import LLMSupervisor, RollingHistory
        thresholds = TriageThresholds()
        history = RollingHistory()
        supervisor = LLMSupervisor(thresholds, history)
        supervisor.start()

    flush_count = 0
    try:
        for frame in stream(config_path):
            buffer.append(frame)

            anomalous = ood.update_and_check(frame.data)
            recent_drift_flags.append(anomalous)
            if len(recent_drift_flags) > RECENT_DRIFT_WINDOW:
                recent_drift_flags.pop(0)

            now = time.time()
            if now - last_flush_time < FLUSH_INTERVAL_S:
                continue
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

            t_compute_start = time.perf_counter()
            if tier is Tier.FULL:
                estimate_batch(_stack(buffer), device=device)
                last_full_update_time = time.time()
            elif tier is Tier.CHEAP:
                estimate_batch(_stack(buffer), device=device, n_members=CHEAP_N_MEMBERS)
            tier_compute_s[tier.name] += time.perf_counter() - t_compute_start

            completion_time = time.time()
            for f in buffer:
                log.record(frame_index=f.frame_index, t=completion_time,
                           lag=completion_time - f.emitted_at, tier=tier.name)
            last_vx, last_vy = buffer[-1].vx, buffer[-1].vy
            buffer = []
            flush_count += 1

            if flush_count % yield_every == 0:
                yield {
                    "done": False, "df": log.to_dataframe(), "tier_counts": dict(tier_counts),
                    "max_q": max_queue_depth_seen, "tier_compute_s": dict(tier_compute_s),
                    "events": list(supervisor.events) if supervisor is not None else None,
                    "frames_seen": frame.frame_index + 1, "vx": last_vx, "vy": last_vy,
                }

        if buffer:
            t_compute_start = time.perf_counter()
            estimate_batch(_stack(buffer), device=device)
            tier_compute_s["FULL"] += time.perf_counter() - t_compute_start
            completion_time = time.time()
            for f in buffer:
                log.record(frame_index=f.frame_index, t=completion_time,
                           lag=completion_time - f.emitted_at, tier="FULL")
            last_vx, last_vy = buffer[-1].vx, buffer[-1].vy
    finally:
        if supervisor is not None:
            supervisor.stop()

    yield {
        "done": True, "df": log.to_dataframe(), "tier_counts": tier_counts,
        "max_q": max_queue_depth_seen, "tier_compute_s": tier_compute_s,
        "events": supervisor.events if supervisor is not None else None,
        "log": log, "vx": last_vx, "vy": last_vy,
    }


def run(mode: Literal["serial", "batched", "batched_triage", "batched_triage_llm"],
         config_path: str, device: str = DEFAULT_DEVICE):
    if mode == "serial":
        return _run_serial(config_path)
    elif mode == "batched":
        log, tier_counts, max_q, _, tier_compute_s = _run_batched(config_path, use_triage=False, device=device)
        return log
    elif mode == "batched_triage":
        log, tier_counts, max_q, _, tier_compute_s = _run_batched(config_path, use_triage=True, device=device)
        print(f"  tier decisions: {tier_counts}  (max queue depth seen: {max_q})")
        print(f"  tier compute time (s): {tier_compute_s}")
        return log
    elif mode == "batched_triage_llm":
        log, tier_counts, max_q, events, tier_compute_s = _run_batched(
            config_path, use_triage=True, llm_supervised=True, device=device)
        print(f"  tier decisions: {tier_counts}  (max queue depth seen: {max_q})")
        print(f"  tier compute time (s): {tier_compute_s}")
        if events:
            print(f"  LLM supervisor made {len(events)} threshold updates:")
            for ev in events:
                print(f"    {ev.old} -> {ev.new}  ({ev.reasoning})")
        else:
            print("  LLM supervisor never fired (run too short for its interval, or no window yet)")
        return log
    else:
        raise ValueError(f"unknown mode: {mode!r}")


def run_detailed(mode: Literal["serial", "batched", "batched_triage", "batched_triage_llm"],
                  config_path: str, device: str = DEFAULT_DEVICE):
    """Like run(), but also returns tier_counts / max_queue_depth / LLM-supervisor
    events / per-tier compute time where applicable. Returns
    (log, tier_counts_or_None, max_queue_depth_or_None, events_or_None,
    tier_compute_s_or_None).
    """
    if mode == "serial":
        return _run_serial(config_path), None, None, None, None
    elif mode == "batched":
        log, tier_counts, max_q, events, tier_compute_s = _run_batched(config_path, use_triage=False, device=device)
        return log, None, max_q, None, tier_compute_s
    elif mode == "batched_triage":
        log, tier_counts, max_q, events, tier_compute_s = _run_batched(config_path, use_triage=True, device=device)
        return log, tier_counts, max_q, events, tier_compute_s
    elif mode == "batched_triage_llm":
        log, tier_counts, max_q, events, tier_compute_s = _run_batched(
            config_path, use_triage=True, llm_supervised=True, device=device)
        return log, tier_counts, max_q, events, tier_compute_s
    else:
        raise ValueError(f"unknown mode: {mode!r}")
