"""Continuous background live session for the WISER26 control console.

This is what makes app.py a real "instrument, not a script": Start spawns
a background thread that keeps pulling frames and running triage/staleness/
drift/estimation until Stop is called, instead of a request/response
"pick settings, click Run, wait, see a final chart" flow. Streamlit itself
stays request/response underneath -- this module is what makes that
irrelevant to the learner: the session keeps running on its own thread
regardless of whether the page is mid-render, and app.py just polls
snapshot() on a timer (see app.py's use of st.fragment(run_every=...)).

Deliberately wiring, not a reimplementation: the triage decision
(agent/triage.py:decide), the estimator (twin/batch_estimator.py:
estimate_batch), the staleness log (twin/staleness.py:StalenessLog), the
drift detector (perception/ood.py:RollingOODDetector), and the optional
LLM supervisor (agent/llm_supervisor.py) are all used completely
unchanged from pipeline.py's own batched-mode loop -- see
pipeline.py's _run_batched_loop(), which the loop below closely mirrors.
The one real difference is HOW frames are pulled: pipeline.py iterates
`for frame in stream(config_path)` directly; this instead drives a
QArrayTwinInstrument's next_frame() in a loop, so the SAME VoltageOverride
the console's Vx/Vy Parameters write into is what every frame in this
pipeline is generated from -- one real QCoDeS instrument feeding the
whole diagnostics stack, not two independent stream() consumers running
side by side that could silently disagree with each other.

Per the WISER26 console design decision: when the scripted trajectory
runs out while the session is still "started," the loop stops itself and
reports status="trajectory complete" -- it does NOT loop/restart the
generator. See app.py for how that's surfaced to the learner.
"""
import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qdot_edu.agent.thresholds import TriageThresholds
from qdot_edu.agent.triage import Tier, decide
from qdot_edu.hardware.qcodes_adapter import QArrayTwinInstrument
from qdot_edu.perception.ood import RollingOODDetector
from qdot_edu.stream.generator import build_model as build_qarray_model
from qdot_edu.stream.trajectory import load_trajectory_config
from qdot_edu.twin.batch_estimator import estimate_batch
from qdot_edu.twin.serial_estimator import CHEAP_N_MEMBERS
from qdot_edu.twin.staleness import StalenessLog
from qdot_edu.viz.potential_well import potential_surface

FLUSH_INTERVAL_S = 0.02   # same micro-batch window pipeline.py's batched modes use
RECENT_DRIFT_WINDOW = 20  # same window pipeline.py uses for the drift signal
LAG_HISTORY_MAXLEN = 1000  # cap so a long-running session's UI payload doesn't grow unbounded


@dataclass
class LiveSnapshot:
    """A full point-in-time read of the session, for app.py to render.

    LiveConsoleSession always replaces fields wholesale (new dict/list/
    array objects), never mutates one in place after it's been stored --
    so a caller holding a LiveSnapshot from .snapshot() never observes a
    half-updated value, without needing its own lock.
    """
    running: bool = False
    paused: bool = False
    done: bool = False
    error: Optional[str] = None
    status: str = "idle"

    frame_index: int = -1
    n_frames_total: int = 0

    vx: float = float("nan")
    vy: float = float("nan")
    vx_override_active: bool = False
    vy_override_active: bool = False

    frame: Optional[np.ndarray] = None       # device feed: latest stability-diagram patch
    fe_x: Optional[np.ndarray] = None        # potential-well surface grids (see viz/potential_well.py)
    fe_y: Optional[np.ndarray] = None
    fe_z: Optional[np.ndarray] = None

    tier: Optional[str] = None
    tier_counts: dict = field(default_factory=lambda: {"FULL": 0, "CHEAP": 0, "SKIP": 0})
    tier_compute_s: dict = field(default_factory=lambda: {"FULL": 0.0, "CHEAP": 0.0, "SKIP": 0.0})

    queue_depth: int = 0
    max_queue_depth: int = 0
    last_lag_s: float = 0.0
    lag_history: list = field(default_factory=list)  # list of (frame_index, lag_s, tier_name)

    drift_active: bool = False
    thresholds: tuple = (10, 12, 0.05)
    llm_supervised: bool = False
    supervisor_events: list = field(default_factory=list)

    started_at: Optional[float] = None
    elapsed_s: float = 0.0


class LiveConsoleSession:
    """Owns one QArrayTwinInstrument and runs the triage/staleness/drift
    pipeline against it continuously in a background thread, from Start
    until Stop (or the trajectory runs out).
    """

    def __init__(self, config_path: str, llm_supervised: bool = True, device: str = "cpu"):
        self.config_path = config_path
        self.llm_supervised = llm_supervised
        self.device = device

        cfg = load_trajectory_config(config_path)
        self.rows, self.cols = cfg.array_size

        self.station = QArrayTwinInstrument(
            name=f"qarray_console_{id(self)}", config_path=config_path,
        )
        # The potential-well panel reads an independent DotArray built from
        # the SAME shared model_params-derived matrices the instrument's
        # stream uses (see stream/generator.py:build_model) -- a second
        # observer of the same physical model, not a second copy of the
        # stream, and not a re-derived/approximated model of its own.
        self._fe_model = build_qarray_model(self.rows, self.cols)

        self._lock = threading.Lock()
        self._state = LiveSnapshot(n_frames_total=cfg.n_frames)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._supervisor = None

    # ---- public control, safe to call from the Streamlit main thread ----

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running():
            return
        self._stop_event.clear()
        self._pause_event.clear()
        with self._lock:
            self._state = LiveSnapshot(
                n_frames_total=self._state.n_frames_total, running=True, status="starting",
                started_at=time.time(), llm_supervised=self.llm_supervised,
            )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop and wait briefly for it.
        Safe to call even if the session already stopped itself (e.g.
        status="trajectory complete") -- join() on a finished thread
        returns immediately.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            self._state.running = False
            if self._state.status not in ("trajectory complete", "error"):
                self._state.status = "stopped"

    def pause(self) -> None:
        """Halt the background loop's frame consumption without tearing
        the session down -- unlike stop(), the thread stays alive and all
        accumulated state (staleness log, tier tally, thresholds) is
        preserved, so resume() picks up exactly where this left off.

        Because stream()'s pacing sleep happens INSIDE the generator at
        call time (not on a fixed wall-clock schedule), simply not calling
        next_frame() for a while has no drift/catch-up side effect --
        the stream is exactly as far along when you resume as it was when
        you paused, just later in wall-clock time.
        """
        self._pause_event.set()
        self._patch(paused=True, status="paused")

    def resume(self) -> None:
        """Undo pause() -- the background loop resumes pulling frames on
        its very next iteration.
        """
        self._pause_event.clear()
        self._patch(paused=False, status="running")

    def set_vx(self, vx: Optional[float]) -> None:
        """Inject a manual Vx, live, whether or not the session is running.
        Goes through the real QCoDeS Parameter, exactly as a lab engineer
        would set it on an actual instrument.
        """
        self.station.vx_override(vx)

    def set_vy(self, vy: Optional[float]) -> None:
        self.station.vy_override(vy)

    def clear_vx(self) -> None:
        """Release Vx back to the scripted trajectory (autopilot), leaving
        Vy's override state (if any) untouched.
        """
        self.station.override.clear_vx()

    def clear_vy(self) -> None:
        self.station.override.clear_vy()

    def snapshot(self) -> LiveSnapshot:
        with self._lock:
            return copy.copy(self._state)

    # ---- background thread body ------------------------------------------

    def _run(self) -> None:
        log = StalenessLog()
        buffer: list = []
        last_flush_time = time.time()
        last_full_update_time = time.time()

        ood = RollingOODDetector()
        recent_drift_flags: list = []
        tier_counts = {"FULL": 0, "CHEAP": 0, "SKIP": 0}
        tier_compute_s = {"FULL": 0.0, "CHEAP": 0.0, "SKIP": 0.0}
        max_queue_depth_seen = 0

        thresholds = TriageThresholds()
        history = None
        if self.llm_supervised:
            from qdot_edu.agent.llm_supervisor import LLMSupervisor, RollingHistory
            history = RollingHistory()
            self._supervisor = LLMSupervisor(thresholds, history)
            self._supervisor.start()

        self._patch(status="running")

        try:
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    # Deliberately just sleep-and-recheck rather than
                    # wait()-on-an-Event -- this loop also needs to wake up
                    # promptly on stop_event, and a plain poll here is
                    # simpler than juggling two Events in one wait call for
                    # a 100ms-granularity control action.
                    time.sleep(0.1)
                    continue

                data = self.station.next_frame()
                if data is None:
                    self._patch(running=False, done=True, status="trajectory complete")
                    return
                frame = self.station.last_frame
                buffer.append(frame)

                anomalous = ood.update_and_check(frame.data)
                recent_drift_flags.append(anomalous)
                if len(recent_drift_flags) > RECENT_DRIFT_WINDOW:
                    recent_drift_flags.pop(0)

                vx_ov, vy_ov = self.station.override.get()
                now = time.time()

                if now - last_flush_time < FLUSH_INTERVAL_S:
                    # Between flushes, still refresh the cheap, always-fresh
                    # fields every frame -- Vx/Vy and the device feed
                    # shouldn't visibly stall at the flush cadence even
                    # though tier/staleness/potential-well only update once
                    # per flush, same as pipeline.py's own flush semantics.
                    self._patch(
                        frame_index=frame.frame_index, vx=frame.vx, vy=frame.vy,
                        vx_override_active=vx_ov is not None, vy_override_active=vy_ov is not None,
                        frame=frame.data, queue_depth=len(buffer),
                        elapsed_s=now - (self._state.started_at or now),
                    )
                    continue

                last_flush_time = now
                if not buffer:
                    continue

                max_queue_depth_seen = max(max_queue_depth_seen, len(buffer))

                queue_depth = len(buffer)
                time_since_full = now - last_full_update_time
                recent_drift_activity = any(recent_drift_flags)
                tier = decide(queue_depth, time_since_full, recent_drift_activity, thresholds=thresholds)
                if history is not None:
                    history.record(queue_depth, time_since_full, recent_drift_activity, tier.name)

                tier_counts[tier.name] += 1

                t_compute_start = time.perf_counter()
                if tier is Tier.FULL:
                    estimate_batch(self._stack(buffer), device=self.device)
                    last_full_update_time = time.time()
                elif tier is Tier.CHEAP:
                    estimate_batch(self._stack(buffer), device=self.device, n_members=CHEAP_N_MEMBERS)
                tier_compute_s[tier.name] += time.perf_counter() - t_compute_start

                completion_time = time.time()
                last_lag = 0.0
                new_lag_entries = []
                for f in buffer:
                    lag = completion_time - f.emitted_at
                    log.record(frame_index=f.frame_index, t=completion_time, lag=lag, tier=tier.name)
                    new_lag_entries.append((f.frame_index, lag, tier.name))
                    last_lag = lag

                last_frame = buffer[-1]
                fe_x, fe_y, fe_z = potential_surface(
                    last_frame.vx, last_frame.vy, self.rows, self.cols, model=self._fe_model,
                )

                lag_history = self._state.lag_history + new_lag_entries
                if len(lag_history) > LAG_HISTORY_MAXLEN:
                    lag_history = lag_history[-LAG_HISTORY_MAXLEN:]

                self._patch(
                    frame_index=last_frame.frame_index, vx=last_frame.vx, vy=last_frame.vy,
                    vx_override_active=vx_ov is not None, vy_override_active=vy_ov is not None,
                    frame=last_frame.data, fe_x=fe_x, fe_y=fe_y, fe_z=fe_z,
                    tier=tier.name, tier_counts=dict(tier_counts), tier_compute_s=dict(tier_compute_s),
                    queue_depth=0, max_queue_depth=max_queue_depth_seen,
                    last_lag_s=last_lag, lag_history=lag_history,
                    drift_active=recent_drift_activity, thresholds=thresholds.snapshot(),
                    supervisor_events=list(self._supervisor.events) if self._supervisor is not None else [],
                    elapsed_s=completion_time - (self._state.started_at or completion_time),
                )
                buffer = []

        except Exception as e:
            self._patch(running=False, error=repr(e), status="error")
            raise
        finally:
            if self._supervisor is not None:
                self._supervisor.stop()

    @staticmethod
    def _stack(frames) -> np.ndarray:
        return np.stack([f.data for f in frames]).astype(np.float32)

    def _patch(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._state, k, v)
