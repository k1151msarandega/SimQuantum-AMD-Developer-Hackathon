"""QArray-backed frame stream: plays the role of 'the real device'.

Wraps QArray's DotArray simulation to emit local stability-diagram patches
centered on a scripted device trajectory, at a rate that ramps up over the
run -- fast enough to eventually outpace the serial baseline. This is the
honest 'fire hose' data source; it does not know or care whether anything
downstream can keep up.

Uses QArray's default Rust implementation, not JAX -- Rust has near-zero
per-call overhead vs. JAX's per-shape JIT-compile cost. This was already
CPU compute in the original repo (the GPU work lived entirely in
twin/batch_estimator.py), so this file needs NO changes for the CPU port.
PORTED UNCHANGED from qdot-live-twin (Act II). See docs/PORTING_NOTES.md.

NEW for WISER26: the model this file drives is now built dynamically from
the config's `array_size` via qdot_edu.model_params (previously hardcoded
to a fixed 2-dot pair) -- see docs/PORTING_NOTES.md item 5 and
model_params.py's module docstring for the physical model and its
verification caveat. This is also the same shared source the
potential-well visualization (viz/potential_well.py) uses, so the two
can no longer silently drift apart.

NEW (live console): build_model() below is a public constructor exposed
so a second, independent reader of the SAME physical model -- the live
console's potential-well panel, specifically -- can build an identical
DotArray without re-deriving its own copy. stream() still owns and builds
its own instance internally; this is purely additive.
"""
import threading
import time
from typing import Iterator, NamedTuple, Optional

import numpy as np
from qarray import DotArray, charge_state_to_scalar

from qdot_edu import model_params
from qdot_edu.stream.trajectory import load_trajectory_config, voltage_at


class VoltageOverride:
    """Thread-safe, optional live override for the scripted trajectory.

    When set, stream() reads (vx, vy) from here each tick instead of the
    scripted trajectory value. Left unset (None, the default), stream()
    behaves exactly as it always has.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._vx: Optional[float] = None
        self._vy: Optional[float] = None

    def set(self, vx: Optional[float] = None, vy: Optional[float] = None) -> None:
        with self._lock:
            if vx is not None:
                self._vx = vx
            if vy is not None:
                self._vy = vy

    def clear(self) -> None:
        with self._lock:
            self._vx = None
            self._vy = None

    def clear_vx(self) -> None:
        """Release just Vx back to the scripted trajectory, leaving Vy
        (if overridden) untouched. Needed because the live console lets a
        learner release one axis to autopilot while still manually driving
        the other -- clear() alone can only release both at once.
        """
        with self._lock:
            self._vx = None

    def clear_vy(self) -> None:
        """Release just Vy back to the scripted trajectory -- see clear_vx()."""
        with self._lock:
            self._vy = None

    def get(self) -> tuple:
        with self._lock:
            return self._vx, self._vy


# Capacitance matrices are now built per-run from the trajectory config's
# array_size (see stream() below) via qdot_edu.model_params, not hardcoded
# here -- see docs/PORTING_NOTES.md item 5.

PATCH_WINDOW = 0.5   # +/- volts around the current trajectory point
PATCH_RES = 32        # patch resolution; small so single-frame Rust generation stays fast


class Frame(NamedTuple):
    data: np.ndarray          # (PATCH_RES, PATCH_RES) scalar stability-diagram patch
    vx: float                 # ground-truth gate voltage at emission time
    vy: float
    frame_index: int
    emitted_at: float         # wall-clock timestamp


def _make_model(rows: int, cols: int) -> DotArray:
    """Build a DotArray for a rows x cols dot grid using the shared
    capacitance model in qdot_edu.model_params.

    implementation left at its default ('rust') deliberately -- see
    module docstring.
    """
    Cdd, Cgd = model_params.dot_grid_matrices(rows, cols)
    return DotArray(Cdd=Cdd, Cgd=Cgd)


def build_model(rows: int, cols: int) -> DotArray:
    """Public constructor for the same QArray model stream() builds
    internally, for a given array_size.

    Exposed so a second, independent reader -- the live console's
    potential-well panel, in particular -- uses the identical capacitance
    configuration rather than re-deriving its own copy and risking the two
    silently drifting apart. stream() keeps building and owning its own
    instance via _make_model(); this does not change that.
    """
    return _make_model(rows, cols)


def _generate_patch(model: DotArray, vx: float, vy: float, x_gate: str, y_gate: str) -> np.ndarray:
    """Generate a small stability-diagram patch centered at (vx, vy),
    sweeping the two named gates. Any other gates in the array are left
    at QArray's default (see model_params.py's verification caveat).
    """
    n = model.do2d_open(
        x_gate=x_gate, x_min=vx - PATCH_WINDOW, x_max=vx + PATCH_WINDOW, x_res=PATCH_RES,
        y_gate=y_gate, y_min=vy - PATCH_WINDOW, y_max=vy + PATCH_WINDOW, y_res=PATCH_RES,
    )
    return charge_state_to_scalar(n)


def stream(config_path: str, override: Optional[VoltageOverride] = None) -> Iterator[Frame]:
    """Yield Frames at the rate specified in the trajectory config.

    Rate ramps linearly from stream_rate_hz.start to stream_rate_hz.end over
    the run, per configs/trajectory.yaml, so early frames are easy to keep
    up with and later frames are deliberately not.

    The array is built from the config's array_size (rows, cols); the
    first two gates (P1, P2) are always the ones swept, regardless of
    array_size -- see model_params.py's verification caveat for arrays
    larger than a 2-dot line.

    `override`, if given, is checked every frame: any (vx, vy) component
    it currently holds replaces the scripted trajectory's value for that
    component only. Default None means "follow the script," identical to
    this function's original behavior.
    """
    cfg = load_trajectory_config(config_path)
    rows, cols = cfg.array_size
    names = model_params.gate_names(rows, cols)
    if len(names) < 2:
        raise ValueError(
            f"array_size={cfg.array_size} gives only {len(names)} gate(s); "
            "need at least 2 to sweep a 2D stability diagram."
        )
    x_gate, y_gate = names[0], names[1]
    model = _make_model(rows, cols)

    for i in range(cfg.n_frames):
        vx, vy = voltage_at(i, cfg)
        if override is not None:
            ovx, ovy = override.get()
            if ovx is not None:
                vx = ovx
            if ovy is not None:
                vy = ovy
        data = _generate_patch(model, vx, vy, x_gate, y_gate)

        progress = i / max(cfg.n_frames - 1, 1)
        rate_hz = cfg.stream_rate_hz_start + progress * (
            cfg.stream_rate_hz_end - cfg.stream_rate_hz_start
        )
        time.sleep(1.0 / rate_hz)

        yield Frame(data=data, vx=vx, vy=vy, frame_index=i, emitted_at=time.time())
