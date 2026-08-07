"""QArray-backed frame stream: plays the role of 'the real device'.

Wraps QArray's DotArray simulation to emit local stability-diagram patches
centered on a scripted device trajectory, at a rate that ramps up over the
run -- fast enough to eventually outpace the serial baseline. This is the
honest 'fire hose' data source; it does not know or care whether anything
downstream can keep up.

Uses QArray's default Rust implementation, not JAX -- the step-1 sanity
check measured Rust as ~180x faster than JAX for a single small diagram
(JAX pays JIT-compile cost per new shape; Rust has near-zero overhead).
GPU work stays concentrated in twin/batch_estimator.py, where it's
actually batched across many frames at once.
"""
import threading
import time
from typing import Iterator, NamedTuple, Optional

import numpy as np
from qarray import DotArray, charge_state_to_scalar

from qdot_twin.stream.trajectory import load_trajectory_config, voltage_at


class VoltageOverride:
    """Thread-safe, optional live override for the scripted trajectory.

    When set, stream() reads (vx, vy) from here each tick instead of the
    scripted trajectory value -- this is the injection point that makes
    QCoDeS control real rather than read-only: a QCoDeS set_cmd
    (hardware/qcodes_adapter.py) writes here, and the very next frame the
    twin/drift-detector processes reflects it, live. Left unset (None,
    the default), stream() behaves exactly as it always has -- this is an
    opt-in injection point, not a replacement for the scripted trajectory,
    so every existing serial/batched/triage run stays byte-for-byte
    unaffected unless a caller explicitly passes an override.
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

    def get(self) -> tuple:
        with self._lock:
            return self._vx, self._vy

# Capacitance matrices for a simple double-dot array. Matches the exact
# configuration verified against the real QArray API during step 1.
_CDD = [[0.0, 0.1], [0.1, 0.0]]
_CGD = [[1.0, 0.2], [0.2, 1.0]]

PATCH_WINDOW = 0.5   # +/- volts around the current trajectory point
PATCH_RES = 32        # patch resolution; small so single-frame Rust generation stays fast


class Frame(NamedTuple):
    data: np.ndarray          # (PATCH_RES, PATCH_RES) scalar stability-diagram patch
    vx: float                 # ground-truth gate voltage at emission time
    vy: float
    frame_index: int
    emitted_at: float         # wall-clock timestamp


def _make_model() -> DotArray:
    # implementation left at its default ('rust') deliberately -- see module docstring.
    return DotArray(Cdd=_CDD, Cgd=_CGD)


def _generate_patch(model: DotArray, vx: float, vy: float) -> np.ndarray:
    """Generate a small stability-diagram patch centered at (vx, vy)."""
    n = model.do2d_open(
        x_gate="P1", x_min=vx - PATCH_WINDOW, x_max=vx + PATCH_WINDOW, x_res=PATCH_RES,
        y_gate="P2", y_min=vy - PATCH_WINDOW, y_max=vy + PATCH_WINDOW, y_res=PATCH_RES,
    )
    return charge_state_to_scalar(n)


def build_model() -> DotArray:
    """Public constructor for the same QArray model stream() builds internally.

    Exposed so a second, independent reader -- the live console's
    free-energy panel, in particular -- uses the identical capacitance
    configuration rather than re-deriving its own copy and risking the two
    silently drifting apart. stream() keeps building and owning its own
    instance via _make_model(); this does not change that.
    """
    return _make_model()


def free_energy_patch(model: DotArray, vx: float, vy: float) -> np.ndarray:
    """The real electrostatic free-energy landscape around (vx, vy) -- the
    potential well the device's current charge configuration actually sits
    in, not a visual stand-in for one.

    Uses QArray's own DotArray.free_energy(n, vg): first reads the ground-
    state charge configuration n at the center point via ground_state_open
    (this is real QArray physics, the same call do2d_open's stability-
    diagram sweep is built on), then evaluates free_energy of that fixed
    configuration against a (PATCH_RES, PATCH_RES) grid of nearby gate
    voltages spanning the same +/-PATCH_WINDOW window _generate_patch uses,
    so the two patches are directly comparable side by side.

    Independent read, same device: this does not reuse or perturb the
    stream()-owned model/generator in any way -- same as qcodes_adapter.py's
    relationship to stream(), a second observer of the same QArray
    DotArray configuration, not a second copy of the stream itself.
    """
    center_vg = np.array([vx, vy])
    n_center = model.ground_state_open(center_vg)

    xs = np.linspace(vx - PATCH_WINDOW, vx + PATCH_WINDOW, PATCH_RES)
    ys = np.linspace(vy - PATCH_WINDOW, vy + PATCH_WINDOW, PATCH_RES)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")
    vg_grid = np.stack([x_grid, y_grid], axis=-1)  # (PATCH_RES, PATCH_RES, n_gate)

    fe = model.free_energy(n_center, vg_grid)  # (PATCH_RES, PATCH_RES, 1)
    return np.asarray(fe)[..., 0]


def stream(config_path: str, override: Optional[VoltageOverride] = None) -> Iterator[Frame]:
    """Yield Frames at the rate specified in the trajectory config.

    Rate ramps linearly from stream_rate_hz.start to stream_rate_hz.end over
    the run, per configs/trajectory.yaml, so early frames are easy to keep
    up with and later frames are deliberately not.

    `override`, if given, is checked every frame: any (vx, vy) component
    it currently holds replaces the scripted trajectory's value for that
    component only (see VoltageOverride's docstring). Default None means
    "follow the script," identical to this function's original behavior --
    every existing serial/batched/triage caller passes no override and is
    unaffected.
    """
    cfg = load_trajectory_config(config_path)
    model = _make_model()

    for i in range(cfg.n_frames):
        vx, vy = voltage_at(i, cfg)
        if override is not None:
            ovx, ovy = override.get()
            if ovx is not None:
                vx = ovx
            if ovy is not None:
                vy = ovy
        data = _generate_patch(model, vx, vy)

        progress = i / max(cfg.n_frames - 1, 1)
        rate_hz = cfg.stream_rate_hz_start + progress * (
            cfg.stream_rate_hz_end - cfg.stream_rate_hz_start
        )
        time.sleep(1.0 / rate_hz)

        yield Frame(data=data, vx=vx, vy=vy, frame_index=i, emitted_at=time.time())
