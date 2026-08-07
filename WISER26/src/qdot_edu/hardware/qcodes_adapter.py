"""QCoDeS-compatible wrapper around the QArray-backed frame stream.

PORTED from qdot-live-twin (Act II), src/qdot_twin/hardware/qcodes_adapter.py,
with import paths updated (qdot_twin -> qdot_edu). Nothing here depended on
the GPU -- this module only ever pulled frames from stream(), never called
batch_estimator.py -- so the CPU port needed no other changes.

Real quantum-dot labs orchestrate instruments through QCoDeS (Instrument /
Station / Parameter). This module exposes the *exact same* stream that
pipeline.py consumes (stream/generator.py's stream(), backed by a real
QArray DotArray) as a QCoDeS Instrument, so a QCoDeS-based lab stack could
subscribe to this twin's data source without touching its own control
software -- and so this console's Vx/Vy controls are genuinely QCoDeS
Parameters, not a bespoke stand-in for one.

Intentionally a thin adapter, not a rebuild: it does not reimplement the
stream, the trajectory, or QArray -- it wraps the same stream() generator
already used by pipeline.py, pulling one frame at a time, and exposes the
result as QCoDeS Parameters. This module is NOT wired into pipeline.py's
own run()/run_live() hot path and changes nothing about how those modes
work; the live console (console/live_session.py) is a second, independent
consumer of stream(), same as this adapter is.
"""
from typing import Optional

import numpy as np
from qcodes.instrument import Instrument

from qdot_edu.stream.generator import VoltageOverride
from qdot_edu.stream.generator import stream as qarray_stream


class QArrayTwinInstrument(Instrument):
    """A QCoDeS Instrument backed by the same QArray stream the twin uses.

    The `frame_index`/`vx`/`vy`/`frame` parameters all read from the most
    recently pulled frame. Call `.next_frame()` to advance the stream by
    one frame -- this mirrors how a real QCoDeS driver's get() triggers a
    hardware read, except here the "hardware" is QArray's DotArray.

    `vx_override`/`vy_override` are real, settable QCoDeS Parameters (not
    just read-only telemetry): setting one writes into a shared
    VoltageOverride that the underlying stream() checks on every frame, so
    the very next `.next_frame()` reflects it -- this is the
    bidirectional-control half of the demo, not just a read pull.
    `clear_overrides()` returns to following the scripted trajectory.
    """

    def __init__(self, name: str, config_path: str, override: Optional[VoltageOverride] = None, **kwargs):
        super().__init__(name, **kwargs)
        self._config_path = config_path
        self._gen = None
        self._last_frame = None
        # override=None (the default) constructs a private VoltageOverride,
        # same as the original behavior. A caller that needs THIS
        # instrument's vx_override/vy_override Parameters to actually
        # affect a second, independent stream() consumer -- e.g. the live
        # console's background triage pipeline in console/live_session.py
        # -- passes that consumer's own VoltageOverride in here instead,
        # so both sides read/write the exact same shared object.
        self._override = override if override is not None else VoltageOverride()

        self.add_parameter(
            "vx_override",
            label="Injected Vx override",
            unit="V",
            get_cmd=lambda: self._override.get()[0],
            set_cmd=lambda v: self._override.set(vx=v),
        )
        self.add_parameter(
            "vy_override",
            label="Injected Vy override",
            unit="V",
            get_cmd=lambda: self._override.get()[1],
            set_cmd=lambda v: self._override.set(vy=v),
        )

        self.add_parameter(
            "frame_index",
            label="Frame index",
            get_cmd=lambda: self._last_frame.frame_index if self._last_frame else -1,
        )
        self.add_parameter(
            "vx",
            label="Gate voltage Vx",
            unit="V",
            get_cmd=lambda: self._last_frame.vx if self._last_frame else float("nan"),
        )
        self.add_parameter(
            "vy",
            label="Gate voltage Vy",
            unit="V",
            get_cmd=lambda: self._last_frame.vy if self._last_frame else float("nan"),
        )
        self.add_parameter(
            "frame",
            label="Stability diagram patch",
            get_cmd=lambda: self._last_frame.data if self._last_frame else None,
            snapshot_value=False,  # patches are arrays; don't dump them into every station snapshot
        )

    @property
    def last_frame(self):
        """The full Frame namedtuple (data, vx, vy, frame_index, emitted_at)
        from the most recent next_frame() call, or None before the first
        pull. The individual vx/vy/frame_index/frame Parameters above each
        read one field of this same object -- exposed as a whole here so a
        caller driving a full staleness/triage loop off this instrument
        (console/live_session.py) can read everything from one pull
        without four separate Parameter .get() calls.
        """
        return self._last_frame

    @property
    def override(self) -> VoltageOverride:
        """The shared VoltageOverride this instrument writes into. Exposed
        so a caller that needs to build its OWN stream() against the same
        live override (e.g. console/live_session.py's background pipeline
        thread) can share exactly this instrument's override state,
        instead of the instrument and the pipeline silently drifting onto
        two different overrides that don't affect each other.
        """
        return self._override

    def next_frame(self) -> Optional[np.ndarray]:
        """Advance the underlying QArray stream by exactly one frame.

        Lazily creates the generator on first call so constructing the
        instrument itself never blocks on QArray or the trajectory config.
        """
        if self._gen is None:
            self._gen = qarray_stream(self._config_path, override=self._override)
        try:
            self._last_frame = next(self._gen)
        except StopIteration:
            return None
        return self._last_frame.data

    def clear_overrides(self) -> None:
        """Stop injecting -- subsequent frames follow the scripted trajectory again."""
        self._override.clear()
