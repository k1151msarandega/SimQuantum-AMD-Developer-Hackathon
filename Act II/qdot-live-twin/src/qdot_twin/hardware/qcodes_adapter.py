"""QCoDeS-compatible wrapper around the QArray-backed frame stream.

Real quantum-dot labs orchestrate instruments through QCoDeS (Instrument /
Station / Parameter). This module exposes the *exact same* stream that
pipeline.py consumes (stream/generator.py's stream(), backed by a real
QArray DotArray -- see that module's docstring) as a QCoDeS Instrument, so
a QCoDeS-based lab stack could subscribe to this twin's data source
without touching its own control software.

Intentionally a thin adapter, not a rebuild: it does not reimplement the
stream, the trajectory, or QArray -- it wraps the same stream() generator
already proven in steps 1-6, pulling one frame at a time, and exposes the
result as QCoDeS Parameters. This module is NOT wired into pipeline.py's
hot path and changes nothing about how the timed serial/batched/triage
runs work -- it exists to demonstrate framework compatibility for the
pitch (Product/Market Potential: "this could plug into a real QCoDeS lab
stack today"), not to replace the pipeline's own proven stream consumption.
"""
from typing import Optional

import numpy as np
from qcodes.instrument import Instrument

from qdot_twin.stream.generator import VoltageOverride
from qdot_twin.stream.generator import stream as qarray_stream


class QArrayTwinInstrument(Instrument):
    """A QCoDeS Instrument backed by the same QArray stream the twin uses.

    The `frame_index`/`vx`/`vy`/`frame` parameters all read from the most
    recently pulled frame. Call `.next_frame()` to advance the stream by
    one frame -- this mirrors how a real QCoDeS driver's get() triggers a
    hardware read, except here the "hardware" is QArray's DotArray
    (see stream/generator.py's own docstring on why Rust, not JAX, backs it).

    `vx_override`/`vy_override` are real, settable QCoDeS Parameters (not
    just read-only telemetry): setting one writes into a shared
    VoltageOverride that the underlying stream() checks on every frame
    (see stream/generator.py's VoltageOverride docstring), so the very
    next `.next_frame()` reflects it -- this is the bidirectional-control
    half of the demo, not just a read pull. `clear_overrides()` returns to
    following the scripted trajectory.
    """

    def __init__(self, name: str, config_path: str, **kwargs):
        super().__init__(name, **kwargs)
        self._config_path = config_path
        self._gen = None
        self._last_frame = None
        self._override = VoltageOverride()

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
