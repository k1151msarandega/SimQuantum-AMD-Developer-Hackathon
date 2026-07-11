"""Staleness tracking: the core evaluation metric.

Logs two parallel series over the run, sharing an x-axis (frame index / time):
  - wall-clock lag: time since the twin's last fully-processed update
  - state-error magnitude: distance between twin's current estimate and
    QArray ground truth at that instant

Both get logged continuously under three regimes -- serial, GPU-batched,
and GPU-batched+triage -- for the comparison chart in the demo.
"""
from dataclasses import dataclass, field


@dataclass
class StalenessLog:
    timestamps: list = field(default_factory=list)
    wall_clock_lag: list = field(default_factory=list)
    state_error: list = field(default_factory=list)

    def record(self, t: float, lag: float, error: float) -> None:
        raise NotImplementedError

    def to_dataframe(self):
        """Return a tidy dataframe for plotting (see metrics.py)."""
        raise NotImplementedError
