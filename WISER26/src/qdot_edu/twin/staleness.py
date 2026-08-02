"""Staleness tracking: the core evaluation metric.

Logs wall-clock lag over the run: how far behind "now" the twin's last
processed frame is, in seconds. This is the headline metric -- visceral,
easy to read off a chart, and doesn't require a calibrated ground-truth
comparison to be meaningful.

PORTED UNCHANGED from qdot-live-twin (Act II), src/qdot_twin/twin/
staleness.py. No GPU dependency -- nothing to adapt for the CPU port.
See docs/PORTING_NOTES.md.
"""
from dataclasses import dataclass, field


@dataclass
class StalenessLog:
    frame_indices: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    wall_clock_lag: list = field(default_factory=list)
    tiers: list = field(default_factory=list)

    def record(self, frame_index: int, t: float, lag: float, tier: str | None = None) -> None:
        self.frame_indices.append(frame_index)
        self.timestamps.append(t)
        self.wall_clock_lag.append(lag)
        self.tiers.append(tier)

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame({
            "frame_index": self.frame_indices,
            "timestamp": self.timestamps,
            "wall_clock_lag": self.wall_clock_lag,
            "tier": self.tiers,
        })
