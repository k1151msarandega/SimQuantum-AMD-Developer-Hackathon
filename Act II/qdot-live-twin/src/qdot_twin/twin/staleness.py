"""Staleness tracking: the core evaluation metric.

Logs wall-clock lag over the run: how far behind "now" the twin's last
processed frame is, in seconds. This is the headline metric -- visceral,
easy to read off a chart, and doesn't require a calibrated ground-truth
comparison to be meaningful.

state_error (distance between the twin's estimate and QArray ground truth)
is deliberately NOT computed yet: the ensemble classifier's weights are
still untrained/random (see perception/ensemble.py), so its predicted_class
has no calibrated relationship to the true charge state. Labeling an
untrained model's output as "error" would be a dishonest number dressed up
as a real one. This gets added once step 4 gives the classifier something
real to be right or wrong about.
"""
from dataclasses import dataclass, field


@dataclass
class StalenessLog:
    frame_indices: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    wall_clock_lag: list = field(default_factory=list)
    # Which tier actually produced this record -- "FULL", "CHEAP", "SKIP",
    # or None for callers (e.g. the serial baseline) that don't have tiers.
    # Added because pipeline.py used to call record() unconditionally
    # regardless of tier, so a SKIP'd (never-computed) frame was
    # indistinguishable in the log from a real FULL/CHEAP completion --
    # the log had no way to say "this one wasn't actually estimated."
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
