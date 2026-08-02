"""Out-of-distribution / drift detector.

Question this answers: does this incoming frame match the twin's *current
rolling expectation* -- a window of recently-accepted frames -- rather than
a fixed training set. This matters: the device legitimately drifts (slow
creep should NOT trip the flag), but a sudden jump should.

Feature used: simple frame-level statistics (mean, std, pixel range) --
deliberately independent of perception/ensemble.py's classifier internals,
so this and ensemble_disagreement() are two genuinely separate signals
feeding the drift flag, not two views of the same computation.

Design decision on window adaptation: the window ALWAYS accepts new
frames, regardless of whether they were flagged anomalous. This means
after a real jump, the detector correctly keeps flagging for roughly one
window's worth of frames (while the window is still dominated by
pre-jump statistics), then naturally quiets down as the window fills with
post-jump frames and the new state becomes the baseline. Simpler and more
honest than trying to selectively decide what counts as "real drift worth
re-baselining to" -- and avoids the failure mode of a real jump getting
stuck as a permanent alarm.

PORTED UNCHANGED from qdot-live-twin (Act II), src/qdot_twin/perception/
ood.py. Pure NumPy, no GPU dependency whatsoever. See docs/PORTING_NOTES.md.
"""
from collections import deque

import numpy as np


def _frame_features(frame: np.ndarray) -> np.ndarray:
    """Cheap, interpretable per-frame summary: mean, std, pixel range."""
    return np.array([frame.mean(), frame.std(), frame.max() - frame.min()])


class RollingOODDetector:
    def __init__(self, window_size: int = 50, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self._window: deque = deque(maxlen=window_size)

    def update_and_check(self, frame: np.ndarray) -> bool:
        """Returns True if `frame` looks anomalous relative to the rolling
        window of recently accepted frames' features.
        """
        feature = _frame_features(frame)

        # Not enough history yet to judge -- treat as normal, just build up
        # the window.
        if len(self._window) < max(5, self.window_size // 5):
            self._window.append(feature)
            return False

        window_arr = np.array(self._window)
        mean = window_arr.mean(axis=0)
        std = window_arr.std(axis=0) + 1e-6  # avoid divide-by-zero

        z = np.abs((feature - mean) / std)
        score = float(z.mean())
        is_anomalous = score > self.z_threshold

        self._window.append(feature)  # always accept -- see module docstring
        return is_anomalous
