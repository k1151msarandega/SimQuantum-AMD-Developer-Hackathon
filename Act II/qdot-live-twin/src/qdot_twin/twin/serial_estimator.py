"""CPU, one-frame-at-a-time state estimator. The honest baseline.

This exists to produce real, measured evidence of the twin falling behind
under load -- not to be a good solution. Do not optimize this file; the
point is that it's slow because it's unbatched, not because it's rigged.

Deliberately does real per-frame work (gradient-based estimate of where the
charge-transition boundary sits in the patch) rather than a placeholder, so
whatever latency shows up when this actually runs is measured, not staged --
same principle applied to the GPU numbers in step 1.
"""
import time

import numpy as np


class TwinState(dict):
    """Estimated device state: transition location, confidence, timestamp."""


def estimate(frame: np.ndarray) -> TwinState:
    """Turn a single raw frame into a state estimate. CPU, serial, no batching.

    Approximates the charge-transition boundary as the pixel with the
    largest local gradient magnitude in the patch -- a simple, real stand-in
    for "where did the charge state change" until the reused ensemble
    classifier (old repo's classifier.py) is repointed in step 4.
    """
    gx, gy = np.gradient(frame)
    grad_mag = np.sqrt(gx**2 + gy**2)
    peak_idx = np.unravel_index(np.argmax(grad_mag), grad_mag.shape)

    return TwinState(
        boundary_row=int(peak_idx[0]),
        boundary_col=int(peak_idx[1]),
        confidence=float(grad_mag[peak_idx]),
        estimated_at=time.time(),
    )
