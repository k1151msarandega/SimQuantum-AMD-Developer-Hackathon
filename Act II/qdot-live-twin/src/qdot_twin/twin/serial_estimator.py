"""CPU, one-frame-at-a-time state estimator. The honest baseline.

This exists to produce real, measured evidence of the twin falling behind
under load -- not to be a good solution. Do not optimize this file; the
point is that it's slow because it's unbatched, not because it's rigged.

Runs the ensemble CNN (perception/ensemble.py) once per frame, serially.
This replaces an earlier gradient-based placeholder that measured ~0.1ms/
frame -- far too cheap to ever create backlog at any point in the demo
trajectory (confirmed empirically, not assumed). The ensemble forward pass
is real, representative compute cost, matching the old repo's classifier.py
docstring target of "<5ms per patch on CPU".
"""
import time

import numpy as np
import torch

from qdot_twin.perception.ensemble import ensemble_forward


class TwinState(dict):
    """Estimated device state: predicted class, ensemble disagreement, timestamp."""


def estimate(frame: np.ndarray) -> TwinState:
    """Turn a single raw frame into a state estimate. CPU, serial, no batching."""
    outputs = ensemble_forward(frame)  # (N_ENSEMBLE_MEMBERS, N_CLASSES)
    probs = torch.softmax(outputs, dim=-1)
    mean_probs = probs.mean(dim=0)
    predicted_class = int(torch.argmax(mean_probs).item())
    disagreement = float(probs.var(dim=0).mean().item())

    return TwinState(
        predicted_class=predicted_class,
        disagreement=disagreement,
        estimated_at=time.time(),
    )
