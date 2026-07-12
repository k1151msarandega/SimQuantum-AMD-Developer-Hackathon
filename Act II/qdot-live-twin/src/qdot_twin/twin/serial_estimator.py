"""CPU, one-frame-at-a-time state estimator. The honest baseline.

This exists to produce real, measured evidence of the twin falling behind
under load -- not to be a good solution. Do not optimize this file; the
point is that it's slow because it's unbatched, not because it's rigged.

Runs the ensemble CNN (perception/ensemble.py) once per frame, serially.

estimate() is the FULL tier: all N_ENSEMBLE_MEMBERS run.
estimate_cheap() is the CHEAP tier the triage agent (agent/triage.py) uses
under backlog: genuinely fewer ensemble members, same weights, same
architecture -- less of the same real computation, not a different
heuristic standing in for it. Cost scales down roughly proportionally with
member count, so the tradeoff the triage agent is reasoning about is real.
"""
import time

import numpy as np
import torch

from qdot_twin.perception.ensemble import ensemble_forward

CHEAP_N_MEMBERS = 2  # vs. the full N_ENSEMBLE_MEMBERS (5) -- roughly 2/5 the cost


class TwinState(dict):
    """Estimated device state: predicted class, ensemble disagreement, timestamp."""


def _estimate_with(frame: np.ndarray, n_members: int | None) -> TwinState:
    outputs = ensemble_forward(frame, n_members=n_members)
    probs = torch.softmax(outputs, dim=-1)
    mean_probs = probs.mean(dim=0)
    predicted_class = int(torch.argmax(mean_probs).item())
    disagreement = float(probs.var(dim=0).mean().item())

    return TwinState(
        predicted_class=predicted_class,
        disagreement=disagreement,
        estimated_at=time.time(),
    )


def estimate(frame: np.ndarray) -> TwinState:
    """FULL tier: turn a single raw frame into a state estimate using the
    entire ensemble. CPU, serial, no batching.
    """
    return _estimate_with(frame, n_members=None)


def estimate_cheap(frame: np.ndarray) -> TwinState:
    """CHEAP tier: same computation, fewer ensemble members. Real, honest,
    proportionally cheaper -- not a placeholder.
    """
    return _estimate_with(frame, n_members=CHEAP_N_MEMBERS)
