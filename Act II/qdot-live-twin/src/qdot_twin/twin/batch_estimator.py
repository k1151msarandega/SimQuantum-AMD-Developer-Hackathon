"""PyTorch/ROCm batched state estimator. The MI300X payoff.

Same job as serial_estimator.estimate, but processes a batch of queued
frames at once on the GPU to reduce per-frame latency under load. This is
the specific, honest GPU story for the pitch: batched twin updates, not a
vague 'powers the AI' claim.

Uses the SAME ensemble weights as serial_estimator (see
perception/ensemble.py's per-device cache, same seed) -- this is the same
model, batched, not a different/smaller one standing in for it.
"""
import time

import numpy as np
import torch

from qdot_twin.perception.ensemble import _get_ensemble


def estimate_batch(frames: np.ndarray, device: str = "cuda") -> list[dict]:
    """Turn a batch of raw frames into a list of state estimates, on GPU.

    frames: (B, H, W) array of B stacked patches.
    `device="cuda"` is correct even on ROCm -- PyTorch's ROCm build keeps
    the CUDA device string as its API surface.

    Returns a list of B dicts with the same keys as serial_estimator's
    TwinState (predicted_class, disagreement, estimated_at), so the two
    can be compared directly.
    """
    models = _get_ensemble(device)
    x = torch.from_numpy(frames).float().unsqueeze(1).to(device)  # (B, 1, H, W)

    with torch.no_grad():
        outputs = torch.stack([m(x) for m in models])  # (N_ENSEMBLE_MEMBERS, B, N_CLASSES)

    probs = torch.softmax(outputs, dim=-1)
    mean_probs = probs.mean(dim=0)                        # (B, N_CLASSES)
    predicted_classes = torch.argmax(mean_probs, dim=-1)   # (B,)
    disagreement = probs.var(dim=0).mean(dim=-1)           # (B,)

    now = time.time()
    return [
        {
            "predicted_class": int(predicted_classes[i].item()),
            "disagreement": float(disagreement[i].item()),
            "estimated_at": now,
        }
        for i in range(frames.shape[0])
    ]
