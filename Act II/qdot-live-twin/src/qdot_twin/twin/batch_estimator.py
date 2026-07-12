"""PyTorch/ROCm batched state estimator. The MI300X payoff.

Same job as serial_estimator.estimate, but processes a batch of queued
frames at once on the GPU to reduce per-frame latency under load. This is
the specific, honest GPU story for the pitch: batched twin updates, not a
vague 'powers the AI' claim.

Uses the SAME ensemble weights as serial_estimator (see
perception/ensemble.py's per-device cache, same seed) -- this is the same
model, batched, not a different/smaller one standing in for it.

n_members mirrors serial_estimator.py's CHEAP tier: a genuine subset of
the ensemble, same weights, run batched -- what pipeline.py's triage-driven
mode uses when the agent picks CHEAP instead of FULL.
"""
import time

import numpy as np
import torch

from qdot_twin.perception.ensemble import _get_ensemble


def estimate_batch(frames: np.ndarray, device: str = "cuda", n_members: int | None = None) -> list[dict]:
    """Turn a batch of raw frames into a list of state estimates, on GPU.

    frames: (B, H, W) array of B stacked patches.
    n_members: None runs the full ensemble (FULL tier); a smaller number
    runs genuinely less compute (CHEAP tier), same as serial_estimator.
    `device="cuda"` is correct even on ROCm -- PyTorch's ROCm build keeps
    the CUDA device string as its API surface.

    Returns a list of B dicts with the same keys as serial_estimator's
    TwinState (predicted_class, disagreement, estimated_at).
    """
    models = _get_ensemble(device)
    if n_members is not None:
        models = models[:n_members]

    x = torch.from_numpy(frames).float().unsqueeze(1).to(device)  # (B, 1, H, W)

    with torch.no_grad():
        outputs = torch.stack([m(x) for m in models])  # (n_members, B, N_CLASSES)

    probs = torch.softmax(outputs, dim=-1)
    mean_probs = probs.mean(dim=0)                        # (B, N_CLASSES)
    predicted_classes = torch.argmax(mean_probs, dim=-1)   # (B,)
    disagreement = probs.var(dim=0).mean(dim=-1)           # (B,)

    predicted_classes_cpu = predicted_classes.cpu().tolist()
    disagreement_cpu = disagreement.cpu().tolist()

    now = time.time()
    return [
        {"predicted_class": int(pc), "disagreement": float(d), "estimated_at": now}
        for pc, d in zip(predicted_classes_cpu, disagreement_cpu)
    ]
