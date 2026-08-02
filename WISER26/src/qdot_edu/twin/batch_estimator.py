"""Batched state estimator -- CPU port of qdot-live-twin's PyTorch/ROCm GPU version.

Same job as serial_estimator.estimate, but processes a batch of queued
frames at once to reduce per-frame latency under load.

PORT NOTE (the actual change from the original file, see docs/PORTING_NOTES.md):
The original (src/qdot_twin/twin/batch_estimator.py) hardcoded
device="cuda" as the default and every internal call site. This version
defaults to device="cpu" and threads that default through consistently.
torch.stack([m(x) for m in models]) and the softmax/argmax/var math are
completely device-agnostic -- PyTorch runs the identical code path on CPU,
just without a GPU's parallelism. "Batched" on CPU still has a real,
teachable meaning: fewer Python-level calls into the model, better cache
locality per batch than N separate single-frame calls -- just a much
smaller speedup than the GPU version got. That gap is itself worth
surfacing to learners (see docs/lessons/ -- "batching helps everywhere,
but how much depends on the hardware" is a real systems lesson).

Uses the SAME ensemble weights as serial_estimator (see
perception/ensemble.py's per-device cache, same seed) -- this is the same
model, batched, not a different/smaller one standing in for it.
"""
import time

import numpy as np
import torch

from qdot_edu.perception.ensemble import _get_ensemble


def estimate_batch(frames: np.ndarray, device: str = "cpu", n_members: int | None = None) -> list[dict]:
    """Turn a batch of raw frames into a list of state estimates.

    frames: (B, H, W) array of B stacked patches.
    n_members: None runs the full ensemble (FULL tier); a smaller number
    runs genuinely less compute (CHEAP tier), same as serial_estimator.
    device: "cpu" by default for this port. If you have a CUDA/ROCm GPU
    available and want to compare, pass device="cuda" explicitly -- the
    code path is unchanged, only the default differs from the original repo.

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
