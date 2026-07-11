"""PyTorch/ROCm batched state estimator. The MI300X payoff.

Same job as serial_estimator.estimate, but processes a batch of queued
frames at once on the GPU to reduce per-frame latency under load. This is
the specific, honest GPU story for the pitch: batched twin updates, not a
vague 'powers the AI' claim.
"""
import torch


def estimate_batch(frames: torch.Tensor, device: str = "cuda") -> list:
    """Turn a batch of raw frames into a list of state estimates, on GPU.

    `device="cuda"` is correct even on ROCm -- PyTorch's ROCm build keeps
    the CUDA device string as its API surface.
    """
    raise NotImplementedError
