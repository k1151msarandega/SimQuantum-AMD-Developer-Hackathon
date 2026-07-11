"""Ensemble-disagreement signal, repointed from the old repo's classifier.py.

Old question: how much do ensemble members disagree on charge-state ID?
New question: same mechanism, fed into the drift flag alongside ood.py --
high disagreement is a second, independent signal that something about
the incoming data no longer matches what the twin expects.

Also serves as the real per-frame compute workload for the serial baseline
(twin/serial_estimator.py) and, batched, for the GPU estimator (step 3) --
this is not a throwaway stand-in, it's the actual shared perception model
used across steps 2-4. Weights are untrained/random: irrelevant for now,
since steps 2/3 only need realistic *compute cost and shape*, matching the
old repo's classifier.py docstring target of "<5ms per patch on CPU".
Step 4 is where classification accuracy would start to matter.
"""
import contextlib
import os

import numpy as np
import torch
import torch.nn as nn

N_ENSEMBLE_MEMBERS = 5
N_CLASSES = 4  # arbitrary placeholder charge-state classes; revisit in step 4


@contextlib.contextmanager
def _suppress_stderr():
    """Redirect the process's stderr file descriptor to /dev/null for the
    duration of the block.

    Used because this build's "Could not initialize NNPACK" warning is a
    low-level C++ TORCH_WARN that ignores torch.backends.nnpack.enabled --
    that documented flag does not suppress it on this hardware/build.
    Redirecting the raw file descriptor works regardless of where the
    warning actually originates. Scoped tightly around just the forward
    pass so it doesn't hide anything else.
    """
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)


class _SmallCNN(nn.Module):
    """Small conv net, deliberately similar in shape to the old repo's
    per-patch classifier -- a few conv layers, global pool, linear head.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


_models: list[_SmallCNN] | None = None  # lazy module-level singleton


def _get_ensemble() -> list[_SmallCNN]:
    global _models
    if _models is None:
        torch.manual_seed(0)  # stable weights across calls within one run
        _models = [_SmallCNN().eval() for _ in range(N_ENSEMBLE_MEMBERS)]
    return _models


def ensemble_forward(frame: np.ndarray) -> torch.Tensor:
    """Run all ensemble members on a single frame (CPU, unbatched).

    Returns a (N_ENSEMBLE_MEMBERS, N_CLASSES) tensor of raw logits. This is
    the real per-frame cost that twin/serial_estimator.py measures.
    """
    models = _get_ensemble()
    x = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    with _suppress_stderr():
        with torch.no_grad():
            outputs = torch.stack([m(x)[0] for m in models])
    return outputs


def ensemble_disagreement(frame: np.ndarray) -> float:
    """Return a disagreement score across ensemble members: mean variance of
    their predicted class probabilities. High variance = members disagree.
    """
    outputs = ensemble_forward(frame)
    probs = torch.softmax(outputs, dim=-1)
    return float(probs.var(dim=0).mean().item())
