"""Ensemble-disagreement signal, repointed from the old repo's classifier.py.

Old question: how much do ensemble members disagree on charge-state ID?
New question: same mechanism, fed into the drift flag alongside ood.py --
high disagreement is a second, independent signal that something about
the incoming data no longer matches what the twin expects.

Also serves as the real per-frame compute workload for the serial baseline
(twin/serial_estimator.py) and, batched, for the GPU estimator
(twin/batch_estimator.py). Weights are untrained/random: irrelevant for
now, since steps 2/3 only need realistic *compute cost and shape*. Step 4
is where classification accuracy would start to matter.

Models are cached per-device (see _get_ensemble), all built from the same
seed, so the CPU serial estimator and the GPU batched estimator are
comparing the *same model*, just batched differently.

ensemble_forward supports running a SUBSET of members (n_members) -- this
is what the triage agent's CHEAP tier uses (twin/serial_estimator.py's
estimate_cheap): genuinely less of the same real computation, not a
different heuristic, so its output stays directly comparable to the FULL
tier's.
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

    Used because this build's "Could not initialize NNPACK" warning (a CPU
    backend selection quirk) is a low-level C++ TORCH_WARN that ignores
    torch.backends.nnpack.enabled -- that documented flag does not suppress
    it on this hardware/build. Redirecting the raw file descriptor works
    regardless of where the warning actually originates. Only relevant to
    CPU execution; irrelevant to the GPU path but harmless there too.
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
    """Conv net, deliberately sized to have enough real FLOPs to be worth
    a GPU trip -- three conv layers, 32/64/128 channels, global pool,
    linear head.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


_model_cache: dict[str, list[_SmallCNN]] = {}  # keyed by device string


def _get_ensemble(device: str = "cpu") -> list[_SmallCNN]:
    """Return the ensemble on the requested device, building it once per
    device and caching. Same seed regardless of device, so CPU and GPU
    copies have identical weights -- the batched/serial comparison is
    apples to apples.
    """
    if device not in _model_cache:
        torch.manual_seed(0)
        models = [_SmallCNN().eval().to(device) for _ in range(N_ENSEMBLE_MEMBERS)]
        _model_cache[device] = models
    return _model_cache[device]


def ensemble_forward(frame: np.ndarray, n_members: int | None = None) -> torch.Tensor:
    """Run ensemble members on a single frame (CPU, unbatched).

    n_members: how many of the N_ENSEMBLE_MEMBERS to actually run. None (or
    N_ENSEMBLE_MEMBERS) runs the full ensemble -- the FULL tier. A smaller
    number runs genuinely less compute, proportionally -- the CHEAP tier.
    Always uses the FIRST n_members models (same seed -> same models every
    call, so results are reproducible/comparable across calls).

    Returns a (n_members, N_CLASSES) tensor of raw logits.
    """
    models = _get_ensemble("cpu")
    if n_members is not None:
        models = models[:n_members]

    x = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    with _suppress_stderr():
        with torch.no_grad():
            outputs = torch.stack([m(x)[0] for m in models])
    return outputs


def ensemble_disagreement(frame: np.ndarray) -> float:
    """Return a disagreement score across ensemble members: mean variance of
    their predicted class probabilities. High variance = members disagree.

    Always uses the full ensemble -- this is the drift-flag signal, not
    subject to the triage agent's cost tradeoffs.
    """
    outputs = ensemble_forward(frame)
    probs = torch.softmax(outputs, dim=-1)
    return float(probs.var(dim=0).mean().item())
