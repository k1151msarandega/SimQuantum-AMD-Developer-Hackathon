"""Step 4: drift-flag signals, measured across the full trajectory.

Logs both independent signals per frame:
  - ensemble_disagreement(frame): ensemble variance (perception/ensemble.py)
  - RollingOODDetector.update_and_check(frame): rolling-window anomaly flag
    (perception/ood.py)

Deliberately does NOT yet fuse them into one final drift flag with a chosen
threshold -- that threshold should be picked from what these numbers
actually look like, not guessed in advance. This script produces the
evidence needed to pick it.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from qdot_twin.perception.ensemble import ensemble_disagreement
from qdot_twin.perception.ood import RollingOODDetector
from qdot_twin.stream.generator import stream

ood = RollingOODDetector(window_size=50, z_threshold=3.0)

frame_indices = []
disagreements = []
ood_flags = []

t0 = time.time()
for frame in stream("configs/trajectory.yaml"):
    d = ensemble_disagreement(frame.data)
    anomalous = ood.update_and_check(frame.data)

    frame_indices.append(frame.frame_index)
    disagreements.append(d)
    ood_flags.append(anomalous)

    if frame.frame_index % 200 == 0:
        print(f"frame {frame.frame_index:4d}  disagreement={d:.6f}  ood_anomalous={anomalous}")

print(f"\nRan {len(frame_indices)} frames in {time.time() - t0:.1f}s")

frame_indices = np.array(frame_indices)
disagreements = np.array(disagreements)
ood_flags = np.array(ood_flags)

n_ood_fired = int(ood_flags.sum())
print(f"OOD flag fired on {n_ood_fired} / {len(ood_flags)} frames")
print(f"disagreement: min={disagreements.min():.6f} max={disagreements.max():.6f} "
      f"mean={disagreements.mean():.6f} std={disagreements.std():.6f}")

fired_indices = frame_indices[ood_flags]
print(f"OOD fired at frame indices (first 30 shown): {fired_indices[:30].tolist()}")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(frame_indices, disagreements)
axes[0].axvline(1200, color="red", linestyle="--", label="injected jump (frame 1200)")
axes[0].set_ylabel("ensemble disagreement")
axes[0].legend()

axes[1].plot(frame_indices, ood_flags.astype(int))
axes[1].axvline(1200, color="red", linestyle="--")
axes[1].set_ylabel("OOD anomalous (0/1)")
axes[1].set_xlabel("frame index")

plt.tight_layout()
plt.savefig("scripts/step4_drift_signals.png", dpi=120)
print("Saved plot to scripts/step4_drift_signals.png")
plt.show()
