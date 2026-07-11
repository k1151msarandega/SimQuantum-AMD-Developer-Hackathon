"""Wider wiring check: samples frames spanning a real charge transition and
the injected jump, and reports both timing AND whether the estimator's
confidence value shows real variation (vs. the constant-artifact pattern
seen in the first 20 frames of the run).
"""
import time

from qdot_twin.stream.generator import stream
from qdot_twin.twin.serial_estimator import estimate

t_start = time.time()
confidences = []
costs = []

for i, frame in enumerate(stream("configs/trajectory.yaml")):
    t0 = time.time()
    state = estimate(frame.data)
    t1 = time.time()

    costs.append(t1 - t0)
    confidences.append(state["confidence"])

    # Print every 100th frame plus a window right around the injected jump.
    if i % 100 == 0 or 1195 <= i <= 1205:
        print(
            f"frame {frame.frame_index:4d}  "
            f"estimate cost={t1 - t0:.5f}s  "
            f"(vx={frame.vx:.3f}, vy={frame.vy:.3f})  "
            f"confidence={state['confidence']:.4f}  "
            f"boundary=({state['boundary_row']},{state['boundary_col']})"
        )

    if i + 1 >= 1300:  # covers the jump at frame 1200 with margin either side
        break

print(f"\nChecked 1300 frames in {time.time() - t_start:.2f}s wall clock.")
print(f"confidence range: min={min(confidences):.4f} max={max(confidences):.4f} "
      f"unique values={len(set(confidences))}")
print(f"estimate cost range: min={min(costs)*1000:.4f}ms max={max(costs)*1000:.4f}ms "
      f"mean={sum(costs)/len(costs)*1000:.4f}ms")
