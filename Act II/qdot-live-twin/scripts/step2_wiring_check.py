"""Wider wiring check: samples frames spanning a real charge transition and
the injected jump, and reports both timing AND whether the estimator's
compute cost is now realistic (real ensemble forward pass, not the old
gradient placeholder that measured ~0.1ms regardless of content).
"""
import time

from qdot_twin.stream.generator import stream
from qdot_twin.twin.serial_estimator import estimate

t_start = time.time()
disagreements = []
costs = []

for i, frame in enumerate(stream("configs/trajectory.yaml")):
    t0 = time.time()
    state = estimate(frame.data)
    t1 = time.time()

    costs.append(t1 - t0)
    disagreements.append(state["disagreement"])

    if i % 100 == 0 or 1195 <= i <= 1205:
        print(
            f"frame {frame.frame_index:4d}  "
            f"estimate cost={t1 - t0:.5f}s  "
            f"(vx={frame.vx:.3f}, vy={frame.vy:.3f})  "
            f"predicted_class={state['predicted_class']}  "
            f"disagreement={state['disagreement']:.6f}"
        )

    if i + 1 >= 1300:
        break

print(f"\nChecked 1300 frames in {time.time() - t_start:.2f}s wall clock.")
print(f"disagreement range: min={min(disagreements):.6f} max={max(disagreements):.6f}")
print(f"estimate cost range: min={min(costs)*1000:.4f}ms max={max(costs)*1000:.4f}ms "
      f"mean={sum(costs)/len(costs)*1000:.4f}ms")
