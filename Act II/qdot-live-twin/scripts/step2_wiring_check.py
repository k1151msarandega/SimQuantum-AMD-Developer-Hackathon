"""Quick manual check that generator.py + serial_estimator.py actually work
together, before wiring them into staleness.py / pipeline.py.

Run this from the notebook (repo installed via `pip install -e .`), not
part of the package itself.
"""
import time

from qdot_twin.stream.generator import stream
from qdot_twin.twin.serial_estimator import estimate

t_start = time.time()
n_checked = 20  # just the first 20 frames -- this is a wiring check, not the real run

for i, frame in enumerate(stream("configs/trajectory.yaml")):
    t0 = time.time()
    state = estimate(frame.data)
    t1 = time.time()

    print(
        f"frame {frame.frame_index:4d}  "
        f"gen->estimate lag={t1 - frame.emitted_at:.4f}s  "
        f"estimate cost={t1 - t0:.4f}s  "
        f"(vx={frame.vx:.3f}, vy={frame.vy:.3f})  "
        f"state={dict(state)}"
    )

    if i + 1 >= n_checked:
        break

print(f"\nChecked {n_checked} frames in {time.time() - t_start:.2f}s wall clock.")
