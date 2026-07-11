"""Step 3 deliverable: real, measured serial-CPU vs GPU-batched benchmark.

Generates a fixed set of frames (no stream pacing -- this measures pure
compute cost, not simulated arrival timing), then times:
  (a) serial: estimate() called once per frame, in a Python loop (CPU)
  (b) batched: estimate_batch() called once on the whole set (MI300X)

Both use the exact same ensemble weights (see perception/ensemble.py) --
this is the actual "what specifically runs on the MI300X and why" evidence
for the pitch, measured rather than assumed.
"""
import time

import numpy as np

from qdot_twin.stream.generator import _generate_patch, _make_model
from qdot_twin.twin.batch_estimator import estimate_batch
from qdot_twin.twin.serial_estimator import estimate

N_FRAMES = 256

# Fixed set of frames across a voltage sweep -- not paced through stream(),
# since this benchmark cares about compute cost only, not arrival timing.
model = _make_model()
vxs = np.linspace(-5, 5, N_FRAMES)
vys = np.linspace(-5, 5, N_FRAMES)
frames = np.stack([_generate_patch(model, vx, vy) for vx, vy in zip(vxs, vys)]).astype(np.float32)
print(f"Generated {frames.shape[0]} frames, patch shape {frames.shape[1:]}")

# (a) serial CPU baseline, one frame at a time
t0 = time.time()
for f in frames:
    estimate(f)
t1 = time.time()
serial_total = t1 - t0
print(f"[serial cpu]   {N_FRAMES} frames in {serial_total:.4f}s "
      f"({serial_total / N_FRAMES * 1000:.4f}ms/frame)")

# Warm-up call on GPU first -- same JIT/kernel-compile caveat as JAX in
# step 1, so the reported number below is steady-state, not cold-start.
_ = estimate_batch(frames[:8], device="cuda")

# (b) GPU-batched, whole set in one call
t0 = time.time()
_ = estimate_batch(frames, device="cuda")
t1 = time.time()
batched_total = t1 - t0
print(f"[gpu batched]  {N_FRAMES} frames in {batched_total:.4f}s "
      f"({batched_total / N_FRAMES * 1000:.4f}ms/frame)")

print(f"\nSpeedup: {serial_total / batched_total:.2f}x")
