"""Step 3 deliverable: real, measured serial-CPU vs GPU-batched benchmark,
swept across batch sizes.

Generates a fixed pool of frames (no stream pacing -- this measures pure
compute cost, not simulated arrival timing), then for each batch size B:
  (a) serial: estimate() called B times in a Python loop (CPU)
  (b) batched: estimate_batch() called once on B frames (MI300X)

Both use the exact same ensemble weights (see perception/ensemble.py) --
this is the actual "what specifically runs on the MI300X and why" evidence
for the pitch, measured rather than assumed. Swept across sizes because a
single batch size can mislead: step 1 already showed a tiny workload can
lose to CPU on pure kernel-launch/transfer overhead (JAX vs Rust). The
real question is where the crossover is, since that's what the triage
agent (step 5) needs to know to decide when batching is worth it.
"""
import time

import numpy as np

from qdot_twin.stream.generator import _generate_patch, _make_model
from qdot_twin.twin.batch_estimator import estimate_batch
from qdot_twin.twin.serial_estimator import estimate

BATCH_SIZES = [8, 32, 128, 512, 2048]
MAX_N = max(BATCH_SIZES)

model = _make_model()
vxs = np.linspace(-5, 5, MAX_N)
vys = np.linspace(-5, 5, MAX_N)
all_frames = np.stack(
    [_generate_patch(model, vx, vy) for vx, vy in zip(vxs, vys)]
).astype(np.float32)
print(f"Generated pool of {all_frames.shape[0]} frames, patch shape {all_frames.shape[1:]}")

# Warm-up GPU call once, outside the timed sweep -- JIT/kernel-compile
# cost, same caveat as JAX in step 1.
_ = estimate_batch(all_frames[:8], device="cuda")

print(f"\n{'N':>6}  {'serial(s)':>10}  {'serial ms/f':>12}  {'gpu(s)':>10}  {'gpu ms/f':>10}  {'speedup':>8}")
for n in BATCH_SIZES:
    frames = all_frames[:n]

    t0 = time.time()
    for f in frames:
        estimate(f)
    t1 = time.time()
    serial_total = t1 - t0

    t0 = time.time()
    _ = estimate_batch(frames, device="cuda")
    t1 = time.time()
    batched_total = t1 - t0

    speedup = serial_total / batched_total
    print(f"{n:>6}  {serial_total:>10.4f}  {serial_total/n*1000:>12.4f}  "
          f"{batched_total:>10.4f}  {batched_total/n*1000:>10.4f}  {speedup:>7.2f}x")
