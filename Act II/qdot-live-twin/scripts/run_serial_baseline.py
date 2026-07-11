"""Step 2 deliverable: the honest serial-CPU baseline staleness curve.

Runs the full trajectory (all frames in configs/trajectory.yaml), processes
each frame serially through the ensemble estimator, and logs wall-clock
lag throughout. Saves both the raw log (CSV) and a plot.

This is the "here's the twin falling behind" evidence -- measured, not
assumed, and not tuned to force a particular outcome.
"""
import time

from qdot_twin.stream.generator import stream
from qdot_twin.stream.trajectory import load_trajectory_config
from qdot_twin.twin.serial_estimator import estimate
from qdot_twin.twin.staleness import StalenessLog

cfg = load_trajectory_config("configs/trajectory.yaml")
log = StalenessLog()
t_run_start = time.time()

for frame in stream("configs/trajectory.yaml"):
    estimate(frame.data)  # the actual work being timed; state itself not needed for staleness
    now = time.time()
    lag = now - frame.emitted_at
    log.record(frame_index=frame.frame_index, t=now, lag=lag)

    if frame.frame_index % 200 == 0:
        elapsed = now - t_run_start
        print(f"frame {frame.frame_index:4d}  lag={lag:.4f}s  run elapsed={elapsed:.1f}s")

df = log.to_dataframe()
df.to_csv("scripts/serial_baseline_staleness.csv", index=False)
print(f"\nSaved {len(df)} rows to scripts/serial_baseline_staleness.csv")
print(f"lag: min={df['wall_clock_lag'].min():.4f}s max={df['wall_clock_lag'].max():.4f}s "
      f"mean={df['wall_clock_lag'].mean():.4f}s")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["frame_index"], df["wall_clock_lag"])
ax.axvline(cfg.jump_at_frame, color="red", linestyle="--",
           label=f"injected jump (frame {cfg.jump_at_frame})")
ax.set_xlabel("frame index")
ax.set_ylabel("wall-clock lag (s)")
ax.set_title("Serial CPU baseline: staleness over the run")
ax.legend()
plt.tight_layout()
plt.savefig("scripts/serial_baseline_staleness.png", dpi=120)
print("Saved plot to scripts/serial_baseline_staleness.png")
plt.show()
