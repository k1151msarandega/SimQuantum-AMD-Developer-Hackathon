"""Step 6 core artifact: run all three regimes and produce the final
staleness comparison chart -- the headline evidence for the pitch.

Runs the full trajectory three times (serial, batched, batched_triage),
each taking roughly the same wall-clock time as the others since all three
are bound by the same real-time frame-arrival pacing in stream(), not by
processing speed -- expect ~45-50s per mode, ~2.5 minutes total.
"""
import numpy as np

from qdot_twin.metrics import plot_staleness_comparison
from qdot_twin.pipeline import run
from qdot_twin.stream.trajectory import load_trajectory_config
from qdot_twin.twin.batch_estimator import estimate_batch
from qdot_twin.twin.serial_estimator import estimate


def _warmup():
    """Pay all one-time costs (CPU ensemble construction, GPU ensemble
    construction + transfer, first-call kernel compilation on both
    devices) BEFORE any timed run starts. Without this, whichever mode
    happens to run first absorbs a large, misleading one-time cost in its
    max lag -- exactly what happened when "batched" ran before
    "batched_triage" and inherited none of the cold-start cost purely
    because of run order, not because either mode is actually better at
    avoiding it.
    """
    dummy_frame = np.zeros((32, 32), dtype=np.float32)
    dummy_batch = np.zeros((4, 32, 32), dtype=np.float32)
    estimate(dummy_frame)
    estimate_batch(dummy_batch, device="cuda")


print("Warming up CPU and GPU ensembles...")
_warmup()

cfg = load_trajectory_config("configs/trajectory.yaml")

print("Running serial...")
serial_log = run("serial", "configs/trajectory.yaml")

print("Running batched...")
batched_log = run("batched", "configs/trajectory.yaml")

print("Running batched_triage...")
triage_log = run("batched_triage", "configs/trajectory.yaml")

logs = {"serial": serial_log, "batched": batched_log, "batched_triage": triage_log}

for name, log in logs.items():
    df = log.to_dataframe()
    df.to_csv(f"scripts/{name}_staleness.csv", index=False)
    print(f"{name:>15}: min={df['wall_clock_lag'].min():.4f}s "
          f"max={df['wall_clock_lag'].max():.4f}s mean={df['wall_clock_lag'].mean():.4f}s")

plot_staleness_comparison(logs, jump_at_frame=cfg.jump_at_frame,
                           save_path="scripts/full_comparison.png")
print("\nSaved scripts/full_comparison.png")
