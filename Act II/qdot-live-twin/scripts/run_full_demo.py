"""Step 6 core artifact: run all three regimes and produce the final
staleness comparison chart -- the headline evidence for the pitch.

Runs the full trajectory three times (serial, batched, batched_triage),
each taking roughly the same wall-clock time as the others since all three
are bound by the same real-time frame-arrival pacing in stream(), not by
processing speed -- expect ~45-50s per mode, ~2.5 minutes total.
"""
from qdot_twin.metrics import plot_staleness_comparison
from qdot_twin.pipeline import run
from qdot_twin.stream.trajectory import load_trajectory_config

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
