#!/usr/bin/env python
"""CLI entry point: run one full demo pass and print a summary.

Example:
    python scripts/run_demo.py --mode batched_triage --config configs/trajectory_quick.yaml

This is the CPU-only, script-driven equivalent of what the (not yet
built -- see README.md Status) Streamlit app does interactively. Useful
for a quick sanity check that the install works end to end, and for
benchmarking (see docs/PORTING_NOTES.md's "Open items" -- this is the
natural place to measure real FULL-tier throughput on a target CPU).
"""
import argparse
import sys

from qdot_edu.pipeline import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["serial", "batched", "batched_triage", "batched_triage_llm"],
        default="batched_triage",
        help="which pipeline mode to run (see pipeline.py's module docstring)",
    )
    parser.add_argument(
        "--config",
        default="configs/trajectory_quick.yaml",
        help="path to a trajectory config (see configs/)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='compute device for the estimator, default "cpu" (this port'
             ' targets CPU-only; pass "cuda" only if you have a GPU and'
             " want to compare against the original repo's numbers)",
    )
    args = parser.parse_args()

    print(f"Running mode={args.mode!r} config={args.config!r} device={args.device!r}...")
    log = run(args.mode, args.config, device=args.device)

    df = log.to_dataframe()
    print(f"\n{len(df)} frames processed.")
    print(f"wall_clock_lag: mean={df['wall_clock_lag'].mean():.4f}s  "
          f"max={df['wall_clock_lag'].max():.4f}s  "
          f"final={df['wall_clock_lag'].iloc[-1]:.4f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
