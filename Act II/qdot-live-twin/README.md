# qdot-live-twin

A GPU-accelerated live digital twin of a quantum dot device, with a
throughput-aware triage agent, built for the AMD Developer Hackathon Act II.

A **twin** passively mirrors the device's current state as it changes; it
does not steer the device anywhere (that's navigation/tuning, deliberately
out of scope). Success is staying synchronized (low staleness), not
"reaching a target."

## What it does
- Streams simulated charge-stability-diagram frames from QArray, playing
  the role of "the real device."
- Estimates device state from each frame, first serially on CPU (the
  honest baseline), then batched on an AMD MI300X via PyTorch/ROCm.
- Tracks staleness (wall-clock lag + state-error magnitude) continuously,
  under three regimes: serial, GPU-batched, and GPU-batched + triage agent.
- Flags drift when incoming data stops matching the twin's rolling
  expectation.
- A rule-based triage agent decides, per incoming frame, whether it's
  worth a full update, a cheap approximate one, or a skip — to keep
  staleness bounded when the twin falls behind under load.

## Setup
```
pip install -e .
```

## Run
```
python scripts/run_demo.py --mode batched_triage --config configs/trajectory.yaml
```

Modes: `serial`, `batched`, `batched_triage` — each produces a staleness
log; run all three to reproduce the comparison chart.

## Status
Scaffolded, implementation in progress. See `docs/pitch_notes.md` for
pitch framing and `configs/trajectory.yaml` for the demo trajectory.
