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

This runs on an AMD GPU droplet (AMD Developer Cloud), not locally:

1. Spin up / resume your team's AMD GPU pod, open its Jupyter environment.
2. In a notebook terminal: `pip install -e .` (ROCm-enabled torch is
   already present on the droplet image).
3. Set `FIREWORKS_API_KEY` as an environment variable (needed for the
   "GPU batched + triage + LLM supervisor" mode).
4. Launch the console: `[YOUR ACTUAL STREAMLIT LAUNCH COMMAND]`
5. Open the Streamlit link Jupyter/the droplet exposes — that's the
   control console shown in the demo video/screenshots.

Note: the app is tied to the droplet's GPU session — killing the droplet
kills the Streamlit link. `scripts/run_full_demo.py` is a secondary,
GPU-required CLI path that reproduces the three-mode staleness comparison
chart headlessly, for anyone who wants numbers without the UI.

## Status
Scaffolded, implementation in progress. See `docs/pitch_notes.md` for
pitch framing and `configs/trajectory.yaml` for the demo trajectory.
