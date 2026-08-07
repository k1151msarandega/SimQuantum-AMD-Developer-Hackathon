# qdot-edu: a live digital twin of a quantum dot device, as a teaching tool

**WISER 2026 Education Challenge submission.**

A CPU-runnable, interactive lesson built around a real systems-engineering
problem: keeping a "digital twin" of a semiconductor quantum dot array in
sync with a live stream of simulated device data, when the data arrives
faster than a naive estimator can process it. Learners watch a serial
baseline fall behind in real time, then see an adaptive triage agent
fix it -- and along the way, learn how gate-defined quantum dot arrays
work, how digital twins are built and used in quantum hardware, and how
to reason about throughput/latency tradeoffs in a live system.

> Forked and rebuilt from `qdot-live-twin`, a prior hackathon project
> (AMD-GPU-accelerated, not education-focused). See
> [`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md) for exactly what
> changed and why. **Status: actively being rebuilt for this
> submission -- see the Status section below before assuming any given
> piece is finished.**

## Status
`app.py` now has two tabs: **Live Console** (a continuously-running
instrument session -- QCoDeS-backed Vx/Vy control, a live device feed, a
real free-energy potential well, and live triage/staleness/drift
diagnostics; see [`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md) for what
that's built on) and **Mode comparison** (the original run-to-completion
serial/batched/triage comparison). `agent/llm_supervisor.py` and
`hardware/qcodes_adapter.py` are both now ported (see PORTING_NOTES).
Still open: `notebooks/` lesson content, and the rate/threshold
re-tuning tracked in PORTING_NOTES' "Open items" -- do that before
assuming triage actually varies tier on the CPU-tuned configs. Treat this
README as accurate about *intent and architecture*; check PORTING_NOTES'
"Open items" before assuming every piece has been run end to end.

## Target audience
Undergraduate or early-graduate students in physics, EE, or CS with
working Python but little/no hands-on exposure to real quantum-hardware
control systems. No prior knowledge of quantum dot devices or digital
twins assumed. Full detail: [`docs/LEARNING_OBJECTIVES.md`](docs/LEARNING_OBJECTIVES.md).

## Learning objectives
By the end of the lesson arc, a learner can explain what a quantum-hardware
digital twin is and why it's used; explain the constant-capacitance model
behind charge-stability diagrams; reason about throughput/latency
tradeoffs and system staleness under load; design and critique a
multi-signal triage policy; interpret drift detection; and connect an
electrostatic energy model to spatial intuition about electron trapping.
Full list with rationale: [`docs/LEARNING_OBJECTIVES.md`](docs/LEARNING_OBJECTIVES.md).

## Methodology
The lesson is built around one continuously-developed running example
(not a series of disconnected demos): a simulated quantum dot array
(via [QArray](https://github.com/b-vanstraaten/qarray)'s constant-
capacitance model) streams charge-stability-diagram frames at an
increasing rate. Learners first see a serial estimator fall behind
(a real, measured effect -- not scripted), then a batched estimator do
better, then a rule-based triage agent make explicit
cost/correctness tradeoffs using real signals (queue depth, staleness,
drift). A 3D potential-well visualization -- QArray's own
`free_energy(n, vg)`, not a schematic stand-in -- gives spatial intuition
for what the abstract charge-stability numbers mean physically: the
electrostatic energy landscape the device's current charge configuration
actually sits in. The Live Console (`app.py`'s first tab) puts all of this
in one continuously-running instrument session, driven by a real
QCoDeS-backed Vx/Vy instrument, instead of a series of one-shot runs. Each
notebook stage is a working notebook that builds on the last, mirroring
how the underlying engineering was actually developed -- see
[`docs/lessons/README.md`](docs/lessons/README.md) for the full arc, and
[`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md) for the engineering
history this was built from.

## Tech stack
- **[QArray](https://github.com/b-vanstraaten/qarray)** -- constant-capacitance quantum dot array simulator (Rust backend, CPU)
- **QCoDeS** -- the instrument/control layer for the Live Console's Vx/Vy panel (`hardware/qcodes_adapter.py`)
- **PyTorch** (CPU) -- the ensemble "perception" model that stands in for a real state-estimation workload
- **NumPy / pandas** -- signal processing, staleness logging
- **Streamlit** -- the interactive app (`app.py`)
- **Plotly / Matplotlib** -- 2D charge-stability plots and the 3D potential-well surface
- **Jupyter notebooks** -- the guided lesson arc (`notebooks/`, not yet built -- see Status)

No GPU required anywhere in this repo -- see
[`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md) for what that took. The
LLM threshold supervisor is the one optional exception that needs network
access (a Fireworks API key) -- everything else, including the rest of
the Live Console, runs fully offline.

## Install & run
Developed and run from Google Colab, not a local install -- there is no
tested local `pip`/venv workflow for this repo (see
[`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md)).

**Use [`notebooks/00_launch_app.ipynb`](notebooks/00_launch_app.ipynb) --
it is the actual, tested install/launch procedure** (install, import
check, a no-UI pipeline smoke test, then the app via a tunnel). Don't
reconstruct the install steps from scratch; a first real run already
caught an install-ordering bug (see
[`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md)'s "Resolved" section) --
install torch and the project **in a single `pip install -e .` call**,
not as two separate `pip install` commands. Installing torch separately
first (e.g. via `pip install torch --index-url .../whl/cpu`, which an
earlier version of this README suggested) collided with `qarray`'s exact
`numpy==2.2.4` pin and silently left `scipy` uninstalled. Also **restart
the Colab runtime after installing, before importing anything** --
Colab keeps already-loaded numpy/scipy in memory, so a pip-level swap
doesn't take effect until the process restarts.

To use the LLM threshold supervisor in the Live Console, set
`FIREWORKS_API_KEY` before launching. Without it, the console still runs
fully -- the supervisor thread just logs an error each tick and leaves
thresholds at their defaults.

## User guide
(To be expanded once `notebooks/` exist -- see
[`docs/lessons/README.md`](docs/lessons/README.md) for the planned
notebook-by-notebook walkthrough in the meantime.) The intended flow: work
through `notebooks/00`-`07` in order, then open the Streamlit app.
**Live Console** tab: click Start to power on a continuously-running
session, adjust Vx/Vy live from the sidebar (or leave them on autopilot),
and watch the device feed, potential well, and triage/staleness/drift
diagnostics update in real time; Stop ends the session. **Mode
comparison** tab: pick a mode and config and click Run for the
side-by-side serial vs. batched vs. triage comparison, run to completion.

## Future improvements / scalability
See [`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md)'s "Open items" for
the concrete near-term TODOs -- rate/threshold re-tuning against real CPU
timing (this also affects whether the Live Console's tier panel visibly
changes tier), and a full end-to-end run of this session's additions
(Live Console, ported QCoDeS adapter and LLM supervisor) since none of it
has been executed in Streamlit yet, only reviewed against the actual
installed QArray API. Longer-term: explore array shapes beyond a 2-dot
line now that `array_size` is live (see `model_params.py`) -- e.g. a
guided exercise on how triage dynamics change with array size; consider a
"build-your-own-triage-policy" exercise as a capstone notebook.

## Credits & licensing
See [`assets/CREDITS.md`](assets/CREDITS.md) for attribution on all
reused imagery/renders. Original engineering (`qdot-live-twin`, Act II
hackathon) rebuilt and extended per [`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md).

## AI-assistance disclosure
This repo was developed with AI coding-assistant support (Claude, via an
MCP filesystem connection, working directly in this project's folder).
Used for: porting the original GPU-specific code to run on CPU,
scaffolding the repo structure and documentation, drafting the
visualization and lab-theming modules, porting `agent/llm_supervisor.py`
and `hardware/qcodes_adapter.py` from the original hackathon repo, replacing
the schematic potential-well visualization with one based on QArray's real
`free_energy` output, and building the Live Console
(`console/live_session.py`, `app.py`'s Live Console tab). All design
decisions, physics/engineering content, and final review remain the
author's responsibility, per WISER's disclosure requirement -- expand
this section with specifics as the build progresses.
