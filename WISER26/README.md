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
### Before you start
Run this from [`notebooks/00_launch_app.ipynb`](notebooks/00_launch_app.ipynb) -- see "Install & run" above. That notebook installs everything, smoke-tests the pipeline with no UI, then launches the app through a tunnel and prints a URL to click.

### Live Console tab
This is the actual deliverable: a continuously-running instrument session, not a "pick settings, click Run, wait" flow.

1. **Pick a config** in the sidebar (Quick/CPU teaching demo/Full) and click **Start**. A background session begins immediately -- frames start streaming from QArray, and the device feed and potential well begin updating live.
2. **Watch it run on autopilot** first. The frame counter, Vx/Vy readout, and both panels update continuously without you touching anything -- this is the scripted trajectory driving the device.
3. **Take manual control**: check "Manual Vx" and/or "Manual Vy" in the sidebar, type a voltage, click **Apply**. The device immediately starts responding to your value instead of the script -- you'll see the device feed and potential well shift accordingly within a second or two. Click "Release both to autopilot" to hand control back to the script.
4. **Pause/Resume**: Pause freezes frame consumption without losing any state (staleness log, tier tally, thresholds all stay exactly where they were) -- useful for stopping to look at a specific moment. Resume picks up exactly where it left off.
5. **Read the diagnostics** under "Twin health / diagnostics": current tier (FULL/CHEAP/SKIP -- see Learning objective #5), a drift flag, the running tier tally, and a staleness-over-time chart. If the LLM supervisor is on, its most recent reasoning and a collapsible log appear here too.
6. **Stop** ends the session for real -- a subsequent Start begins a fresh trajectory from frame 0, not a resume.

### Mode comparison tab
The original, simpler flow: pick a pipeline mode (serial/batched/batched_triage/batched_triage_llm) and a config, click **Run**, and watch it run to completion. This is the more direct way to see the serial-vs-batched-vs-triage story side by side (see Learning objectives #3-4) without the Live Console's extra controls in the way.

### What to expect (read this before assuming something's broken)
- **Tiers may stay at FULL the whole run.** This is a known, documented tuning gap (see [`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md)'s "Open items") -- the triage thresholds were tuned against the original GPU-batched timing, not this CPU port's, and haven't been re-measured yet. It is not a bug in the triage logic itself.
- **The potential well can look nearly flat** at some voltages and clearly curved (a real saddle shape) at others -- this is real physics (see `viz/potential_well.py`'s module docstring for the numeric reasoning), not inconsistent rendering.
- **Occasional `Failed to fetch dynamically imported module` errors** are a known `localtunnel`-through-Colab fragility, not an app bug -- a hard refresh usually clears it. See "Future improvements" below for the plan to remove this dependency entirely.

## Future improvements / scalability

### Near-term (already scoped, not yet done)
- **Re-tune triage against real CPU timing.** `STALE_THRESHOLD_S` and the queue-depth thresholds were carried over from the original GPU-batched run's measured timing; a real CPU benchmark is needed before the triage tier ever visibly leaves FULL on the CPU-tuned configs (see `docs/PORTING_NOTES.md`'s "Open items").
- **Framework migration to [Solara](https://solara.dev/)**, in progress as of this submission. Streamlit's request/response model needed a polling workaround (`st.fragment(run_every=...)`) to drive the Live Console's continuous updates; Solara's reactive model lets a background thread push state directly, which is a more natural fit and removes a source of UI fragility observed during development (occasional failed asset loads under Streamlit + a Colab `localtunnel` connection specifically). Retaining the Streamlit version as a tested fallback until the Solara port has run end-to-end against the real dependency stack.
- **Hosting off Colab+tunnel entirely** -- Streamlit Community Cloud or a Hugging Face Space, both free, would remove the tunnel-fragility class of issue outright and give a stable public URL instead of a per-session tunnel link. Both are viable today (a `requirements.txt` is already included for this); not yet done only for lack of time.

### Educational content
- **Notebooks 01-07**: the guided arc `docs/lessons/README.md` lays out, walking through each learning objective in order using the same running example the app demonstrates live.
- **A "build-your-own-triage-policy" capstone notebook**, where a learner writes and tests their own multi-signal rule against the same staleness/drift signals the built-in agent uses -- directly exercises learning objective #5 (design and critique a triage policy) as a hands-on exercise rather than only an observation.
- **Automated assessments** (WISER's optional advanced task): short adaptive checks after each notebook -- e.g. "given this queue-depth/staleness trace, what tier should trigger next" -- auto-graded against the real `agent/triage.py` logic rather than a hand-written answer key, so the check stays correct even if thresholds are re-tuned later.
- **Multilingual content** (also an optional advanced task): the notebook narrative text is the only part that would need translation -- all code, config, and the app itself are language-independent already.

### Scalability
- **Array shapes beyond a 2-dot line.** `array_size` (rows x cols) is already a live, working config value via `model_params.py` -- extending the guided exercises to larger arrays would let learners see how triage dynamics and drift detection change as the system gets more complex, without any new simulation code.
- **Adoption by educators with minimal setup** (WISER's optional advanced task): the whole stack is CPU-only and open-source with no GPU or paid-API dependency required for the core lesson (the LLM supervisor is opt-in and degrades gracefully without a key). A Binder or `nbgitpuller` launch link, once the notebook arc is complete, would let an instructor point students at a zero-install link instead of a local setup.
- **Community contribution path**: `docs/PORTING_NOTES.md`'s practice of documenting every real bug found and its fix, not just claiming things work, is meant to double as a model for contributors -- the same discipline this project asks learners to practice in objective #8.

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
