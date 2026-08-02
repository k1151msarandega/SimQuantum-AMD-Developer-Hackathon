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
This README and the module-level scaffolding are in place; several pieces
are explicitly marked TODO/not-yet-built in the code and in
[`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md) and
[`docs/lessons/README.md`](docs/lessons/README.md) -- most notably, the
actual `notebooks/` lesson content and `app.py` Streamlit app don't exist
yet. Treat this README as accurate about *intent and architecture*, not
yet as a claim that the full learner experience is built and tested.

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
drift). A 3D potential-well visualization gives spatial intuition for
what the abstract charge-stability numbers mean physically. Each stage is
a working notebook that builds on the last, mirroring how the underlying
engineering was actually developed -- see
[`docs/lessons/README.md`](docs/lessons/README.md) for the full arc, and
[`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md) for the engineering
history this was built from.

## Tech stack
- **[QArray](https://github.com/b-vanstraaten/qarray)** -- constant-capacitance quantum dot array simulator (Rust backend, CPU)
- **PyTorch** (CPU) -- the ensemble "perception" model that stands in for a real state-estimation workload
- **NumPy / pandas** -- signal processing, staleness logging
- **Streamlit** -- the interactive app (`app.py`, not yet built -- see Status)
- **Plotly / Matplotlib** -- 2D charge-stability plots and the 3D potential-well surface
- **Jupyter notebooks** -- the guided lesson arc (`notebooks/`, not yet built -- see Status)

No GPU required anywhere in this repo -- see
[`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md) for what that took.

## Install & run
Developed and run from Google Colab, not a local install -- there is no
tested local `pip`/venv workflow for this repo (see
[`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md)).

**Notebooks (`notebooks/00`-`07`, once built -- see Status):** open
directly in Colab (upload this repo or mount/clone it into a Colab
session), then in the first cell:
```python
!pip install -e .
```
and run cells top to bottom.

**The Streamlit app (`app.py`, once built -- see Status):** Colab can't
serve Streamlit directly, so run it and expose it through a tunnel from a
Colab cell, e.g.:
```python
!pip install -e . streamlit
!streamlit run app.py &>/content/logs.txt &
!npx --yes localtunnel --port 8501
```
then open the printed `localtunnel` URL -- that's the "hosting link" to
share/open. (Any equivalent tunnel, e.g. `pyngrok`, works too -- this is
just one concrete option pending an actual test run.)

If `pip install torch` pulls in an unwanted CUDA build in your Colab
runtime, install the CPU-only wheel explicitly first:
```python
!pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## User guide
(To be expanded once `app.py` and `notebooks/` exist -- see
[`docs/lessons/README.md`](docs/lessons/README.md) for the planned
notebook-by-notebook walkthrough in the meantime.) The intended flow: work
through `notebooks/00`-`07` in order, then explore the same ideas
interactively in the Streamlit app, which lets you switch between serial/
batched/triage modes and tune triage thresholds live while watching
staleness and the potential-well visualization respond.

## Future improvements / scalability
See [`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md)'s "Open items" for
the concrete near-term TODOs (rate/threshold re-tuning, deciding the fate
of the LLM-supervisor mode, and a real end-to-end test run since none of
this has been executed yet -- only statically reviewed). Longer-term:
add a real QArray potential-query API to `viz/potential_well.py` if one
exists, replacing the current schematic interpolation; explore array
shapes beyond a 2-dot line now that `array_size` is live (see
`model_params.py`) -- e.g. a guided exercise on how triage dynamics
change with array size; consider a "build-your-own-triage-policy"
exercise as a capstone notebook.

## Credits & licensing
See [`assets/CREDITS.md`](assets/CREDITS.md) for attribution on all
reused imagery/renders. Original engineering (`qdot-live-twin`, Act II
hackathon) rebuilt and extended per [`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md).

## AI-assistance disclosure
This repo was developed with AI coding-assistant support (Claude, via an
MCP filesystem connection, working directly in this project's folder).
Used for: porting the original GPU-specific code to run on CPU, scaffolding
the repo structure and documentation, and drafting the new visualization
and lab-theming modules. All design decisions, physics/engineering
content, and final review remain the author's responsibility, per WISER's
disclosure requirement -- expand this section with specifics as the build
progresses.
