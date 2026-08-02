# Lesson map (planned notebook arc)

NOT YET BUILT -- this is the plan, ported and adapted from qdot-live-twin's
original 6-notebook progression. Each entry below becomes
`notebooks/0N_*.ipynb`, and should open with the specific learning
objective(s) from ../LEARNING_OBJECTIVES.md it targets.

| # | Notebook | Targets objective(s) | Core idea |
|---|---|---|---|
| 00 | `start_here` | -- | Orientation: what a digital twin is, why this project exists, how to run the app. |
| 01 | `sanity_check` | 2 | Drive QArray directly, plot a single charge-stability diagram, explain what you're looking at. |
| 02 | `serial_baseline` | 3, 4 | Run the serial estimator against a live-rate stream; watch it fall behind; plot staleness over time. |
| 03 | `batched` | 3, 4 | Same stream, batched estimator; compare staleness curves against 02. Discuss the CPU-vs-GPU batching gap. |
| 04 | `drift_detection` | 6 | Rolling OOD detector; inject a jump; gradual creep vs. sudden jump. |
| 05 | `triage_agent` | 5 | Wire in the triage agent; learner tunes thresholds and watches tier_counts respond live. |
| 06 | `potential_well` | 1, 7 | 3D confinement-potential visualization; explicit discussion of what it does/doesn't represent. |
| 07 | `full_demo` | all | Everything together in the lab-themed Streamlit app; wrap-up / reflection prompts. |

## Design principle for every notebook
Mirror the honest-engineering voice already present throughout
`src/qdot_edu/` (see e.g. `agent/triage.py`'s docstring): state the design
decision, the tradeoff, and why it was made that way. That voice is a
real strength of the underlying codebase and should carry into the
teaching material, not get sanded off.

Each notebook should end with 1-2 short reflection/prediction prompts
("before running the next cell, guess what happens if...") rather than
pure narration.
