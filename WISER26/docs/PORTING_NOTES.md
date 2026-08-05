# Porting notes: qdot-live-twin (Act II hackathon) -> qdot_edu (WISER26)

What changed, what didn't, and what's still open. Kept up to date as the
port progresses -- if you change something not reflected here, add a line.

## Unchanged (ported as-is, only `qdot_twin` -> `qdot_edu` import paths updated)
These were already CPU-native in the original repo; nothing about them
depended on the AMD GPU:
- `twin/staleness.py`, `twin/serial_estimator.py`
- `agent/triage.py`, `agent/thresholds.py`
- `perception/ensemble.py`, `perception/ood.py` -- already defaulted to device="cpu"

## Actually changed for the CPU port
- `twin/batch_estimator.py`: original hardcoded `device="cuda"` everywhere.
  Now defaults to `device="cpu"`, parameterized throughout.
- `pipeline.py`: threaded a `device` parameter (default `DEFAULT_DEVICE =
  "cpu"`) through `run()`, `run_detailed()`, `run_live()`, and every
  `estimate_batch()` call site, replacing the original's hardcoded `"cuda"`.
  Also (see "Resolved" below): `run_live()` now yields real per-frame
  `vx`/`vy`.
- `stream/trajectory.py`: `array_size` is now actually read into
  `TrajectoryConfig` (see "Resolved" below -- it used to be parsed by
  nothing and silently ignored).
- `stream/generator.py`: model is now built dynamically per-run from
  `array_size` via the new `model_params` module, instead of hardcoded
  `Cdd`/`Cgd` (see "Resolved" below). QArray's Rust backend itself is
  still CPU-only either way -- no change there.

## New for WISER26 (no equivalent in the original repo)
- `model_params.py` -- single source of truth for dot-grid layout,
  `Cdd`/`Cgd`, and gate names, derived from `array_size`. Used by both
  `stream/generator.py` and `viz/potential_well.py`.
- `viz/potential_well.py` -- schematic 3D confinement-potential surface,
  derived from the same `model_params` matrices `stream/generator.py`
  uses. See its module docstring: this is a pedagogical interpolation,
  NOT a Poisson-equation solve. Must be labeled as such in learner-facing
  material.
- `viz/lab_theme.py` -- Streamlit CSS/theme helpers for the lab-console framing.

## Not yet ported -- will break if used
- `agent/llm_supervisor.py` does not exist in this repo yet. `pipeline.py`'s
  `"batched_triage_llm"` mode still references it (matching the original's
  structure) and will raise `ImportError` if selected. Decide whether to
  port it, simplify it, or drop that mode entirely from the submission --
  it also adds an external API dependency (Fireworks) that cuts against
  "easy to reproduce on a laptop."
- No `hardware/qcodes_adapter.py` equivalent checked/ported yet.

## Open items that need real benchmarking, not guessing
1. Retune `configs/trajectory.yaml`'s `stream_rate_hz.end` (2000) and/or
   `configs/trajectory_cpu_edu.yaml`'s placeholder `end: 60` against
   measured FULL-tier throughput of `perception/ensemble.py` on a target
   laptop CPU.
2. Re-ground `agent/triage.py`'s `STALE_THRESHOLD_S` (0.05s) -- came from
   the original GPU-batched run's measured worst-case lag; re-measure
   against the CPU `batch_estimator`.
3. `tier_counts` confirmed NOT varying on a real run: `batched_triage`
   mode against `configs/trajectory_cpu_edu.yaml` returned
   `{'FULL': 215, 'CHEAP': 0, 'SKIP': 0}` -- triage never once chose
   anything but FULL. This was flagged as a risk before any code ran;
   now confirmed with real data. Likely cause: `agent/triage.py`'s
   `STALE_THRESHOLD_S`/queue-depth thresholds were tuned against the
   original GPU-batched run's timing (see item 2) and may simply never
   trigger under `trajectory_cpu_edu.yaml`'s current (still-placeholder,
   see that file's own TODO) rate. Needs real re-tuning, not a guess.
4. `viz/potential_well.py`'s TODO: check whether the installed QArray
   version exposes a real potential-query API to replace the Gaussian-well
   interpolation.
5. Untested end to end beyond a partial first run: `notebooks/00_launch_app.ipynb`
   made it all the way to a real Streamlit app launch after the install
   fixes above -- QArray's actual API (`do2d_open`, `DotArray`, gate
   names) works as `stream/generator.py` assumed, which was the biggest
   open unknown. Not yet confirmed: whether the potential-well panel
   renders correctly once the tunnel-timing fix (see "Resolved") lands,
   since the Plotly chart failed to load on the first run before that fix.

## Resolved
- ~~De-duplicate capacitance matrices~~ -- done. `model_params.py` is now
  the single shared source for `Cdd`/`Cgd`/dot positions/gate names, used
  by both `stream/generator.py` and `viz/potential_well.py`.
- ~~`array_size` in the YAML configs was parsed by nothing~~ -- real bug,
  caught during review, now fixed: `trajectory.py` reads it, and
  `generator.py`/`potential_well.py` both build their models from it via
  `model_params.py`. Also corrected the configs' actual values: the
  original repo's `array_size: [2, 2]` never matched the hardcoded 2-dot
  matrices that were *actually* simulated -- now set to `[1, 2]`, which
  `model_params.dot_grid_matrices(1, 2)` reproduces exactly (verified in
  that function's docstring). array_size is now a real, live control.
- ~~`app.py`'s potential-well panel was a static placeholder~~ -- fixed.
  `pipeline.run_live()`'s yielded dicts now carry real per-frame `vx`/`vy`
  (both partial updates and the final dict), and `app.py` re-renders the
  potential-well figure on every yield using those real values plus the
  selected config's `array_size`.
- ~~Colab install: separate torch + project install broke numpy/scipy~~ --
  real bug, caught on the first actual Colab run (not a review finding).
  `notebooks/00_launch_app.ipynb`'s Step 1 used to run
  `pip install torch --index-url .../whl/cpu` followed by a separate
  `pip install -e .`. Colab's preinstalled `numpy 2.2.4`/`scipy 1.16.3`
  already satisfied `qarray`'s exact `numpy==2.2.4`/`scipy>=1.11` pins --
  but the first pip call downgraded numpy to `1.26.4` for torch's sake and
  uninstalled scipy as a side effect; the second call bumped numpy back up
  for qarray but never restored scipy, leaving imports broken. Fixed by
  combining both installs into one `pip install -e .` call (see the
  notebook and README.md's Install & run section) so pip resolves the
  whole dependency graph in one pass, plus an explicit "restart the Colab
  runtime before importing" step (Colab caches already-loaded numpy/scipy
  in memory across a pip-level swap). Not yet re-verified end to end after
  this fix -- see "Open items" #5.
- ~~Even a clean install left `_center` ImportError from numpy/scipy~~ --
  a second real bug: the numpy<->scipy churn from the fix above left
  genuinely corrupted compiled files on disk (mixed-version artifacts pip
  doesn't always fully clean up), which a runtime restart alone can't fix
  since restart only clears in-memory state, not broken files on disk.
  Fixed with a proper repair cell in the notebook: uninstall numpy/scipy
  first (not just `--force-reinstall`, which can still reuse the same bad
  cached wheel), then reinstall with `--no-cache-dir` to force a genuinely
  fresh download.
- ~~`%cd` from Step 0 silently lost after every runtime restart~~ -- third
  real bug from the same first run: Colab resets cwd to `/content` on
  restart, breaking every relative path in Steps 2-5 (root cause of why
  Step 5's Streamlit app wasn't opening -- `streamlit run app.py` was
  running from the wrong directory and never actually starting a server).
  Fixed by persisting the resolved path with `%store` in Step 0 and adding
  a "restore working directory" cell to run after every restart. Also
  folded in the real GitHub URL for this repo now that it's been pushed:
  `k1151msarandega/SimQuantum-AMD-Developer-Hackathon`.
- ~~Blank page + `Failed to fetch dynamically imported module` for chart
  widgets on first load~~ -- fourth real bug, and the good news attached
  to it: it surfaced only AFTER the pipeline itself had already run
  correctly end to end (`app.py` printed real `tier_counts`), so this was
  purely a frontend-loading issue, not a pipeline bug. Root cause:
  `notebooks/00_launch_app.ipynb`'s Step 5 used a fixed `sleep 3` before
  starting the `localtunnel` tunnel, which wasn't reliably long enough for
  Streamlit to finish starting -- the tunnel began routing traffic before
  the server could serve its heavier widget JS bundles (Vega-Lite chart,
  Plotly chart), while lighter widgets (selectboxes) eventually loaded
  after manual refreshes. Fixed by polling `localhost:8501` until it
  actually responds before starting the tunnel, instead of guessing a
  fixed delay.
