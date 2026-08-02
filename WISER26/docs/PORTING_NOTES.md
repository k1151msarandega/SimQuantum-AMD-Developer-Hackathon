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
3. Verify `tier_counts` actually varies (triage doing something, not
   always FULL or always SKIP) once real CPU timing is in place.
4. `viz/potential_well.py`'s TODO: check whether the installed QArray
   version exposes a real potential-query API to replace the Gaussian-well
   interpolation.
5. Reproducibility: this repo is run from Colab notebooks, not local pip
   installs (confirmed with the author) -- update README.md's install/run
   instructions to match that actual workflow rather than a generic
   `pip install -e .` + local `streamlit run`.
6. Untested end to end: none of this code has actually been executed
   (only statically reviewed) -- QArray's exact `do2d_open`/`DotArray` API
   shape, and whether un-swept gates default to 0V during `do2d_open`
   (see `model_params.py`'s verification caveat), are unverified against
   a real installed QArray version.

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
