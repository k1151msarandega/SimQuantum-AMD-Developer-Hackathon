"""Solara app: the interactive 'virtual lab' dashboard.

Solara port of app.py (Streamlit). See docs/PORTING_NOTES.md for the
Streamlit -> Solara migration notes.

app.py (Streamlit) is left untouched and fully working -- this is a
separate, additive file, not a replacement. Run `streamlit run app.py`
for the original, or `solara run solara_app.py` for this port. Keep
both around; treat Streamlit as the known-good fallback until this has
been exercised against the real hardware/twin stack.

TWO TABS, same as the Streamlit version:

  "Live Console" -- the actual deliverable. A continuously-running
  instrument session (console/live_session.py), not a request/response
  "pick settings, click Run, wait, see a final chart" flow. Start begins
  a background thread; Vx/Vy are real, settable QCoDeS Parameters
  (hardware/qcodes_adapter.py) writable at any point while it's live;
  the device feed, the real free-energy potential well
  (viz/potential_well.py), and the triage/staleness/drift diagnostics all
  update continuously while the session runs; Stop ends it.

  Streamlit's st.fragment(run_every="0.5s") polled session.snapshot() on
  a timer. Solara's equivalent is a background polling thread
  (solara.use_thread) that copies session.snapshot() into a
  solara.reactive every ~0.5s -- writing a reactive from a background
  thread is exactly what triggers Solara's UI to re-render, so this is
  the natural translation, not a workaround.

  "Mode comparison" -- the original WISER26 MVP flow, preserved
  unchanged in spirit: pick a mode (serial/batched/batched_triage[_llm])
  and a config, click Run, watch it complete. pipeline.run_live() is a
  blocking generator, same as before; here it's driven from a background
  thread (solara.use_thread) instead of a plain for-loop in the Streamlit
  script body, since Solara has no direct equivalent of Streamlit's
  block-until-done script rerun -- the UI needs to stay responsive (e.g.
  to a future Cancel button) while the generator runs.

Run with: solara run solara_app.py
(Streamlit's `streamlit run app.py` becomes `solara run solara_app.py`.)

NOTE on state scope: the reactive variables below are declared at module
level, so (matching Streamlit's single-session assumption for this
teaching app -- there is only one real QArrayTwinInstrument "instrument"
at a time) they are shared process-wide. If this app is ever served to
multiple simultaneous users/tabs from one `solara run` process, wrap this
state in a per-session container instead (e.g. solara.use_memo(lambda:
SessionState(), []) passed down as a prop) -- see Solara's docs on
per-user vs. global reactive state.
"""
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import solara
import solara.lab

from qdot_edu.console.live_session import LiveConsoleSession, LiveSnapshot
from qdot_edu.pipeline import run_live
from qdot_edu.stream.trajectory import load_trajectory_config
from qdot_edu.viz.potential_well import render_plotly_figure

CONFIG_OPTIONS = {
    "Quick (300 frames, ~seconds)": "configs/trajectory_quick.yaml",
    "CPU teaching demo": "configs/trajectory_cpu_edu.yaml",
    "Full (2000 frames, ~45-50s)": "configs/trajectory.yaml",
}
CONFIG_LABELS = list(CONFIG_OPTIONS.keys())

TIER_COLORS = {"FULL": "#3ddc84", "CHEAP": "#e0b152", "SKIP": "#e05d5d"}

# lab_theme.py's CSS/status-badge helpers were Streamlit-markup-specific
# (unsafe_allow_html strings); reproduced here as plain CSS + a small
# Solara component instead of reusing lab_theme.py directly.
LAB_CSS = """
.lab-app-bg { background-color: #0b0f14; }
.lab-readout { font-family: 'Courier New', monospace; }
"""


@solara.component
def StatusBadge(label: str, ok: bool):
    color = "#3ddc84" if ok else "#e05d5d"
    solara.HTML(tag="span", unsafe_innerHTML=label, classes=["lab-readout"], style=f"color:{color}")


# =============================================================================
# Module-level reactive state (see NOTE on state scope in the module docstring)
# =============================================================================
console_config_label = solara.reactive(CONFIG_LABELS[0])
llm_on = solara.reactive(True)

live_session: solara.Reactive[Optional[LiveConsoleSession]] = solara.reactive(None)
session_generation = solara.reactive(0)  # bumped on Start/Stop to restart the poller thread

vx_on = solara.reactive(False)
vx_val = solara.reactive(0.0)
vy_on = solara.reactive(False)
vy_val = solara.reactive(0.0)

compare_mode = solara.reactive("batched_triage")
compare_config_label = solara.reactive(CONFIG_LABELS[0])
compare_run_generation = solara.reactive(0)
compare_running = solara.reactive(False)


# =============================================================================
# Live Console
# =============================================================================
@solara.component
def LiveConsoleSidebar():
    session = live_session.value
    is_running = session is not None and session.is_running()
    is_paused = is_running and session.snapshot().paused

    solara.Markdown("### Live Console")

    solara.Select(
        label="Trajectory config", value=console_config_label, values=CONFIG_LABELS,
        disabled=is_running,
    )

    fireworks_ready = bool(os.environ.get("FIREWORKS_API_KEY"))
    solara.Checkbox(
        label="LLM threshold supervisor", value=llm_on, disabled=is_running,
    )
    solara.Markdown(
        "Background thread that tunes the triage thresholds from recent backlog trends "
        "(agent/llm_supervisor.py). Requires FIREWORKS_API_KEY. The rest of the console "
        "works fully offline with this off.",
        style="font-size: 0.8em; opacity: 0.8;",
    )
    if llm_on.value and not fireworks_ready:
        solara.Warning(
            "FIREWORKS_API_KEY is not set. The console will still run fully -- the supervisor "
            "thread just logs an error each tick and leaves thresholds at their defaults instead "
            "of tuning them."
        )

    def start_clicked():
        new_session = LiveConsoleSession(
            CONFIG_OPTIONS[console_config_label.value], llm_supervised=llm_on.value, device="cpu",
        )
        new_session.start()
        live_session.value = new_session
        session_generation.value += 1

    def stop_clicked():
        if session is not None:
            session.stop()
        session_generation.value += 1

    def pause_clicked():
        if session is not None:
            session.pause()
        session_generation.value += 1

    def resume_clicked():
        if session is not None:
            session.resume()
        session_generation.value += 1

    with solara.Row():
        solara.Button("\u25B6 Start", on_click=start_clicked, disabled=is_running, color="primary")
        solara.Button("\u25A0 Stop", on_click=stop_clicked, disabled=not is_running)
    with solara.Row():
        solara.Button("\u23F8 Pause", on_click=pause_clicked, disabled=not is_running or is_paused)
        solara.Button("\u25B6 Resume", on_click=resume_clicked, disabled=not is_running or not is_paused)

    solara.Markdown("---")
    solara.Markdown("**Instrument panel -- Vx / Vy**")
    solara.Markdown(
        "Real QCoDeS Parameters (hardware/qcodes_adapter.py). Grab manual control at any point "
        "while the console is live, or leave an axis on autopilot to follow the scripted trajectory.",
        style="font-size: 0.8em; opacity: 0.8;",
    )
    solara.Checkbox(label="Manual Vx", value=vx_on, disabled=session is None)
    solara.InputFloat("Vx (V)", value=vx_val, disabled=session is None or not vx_on.value)
    solara.Checkbox(label="Manual Vy", value=vy_on, disabled=session is None)
    solara.InputFloat("Vy (V)", value=vy_val, disabled=session is None or not vy_on.value)

    def apply_clicked():
        if session is None:
            return
        if vx_on.value:
            session.set_vx(vx_val.value)
        else:
            session.clear_vx()
        if vy_on.value:
            session.set_vy(vy_val.value)
        else:
            session.clear_vy()
        session_generation.value += 1

    def release_clicked():
        if session is None:
            return
        session.clear_vx()
        session.clear_vy()
        session_generation.value += 1

    solara.Button("Apply", on_click=apply_clicked, disabled=session is None)
    solara.Button("Release both to autopilot", on_click=release_clicked, disabled=session is None)


@solara.component
def LiveConsoleBody():
    session = live_session.value

    # Per-instance reactive snapshot, refreshed by a background poller
    # thread -- the direct translation of Streamlit's
    # st.fragment(run_every="0.5s"). See module docstring.
    snapshot: solara.Reactive[LiveSnapshot] = solara.use_reactive(LiveSnapshot())

    def poll():
        # Restarted (old thread cancelled) whenever session_generation
        # changes, i.e. on every Start/Stop/Pause/Resume/Apply -- so a
        # control action is reflected immediately via the fresh
        # snapshot() call below, not just on the next 0.5s tick.
        current = live_session.value
        if current is None:
            return
        while True:
            snapshot.value = current.snapshot()
            time.sleep(0.5)

    solara.use_thread(poll, dependencies=[session_generation.value])

    if session is None:
        solara.Info("Choose a config and click **Start** in the sidebar to power on the console.")
        return

    snap = snapshot.value

    # ---- status strip -------------------------------------------------
    with solara.Row(justify="space-between"):
        StatusBadge(f"STATUS: {snap.status.upper()}", ok=snap.running or snap.status == "trajectory complete")
        solara.Text(f"Frame {max(snap.frame_index, 0)}/{snap.n_frames_total}")
        solara.Text(f"Elapsed {snap.elapsed_s:.1f}s")
        solara.Text(f"Vx/Vy {snap.vx:.3f} / {snap.vy:.3f} V" if not np.isnan(snap.vx) else "Vx/Vy --")

    if snap.vx_override_active or snap.vy_override_active:
        which = " + ".join(
            n for n, active in (("Vx", snap.vx_override_active), ("Vy", snap.vy_override_active)) if active
        )
        solara.Markdown(f"**Manual control active:** {which} driven from the sidebar, not the scripted trajectory.")
    if snap.error:
        solara.Error(f"Session error: {snap.error}")
    if snap.paused:
        solara.Info("Paused -- frame consumption is halted; nothing is lost. Click **Resume** to continue.")
    if snap.status == "trajectory complete":
        solara.Success("Trajectory complete -- the scripted run finished. Click Start for a fresh session.")

    # ---- device feed vs. potential well --------------------------------
    with solara.Columns([1, 1]):
        with solara.Column():
            solara.Markdown("**Live device feed -- charge-stability diagram**")
            if snap.frame is not None:
                frame = np.nan_to_num(np.asarray(snap.frame, dtype=float))
                zmin, zmax = float(frame.min()), float(frame.max())
                if zmin == zmax:
                    zmin, zmax = zmin - 0.5, zmax + 0.5
                heat = go.Figure(data=go.Heatmap(
                    z=frame, colorscale="Viridis", zmin=zmin, zmax=zmax,
                    colorbar=dict(title="charge state (a.u.)"),
                ))
                heat.update_layout(
                    template="plotly_white", height=380, margin=dict(l=40, r=20, t=20, b=40),
                    xaxis_title="gate sweep index (x)", yaxis_title="gate sweep index (y)",
                )
                solara.FigurePlotly(heat)
            else:
                solara.Text("Waiting for the first frame...")

        with solara.Column():
            solara.Markdown("**Potential well -- real QArray free_energy**")
            if snap.fe_z is not None:
                surf = go.Figure(data=[go.Surface(
                    x=snap.fe_x, y=snap.fe_y, z=snap.fe_z, colorscale="Viridis_r",
                    colorbar=dict(title="free energy (a.u.)"),
                )])
                surf.update_layout(
                    scene=dict(xaxis_title="Vx (V)", yaxis_title="Vy (V)", zaxis_title="F(n, Vg)"),
                    height=380, margin=dict(l=0, r=0, t=20, b=0),
                )
                solara.FigurePlotly(surf)
                solara.Markdown("Deep well = current charge configuration is stable here. See viz/potential_well.py.")
            else:
                solara.Text("Waiting for the first flush...")

    # ---- twin health / diagnostics -------------------------------------
    solara.Markdown("**Twin health / diagnostics**")
    with solara.Columns([1, 1, 1, 2]):
        with solara.Column():
            tier_color = TIER_COLORS.get(snap.tier, "#8a8f98")
            solara.HTML(tag="span", unsafe_innerHTML=f"<b>Current tier:</b> {snap.tier or '--'}",
                        style=f"color:{tier_color}")
            solara.Text(f"queue depth: {snap.queue_depth}  (max seen: {snap.max_queue_depth})")
        with solara.Column():
            StatusBadge("DRIFT: ACTIVE" if snap.drift_active else "DRIFT: nominal", ok=not snap.drift_active)
            solara.Text(f"last lag: {snap.last_lag_s * 1000:.1f} ms")
        with solara.Column():
            tc = snap.tier_counts
            solara.Text(f"FULL {tc.get('FULL', 0)} \u00b7 CHEAP {tc.get('CHEAP', 0)} \u00b7 SKIP {tc.get('SKIP', 0)}")
            cheap_q, skip_q, stale_s = snap.thresholds
            solara.Text(f"thresholds: cheap_q={cheap_q} skip_q={skip_q} stale_s={stale_s:.3f}")
        with solara.Column():
            if snap.llm_supervised and snap.supervisor_events:
                ev = snap.supervisor_events[-1]
                solara.Markdown(f"**Supervisor:** {ev.reasoning}")
            elif snap.llm_supervised:
                solara.Text("Supervisor: no threshold update yet.")

    if snap.lag_history:
        df = pd.DataFrame(snap.lag_history, columns=["frame_index", "wall_clock_lag", "tier"])
        lag_fig = go.Figure(data=go.Scatter(x=df["frame_index"], y=df["wall_clock_lag"], mode="lines"))
        lag_fig.update_layout(height=220, margin=dict(l=40, r=20, t=10, b=30),
                               xaxis_title="frame index", yaxis_title="wall-clock lag (s)")
        solara.FigurePlotly(lag_fig)
    else:
        solara.Text("Staleness chart appears after the first micro-batch flush.")

    if snap.llm_supervised and snap.supervisor_events:
        with solara.Details(summary=f"Supervisor log ({len(snap.supervisor_events)} updates)"):
            t0 = snap.supervisor_events[0].t
            collapsed = []  # list of [first_ev, last_ev, count]
            for ev in snap.supervisor_events[-200:]:
                if collapsed and collapsed[-1][0].reasoning == ev.reasoning:
                    collapsed[-1][1] = ev
                    collapsed[-1][2] += 1
                else:
                    collapsed.append([ev, ev, 1])

            for first_ev, last_ev, count in collapsed[-20:]:
                span = (
                    f"t={first_ev.t - t0:.2f}s"
                    if count == 1
                    else f"t={first_ev.t - t0:.2f}s\u2013{last_ev.t - t0:.2f}s ({count}\u00d7)"
                )
                ev = last_ev
                solara.Text(
                    f"{span} -- {ev.reasoning}  "
                    f"(cheap {ev.old[0]}\u2192{ev.new[0]}, skip {ev.old[1]}\u2192{ev.new[1]}, "
                    f"stale {ev.old[2]:.3f}\u2192{ev.new[2]:.3f})"
                )


# =============================================================================
# Mode comparison
# =============================================================================
@solara.component
def ModeComparisonTab():
    result_state = solara.use_reactive({"df": None, "tier_counts": None, "vx": 0.0, "vy": 0.0, "done": False})

    def do_run():
        # use_thread (unlike use_task) always needs a real dependency list
        # -- dependencies=None doesn't mean "don't run" here. Gate on
        # compare_run_generation instead: 0 means "Run has never been
        # clicked", so do nothing on the initial mount.
        if compare_run_generation.value == 0:
            return
        array_size = load_trajectory_config(CONFIG_OPTIONS[compare_config_label.value]).array_size
        rows, cols = array_size
        result_state.value = {"df": pd.DataFrame(), "tier_counts": None, "vx": 0.0, "vy": 0.0, "done": False,
                               "rows": rows, "cols": cols}
        for update in run_live(compare_mode.value, CONFIG_OPTIONS[compare_config_label.value],
                                yield_every=5, device="cpu"):
            result_state.value = {**update, "rows": rows, "cols": cols}
            if update["done"]:
                break
        compare_running.value = False

    solara.use_thread(do_run, dependencies=[compare_run_generation.value])

    solara.Markdown(
        "Run a mode to completion and watch it stream in -- useful for the side-by-side "
        "serial vs. batched vs. triage comparison the notebooks build up to."
    )
    with solara.Columns([1, 1, 1]):
        solara.Select(
            label="Pipeline mode", value=compare_mode,
            values=["serial", "batched", "batched_triage", "batched_triage_llm"],
        )
        solara.Select(label="Trajectory config", value=compare_config_label, values=CONFIG_LABELS)

        def run_clicked():
            compare_run_generation.value += 1
            compare_running.value = True

        solara.Button("Run", on_click=run_clicked, color="primary", disabled=compare_running.value)

    result = result_state.value
    with solara.Columns([1, 1]):
        with solara.Column():
            solara.Markdown("### Staleness over time")
            df = result.get("df")
            if df is not None and len(df) > 0:
                fig = go.Figure(data=go.Scatter(x=df["frame_index"], y=df["wall_clock_lag"], mode="lines"))
                fig.update_layout(height=320, margin=dict(l=40, r=20, t=10, b=30),
                                   xaxis_title="frame index", yaxis_title="wall-clock lag (s)")
                solara.FigurePlotly(fig)
            if result.get("tier_counts") is not None:
                StatusBadge(f"tiers so far: {result['tier_counts']}", ok=True)
            if result.get("done") and df is not None:
                solara.Success(f"Run complete -- {len(df)} frames processed.")
        with solara.Column():
            solara.Markdown("### Potential well -- real QArray free_energy")
            if result.get("df") is not None:
                rows, cols = result.get("rows", 1), result.get("cols", 2)
                well_fig = render_plotly_figure(result.get("vx", 0.0), result.get("vy", 0.0), rows, cols)
                solara.FigurePlotly(well_fig)

    if result.get("df") is None:
        solara.Info("Choose a mode and config above, then click Run.")


# =============================================================================
# Page
# =============================================================================
@solara.component
def Page():
    solara.Style(LAB_CSS)
    solara.Title("qdot-edu: live twin")

    with solara.Column(classes=["lab-app-bg"]) as main:
        solara.Markdown("# Quantum Dot Digital Twin -- Live Console")
        solara.Markdown(
            "A CPU-only teaching build. See README.md for the full write-up and "
            "docs/lessons/ for the guided notebook arc this app complements."
        )

        with solara.Sidebar():
            LiveConsoleSidebar()

        with solara.lab.Tabs():
            with solara.lab.Tab("Live Console"):
                LiveConsoleBody()
            with solara.lab.Tab("Mode comparison"):
                ModeComparisonTab()

    return main
