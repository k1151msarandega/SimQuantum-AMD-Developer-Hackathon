"""Streamlit app: the interactive 'virtual lab' dashboard.

TWO TABS:

  "Live Console" -- the actual deliverable. A continuously-running
  instrument session (console/live_session.py), not a request/response
  "pick settings, click Run, wait, see a final chart" flow. Start begins
  a background stream; Vx/Vy are real, settable QCoDeS Parameters
  (hardware/qcodes_adapter.py) writable at any point while it's live;
  the device feed, the real free-energy potential well
  (viz/potential_well.py), and the triage/staleness/drift diagnostics all
  update continuously while the session runs; Stop ends it. Streamlit
  itself is still request/response underneath, but the session keeps
  running on its own background thread regardless -- st.fragment's
  run_every is what makes the *page* poll that live state on a timer
  instead of the learner having to manually rerun.

  "Mode comparison" -- the original WISER26 MVP flow, preserved
  unchanged: pick a mode (serial/batched/batched_triage[_llm]) and a
  config, click Run, watch it complete. Still useful for the side-by-side
  "does triage actually help" comparison the notebooks build up to (see
  docs/LEARNING_OBJECTIVES.md) -- the Live Console tab doesn't replace
  that comparison, it's a different, complementary view: one instrument
  running continuously vs. several completed runs compared side by side.

Run with: streamlit run app.py
"""
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qdot_edu.console.live_session import LiveConsoleSession
from qdot_edu.pipeline import run_live
from qdot_edu.stream.trajectory import load_trajectory_config
from qdot_edu.viz.lab_theme import inject_lab_css, status_badge
from qdot_edu.viz.potential_well import render_plotly_figure

st.set_page_config(page_title="qdot-edu: live twin", layout="wide", initial_sidebar_state="expanded")
st.markdown(inject_lab_css(), unsafe_allow_html=True)

st.title("Quantum Dot Digital Twin -- Live Console")
st.caption(
    "A CPU-only teaching build. See README.md for the full write-up and "
    "docs/lessons/ for the guided notebook arc this app complements."
)

CONFIG_OPTIONS = {
    "Quick (300 frames, ~seconds)": "configs/trajectory_quick.yaml",
    "CPU teaching demo": "configs/trajectory_cpu_edu.yaml",
    "Full (2000 frames, ~45-50s)": "configs/trajectory.yaml",
}

tab_live, tab_compare = st.tabs(["Live Console", "Mode comparison"])

# =============================================================================
# Live Console
# =============================================================================
with tab_live:
    with st.sidebar:
        st.markdown("### Live Console")

        console_config_label = st.selectbox(
            "Trajectory config", list(CONFIG_OPTIONS.keys()), index=0, key="console_config",
        )
        console_config_path = CONFIG_OPTIONS[console_config_label]

        import os
        fireworks_ready = bool(os.environ.get("FIREWORKS_API_KEY"))
        llm_on = st.checkbox(
            "LLM threshold supervisor", value=True, key="console_llm_on",
            help="Background thread that tunes the triage thresholds from recent backlog trends "
                 "(agent/llm_supervisor.py). Requires FIREWORKS_API_KEY. The rest of the console "
                 "works fully offline with this off.",
        )
        if llm_on and not fireworks_ready:
            st.warning(
                "FIREWORKS_API_KEY is not set. The console will still run fully -- the supervisor "
                "thread just logs an error each tick and leaves thresholds at their defaults instead "
                "of tuning them.", icon="\u26A0\uFE0F",
            )

        if "live_session" not in st.session_state:
            st.session_state.live_session = None
        session: LiveConsoleSession | None = st.session_state.live_session
        is_running = session is not None and session.is_running()

        c1, c2 = st.columns(2)
        start_clicked = c1.button("\u25B6 Start", type="primary", use_container_width=True, disabled=is_running)
        stop_clicked = c2.button("\u25A0 Stop", use_container_width=True, disabled=not is_running)

        if start_clicked:
            st.session_state.live_session = LiveConsoleSession(
                console_config_path, llm_supervised=llm_on, device="cpu",
            )
            st.session_state.live_session.start()
            st.rerun()
        if stop_clicked:
            session.stop()
            st.rerun()

        st.markdown("---")
        st.markdown("**Instrument panel -- Vx / Vy**")
        st.caption(
            "Real QCoDeS Parameters (hardware/qcodes_adapter.py). Grab manual control at any point "
            "while the console is live, or leave an axis on autopilot to follow the scripted trajectory."
        )
        vx_on = st.checkbox("Manual Vx", key="vx_on", disabled=session is None)
        vx_val = st.number_input("Vx (V)", value=0.0, step=0.05, key="vx_val", disabled=session is None or not vx_on)
        vy_on = st.checkbox("Manual Vy", key="vy_on", disabled=session is None)
        vy_val = st.number_input("Vy (V)", value=0.0, step=0.05, key="vy_val", disabled=session is None or not vy_on)

        apply_clicked = st.button("Apply", use_container_width=True, key="apply_vxy", disabled=session is None)
        release_clicked = st.button(
            "Release both to autopilot", use_container_width=True, key="release_vxy", disabled=session is None,
        )

        if session is not None and apply_clicked:
            if vx_on:
                session.set_vx(vx_val)
            else:
                session.clear_vx()
            if vy_on:
                session.set_vy(vy_val)
            else:
                session.clear_vy()
            st.rerun()
        if session is not None and release_clicked:
            session.clear_vx()
            session.clear_vy()
            st.rerun()

    session = st.session_state.get("live_session")

    if session is None:
        st.info("Choose a config and click **Start** in the sidebar to power on the console.")
    else:
        @st.fragment(run_every="0.5s")
        def render_live_console():
            snap = session.snapshot()

            # ---- status strip -------------------------------------------
            status_cols = st.columns([2, 1, 1, 1])
            with status_cols[0]:
                running_ok = snap.running
                st.markdown(
                    status_badge(f"STATUS: {snap.status.upper()}", ok=running_ok or snap.status == "trajectory complete"),
                    unsafe_allow_html=True,
                )
            status_cols[1].metric("Frame", f"{max(snap.frame_index, 0)}/{snap.n_frames_total}")
            status_cols[2].metric("Elapsed", f"{snap.elapsed_s:.1f}s")
            status_cols[3].metric(
                "Vx / Vy",
                f"{snap.vx:.3f} / {snap.vy:.3f} V" if not np.isnan(snap.vx) else "--",
            )
            if snap.vx_override_active or snap.vy_override_active:
                st.caption(
                    f"**Manual control active:** "
                    f"{'Vx' if snap.vx_override_active else ''}"
                    f"{' + ' if snap.vx_override_active and snap.vy_override_active else ''}"
                    f"{'Vy' if snap.vy_override_active else ''} "
                    "driven from the sidebar, not the scripted trajectory."
                )
            if snap.error:
                st.error(f"Session error: {snap.error}")
            if snap.status == "trajectory complete":
                st.success("Trajectory complete -- the scripted run finished. Click Start for a fresh session.")

            st.markdown("---")

            # ---- device feed vs. potential well ---------------------------
            feed_col, well_col = st.columns(2)
            with feed_col:
                st.markdown("**Live device feed -- charge-stability diagram**")
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
                    st.plotly_chart(heat, use_container_width=True, key="live_feed_chart")
                else:
                    st.caption("Waiting for the first frame...")

            with well_col:
                st.markdown("**Potential well -- real QArray free_energy**")
                if snap.fe_z is not None:
                    surf = go.Figure(data=[go.Surface(
                        x=snap.fe_x, y=snap.fe_y, z=snap.fe_z, colorscale="Viridis_r",
                        colorbar=dict(title="free energy (a.u.)"),
                    )])
                    surf.update_layout(
                        scene=dict(xaxis_title="Vx (V)", yaxis_title="Vy (V)", zaxis_title="F(n, Vg)"),
                        height=380, margin=dict(l=0, r=0, t=20, b=0),
                    )
                    st.plotly_chart(surf, use_container_width=True, key="live_well_chart")
                    st.caption("Deep well = current charge configuration is stable here. See viz/potential_well.py.")
                else:
                    st.caption("Waiting for the first flush...")

            st.markdown("---")

            # ---- twin health / diagnostics ---------------------------------
            st.markdown("**Twin health / diagnostics**")
            diag_cols = st.columns([1, 1, 1, 2])
            tier_color = {"FULL": "#3ddc84", "CHEAP": "#e0b152", "SKIP": "#e05d5d"}.get(snap.tier, "#8a8f98")
            with diag_cols[0]:
                st.markdown(f"**Current tier:** <span style='color:{tier_color}'>{snap.tier or '--'}</span>",
                            unsafe_allow_html=True)
                st.caption(f"queue depth: {snap.queue_depth}  (max seen: {snap.max_queue_depth})")
            with diag_cols[1]:
                st.markdown(
                    status_badge("DRIFT: ACTIVE" if snap.drift_active else "DRIFT: nominal", ok=not snap.drift_active),
                    unsafe_allow_html=True,
                )
                st.caption(f"last lag: {snap.last_lag_s * 1000:.1f} ms")
            with diag_cols[2]:
                tc = snap.tier_counts
                st.caption(f"FULL {tc.get('FULL', 0)} \u00b7 CHEAP {tc.get('CHEAP', 0)} \u00b7 SKIP {tc.get('SKIP', 0)}")
                cheap_q, skip_q, stale_s = snap.thresholds
                st.caption(f"thresholds: cheap_q={cheap_q} skip_q={skip_q} stale_s={stale_s:.3f}")
            with diag_cols[3]:
                if snap.llm_supervised and snap.supervisor_events:
                    ev = snap.supervisor_events[-1]
                    st.caption(f"**Supervisor:** {ev.reasoning}")
                elif snap.llm_supervised:
                    st.caption("Supervisor: no threshold update yet.")

            if snap.lag_history:
                df = pd.DataFrame(snap.lag_history, columns=["frame_index", "wall_clock_lag", "tier"])
                st.line_chart(df.set_index("frame_index")["wall_clock_lag"])
            else:
                st.caption("Staleness chart appears after the first micro-batch flush.")

            if snap.llm_supervised and snap.supervisor_events:
                with st.expander(f"Supervisor log ({len(snap.supervisor_events)} updates)"):
                    t0 = snap.supervisor_events[0].t
                    for ev in snap.supervisor_events[-20:]:
                        st.caption(
                            f"t={ev.t - t0:.2f}s -- {ev.reasoning}  "
                            f"(cheap {ev.old[0]}\u2192{ev.new[0]}, skip {ev.old[1]}\u2192{ev.new[1]}, "
                            f"stale {ev.old[2]:.3f}\u2192{ev.new[2]:.3f})"
                        )

        render_live_console()

# =============================================================================
# Mode comparison (original WISER26 MVP flow, preserved)
# =============================================================================
with tab_compare:
    st.caption(
        "Run a mode to completion and watch it stream in, one panel at a time -- useful for the "
        "side-by-side serial vs. batched vs. triage comparison the notebooks build up to."
    )
    cc1, cc2, cc3 = st.columns([1, 1, 1])
    with cc1:
        compare_mode = st.selectbox(
            "Pipeline mode",
            ["serial", "batched", "batched_triage", "batched_triage_llm"],
            index=2, key="compare_mode",
            help="serial = naive baseline (watch it fall behind); batched = micro-batched, no triage; "
                 "batched_triage = adaptive agent picking FULL/CHEAP/SKIP; batched_triage_llm = same, "
                 "with an LLM tuning the thresholds live.",
        )
    with cc2:
        compare_config_label = st.selectbox(
            "Trajectory config", list(CONFIG_OPTIONS.keys()), index=0, key="compare_config",
        )
        compare_config_path = CONFIG_OPTIONS[compare_config_label]
    with cc3:
        st.write("")
        st.write("")
        run_clicked = st.button("Run", type="primary", key="compare_run")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Staleness over time")
        staleness_placeholder = st.empty()
        status_placeholder = st.empty()
    with col_right:
        st.subheader("Potential well -- real QArray free_energy")
        potential_placeholder = st.empty()

    if run_clicked:
        array_size = load_trajectory_config(compare_config_path).array_size
        rows, cols = array_size

        potential_placeholder.plotly_chart(
            render_plotly_figure(0.0, 0.0, rows, cols), use_container_width=True, key="compare_well_initial",
        )

        chart_i = 0
        for update in run_live(compare_mode, compare_config_path, yield_every=5, device="cpu"):
            df = update["df"]
            if len(df) > 0:
                staleness_placeholder.line_chart(df.set_index("frame_index")["wall_clock_lag"])

            if update["tier_counts"] is not None:
                status_placeholder.markdown(
                    status_badge(f"tiers so far: {update['tier_counts']}", ok=True),
                    unsafe_allow_html=True,
                )

            chart_i += 1
            potential_placeholder.plotly_chart(
                render_plotly_figure(update["vx"], update["vy"], rows, cols),
                use_container_width=True, key=f"compare_well_{chart_i}",
            )

            if update["done"]:
                st.success(f"Run complete -- {len(df)} frames processed.")
                break
    else:
        st.info("Choose a mode and config above, then click Run.")
