"""Streamlit app: the interactive 'virtual lab' dashboard.

NEW for WISER26 -- no equivalent in the original qdot-live-twin repo
(which had a Streamlit dependency listed but only a placeholder launch
command and no working app file, per the original README -- see
docs/PORTING_NOTES.md).

STATUS: functional MVP skeleton, not yet polished. Lets a learner pick a
mode and config, run it, and watch staleness update live alongside the
schematic 3D potential well. TODO before submission: richer lab-theme
styling (viz/lab_theme.py is currently placeholder CSS), hardware
reference imagery panel (assets/hardware_refs/, assets/renders/), and
learner-facing explanatory text tied to docs/LEARNING_OBJECTIVES.md.

Run with: streamlit run app.py
"""
import streamlit as st

from qdot_edu.pipeline import run_live
from qdot_edu.stream.trajectory import load_trajectory_config
from qdot_edu.viz.lab_theme import inject_lab_css, status_badge
from qdot_edu.viz.potential_well import render_plotly_figure

st.set_page_config(page_title="qdot-edu: live twin", layout="wide")
st.markdown(inject_lab_css(), unsafe_allow_html=True)

st.title("Quantum Dot Digital Twin -- Live Console")
st.caption(
    "A CPU-only teaching build. See README.md for the full write-up and "
    "docs/lessons/ for the guided notebook arc this app complements."
)

with st.sidebar:
    st.header("Controls")
    mode = st.selectbox(
        "Pipeline mode",
        ["serial", "batched", "batched_triage"],  # "batched_triage_llm" omitted -- not yet ported, see docs/PORTING_NOTES.md
        index=2,
        help="serial = naive baseline (watch it fall behind); batched = "
             "micro-batched, no triage; batched_triage = adaptive agent "
             "picking FULL/CHEAP/SKIP per micro-batch.",
    )
    config_path = st.selectbox(
        "Trajectory config",
        ["configs/trajectory_quick.yaml", "configs/trajectory_cpu_edu.yaml", "configs/trajectory.yaml"],
        help="quick = fastest for iterating; cpu_edu = tuned for CPU teaching demos "
             "(rate still a placeholder, see its own TODO); trajectory.yaml = full original run. "
             "Each config also sets array_size (rows, cols) -- see configs/*.yaml.",
    )
    run_clicked = st.button("Run", type="primary")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Staleness over time")
    staleness_placeholder = st.empty()
    status_placeholder = st.empty()

with col_right:
    st.subheader("Schematic confinement potential")
    st.caption(
        "Gaussian-well interpolation over the array's derived gate energies -- "
        "NOT a Poisson-equation solve. See viz/potential_well.py's module docstring."
    )
    potential_placeholder = st.empty()

if run_clicked:
    array_size = load_trajectory_config(config_path).array_size
    rows, cols = array_size

    potential_placeholder.plotly_chart(
        render_plotly_figure(0.0, 0.0, rows, cols), use_container_width=True
    )

    for update in run_live(mode, config_path, yield_every=5, device="cpu"):
        df = update["df"]
        if len(df) > 0:
            staleness_placeholder.line_chart(df.set_index("frame_index")["wall_clock_lag"])

        if update["tier_counts"] is not None:
            status_placeholder.markdown(
                status_badge(f"tiers so far: {update['tier_counts']}", ok=True),
                unsafe_allow_html=True,
            )

        potential_placeholder.plotly_chart(
            render_plotly_figure(update["vx"], update["vy"], rows, cols),
            use_container_width=True,
        )

        if update["done"]:
            st.success(f"Run complete -- {len(df)} frames processed.")
            break
else:
    st.info("Choose a mode and config in the sidebar, then click Run.")
