"""CSS/layout helpers for the Streamlit app's 'virtual lab' framing.

NEW for WISER26. Purely presentational -- no physics lives here. Purpose:
make the dashboard read as a lab console (cryostat status, instrument
readouts) rather than a generic data-science notebook UI, so the
hardware-reference imagery in assets/hardware_refs/ (see CREDITS.md) sits
in a consistent visual context instead of looking bolted on.

TODO: actual CSS/theming to be filled in once the Streamlit app layout
(app.py) is further along -- placeholder function signatures for now so
app.py has a stable import target to build against.
"""


def inject_lab_css() -> str:
    """Return a <style> block to inject into the Streamlit app via
    st.markdown(inject_lab_css(), unsafe_allow_html=True).

    Currently a minimal placeholder (dark background, monospace readouts)
    -- expand once the visual direction is settled. Keep this file the
    single source of truth for the lab theme so app.py doesn't accumulate
    inline style strings.
    """
    return """
    <style>
    /* PLACEHOLDER -- lab console visual direction TBD */
    .stApp { background-color: #0b0f14; }
    .lab-readout { font-family: 'Courier New', monospace; color: #9fe6a0; }
    </style>
    """


def status_badge(label: str, ok: bool) -> str:
    """Small HTML status-light snippet, e.g. for 'FRIDGE: NOMINAL' /
    'FRIDGE: DRIFT DETECTED' style readouts next to the live plots.
    """
    color = "#3ddc84" if ok else "#e05d5d"
    return f'<span class="lab-readout" style="color:{color}">{label}</span>'
