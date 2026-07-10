cat > src/qdot_twin/metrics.py << 'EOF'
"""Staleness-curve plotting, shared across all three run modes."""
import matplotlib.pyplot as plt


def plot_staleness_comparison(logs: dict, save_path: str | None = None):
    """logs: {"serial": StalenessLog, "batched": StalenessLog, "batched_triage": StalenessLog}

    Plots wall-clock lag as the headline line, state-error magnitude as a
    secondary line, same x-axis (frame index), one subplot per regime or
    overlaid -- decide during the demo-polish pass.
    """
    raise NotImplementedError
EOF
