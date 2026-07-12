"""Staleness-curve plotting, shared across all three run modes."""
import matplotlib.pyplot as plt


def plot_staleness_comparison(logs: dict, jump_at_frame: int, save_path: str | None = None):
    """logs: {"serial": StalenessLog, "batched": StalenessLog, "batched_triage": StalenessLog}

    One subplot per regime, sharing both x and y axis scale, so the three
    are directly, visually comparable -- same y-scale is deliberate: it
    would be easy to make batched/batched_triage look artificially good by
    letting each subplot auto-scale its own axis.
    """
    fig, axes = plt.subplots(len(logs), 1, figsize=(10, 3 * len(logs)), sharex=True, sharey=True)
    if len(logs) == 1:
        axes = [axes]

    for ax, (name, log) in zip(axes, logs.items()):
        df = log.to_dataframe()
        ax.plot(df["frame_index"], df["wall_clock_lag"])
        ax.axvline(jump_at_frame, color="red", linestyle="--")
        ax.set_ylabel(f"{name}\nlag (s)")

    axes[-1].set_xlabel("frame index")
    axes[0].set_title("Staleness comparison: serial vs batched vs batched+triage")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()
