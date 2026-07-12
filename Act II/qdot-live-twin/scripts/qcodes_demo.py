"""Demo: register the twin's QArray stream as a QCoDeS Station instrument.

Shows this pipeline's data source is compatible with real lab control
software (QCoDeS), without changing pipeline.py's own already-proven
stream consumption. Standalone credibility demo for the pitch/README --
not part of the timed serial/batched/triage runs.

Usage:
    python scripts/qcodes_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qcodes.station import Station

from qdot_twin.hardware.qcodes_adapter import QArrayTwinInstrument

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "trajectory.yaml")


def main():
    twin = QArrayTwinInstrument("qdot_twin_source", CONFIG_PATH)
    station = Station(twin)

    print("QCoDeS Station registered:")
    print(f"  instruments: {list(station.components.keys())}")
    print()

    print("Pulling 5 frames through the QCoDeS Parameter interface...")
    for _ in range(5):
        twin.next_frame()
        print(
            f"  frame_index={twin.frame_index()}  "
            f"vx={twin.vx():.4f}  vy={twin.vy():.4f}  "
            f"frame_shape={twin.frame().shape}"
        )

    print()
    print("Station snapshot (metadata only -- frame arrays excluded via snapshot_value=False):")
    snap = station.snapshot()
    print(list(snap["instruments"]["qdot_twin_source"]["parameters"].keys()))

    twin.close()


if __name__ == "__main__":
    main()
