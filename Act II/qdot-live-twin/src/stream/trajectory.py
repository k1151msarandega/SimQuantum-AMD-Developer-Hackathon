cat > src/qdot_twin/stream/trajectory.py << 'EOF'
"""Scripted device trajectory: gate voltages as a function of frame index.

Pure function of (frame_index, config) -> (Vx, Vy). No I/O, no state —
this makes it trivially testable and reusable from both the serial and
batched pipelines.
"""
from dataclasses import dataclass


@dataclass
class TrajectoryConfig:
    n_frames: int
    vx_range: tuple[float, float]
    vy_range: tuple[float, float]
    noise_std: float
    jump_at_frame: int
    jump_vx_delta: float
    jump_vy_delta: float


def voltage_at(frame_index: int, cfg: TrajectoryConfig) -> tuple[float, float]:
    """Return (Vx, Vy) ground-truth gate voltages at a given frame index.

    Linear creep across the full run, plus a step-function offset applied
    for all frames after cfg.jump_at_frame (the injected discrete drift event).
    """
    raise NotImplementedError


def load_trajectory_config(path: str) -> TrajectoryConfig:
    """Load a TrajectoryConfig from a YAML file (e.g. configs/trajectory.yaml)."""
    raise NotImplementedError
EOF
