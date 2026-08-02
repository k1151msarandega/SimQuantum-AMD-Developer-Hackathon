"""Scripted device trajectory: gate voltages as a function of frame index.

Pure function of (frame_index, config) -> (Vx, Vy). No I/O, no state --
this makes it trivially testable and reusable from both the serial and
batched pipelines.

PORTED UNCHANGED from qdot-live-twin (Act II hackathon), src/qdot_twin/
stream/trajectory.py. No GPU dependency in this file -- nothing to adapt.
See docs/PORTING_NOTES.md.
"""
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class TrajectoryConfig:
    n_frames: int
    array_size: tuple[int, int]   # (rows, cols) -- NEW, was previously parsed but unused (see docs/PORTING_NOTES.md)
    vx_range: tuple[float, float]
    vy_range: tuple[float, float]
    noise_std: float
    jump_at_frame: int
    jump_vx_delta: float
    jump_vy_delta: float
    stream_rate_hz_start: float
    stream_rate_hz_end: float


def voltage_at(frame_index: int, cfg: TrajectoryConfig) -> tuple[float, float]:
    """Return (Vx, Vy) ground-truth gate voltages at a given frame index.

    Linear creep across the full run, plus a step-function offset applied
    for all frames after cfg.jump_at_frame (the injected discrete drift event).
    """
    progress = frame_index / max(cfg.n_frames - 1, 1)
    vx = cfg.vx_range[0] + progress * (cfg.vx_range[1] - cfg.vx_range[0])
    vy = cfg.vy_range[0] + progress * (cfg.vy_range[1] - cfg.vy_range[0])

    vx += float(np.random.normal(0.0, cfg.noise_std))
    vy += float(np.random.normal(0.0, cfg.noise_std))

    if frame_index >= cfg.jump_at_frame:
        vx += cfg.jump_vx_delta
        vy += cfg.jump_vy_delta

    return vx, vy


def load_trajectory_config(path: str) -> TrajectoryConfig:
    """Load a TrajectoryConfig from a YAML file (e.g. configs/trajectory.yaml)."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    return TrajectoryConfig(
        n_frames=raw["n_frames"],
        array_size=tuple(raw["array_size"]),
        vx_range=tuple(raw["creep"]["vx_range"]),
        vy_range=tuple(raw["creep"]["vy_range"]),
        noise_std=raw["creep"]["noise_std"],
        jump_at_frame=raw["jump"]["at_frame"],
        jump_vx_delta=raw["jump"]["vx_delta"],
        jump_vy_delta=raw["jump"]["vy_delta"],
        stream_rate_hz_start=raw["stream_rate_hz"]["start"],
        stream_rate_hz_end=raw["stream_rate_hz"]["end"],
    )
