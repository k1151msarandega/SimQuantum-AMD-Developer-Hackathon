"""Wires stream -> twin -> staleness -> drift -> triage into three runnable modes.

Modes: "serial", "batched", "batched_triage" -- run each over the same
trajectory config to produce the three-regime staleness comparison chart.
"""
from typing import Literal


def run(mode: Literal["serial", "batched", "batched_triage"], config_path: str):
    raise NotImplementedError
