"""qdot_twin: a live digital twin of a quantum dot device.

A twin passively mirrors the device's current state as it changes.
It does not steer the device anywhere (that's navigation/tuning — out of scope).
Success is measured as staleness (how far behind the twin's belief is from
ground truth), not as "reaching a target."
"""
