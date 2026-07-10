cat > src/qdot_twin/twin/serial_estimator.py << 'EOF'
"""CPU, one-frame-at-a-time state estimator. The honest baseline.

This exists to produce real, measured evidence of the twin falling behind
under load -- not to be a good solution. Do not optimize this file; the
point is that it's slow, and the batched estimator's win is only credible
if this baseline is real and unoptimized.
"""
import numpy as np


class TwinState(dict):
    """Estimated device state: charge occupation, confidence, timestamp, etc."""


def estimate(frame: np.ndarray) -> TwinState:
    """Turn a single raw frame into a state estimate. CPU, serial, no batching."""
    raise NotImplementedError
EOF

