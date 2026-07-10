cat > src/qdot_twin/stream/generator.py << 'EOF'
"""QArray-backed frame stream: plays the role of 'the real device'.

Wraps QArray's charge-sensor simulation to emit (frame, ground_truth_state,
timestamp) tuples at a configurable, ramping rate — fast enough to eventually
outpace the serial baseline. This is the honest 'fire hose' data source; it
does not know or care whether anything downstream can keep up.
"""
from typing import Iterator, NamedTuple
import numpy as np


class Frame(NamedTuple):
    data: np.ndarray          # the simulated stability-diagram patch
    vx: float                 # ground-truth gate voltage at emission time
    vy: float
    frame_index: int
    emitted_at: float         # wall-clock timestamp


def stream(config_path: str) -> Iterator[Frame]:
    """Yield Frames at the rate specified in the trajectory config.

    Rate ramps from stream_rate_hz.start to stream_rate_hz.end over the run,
    per configs/trajectory.yaml, so early frames are easy to keep up with and
    later frames are deliberately not.
    """
    raise NotImplementedError
EOF
