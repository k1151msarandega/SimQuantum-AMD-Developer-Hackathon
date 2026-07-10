cat > src/qdot_twin/perception/ood.py << 'EOF'
"""Out-of-distribution detector, repointed from the old repo.

Old question: does this frame match the training distribution of valid
charge states?
New question: does this incoming frame match the twin's *current rolling
expectation* -- a window of recently-accepted frames -- rather than a fixed
training set. This adaptation matters: the device legitimately drifts
(slow creep should NOT trip the flag), but a sudden jump should.

Open design point carried over from planning: window size, and how we stop
the reference window from silently absorbing a real drift event as normal.
"""
import numpy as np


class RollingOODDetector:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        raise NotImplementedError

    def update_and_check(self, frame: np.ndarray) -> bool:
        """Returns True if `frame` looks anomalous relative to the rolling window."""
        raise NotImplementedError
EOF

