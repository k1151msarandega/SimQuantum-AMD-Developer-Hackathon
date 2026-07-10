cat > src/qdot_twin/perception/ensemble.py << 'EOF'
"""Ensemble-disagreement signal, repointed from the old repo's classifier.py.

Old question: how much do ensemble members disagree on charge-state ID?
New question: same mechanism, fed into the drift flag alongside ood.py --
high disagreement is a second, independent signal that something about
the incoming data no longer matches what the twin expects.
"""
import numpy as np


def ensemble_disagreement(frame: np.ndarray) -> float:
    """Return a disagreement score in [0, 1] across ensemble members."""
    raise NotImplementedError
EOF
