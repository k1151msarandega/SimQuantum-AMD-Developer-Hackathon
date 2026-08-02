"""Shared quantum dot array physical model parameters.

Single source of truth for the array layout and constant-capacitance
matrices, derived from `array_size`, used by BOTH stream/generator.py
(drives the real QArray simulation) and viz/potential_well.py (drives the
schematic 3D visualization). Closes docs/PORTING_NOTES.md item 5
(previously these were two independent hardcoded copies).

MODEL: dots arranged in a `rows x cols` grid, one plunger gate per dot.
Capacitive coupling is nearest-neighbor only (4-connected grid: up/down/
left/right). This generalizes the original hackathon repo's hand-picked
two-dot matrices -- Cdd=[[0,0.1],[0.1,0]], Cgd=[[1,0.2],[0.2,1]] -- rather
than inventing new physics: self-coupling 1.0, nearest-neighbor gate
cross-talk 0.2, nearest-neighbor dot-dot coupling 0.1. array_size=(1, 2)
reproduces the original matrices exactly (see dot_grid_matrices).

VERIFY BEFORE TRUSTING FOR THE SUBMISSION: stream/generator.py only ever
sweeps the array's first two gates (P1, P2) via QArray's do2d_open, no
matter how many dots are configured. What voltage un-swept gates default
to during do2d_open has NOT been verified against the installed QArray
version -- assumed 0V here (QArray's typical default). If wrong, results
for array_size beyond a 2-dot line will be internally consistent but not
physically correct. See docs/PORTING_NOTES.md.
"""
import numpy as np

SELF_CGD = 1.0
NEIGHBOR_CGD = 0.2
NEIGHBOR_CDD = 0.1
GRID_SPACING = 1.0  # arbitrary length unit, viz layout only -- no physical calibration


def dot_grid_positions(rows: int, cols: int) -> np.ndarray:
    """Physical (x, y) layout of each dot, centered at the origin.

    Row-major dot ordering (dot index = r*cols + c) -- MUST match the
    ordering used by dot_grid_matrices() and gate_names() below, since
    all three are indexed together elsewhere (e.g. potential_well.py
    zips positions with per-dot well depths).
    """
    xs = (np.arange(cols) - (cols - 1) / 2) * GRID_SPACING
    ys = (np.arange(rows) - (rows - 1) / 2) * GRID_SPACING
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel()], axis=-1)


def _neighbor_pairs(rows: int, cols: int):
    """Yield 4-connected grid-neighbor (i, j) index pairs, row-major indexing."""
    def idx(r, c):
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            if c + 1 < cols:
                yield i, idx(r, c + 1)
            if r + 1 < rows:
                yield i, idx(r + 1, c)


def dot_grid_matrices(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (Cdd, Cgd) for a rows x cols grid, one plunger gate per dot.

    Cdd: (n_dot, n_dot) -- 0 on the diagonal, NEIGHBOR_CDD for 4-connected
    neighbor pairs, 0 elsewhere.
    Cgd: (n_dot, n_gate) with n_gate == n_dot -- SELF_CGD on the diagonal
    (each gate's own dot), NEIGHBOR_CGD for 4-connected neighbor
    cross-talk, 0 elsewhere.

    Sanity check (see module docstring): dot_grid_matrices(1, 2) returns
    Cdd=[[0, 0.1], [0.1, 0]], Cgd=[[1, 0.2], [0.2, 1]] -- exactly the
    original hardcoded two-dot matrices.
    """
    n = rows * cols
    Cdd = np.zeros((n, n))
    Cgd = np.eye(n) * SELF_CGD

    for i, j in _neighbor_pairs(rows, cols):
        Cdd[i, j] = Cdd[j, i] = NEIGHBOR_CDD
        Cgd[i, j] = Cgd[j, i] = NEIGHBOR_CGD

    return Cdd, Cgd


def gate_names(rows: int, cols: int) -> list[str]:
    """One plunger gate name per dot, same row-major order as the other functions here."""
    return [f"P{i + 1}" for i in range(rows * cols)]
