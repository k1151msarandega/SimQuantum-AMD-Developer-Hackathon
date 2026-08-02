"""Live 3D electrostatic potential-well surface for the dot array.

NEW for WISER26 -- no equivalent in the original qdot-live-twin repo.

WHAT THIS IS: a schematic rendering of the confinement landscape implied
by the SAME capacitance parameters (qdot_edu.model_params) that
stream/generator.py uses to drive QArray's actual charge-stability
simulation. Each dot's "well depth" at a given gate voltage is derived
from the real Cgd coupling used elsewhere in this codebase -- it is not a
made-up number, and it is now the same shared source as generator.py's
(see docs/PORTING_NOTES.md item 5 -- this used to be a separate hardcoded
copy that could silently drift out of sync; not anymore).

WHAT THIS IS NOT, and must not be presented as in the submission: a
solution to Poisson's equation over continuous space. The constant-
capacitance model (what QArray/this repo actually simulates) gives
energies AT the discrete dot sites, not a continuum potential between
them. The smooth surface below fills the space between dot sites with
Gaussian wells centered on each dot's physical position, depth set by
that dot's derived energy -- this is a standard *pedagogical* device
(the same kind of cartoon used in textbooks/talks to give intuition for
"electron trapped here, not there") but it is an interpolation choice,
not physics between the dot sites. Learner-facing text (see
docs/lessons/) must say this explicitly, or a technically literate judge
will reasonably read it as claiming more rigor than it has.

Only the array's first two gates (P1, P2) are ever driven here, matching
stream/generator.py's do2d_open sweep -- any additional dots/gates beyond
the first two are rendered at their model_params-derived coupling to
those same two swept gates, with all other gates assumed at 0V. See
model_params.py's verification caveat.

TODO before relying on this for the submission: check whether the
installed QArray version exposes an actual potential-landscape/point
query API (something like a single-point equivalent of the do2d_open
sweep in stream/generator.py) that could replace the Gaussian-well
interpolation below with the model's own computed values at a finer
grid. Not yet verified against the installed version.
"""
import numpy as np

from qdot_edu import model_params

WELL_SIGMA = 0.4        # spatial spread of each Gaussian well, arbitrary units
GRID_RES = 40             # surface resolution; keep low for live/interactive rendering
GRID_MARGIN = 1.0         # extra +/- range beyond the dot layout's own extent


def dot_well_depths(vx: float, vy: float, rows: int, cols: int) -> np.ndarray:
    """Per-dot 'well depth' proxy at the given gate voltages, for a rows x
    cols array (see qdot_edu.model_params).

    depth_i = -(Cgd[:, :2] @ [vx, vy])_i -- only the first two gate
    columns (P1, P2) are driven, matching stream/generator.py's sweep;
    more negative gate-induced potential at a dot reads as a DEEPER well
    (more attractive to an electron) in this convention. Sign and scale
    are a modeling choice for the visualization, consistent with "more
    gate voltage pulling this dot's potential down => more likely to trap
    a charge," but not calibrated against a specific real-device energy
    scale.
    """
    _, Cgd = model_params.dot_grid_matrices(rows, cols)
    v = np.array([vx, vy])
    return -(Cgd[:, :2] @ v)


def potential_surface(vx: float, vy: float, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Y, Z) grids for a 3D surface plot of the schematic
    confinement potential at the given gate voltages, for a rows x cols array.

    See module docstring for what this is and is not.
    """
    depths = dot_well_depths(vx, vy, rows, cols)
    positions = model_params.dot_grid_positions(rows, cols)

    extent = max(rows, cols) * model_params.GRID_SPACING / 2 + GRID_MARGIN
    lin = np.linspace(-extent, extent, GRID_RES)
    X, Y = np.meshgrid(lin, lin)
    Z = np.zeros_like(X)

    for (dx, dy), depth in zip(positions, depths):
        r2 = (X - dx) ** 2 + (Y - dy) ** 2
        Z += depth * np.exp(-r2 / (2 * WELL_SIGMA ** 2))

    return X, Y, Z


def electron_trapped_mask(Z: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Return a boolean grid: True where the well is deep enough to plausibly
    trap an electron, for the "trapped / not trapped" visual callout.

    threshold=None picks the midpoint between the surface's min and max at
    render time -- a relative, per-frame threshold, not a calibrated
    physical one. Good enough for "deeper = more likely trapped" intuition;
    NOT a claim about a real occupation number (that's what the actual
    QArray charge_state_to_scalar output in stream/generator.py is for).
    """
    if threshold is None:
        threshold = (Z.min() + Z.max()) / 2
    return Z < threshold


def render_plotly_figure(vx: float, vy: float, rows: int, cols: int):
    """Build a Plotly 3D surface figure for the current gate voltages and
    array shape.

    Import is local to this function so importing this module doesn't
    hard-require plotly if a caller only wants the raw arrays above
    (e.g. for a matplotlib-based notebook instead).
    """
    import plotly.graph_objects as go

    X, Y, Z = potential_surface(vx, vy, rows, cols)
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
    fig.update_layout(
        scene=dict(
            xaxis_title="x (a.u.)",
            yaxis_title="y (a.u.)",
            zaxis_title="schematic potential (a.u.)",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig
