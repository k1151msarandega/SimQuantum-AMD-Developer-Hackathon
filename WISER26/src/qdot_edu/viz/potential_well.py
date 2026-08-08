"""Live 3D electrostatic potential-well surface for the dot array.

REPLACES the original WISER26 version of this file, which rendered a
schematic Gaussian-well interpolation and carried an explicit TODO asking
whether the installed QArray version exposes a real potential-query API
to replace it with. It does: QArray's DotArray.free_energy(n, vg) --
verified directly against the installed package -- computes the real
electrostatic free energy of a fixed charge configuration n across a grid
of nearby gate voltages vg, using the exact same constant-capacitance
model (qdot_edu.model_params) that stream/generator.py already drives
QArray's actual charge-stability simulation with. This is that same
physics, not a second, independent approximation of it.

WHAT THIS SHOWS: the free-energy landscape the device's CURRENT charge
configuration sits in, as a function of nearby (Vx, Vy) -- i.e. "if the
gates were nudged slightly from here, how energetically costly would that
be relative to this configuration." A deep well (low free energy) at a
point means that region of gate-voltage space is a stable place for the
CURRENT charge configuration to sit; a shallow/flat region means the
configuration is only weakly favored there. This is the real quantity a
learner should read as "electron trapped here, not there."

WHAT THIS IS STILL NOT, and must not be presented as: a solution to
Poisson's equation over continuous space. The constant-capacitance model
(what QArray/this repo actually simulates) gives free energies for
DISCRETE charge configurations n at each gate-voltage point -- there is no
continuous charge density between dot sites in this model. The surface
below is real QArray output evaluated on a grid, not an interpolation
between hand-picked points, but the underlying model itself is still the
same constant-capacitance approximation the rest of this repo uses, not a
full electrostatics solve. Say so in learner-facing material.

Only the array's first two gates (P1, P2) are ever swept here, matching
stream/generator.py's do2d_open sweep. Any additional gates beyond the
first two are held at 0V for both the sweep grid and the reference
charge configuration -- see model_params.py's verification caveat on
what QArray assumes for un-swept gates.

WINDOW SIZE, and why it's no longer tied to generator.PATCH_WINDOW: this
module originally reused stream/generator.py's +/-PATCH_WINDOW so the two
panels were directly, spatially comparable side by side. In practice that
made the well look like a flat tilted sheet almost everywhere except very
early in a trajectory -- verified numerically, not just by eyeballing a
screenshot: Cdd_inv for this repo's 2-dot matrices has eigenvalues
[-10, +10], i.e. free_energy is a SADDLE in (Vx, Vy), not a bowl, and its
curvature magnitude over a window is roughly fixed while the surface's
local linear gradient grows with distance from the saddle center -- so a
small fixed window centered far from that point is dominated by the
gradient and looks nearly planar (measured ~1.7% deviation from a planar
fit at Vx~9V with a +/-0.5V window, vs ~10% at +/-3V, same point -- see
docs/PORTING_NOTES.md). WELL_WINDOW below trades exact side-by-side
comparability with the device-feed patch for the well actually being
readable most of the time. It is still a real, exact evaluation of
free_energy over that wider window, not an approximation -- just a wider
one than the device-feed patch uses.
"""
import numpy as np

from qdot_edu import model_params
from qdot_edu.stream.generator import build_model

WELL_WINDOW = 3.0  # +/- volts; deliberately wider than generator.PATCH_WINDOW -- see module docstring
GRID_RES = 40  # surface resolution; cheap linear algebra, not GPU/CNN work -- fine to keep live-interactive


def potential_surface(
    vx: float, vy: float, rows: int, cols: int, model=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Y, Z) grids of the real QArray free-energy landscape
    around (vx, vy), for a rows x cols array.

    Z = model.free_energy(n_center, vg_grid): the free energy of the
    ground-state charge configuration AT (vx, vy) (n_center, from
    QArray's own ground_state_open), evaluated across a grid of nearby
    (Vx, Vy) points spanning +/-WELL_WINDOW (see module docstring for why
    this is wider than, and independent from, generator.PATCH_WINDOW).
    All gates beyond the first two (P1, P2) are held at 0V in both
    n_center and the sweep grid, matching stream()'s own sweep.

    `model`, if given, reuses a caller-provided DotArray (e.g. the live
    console keeps one alive across frames rather than rebuilding it every
    render) instead of building a fresh one via build_model() each call.
    """
    if model is None:
        model = build_model(rows, cols)

    n_gate = rows * cols
    xs = np.linspace(vx - WELL_WINDOW, vx + WELL_WINDOW, GRID_RES)
    ys = np.linspace(vy - WELL_WINDOW, vy + WELL_WINDOW, GRID_RES)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")

    vg_grid = np.zeros(x_grid.shape + (n_gate,))
    vg_grid[..., 0] = x_grid
    vg_grid[..., 1] = y_grid

    center_vg = np.zeros(n_gate)
    center_vg[0] = vx
    center_vg[1] = vy
    n_center = model.ground_state_open(center_vg)

    z = model.free_energy(n_center, vg_grid)  # (GRID_RES, GRID_RES, 1)
    return x_grid, y_grid, np.asarray(z)[..., 0]


def electron_trapped_mask(Z: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Return a boolean grid: True where free energy is low enough to read
    as "the current charge configuration is stable here" -- the
    "trapped / not trapped" visual callout.

    threshold=None picks the midpoint between the surface's min and max at
    render time -- a relative, per-frame threshold, not a calibrated
    physical one. Real free energy, relative threshold: good enough for
    "deeper (lower F) = more stable here" intuition, NOT a claim about a
    calibrated real occupation number (that's what stream/generator.py's
    charge_state_to_scalar output is for).
    """
    if threshold is None:
        threshold = (Z.min() + Z.max()) / 2
    return Z < threshold


def render_plotly_figure(vx: float, vy: float, rows: int, cols: int, model=None):
    """Build a Plotly 3D surface figure of the real free-energy landscape
    at the given gate voltages and array shape.

    Import is local to this function so importing this module doesn't
    hard-require plotly if a caller only wants the raw arrays above
    (e.g. for a matplotlib-based notebook instead).
    """
    import plotly.graph_objects as go

    X, Y, Z = potential_surface(vx, vy, rows, cols, model=model)
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z, colorscale="Viridis_r",  # reversed: darker = lower free energy = deeper well
        colorbar=dict(title="free energy (a.u.)"),
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title="Vx (V)",
            yaxis_title="Vy (V)",
            zaxis_title="free energy F(n, Vg) (a.u.)",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"Potential well -- real QArray free_energy at Vx={vx:.3f} V, Vy={vy:.3f} V",
    )
    return fig
