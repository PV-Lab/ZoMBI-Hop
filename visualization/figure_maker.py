"""Design studio: hand-build a 2D synthetic landscape and dress it for a figure.

A single Dash app showing a :class:`~synthetic_data.ensemble.CartesianEnsemble`
on the unit square as a rotatable 3D surface whose height *and* colour are the
objective (yellow high, blue low).  This is the mode for hand-building a
landscape; the benchmark views of the same objective (ternary heatmap,
tetrahedron point cloud) live in ``synthetic_data/plot_ensemble.py``.

The ensemble itself is **fixed** — see :data:`ENSEMBLE` — so the landscape
underneath is the same every time the app is opened and only the hand-placed
features and the drawing controls change.  Optima are never sampled: the only
ones present are the ones clicked in.

The **interaction** radio picks what the mouse does:

  * *Rotate / inspect* — normal 3D navigation (drag to orbit, scroll to zoom).
  * *Place optima* — the camera is pinned to whatever orientation you left it
    in, so the surface holds still (a drag snaps straight back on the next
    redraw), and clicking pins an optimum where you clicked, raising a hill
    there; click the same spot again to drop it, or use **Clear Optima**.
    **Optima height** sets how far every placed optimum towers over the
    roughness: the peaks are raised while the background's cap stays where it
    is, and the objective is normalized by its own largest value, so turning it
    up squeezes the Perlin noise into an ever thinner band below the summits.
    Each optimum then gets its own slider under **individual optimum heights**,
    a multiple of that in ``[0.05, 3]``, so a landscape can mix towering optima
    with lesser ones; the tallest always reaches the top of the range, so the
    sliders set heights *relative* to one another.
  * *Red penalization* / *Grey penalization* — clicking a
    placed optimum wraps it in a penalization region, the 3D counterpart of the
    ternary penalty zones drawn by ``interface/app.py``.  Both are the same
    shape: a single translucent cylinder of even tint running the full height of
    the plot — no falloff, no shells — in red for one and light grey for the
    other.  Each colour carries its own **penalization radius** and
    **penalization cylinder opacity** sliders (0.15 and 0.5 by default), and
    both pairs stay on screen in either mode, so one column can be sized against
    the other without switching modes to reach its slider.  The two regions are
    independent, so an optimum can carry both at once — whichever is drawn
    second is then built imperceptibly narrower and shorter, since two columns
    at the same radius would otherwise share a wall, and coincident translucent
    walls tear into stripes.
    Clicking again in the same mode clears
    that colour's region, leaving the other in place.  A click anywhere on a
    column resolves to the optimum it stands on — the click itself reports the
    wall it landed on, which is nowhere near that summit.  A solid column is
    also opaque to plotly's pick pass, which would leave anything behind it
    unclickable, so while a placing mode is live the column is drawn as its
    outline alone and the ray passes between the strokes; switch back to
    *Rotate / inspect* to see it solid again.  A column stands on a
    summit, so most of it is inside its own hill — drop the **background
    opacity** to see the buried part, or tick **wireframe edges on penalization
    cylinders** to ink that same outline over the solid column, which reads the
    penalized footprint straight off the surface without having to see through
    the hill.

    The outline is cel-shaded: only the strokes a cartoonist would draw — the
    top circle, the bottom circle and the closed contour where the wall cuts the
    landscape, struck on the radius itself so the outline is the cylinder's own
    edge, and inked opaque so it does not composite through the wall behind it.
    Over a filled column that is all of them: the wall is its own silhouette
    already.  While *placing*, where the column is nothing but strokes, it also
    gets the two vertical *silhouette* edges where the wall would turn away from
    the viewer — a quarter turn either side of the direction the camera looks
    from, so they follow the view as it orbits, at an angle rounded to 5° so a
    drag redraws the figure a handful of times rather than once a frame.
    **Penalization wireframe thickness** sets how heavily all of it is inked.
  * *Draw printed line* — two clicks make a line between two boundary points of
    the square (both endpoints snap to the nearest side).  It is drawn as evenly
    spaced samples styled like the measured points in
    ``visualization/plot_run.py``, riding the height map so it visibly bends over
    the contours.  Every sample also casts a sticker onto the floor of the plot
    — the point's black outline alone, drawn flat as if stuck to the plane — with
    a skinny vertical drop tying it to the point it belongs to, so the samples
    read as a line laid across the domain rather than dots afloat above it.
    **Printed line samples** sets how many points a line is drawn from, **point
    size** how big each one is, **outline width** how heavy its black rim (and
    so the floor ring) is, **floor projection opacity** how strongly the stickers
    sit on the plane, and **connector line width** / **opacity** the drops.  An
    outline width of 0 leaves nothing to project, so the stickers go with it.
    **Point height above surface** lifts the coloured samples (and the black rim
    that belongs to each one) clear of the landscape so they read on top of it
    rather than half-sunk into the slope in front of them — plotly depth-tests
    every 3D trace against every other and offers no draw-order override, so the
    clearance is what buys it.  The floor stickers and the drops keep their own
    depth and so still pass behind the landscape.  Wiped by **Clear Lines**.

Placed optima carry no marker of their own — the hill each one raises is where
it is, and clicks resolve against the stored position, so the landscape is only
ever shown with the penalization that has been pinned onto it.

Printed lines are an overlay only: they do not alter the objective.  A
**background opacity** slider fades the surface underneath both overlays so they
read clearly against it, and a **peak smoothing** slider rounds the cusp at the
tip of every basin so optima render as hills rather than spikes.  A **colour
scale floored at the 10th percentile** checkbox lifts the bottom of the colour
limits off the objective's own minimum, which is what keeps the background
legible once one deep trough is enough to own the low end of the range.  The top
of the scale is left alone — the largest value in the data is always the top
colour — and heights are always the real objective either way.

Usage:
    python visualization/figure_maker.py
"""

import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # repo root, so synthetic_data is importable
from synthetic_data.ensemble import (  # noqa: E402
    _BASE,
    _PEAK,
    CartesianEnsemble,
    _negated_ackley_env01,
)
from synthetic_data.plot_ackley import FIG_H, FIG_W  # noqa: E402

# The landscape under the hand-placed features, fixed rather than exposed as
# sliders: this app is for dressing one known background, not for exploring the
# ensemble's parameter space (that is what ``plot_ensemble.py`` is for).  Weak
# optima, ridges, anisotropy, plateaus and the edge / structural bias are all
# switched off by their disabling values, so what is left is Perlin roughness
# plus whatever optima are clicked in.  ``n_optima=0`` is what keeps the
# sampled ones out; :class:`DesignEnsemble` prepends the pinned ones.
ENSEMBLE = dict(
    n_optima=0,
    basin_width=12.4,
    optima_margin=0.2,
    n_weak=0,
    n_ridges=0,
    noise_freq=8.0,
    noise_amp=1100.0,
    noise_octaves=2,
    aniso_strength=0.0,
    n_plateaus=0,
    edge_region=None,
    neg_frac=0.5,
    seed=11,
)
# Lattice the square is drawn on, and the cut below which the surface goes black
# (off by default, as everywhere else here — both were ensemble-level knobs).
GRID_N = 160
BASIN_THRESHOLD = 0.0

# The square is drawn 1:1 in x/y with the objective squashed into a shallower z
# so the height map reads as a landscape rather than a tower.
DESIGN_ASPECT_Z = 0.55
# Overlays are lifted this fraction of the objective range above the surface so
# they are not z-fought by the polygon they sit on.
OVERLAY_LIFT = 0.02
DESIGN_AXIS_LABELS = ("x1", "x2")
# Colour of the xy floor plane.  Printed-line stickers punch a hole of exactly
# this colour to read as rings, so the plane cannot be left at plotly's default.
FLOOR_COLOR = "rgb(229, 236, 246)"
# Penalization regions.  ``PENALTY_ALPHA`` is the opacity the outlines and the
# cylinders are drawn at, matching ``_draw_penalty_gradient`` in
# ``interface/app.py`` so these read at the same strength as the ternary rings
# they mirror.
PENALTY_ALPHA = 0.5
# Samples around the circle the solid cylinder's wall is tessellated from, and
# how far its caps are held off the ends of the column, as a fraction of the
# column's height.  The z axis is ranged to exactly the span the cylinder is
# built to, so a cap laid on either end sits *on* the scene's clip plane and
# plotly clips it a triangle at a time — which shows up as a starburst of the
# background through the cap's own triangulation.  Holding the caps a hair
# inside is what keeps them whole; at this size the shortening is invisible.
PENALTY_SOLID_N = 96
PENALTY_SOLID_INSET = 2e-3
# Samples around the circle of the wireframe's rings and its landscape contour.
PENALTY_CONTOUR_N = 145
# How finely the silhouette is allowed to lag the camera, in degrees of azimuth.
# The two vertical wires are the edges of the column *as seen from where you are
# standing*, so they have to be redrawn as the view turns; quantizing the angle
# is what keeps that from rebuilding the whole figure on every frame of a drag.
WIRE_AZIMUTH_STEP = 5.0
# The two penalization styles.  They are the same column in different ink, and
# they are independent: a single optimum can carry both at once, one toggled by
# each mode.
PENALTY_MODES = {"pen-solid": "solid", "pen-grey": "grey"}
PENALTY_COLORS = {"solid": "red", "grey": "rgb(190, 190, 190)"}
# Interaction modes that *freeze* the view: the camera is pinned and zoom is
# disabled so a click lands exactly where the surface is drawn.  Penalization is
# deliberately NOT here — it toggles a volume on an existing optimum (resolved
# back to its owner even through the column wall, see :func:`_cylinder_owner`),
# so it can stay fully navigable and show the solid, live-opacity cylinder while
# you orbit and zoom around it.
LOCKED_MODES = ("optima", "line")
# Draw order, outermost first.  Two columns pinned on the same optimum would be
# built to exactly the same wall, caps and rings, and coincident translucent
# surfaces tear into stripes as the depth test picks a winner pixel by pixel.
# Each style after the first is therefore nested a hair inside the one before
# it — ``PENALTY_NEST`` of the radius and of the height at each end — so the red
# column stands imperceptibly wider and taller than the grey one and the two
# never share a surface.  ``PENALTY_NEST_LIFT`` does the same for the landscape
# contours, which are separated in height instead: their radii are already too
# close to pull apart on the surface.
PENALTY_STACK = ("solid", "grey")
PENALTY_NEST = 4e-3
PENALTY_NEST_LIFT = 0.35
# Where the camera sits until the user has rotated the height map themselves.
DESIGN_CAMERA = {"eye": {"x": 1.25, "y": 1.25, "z": 1.0},
                 "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                 "up": {"x": 0.0, "y": 0.0, "z": 1.0}}
# Client-side PNG export via the plot's camera button.  A PNG carries no
# intrinsic DPI, so a "300 dpi" figure is bought with pixel count: the figure is
# laid out at roughly SCREEN_DPI on screen, so rendering every dimension scaled
# by EXPORT_DPI / SCREEN_DPI hands the browser a raster dense enough to print at
# 300 dpi at the same physical size (950x850 -> ~2969x2656 px).
EXPORT_DPI = 300.0
SCREEN_DPI = 96.0
EXPORT_SCALE = EXPORT_DPI / SCREEN_DPI


def _colorscale_stops(name, reverse=False, n=32):
    """A named plotly colour scale as an explicit ``[[frac, rgb], ...]`` list.

    Sampled to explicit stops (rather than passed by name) so the same list
    feeds :class:`plotly.graph_objects.Surface`, the printed-line markers *and*
    the sub-threshold blackout in :func:`_threshold_colorscale` without any of
    them having to re-resolve the name — and so a diverging scale can be handed
    back *reversed*, which the bare name cannot express.
    """
    fr = np.linspace(0.0, 1.0, int(n))
    cols = sample_colorscale(name, [float(f) for f in fr])
    if reverse:
        cols = cols[::-1]
    return [[float(f), c] for f, c in zip(fr, cols)]


# Colour scales offered in the UI.  "Loss (teal–brown)" is the diverging BrBG
# ramp reversed so it reads like the reference loss landscape — deep teal at the
# low (good) end, white through the middle, brown at the high (bad) end; Viridis
# is the original yellow-high / blue-low objective ramp.  Each value is an
# explicit stop list (see :func:`_colorscale_stops`).
COLORSCALES = {
    "Viridis": _colorscale_stops("Viridis"),
    "Loss (teal–brown)": _colorscale_stops("BrBG", reverse=True),
    "Terrain (Earth)": _colorscale_stops("Earth"),
    "Blue–white–red": _colorscale_stops("RdBu", reverse=True),
    "Cividis": _colorscale_stops("Cividis"),
}
DEFAULT_COLORSCALE = "Viridis"


# ── Design-studio objective ──────────────────────────────────────────────────

class DesignEnsemble(CartesianEnsemble):
    """The unit-square objective with per-optimum peak heights.

    :class:`~synthetic_data.ensemble.Ensemble` puts every basin at the same raw
    peak, ``_PEAK``, one span above the neutral level, and normalizes the whole
    raw range by its own largest magnitude.  Here each basin instead peaks at
    ``gain * height_i * _PEAK``, so raising ``gain`` lifts the optima higher —
    the normalization divides by the taller peak, which squashes the Perlin
    noise toward the middle of the colour scale — and ``height_i`` sets how tall
    each individual optimum stands within that.

    The roughness is *added* onto the optima rather than max-combined beside
    them (see :meth:`_predict_raw`), so a placed hill carries the same Perlin
    texture as the flats around it and blends into the landscape instead of
    reading as a glassy smooth dome sitting on top of it.

    ``heights`` lines up with the *pinned* optima, which the base class prepends
    to the sampled ones, so it indexes the design studio's placed points in
    click order; anything unlisted keeps a height of 1.
    """

    def __init__(self, *args, optima_gain=1.0, optima_heights=None, **kwargs):
        # Set before ``super().__init__``: it estimates the raw range at the end
        # of construction, which runs straight through ``_true_field`` below.
        self.optima_gain = max(0.0, float(optima_gain))
        self.optima_heights = np.asarray(optima_heights if optima_heights is not None
                                         else [], dtype=float).ravel()
        super().__init__(*args, **kwargs)

    @property
    def peak_heights(self) -> np.ndarray:
        """Per-basin multiplier on ``_PEAK``, one per row of ``peak_centers``."""
        h = np.ones(len(self.peak_centers))
        k = min(len(h), len(self.optima_heights))
        h[:k] = self.optima_heights[:k]
        return self.optima_gain * h

    def _true_field(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if not len(self.peak_centers):
            return np.full(X.shape[0], _BASE)
        heights = self.peak_heights
        # Each basin rises from the same floor to its own peak, and the tallest
        # one wins wherever two overlap — so a short optimum next to a tall one
        # reads as a shoulder on it rather than punching a dent in it.
        return np.stack(
            [_BASE + (h * _PEAK - _BASE)
             * _negated_ackley_env01(X, c, self.basin_width, self.axis_scale,
                                     self.basin_smoothing)
             for c, h in zip(self.peak_centers, heights)],
            axis=0,
        ).max(axis=0)

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Roughness laid *over* the optima, not max-combined beside them.

        The base class combines the smooth optima field with the background by
        ``max(true, min(background, cap))``: a placed optimum's summit rises
        above the roughness, so ``max`` keeps the smooth hill there and the
        Perlin texture never reaches it — the optimum reads as a glassy dome on
        an otherwise rugged landscape, which is the "why is my optimum smooth?"
        the design studio kept hitting.  In this studio every other distractor
        is off (see :data:`ENSEMBLE`), so the background is pure roughness;
        *adding* it to the optima field lays that same roughness over the hills
        and the flats alike, so a placed optimum blends into the terrain around
        it.  The peak still towers by ``gain * height * _PEAK``, so it stays the
        tallest point and reads as an optimum — it just wears the texture now.
        """
        return self._true_field(X) + self._background_field(X)


# ── Slider helper ────────────────────────────────────────────────────────────

def _slider(label, sid, lo, hi, step, value, marks_every, fmt=str):
    n_marks = int(round((hi - lo) / marks_every)) + 1
    marks = {}
    for i in range(n_marks):
        v = lo + i * marks_every
        v = round(v, 6)
        marks[v] = fmt(v)
    return html.Div([
        html.Label(label),
        dcc.Slider(id=sid, min=lo, max=hi, step=step, value=value, marks=marks,
                   tooltip={"placement": "bottom", "always_visible": True}),
    ], style={"padding": "8px"})


# ── Dash app ─────────────────────────────────────────────────────────────────

def build_app():
    global html, dcc  # used by the helper above after Dash import
    from dash import (  # noqa: F811
        ALL, Dash, Input, Output, State, callback, ctx, dcc, html, no_update,
    )

    app = Dash(__name__)

    controls = html.Div([
        html.Div([
            html.Div([
                html.Label("Interaction"),
                dcc.RadioItems(
                    id="click-action",
                    options=[{"label": "  Rotate / inspect", "value": "rotate"},
                             {"label": "  Place optima", "value": "optima"},
                             {"label": "  Draw printed line", "value": "line"},
                             {"label": "  Red penalization volume",
                              "value": "pen-solid"},
                             {"label": "  Grey penalization volume",
                              "value": "pen-grey"}],
                    value="rotate", labelStyle={"display": "block"}),
            ], style={"paddingTop": "6px"}),
            html.Div(id="place-hint",
                     style={"fontSize": "13px", "color": "#555",
                            "paddingTop": "6px"}),
            _slider("Printed Line Samples", "line-samples", 10, 100, 1, 25, 30,
                    str),
            _slider("Printed Line Point Size", "line-size", 2, 20, 1, 5, 3, str),
            _slider("Printed Line Point Height Above Surface", "line-lift",
                    0.5, 20.0, 0.5, 3.0, 5.0, lambda v: f"{v:.0f}"),
            _slider("Printed Line Outline Width", "line-outline",
                    0.0, 6.0, 0.25, 1.5, 1.0, lambda v: f"{v:.1f}"),
            _slider("Floor Projection Opacity", "line-floor-opacity",
                    0.0, 1.0, 0.05, 1.0, 0.25, lambda v: f"{v:.2f}"),
            _slider("Connector Line Width", "line-drop-width",
                    0.0, 10.0, 0.5, 1.5, 2.0, lambda v: f"{v:.1f}"),
            _slider("Connector Line Opacity", "line-drop-opacity",
                    0.0, 1.0, 0.05, 0.6, 0.25, lambda v: f"{v:.2f}"),
            _slider("Background Opacity", "bg-opacity", 0.1, 1.0, 0.05, 1.0,
                    0.1, lambda v: f"{v:.1f}"),
            _slider("Peak Smoothing (rounds the tip of every basin)",
                    "basin-smooth", 0.0, 6.0, 0.1, 1.5, 1.0,
                    lambda v: f"{v:.0f}"),
            _slider("Optima Height (peak height above the background)",
                    "optima-gain", 1.0, 30.0, 0.5, 1.0, 5.0,
                    lambda v: f"{v:.0f}"),
            html.Div([
                html.Label("Individual Optimum Heights "
                           "(× the Optima Height above)"),
                html.Div(id="optima-height-list"),
            ], style={"padding": "8px"}),
            # ── Landscape (HD roughness) ──────────────────────────────────────
            # The fixed background is Perlin roughness; turning the mesh finer
            # and the noise deeper / more octaved is what gives the sharp,
            # fractal, HD look of a real loss landscape.
            html.Div(html.Label("Landscape (HD roughness)",
                                style={"fontWeight": "bold"}),
                     style={"padding": "12px 8px 0"}),
            _slider("Mesh Resolution (higher = sharper / HD, slower)",
                    "grid-n", 80, 400, 20, GRID_N, 80, lambda v: f"{v:.0f}"),
            _slider("Roughness Amplitude (height of the noise)", "noise-amp",
                    0, 2000, 50, ENSEMBLE["noise_amp"], 500,
                    lambda v: f"{v:.0f}"),
            _slider("Roughness Detail (noise octaves — fractal sharpness)",
                    "noise-octaves", 1, 6, 1, ENSEMBLE["noise_octaves"], 1,
                    lambda v: f"{v:.0f}"),
            _slider("Feature Scale (noise frequency — finer = busier)",
                    "noise-freq", 0, 40, 1, ENSEMBLE["noise_freq"], 8,
                    lambda v: f"{v:.0f}"),
            # ── Colour scale ──────────────────────────────────────────────────
            html.Div([
                html.Label("Colour Scale"),
                dcc.Dropdown(id="colorscale",
                             options=[{"label": k, "value": k}
                                      for k in COLORSCALES],
                             value=DEFAULT_COLORSCALE, clearable=False),
            ], style={"padding": "8px"}),
            # The two cut-offs are dialled in by hand instead of a fixed
            # percentile: the floor and the ceiling of the colour range are each
            # a percentile of the objective's own distribution, so exactly which
            # span the colours spend themselves on is set with these sliders.
            _slider("Colour Scale Floor (percentile of objective)",
                    "color-lo-pct", 0, 50, 1, 0, 10, lambda v: f"{v:.0f}"),
            _slider("Colour Scale Ceiling (percentile of objective)",
                    "color-hi-pct", 50, 100, 1, 100, 10, lambda v: f"{v:.0f}"),
            # ── Surface look ──────────────────────────────────────────────────
            # Directional shading (relief) so peaks read from the side instead
            # of blending into the colour ramp, plus an optional black contour
            # grid draped over the surface to outline the terrain.
            html.Div(html.Label("Surface look", style={"fontWeight": "bold"}),
                     style={"padding": "12px 8px 0"}),
            _slider("Surface Shading (relief — makes peaks pop laterally)",
                    "surface-shading", 0.0, 1.0, 0.05, 0.6, 0.25,
                    lambda v: f"{v:.2f}"),
            html.Div(dcc.Checklist(
                id="surface-toggles",
                options=[{"label": "  Black edges on the landscape "
                                   "(contour grid draped on the surface)",
                          "value": "edges"}],
                value=[], labelStyle={"display": "block"},
                style={"paddingTop": "4px"}),
                style={"padding": "8px"}),
            _slider("Black Edge Spacing (smaller = denser grid)",
                    "edge-spacing", 0.02, 0.25, 0.01, 0.05, 0.05,
                    lambda v: f"{v:.2f}"),
            html.Div(dcc.Checklist(
                id="design-toggles",
                options=[{"label": "  Wireframe edges on penalization cylinders, "
                                   "plus a contour where they cut the landscape",
                          "value": "cage"}],
                value=["cage"], labelStyle={"display": "block"},
                style={"paddingTop": "4px"}),
                style={"padding": "8px"}),
            # ── Saved-figure appearance ───────────────────────────────────────
            # These ride on the *live* figure, so what is on screen is what the
            # camera button saves.  A transparent background drops the figure
            # onto a slide or poster cleanly; hiding the grid strips the axes,
            # zero lines and the shaded background planes (the floor included)
            # for a clean floating height map.
            html.Div([
                html.Label("Saved figure (camera icon on the plot toolbar)",
                           style={"fontWeight": "bold"}),
                dcc.Checklist(
                    id="export-toggles",
                    options=[{"label": "  Transparent background",
                              "value": "transparent"},
                             {"label": "  Hide grid, axes & background planes",
                              "value": "nogrid"}],
                    value=["transparent"], labelStyle={"display": "block"},
                    style={"paddingTop": "4px"}),
                html.Div("Saved as a 300 dpi PNG.",
                         style={"fontSize": "12px", "color": "#555",
                                "paddingTop": "2px"}),
            ], style={"padding": "12px 8px 4px"}),
            _slider("Penalization Wireframe Thickness", "pen-wire-width",
                    0.5, 12.0, 0.5, 3.0, 2.5, lambda v: f"{v:.1f}"),
            # A radius and an opacity per colour: the two columns are
            # independent regions that happen to share a shape, so sizing one
            # must not drag the other with it.  All four show together in either
            # penalization mode, so a column can be tuned against the one it is
            # sitting inside without switching modes to reach its slider.
            html.Div(_slider("Red Penalization Radius", "pen-radius-solid",
                             0.02, 0.5, 0.01, 0.15, 0.12, lambda v: f"{v:.2f}"),
                     id="pen-radius-solid-wrap", style={"display": "none"}),
            html.Div(_slider("Red Penalization Cylinder Opacity",
                             "pen-opacity-solid",
                             0.0, 1.0, 0.05, PENALTY_ALPHA, 0.25,
                             lambda v: f"{v:.2f}"),
                     id="pen-opacity-solid-wrap", style={"display": "none"}),
            html.Div(_slider("Grey Penalization Radius", "pen-radius-grey",
                             0.02, 0.5, 0.01, 0.15, 0.12, lambda v: f"{v:.2f}"),
                     id="pen-radius-grey-wrap", style={"display": "none"}),
            html.Div(_slider("Grey Penalization Cylinder Opacity",
                             "pen-opacity-grey",
                             0.0, 1.0, 0.05, PENALTY_ALPHA, 0.25,
                             lambda v: f"{v:.2f}"),
                     id="pen-opacity-grey-wrap", style={"display": "none"}),
        ], style={"padding": "8px"}),

        html.Div([
            html.Button("Clear Optima", id="clear-optima-btn", n_clicks=0,
                        style={"fontSize": "16px", "padding": "8px 24px",
                               "margin": "12px 8px", "cursor": "pointer"}),
            html.Button("Clear Lines", id="clear-lines-btn", n_clicks=0,
                        style={"fontSize": "16px", "padding": "8px 24px",
                               "margin": "12px 8px", "cursor": "pointer"}),
        ], style={"textAlign": "center"}),
    ], style={"width": "60%", "margin": "0 auto"})

    app.layout = html.Div([
        html.H2("Figure Maker — Design Studio (2D square height map)",
                style={"textAlign": "center"}),
        controls,
        dcc.Graph(id="cloud-plot", style={"height": f"{FIG_H}px"}),
        # Points pinned by clicking; kept out of the figure so they survive
        # redraws.
        dcc.Store(id="placed-optima", data=[]),
        # Per-optimum height multipliers, parallel to ``placed-optima``.
        dcc.Store(id="optima-heights", data=[]),
        # Penalization regions pinned onto placed optima, as
        # ``{"xy": [x, y], "kind": "solid"|"grey"}``.  One entry per colour, so
        # an optimum wearing both appears twice.
        dcc.Store(id="penalized", data=[]),
        # Finished printed lines as [[start_xy], [end_xy]] pairs, plus the first
        # endpoint of a line still being drawn (None between lines).
        dcc.Store(id="print-lines", data=[]),
        dcc.Store(id="line-start", data=None),
        # Orientation the height map was left in while free to rotate; placing
        # modes pin the camera back to it.
        dcc.Store(id="design-camera", data=None),
        # The same orientation quantized to WIRE_AZIMUTH_STEP, which is what the
        # cylinder silhouettes are drawn against.  Separate from the camera
        # itself because the figure has to be *rebuilt* when it changes, and a
        # rebuild per frame of an orbit would be unusable.
        dcc.Store(id="wire-azimuth", data=None),
    ])

    @callback(
        Output("design-camera", "data"),
        Output("wire-azimuth", "data"),
        Input("cloud-plot", "relayoutData"),
        State("click-action", "value"),
        State("wire-azimuth", "data"),
        prevent_initial_call=True,
    )
    def remember_camera(relayout, action, azimuth):
        # Tracked whenever the view is free to move — rotate *and* the
        # penalization modes, which now orbit freely too — so the orientation a
        # locked placing mode (optima / line) later pins to is always the latest
        # one the user left.  A stray drag during a locked mode must not move the
        # goalposts, so those are skipped.
        if action in LOCKED_MODES:
            return no_update, no_update
        camera = (relayout or {}).get("scene.camera")
        if not camera:
            return no_update, no_update
        # The camera itself is stored at full precision — it is what placing
        # pins the view to.  The silhouette angle is stored rounded and only
        # when the rounding actually moved, so orbiting rebuilds the figure once
        # every WIRE_AZIMUTH_STEP degrees instead of once every frame.
        step = WIRE_AZIMUTH_STEP
        rounded = float(round(_view_azimuth(camera) / step) * step)
        return camera, (no_update if rounded == azimuth else rounded)

    @callback(
        Output("cloud-plot", "config"),
        Output("pen-radius-solid-wrap", "style"),
        Output("pen-opacity-solid-wrap", "style"),
        Output("pen-radius-grey-wrap", "style"),
        Output("pen-opacity-grey-wrap", "style"),
        Input("click-action", "value"),
    )
    def toggle_mode_controls(action):
        show, hide = {}, {"display": "none"}
        # The camera button downloads the figure as a 300 dpi PNG: no intrinsic
        # DPI on a PNG, so the density is bought by rendering every dimension
        # scaled up (see EXPORT_SCALE).  Transparency and grid come from the
        # figure itself (paper_bgcolor / the axes), not from here.
        cfg = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": "figure_maker",
                "width": FIG_W,
                "height": FIG_H,
                "scale": EXPORT_SCALE,
            },
            "displaylogo": False,
        }
        # Only the locked placing modes (optima / line) freeze the view; rotate
        # and the penalization modes keep scroll-zoom and double-click reset so
        # you can pan, orbit and zoom around the columns while you place them.
        if action in LOCKED_MODES:
            cfg["scrollZoom"] = False
            cfg["doubleClick"] = False
        # The per-colour radius and opacity sliders belong to the two
        # penalization modes; everything else here is on screen always.
        return (cfg, *((show if action in PENALTY_MODES else hide,) * 4))

    @callback(
        Output("place-hint", "children"),
        Input("click-action", "value"),
    )
    def place_hint_text(action):
        if action == "line":
            return ("The view is locked to the orientation you left it in.  "
                    "Click the start and end of a printed line: both endpoints "
                    "snap to the nearest side of the square, and the finished "
                    "line is sampled along its length, coloured by the objective "
                    "and laid on the surface.")
        if action == "optima":
            return ("The view is locked to the orientation you left it in.  "
                    "Click the surface to pin an optimum there; click the peak "
                    "of a pinned one again to remove it.")
        if action in PENALTY_MODES:
            colour = "red" if action == "pen-solid" else "light grey"
            shape = (f"a single solid translucent {colour} cylinder spanning "
                     f"the height of the plot, at the {colour} radius and tint "
                     f"you set with the sliders below")
            return (f"Drag to orbit, scroll to zoom, right-drag to pan — the "
                    f"view stays free here.  Click the peak of a placed optimum "
                    f"to wrap it in {shape}, shown solid so the opacity slider "
                    f"updates it live.  Clicking it "
                    f"again in this mode clears it; the other penalization "
                    f"mode is independent, so an optimum can carry both "
                    f"colours at once.  A click anywhere on an existing "
                    f"column counts as its own optimum, so a column can be "
                    f"clicked back off without hunting for the summit inside it.")
        return ("Drag to orbit the height map, scroll to zoom.  Switch to a "
                "placing mode to lock the view and click features in.")

    @callback(
        Output("placed-optima", "data"),
        Output("print-lines", "data"),
        Output("line-start", "data"),
        Output("penalized", "data"),
        Output("optima-heights", "data"),
        Input("cloud-plot", "clickData"),
        Input("clear-optima-btn", "n_clicks"),
        Input("clear-lines-btn", "n_clicks"),
        State("placed-optima", "data"),
        State("print-lines", "data"),
        State("line-start", "data"),
        State("penalized", "data"),
        State("optima-heights", "data"),
        State("click-action", "value"),
        State("pen-radius-solid", "value"),
        State("pen-radius-grey", "value"),
        prevent_initial_call=True,
    )
    def edit_placed_items(click_data, _clear_optima, _clear_lines,
                          placed, lines, start, penalized, heights, action,
                          pen_r_solid, pen_r_grey):
        keep = (no_update,) * 5
        if ctx.triggered_id == "clear-optima-btn":
            # Penalization is pinned *on* an optimum, so it goes with it, and so
            # does the optimum's own height.
            return [], no_update, no_update, [], []
        if ctx.triggered_id == "clear-lines-btn":
            # A half-drawn line goes with the finished ones, otherwise the next
            # click would silently close a line against a stale start point.
            return no_update, [], None, no_update, no_update
        if action == "rotate":
            return keep
        xy = _click_xy(click_data)
        if xy is None:
            return keep

        if action == "line":
            # Printed lines run between two boundary points, so both endpoints
            # are pulled onto the edge of the square.  The first click parks a
            # start point; the second closes the line and re-arms for the next.
            end = [float(v) for v in _snap_to_square_edge(xy)]
            if start is None:
                return no_update, no_update, end, no_update, no_update
            return (no_update, [*(lines or []), [list(start), end]], None,
                    no_update, no_update)

        placed = [list(p) for p in (placed or [])]
        # Heights are parallel to ``placed``; a list that has fallen short (a
        # point placed before this store existed) reads as the neutral 1.
        heights = [float(h) for h in (heights or [])]
        heights += [1.0] * (len(placed) - len(heights))
        del heights[len(placed):]
        # A click within about one grid cell of a pinned optimum is a click *on*
        # that optimum, which is what both placing and penalizing act on.
        tol = max(0.02, 1.5 / GRID_N)
        hit = next((i for i, p in enumerate(placed)
                    if float(np.linalg.norm(np.asarray(p, dtype=float) - xy)) <= tol),
                   None)
        if hit is None:
            # A click that landed on a cylinder wall reports the wall, not the
            # optimum underneath it, so resolve it back to the column's owner.
            # This is also what stops a click on a column from being read as bare
            # landscape and dropping a stray optimum onto its side.
            hit = _cylinder_owner(xy, placed, penalized,
                                  {"solid": float(pen_r_solid or 0.0),
                                   "grey": float(pen_r_grey or 0.0)})

        if action in PENALTY_MODES:
            # Penalization only ever attaches to an optimum that is already
            # placed — a click on bare landscape does nothing here.
            if hit is None:
                return keep
            kind = PENALTY_MODES[action]
            pen = [dict(e) for e in (penalized or [])]
            # Each colour is toggled on its own: a mode only ever adds or
            # removes its own entry, so the other colour on the same optimum is
            # left exactly as it was.
            for i, e in enumerate(pen):
                if e.get("kind") == kind and _same_point(e["xy"], placed[hit]):
                    pen.pop(i)
                    return no_update, no_update, no_update, pen, no_update
            pen.append({"xy": list(placed[hit]), "kind": kind})
            return no_update, no_update, no_update, pen, no_update

        # Placing: the same gesture places and un-places.  A removed optimum
        # takes its own height slider with it, so the ones left keep theirs
        # rather than every slider below the gap sliding up one place.
        if hit is not None:
            gone = placed.pop(hit)
            heights.pop(hit)
            pen = [e for e in (penalized or []) if not _same_point(e["xy"], gone)]
            return placed, no_update, no_update, pen, heights
        placed.append([float(v) for v in xy])
        return placed, no_update, no_update, no_update, [*heights, 1.0]

    @callback(
        Output("optima-height-list", "children"),
        Input("placed-optima", "data"),
        State("optima-heights", "data"),
    )
    def build_height_sliders(placed, heights):
        """One slider per placed optimum, rebuilt whenever the set of them
        changes.

        Driven by ``placed-optima`` alone — the *structure* — so dragging a
        slider does not tear its own row out from under the mouse; the values
        come from the store as state, which is what carries a height across the
        rebuild that placing the next optimum triggers.
        """
        placed = placed or []
        heights = list(heights or [])
        if not placed:
            return html.Div("Place an optimum to give it its own height.",
                            style={"fontSize": "13px", "color": "#555"})
        rows = []
        for i, pt in enumerate(placed):
            x, y = (float(v) for v in pt)
            value = float(heights[i]) if i < len(heights) else 1.0
            rows.append(html.Div([
                html.Label(f"#{i + 1} at ({x:.2f}, {y:.2f})",
                           style={"fontSize": "13px"}),
                dcc.Slider(id={"type": "optimum-height", "index": i},
                           min=0.05, max=3.0, step=0.05, value=value,
                           marks={0.05: "0.05", 1: "1", 2: "2", 3: "3"},
                           tooltip={"placement": "bottom",
                                    "always_visible": True}),
            ], style={"padding": "4px 0"}))
        return rows

    @callback(
        Output("optima-heights", "data", allow_duplicate=True),
        Input({"type": "optimum-height", "index": ALL}, "value"),
        prevent_initial_call=True,
    )
    def store_heights(values):
        # The sliders are the live control, the store is what survives their
        # rebuild — and what the figure reads, so it stays the single source of
        # truth even in the moment between a click and the rebuilt slider row.
        return [1.0 if v is None else float(v) for v in (values or [])]

    @callback(
        Output("cloud-plot", "figure"),
        Input("placed-optima", "data"),
        Input("print-lines", "data"),
        Input("line-start", "data"),
        Input("line-samples", "value"),
        Input("line-size", "value"),
        Input("line-lift", "value"),
        Input("line-outline", "value"),
        Input("line-floor-opacity", "value"),
        Input("line-drop-width", "value"),
        Input("line-drop-opacity", "value"),
        Input("bg-opacity", "value"),
        Input("basin-smooth", "value"),
        Input("penalized", "data"),
        Input("pen-radius-solid", "value"),
        Input("pen-opacity-solid", "value"),
        Input("pen-radius-grey", "value"),
        Input("pen-opacity-grey", "value"),
        Input("pen-wire-width", "value"),
        Input("wire-azimuth", "data"),
        Input("optima-gain", "value"),
        Input("optima-heights", "data"),
        Input("design-toggles", "value"),
        Input("colorscale", "value"),
        Input("color-lo-pct", "value"),
        Input("color-hi-pct", "value"),
        Input("grid-n", "value"),
        Input("noise-amp", "value"),
        Input("noise-octaves", "value"),
        Input("noise-freq", "value"),
        Input("surface-shading", "value"),
        Input("surface-toggles", "value"),
        Input("edge-spacing", "value"),
        Input("export-toggles", "value"),
        Input("click-action", "value"),
        State("design-camera", "data"),
    )
    def update_plot(placed, print_lines, line_start, line_samples, line_size,
                    line_lift, line_outline, line_floor_opacity,
                    line_drop_width, line_drop_opacity, bg_opacity,
                    basin_smooth, penalized, pen_r_solid, pen_a_solid,
                    pen_r_grey, pen_a_grey, pen_wire_width, wire_azimuth,
                    optima_gain, optima_heights, design_toggles, colorscale,
                    color_lo_pct, color_hi_pct, grid_n, noise_amp,
                    noise_octaves, noise_freq, surface_shading, surface_toggles,
                    edge_spacing, export_toggles, action, camera):
        placed_arr = (np.asarray(placed, dtype=float).reshape(-1, 2)
                      if placed else np.empty((0, 2)))
        # Printed lines are an overlay only — they never feed the objective.
        lines = print_lines or []
        # The roughness knobs override the fixed ENSEMBLE background so the mesh
        # can be driven up to the sharp, fractal, HD look of a real landscape.
        ens = {**ENSEMBLE,
               "noise_amp": float(noise_amp),
               "noise_octaves": int(noise_octaves),
               "noise_freq": float(noise_freq)}
        fn = DesignEnsemble(
            dim=2,
            optima_gain=float(optima_gain),
            optima_heights=[float(h) for h in (optima_heights or [])],
            basin_smoothing=float(basin_smooth),
            pinned_optima=placed_arr if len(placed_arr) else None,
            **ens,
        )
        pen = penalized or []
        title = (f"Ensemble — {len(placed_arr)} placed, {len(lines)} printed, "
                 f"{len(pen)} penalized")
        toggles = design_toggles or []
        exports = export_toggles or []
        stops = COLORSCALES.get(colorscale, COLORSCALES[DEFAULT_COLORSCALE])
        return _design_figure(fn, int(grid_n), BASIN_THRESHOLD, title,
                              placed_arr, lines, line_start, int(line_samples),
                              float(bg_opacity), action in LOCKED_MODES, camera,
                              penalized=pen,
                              penalty_radius={"solid": float(pen_r_solid),
                                              "grey": float(pen_r_grey)},
                              colorscale=stops,
                              color_lo_pct=float(color_lo_pct),
                              color_hi_pct=float(color_hi_pct),
                              shading=float(surface_shading),
                              black_edges="edges" in (surface_toggles or []),
                              edge_spacing=float(edge_spacing),
                              transparent="transparent" in exports,
                              hide_grid="nogrid" in exports,
                              cylinder_edges="cage" in toggles,
                              wire_width=float(pen_wire_width),
                              solid_opacity={"solid": float(pen_a_solid),
                                             "grey": float(pen_a_grey)},
                              # The stored angle is the rotated view; before
                              # the first orbit there is none, so fall back
                              # to whatever camera the figure is drawn with.
                              view_azimuth=(float(wire_azimuth)
                                            if wire_azimuth is not None
                                            else _view_azimuth(camera)),
                              line_size=float(line_size),
                              line_lift=float(line_lift),
                              line_outline=float(line_outline),
                              line_floor_opacity=float(line_floor_opacity),
                              line_drop_width=float(line_drop_width),
                              line_drop_opacity=float(line_drop_opacity))

    return app


# ── Design-studio geometry ───────────────────────────────────────────────────

def build_square_grid(grid_n):
    """Regular lattice over the unit square.

    Returns ``(axis, shape, points)``: the shared 1-D axis ticks, the ``(ny, nx)``
    shape a per-point value must be reshaped to for :class:`plotly.graph_objects.Surface`,
    and the ``(N, 2)`` points themselves in that same row-major order.
    """
    axis = np.linspace(0.0, 1.0, int(grid_n) + 1)
    xx, yy = np.meshgrid(axis, axis)  # "xy" indexing: shape (ny, nx)
    return axis, xx.shape, np.column_stack([xx.ravel(), yy.ravel()])


def _snap_to_square_edge(xy):
    """Nearest point on the boundary of the unit square to ``xy``.

    A printed line runs between two boundary points, so an endpoint clicked in
    the interior is pushed out to whichever of the four sides is closest.
    """
    x, y = (float(v) for v in np.clip(np.asarray(xy, dtype=float), 0.0, 1.0))
    sides = ((x, (0.0, y)), (1.0 - x, (1.0, y)),
             (y, (x, 0.0)), (1.0 - y, (x, 1.0)))
    return np.asarray(min(sides, key=lambda s: s[0])[1], dtype=float)


def _line_samples(lines, n_samples):
    """Evenly spaced points along every printed line, stacked ``(M, 2)``."""
    if not lines:
        return np.empty((0, 2))
    t = np.linspace(0.0, 1.0, max(int(n_samples), 2))[:, None]
    segs = []
    for start, end in lines:
        a = np.asarray(start, dtype=float)
        b = np.asarray(end, dtype=float)
        segs.append(a + t * (b - a))
    return np.vstack(segs)


def _click_xy(click_data):
    """``(x, y)`` of the clicked point, or ``None`` if unavailable.

    Every trace lives in the same square, so the click resolves the same way
    whether it landed on the surface or on an overlay marker sitting above it.
    """
    points = (click_data or {}).get("points") or []
    if not points:
        return None
    p = points[0]
    if "x" not in p or "y" not in p:
        return None
    xy = np.array([p["x"], p["y"]], dtype=float)
    if not np.isfinite(xy).all():
        return None
    return np.clip(xy, 0.0, 1.0)


# ── Penalization volumes ─────────────────────────────────────────────────────

def _same_point(a, b, tol=1e-9):
    """Whether two stored points are the same pinned optimum."""
    return float(np.linalg.norm(np.asarray(a, dtype=float)
                                - np.asarray(b, dtype=float))) <= tol


def _cylinder_owner(xy, placed, penalized, radii):
    """Index in ``placed`` of the cylinder the click landed on, or ``None``.

    A cylinder is a *volume*: the click resolves to a point on its wall or cap,
    which is up to its own radius away from the optimum the column stands on, so
    the usual "within a grid cell of an optimum" test never matches it.  Any
    click inside the footprint of a column is therefore read as a click on that
    column's optimum — the nearest one, if two overlap.  Both colours are
    volumes in this sense, so both are resolved here, each against the radius
    ``radii`` gives its own kind.
    """
    xy = np.asarray(xy, dtype=float)
    best, best_d = None, float("inf")
    for entry in penalized or []:
        if entry.get("kind") not in PENALTY_COLORS:
            continue
        # A click on the rim sits at exactly the radius, which floating point
        # puts on either side of the cut, so the footprint is widened by a hair.
        reach = _by_kind(radii, entry["kind"], 0.0) * (1.0 + 1e-6)
        center = np.asarray(entry["xy"], dtype=float)
        d = float(np.linalg.norm(xy - center))
        if d > reach or d >= best_d:
            continue
        for i, p in enumerate(placed):
            if _same_point(p, center):
                best, best_d = i, d
                break
    return best


def _by_kind(value, kind, default):
    """One penalization style's share of a per-colour setting.

    The radius and the opacity are set per colour, so they travel as
    ``{kind: value}`` mappings; a bare number is still accepted and read as the
    same value for every colour.
    """
    if isinstance(value, dict):
        value = value.get(kind, default)
    return float(default if value is None else value)


def _penalty_kinds(point, penalized, tol=1e-9):
    """Which penalization colours are pinned on ``point``, in draw order.

    An optimum can wear both at once, so this is a set rather than a single
    style; anything not in :data:`PENALTY_COLORS` is ignored, which is what
    keeps a store left over from an older layout from being drawn.
    """
    kinds = {entry["kind"] for entry in penalized or []
             if entry.get("kind") in PENALTY_COLORS
             and _same_point(entry["xy"], point, tol)}
    return [k for k in PENALTY_STACK if k in kinds]


def _penalty_nesting(kind, radius, z_lo, z_hi, lift):
    """Geometry the column of ``kind`` is built to, nested by its draw order.

    The first style in :data:`PENALTY_STACK` gets the radius and the height
    exactly as asked; each one after it is drawn a :data:`PENALTY_NEST` fraction
    narrower and shorter, so two columns on the same optimum never share a wall
    or a cap to fight over.  Their landscape contours are near enough in radius
    to still land on the same pixels, so those are separated in *height*
    instead: the outer column's contour rides the full overlay lift, the ones
    inside it proportionally less.
    """
    k = PENALTY_STACK.index(kind)
    pad = PENALTY_NEST * (float(z_hi) - float(z_lo)) * k
    return (float(radius) * (1.0 - PENALTY_NEST * k),
            float(z_lo) + pad, float(z_hi) - pad,
            float(lift) * (1.0 - PENALTY_NEST_LIFT * k))


def _view_azimuth(camera):
    """Compass bearing the scene camera looks from, in degrees.

    Only the horizontal part of the eye-to-target vector matters, and the scene
    maps x and y identically (both unit aspect over ``[0, 1]``), so the angle in
    camera coordinates is the angle in the square — no unscaling needed.
    """
    cam = camera or DESIGN_CAMERA
    eye = cam.get("eye") or DESIGN_CAMERA["eye"]
    center = cam.get("center") or {"x": 0.0, "y": 0.0}
    dx = float(eye.get("x", 0.0)) - float(center.get("x", 0.0))
    dy = float(eye.get("y", 0.0)) - float(center.get("y", 0.0))
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return 45.0
    return float(np.degrees(np.arctan2(dy, dx)))


def _cylinder_cage_trace(center, radius, z_lo, z_hi, azimuth=45.0, width=3.0,
                         verticals=True, opacity=PENALTY_ALPHA, color="red"):
    """The penalization column drawn as a cel-shaded outline.

    Only the lines a cartoonist would ink: the top and bottom circles, and the
    two vertical edges *of the silhouette* — the points on the circle where the
    wall turns away from the viewer, which is where the column reads as having a
    side at all.  Those two sit perpendicular to the direction the camera looks
    from, so they move as the view turns; ``azimuth`` is that direction (see
    :func:`_view_azimuth`), quantized upstream so orbiting does not redraw the
    figure continuously.  ``verticals=False`` drops them and leaves the two
    rings: the solid column already draws its own silhouette in tinted glass, so
    inking the edges over it only doubles a line that is there anyway.

    ``opacity`` is the stroke's own — a translucent line over a translucent wall
    composites twice and reads as a ragged, banded edge, so the solid column
    inks its rings at full opacity instead.

    This is also what a column is drawn as while a placing mode is live.  Plotly
    resolves a 3D click in the GPU's pick buffer and keeps a single topmost hit,
    so a solid volume standing over its own optimum swallows every click aimed at
    that optimum (``hoverinfo="skip"`` only discards the hit — it does not let
    the ray continue).  Four strokes occlude almost nothing, so the ray reaches
    the landscape underneath, and the wires themselves still resolve to the
    column's optimum via :func:`_cylinder_owner`.
    """
    center = np.asarray(center, dtype=float).ravel()
    radius = float(radius)
    if radius <= 0:
        return None
    theta = np.linspace(0.0, 2.0 * np.pi, 97)
    xs, ys, zs = [], [], []
    # ``None`` breaks the polyline so the rings and wires stay separate strokes
    # instead of being threaded together.
    for z in (float(z_lo), float(z_hi)):
        xs += [*(center[0] + radius * np.cos(theta)), None]
        ys += [*(center[1] + radius * np.sin(theta)), None]
        zs += [z] * len(theta) + [None]
    # The silhouette edges are a quarter turn either side of the view bearing.
    if verticals:
        for a in (np.radians(azimuth) + np.pi / 2.0,
                  np.radians(azimuth) - np.pi / 2.0):
            xs += [center[0] + radius * np.cos(a)] * 2 + [None]
            ys += [center[1] + radius * np.sin(a)] * 2 + [None]
            zs += [float(z_lo), float(z_hi), None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines", opacity=float(opacity),
        line=dict(color=color, width=float(width)),
        hoverinfo="name", showlegend=False, name="penalization",
    )


def _cylinder_contour_trace(fn, center, radius, lift, width=3.0,
                            opacity=PENALTY_ALPHA, color="red"):
    """The curve where a penalization cylinder cuts the landscape.

    The cylinder's wall is the circle of radius ``radius`` about ``center``
    extruded in z, so the intersection is that circle *lifted onto the height
    map* — a closed loop that rides up and over whatever the column is standing
    on, marking the penalized footprint on the surface itself rather than only
    in the air above it.  The arc outside the square is dropped (the column is
    clipped by the domain there, so there is no landscape to draw on).

    ``opacity`` is the stroke's own, for the same reason as in
    :func:`_cylinder_cage_trace`: under the solid column the loop is seen
    through the tinted wall, and a translucent stroke there composites twice.
    """
    center = np.asarray(center, dtype=float).ravel()
    radius = float(radius)
    if radius <= 0:
        return None
    theta = np.linspace(0.0, 2.0 * np.pi, PENALTY_CONTOUR_N)
    ring = np.column_stack([center[0] + radius * np.cos(theta),
                            center[1] + radius * np.sin(theta)])
    inside = ((ring >= 0.0) & (ring <= 1.0)).all(axis=1)
    if not inside.any():
        return None
    z = np.full(len(ring), np.nan)
    z[inside] = fn.predict(ring[inside]) + lift
    # ``None`` where the ring leaves the square, so the loop breaks there instead
    # of being closed across the gap by a chord.
    zs = [None if not ok else float(v) for ok, v in zip(inside, z)]
    return go.Scatter3d(
        x=ring[:, 0], y=ring[:, 1], z=zs, mode="lines",
        line=dict(color=color, width=float(width)),
        opacity=float(opacity), hoverinfo="name", showlegend=False,
        name="penalization",
    )


def _solid_cylinder_trace(center, radius, z_lo, z_hi, opacity=PENALTY_ALPHA,
                          color="red"):
    """A penalization region as one solid translucent cylinder.

    A closed :class:`plotly.graph_objects.Mesh3d` — wall plus both caps — of
    even ``color`` at ``opacity``, standing on ``center`` at exactly ``radius``
    and running the full height of the plot.  No falloff, so it reads as a
    single piece of tinted glass rather than as a gradient (its far wall still
    composites through the near one, which is what gives it any shading at all).
    Lit flat for the same reason: the tint is the whole point, and plotly's
    default lighting would rake it with a highlight.

    The caps are held :data:`PENALTY_SOLID_INSET` of the height inside
    ``z_lo``/``z_hi`` rather than laid on them — see that constant for why.
    """
    center = np.asarray(center, dtype=float).ravel()
    radius = float(radius)
    if radius <= 0:
        return None
    inset = PENALTY_SOLID_INSET * (float(z_hi) - float(z_lo))
    z_lo, z_hi = float(z_lo) + inset, float(z_hi) - inset
    n = PENALTY_SOLID_N
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    cx = center[0] + radius * np.cos(theta)
    cy = center[1] + radius * np.sin(theta)
    # Bottom ring, top ring, then the two cap centres the caps fan out from.
    x = np.concatenate([cx, cx, [center[0], center[0]]])
    y = np.concatenate([cy, cy, [center[1], center[1]]])
    z = np.concatenate([np.full(n, float(z_lo)), np.full(n, float(z_hi)),
                        [float(z_lo), float(z_hi)]])
    i0 = np.arange(n)
    i1 = (i0 + 1) % n
    # Two triangles per wall quad, plus one per cap wedge at each end.
    faces_i = np.concatenate([i0, i1, np.full(n, 2 * n), np.full(n, 2 * n + 1)])
    faces_j = np.concatenate([i1, n + i1, i0, n + i0])
    faces_k = np.concatenate([n + i0, n + i0, i1, n + i1])
    return go.Mesh3d(
        x=x, y=y, z=z, i=faces_i, j=faces_j, k=faces_k,
        color=color, opacity=float(opacity), flatshading=True,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
        # Not ``hoverinfo="skip"``: a skipped trace is dropped from the 3D pick
        # pass, and the column stands over its own optimum, so skipping it would
        # swallow the click that would take the region back off.
        hoverinfo="name", showlegend=False, name="penalization",
    )


def _threshold_colorscale(threshold, lo, hi, base="Viridis", n_stops=16):
    """``base`` colour scale with everything below ``threshold`` flattened to
    black.

    Matches the ternary view in ``synthetic_data/plot_ensemble.py``, where
    sub-threshold compositions are blacked out; a surface has no per-point
    colour override, so the cut is baked into the colourscale instead.  ``base``
    is the scale to cut into — a named plotly scale or an explicit stop list
    (see :func:`_colorscale_stops`) — and is returned untouched when the cut is
    off, which is the usual case here (``BASIN_THRESHOLD`` is 0).
    """
    if hi - lo < 1e-12 or not np.isfinite(threshold) or threshold <= lo:
        return base
    if threshold >= hi:
        return [[0.0, "black"], [1.0, "black"]]
    t = float((threshold - lo) / (hi - lo))
    fracs = np.linspace(0.0, 1.0, n_stops)
    colors = sample_colorscale(base, [float(f) for f in fracs])
    # The duplicated stop at ``t`` is what makes the cut a hard edge.
    return ([[0.0, "black"], [t, "black"]]
            + [[t + (1.0 - t) * float(f), c] for f, c in zip(fracs, colors)])


# ── Figure builder ───────────────────────────────────────────────────────────

def _design_figure(fn, grid_n, basin_threshold, title, placed=None, lines=None,
                   line_start=None, n_samples=25, bg_opacity=1.0, placing=False,
                   camera=None, penalized=None, penalty_radius=0.15,
                   colorscale="Viridis", color_lo_pct=0.0, color_hi_pct=100.0,
                   shading=0.6, black_edges=False, edge_spacing=0.05,
                   transparent=True, hide_grid=False, cylinder_edges=True,
                   wire_width=3.0, view_azimuth=45.0,
                   solid_opacity=PENALTY_ALPHA,
                   line_size=5.0, line_lift=5.0, line_outline=1.5,
                   line_floor_opacity=1.0,
                   line_drop_width=1.5, line_drop_opacity=0.6):
    """The unit-square objective as a rotatable 3D height map.

    Height *and* colour are both the objective, so the surface stands up in 3D
    while colour reads off ``colorscale`` (a named plotly scale or an explicit
    stop list — see :data:`COLORSCALES`).

    ``color_lo_pct`` / ``color_hi_pct`` are the colour scale's floor and ceiling
    as percentiles of the objective's own distribution, so exactly which span
    the colours are spent on is set by hand; an inverted or collapsed pick falls
    back to the full data range.  Only the *colours* move — heights are always
    the real objective.

    ``shading`` (0..1) drives the surface's directional lighting: 0 is flat
    (colour only), 1 is strong relief — low ambient, high diffuse — so peaks
    throw light and shadow and stand out when the height map is viewed from the
    side instead of blending into the colour ramp.  ``black_edges`` drapes a
    black contour grid over the surface at ``edge_spacing`` in x and y to outline
    the terrain.

    ``transparent`` gives the saved PNG a see-through paper background;
    ``hide_grid`` strips the grid lines, zero lines and the shaded background
    planes (the floor included) for a clean floating height map.

    ``penalty_radius`` and ``solid_opacity`` are per-colour: either a
    ``{kind: value}`` mapping or a single number standing for every colour (see
    :func:`_by_kind`).
    """
    axis, shape, pts = build_square_grid(grid_n)
    obj = fn.predict(pts)
    obj_min, obj_max = float(np.nanmin(obj)), float(np.nanmax(obj))
    span = max(obj_max - obj_min, 1e-6)
    lift = OVERLAY_LIFT * span
    # Colour limits: the floor and ceiling are each a percentile of the
    # objective's own values, dialled in with the two sliders, so the exact cut
    # is set by hand.  A collapsed or inverted pick (ceiling at or below floor)
    # falls back to the full data range rather than showing a flat colour.
    lo_p, hi_p = float(color_lo_pct), float(color_hi_pct)
    c_lo = float(np.nanpercentile(obj, np.clip(min(lo_p, hi_p), 0.0, 100.0)))
    c_hi = float(np.nanpercentile(obj, np.clip(max(lo_p, hi_p), 0.0, 100.0)))
    if c_hi - c_lo < 1e-9:
        c_lo, c_hi = obj_min, obj_max
    # Pinned rather than auto-ranged, so "top to bottom of the plot" is a height
    # the penalization cylinders can actually be built to.  The headroom above
    # the landscape is what makes a cylinder read at all: an optimum sits on a
    # summit, so without it the column would be buried inside its own hill.
    z_lo, z_hi = obj_min - 0.04 * span, obj_max + 0.3 * span

    # Directional lighting: ``shading`` fades ambient down and diffuse up, so a
    # higher value throws sharper light and shadow across slopes and the peaks
    # read from the side rather than washing into the colour ramp.  A little
    # specular on top gives the summits a catch-light as it climbs.
    s = float(np.clip(shading, 0.0, 1.0))
    lighting = dict(ambient=0.9 - 0.6 * s, diffuse=0.3 + 0.7 * s,
                    specular=0.04 + 0.16 * s, roughness=0.5, fresnel=0.2)
    # Optional black contour grid draped over the surface — lines at fixed x and
    # at fixed y, each riding the height map, so the terrain gets an inked mesh.
    contours = None
    if black_edges:
        step = float(np.clip(edge_spacing, 1e-3, 1.0))
        edge = lambda: dict(show=True, color="black", width=1.5, highlight=False,
                            start=0.0, end=1.0, size=step)
        contours = dict(x=edge(), y=edge())
    surface = go.Surface(
        x=axis, y=axis, z=obj.reshape(shape), name="objective",
        colorscale=_threshold_colorscale(basin_threshold, c_lo, c_hi, colorscale),
        cmin=c_lo, cmax=c_hi, opacity=float(bg_opacity), showscale=True,
        lighting=lighting, contours=contours,
        hovertemplate=("x1=%{x:.3f}<br>x2=%{y:.3f}<br>"
                       "objective=%{z:.4f}<extra></extra>"),
        colorbar=dict(title=dict(text="Objective", side="top", font=dict(size=18)),
                      tickfont=dict(size=15), len=0.75),
    )
    traces = [surface]

    if placed is not None and len(placed):
        placed = np.atleast_2d(np.asarray(placed, dtype=float))
        for pt in placed:
            regions = []
            # Outermost colour first, each one after it nested a hair inside so
            # two columns on the same optimum never share a surface.
            for kind in _penalty_kinds(pt, penalized):
                color = PENALTY_COLORS[kind]
                pen_alpha = _by_kind(solid_opacity, kind, PENALTY_ALPHA)
                pen_r, pen_lo, pen_hi, pen_lift = _penalty_nesting(
                    kind, _by_kind(penalty_radius, kind, 0.15),
                    z_lo, z_hi, lift)
                # Solid while inspecting, an outline while placing — a filled
                # column is opaque to the pick pass, so leaving it filled would
                # keep every click that lands on it from reaching what is behind.
                if placing:
                    regions.append(_cylinder_cage_trace(
                        pt, pen_r, pen_lo, pen_hi, view_azimuth, wire_width,
                        color=color))
                else:
                    regions.append(_solid_cylinder_trace(
                        pt, pen_r, pen_lo, pen_hi, pen_alpha, color=color))
                if cylinder_edges:
                    # Struck opaque over the filled column, where a translucent
                    # stroke would composite through the wall as well and band.
                    # (Not while placing: the column is only strokes then, with
                    # no wall to composite against.)
                    wire_alpha = PENALTY_ALPHA if placing else 1.0
                    # The outline doubles as the wireframe edging; while placing
                    # it is already the whole column, so only the contour is new.
                    if not placing:
                        # No silhouette edges over a filled column: its own wall
                        # already reads as the silhouette there.
                        regions.append(_cylinder_cage_trace(
                            pt, pen_r, pen_lo, pen_hi, view_azimuth, wire_width,
                            verticals=False, opacity=wire_alpha, color=color))
                    regions.append(_cylinder_contour_trace(
                        fn, pt, pen_r, pen_lift, wire_width, wire_alpha,
                        color=color))
            traces.extend(r for r in regions if r is not None)
    # Printed lines: sampled points styled like the measured points in
    # visualization/plot_run.py — viridis circles with a black outline, on the
    # same colour scale as the surface, riding it so the line bends with the
    # contours underneath.
    line_pts = _line_samples(lines, n_samples)
    if len(line_pts):
        line_obj = fn.predict(line_pts)
        # How far the sample points ride above the surface.  Plotly's 3D
        # renderer depth-tests every trace against every other and exposes no
        # draw-order override, so "on top" has to be bought with clearance: a
        # marker is a screen-space disc hung at its centre's depth, and at the
        # default lift the landscape just downhill of a point is close enough to
        # win the test and eat half the disc.  Lifting the coloured points (and
        # the black rim that belongs to them) clear of the surface is what stops
        # that; the floor stickers and the drops keep their own depth, so they
        # still pass behind the landscape the way they should.
        point_lift = lift * float(line_lift)
        # The floor stickers and the drops that tie them to the real points: the
        # samples read as a *line* laid across the domain, not just a string of
        # dots hovering somewhere over it.  Both hang off the floor plane, lifted
        # clear of it so neither z-fights the plane it sits on.
        floor_z = z_lo + 0.01 * span
        if line_drop_width > 0:
            xs, ys, zs = [], [], []
            for (px, py), pz in zip(line_pts, line_obj):
                # ``None`` breaks the polyline, so each drop is its own segment
                # rather than a zigzag threaded through all of them.
                xs += [px, px, None]
                ys += [py, py, None]
                zs += [floor_z, pz + point_lift, None]
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines", name="printed line drop",
                hoverinfo="skip", showlegend=False,
                opacity=float(line_drop_opacity),
                line=dict(color="black", width=float(line_drop_width)),
            ))
        if line_floor_opacity > 0 and line_outline > 0:
            # The sticker is the point's *outline* alone, so it is a ring: a black
            # disc with a floor-coloured one punched out of its middle.  The hole
            # is drawn at full opacity because it is exactly the floor colour —
            # fading it would let the black underneath bleed through and fill the
            # ring back in — so the opacity slider rides on the black alone.
            ring = float(line_size) + 2.0 * float(line_outline)
            traces.append(go.Scatter3d(
                x=line_pts[:, 0], y=line_pts[:, 1],
                z=np.full(len(line_pts), floor_z), mode="markers",
                name="printed line floor", hoverinfo="skip", showlegend=False,
                opacity=float(line_floor_opacity),
                marker=dict(symbol="circle", color="black", size=ring),
            ))
            traces.append(go.Scatter3d(
                x=line_pts[:, 0], y=line_pts[:, 1],
                z=np.full(len(line_pts), floor_z + 0.002 * span),
                mode="markers", name="printed line floor hole",
                hoverinfo="skip", showlegend=False,
                marker=dict(symbol="circle", color=FLOOR_COLOR,
                            size=float(line_size)),
            ))
        if line_outline > 0:
            # ``marker.line`` is ignored by Scatter3d, so the outline is its own
            # trace: a black dot ``line_outline`` wider on every side, parked a
            # hair below the coloured one (still well clear of the surface) so it
            # cannot win the depth tie and cover the point it is ringing.
            traces.append(go.Scatter3d(
                x=line_pts[:, 0], y=line_pts[:, 1],
                z=line_obj + 0.98 * point_lift, mode="markers",
                name="printed line outline", hoverinfo="skip", showlegend=False,
                marker=dict(symbol="circle", color="black",
                            size=float(line_size) + 2.0 * float(line_outline)),
            ))
        traces.append(go.Scatter3d(
            x=line_pts[:, 0], y=line_pts[:, 1], z=line_obj + point_lift,
            mode="markers", name="printed line",
            customdata=np.column_stack([line_pts, line_obj]),
            hovertemplate=("x1=%{customdata[0]:.3f}<br>x2=%{customdata[1]:.3f}<br>"
                           "objective=%{customdata[2]:.4f}<extra></extra>"),
            marker=dict(symbol="circle", size=float(line_size), color=line_obj,
                        colorscale=colorscale, cmin=c_lo, cmax=c_hi,
                        showscale=False),
        ))
    if line_start is not None:
        # The first endpoint of a line still being drawn, so a half-finished line
        # is visible rather than silently pending.
        start = np.asarray(line_start, dtype=float).reshape(1, 2)
        traces.append(go.Scatter3d(
            x=start[:, 0], y=start[:, 1], z=fn.predict(start) + lift,
            mode="markers", name="line start", hoverinfo="name",
            marker=dict(symbol="circle-open", color="black", size=9,
                        line=dict(width=4)),
        ))

    # For a clean saved figure the grid lines, zero lines and the shaded
    # background planes are all switched off, so the height map floats on its
    # own; otherwise the floor stays FLOOR_COLOR so the printed-line stickers
    # punch a matching hole (see FLOOR_COLOR).
    grid_off = dict(showgrid=False, zeroline=False, showspikes=False,
                    showbackground=False) if hide_grid else {}
    scene = dict(
        xaxis=dict(title=DESIGN_AXIS_LABELS[0], range=[0.0, 1.0], **grid_off),
        yaxis=dict(title=DESIGN_AXIS_LABELS[1], range=[0.0, 1.0], **grid_off),
        # ``zaxis`` styles the xy floor plane, pinned to a known colour so the
        # hole punched in each floor sticker matches it exactly.
        zaxis=dict(title="Objective", range=[z_lo, z_hi],
                   backgroundcolor=FLOOR_COLOR, showbackground=not hide_grid,
                   **{k: v for k, v in grid_off.items() if k != "showbackground"}),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=DESIGN_ASPECT_Z),
        # Always orbit, even while placing.  Plotly emits a 3D click only from a
        # render that happens while a mouse button is held, and only an orbiting
        # camera marks itself dirty on a press — under ``dragmode`` of "pan" or
        # ``False`` the click event never fires at all.  Placing therefore freezes
        # the view by pinning the camera below rather than by disabling the drag.
        dragmode="orbit",
    )
    if placing:
        # Locked to wherever the user left it while free to rotate, and re-applied
        # on every redraw (a fresh uirevision is what makes plotly honour a camera
        # in the layout), so a drag during placing snaps straight back.
        scene["camera"] = camera or DESIGN_CAMERA

    fig = go.Figure(data=traces)
    # Identity for plotly's client-side diff.  A redraw hands the browser a whole
    # new trace list, which it reconciles against the one on screen; with no uid
    # it matches by position, so a slot that *changes type* between redraws — the
    # penalization column is a cage while placing, a mesh or a volume while
    # inspecting — is reconciled as a mutation of the trace already there rather
    # than as a replacement, and a 3D trace mutated across types can come back
    # empty.  Keying the uid on the type as well as the slot makes any such
    # change read as remove-and-add, which always rebuilds cleanly.
    for idx, trace in enumerate(fig.data):
        trace.uid = f"design-{idx}-{trace.type}"
    fig.update_layout(
        title=title,
        scene=scene,
        # While rotating this is constant, so the camera survives every slider
        # move and every placed point; while placing it must differ each time or
        # plotly keeps the user's camera and ignores the locked one.
        uirevision=f"design-{uuid4().hex}" if placing else "design",
        legend=dict(x=0.0, y=1.0), width=FIG_W, height=FIG_H, margin=dict(t=60),
        clickmode="event",
        # Transparent paper is what makes the downloaded PNG see-through; the
        # 3D scene has no separate background of its own once the axis planes
        # are hidden, so this alone carries the transparency through export.
        paper_bgcolor="rgba(0, 0, 0, 0)" if transparent else "white",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("FIGURE_MAKER_PORT", 8061)),
                        help="Port to serve the Dash app on (default: 8061, "
                             "or $FIGURE_MAKER_PORT).")
    args = parser.parse_args()
    print(f"Starting Dash app at http://127.0.0.1:{args.port}")
    build_app().run(debug=True, port=args.port)


if __name__ == "__main__":
    main()
