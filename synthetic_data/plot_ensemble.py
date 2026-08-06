"""Interactive viewer for the layered :class:`~synthetic_data.ensemble.Ensemble`
objective.

A single Dash app shows **one plot at a time**, chosen by the **view** dropdown:

  * **3D (ternary heatmap)** — the 3-simplex drawn as a ternary heatmap.
  * **4D (tetrahedron point cloud)** — the 4-simplex as a point cloud, objective
    mapped to colour.
  * **Design studio (2D square height map)** — a
    :class:`~synthetic_data.ensemble.CartesianEnsemble` on the unit square,
    drawn as a rotatable 3D surface whose height *and* colour are the objective
    (yellow high, blue low).  This is the mode for hand-building a landscape.

Design studio
-------------
The **interaction** radio picks what the mouse does:

  * *Rotate / inspect* — normal 3D navigation (drag to orbit, scroll to zoom).
  * *Place optima* — the camera is pinned to whatever orientation you left it
    in, so the surface holds still (a drag snaps straight back on the next
    redraw), and clicking pins an optimum where you clicked, raising a hill
    there; click the same spot again to drop it, or use **Clear Optima**.
    Pinned optima survive **Randomize** — only the random features are redrawn
    around them.
  * *Cylindrical penalization* / *Surface penalization* — clicking a placed
    optimum wraps it in a penalization region, the 3D counterpart of the ternary
    penalty zones drawn by ``interface/app.py``.  Both carry the optimizer's own
    ``(1 - s**2)**2`` falloff, opaque at the centre and gone at the **penalization
    radius**: the cylindrical one as a translucent red column running the full
    height of the plot, the surface one as a red wash painted onto the landscape
    itself, bending over the contours.  Clicking again in the same mode clears
    the region; clicking in the other mode restyles it.  A click anywhere on a
    column resolves to the optimum it stands on — the click itself reports the
    wall it landed on, which is nowhere near that summit.  A solid column is
    also opaque to plotly's pick pass, which would leave anything behind it
    unclickable, so while a placing mode is live the column is drawn as an open
    wireframe cage instead and the ray passes between the wires; switch back to
    *Rotate / inspect* to see it solid again.  A column stands on a
    summit, so most of it is inside its own hill — drop the **background
    opacity** to see the buried part.
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
    Wiped by **Clear Lines**.

Placed optima carry no marker of their own — the hill each one raises is where
it is, and clicks resolve against the stored position, so the landscape is only
ever shown with the penalization that has been pinned onto it.

Printed lines are an overlay only: they do not alter the objective.  A
**background opacity** slider fades the surface underneath both overlays so they
read clearly against it, and a **peak smoothing** slider rounds
the cusp at the tip of every basin so optima render as hills rather than spikes.
Both are design-studio-only controls; smoothing is not applied in the other two
views, which are meant to show the benchmark landscape as the optimizers see it.

Every parameter of every feature (true optima + their placement/clustering, weak
optima, ridges incl. length, roughness, anisotropy, plateaus, and the edge /
structural bias) has its own control, plus a **negative-fraction** slider (share
of features that subtract mass), a master **random seed** slider, **grid
resolution** slider, and a **basin threshold**.  They all live in a collapsible
**Ensemble settings** section (available in every view) so the design studio's
own controls stay uncluttered.  Each optional feature has an on/off checkbox;
unchecking it passes the disabling value (count/amplitude 0) without losing the
slider position.  There is intentionally no save/load of defaults.

Geometry helpers (simplex lattices, tetra rendering) are reused from
``plot_ackley.py`` so the two viewers stay visually consistent.

Usage:
    python plot_ensemble.py
"""

import argparse
import os
import random
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from synthetic_data.ensemble import (  # noqa: E402
    CartesianEnsemble,
    Ensemble,
    random_ensemble_config,
)
from synthetic_data.plot_ackley import (  # noqa: E402
    BASIN_THRESHOLD,
    DEFAULT_GRID_N,
    DEFAULT_GRID_N_3D,
    FIG_H,
    FIG_W,
    MARKER_OPACITY,
    MARKER_SIZE,
    TERNARY_MARKER_SIZE,
    basin_mask,
    build_simplex_lattice,
    build_ternary_grid,
    tetra_edges_trace,
    to_3d,
    vertex_labels_trace,
)

# Design studio: the square is drawn 1:1 in x/y with the objective squashed into
# a shallower z so the height map reads as a landscape rather than a tower.
DESIGN_ASPECT_Z = 0.55
# Overlays are lifted this fraction of the objective range above the surface so
# they are not z-fought by the polygon they sit on.
OVERLAY_LIFT = 0.02
DESIGN_AXIS_LABELS = ("x1", "x2")
# Colour of the xy floor plane.  Printed-line stickers punch a hole of exactly
# this colour to read as rings, so the plane cannot be left at plotly's default.
FLOOR_COLOR = "rgb(229, 236, 246)"
# Penalization regions.  ``PENALTY_ALPHA`` is the opacity at the centre of a
# surface-hugging region, matching ``_draw_penalty_gradient`` in
# ``interface/app.py`` so these read at the same strength as the ternary rings
# they mirror.  A cylinder is a *volume*, where every slab along the sight line
# composites, so its per-slab alpha is much lower to land in the same place.
PENALTY_ALPHA = 0.5
PENALTY_VOLUME_ALPHA = 0.2
# Grid a cylinder's field is sampled on (nx == ny, and nz), how many isosurfaces
# plotly composites it from, and the resolution of a surface-hugging patch.
PENALTY_VOLUME_N = (26, 14)
PENALTY_VOLUME_ISO = 21
PENALTY_PATCH_N = 36
# The cage a cylinder is drawn as while a placing mode is live: vertical wires,
# horizontal rings, and the line width of both.  A volume is solid to the GPU's
# pick pass, so it swallows every click aimed at what stands behind it; a cage
# only occludes along its own wires, so the ray reaches the landscape between
# them.  Kept sparse for that reason — more wires means less to click through.
PENALTY_CAGE = (12, 3, 2)
PENALTY_MODES = {"pen-cyl": "cyl", "pen-surf": "surf"}
# Where the camera sits until the user has rotated the height map themselves.
DESIGN_CAMERA = {"eye": {"x": 1.25, "y": 1.25, "z": 1.0},
                 "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                 "up": {"x": 0.0, "y": 0.0, "z": 1.0}}


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


def _feature_block(title, toggle_id, children, enabled=True):
    return html.Div([
        dcc.Checklist(id=toggle_id, options=[{"label": f"  {title}", "value": "on"}],
                      value=["on"] if enabled else [],
                      style={"fontWeight": "bold", "marginTop": "6px"}),
        html.Div(children, style={"paddingLeft": "14px"}),
    ], style={"borderTop": "1px solid #ddd", "padding": "4px 0"})


def _collapsible(title, children, open_=False):
    """A native ``<details>`` disclosure — the children stay mounted while it is
    shut, so every callback wired to them keeps working."""
    return html.Details([
        html.Summary(title, style={"fontWeight": "bold", "fontSize": "17px",
                                   "cursor": "pointer", "padding": "8px 0"}),
        html.Div(children),
    ], open=open_, style={"borderTop": "2px solid #999", "marginTop": "8px"})


# ── Dash app ─────────────────────────────────────────────────────────────────

def build_app():
    global html, dcc  # used by the helpers above after Dash import
    from dash import (  # noqa: F811
        Dash, Input, Output, State, callback, ctx, dcc, html, no_update,
    )

    app = Dash(__name__)

    settings = [
        # True optima (always on).
        html.Div([
            html.Label("True Optima", style={"fontWeight": "bold"}),
            _slider("Number of Optima", "n-optima", 0, 150, 1, 4, 25, str),
            _slider("Basin Width (b)", "basin-width", 2.2, 15, 0.1, 5, 2.56,
                    lambda v: f"{v:.1f}"),
            _slider("Optima Margin (normalized gap above background)",
                    "optima-margin", 0.0, 0.5, 0.01, 0.2, 0.1,
                    lambda v: f"{v:.2f}"),
            html.Div([
                html.Label("Placement / Clustering"),
                dcc.Dropdown(id="optima-layout", clearable=False,
                             options=[{"label": "Scatter (uniform)", "value": "scatter"},
                                      {"label": "Cluster @ corners", "value": "corners"},
                                      {"label": "Cluster @ edges", "value": "edges"},
                                      {"label": "Cluster @ faces", "value": "faces"},
                                      {"label": "Cluster @ middle", "value": "middle"}],
                             value="scatter"),
            ], style={"padding": "8px"}),
            _slider("Cluster Count", "n-clusters", 1, 6, 1, 3, 1, str),
            _slider("Cluster Concentration (higher = tighter)",
                    "cluster-conc", 20, 250, 5, 80, 50, str),
            _slider("Cluster Spread (higher = looser hug of region)",
                    "cluster-spread", 0.0, 1.0, 0.05, 0.0, 0.25, str),
        ], style={"borderTop": "2px solid #999", "padding": "4px 0"}),

        _feature_block("Weak Optima (distractors)", "tog-weak", [
            _slider("Count", "n-weak", 0, 30, 1, 6, 5, str),
            _slider("Basin Width (b)", "weak-width", 5, 300, 1, 120, 50, str),
            _slider("Prominence", "weak-amp", 0.0, 1.0, 0.02, 0.6, 0.25,
                    lambda v: f"{v:.2f}"),
        ]),

        _feature_block("Ridges", "tog-ridges", [
            _slider("Count", "n-ridges", 0, 8, 1, 2, 2, str),
            _slider("Tube Width", "ridge-width", 0.01, 0.25, 0.005, 0.06, 0.06,
                    lambda v: f"{v:.2f}"),
            _slider("Prominence", "ridge-amp", 0.0, 1.0, 0.02, 0.6, 0.25,
                    lambda v: f"{v:.2f}"),
            _slider("Length", "ridge-length", 0.1, 1.0, 0.05, 1.0, 0.3,
                    lambda v: f"{v:.2f}"),
        ]),

        _feature_block("Roughness (Perlin noise)", "tog-rough", [
            _slider("Frequency", "noise-freq", 0, 40, 0.5, 8, 10, str),
            _slider("Amplitude (raw)", "noise-amp", 0, 2000, 100, 120, 200, str),
            _slider("Octaves", "noise-oct", 1, 6, 1, 4, 1, str),
        ]),

        _feature_block("Anisotropy (stretch axes)", "tog-aniso", [
            _slider("Strength", "aniso-strength", 0.0, 50, 1, 0.0, 10,
                    lambda v: f"{v:.1f}"),
        ]),

        _feature_block("Plateaus", "tog-plateaus", [
            _slider("Count", "n-plateaus", 0, 8, 1, 2, 2, str),
            _slider("Radius", "plateau-radius", 0.02, 0.40, 0.01, 0.12, 0.1,
                    lambda v: f"{v:.2f}"),
            _slider("Mesa Height", "plateau-amp", 0.0, 1.0, 0.02, 0.7, 0.25,
                    lambda v: f"{v:.2f}"),
        ]),

        _feature_block("Edge / Structural Bias", "tog-edge", [
            html.Div([
                html.Label("Target Region"),
                dcc.Dropdown(id="edge-region", clearable=False,
                             options=[{"label": "Corners", "value": "corners"},
                                      {"label": "Edges", "value": "edges"},
                                      {"label": "Faces", "value": "faces"},
                                      {"label": "Middle", "value": "middle"}],
                             value="corners"),
            ], style={"padding": "8px"}),
            _slider("Prominence", "edge-amp", 0.0, 1.0, 0.02, 0.4, 0.25,
                    lambda v: f"{v:.2f}"),
            _slider("Reach", "edge-reach", 0.05, 0.80, 0.05, 0.3, 0.25,
                    lambda v: f"{v:.2f}"),
        ], enabled=False),

        # Global controls.
        html.Div([
            html.Label("Global", style={"fontWeight": "bold"}),
            _slider("Random Seed", "seed", 0, 100, 1, 0, 20, str),
            _slider("Negative Fraction (share of features that subtract mass)",
                    "neg-frac", 0.0, 1.0, 0.02, 0.5, 0.25, lambda v: f"{v:.2f}"),
            html.Div(_slider("Grid Resolution", "grid-res-3d", 30, 180, 10,
                             160, 30, str), id="grid-3d-wrap"),
            html.Div(_slider("Grid Resolution", "grid-res-4d", 15, 50, 5,
                             DEFAULT_GRID_N, 5, str), id="grid-4d-wrap",
                     style={"display": "none"}),
            _slider("Basin Threshold (below: blacked out 3D / transparent 4D)",
                    "basin-threshold", 0.0, 1.0, 0.01, 0.0, 0.1,
                    lambda v: f"{v:.2f}"),
        ], style={"borderTop": "2px solid #999", "padding": "4px 0"}),
    ]

    controls = html.Div([
        html.Div([
            html.Label("View"),
            dcc.Dropdown(id="dim-select", clearable=False,
                         options=[{"label": "3D (ternary heatmap)", "value": "3d"},
                                  {"label": "4D (tetrahedron point cloud)", "value": "4d"},
                                  {"label": "Design studio (2D square height map)",
                                   "value": "design"}],
                         value="3d"),
            html.Div([
                html.Label("Interaction"),
                dcc.RadioItems(
                    id="click-action",
                    options=[{"label": "  Rotate / inspect", "value": "rotate"},
                             {"label": "  Place optima", "value": "optima"},
                             {"label": "  Draw printed line", "value": "line"},
                             {"label": "  Cylindrical penalization",
                              "value": "pen-cyl"},
                             {"label": "  Surface penalization",
                              "value": "pen-surf"}],
                    value="rotate", labelStyle={"display": "block"}),
            ], id="click-action-wrap", style={"display": "none",
                                              "paddingTop": "6px"}),
            html.Div(id="place-hint",
                     style={"display": "none", "fontSize": "13px",
                            "color": "#555", "paddingTop": "6px"}),
            html.Div(_slider("Printed Line Samples", "line-samples",
                             10, 100, 1, 25, 30, str),
                     id="line-samples-wrap", style={"display": "none"}),
            html.Div(_slider("Printed Line Point Size", "line-size",
                             2, 20, 1, 5, 3, str),
                     id="line-size-wrap", style={"display": "none"}),
            html.Div(_slider("Printed Line Outline Width", "line-outline",
                             0.0, 6.0, 0.25, 1.5, 1.0, lambda v: f"{v:.1f}"),
                     id="line-outline-wrap", style={"display": "none"}),
            html.Div(_slider("Floor Projection Opacity", "line-floor-opacity",
                             0.0, 1.0, 0.05, 1.0, 0.25, lambda v: f"{v:.2f}"),
                     id="line-floor-wrap", style={"display": "none"}),
            html.Div(_slider("Connector Line Width", "line-drop-width",
                             0.0, 10.0, 0.5, 1.5, 2.0, lambda v: f"{v:.1f}"),
                     id="line-drop-width-wrap", style={"display": "none"}),
            html.Div(_slider("Connector Line Opacity", "line-drop-opacity",
                             0.0, 1.0, 0.05, 0.6, 0.25, lambda v: f"{v:.2f}"),
                     id="line-drop-opacity-wrap", style={"display": "none"}),
            html.Div(_slider("Background Opacity", "bg-opacity",
                             0.1, 1.0, 0.05, 1.0, 0.1, lambda v: f"{v:.1f}"),
                     id="bg-opacity-wrap", style={"display": "none"}),
            html.Div(_slider("Peak Smoothing (rounds the tip of every basin)",
                             "basin-smooth", 0.0, 6.0, 0.1, 1.5, 1.0,
                             lambda v: f"{v:.0f}"),
                     id="basin-smooth-wrap", style={"display": "none"}),
            html.Div(_slider("Penalization Radius", "pen-radius",
                             0.02, 0.5, 0.01, 0.15, 0.12, lambda v: f"{v:.2f}"),
                     id="pen-radius-wrap", style={"display": "none"}),
        ], style={"padding": "8px"}),

        html.Div([
            html.Button("Randomize", id="randomize-btn", n_clicks=0,
                        style={"fontSize": "16px", "padding": "8px 24px",
                               "margin": "12px 8px", "cursor": "pointer"}),
            html.Button("Clear Optima", id="clear-optima-btn", n_clicks=0,
                        style={"fontSize": "16px", "padding": "8px 24px",
                               "margin": "12px 8px", "cursor": "pointer"},
                        hidden=True),
            html.Button("Clear Lines", id="clear-lines-btn", n_clicks=0,
                        style={"fontSize": "16px", "padding": "8px 24px",
                               "margin": "12px 8px", "cursor": "pointer"},
                        hidden=True),
        ], style={"textAlign": "center"}),

        _collapsible("Ensemble settings", settings),
    ], style={"width": "60%", "margin": "0 auto"})

    app.layout = html.Div([
        html.H2("Layered Synthetic Objective (Ensemble)",
                style={"textAlign": "center"}),
        controls,
        dcc.Graph(id="cloud-plot", style={"height": f"{FIG_H}px"}),
        # Points pinned by clicking in the design studio; kept out of the figure
        # so they survive redraws and randomization.
        dcc.Store(id="placed-optima", data=[]),
        # Remembers the optima count while the design studio forces it to 0.
        dcc.Store(id="n-optima-memo", data=None),
        # Penalization regions pinned onto placed optima, as
        # ``{"xy": [x, y], "kind": "cyl"|"surf"}``.
        dcc.Store(id="penalized", data=[]),
        # Finished printed lines as [[start_xy], [end_xy]] pairs, plus the first
        # endpoint of a line still being drawn (None between lines).
        dcc.Store(id="print-lines", data=[]),
        dcc.Store(id="line-start", data=None),
        # Orientation the height map was left in while free to rotate; placing
        # modes pin the camera back to it.
        dcc.Store(id="design-camera", data=None),
    ])

    @callback(
        Output("design-camera", "data"),
        Input("cloud-plot", "relayoutData"),
        State("dim-select", "value"),
        State("click-action", "value"),
        prevent_initial_call=True,
    )
    def remember_camera(relayout, dim_sel, action):
        # Only while rotating: this is the orientation a placing mode then locks
        # to, so a stray drag made *during* placing must not move the goalposts.
        if dim_sel != "design" or action != "rotate":
            return no_update
        return (relayout or {}).get("scene.camera") or no_update

    @callback(
        Output("grid-3d-wrap", "style"),
        Output("grid-4d-wrap", "style"),
        Output("clear-optima-btn", "hidden"),
        Output("clear-lines-btn", "hidden"),
        Output("place-hint", "style"),
        Output("cloud-plot", "config"),
        Output("click-action-wrap", "style"),
        Output("line-samples-wrap", "style"),
        Output("line-size-wrap", "style"),
        Output("line-outline-wrap", "style"),
        Output("line-floor-wrap", "style"),
        Output("line-drop-width-wrap", "style"),
        Output("line-drop-opacity-wrap", "style"),
        Output("bg-opacity-wrap", "style"),
        Output("basin-smooth-wrap", "style"),
        Output("pen-radius-wrap", "style"),
        Input("dim-select", "value"),
        Input("click-action", "value"),
    )
    def toggle_grid_controls(dim_sel, action):
        show, hide = {}, {"display": "none"}
        design = dim_sel == "design"
        placing = design and action != "rotate"
        hint = {"fontSize": "13px", "color": "#555", "paddingTop": "6px"}
        act = {"paddingTop": "6px"}
        grid = (hide, show) if dim_sel == "4d" else (show, hide)
        # Placing freezes the view, so zoom goes with rotation.
        cfg = {"scrollZoom": False, "doubleClick": False} if placing else {}
        # Everything between the interaction radio and the penalization radius is
        # design-studio-only and shown together; the radius narrows further to
        # the two penalization modes.
        return (*grid, not design, not design, hint if design else hide, cfg,
                act if design else hide,
                *((show if design else hide,) * 8),
                show if design and action in PENALTY_MODES else hide)

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
                    "of a pinned one again to remove it.  Pinned optima stay "
                    "put when you randomize.")
        if action in PENALTY_MODES:
            shape = ("a red cylinder spanning the height of the plot"
                     if action == "pen-cyl" else
                     "a red region painted onto the landscape around it")
            return (f"The view is locked to the orientation you left it in.  "
                    f"Click the peak of a placed optimum to wrap it in {shape}, "
                    f"fading out "
                    f"with the optimizer's own penalty falloff.  Clicking it "
                    f"again in this mode clears it; clicking it in the other "
                    f"penalization mode restyles it.  Anywhere on an existing "
                    f"column counts as its own optimum, so a column can be "
                    f"clicked back off without hunting for the summit inside it."
                    f"  Columns show as wireframe cages while you are placing, "
                    f"so you can click straight through them to the landscape "
                    f"behind; they go solid again in Rotate / inspect.")
        return ("Drag to orbit the height map, scroll to zoom.  Switch to a "
                "placing mode to lock the view and click features in.")

    @callback(
        Output("n-optima", "value", allow_duplicate=True),
        Output("n-optima-memo", "data"),
        Input("dim-select", "value"),
        State("n-optima", "value"),
        State("n-optima-memo", "data"),
        prevent_initial_call=True,
    )
    def sync_optima_count(dim_sel, n_optima, memo):
        # Entering the design studio: the landscape starts with no optima at all
        # so the only ones present are the ones clicked in.  Leaving restores it.
        if dim_sel == "design":
            return 0, n_optima
        if memo is None:
            return no_update, None
        return memo, None

    @callback(
        Output("placed-optima", "data"),
        Output("print-lines", "data"),
        Output("line-start", "data"),
        Output("penalized", "data"),
        Input("cloud-plot", "clickData"),
        Input("clear-optima-btn", "n_clicks"),
        Input("clear-lines-btn", "n_clicks"),
        Input("dim-select", "value"),
        State("placed-optima", "data"),
        State("print-lines", "data"),
        State("line-start", "data"),
        State("penalized", "data"),
        State("click-action", "value"),
        State("grid-res-3d", "value"),
        State("pen-radius", "value"),
        prevent_initial_call=True,
    )
    def edit_placed_items(click_data, _clear_optima, _clear_lines, dim_sel,
                          placed, lines, start, penalized, action, grid_n,
                          pen_radius):
        keep = (no_update, no_update, no_update, no_update)
        if ctx.triggered_id == "clear-optima-btn":
            # Penalization is pinned *on* an optimum, so it goes with it.
            return [], no_update, no_update, []
        if ctx.triggered_id == "clear-lines-btn":
            # A half-drawn line goes with the finished ones, otherwise the next
            # click would silently close a line against a stale start point.
            return no_update, [], None, no_update
        if ctx.triggered_id == "dim-select":
            # Overlays are stored as points of whichever domain placed them, so
            # they cannot carry across a view switch.
            return [], [], None, []
        if dim_sel != "design" or action == "rotate":
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
                return no_update, no_update, end, no_update
            return no_update, [*(lines or []), [list(start), end]], None, no_update

        placed = [list(p) for p in (placed or [])]
        # A click within about one grid cell of a pinned optimum is a click *on*
        # that optimum, which is what both placing and penalizing act on.
        tol = max(0.02, 1.5 / max(int(grid_n or 1), 1))
        hit = next((i for i, p in enumerate(placed)
                    if float(np.linalg.norm(np.asarray(p, dtype=float) - xy)) <= tol),
                   None)
        if hit is None:
            # A click that landed on a cylinder wall reports the wall, not the
            # optimum underneath it, so resolve it back to the column's owner.
            # This is also what stops a click on a column from being read as bare
            # landscape and dropping a stray optimum onto its side.
            hit = _cylinder_owner(xy, placed, penalized,
                                  float(pen_radius or 0.0))

        if action in PENALTY_MODES:
            # Penalization only ever attaches to an optimum that is already
            # placed — a click on bare landscape does nothing here.
            if hit is None:
                return keep
            kind = PENALTY_MODES[action]
            pen = [dict(e) for e in (penalized or [])]
            for i, e in enumerate(pen):
                if _same_point(e["xy"], placed[hit]):
                    # Same mode again clears it; the other mode restyles it.
                    if e["kind"] == kind:
                        pen.pop(i)
                    else:
                        pen[i] = {"xy": e["xy"], "kind": kind}
                    return no_update, no_update, no_update, pen
            pen.append({"xy": list(placed[hit]), "kind": kind})
            return no_update, no_update, no_update, pen

        # Placing: the same gesture places and un-places.
        if hit is not None:
            gone = placed.pop(hit)
            pen = [e for e in (penalized or []) if not _same_point(e["xy"], gone)]
            return placed, no_update, no_update, pen
        placed.append([float(v) for v in xy])
        return placed, no_update, no_update, no_update

    @callback(
        Output("n-optima", "value"),
        Output("basin-width", "value"),
        Output("optima-layout", "value"),
        Output("n-clusters", "value"),
        Output("cluster-conc", "value"),
        Output("cluster-spread", "value"),
        Output("n-weak", "value"),
        Output("weak-width", "value"),
        Output("weak-amp", "value"),
        Output("n-ridges", "value"),
        Output("ridge-width", "value"),
        Output("ridge-amp", "value"),
        Output("ridge-length", "value"),
        Output("noise-freq", "value"),
        Output("noise-amp", "value"),
        Output("noise-oct", "value"),
        Output("aniso-strength", "value"),
        Output("n-plateaus", "value"),
        Output("plateau-radius", "value"),
        Output("plateau-amp", "value"),
        Output("edge-region", "value"),
        Output("edge-amp", "value"),
        Output("edge-reach", "value"),
        Output("neg-frac", "value"),
        Output("tog-weak", "value"),
        Output("tog-ridges", "value"),
        Output("tog-rough", "value"),
        Output("tog-aniso", "value"),
        Output("tog-plateaus", "value"),
        Output("tog-edge", "value"),
        Input("randomize-btn", "n_clicks"),
        State("dim-select", "value"),
        prevent_initial_call=True,
    )
    def randomize(_n_clicks, dim_sel):
        # Draw exactly what optimize/run_mobo.py and optimize/evaluate.py generate
        # per run, so the viewer's "Randomize" matches the benchmark landscapes.
        # A random Sobol' index + scramble seed gives a fresh landscape per click.
        cfg = random_ensemble_config(_dim_of(dim_sel),
                                     index=random.randrange(1 << 20),
                                     seed=random.randrange(1 << 16))
        on = lambda v: ["on"] if v else []  # noqa: E731
        return (
            # The design studio keeps its hand-placed-only landscape: randomize
            # refreshes every other feature but never adds random optima back in.
            no_update if dim_sel == "design" else cfg["n_optima"],  # n-optima
            cfg["basin_width"],                          # basin-width
            cfg["optima_layout"],                        # optima-layout
            cfg["n_optima_clusters"],                    # n-clusters
            cfg["optima_cluster_conc"],                  # cluster-conc
            cfg["optima_cluster_spread"],                # cluster-spread
            cfg["n_weak"],                               # n-weak
            cfg["weak_width"],                           # weak-width
            cfg["weak_amp"],                             # weak-amp
            cfg["n_ridges"],                             # n-ridges
            cfg["ridge_width"],                          # ridge-width
            cfg["ridge_amp"],                            # ridge-amp
            cfg["ridge_length"],                         # ridge-length
            cfg["noise_freq"],                           # noise-freq
            cfg["noise_amp"],                            # noise-amp
            cfg["noise_octaves"],                        # noise-oct
            cfg["aniso_strength"],                       # aniso-strength
            cfg["n_plateaus"],                           # n-plateaus
            cfg["plateau_radius"],                       # plateau-radius
            cfg["plateau_amp"],                          # plateau-amp
            cfg["edge_region"] or "corners",             # edge-region
            cfg["edge_amp"],                             # edge-amp
            cfg["edge_reach"],                           # edge-reach
            cfg["neg_frac"],                             # neg-frac
            on(cfg["n_weak"] > 0),                       # tog-weak
            on(cfg["n_ridges"] > 0),                     # tog-ridges
            on(cfg["noise_amp"] > 0),                    # tog-rough
            on(cfg["aniso_strength"] > 0),               # tog-aniso
            on(cfg["n_plateaus"] > 0),                   # tog-plateaus
            on(cfg["edge_region"] is not None),          # tog-edge
        )

    @callback(
        Output("cloud-plot", "figure"),
        Input("dim-select", "value"),
        Input("n-optima", "value"),
        Input("basin-width", "value"),
        Input("optima-margin", "value"),
        Input("optima-layout", "value"),
        Input("n-clusters", "value"),
        Input("cluster-conc", "value"),
        Input("cluster-spread", "value"),
        Input("tog-weak", "value"),
        Input("n-weak", "value"),
        Input("weak-width", "value"),
        Input("weak-amp", "value"),
        Input("tog-ridges", "value"),
        Input("n-ridges", "value"),
        Input("ridge-width", "value"),
        Input("ridge-amp", "value"),
        Input("ridge-length", "value"),
        Input("tog-rough", "value"),
        Input("noise-freq", "value"),
        Input("noise-amp", "value"),
        Input("noise-oct", "value"),
        Input("tog-aniso", "value"),
        Input("aniso-strength", "value"),
        Input("tog-plateaus", "value"),
        Input("n-plateaus", "value"),
        Input("plateau-radius", "value"),
        Input("plateau-amp", "value"),
        Input("tog-edge", "value"),
        Input("edge-region", "value"),
        Input("edge-amp", "value"),
        Input("edge-reach", "value"),
        Input("neg-frac", "value"),
        Input("seed", "value"),
        Input("grid-res-3d", "value"),
        Input("grid-res-4d", "value"),
        Input("basin-threshold", "value"),
        Input("placed-optima", "data"),
        Input("print-lines", "data"),
        Input("line-start", "data"),
        Input("line-samples", "value"),
        Input("line-size", "value"),
        Input("line-outline", "value"),
        Input("line-floor-opacity", "value"),
        Input("line-drop-width", "value"),
        Input("line-drop-opacity", "value"),
        Input("bg-opacity", "value"),
        Input("basin-smooth", "value"),
        Input("penalized", "data"),
        Input("pen-radius", "value"),
        Input("click-action", "value"),
        State("design-camera", "data"),
    )
    def update_plot(dim_sel, n_optima, basin_width, optima_margin,
                    optima_layout, n_clusters, cluster_conc, cluster_spread,
                    tog_weak, n_weak, weak_width, weak_amp,
                    tog_ridges, n_ridges, ridge_width, ridge_amp, ridge_length,
                    tog_rough, noise_freq, noise_amp, noise_oct,
                    tog_aniso, aniso_strength,
                    tog_plateaus, n_plateaus, plateau_radius, plateau_amp,
                    tog_edge, edge_region, edge_amp, edge_reach, neg_frac,
                    seed, grid_res_3d, grid_res_4d, basin_threshold, placed,
                    print_lines, line_start, line_samples, line_size,
                    line_outline, line_floor_opacity, line_drop_width,
                    line_drop_opacity, bg_opacity,
                    basin_smooth, penalized, pen_radius, action, camera):
        dim = _dim_of(dim_sel)
        on = lambda t: bool(t)  # noqa: E731
        design = dim_sel == "design"
        placed_arr = (np.asarray(placed, dtype=float).reshape(-1, dim)
                      if design and placed else np.empty((0, dim)))
        pinned = placed_arr if len(placed_arr) else None
        # Printed lines are an overlay only — they never feed the objective.
        lines = (print_lines or []) if design else []
        pending = line_start if design else None

        cls = CartesianEnsemble if design else Ensemble
        fn = cls(
            dim=dim,
            n_optima=int(n_optima),
            basin_width=float(basin_width),
            # The smoothing slider is a design-studio control, so it must not
            # quietly reshape the ternary / tetrahedron landscapes behind it.
            basin_smoothing=float(basin_smooth) if design else 0.0,
            optima_margin=float(optima_margin),
            optima_layout=str(optima_layout),
            n_optima_clusters=int(n_clusters),
            optima_cluster_conc=float(cluster_conc),
            optima_cluster_spread=float(cluster_spread),
            pinned_optima=pinned,
            n_weak=int(n_weak) if on(tog_weak) else 0,
            weak_width=float(weak_width),
            weak_amp=float(weak_amp),
            n_ridges=int(n_ridges) if on(tog_ridges) else 0,
            ridge_width=float(ridge_width),
            ridge_amp=float(ridge_amp),
            ridge_length=float(ridge_length),
            noise_freq=float(noise_freq),
            noise_amp=float(noise_amp) if on(tog_rough) else 0.0,
            noise_octaves=int(noise_oct),
            aniso_strength=float(aniso_strength) if on(tog_aniso) else 0.0,
            n_plateaus=int(n_plateaus) if on(tog_plateaus) else 0,
            plateau_radius=float(plateau_radius),
            plateau_amp=float(plateau_amp),
            edge_region=str(edge_region) if on(tog_edge) else None,
            edge_amp=float(edge_amp),
            edge_reach=float(edge_reach),
            neg_frac=float(neg_frac),
            seed=int(seed),
        )
        peaks = np.asarray(fn.centers)
        title = (f"Ensemble — {len(peaks)} optima, margin={optima_margin:g} "
                 f"(dim {dim}, seed {seed})")
        if design:
            pen = penalized or []
            title += (f" — {len(placed_arr)} placed, {len(lines)} printed, "
                      f"{len(pen)} penalized")
            return _design_figure(fn, int(grid_res_3d), basin_threshold, title,
                                  placed_arr, lines, pending, int(line_samples),
                                  float(bg_opacity), action != "rotate", camera,
                                  penalized=pen, penalty_radius=float(pen_radius),
                                  line_size=float(line_size),
                                  line_outline=float(line_outline),
                                  line_floor_opacity=float(line_floor_opacity),
                                  line_drop_width=float(line_drop_width),
                                  line_drop_opacity=float(line_drop_opacity))
        if dim == 3:
            return _ternary_figure(fn, peaks, int(grid_res_3d), basin_threshold,
                                   title)
        return _point_cloud_figure(fn, peaks, int(grid_res_4d), basin_threshold, title)

    return app


def _dim_of(dim_sel):
    """Input dimensionality behind each view."""
    if dim_sel == "4d":
        return 4
    if dim_sel == "design":
        return 2
    return 3


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

    Every trace in the design studio lives in the same square, so the click
    resolves the same way whether it landed on the surface or on an overlay
    marker sitting above it.
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

def _penalty_profile(s):
    """The optimizer's smooth repulsion strength at normalized radius ``s``.

    Identical to the ternary overlays in ``interface/app.py``: ``(1 - s**2)**2``,
    which is 1 at the centre of the region and eases to 0 at its boundary.
    """
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    return (1.0 - s ** 2) ** 2


def _same_point(a, b, tol=1e-9):
    """Whether two stored points are the same pinned optimum."""
    return float(np.linalg.norm(np.asarray(a, dtype=float)
                                - np.asarray(b, dtype=float))) <= tol


def _cylinder_owner(xy, placed, penalized, radius):
    """Index in ``placed`` of the cylinder the click landed on, or ``None``.

    A cylinder is a *volume*: the click resolves to a point on its wall or cap,
    which is up to ``radius`` away from the optimum the column stands on, so the
    usual "within a grid cell of an optimum" test never matches it.  Any click
    inside the footprint of a column is therefore read as a click on that
    column's optimum — the nearest one, if two overlap.

    Surface-hugging regions need none of this: they ride the landscape, so a
    click on one already carries the x/y the user aimed at.
    """
    xy = np.asarray(xy, dtype=float)
    # A click on the rim sits at exactly ``radius``, which floating point puts on
    # either side of the cut, so the footprint is widened by a hair.
    reach = float(radius) * (1.0 + 1e-6)
    best, best_d = None, float("inf")
    for entry in penalized or []:
        if entry.get("kind") != "cyl":
            continue
        center = np.asarray(entry["xy"], dtype=float)
        d = float(np.linalg.norm(xy - center))
        if d > reach or d >= best_d:
            continue
        for i, p in enumerate(placed):
            if _same_point(p, center):
                best, best_d = i, d
                break
    return best


def _penalty_kind(point, penalized, tol=1e-9):
    """Which penalization style is pinned on ``point``, or ``None``."""
    for entry in penalized or []:
        if _same_point(entry["xy"], point, tol):
            return entry["kind"]
    return None


def _cylinder_trace(center, radius, z_lo, z_hi):
    """A penalization *volume*: a translucent red cylinder on ``center``, running
    the full height of the plot.

    The field is :func:`_penalty_profile` of the horizontal distance to the axis
    — constant in z, so its isosurfaces are cylinders — handed to
    :class:`plotly.graph_objects.Volume`, whose ``opacityscale`` turns that
    penalty directly into alpha.  Every slab along a sight line composites, so
    the eye accumulates the penalty *integrated* through the column: solid in the
    middle, feathering to nothing at the rim, exactly the way the ternary rings
    in ``interface/app.py`` fade.
    """
    center = np.asarray(center, dtype=float).ravel()
    radius = float(radius)
    if radius <= 0:
        return None
    n_xy, n_z = PENALTY_VOLUME_N
    gx = np.linspace(center[0] - radius, center[0] + radius, n_xy)
    gy = np.linspace(center[1] - radius, center[1] + radius, n_xy)
    gz = np.linspace(float(z_lo), float(z_hi), n_z)
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    s = np.hypot(X - center[0], Y - center[1]) / radius
    return go.Volume(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        value=_penalty_profile(s).ravel(), isomin=0.0, isomax=1.0,
        colorscale=[[0.0, "red"], [1.0, "red"]], showscale=False,
        opacity=PENALTY_VOLUME_ALPHA, opacityscale=[[0.0, 0.0], [1.0, 1.0]],
        surface_count=PENALTY_VOLUME_ISO,
        # The z caps are what the column reads as when looked at straight down
        # the axis; the x/y caps would just be flat slabs through it.
        caps=dict(x_show=False, y_show=False, z_show=True),
        # Not ``hoverinfo="skip"``: a skipped trace is dropped from the 3D pick
        # pass, and because the column stands over its own optimum it is what the
        # ray hits first — so skipping it means *no* click event fires at all and
        # the region can never be clicked back off.
        hoverinfo="name", showlegend=False, name="penalization",
    )


def _cylinder_cage_trace(center, radius, z_lo, z_hi):
    """The same penalization column drawn as an open wireframe.

    Plotly resolves a 3D click in the GPU's pick buffer and keeps a single
    topmost hit, so a solid volume standing over its own optimum swallows every
    click aimed at that optimum (``hoverinfo="skip"`` only discards the hit — it
    does not let the ray continue).  While a placing mode is live the column is
    therefore drawn as a cage: the ray passes between the wires and lands on the
    landscape underneath, and the wires themselves still resolve to the column's
    optimum via :func:`_cylinder_owner`.
    """
    center = np.asarray(center, dtype=float).ravel()
    radius = float(radius)
    if radius <= 0:
        return None
    n_wires, n_rings, width = PENALTY_CAGE
    theta = np.linspace(0.0, 2.0 * np.pi, 49)
    xs, ys, zs = [], [], []
    # ``None`` breaks the polyline so the rings and wires stay separate strokes
    # instead of being threaded together.
    for z in np.linspace(float(z_lo), float(z_hi), n_rings):
        xs += [*(center[0] + radius * np.cos(theta)), None]
        ys += [*(center[1] + radius * np.sin(theta)), None]
        zs += [z] * len(theta) + [None]
    for a in np.linspace(0.0, 2.0 * np.pi, n_wires, endpoint=False):
        xs += [center[0] + radius * np.cos(a)] * 2 + [None]
        ys += [center[1] + radius * np.sin(a)] * 2 + [None]
        zs += [float(z_lo), float(z_hi), None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines", opacity=PENALTY_ALPHA,
        line=dict(color="red", width=width),
        hoverinfo="name", showlegend=False, name="penalization",
    )


def _surface_penalty_trace(fn, center, radius, lift):
    """A penalization region painted onto the landscape itself.

    A patch of the height map around ``center`` re-sampled and drawn just above
    it, tinted red with the opacity carrying :func:`_penalty_profile` — solid in
    the middle, invisible past ``radius`` — so the region bends over the contours
    instead of floating above them.
    """
    center = np.asarray(center, dtype=float).ravel()
    radius = float(radius)
    if radius <= 0:
        return None
    # Clipped to the square, so a region near a side is cut by the domain rather
    # than hanging off it (the duplicated edge ticks are degenerate, not drawn).
    ax_x = np.clip(np.linspace(center[0] - radius, center[0] + radius,
                               PENALTY_PATCH_N), 0.0, 1.0)
    ax_y = np.clip(np.linspace(center[1] - radius, center[1] + radius,
                               PENALTY_PATCH_N), 0.0, 1.0)
    xx, yy = np.meshgrid(ax_x, ax_y)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    s = np.linalg.norm(pts - center.reshape(1, -1), axis=1) / radius
    pen = _penalty_profile(s).reshape(xx.shape)
    z = fn.predict(pts).reshape(xx.shape) + lift
    return go.Surface(
        x=ax_x, y=ax_y, z=z, surfacecolor=pen, cmin=0.0, cmax=1.0,
        colorscale=[[0.0, "red"], [1.0, "red"]],
        # ``opacityscale`` turns the penalty into alpha; the trace opacity caps
        # it, so the centre lands at exactly PENALTY_ALPHA and the rim at 0.
        opacityscale=[[0.0, 0.0], [1.0, 1.0]], opacity=PENALTY_ALPHA,
        # Clickable for the same reason as the cylinder — it covers the summit it
        # is painted on, so a skipped pick would swallow the click.  Unlike the
        # cylinder its x/y is honest: the patch rides the landscape, so a click
        # on it lands where the user thinks it did.
        showscale=False, showlegend=False, hoverinfo="name",
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
        name="penalization",
    )


def _threshold_colorscale(threshold, lo, hi, n_stops=16):
    """Viridis with everything below ``threshold`` flattened to black.

    Matches the ternary view, where sub-threshold compositions are blacked out;
    a surface has no per-point colour override, so the cut is baked into the
    colourscale instead.
    """
    if hi - lo < 1e-12 or not np.isfinite(threshold) or threshold <= lo:
        return "Viridis"
    if threshold >= hi:
        return [[0.0, "black"], [1.0, "black"]]
    from plotly.colors import sample_colorscale

    t = float((threshold - lo) / (hi - lo))
    fracs = np.linspace(0.0, 1.0, n_stops)
    colors = sample_colorscale("Viridis", [float(f) for f in fracs])
    # The duplicated stop at ``t`` is what makes the cut a hard edge.
    return ([[0.0, "black"], [t, "black"]]
            + [[t + (1.0 - t) * float(f), c] for f, c in zip(fracs, colors)])


# ── Figure builders ──────────────────────────────────────────────────────────

def _design_figure(fn, grid_n, basin_threshold, title, placed=None, lines=None,
                   line_start=None, n_samples=25, bg_opacity=1.0, placing=False,
                   camera=None, penalized=None, penalty_radius=0.15,
                   line_size=5.0, line_outline=1.5, line_floor_opacity=1.0,
                   line_drop_width=1.5, line_drop_opacity=0.6):
    """The unit-square objective as a rotatable 3D height map.

    Height *and* colour are both the objective, so the surface reads the same way
    the ternary heatmap does (yellow high, blue low) while also standing up in 3D.
    """
    axis, shape, pts = build_square_grid(grid_n)
    obj = fn.predict(pts)
    obj_min, obj_max = float(np.nanmin(obj)), float(np.nanmax(obj))
    span = max(obj_max - obj_min, 1e-6)
    lift = OVERLAY_LIFT * span
    # Pinned rather than auto-ranged, so "top to bottom of the plot" is a height
    # the penalization cylinders can actually be built to.  The headroom above
    # the landscape is what makes a cylinder read at all: an optimum sits on a
    # summit, so without it the column would be buried inside its own hill.
    z_lo, z_hi = obj_min - 0.04 * span, obj_max + 0.3 * span

    surface = go.Surface(
        x=axis, y=axis, z=obj.reshape(shape), name="objective",
        colorscale=_threshold_colorscale(basin_threshold, obj_min, obj_max),
        cmin=obj_min, cmax=obj_max, opacity=float(bg_opacity), showscale=True,
        hovertemplate=("x1=%{x:.3f}<br>x2=%{y:.3f}<br>"
                       "objective=%{z:.4f}<extra></extra>"),
        colorbar=dict(title=dict(text="Objective", side="top", font=dict(size=18)),
                      tickfont=dict(size=15), len=0.75),
    )
    traces = [surface]

    if placed is not None and len(placed):
        placed = np.atleast_2d(np.asarray(placed, dtype=float))
        # A surface-hugging region rides just above the height map — any less
        # and the two z-fight into moire rings.
        for pt in placed:
            kind = _penalty_kind(pt, penalized)
            if kind == "cyl":
                # Solid while inspecting, a cage while placing — a volume is
                # opaque to the pick pass, so leaving it solid would keep every
                # click that lands on it from reaching what is behind it.
                region = (_cylinder_cage_trace(pt, penalty_radius, z_lo, z_hi)
                          if placing else
                          _cylinder_trace(pt, penalty_radius, z_lo, z_hi))
            elif kind == "surf":
                region = _surface_penalty_trace(fn, pt, penalty_radius, lift)
            else:
                continue
            if region is not None:
                traces.append(region)
    # Printed lines: sampled points styled like the measured points in
    # visualization/plot_run.py — viridis circles with a black outline, on the
    # same colour scale as the surface, riding it so the line bends with the
    # contours underneath.
    line_pts = _line_samples(lines, n_samples)
    if len(line_pts):
        line_obj = fn.predict(line_pts)
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
                zs += [floor_z, pz + lift, None]
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
                z=line_obj + 0.9 * lift, mode="markers",
                name="printed line outline", hoverinfo="skip", showlegend=False,
                marker=dict(symbol="circle", color="black",
                            size=float(line_size) + 2.0 * float(line_outline)),
            ))
        traces.append(go.Scatter3d(
            x=line_pts[:, 0], y=line_pts[:, 1], z=line_obj + lift,
            mode="markers", name="printed line",
            customdata=np.column_stack([line_pts, line_obj]),
            hovertemplate=("x1=%{customdata[0]:.3f}<br>x2=%{customdata[1]:.3f}<br>"
                           "objective=%{customdata[2]:.4f}<extra></extra>"),
            marker=dict(symbol="circle", size=float(line_size), color=line_obj,
                        colorscale="Viridis", cmin=obj_min, cmax=obj_max,
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

    scene = dict(
        xaxis=dict(title=DESIGN_AXIS_LABELS[0], range=[0.0, 1.0]),
        yaxis=dict(title=DESIGN_AXIS_LABELS[1], range=[0.0, 1.0]),
        # ``zaxis`` styles the xy floor plane, pinned to a known colour so the
        # hole punched in each floor sticker matches it exactly.
        zaxis=dict(title="Objective", range=[z_lo, z_hi],
                   showbackground=True, backgroundcolor=FLOOR_COLOR),
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
    fig.update_layout(
        title=title,
        scene=scene,
        # While rotating this is constant, so the camera survives every slider
        # move and every placed point; while placing it must differ each time or
        # plotly keeps the user's camera and ignores the locked one.
        uirevision=f"design-{uuid4().hex}" if placing else "design",
        legend=dict(x=0.0, y=1.0), width=FIG_W, height=FIG_H, margin=dict(t=60),
        clickmode="event",
    )
    return fig


def _ternary_figure(fn, peaks, grid_n, basin_threshold, title):
    comp = build_ternary_grid(grid_n)
    obj = fn.predict(comp)
    obj_min, obj_max = float(np.nanmin(obj)), float(np.nanmax(obj))
    above = basin_mask(obj, basin_threshold)

    traces = []
    if (~above).any():
        traces.append(go.Scatterternary(
            a=comp[~above, 2], b=comp[~above, 0], c=comp[~above, 1],
            mode="markers", name="below basin threshold", hoverinfo="skip",
            showlegend=False,
            marker=dict(color="black", size=TERNARY_MARKER_SIZE),
        ))
    traces.append(go.Scatterternary(
        a=comp[above, 2], b=comp[above, 0], c=comp[above, 1], mode="markers",
        name="objective", hoverinfo="skip",
        marker=dict(color=obj[above], colorscale="Viridis", cmin=obj_min, cmax=obj_max,
                    size=TERNARY_MARKER_SIZE, showscale=True,
                    colorbar=dict(
                        title=dict(text="Objective", side="top", font=dict(size=20)),
                        tickfont=dict(size=18), len=0.8, x=1.12)),
    ))
    if len(peaks):
        traces.append(go.Scatterternary(
            a=peaks[:, 2], b=peaks[:, 0], c=peaks[:, 1], mode="markers",
            name="known peak", visible="legendonly",
            marker=dict(symbol="star", color="red", size=14,
                        line=dict(color="white", width=1)),
        ))

    fig = go.Figure(data=traces)
    axis_title_font = dict(size=22)
    axis_tick_font = dict(size=18)
    fig.update_layout(
        title=title,
        ternary=dict(
            sum=1,
            # a-axis is the top vertex; b/c are the two bottom vertices, so pad
            # their titles with a leading line break to clear the tick labels.
            aaxis=dict(title=dict(text="FAPbI3", font=axis_title_font),
                       tickfont=axis_tick_font),
            baxis=dict(title=dict(text="<br>MAPbI3", font=axis_title_font),
                       tickfont=axis_tick_font),
            caxis=dict(title=dict(text="<br>MAPbBr3", font=axis_title_font),
                       tickfont=axis_tick_font),
        ),
        legend=dict(x=1.28, y=1.0), width=FIG_W, height=FIG_H, margin=dict(t=60),
    )
    return fig


def _point_cloud_figure(fn, peaks, grid_n, basin_threshold, title):
    comp = build_simplex_lattice(grid_n)
    obj = fn.predict(comp)
    xyz = to_3d(comp)
    obj_min, obj_max = float(obj.min()), float(obj.max())

    above = basin_mask(obj, basin_threshold)
    xyz_v, obj_v, comp_v = xyz[above], obj[above], comp[above]
    hover = [f"x=[{a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}]<br>obj={v:.2f}"
             for (a, b, c, d), v in zip(comp_v, obj_v)]

    cloud = go.Scatter3d(
        x=xyz_v[:, 0], y=xyz_v[:, 1], z=xyz_v[:, 2], mode="markers",
        name="objective", text=hover, hoverinfo="text",
        marker=dict(color=obj_v, colorscale="Viridis", cmin=obj_min, cmax=obj_max,
                    size=MARKER_SIZE, opacity=MARKER_OPACITY, showscale=True,
                    colorbar=dict(title="Objective")),
    )
    data = [cloud, tetra_edges_trace(), vertex_labels_trace()]
    if len(peaks):
        peaks_xyz = to_3d(peaks)
        data.append(go.Scatter3d(
            x=peaks_xyz[:, 0], y=peaks_xyz[:, 1], z=peaks_xyz[:, 2], mode="markers",
            name="known peak", visible="legendonly",
            marker=dict(symbol="diamond", color="red", size=6,
                        line=dict(color="white", width=1)),
            hoverinfo="name",
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title,
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), aspectmode="data"),
        legend=dict(x=0.0, y=1.0), width=FIG_W, height=FIG_H,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PLOT_ENSEMBLE_PORT", 8060)),
                        help="Port to serve the Dash app on (default: 8060, "
                             "or $PLOT_ENSEMBLE_PORT).")
    args = parser.parse_args()
    print(f"Starting Dash app at http://127.0.0.1:{args.port}")
    build_app().run(debug=True, port=args.port)


if __name__ == "__main__":
    main()
