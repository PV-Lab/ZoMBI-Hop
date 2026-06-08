"""Interactive 3D point-cloud view of a 4D negated-Ackley objective.

The companion to ``synthetic_data/plot_4d.py``. Where that script slices the
objective by x4 and shows one ternary cross-section at a time behind a slider,
this one shows the *whole* 4D function at once as a single semi-transparent 3D
point cloud you can orbit, pan, and zoom.

The trick is that the 4-element probability simplex (x1 + x2 + x3 + x4 = 1) is a
regular tetrahedron: every composition is a convex combination of the four
vertices, so we map each lattice point (a, b, c, d) -> a*V1 + b*V2 + c*V3 + d*V4
into 3D and colour it by the objective. No dimension is hidden behind a slider --
all four coordinates are encoded in the 3D position, and the objective is the
colour. Output is a self-contained interactive HTML file that opens in your
default browser (rotatable WebGL scene; no notebook, kernel, or GUI backend).

Usage:
    python point_cloud_4d.py --ackley {centroid|edge|vertex|multimodal|realistic}
    python point_cloud_4d.py --ackley centroid --example

The variant catalogue is defined once in ``synthetic_data/ackley.py``; adding a
new mode there makes it available here (and in plot_4d.py) automatically.

Overlay API
-----------
``add_simplex_overlays`` lets a caller draw ZoMBI-Hop-style annotations on top
of the point cloud, all specified as plain simplex compositions (no ZoMBI data
structures required). It supports:

  * **pared points**  — discrete sampled compositions, coloured by objective and
    optionally size-faded by recency (newest largest);
  * **main / cache lines** — LineBO's suggested lines as 3D segments through the
    tetrahedron (orange solid + cornflower-blue dotted);
  * **needles** — discovered-needle markers plus translucent purple penalization
    ellipsoids (ILR-space ``M`` matrices, same convention as the 2D plots).

``--example`` stacks a synthetic demonstration of the above on top of the chosen
``--ackley`` landscape (the example data is derived from that objective, so the
pared points and needle sit on real peaks).
"""

import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
# Make the repo root importable so this runs from any working directory.
sys.path.insert(0, str(HERE.parent))
from synthetic_data.ackley import Ackley  # noqa: E402

DIM = 4           # the 4-element simplex (x1 + x2 + x3 + x4 = 1) is a tetrahedron
GRID_N = 44       # lattice resolution: points where the four integer counts sum to GRID_N
MARKER_SIZE = 3.5
MARKER_OPACITY = 0.18   # low so the dense interior reads as a translucent volume
FIG_W, FIG_H = 950, 850
GRID_RESOLUTIONS = [20, 30, 44, 60]  # slider steps for background cloud density

# ── Overlay styling (annotations drawn on top of the cloud) ────────────────────
PARED_SIZE = 8.0                 # base marker size for pared points
MAIN_LINE_COLOR = "orange"       # LineBO main suggested line
CACHE_LINE_COLOR = "deepskyblue"     # LineBO cache line
NEEDLE_MARKER_COLOR = "red"      # discovered-needle marker
NEEDLE_ELL_COLOR = "purple"      # penalization ellipsoid
NEEDLE_ELL_OPACITY = 0.16

# Vertices of a regular tetrahedron, one per simplex coordinate (x1..x4). Any
# affine-independent placement works; these are the canonical "alternating cube
# corners" set, recentred on the origin for nice default framing.
TETRA_VERTICES = np.array([
    [1.0,  1.0,  1.0],
    [1.0, -1.0, -1.0],
    [-1.0,  1.0, -1.0],
    [-1.0, -1.0,  1.0],
])
TETRA_VERTICES = TETRA_VERTICES - TETRA_VERTICES.mean(axis=0)
VERTEX_LABELS = ["x1", "x2", "x3", "x4"]


def build_simplex_lattice(grid_n):
    """All (x1, x2, x3, x4) on the 4-simplex with denominator ``grid_n``.

    Returns an (N, 4) array of compositions summing to 1, one per integer
    4-tuple (i, j, k, l) with i + j + k + l = grid_n.
    """
    pts = [
        (i, j, k, grid_n - i - j - k)
        for i in range(grid_n + 1)
        for j in range(grid_n + 1 - i)
        for k in range(grid_n + 1 - i - j)
    ]
    return np.array(pts, dtype=float) / grid_n


def to_3d(comp):
    """Map simplex compositions (N, 4) into 3D tetrahedron coordinates (N, 3)."""
    return comp @ TETRA_VERTICES


def tetra_edges_trace():
    """Wireframe of the tetrahedron's six edges, for spatial orientation."""
    xs, ys, zs = [], [], []
    for i in range(4):
        for j in range(i + 1, 4):
            xs += [TETRA_VERTICES[i, 0], TETRA_VERTICES[j, 0], None]
            ys += [TETRA_VERTICES[i, 1], TETRA_VERTICES[j, 1], None]
            zs += [TETRA_VERTICES[i, 2], TETRA_VERTICES[j, 2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines", name="simplex edges",
        line=dict(color="rgba(60,60,60,0.6)", width=3), hoverinfo="skip",
    )


def vertex_labels_trace():
    """Corner labels x1..x4, nudged outward so they clear the cloud."""
    pos = TETRA_VERTICES * 1.12
    return go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="text", text=VERTEX_LABELS,
        textfont=dict(size=18, color="black"), name="vertices", hoverinfo="skip",
        showlegend=False,
    )


# ── Overlay API ────────────────────────────────────────────────────────────────
# Everything below takes plain simplex compositions (or, for zoom zones, box
# bounds, and for needle ellipsoids, ILR-space M matrices) and returns / appends
# Plotly traces in the same tetrahedron frame as the objective cloud. No ZoMBI
# objects are required, so any caller can describe a run with numpy arrays.


def composition_to_tetra(comp: np.ndarray) -> np.ndarray:
    """Public mapping (N, 4) simplex compositions → (N, 3) tetra coords.

    The same map ``to_3d`` uses internally, exposed for callers building
    overlays. Accepts a single composition (4,) or a batch (N, 4).
    """
    comp = np.atleast_2d(np.asarray(comp, dtype=float))
    n_vert = TETRA_VERTICES.shape[0]
    if comp.shape[1] != n_vert:
        raise ValueError(
            f"expected compositions with {n_vert} components, got shape {comp.shape}"
        )
    return comp @ TETRA_VERTICES


def pared_points_trace(
    comp: np.ndarray,
    values: np.ndarray | None = None,
    *,
    recency: np.ndarray | None = None,
    cmin: float | None = None,
    cmax: float | None = None,
    size: float = PARED_SIZE,
    name: str = "pared points",
) -> go.Scatter3d:
    """Pared (deduplicated) sample points as opaque markers.

    ``values`` (optional, per-point objective) colours the markers on the same
    Viridis scale as the cloud — pass ``cmin`` / ``cmax`` to share the cloud's
    colour limits. ``recency`` (optional, any monotonic per-point order, newest
    largest) fades the points by size so the run's trajectory reads at a glance.
    """
    xyz = composition_to_tetra(comp)
    n = xyz.shape[0]

    if recency is not None:
        r = np.asarray(recency, dtype=float).ravel()
        rng = r.max() - r.min()
        norm = (r - r.min()) / rng if rng > 0 else np.ones(n)
        sizes = size * (0.6 + 0.9 * norm)   # oldest 0.6×, newest 1.5×
    else:
        sizes = size

    marker = dict(size=sizes, opacity=0.95, line=dict(color="white", width=0.5))
    if values is not None:
        values = np.asarray(values, dtype=float).ravel()
        marker.update(
            color=values, colorscale="Viridis",
            cmin=cmin if cmin is not None else float(values.min()),
            cmax=cmax if cmax is not None else float(values.max()),
            showscale=False,   # the cloud already owns the colorbar
        )
        text = [f"obj={v:.3f}" for v in values]
        hoverinfo = "text"
    else:
        marker.update(color=MAIN_LINE_COLOR)
        text = None
        hoverinfo = "name"

    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        name=name, text=text, hoverinfo=hoverinfo, marker=marker,
    )


def line_trace(
    endpoints: np.ndarray,
    *,
    name: str,
    color: str,
    dash: str | None = None,
    width: float = 6.0,
) -> go.Scatter3d:
    """A LineBO line segment from a (2, 4) array of simplex endpoints."""
    pts = composition_to_tetra(endpoints)   # (2, 3)
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
        name=name, hoverinfo="name",
        line=dict(color=color, width=width, dash=dash),
    )


def _fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` roughly-even points on the unit 2-sphere (for ILR-3 ellipsoids)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])


def needle_marker_trace(
    centers: np.ndarray,
    *,
    name: str = "needle",
    color: str = NEEDLE_MARKER_COLOR,
) -> go.Scatter3d:
    """Discovered-needle locations as red crosses (distinct from the peak diamonds)."""
    xyz = composition_to_tetra(centers)
    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        name=name, hoverinfo="name",
        marker=dict(symbol="x", color=color, size=7,
                    line=dict(color="darkred", width=1)),
    )


def needle_ellipsoid_mesh(
    center: np.ndarray,
    M: np.ndarray,
    *,
    name: str,
    color: str = NEEDLE_ELL_COLOR,
    opacity: float = NEEDLE_ELL_OPACITY,
    show_legend: bool = False,
    n: int = 600,
) -> go.Mesh3d | None:
    """Penalization ellipsoid for one needle as a translucent solid.

    ``M`` is the ILR-space matrix whose boundary is ``{u : uᵀ M u = 1}`` (same
    convention as ``_draw_needle_ellipsoid`` in the 2D plots). The boundary
    sphere is mapped ILR → composition → tetrahedron and hulled into a blob.
    """
    import torch
    from src.utils.simplex import composition_to_ilr, ilr_to_composition

    center = np.asarray(center, dtype=float).ravel()
    d = center.shape[0]
    M = np.asarray(M, dtype=float)

    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-12)
    sphere = _fibonacci_sphere(n)                              # (n, d-1) on unit 2-sphere
    u = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ sphere.T).T   # (n, d-1), uᵀ M u = 1

    c_ilr = composition_to_ilr(
        torch.as_tensor(center.reshape(1, -1), dtype=torch.float64)
    ).squeeze(0).cpu().numpy()                                 # (d-1,)
    z = c_ilr + u
    comp = ilr_to_composition(torch.as_tensor(z, dtype=torch.float64), d).cpu().numpy()
    xyz = composition_to_tetra(comp)
    return go.Mesh3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        alphahull=0, color=color, opacity=opacity, flatshading=True,
        name=name, showlegend=show_legend, hoverinfo="skip",
    )


def add_simplex_overlays(
    fig: go.Figure,
    *,
    pared_points: np.ndarray | None = None,
    pared_values: np.ndarray | None = None,
    recency: np.ndarray | None = None,
    main_line: np.ndarray | None = None,
    cache_line: np.ndarray | None = None,
    needles: np.ndarray | None = None,
    needle_ell_M: list | None = None,
    obj_cmin: float | None = None,
    obj_cmax: float | None = None,
) -> go.Figure:
    """Append ZoMBI-Hop-style overlays to a tetrahedron figure (mutates ``fig``).

    Parameters
    ----------
    pared_points : (N, 4) compositions, or None.
    pared_values : (N,) per-point objective for colour, or None.
    recency      : (N,) monotonic order (newest largest) to size-fade points, or None.
    main_line, cache_line : (2, 4) endpoint pairs, or None.
    needles      : (k, 4) needle centres, or None.
    needle_ell_M : list of ILR ``M`` matrices ((d-1, d-1)) parallel to ``needles``;
                   entries may be None to skip a single ellipsoid.
    obj_cmin, obj_cmax : shared colour limits so pared points match the cloud.

    Returns the same ``fig`` for chaining.
    """
    traces: list = []

    if needles is not None:
        needles = np.atleast_2d(np.asarray(needles, dtype=float))
        M_list = needle_ell_M or []
        first_ell = True
        for i, c in enumerate(needles):
            Mi = M_list[i] if i < len(M_list) else None
            if Mi is None:
                continue
            mesh = needle_ellipsoid_mesh(
                c, Mi, name="needle region", show_legend=first_ell,
            )
            if mesh is not None:
                traces.append(mesh)
                first_ell = False
        traces.append(needle_marker_trace(needles))

    if pared_points is not None:
        traces.append(pared_points_trace(
            pared_points, pared_values, recency=recency,
            cmin=obj_cmin, cmax=obj_cmax,
        ))

    if main_line is not None:
        traces.append(line_trace(
            main_line, name="LineBO (main)", color=MAIN_LINE_COLOR, width=7,
        ))
    if cache_line is not None:
        traces.append(line_trace(
            cache_line, name="LineBO (cache)", color=CACHE_LINE_COLOR,
            dash="dot", width=9,
        ))

    fig.add_traces(traces)
    return fig


def build_example_overlays(fn, *, seed: int = 0) -> dict:
    """Synthesize a demonstration overlay set derived from an ``Ackley`` objective.

    Places pared points (a cluster near a peak plus a scatter), a main and cache
    LineBO line crossing that peak, a zoom-zone box around it, and a needle with
    an ILR-sphere penalization ellipsoid. Returns kwargs for
    ``add_simplex_overlays``.
    """
    rng = np.random.default_rng(seed)
    centers = np.atleast_2d(np.array(fn.centers, dtype=float))
    peak = centers[0]

    # Pared points: scatter (older) then a tight cluster around the peak (newer).
    scatter = rng.dirichlet(np.ones(DIM), size=30)
    cluster = rng.dirichlet(peak * 200.0 + 1.0, size=70)
    pared = np.vstack([scatter, cluster])
    values = np.asarray(fn.predict(pared)).ravel()
    recency = np.arange(pared.shape[0], dtype=float)   # newest = cluster, plotted largest

    # Main / cache lines: segments through the peak along random tangent directions.
    def _segment(half_len: float) -> np.ndarray:
        v = rng.normal(size=DIM)
        v -= v.mean()                       # stay on the zero-sum (tangent) plane
        v /= np.linalg.norm(v)
        ends = []
        for sign in (-1.0, 1.0):
            p = np.clip(peak + sign * half_len * v, 1e-6, None)
            ends.append(p / p.sum())
        return np.vstack(ends)

    main_line = _segment(0.28)
    cache_line = _segment(0.22)

    # Needle at the peak with an isotropic ILR ellipsoid (sphere of radius r).
    r = 0.45
    needle_ell_M = [np.eye(DIM - 1) / (r ** 2)]
    needles = peak.reshape(1, -1)

    return dict(
        pared_points=pared, pared_values=values, recency=recency,
        main_line=main_line, cache_line=cache_line,
        needles=needles, needle_ell_M=needle_ell_M,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot a 4D negated-Ackley objective as a rotatable 3D point cloud."
    )
    parser.add_argument(
        "--ackley", required=True, choices=sorted(Ackley.VARIANTS),
        help="Which analytic Ackley variant to plot (lifted to the 4-element simplex).",
    )
    parser.add_argument(
        "--example", action="store_true",
        help="Stack a synthetic demonstration of the overlay API (pared points, "
             "main/cache lines, and a needle + ellipsoid) on top of "
             "the chosen --ackley landscape.",
    )
    args = parser.parse_args()

    variant = args.ackley
    fn = Ackley(variant, dim=DIM)

    # Pre-compute objective clouds at each resolution for the slider.
    resolutions = GRID_RESOLUTIONS
    clouds_data = []
    obj_min_global, obj_max_global = np.inf, -np.inf
    for res in resolutions:
        comp = build_simplex_lattice(res)
        obj = fn.predict(comp)
        xyz = to_3d(comp)
        obj_min_global = min(obj_min_global, float(obj.min()))
        obj_max_global = max(obj_max_global, float(obj.max()))
        clouds_data.append((xyz, obj, comp))

    # Build one cloud trace per resolution; only the default (GRID_N) is visible.
    default_idx = resolutions.index(GRID_N) if GRID_N in resolutions else len(resolutions) - 1
    cloud_traces = []
    for i, (xyz, obj, comp) in enumerate(clouds_data):
        hover = [
            f"x=[{a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}]<br>obj={v:.2f}"
            for (a, b, c, d), v in zip(comp, obj)
        ]
        cloud_traces.append(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
            name="objective", text=hover, hoverinfo="text",
            visible=(i == default_idx),
            marker=dict(
                color=obj, colorscale="Viridis",
                cmin=obj_min_global, cmax=obj_max_global,
                size=MARKER_SIZE, opacity=MARKER_OPACITY,
                showscale=True, colorbar=dict(title="Objective"),
            ),
        ))

    n_clouds = len(cloud_traces)

    # Known analytic peaks, mapped into the same 3D frame.
    peaks = np.array(fn.centers)
    peaks_xyz = to_3d(peaks)
    peaks_trace = go.Scatter3d(
        x=peaks_xyz[:, 0], y=peaks_xyz[:, 1], z=peaks_xyz[:, 2], mode="markers",
        name="known peak",
        marker=dict(symbol="diamond", color="red", size=6,
                    line=dict(color="white", width=1)),
        hoverinfo="name",
    )

    fixed_traces = [tetra_edges_trace(), vertex_labels_trace(), peaks_trace]
    fig = go.Figure(data=cloud_traces + fixed_traces)

    # Build slider steps: each step shows exactly one cloud trace and all fixed traces.
    n_total = len(fig.data)
    steps = []
    for i, res in enumerate(resolutions):
        n_pts = clouds_data[i][0].shape[0]
        vis = [False] * n_total
        vis[i] = True  # show this resolution's cloud
        for j in range(n_clouds, n_total):
            vis[j] = True  # always show fixed traces
        steps.append(dict(
            method="restyle",
            args=[{"visible": vis}],
            label=f"{res} ({n_pts:,} pts)",
        ))

    fig.update_layout(
        title=f"Negated Ackley ('{variant}') on the 4-simplex as a 3D point cloud",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data",
        ),
        legend=dict(x=0.0, y=1.0),
        width=FIG_W, height=FIG_H,
        sliders=[dict(
            active=default_idx,
            currentvalue=dict(prefix="Grid resolution: "),
            pad=dict(t=40),
            steps=steps,
        )],
    )

    if args.example:
        add_simplex_overlays(
            fig, obj_cmin=obj_min_global, obj_cmax=obj_max_global,
            **build_example_overlays(fn),
        )
        # Update slider steps to keep overlay traces visible too.
        n_total_new = len(fig.data)
        new_steps = []
        for i, res in enumerate(resolutions):
            n_pts = clouds_data[i][0].shape[0]
            vis = [False] * n_total_new
            vis[i] = True
            for j in range(n_clouds, n_total_new):
                vis[j] = True
            new_steps.append(dict(
                method="restyle",
                args=[{"visible": vis}],
                label=f"{res} ({n_pts:,} pts)",
            ))
        fig.layout.sliders[0].steps = new_steps
        print("  --example: added pared points, main/cache lines, "
              "and a needle + ellipsoid overlay.")

    out_html = HERE / "point_cloud_plot.html"
    fig.write_html(out_html, include_plotlyjs="cdn", auto_open=False)
    print(f"Wrote {out_html}")
    webbrowser.open(out_html.as_uri())


if __name__ == "__main__":
    main()
