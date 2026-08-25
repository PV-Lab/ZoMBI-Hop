"""
visualization/min_zoom_box.py
=============================
What the **smallest zoom box ZoMBI can ask for** actually covers on a simplex.

    min_zoom_box_3d.png    ternary triangle   (d=3, the 2-simplex)
    min_zoom_box_4d.png    3D tetrahedron     (d=4, the 3-simplex)

The mechanic
------------
``DataHandler.min_box_width`` floors every axis of a zoom box at

    input_noise_threshold_mult x input_noise

and ``_apply_min_box_width`` widens any narrower axis about its centre (then
translates it back inside the global domain). The floor exists because a box
narrower than the printer's input noise asks for a resolution the hardware does
not have: the requested compositions land inside the box, but the *realised*
ones — the only ones the GP ever sees — scatter outside it.

That floor is a box in ``[0,1]^d``. The samples, however, live on the
composition simplex ``{x : sum(x) = 1, x >= 0}``, and it is the INTERSECTION of
the two that the acquisition actually searches. This script draws that
intersection, because the floor's width in composition units says very little
on its own about how much of the reachable space a "fully zoomed in" run is
still spread over.

Why now
-------
Two hyperparameters moved together, and they multiply:

    before:  0.5 x 0.064 = 0.032     (mult 0.5, noise from data/2nd_real_run.db)
    after:   3.0 x 0.128 = 0.384     (mult 3.0, noise from runs/run_39af)

so the minimum box width grew **12x**. On the simplex that is not a 12x change
in footprint — the intersection is a polytope whose size grows much faster than
the width — which is exactly what the figures are for.

What each figure shows
----------------------
Top row is the current floor, bottom row the pre-``a2deba7`` one, at three
placements of the same minimum-width box:

  * **vertex**   — box centred on a pure-component corner. This is the SMALLEST
    footprint the floor can produce: the intersection is a scaled copy of the
    simplex with edge ratio ``w``, i.e. a fraction ``w^(d-1)``.
  * **centroid** — box centred on the barycentre, the LARGEST footprint, and the
    placement a run converging to the middle of composition space ends at.

"Smallest" is over the boxes a run can actually produce, which is the restriction
that makes the claim true and is checked at runtime by ``search_min_fraction``. A
box floating anywhere in ``[0,1]^d`` could be slid until it merely grazed the
simplex, with a footprint arbitrarily close to zero; the floor is applied to the
bounding box of measured top-m points, which lie on the simplex, so the reachable
family is the boxes CENTRED on a simplex point.

Those two bracket every other placement, so the wide bottom panel sweeps the
floor width continuously and shades the band between them, with the before/after
widths marked — the two campaigns read off one axis instead of being compared
across figures.

Percentages are fractions of the simplex's own (d-1)-dimensional measure — area
for d=3, volume for d=4 — not of the enclosing box or cube.

The diagrams follow the project's conventions: the ternary matches plot_run.py's
``comp_to_xy`` (unit triangle, same corner order) and the tetrahedron matches its
``TETRA_VERTS`` corner mapping. Both are mirrored locally rather than imported,
since plot_run.py pulls in sklearn/dash at module scope.

Run:
  uv run python visualization/min_zoom_box.py
  # one dimension only, or a hypothetical multiplier:
  uv run python visualization/min_zoom_box.py --dims 4
  uv run python visualization/min_zoom_box.py --mult 1.5
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # repo root, so src is importable

# Single source of truth for the two factors that set the floor.
from src.default_hparams import (  # noqa: E402
    DEFAULT_HPARAMS,
    DEFAULT_INPUT_NOISE,
)

OUT_TEMPLATE = "min_zoom_box_{dim}d.png"

# Current floor, straight from the canonical defaults.
NEW_MULT = float(DEFAULT_HPARAMS["input_noise_threshold_mult"])
NEW_NOISE = float(DEFAULT_INPUT_NOISE)

# The pre-``a2deba7`` pair, hard-coded because it no longer exists anywhere in
# the tree: the multiplier was 0.5 and the noise 0.064, the latter inferred from
# data/2nd_real_run.db by fitting a line through each print line's realised
# endpoints (a fit that absorbs the requested-vs-realised offset and so
# under-reports by ~3x — the reason for the change).
OLD_MULT, OLD_NOISE = 0.5, 0.064

_SQRT3_2 = np.sqrt(3) / 2

# Vertices of a regular (unit-edge) tetrahedron; composition column i maps to
# corner TETRA_VERTS[i] (mirrors plot_run.TETRA_VERTS / comp_to_xyz).
TETRA_VERTS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, _SQRT3_2, 0.0],
    [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
], dtype=float)
TETRA_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# Corner labels, matching plot_run.DIM_LABELS' physical identities.
LABELS = {
    3: ["MAPbBr3", "MAPbI3", "FAPbI3"],
    4: ["MAPbBr3", "MAPbI3", "FAPbI3", "CsPbI3"],
}

NEW_COLOR = "#c1121f"
OLD_COLOR = "#0353a4"
# The two extremes of where a fixed-width box can sit. Everything in between
# falls inside the band they bound, so a third sample placement adds no
# information — see the module docstring.
PLACEMENTS = ("vertex", "centroid")


# ── simplex coordinates (mirror plot_run.py) ──────────────────────────────────

def comp_to_xy(comp: np.ndarray) -> np.ndarray:
    """(N, 3) simplex compositions -> (N, 2) Cartesian ternary coordinates."""
    p = np.atleast_2d(np.asarray(comp, dtype=float))
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def comp_to_xyz(comp: np.ndarray) -> np.ndarray:
    """(N, 4) simplex compositions -> (N, 3) Cartesian tetrahedron coordinates."""
    p = np.atleast_2d(np.asarray(comp, dtype=float))
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return p @ TETRA_VERTS


def to_cartesian(comp: np.ndarray, d: int) -> np.ndarray:
    return comp_to_xy(comp) if d == 3 else comp_to_xyz(comp)


# ── the box, and its footprint on the simplex ─────────────────────────────────

def min_box_width(mult: float = NEW_MULT, noise: float = NEW_NOISE) -> float:
    """Mirror of ``DataHandler.min_box_width``: the per-axis floor, or 0 if disabled."""
    floor = float(mult) * float(noise)
    return floor if floor > 0.0 else 0.0


def box_simplex_vertices(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Exact vertices of ``{x : sum(x) = 1, lo <= x <= hi}``.

    On the simplex hyperplane one coordinate is determined by the other d-1, so
    every vertex of the intersection has d-1 coordinates sitting on a bound and
    the remaining one taking up the slack. Enumerating those candidates — choose
    the free index, then each fixed coordinate's lo/hi — and keeping the feasible
    ones is exact and, at d <= 4 (at most 32 candidates), far cheaper and more
    robust than a general halfspace intersection, which needs a strictly interior
    point that may not exist when the polytope is degenerate.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    d = lo.size
    tol = 1e-12
    verts: list[np.ndarray] = []
    for free in range(d):
        fixed = [i for i in range(d) if i != free]
        for choice in itertools.product((0, 1), repeat=d - 1):
            x = np.empty(d)
            for i, c in zip(fixed, choice):
                x[i] = hi[i] if c else lo[i]
            x[free] = 1.0 - x[fixed].sum()
            if lo[free] - tol <= x[free] <= hi[free] + tol:
                verts.append(np.clip(x, lo, hi))
    if not verts:
        return np.empty((0, d))
    V = np.array(verts)
    # Distinct vertices only; the enumeration hits each one once per set of
    # active constraints, so a corner where more than d-1 bounds are tight
    # appears several times.
    return np.unique(np.round(V, 10), axis=0)


def simplex_fraction(lo: np.ndarray, hi: np.ndarray, d: int) -> float:
    """Fraction of the simplex's (d-1)-measure covered by the box.

    Computed in the Cartesian embedding, which is affine, so the ratio is exact
    regardless of the (arbitrary) corner placement. A polytope of lower dimension
    than d-1 has measure zero and Qhull refuses it — that is the correct answer,
    not an error.
    """
    V = box_simplex_vertices(lo, hi)
    if len(V) < d:
        return 0.0
    pts = to_cartesian(V, d)
    full = to_cartesian(np.eye(d), d)
    try:
        part_measure = ConvexHull(pts).volume
    except QhullError:
        return 0.0
    return float(part_measure / ConvexHull(full).volume)


# ── placements of a fixed-width box ───────────────────────────────────────────

def place_box_at(center: np.ndarray, width: float, d: int) -> tuple[np.ndarray, np.ndarray]:
    """A ``width``-wide box centred on ``center``, shifted back inside ``[0,1]^d``.

    Mirrors ``DataHandler._apply_min_box_width``: the widening is symmetric about
    the centre, and a box that overhangs the domain is TRANSLATED back in rather
    than truncated, so it keeps its full floor width wherever the domain allows.
    """
    w = min(float(width), 1.0)
    lo = np.asarray(center, dtype=float) - w / 2.0
    lo = lo + np.clip(0.0 - lo, 0.0, None)          # shift up off the lower wall
    lo = lo - np.clip((lo + w) - 1.0, 0.0, None)    # shift down off the upper wall
    return np.clip(lo, 0.0, 1.0), np.clip(lo + w, 0.0, 1.0)


def place_box(width: float, d: int, where: str) -> tuple[np.ndarray, np.ndarray]:
    """The minimum-width box at one of the two extreme reachable placements.

    ``vertex`` centres it on the pure-component corner ``e_0``; ``centroid`` on
    the barycentre. Both centres are points OF the simplex, which is the whole
    constraint that makes these the extremes — see ``search_min_fraction``.
    """
    if where == "vertex":
        center = np.zeros(d)
        center[0] = 1.0
    elif where == "centroid":
        center = np.full(d, 1.0 / d)
    else:
        raise ValueError(f"unknown placement {where!r}")
    return place_box_at(center, width, d)


def search_min_fraction(width: float, d: int, n: int = 20000,
                        seed: int = 0) -> tuple[float, np.ndarray]:
    """Smallest simplex footprint a ``width``-wide box can reach, by random search.

    The vertex placement is *claimed* above to be the minimum; this checks it
    rather than trusting it. Returns the best fraction found and the centre that
    achieved it, so a caller can flag a placement that beats the corner.

    Centres are drawn from the SIMPLEX, not from the cube, and that restriction is
    the point. A box floating free in ``[0,1]^d`` can be slid until it merely
    grazes the simplex, making the intersection arbitrarily close to zero — but no
    such box is reachable here. The floor is applied to the axis-aligned bounding
    box of the run's top-m points, which are measured compositions and therefore
    lie ON the simplex, and the widening is about that box's own centre. So the
    reachable family is the boxes centred on a simplex point, and over that family
    the corner really is the floor.
    """
    rng = np.random.default_rng(seed)
    centers = np.vstack([rng.dirichlet(np.ones(d), size=n), np.eye(d)])
    best_frac, best_c = np.inf, None
    for c in centers:
        f = simplex_fraction(*place_box_at(c, width, d), d)
        if 0.0 < f < best_frac:
            best_frac, best_c = f, c
    return best_frac, best_c


# ── drawing ───────────────────────────────────────────────────────────────────

def _draw_ternary(ax, lo, hi, color, labels) -> None:
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "-", color="0.25", lw=1.4)
    V = box_simplex_vertices(lo, hi)
    if len(V) >= 3:
        P = comp_to_xy(V)
        # Order the hull's own vertices by angle so the polygon is drawn convex;
        # ConvexHull.vertices is already CCW in 2D, but the raw enumeration is not.
        hull = ConvexHull(P)
        ring = P[hull.vertices]
        ax.fill(ring[:, 0], ring[:, 1], color=color, alpha=0.45, lw=0, zorder=2)
        ax.plot(np.append(ring[:, 0], ring[0, 0]),
                np.append(ring[:, 1], ring[0, 1]),
                color=color, lw=1.6, zorder=3)
    corners = comp_to_xy(np.eye(3))
    mid = corners.mean(axis=0)
    for (x, y), name in zip(corners, labels):
        # Push each label along the centroid->corner ray so it never sits on an
        # edge or on the shaded polytope.
        v = np.array([x, y]) - mid
        v = v / max(1e-12, np.linalg.norm(v))
        ax.annotate(name, (x + 0.10 * v[0], y + 0.09 * v[1]), ha="center",
                    va="center", fontsize=7, color="0.35", bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6))
    ax.set_xlim(-0.20, 1.20)
    ax.set_ylim(-0.20, _SQRT3_2 + 0.22)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_tetra(ax, lo, hi, color, labels) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for i, j in TETRA_EDGES:
        seg = TETRA_VERTS[[i, j]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="0.35", lw=1.0, zorder=1)
    V = box_simplex_vertices(lo, hi)
    if len(V) >= 4:
        P = comp_to_xyz(V)
        try:
            hull = ConvexHull(P)
            faces = [P[s] for s in hull.simplices]
            poly = Poly3DCollection(faces, alpha=0.42, facecolor=color,
                                    edgecolor=color, linewidths=0.5)
            ax.add_collection3d(poly)
        except QhullError:
            # Degenerate (measure-zero) intersection — the wireframe alone is the
            # honest picture; a flat patch would imply an area that is not there.
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=8, color=color)
    mid = TETRA_VERTS.mean(axis=0)
    for v, name in zip(TETRA_VERTS, labels):
        u = v - mid
        u = u / max(1e-12, np.linalg.norm(u))
        q = v + 0.42 * u
        # A tetrahedron always projects one corner inside its own silhouette from
        # any camera, so the label gets a background rather than a lucky angle.
        ax.text(q[0], q[1], q[2], name, fontsize=6.5, color="0.35",
                ha="center", va="center", bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6))
    ax.set_xlim(-0.34, 1.34)
    ax.set_ylim(-0.34, _SQRT3_2 + 0.34)
    ax.set_zlim(-0.30, np.sqrt(6) / 3 + 0.30)
    ax.set_box_aspect((1, _SQRT3_2, np.sqrt(6) / 3))
    ax.view_init(elev=18, azim=-62)
    ax.set_axis_off()


def draw_simplex(ax, lo, hi, color, d, labels) -> None:
    (_draw_ternary if d == 3 else _draw_tetra)(ax, lo, hi, color, labels)


def coverage_curve(d: int, n: int = 90) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Covered fraction vs box width, one series per placement."""
    widths = np.linspace(0.005, 1.0, n)
    out = {}
    for where in PLACEMENTS:
        out[where] = np.array([simplex_fraction(*place_box(w, d, where), d)
                               for w in widths])
    return widths, out


def render(path: Path, d: int, *, mult: float, noise: float,
           verify: bool = True) -> None:
    import matplotlib.pyplot as plt

    labels = LABELS[d]
    w_new = min_box_width(mult, noise)
    w_old = min_box_width(OLD_MULT, OLD_NOISE)
    rows = [("current", w_new, mult, noise, NEW_COLOR),
            ("pre-a2deba7", w_old, OLD_MULT, OLD_NOISE, OLD_COLOR)]

    proj = {"projection": "3d"} if d == 4 else {}
    fig = plt.figure(figsize=(9.0, 9.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.05],
                          hspace=0.22, wspace=0.02)

    for r, (tag, w, m, nz, color) in enumerate(rows):
        for c, where in enumerate(PLACEMENTS):
            ax = fig.add_subplot(gs[r, c], **proj)
            lo, hi = place_box(w, d, where)
            frac = simplex_fraction(lo, hi, d)
            draw_simplex(ax, lo, hi, color, d, labels)
            # Each panel names its own row. A separate row header — above the
            # panels or rotated in the margin — collides with these titles or with
            # the neighbouring row at any fixed offset, and there are only four.
            title = (f"{tag}: {m:g} x {nz:g} = {w:.3f} wide\n"
                     f"{where} — {frac * 100:.2g}% of simplex")
            if frac < 3e-4:
                # At this scale the polytope is well under a pixel. Saying so beats
                # letting an empty-looking diagram read as a failed render.
                title += "\n(too small to see — under a pixel here)"
            ax.set_title(title, fontsize=9, color=color, pad=2)


    # ── sweep panel ──
    ax = fig.add_subplot(gs[2, :])
    widths, curves = coverage_curve(d)
    ax.fill_between(widths, 100 * curves["vertex"], 100 * curves["centroid"],
                    color="0.55", alpha=0.22, lw=0,
                    label="every other placement lands in here")
    ax.plot(widths, 100 * curves["centroid"], "-", color="0.25", lw=1.6,
            label="centroid placement (largest footprint)")
    ax.plot(widths, 100 * curves["vertex"], "--", color="0.25", lw=1.6,
            label="vertex placement (smallest footprint)")
    for w, color, tag in ((w_old, OLD_COLOR, "pre-a2deba7"), (w_new, NEW_COLOR, "current")):
        ax.axvline(w, color=color, lw=1.4, alpha=0.9)
        ax.annotate(f"{tag}\nw = {w:.3f}", (w, 100), xytext=(4, -2),
                    textcoords="offset points", fontsize=8, color=color,
                    ha="left", va="top")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylim(0, 130)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("minimum zoom-box width (composition units, every axis)", fontsize=9)
    ax.set_ylabel("% of simplex covered\n(symlog)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    growth = (simplex_fraction(*place_box(w_new, d, "vertex"), d) /
              max(1e-30, simplex_fraction(*place_box(w_old, d, "vertex"), d)))
    fig.suptitle(
        f"Smallest zoom box ZoMBI can request, on the {d - 1}-simplex (d={d})\n"
        f"width floor grew {w_new / w_old:.0f}x "
        f"({w_old:.3f} -> {w_new:.3f}); its smallest simplex footprint grew "
        f"{growth:,.0f}x",
        fontsize=12, y=0.985)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")

    for tag, w, *_ in rows:
        line = f"    d={d} {tag:12s} w={w:.4f}  " + "  ".join(
            f"{p}={simplex_fraction(*place_box(w, d, p), d) * 100:.4g}%"
            for p in PLACEMENTS)
        if verify:
            found, _ = search_min_fraction(w, d)
            corner = simplex_fraction(*place_box(w, d, "vertex"), d)
            flag = "" if found >= corner - 1e-9 else f"  !! random search found {found * 100:.4g}%"
            line += f"   [min over 20k random placements: {found * 100:.4g}%{flag}]"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static PNGs of the minimum zoom box's footprint on the simplex.")
    parser.add_argument("--dims", default="3,4",
                        help="Comma-separated dimensions to render (default: 3,4).")
    parser.add_argument("--mult", type=float, default=NEW_MULT,
                        help=f"input_noise_threshold_mult (default: {NEW_MULT:g}).")
    parser.add_argument("--noise", type=float, default=NEW_NOISE,
                        help=f"input_noise (default: {NEW_NOISE:g}).")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the random search that checks the vertex placement "
                             "really is the smallest footprint.")
    parser.add_argument("--out-dir", default=str(_HERE),
                        help="Directory for the PNGs (default: alongside this script).")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for token in args.dims.split(","):
        d = int(token.strip())
        if d not in LABELS:
            raise SystemExit(f"No simplex diagram for d={d} (have {sorted(LABELS)})")
        render(out_dir / OUT_TEMPLATE.format(dim=d), d,
               mult=args.mult, noise=args.noise, verify=not args.no_verify)


if __name__ == "__main__":
    main()
