"""
visualization/point_cloud_fig.py
================================
A 3D illustrative point cloud made of two overlapping "funnels".

Both funnels start from nearly the same wide base (their base clouds overlap)
and taper as they sweep off toward the right of the figure in two different
directions. "Funnel" is meant loosely: the points are randomly scattered and
semi-transparent, so the shape only reads as a funnel in aggregate.

  * Funnel A is shaded dark grey (base) -> light grey (tip).
  * Funnel B is shaded blue (base) -> yellow (tip).
  * A red star marks the tip of each funnel.

Purely illustrative — no run data involved. The axes are labelled
"Objective 1/2/3" with no tick values, in the spirit of ``triangles.py``.

Usage
-----
  python visualization/point_cloud_fig.py
  python visualization/point_cloud_fig.py --n 3000 --seed 1 --out cloud.png

Flags
-----
  --n N      Points per funnel (default: 2200).
  --seed S   RNG seed (default: 0).
  --out PATH Save to PATH instead of showing the window (default: show).
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Base of both funnels: same neighbourhood, offset just enough that the two
# clouds interpenetrate near the base rather than sitting on top of each other.
BASE_COLOR = np.array([0.15, 0.45, 0.55])
BASE_GREY = np.array([0.05, 0.60, 0.42])

# Tips: both toward the right (+x), fanning sharply apart in z — the colour
# funnel climbs, the grey one dives, so the angle between them is wide.
TIP_COLOR = np.array([1.00, 0.18, 1.15])
TIP_GREY = np.array([0.92, 0.86, -0.25])

# Bend of each funnel's centre line: offsets applied to the two interior
# control points of a cubic Bezier from base to tip. Opposite signs give an
# S-shaped sweep rather than a straight run.
BEND_COLOR = (np.array([-0.16, -0.38, 0.34]), np.array([0.22, 0.32, -0.26]))
BEND_GREY = (np.array([-0.10, 0.36, 0.30]), np.array([0.20, -0.28, -0.24]))

BASE_RADIUS = 0.34   # spread of the cloud at the base
TIP_RADIUS = 0.14    # spread at the tip (still a broad cloud, not a needle)
TAPER = 0.85         # <1 keeps the funnel wide well past the base

# Point count that holds the *density* of the cloud fixed at the radii above.
# Cross-sectional area goes as radius**2, so if you change the radii, scale
# this by the same factor squared or the cloud will thicken/thin visually.
DEFAULT_N = 1450

GREY_CMAP = LinearSegmentedColormap.from_list("grey_funnel", ["#7a7a7a", "#d9d9d9"])
BLUE_YELLOW_CMAP = LinearSegmentedColormap.from_list(
    "blue_yellow_funnel", ["#1f3fd6", "#2f9fd0", "#5fc48a", "#ffd633"]
)


def _bezier(ctrl: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cubic Bezier positions and unit tangents at parameters ``t``.

    ``ctrl`` is (4, 3); returns ``(pos, tangent)``, each (len(t), 3).
    """
    p0, p1, p2, p3 = ctrl
    s = (1.0 - t)[:, None]
    tt = t[:, None]
    pos = (s**3) * p0 + 3 * (s**2) * tt * p1 + 3 * s * (tt**2) * p2 + (tt**3) * p3
    tan = 3 * (s**2) * (p1 - p0) + 6 * s * tt * (p2 - p1) + 3 * (tt**2) * (p3 - p2)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    return pos, tan


def funnel_cloud(
    base: np.ndarray,
    tip: np.ndarray,
    bend: tuple[np.ndarray, np.ndarray],
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Random points filling a curved funnel from ``base`` to ``tip``.

    The centre line is a cubic Bezier whose interior control points are nudged
    off the straight base->tip line by ``bend``, so the cloud sweeps and bends
    instead of running dead straight. Returns ``(pts, t)`` where ``t`` in
    [0, 1] is each point's progress along the funnel and doubles as the colour
    value.
    """
    straight = tip - base
    ctrl = np.array([
        base,
        base + straight / 3.0 + bend[0],
        base + 2.0 * straight / 3.0 + bend[1],
        tip,
    ])

    # Bunch points toward the base so the wide end looks denser than the neck.
    t = rng.uniform(0.0, 1.0, n) ** 1.6
    pos, tan = _bezier(ctrl, t)
    radius = TIP_RADIUS + (BASE_RADIUS - TIP_RADIUS) * (1.0 - t) ** TAPER

    # Per-point frame perpendicular to the local tangent (the tangent turns as
    # the curve bends, so the cross-sections follow the sweep).
    helper = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    steep = np.abs(tan[:, 2]) > 0.9
    helper[steep] = np.array([0.0, 1.0, 0.0])
    u = np.cross(tan, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(tan, u)

    # Gaussian (not uniform-disc) offsets keep the edges soft and scattered.
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    r = np.abs(rng.normal(0.0, 0.75, n)) * radius
    offset = (r * np.cos(theta))[:, None] * u + (r * np.sin(theta))[:, None] * v

    # Round off the base into a dome instead of a flat disc: points near the
    # base are pushed back along the tangent by the height of a hemisphere of
    # radius BASE_RADIUS at their own radial distance, so on-axis points bulge
    # out furthest and rim points stay put. The (1-t)**3 weight confines the
    # cap to the base end and leaves the taper untouched.
    cap = np.sqrt(np.maximum(0.0, BASE_RADIUS**2 - r**2))
    cap_weight = (1.0 - t) ** 3

    # A little jitter along the curve too, so the tip is a fuzzy cluster.
    along = -cap_weight * cap + rng.normal(0.0, 0.02, n)
    pts = pos + along[:, None] * tan + offset
    return pts, t


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="Points per funnel.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed.")
    ap.add_argument("--out", type=str, default=None,
                    help="Save to this path instead of showing a window.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    pts_grey, t_grey = funnel_cloud(BASE_GREY, TIP_GREY, BEND_GREY, args.n, rng)
    pts_col, t_col = funnel_cloud(BASE_COLOR, TIP_COLOR, BEND_COLOR, args.n, rng)

    fig = plt.figure(figsize=(8, 7))
    # computed_zorder=False: honour our explicit zorder instead of matplotlib's
    # depth sort, so the tip stars always draw over both clouds.
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # Grey funnel first, blue->yellow on top; both faint enough to show overlap.
    ax.scatter(pts_grey[:, 0], pts_grey[:, 1], pts_grey[:, 2], c=t_grey,
               cmap=GREY_CMAP, s=22, alpha=0.30, edgecolors="none",
               depthshade=False, zorder=1)
    ax.scatter(pts_col[:, 0], pts_col[:, 1], pts_col[:, 2], c=t_col,
               cmap=BLUE_YELLOW_CMAP, s=22, alpha=0.30, edgecolors="none",
               depthshade=False, zorder=2)

    for tip in (TIP_GREY, TIP_COLOR):
        ax.scatter(*tip, marker="*", s=520, c="red", edgecolors="k",
                   linewidths=0.6, depthshade=False, zorder=6)

    ax.set_xlabel("Objective 1", labelpad=6)
    ax.set_ylabel("Objective 2", labelpad=6)
    ax.set_zlabel("Objective 3", labelpad=6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([])
        axis.set_ticks([])
    # Clip the axes to the bulk of the cloud (a few sparse outliers fall
    # outside) so the funnels fill the box instead of floating in whitespace.
    allpts = np.vstack([pts_grey, pts_col])
    lo = np.percentile(allpts, 1.0, axis=0)
    hi = np.percentile(allpts, 99.0, axis=0)
    pad = 0.04 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    # zoom>1 fills the frame; much past this the axis labels fall off the edge.
    ax.set_box_aspect((1, 1, 1), zoom=1.05)
    # +x tilted toward the viewer (out of the screen) and to the right, so the
    # funnels read as coming forward without needing to rotate the window.
    ax.view_init(elev=18, azim=-58)

    # Leave room on the right for the z label: matplotlib's tight bbox does not
    # account for 3D axis labels, so at this zoom they get cropped otherwise.
    fig.subplots_adjust(left=0.0, right=0.88, bottom=0.03, top=1.0)

    if args.out:
        # No bbox_inches="tight": it crops to the axes and would clip the z
        # label; the margins set above already keep whitespace down.
        fig.savefig(args.out, dpi=200)
        print(f"wrote {args.out}  ({len(pts_grey)} + {len(pts_col)} points)")
    else:
        plt.show()


if __name__ == "__main__":
    main()
