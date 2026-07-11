"""
visualization/triangles.py
===========================
Three point-cloud shapes rendered as semi-transparent blue dots:

  1. A filled 2D triangle on a regular grid.
  2. A solid 3D triangular pyramid (tetrahedron) on a regular grid.
  3. A swirly, non-uniform random cloud (a noisy 3D spiral).

Purely illustrative — no run data involved. Each panel is a matplotlib
scatter of uniformly/grid-placed (panels 1-2) or randomly-placed (panel 3)
points.

Usage
-----
  conda activate zombi-hop
  python visualization/triangles.py
  python visualization/triangles.py --density 40 --seed 0 --out triangles.png

Flags
-----
  --density N   Grid resolution per axis for the two grid shapes (default: 36).
  --n-cloud N   Number of points in the swirly cloud (default: 4000).
  --seed S      RNG seed for the cloud (default: 0).
  --out PATH    Save to PATH instead of showing the window (default: show).
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

BLUE = "#1f6fd6"


def triangle_grid(density: int) -> np.ndarray:
    """Regular-grid points inside the 2D triangle (0,0)-(1,0)-(0.5,1)."""
    xs = np.linspace(0.0, 1.0, density)
    ys = np.linspace(0.0, 1.0, density)
    gx, gy = np.meshgrid(xs, ys)
    px, py = gx.ravel(), gy.ravel()
    # Triangle with apex at (0.5, 1): edges y <= 2x and y <= 2(1-x).
    inside = (py <= 2.0 * px) & (py <= 2.0 * (1.0 - px))
    return np.column_stack([px[inside], py[inside]])


def tetra_grid(density: int) -> np.ndarray:
    """Regular-grid points inside a triangular pyramid (tetrahedron)."""
    # Base triangle in z=0 plane, apex at the top. Use barycentric-style
    # constraints on a cubic grid and keep points inside the solid.
    lin = np.linspace(0.0, 1.0, density)
    gx, gy, gz = np.meshgrid(lin, lin, lin)
    px, py, pz = gx.ravel(), gy.ravel(), gz.ravel()
    # Tetrahedron with vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1):
    # x>=0, y>=0, z>=0, x+y+z<=1.
    inside = (px + py + pz) <= 1.0
    return np.column_stack([px[inside], py[inside], pz[inside]])


def swirly_cloud(n: int, rng: np.random.Generator) -> np.ndarray:
    """A non-uniform, swirly random 3D cloud (noisy vertical spiral)."""
    # Angle grows with height; radius wanders, giving an uneven swirl.
    t = rng.uniform(0.0, 1.0, n) ** 0.7          # bunch points near the base
    theta = t * 6.0 * np.pi + rng.normal(0, 0.35, n)
    radius = (0.15 + 0.85 * t) * rng.uniform(0.4, 1.0, n)
    x = radius * np.cos(theta) + rng.normal(0, 0.08, n)
    y = radius * np.sin(theta) + rng.normal(0, 0.08, n)
    z = t + rng.normal(0, 0.05, n)
    return np.column_stack([x, y, z])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--density", type=int, default=36,
                    help="Grid resolution per axis for the grid shapes.")
    ap.add_argument("--n-cloud", type=int, default=4000,
                    help="Number of points in the swirly cloud.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the cloud.")
    ap.add_argument("--out", type=str, default=None,
                    help="Save to this path instead of showing a window.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    tri = triangle_grid(args.density)
    tet = tetra_grid(args.density)
    cloud = swirly_cloud(args.n_cloud, rng)

    fig = plt.figure(figsize=(15, 5))

    # 1. Flat triangle.
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(tri[:, 0], tri[:, 1], s=12, c=BLUE, alpha=0.35, edgecolors="none")
    ax1.set_title("Triangle")
    ax1.set_aspect("equal")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # 2. Triangular pyramid.
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(tet[:, 0], tet[:, 1], tet[:, 2], s=10, c=BLUE,
                alpha=0.30, edgecolors="none")
    ax2.set_title("Triangular pyramid")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    # 3. Swirly random cloud.
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=8, c=BLUE,
                alpha=0.25, edgecolors="none")
    ax3.set_title("Swirly cloud")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")

    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"wrote {args.out}  "
              f"(triangle={len(tri)}, pyramid={len(tet)}, cloud={len(cloud)} pts)")
    else:
        plt.show()


if __name__ == "__main__":
    main()
