"""
visualization/triangles.py
===========================
Three point-cloud shapes rendered as semi-transparent blue dots:

  1. A filled 2D triangle (equilateral 3-simplex) on a barycentric grid.
  2. A solid regular tetrahedron (4-simplex) on a barycentric grid.
  3. A swirly, non-uniform random cloud (a noisy 3D spiral).

Purely illustrative — no run data involved, and no axes, grid lines, or
ticks. The triangle and tetrahedron use the same simplex geometry as the
ternary / point-cloud viewers in ``synthetic_data/plot_ensemble.py`` (an
equilateral triangle and the regular tetrahedron ``TETRA_VERTICES``), so
their side-length ratios and angles match those plots.

Usage
-----
  python visualization/triangles.py
  python visualization/triangles.py --density 40 --seed 0 --out triangles.png

Flags
-----
  --density N   Barycentric resolution per edge for the two simplex shapes
                (default: 36).
  --n-cloud N   Number of points in the swirly cloud (default: 4000).
  --seed S      RNG seed for the cloud (default: 0).
  --out PATH    Save to PATH instead of showing the window (default: show).
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

BLUE = "#1f6fd6"

# Equilateral triangle (3-simplex): base [0,1] on the x-axis, apex centred.
TRI_VERTICES = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, np.sqrt(3.0) / 2.0],
])

# Regular tetrahedron (4-simplex): the cube-corner vertices used by the
# simplex viewers in synthetic_data/plot_ackley.py, recentred on the origin.
TETRA_VERTICES = np.array([
    [1.0,  1.0,  1.0],
    [1.0, -1.0, -1.0],
    [-1.0,  1.0, -1.0],
    [-1.0, -1.0,  1.0],
])
TETRA_VERTICES = TETRA_VERTICES - TETRA_VERTICES.mean(axis=0)


def _simplex_lattice(density: int, n_vertices: int) -> np.ndarray:
    """Barycentric lattice of an ``n_vertices``-simplex at edge resolution
    ``density``, returned as (N, n_vertices) weights summing to 1."""
    if n_vertices == 3:
        pts = [
            (i, j, density - i - j)
            for i in range(density + 1)
            for j in range(density + 1 - i)
        ]
    elif n_vertices == 4:
        pts = [
            (i, j, k, density - i - j - k)
            for i in range(density + 1)
            for j in range(density + 1 - i)
            for k in range(density + 1 - i - j)
        ]
    else:
        raise ValueError("n_vertices must be 3 or 4")
    return np.array(pts, dtype=float) / density


def triangle_grid(density: int) -> np.ndarray:
    """Grid points filling the equilateral triangle ``TRI_VERTICES``."""
    return _simplex_lattice(density, 3) @ TRI_VERTICES


def tetra_grid(density: int) -> np.ndarray:
    """Grid points filling the regular tetrahedron ``TETRA_VERTICES``."""
    return _simplex_lattice(density, 4) @ TETRA_VERTICES


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
    ap.add_argument("--density", type=int, default=11,
                    help="Barycentric resolution per edge for the tetrahedron.")
    ap.add_argument("--tri-density", type=int, default=14,
                    help="Barycentric resolution per edge for the triangle "
                         "(denser than the tetrahedron).")
    ap.add_argument("--n-cloud", type=int, default=650,
                    help="Number of points in the swirly cloud.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the cloud.")
    ap.add_argument("--out", type=str, default=None,
                    help="Save to this path instead of showing a window.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    tri = triangle_grid(args.tri_density)
    tet = tetra_grid(args.density)
    cloud = swirly_cloud(args.n_cloud, rng)

    fig = plt.figure(figsize=(15, 5))

    # 1. Flat equilateral triangle, with a black outline.
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(tri[:, 0], tri[:, 1], s=90, c=BLUE, alpha=0.6, edgecolors="none")
    tri_loop = np.vstack([TRI_VERTICES, TRI_VERTICES[0]])
    ax1.plot(tri_loop[:, 0], tri_loop[:, 1], color="black", lw=1.5)
    ax1.set_title("Triangle")
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.margins(0.30)  # shrink the triangle ~30% within its panel

    # 2. Regular tetrahedron, with black edge outlines.
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(tet[:, 0], tet[:, 1], tet[:, 2], s=70, c=BLUE,
                alpha=0.30, edgecolors="none")
    for i in range(4):
        for j in range(i + 1, 4):
            ax2.plot(*zip(TETRA_VERTICES[i], TETRA_VERTICES[j]),
                     color="black", lw=1.2)
    ax2.set_title("Tetrahedron")
    ax2.set_box_aspect((1, 1, 1))
    ax2.set_axis_off()

    # 3. Swirly random cloud.
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=70, c=BLUE,
                alpha=0.25, edgecolors="none")
    ax3.set_title("Swirly cloud")
    ax3.set_axis_off()
    ax3.set_box_aspect(None, zoom=1.5)  # render the cloud ~50% larger

    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"wrote {args.out}  "
              f"(triangle={len(tri)}, tetrahedron={len(tet)}, cloud={len(cloud)} pts)")
    else:
        plt.show()


if __name__ == "__main__":
    main()
