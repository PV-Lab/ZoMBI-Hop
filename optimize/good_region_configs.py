"""
optimize/good_region_configs.py
===============================
Emit hyperparameter configurations positioned relative to a dimensionality's
"good region", for a showdown that tests how sharply that region is defined.

The premise (from ``visualization/hparam_sensitivity.py``) is that 10d has the
most clearly delineated good region: its best configurations cluster far tighter
than a random group of the same size. If that is real, configurations drawn from
the centre of the cluster should beat configurations drawn from as far outside it
as the search space allows. This script produces both, so the showdown harness can
run all four on identical landscapes.

The good region
---------------
Same construction ``hparam_sensitivity`` uses for its ``top`` group: pool the
signature-matching runs for the dimensionality, take the Pareto front, then the
best ``--top-frac`` of that front by ``dist_to_needles``. Every configuration is a
point in the 16-D unit cube (each hyperparameter normalised within its canonical
``HPARAM_SPACE`` bounds, log-scaled where the space is). The good region's
**centre** is that group's centroid.

The four configurations
-----------------------
  ``inside_center``   the centroid itself — the closest any point can be to the
                      centre, though it is a synthesised config that no run has
                      ever executed
  ``inside_nearest``  the observed top-group configuration nearest the centroid —
                      a real, already-validated config, as a control against the
                      centroid being an artifact of averaging
  ``outside_far``     the cube corner farthest from the centroid: every
                      hyperparameter pushed to whichever canonical bound is
                      farther from the centre. This is the maximum-distance point
                      in the space, so it is as far outside as it is possible to be
  ``outside_far2``    the farthest corner subject to differing from ``outside_far``
                      in the ``--flip`` coordinates that contribute least to the
                      distance. Nearly as far, but meaningfully different — two
                      corners differing in one coordinate would be the same test run
                      twice

Corners are extreme by construction, which is the point: they are the honest
answer to "as far away from the centre as possible". Expect them to perform badly
or to terminate early; that IS the measurement. Integer-valued hyperparameters are
rounded after denormalisation, so every emitted config is directly runnable.

Usage
-----
  python optimize/good_region_configs.py --dim 10 --out optimize/runs/showdown_10d/configs
  python optimize/good_region_configs.py --dim 10 --top-frac 0.2 --flip 4
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "visualization"))

from pareto import DIST_KEY, pareto_mask_min  # noqa: E402
from hparam_sensitivity import (  # noqa: E402
    load_canonical_space,
    load_pool,
    encode,
    objective_matrix,
)


def denormalise(v: float, bounds: tuple) -> float | int:
    """Map a normalised [0, 1] coordinate back to a raw hyperparameter value.

    The inverse of ``hparam_sensitivity.normalise``, with ``int`` hyperparameters
    rounded (and clamped) so the emitted config is directly runnable rather than
    carrying a fractional count.
    """
    lo, hi, tfm = bounds
    v = float(np.clip(v, 0.0, 1.0))
    if tfm == "log":
        raw = math.exp(math.log(lo) + v * (math.log(hi) - math.log(lo)))
    else:
        raw = lo + v * (hi - lo)
    if tfm == "int":
        return int(np.clip(round(raw), math.ceil(lo), math.floor(hi)))
    return float(raw)


def to_hparams(x: np.ndarray, names: list[str], space: dict) -> dict:
    """A normalised cube point -> a raw hyperparameter dict."""
    return {n: denormalise(x[j], space[n]) for j, n in enumerate(names)}


def farthest_corner(c: np.ndarray) -> np.ndarray:
    """The cube corner farthest from centroid *c*.

    Per coordinate the farther bound is whichever of 0 or 1 is more distant, and
    because the squared distance separates across coordinates, choosing each
    independently maximises the total — no search needed.
    """
    return (c < 0.5).astype(float)


def second_farthest_corner(c: np.ndarray, n_flip: int) -> np.ndarray:
    """The farthest corner constrained to differ from it in *n_flip* coordinates.

    Flipping coordinate j back costs ``|2*c_j - 1|`` in the per-coordinate distance,
    so flipping the *smallest*-cost coordinates gives the farthest corner at that
    Hamming distance. This keeps the second "outside" config genuinely far out while
    making it a different configuration rather than a near-duplicate of the first.
    """
    corner = farthest_corner(c).copy()
    cost = np.abs(2.0 * c - 1.0)
    for j in np.argsort(cost)[:max(0, int(n_flip))]:
        corner[j] = 1.0 - corner[j]
    return corner


def build_configs(dim: int, runs_dir: str, top_frac: float, n_flip: int,
                  bounds_mode: str = "clip", own_only: bool = False) -> dict:
    """Assemble the four configurations plus the geometry that justifies them."""
    space = load_canonical_space()
    names = list(space)

    pool = load_pool(dim, runs_dir, own_only=own_only)
    X, kept, enc = encode(pool["records"], names, space, bounds_mode)
    if len(X) < 3:
        raise SystemExit(f"{dim}D pool has too few usable trials ({len(X)})")

    M, obj_cols = objective_matrix(kept)
    pmask = pareto_mask_min(M)
    pareto_idx = np.where(pmask)[0]
    dist = np.asarray([float(r["metrics"][DIST_KEY]) for r in kept])
    k_top = max(2, int(math.ceil(top_frac * len(pareto_idx))))
    top_idx = pareto_idx[np.argsort(dist[pareto_idx])[:k_top]]
    Xtop = X[top_idx]

    centre = Xtop.mean(axis=0)
    d_to_centre = np.linalg.norm(Xtop - centre, axis=1)
    nearest = int(np.argmin(d_to_centre))
    near_rec = kept[top_idx[nearest]]

    corner1 = farthest_corner(centre)
    corner2 = second_farthest_corner(centre, n_flip)

    print(f"[{dim}d] pool={len(kept)}  pareto={len(pareto_idx)}  "
          f"top{top_frac:.0%}-of-pareto={len(top_idx)}")
    print(f"[{dim}d] good-region centre: mean distance of its own members "
          f"{d_to_centre.mean():.3f}, nearest member {d_to_centre.min():.3f}")
    print(f"[{dim}d] outside_far  distance from centre "
          f"{np.linalg.norm(corner1 - centre):.3f}")
    print(f"[{dim}d] outside_far2 distance from centre "
          f"{np.linalg.norm(corner2 - centre):.3f} "
          f"({n_flip} coordinate(s) flipped vs outside_far)")

    def entry(name, x, selected_for, note, src=None):
        return {
            "name": name,
            "selected_for": selected_for,
            "hparams": to_hparams(x, names, space),
            "note": note,
            "distance_from_centre": float(np.linalg.norm(x - centre)),
            "source_run": (src or {}).get("source_run"),
            "trial": (src or {}).get("trial"),
            "metrics": (src or {}).get("metrics", {}),
        }

    configs = [
        entry("inside_center", centre, "inside",
              "centroid of the top-of-Pareto good region (synthesised)"),
        entry("inside_nearest", Xtop[nearest], "inside",
              "observed top-group config nearest the centroid", near_rec),
        entry("outside_far", corner1, "outside",
              "cube corner farthest from the good-region centre"),
        entry("outside_far2", corner2, "outside",
              f"farthest corner with the {n_flip} lowest-cost coordinate(s) flipped"),
    ]
    return {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "dim": dim,
        "top_frac": top_frac,
        "n_flip": n_flip,
        "bounds_mode": bounds_mode,
        "pool_note": pool["note"],
        "n_pool": len(kept),
        "n_pareto": int(len(pareto_idx)),
        "n_top": int(len(top_idx)),
        "objectives": obj_cols,
        "hparam_names": names,
        "centre_normalised": centre.tolist(),
        "top_group_mean_radius": float(d_to_centre.mean()),
        "configs": configs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--runs-dir", default=os.path.join(_REPO, "optimize", "runs"))
    ap.add_argument("--top-frac", type=float, default=0.10,
                    help="fraction of the Pareto front (best by dist_to_needles) "
                         "that defines the good region (default: %(default)s)")
    ap.add_argument("--flip", type=int, default=3,
                    help="coordinates by which the second outside config differs "
                         "from the first (default: %(default)s)")
    ap.add_argument("--bounds", default="clip", choices=["clip", "drop", "union"])
    ap.add_argument("--own-runs-only", action="store_true")
    ap.add_argument("--out", required=True,
                    help="directory to write the four <name>.json configs into")
    args = ap.parse_args()

    info = build_configs(args.dim, args.runs_dir, args.top_frac, args.flip,
                         bounds_mode=args.bounds, own_only=args.own_runs_only)
    os.makedirs(args.out, exist_ok=True)
    for c in info["configs"]:
        path = os.path.join(args.out, f"{c['name']}.json")
        with open(path, "w") as f:
            json.dump({k: v for k, v in c.items() if k != "name"}, f, indent=2)
        print(f"  wrote {path}")
    meta = os.path.join(args.out, "good_region.json")
    with open(meta, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  wrote {meta}")


if __name__ == "__main__":
    main()
