"""Uniform random samples of needles landscapes, for an RF fit like run_69af's.

Draws N points uniformly on the simplex (Dirichlet(1), no space-filling design),
scores them on each requested needles landscape (evaluate.needles_config, the same
landscape ``--dataset needles`` runs), and writes one ``.npz`` (``X``, ``Y`` + the
landscape's settings) per landscape. ``evaluate.py --dataset runrf --gp-run <file>.npz``
then fits its 500-tree forest to those points exactly as it does to a run directory.

N defaults to the scored-point count of ``--match-run``, so the forest sees as much
data as the run_69af forest. Every landscape gets the SAME X (one draw, seed
``--sample-seed``), so the RFs differ only through the landscape.

    uv run python optimize/scripts/needles_rf_points.py --match-run runs/run_69af \\
        --needles n_optima=2,basin_width=2.2 --needles n_optima=3,basin_width=6 \\
        --out analysis/needles_rf_points_20261026
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "optimize"))

import evaluate  # noqa: E402
from synthetic_data.ensemble import Ensemble  # noqa: E402
from warm_start.warm_gp_landscape import _load_points  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--needles", action="append", required=True, metavar="KEY=VAL[,...]",
                    help="one needles landscape, 'n_optima=N,basin_width=B[,index=I]' "
                         "(repeatable; index = the sweep's draw, default 0)")
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--landscape-seed", type=int, default=0,
                    help="evaluate.py --ensemble-seed of the needles landscapes")
    ap.add_argument("--match-run", default=None, metavar="RUN_DIR",
                    help="take N from this run's scored-point count")
    ap.add_argument("--n-points", type=int, default=None, help="N, if no --match-run")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output directory for the .npz files")
    a = ap.parse_args()

    if a.match_run:
        X_run, _, snap = _load_points(Path(a.match_run).resolve(), None)
        n = len(X_run)
        print(f"  {a.match_run} ({snap}): {n} scored points, dim {X_run.shape[1]}")
        if X_run.shape[1] != a.dim:
            sys.exit(f"--match-run has dim {X_run.shape[1]}, not --dim {a.dim}")
    elif a.n_points:
        n = a.n_points
    else:
        sys.exit("give --match-run or --n-points")

    X = np.random.default_rng(a.sample_seed).dirichlet(np.ones(a.dim), n)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.needles:
        ov = {k.strip(): json.loads(v) for k, v in
              (t.split("=", 1) for t in spec.split(",") if t.strip())}
        index = int(ov.pop("index", 0))
        cfg = evaluate.needles_config(a.dim, index, a.landscape_seed, ov)
        fn = Ensemble(**cfg)
        Y = np.asarray(fn.predict(X), dtype=float).ravel()
        plain = float(np.min(Y))
        peak = max(v for _, v in fn.known_maxima)
        name = (f"needles_n{ov['n_optima']}_bw{ov['basin_width']}_i{index}"
                + ("_unit" if ov.get("unit_range") else "")
                + f"_N{n}_s{a.sample_seed}.npz")
        np.savez(out / name, X=X, Y=Y, n_optima=ov["n_optima"],
                 basin_width=ov["basin_width"], index=index,
                 unit_range=bool(ov.get("unit_range")),
                 landscape_seed=a.landscape_seed, sample_seed=a.sample_seed,
                 centers=np.array([np.ravel(c) for c, _ in fn.known_maxima], dtype=float))
        print(f"  {name}: {len(fn.known_maxima)} optima; Y min {plain:.4f} "
              f"median {np.median(Y):.4f} max {Y.max():.4f}; "
              f"{int((Y > plain + 1e-6).sum())}/{n} points above the plain, "
              f"{int((Y > plain + 0.5 * (peak - plain)).sum())} above half-height")


if __name__ == "__main__":
    main()
