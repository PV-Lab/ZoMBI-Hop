"""
uniform_baseline_conet.py
=========================
Render a CoNet visualization of a UNIFORM sampling of a run's ensemble
landscape, as a neutral reference for judging a real run's CoNet.

Motivation
----------
A run's CoNet only shows where the optimizer *chose* to sample. On its own you
cannot tell whether the run covered many of the landscape's optima or just a
handful. This is the "no-optimizer" baseline: it reconstructs the exact ensemble
landscape from a run's ``ensemble_config.json``, draws N points UNIFORMLY from
the composition simplex (Dirichlet(1,...,1)), evaluates the true objective, and
renders the same CoNet machinery used for real runs. The landscape's TRUE optima
are overlaid as stars, so a single image shows both what unbiased coverage looks
like and exactly where every optimum sits.

Nothing is written to disk except the PNG: samples/optima are built in memory and
handed straight to the CoNet builder. The output lands next to the run's own
``conet.png`` as ``conet_uniform.png`` by default.

Usage
-----
  python visualization/uniform_baseline_conet.py --run <RUN_DIR> [--save PNG]

``<RUN_DIR>`` must contain an ``ensemble_config.json``; its ``points.csv`` row
count sets the default sample budget so the baseline matches the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Repo root on sys.path so ``synthetic_data`` imports resolve when this script is
# run as ``python visualization/uniform_baseline_conet.py`` (only the script's
# own directory is auto-added otherwise).
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
# ``plot_10d`` lives beside this file; importing it reuses the exact CoNet
# build + render path that real runs use (no duplicated plotting logic).
import plot_10d as p10  # noqa: E402
from synthetic_data import Ensemble  # noqa: E402

# ``ensemble_config.json`` keys are exactly the Ensemble(...) constructor kwargs,
# so the config round-trips straight into the landscape.


def _count_csv_rows(path: Path) -> int:
    """Data-row count of a CSV (excludes the header), or 0 if absent."""
    if not path.exists():
        return 0
    with open(path, newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def build_ensemble(run_dir: Path) -> tuple[Ensemble, dict]:
    """Reconstruct the run's ensemble landscape from its ``ensemble_config.json``."""
    cfg_path = Path(run_dir) / "ensemble_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"no ensemble_config.json in {run_dir} — is this an ensemble run?")
    cfg = json.loads(cfg_path.read_text())
    return Ensemble(**cfg), cfg


def uniform_samples(dim: int, n: int, seed: int) -> np.ndarray:
    """N points drawn uniformly from the (dim-1)-simplex (Dirichlet with all-ones)."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(dim), size=n)


def render_uniform_conet(run_dir, save_path=None, *, n=None, seed=0, show_optima=True,
                         purity=None, spread=None, gap_reach=None):
    """Render the uniform-sampling baseline CoNet for an ensemble ``run_dir``.

    Builds everything in memory (no intermediate files) and writes the PNG to
    ``save_path`` (default: ``<run_dir>/conet_uniform.png``). Returns
    ``(png_path, n_stars_drawn)``.
    """
    run_dir = Path(run_dir)
    save_path = Path(save_path) if save_path else run_dir / "conet_uniform.png"

    ens, cfg = build_ensemble(run_dir)
    dim = int(cfg["dim"])
    if n is None:
        n = _count_csv_rows(run_dir / "points.csv")
    if n <= 0:
        raise ValueError("could not determine sample count from points.csv; pass n explicitly")

    X = uniform_samples(dim, n, seed)
    Y = ens.predict(X)

    # Same reduced composition space the real-run loader uses.
    comp, active = p10._reduce_comp(X)
    names = [f"x{i}" for i, a in enumerate(active) if a]

    needles_comp = None
    if show_optima:
        # True optima ranked best-value first, mapped into the same reduced space.
        centers = np.asarray(ens.centers, dtype=float)
        vals = ens.predict(centers)
        needles_comp = p10._reduce_needles(centers[np.argsort(-vals)], active)

    resp = Y.reshape(-1, 1)
    iters = np.arange(len(comp), dtype=float)
    title = f"UNIFORM baseline — {run_dir.name} · {len(comp)} samples · coloured by Objective"
    M, F = p10.build_conet(
        comp, names, resp, iters,
        umap_md=spread if spread is not None else p10.CN_UMAP_MD,
        gap_reach=gap_reach if gap_reach is not None else p10.CN_GAP_REACH,
        purity_thr=purity if purity is not None else p10.CN_PURITY_THR,
        needles_comp=needles_comp, show_needles=show_optima,
    )
    bounds = p10.conet_bounds(resp)
    out = p10.save_png(M, F, bounds, title, str(save_path))
    return out, len(M["needles"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="run directory containing ensemble_config.json (+ points.csv)")
    ap.add_argument("--save", default=None, metavar="PNG",
                    help="output PNG (default: <run>/conet_uniform.png)")
    ap.add_argument("--n", type=int, default=None,
                    help="number of uniform samples (default: match the run's points.csv row count)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the uniform draw (default: %(default)s)")
    ap.add_argument("--no-optima-stars", action="store_true",
                    help="do not overlay the landscape's true optima as stars")
    ap.add_argument("--purity", type=float, default=None,
                    help="CoNet purity layout threshold (default: plot_10d's CN_PURITY_THR)")
    ap.add_argument("--spread", type=float, default=None,
                    help="UMAP min_dist / point spread (default: plot_10d's CN_UMAP_MD)")
    ap.add_argument("--gap-reach", type=float, default=None,
                    help="far-satellite squash factor (default: plot_10d's CN_GAP_REACH)")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    ens, cfg = build_ensemble(run_dir)
    n = args.n if args.n is not None else _count_csv_rows(run_dir / "points.csv")
    print(f"landscape : dim={cfg.get('dim')}, n_optima={cfg.get('n_optima')}, seed={cfg.get('seed')}")
    print(f"sampling  : {n} uniform simplex points (seed={args.seed})")
    out, n_stars = render_uniform_conet(
        run_dir, args.save, n=args.n, seed=args.seed,
        show_optima=not args.no_optima_stars,
        purity=args.purity, spread=args.spread, gap_reach=args.gap_reach,
    )
    print(f"stars     : {n_stars} true optima drawn")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
