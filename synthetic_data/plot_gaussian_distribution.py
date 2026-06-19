"""Objective-value distribution over the simplex: realistic Gaussian vs campaign1a RF.

For each dimensionality (default 3, 4, 10 — matching ``gaussian3d`` / ``gaussian4d`` /
``gaussian10d`` in ``optimize/evaluate.py``), draws uniform simplex samples,
evaluates ``RealisticGaussian(dim=…)`` with current ``gaussian/defaults.json``
parameters, and histograms objective values.  An optional final panel compares
the campaign1a 3D RF surrogate.

Peak count scales with dimension via ``--peak-scaling`` (default multiplicative:
``n_base × dim/3``; linear uses ``n_base × (d−1)/2`` as in evaluate.py).

Usage:
    python synthetic_data/plot_gaussian_distribution.py
    MPLBACKEND=Agg python synthetic_data/plot_gaussian_distribution.py --no-show
    python synthetic_data/plot_gaussian_distribution.py --peak-scaling linear
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

if not sys.stdin.isatty():
    matplotlib.use("Agg")

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from synthetic_data.compare_campaign_datasets import resolve_campaign1a_path
from synthetic_data.gaussian_landscape import RealisticGaussian, load_config
from synthetic_data.plot_distribution_common import (
    SAMPLE_SEED,
    add_peak_scaling_args,
    load_campaign1a_rf,
    make_distribution_figure,
    parse_dims,
    sample_simplex,
    scaled_peak_count,
)

DEFAULT_DIMENSIONS = (3, 4, 10)
N_SAMPLES = 200_000
PEAK_SEED = 42
PLOTS_DIR = Path(__file__).resolve().parent / "data" / "plots"
DEFAULT_OUTPUT = PLOTS_DIR / "gaussian_distribution.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dims",
        default=",".join(str(d) for d in DEFAULT_DIMENSIONS),
        help=f"Comma-separated simplex dimensions (default: {','.join(map(str, DEFAULT_DIMENSIONS))})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=PEAK_SEED,
        help="Peak-location RNG seed (default: 42, as in evaluate.py gaussian3d/4d/10d)",
    )
    add_peak_scaling_args(parser)
    parser.add_argument("--campaign1a", default=None, help="Path to campaign1a.csv")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--log", action="store_true", help="Log-scale y axis")
    parser.add_argument("--no-rf", action="store_true", help="Skip campaign1a RF panel")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dims = parse_dims(args.dims)
    cfg = load_config()
    n_base = int(cfg["n_peaks"])
    rng = np.random.default_rng(SAMPLE_SEED)

    panels: list[tuple[str, np.ndarray, str]] = []
    for dim in dims:
        n_peaks = scaled_peak_count(n_base, dim, mode=args.peak_scaling)
        fn = RealisticGaussian(dim, n_peaks=n_peaks, peak_seed=args.seed)
        pts = sample_simplex(dim, args.n_samples, rng)
        y = fn.predict(pts).astype(float)
        sigma = fn.sigmas[0] if fn.sigmas else float("nan")
        title = (
            f"Realistic Gaussian\n"
            f"dim = {dim}, {n_peaks} peaks, σ ≈ {sigma:.3g}"
        )
        if fn._noise_amp > 0:
            title += f", noise_amp = {fn._noise_amp:g}"
        print(f"  dim={dim}: y ∈ [{y.min():.3g}, {y.max():.3g}], {n_peaks} peaks")
        panels.append((title, y, "steelblue"))

    if not args.no_rf:
        campaign_path = resolve_campaign1a_path(args.campaign1a)
        rf, n_rows = load_campaign1a_rf(campaign_path)
        pts3 = sample_simplex(3, args.n_samples, rng)
        y_rf = rf.predict(pts3).astype(float)
        title = f"campaign1a RF surrogate\n(dim = 3, n = {n_rows} training rows)"
        print(f"  RF ({campaign_path}): y ∈ [{y_rf.min():.3g}, {y_rf.max():.3g}]")
        panels.append((title, y_rf, "indianred"))

    make_distribution_figure(
        panels,
        n_samples=args.n_samples,
        peak_scaling=args.peak_scaling,
        output_png=Path(args.output),
        show=not args.no_show,
        log_y=args.log,
    )


if __name__ == "__main__":
    main()
