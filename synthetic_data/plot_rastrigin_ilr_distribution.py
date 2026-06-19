"""Objective-value distribution over the simplex: Rastrigin-in-ILR vs campaign1a RF.

For each dimensionality, draws uniform simplex samples, evaluates the analytic
Rastrigin-in-ILR oracle (current ``rastrigin_ilr/defaults.json`` parameters), and
histograms objective values.  An optional final panel compares the campaign1a
3D RF surrogate on the same footing (dim = 3 only).

Usage:
    python synthetic_data/plot_rastrigin_ilr_distribution.py
    MPLBACKEND=Agg python synthetic_data/plot_rastrigin_ilr_distribution.py --no-show
    python synthetic_data/plot_rastrigin_ilr_distribution.py --dims 3,4,10 --no-rf
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

if not sys.stdin.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from synthetic_data.campaign_datasets import RF_N_ESTIMATORS, normalize_rows, train_rf
from synthetic_data.compare_campaign_datasets import (
    CAMPAIGN1A_COLS,
    OBJECTIVE_COL,
    resolve_campaign1a_path,
)
from synthetic_data.ackley import scaled_n_optima
from synthetic_data.oracles import load_rastrigin_config

DEFAULT_DIMENSIONS = (3, 4, 10)
N_SAMPLES = 200_000
N_BINS = 80
SEED = 0
PLOTS_DIR = Path(__file__).resolve().parent / "data" / "plots"
DEFAULT_OUTPUT = PLOTS_DIR / "rastrigin_ilr_distribution.png"


def sample_simplex(dim: int, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.dirichlet(np.ones(dim), size=n)


def eval_rastrigin_ilr_batch(X: np.ndarray, amplitude: float) -> np.ndarray:
    """Vectorized Rastrigin-in-ILR (negated, maximize)."""
    X = np.asarray(X, dtype=float)
    eps = 1e-10
    log_x = np.log(X + eps)
    d = X.shape[1]
    ilr = np.empty((X.shape[0], d - 1), dtype=float)
    for i in range(d - 1):
        coef = math.sqrt((i + 1) / (i + 2))
        ilr[:, i] = coef * (log_x[:, : i + 1].sum(axis=1) / (i + 1) - log_x[:, i + 1])
    n = d - 1
    rastrigin = amplitude * n + np.sum(
        ilr ** 2 - amplitude * np.cos(2.0 * math.pi * ilr), axis=1,
    )
    return -rastrigin


def load_campaign1a_rf(csv_path: Path):
    df = pd.read_csv(csv_path).dropna(subset=CAMPAIGN1A_COLS + [OBJECTIVE_COL])
    X = normalize_rows(df[CAMPAIGN1A_COLS].values.astype(float))
    y = df[OBJECTIVE_COL].values.astype(float)
    rf = train_rf(X, y, n_estimators=RF_N_ESTIMATORS)
    return rf, len(df)


def _parse_dims(raw: str) -> list[int]:
    dims = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        d = int(tok)
        if d < 2:
            raise ValueError(f"dimension must be >= 2, got {d}")
        dims.append(d)
    if not dims:
        raise ValueError("no dimensions parsed")
    return dims


def make_figure(
    panels: list[tuple[str, np.ndarray, str]],
    *,
    n_samples: int,
    output_png: Path,
    show: bool,
    log_y: bool,
) -> None:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (title, y, color) in zip(axes, panels):
        ax.hist(y, bins=N_BINS, color=color, edgecolor="white", linewidth=0.3)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("objective value")
        ax.ticklabel_format(axis="x", useOffset=False)
        if log_y:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        ax.axvline(y.max(), color="crimson", linestyle="--", linewidth=1.2)
        ax.text(
            0.97, 0.95,
            f"best sampled\n{y.max():.3g}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="crimson",
        )

    axes[0].set_ylabel("count")
    fig.suptitle(
        f"Objective distribution over the simplex "
        f"({n_samples:,} uniform samples each)",
        fontsize=13,
    )
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_png}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dims",
        default=",".join(str(d) for d in DEFAULT_DIMENSIONS),
        help=f"Comma-separated simplex dimensions (default: {','.join(map(str, DEFAULT_DIMENSIONS))})",
    )
    parser.add_argument("--campaign1a", default=None, help="Path to campaign1a.csv")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--log", action="store_true", help="Log-scale y axis")
    parser.add_argument("--no-rf", action="store_true", help="Skip campaign1a RF panel")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dims = _parse_dims(args.dims)
    cfg = load_rastrigin_config()
    amplitude = float(cfg["amplitude"])
    n_base = int(cfg["n_optima"])
    rng = np.random.default_rng(SEED)

    panels: list[tuple[str, np.ndarray, str]] = []
    for dim in dims:
        n_optima = scaled_n_optima(n_base, dim)
        pts = sample_simplex(dim, args.n_samples, rng)
        y = eval_rastrigin_ilr_batch(pts, amplitude)
        title = (
            f"Rastrigin in ILR\n"
            f"dim = {dim}, A = {amplitude:g}, {n_optima} optima"
        )
        print(f"  dim={dim}: y ∈ [{y.min():.3g}, {y.max():.3g}], {n_optima} optima")
        panels.append((title, y, "steelblue"))

    if not args.no_rf:
        campaign_path = resolve_campaign1a_path(args.campaign1a)
        rf, n_rows = load_campaign1a_rf(campaign_path)
        pts3 = sample_simplex(3, args.n_samples, rng)
        y_rf = rf.predict(pts3).astype(float)
        title = f"campaign1a RF surrogate\n(dim = 3, n = {n_rows} training rows)"
        print(f"  RF ({campaign_path}): y ∈ [{y_rf.min():.3g}, {y_rf.max():.3g}]")
        panels.append((title, y_rf, "indianred"))

    make_figure(
        panels,
        n_samples=args.n_samples,
        output_png=Path(args.output),
        show=not args.no_show,
        log_y=args.log,
    )


if __name__ == "__main__":
    main()
