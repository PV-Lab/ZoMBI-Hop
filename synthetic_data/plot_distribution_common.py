"""Shared helpers for simplex objective-distribution plot scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthetic_data.ackley import resolve_scaled_n_optima
from synthetic_data.campaign_datasets import RF_N_ESTIMATORS, normalize_rows, train_rf
from synthetic_data.compare_campaign_datasets import (
    CAMPAIGN1A_COLS,
    OBJECTIVE_COL,
    resolve_campaign1a_path,
)

N_BINS = 80
SAMPLE_SEED = 0


def sample_simplex(dim: int, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.dirichlet(np.ones(dim), size=n)


def load_campaign1a_rf(csv_path: Path):
    df = pd.read_csv(csv_path).dropna(subset=CAMPAIGN1A_COLS + [OBJECTIVE_COL])
    X = normalize_rows(df[CAMPAIGN1A_COLS].values.astype(float))
    y = df[OBJECTIVE_COL].values.astype(float)
    rf = train_rf(X, y, n_estimators=RF_N_ESTIMATORS)
    return rf, len(df)


def parse_dims(raw: str) -> list[int]:
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


def scaled_peak_count(n_base: int, dim: int, *, mode: str) -> int:
    return resolve_scaled_n_optima(n_base, dim, mode=mode)


def add_peak_scaling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--peak-scaling",
        choices=("linear", "multiplicative"),
        default="multiplicative",
        help=(
            "How to scale peak/optima count with dimension from the d=3 baseline "
            "(default: multiplicative dim/3; linear uses (d-1)/2 as in evaluate.py)"
        ),
    )


def make_distribution_figure(
    panels: list[tuple[str, np.ndarray, str]],
    *,
    n_samples: int,
    peak_scaling: str,
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
        f"({n_samples:,} uniform samples each, {peak_scaling} peak scaling)",
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
