"""Analyze the objective-value distribution of the Ackley objective across
different simplex dimensionalities, alongside the campaign1a RF surrogate.

For each dimensionality in ``DIMENSIONS`` we draw a large uniform sample from the
probability simplex, evaluate the analytic Ackley objective (synthetic_data/ackley.py)
at every point, and histogram the resulting objective values. One extra panel does
the same for the Random-Forest surrogate that interactive_testing/interactive_test_zombi.py
trains on campaign1a.csv (3-component composition -> Objective), so its landscape can
be compared on equal footing. Each panel sits in a single shared figure.

Reading the plot as a basin-volume diagnostic: the objective is highest at the
optima and falls off away from them, so the histogram is the distribution of "how
good" a uniformly random point is. Mass piled up at the low end with only a thin
tail reaching the top means the near-optimal basins occupy a tiny fraction of the
simplex volume -- and for Ackley that fraction shrinks as the dimensionality grows
(the curse of dimensionality, made visible).

The figure is shown interactively (plt.show()); nothing is written to disk.

Usage:
    python analyze_basin_vol.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

HERE = Path(__file__).resolve().parent
# Make the repo root importable so this runs from any working directory.
sys.path.insert(0, str(HERE.parent))
from synthetic_data.ackley import Ackley  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────
BASIN_THRESHOLD = 0.67     # objective cutoff drawn as the "basin threshold" line
DIMENSIONS = [3, 4, 10]   # simplex dimensionalities to examine (one panel each)
N_SAMPLES = 500_000       # uniform simplex samples used to estimate the distribution
N_BINS = 80               # histogram bins per panel
SEED = 0                  # RNG seed for reproducible sampling

INCLUDE_RF = True         # add a panel for the campaign1a RF surrogate (dim = 3)
# RF surrogate settings -- mirror interactive_testing/interactive_test_zombi.py.
RF_CSV = HERE.parent / "interactive_testing" / "campaign1a.csv"
RF_COMPOSITION_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]
RF_OBJECTIVE_COL = "Objective"
RF_N_ESTIMATORS = 500
RF_RANDOM_STATE = 42


def sample_simplex(dim: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` points uniformly from the ``dim``-element probability simplex."""
    return rng.dirichlet(np.ones(dim), size=n)


def load_rf_surrogate() -> RandomForestRegressor:
    """Train the campaign1a Random-Forest surrogate (same recipe as the
    interactive ZoMBI-Hop test: 3 composition columns row-normalized to the
    simplex -> Objective, 500 trees, fixed seed)."""
    df = pd.read_csv(RF_CSV).dropna(subset=RF_COMPOSITION_COLS + [RF_OBJECTIVE_COL])
    X = df[RF_COMPOSITION_COLS].to_numpy(dtype=float)
    X = X / np.where(X.sum(axis=1, keepdims=True) == 0, 1.0, X.sum(axis=1, keepdims=True))
    y = df[RF_OBJECTIVE_COL].to_numpy(dtype=float)
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=RF_RANDOM_STATE
    )
    rf.fit(X, y)
    return rf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="store_true", help="Use a log scale for the y axis")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)

    panels = []
    for dim in DIMENSIONS:
        fn = Ackley("realistic", dim=dim)
        y = fn.predict(sample_simplex(dim, N_SAMPLES, rng))
        panels.append((f"Ackley 'realistic'\ndim = {dim}", y, "steelblue"))

    if INCLUDE_RF:
        rf = load_rf_surrogate()
        y_rf = rf.predict(sample_simplex(len(RF_COMPOSITION_COLS), N_SAMPLES, rng))
        panels.append((f"RF surrogate\n(campaign1a, dim = {len(RF_COMPOSITION_COLS)})",
                       y_rf, "indianred"))

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5 * len(panels), 4.5), sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, (title, y, color) in zip(axes, panels):
        ax.hist(y, bins=N_BINS, color=color, edgecolor="white", linewidth=0.3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        # Pin every panel to the full objective range and disable matplotlib's
        # offset notation.  Otherwise a high-dim panel whose values all collapse
        # to the floor (~0.5) gets auto-zoomed to a ~1e-5 sliver and relabelled
        # "0..6  1e-5+5e-1", which reads like values of 0-6 instead of a spike
        # at 0.5.
        ax.set_xlim(0.5, 1.0)
        ax.ticklabel_format(axis="x", useOffset=False)
        if args.log:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        ax.axvline(BASIN_THRESHOLD, color="red", linestyle=":", linewidth=1.4,
                   label="basin threshold")
        ax.legend(loc="upper left", fontsize=8)
        # Uniform simplex samples => the share at/above the threshold estimates
        # the fraction of simplex volume occupied by the basins.
        pct_above = 100.0 * np.mean(y >= BASIN_THRESHOLD)
        ax.set_xlabel(f"objective value\n{pct_above:.2g}% of simplex above threshold")

    axes[0].set_ylabel("count")
    fig.suptitle(
        f"Objective distribution over the simplex "
        f"(Ackley 'realistic' vs. campaign1a RF, {N_SAMPLES:,} uniform samples each)",
        fontsize=13,
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
