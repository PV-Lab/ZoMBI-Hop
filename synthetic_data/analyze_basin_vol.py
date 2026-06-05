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
DIMENSIONS = [3, 4, 10]   # simplex dimensionalities to examine (one panel each)
VARIANT = "realistic"     # which Ackley variant from Ackley.VARIANTS to analyze
N_SAMPLES = 200_000       # uniform simplex samples used to estimate the distribution
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
    rng = np.random.default_rng(SEED)

    # Build the list of (title, objective-values, colour) panels.
    panels = []
    for dim in DIMENSIONS:
        fn = Ackley(VARIANT, dim=dim)
        y = fn.predict(sample_simplex(dim, N_SAMPLES, rng))
        panels.append((f"Ackley '{VARIANT}'\ndim = {dim}", y, "steelblue"))

    if INCLUDE_RF:
        rf = load_rf_surrogate()
        y_rf = rf.predict(sample_simplex(len(RF_COMPOSITION_COLS), N_SAMPLES, rng))
        panels.append((f"RF surrogate\n(campaign1a, dim = {len(RF_COMPOSITION_COLS)})",
                       y_rf, "indianred"))

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5 * len(panels), 4.5), sharey=True
    )
    # Keep axes iterable even when there is a single panel.
    axes = np.atleast_1d(axes)

    for ax, (title, y, color) in zip(axes, panels):
        ax.hist(y, bins=N_BINS, color=color, edgecolor="white", linewidth=0.3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("objective value")
        ax.grid(axis="y", alpha=0.25)

        # Annotate with the best (highest) sampled value as a basin-reach cue.
        ax.axvline(y.max(), color="crimson", linestyle="--", linewidth=1.2)
        ax.text(
            0.97, 0.95, f"best sampled\n{y.max():.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="crimson",
        )

    axes[0].set_ylabel("count")
    fig.suptitle(
        f"Objective distribution over the simplex "
        f"(Ackley '{VARIANT}' vs. campaign1a RF, {N_SAMPLES:,} uniform samples each)",
        fontsize=13,
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
