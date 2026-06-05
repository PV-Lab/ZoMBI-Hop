"""Analyze the objective-value distribution of the Ackley objective across
different simplex dimensionalities.

For each dimensionality in ``DIMENSIONS`` we draw a large uniform sample from the
probability simplex, evaluate the analytic Ackley objective (synthetic_data/ackley.py)
at every point, and histogram the resulting objective values. Each dimensionality
gets its own panel in a single figure.

Reading the plot as a basin-volume diagnostic: the objective peaks at 0 (the
optima) and grows more negative away from them, so the histogram is the
distribution of "how good" a uniformly random point is. Mass piled up at the far
(negative) end and a thin tail reaching toward 0 means the near-optimal basins
occupy only a tiny fraction of the simplex volume -- and that fraction shrinks as
the dimensionality grows (the curse of dimensionality, made visible).

The figure is shown interactively (plt.show()); nothing is written to disk.

Usage:
    python analyze_basin_vol.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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


def sample_simplex(dim: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` points uniformly from the ``dim``-element probability simplex."""
    return rng.dirichlet(np.ones(dim), size=n)


def main():
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(
        1, len(DIMENSIONS), figsize=(5 * len(DIMENSIONS), 4.5), sharey=True
    )
    # Keep axes iterable even when DIMENSIONS has a single entry.
    axes = np.atleast_1d(axes)

    for ax, dim in zip(axes, DIMENSIONS):
        fn = Ackley(VARIANT, dim=dim)
        X = sample_simplex(dim, N_SAMPLES, rng)
        y = fn.predict(X)

        ax.hist(y, bins=N_BINS, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_title(f"dim = {dim}", fontsize=12, fontweight="bold")
        ax.set_xlabel("objective value")
        ax.grid(axis="y", alpha=0.25)

        # Annotate with the best (closest-to-0) sampled value as a basin-reach cue.
        ax.axvline(y.max(), color="crimson", linestyle="--", linewidth=1.2)
        ax.text(
            0.97, 0.95, f"best sampled\n{y.max():.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="crimson",
        )

    axes[0].set_ylabel("count")
    fig.suptitle(
        f"Ackley ('{VARIANT}') objective distribution over the simplex "
        f"({N_SAMPLES:,} uniform samples)",
        fontsize=13,
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
