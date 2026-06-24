"""Histogram the objective-value distribution of a synthetic benchmark over the
probability simplex, alongside the campaign1a RF surrogate.

Two modes, selected by a required flag:

``--ackley``
    For each dimensionality in ``DIMENSIONS`` draw a large uniform sample from the
    probability simplex, evaluate the analytic Ackley objective
    (synthetic_data/ackley.py) at every point, and histogram the resulting
    objective values -- one panel per dimensionality.  A final panel does the same
    for the Random-Forest surrogate that
    interactive_testing/interactive_test_zombi.py trains on campaign1a.csv
    (3-component composition -> Objective), so its landscape can be compared on
    equal footing.

``--ensemble``
    Two panels.  On the **left**, a *combined* histogram of ``ENSEMBLE_N_RUNS``
    random :class:`~synthetic_data.ensemble.Ensemble` landscapes (all 3-D): every
    run is sampled uniformly on the simplex and all the objective values across the
    runs are pooled into one shared set of bins, so the panel shows what the
    distribution looks like aggregated over many random benchmark instances.  On
    the **right**, the same campaign1a RF surrogate panel as the Ackley mode.

Reading the plot as a basin-volume diagnostic: the objective is highest at the
optima and falls off away from them, so the histogram is the distribution of "how
good" a uniformly random point is.  Mass piled up at the low end with only a thin
tail reaching the top means the near-optimal basins occupy a tiny fraction of the
simplex volume.

The figure is shown interactively (plt.show()); nothing is written to disk.

Usage:
    python histogram.py --ackley
    python histogram.py --ensemble
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
from synthetic_data.ensemble import Ensemble, random_ensemble_config  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────
DIMENSIONS = [3, 4, 10]   # simplex dimensionalities to examine (--ackley, one panel each)
N_SAMPLES = 500_000       # uniform simplex samples used to estimate the distribution
N_BINS = 80               # histogram bins per panel
SEED = 0                  # RNG seed for reproducible sampling

# Ensemble mode: pool this many random 3-D landscapes into one combined histogram.
ENSEMBLE_DIM = 3
ENSEMBLE_N_RUNS = 10
ENSEMBLE_SEED = 0         # seeds the per-run random configs (reproducible)

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


def rf_panel(rng: np.random.Generator) -> tuple:
    """The campaign1a RF-surrogate histogram panel (shared by both modes)."""
    rf = load_rf_surrogate()
    y_rf = rf.predict(sample_simplex(len(RF_COMPOSITION_COLS), N_SAMPLES, rng))
    return (f"RF surrogate\n(campaign1a, dim = {len(RF_COMPOSITION_COLS)})",
            y_rf, "indianred", 1.0)


def ackley_panels(rng: np.random.Generator) -> list:
    """One Ackley panel per dimensionality in ``DIMENSIONS``."""
    panels = []
    for dim in DIMENSIONS:
        fn = Ackley("realistic", dim=dim)
        y = fn.predict(sample_simplex(dim, N_SAMPLES, rng))
        panels.append((f"Ackley 'realistic'\ndim = {dim}", y, "steelblue", 1.0))
    return panels


def ensemble_combined_panel(rng: np.random.Generator) -> tuple:
    """Pool ``ENSEMBLE_N_RUNS`` random 3-D Ensemble landscapes into one panel.

    Each run draws a fresh random Ensemble config (same recipe the benchmark and
    the plot_ensemble.py "Randomize" button use): walking the Sobol' sweep
    ``index = 0, 1, ..., ENSEMBLE_N_RUNS - 1`` at ``seed=ENSEMBLE_SEED`` gives a
    low-discrepancy spread of landscapes.  Each is sampled uniformly on the
    simplex, and the objective values from every run are concatenated so they all
    fall into one shared set of bins in :func:`plot_panels`.

    Each sample is weighted by ``1 / ENSEMBLE_N_RUNS`` so the bin heights are the
    *average* per-run count -- the same magnitude as the single-run RF panel, so
    both panels read on the same scale.
    """
    ys = []
    for i in range(ENSEMBLE_N_RUNS):
        cfg = random_ensemble_config(ENSEMBLE_DIM, index=i, seed=ENSEMBLE_SEED)
        fn = Ensemble(**cfg)
        ys.append(fn.predict(sample_simplex(ENSEMBLE_DIM, N_SAMPLES, rng)))
    y = np.concatenate(ys)
    return (f"Ensemble (combined)\n{ENSEMBLE_N_RUNS} random runs, dim = {ENSEMBLE_DIM}",
            y, "seagreen", 1.0 / ENSEMBLE_N_RUNS)


def plot_panels(panels: list, suptitle: str, log: bool) -> None:
    """Render ``panels`` as a row of shared-y histograms over the [0.5, 1] range."""
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5 * len(panels), 4.5), sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, (title, y, color, weight) in zip(axes, panels):
        # Fixed bin range so a combined/pooled panel uses the same shared bins
        # for every run and panels stay comparable.  ``weight`` rescales the bin
        # heights (e.g. 1/N_RUNS for the pooled Ensemble panel -> per-run average,
        # matching the single-run RF panel's scale).
        ax.hist(y, bins=N_BINS, range=(0.5, 1.0), color=color,
                edgecolor="white", linewidth=0.3,
                weights=np.full(y.shape, weight))
        ax.set_title(title, fontsize=12, fontweight="bold")
        # Pin every panel to the full objective range and disable matplotlib's
        # offset notation.  Otherwise a panel whose values all collapse to the
        # floor (~0.5) gets auto-zoomed to a ~1e-5 sliver and relabelled
        # "0..6  1e-5+5e-1", which reads like values of 0-6 instead of a spike
        # at 0.5.
        ax.set_xlim(0.5, 1.0)
        ax.ticklabel_format(axis="x", useOffset=False)
        if log:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlabel("objective value")

    axes[0].set_ylabel("count")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ackley", action="store_true",
                      help="Ackley panels (one per dim in DIMENSIONS) + RF surrogate")
    mode.add_argument("--ensemble", action="store_true",
                      help="combined histogram of 10 random 3-D ensembles + RF surrogate")
    parser.add_argument("--log", action="store_true", help="Use a log scale for the y axis")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)

    if args.ensemble:
        panels = [ensemble_combined_panel(rng), rf_panel(rng)]
        suptitle = "Objective distribution: Ensemble vs. campaign1a RF"
    else:
        panels = ackley_panels(rng) + [rf_panel(rng)]
        suptitle = "Objective distribution: Ackley vs. campaign1a RF"

    plot_panels(panels, suptitle, args.log)


if __name__ == "__main__":
    main()
