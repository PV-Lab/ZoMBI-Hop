"""
benchmarks/methods/calibrate.py
===============================
Check a needle extractor on sample sets where the right answer is known.

    python -m benchmarks.methods calibrate [--extractor gp_peaks] [--out DIR]

Each scenario builds a bumps-only cube landscape (or one with NO needles), draws a
sample set the way a kind of optimiser would — uniformly, or with most of the budget
piled onto four true peaks and two decoy spots on the plain — adds the benchmark's
4.5% multiplicative noise, and scores what the extractor declares. A good extractor
has no false positives on the plains, never declares the decoys, and recovers what
the samples actually resolve. Writes ``calibration.csv`` to ``--out``.

A model fit per scenario, so run it on a compute node, not the login node.
"""

from __future__ import annotations

import os
import time

import numpy as np

from .extract import make_extractor
from .runner import GroundTruth, score_needles

#: (label, dim, n_needles, basin_width, sampler, n_points); n_needles 0 = plain only.
SCENARIOS = [
    ("uniform d3 n10 b6", 3, 10, 6.0, "uniform", 3000),
    ("uniform d3 n10 b15", 3, 10, 15.0, "uniform", 3000),
    ("uniform d3 n30 b2.2", 3, 30, 2.2, "uniform", 3000),
    ("uniform d6 n10 b6", 6, 10, 6.0, "uniform", 3000),
    ("concentrated d3 n10 b6", 3, 10, 6.0, "concentrated", 3000),
    ("concentrated d10 n10 b6", 10, 10, 6.0, "concentrated", 3000),
    ("uniform d3 n2 b6 N48", 3, 2, 6.0, "uniform", 48),
    ("uniform d3 n10 b6 N96", 3, 10, 6.0, "uniform", 96),
    ("uniform d10 n10 b6 N96", 10, 10, 6.0, "uniform", 96),
    ("PLAIN uniform d3", 3, 0, 6.0, "uniform", 3000),
    ("PLAIN uniform d3 N48", 3, 0, 6.0, "uniform", 48),
    ("PLAIN uniform d3 N96", 3, 0, 6.0, "uniform", 96),
    ("PLAIN uniform d10", 10, 0, 6.0, "uniform", 3000),
]
NOISE_FRAC = 0.045
SEPARATION = 0.13


def _landscape(dim: int, n: int, b: float, seed: int):
    from synthetic_data.ensemble import CartesianEnsemble

    rng = np.random.default_rng(seed)
    centers: list[np.ndarray] = []
    while len(centers) < n:
        p = rng.random(dim)
        if not centers or np.min(np.linalg.norm(np.asarray(centers) - p, axis=1)) >= SEPARATION:
            centers.append(p)
    C = np.asarray(centers).reshape(-1, dim)
    # A plain-only landscape still needs one basin to build; park it far outside
    # the box so it contributes nothing inside.
    pinned = C if n else np.full((1, dim), 5.0)
    fn = CartesianEnsemble(
        dim=dim, n_optima=0, pinned_optima=pinned, basin_width=b, n_weak=0,
        weak_amp=0.0, n_ridges=0, ridge_amp=0.0, noise_amp=0.0, aniso_strength=0.0,
        n_plateaus=0, plateau_amp=0.0, edge_region=None, edge_amp=0.0,
        input_noise=0.0 if not n else 0.128, seed=seed)
    return fn, C


def _samples(dim: int, C: np.ndarray, sampler: str, N: int, rng) -> np.ndarray:
    if sampler == "uniform" or not len(C):
        return rng.random((N, dim))
    k = N - 300
    cents = np.vstack([C[:4], rng.random((2, dim))])   # 4 true peaks + 2 decoys
    lab = rng.integers(0, len(cents), k)
    return np.vstack([rng.random((300, dim)),
                      np.clip(cents[lab] + rng.normal(0, 0.04, (k, dim)), 0, 1)])


def calibrate(extractor: str = "gp_peaks", seeds=(0, 1), device: str = "cpu",
              out: str | None = None) -> list[dict]:
    rows = []
    for seed in seeds:
        for label, dim, n, b, sampler, N in SCENARIOS:
            rng = np.random.default_rng(seed + 11)
            fn, C = _landscape(dim, n, b, seed)
            X = _samples(dim, C, sampler, N, rng)
            F = fn.predict(X)
            Y = F * (1.0 + rng.normal(0.0, NOISE_FRAC, F.shape))
            ex = make_extractor(extractor, device=device, seed=seed, noise_frac=NOISE_FRAC)
            t0 = time.time()
            needles, _ = ex(X, Y)
            row = {"seed": seed, "scenario": label, "n_true": n, "n_points": N,
                   "n_declared": int(len(needles)), "seconds": round(time.time() - t0, 2)}
            if n:
                s = score_needles(needles, GroundTruth(C, 1.0))
                row.update(recall=s["frac_optima_found"], precision=s["needle_precision"],
                           dist_to_needles=s["dist_to_needles"])
                print(f"  seed {seed}  {label:<26} declared {len(needles):>3}/{n:<3} "
                      f"recall {s['frac_optima_found']:.2f}  precision "
                      f"{s['needle_precision']:.2f}  dist {s['dist_to_needles']:.3f}  "
                      f"({row['seconds']}s)", flush=True)
            else:
                row["false_positives"] = int(len(needles))
                print(f"  seed {seed}  {label:<26} FALSE POSITIVES {len(needles)}  "
                      f"({row['seconds']}s)", flush=True)
            rows.append(row)
    if out:
        import pandas as pd

        os.makedirs(out, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(out, "calibration.csv"), index=False)
        print(f"  -> {os.path.join(out, 'calibration.csv')}")
    return rows
