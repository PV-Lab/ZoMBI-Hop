"""
Warm-start seed prior: seeding the GP with partially-scored measurements
========================================================================

The campaign objective is an equally-weighted mean of three properties

    Objective = 1/3 * merit(Bandgap) + 1/3 * Photoconductance + 1/3 * Stability

but at warm-start time the stability assay is not ready, so a warm-start line can
only be scored on the first two thirds.  This module answers the question that
makes such a partial score usable: *how much should the GP distrust it?*

The idea
--------
A GP's uncertainty is not a dial you set after the fact — it is computed from how
far you are from data and from how noisy you declared each observation to be.  So
"seed the point, then inflate the uncertainty around it" is one action, not two:
you hand the point in with a large observation noise from the start.

That is what :func:`GPSimplex.set_seed_prior <src.utils.gp_simplex.GPSimplex.set_seed_prior>`
does with the constants below.  The GP uses the seeds to shape its guess where it
has nothing else, never becomes confident there (so the acquisition still wants to
measure those regions for real), and lets a later clean measurement overrule a
nearby seed.

Where the numbers come from (data/2nd_real_run.db, 644 complete rows)
--------------------------------------------------------------------
The scalarization was verified empirically rather than assumed: subtracting
``(Photoconductance + Stability)/3`` from ``Objective`` leaves a quantity that is a
*deterministic* function of Bandgap (within-bandgap std = 0.0 across 54 repeated
bandgap groups), confirming the 1/3 weights and a tent-shaped bandgap merit peaking
near 1.75 eV.

The piece a seed is missing is therefore exactly ``1/3 * Stability``:

    mean 0.2491,  std 0.0393,  var 0.001543,  range [0.107, 0.333]

Two consequences drive the design:

1. **Add the mean back** (:data:`MISSING_MEAN`).  A seed is missing a term whose
   mean is 0.249, so a bare partial score sits a constant 0.249 below every real
   measurement.  The GP reads that offset as real structure and the acquisition
   then *avoids* precisely the regions the warm start paid to explore.  Seeding
   ``partial + MISSING_MEAN`` puts seeds and real points on one scale, leaving only
   the genuine uncertainty.

2. **The missing term is very nearly iid noise.**  A 5-nearest-neighbour predictor
   built on composition explains only ~1% of the variance of ``1/3 * Stability``
   (residual std 0.0391 vs marginal 0.0393).  Stability is essentially spatially
   unstructured in this campaign.  That matters because "inflate the noise" encodes
   an *iid* error assumption; had stability been strongly composition-dependent the
   seeds would carry a systematic regional bias that a noise term cannot express.
   Here they do not, so the noise model is honest rather than a compromise.

The resulting inflation is mild: seed variance is only ~2x the real output-noise
variance (0.001543 vs 0.000758 at the campaign's mean |y| = 0.612).

Transferring to a synthetic landscape
-------------------------------------
A benchmark landscape (e.g. the layered Ensemble objective) has no literal
stability third, so the absolute variance above is meaningless there.  What
transfers is the *ratio* of the missing term's spread to the objective's own
spread, :data:`MISSING_STD_FRAC` = 0.0393 / 0.1151 = 0.341 — "a seed is uncertain
to about a third of an objective standard deviation".  :func:`seed_noise_for_scale`
applies it to a landscape's own output std.
"""

from __future__ import annotations

import os
import sqlite3

import numpy as np

# --- Derived from data/2nd_real_run.db; see module docstring for the derivation.
# Regenerate with:  python -m warm_start.seed_prior --recompute

#: Mean of the missing ``1/3 * Stability`` term.  Added to a partial score so seeds
#: and fully-measured points share one scale.
MISSING_MEAN = 0.2491
#: Std of the missing term, in campaign objective units.
MISSING_STD = 0.0393
#: Variance of the missing term (``MISSING_STD ** 2``).
MISSING_VAR = 0.001543
#: Missing-term std as a fraction of the objective's own std — the scale-free form
#: used to carry this prior onto a synthetic landscape.
MISSING_STD_FRAC = 0.3414

#: Multiplicative output-noise fraction applied to real measurements
#: (``optimize/run_mobo.py: OUTPUT_NOISE_FRAC``).
OUTPUT_NOISE_FRAC = 0.045

_DB_RELPATH = os.path.join("data", "2nd_real_run.db")


def seed_noise_for_scale(y_std: float) -> tuple[float, float]:
    """Return ``(seed_var, real_var)`` for a landscape whose outputs have std `y_std`.

    ``real_var`` is the variance a fully-measured point carries — the simulated
    multiplicative output noise, whose typical magnitude is
    ``OUTPUT_NOISE_FRAC * y_std``.

    ``seed_var`` adds the missing third's variance on top: a seed suffers the same
    measurement noise as any point *plus* the unknown stability contribution.  The
    two add because they are independent sources of error, so a seed is strictly
    noisier than a real point rather than merely differently noisy.
    """
    if not y_std > 0:
        raise ValueError(f"y_std must be positive, got {y_std}")
    real_var = (OUTPUT_NOISE_FRAC * y_std) ** 2
    missing_var = (MISSING_STD_FRAC * y_std) ** 2
    return real_var + missing_var, real_var


def partial_score_to_seed_y(partial: np.ndarray) -> np.ndarray:
    """Convert campaign partial scores ``1/3*merit(Eg) + 1/3*PC`` into seed values.

    Adds back the mean of the missing stability third so the seeds sit on the same
    scale as fully-measured points (see consequence 1 in the module docstring).
    This is the real-hardware entry point; synthetic benchmarks use
    :func:`warm_start.compare.simulate_partial_score` instead.
    """
    return np.asarray(partial, dtype=float) + MISSING_MEAN


# ---------------------------------------------------------------------------
# Re-derivation from the campaign database
# ---------------------------------------------------------------------------

def recompute_from_db(db_path: str | None = None) -> dict:
    """Re-derive the constants above from the campaign database.

    Returns a dict with the missing-term statistics, the verification that the
    objective really is an equally-weighted third of each property, and the
    spatial-structure check that justifies the iid noise model.
    """
    if db_path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(repo_root, _DB_RELPATH)
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"campaign database not found: {db_path}")

    cols = ("FAPbI3, MAPbI3, MAPbBr3, Bandgap, Photoconductance, "
            "Stability, Objective")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(f"SELECT {cols} FROM results").fetchall()
    a = np.array([[np.nan if v is None else v for v in r] for r in rows], float)
    b = a[~np.isnan(a).any(axis=1)]
    if b.shape[0] == 0:
        raise ValueError("no complete rows in campaign database")

    X, bandgap, pc, stability, objective = b[:, :3], b[:, 3], b[:, 4], b[:, 5], b[:, 6]
    missing = stability / 3.0

    # Verify the 1/3 weights: the residual after removing 1/3*PC and 1/3*Stability
    # must be a deterministic function of bandgap alone.
    merit = 3.0 * (objective - (pc + stability) / 3.0)
    keys, inv = np.unique(np.round(bandgap, 3), return_inverse=True)
    spreads = [merit[inv == k].std() for k in range(len(keys)) if (inv == k).sum() > 2]
    merit_determinism = float(np.median(spreads)) if spreads else float("nan")

    # Spatial-structure check: how much of the missing term does composition predict?
    from scipy.spatial import cKDTree

    Xn = X / X.sum(axis=1, keepdims=True)
    _, idx = cKDTree(Xn).query(Xn, k=6)
    residual = missing - missing[idx[:, 1:]].mean(axis=1)
    explained = 1.0 - residual.var() / missing.var()

    real_std = OUTPUT_NOISE_FRAC * np.abs(objective).mean()
    return {
        "n_rows": int(b.shape[0]),
        "missing_mean": float(missing.mean()),
        "missing_std": float(missing.std()),
        "missing_var": float(missing.var()),
        "objective_std": float(objective.std()),
        "missing_std_frac": float(missing.std() / objective.std()),
        "merit_within_bandgap_std": merit_determinism,
        "spatially_explained_frac": float(explained),
        "real_var": float(real_std ** 2),
        "seed_over_real_ratio": float(missing.var() / real_std ** 2 + 1.0),
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--db", default=None, help="path to the campaign .db")
    args = p.parse_args()

    s = recompute_from_db(args.db)
    print(f"complete rows                     {s['n_rows']}")
    print(f"missing term 1/3*Stability  mean  {s['missing_mean']:.4f}")
    print(f"                            std   {s['missing_std']:.4f}")
    print(f"                            var   {s['missing_var']:.6f}")
    print(f"objective std                     {s['objective_std']:.4f}")
    print(f"missing/objective std ratio       {s['missing_std_frac']:.4f}")
    print()
    print(f"merit(Eg) within-bandgap std      {s['merit_within_bandgap_std']:.2e}"
          f"   (0 confirms the 1/3 weights)")
    print(f"missing term explained by comp.   {s['spatially_explained_frac']:.1%}"
          f"   (low confirms the iid noise model)")
    print()
    print(f"seed variance / real variance     {s['seed_over_real_ratio']:.1f}x")


if __name__ == "__main__":
    main()
