"""hparam_sensitivity.py

How tightly clustered are our chosen hyperparameter configurations, relative to
the whole search space?

Each configuration of the ``run_mobo.HPARAM_SPACE`` hyperparameters is one point
in a 16-dimensional cube: every hyperparameter is mapped to [0, 1] within its
canonical bounds (log-scaled where the space is log-scaled), so all distances are
commensurate and "the entire search space" is literally the unit cube.

Four groups are compared per dimensionality:

  pool     every trial of the reference tuning run/pool — where the optimiser
           actually looked (an empirical null that is NOT uniform)
  pareto   the Pareto-optimal trials of that pool (pareto.pareto_mask_min)
  top      the best ``--top-frac`` of that Pareto front by dist_to_needles — the
           sharp end of the front, so a subset of `pareto`. At the default 10%
           this is a handful of points, which is enough to ask "where do they
           sit" but not enough to say much about how tightly they cluster.
  deploy   the configurations the deploy_* evaluations actually ran
           (configs/{warm_best,mixture_best,best_hparams})

In 16 dimensions raw Euclidean distances are uninterpretable — concentration of
measure puts uniform points at a mean pairwise distance of ~sqrt(16/6) ~ 1.63
with a narrow spread, so *everything* looks far apart. Every statistic here is
therefore reported against a null: a Sobol sample of the cube (uniform) and the
trial pool itself (explored). Because the groups are small (deploy is 3 distinct
configs per dimensionality), all nulls are SIZE-MATCHED — random subsets of the
same size — and significance comes from permutation, never from an asymptotic
test. With k=3 in 16-D you cannot estimate a covariance; only distance statistics
are honest at that size.

Reference pools (see the module constants):
  3d   archived_runs/mobo_05_06_15_32 — 320 trials. Its run_config.json is
       essentially empty (no dim/dataset/hparam_space), so it cannot be
       signature-matched; it is used as an explicit single run with dim asserted.
       It scores the legacy runtime_s rather than avg_time_per_iter_s.
  4d   the signature-matched pool around archived_runs/mobo_ensemble_4d_job17300587
       (that run alone is 9 trials — too few for a cluster statistic), which comes
       out at 20 runs / 148 trials across both accounts. eve_lal's older-fork runs
       never stored the ``variant`` key, but pareto._load_run_signature backfills
       it from the folder name (``mobo_hebo_*`` -> hebo, else ensemble), so they do
       join the pool — exactly as they do in the Pareto plots. --own-runs-only
       restricts the pool to this account for a robustness check.
  6d   the signature-matched pool around mobo_ensemble_6d_job19202380 — the same
       rule, and the same pool ``python optimize/pareto.py <that run dir>``
       collects (single run dir -> pool its config-matching siblings).
  10d  the signature-matched pool around mobo_ensemble_10d_job18275950. All 16 of
       this account's 10d runs share one signature (time_limit_hours=0.7), so the
       choice of reference among them does not change the pool. eve_lal's
       mobo_19_06_15_45_28_124009_10d has no run_config.json at all, so it has no
       signature to match and stays out (4 trials).

Cross-dimension comparisons are run for every PAIR of the analysed
dimensionalities, so the 3-vs-4 conclusion stays on screen next to 3-vs-6 and
4-vs-6 rather than being replaced by them.

Bounds drift: the canonical space has changed since the older runs were sampled
(3d has max_zooms=2 in 48% of trials vs a canonical low of 3, and raw>300 in 16%;
the 4d pool has one such max_zooms). --bounds controls this; the default clips and
flags the affected dimensions, because clipping piles those trials onto exactly
0.0 in that coordinate and would make the 3d groups look artificially tight there.
Any max_zooms/raw conclusion for 3d carries that caveat in the report.

Usage:
  python visualization/hparam_sensitivity.py                    # 3d + 4d + 6d + 10d
  python visualization/hparam_sensitivity.py --dim 6
  python visualization/hparam_sensitivity.py --dim both         # 3d + 4d only
  python visualization/hparam_sensitivity.py --top-frac 0.2
  python visualization/hparam_sensitivity.py --bounds union     # drift sensitivity
  python visualization/hparam_sensitivity.py --own-runs-only
  python visualization/hparam_sensitivity.py --out /tmp/hps
"""

from __future__ import annotations

import os
import sys
import ast
import glob
import json
import math
import argparse
import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pareto.py owns trial collection, run-config signatures and the Pareto mask; reuse
# them so this script cannot drift from what the Pareto plots call "optimal".
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "optimize"))

from pareto import (  # noqa: E402
    DIST_KEY,
    DUP_KEY,
    NPTS_KEY,
    collect_trials,
    pareto_mask_min,
    _run_signature,
    _load_run_signature,
    _signatures_match,
    _dedup_realpath,
)
from collab_dirs import COLLABORATOR_RUNS_DIRS  # noqa: E402


# ─── Reference pools ─────────────────────────────────────────────────────────────

RUNS_DIR_DEFAULT = os.path.join(_REPO, "optimize", "runs")

# The long 3d tuning run. Single explicit run dir: its run_config.json predates
# config recording, so there is no signature to pool siblings on.
REF_3D = os.path.join("archived_runs", "mobo_05_06_15_32")

# Reference runs whose *signature* defines the pool for their dimensionality (the
# run is never used alone — see load_pool). 3d is absent by construction: it has no
# usable run_config.json, so it is handled as an explicit single run.
REF_RUNS = {
    4: os.path.join("archived_runs", "mobo_ensemble_4d_job17300587"),
    6: "mobo_ensemble_6d_job19202380",
    # All 16 of this account's 10d runs carry one identical signature, so any of
    # them defines the same pool; the largest is named for concreteness.
    10: "mobo_ensemble_10d_job18275950",
}

DIMS = [3, 4, 6, 10]

# Uniform std in one dimension, the reference for a per-hyperparameter spread.
UNIFORM_STD_1D = 1.0 / math.sqrt(12.0)


# ─── Canonical hyperparameter space ──────────────────────────────────────────────

def load_canonical_space(repo: str = _REPO) -> dict[str, tuple]:
    """The ``HPARAM_SPACE`` literal from optimize/run_mobo.py — the canonical bounds.

    Read by parsing the assignment out of the source rather than importing
    run_mobo, which would drag in torch/botorch and the whole ZoMBI stack for one
    dict. run_mobo.py stays the single source of truth either way; a structural
    change there surfaces here as a loud parse failure rather than a stale copy.
    """
    path = os.path.join(repo, "optimize", "run_mobo.py")
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "HPARAM_SPACE":
                space = ast.literal_eval(node.value)
                return {k: tuple(v) for k, v in space.items()}
    raise RuntimeError(f"HPARAM_SPACE not found in {path}")


def normalise(value: float, bounds: tuple) -> float:
    """Map a raw hyperparameter value into [0, 1] within *bounds* (unclipped)."""
    lo, hi, tfm = bounds
    if tfm == "log":
        return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (value - lo) / (hi - lo)


def encode(records: list[dict], names: list[str], space: dict[str, tuple],
           mode: str) -> tuple[np.ndarray, list[dict], dict]:
    """Records -> (X in [0,1]^n_hparams, kept records, drift report).

    *mode* is the --bounds policy: ``clip`` keeps every record and clamps
    out-of-range coordinates (flagging which dimensions were touched), ``drop``
    discards any record with an out-of-range value, ``union`` widens the bounds to
    cover the observed data so nothing piles onto an endpoint.
    """
    rows, kept, oob = [], [], {}
    missing: set[str] = set()
    extra: set[str] = set()

    eff = dict(space)
    if mode == "union":
        for name in names:
            lo, hi, tfm = space[name]
            vals = [float(r["hparams"][name]) for r in records
                    if name in r["hparams"]]
            if vals:
                eff[name] = (min(lo, min(vals)), max(hi, max(vals)), tfm)

    for rec in records:
        hp = rec["hparams"]
        extra |= set(hp) - set(names)
        if any(n not in hp for n in names):
            missing |= {n for n in names if n not in hp}
            continue
        row, bad = [], []
        for name in names:
            v = float(hp[name])
            lo, hi, _ = eff[name]
            if v < lo - 1e-9 or v > hi + 1e-9:
                bad.append(name)
            row.append(normalise(v, eff[name]))
        if bad:
            for name in bad:
                oob[name] = oob.get(name, 0) + 1
            if mode == "drop":
                continue
        rows.append(np.clip(np.asarray(row, dtype=float), 0.0, 1.0))
        kept.append(rec)

    X = np.asarray(rows, dtype=float) if rows else np.zeros((0, len(names)))
    report = {
        "mode": mode,
        "n_in": len(records),
        "n_kept": len(kept),
        "out_of_bounds_counts": oob,
        "dropped_extra_hparams": sorted(extra),
        "records_missing_hparams": sorted(missing),
        "widened_bounds": ({k: list(eff[k]) for k in names
                            if eff[k] != space[k]} if mode == "union" else {}),
    }
    return X, kept, report


# ─── Pool assembly ───────────────────────────────────────────────────────────────

def _crawl_dirs(runs_dir: str, *, own_only: bool = False) -> list[str]:
    """Every runs dir worth crawling for a signature pool: ours (live + archived) + collabs.

    collab_dirs.SHARE_COLLABORATOR_HISTORY is deliberately NOT consulted: that flag
    governs live GP pooling for run_mobo/pareto/sync_runs, and flipping it would
    change their behaviour too. A read-only analysis opts in on its own.
    """
    dirs = [runs_dir, os.path.join(runs_dir, "archived_runs")]
    if not own_only:
        for d in COLLABORATOR_RUNS_DIRS:
            dirs += [d, os.path.join(d, "archived_runs")]
    return [d for d in _dedup_realpath(dirs) if os.path.isdir(d)]


def load_pool(dim: int, runs_dir: str, *, own_only: bool = False) -> dict:
    """The reference trial pool for *dim*: {records, signature, note, dirs}."""
    if dim == 3:
        run_dir = os.path.join(runs_dir, REF_3D)
        if not os.path.isdir(run_dir):
            raise SystemExit(f"3d reference run not found: {run_dir}")
        print(f"[pool 3d] single run (empty run_config, dim asserted): {run_dir}")
        records = collect_trials(run_dir)
        for r in records:
            r["source_run"] = os.path.basename(run_dir)
        return {
            "records": records,
            "signature": None,
            "dirs": [run_dir],
            "note": ("single archived run; run_config.json records no dim/dataset/"
                     "hparam_space, so no signature pooling and dim is asserted "
                     "from the CLI. Scores legacy runtime_s."),
        }

    if dim not in REF_RUNS:
        raise SystemExit(f"no reference run configured for dim {dim} "
                         f"(known: {sorted(REF_RUNS) + [3]})")
    ref = os.path.join(runs_dir, REF_RUNS[dim])
    sig = _load_run_signature(ref)
    if sig is None:
        raise SystemExit(f"{dim}d reference run has no usable run_config.json: {ref}")
    if sig.get("dim") != dim:
        raise SystemExit(f"{dim}d reference run records dim={sig.get('dim')}: {ref}")
    match_sig = dict(sig)
    note = (f"signature-matched pool around {os.path.basename(ref)} "
            f"({json.dumps(sig, sort_keys=True)})")
    if own_only:
        note += " | OWN RUNS ONLY: collaborator runs dirs excluded."
    print(f"[pool {dim}d] {note}")

    dirs = _crawl_dirs(runs_dir, own_only=own_only)
    records: list[dict] = []
    for d in dirs:
        records += collect_trials(d, only_signature=match_sig)
    if not records:
        raise SystemExit(f"{dim}d pool is empty — no signature-matching runs found.")
    return {"records": records, "signature": match_sig, "dirs": dirs, "note": note}


def load_deploy_configs(dim: int, runs_dir: str) -> list[dict]:
    """The hyperparameter sets the deploy_* evaluations ran, as trial-like records.

    Each deploy run writes three configs: warm_best and mixture_best (selected from
    the mobo_warmgp_*/mobo_mixture_* tuning runs) and best_hparams (the fixed
    production BEST_HPARAMS.md baseline, identical across every deploy run). The
    same config recurs across the L-budget variants, so identical vectors are
    collapsed and every source path is recorded on the survivor.
    """
    method_of = {"warm_best": "warm", "mixture_best": "mixture",
                 "best_hparams": "baseline"}
    found: dict[tuple, dict] = {}
    pattern = os.path.join(runs_dir, f"deploy_{dim}d*_job*", "configs", "*")
    for cfg_dir in sorted(glob.glob(pattern)):
        kind = os.path.basename(cfg_dir)
        for fname in ("hparams.json", "trial.json"):
            path = os.path.join(cfg_dir, fname)
            if not os.path.isfile(path):
                continue
            try:
                with open(path) as f:
                    blob = json.load(f)
            except Exception as exc:
                print(f"  [deploy] {path}: unreadable ({exc}); skipping.")
                continue
            hp = blob.get("hparams")
            if not hp:
                continue
            run = os.path.basename(os.path.dirname(os.path.dirname(cfg_dir)))
            key = tuple(sorted((k, round(float(v), 10)) for k, v in hp.items()))
            rec = found.get(key)
            if rec is None:
                found[key] = {
                    "source_run": run,
                    "trial": blob.get("trial"),
                    "method": method_of.get(kind, kind),
                    "source_tuning_run": blob.get("source_run") or blob.get("source"),
                    "hparams": hp,
                    "metrics": blob.get("metrics", {}),
                    "seen_in": [f"{run}/{kind}"],
                }
            else:
                rec["seen_in"].append(f"{run}/{kind}")
            break
    out = list(found.values())
    print(f"[deploy {dim}d] {len(out)} distinct config(s) across "
          f"{sum(len(r['seen_in']) for r in out)} deploy config slot(s)")
    for r in out:
        print(f"  [deploy] {r['method']:9s} trial={r['trial']} "
              f"used by {len(r['seen_in'])} slot(s)")
    return out


def objective_matrix(records: list[dict]) -> tuple[np.ndarray, list[str]]:
    """(n, n_obj) minimisation matrix for the Pareto mask, mirroring pareto.main.

    dist_to_needles, dup_fraction and the time objective always; n_points_penalty
    only when every trial in the pool has it (all-or-nothing, as pareto.py does).
    """
    time_key = records[0]["time_key"]
    cols = [DIST_KEY, DUP_KEY, time_key]
    if all(r.get("npts_value") is not None for r in records):
        cols.append(NPTS_KEY)
    M = np.asarray([[float(r["metrics"][c]) for c in cols] for r in records])
    return M, cols


# ─── Distance geometry ───────────────────────────────────────────────────────────

def pdist_mean(X: np.ndarray) -> float:
    """Mean pairwise Euclidean distance within *X* (nan for fewer than 2 points)."""
    if len(X) < 2:
        return float("nan")
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    iu = np.triu_indices(len(X), k=1)
    return float(D[iu].mean())


def pdist_median(X: np.ndarray) -> float:
    if len(X) < 2:
        return float("nan")
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    iu = np.triu_indices(len(X), k=1)
    return float(np.median(D[iu]))


def centroid_radius(X: np.ndarray) -> float:
    """Mean distance to the group centroid (radius of gyration)."""
    if len(X) < 1:
        return float("nan")
    return float(np.linalg.norm(X - X.mean(axis=0), axis=1).mean())


def cross_dmin(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """For each row of *A*, the distance to the nearest row of *B*."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros(len(A))
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    return D.min(axis=1)


def sobol_null(n: int, d: int, seed: int) -> np.ndarray:
    """*n* quasi-uniform points in [0,1]^d (Sobol, falling back to plain uniform)."""
    try:
        from scipy.stats import qmc
        return qmc.Sobol(d=d, scramble=True, seed=seed).random(n)
    except Exception as exc:  # pragma: no cover - scipy always present in practice
        print(f"  [null] Sobol unavailable ({exc}); using uniform random.")
        return np.random.default_rng(seed).random((n, d))


def subset_dispersion_null(P: np.ndarray, k: int, n_perm: int,
                           rng: np.random.Generator) -> dict:
    """Dispersion of *n_perm* random k-subsets of *P*: the size-matched null.

    Returns the mean-pairwise-distance and centroid-radius samples. Matching k is
    essential — both statistics shrink with k, so an unmatched comparison would
    report small groups as tight no matter where they sit.
    """
    pair, rad = [], []
    if k < 2 or len(P) < k:
        return {"pairwise": np.array([]), "radius": np.array([])}
    for _ in range(n_perm):
        S = P[rng.choice(len(P), size=k, replace=False)]
        pair.append(pdist_mean(S))
        rad.append(centroid_radius(S))
    return {"pairwise": np.asarray(pair), "radius": np.asarray(rad)}


def _pct_le(sample: np.ndarray, value: float) -> float:
    """Fraction of *sample* at or below *value* (nan on an empty sample)."""
    if sample.size == 0 or not np.isfinite(value):
        return float("nan")
    return float(np.mean(sample <= value))


def effective_dim(X: np.ndarray) -> dict:
    """PCA shape of a group: participation ratio and #PCs for 90% of the variance.

    A group that is tight overall but spread along a couple of directions is a
    low-dimensional manifold of good configurations, not a ball — worth telling
    apart, since the former means most hyperparameters are pinned.
    """
    if len(X) < 3:
        return {"participation_ratio": float("nan"), "n_pc_90": None}
    Xc = X - X.mean(axis=0)
    ev = np.linalg.svd(Xc, compute_uv=False) ** 2
    if ev.sum() <= 0:
        return {"participation_ratio": float("nan"), "n_pc_90": None}
    p = ev / ev.sum()
    pr = float((p.sum() ** 2) / (p ** 2).sum())
    n90 = int(np.searchsorted(np.cumsum(p), 0.90) + 1)
    return {"participation_ratio": pr, "n_pc_90": n90}


def energy_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Two-sample energy distance: 2*E|a-b| - E|a-a'| - E|b-b'| (0 iff same law)."""
    if len(A) < 2 or len(B) < 2:
        return float("nan")
    ab = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1).mean()
    aa = pdist_mean(A)
    bb = pdist_mean(B)
    return float(2 * ab - aa - bb)


def energy_test(A: np.ndarray, B: np.ndarray, n_perm: int,
                rng: np.random.Generator) -> dict:
    """Label-shuffle permutation test on the energy distance between two groups."""
    obs = energy_distance(A, B)
    if not np.isfinite(obs):
        return {"energy": obs, "p_value": float("nan"), "n_perm": 0}
    P = np.vstack([A, B])
    nA = len(A)
    null = []
    for _ in range(n_perm):
        idx = rng.permutation(len(P))
        null.append(energy_distance(P[idx[:nA]], P[idx[nA:]]))
    null = np.asarray(null)
    # +1 in numerator and denominator: never report p=0 off a finite permutation set.
    p = float((1 + np.sum(null >= obs)) / (1 + len(null)))
    return {"energy": obs, "p_value": p, "n_perm": int(n_perm)}


# ─── Per-group analysis ──────────────────────────────────────────────────────────

def analyse_group(name: str, X: np.ndarray, names: list[str], U: np.ndarray,
                  pool: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    """Tightness of one group, expressed against the uniform and pool nulls."""
    k = len(X)
    pair = pdist_mean(X)
    rad = centroid_radius(X)
    unull = subset_dispersion_null(U, k, n_perm, rng)
    pnull = subset_dispersion_null(pool, k, n_perm, rng)

    per_dim_std = X.std(axis=0, ddof=0) if k > 1 else np.full(len(names), np.nan)
    per_dim_range = (X.max(axis=0) - X.min(axis=0)) if k > 0 else np.full(len(names), np.nan)

    out = {
        "name": name,
        "n": int(k),
        "mean_pairwise": pair,
        "median_pairwise": pdist_median(X),
        "centroid_radius": rad,
        "uniform_null": {
            "mean_pairwise": (float(unull["pairwise"].mean())
                              if unull["pairwise"].size else float("nan")),
            "mean_radius": (float(unull["radius"].mean())
                            if unull["radius"].size else float("nan")),
            "ratio_pairwise": (pair / float(unull["pairwise"].mean())
                               if unull["pairwise"].size else float("nan")),
            "ratio_radius": (rad / float(unull["radius"].mean())
                             if unull["radius"].size else float("nan")),
            "p_tighter": _pct_le(unull["pairwise"], pair),
        },
        "pool_null": {
            "mean_pairwise": (float(pnull["pairwise"].mean())
                              if pnull["pairwise"].size else float("nan")),
            "ratio_pairwise": (pair / float(pnull["pairwise"].mean())
                               if pnull["pairwise"].size else float("nan")),
            "p_tighter": _pct_le(pnull["pairwise"], pair),
        },
        "per_hparam_std_ratio": {
            n: (float(s / UNIFORM_STD_1D) if np.isfinite(s) else None)
            for n, s in zip(names, per_dim_std)
        },
        "per_hparam_range": {
            n: (float(r) if np.isfinite(r) else None)
            for n, r in zip(names, per_dim_range)
        },
        "effective_dim": effective_dim(X),
    }
    return out


def classify_hparams(res: dict, names: list[str], pinned: float = 0.35,
                     free: float = 0.75) -> dict:
    """Split hyperparameters into pinned / intermediate / free by spread ratio.

    The ratio is the group's per-dimension std over the uniform 1-D std
    (1/sqrt(12)); a pinned dimension is one the good configurations agree on, i.e.
    a hyperparameter the objective is sensitive to.
    """
    ratios = res["per_hparam_std_ratio"]
    buckets = {"pinned": [], "intermediate": [], "free": []}
    for n in names:
        r = ratios.get(n)
        if r is None:
            continue
        buckets["pinned" if r < pinned else
                "free" if r > free else "intermediate"].append((n, r))
    for b in buckets:
        buckets[b].sort(key=lambda kv: kv[1])
    return buckets


def deploy_proximity(Xd: np.ndarray, deploy: list[dict], targets: dict,
                     U: np.ndarray) -> list[dict]:
    """Per-deploy-config distance to each good set, as a percentile of the null.

    ``percentile`` is the fraction of uniform points whose nearest-neighbour
    distance to the same target set is at or below this config's — so a low value
    means the config sits genuinely inside the good region, and ~0.5 means it is no
    closer to it than a config drawn at random.
    """
    rows = []
    for i, rec in enumerate(deploy):
        row = {
            "method": rec["method"],
            "trial": rec.get("trial"),
            "source_tuning_run": rec.get("source_tuning_run"),
            "n_slots": len(rec.get("seen_in", [])),
            "targets": {},
        }
        for tname, T in targets.items():
            if len(T) == 0:
                continue
            d = float(cross_dmin(Xd[i:i + 1], T)[0])
            null = cross_dmin(U, T)
            row["targets"][tname] = {
                "d_min": d,
                "null_mean_d_min": float(null.mean()),
                "percentile": _pct_le(null, d),
            }
        rows.append(row)
    return rows


def hparam_ranges(group_records: dict[str, list[dict]], groups: dict[str, np.ndarray],
                  names: list[str], space: dict[str, tuple]) -> list[dict]:
    """Per-hyperparameter raw ranges per group, with the fraction of the axis used.

    Raw min/max are reported in the hyperparameter's own units (what you would put
    in a config), while ``frac`` is measured in the NORMALISED coordinate — so for
    the one log-scaled hyperparameter (nat_grad_step) the fraction is a fraction of
    the log axis, not of the linear span. That is the honest reading: the search
    treats that axis as logarithmic, so half the axis means half the decades.

    ``frac`` is the width of the group's observed range as a fraction of the
    canonical bounds, i.e. the share of that one axis the group occupies. Multiply
    across dimensions at your peril — see the volume caveat in the report.
    """
    rows = []
    for j, n in enumerate(names):
        lo, hi, tfm = space[n]
        row = {"hparam": n, "transform": tfm,
               "canonical": [float(lo), float(hi)], "groups": {}}
        for g, recs in group_records.items():
            vals = [float(r["hparams"][n]) for r in recs if n in r["hparams"]]
            if not vals or g not in groups or len(groups[g]) == 0:
                continue
            # Width in the normalised coordinate, which is what "share of the
            # axis" means for a log-scaled hyperparameter.
            span = groups[g][:, j]
            p10, p90 = np.percentile(vals, [10.0, 90.0])
            n10, n90 = np.percentile(span, [10.0, 90.0])
            row["groups"][g] = {
                "n": len(vals),
                "min": min(vals),
                "max": max(vals),
                "frac": float(span.max() - span.min()),
                # 10-90 band: min/max is driven by single stragglers, so the
                # central band is the robust read of where the group actually is.
                "p10": float(p10),
                "p90": float(p90),
                "frac_10_90": float(n90 - n10),
                # Raw values outside the canonical bounds were clamped when
                # normalised, so `frac` understates the true spread here.
                "out_of_canonical": bool(min(vals) < lo - 1e-9
                                         or max(vals) > hi + 1e-9),
            }
        rows.append(row)
    return rows


def cross_dim_agreement(A: np.ndarray, B: np.ndarray, names: list[str],
                        label_a: str = "a", label_b: str = "b",
                        lo_pct: float = 10.0, hi_pct: float = 90.0,
                        wide: float = 0.5) -> list[dict]:
    """Per-hyperparameter overlap of two good sets' central ranges.

    Uses the 10-90 percentile band rather than min/max so one stray trial cannot
    manufacture an overlap. Verdicts distinguish *evidence of* agreement from the
    mere absence of a contradiction, since a band overlap is only informative when
    both bands are actually narrow:

      both free   both bands cover more than *wide* of the axis — neither
                  dimensionality constrains this hyperparameter, so the overlap
                  says nothing
      agrees      both narrow and overlapping by at least half the narrower band —
                  the real "same place in both" verdict
      compatible  one narrow, one free: the constrained band sits inside the
                  unconstrained one. Not contradicted, but not corroborated —
                  the free side simply has no opinion
      disagrees   bands overlap by less than half the narrower one
    """
    rows = []
    for j, n in enumerate(names):
        a_lo, a_hi = np.percentile(A[:, j], [lo_pct, hi_pct])
        b_lo, b_hi = np.percentile(B[:, j], [lo_pct, hi_pct])
        wa, wb = a_hi - a_lo, b_hi - b_lo
        ov = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
        narrow = min(wa, wb)
        frac = (1.0 if narrow <= 1e-12 and ov > 0 else
                ov / narrow if narrow > 1e-12 else
                1.0 if abs(a_lo - b_lo) < 1e-9 else 0.0)
        a_wide, b_wide = wa > wide, wb > wide
        if a_wide and b_wide:
            verdict = "both free"
        elif frac < 0.5:
            verdict = "disagrees"
        elif a_wide != b_wide:
            verdict = f"compatible ({label_b if b_wide else label_a} free)"
        else:
            verdict = "agrees"
        rows.append({
            "hparam": n,
            "band_a": [float(a_lo), float(a_hi)],
            "band_b": [float(b_lo), float(b_hi)],
            "overlap_frac": float(frac),
            "verdict": verdict,
        })
    return rows


# ─── Figures ─────────────────────────────────────────────────────────────────────

GROUP_COLORS = {
    "pool": "#b0b0b0",
    "pareto": "#1f77b4",
    "top": "#2ca02c",
    "deploy": "#d62728",
}


def plot_std_heatmap(per_dim: dict[str, dict], names: list[str], dim: int,
                     out: str) -> None:
    """Per-hyperparameter spread ratio per group — which hyperparameters are pinned."""
    groups = list(per_dim)
    # `or np.nan` would swallow a ratio of exactly 0.0 — a fully pinned
    # hyperparameter, the most interesting cell in the grid — so test for None.
    def _cell(g: str, n: str) -> float:
        v = per_dim[g]["per_hparam_std_ratio"].get(n)
        return np.nan if v is None else float(v)

    M = np.array([[_cell(g, n) for n in names] for g in groups], dtype=float)
    fig, ax = plt.subplots(figsize=(0.55 * len(names) + 4, 0.6 * len(groups) + 2.4))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{g} (n={per_dim[g]['n']})" for g in groups], fontsize=8)
    for i in range(len(groups)):
        for j in range(len(names)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=5.5,
                        color="white" if M[i, j] < 0.7 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("std / uniform std  (1.0 = spread like the whole space)", fontsize=8)
    ax.set_title(f"{dim}D per-hyperparameter spread, normalised search space",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  [plot] {out}")


def plot_parallel(groups: dict[str, np.ndarray], names: list[str], dim: int,
                  out: str) -> None:
    """Parallel coordinates in the normalised cube — the most direct visual answer."""
    fig, ax = plt.subplots(figsize=(0.6 * len(names) + 4, 5))
    order = [g for g in ("pool", "pareto", "top", "deploy") if g in groups]
    for g in order:
        X = groups[g]
        if len(X) == 0:
            continue
        alpha = 0.06 if g == "pool" else 0.35 if g == "pareto" else 0.8
        lw = 0.5 if g == "pool" else 0.9 if g == "pareto" else 1.8
        for row in X:
            ax.plot(range(len(names)), row, color=GROUP_COLORS[g], alpha=alpha,
                    lw=lw, zorder=order.index(g))
        ax.plot([], [], color=GROUP_COLORS[g], lw=1.8,
                label=f"{g} (n={len(X)})")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("normalised value in canonical bounds")
    ax.set_title(f"{dim}D hyperparameter configurations", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  [plot] {out}")


def plot_null_hist(res: dict[str, dict], U: np.ndarray, n_perm: int,
                   rng: np.random.Generator, dim: int, out: str) -> None:
    """Size-matched uniform nulls with each group's observed dispersion overlaid."""
    order = [g for g in ("pareto", "top", "deploy") if g in res
             and res[g]["n"] >= 2]
    if not order:
        return
    fig, axes = plt.subplots(1, len(order), figsize=(4.2 * len(order), 3.4),
                             squeeze=False)
    for ax, g in zip(axes[0], order):
        k = res[g]["n"]
        null = subset_dispersion_null(U, k, n_perm, rng)["pairwise"]
        ax.hist(null, bins=40, color="#b0b0b0", alpha=0.85)
        ax.axvline(res[g]["mean_pairwise"], color=GROUP_COLORS[g], lw=2.2,
                   label=(f"observed {res[g]['mean_pairwise']:.3f}\n"
                          f"p={res[g]['uniform_null']['p_tighter']:.3f}"))
        ax.set_title(f"{g} (n={k}) vs size-matched uniform", fontsize=9)
        ax.set_xlabel("mean pairwise distance")
        ax.legend(fontsize=7)
    fig.suptitle(f"{dim}D dispersion against the whole search space", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  [plot] {out}")


def plot_deploy_proximity(rows: list[dict], dim: int, out: str) -> None:
    """Deploy configs' nearest-good-set distance as a percentile of the null."""
    if not rows:
        return
    tnames = sorted({t for r in rows for t in r["targets"]})
    if not tnames:
        return
    labels = [f"{r['method']}"
              + (f" (t{r['trial']})" if isinstance(r["trial"], int) else "")
              for r in rows]
    x = np.arange(len(rows))
    w = 0.8 / len(tnames)
    fig, ax = plt.subplots(figsize=(1.5 * len(rows) + 3.5, 3.6))
    for i, t in enumerate(tnames):
        vals = [r["targets"].get(t, {}).get("percentile", np.nan) for r in rows]
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w, label=f"vs {t}")
    ax.axhline(0.5, color="grey", ls="--", lw=1,
               label="no closer than a random config")
    ax.axhline(0.05, color="#2ca02c", ls=":", lw=1, label="inside good region (5%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("percentile of null nearest-neighbour distance\n(lower = deeper inside)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{dim}D deploy configs vs the good sets", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  [plot] {out}")


def plot_cross_dim(rows: list[dict], label_a: str, label_b: str, out: str) -> None:
    """Per-hyperparameter 10-90 bands of two dimensionalities' good sets, side by side."""
    if not rows:
        return
    names = [r["hparam"] for r in rows]
    y = np.arange(len(names))
    color = {"agrees": "#2ca02c", "disagrees": "#d62728", "both free": "#8c8c8c"}

    def vcolor(v: str) -> str:
        return color.get(v, "#7f7fbf")  # "compatible (…)" variants
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(names) + 2.2))
    for i, r in enumerate(rows):
        # A fully pinned band has zero width and would draw as nothing, so the
        # degenerate case gets a marker instead of an invisible line.
        for band, off, col in ((r["band_a"], 0.16, "#1f77b4"),
                               (r["band_b"], -0.16, "#ff7f0e")):
            if band[1] - band[0] < 0.01:
                ax.plot([band[0]], [y[i] + off], marker="|", ms=9, mew=2.5,
                        color=col)
            else:
                ax.plot(band, [y[i] + off] * 2, lw=5, color=col,
                        solid_capstyle="butt")
        ax.text(1.02, y[i], r["verdict"], fontsize=6.5, va="center",
                color=vcolor(r["verdict"]))
    ax.plot([], [], lw=5, color="#1f77b4", label=f"{label_a} good set (10-90%)")
    ax.plot([], [], lw=5, color="#ff7f0e", label=f"{label_b} good set (10-90%)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    # Room to the right of the [0, 1] cube for the verdict labels.
    ax.set_xlim(0, 1.5)
    ax.axvline(1.0, color="black", lw=0.8)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("normalised value in canonical bounds")
    ax.set_title(f"Does the good region sit in the same place in "
                 f"{label_a.upper()} and {label_b.upper()}?", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  [plot] {out}")


# ─── Report ──────────────────────────────────────────────────────────────────────

def _f(v, nd=3):
    """Format a possibly-missing float for the markdown tables."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


def write_report(summary: dict, names: list[str], out_dir: str) -> str:
    L: list[str] = []
    A = L.append
    A("# Hyperparameter clustering vs the whole search space")
    A("")
    A(f"Generated {summary['generated']}")
    A("")
    A(f"Search space: **{len(names)} hyperparameters** from "
      "`optimize/run_mobo.py:HPARAM_SPACE`, each mapped to [0, 1] within its "
      "canonical bounds. Distances are Euclidean in that unit cube.")
    A("")
    A(f"Nulls: {summary['n_null']} Sobol points (uniform = the whole space) and the "
      f"trial pool itself (explored). All dispersion statistics are size-matched "
      f"over {summary['n_perm']} random subsets; `p_tighter` is the fraction of "
      "size-matched random groups that are at least as tight as the observed one, "
      "so **small p = genuinely clustered**.")
    A("")
    A(f"Bounds policy: `{summary['bounds_mode']}`.")
    A("")

    for dim in summary["dims"]:
        d = summary["per_dim"][str(dim)]
        A(f"## {dim}D")
        A("")
        A(f"Reference pool: {d['pool_note']}")
        A("")
        A(f"- pool trials used: **{d['n_pool']}** "
          f"(from {d['n_source_runs']} run(s))")
        A(f"- objectives for the Pareto front: `{', '.join(d['objectives'])}`")
        A(f"- top set: best {summary['top_frac']:.0%} of the "
          f"{d['groups']['pareto']['n']}-point Pareto front by `dist_to_needles` "
          f"= **{d['groups']['top']['n']}** trial(s)")
        if d["groups"]["top"]["n"] < 5:
            A(f"  - with only {d['groups']['top']['n']} points, this group's "
              "dispersion row below has almost no statistical power — read it as "
              "*where* those configs sit, not as how tightly they cluster")
        A("")
        drift = d["encoding"]["out_of_bounds_counts"]
        if drift:
            A("**Bounds drift** (values outside the canonical space, "
              f"policy `{summary['bounds_mode']}`):")
            for k, v in sorted(drift.items(), key=lambda kv: -kv[1]):
                A(f"- `{k}`: {v} trial(s) ({100 * v / max(1, d['encoding']['n_in']):.0f}%)"
                  " — conclusions on this dimension are the least trustworthy")
            A("")
        if d["encoding"]["dropped_extra_hparams"]:
            A("Hyperparameters present in the pool but no longer tuned (dropped): "
              + ", ".join(f"`{h}`" for h in d["encoding"]["dropped_extra_hparams"]))
            A("")

        A("### Tightness relative to the whole space")
        A("")
        A("| group | n | mean pairwise | vs uniform | p_tighter (uniform) | "
          "centroid radius | vs uniform | vs pool | eff. dim (PR) | PCs for 90% |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for g in ("pool", "pareto", "top", "deploy"):
            if g not in d["groups"]:
                continue
            r = d["groups"][g]
            A(f"| {g} | {r['n']} | {_f(r['mean_pairwise'])} | "
              f"{_f(r['uniform_null']['ratio_pairwise'])}× | "
              f"{_f(r['uniform_null']['p_tighter'])} | "
              f"{_f(r['centroid_radius'])} | "
              f"{_f(r['uniform_null']['ratio_radius'])}× | "
              f"{_f(r['pool_null']['ratio_pairwise'])}× | "
              f"{_f(r['effective_dim']['participation_ratio'], 2)} | "
              f"{r['effective_dim']['n_pc_90'] or '—'} |")
        A("")
        A("`vs uniform` under 1.0 means tighter than a random group of the same "
          "size; `vs pool` compares against what the optimiser actually explored. "
          "Participation ratio is the effective number of directions the group "
          f"spreads along, out of {len(names)}.")
        A("")

        A("### Which hyperparameters are pinned")
        A("")
        for g in ("pareto", "top", "deploy"):
            if g not in d["groups"]:
                continue
            b = d["classified"][g]
            A(f"**{g}** — spread ratio = per-dimension std / uniform std:")
            for bucket, blurb in (("pinned", "agreed on (sensitive)"),
                                  ("intermediate", "partly constrained"),
                                  ("free", "unconstrained (insensitive)")):
                items = b[bucket]
                A(f"- *{bucket}* ({blurb}): "
                  + (", ".join(f"`{n}` {r:.2f}" for n, r in items) if items else "none"))
            A("")

        A("### Per-hyperparameter ranges")
        A("")
        A("In raw units, with the share of each axis the Pareto front occupies: "
          "see **[hparam_ranges.md](hparam_ranges.md)**.")
        A("")

        A("### Deploy configurations vs the good sets")
        A("")
        if d["deploy_proximity"]:
            tnames = sorted({t for r in d["deploy_proximity"] for t in r["targets"]})
            A("| deploy config | slots | source tuning run | "
              + " | ".join(f"d→{t} (percentile)" for t in tnames) + " |")
            A("|---" * (3 + len(tnames)) + "|")
            for r in d["deploy_proximity"]:
                cells = []
                for t in tnames:
                    tv = r["targets"].get(t)
                    cells.append("—" if tv is None else
                                 f"{_f(tv['d_min'])} ({_f(tv['percentile'])})")
                src = r["source_tuning_run"] or "—"
                A(f"| {r['method']}"
                  + (f" t{r['trial']}" if isinstance(r["trial"], int) else "")
                  + f" | {r['n_slots']} | `{os.path.basename(str(src))}` | "
                  + " | ".join(cells) + " |")
            A("")
            A("Percentile = fraction of uniform configs at least as close to that "
              "good set. **≲0.05 = sits inside the good region; ≈0.5 = no closer "
              "than a config drawn at random.**")
        else:
            A("No deploy configs found for this dimensionality.")
        A("")
        A(f"Circularity check: the deploy configs come from "
          f"`mobo_warmgp_{dim}d_*`/`mobo_mixture_{dim}d_*` tuning runs, whose "
          f"signatures differ from this reference pool, so no deploy point is a "
          f"member of the pool and these distances are not degenerate.")
        A("")

    if summary.get("cross_dim"):
        A("## Is the good region in the same place across dimensionalities?")
        A("")
        A(f"Each pair is compared on its best-{summary['top_frac']:.0%}-of-Pareto "
          "sets. `dist_to_needles` is the one objective defined comparably across "
          "pools; Pareto membership is not, since the 3d run scores `runtime_s` "
          "and the 4d/6d pools `avg_time_per_iter_s`. The larger set is "
          "subsampled to the smaller wherever a raw comparison would be "
          "confounded by group size.")
        A("")
        for cd in summary["cross_dim"]:
            la, lb = cd["label_a"], cd["label_b"]
            A(f"### {la.upper()} vs {lb.upper()}")
            A("")
            A(f"Sets: n_{la}={cd['n_a']}, n_{lb}={cd['n_b']}.")
            A("")
            A(f"- energy distance {_f(cd['energy']['energy'])}, permutation "
              f"p = **{_f(cd['energy']['p_value'])}** "
              f"({cd['energy']['n_perm']} shuffles) — small p means the two good "
              "sets occupy *different* regions")
            A(f"- {la}→{lb} mean nearest-neighbour distance "
              f"{_f(cd['nn_a_to_b']['d'])}, percentile of null "
              f"**{_f(cd['nn_a_to_b']['percentile'])}**")
            A(f"- {lb}→{la} mean nearest-neighbour distance "
              f"{_f(cd['nn_b_to_a']['d'])}, percentile of null "
              f"**{_f(cd['nn_b_to_a']['percentile'])}**")
            A(f"- size-matched dispersion: {cd['subsampled']} subsampled to "
              f"n={cd['n_matched']} gives mean pairwise "
              f"{_f(cd['matched']['mean_subsampled'])} "
              f"(±{_f(cd['matched']['std_subsampled'])}) vs "
              f"{_f(cd['matched']['mean_other'])} for the other")
            A("")
            A(f"| hyperparameter | {la} band (10-90%) | {lb} band (10-90%) | "
              "overlap | verdict |")
            A("|---|---|---|---|---|")
            for r in cd["agreement"]:
                A(f"| `{r['hparam']}` | "
                  f"[{r['band_a'][0]:.2f}, {r['band_a'][1]:.2f}] | "
                  f"[{r['band_b'][0]:.2f}, {r['band_b'][1]:.2f}] | "
                  f"{_f(r['overlap_frac'], 2)} | {r['verdict']} |")
            A("")
            counts = {}
            for r in cd["agreement"]:
                counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
            A("Summary: "
              + ", ".join(f"**{v}** {k}" for k, v in sorted(counts.items())))
            A("")

    A("## Caveats")
    A("")
    for c in summary["caveats"]:
        A(f"- {c}")
    A("")

    path = os.path.join(out_dir, "hparam_sensitivity.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  [report] {path}")
    return path


def write_ranges(summary: dict, names: list[str], out_dir: str) -> str:
    """Standalone per-hyperparameter range tables, one per dimensionality.

    Rows are ordered most-constrained first (by the 10-90 share of the axis), so
    the hyperparameters the objective is sensitive to come out on top.
    """
    L: list[str] = []
    A = L.append
    A("# Per-hyperparameter ranges of the good configurations")
    A("")
    A(f"Generated {summary['generated']} — companion to "
      "[hparam_sensitivity.md](hparam_sensitivity.md).")
    A("")
    A("Ranges are in **raw units**, as you would write them into a config. Two "
      "shares of each axis are given, both measured in the *normalised* "
      "coordinate (so for the log-scaled `nat_grad_step` a share is a share of "
      "the decades, not of the linear span):")
    A("")
    A("- **% axis (10-90)** — the robust one. Use this.")
    A("- **% axis (full)** — from the min/max, so a single straggler can inflate "
      "it. Where the two disagree badly, the front is concentrated with outliers.")
    A("")
    A("The pool row bounds what the optimiser actually sampled, so a Pareto range "
      "can never exceed it — a wide Pareto range on a narrow pool axis says more "
      "about the search than about the objective.")
    A("")
    A("Rows are sorted most-constrained first. **Bold** marks the "
      "hyperparameters the good configurations genuinely agree on "
      "(10-90 share below 35%).")
    A("")

    for dim in summary["dims"]:
        d = summary["per_dim"][str(dim)]
        rows = sorted(d["hparam_ranges"],
                      key=lambda r: (r["groups"].get("pareto", {})
                                     .get("frac_10_90", 1.0)))
        npareto = d["groups"]["pareto"]["n"]
        A(f"## {dim}D — Pareto front of {npareto} trials "
          f"(pool of {d['n_pool']})")
        A("")
        A("| hyperparameter | canonical range | pool range | Pareto range (full) | "
          "Pareto 10-90% | % axis (10-90) | % axis (full) |")
        A("|---|---|---|---|---|---|---|")
        for r in rows:
            c = r["canonical"]
            pool = r["groups"].get("pool")
            par = r["groups"].get("pareto")
            fmt = ((lambda v: f"{v:g}") if r["transform"] == "int"
                   else (lambda v: f"{v:.4g}"))
            pfmt = (lambda v: f"{v:.3g}")   # percentiles interpolate off-grid
            warn = (" ⚠" if (par and par.get("out_of_canonical"))
                    or (pool and pool.get("out_of_canonical")) else "")
            band = (f"**{100 * par['frac_10_90']:.0f}%**"
                    if par and par["frac_10_90"] < 0.35
                    else f"{100 * par['frac_10_90']:.0f}%" if par else "—")
            A(f"| `{r['hparam']}`{' (log)' if r['transform'] == 'log' else ''} | "
              f"{fmt(c[0])} – {fmt(c[1])} | "
              + (f"{fmt(pool['min'])} – {fmt(pool['max'])}" if pool else "—")
              + " | "
              + (f"{fmt(par['min'])} – {fmt(par['max'])}" if par else "—")
              + " | "
              + (f"{pfmt(par['p10'])} – {pfmt(par['p90'])}" if par else "—")
              + f" | {band} | "
              + (f"{100 * par['frac']:.0f}%{warn}" if par else "—") + " |")
        A("")
        flagged = [r["hparam"] for r in rows
                   if any(g.get("out_of_canonical") for g in r["groups"].values())]
        if flagged:
            A("⚠ = observed values run outside the canonical bounds "
              f"(policy `{summary['bounds_mode']}`), so they were clamped when "
              "normalised and both percentages **understate** the true spread: "
              + ", ".join(f"`{h}`" for h in flagged)
              + ". Re-run with `--bounds union` to widen those axes to the data.")
            A("")

    A("## Why these shares do not multiply into a volume")
    A("")
    A(f"Multiplying the shares across all {len(names)} axes would treat the good "
      "region as an axis-aligned box, and any single fully-pinned hyperparameter "
      "drives the product to zero. The good region is a lower-dimensional slab, "
      "not a small box — the effective-dimension column in the main report is the "
      "honest summary of its shape. Per-axis shares are interpretable; their "
      "product is not.")
    A("")

    path = os.path.join(out_dir, "hparam_ranges.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  [report] {path}")
    return path


# ─── Main ────────────────────────────────────────────────────────────────────────

def analyse_dim(dim: int, args, space: dict, names: list[str],
                U: np.ndarray, rng: np.random.Generator) -> dict:
    print("=" * 70)
    print(f"[{dim}D]")
    pool_info = load_pool(dim, args.runs_dir, own_only=args.own_runs_only)
    records = pool_info["records"]
    print(f"[pool {dim}d] {len(records)} trial(s) from "
          f"{len({r['source_run'] for r in records})} run(s)")

    X, kept, enc = encode(records, names, space, args.bounds)
    if len(X) < 3:
        raise SystemExit(f"{dim}D pool has too few usable trials ({len(X)}).")
    if enc["out_of_bounds_counts"]:
        print(f"  [bounds] out-of-canonical values ({args.bounds}): "
              + ", ".join(f"{k}×{v}" for k, v in
                          sorted(enc["out_of_bounds_counts"].items(),
                                 key=lambda kv: -kv[1])))
    if enc["dropped_extra_hparams"]:
        print("  [schema] dropped no-longer-tuned hparams: "
              + ", ".join(enc["dropped_extra_hparams"]))

    M, obj_cols = objective_matrix(kept)
    pmask = pareto_mask_min(M)
    dist = np.asarray([float(r["metrics"][DIST_KEY]) for r in kept])
    # The `top` group is the best --top-frac *of the Pareto front* by
    # dist_to_needles, not of the whole pool: it is the sharp end of the front, so
    # it is a subset of `pareto` by construction.
    pareto_idx = np.where(pmask)[0]
    k_top = max(2, int(math.ceil(args.top_frac * len(pareto_idx))))
    top_idx = pareto_idx[np.argsort(dist[pareto_idx])[:k_top]]
    print(f"  [groups] pareto={len(pareto_idx)}  "
          f"top{args.top_frac:.0%}-of-pareto={len(top_idx)} "
          f"(dist_to_needles ≤ {dist[top_idx].max():.3f})")
    if len(top_idx) < 5:
        print(f"  [groups] WARNING: top set is only {len(top_idx)} point(s) — its "
              "dispersion statistics carry almost no power; consider a larger "
              "--top-frac.")

    deploy = load_deploy_configs(dim, args.runs_dir)
    Xd, deploy_kept, denc = (encode(deploy, names, space, args.bounds)
                             if deploy else
                             (np.zeros((0, len(names))), [], {}))

    groups = {
        "pool": X,
        "pareto": X[pmask],
        "top": X[top_idx],
        "deploy": Xd,
    }
    group_records = {
        "pool": kept,
        "pareto": [kept[i] for i in np.where(pmask)[0]],
        "top": [kept[i] for i in top_idx],
        "deploy": deploy_kept,
    }
    res, classified = {}, {}
    for g, Xg in groups.items():
        if len(Xg) == 0:
            continue
        res[g] = analyse_group(g, Xg, names, U, X, args.n_perm, rng)
        classified[g] = classify_hparams(res[g], names)
        print(f"  [{g:6s}] n={res[g]['n']:3d}  mean pairwise "
              f"{res[g]['mean_pairwise']:.3f}  "
              f"({res[g]['uniform_null']['ratio_pairwise']:.2f}× uniform, "
              f"p_tighter={res[g]['uniform_null']['p_tighter']:.3f})")

    prox = (deploy_proximity(Xd, deploy_kept,
                             {"pareto": groups["pareto"], "top": groups["top"]}, U)
            if len(Xd) else [])
    for r in prox:
        for t, tv in r["targets"].items():
            print(f"  [deploy] {r['method']:9s} → {t:6s} d_min={tv['d_min']:.3f} "
                  f"percentile={tv['percentile']:.3f}")

    tag = f"{dim}d"
    plot_std_heatmap(res, names, dim,
                     os.path.join(args.out, f"spread_heatmap_{tag}.png"))
    plot_parallel(groups, names, dim,
                  os.path.join(args.out, f"parallel_coords_{tag}.png"))
    plot_null_hist(res, U, args.n_perm, rng, dim,
                   os.path.join(args.out, f"null_dispersion_{tag}.png"))
    plot_deploy_proximity(prox, dim,
                          os.path.join(args.out, f"deploy_proximity_{tag}.png"))

    return {
        "pool_note": pool_info["note"],
        "n_pool": len(kept),
        "n_source_runs": len({r["source_run"] for r in kept}),
        "objectives": obj_cols,
        "encoding": enc,
        "deploy_encoding": denc,
        "groups": res,
        "classified": classified,
        "deploy_proximity": prox,
        "hparam_ranges": hparam_ranges(group_records, groups, names, space),
        "_X": groups,
    }


def compare_dims(per_dim: dict, dim_a: int, dim_b: int, names: list[str],
                 U: np.ndarray, args, rng: np.random.Generator) -> dict:
    """One pairwise comparison of two dimensionalities' good sets."""
    la, lb = f"{dim_a}d", f"{dim_b}d"
    A = per_dim[str(dim_a)]["_X"]["top"]
    B = per_dim[str(dim_b)]["_X"]["top"]
    print("=" * 70)
    print(f"[cross-dim] {la} top n={len(A)} vs {lb} top n={len(B)}")

    energy = energy_test(A, B, args.n_perm, rng)
    nn_ab = float(cross_dmin(A, B).mean())
    nn_ba = float(cross_dmin(B, A).mean())
    null_ab = cross_dmin(U, B)
    null_ba = cross_dmin(U, A)

    # Size-match the larger set to the smaller before comparing dispersion: mean
    # pairwise distance shrinks with n, so the raw numbers are not comparable.
    k = min(len(A), len(B))
    big, small_is_a = (A, False) if len(A) >= len(B) else (B, True)
    sub = [pdist_mean(big[rng.choice(len(big), size=k, replace=False)])
           for _ in range(args.n_subsample)]

    out = {
        "dim_a": dim_a,
        "dim_b": dim_b,
        "label_a": la,
        "label_b": lb,
        "n_a": len(A),
        "n_b": len(B),
        "n_matched": k,
        "n_subsample": args.n_subsample,
        "subsampled": lb if small_is_a else la,
        "energy": energy,
        "nn_a_to_b": {"d": nn_ab, "percentile": _pct_le(null_ab, nn_ab),
                      "null_mean": float(null_ab.mean())},
        "nn_b_to_a": {"d": nn_ba, "percentile": _pct_le(null_ba, nn_ba),
                      "null_mean": float(null_ba.mean())},
        "matched": {"mean_subsampled": float(np.mean(sub)),
                    "std_subsampled": float(np.std(sub)),
                    "mean_other": pdist_mean(A if small_is_a else B)},
        "agreement": cross_dim_agreement(A, B, names, la, lb),
    }
    print(f"  [cross-dim] energy={energy['energy']:.4f} p={energy['p_value']:.3f}  "
          f"{la}→{lb} nn={nn_ab:.3f} (pct {out['nn_a_to_b']['percentile']:.3f})")
    counts = {}
    for r in out["agreement"]:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("  [cross-dim] per-hparam verdicts: "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    plot_cross_dim(out["agreement"], la, lb,
                   os.path.join(args.out, f"cross_dim_agreement_{la}_{lb}.png"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cluster analysis of hyperparameter configurations in the "
                    "normalised search space.")
    p.add_argument("--dim", default="all",
                   choices=["3", "4", "6", "10", "both", "all"],
                   help="dimensionality to analyse: one dim, `both` (3+4, the "
                        "historical default) or `all` (3+4+6+10, the default)")
    p.add_argument("--top-frac", type=float, default=0.10,
                   help="fraction of the PARETO FRONT, best by dist_to_needles, "
                        "that forms the 'top' group (default: 0.10). At 10% this "
                        "is only a few points; raise it for a group large enough "
                        "to support a dispersion claim.")
    p.add_argument("--n-null", type=int, default=20000,
                   help="Sobol points sampled from the cube as the uniform null")
    p.add_argument("--n-perm", type=int, default=2000,
                   help="size-matched subsets / label shuffles per test")
    p.add_argument("--n-subsample", type=int, default=2000,
                   help="draws when size-matching the 3d set to the 4d set")
    p.add_argument("--bounds", default="clip", choices=["clip", "drop", "union"],
                   help="policy for values outside the canonical space "
                        "(default: clip, and flag the affected dimensions)")
    p.add_argument("--own-runs-only", action="store_true",
                   help="restrict the signature-matched pools to this account's "
                        "runs dirs "
                        "(collaborator runs are pooled by default, matching "
                        "pareto.py's signature rule)")
    p.add_argument("--runs-dir", default=RUNS_DIR_DEFAULT,
                   help="runs directory to resolve the reference pools against")
    p.add_argument("--out", default=os.path.join(_HERE, "hparam_sensitivity"),
                   help="output directory for the report and figures")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not 0.0 < args.top_frac <= 1.0:
        raise SystemExit("--top-frac must be in (0, 1]")
    os.makedirs(args.out, exist_ok=True)

    space = load_canonical_space()
    names = list(space)
    rng = np.random.default_rng(args.seed)
    print(f"[space] {len(names)} canonical hyperparameters from "
          "optimize/run_mobo.py")
    print(f"[null] {args.n_null} Sobol points in [0,1]^{len(names)} "
          f"(uniform mean pairwise distance ~ {math.sqrt(len(names) / 6):.3f})")
    U = sobol_null(args.n_null, len(names), args.seed)

    dims = ([3, 4] if args.dim == "both" else
            list(DIMS) if args.dim == "all" else [int(args.dim)])
    per_dim = {}
    for dim in dims:
        per_dim[str(dim)] = analyse_dim(dim, args, space, names, U, rng)

    # Every pair of the analysed dimensionalities, so 3-vs-4 conclusions stay
    # comparable with the newer 3-vs-6 / 4-vs-6 ones.
    cross = [compare_dims(per_dim, a, b, names, U, args, rng)
             for i, a in enumerate(dims) for b in dims[i + 1:]]

    caveats = [
        "Euclidean distance in the normalised cube weights all "
        f"{len(names)} hyperparameters equally and treats integers as continuous; "
        "the per-hyperparameter spread table is what keeps a single encoding "
        "choice from driving the conclusion.",
        "The 3d reference is one archived run whose run_config.json records no "
        "dim/dataset/hparam_space, so its dimensionality is asserted, not "
        "verified, and it cannot be signature-pooled with siblings.",
        "The 3d pool scores the legacy `runtime_s` while the 4d and 6d pools "
        "score `avg_time_per_iter_s`, so Pareto membership is not defined on the "
        "same objective triple across dimensionalities — cross-dim claims use the "
        "best-of-Pareto-by-dist_to_needles sets instead.",
        "Each 4d/6d run randomises its ensemble landscape (the signature "
        "deliberately excludes ensemble_seed), so those good sets are averaged "
        "over landscapes while the 3d run is a single fixed one.",
        "`dist_to_needles` is not comparable in magnitude across "
        "dimensionalities — the needles differ and the space is larger — so the "
        "`top` cut selects the sharp end *within* each pool, never across pools.",
        "A group's apparent spread partly reflects which bounds its sampler drew "
        "from, not only where good configurations live; newer runs record their "
        "own space at run_config.json:invocation.hparam_space, the 3d reference "
        "does not.",
        "Group sizes are small by construction, so every statistic is a "
        "distance one with a size-matched permutation null; no covariance is "
        "estimated and no asymptotic test is used.",
    ]
    caveats.append(
        "The signature-matched pools span both accounts. eve_lal's older-fork "
        "runs never stored "
        "the `variant` key; pareto._load_run_signature backfills it from the "
        "folder name, which is what admits them — the same rule the Pareto plots "
        "use, but it is a name-based inference, not a recorded fact."
        if not args.own_runs_only else
        "--own-runs-only was set: the signature-matched pools exclude collaborator "
        "runs, so they are smaller than the pools the Pareto plots use.")

    summary = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "dims": dims,
        "top_frac": args.top_frac,
        "n_null": args.n_null,
        "n_perm": args.n_perm,
        "bounds_mode": args.bounds,
        "hparams": names,
        "canonical_space": {k: list(v) for k, v in space.items()},
        "per_dim": per_dim,
        "cross_dim": cross,
        "caveats": caveats,
    }
    write_report(summary, names, args.out)
    write_ranges(summary, names, args.out)

    for d in summary["per_dim"].values():
        d.pop("_X", None)
    jpath = os.path.join(args.out, "hparam_sensitivity.json")
    with open(jpath, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  [json] {jpath}")
    print("=" * 70)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
