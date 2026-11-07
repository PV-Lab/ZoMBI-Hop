"""
benchmarks/methods/runner.py
============================
Run one method on one problem, score it against the ground truth, write the cell.

A cell directory holds::

    points.csv               every measured sample: batch, noisy y, noiseless f,
                             coordinates (x0..), requested coordinates (rx0..) when
                             input noise is on, and any per-batch tags / per-point
                             columns the method adds (ZoMBI-Hop: activation, zoom,
                             penalized)
    needles.csv              the needles the cell is scored on: the method's own
                             declarations for a declaring method, the extractor's
                             for the rest, with the distance to the nearest true optimum
    needles_extracted.csv    declaring methods only: the extractor's needles from
                             the same samples (the controlled comparison)
    metrics_over_time.csv    one row per measured batch (see below)
    method.json              method, full config, seed, device, extractor, problem
    metrics.json             final scalars. Written LAST and atomically, so its
                             presence is the completion marker.
    error.log                only if the method raised

Scoring
-------
``dist_to_needles`` is ``optimize/eval_metrics.metric_dist_to_needles`` unchanged —
the repo's one definition (optimal one-to-one matching, distances capped at and
unmatched members charged ``UNMATCHED_PENALTY``). Alongside it:

    n_needles             size of the scored needle set
    frac_optima_found     true optima with a needle within MATCH_RADIUS (recall)
    needle_precision      needles within MATCH_RADIUS of a true optimum
    frac_optima_visited   true optima with a SAMPLE within MATCH_RADIUS — whether
                          the method ever measured there, needles aside
    best_f                best noiseless value sampled
    median_nn_spacing     eval_metrics.metric_median_nn_spacing of the samples

``*_extracted`` variants score the extractor's needles for every method.

Trajectories
------------
Every row of ``metrics_over_time.csv`` is one measured batch. Cheap columns
(best_f, visits, a declaring method's own needles) are filled on every row; the
extractor is a GP fit, so its columns are filled only every ``trace_every`` batches
and on the last one, and are blank elsewhere.
"""

from __future__ import annotations

import json
import os
import random
import time
import traceback
from dataclasses import dataclass

import numpy as np

from ._paths import ensure_paths
from .base import BudgetExhausted, Method, Problem, TimeLimitReached

ensure_paths()

from eval_metrics import (  # noqa: E402
    MATCH_RADIUS,
    UNMATCHED_PENALTY,
    metric_dist_to_needles,
    metric_median_nn_spacing,
)

METRICS_FILE = "metrics.json"
REQUIRED_METRIC_KEYS = ("method", "dist_to_needles", "n_points", "budget_hit")


@dataclass
class GroundTruth:
    """What the runner scores against, and the method never sees."""

    optima: np.ndarray        # (k, dim)
    peak_value: float         # noiseless global maximum (1.0 on the needle landscapes)


# ─── Scoring ─────────────────────────────────────────────────────────────────────

def _nearest(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """For each row of A, distance to the nearest row of B (inf if B is empty)."""
    if len(A) == 0:
        return np.empty(0)
    if len(B) == 0:
        return np.full(len(A), np.inf)
    from scipy.spatial import cKDTree

    return cKDTree(B).query(A, k=1)[0]


def score_needles(needles: np.ndarray, truth: GroundTruth) -> dict:
    opt = truth.optima
    n = int(len(needles))
    return {
        "dist_to_needles": round(metric_dist_to_needles(needles, list(opt)), 6),
        "n_needles": n,
        "frac_optima_found": round(float((_nearest(opt, needles) <= MATCH_RADIUS).mean())
                                   if len(opt) else 0.0, 6),
        "needle_precision": round(float((_nearest(needles, opt) <= MATCH_RADIUS).mean())
                                  if n else 0.0, 6),
    }


class _VisitTracker:
    """Running min distance from each true optimum to the samples, batch by batch.

    ``frac_optima_visited`` is the fraction of optima whose minimum is inside
    ``MATCH_RADIUS``. Keeping it incremental makes the trajectory free instead of
    re-scanning the whole prefix at every batch.
    """

    def __init__(self, optima: np.ndarray) -> None:
        self.optima = optima
        self.dmin = np.full(len(optima), np.inf)

    def add(self, X: np.ndarray) -> float:
        if len(X) and len(self.optima):
            self.dmin = np.minimum(self.dmin, _nearest(self.optima, X))
        return float((self.dmin <= MATCH_RADIUS).mean()) if len(self.optima) else 0.0


# ─── Seeding / IO ────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    """Python, NumPy and torch global RNGs (HEBO and parts of BoTorch use them)."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def atomic_write_json(path: str, obj) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_metrics(trial_dir: str) -> dict | None:
    """A finished cell's metrics, or None if absent or truncated."""
    path = os.path.join(trial_dir, METRICS_FILE)
    try:
        with open(path) as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return m if all(k in m for k in REQUIRED_METRIC_KEYS) else None


def is_complete(trial_dir: str) -> bool:
    return load_metrics(trial_dir) is not None


def _xcols(prefix: str, d: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(d)]


def _write_needles(path: str, needles: np.ndarray, records: list[dict],
                   truth: GroundTruth, d: int) -> None:
    import pandas as pd

    df = pd.DataFrame(np.asarray(needles).reshape(-1, d), columns=_xcols("x", d))
    df.insert(0, "needle_idx", np.arange(len(df)))
    if records and len(records) == len(df):
        extra = pd.DataFrame(records)
        for c in extra.columns:
            df[c] = extra[c].to_numpy()
    df["dist_to_nearest_optimum"] = _nearest(np.asarray(needles).reshape(-1, d),
                                             truth.optima)
    df.to_csv(path, index=False)


# ─── The run ─────────────────────────────────────────────────────────────────────

def run_method(method: Method, problem: Problem, truth: GroundTruth, trial_dir: str, *,
               extractor, trace_every: int = 10, extra_record: dict | None = None,
               verbose: bool = True) -> dict:
    """Run ``method`` on ``problem`` until it stops, score it, write the cell.

    Returns the metrics dict. Raises whatever the method raises other than the
    budget/time stop — after writing ``error.log`` and WITHOUT writing
    ``metrics.json``, so a crashed cell is never counted as finished.
    """
    import pandas as pd

    os.makedirs(trial_dir, exist_ok=True)
    stale = os.path.join(trial_dir, METRICS_FILE)
    if os.path.exists(stale):
        os.remove(stale)
    d = problem.dim
    seed_everything(method.seed)

    # Declared-needle snapshots, keyed by the number of batches measured so far.
    snapshots: dict[int, np.ndarray] = {}
    if method.declares_needles:
        def _snap(p: Problem) -> None:
            nd = method.declared_needles()
            snapshots[p.n_batches] = (np.empty((0, d)) if nd is None
                                      else np.array(nd, dtype=float).reshape(-1, d))
        problem.before_batch.append(_snap)

    run_record = {
        "method": method.name, "class": f"{type(method).__module__}.{type(method).__name__}",
        "config": method.config, "seed": method.seed, "device": method.device,
        "declares_needles": method.declares_needles,
        "extractor": extractor.config(),
        "problem": {"dim": d, "budget": problem.budget, "batch_size": problem.batch_size,
                    "input_noise": problem.input_noise,
                    "output_noise_frac": problem.output_noise_frac},
        **(extra_record or {}),
    }
    with open(os.path.join(trial_dir, "method.json"), "w") as f:
        json.dump(run_record, f, indent=2, default=_json_default)

    if verbose:
        print(f"  [run] {method.name}  dim={d}  budget={problem.budget}  "
              f"batch={problem.batch_size}  seed={method.seed}  device={method.device}",
              flush=True)
    t0 = time.time()
    stop = "method_returned"
    try:
        method.run(problem)
    except TimeLimitReached as exc:
        stop = "time"
        if verbose:
            print(f"    [stop] {exc}", flush=True)
    except BudgetExhausted as exc:
        stop = "budget"
        if verbose:
            print(f"    [stop] {exc}", flush=True)
    except Exception:
        with open(os.path.join(trial_dir, "error.log"), "a") as f:
            f.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            traceback.print_exc(file=f)
        raise
    runtime = time.time() - t0
    if stop == "method_returned" and problem.remaining == 0:
        stop = "budget"
    if method.declares_needles:
        _snap(problem)   # the final state, after the last batch

    # ── score ──
    t1 = time.time()
    X, Y, F = problem.X, problem.Y, problem._true_values()
    batch = problem._batch_index()
    n_batches = problem.n_batches
    ends = np.cumsum(np.bincount(batch, minlength=n_batches)) if n_batches else np.empty(0, int)
    every = max(0, int(trace_every))
    checkpoints = {k for k in range(1, n_batches + 1) if every and k % every == 0}
    if n_batches:
        checkpoints.add(n_batches)

    extracted: dict[int, tuple[np.ndarray, list[dict]]] = {}
    for k in sorted(checkpoints):
        n = int(ends[k - 1])
        extracted[k] = extractor(X[:n], Y[:n])

    visits = _VisitTracker(truth.optima)
    best_f = -np.inf
    rows = []
    for k in range(1, n_batches + 1):
        lo = int(ends[k - 2]) if k > 1 else 0
        n = int(ends[k - 1])
        frac_visited = visits.add(X[lo:n])
        best_f = max(best_f, float(F[lo:n].max()))
        row = {"batch": k, "n_points": n, "best_f": round(best_f, 6),
               "frac_optima_visited": round(frac_visited, 6)}
        native = None
        if method.declares_needles:
            # The snapshot taken before batch k+1 is the state after batch k.
            native = snapshots.get(k)
        elif k in extracted:
            native = extracted[k][0]
        if native is not None:
            row.update(score_needles(native, truth))
        if k in extracted:
            s = score_needles(extracted[k][0], truth)
            row.update({f"{key}_extracted": v for key, v in s.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(trial_dir, "metrics_over_time.csv"), index=False)

    ex_needles, ex_records = (extracted[n_batches] if n_batches
                              else (np.empty((0, d)), []))
    if method.declares_needles:
        needles = snapshots.get(n_batches, np.empty((0, d)))
        records = method.needle_records()
        _write_needles(os.path.join(trial_dir, "needles_extracted.csv"),
                       ex_needles, ex_records, truth, d)
    else:
        needles, records = ex_needles, ex_records
    _write_needles(os.path.join(trial_dir, "needles.csv"), needles, records, truth, d)

    # points.csv
    pts = {"sample_idx": np.arange(len(Y)), "batch": batch + 1, "y": Y, "f": F}
    for i in range(d):
        pts[f"x{i}"] = X[:, i]
    if problem.input_noise > 0:
        Xr = problem.X_requested
        for i in range(d):
            pts[f"rx{i}"] = Xr[:, i]
    pts.update(problem._tag_columns())
    for key, col in method.point_columns().items():
        if len(col) == len(Y):
            pts[key] = col
    pd.DataFrame(pts).to_csv(os.path.join(trial_dir, "points.csv"), index=False)

    try:
        method.write_artifacts(trial_dir)
    except Exception as exc:  # noqa: BLE001 — an optional artifact never fails a cell
        print(f"    [run] {method.name}.write_artifacts failed: {exc}", flush=True)

    final = score_needles(needles, truth)
    final_ex = score_needles(ex_needles, truth)
    metrics = {
        "method": method.name,
        "dim": d,
        "seed": method.seed,
        "needles_source": "declared" if method.declares_needles else "extracted",
        **final,
        **{f"{k}_extracted": v for k, v in final_ex.items()},
        "frac_optima_visited": rows[-1]["frac_optima_visited"] if rows else 0.0,
        "best_f": rows[-1]["best_f"] if rows else None,
        "median_nn_spacing": (round(metric_median_nn_spacing(X), 8) if len(X) > 1 else None),
        "n_points": int(problem.n_evaluated),
        "n_batches": int(n_batches),
        "budget": int(problem.budget),
        "budget_hit": bool(problem.remaining == 0),
        "stop_reason": stop,
        "n_clipped_coords": int(problem.n_clipped),
        "runtime_s": round(runtime, 3),
        "scoring_s": round(time.time() - t1, 3),
        "match_radius": MATCH_RADIUS,
        "unmatched_penalty": UNMATCHED_PENALTY,
        "method_summary": method.summary(),
    }
    atomic_write_json(os.path.join(trial_dir, METRICS_FILE), metrics)
    if verbose:
        print(f"  [run] {method.name} done — dist={metrics['dist_to_needles']:.4f} "
              f"(extracted {metrics['dist_to_needles_extracted']:.4f})  "
              f"needles={metrics['n_needles']}  "
              f"points={metrics['n_points']}/{problem.budget}  stop={stop}  "
              f"({runtime:.1f}s run, {metrics['scoring_s']:.1f}s scoring)", flush=True)
    return metrics
