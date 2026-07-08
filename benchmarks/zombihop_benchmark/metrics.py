from __future__ import annotations

import math
from typing import Any

import numpy as np

from .spaces import ilr_distance
from .types import ObjectiveInfo


def best_y_so_far(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    return np.maximum.accumulate(arr)


def dup_fraction(X: np.ndarray, duplicate_radius_ilr: float) -> float:
    arr = np.asarray(X, dtype=float)
    if arr.shape[0] <= 1:
        return 0.0
    dists = ilr_distance(arr, arr)
    duplicates = 0
    for i in range(1, arr.shape[0]):
        if np.any(dists[i, :i] < duplicate_radius_ilr):
            duplicates += 1
    return duplicates / arr.shape[0]


def dist_to_needles(X: np.ndarray, true_needles: np.ndarray | None) -> float:
    if true_needles is None or len(true_needles) == 0:
        return math.nan
    dists = ilr_distance(true_needles, X)
    return float(np.mean(np.min(dists, axis=1)))


def pct_matched(X: np.ndarray, true_needles: np.ndarray | None, match_radius_ilr: float | None) -> float:
    if true_needles is None or len(true_needles) == 0 or match_radius_ilr is None:
        return math.nan
    dists = ilr_distance(true_needles, X)
    matched = np.min(dists, axis=1) <= match_radius_ilr
    return float(100.0 * np.mean(matched))


def compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    objective_info: ObjectiveInfo,
    duplicate_radius_ilr: float,
    match_radius_ilr: float | None,
    runtime_s: float,
    step: int,
) -> dict[str, Any]:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    radius = match_radius_ilr if match_radius_ilr is not None else objective_info.match_radius_ilr
    best = float(np.max(y_arr)) if y_arr.size else math.nan
    return {
        "step": int(step),
        "best_y_so_far": best,
        "dist_to_needles": dist_to_needles(X, objective_info.true_needles),
        "pct_matched": pct_matched(X, objective_info.true_needles, radius),
        "dup_fraction": dup_fraction(X, duplicate_radius_ilr),
        "runtime_s": float(runtime_s),
        "num_points": int(len(y_arr)),
    }

