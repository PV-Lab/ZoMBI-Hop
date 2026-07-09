from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .spaces import ilr_distance
from .types import ObjectiveInfo

DEFAULT_MATCH_RADIUS_COMP = 0.05
DEFAULT_DUPLICATE_RADIUS_COMP = 0.032


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


def dup_fraction_comp(X: np.ndarray, duplicate_radius_comp: float) -> float:
    arr = np.asarray(X, dtype=float)
    if arr.shape[0] <= 1:
        return 0.0
    dists = composition_l2_distance(arr, arr)
    duplicates = 0
    for i in range(1, arr.shape[0]):
        if np.any(dists[i, :i] < duplicate_radius_comp):
            duplicates += 1
    return duplicates / arr.shape[0]


def dup_fraction_comp_cross_line(
    X: np.ndarray,
    duplicate_radius_comp: float,
    line_group_ids: Sequence[Any] | None,
) -> float:
    if line_group_ids is None:
        return dup_fraction_comp(X, duplicate_radius_comp)
    arr = np.asarray(X, dtype=float)
    if arr.shape[0] <= 1:
        return 0.0
    groups = list(line_group_ids)
    if len(groups) != arr.shape[0]:
        raise ValueError("line_group_ids must have the same length as X")
    dists = composition_l2_distance(arr, arr)
    duplicates = 0
    for i in range(1, arr.shape[0]):
        prior_cross_line = [j for j in range(i) if groups[j] != groups[i]]
        if prior_cross_line and np.any(dists[i, prior_cross_line] < duplicate_radius_comp):
            duplicates += 1
    return duplicates / arr.shape[0]


def composition_l2_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(Y, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(1, -1)
    if x_arr.ndim != 2 or y_arr.ndim != 2 or x_arr.shape[1] != y_arr.shape[1]:
        raise ValueError("X and Y must be 1D/2D arrays with the same number of simplex components")
    diff = x_arr[:, None, :] - y_arr[None, :, :]
    return np.linalg.norm(diff, axis=2)


def dist_to_needles_ilr(X: np.ndarray, true_needles: np.ndarray | None) -> float:
    if true_needles is None or len(true_needles) == 0:
        return math.nan
    dists = ilr_distance(true_needles, X)
    return float(np.mean(np.min(dists, axis=1)))


def dist_to_needles_comp(X: np.ndarray, true_needles: np.ndarray | None) -> float:
    if true_needles is None or len(true_needles) == 0:
        return math.nan
    dists = composition_l2_distance(true_needles, X)
    return float(np.mean(np.min(dists, axis=1)))


def pct_matched_ilr(X: np.ndarray, true_needles: np.ndarray | None, match_radius_ilr: float | None) -> float:
    if true_needles is None or len(true_needles) == 0 or match_radius_ilr is None:
        return math.nan
    dists = ilr_distance(true_needles, X)
    matched = np.min(dists, axis=1) <= match_radius_ilr
    return float(100.0 * np.mean(matched))


def pct_matched_comp(X: np.ndarray, true_needles: np.ndarray | None, match_radius_comp: float | None) -> float:
    if true_needles is None or len(true_needles) == 0 or match_radius_comp is None:
        return math.nan
    dists = composition_l2_distance(true_needles, X)
    matched = np.min(dists, axis=1) <= match_radius_comp
    return float(100.0 * np.mean(matched))


# Backward-compatible aliases. Before Step 9 these headline names were ILR based.
dist_to_needles = dist_to_needles_ilr
pct_matched = pct_matched_ilr


def compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    objective_info: ObjectiveInfo,
    duplicate_radius_ilr: float,
    match_radius_ilr: float | None,
    runtime_s: float,
    step: int,
    duplicate_radius_comp: float = DEFAULT_DUPLICATE_RADIUS_COMP,
    match_radius_comp: float | None = None,
    line_group_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    ilr_radius = match_radius_ilr if match_radius_ilr is not None else objective_info.match_radius_ilr
    comp_radius = (
        match_radius_comp
        if match_radius_comp is not None
        else (
            objective_info.match_radius_comp
            if objective_info.match_radius_comp is not None
            else DEFAULT_MATCH_RADIUS_COMP
        )
    )
    best = float(np.max(y_arr)) if y_arr.size else math.nan
    dist_ilr = dist_to_needles_ilr(X, objective_info.true_needles)
    pct_ilr = pct_matched_ilr(X, objective_info.true_needles, ilr_radius)
    dup_ilr = dup_fraction(X, duplicate_radius_ilr)
    dist_comp = dist_to_needles_comp(X, objective_info.true_needles)
    pct_comp = pct_matched_comp(X, objective_info.true_needles, comp_radius)
    dup_comp = dup_fraction_comp(X, duplicate_radius_comp)
    dup_comp_cross = dup_fraction_comp_cross_line(X, duplicate_radius_comp, line_group_ids)
    return {
        "step": int(step),
        "best_y_so_far": best,
        "dist_to_needles": dist_ilr,
        "pct_matched": pct_ilr,
        "dup_fraction": dup_ilr,
        "dist_to_needles_ilr": dist_ilr,
        "pct_matched_ilr": pct_ilr,
        "dup_fraction_ilr": dup_ilr,
        "dist_to_needles_comp": dist_comp,
        "pct_matched_comp": pct_comp,
        "dup_fraction_comp": dup_comp,
        "dup_fraction_comp_all_points": dup_comp,
        "dup_fraction_comp_cross_line": dup_comp_cross,
        "runtime_s": float(runtime_s),
        "num_points": int(len(y_arr)),
    }

