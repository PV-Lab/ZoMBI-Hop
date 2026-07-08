from __future__ import annotations

import math
from typing import Any

import numpy as np

from .spaces import composition_to_ilr_np, project_simplex


SIMPLEX_DIAMETER_L2 = math.sqrt(2.0)
ILR_EPS = 1e-12


def audit_line_endpoints(endpoints: np.ndarray, atol: float = 1e-6) -> dict[str, Any]:
    arr = np.asarray(endpoints, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(f"line endpoints must have shape (2, n_components), got {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError("line endpoints must have at least two simplex components")

    finite = bool(np.isfinite(arr).all())
    row_sums = arr.sum(axis=1) if finite else np.asarray([math.nan, math.nan])
    min_by_endpoint = np.min(arr, axis=1) if finite else np.asarray([math.nan, math.nan])
    min_component = float(np.min(min_by_endpoint)) if finite else math.nan
    sum_deviation = float(np.max(np.abs(row_sums - 1.0))) if finite else math.nan
    nonnegative = bool(finite and np.min(arr) >= -atol)
    normalized = bool(finite and np.allclose(row_sums, 1.0, atol=atol))
    valid_simplex = bool(nonnegative and normalized)

    projected = project_simplex(arr) if finite else arr
    length_l2 = float(np.linalg.norm(projected[1] - projected[0])) if finite else math.nan
    try:
        endpoint_ilr = composition_to_ilr_np(projected, eps=ILR_EPS)
        length_ilr = float(np.linalg.norm(endpoint_ilr[1] - endpoint_ilr[0]))
        ilr_finite = bool(np.isfinite(length_ilr))
    except Exception:
        length_ilr = math.nan
        ilr_finite = False

    return {
        "line_endpoint_min": min_component,
        "line_endpoint_min_left": float(min_by_endpoint[0]),
        "line_endpoint_min_right": float(min_by_endpoint[1]),
        "line_endpoint_sum_deviation": sum_deviation,
        "line_endpoints_finite": finite,
        "line_endpoints_nonnegative": nonnegative,
        "line_endpoints_normalized": normalized,
        "line_endpoints_valid_simplex": valid_simplex,
        "line_length_l2_audit": length_l2,
        "line_length_ilr_audit": length_ilr,
        "line_length_ilr_finite": ilr_finite,
        "line_length_l2_coordinate_system": "raw_simplex_l2",
        "line_length_ilr_coordinate_system": f"helmert_ilr_l2_eps_{ILR_EPS:g}",
        "line_length_l2_simplex_diameter": SIMPLEX_DIAMETER_L2,
        "line_length_l2_within_simplex_diameter": bool(
            np.isfinite(length_l2) and length_l2 <= SIMPLEX_DIAMETER_L2 + atol
        ),
    }
