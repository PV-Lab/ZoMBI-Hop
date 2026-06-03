"""
run_zombi_test_v2.py
====================

Extension of run_zombi_test.py with three changes:

1. **Region-based distance metric.**
   Instead of measuring distance to a single known-maximum point, the
   evaluation uses the max/min regions produced by
   ``interactive_maxima_selector.py``.  A needle whose function value falls
   within ε of any max-region seed is considered distance-0 from that region.
   When no region file is supplied the script falls back to the V1 point-based
   metric automatically.

2. **Three MOBO objectives** (all minimised jointly):
     (a) avg_region_dist  — average distance from each max region to the
                            nearest needle across all benchmark runs
     (b) dup_frac         — fraction of sampled points that are near-duplicate
                            (wasted) evaluations
     (c) total_penalty_ball_volume — sum of Euclidean d-ball volumes
         ``Σ_i Vol(B_d(r_i))`` for each needle's penalty radius ``r_i`` (same
         metric as ``DataHandler`` penalty balls). ``penalty_max_radius`` remains
         a *tuned hyperparameter*, not an objective.

3. **Noisy objectives preserved.**
   The same four (input_noise, output_noise) combinations from V1 are swept
   for every benchmark objective.

Usage
-----
    # Single benchmark run (no MOBO):
    python scripts/run_zombi_test_v2.py

    # MOBO hyperparameter tuning (e.g. 4 configs per round, 4 parallel processes):
    python scripts/run_zombi_test_v2.py --mobo \\
        --regions max_min_regions.json \\
        --mobo-init 8 --mobo-iters 20 --batch 4 --workers 4

    # MOBO without a regions file (falls back to V1 point distance):
    python scripts/run_zombi_test_v2.py --mobo
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

# ── project path ──────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RF_CACHE_DIR = os.path.join(_REPO_ROOT, "test_rfs")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import everything reusable from V1
from scripts.run_zombi_test import (
    # Classes
    NaturalGradGPSimplex,
    NaturalGradZoMBIHop,
    # Functions
    build_csv_rf_objectives,
    make_zombi_objective,
    # Ackley
    ackley_equal,
    ackley_edge,
    ackley_vertex,
    multimodal_ackley,
    ACKLEY_CENTER_EQUAL,
    ACKLEY_CENTER_EDGE,
    ACKLEY_CENTER_VERTEX,
    MULTIMODAL_CENTERS,
    # CSV constants
    CSV_ELEMENT_TRIPLE,
    CSV_OBJECTIVES,
    CSV_PEROVSKITE_PATH,
    # MOBO helpers
    _NumpyEncoder,
    _DEFAULT_PARAM_BOUNDS,
    _DEFAULT_LOG_PARAMS,
    _DEFAULT_INT_PARAMS,
    _config_from_unit,
    _unit_from_config,
    # Misc
    RF_CACHE_DIR as _V1_RF_CACHE_DIR,
)
from src.utils.simplex import random_simplex

_D = 3

# MOBO / needle-count fallbacks (match region NEEDLE_PENALTY scale order of magnitude)
_NEEDLE_VOL_PENALTY = 1e6


def total_penalty_euclidean_ball_volume(radii: np.ndarray, d: int) -> float:
    """
    Sum of Lebesgue volumes of Euclidean d-balls with radii ``radii``.

    Uses the same distance geometry as ``DataHandler`` penalty regions
    (Euclidean norm in R^d). Volume of one ball: π^(d/2) r^d / Γ(d/2 + 1).
    """
    radii = np.asarray(radii, dtype=np.float64).ravel()
    if radii.size == 0:
        return 0.0
    from math import gamma, pi

    coeff = (pi ** (d / 2)) / gamma(d / 2 + 1)
    return float(np.sum(coeff * (radii**d)))


def _resolve_device(device: Optional[str]) -> str:
    """Default to CUDA when available; fall back to CPU."""
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available; using CPU.", stacklevel=2)
        return "cpu"
    return device


# =============================================================================
# Region loading and region-based distance
# =============================================================================

_TOP_L_MAX = 10   # hard cap on L


def _get_top_seeds(
    regions_data: Optional[Dict],
    obj_name: str,
    mode: str,
    L_max: int = _TOP_L_MAX,
) -> Tuple[List[np.ndarray], int]:
    """
    Return the top-L seeds for *obj_name* in *mode* ("max" or "min"), along
    with the actual L used.

    Seeds are sorted best-first (descending by value for max, ascending for
    min) and capped at *L_max*.  Returns ([], 1) when no region data is
    available so the caller always gets a valid L for adaptive activations.

    Returns
    -------
    coords : List[np.ndarray]   — up to L seed coordinates (each shape (3,))
    L      : int                — min(L_max, n_seeds), at least 1
    """
    if regions_data is None:
        return [], 1

    entry = regions_data.get("objectives", {}).get(obj_name)
    if entry is None:
        return [], 1

    seeds = entry.get(f"{mode}_seeds", [])
    vals  = entry.get(f"{mode}_seed_vals", [])
    if not seeds:
        return [], 1

    descending = (mode == "max")
    pairs = sorted(zip(vals, seeds), reverse=descending)
    top   = pairs[:L_max]

    coords = [np.array(coord, dtype=np.float64) for _, coord in top]
    return coords, len(coords)

def load_regions(path: str) -> Optional[Dict]:
    """
    Load a ``max_min_regions.json`` file produced by
    ``interactive_maxima_selector.py``.

    Returns the parsed dict, or None if the file doesn't exist / is invalid.

    The dict schema is::

        {
          "epsilon_frac": float,
          "objectives": {
            "<name>": {
              "epsilon_abs": float,
              "max_seeds": [[x0,x1,x2], ...],
              "max_seed_vals": [float, ...],
              "min_seeds": [...],
              "min_seed_vals": [...],
              "max_region_coords": [[x0,x1,x2], ...],   # may be []
              "min_region_coords": [...],
            },
            ...
          }
        }
    """
    if path is None or not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data
    except Exception as exc:
        warnings.warn(f"Could not load regions file '{path}': {exc}", stacklevel=2)
        return None


def _dist_one_seed_to_needles(
    needles_np: np.ndarray,
    seed_coord: np.ndarray,
    seed_val: float,
    region_coords_for_seed: List,
    epsilon_abs: float,
    needle_fn_vals: Optional[np.ndarray],
) -> float:
    """
    Distance from the nearest needle to ONE seed's region.

    Priority order:
    1. Level-set: any needle with |f(needle) - seed_val| < epsilon_abs → 0.
    2. Spatial KD-tree on this seed's region blob (Voronoi-attributed).
    3. Euclidean distance to the seed coordinate itself.
    """
    if needle_fn_vals is not None:
        if np.any(np.abs(needle_fn_vals - seed_val) < epsilon_abs):
            return 0.0
    if region_coords_for_seed:
        rc_arr = np.asarray(region_coords_for_seed, dtype=np.float64)
        return float(cKDTree(rc_arr).query(needles_np)[0].min())
    return float(np.linalg.norm(needles_np - seed_coord[None, :], axis=1).min())


def compute_region_avg_dist(
    needles_np: np.ndarray,        # (n_needles, d)
    obj_name: str,
    regions_data: Optional[Dict],
    fallback_known_extrema: List[np.ndarray],
    obj_fn: Optional[Callable] = None,
    mode: str = "max",
    epsilon_frac: float = 0.2,
) -> float:
    """
    Average distance from each labeled seed's region to its nearest needle,
    over all L seeds.

    For each seed i:
      1. Level-set: if any needle has |f(needle) - val_i| < epsilon_abs → dist_i = 0.
      2. Spatial: min distance from any needle to seed i's Voronoi-attributed
         region blob (subset of ``{mode}_region_coords`` nearest to seed i).
      3. Euclidean fallback: min distance from any needle to the seed coord itself.

    The return value is ``mean(dist_i for i in 1…L)``, so every seed must be
    covered for the score to be low.  This mirrors the V1 per-extremum
    averaging below and fixes the previous bug where all blobs were pooled
    into one KD-tree (a single needle near any region gave dist ≈ 0).

    Falls back to V1 greedy point distance when no region data is available.
    """
    if needles_np.shape[0] == 0:
        return float("inf")

    # ── Region-based distance: per-seed averaging ──────────────────────────────
    if regions_data is not None:
        obj_regions = regions_data.get("objectives", {}).get(obj_name)
        seed_key = f"{mode}_seeds"
        if obj_regions is not None and obj_regions.get(seed_key):
            val_range   = float(obj_regions.get("val_range", 1.0))
            epsilon_abs = epsilon_frac * val_range

            seed_coords_raw = obj_regions.get(seed_key, [])
            seed_vals       = [float(v) for v in obj_regions.get(f"{mode}_seed_vals", [])]
            all_rc          = obj_regions.get(f"{mode}_region_coords", [])

            seed_coords = [np.asarray(sc, dtype=np.float64) for sc in seed_coords_raw]
            L = len(seed_coords)
            if L == 0:
                pass  # fall through to V1
            else:
                # Voronoi-split the combined region blob back to each seed.
                # Each region point is assigned to its spatially nearest seed.
                per_seed_rc: List[List] = [[] for _ in range(L)]
                if all_rc:
                    if L == 1:
                        per_seed_rc[0] = all_rc
                    else:
                        seed_arr = np.array(seed_coords, dtype=np.float64)
                        seed_tree = cKDTree(seed_arr)
                        rc_arr = np.asarray(all_rc, dtype=np.float64)
                        _, assigns = seed_tree.query(rc_arr)
                        for pt, s_idx in zip(all_rc, assigns.tolist()):
                            per_seed_rc[s_idx].append(pt)

                # Evaluate fn at all needles once (for level-set membership).
                needle_fn_vals = None
                if obj_fn is not None:
                    needle_fn_vals = np.array(
                        [obj_fn(needles_np[k]) for k in range(len(needles_np))],
                        dtype=np.float64,
                    )

                per_seed_dists = [
                    _dist_one_seed_to_needles(
                        needles_np, sc, sv, per_seed_rc[i],
                        epsilon_abs, needle_fn_vals,
                    )
                    for i, (sc, sv) in enumerate(zip(seed_coords, seed_vals))
                ]
                return float(np.mean(per_seed_dists))

    # ── Fallback: greedy point distance (V1 behaviour) ────────────────────────
    if not fallback_known_extrema:
        return float("inf")
    total = 0.0
    for km in fallback_known_extrema:
        km_arr = np.asarray(km)
        dists  = np.linalg.norm(needles_np - km_arr[None, :], axis=1)
        total += float(dists.min())
    return total / len(fallback_known_extrema)


# =============================================================================
# Single-objective ZoMBI-Hop runner  (V2: region-aware distance)
# =============================================================================

def run_zombi_on_objective_v2(
    fn: Callable,
    known_extrema: List[np.ndarray],
    name: str,
    obj_name_for_region: str,
    regions_data: Optional[Dict],
    *,
    mode: str = "max",
    L: int = 1,
    epsilon_frac: float = 0.2,
    num_init_data: int = 4,
    max_zooms: int = 3,
    max_iterations: int = 10,
    n_restarts: int = 30,
    raw_samples: int = 500,
    top_m_points: Optional[int] = None,
    penalization_threshold: float = 1e-3,
    penalty_max_radius: float = 0.3,
    convergence_pi_threshold: float = 0.01,
    n_consecutive_converged: int = 2,
    max_gp_points: int = 3000,
    ucb_beta: float = 0.1,
    repulsion_lambda: Optional[float] = None,
    nat_grad_step: float = 0.02,
    nat_grad_max_steps: int = 50,
    num_points_per_line: int = 100,
    input_noise: float = 0.0,
    output_noise: float = 0.0,
    num_lines: int = 30,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    seed: int = 0,
) -> Dict:
    """
    Run NaturalGradZoMBIHop on a single objective and return a result dict
    with region-based distance (V2) and V1 point distance for reference.

    Parameters
    ----------
    fn : Callable
        Original (un-negated) scalar objective.  ZoMBI maximises it when
        ``mode="max"`` and maximises ``-fn`` when ``mode="min"``.
    known_extrema : list of np.ndarray
        Top-L seed coordinates for this mode (used as fallback targets).
    mode : "max" or "min"
        Which extremum to chase.
    L : int
        Number of target optima for this run.  Drives:
          - ``max_activations = ceil(1.2 * L)``
          - Needle-count penalty: if ``n_needles < L`` → ``region_dist = 10``
    epsilon_frac : float
        Level-set width as a fraction of value range (default 0.2).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Adaptive activations: ceil(1.2 * L), capped implicitly by L ≤ 10
    max_activations = math.ceil(1.2 * L)

    device_t = torch.device(device)
    lo = torch.zeros(_D, device=device_t, dtype=dtype)
    hi = torch.ones(_D,  device=device_t, dtype=dtype)
    bounds = torch.stack([lo, hi])

    # ── Noise wrapper (applied to the direction fn, not the negation) ─────────
    def _noisy_fn(x: np.ndarray) -> float:
        x_eval = x.copy()
        if input_noise > 0.0:
            perturbed = x_eval + np.random.normal(0.0, input_noise, size=x_eval.shape)
            perturbed = np.clip(perturbed, 0.0, None)
            s = perturbed.sum()
            x_eval = perturbed / s if s > 1e-12 else x_eval
        y = fn(x_eval)
        if output_noise > 0.0:
            y += np.random.normal(0.0, output_noise)
        return float(y)

    noisy_fn = _noisy_fn if (input_noise > 0.0 or output_noise > 0.0) else fn

    # ZoMBI always maximises; negate for min mode
    if mode == "min":
        zombi_fn = lambda x: -noisy_fn(x)   # noqa: E731
    else:
        zombi_fn = noisy_fn

    # ── Initial data ──────────────────────────────────────────────────────────
    X_init = random_simplex(num_init_data, lo, hi, device=device, torch_dtype=dtype)
    Y_init = torch.tensor(
        [zombi_fn(X_init[k].cpu().numpy()) for k in range(num_init_data)],
        device=device_t, dtype=dtype,
    ).unsqueeze(1)

    objective = make_zombi_objective(zombi_fn, num_points_per_line, num_lines, device_t, dtype)

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"Running ZoMBI-Hop (V2) on: {name}  [mode={mode}  L={L}"
              f"  max_activations={max_activations}]")
        print(f"  Known {mode} extrema: {[km.tolist() for km in known_extrema]}")
        print(f"{'─' * 60}")

    optimizer = NaturalGradZoMBIHop(
        objective=objective,
        bounds=bounds,
        X_init_actual=X_init,
        X_init_expected=X_init.clone(),
        Y_init=Y_init,
        max_zooms=max_zooms,
        max_iterations=max_iterations,
        top_m_points=top_m_points,
        n_restarts=n_restarts,
        raw=raw_samples,
        penalization_threshold=penalization_threshold,
        penalty_max_radius=penalty_max_radius,
        convergence_pi_threshold=convergence_pi_threshold,
        n_consecutive_converged=n_consecutive_converged,
        max_gp_points=max_gp_points,
        acquisition_type="ucb",
        ucb_beta=ucb_beta,
        repulsion_lambda=repulsion_lambda,
        device=device,
        dtype=dtype,
        checkpoint_dir=None,
        max_snapshots=0,
        verbose=verbose,
        nat_grad_step=nat_grad_step,
        nat_grad_max_steps=nat_grad_max_steps,
    )

    needles_results, needles, needle_vals, _X_all, _Y_all = optimizer.run(
        max_activations=max_activations
    )

    # ── Duplicate analysis ────────────────────────────────────────────────────
    X_all_np = _X_all.cpu().numpy()
    n_total = X_all_np.shape[0]
    n_redundant = 0
    n_in_any_dup = 0
    if n_total > 1:
        from scipy.spatial.distance import pdist, squareform
        dist_mat = squareform(pdist(X_all_np, metric="chebyshev"))
        np.fill_diagonal(dist_mat, np.inf)
        near = dist_mat < 1e-4
        n_in_any_dup = int(np.any(near, axis=1).sum())
        assigned = np.zeros(n_total, dtype=bool)
        for k in range(n_total):
            if not assigned[k]:
                nb = np.where(near[k])[0]
                nb = nb[nb > k]
                assigned[nb] = True
                n_redundant += len(nb)

    n_needles = needles.shape[0] if torch.is_tensor(needles) else 0
    needles_np = needles.cpu().numpy() if n_needles > 0 else np.empty((0, _D))

    # ── Needle-count penalty ──────────────────────────────────────────────────
    # If ZoMBI found fewer needles than the number of target optima, penalise
    # hard regardless of where those needles are.
    NEEDLE_PENALTY = 10.0
    if n_needles < L:
        region_dist = NEEDLE_PENALTY
        total_penalty_ball_volume = float(_NEEDLE_VOL_PENALTY)
        if verbose:
            print(f"[{name}] PENALTY: only {n_needles} needle(s) found, need {L}")
    else:
        # ── Region-based distance (primary V2 metric) ─────────────────────────
        region_dist = compute_region_avg_dist(
            needles_np,
            obj_name_for_region,
            regions_data,
            known_extrema,
            obj_fn=fn,          # original un-negated fn for level-set membership
            mode=mode,
            epsilon_frac=epsilon_frac,
        )
        _nr, radii_t = optimizer.data_handler.get_needles_and_penalty_radii()
        total_penalty_ball_volume = total_penalty_euclidean_ball_volume(
            radii_t.cpu().numpy(), _D
        )

    # V1 fallback distance kept for reference (always uses original fn)
    if n_needles == 0 or not known_extrema:
        v1_dist = float("inf")
    else:
        v1_dist = 0.0
        for km in known_extrema:
            km_arr = np.asarray(km)
            dists  = np.linalg.norm(needles_np - km_arr[None, :], axis=1)
            v1_dist += float(dists.min())
        v1_dist /= len(known_extrema)

    if verbose:
        print(f"[{name}] {n_total} total pts  |  {n_redundant} redundant")
        print(f"[{name}] {n_needles}/{L} needle(s)  |  region_dist={region_dist:.5f}"
              f"  v1_dist={v1_dist:.5f}")
        print(f"[{name}] total_penalty_ball_volume (Σ Vol(B_d(r_i))) = {total_penalty_ball_volume:.6e}")

    return {
        "name":                         name,
        "obj_name_for_region":          obj_name_for_region,
        "mode":                         mode,
        "L":                            L,
        "max_activations":              max_activations,
        "needles":                      needles_np,
        "needle_vals":                  (
            needle_vals.cpu().numpy()
            if torch.is_tensor(needle_vals) and needle_vals.numel() > 0
            else np.array([])
        ),
        "avg_region_dist":              region_dist,
        "avg_distance_to_known_extrema": v1_dist,
        "known_extrema":                [np.asarray(km) for km in known_extrema],
        "needles_results":              needles_results,
        "n_total_points":               n_total,
        "n_in_any_dup":                 n_in_any_dup,
        "n_redundant":                  n_redundant,
        "input_noise":                  input_noise,
        "output_noise":                 output_noise,
        "total_penalty_ball_volume":    total_penalty_ball_volume,
    }


# =============================================================================
# Top-level benchmark runner  (V2)
# =============================================================================

def run_zombi_test_v2(
    *,
    regions_path: Optional[str] = None,
    epsilon_frac: float = 0.2,
    # ── ZoMBI hyperparams ──────────────────────────────────────────────────────
    max_zooms: int = 3,
    max_iterations: int = 10,
    n_restarts: int = 30,
    raw_samples: int = 500,
    top_m_points: Optional[int] = None,
    penalization_threshold: float = 1e-3,
    penalty_max_radius: float = 0.3,
    convergence_pi_threshold: float = 0.01,
    n_consecutive_converged: int = 2,
    max_gp_points: int = 3000,
    # ── GP / acquisition hyperparams ──────────────────────────────────────────
    ucb_beta: float = 0.1,
    repulsion_lambda: Optional[float] = None,
    nat_grad_step: float = 0.02,
    nat_grad_max_steps: int = 50,
    # ── LineBO hyperparams ─────────────────────────────────────────────────────
    num_points_per_line: int = 100,
    num_lines: int = 30,
    # ── Run settings ──────────────────────────────────────────────────────────
    # max_activations is now computed per-run as ceil(1.2 * L)
    num_init_data: int = 4,
    rf_global_samples: int = 10_000_000,
    rf_cache_dir: Optional[str] = RF_CACHE_DIR,
    csv_path: str = CSV_PEROVSKITE_PATH,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    show_plot: bool = False,
) -> List[Dict]:
    """
    Full V2 benchmark pipeline.

    Each base objective is run 8 times:
      2 modes (max + min)  ×  4 noise combos  =  8 runs per objective

    - ``max_activations`` is computed adaptively per run as
      ``ceil(1.2 * L)`` where ``L = min(10, n_labeled_seeds)``.
    - Region distance uses ``epsilon_frac=0.2`` (overrides stored value).
    - Runs with ``n_needles < L`` receive ``avg_region_dist = 10`` (penalty).
    """
    device = _resolve_device(device)
    regions_data = load_regions(regions_path)
    if regions_data is not None:
        print(f"  Loaded regions from: {regions_path}  "
              f"({len(regions_data.get('objectives', {}))} objectives)")
    else:
        print("  No regions file — using V1 point-based distance (L=1 fallback).")

    # Shared ZoMBI kwargs — max_activations is intentionally absent; it is
    # computed per-run from L and injected in the loop below.
    _zombi_kw = dict(
        num_init_data=num_init_data,
        max_zooms=max_zooms,
        max_iterations=max_iterations,
        n_restarts=n_restarts,
        raw_samples=raw_samples,
        top_m_points=top_m_points,
        penalization_threshold=penalization_threshold,
        penalty_max_radius=penalty_max_radius,
        convergence_pi_threshold=convergence_pi_threshold,
        n_consecutive_converged=n_consecutive_converged,
        max_gp_points=max_gp_points,
        ucb_beta=ucb_beta,
        repulsion_lambda=repulsion_lambda,
        nat_grad_step=nat_grad_step,
        nat_grad_max_steps=nat_grad_max_steps,
        num_points_per_line=num_points_per_line,
        num_lines=num_lines,
        device=device,
        dtype=dtype,
        verbose=verbose,
        epsilon_frac=epsilon_frac,
    )

    # ── Step 1: CSV RF objectives ──────────────────────────────────────────────
    csv_rf_objectives: Dict = {}
    if os.path.isfile(csv_path):
        try:
            csv_rf_objectives = build_csv_rf_objectives(
                csv_path=csv_path,
                objectives=CSV_OBJECTIVES,
                rf_global_samples=rf_global_samples,
                cache_dir=rf_cache_dir,
            )
        except Exception as exc:
            warnings.warn(f"CSV RF build failed: {exc}", stacklevel=2)
    else:
        warnings.warn(f"CSV not found: {csv_path}", stacklevel=2)

    # ── Step 2: Ackley objectives ──────────────────────────────────────────────
    ackley_fns: List[Tuple[str, Callable, List[np.ndarray], Optional[np.ndarray]]] = [
        ("Ackley-Centroid",    ackley_equal,      [ACKLEY_CENTER_EQUAL],  None),
        ("Ackley-Edge",        ackley_edge,        [ACKLEY_CENTER_EDGE],   None),
        ("Ackley-Vertex",      ackley_vertex,      [ACKLEY_CENTER_VERTEX], None),
        ("Ackley-Multi-modal", multimodal_ackley,  MULTIMODAL_CENTERS,     None),
    ]

    # ── Step 3: Noise combos ───────────────────────────────────────────────────
    noise_combos: List[Tuple[float, float]] = [
        (0.01,  0.01),
        (0.01,  0.001),
        (0.001, 0.01),
        (0.001, 0.001),
    ]

    # Flat list of (base_name, region_key, fn, extrema, mode, L)
    # Two entries per base objective: one for max, one for min.
    objectives_to_run: List[Tuple[str, str, Callable, List[np.ndarray], str, int]] = []

    for obj_col, data in csv_rf_objectives.items():
        rf = data["rf"]
        def _csv_rf_fn(x: np.ndarray, _rf=rf) -> float:
            return float(_rf.predict(x.reshape(1, -1))[0])
        base_name  = f"CSV-RF-{obj_col} ({'/'.join(CSV_ELEMENT_TRIPLE)})"
        region_key = f"RF-{obj_col}"

        for mode in ("max", "min"):
            top_seeds, L = _get_top_seeds(regions_data, region_key, mode)
            if not top_seeds:
                # No labeled seeds → fall back to RF global extremum
                if mode == "max":
                    top_seeds = [data["global_max_x"]]
                else:
                    gmin = data.get("global_min_x")
                    top_seeds = [gmin] if gmin is not None else []
                L = max(len(top_seeds), 1)
            objectives_to_run.append((base_name, region_key, _csv_rf_fn, top_seeds, mode, L))

    for aname, afn, analytic_max, _ in ackley_fns:
        region_key = aname

        for mode in ("max", "min"):
            top_seeds, L = _get_top_seeds(regions_data, region_key, mode)
            if not top_seeds:
                # No labeled seeds → use analytic maxima (only available for max)
                if mode == "max":
                    top_seeds = [np.asarray(km) for km in analytic_max]
                else:
                    top_seeds = []   # Ackley minima not analytically provided
                L = max(len(top_seeds), 1)
            objectives_to_run.append((aname, region_key, afn, top_seeds, mode, L))

    all_results: List[Dict] = []
    total_runs = len(objectives_to_run) * len(noise_combos)
    run_idx = 0

    for base_name, region_key, fn, known_extrema, mode, L in objectives_to_run:
        for inp_noise, out_noise in noise_combos:
            run_idx += 1
            noise_tag = f"in={inp_noise:.3f}/out={out_noise:.3f}"
            full_name = f"{base_name} [{mode}] [{noise_tag}]"
            if verbose:
                print(f"\n  [{run_idx}/{total_runs}] {full_name}  "
                      f"L={L}  activations={math.ceil(1.2*L)}")
            res = run_zombi_on_objective_v2(
                fn=fn,
                known_extrema=known_extrema,
                name=full_name,
                obj_name_for_region=region_key,
                regions_data=regions_data,
                mode=mode,
                L=L,
                input_noise=inp_noise,
                output_noise=out_noise,
                **_zombi_kw,
            )
            all_results.append(res)

    return all_results


# =============================================================================
# MOBO evaluation helpers  (V2: 3 objectives)
# =============================================================================

def _evaluate_config_v2(
    config: Dict,
    *,
    fixed_kw: Dict,
) -> Tuple[float, float, float]:
    """
    Run the full V2 benchmark for *config* merged into *fixed_kw*.

    Returns
    -------
    (avg_region_dist, dup_frac, mean_total_penalty_ball_volume)
    All three are to be minimised. If no sub-run has finite ``avg_region_dist``,
    the first scalar is ``100.0`` (total-evaluation failure fallback).
    """
    merged = {**fixed_kw, **config}
    results = run_zombi_test_v2(**merged)

    dists = [
        r["avg_region_dist"]
        for r in results
        if np.isfinite(r["avg_region_dist"])
    ]
    avg_dist = float(np.mean(dists)) if dists else 100.0

    dup_fracs = [
        r["n_redundant"] / max(r["n_total_points"], 1)
        for r in results
    ]
    avg_dup = float(np.mean(dup_fracs)) if dup_fracs else 1.0

    vols = [
        float(v)
        for r in results
        for v in [r.get("total_penalty_ball_volume")]
        if v is not None and np.isfinite(v)
    ]
    avg_vol = float(np.mean(vols)) if vols else float(_NEEDLE_VOL_PENALTY)

    return avg_dist, avg_dup, avg_vol


def _run_batch_sequential_v2(
    configs: List[Dict],
    x_unit_batch: List[np.ndarray],
    fixed_kw: Dict,
    label: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
    """Evaluate each hyperparameter config one after another (GPU ZoMBI; no process pool)."""
    dev = fixed_kw.get("device", "cpu")
    print(f"\n  Evaluating {len(configs)} config(s) sequentially on {dev} …")

    new_X: List[torch.Tensor] = []
    new_Y: List[torch.Tensor] = []
    for i, cfg in enumerate(configs):
        try:
            dist, dup, vol = _evaluate_config_v2(cfg, fixed_kw=fixed_kw)
        except Exception as exc:
            warnings.warn(f"  [{label} {i+1}/{len(configs)}] failed: {exc}", stacklevel=2)
            dist, dup, vol = (1.0, 1.0, float(_NEEDLE_VOL_PENALTY))
        print(
            f"  [{label} {i+1}/{len(configs)}] "
            f"region_dist={dist:.5f}  dup_frac={dup:.4f}  "
            f"total_penalty_vol={vol:.6e}  config={cfg}",
        )
        new_X.append(torch.tensor(x_unit_batch[i], dtype=torch.float64))
        new_Y.append(torch.tensor([dist, dup, vol], dtype=torch.float64))
        if dev == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return new_X, new_Y, configs


def _mobo_worker_v2(task: Tuple[Dict, Dict]) -> Tuple[float, float, float]:
    """Picklable worker: evaluate one MOBO hyperparameter config (full V2 benchmark)."""
    config, fixed_kw = task
    return _evaluate_config_v2(config, fixed_kw=fixed_kw)


def _run_batch_mobo_v2(
    configs: List[Dict],
    x_unit_batch: List[np.ndarray],
    fixed_kw: Dict,
    label: str,
    mobo_workers: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
    """
    Evaluate a batch of hyperparameter configs.

    * ``mobo_workers == 1``: sequential (single process; simplest GPU use).
    * ``mobo_workers > 1``: ``ProcessPoolExecutor`` with up to ``mobo_workers``
      processes (default 4). Each run loads its own ZoMBI stack; use ``--device cpu``
      if multiple CUDA processes exhaust GPU memory.
    """
    mobo_workers = max(1, int(mobo_workers))
    ncfg = len(configs)
    if mobo_workers == 1 or ncfg == 1:
        return _run_batch_sequential_v2(configs, x_unit_batch, fixed_kw, label)

    import concurrent.futures

    workers = min(mobo_workers, ncfg)
    tasks = [(cfg, fixed_kw) for cfg in configs]
    print(
        f"\n  Evaluating {ncfg} config(s) with ProcessPoolExecutor "
        f"(max_workers={workers}) …",
    )

    results_xyz: List[Optional[Tuple[float, float, float]]] = [None] * len(tasks)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_mobo_worker_v2, t): i for i, t in enumerate(tasks)}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                results_xyz[idx] = fut.result()
            except Exception as exc:
                warnings.warn(f"  [{label} worker {idx}] failed: {exc}", stacklevel=2)
                results_xyz[idx] = (1.0, 1.0, float(_NEEDLE_VOL_PENALTY))

    new_X: List[torch.Tensor] = []
    new_Y: List[torch.Tensor] = []
    for i, triple in enumerate(results_xyz):
        dist, dup, vol = triple  # type: ignore[misc]
        print(
            f"  [{label} {i+1}/{len(tasks)}] "
            f"region_dist={dist:.5f}  dup_frac={dup:.4f}  "
            f"total_penalty_vol={vol:.6e}  config={configs[i]}",
        )
        new_X.append(torch.tensor(x_unit_batch[i], dtype=torch.float64))
        new_Y.append(torch.tensor([dist, dup, vol], dtype=torch.float64))

    return new_X, new_Y, configs


# =============================================================================
# MOBO tuner  (V2: 3-objective qLogNEHVI)
# =============================================================================

def mobo_tune_zombi_v2(
    *,
    regions_path: Optional[str] = None,
    n_initial: int = 8,
    n_mobo_iterations: int = 20,
    n_parallel: int = 4,
    mobo_workers: int = 4,
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    log_scale_params: Optional[set] = None,
    integer_params: Optional[set] = None,
    # ── Fixed run settings (not tuned) ────────────────────────────────────────
    csv_path: str = CSV_PEROVSKITE_PATH,
    # max_activations is absent: computed per-run as ceil(1.2 * L)
    num_init_data: int = 4,
    # max_zooms, max_iterations, n_restarts, raw_samples, penalty_max_radius
    # are tunable — values here serve as fallback defaults only.
    max_zooms: int = 3,
    max_iterations: int = 10,
    n_restarts: int = 30,
    raw_samples: int = 500,
    penalty_max_radius: float = 0.3,
    top_m_points: Optional[int] = None,
    n_consecutive_converged: int = 2,
    max_gp_points: int = 3000,
    repulsion_lambda: Optional[float] = None,
    rf_global_samples: int = 10_000_000,
    rf_cache_dir: Optional[str] = RF_CACHE_DIR,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float64,
    verbose_zombi: bool = False,
    results_json: str = "hyperparam_results_v2.json",
    seed: int = 0,
) -> Dict:
    """
    3-objective MOBO tuning of ZoMBI-Hop hyperparameters.

    Minimises simultaneously:
      1. avg_region_dist — region-based distance (0 if needle inside region)
      2. dup_frac — fraction of wasted/duplicate evaluations
      3. mean total_penalty_ball_volume — mean over benchmark runs of
         Σ_i Vol(B_d(r_i)) for needle penalty radii (Euclidean d-balls)

    ``penalty_max_radius`` remains a tuned *hyperparameter* in ``config``,
    not an objective.

    Uses ``qLogNoisyExpectedHypervolumeImprovement`` with one GP per objective.
    Each MOBO round proposes ``n_parallel`` candidate configs. They are
    evaluated with up to ``mobo_workers`` parallel processes (default 4), or
    sequentially if ``mobo_workers=1``.

    Parameters
    ----------
    regions_path : str or None
        Path to ``max_min_regions.json`` from ``interactive_maxima_selector.py``.
        If None, distance objective falls back to V1 point-based metric.
    n_parallel : int
        Batch size ``q`` for acquisition optimization (configs per round).
    mobo_workers : int
        Max parallel processes for evaluating those configs. Use ``1`` to force
        sequential execution (e.g. single-GPU memory limits).

    Returns
    -------
    dict with keys:
        X, Y (n×3), configs, pareto_mask, pareto_configs, pareto_Y, param_names
    """
    device = _resolve_device(device)
    from botorch.models import SingleTaskGP
    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement,
    )
    from botorch.optim import optimize_acqf
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from botorch.utils.sampling import draw_sobol_samples
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    raw_bounds  = param_bounds    or dict(_DEFAULT_PARAM_BOUNDS)
    log_params  = log_scale_params or set(_DEFAULT_LOG_PARAMS)
    int_params  = integer_params   or set(_DEFAULT_INT_PARAMS)
    param_names = sorted(raw_bounds.keys())
    d           = len(param_names)

    n_initial = int(np.ceil(n_initial / n_parallel)) * n_parallel

    fixed_kw: Dict = dict(
        regions_path=regions_path,
        csv_path=csv_path,
        epsilon_frac=0.2,
        # max_activations is intentionally absent — computed per-run from L
        num_init_data=num_init_data,
        max_zooms=max_zooms,
        max_iterations=max_iterations,
        n_restarts=n_restarts,
        raw_samples=raw_samples,
        penalty_max_radius=penalty_max_radius,
        top_m_points=top_m_points,
        n_consecutive_converged=n_consecutive_converged,
        max_gp_points=max_gp_points,
        repulsion_lambda=repulsion_lambda,
        rf_global_samples=rf_global_samples,
        rf_cache_dir=rf_cache_dir,
        device=device,
        dtype=dtype,
        verbose=verbose_zombi,
        show_plot=False,
    )

    # ── Incremental JSON writer ────────────────────────────────────────────────
    def _flush_json(
        evaluations: List[Dict],
        configs: List[Dict],
        Y_list: List[torch.Tensor],
        path: str,
        *,
        pareto_mask: Optional[np.ndarray] = None,
    ) -> None:
        records = []
        for k, (cfg, y_t) in enumerate(zip(configs, Y_list)):
            avg_dist = float(y_t[0])
            dup_frac = float(y_t[1])
            vol      = float(y_t[2])
            records.append({
                "eval_idx":                    k,
                "phase":                       evaluations[k]["phase"],
                "config":                      cfg,
                "avg_region_dist":             avg_dist,
                "dup_frac":                    dup_frac,
                "total_penalty_ball_volume":   vol,
                "score": float(
                    np.sqrt(
                        avg_dist ** 2
                        + dup_frac ** 2
                        + np.log1p(max(vol, 0.0)) ** 2,
                    )
                ),
            })
        best_idx   = int(np.argmin([r["score"] for r in records]))
        best_so_far = {**records[best_idx]}
        out: Dict = {"evaluations": records, "best_so_far": best_so_far}
        if pareto_mask is not None:
            out["pareto_front"] = [r for r, m in zip(records, pareto_mask) if m]
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2, cls=_NumpyEncoder)

    all_phases:  List[Dict]          = []
    all_X:       List[torch.Tensor]  = []
    all_Y:       List[torch.Tensor]  = []
    all_configs: List[Dict]          = []

    device_t = torch.device(device)

    # Unit-cube BoTorch bounds [0,1]^d (keep MOBO on the requested device)
    bo_bounds = torch.zeros(2, d, dtype=torch.float64, device=device_t)
    bo_bounds[1] = 1.0

    # Reference point (maximization space = -Y): pessimistic vs typical ranges.
    # Third axis: -total_penalty_ball_volume can be very negative for large volumes.
    ref_point = torch.tensor([-100.0, -100.0, -1e12], dtype=torch.float64, device=device_t)

    n_total_evals = n_initial + n_mobo_iterations * n_parallel
    print("=" * 70)
    print("MOBO V2 TUNING  (3 objectives: region_dist, dup_frac, total_penalty_ball_volume)")
    print(f"  {n_initial} Sobol init + {n_mobo_iterations} rounds × {n_parallel} batch"
          f" = {n_total_evals} total evals (mobo_workers={mobo_workers})")
    print(f"  Tunable params ({d}): {param_names}")
    if regions_path:
        print(f"  Regions file: {regions_path}")
    else:
        print("  Regions file: None  (V1 point-distance fallback)")
    print(f"  Device: {device}")
    print("=" * 70)

    torch.manual_seed(seed)

    X_sobol = draw_sobol_samples(
        bounds=bo_bounds, n=n_initial, q=1, seed=seed,
    ).squeeze(1)

    # ── Resume from existing results_json ─────────────────────────────────────
    n_sobol_already_done     = 0
    n_mobo_rounds_already_done = 0
    if os.path.isfile(results_json):
        try:
            with open(results_json) as fh:
                saved = json.load(fh)
            for rec in saved.get("evaluations", []):
                cfg = rec["config"]
                x_unit = np.array([
                    _unit_from_config(cfg[name], name, raw_bounds, log_params)
                    for name in param_names
                ], dtype=np.float64)
                all_X.append(torch.tensor(x_unit, dtype=torch.float64, device=device_t))
                third = rec.get("total_penalty_ball_volume")
                if third is None:
                    third = rec.get("penalty_max_radius", 0.3)
                all_Y.append(torch.tensor([
                    rec["avg_region_dist"],
                    rec["dup_frac"],
                    float(third),
                ], dtype=torch.float64, device=device_t))
                all_configs.append(cfg)
                all_phases.append({"phase": rec["phase"]})
                if rec["phase"] == "sobol_init":
                    n_sobol_already_done += 1
                elif rec["phase"].startswith("mobo_round_"):
                    rnd = int(rec["phase"].split("_")[-1])
                    n_mobo_rounds_already_done = max(n_mobo_rounds_already_done, rnd)
            print(f"  Resumed {len(all_configs)} evals from {results_json} "
                  f"({n_sobol_already_done} Sobol, "
                  f"{n_mobo_rounds_already_done} MOBO rounds done).")
        except Exception as exc:
            print(f"  [warn] Could not load {results_json}: {exc}. Starting fresh.")

    # ── Phase 1: Sobol init ────────────────────────────────────────────────────
    for batch_start in range(0, n_initial, n_parallel):
        if batch_start + n_parallel <= n_sobol_already_done:
            print(f"  [resume] Skipping Sobol batch "
                  f"{batch_start+1}-{batch_start+n_parallel}.")
            continue
        batch_idx  = X_sobol[batch_start: batch_start + n_parallel]
        batch_cfgs = [
            _config_from_unit(batch_idx[j].detach().cpu().numpy(), param_names,
                              raw_bounds, log_params, int_params)
            for j in range(len(batch_idx))
        ]
        label = f"Init {batch_start+1}-{batch_start+len(batch_idx)}/{n_initial}"
        print(f"\n{'─'*60}\n{label}")
        new_X, new_Y, new_cfgs = _run_batch_mobo_v2(
            batch_cfgs,
            [batch_idx[j].detach().cpu().numpy() for j in range(len(batch_idx))],
            fixed_kw,
            label,
            mobo_workers,
        )
        all_X.extend(new_X)
        all_Y.extend(new_Y)
        all_configs.extend(new_cfgs)
        all_phases.extend([{"phase": "sobol_init"}] * len(new_cfgs))
        _flush_json(all_phases, all_configs, all_Y, results_json)

    # ── Phase 2: MOBO loop ─────────────────────────────────────────────────────
    for it in range(n_mobo_iterations):
        if it + 1 <= n_mobo_rounds_already_done:
            print(f"  [resume] Skipping MOBO round {it+1}.")
            continue
        print(f"\n{'─'*60}\nMOBO round {it+1}/{n_mobo_iterations}\n{'─'*60}")

        train_X = torch.stack(all_X).to(device_t)        # (n, d)
        train_Y = -torch.stack(all_Y).to(device_t)       # (n, 3) — negated for maximisation

        models = []
        for obj_idx in range(3):
            gp  = SingleTaskGP(train_X, train_Y[:, obj_idx: obj_idx + 1])
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            models.append(gp)
        model = ModelListGP(*models)

        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X,
            prune_baseline=True,
        )
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bo_bounds,
            q=n_parallel,
            num_restarts=16,
            raw_samples=256,
        )  # (n_parallel, d)

        batch_cfgs = [
            _config_from_unit(candidates[j].detach().cpu().numpy(), param_names,
                              raw_bounds, log_params, int_params)
            for j in range(n_parallel)
        ]
        label = f"Round {it+1}"
        new_X, new_Y, new_cfgs = _run_batch_mobo_v2(
            batch_cfgs,
            [candidates[j].detach().cpu().numpy() for j in range(n_parallel)],
            fixed_kw,
            label,
            mobo_workers,
        )
        all_X.extend(new_X)
        all_Y.extend(new_Y)
        all_configs.extend(new_cfgs)
        all_phases.extend([{"phase": f"mobo_round_{it+1}"}] * len(new_cfgs))
        _flush_json(all_phases, all_configs, all_Y, results_json)

    # ── Assemble and report ────────────────────────────────────────────────────
    Y_np = torch.stack(all_Y).numpy()   # (n, 3)
    X_np = torch.stack(all_X).numpy()
    pareto_mask = is_non_dominated(-torch.stack(all_Y)).numpy()
    _flush_json(all_phases, all_configs, all_Y, results_json, pareto_mask=pareto_mask)
    print(f"\nResults saved → {os.path.abspath(results_json)}")

    print("\n" + "=" * 70)
    print("V2 MOBO COMPLETE — Pareto-optimal configurations:")
    print("=" * 70)
    header = f"  {'region_dist':>12}  {'dup_frac':>10}  {'ΣVol':>12}  config"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for idx in np.where(pareto_mask)[0]:
        print(f"  {Y_np[idx,0]:>12.5f}  {Y_np[idx,1]:>10.4f}  "
              f"{Y_np[idx,2]:>12.6e}  {all_configs[idx]}")
    print("=" * 70)

    return {
        "X":             X_np,
        "Y":             Y_np,
        "configs":       all_configs,
        "pareto_mask":   pareto_mask,
        "pareto_configs": [c for c, m in zip(all_configs, pareto_mask) if m],
        "pareto_Y":      Y_np[pareto_mask],
        "param_names":   param_names,
    }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="ZoMBI-Hop V2 benchmark: region-based distance + 3-objective MOBO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mobo",         action="store_true",
                    help="Run MOBO hyperparameter tuning.")
    ap.add_argument("--regions",      type=str, default=None,
                    help="Path to max_min_regions.json from interactive_maxima_selector.py.")
    ap.add_argument("--mobo-init",    type=int, default=8)
    ap.add_argument("--mobo-iters",   type=int, default=20)
    ap.add_argument(
        "--batch", "--parallel",
        dest="batch",
        type=int,
        default=4,
        help="MOBO: acquisition batch size (q).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="MOBO: parallel processes to evaluate each batch (1 = sequential).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="ZoMBI device: cuda or cpu. Omit to auto-select (prefer CUDA).",
    )
    ap.add_argument("--results-json", type=str, default="hyperparam_results_v2.json")
    args = ap.parse_args()

    if args.device is not None and args.device not in ("cuda", "cpu"):
        ap.error("--device must be cuda or cpu")

    dev_kw: Dict[str, Optional[str]] = {}
    if args.device is not None:
        dev_kw["device"] = args.device

    if args.mobo:
        mobo_tune_zombi_v2(
            regions_path=args.regions,
            n_initial=args.mobo_init,
            n_mobo_iterations=args.mobo_iters,
            n_parallel=args.batch,
            mobo_workers=args.workers,
            csv_path=CSV_PEROVSKITE_PATH,
            verbose_zombi=False,
            results_json=args.results_json,
            **dev_kw,
        )
    else:
        results = run_zombi_test_v2(
            regions_path=args.regions,
            epsilon_frac=0.2,
            max_zooms=3,
            max_iterations=10,
            n_restarts=30,
            raw_samples=500,
            convergence_pi_threshold=0.01,
            n_consecutive_converged=2,
            ucb_beta=0.1,
            nat_grad_step=0.02,
            nat_grad_max_steps=50,
            num_points_per_line=100,
            num_lines=30,
            num_init_data=4,
            rf_global_samples=10_000_000,
            csv_path=CSV_PEROVSKITE_PATH,
            verbose=True,
            **dev_kw,
        )
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY (V2)")
        print("=" * 70)
        header = (f"{'Objective':<50}  {'Ndl':>4}  {'RegDist':>9}"
                  f"  {'Tot':>6}  {'Redund':>6}  {'Wst%':>5}")
        print(header)
        print("-" * len(header))
        for res in results:
            n   = len(res["needles"])
            d   = res["avg_region_dist"]
            d_s = f"{d:.5f}" if np.isfinite(d) else "    n/a"
            nt  = res["n_total_points"]
            nr  = res["n_redundant"]
            wp  = 100.0 * nr / nt if nt > 0 else 0.0
            print(f"  {res['name']:<48}  {n:>4}  {d_s:>9}"
                  f"  {nt:>6}  {nr:>6}  {wp:>4.1f}%")
        print("=" * 70)
