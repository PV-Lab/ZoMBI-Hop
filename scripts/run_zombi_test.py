"""
run_zombi_test.py
=================

Self-contained benchmark harness for ZoMBI-Hop + LineBO on a 3-element simplex.

Workflow
--------
1. Load the FAPbI3/MAPbI3/MAPbBr3 perovskite campaign CSV and train one
   Random Forest per objective (Bandgap, Photoconductance, Stability, Objective).
2. Find each RF's global extrema by drawing 10 M random simplex samples.
3. Define four synthetic Ackley test functions on the same 3-simplex.
4. Visualise all objectives on a ternary plot, then run ZoMBI-Hop (max
   5 activations) on each one.
5. Report the needles found and the average distance from each known maximum
   to its nearest needle (greedy matching).

Usage
-----
    python scripts/run_zombi_test.py
"""

from __future__ import annotations

import sys
import json
import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import joblib
from sklearn.ensemble import RandomForestRegressor

# ── project imports ────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RF_CACHE_DIR = os.path.join(_REPO_ROOT, "test_rfs")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.core.zombihop import ZoMBIHop
from src.core.linebo import LineBO
from src.utils.gp_simplex import GPSimplex
from src.utils.simplex import random_simplex, proj_simplex

# ── Perovskite CSV dataset ─────────────────────────────────────────────────────
# FAPbI3 / MAPbI3 / MAPbBr3 validation campaign dataset
_CSV_DEFAULT = os.path.join(
    os.path.dirname(__file__),
    "campaign_1a_validation_set_1_766_compositions_MAFAPbIBr (2).csv",
)
CSV_PEROVSKITE_PATH: str = _CSV_DEFAULT

# ── Simplex / optimisation constants ──────────────────────────────────────────
_D = 3          # dimensionality (3-element simplex)
_ACKLEY_A = 20.0
_ACKLEY_B = 0.2
_ACKLEY_B_SKINNY = 1.2   # larger b → skinnier peaks (used in multimodal_ackley)
_ACKLEY_C = 2.0 * np.pi
_ACKLEY_SCALE = 30.0   # vertical scaling so Y values are numerically healthy

ACKLEY_CENTER_EQUAL  = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
ACKLEY_CENTER_EDGE   = np.array([0.5, 0.5, 0.0])
ACKLEY_CENTER_VERTEX = np.array([1.0, 0.0, 0.0])
MULTIMODAL_CENTERS   = [ACKLEY_CENTER_EQUAL, ACKLEY_CENTER_EDGE, ACKLEY_CENTER_VERTEX]


# =============================================================================
# 1.  NaturalGradGPSimplex  –  Method D (Fisher-Rao mirror descent) candidate
# =============================================================================

class NaturalGradGPSimplex(GPSimplex):
    """
    GPSimplex variant that uses natural gradient ascent (Method D) to climb
    the acquisition surface instead of the default projected Euclidean step.

    Update rule (per step):

        g     = ∇_x acq(x)
        ḡ_x   = Σ_i x_i * g_i          (Fisher metric weighted mean)
        x_new = x * exp( α * (g - ḡ_x) )
        x_new = x_new / Σ x_new          (renormalise onto simplex)
        x_new = clamp(x_new, lo, hi) / Σ (renormalise after box clamp)

    Parameters
    ----------
    nat_grad_step : float
        Step size α.  Default: 0.02.
    nat_grad_max_steps : int
        Maximum ascent steps per restart.  Default: 50.
    All other parameters are forwarded to ``GPSimplex.__init__``.
    """

    def __init__(
        self,
        *args,
        nat_grad_step: float = 0.02,
        nat_grad_max_steps: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nat_grad_step = nat_grad_step
        self.nat_grad_max_steps = nat_grad_max_steps

    def _optimize_acquisition(
        self,
        acq,
        bounds: torch.Tensor,
        initial_conditions: torch.Tensor,
        **_ignored_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Natural gradient ascent over all restarts.

        Parameters
        ----------
        acq : nn.Module
            Acquisition function (already wraps needle repulsion).
        bounds : torch.Tensor
            (2, d) search bounds.
        initial_conditions : torch.Tensor
            (num_restarts, 1, d) starting points.

        Returns
        -------
        candidates : torch.Tensor
            (R, d) — one per successful restart.
        values : torch.Tensor
            (R,)  — acquisition value at each candidate.
        """
        lo = bounds[0].to(device=self.device, dtype=self.dtype)  # (d,)
        hi = bounds[1].to(device=self.device, dtype=self.dtype)  # (d,)
        d = initial_conditions.shape[-1]

        candidates_list: list = []
        values_list: list = []

        for r in range(initial_conditions.shape[0]):
            # Squeeze to (d,)
            x_raw = initial_conditions[r]  # (1, d) or (1, 1, d)
            x = x_raw.reshape(d).clone().to(device=self.device, dtype=self.dtype)
            x = proj_simplex(x.unsqueeze(0)).squeeze(0)  # ensure on simplex

            for _ in range(self.nat_grad_max_steps):
                x = x.detach().requires_grad_(True)
                x_in = x.unsqueeze(0).unsqueeze(0)  # (1, 1, d)

                try:
                    val = acq(x_in)
                    g = torch.autograd.grad(val.sum(), x)[0]  # (d,)
                except RuntimeError:
                    break

                with torch.no_grad():
                    x_det = x.detach()
                    g_bar = (x_det * g).sum()              # scalar: Σ xᵢ gᵢ
                    shift = self.nat_grad_step * (g - g_bar)
                    shift = torch.clamp(shift, -10.0, 10.0)  # guard against overflow
                    x_new = x_det * torch.exp(shift)

                    s = x_new.sum()
                    if s < 1e-12:
                        break
                    x_new = x_new / s                       # renormalise

                    # Enforce box bounds, then renormalise again
                    x_new = torch.clamp(x_new, lo, hi)
                    s2 = x_new.sum()
                    if s2 < 1e-12:
                        break
                    x = x_new / s2

            # Evaluate final value
            x = x.detach()
            x_in = x.unsqueeze(0).unsqueeze(0)
            try:
                with torch.no_grad():
                    final_val = acq(x_in)
            except RuntimeError:
                continue

            candidates_list.append(x)                         # (d,)
            values_list.append(final_val.squeeze())           # scalar tensor

        if not candidates_list:
            empty = torch.empty(0, d, device=bounds.device, dtype=bounds.dtype)
            return empty, torch.empty(0, device=bounds.device, dtype=bounds.dtype)

        return torch.stack(candidates_list), torch.stack(values_list)


# =============================================================================
# 2.  NaturalGradZoMBIHop  –  ZoMBIHop wired to NaturalGradGPSimplex
# =============================================================================

class NaturalGradZoMBIHop(ZoMBIHop):
    """
    ZoMBIHop subclass that replaces the default ``GPSimplex`` with
    ``NaturalGradGPSimplex`` (Method D natural gradient ascent).

    Extra parameters
    ----------------
    nat_grad_step : float
        Natural gradient step size.  Default: 0.02.
    nat_grad_max_steps : int
        Max ascent steps per restart.  Default: 50.
    All other parameters are forwarded to ``ZoMBIHop.__init__``.
    """

    def __init__(
        self,
        *args,
        nat_grad_step: float = 0.02,
        nat_grad_max_steps: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Swap out the GP handler created by the parent constructor
        self.gp_handler = NaturalGradGPSimplex(
            data_handler=self.data_handler,
            proj_fn=self.proj_fn,
            random_sampler=self.random_sampler,
            num_restarts=self.data_handler.n_restarts,
            raw_samples=self.data_handler.raw,
            repulsion_lambda=self.data_handler.repulsion_lambda,
            acquisition_type=self.data_handler.acquisition_type,
            ucb_beta=self.data_handler.ucb_beta,
            device=str(self.device),
            dtype=self.dtype,
            nat_grad_step=nat_grad_step,
            nat_grad_max_steps=nat_grad_max_steps,
        )


# =============================================================================
# 3.  (Materials Project integration — kept as reference, not active)
#
# To re-enable MP objectives:
#   1. pip install mp-api>=0.41.0
#   2. Uncomment the block below and supply your API key.
#   3. Call fetch_materials_data / find_best_element_triple / build_rf_objective
#      from run_zombi_test() the same way build_csv_rf_objectives() is called.
#
# import itertools
# from collections import defaultdict
#
# MP_API_KEY = "your-key-here"
#
# def fetch_materials_data(api_key, cache_dir=None):
#     """Fetch bandgap / dielectric / work-function records from MP (1-3 elements)."""
#     from mp_api.client import MPRester
#     results = {"bandgap": [], "dielectric": [], "workfn": []}
#     if cache_dir:
#         os.makedirs(cache_dir, exist_ok=True)
#         for key in results:
#             p = os.path.join(cache_dir, f"mp_records_{key}.joblib")
#             if os.path.isfile(p):
#                 results[key] = joblib.load(p)
#     if all(results.values()):
#         return results
#     with MPRester(api_key) as mpr:
#         if not results["bandgap"]:
#             docs = mpr.materials.summary.search(
#                 num_elements=(1, 3),
#                 fields=["material_id", "formula_pretty", "elements",
#                         "composition", "band_gap", "nelements"],
#             )
#             for doc in docs:
#                 bg = getattr(doc, "band_gap", None)
#                 if bg is None or np.isnan(float(bg)):
#                     continue
#                 comp = getattr(doc, "composition", None)
#                 if comp is None:
#                     continue
#                 results["bandgap"].append({
#                     "material_id": doc.material_id,
#                     "formula_pretty": doc.formula_pretty,
#                     "elements": [str(e) for e in doc.elements],
#                     "composition": comp,
#                     "value": float(bg),
#                 })
#             if cache_dir:
#                 joblib.dump(results["bandgap"],
#                             os.path.join(cache_dir, "mp_records_bandgap.joblib"))
#         # (add dielectric / workfn blocks here following the same pattern)
#     return results
#
# def encode_composition(composition, element_triple):
#     """pymatgen Composition → 3-component simplex vector for the given triple."""
#     e1, e2, e3 = element_triple
#     allowed = {e1, e2, e3}
#     amounts = {}
#     for elem in composition.elements:
#         key = str(elem)
#         if key not in allowed:
#             return None
#         amounts[key] = composition[elem]
#     total = sum(amounts.values())
#     if total < 1e-10:
#         return None
#     return np.array([amounts.get(e, 0.0) / total for e in (e1, e2, e3)])
#
# def find_best_element_triple(records):
#     """Return (triple, filtered_records) maximising dataset size."""
#     set_counts = defaultdict(int)
#     for rec in records:
#         set_counts[frozenset(rec["elements"])] += 1
#     all_elems = sorted({e for rec in records for e in rec["elements"]})
#     best, best_triple = -1, None
#     for triple in itertools.combinations(all_elems, 3):
#         ts = frozenset(triple)
#         cnt = sum(v for k, v in set_counts.items() if k.issubset(ts))
#         if cnt > best:
#             best, best_triple = cnt, triple
#     filtered = [r for r in records if frozenset(r["elements"]).issubset(frozenset(best_triple))]
#     return best_triple, filtered
#
# def build_rf_objective(records, element_triple, rf_global_samples=10_000_000,
#                        rf_kwargs=None, cache_dir=None, cache_key=None):
#     """Train RF on simplex coords → property; find global max/min via Dirichlet sampling."""
#     if cache_dir and cache_key:
#         p = os.path.join(cache_dir, f"rf_{cache_key}.joblib")
#         if os.path.isfile(p):
#             return joblib.load(p)
#     X = np.array([enc for rec in records
#                   for enc in [encode_composition(rec["composition"], element_triple)]
#                   if enc is not None], dtype=np.float64)
#     y = np.array([rec["value"] for rec in records
#                   if encode_composition(rec["composition"], element_triple) is not None],
#                  dtype=np.float64)
#     rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42,
#                                **(rf_kwargs or {}))
#     rf.fit(X, y)
#     best_max_val, best_max_x, best_min_val, best_min_x = -np.inf, None, np.inf, None
#     for _ in range(rf_global_samples // 500_000):
#         raw = np.random.exponential(1.0, (500_000, 3))
#         Xc = raw / raw.sum(1, keepdims=True)
#         p = rf.predict(Xc)
#         if p.max() > best_max_val:
#             best_max_val, best_max_x = float(p.max()), Xc[p.argmax()].copy()
#         if p.min() < best_min_val:
#             best_min_val, best_min_x = float(p.min()), Xc[p.argmin()].copy()
#     out = dict(rf=rf, element_triple=element_triple,
#                global_max_x=best_max_x, global_max_val=best_max_val,
#                global_min_x=best_min_x, global_min_val=best_min_val)
#     if cache_dir and cache_key:
#         joblib.dump(out, os.path.join(cache_dir, f"rf_{cache_key}.joblib"))
#     return out
# =============================================================================


# =============================================================================
# 5b.  Perovskite CSV RF objectives (FAPbI3 / MAPbI3 / MAPbBr3)
# =============================================================================

# Axis labels for perovskite ternary plots
CSV_ELEMENT_TRIPLE: Tuple[str, str, str] = ("FAPbI3", "MAPbI3", "MAPbBr3")

# Property columns to use as objectives
CSV_OBJECTIVES: List[str] = ["Bandgap", "Photoconductance", "Stability", "Objective"]


def build_csv_rf_objectives(
    csv_path: str = CSV_PEROVSKITE_PATH,
    objectives: List[str] = CSV_OBJECTIVES,
    rf_global_samples: int = 10_000_000,
    rf_kwargs: Optional[dict] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Load the perovskite CSV dataset and train one RF per objective.

    The input features are the three stoichiometric mixing ratios
    (FAPbI3, MAPbI3, MAPbBr3) which already live on the simplex
    (each row sums to ~1).

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    objectives : list of str
        Column names to use as prediction targets.
    rf_global_samples : int
        Number of random Dirichlet samples for global-extremum search.
    rf_kwargs : dict, optional
        Extra kwargs forwarded to ``RandomForestRegressor``.
    cache_dir : str, optional
        Directory for caching built models (``rf_csv_{key}.joblib``).

    Returns
    -------
    dict mapping objective name → RF result dict (same schema as
    ``build_rf_objective``).
    """
    import pandas as pd

    print(f"  Loading perovskite CSV: {os.path.basename(csv_path)} …")
    df = pd.read_csv(csv_path)

    feature_cols = ["FAPbI3", "MAPbI3", "MAPbBr3"]
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"CSV is missing expected column '{col}'")

    # Renormalise rows so each composition sums exactly to 1
    X_raw = df[feature_cols].values.astype(np.float64)
    row_sums = X_raw.sum(axis=1, keepdims=True)
    X_raw = X_raw / np.where(row_sums > 0, row_sums, 1.0)
    print(f"  {len(X_raw)} compositions loaded.")

    rf_results: Dict[str, Dict] = {}
    rf_kwargs = rf_kwargs or {}

    for obj_col in objectives:
        if obj_col not in df.columns:
            warnings.warn(f"Column '{obj_col}' not found in CSV; skipping.", stacklevel=2)
            continue

        cache_key = f"csv_{obj_col.lower()}"
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"rf_{cache_key}.joblib")
            if os.path.isfile(cache_path):
                print(f"  [{obj_col}] Loaded from cache")
                rf_results[obj_col] = joblib.load(cache_path)
                continue

        y_raw = df[obj_col].values
        valid = ~pd.isna(y_raw)
        X_train = X_raw[valid]
        y_train = y_raw[valid].astype(np.float64)

        if len(X_train) < 10:
            warnings.warn(
                f"Too few valid rows ({len(X_train)}) for '{obj_col}'; skipping.",
                stacklevel=2,
            )
            continue

        print(f"\n  [{obj_col}] Training RF on {len(X_train)} samples …")
        kw = dict(rf_kwargs)
        rf = RandomForestRegressor(
            n_estimators=kw.pop("n_estimators", 500),
            max_depth=kw.pop("max_depth", None),
            n_jobs=-1,
            random_state=42,
            **kw,
        )
        rf.fit(X_train, y_train)
        print(f"  [{obj_col}] RF training complete.")

        # Global extremum search
        print(f"  [{obj_col}] Searching global extremum ({rf_global_samples:,} samples) …")
        chunk = 500_000
        best_max_val, best_max_x = -np.inf, None
        best_min_val, best_min_x = np.inf, None

        for start in range(0, rf_global_samples, chunk):
            n = min(chunk, rf_global_samples - start)
            raw = np.random.exponential(1.0, size=(n, 3))
            X_chunk = raw / raw.sum(axis=1, keepdims=True)
            preds = rf.predict(X_chunk)

            idx_max = preds.argmax()
            if preds[idx_max] > best_max_val:
                best_max_val = float(preds[idx_max])
                best_max_x = X_chunk[idx_max].copy()

            idx_min = preds.argmin()
            if preds[idx_min] < best_min_val:
                best_min_val = float(preds[idx_min])
                best_min_x = X_chunk[idx_min].copy()

        print(f"  [{obj_col}] Global max: {best_max_val:.4f} at {best_max_x}")
        print(f"  [{obj_col}] Global min: {best_min_val:.4f} at {best_min_x}")

        out = {
            "rf": rf,
            "element_triple": CSV_ELEMENT_TRIPLE,
            "n_train": len(X_train),
            "global_max_x": best_max_x,
            "global_max_val": best_max_val,
            "global_min_x": best_min_x,
            "global_min_val": best_min_val,
        }
        if cache_dir:
            joblib.dump(out, cache_path)
            print(f"  [{obj_col}] Cached → {cache_path}")

        rf_results[obj_col] = out

    return rf_results


# =============================================================================
# 6.  Ackley test functions
# =============================================================================

def _ackley_negated(
    x: np.ndarray,
    center: np.ndarray,
    a: float = _ACKLEY_A,
    b: float = _ACKLEY_B,
    c: float = _ACKLEY_C,
    scale: float = _ACKLEY_SCALE,
) -> float:
    """
    Negated Ackley centred at ``center``.

    Maximum is at ``x == center`` (value ≈ 0).  The function is negative
    away from the centre; ZoMBI-Hop's maximisation finds the peak correctly.

    Parameters
    ----------
    x : np.ndarray (3,)
        Query point on the 3-simplex.
    center : np.ndarray (3,)
        Peak location.
    """
    x = np.asarray(x, dtype=float)
    d = x.shape[0]
    delta = x - center
    t1 = -a * np.exp(-b * np.sqrt(np.sum(delta ** 2) / d))
    t2 = -np.exp(np.sum(np.cos(c * delta)) / d)
    return scale * (t1 + t2 + a + np.e)


def multimodal_ackley(x: np.ndarray) -> float:
    """
    Sum of three negated Ackley functions, each centred at one of the three
    canonical simplex positions.  Uses skinnier peaks (larger b) so the three
    maxima are more separated.  Known maxima:

        [1/3, 1/3, 1/3],  [0.5, 0.5, 0],  [1, 0, 0]
    """
    return sum(
        _ackley_negated(x, c, b=_ACKLEY_B_SKINNY) for c in MULTIMODAL_CENTERS
    )


def ackley_equal(x: np.ndarray) -> float:
    """Negated Ackley peaked at the centroid [1/3, 1/3, 1/3]."""
    return _ackley_negated(x, ACKLEY_CENTER_EQUAL)


def ackley_edge(x: np.ndarray) -> float:
    """Negated Ackley peaked at the edge midpoint [0.5, 0.5, 0]."""
    return _ackley_negated(x, ACKLEY_CENTER_EDGE)


def ackley_vertex(x: np.ndarray) -> float:
    """Negated Ackley peaked at a simplex vertex [1, 0, 0]."""
    return _ackley_negated(x, ACKLEY_CENTER_VERTEX)


# =============================================================================
# 7.  Ternary visualisation helper
# =============================================================================

def _to_cart(pts: np.ndarray) -> np.ndarray:
    """
    Convert barycentric coordinates (n, 3) to 2-D Cartesian (n, 2).

    Vertex mapping:
        axis-0 (b1=1) → (0,      0)
        axis-1 (b2=1) → (1,      0)
        axis-2 (b3=1) → (0.5,  √3/2)
    """
    pts = np.asarray(pts)
    x = pts[:, 1] + 0.5 * pts[:, 2]
    y = (np.sqrt(3) / 2) * pts[:, 2]
    return np.column_stack([x, y])


def _draw_triangle(ax, labels: Tuple[str, str, str] = ("A", "B", "C")):
    """Draw the simplex triangle and label the vertices."""
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    tri = plt.Polygon(corners, fill=False, edgecolor="black", linewidth=1.5)
    ax.add_patch(tri)
    offsets = [(-0.07, -0.05), (1.02, -0.05), (0.50, np.sqrt(3) / 2 + 0.04)]
    for (ox, oy), lbl in zip(offsets, labels):
        ax.text(ox, oy, lbl, ha="center", va="center", fontsize=9,
                fontweight="bold")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.10, np.sqrt(3) / 2 + 0.15)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_objectives(
    rf_data: Dict[str, Dict],
    ackley_fns: List[Tuple[str, Callable, List[np.ndarray]]],
    element_triple: Tuple[str, str, str],
    n_vis_pts: int = 5_000,
    save_path: str = "zombi_test_objectives.png",
) -> None:
    """
    Plot all objectives on ternary scatter plots before running ZoMBI-Hop.

    Parameters
    ----------
    rf_data : dict
        Mapping from objective name to ``build_rf_objective`` result dict.
        Keys: "bandgap", "dielectric", "workfn".
    ackley_fns : list of (name, callable, list_of_known_maxima)
    element_triple : tuple of 3 str
        Used to label the RF ternary axes.
    n_vis_pts : int
        Random simplex points to draw per panel.
    save_path : str
        File path for the output figure.
    """
    n_panels = len(rf_data) + len(ackley_fns)
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes_flat = axes.flatten() if nrows > 1 else np.array(axes).flatten()

    # Random simplex samples for all panels
    rng = np.random.default_rng(0)
    raw = rng.exponential(1.0, size=(n_vis_pts, 3))
    pts = raw / raw.sum(axis=1, keepdims=True)  # uniform Dirichlet
    cart = _to_cart(pts)

    panel_idx = 0

    # ── RF panels ─────────────────────────────────────────────────────────────
    rf_labels = {"bandgap": "Band Gap (eV)",
                 "dielectric": "Dielectric Constant",
                 "workfn": "Work Function (eV)"}

    for name, data in rf_data.items():
        ax = axes_flat[panel_idx]
        panel_idx += 1

        rf_vals = data["rf"].predict(pts)
        sc = ax.scatter(cart[:, 0], cart[:, 1], c=rf_vals, cmap="viridis",
                        s=8, alpha=0.7, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        _draw_triangle(ax, labels=element_triple)

        # Mark global max
        gmax_cart = _to_cart(data["global_max_x"].reshape(1, 3))
        ax.scatter(gmax_cart[:, 0], gmax_cart[:, 1], marker="*",
                   color="red", s=200, zorder=10, label="global max")
        ax.set_title(f"RF – {rf_labels.get(name, name)}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")

    # ── Ackley panels ──────────────────────────────────────────────────────────
    for name, fn, known_maxima in ackley_fns:
        ax = axes_flat[panel_idx]
        panel_idx += 1

        fn_vals = np.array([fn(pts[i]) for i in range(n_vis_pts)])
        sc = ax.scatter(cart[:, 0], cart[:, 1], c=fn_vals, cmap="plasma",
                        s=8, alpha=0.7, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        _draw_triangle(ax, labels=("A", "B", "C"))

        for km in known_maxima:
            km_cart = _to_cart(np.array(km).reshape(1, 3))
            ax.scatter(km_cart[:, 0], km_cart[:, 1], marker="*",
                       color="red", s=200, zorder=10)
        ax.set_title(f"Ackley – {name}", fontsize=10)

    # Hide unused panels
    for i in range(panel_idx, len(axes_flat)):
        axes_flat[i].axis("off")

    fig.suptitle("ZoMBI-Hop Benchmark Objectives", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Objectives plot saved → {save_path}")
    print("Close the plot window to begin optimisation …")
    plt.show()
    plt.close(fig)


# =============================================================================
# 8.  LineBO objective factory
# =============================================================================

def _make_inner_linebo_obj(
    fn: Callable,
    n_line_pts: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Return a LineBO-compatible inner objective for any scalar function ``fn``.

    The closure receives ``endpoints: (num_lines, 2, d)`` (ranked best-first)
    and evaluates ``fn`` along ``n_line_pts`` evenly-spaced points on the
    highest-ranked line.

    Returns
    -------
    Callable: endpoints → (x_actual: Tensor(n, d), y: Tensor(n,))
    """
    def inner_obj(endpoints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_left  = endpoints[0, 0].to(device=device, dtype=dtype)  # (d,)
        x_right = endpoints[0, 1].to(device=device, dtype=dtype)  # (d,)

        t = torch.linspace(0, 1, n_line_pts, device=device, dtype=dtype)
        x_line = (
            x_left.unsqueeze(0) * (1 - t).unsqueeze(1)
            + x_right.unsqueeze(0) * t.unsqueeze(1)
        )  # (n_line_pts, d)

        y_vals = np.array([fn(x_line[i].cpu().numpy()) for i in range(n_line_pts)])
        y = torch.tensor(y_vals, device=device, dtype=dtype)  # (n_line_pts,)
        return x_line, y

    return inner_obj


def make_zombi_objective(
    fn: Callable,
    num_points_per_line: int,
    num_lines: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Build a ZoMBI-Hop-compatible objective wrapping ``fn`` via LineBO.

    The returned callable matches the signature expected by ``ZoMBIHop``:
        ``objective(X: Tensor(d,), bounds, acq_fn) → (X_expected, X_actual, Y)``
    where Y is a 1-D tensor of shape ``(n,)``.
    """
    inner_obj = _make_inner_linebo_obj(fn, num_points_per_line, device, dtype)
    linebo = LineBO(
        objective_function=inner_obj,
        dimensions=_D,
        num_points_per_line=num_points_per_line,
        num_lines=num_lines,
        device=str(device),
    )

    def objective(
        X: torch.Tensor,
        bounds: torch.Tensor,
        acquisition_function,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_req, x_act, y = linebo.sampler(X, bounds, acquisition_function)
        if y.dim() == 2:
            y = y.squeeze(-1)
        return x_req, x_act, y

    return objective


# =============================================================================
# 9.  Single-objective ZoMBI-Hop runner
# =============================================================================

def run_zombi_on_objective(
    fn: Callable,
    known_maxima: List[np.ndarray],
    name: str,
    *,
    num_init_data: int = 4,
    max_activations: int = 5,
    # ZoMBI hyperparams
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
    # GP hyperparams
    ucb_beta: float = 0.1,
    repulsion_lambda: Optional[float] = None,
    nat_grad_step: float = 0.02,
    nat_grad_max_steps: int = 50,
    # LineBO hyperparams
    num_points_per_line: int = 100,
    # Noise
    input_noise: float = 0.0,
    output_noise: float = 0.0,
    num_lines: int = 30,
    # Runtime
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    seed: int = 0,
) -> Dict:
    """
    Run ``NaturalGradZoMBIHop`` on a single objective.

    Parameters
    ----------
    fn : Callable
        Scalar objective ``f(x: np.ndarray) → float``.  ZoMBI maximises this.
    known_maxima : list of np.ndarray
        True peak locations (used only for evaluation, not during optimisation).
    name : str
        Human-readable label for logging.
    input_noise : float
        Std-dev of Gaussian noise added to the query point before evaluating
        ``fn`` (simplex-projected after perturbation).  0 = noiseless.
    output_noise : float
        Std-dev of Gaussian noise added to the scalar output of ``fn``.
        0 = noiseless.
    ... (see module docstring for full hyperparameter list)

    Returns
    -------
    dict with keys:
        name, needles, needle_vals, avg_distance_to_known_maxima,
        known_maxima, needles_results, n_total_points, n_duplicate_points,
        input_noise, output_noise
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_t = torch.device(device)

    lo = torch.zeros(_D, device=device_t, dtype=dtype)
    hi = torch.ones(_D, device=device_t, dtype=dtype)
    bounds = torch.stack([lo, hi])  # (2, d)

    # ── Noise wrapper ──────────────────────────────────────────────────────────
    def _noisy_fn(x: np.ndarray) -> float:
        x_eval = x.copy()
        if input_noise > 0.0:
            perturbed = x_eval + np.random.normal(0.0, input_noise, size=x_eval.shape)
            # Project back onto simplex: clip negatives, renormalise
            perturbed = np.clip(perturbed, 0.0, None)
            s = perturbed.sum()
            x_eval = perturbed / s if s > 1e-12 else x_eval
        y = fn(x_eval)
        if output_noise > 0.0:
            y += np.random.normal(0.0, output_noise)
        return float(y)

    eval_fn = _noisy_fn if (input_noise > 0.0 or output_noise > 0.0) else fn

    # ── Initial random simplex data ────────────────────────────────────────────
    X_init = random_simplex(
        num_init_data, lo, hi, device=device, torch_dtype=dtype
    )  # (n_init, d)

    Y_init = torch.tensor(
        [eval_fn(X_init[i].cpu().numpy()) for i in range(num_init_data)],
        device=device_t, dtype=dtype,
    ).unsqueeze(1)  # (n_init, 1)  — ZoMBIHop requires (n, 1) at init

    # ── Build ZoMBI objective (LineBO wrapper) ─────────────────────────────────
    objective = make_zombi_objective(eval_fn, num_points_per_line, num_lines, device_t, dtype)

    print(f"\n{'─' * 60}")
    print(f"Running ZoMBI-Hop on: {name}")
    print(f"  Known maxima: {[km.tolist() for km in known_maxima]}")
    print(f"{'─' * 60}")

    # ── Instantiate NaturalGradZoMBIHop ───────────────────────────────────────
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
        checkpoint_dir=None,    # no checkpointing
        max_snapshots=0,
        verbose=verbose,
        nat_grad_step=nat_grad_step,
        nat_grad_max_steps=nat_grad_max_steps,
    )

    needles_results, needles, needle_vals, _X_all, _Y_all = optimizer.run(
        max_activations=max_activations
    )

    # ── Duplicate-point analysis (within tolerance 1e-4) ──────────────────────
    # "redundant" = wasted evaluations: for each cluster of near-identical
    # points, the first is kept and the rest are counted as redundant.
    # e.g. if A, B, C are all within 1e-4 of each other → 2 redundant evals.
    X_all_np = _X_all.cpu().numpy()  # (n_total, d)
    n_total = X_all_np.shape[0]
    n_redundant = 0   # number of *wasted* evaluations (cluster size - 1 per cluster)
    n_in_any_dup = 0  # number of individual points that have ≥1 near-neighbour
    if n_total > 1:
        from scipy.spatial.distance import pdist, squareform
        dist_mat = squareform(pdist(X_all_np, metric="chebyshev"))
        np.fill_diagonal(dist_mat, np.inf)
        near = dist_mat < 1e-4  # (n, n) bool, diagonal=False

        # Points that have at least one near-neighbour
        n_in_any_dup = int(np.any(near, axis=1).sum())

        # Redundant count: greedily assign each point to the earliest point
        # in its cluster; all later arrivals are redundant.
        assigned = np.zeros(n_total, dtype=bool)
        for i in range(n_total):
            if not assigned[i]:
                neighbours = np.where(near[i])[0]
                neighbours = neighbours[neighbours > i]   # only later points
                assigned[neighbours] = True
                n_redundant += len(neighbours)

    print(f"\n[{name}] {n_total} total sampled points")
    print(f"[{name}]   {n_in_any_dup} points have ≥1 near-neighbour (tol=1e-4)")
    print(f"[{name}]   {n_redundant} redundant evaluations (wasted re-samples)")

    # ── Distance metric: greedy nearest-needle per known maximum ───────────────
    n_needles = needles.shape[0] if torch.is_tensor(needles) else 0
    if n_needles == 0:
        avg_dist = float("inf")
    else:
        needles_np = needles.cpu().numpy()  # (n_needles, d)
        total_dist = 0.0
        for km in known_maxima:
            km_arr = np.asarray(km)
            dists = np.linalg.norm(needles_np - km_arr[None, :], axis=1)  # (n_needles,)
            total_dist += float(dists.min())
        avg_dist = total_dist / len(known_maxima)

    print(f"[{name}] Found {n_needles} needle(s)")
    print(f"[{name}] Avg distance from known maxima to nearest needle: {avg_dist:.5f}")

    return {
        "name": name,
        "needles": needles.cpu().numpy() if n_needles > 0 else np.empty((0, _D)),
        "needle_vals": (
            needle_vals.cpu().numpy()
            if torch.is_tensor(needle_vals) and needle_vals.numel() > 0
            else np.array([])
        ),
        "avg_distance_to_known_maxima": avg_dist,
        "known_maxima": [np.asarray(km) for km in known_maxima],
        "needles_results": needles_results,
        "n_total_points": n_total,
        "n_in_any_dup": n_in_any_dup,       # points that have ≥1 near-neighbour
        "n_redundant": n_redundant,          # wasted re-evaluations (cluster size - 1)
        "input_noise": input_noise,
        "output_noise": output_noise,
    }


# =============================================================================
# 10.  Top-level entry point
# =============================================================================

def run_zombi_test(
    *,
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
    max_activations: int = 5,
    num_init_data: int = 4,
    rf_global_samples: int = 10_000_000,
    rf_cache_dir: Optional[str] = RF_CACHE_DIR,
    csv_path: str = CSV_PEROVSKITE_PATH,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    vis_n_pts: int = 5_000,
    plot_save_path: str = "zombi_test_objectives.png",
    show_plot: bool = True,
) -> List[Dict]:
    """
    Full benchmark pipeline.

    1. Load the perovskite CSV and train RF models for Bandgap,
       Photoconductance, Stability, and Objective.
    2. Define four synthetic Ackley objectives.
    3. Visualise all objectives on ternary plots.
    4. Run ZoMBI-Hop (max ``max_activations`` activations) on each objective.
    5. Print a summary table and return the list of result dicts.

    Returns
    -------
    list of dict, one per objective, with keys:
        name, needles, needle_vals, avg_distance_to_known_maxima,
        known_maxima, needles_results
    """
    _zombi_kw = dict(
        num_init_data=num_init_data,
        max_activations=max_activations,
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
    )

    # ── Step 1: Perovskite CSV RF objectives ───────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: Perovskite CSV (FAPbI3/MAPbI3/MAPbBr3) RF objectives")
    print("=" * 70)

    csv_rf_objectives: Dict[str, Dict] = {}
    if os.path.isfile(csv_path):
        try:
            csv_rf_objectives = build_csv_rf_objectives(
                csv_path=csv_path,
                objectives=CSV_OBJECTIVES,
                rf_global_samples=rf_global_samples,
                cache_dir=rf_cache_dir,
            )
            print(f"  Built {len(csv_rf_objectives)} CSV RF objective(s): "
                  f"{list(csv_rf_objectives.keys())}")
        except Exception as exc:
            warnings.warn(f"CSV RF build failed: {exc}", stacklevel=2)
    else:
        warnings.warn(
            f"Perovskite CSV not found at:\n    {csv_path}\n"
            "Skipping CSV RF objectives.  Set csv_path= to the correct path.",
            stacklevel=2,
        )

    # ── Step 2: Synthetic Ackley objectives ────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Synthetic Ackley objectives")
    print("=" * 70)

    # (name, callable, list_of_known_maxima)
    ackley_fns: List[Tuple[str, Callable, List[np.ndarray]]] = [
        ("Centroid",    ackley_equal,       [ACKLEY_CENTER_EQUAL]),
        ("Edge",        ackley_edge,        [ACKLEY_CENTER_EDGE]),
        ("Vertex",      ackley_vertex,      [ACKLEY_CENTER_VERTEX]),
        ("Multi-modal", multimodal_ackley,  MULTIMODAL_CENTERS),
    ]
    print(f"  Defined {len(ackley_fns)} Ackley test function(s).")

    # ── Step 3: Visualise all objectives ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Visualising objectives …")
    print("=" * 70)

    if show_plot:
        plot_objectives(
            csv_rf_objectives,
            ackley_fns,
            element_triple=CSV_ELEMENT_TRIPLE,
            n_vis_pts=vis_n_pts,
            save_path=plot_save_path,
        )
    else:
        print("  (plot skipped — show_plot=False)")

    # ── Step 4: Run ZoMBI on every objective × noise level ────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: Running ZoMBI-Hop on all objectives")
    print("=" * 70)

    # All (input_noise, output_noise) combinations to sweep
    noise_combos: List[Tuple[float, float]] = [
        (0.01,  0.01),
        (0.01,  0.001),
        (0.001, 0.01),
        (0.001, 0.001),
    ]

    all_results: List[Dict] = []

    # Build a flat list of (base_name, fn, known_maxima) for all objectives
    objectives_to_run: List[Tuple[str, Callable, List[np.ndarray]]] = []

    for obj_col, data in csv_rf_objectives.items():
        rf = data["rf"]
        def _csv_rf_fn(x: np.ndarray, _rf=rf) -> float:
            return float(_rf.predict(x.reshape(1, -1))[0])
        objectives_to_run.append((
            f"CSV-RF-{obj_col} ({'/'.join(CSV_ELEMENT_TRIPLE)})",
            _csv_rf_fn,
            [data["global_max_x"]],
        ))

    for aname, afn, amax in ackley_fns:
        objectives_to_run.append((
            f"Ackley-{aname}",
            afn,
            [np.asarray(km) for km in amax],
        ))

    total_runs = len(objectives_to_run) * len(noise_combos)
    run_idx = 0
    for base_name, fn, known_maxima in objectives_to_run:
        for inp_noise, out_noise in noise_combos:
            run_idx += 1
            noise_tag = f"in={inp_noise:.3f}/out={out_noise:.3f}"
            full_name = f"{base_name} [{noise_tag}]"
            print(f"\n  [{run_idx}/{total_runs}] {full_name}")
            res = run_zombi_on_objective(
                fn=fn,
                known_maxima=known_maxima,
                name=full_name,
                input_noise=inp_noise,
                output_noise=out_noise,
                **_zombi_kw,
            )
            all_results.append(res)

    # ── Step 5: Summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    header = (f"{'Objective':<50}  {'Ndl':>4}  {'AvgDist':>9}"
              f"  {'Tot':>6}  {'InDup':>6}  {'Redund':>6}  {'Wst%':>5}"
              f"  {'InNoise':>8}  {'OutNoise':>9}")
    print(header)
    print("-" * len(header))
    for res in all_results:
        n      = len(res["needles"])
        d      = res["avg_distance_to_known_maxima"]
        d_str  = f"{d:.5f}" if np.isfinite(d) else "    n/a"
        n_tot  = res["n_total_points"]
        n_any  = res["n_in_any_dup"]     # points touching a duplicate
        n_red  = res["n_redundant"]      # wasted evaluations
        wst_pct = 100.0 * n_red / n_tot if n_tot > 0 else 0.0
        print(f"  {res['name']:<48}  {n:>4}  {d_str:>9}"
              f"  {n_tot:>6}  {n_any:>6}  {n_red:>6}  {wst_pct:>4.1f}%"
              f"  {res['input_noise']:>8.3f}  {res['output_noise']:>9.3f}")
    print("=" * 70)

    return all_results


# =============================================================================
# 11.  MOBO hyperparameter tuning
# =============================================================================

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""
    def default(self, obj):  # noqa: D102
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# Default tunable parameters: (lower, upper)
_DEFAULT_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "ucb_beta":                   (0.01,  2.0),
    "penalty_max_radius":         (0.05,  0.5),
    "penalization_threshold":     (1e-4,  1e-1),   # log scale
    "convergence_pi_threshold":   (1e-3,  0.1),    # log scale
    "nat_grad_step":              (0.005, 0.1),
    "nat_grad_max_steps":         (10,    100),     # integer
    "max_zooms":                  (1,     6),       # integer
    "max_iterations":             (3,     20),      # integer
    "n_restarts":                 (5,     100),     # integer
    "raw_samples":                (50,    2000),    # integer
}
_DEFAULT_LOG_PARAMS  = {"penalization_threshold", "convergence_pi_threshold"}
_DEFAULT_INT_PARAMS  = {"nat_grad_max_steps", "max_zooms", "max_iterations",
                        "n_restarts", "raw_samples"}


def _config_from_unit(
    x: np.ndarray,
    param_names: List[str],
    raw_bounds: Dict[str, Tuple[float, float]],
    log_params: set,
    int_params: set,
) -> Dict:
    """Map a point in [0, 1]^d to a concrete hyperparameter dict."""
    cfg: Dict = {}
    for i, name in enumerate(param_names):
        lo, hi = raw_bounds[name]
        t = float(x[i])
        if name in log_params:
            val = np.exp(np.log(lo) + t * (np.log(hi) - np.log(lo)))
        else:
            val = lo + t * (hi - lo)
        if name in int_params:
            val = int(round(val))
        cfg[name] = val
    return cfg


def _unit_from_config(
    val: float,
    name: str,
    raw_bounds: Dict[str, Tuple[float, float]],
    log_params: set,
) -> float:
    """Inverse of _config_from_unit for a single parameter (ignoring int rounding)."""
    lo, hi = raw_bounds[name]
    if name in log_params:
        return float(np.clip(
            (np.log(val) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0.0, 1.0
        ))
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))


def _evaluate_config(
    config: Dict,
    *,
    fixed_kw: Dict,
) -> Tuple[float, float]:
    """
    Run the full ZoMBI benchmark with *config* merged into *fixed_kw*,
    then return ``(avg_distance, duplicate_fraction)`` — both to minimise.
    """
    merged = {**fixed_kw, **config}
    results = run_zombi_test(**merged)

    dists = [
        r["avg_distance_to_known_maxima"]
        for r in results
        if np.isfinite(r["avg_distance_to_known_maxima"])
    ]
    avg_dist = float(np.mean(dists)) if dists else 1.0

    dup_fracs = [
        r["n_redundant"] / max(r["n_total_points"], 1)
        for r in results
    ]
    avg_dup = float(np.mean(dup_fracs)) if dup_fracs else 1.0

    return avg_dist, avg_dup


def _mobo_worker(task: Tuple[Dict, Dict, int]) -> Tuple[float, float]:
    """
    Top-level (picklable) worker for parallel MOBO evaluation.

    Parameters
    ----------
    task : (config, fixed_kw, threads_per_worker)
        ``config`` is the ZoMBI hyperparameter dict for this trial;
        ``fixed_kw`` contains the fixed benchmark settings;
        ``threads_per_worker`` caps intra-op parallelism per process.
    """
    config, fixed_kw, threads_per_worker = task

    # Cap thread usage so that n_parallel workers don't over-subscribe CPUs
    os.environ["OMP_NUM_THREADS"]      = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"]      = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
    os.environ["BLAS_NUM_THREADS"]     = str(threads_per_worker)
    torch.set_num_threads(threads_per_worker)

    return _evaluate_config(config, fixed_kw=fixed_kw)


def _run_batch_parallel(
    configs: List[Dict],
    x_unit_batch: List[np.ndarray],
    fixed_kw: Dict,
    n_parallel: int,
    threads_per_worker: int,
    label: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
    """
    Evaluate a batch of configs in parallel using ``ProcessPoolExecutor``.
    Returns (list_of_x_tensors, list_of_y_tensors, configs) in the same order.
    """
    import concurrent.futures

    tasks = [(cfg, fixed_kw, threads_per_worker) for cfg in configs]

    print(f"\n  Submitting {len(tasks)} workers (parallelism={n_parallel}, "
          f"threads/worker={threads_per_worker}) …")

    results_xy: List[Tuple[float, float]] = [None] * len(tasks)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_parallel) as pool:
        futures = {pool.submit(_mobo_worker, t): i for i, t in enumerate(tasks)}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                results_xy[idx] = fut.result()
            except Exception as exc:
                warnings.warn(f"  [{label} worker {idx}] failed: {exc}", stacklevel=2)
                results_xy[idx] = (1.0, 1.0)  # pessimistic fallback

    new_X, new_Y = [], []
    for i, (dist, dup) in enumerate(results_xy):
        print(f"  [{label} {i+1}/{len(tasks)}] avg_dist={dist:.5f}  "
              f"dup_frac={dup:.4f}  config={configs[i]}")
        new_X.append(torch.tensor(x_unit_batch[i], dtype=torch.float64))
        new_Y.append(torch.tensor([dist, dup], dtype=torch.float64))

    return new_X, new_Y, configs


def mobo_tune_zombi(
    *,
    n_initial: int = 8,
    n_mobo_iterations: int = 20,
    n_parallel: int = 4,
    threads_per_worker: int = 2,
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    log_scale_params: Optional[set] = None,
    integer_params: Optional[set] = None,
    # ── Fixed run settings (not tuned) ─────────────────────────────────────────
    csv_path: str = CSV_PEROVSKITE_PATH,
    max_activations: int = 3,
    num_init_data: int = 4,
    # max_zooms, max_iterations, n_restarts, raw_samples are tunable — their
    # values here serve only as fallback defaults when param_bounds is overridden.
    max_zooms: int = 3,
    max_iterations: int = 10,
    n_restarts: int = 30,
    raw_samples: int = 500,
    top_m_points: Optional[int] = None,
    n_consecutive_converged: int = 2,
    max_gp_points: int = 3000,
    repulsion_lambda: Optional[float] = None,
    rf_global_samples: int = 10_000_000,
    rf_cache_dir: Optional[str] = RF_CACHE_DIR,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose_zombi: bool = False,
    show_plot: bool = False,
    results_json: str = "hyperparam_results.json",
    seed: int = 0,
) -> Dict:
    """
    High-throughput multi-objective Bayesian optimisation of ZoMBI-Hop
    hyperparameters.

    Each MOBO iteration proposes ``n_parallel`` candidates at once (batch
    qNEHVI with ``q=n_parallel``) and evaluates them simultaneously in
    separate processes, each limited to ``threads_per_worker`` CPU threads.
    The same parallel scheme is used for the initial Sobol phase.

    Minimises two objectives simultaneously:
      1. Average greedy distance from known maxima to nearest needle.
      2. Fraction of sampled points that are duplicates (tol 1e-4).

    Parameters
    ----------
    n_initial : int
        Total number of quasi-random Sobol initial evaluations.
        Rounded up to the nearest multiple of ``n_parallel``.
    n_mobo_iterations : int
        Number of MOBO acquisition rounds after initialisation.
        Each round evaluates ``n_parallel`` configs simultaneously.
    n_parallel : int
        Number of worker processes to run in parallel (default 4).
    threads_per_worker : int
        OMP/MKL/torch thread cap per worker process (default 2).
    param_bounds : dict, optional
        ``{param_name: (lo, hi)}``. Defaults to ``_DEFAULT_PARAM_BOUNDS``.
    log_scale_params : set, optional
        Names to search in log space. Defaults to ``_DEFAULT_LOG_PARAMS``.
    integer_params : set, optional
        Names to round to int. Defaults to ``_DEFAULT_INT_PARAMS``.
    (remaining kwargs)
        Fixed settings forwarded to every ``run_zombi_test`` call.

    Returns
    -------
    dict with keys:
        ``X`` : np.ndarray (n, d) — evaluated configs in unit hypercube
        ``Y`` : np.ndarray (n, 2) — [avg_dist, dup_frac] per evaluation
        ``configs`` : list of dict — concrete hyperparameter dicts
        ``pareto_mask`` : np.ndarray (n,) bool — Pareto-optimal rows
        ``pareto_configs`` : list of dict — Pareto-optimal configs
        ``pareto_Y`` : np.ndarray (k, 2) — Pareto-optimal objectives
        ``param_names`` : list of str — parameter names (column order for X)
    """
    import concurrent.futures  # noqa: F401  (imported here to keep top-level clean)
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

    # Round n_initial up to a full multiple of n_parallel
    n_initial = int(np.ceil(n_initial / n_parallel)) * n_parallel

    # Fixed kwargs forwarded verbatim to every run_zombi_test call.
    # torch.dtype is picklable, so this dict can cross process boundaries.
    fixed_kw: Dict = dict(
        csv_path=csv_path,
        max_activations=max_activations,
        num_init_data=num_init_data,
        max_zooms=max_zooms,
        max_iterations=max_iterations,
        n_restarts=n_restarts,
        raw_samples=raw_samples,
        top_m_points=top_m_points,
        n_consecutive_converged=n_consecutive_converged,
        max_gp_points=max_gp_points,
        repulsion_lambda=repulsion_lambda,
        rf_global_samples=rf_global_samples,
        rf_cache_dir=rf_cache_dir,
        device=device,
        dtype=dtype,
        verbose=verbose_zombi,
        show_plot=show_plot,
    )

    # ── Incremental JSON writer ──────────────────────────────────────────────
    def _flush_json(
        evaluations: List[Dict],
        configs: List[Dict],
        Y_list: List[torch.Tensor],
        path: str,
        *,
        pareto_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Write all evaluations, best_so_far, and (optionally) Pareto front."""
        records = []
        for i, (cfg, y_t) in enumerate(zip(configs, Y_list)):
            avg_dist = float(y_t[0])
            dup_frac = float(y_t[1])
            records.append({
                "eval_idx": i,
                "phase": evaluations[i]["phase"],
                "config": cfg,
                "avg_dist": avg_dist,
                "dup_frac": dup_frac,
                "score": float(np.sqrt(avg_dist ** 2 + dup_frac ** 2)),
            })

        # best_so_far: minimum Euclidean distance to the ideal point (0, 0)
        best_idx = int(np.argmin([r["score"] for r in records]))
        best_so_far = {**records[best_idx]}

        out: Dict = {
            "evaluations": records,
            "best_so_far": best_so_far,
        }
        if pareto_mask is not None:
            out["pareto_front"] = [r for r, m in zip(records, pareto_mask) if m]

        with open(path, "w") as fh:
            json.dump(out, fh, indent=2, cls=_NumpyEncoder)

    # ── Evaluation phase tracker (parallel to all_configs / all_Y) ──────────
    all_phases: List[Dict] = []

    # Unit-cube BoTorch bounds: [0, 1]^d
    bo_bounds = torch.zeros(2, d, dtype=torch.float64)
    bo_bounds[1] = 1.0

    # Worst-case reference point (negated objectives, since BoTorch maximises)
    ref_point = torch.tensor([-1.5, -1.5], dtype=torch.float64)

    n_total_evals = n_initial + n_mobo_iterations * n_parallel
    print("=" * 70)
    print(f"MOBO TUNING (high-throughput)")
    print(f"  {n_initial} Sobol init + {n_mobo_iterations} rounds × {n_parallel} parallel "
          f"= {n_total_evals} total evals")
    print(f"  {n_parallel} workers, {threads_per_worker} threads/worker")
    print(f"  Tunable params ({d}): {param_names}")
    print("=" * 70)

    torch.manual_seed(seed)

    # Draw all Sobol init points at once (n_initial, d)
    X_sobol = draw_sobol_samples(
        bounds=bo_bounds, n=n_initial, q=1, seed=seed,
    ).squeeze(1)

    all_X: List[torch.Tensor] = []
    all_Y: List[torch.Tensor] = []
    all_configs: List[Dict] = []

    # ── Resume: load any previously saved evaluations from results_json ─────────
    n_sobol_already_done = 0
    n_mobo_rounds_already_done = 0
    if os.path.isfile(results_json):
        try:
            with open(results_json) as fh:
                saved = json.load(fh)
            prev_evals = saved.get("evaluations", [])
            if prev_evals:
                for rec in prev_evals:
                    # Reconstruct unit-cube X from config via inverse mapping
                    cfg = rec["config"]
                    x_unit = np.array([
                        _unit_from_config(cfg[name], name, raw_bounds,
                                          log_params)
                        for name in param_names
                    ], dtype=np.float64)
                    all_X.append(torch.tensor(x_unit, dtype=torch.float64))
                    all_Y.append(torch.tensor(
                        [rec["avg_dist"], rec["dup_frac"]], dtype=torch.float64
                    ))
                    all_configs.append(cfg)
                    all_phases.append({"phase": rec["phase"]})
                    if rec["phase"] == "sobol_init":
                        n_sobol_already_done += 1
                    elif rec["phase"].startswith("mobo_round_"):
                        rnd = int(rec["phase"].split("_")[-1])
                        n_mobo_rounds_already_done = max(
                            n_mobo_rounds_already_done, rnd
                        )
                print(
                    f"  Resumed {len(prev_evals)} evaluations from {results_json} "
                    f"({n_sobol_already_done} Sobol, "
                    f"{n_mobo_rounds_already_done} MOBO rounds done)."
                )
        except Exception as exc:
            print(f"  [warn] Could not load {results_json}: {exc}. Starting fresh.")

    # ── Phase 1: Sobol init — evaluated in parallel batches of n_parallel ──────
    for batch_start in range(0, n_initial, n_parallel):
        # Skip batches already covered by resumed data
        if batch_start + n_parallel <= n_sobol_already_done:
            print(f"  [resume] Skipping Sobol batch {batch_start+1}-"
                  f"{batch_start+n_parallel} (already done).")
            continue

        batch_idx = X_sobol[batch_start : batch_start + n_parallel]
        batch_cfgs = [
            _config_from_unit(batch_idx[j].numpy(), param_names,
                              raw_bounds, log_params, int_params)
            for j in range(len(batch_idx))
        ]
        label = f"Init {batch_start+1}-{batch_start+len(batch_idx)}/{n_initial}"
        print(f"\n{'─' * 60}\n{label}")

        new_X, new_Y, new_cfgs = _run_batch_parallel(
            batch_cfgs,
            [batch_idx[j].numpy() for j in range(len(batch_idx))],
            fixed_kw, n_parallel, threads_per_worker, label,
        )
        all_X.extend(new_X)
        all_Y.extend(new_Y)
        all_configs.extend(new_cfgs)
        all_phases.extend([{"phase": "sobol_init"}] * len(new_cfgs))
        _flush_json(all_phases, all_configs, all_Y, results_json)

    # ── Phase 2: MOBO loop — each round proposes q=n_parallel candidates ───────
    for it in range(n_mobo_iterations):
        # Skip rounds already covered by resumed data
        if it + 1 <= n_mobo_rounds_already_done:
            print(f"  [resume] Skipping MOBO round {it+1} (already done).")
            continue

        print(f"\n{'─' * 60}")
        print(f"MOBO round {it+1}/{n_mobo_iterations}")
        print(f"{'─' * 60}")

        train_X = torch.stack(all_X)       # (n, d)
        train_Y = -torch.stack(all_Y)      # (n, 2) — negated for maximisation

        # Fit one independent GP per objective
        models = []
        for obj_idx in range(2):
            gp = SingleTaskGP(train_X, train_Y[:, obj_idx : obj_idx + 1])
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            models.append(gp)
        model = ModelListGP(*models)

        # Batch qNEHVI: propose n_parallel candidates simultaneously
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
            _config_from_unit(candidates[j].detach().numpy(), param_names,
                              raw_bounds, log_params, int_params)
            for j in range(n_parallel)
        ]
        label = f"Round {it+1}"
        new_X, new_Y, new_cfgs = _run_batch_parallel(
            batch_cfgs,
            [candidates[j].detach().numpy() for j in range(n_parallel)],
            fixed_kw, n_parallel, threads_per_worker, label,
        )
        all_X.extend(new_X)
        all_Y.extend(new_Y)
        all_configs.extend(new_cfgs)
        all_phases.extend([{"phase": f"mobo_round_{it+1}"}] * len(new_cfgs))
        _flush_json(all_phases, all_configs, all_Y, results_json)

    # ── Assemble and report ─────────────────────────────────────────────────────
    Y_np = torch.stack(all_Y).numpy()
    X_np = torch.stack(all_X).numpy()
    pareto_mask = is_non_dominated(-torch.stack(all_Y)).numpy()

    # Final flush with Pareto info
    _flush_json(all_phases, all_configs, all_Y, results_json, pareto_mask=pareto_mask)
    print(f"\nHyperparameter results saved → {os.path.abspath(results_json)}")

    print("\n" + "=" * 70)
    print("MOBO TUNING COMPLETE — Pareto-optimal configurations:")
    print("=" * 70)
    header = f"  {'avg_dist':>10}  {'dup_frac':>10}  config"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for idx in np.where(pareto_mask)[0]:
        print(f"  {Y_np[idx, 0]:>10.5f}  {Y_np[idx, 1]:>10.4f}  {all_configs[idx]}")
    print("=" * 70)

    return {
        "X": X_np,
        "Y": Y_np,
        "configs": all_configs,
        "pareto_mask": pareto_mask,
        "pareto_configs": [c for c, m in zip(all_configs, pareto_mask) if m],
        "pareto_Y": Y_np[pareto_mask],
        "param_names": param_names,
    }


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    # ProcessPoolExecutor requires the __main__ guard on macOS / Windows
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mobo", action="store_true",
                    help="Run MOBO hyperparameter tuning instead of a single test.")
    ap.add_argument("--mobo-init",    type=int, default=8,
                    help="Sobol init evals (rounded up to multiple of --parallel).")
    ap.add_argument("--mobo-iters",   type=int, default=20,
                    help="MOBO acquisition rounds.")
    ap.add_argument("--parallel",     type=int, default=4,
                    help="Number of parallel worker processes per round.")
    ap.add_argument("--threads",      type=int, default=2,
                    help="CPU threads per worker process.")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip objective visualisation for single-run mode.")
    ap.add_argument("--plot", action="store_true",
                    help="Enable objective visualisation in MOBO mode (off by default).")
    ap.add_argument("--results-json", type=str, default="hyperparam_results.json",
                    help="Path to write incremental MOBO results JSON.")
    args = ap.parse_args()

    if args.mobo:
        mobo_result = mobo_tune_zombi(
            n_initial=args.mobo_init,
            n_mobo_iterations=args.mobo_iters,
            n_parallel=args.parallel,
            threads_per_worker=args.threads,
            csv_path=CSV_PEROVSKITE_PATH,
            max_activations=3,
            device="cpu",
            verbose_zombi=False,
            show_plot=args.plot,   # off by default in MOBO; use --plot to enable
            results_json=args.results_json,
        )
    else:
        results = run_zombi_test(
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
            max_activations=5,
            num_init_data=4,
            rf_global_samples=10_000_000,
            csv_path=CSV_PEROVSKITE_PATH,
            device="cpu",
            verbose=True,
            show_plot=not args.no_plot,
        )
