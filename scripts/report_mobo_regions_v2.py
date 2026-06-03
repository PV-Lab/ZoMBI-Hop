#!/usr/bin/env python3
"""
report_mobo_regions_v2.py
=========================

Build the same *objectives_to_run* region / seed / L choices that
``run_zombi_test_v2.run_zombi_test_v2`` and MOBO evaluation use, then plot them
with the **same ternary graphing logic** as ``interactive_maxima_selector``
(dense Dirichlet grid, ``viridis`` scatter, red / royalblue region overlays,
star seeds, colorbar).  Prints a **top-L optima** table to stdout (rank, value,
barycentric xyz) and labels each ★ on the figure with its rank.

Example
-------
    python scripts/report_mobo_regions_v2.py \\
        --regions max_min_regions.json \\
        --save mobo_region_plan_v2.png

    # Browse one panel at a time (Prev/Next, ←/→ / a/d — like interactive_maxima_selector):
    python scripts/report_mobo_regions_v2.py --regions max_min_regions.json --interactive
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RF_CACHE_DIR = os.path.join(_REPO_ROOT, "test_rfs")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.run_zombi_test import (  # noqa: E402
    CSV_ELEMENT_TRIPLE,
    CSV_OBJECTIVES,
    CSV_PEROVSKITE_PATH,
    RF_CACHE_DIR as _V1_RF_CACHE_DIR,
    _ACKLEY_B,
    _ACKLEY_B_SKINNY,
    ackley_equal,
    ackley_edge,
    ackley_vertex,
    multimodal_ackley,
    ACKLEY_CENTER_EQUAL,
    ACKLEY_CENTER_EDGE,
    ACKLEY_CENTER_VERTEX,
    MULTIMODAL_CENTERS,
    build_csv_rf_objectives,
    _NumpyEncoder,
)
from scripts.run_zombi_test_v2 import load_regions, _get_top_seeds  # noqa: E402
from scripts.interactive_maxima_selector import (  # noqa: E402
    N_GRID,
    _ackley_vec,
    _draw_triangle,
    _to_cart,
    make_grid,
)


_TOP_L_MAX = 10
_NOISE_COMBOS: List[Tuple[float, float]] = [
    (0.01, 0.01),
    (0.01, 0.001),
    (0.001, 0.01),
    (0.001, 0.001),
]


def _barycentric_region_cloud(
    regions_data: Optional[Dict],
    region_key: str,
    mode: str,
) -> np.ndarray:
    """
    Points (n, 3) saved in the regions JSON for shading / context, mirroring
    ``compute_region_avg_dist`` (region patch, else seeds as fallback).
    """
    if regions_data is None:
        return np.empty((0, 3))
    entry = regions_data.get("objectives", {}).get(region_key)
    if entry is None:
        return np.empty((0, 3))
    rc = entry.get(f"{mode}_region_coords", []) or []
    if not rc:
        rc = entry.get(f"{mode}_seeds", []) or []
    if not rc:
        return np.empty((0, 3))
    return np.asarray(rc, dtype=np.float64)


def _sorted_top_l_pairs_from_json(
    regions_data: Optional[Dict],
    region_key: str,
    mode: str,
    L_max: int = _TOP_L_MAX,
) -> List[Tuple[float, np.ndarray]]:
    """
    Same ordering as ``run_zombi_test_v2._get_top_seeds``:
    sort by objective value (best first), keep at most *L_max* pairs.
    """
    if regions_data is None:
        return []
    entry = regions_data.get("objectives", {}).get(region_key)
    if entry is None:
        return []
    seeds = entry.get(f"{mode}_seeds", [])
    vals = entry.get(f"{mode}_seed_vals", [])
    if not seeds:
        return []
    descending = mode == "max"
    pairs = sorted(zip(vals, seeds), reverse=descending)[:L_max]
    return [(float(v), np.asarray(coord, dtype=np.float64)) for v, coord in pairs]


def _make_top_l_optima(
    *,
    regions_data: Optional[Dict],
    region_key: str,
    mode: str,
    top_seeds: List[np.ndarray],
    seed_source: str,
    value_fn: Optional[Callable[[np.ndarray], float]],
) -> List[Dict[str, Any]]:
    """
    Ranked list of the *L* targets (1 … L) with barycentric coords and scalar
    *value* when available (stored seed vals / RF pred / Ackley eval).
    """
    pj = _sorted_top_l_pairs_from_json(regions_data, region_key, mode)
    if seed_source == "from_regions_json" and pj:
        return [
            {"rank": i + 1, "barycentric": c.tolist(), "value": v}
            for i, (v, c) in enumerate(pj)
        ]
    if not top_seeds or value_fn is None:
        return []
    out: List[Dict[str, Any]] = []
    for i, coord in enumerate(top_seeds):
        c = np.asarray(coord, dtype=np.float64).ravel()
        out.append({
            "rank": i + 1,
            "barycentric": c.tolist(),
            "value": float(value_fn(c)),
        })
    return out


def print_top_l_optima_table(report: Dict[str, Any]) -> None:
    """Stdout table: ranked top-L barycentric targets per benchmark row."""
    print("\n" + "=" * 88)
    print(
        "Top-L optima per row (rank 1 = most extreme in JSON sort; "
        f"L ≤ {_TOP_L_MAX}; matches run_zombi_test_v2)"
    )
    print("=" * 88)
    for r in report["objective_rows"]:
        hdr = f"{r['region_key']}  [{r['mode']}]  L={r['L']}  ({r['seed_source']})"
        print(f"\n{hdr}")
        print("-" * len(hdr))
        ops = r.get("top_L_optima") or []
        if not ops:
            print("  (no target points — L=1 empty-extrema case or missing data)")
            continue
        for op in ops:
            v = op["value"]
            vs = f"{v:.8g}" if v is not None else "n/a"
            x, y, z = op["barycentric"]
            print(
                f"  rank {op['rank']:>2}  value={vs:>16}  "
                f"x={x:.6f}  y={y:.6f}  z={z:.6f}"
            )
    print()


def _summarize_json_objective(regions_data: Optional[Dict], region_key: str) -> Optional[Dict[str, Any]]:
    if regions_data is None:
        return None
    entry = regions_data.get("objectives", {}).get(region_key)
    if entry is None:
        return {"present": False, "region_key": region_key}
    def _n_coords(key: str) -> int:
        v = entry.get(key)
        return len(v) if isinstance(v, list) else 0
    return {
        "present": True,
        "region_key": region_key,
        "epsilon_abs_stored": entry.get("epsilon_abs"),
        "val_range": entry.get("val_range"),
        "n_max_seeds": _n_coords("max_seeds"),
        "n_min_seeds": _n_coords("min_seeds"),
        "n_max_region_points": _n_coords("max_region_coords"),
        "n_min_region_points": _n_coords("min_region_coords"),
    }


# Styling copied from ``MaximaSelector._redraw_current`` (interactive_maxima_selector)
_REGION_STYLE = {
    "max": ("red", 0.30),
    "min": ("royalblue", 0.30),
}
_SEED_STYLE = {
    "max": ("red", "darkred"),
    "min": ("royalblue", "navy"),
}


def build_ternary_objectives_like_interactive(
    *,
    csv_path: str,
    rf_cache_dir: Optional[str],
    rf_global_samples: int,
    n_pts: int,
) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
    """
    Same grid + RF / Ackley evaluation pipeline as
    ``interactive_maxima_selector.main`` (for static plotting only).
    """
    grid = make_grid(n_pts)
    cart = _to_cart(grid)
    objectives: List[Dict[str, Any]] = []

    if os.path.isfile(csv_path):
        try:
            csv_rf = build_csv_rf_objectives(
                csv_path=csv_path,
                objectives=CSV_OBJECTIVES,
                rf_global_samples=rf_global_samples,
                cache_dir=rf_cache_dir,
            )
            for name, data in csv_rf.items():
                rf = data["rf"]
                vals = rf.predict(grid).astype(np.float64)
                objectives.append({
                    "name": f"RF-{name}",
                    "vals": vals,
                    "labels": CSV_ELEMENT_TRIPLE,
                })
        except Exception as exc:
            warnings.warn(f"CSV RF build failed: {exc}", stacklevel=2)
    else:
        warnings.warn(f"CSV not found: {csv_path}", stacklevel=2)

    ackley_defs = [
        ("Ackley-Centroid", ACKLEY_CENTER_EQUAL, _ACKLEY_B),
        ("Ackley-Edge", ACKLEY_CENTER_EDGE, _ACKLEY_B),
        ("Ackley-Vertex", ACKLEY_CENTER_VERTEX, _ACKLEY_B),
    ]
    for aname, center, b in ackley_defs:
        vals = _ackley_vec(grid, center, b=b)
        objectives.append({
            "name": aname,
            "vals": vals.astype(np.float64),
            "labels": ("A", "B", "C"),
        })

    mm_vals = sum(
        _ackley_vec(grid, c, b=_ACKLEY_B_SKINNY) for c in MULTIMODAL_CENTERS
    )
    objectives.append({
        "name": "Ackley-Multi-modal",
        "vals": mm_vals.astype(np.float64),
        "labels": ("A", "B", "C"),
    })

    by_name = {o["name"]: o for o in objectives}
    return cart, by_name


def build_report(
    *,
    regions_path: Optional[str],
    csv_path: str,
    rf_cache_dir: Optional[str],
    rf_global_samples: int,
) -> Dict[str, Any]:
    # Same as V2: missing path or missing file → None (see run_zombi_test_v2.load_regions)
    regions_data = load_regions(regions_path)

    csv_rf_objectives: Dict = {}
    csv_error: Optional[str] = None
    if os.path.isfile(csv_path):
        try:
            csv_rf_objectives = build_csv_rf_objectives(
                csv_path=csv_path,
                objectives=CSV_OBJECTIVES,
                rf_global_samples=rf_global_samples,
                cache_dir=rf_cache_dir,
            )
        except Exception as exc:
            csv_error = str(exc)
    else:
        csv_error = f"CSV not found: {csv_path}"

    ackley_fns: List[Tuple[str, Any, List[np.ndarray], Optional[np.ndarray]]] = [
        ("Ackley-Centroid", ackley_equal, [ACKLEY_CENTER_EQUAL], None),
        ("Ackley-Edge", ackley_edge, [ACKLEY_CENTER_EDGE], None),
        ("Ackley-Vertex", ackley_vertex, [ACKLEY_CENTER_VERTEX], None),
        ("Ackley-Multi-modal", multimodal_ackley, MULTIMODAL_CENTERS, None),
    ]

    rows: List[Dict[str, Any]] = []

    # ── CSV RF objectives (same loop as run_zombi_test_v2) ─────────────────
    for obj_col, data in csv_rf_objectives.items():
        base_name = f"CSV-RF-{obj_col} ({'/'.join(CSV_ELEMENT_TRIPLE)})"
        region_key = f"RF-{obj_col}"

        for mode in ("max", "min"):
            top_seeds, L_from_json = _get_top_seeds(regions_data, region_key, mode)
            seed_source: str
            if top_seeds:
                seed_source = "from_regions_json"
                L = L_from_json
            elif mode == "max":
                top_seeds = [data["global_max_x"]]
                L = max(len(top_seeds), 1)
                seed_source = "from_rf_global_max"
            else:
                gmin = data.get("global_min_x")
                top_seeds = [gmin] if gmin is not None else []
                L = max(len(top_seeds), 1)
                seed_source = "from_rf_global_min" if gmin is not None else "from_rf_global_min_missing"

            cloud = _barycentric_region_cloud(regions_data, region_key, mode)

            def _rf_val(c: np.ndarray) -> float:
                return float(data["rf"].predict(c.reshape(1, -1))[0])

            top_L_optima = _make_top_l_optima(
                regions_data=regions_data,
                region_key=region_key,
                mode=mode,
                top_seeds=top_seeds,
                seed_source=seed_source,
                value_fn=_rf_val,
            )
            rows.append({
                "kind": "csv_rf",
                "base_name": base_name,
                "region_key": region_key,
                "mode": mode,
                "L": L,
                "max_activations": int(math.ceil(1.2 * L)),
                "seed_source": seed_source,
                "known_extrema": [s.tolist() for s in top_seeds],
                "top_L_optima": top_L_optima,
                "region_plot_coords": cloud.tolist(),
                "regions_json_summary": _summarize_json_objective(regions_data, region_key),
            })

    # ── Ackley objectives ─────────────────────────────────────────────────
    for aname, afn, analytic_max, _ in ackley_fns:
        region_key = aname

        for mode in ("max", "min"):
            top_seeds, L_from_json = _get_top_seeds(regions_data, region_key, mode)
            seed_source: str
            if top_seeds:
                seed_source = "from_regions_json"
                L = L_from_json
            elif mode == "max":
                top_seeds = [np.asarray(km) for km in analytic_max]
                L = max(len(top_seeds), 1)
                seed_source = "from_analytic_maxima"
            else:
                top_seeds = []
                L = max(len(top_seeds), 1)
                seed_source = "ackley_min_no_analytic_fallback"

            cloud = _barycentric_region_cloud(regions_data, region_key, mode)

            top_L_optima = _make_top_l_optima(
                regions_data=regions_data,
                region_key=region_key,
                mode=mode,
                top_seeds=top_seeds,
                seed_source=seed_source,
                value_fn=lambda c, f=afn: float(f(c)),
            )
            rows.append({
                "kind": "ackley",
                "base_name": aname,
                "region_key": region_key,
                "mode": mode,
                "L": L,
                "max_activations": int(math.ceil(1.2 * L)),
                "seed_source": seed_source,
                "known_extrema": [s.tolist() for s in top_seeds],
                "top_L_optima": top_L_optima,
                "region_plot_coords": cloud.tolist(),
                "regions_json_summary": _summarize_json_objective(regions_data, region_key),
            })

    n_sub_runs = len(rows) * len(_NOISE_COMBOS)

    return {
        "regions_path": os.path.abspath(regions_path) if regions_path and os.path.isfile(regions_path) else regions_path,
        "regions_loaded": regions_data is not None,
        "csv_path": os.path.abspath(csv_path) if os.path.isfile(csv_path) else csv_path,
        "csv_rf_columns_built": list(csv_rf_objectives.keys()),
        "csv_build_error": csv_error,
        "rf_cache_dir": rf_cache_dir,
        "rf_global_samples": rf_global_samples,
        "top_L_cap": _TOP_L_MAX,
        "noise_combos": [list(p) for p in _NOISE_COMBOS],
        "n_objective_modes": len(rows),
        "n_sub_runs_per_mobo_eval": n_sub_runs,
        "objective_rows": rows,
    }


def _draw_mobo_row_on_axes(
    ax,
    r: Dict[str, Any],
    cart: np.ndarray,
    objectives_by_name: Dict[str, Dict[str, Any]],
    *,
    ax_cbar: Optional[Any] = None,
    title_fontsize: int = 7,
    cbar_high_low: bool = False,
) -> None:
    """
    Draw one MOBO benchmark row on *ax* (ternary + viridis + region + ranked ★).
    If *ax_cbar* is given, put the colorbar there; else attach a thin colorbar to *ax*.
    """
    ax.cla()
    if ax_cbar is not None:
        ax_cbar.cla()

    mode = r["mode"]
    assert mode in ("max", "min")
    obj = objectives_by_name.get(r["region_key"])
    sc: Optional[Any] = None
    vmin_val: Optional[float] = None
    vmax_val: Optional[float] = None

    if obj is not None:
        vals = obj["vals"]
        vmin_val = float(vals.min())
        vmax_val = float(vals.max())
        sc = ax.scatter(
            cart[:, 0], cart[:, 1], c=vals, cmap="viridis",
            s=6, alpha=0.55, linewidths=0, zorder=1,
            vmin=vmin_val, vmax=vmax_val,
        )
        labels = obj["labels"]
    else:
        ax.text(
            0.5, 0.55, "no grid objective\n(RF / data missing)",
            ha="center", va="center", fontsize=8, transform=ax.transAxes,
        )
        labels = (
            tuple(CSV_ELEMENT_TRIPLE)
            if r["kind"] == "csv_rf" else ("A", "B", "C")
        )

    if sc is not None:
        if ax_cbar is not None:
            plt.colorbar(sc, cax=ax_cbar)
            ax_cbar.tick_params(labelsize=8)
            if cbar_high_low and vmin_val is not None and vmax_val is not None:
                ax_cbar.text(
                    0.5, 1.035, f"▲ HIGH\n{vmax_val:.4g}",
                    transform=ax_cbar.transAxes,
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color="#2ca02c",
                )
                ax_cbar.text(
                    0.5, -0.035, f"{vmin_val:.4g}\nLOW ▼",
                    transform=ax_cbar.transAxes,
                    ha="center", va="top", fontsize=8, fontweight="bold",
                    color="#440154",
                )
        else:
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

    colour, alpha_r = _REGION_STYLE[mode]
    cloud = np.asarray(r.get("region_plot_coords") or [], dtype=np.float64)
    if cloud.shape[0] > 0:
        cc = _to_cart(cloud)
        ax.scatter(
            cc[:, 0], cc[:, 1], color=colour,
            alpha=alpha_r, s=8, linewidths=0, zorder=2,
        )

    face, edge = _SEED_STYLE[mode]
    optima = r.get("top_L_optima") or []
    if optima:
        for op in optima:
            sc_coord = np.asarray(op["barycentric"], dtype=np.float64).reshape(1, 3)
            sp = _to_cart(sc_coord)
            ax.scatter(
                sp[:, 0], sp[:, 1], marker="*", color=face,
                s=400, zorder=6, edgecolors=edge, linewidths=0.7,
            )
            ax.annotate(
                str(op["rank"]),
                xy=(float(sp[0, 0]), float(sp[0, 1])),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="white",
                zorder=8,
                bbox=dict(
                    boxstyle="round,pad=0.12",
                    facecolor="0.12",
                    edgecolor="0.35",
                    alpha=0.92,
                ),
            )
    else:
        for coord_list in r.get("known_extrema") or []:
            sc_coord = np.asarray(coord_list, dtype=np.float64).reshape(1, 3)
            sp = _to_cart(sc_coord)
            ax.scatter(
                sp[:, 0], sp[:, 1], marker="*", color=face,
                s=400, zorder=6, edgecolors=edge, linewidths=0.7,
            )

    _draw_triangle(ax, labels=labels)

    mode_colour = "darkred" if mode == "max" else "navy"
    rk = str(r["region_key"])
    show_rk = rk if len(rk) <= 40 else rk[:37] + "…"
    ax.set_title(
        f"{show_rk}  [{mode}]\n"
        f"L={r['L']}  act={r['max_activations']}  |  {r['seed_source']}",
        fontsize=title_fontsize,
        color=mode_colour,
        fontweight="bold",
    )


class MoboRegionPlanViewer:
    """
    Single-panel navigator (Prev / Next) matching the layout of
    ``interactive_maxima_selector.MaximaSelector`` — read-only, no seed editing.
    """

    def __init__(
        self,
        report: Dict[str, Any],
        cart: np.ndarray,
        objectives_by_name: Dict[str, Dict[str, Any]],
    ) -> None:
        self.rows: List[Dict[str, Any]] = report["objective_rows"]
        self.cart = cart
        self.objectives_by_name = objectives_by_name
        if not self.rows:
            raise ValueError("No objective rows to browse.")

        self.current_idx = 0

        self.fig = plt.figure(figsize=(9, 10))

        self.ax_main = self.fig.add_axes([0.08, 0.28, 0.78, 0.64])
        self.ax_cbar = self.fig.add_axes([0.88, 0.28, 0.025, 0.64])

        ax_prev = self.fig.add_axes([0.08, 0.205, 0.10, 0.045])
        ax_next = self.fig.add_axes([0.82, 0.205, 0.10, 0.045])
        ax_nav_lbl = self.fig.add_axes([0.19, 0.205, 0.62, 0.045])
        ax_nav_lbl.axis("off")

        self.btn_prev = Button(ax_prev, "◄  Prev")
        self.btn_next = Button(ax_next, "Next  ►")
        self.nav_txt = ax_nav_lbl.text(
            0.5, 0.5, "",
            ha="center", va="center", fontsize=10, fontweight="bold",
            transform=ax_nav_lbl.transAxes,
        )

        ax_strip = self.fig.add_axes([0.06, 0.158, 0.88, 0.042])
        ax_strip.axis("off")
        self.strip_txt = ax_strip.text(
            0.5, 0.5, "",
            ha="center", va="center", fontsize=7.5,
            transform=ax_strip.transAxes,
            family="monospace",
        )

        ax_status = self.fig.add_axes([0.00, 0.010, 1.00, 0.036])
        ax_status.axis("off")
        self.status_txt = ax_status.text(
            0.5, 0.5, "",
            ha="center", va="center", fontsize=8,
            transform=ax_status.transAxes,
        )

        self.btn_prev.on_clicked(self._prev)
        self.btn_next.on_clicked(self._next)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.fig.suptitle(
            "MOBO V2 region plan — browse targets (read-only)",
            fontsize=11,
            fontweight="bold",
            y=0.98,
        )
        self._goto(0)

    def _prev(self, _event: Any) -> None:
        self._goto((self.current_idx - 1) % len(self.rows))

    def _next(self, _event: Any) -> None:
        self._goto((self.current_idx + 1) % len(self.rows))

    def _goto(self, i: int) -> None:
        self.current_idx = i
        r = self.rows[i]
        _draw_mobo_row_on_axes(
            self.ax_main,
            r,
            self.cart,
            self.objectives_by_name,
            ax_cbar=self.ax_cbar,
            title_fontsize=10,
            cbar_high_low=True,
        )
        name = r["region_key"]
        self.nav_txt.set_text(
            f"[{i + 1} / {len(self.rows)}]   {name}   ({r['mode']})   "
            f"L={r['L']}   act={r['max_activations']}"
        )
        self._refresh_strip()
        self.status_txt.set_text(
            "← or a = previous  |  → or d = next  (same as interactive_maxima_selector)  |  q = close"
        )
        self.fig.canvas.draw_idle()

    def _refresh_strip(self) -> None:
        parts: List[str] = []
        for j, row in enumerate(self.rows):
            short = (
                row["region_key"]
                .replace("Ackley-", "A-")
                .replace("RF-", "")
            )
            if len(short) > 14:
                short = short[:11] + "…"
            if j == self.current_idx:
                parts.append(f"[{short}:{row['mode'][0]}]")
            else:
                parts.append(f" {short}:{row['mode'][0]} ")
        self.strip_txt.set_text("  ".join(parts))

    def _on_key(self, event: Any) -> None:
        key = event.key
        if key in ("right", "d"):
            self._next(None)
        elif key in ("left", "a"):
            self._prev(None)
        elif key == "q":
            plt.close(self.fig)

    def show(self) -> None:
        plt.show()


def plot_mobo_region_plan(
    report: Dict[str, Any],
    *,
    cart: np.ndarray,
    objectives_by_name: Dict[str, Dict[str, Any]],
    save_path: str,
    show: bool = False,
    do_save: bool = True,
) -> None:
    """
    One ternary panel per MOBO row. Drawing matches
    ``MaximaSelector._redraw_current`` in interactive_maxima_selector.py:
    viridis heatmap on the shared grid, then region overlay, then seeds.
    """
    rows: List[Dict[str, Any]] = report["objective_rows"]
    if not rows:
        print("No objective rows to plot.")
        return

    n = len(rows)
    ncols = 4
    nrows_p = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows_p, ncols, figsize=(4.2 * ncols, 3.7 * nrows_p))
    axes_flat = np.atleast_1d(axes).ravel()

    for i, r in enumerate(rows):
        _draw_mobo_row_on_axes(
            axes_flat[i], r, cart, objectives_by_name,
            ax_cbar=None, title_fontsize=7, cbar_high_low=False,
        )

    for j in range(len(rows), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "MOBO V2 region plan — plot style: interactive_maxima_selector "
        "(viridis field, red=max / blue=min overlays, ★ seeds)",
        fontsize=10,
        fontweight="bold",
        y=1.002,
    )
    plt.tight_layout()
    if do_save:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        print(f"Figure saved → {os.path.abspath(save_path)}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot region targets used by run_zombi_test_v2 / MOBO (ternary panels).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--regions", type=str, default=None,
                    help="Path to max_min_regions.json (same as --regions on run_zombi_test_v2.py).")
    ap.add_argument("--csv", dest="csv_path", type=str, default=CSV_PEROVSKITE_PATH)
    ap.add_argument("--rf-cache", type=str, default=_V1_RF_CACHE_DIR)
    ap.add_argument("--rf-global-samples", type=int, default=10_000_000)
    ap.add_argument("--n-pts", type=int, default=N_GRID,
                    help="Dense simplex grid size (same as interactive --n-pts).")
    ap.add_argument("--save", type=str, default="mobo_region_plan_v2.png",
                    help="Output image path (PNG).")
    ap.add_argument("--show", action="store_true",
                    help="Also display the static multi-panel figure (non-blocking depends on backend).")
    ap.add_argument("--interactive", action="store_true",
                    help="Single-panel navigator (◄ ► buttons; ←/→ or a/d keys), "
                         "same layout style as interactive_maxima_selector.py.")
    ap.add_argument("--no-static", action="store_true",
                    help="Skip writing the multi-panel PNG.")
    ap.add_argument("--json-out", type=str, default=None,
                    help="Optional path to dump the same data as JSON.")
    args = ap.parse_args()

    report = build_report(
        regions_path=args.regions,
        csv_path=args.csv_path,
        rf_cache_dir=args.rf_cache,
        rf_global_samples=args.rf_global_samples,
    )
    print_top_l_optima_table(report)

    cart, objectives_by_name = build_ternary_objectives_like_interactive(
        csv_path=args.csv_path,
        rf_cache_dir=args.rf_cache,
        rf_global_samples=args.rf_global_samples,
        n_pts=args.n_pts,
    )
    want_static_file = not args.no_static
    want_static_show = args.show and not args.interactive

    if want_static_file or want_static_show:
        plot_mobo_region_plan(
            report,
            cart=cart,
            objectives_by_name=objectives_by_name,
            save_path=args.save,
            show=want_static_show,
            do_save=want_static_file,
        )

    if args.interactive:
        try:
            MoboRegionPlanViewer(report, cart, objectives_by_name).show()
        except ValueError as exc:
            print(f"Interactive viewer: {exc}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2, cls=_NumpyEncoder)
        print(f"JSON also written → {os.path.abspath(args.json_out)}")

    print(f"  regions_loaded={report['regions_loaded']}  "
          f"objective_rows={report['n_objective_modes']}  "
          f"sub_runs_per_MOBO_eval={report['n_sub_runs_per_mobo_eval']}")
    if report.get("csv_build_error") and not report["csv_rf_columns_built"]:
        print(f"  [warn] CSV: {report['csv_build_error']}")


if __name__ == "__main__":
    main()
