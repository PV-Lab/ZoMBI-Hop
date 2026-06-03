"""
interactive_maxima_selector.py
==============================

Interactive ternary tool for manually labelling "maximum regions" and
"minimum regions" on every ZoMBI-Hop benchmark objective.

How it works
------------
1. A dense grid of simplex points is sampled once at startup.
2. Each objective is evaluated on the grid and displayed as a ternary
   scatter plot.
3. You click on a panel to plant a "seed" point.  The tool flood-fills
   the connected component of the level set passing through that seed
   (all grid points reachable via the KNN graph whose function value
   lies within ε of the seed's value).  That connected blob is the
   "region".
4. Seeds and computed regions are saved to a JSON file for later use
   in validation (any needle inside a region counts as distance-0).

Controls
--------
  Left-click  on plot     →  add seed in current mode (max / min)
  Right-click on plot     →  remove the nearest seed in current mode
  ◄ / ► buttons           →  navigate between objectives
  ε slider                →  adjust level-set width (fraction of value range)
  Undo button             →  remove the most recently added seed (current obj)
  Clear button            →  clear seeds for the current objective only
  Clear All button        →  wipe every seed and region across all objectives
  Save button             →  write regions.json (or --save-path)

Usage
-----
    python scripts/interactive_maxima_selector.py [options]

    --epsilon    FLOAT   initial ε as fraction of value range  [0.05]
    --n-pts      INT     dense grid size                        [80000]
    --save-path  PATH    output JSON path                       [max_min_regions.json]
    --csv-path   PATH    perovskite CSV (auto-detected by default)
    --rf-cache   PATH    RF model cache dir                     [test_rfs/]
    --rf-samples INT     Dirichlet samples for RF global-max    [10000000]
    --no-ackley          skip synthetic Ackley objectives
    --no-rf              skip CSV / RF objectives

Output JSON schema
------------------
{
  "epsilon_frac": 0.05,
  "grid_n": 80000,
  "grid_seed": 42,
  "objectives": {
    "<name>": {
      "epsilon_abs":        float,
      "val_range":          float,
      "max_seeds":          [[x0,x1,x2], ...],
      "max_seed_vals":      [float, ...],
      "min_seeds":          [[x0,x1,x2], ...],
      "min_seed_vals":      [float, ...],
      "max_region_coords":  [[x0,x1,x2], ...],   # grid pts in max region
      "min_region_coords":  [[x0,x1,x2], ...]    # grid pts in min region
    },
    ...
  }
}

Validation hint (future use)
-----------------------------
To check if a needle x is "in the max region" of objective f:
    v_needle = f(x)
    in_region = any(abs(v_needle - v_seed) < epsilon_abs
                    for v_seed in max_seed_vals)
(This is a level-set membership check, not a strict connectivity check,
but for small ε on a smooth surface the two are equivalent.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, RadioButtons, Slider
from scipy.spatial import cKDTree

# ── project path ──────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RF_CACHE_DIR = os.path.join(_REPO_ROOT, "test_rfs")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.run_zombi_test import (
    _ACKLEY_A,
    _ACKLEY_B,
    _ACKLEY_B_SKINNY,
    _ACKLEY_C,
    _ACKLEY_SCALE,
    ACKLEY_CENTER_EDGE,
    ACKLEY_CENTER_EQUAL,
    ACKLEY_CENTER_VERTEX,
    CSV_ELEMENT_TRIPLE,
    CSV_OBJECTIVES,
    CSV_PEROVSKITE_PATH,
    MULTIMODAL_CENTERS,
    build_csv_rf_objectives,
)

# ── constants ─────────────────────────────────────────────────────────────────
N_GRID          = 80_000
KNN_K           = 14        # neighbours per grid point for level-set BFS
BFS_MAX_SIZE    = 30_000    # cap: flood-fill stops after this many grid points
DEFAULT_EPSILON = 0.05      # initial ε as fraction of each obj's value range


# =============================================================================
# Geometry helpers
# =============================================================================

def _to_cart(pts: np.ndarray) -> np.ndarray:
    """Barycentric (N, 3) → Cartesian (N, 2)."""
    pts = np.asarray(pts)
    x = pts[:, 1] + 0.5 * pts[:, 2]
    y = (np.sqrt(3) / 2) * pts[:, 2]
    return np.column_stack([x, y])


def _draw_triangle(ax, labels: Tuple[str, str, str] = ("A", "B", "C")) -> None:
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    tri = plt.Polygon(corners, fill=False, edgecolor="black", linewidth=1.2)
    ax.add_patch(tri)
    offsets = [(-0.09, -0.06), (1.04, -0.06), (0.50, np.sqrt(3) / 2 + 0.06)]
    for (ox, oy), lbl in zip(offsets, labels):
        ax.text(ox, oy, lbl, ha="center", va="center",
                fontsize=7.5, fontweight="bold")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.10, np.sqrt(3) / 2 + 0.17)
    ax.set_aspect("equal")
    ax.axis("off")


# =============================================================================
# Grid & graph
# =============================================================================

def make_grid(n: int, seed: int = 42) -> np.ndarray:
    """Uniform Dirichlet sample on the 3-simplex → (n, 3) float64."""
    rng = np.random.default_rng(seed)
    raw = rng.exponential(1.0, (n, 3))
    return (raw / raw.sum(axis=1, keepdims=True)).astype(np.float64)


def make_knn_idx(grid: np.ndarray, k: int) -> np.ndarray:
    """Return (N, k) int32 array of k nearest neighbours per grid point."""
    tree = cKDTree(grid)
    _, idx = tree.query(grid, k=k + 1)   # k+1 because first result is self
    return idx[:, 1:].astype(np.int32)


def graph_ascent(
    start_idx: int,
    vals: np.ndarray,
    nn_idx: np.ndarray,
    mode: str = "max",
    max_steps: int = 300,
) -> int:
    """
    Greedy hill-climb (or hill-descent) on the KNN graph starting from
    *start_idx*.

    At each step, move to whichever neighbour has the highest (mode="max") or
    lowest (mode="min") function value among all unvisited neighbours of the
    current node.  Stop when no improvement is possible or *max_steps* is
    reached.

    Returns the index of the local extremum found.
    """
    cur   = start_idx
    cur_v = float(vals[cur])
    seen  = {cur}
    better = (lambda a, b: a > b) if mode == "max" else (lambda a, b: a < b)

    for _ in range(max_steps):
        best_nb  = None
        best_v   = cur_v
        for nb in nn_idx[cur]:
            if nb not in seen and better(float(vals[nb]), best_v):
                best_v  = float(vals[nb])
                best_nb = nb
        if best_nb is None:
            break
        seen.add(best_nb)
        cur   = best_nb
        cur_v = best_v

    return cur


def level_set_bfs(
    seed_idx: int,
    vals: np.ndarray,
    nn_idx: np.ndarray,
    epsilon: float,
) -> Set[int]:
    """
    Flood-fill the connected component of the level set at vals[seed_idx].

    Traverses the KNN graph, accepting a neighbour only when its value is
    within *epsilon* (absolute) of the seed's value.  Stops when the region
    reaches BFS_MAX_SIZE points.
    """
    seed_val = float(vals[seed_idx])
    visited: Set[int] = {seed_idx}
    queue: deque[int] = deque([seed_idx])
    while queue and len(visited) < BFS_MAX_SIZE:
        cur = queue.popleft()
        for nb in nn_idx[cur]:
            if nb not in visited and abs(float(vals[nb]) - seed_val) < epsilon:
                visited.add(nb)
                queue.append(nb)
    return visited


# =============================================================================
# Vectorised Ackley (much faster than per-point calls)
# =============================================================================

def _ackley_vec(
    grid: np.ndarray,
    center: np.ndarray,
    a: float = _ACKLEY_A,
    b: float = _ACKLEY_B,
    c: float = _ACKLEY_C,
    scale: float = _ACKLEY_SCALE,
) -> np.ndarray:
    delta = grid - center[np.newaxis, :]
    d = grid.shape[1]
    t1 = -a * np.exp(-b * np.sqrt((delta ** 2).sum(1) / d))
    t2 = -np.exp(np.cos(c * delta).sum(1) / d)
    return scale * (t1 + t2 + a + np.e)


# =============================================================================
# Main interactive selector
# =============================================================================

# Seed entry: (grid_index, simplex_coord [3,], function_value)
_Seed = Tuple[int, np.ndarray, float]


class MaximaSelector:
    """
    Interactive matplotlib figure for labelling max/min regions on
    ternary plots of all benchmark objectives, one at a time.

    Parameters
    ----------
    objectives : list of dict
        Each dict must have keys:
            name   (str)        – panel title
            vals   (np.ndarray) – function values on *grid*, shape (N,)
            labels (tuple[str]) – 3 vertex labels for the ternary triangle
    grid : np.ndarray (N, 3)
        Simplex grid points shared by all objectives.
    nn_idx : np.ndarray (N, K)
        Pre-computed K-NN graph indices.
    epsilon : float
        Initial ε as a *fraction* of each objective's value range.
    save_path : str
        File path for JSON output (and automatic resume if it already exists).
    load_path : str or None
        Explicit path to load previously saved regions from.  Defaults to
        ``save_path`` so the same file is used for both save and resume.
        Pass a different path to seed from one file and save to another.
    """

    def __init__(
        self,
        objectives: List[Dict],
        grid: np.ndarray,
        nn_idx: np.ndarray,
        epsilon: float = DEFAULT_EPSILON,
        save_path: str = "max_min_regions.json",
        load_path: Optional[str] = None,
    ) -> None:
        self.objectives   = objectives
        self.grid         = grid
        self.nn_idx       = nn_idx
        self.cart         = _to_cart(grid)
        self.cart_tree    = cKDTree(self.cart)
        self.save_path    = save_path
        self.load_path    = load_path if load_path is not None else save_path
        self.mode         = "max"
        self.epsilon_frac = epsilon
        self.current_idx  = 0

        # Per-objective mutable state
        self.state: List[Dict] = []
        for obj in objectives:
            vr = float(obj["vals"].max() - obj["vals"].min())
            self.state.append({
                "max_seeds":   [],       # list of _Seed
                "min_seeds":   [],
                "max_region":  set(),    # grid indices
                "min_region":  set(),
                "epsilon":     epsilon * vr,
                "val_range":   vr,
            })

        # ── Figure layout ─────────────────────────────────────────────────────
        # Single large ternary + controls at the bottom
        self.fig = plt.figure(figsize=(9, 10))

        # Main ternary axis — tall, nearly square
        self.ax_main = self.fig.add_axes([0.08, 0.28, 0.78, 0.64])

        # Colorbar axis to the right of the ternary
        self.ax_cbar = self.fig.add_axes([0.88, 0.28, 0.025, 0.64])

        # ── Navigation row  (y ≈ 0.21) ───────────────────────────────────────
        ax_prev    = self.fig.add_axes([0.08, 0.205, 0.10, 0.045])
        ax_next    = self.fig.add_axes([0.82, 0.205, 0.10, 0.045])
        ax_nav_lbl = self.fig.add_axes([0.19, 0.205, 0.62, 0.045])
        ax_nav_lbl.axis("off")

        self.btn_prev = Button(ax_prev, "◄  Prev")
        self.btn_next = Button(ax_next, "Next  ►")
        self.nav_txt  = ax_nav_lbl.text(
            0.5, 0.5, "",
            ha="center", va="center", fontsize=10, fontweight="bold",
            transform=ax_nav_lbl.transAxes,
        )

        # ── Overview dot-row  (y ≈ 0.175) ────────────────────────────────────
        # Small coloured dots, one per objective, showing seed status at a glance
        ax_dots = self.fig.add_axes([0.08, 0.170, 0.84, 0.028])
        ax_dots.axis("off")
        self.dots_txt = ax_dots.text(
            0.5, 0.5, "",
            ha="center", va="center", fontsize=8,
            transform=ax_dots.transAxes,
            family="monospace",
        )

        # ── Controls row  (y ≈ 0.10) ─────────────────────────────────────────
        ax_radio     = self.fig.add_axes([0.02, 0.060, 0.10, 0.095])
        ax_slider    = self.fig.add_axes([0.17, 0.105, 0.35, 0.025])
        ax_undo      = self.fig.add_axes([0.57, 0.088, 0.09, 0.042])
        ax_clear_cur = self.fig.add_axes([0.67, 0.088, 0.09, 0.042])
        ax_clear_all = self.fig.add_axes([0.77, 0.088, 0.09, 0.042])
        ax_save      = self.fig.add_axes([0.87, 0.088, 0.09, 0.042])

        # ── Status bar  (y ≈ 0.01) ────────────────────────────────────────────
        ax_status = self.fig.add_axes([0.00, 0.010, 1.00, 0.030])
        ax_status.axis("off")

        self.radio  = RadioButtons(
            ax_radio, ["max", "min"], active=0,
            label_props={"fontsize": [9, 9]},
        )
        self.slider = Slider(
            ax_slider, "ε (frac.)", 0.001, 0.30,
            valinit=epsilon, valstep=0.001,
        )
        self.btn_undo      = Button(ax_undo,      "Undo")
        self.btn_clear_cur = Button(ax_clear_cur, "Clear")
        self.btn_clear_all = Button(ax_clear_all, "Clear All")
        self.btn_save      = Button(ax_save,      "Save")

        self.radio.on_clicked(self._set_mode)
        self.slider.on_changed(self._update_epsilon)
        self.btn_prev.on_clicked(self._prev)
        self.btn_next.on_clicked(self._next)
        self.btn_undo.on_clicked(self._undo)
        self.btn_clear_cur.on_clicked(self._clear_current)
        self.btn_clear_all.on_clicked(self._clear_all)
        self.btn_save.on_clicked(self._save)

        self.status_txt = ax_status.text(
            0.5, 0.5,
            "Left-click: add seed  |  Right-click: remove nearest  |  mode=max",
            ha="center", va="center", fontsize=8,
            transform=ax_status.transAxes,
        )

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event",    self._on_key)
        self._load_existing()
        self._goto(0)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _prev(self, _event) -> None:
        self._goto((self.current_idx - 1) % len(self.objectives))

    def _next(self, _event) -> None:
        self._goto((self.current_idx + 1) % len(self.objectives))

    def _goto(self, i: int) -> None:
        self.current_idx = i
        self._redraw_current()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _redraw_current(self) -> None:
        i   = self.current_idx
        ax  = self.ax_main
        obj = self.objectives[i]
        st  = self.state[i]
        n   = len(self.objectives)

        ax.cla()
        self.ax_cbar.cla()

        c    = self.cart
        vals = obj["vals"]
        vmin_val = float(vals.min())
        vmax_val = float(vals.max())

        # Base objective heatmap
        sc = ax.scatter(
            c[:, 0], c[:, 1], c=vals, cmap="viridis",
            s=6, alpha=0.55, linewidths=0, zorder=1,
            vmin=vmin_val, vmax=vmax_val,
        )

        # Coloured region blobs
        _region_style = {
            "max": ("red",       0.30),
            "min": ("royalblue", 0.30),
        }
        for mode, (colour, alpha) in _region_style.items():
            region = st[f"{mode}_region"]
            if region:
                ri = list(region)
                ax.scatter(c[ri, 0], c[ri, 1], color=colour,
                           alpha=alpha, s=8, linewidths=0, zorder=2)

        # Seed stars
        _seed_style = {
            "max": ("red",       "darkred"),
            "min": ("royalblue", "navy"),
        }
        for mode, (face, edge) in _seed_style.items():
            for sidx, sc_coord, sv in st[f"{mode}_seeds"]:
                sp = _to_cart(sc_coord.reshape(1, 3))
                ax.scatter(
                    sp[:, 0], sp[:, 1], marker="*", color=face,
                    s=400, zorder=6, edgecolors=edge, linewidths=0.7,
                )

        _draw_triangle(ax, labels=obj["labels"])

        # ── Colorbar with HIGH / LOW labels ───────────────────────────────────
        self.fig.colorbar(sc, cax=self.ax_cbar)
        self.ax_cbar.tick_params(labelsize=8)
        # "HIGH" annotation above colorbar (yellow-green = high end of viridis)
        self.ax_cbar.text(
            0.5, 1.035, f"▲ HIGH\n{vmax_val:.4g}",
            transform=self.ax_cbar.transAxes,
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            color="#2ca02c",
        )
        # "LOW" annotation below colorbar (dark purple = low end of viridis)
        self.ax_cbar.text(
            0.5, -0.035, f"{vmin_val:.4g}\nLOW ▼",
            transform=self.ax_cbar.transAxes,
            ha="center", va="top", fontsize=8, fontweight="bold",
            color="#440154",
        )

        # ── Navigation label ──────────────────────────────────────────────────
        self.nav_txt.set_text(f"[{i + 1} / {n}]   {obj['name']}")

        # Overview dots — one per objective, showing ▲/▼/✓ status
        self._refresh_overview()

        # ── Title: show current mode prominently with colour ──────────────────
        nmax = len(st["max_seeds"]); rmax = len(st["max_region"])
        nmin = len(st["min_seeds"]); rmin = len(st["min_region"])

        mode_colour = "darkred" if self.mode == "max" else "navy"
        mode_label  = "MAX (▲ red)"  if self.mode == "max" else "MIN (▼ blue)"
        both_done   = "  ✓ both labelled" if (nmax > 0 and nmin > 0) else ""

        ax.set_title(
            f"Placing: {mode_label}{both_done}\n"
            f"▲ max seeds: {nmax} → {rmax} pts    "
            f"▼ min seeds: {nmin} → {rmin} pts",
            fontsize=10, pad=8, color=mode_colour, fontweight="bold",
        )

        self.fig.canvas.draw_idle()

    def _refresh_overview(self) -> None:
        """Update the one-line overview strip: shows ✓/▲/▼/· per objective."""
        parts = []
        for j, (obj, st) in enumerate(zip(self.objectives, self.state)):
            has_max = bool(st["max_seeds"])
            has_min = bool(st["min_seeds"])
            if has_max and has_min:
                marker = "✓ "     # both done
            elif has_max:
                marker = "▲ "     # max only
            elif has_min:
                marker = " ▼"     # min only
            else:
                marker = "· "     # nothing yet
            label = obj["name"].replace("RF-", "").replace("Ackley-", "A-")
            if j == self.current_idx:
                parts.append(f"[{label}:{marker}]")
            else:
                parts.append(f" {label}:{marker} ")
        n_done = sum(
            bool(st["max_seeds"]) and bool(st["min_seeds"])
            for st in self.state
        )
        summary = f"  {n_done}/{len(self.objectives)} fully labelled"
        self.dots_txt.set_text("  ".join(parts) + summary)

    # ── Click handler ─────────────────────────────────────────────────────────

    def _on_click(self, event) -> None:
        if event.inaxes is not self.ax_main:
            return

        dist, grid_idx = self.cart_tree.query([[event.xdata, event.ydata]])
        grid_idx = int(grid_idx[0])
        if float(dist[0]) > 0.07:
            return  # outside the triangle

        i     = self.current_idx
        obj   = self.objectives[i]
        st    = self.state[i]
        key   = self.mode

        if event.button == 1:                                   # left → add
            # Climb (or descend) from the clicked point to the local extremum
            peak_idx = graph_ascent(
                grid_idx, obj["vals"], self.nn_idx, mode=key,
            )
            coord = self.grid[peak_idx].copy()
            val   = float(obj["vals"][peak_idx])

            seeds = st[f"{key}_seeds"]
            if any(s[0] == peak_idx for s in seeds):
                self._set_status(
                    f"Already have {key} seed at that peak  f={val:.4f}"
                )
                return
            seeds.append((peak_idx, coord, val))
            self._recompute_region(i, key)
            moved = peak_idx != grid_idx
            self._set_status(
                f"Added {key} seed"
                + (" (snapped to peak)" if moved else "")
                + f"  f={val:.4f}  {coord.round(3)}"
            )

        elif event.button == 3:                                 # right → remove
            seeds = st[f"{key}_seeds"]
            if not seeds:
                return
            click_cart = np.array([event.xdata, event.ydata])
            sc_arr = np.array([_to_cart(s[1].reshape(1, 3))[0] for s in seeds])
            rm = int(np.linalg.norm(sc_arr - click_cart, axis=1).argmin())
            removed = seeds.pop(rm)
            self._recompute_region(i, key)
            self._set_status(
                f"Removed {key} seed  f={removed[2]:.4f}  {removed[1].round(3)}"
            )

        else:
            return

        self._redraw_current()

    # ── Region helpers ────────────────────────────────────────────────────────

    def _recompute_region(self, i: int, mode: str) -> None:
        """Union of level-set BFS from every seed in *mode* for objective *i*."""
        st  = self.state[i]
        obj = self.objectives[i]
        region: Set[int] = set()
        for sidx, _, _ in st[f"{mode}_seeds"]:
            region |= level_set_bfs(
                sidx, obj["vals"], self.nn_idx, st["epsilon"]
            )
        st[f"{mode}_region"] = region

    def _recompute_all_regions(self) -> None:
        for i in range(len(self.objectives)):
            for mode in ("max", "min"):
                self._recompute_region(i, mode)

    # ── Widget callbacks ──────────────────────────────────────────────────────

    def _set_mode(self, label: str) -> None:
        self.mode = label
        self._redraw_current()
        self._set_status(f"Mode → {label}  (m = max, n = min)")

    def _on_key(self, event) -> None:
        """Keyboard shortcuts for faster navigation and mode switching."""
        key = event.key
        if key in ("right", "d"):
            self._next(None)
        elif key in ("left", "a"):
            self._prev(None)
        elif key == "m":
            self.radio.set_active(0)   # triggers _set_mode("max")
        elif key in ("n", "v"):
            self.radio.set_active(1)   # triggers _set_mode("min")
        elif key == "z":
            self._undo(None)
        elif key == "s":
            self._save(None)

    def _update_epsilon(self, val: float) -> None:
        self.epsilon_frac = float(val)
        for st in self.state:
            st["epsilon"] = self.epsilon_frac * st["val_range"]
        self._recompute_all_regions()
        self._redraw_current()
        self._set_status(f"ε updated to {self.epsilon_frac:.3f}")

    def _undo(self, _event) -> None:
        """Remove the most recently added seed on the current objective."""
        i   = self.current_idx
        st  = self.state[i]
        obj = self.objectives[i]
        for mode in (self.mode, "max", "min"):
            seeds = st[f"{mode}_seeds"]
            if seeds:
                removed = seeds.pop()
                self._recompute_region(i, mode)
                self._redraw_current()
                self._set_status(
                    f"Undo: removed last {mode} seed from {obj['name']}"
                )
                return
        self._set_status("Nothing to undo on this objective.")

    def _clear_current(self, _event) -> None:
        """Clear seeds and regions for the currently viewed objective only."""
        i  = self.current_idx
        st = self.state[i]
        for mode in ("max", "min"):
            st[f"{mode}_seeds"]  = []
            st[f"{mode}_region"] = set()
        self._redraw_current()
        self._set_status(f"Cleared: {self.objectives[i]['name']}")

    def _clear_all(self, _event) -> None:
        for st in self.state:
            for mode in ("max", "min"):
                st[f"{mode}_seeds"]  = []
                st[f"{mode}_region"] = set()
        self._redraw_current()
        self._set_status("Cleared all seeds and regions.")

    def _load_existing(self) -> None:
        """
        If ``self.save_path`` already exists on disk, restore every seed from
        it.  Seeds are matched back to the nearest grid point so the BFS
        regions regenerate exactly as they were.

        Silently skips objectives whose name is not found in the loaded file
        (e.g. if the grid or objective list has changed).
        """
        if not os.path.isfile(self.load_path):
            return
        try:
            with open(self.load_path) as fh:
                data = json.load(fh)
        except Exception as exc:
            warnings.warn(
                f"Could not load existing regions from '{self.load_path}': {exc}",
                stacklevel=2,
            )
            return

        saved_objs = data.get("objectives", {})
        n_loaded   = 0

        for i, (obj, st) in enumerate(zip(self.objectives, self.state)):
            entry = saved_objs.get(obj["name"])
            if entry is None:
                continue

            # Restore ε from the saved value if it's available
            if "epsilon_abs" in entry:
                st["epsilon"] = float(entry["epsilon_abs"])

            for mode in ("max", "min"):
                seed_coords = entry.get(f"{mode}_seeds", [])
                seed_vals   = entry.get(f"{mode}_seed_vals", [])

                for coord_list, sv in zip(seed_coords, seed_vals):
                    coord     = np.array(coord_list, dtype=np.float64)
                    coord_c   = _to_cart(coord.reshape(1, 3))
                    _, g_idx  = self.cart_tree.query(coord_c)
                    g_idx     = int(g_idx[0])
                    # Use the stored function value (not re-evaluated from grid)
                    # so the level set is reproduced at the same threshold.
                    st[f"{mode}_seeds"].append((g_idx, self.grid[g_idx].copy(), float(sv)))
                    n_loaded += 1

                self._recompute_region(i, mode)

        if n_loaded:
            print(f"  Loaded {n_loaded} seed(s) from '{self.load_path}'.")
        else:
            print(f"  Found '{self.load_path}' but no matching seeds to restore.")

    def _save(self, _event) -> None:
        out: Dict = {
            "epsilon_frac": self.epsilon_frac,
            "grid_n":       int(len(self.grid)),
            "grid_seed":    42,
            "objectives":   {},
        }
        for obj, st in zip(self.objectives, self.state):
            out["objectives"][obj["name"]] = {
                "epsilon_abs":       float(st["epsilon"]),
                "val_range":         float(st["val_range"]),
                "max_seeds":         [s[1].tolist() for s in st["max_seeds"]],
                "max_seed_vals":     [s[2]          for s in st["max_seeds"]],
                "min_seeds":         [s[1].tolist() for s in st["min_seeds"]],
                "min_seed_vals":     [s[2]          for s in st["min_seeds"]],
                "max_region_coords": self.grid[list(st["max_region"])].tolist()
                                     if st["max_region"] else [],
                "min_region_coords": self.grid[list(st["min_region"])].tolist()
                                     if st["min_region"] else [],
            }
        with open(self.save_path, "w") as fh:
            json.dump(out, fh, indent=2)
        abs_path = os.path.abspath(self.save_path)
        print(f"\nRegions saved → {abs_path}")
        self._set_status(f"Saved → {self.save_path}")

    def _set_status(self, msg: str) -> None:
        self.status_txt.set_text(
            f"{msg}  |  mode={self.mode}  ε={self.epsilon_frac:.3f}"
        )
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Interactive max/min region selector for ZoMBI-Hop objectives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--epsilon",    type=float, default=DEFAULT_EPSILON,
                    help="Initial ε as fraction of each objective's value range.")
    ap.add_argument("--n-pts",      type=int,   default=N_GRID,
                    help="Number of simplex grid points.")
    ap.add_argument("--csv-path",   type=str,   default=CSV_PEROVSKITE_PATH,
                    help="Path to perovskite campaign CSV.")
    ap.add_argument("--rf-cache",   type=str,   default=RF_CACHE_DIR,
                    help="Directory for caching trained RF models.")
    ap.add_argument("--rf-samples", type=int,   default=10_000_000,
                    help="Dirichlet samples for RF global-max search.")
    ap.add_argument("--save-path",  type=str,   default="max_min_regions.json",
                    help="Output JSON path for saved regions.")
    ap.add_argument("--load-path",  type=str,   default=None,
                    help="JSON file to resume from (defaults to --save-path).")
    ap.add_argument("--no-rf",      action="store_true",
                    help="Skip CSV / RF objectives.")
    ap.add_argument("--no-ackley",  action="store_true",
                    help="Skip synthetic Ackley objectives.")
    args = ap.parse_args()

    # ── Grid ──────────────────────────────────────────────────────────────────
    print(f"Generating {args.n_pts:,}-point simplex grid …")
    grid   = make_grid(args.n_pts)

    print(f"Building {KNN_K}-NN graph …")
    nn_idx = make_knn_idx(grid, KNN_K)

    objectives: List[Dict] = []

    # ── CSV / RF objectives ───────────────────────────────────────────────────
    if not args.no_rf:
        if os.path.isfile(args.csv_path):
            print("Loading / training CSV RF objectives …")
            try:
                csv_rf = build_csv_rf_objectives(
                    csv_path=args.csv_path,
                    objectives=CSV_OBJECTIVES,
                    rf_global_samples=args.rf_samples,
                    cache_dir=args.rf_cache,
                )
                for name, data in csv_rf.items():
                    rf   = data["rf"]
                    vals = rf.predict(grid).astype(np.float64)
                    objectives.append({
                        "name":   f"RF-{name}",
                        "vals":   vals,
                        "labels": CSV_ELEMENT_TRIPLE,
                    })
                print(f"  Loaded {len(csv_rf)} RF objective(s).")
            except Exception as exc:
                warnings.warn(f"CSV RF build failed: {exc}", stacklevel=2)
        else:
            warnings.warn(
                f"Perovskite CSV not found at:\n  {args.csv_path}\n"
                "Skipping RF objectives.  Pass --csv-path to point to the file.",
                stacklevel=2,
            )

    # ── Ackley objectives (vectorised) ────────────────────────────────────────
    if not args.no_ackley:
        print("Computing Ackley objectives on grid …")
        ackley_defs = [
            ("Ackley-Centroid",    ACKLEY_CENTER_EQUAL,  _ACKLEY_B),
            ("Ackley-Edge",        ACKLEY_CENTER_EDGE,   _ACKLEY_B),
            ("Ackley-Vertex",      ACKLEY_CENTER_VERTEX, _ACKLEY_B),
        ]
        for aname, center, b in ackley_defs:
            vals = _ackley_vec(grid, center, b=b)
            objectives.append({"name": aname, "vals": vals.astype(np.float64),
                                "labels": ("A", "B", "C")})

        # Multi-modal: sum of three skinny Ackley peaks
        mm_vals = sum(
            _ackley_vec(grid, c, b=_ACKLEY_B_SKINNY) for c in MULTIMODAL_CENTERS
        )
        objectives.append({
            "name":   "Ackley-Multi-modal",
            "vals":   mm_vals.astype(np.float64),
            "labels": ("A", "B", "C"),
        })

    if not objectives:
        print("No objectives loaded — nothing to display.  Exiting.")
        return

    print(f"\nObjectives ({len(objectives)}): {[o['name'] for o in objectives]}")
    print("\nLaunching interactive selector …")
    print("  Left-click      → add seed in current mode (max / min)")
    print("  Right-click     → remove nearest seed in current mode")
    print("  ← / → (or a/d) → navigate between objectives")
    print("  m               → switch to MAX mode (red ▲ seeds)")
    print("  n               → switch to MIN mode (blue ▼ seeds)")
    print("  z               → undo last seed on current objective")
    print("  s               → save regions to JSON")
    print("  ε slider        → adjust level-set width (fraction of value range)")
    print("  Clear           → clear current objective only")
    print("  Clear All       → reset all objectives")
    print("")
    print("  Colorbar: HIGH (bright yellow-green) = large values")
    print("            LOW  (dark purple)          = small values")
    print("  Red   ★ stars / blobs  = MAX regions")
    print("  Blue  ★ stars / blobs  = MIN regions")
    print("  ✓ in overview strip    = objective has BOTH max and min seeds\n")

    selector = MaximaSelector(
        objectives, grid, nn_idx,
        epsilon=args.epsilon,
        save_path=args.save_path,
        load_path=args.load_path,
    )
    selector.show()


if __name__ == "__main__":
    main()
