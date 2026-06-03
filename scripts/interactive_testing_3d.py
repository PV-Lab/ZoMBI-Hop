"""
Interactive testing framework for ZoMBI-Hop + LineBO — 3D simplex / Ackley.

Each ZoMBI-Hop iteration shows three sequential ternary frames:
  1. GP Sampling     – all raw candidates coloured by acquisition value
  2. Gradient Ascent – 2×2 comparison of four simplex ascent methods
  3. LineBO Line     – 24 evenly-spaced points on the chosen line + ZoMBI star

The four ascent methods compared side-by-side:
  A) Softmax reparametrisation  - unconstrained θ, x = softmax(θ)
  B) Riemannian gradient        – mean-centred (zero-sum) Euclidean gradient
  C) Riemannian + log-barrier   – Riemannian gradient + μ·Σlog(xᵢ−lᵢ)+log(hᵢ−xᵢ)
  D) Natural gradient / Exp-map – multiplicative weights / Fisher-Rao geodesic

Run from the repo root:
    python -u -m scripts.interactive_testing_3d
"""

from __future__ import annotations

import sys
import time
import pathlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch

from src import ZoMBIHop, LineBO
from src.core.linebo import line_simplex_segment, zero_sum_dirs
from src.utils.gp_simplex import GPSimplex

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# ── constants ────────────────────────────────────────────────────────────────
DIMENSIONS           = 3
NUM_LINE_POINTS      = 24
NUM_INIT_DATA        = 2
DELAY_SECONDS        = 1.0    # pause for raw-sampling frame
ASCENT_HOLD_SECONDS  = 10.0   # hold for the 4-panel gradient-ascent frame
DEVICE               = "cpu"
DTYPE                = torch.float64
BOUNDARY_TOL         = 1e-3   # internal  ⟺  lo+tol < xᵢ < hi-tol for all i
LOG_BARRIER_MU       = 0.01   # strength of the log-barrier in method C
GRADIENT_LOG_DIR     = pathlib.Path("gradient_log")  # saved ascent figures

# Where the Ackley maximum lives on the simplex.
# Change this to test boundary/edge convergence.
# e.g. centroid=[1/3,1/3,1/3], edge=[0.5,0.5,0], vertex=[1,0,0]
# ACKLEY_CENTER        = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])   # vertex
ACKLEY_CENTER        = np.array([0.5, 0.5, 0.0])   # midpoint of one edge
# ACKLEY_CENTER        = np.array([1.0, 0.0, 0.0])   # centroid

# Step sizes per method — tuned so each method makes comparable progress
STEP_SOFTMAX   = 0.05   # θ-space; effective x-step is smaller due to Jacobian
STEP_RIEMANNIAN= 0.005  # small — no barrier to stabilise near faces
STEP_BARRIER   = 0.01   # barrier provides stability
STEP_NATGRAD   = 0.02   # multiplicative; exp-map compresses large steps naturally
MAX_ASCENT_STEPS = 50


# ============================================================================
# Synthetic objective: Ackley on the 3-D simplex
# ============================================================================

def ackley_simplex(x: np.ndarray, a: float = 20.0, b: float = 0.2,
                   c: float = 2 * np.pi) -> np.ndarray:
    """Negated Ackley centred at ACKLEY_CENTER. ZoMBI-Hop maximises this."""
    x      = np.atleast_2d(x).astype(float)
    d      = x.shape[-1]
    center = ACKLEY_CENTER[:d] if len(ACKLEY_CENTER) >= d else np.ones(d) / d
    z = (x - center) * 30.0
    sum_sq  = np.sum(z ** 2,        axis=-1) / d
    sum_cos = np.sum(np.cos(c * z), axis=-1) / d
    val = -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(sum_cos) + a + np.exp(1.0)
    return -val


# ============================================================================
# Ternary helpers
# ============================================================================

_TV = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])


def _to_cart(pts: np.ndarray) -> np.ndarray:
    return np.atleast_2d(pts) @ _TV


def _draw_triangle(ax: Any) -> None:
    tri = np.vstack([_TV, _TV[:1]])
    ax.plot(tri[:, 0], tri[:, 1], "k-", lw=1.8, zorder=1)
    labels  = ["x₁", "x₂", "x₃"]
    offsets = [(-0.07, -0.05), (1.06, -0.05),
               (0.5, np.sqrt(3.0) / 2.0 + 0.06)]
    for lab, (ox, oy) in zip(labels, offsets):
        ax.text(ox, oy, lab, ha="center", va="center",
                fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_bounds_region(ax: Any, lo: np.ndarray, hi: np.ndarray,
                        n: int = 3000) -> None:
    rng  = np.random.default_rng(42)
    pts  = rng.dirichlet(np.ones(3), n)
    mask = np.all((pts >= lo - 1e-9) & (pts <= hi + 1e-9), axis=1)
    if mask.sum() < 3:
        return
    cart = _to_cart(pts[mask])
    ax.scatter(cart[:, 0], cart[:, 1],
               c="khaki", s=3, alpha=0.3, zorder=0.5, marker="s")


# ============================================================================
# Simplex-enforcement helper (shared by all path methods)
# ============================================================================

def _enforce_bounds_tensor(
    x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Clamp xᵢ to [lo+eps, hi-eps] then renormalise to sum=1."""
    x = torch.clamp(x, min=lo + eps, max=hi - eps)
    s = x.sum()
    if s < 1e-12:
        return torch.ones_like(x) / x.shape[0]
    return x / s


# ============================================================================
# _TernaryPlot — three sequential frames per iteration
# ============================================================================

class _TernaryPlot:
    def __init__(self, run_dir: Optional[pathlib.Path] = None) -> None:
        self._fig: Any = None
        self._history_line_pts:   List[np.ndarray] = []
        self._history_line_y:     List[np.ndarray] = []
        self._history_candidates: List[np.ndarray] = []
        self._run_dir = run_dir  # one directory for the whole run

    def _setup(self) -> None:
        if _HAS_MPL:
            plt.ion()

    def _close(self) -> None:
        if self._fig is not None:
            try:
                plt.close(self._fig)
            except Exception:
                pass
            self._fig = None

    def _draw_history(self, ax: Any) -> None:
        for pts in self._history_line_pts:
            cart = _to_cart(pts)
            ax.scatter(cart[:, 0], cart[:, 1],
                       c="lightgrey", s=8, alpha=0.35, zorder=2)
        if self._history_candidates:
            old_c  = np.array(self._history_candidates)
            c_cart = _to_cart(old_c)
            ax.scatter(c_cart[:, 0], c_cart[:, 1],
                       marker="*", s=120, c="salmon", alpha=0.4, zorder=3)

    # ── Frame 1: raw GP candidates ────────────────────────────────────────────
    def show_random_points(
        self,
        pts: np.ndarray,
        acq_vals: np.ndarray,
        unpen_mask: np.ndarray,
        bounds: torch.Tensor,
    ) -> None:
        if not _HAS_MPL:
            return
        self._close()
        lo = bounds[0].cpu().numpy()
        hi = bounds[1].cpu().numpy()

        fig, ax = plt.subplots(figsize=(7, 6.5))
        fig.suptitle(
            "ZoMBI-Hop  •  GP Sampling  —  raw candidates\n"
            f"coloured by acquisition value  "
            f"(iter {len(self._history_candidates) + 1})",
            fontsize=11,
        )
        _draw_triangle(ax)
        _draw_bounds_region(ax, lo, hi)
        self._draw_history(ax)

        pen_pts  = pts[~unpen_mask]
        upen_pts = pts[ unpen_mask]
        upen_acq = acq_vals[unpen_mask]

        if len(pen_pts):
            c = _to_cart(pen_pts)
            ax.scatter(c[:, 0], c[:, 1], marker="x", s=14,
                       c="lightgrey", alpha=0.4, zorder=2, label="penalised")

        if len(upen_pts):
            vlo, vhi = upen_acq.min(), upen_acq.max()
            norm = mcolors.Normalize(vmin=vlo,
                                     vmax=vhi if vhi > vlo else vlo + 1e-8)
            c  = _to_cart(upen_pts)
            sc = ax.scatter(c[:, 0], c[:, 1], c=upen_acq, cmap="viridis",
                            norm=norm, s=18, alpha=0.75, zorder=3,
                            edgecolors="none", label="unpenalised")
            fig.colorbar(sc, ax=ax, fraction=0.042, pad=0.02,
                         label="acquisition value")

        ax.legend(loc="upper right", fontsize=8, markerscale=0.9)
        self._fig = fig
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(DELAY_SECONDS)

    # ── Frame 2: 2×2 comparison of four ascent methods ───────────────────────
    def show_ascent_4panel(
        self,
        method_results: Dict[str, Tuple[List[np.ndarray], np.ndarray]],
        bounds: torch.Tensor,
        overall_best_pt: np.ndarray,
    ) -> None:
        """
        method_results: {method_name: (paths, best_pt_for_that_method)}
        overall_best_pt: the candidate chosen across all methods (marked distinctly)
        """
        if not _HAS_MPL:
            return
        self._close()
        lo = bounds[0].cpu().numpy()
        hi = bounds[1].cpu().numpy()

        method_names = list(method_results.keys())
        n_methods    = len(method_names)
        ncols        = 3 if n_methods > 4 else min(n_methods, 2)
        nrows        = (n_methods + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(7 * ncols, 6.5 * nrows),
                                 squeeze=False)
        fig.suptitle(
            f"ZoMBI-Hop  •  Gradient Ascent — {n_methods} simplex methods compared\n"
            f"iteration {len(self._history_candidates) + 1}  |  "
            f"chosen → {np.round(overall_best_pt, 3).tolist()}",
            fontsize=11,
        )

        legend_proxies = [
            Line2D([0], [0], color="steelblue", lw=1.2, label="ascent path"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="royalblue",  markersize=6, label="start"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="darkorange", markersize=6, label="end"),
            Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="gold",
                   markeredgecolor="darkgoldenrod",
                   markersize=10, label="method best"),
            Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="red",
                   markeredgecolor="darkred",
                   markersize=12, label="chosen (overall)"),
        ]

        overall_cart = _to_cart(overall_best_pt.reshape(1, -1))

        for panel_idx, name in enumerate(method_names):
            row, col  = divmod(panel_idx, ncols)
            ax        = axes[row][col]
            paths, method_best = method_results[name]

            _draw_triangle(ax)
            _draw_bounds_region(ax, lo, hi)
            self._draw_history(ax)

            for path in paths:
                if len(path) < 2:
                    continue
                cart = _to_cart(path)
                ax.plot(cart[:, 0], cart[:, 1],
                        c="steelblue", lw=0.9, alpha=0.35, zorder=3)
                ax.scatter(cart[0, 0],  cart[0, 1],
                           c="royalblue",  s=16, alpha=0.55,
                           zorder=4, edgecolors="none")
                ax.scatter(cart[-1, 0], cart[-1, 1],
                           c="darkorange", s=20, alpha=0.70,
                           zorder=4, edgecolors="none")

            if method_best is not None:
                mb_cart = _to_cart(method_best.reshape(1, -1))
                ax.scatter(mb_cart[0, 0], mb_cart[0, 1],
                           marker="*", s=320, c="gold", zorder=6,
                           edgecolors="darkgoldenrod", linewidths=0.8)

            # Mark the overall chosen candidate (red) on every panel
            ax.scatter(overall_cart[0, 0], overall_cart[0, 1],
                       marker="*", s=380, c="red", zorder=7,
                       edgecolors="darkred", linewidths=0.9)

            ax.set_title(name, fontsize=9, fontweight="bold", pad=4)

        # Hide unused panels
        for panel_idx in range(n_methods, nrows * ncols):
            row, col = divmod(panel_idx, ncols)
            axes[row][col].set_visible(False)

        fig.legend(handles=legend_proxies, loc="lower center",
                   ncol=5, fontsize=8, frameon=True)
        self._fig = fig
        plt.tight_layout(rect=[0, 0.04, 1, 1])

        # ── save figure ───────────────────────────────────────────────────────
        if self._run_dir is not None:
            n_iter   = len(self._history_candidates) + 1
            out_path = self._run_dir / f"ascent_iter{n_iter:04d}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"  [plot] saved → {out_path}", flush=True)

        plt.show(block=False)
        plt.pause(ASCENT_HOLD_SECONDS)

    # ── Frame 3: chosen LineBO line ───────────────────────────────────────────
    def update(self, candidate: np.ndarray,
               line_pts: np.ndarray, line_y: np.ndarray) -> None:
        self._history_candidates.append(candidate.copy())
        self._history_line_pts.append(line_pts.copy())
        self._history_line_y.append(line_y.copy())

        if not _HAS_MPL:
            return
        self._close()

        fig, ax  = plt.subplots(figsize=(7, 6.5))
        n_iter   = len(self._history_candidates)
        best_neg = max(float(y) for yb in self._history_line_y for y in yb)
        fig.suptitle(
            f"ZoMBI-Hop  •  LineBO line\n"
            f"iteration {n_iter}   |   best −Ackley so far: {best_neg:.4f}",
            fontsize=11,
        )
        _draw_triangle(ax)

        for pts in self._history_line_pts[:-1]:
            cart = _to_cart(pts)
            ax.scatter(cart[:, 0], cart[:, 1],
                       c="lightgrey", s=12, alpha=0.45, zorder=2)
        if len(self._history_candidates) > 1:
            old_c  = np.array(self._history_candidates[:-1])
            c_cart = _to_cart(old_c)
            ax.scatter(c_cart[:, 0], c_cart[:, 1],
                       marker="*", s=180, c="salmon", alpha=0.55, zorder=4)

        curr_cart = _to_cart(line_pts)
        y_lo, y_hi = line_y.min(), line_y.max()
        norm = mcolors.Normalize(vmin=y_lo,
                                 vmax=y_hi if y_hi > y_lo else y_lo + 1e-8)
        sc = ax.scatter(curr_cart[:, 0], curr_cart[:, 1],
                        c=line_y, cmap="plasma", norm=norm,
                        s=38, zorder=3, edgecolors="k", linewidths=0.35,
                        label="line pts")
        ax.plot(curr_cart[[0, -1], 0], curr_cart[[0, -1], 1],
                c="steelblue", lw=1.3, alpha=0.65, zorder=2.5)

        cand_cart = _to_cart(candidate.reshape(1, -1))
        ax.scatter(cand_cart[0, 0], cand_cart[0, 1],
                   marker="*", s=420, c="red", zorder=5,
                   edgecolors="darkred", linewidths=0.9,
                   label="ZoMBI candidate")

        fig.colorbar(sc, ax=ax, fraction=0.042, pad=0.02,
                     label="−Ackley (higher = better)")
        ax.legend(loc="upper right", fontsize=8, markerscale=0.75)

        self._fig = fig
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.05)


# ============================================================================
# InstrumentedGPSimplex — four gradient-ascent methods + visualisation
# ============================================================================

class InstrumentedGPSimplex(GPSimplex):
    """
    GPSimplex subclass used only in interactive_testing_3d.

    Overrides get_candidate to:
      1. After raw sampling  → show_random_points           (frame 1)
      2. Run four ascent methods from the same seeds
      3. After ascent        → show_ascent_4panel            (frame 2)
      4. Return the best strictly-internal candidate across all methods.
    """

    def __init__(self, plot_state: _TernaryPlot, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._plot_state = plot_state

    # ── internality check ─────────────────────────────────────────────────────
    @staticmethod
    def _internal_mask(
        candidates: torch.Tensor,
        bounds: torch.Tensor,
        tol: float = BOUNDARY_TOL,
    ) -> torch.Tensor:
        lo, hi = bounds[0], bounds[1]
        return ((candidates > lo.unsqueeze(0) + tol) &
                (candidates < hi.unsqueeze(0) - tol)).all(dim=1)

    # ── shared helpers ─────────────────────────────────────────────────────────
    def _acq_grad_flat(
        self, acq: Any, x_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (∂acq/∂x, acq_value) for x_flat (d,). No side-effects."""
        x = x_flat.detach().requires_grad_(True)
        val = acq(x.unsqueeze(0).unsqueeze(0))
        g   = torch.autograd.grad(val.sum(), x)[0]
        return g.detach(), val.detach().squeeze()

    def _init_x(self, init_cond_r: torch.Tensor,
                lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """Pull one restart seed to a clean (d,) simplex interior point."""
        x = init_cond_r.clone().squeeze()
        x = torch.clamp(x, min=1e-10)
        x = x / x.sum()
        return _enforce_bounds_tensor(x, lo, hi)

    # ── Method A: Softmax reparametrisation ───────────────────────────────────
    def _paths_softmax(
        self, acq: Any, bounds: torch.Tensor,
        init_cond: torch.Tensor,
        max_steps: int = MAX_ASCENT_STEPS,
        step_size: float = STEP_SOFTMAX,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[np.ndarray]]:
        lo, hi = bounds[0], bounds[1]
        d      = init_cond.shape[-1]
        cands, vals, paths = [], [], []

        for r in range(init_cond.shape[0]):
            x = self._init_x(init_cond[r], lo, hi)
            # Initialise θ from x (log-space, gauge-fixed)
            theta = torch.log(x.clamp(min=1e-10)).detach()
            theta = (theta - theta.mean()).clone()

            path = [torch.softmax(theta, dim=0).detach().cpu().numpy()]

            for _ in range(max_steps):
                theta = theta.detach().requires_grad_(True)
                x_soft = torch.softmax(theta, dim=0)
                val    = acq(x_soft.unsqueeze(0).unsqueeze(0))
                try:
                    g_theta = torch.autograd.grad(val.sum(), theta)[0]
                    with torch.no_grad():
                        theta = theta + step_size * g_theta
                        # Re-encode bounds: reflect them back into θ
                        x_new = torch.softmax(theta, dim=0)
                        x_new = _enforce_bounds_tensor(x_new, lo, hi)
                        theta = torch.log(x_new.clamp(min=1e-10))
                        theta = theta - theta.mean()
                    path.append(torch.softmax(theta.detach(), dim=0).cpu().numpy())
                except RuntimeError:
                    break

            with torch.no_grad():
                x_final = torch.softmax(theta.detach(), dim=0)
                x_final = _enforce_bounds_tensor(x_final, lo, hi)
                try:
                    final_val = acq(x_final.unsqueeze(0).unsqueeze(0)).squeeze()
                except RuntimeError:
                    continue

            cands.append(x_final)
            vals.append(final_val)
            paths.append(np.array(path))

        return self._stack(cands, vals, paths, d, bounds)

    # ── Method B: Riemannian gradient (mean-centred) ──────────────────────────
    def _paths_riemannian(
        self, acq: Any, bounds: torch.Tensor,
        init_cond: torch.Tensor,
        max_steps: int = MAX_ASCENT_STEPS,
        step_size: float = STEP_RIEMANNIAN,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[np.ndarray]]:
        lo, hi = bounds[0], bounds[1]
        d      = init_cond.shape[-1]
        cands, vals, paths = [], [], []

        for r in range(init_cond.shape[0]):
            x    = self._init_x(init_cond[r], lo, hi)
            path = [x.detach().cpu().numpy()]

            for _ in range(max_steps):
                try:
                    g, _ = self._acq_grad_flat(acq, x)
                    # Project to tangent space (zero-sum)
                    v = g - g.mean()
                    with torch.no_grad():
                        x = x + step_size * v
                        x = _enforce_bounds_tensor(x, lo, hi)
                    path.append(x.detach().cpu().numpy())
                except RuntimeError:
                    break

            try:
                _, final_val = self._acq_grad_flat(acq, x)
            except RuntimeError:
                continue

            cands.append(x.detach())
            vals.append(final_val.detach())
            paths.append(np.array(path))

        return self._stack(cands, vals, paths, d, bounds)

    # ── Method C: Riemannian + log-barrier ────────────────────────────────────
    def _paths_riemannian_barrier(
        self, acq: Any, bounds: torch.Tensor,
        init_cond: torch.Tensor,
        max_steps: int = MAX_ASCENT_STEPS,
        step_size: float = STEP_BARRIER,
        mu: float = LOG_BARRIER_MU,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[np.ndarray]]:
        lo, hi = bounds[0], bounds[1]
        d      = init_cond.shape[-1]
        cands, vals, paths = [], [], []

        for r in range(init_cond.shape[0]):
            x    = self._init_x(init_cond[r], lo, hi)
            path = [x.detach().cpu().numpy()]

            for _ in range(max_steps):
                try:
                    g_acq, _ = self._acq_grad_flat(acq, x)
                    with torch.no_grad():
                        # Barrier gradient: μ/(xᵢ−lᵢ) − μ/(hᵢ−xᵢ)
                        g_bar = (mu / (x - lo).clamp(min=1e-10)
                                 - mu / (hi - x).clamp(min=1e-10))
                        g_full = g_acq + g_bar
                        v = g_full - g_full.mean()
                        x = x + step_size * v
                        x = _enforce_bounds_tensor(x, lo, hi)
                    path.append(x.detach().cpu().numpy())
                except RuntimeError:
                    break

            try:
                _, final_val = self._acq_grad_flat(acq, x)
            except RuntimeError:
                continue

            cands.append(x.detach())
            vals.append(final_val.detach())
            paths.append(np.array(path))

        return self._stack(cands, vals, paths, d, bounds)

    # ── Method D: Natural gradient / Fisher-Rao exponential map ──────────────
    def _paths_natural_grad(
        self, acq: Any, bounds: torch.Tensor,
        init_cond: torch.Tensor,
        max_steps: int = MAX_ASCENT_STEPS,
        step_size: float = STEP_NATGRAD,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[np.ndarray]]:
        """
        Update rule:  xᵢ ← xᵢ · exp(α · (gᵢ − ḡₓ)) / Z
        where ḡₓ = Σⱼ xⱼ·gⱼ  (x-weighted mean).
        Equivalent to multiplicative weights / Fisher-Rao geodesic.
        """
        lo, hi = bounds[0], bounds[1]
        d      = init_cond.shape[-1]
        cands, vals, paths = [], [], []

        for r in range(init_cond.shape[0]):
            x    = self._init_x(init_cond[r], lo, hi)
            path = [x.detach().cpu().numpy()]

            for _ in range(max_steps):
                try:
                    g, _ = self._acq_grad_flat(acq, x)
                    with torch.no_grad():
                        g_bar_x = (x * g).sum()          # x-weighted mean
                        x = x * torch.exp(step_size * (g - g_bar_x))
                        x = _enforce_bounds_tensor(x, lo, hi)
                    path.append(x.detach().cpu().numpy())
                except RuntimeError:
                    break

            try:
                _, final_val = self._acq_grad_flat(acq, x)
            except RuntimeError:
                continue

            cands.append(x.detach())
            vals.append(final_val.detach())
            paths.append(np.array(path))

        return self._stack(cands, vals, paths, d, bounds)

    # ── Method E: Projected gradient (original baseline) ─────────────────────
    def _paths_projected(
        self, acq: Any, bounds: torch.Tensor,
        init_cond: torch.Tensor,
        max_steps: int = MAX_ASCENT_STEPS,
        step_size: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[np.ndarray]]:
        """
        Original method: Euclidean gradient step → box clamp → proj_simplex.
        Kept as a baseline panel for direct comparison with the geometric methods.
        """
        lo, hi = bounds[0], bounds[1]
        d      = init_cond.shape[-1]
        cands, vals, paths = [], [], []

        for r in range(init_cond.shape[0]):
            x    = self._init_x(init_cond[r], lo, hi)
            path = [x.detach().cpu().numpy()]

            for _ in range(max_steps):
                try:
                    g, _ = self._acq_grad_flat(acq, x)
                    with torch.no_grad():
                        x = x + step_size * g
                        x = torch.clamp(x, min=lo, max=hi)  # box clamp
                        x = self.proj_fn(x.unsqueeze(0).unsqueeze(0)).squeeze()
                        x = x.clamp(min=1e-10)
                        x = x / x.sum()
                    path.append(x.detach().cpu().numpy())
                except RuntimeError:
                    break

            try:
                _, final_val = self._acq_grad_flat(acq, x)
            except RuntimeError:
                continue

            cands.append(x.detach())
            vals.append(final_val.detach())
            paths.append(np.array(path))

        return self._stack(cands, vals, paths, d, bounds)

    # ── utility: stack results ────────────────────────────────────────────────
    @staticmethod
    def _stack(
        cands: List[torch.Tensor],
        vals:  List[torch.Tensor],
        paths: List[np.ndarray],
        d:     int,
        bounds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[np.ndarray]]:
        if not cands:
            return (
                torch.empty(0, d,  device=bounds.device, dtype=bounds.dtype),
                torch.empty(0,     device=bounds.device, dtype=bounds.dtype),
                [],
            )
        return torch.stack(cands), torch.stack(vals), paths

    # ── main get_candidate ────────────────────────────────────────────────────
    def get_candidate(
        self,
        bounds: torch.Tensor,
        best_f: Optional[float] = None,
        max_attempts: int = 5,
        exclude_near: Optional[torch.Tensor] = None,
        exclude_near_tol: float = 1e-8,
    ) -> Optional[torch.Tensor]:

        if self.gp is None:
            raise RuntimeError("GP not fitted.")

        bounds = bounds.to(device=self.device, dtype=self.dtype)
        acq    = self.create_acquisition(best_f=best_f)

        # ── raw sampling ───────────────────────────────────────────────────────
        ic_cands   = self._sample_random(self.raw_samples, bounds)
        ic_cands3d = ic_cands.unsqueeze(1)
        with torch.no_grad():
            acq_vals = acq(ic_cands3d).squeeze()

        upen_mask = self.data_handler.get_penalty_mask(ic_cands)
        upen_idx  = torch.where(upen_mask.squeeze())[0]

        cur_cands, cur_cands3d = ic_cands, ic_cands3d
        cur_acq,   cur_upen    = acq_vals, upen_idx

        attempt = 0
        while len(cur_upen) < self.num_restarts and attempt < max_attempts:
            attempt += 1
            extra   = self._sample_random(self.raw_samples, bounds)
            extra3d = extra.unsqueeze(1)
            with torch.no_grad():
                extra_acq  = acq(extra3d).squeeze()
            extra_upen = torch.where(
                self.data_handler.get_penalty_mask(extra).squeeze())[0]
            offset = cur_cands.shape[0]
            cur_cands   = torch.cat([cur_cands,   extra],           dim=0)
            cur_cands3d = torch.cat([cur_cands3d, extra3d],         dim=0)
            cur_acq     = torch.cat([cur_acq,     extra_acq],       dim=0)
            cur_upen    = torch.cat([cur_upen,    extra_upen + offset])

        if len(cur_upen) == 0 or len(cur_upen) < 0.1 * self.num_restarts:
            return None

        # ── frame 1 ────────────────────────────────────────────────────────────
        all_upen_np = self.data_handler.get_penalty_mask(cur_cands).cpu().numpy()
        self._plot_state.show_random_points(
            pts        = cur_cands.detach().cpu().numpy(),
            acq_vals   = cur_acq.detach().cpu().numpy(),
            unpen_mask = all_upen_np,
            bounds     = bounds,
        )

        # ── select top-k restart seeds ─────────────────────────────────────────
        n_use     = min(self.num_restarts, len(cur_upen))
        top_idx   = torch.argsort(cur_acq[cur_upen], descending=True)[:n_use]
        sel_idx   = cur_upen[top_idx]
        init_cond = cur_cands3d[sel_idx]   # (n_use, 1, d)

        # ── methods A–D: visualization only, no internality filter ──────────────
        VISUAL_METHODS = {
            "A  Softmax reparam":    self._paths_softmax,
            "B  Riemannian grad":    self._paths_riemannian,
            "C  Riemannian+barrier": self._paths_riemannian_barrier,
            "D  Natural grad (Exp)": self._paths_natural_grad,
        }
        method_results: Dict[str, Tuple[List[np.ndarray], Optional[np.ndarray]]] = {}

        for name, fn in VISUAL_METHODS.items():
            cands, vals, paths = fn(acq, bounds, init_cond,
                                    max_steps=MAX_ASCENT_STEPS)
            if cands.shape[0] > 0:
                order = torch.argsort(vals, descending=True)
                cands = cands[order]
                paths = [paths[i] for i in order.cpu().numpy()]
                # Show raw best regardless of whether it is on the boundary
                method_results[name] = (paths, cands[0].detach().cpu().numpy())
            else:
                method_results[name] = ([], None)

        # ── method E: internality check + retry logic (actual candidate) ──────
        d_dim        = init_cond.shape[-1]
        ascent_steps = MAX_ASCENT_STEPS
        e_cands      = torch.empty(0, d_dim, device=bounds.device, dtype=bounds.dtype)
        e_vals       = torch.empty(0,        device=bounds.device, dtype=bounds.dtype)

        while True:
            cands, vals, paths = self._paths_projected(
                acq, bounds, init_cond, max_steps=ascent_steps)

            if cands.shape[0] > 0:
                order  = torch.argsort(vals, descending=True)
                cands  = cands[order]
                vals   = vals[order]
                paths  = [paths[i] for i in order.cpu().numpy()]
                int_m  = self._internal_mask(cands, bounds)
                if int_m.any():
                    e_cands = cands[int_m]
                    e_vals  = vals[int_m]
                    method_results["E  Projected (baseline)"] = (
                        paths, e_cands[0].detach().cpu().numpy())
                    break
                else:
                    method_results["E  Projected (baseline)"] = (
                        paths, cands[0].detach().cpu().numpy())

            if ascent_steps == 0:
                break

            print(
                f"  [GP] E (projected): all candidates on boundary "
                f"(steps={ascent_steps}) — reducing by 50",
                flush=True,
            )
            ascent_steps = max(0, ascent_steps - 50)

        # ── select final candidate exclusively from E's internal pool ─────────
        if e_cands.shape[0] > 0:
            all_c, all_v = e_cands, e_vals
        else:
            # Final fallback: best raw unpenalised internal point
            raw_upen = cur_cands[cur_upen]
            raw_acq  = cur_acq[cur_upen]
            int_raw  = self._internal_mask(raw_upen, bounds)
            if int_raw.any():
                all_c = raw_upen[int_raw]
                all_v = raw_acq[int_raw]
                method_results["E  Projected (baseline)"] = (
                    [all_c[0].detach().cpu().numpy().reshape(1, -1)],
                    all_c[0].detach().cpu().numpy(),
                )
                print("  [GP] E fallback → best internal raw candidate", flush=True)
            else:
                return None

        # Deduplicate and sort
        order  = torch.argsort(all_v, descending=True)
        all_c  = all_c[order]
        all_v  = all_v[order]

        # exclude_near filtering
        if exclude_near is not None:
            excl = exclude_near.to(device=all_c.device, dtype=all_c.dtype)
            if excl.dim() == 1:
                excl = excl.unsqueeze(0)
            dists   = torch.norm(all_c.unsqueeze(1) - excl.unsqueeze(0), dim=2)
            allowed = (dists >= exclude_near_tol).all(dim=1)
            idx     = int(torch.where(allowed)[0][0]) if allowed.any() else 0
        else:
            idx = 0

        best_candidate = all_c[idx]
        best_value     = all_v[idx]
        best_np        = best_candidate.detach().cpu().numpy()

        # ── frame 2: 4-panel ascent comparison ────────────────────────────────
        self._plot_state.show_ascent_4panel(
            method_results  = method_results,
            bounds          = bounds,
            overall_best_pt = best_np,
        )

        print(
            f"  [GP] best = {best_np.tolist()}  acq = {best_value.item():.6f}",
            flush=True,
        )

        is_valid = self.data_handler.get_penalty_mask(best_candidate.unsqueeze(0))
        if not is_valid.any():
            return None
        return best_candidate


# ============================================================================
# LineBO objective closure
# ============================================================================

def make_linebo_objective(
    plot_state:    _TernaryPlot,
    candidate_ref: List[Optional[np.ndarray]],
    bounds_ref:    List[Optional[torch.Tensor]],
    device:        torch.device,
    dtype:         torch.dtype,
):
    def _objective(endpoints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert endpoints.dim() == 3 and endpoints.shape[1] == 2
        left  = endpoints[0, 0].cpu().numpy()
        right = endpoints[0, 1].cpu().numpy()

        t_vals = np.linspace(0.0, 1.0, NUM_LINE_POINTS)
        x_line = np.stack([left + t * (right - left) for t in t_vals])
        y_line = ackley_simplex(x_line)

        candidate = candidate_ref[0]
        if candidate is None:
            candidate = x_line[np.argmax(y_line)]

        best_idx = int(np.argmax(y_line))
        b        = bounds_ref[0]
        lo_str   = np.round(b[0].cpu().numpy(), 4).tolist() if b is not None else "?"
        hi_str   = np.round(b[1].cpu().numpy(), 4).tolist() if b is not None else "?"
        print(
            f"\n[LineBO objective]"
            f"\n  zoom lo = {lo_str}"
            f"\n  zoom hi = {hi_str}"
            f"\n  left    = {np.round(left,  4).tolist()}"
            f"\n  right   = {np.round(right, 4).tolist()}"
            f"\n  best pt = {np.round(x_line[best_idx], 4).tolist()}"
            f"  −Ackley={y_line[best_idx]:.4f}",
            flush=True,
        )

        plot_state.update(candidate, x_line, y_line)
        time.sleep(DELAY_SECONDS)

        return (
            torch.tensor(x_line, device=device, dtype=dtype),
            torch.tensor(y_line, device=device, dtype=dtype),
        )
    return _objective


# ============================================================================
# Sampler wrapper
# ============================================================================

def _fmt_bounds(bounds: torch.Tensor) -> str:
    lo = np.round(bounds[0].cpu().numpy(), 5)
    hi = np.round(bounds[1].cpu().numpy(), 5)
    w  = max(len(f"{v:.5f}") for v in list(lo) + list(hi))
    return (
        f"  lo: [{'  '.join(f'{v:{w}.5f}' for v in lo)}]\n"
        f"  hi: [{'  '.join(f'{v:{w}.5f}' for v in hi)}]"
    )


def make_sampler_wrapper(
    dimensions: int,
    num_lines:  int          = 10,
    device:     torch.device = torch.device("cpu"),
    dtype:      torch.dtype  = torch.float64,
) -> Tuple[Any, _TernaryPlot]:
    run_ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = GRADIENT_LOG_DIR / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  gradient log → {run_dir}", flush=True)

    plot_state    = _TernaryPlot(run_dir=run_dir)
    plot_state._setup()

    candidate_ref:   List[Optional[np.ndarray]]   = [None]
    bounds_ref:      List[Optional[torch.Tensor]] = [None]
    call_count       = [0]
    prev_bounds_repr = [None]

    inner_obj = make_linebo_objective(plot_state, candidate_ref, bounds_ref,
                                      device, dtype)
    linebo = LineBO(inner_obj, dimensions,
                   num_points_per_line=100, num_lines=num_lines,
                   device=str(device))

    def wrapper(
        x_tell: torch.Tensor,
        bounds: torch.Tensor,
        acquisition_function: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_ref[0] = x_tell.cpu().numpy()
        bounds_ref[0]    = bounds
        call_count[0]   += 1

        bounds_repr    = _fmt_bounds(bounds)
        bounds_changed = bounds_repr != prev_bounds_repr[0]
        prev_bounds_repr[0] = bounds_repr

        print("\n" + "━" * 60, flush=True)
        print(f"  Objective call #{call_count[0]}", flush=True)
        print(f"  candidate = {np.round(candidate_ref[0], 5).tolist()}", flush=True)
        tag = " (UPDATED)" if bounds_changed else ""
        print(f"  bounds{tag}:\n{bounds_repr}", flush=True)
        print("━" * 60, flush=True)

        x_requested, x_actual, y = linebo.sampler(x_tell, bounds,
                                                   acquisition_function)
        y_flat = y.reshape(-1)
        print(
            f"  → {y_flat.shape[0]} pts | "
            f"y ∈ [{y_flat.min().item():.4f}, {y_flat.max().item():.4f}]",
            flush=True,
        )
        return x_requested, x_actual, y_flat

    return wrapper, plot_state


# ============================================================================
# Simplex utilities
# ============================================================================

def initial_lines_on_boundary(num_lines, bounds, device, dtype=torch.float64,
                               max_retries=10):
    d = bounds.shape[1]
    low, high = bounds[0], bounds[1]
    points = ZoMBIHop.random_simplex(num_lines, low, high, device=str(device),
                                      torch_dtype=dtype)
    eps_list = []
    for i in range(num_lines):
        x0 = points[i]
        for _ in range(max_retries):
            direction = zero_sum_dirs(1, d, device=str(device), dtype=dtype).squeeze(0)
            seg = line_simplex_segment(x0, direction)
            if seg is not None:
                _, _, xl, xr = seg
                eps_list.append([xl.cpu().numpy(), xr.cpu().numpy()])
                break
        else:
            ei = torch.zeros(d, device=device, dtype=dtype)
            ej = torch.zeros(d, device=device, dtype=dtype)
            ei[i % d] = 1.0;  ej[(i + 1) % d] = 1.0
            eps_list.append([ei.cpu().numpy(), ej.cpu().numpy()])
    return np.array(eps_list)


def expected_from_actual(x_actual: torch.Tensor) -> torch.Tensor:
    if x_actual.shape[0] > 1:
        x_c = x_actual - x_actual.mean(dim=0, keepdim=True)
        _, _, V = torch.linalg.svd(x_c, full_matrices=False)
        direction = V[0]
        proj = (x_c @ direction.unsqueeze(1)).squeeze(1)
        t = torch.linspace(proj.min().item(), proj.max().item(),
                           x_actual.shape[0],
                           device=x_actual.device, dtype=x_actual.dtype)
        return x_actual.mean(0, keepdim=True) + t.unsqueeze(1) * direction.unsqueeze(0)
    return x_actual.clone()


# ============================================================================
# Main runner
# ============================================================================

def run_interactive_3d() -> None:
    device = torch.device(DEVICE)
    dtype  = DTYPE

    bounds = torch.zeros((2, DIMENSIONS), device=device, dtype=dtype)
    bounds[1] = 1.0

    sampler_wrapper, plot_state = make_sampler_wrapper(
        dimensions=DIMENSIONS, num_lines=10, device=device, dtype=dtype,
    )

    print("=" * 70, flush=True)
    print("INTERACTIVE TEST  |  Ackley 3D  |  4 ascent methods compared",
          flush=True)
    print("=" * 70, flush=True)
    print(f"  dimensions : {DIMENSIONS}",          flush=True)
    print(f"  device     : {DEVICE}",               flush=True)
    print(f"  initial bounds:\n{_fmt_bounds(bounds)}", flush=True)
    print(f"  line pts / iter       : {NUM_LINE_POINTS}",    flush=True)
    print(f"  delay (sampling)      : {DELAY_SECONDS}s",     flush=True)
    print(f"  hold  (ascent 4-panel): {ASCENT_HOLD_SECONDS}s", flush=True)
    print(f"  step sizes: softmax={STEP_SOFTMAX}  riemannian={STEP_RIEMANNIAN}"
          f"  barrier={STEP_BARRIER}  natgrad={STEP_NATGRAD}", flush=True)
    print("=" * 70, flush=True)

    print("\nGenerating initial seed lines...", flush=True)
    ordered_ep = initial_lines_on_boundary(
        2 * NUM_INIT_DATA, bounds, device, dtype=dtype
    )
    n_total = len(ordered_ep)
    x_act_list, x_exp_list, y_list = [], [], []

    for i in range(NUM_INIT_DATA):
        idx0, idx1 = 2 * i, 2 * i + 1
        line0, line1 = ordered_ep[idx0], ordered_ep[idx1]
        if np.allclose(line0, line1, rtol=1e-6, atol=1e-8):
            idx1 = (2 * i + 2) % n_total
            if idx1 == idx0:
                idx1 = (idx0 + 1) % n_total
            line1 = ordered_ep[idx1]

        left, right = line0[0], line0[1]
        t_vals = np.linspace(0.0, 1.0, NUM_LINE_POINTS)
        x_line = np.stack([left + t * (right - left) for t in t_vals])
        y_line = ackley_simplex(x_line)

        x_act = torch.tensor(x_line, device=device, dtype=dtype)
        y_act = torch.tensor(y_line, device=device, dtype=dtype)
        x_act_list.append(x_act)
        x_exp_list.append(expected_from_actual(x_act))
        y_list.append(y_act)
        print(
            f"  seed {i+1}/{NUM_INIT_DATA}  "
            f"left={np.round(left, 4).tolist()}  "
            f"right={np.round(right, 4).tolist()}  "
            f"best −Ackley={y_line.max():.4f}",
            flush=True,
        )

    X_init_actual   = torch.cat(x_act_list, dim=0)
    X_init_expected = torch.cat(x_exp_list, dim=0)
    Y_init          = torch.cat(y_list,     dim=0).reshape(-1, 1)

    optimizer = ZoMBIHop(
        objective                  = sampler_wrapper,
        bounds                     = bounds,
        X_init_actual              = X_init_actual,
        X_init_expected            = X_init_expected,
        Y_init                     = Y_init,
        max_zooms                  = 3,
        max_iterations             = 6,
        top_m_points               = max(DIMENSIONS + 1, 4),
        n_restarts                 = 30,
        raw                        = 400,
        penalization_threshold     = 6.5e-5,
        penalty_num_directions     = 10 * DIMENSIONS,
        penalty_max_radius         = 0.33633,
        penalty_radius_step        = None,
        convergence_pi_threshold   = 4.8e-5,
        input_noise_threshold_mult = 2.0,
        output_noise_threshold_mult= 0.5,
        n_consecutive_converged    = 5,
        max_gp_points              = 2000,
        acquisition_type           = "ucb",
        ucb_beta                   = 0.1,
        device                     = str(device),
        dtype                      = dtype,
        run_uuid                   = None,
        checkpoint_dir             = None,
        verbose                    = True,
    )

    dh = optimizer.data_handler
    optimizer.gp_handler = InstrumentedGPSimplex(
        plot_state       = plot_state,
        data_handler     = dh,
        proj_fn          = optimizer.proj_fn,
        random_sampler   = optimizer.random_sampler,
        num_restarts     = dh.n_restarts,
        raw_samples      = dh.raw,
        repulsion_lambda = dh.repulsion_lambda,
        acquisition_type = dh.acquisition_type,
        ucb_beta         = dh.ucb_beta,
        device           = str(device),
        dtype            = dtype,
    )

    print(f"\n✅  ZoMBI-Hop initialised  (UUID: {optimizer.run_uuid})", flush=True)
    print("=" * 70, flush=True)
    print("STARTING OPTIMISATION  (Ctrl-C to stop)", flush=True)
    print("=" * 70 + "\n", flush=True)

    optimizer.run(max_activations=float("inf"), time_limit_hours=None)


if __name__ == "__main__":
    run_interactive_3d()
