"""Visualize 3D S1 pilot runs: ternary landscapes, evolution, Tier-1 recovery."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ela.evolve_context import EvolutionContext, load_context_from_run
from ela.features import composition_to_ilr, train_rf_surrogate
from ela.gp_tree import Node, predict_calibrated, predict_tree, tree_from_jsonable
from ela.tier1 import TIER1_NAMES
from visualization.needle_overlay import comp_to_xy, ternary_grid

logger = logging.getLogger(__name__)

CORNER_LABELS = ("FAPbI3", "MAPbI3", "MAPbBr3")
_SQRT3_2 = np.sqrt(3) / 2


@dataclass
class LandscapePlotCache:
    """Fixed ternary grid + RF target for fast per-generation landscape plots."""

    grid_pts: np.ndarray
    z_grid: np.ndarray
    y_target_grid: np.ndarray
    tri: mtri.Triangulation
    vmin: float
    vmax: float
    y_min: float
    y_max: float

    @classmethod
    def build(cls, ctx: EvolutionContext, *, grid_n: int = 100) -> LandscapePlotCache:
        grid_pts = ternary_grid(grid_n)
        z_grid = composition_to_ilr(grid_pts)
        rf = train_rf_surrogate(ctx.x_campaign, ctx.y_campaign)
        y_target_grid = rf.predict(grid_pts)
        xy = comp_to_xy(grid_pts)
        vmin, vmax = float(y_target_grid.min()), float(y_target_grid.max())
        return cls(
            grid_pts=grid_pts,
            z_grid=z_grid,
            y_target_grid=y_target_grid,
            tri=mtri.Triangulation(xy[:, 0], xy[:, 1]),
            vmin=vmin,
            vmax=vmax,
            y_min=ctx.y_min,
            y_max=ctx.y_max,
        )

    def plot_generation(
        self,
        tree: Node,
        out_path: Path,
        *,
        generation: int,
        fitness: float,
        tier1_loss: float,
        subspace_rmse: float,
        accepted: bool,
        calib: tuple[float, float] | None = None,
        y_ref: np.ndarray | None = None,
    ) -> None:
        """Triptych: RF | best GP this generation | |GP−RF| diagnostic."""
        if calib is not None:
            y_evolved, _ = predict_calibrated(tree, self.z_grid, calib=calib)
        elif y_ref is not None:
            y_evolved, _ = predict_calibrated(tree, self.z_grid, y_ref=y_ref)
        else:
            y_evolved = predict_tree(tree, self.z_grid, y_min=self.y_min, y_max=self.y_max)
        residual = np.abs(y_evolved - self.y_target_grid)
        rmax = float(np.percentile(residual, 99))

        fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
        panels = [
            (self.y_target_grid, "RF target", "viridis", self.vmin, self.vmax),
            (y_evolved, "Best GP", "viridis", self.vmin, self.vmax),
            (residual, "|GP − RF|", "magma", 0.0, rmax),
        ]
        for ax, (vals, subtitle, cmap, lo, hi) in zip(axes, panels):
            _draw_ternary_frame(ax, pad=0.03)
            pc = ax.tripcolor(
                self.tri, vals, cmap=cmap, vmin=lo, vmax=hi,
                shading="gouraud", rasterized=True,
            )
            ax.set_title(subtitle, fontsize=9)
            fig.colorbar(pc, ax=ax, fraction=0.05, pad=0.02)

        status = "accepted" if accepted else "in progress"
        fig.suptitle(
            f"gen {generation:03d} | fitness={fitness:.3f} tier1={tier1_loss:.3f} "
            f"rmse={subspace_rmse:.4f} | {status}",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)


def _draw_ternary_frame(ax: plt.Axes, *, pad: float = 0.04) -> None:
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.axis("off")
    ax.text(-pad, -pad, CORNER_LABELS[0], ha="right", va="top", fontsize=9)
    ax.text(1 + pad, -pad, CORNER_LABELS[1], ha="left", va="top", fontsize=9)
    ax.text(0.5, _SQRT3_2 + pad, CORNER_LABELS[2], ha="center", va="bottom", fontsize=9)


def _load_expression(run_dir: Path):
    expr_path = run_dir / "best" / "expression.json"
    if not expr_path.is_file():
        raise FileNotFoundError(f"Missing {expr_path}")
    with expr_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return tree_from_jsonable(data["expression"])


def _load_history(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "evolution" / "history.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _shared_vmin_vmax(*arrays: np.ndarray) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    return float(np.min(vals)), float(np.max(vals))


def plot_ternary_triptych(
    grid_pts: np.ndarray,
    y_target: np.ndarray,
    y_evolved: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> None:
    """RF target | evolved GP | absolute residual on a ternary grid."""
    residual = np.abs(y_evolved - y_target)
    vmin, vmax = _shared_vmin_vmax(y_target, y_evolved)
    rmax = float(np.percentile(residual, 99))

    xy = comp_to_xy(grid_pts)
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    panels = [
        (y_target, "RF surrogate (target)", "viridis", vmin, vmax),
        (y_evolved, "Evolved GP landscape", "viridis", vmin, vmax),
        (residual, "|evolved − RF|", "magma", 0.0, rmax),
    ]
    for ax, (vals, subtitle, cmap, lo, hi) in zip(axes, panels):
        _draw_ternary_frame(ax)
        pc = ax.tripcolor(tri, vals, cmap=cmap, vmin=lo, vmax=hi, shading="gouraud", rasterized=True)
        ax.set_title(subtitle, fontsize=10)
        fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_ternary_campaign_overlay(
    grid_pts: np.ndarray,
    y_evolved: np.ndarray,
    x_campaign: np.ndarray,
    y_campaign: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> None:
    xy_g = comp_to_xy(grid_pts)
    xy_c = comp_to_xy(x_campaign)
    tri = mtri.Triangulation(xy_g[:, 0], xy_g[:, 1])
    vmin, vmax = float(y_evolved.min()), float(y_evolved.max())

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    _draw_ternary_frame(ax)
    pc = ax.tripcolor(
        tri, y_evolved, cmap="viridis", vmin=vmin, vmax=vmax,
        shading="gouraud", rasterized=True, zorder=2,
    )
    sc = ax.scatter(
        xy_c[:, 0], xy_c[:, 1], c=y_campaign, cmap="viridis",
        vmin=vmin, vmax=vmax, s=22, edgecolors="white", linewidths=0.4,
        zorder=5, label="measured campaign",
    )
    ax.set_title(title, fontsize=11)
    fig.colorbar(pc, ax=ax, label="Objective", fraction=0.046, pad=0.04)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_dense_scatter(
    y_target: np.ndarray,
    y_evolved: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.scatter(y_target, y_evolved, s=8, alpha=0.35, c="#4477aa", linewidths=0)
    lo = float(min(y_target.min(), y_evolved.min()))
    hi = float(max(y_target.max(), y_evolved.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.6, label="y = x")
    rmse = float(np.sqrt(np.mean((y_evolved - y_target) ** 2)))
    ax.set_xlabel("RF surrogate (dense sample)")
    ax.set_ylabel("Evolved GP (dense sample)")
    ax.set_title(f"{title}\nRMSE = {rmse:.5f}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_evolution_history(history: list[dict[str, Any]], out_path: Path) -> None:
    if not history:
        return
    gens = [int(row["generation"]) for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))

    series = [
        (axes[0, 0], "best_fitness", "Best fitness", "#cc6677"),
        (axes[0, 1], "best_tier1_loss", "Best Tier-1 loss", "#44aa99"),
        (axes[1, 0], "best_subspace_rmse", "Best subspace RMSE", "#117733"),
        (axes[1, 1], "best_complexity", "Best tree size", "#332288"),
    ]
    for ax, key, label, color in series:
        ys = [float(row[key]) for row in history]
        ax.plot(gens, ys, "-o", ms=3, lw=1.5, color=color)
        ax.set_xlabel("generation")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Evolution progress", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tier1_recovery(recovery: dict[str, Any], out_path: Path) -> None:
    names = list(TIER1_NAMES)
    targets = [float(recovery[n]["target"]) for n in names]
    achieved = [float(recovery[n]["achieved"]) for n in names]
    rel_err = [float(recovery[n]["rel_err"]) for n in names]

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.bar(x - width / 2, targets, width, label="target (RF λ_T)", color="#4477aa")
    ax.bar(x + width / 2, achieved, width, label="evolved g", color="#cc6677")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_title("Tier-1 feature recovery")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for i, err in enumerate(rel_err):
        label = f"{err:.0%}" if err <= 1.5 else f"{err:.2f}"
        ymax = max(targets[i], achieved[i])
        pad = 0.05 * abs(ymax) if ymax != 0 else 0.05
        ax.text(i, ymax + pad, label, ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ilr_residual(
    z_dense: np.ndarray,
    residual: np.ndarray,
    out_path: Path,
) -> None:
    if z_dense.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sc = ax.scatter(
        z_dense[:, 0], z_dense[:, 1], c=residual, cmap="magma",
        s=10, alpha=0.6, linewidths=0,
    )
    ax.set_xlabel("ILR z₀")
    ax.set_ylabel("ILR z₁")
    ax.set_title("Dense-sample residual in ILR coordinates")
    fig.colorbar(sc, ax=ax, label="|evolved − RF|")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_run(
    run_dir: str | Path,
    *,
    grid_n: int = 200,
    out_subdir: str = "viz",
) -> Path:
    """Generate all 3D pilot visualizations for a completed run."""
    run_dir = Path(run_dir).resolve()
    viz_dir = run_dir / out_subdir
    viz_dir.mkdir(parents=True, exist_ok=True)

    ctx = load_context_from_run(run_dir)
    expr_path = run_dir / "best" / "expression.json"
    with expr_path.open(encoding="utf-8") as f:
        expr_meta = json.load(f)
    tree = tree_from_jsonable(expr_meta["expression"])
    cal = expr_meta.get("linear_calibration", {})
    calib = (float(cal.get("a", 1.0)), float(cal.get("b", 0.0)))

    y_evolved_dense, _ = predict_calibrated(tree, ctx.z_dense, calib=calib)

    grid_pts = ternary_grid(grid_n)
    z_grid = composition_to_ilr(grid_pts)
    rf = train_rf_surrogate(ctx.x_campaign, ctx.y_campaign)
    y_target_grid = rf.predict(grid_pts)
    y_evolved_grid, _ = predict_calibrated(tree, z_grid, calib=calib)

    run_name = run_dir.name
    plot_ternary_triptych(
        grid_pts, y_target_grid, y_evolved_grid,
        viz_dir / "ternary_target_vs_evolved.png",
        title=f"{run_name} — RF target vs evolved GP",
    )
    plot_ternary_campaign_overlay(
        grid_pts, y_evolved_grid, ctx.x_campaign, ctx.y_campaign,
        viz_dir / "ternary_campaign_overlay.png",
        title=f"{run_name} — evolved landscape + measured campaign",
    )
    plot_dense_scatter(
        ctx.y_target, y_evolved_dense,
        viz_dir / "dense_scatter.png",
        title=f"{run_name} — dense Sobol sample",
    )
    plot_ilr_residual(
        ctx.z_dense,
        np.abs(y_evolved_dense - ctx.y_target),
        viz_dir / "ilr_residual.png",
    )

    history = _load_history(run_dir)
    if history:
        plot_evolution_history(history, viz_dir / "evolution_history.png")

    recovery_path = run_dir / "best" / "recovery.json"
    if recovery_path.is_file():
        with recovery_path.open(encoding="utf-8") as f:
            recovery = json.load(f)
        plot_tier1_recovery(recovery, viz_dir / "tier1_recovery.png")

    manifest = {
        "run_dir": str(run_dir),
        "grid_n": grid_n,
        "plots": sorted(p.name for p in viz_dir.glob("*.png")),
    }
    with (viz_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    logger.info("Wrote %d plots to %s", len(manifest["plots"]), viz_dir)
    return viz_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize a 3D S1 pilot run")
    parser.add_argument("run_dir", type=Path, help="Pilot run directory")
    parser.add_argument("--grid-n", type=int, default=200, help="Ternary grid resolution")
    parser.add_argument("--out-subdir", default="viz", help="Subdirectory under run_dir for plots")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    viz_dir = visualize_run(args.run_dir, grid_n=args.grid_n, out_subdir=args.out_subdir)
    print(f"Plots -> {viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
