#!/usr/bin/env python3
"""Plot measured 3D data, an ELA landscape, and an RF sampled from the ELA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ela.features import (  # noqa: E402
    composition_to_ilr,
    load_campaign_rows,
    sample_simplex_sobol,
    train_rf_surrogate,
)
from ela.gp_tree import predict_raw_clipped, tree_from_jsonable  # noqa: E402
from visualization.needle_overlay import comp_to_xy, ternary_grid  # noqa: E402

CORNER_LABELS = ("FAPbI3", "MAPbI3", "MAPbBr3")
SQRT3_2 = np.sqrt(3.0) / 2.0


def draw_ternary_frame(ax: plt.Axes) -> None:
    ax.plot([0, 1, 0.5, 0], [0, 0, SQRT3_2, 0], color="black", lw=1.1)
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.16, SQRT3_2 + 0.12)
    ax.axis("off")
    ax.text(0, -0.075, CORNER_LABELS[0], ha="center", va="top", fontsize=9)
    ax.text(1, -0.075, CORNER_LABELS[1], ha="center", va="top", fontsize=9)
    ax.text(0.5, SQRT3_2 + 0.035, CORNER_LABELS[2], ha="center", va="bottom", fontsize=9)


def load_generation_tree(run_dir: Path, generation: int):
    expression_path = run_dir / "best" / "expression.json"
    data = json.loads(expression_path.read_text(encoding="utf-8"))
    actual_generation = int(data.get("generation", -1))
    if actual_generation != generation:
        raise ValueError(
            f"{expression_path} is generation {actual_generation}, not requested "
            f"generation {generation}"
        )
    return tree_from_jsonable(data["expression"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT / "ela" / "runs" / "ela_3d_18430082",
        help="ELA run directory",
    )
    parser.add_argument(
        "--campaign-db",
        type=Path,
        default=ROOT / "data" / "2nd_real_run.db",
        help="Measured 3D campaign database",
    )
    parser.add_argument("--generation", type=int, default=75)
    parser.add_argument("--n-rf-samples", type=int, default=650)
    parser.add_argument("--sample-seed", type=int, default=75)
    parser.add_argument("--grid-n", type=int, default=180)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--rf-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run.resolve()
    tree = load_generation_tree(run_dir, args.generation)
    x_campaign, y_campaign = load_campaign_rows(args.campaign_db.resolve())

    x_train = sample_simplex_sobol(3, args.n_rf_samples, seed=args.sample_seed)
    y_train = predict_raw_clipped(tree, composition_to_ilr(x_train))
    rf = train_rf_surrogate(
        x_train,
        y_train,
        n_estimators=args.n_estimators,
        random_state=args.rf_seed,
    )

    grid = ternary_grid(args.grid_n)
    grid_xy = comp_to_xy(grid)
    triangulation = mtri.Triangulation(grid_xy[:, 0], grid_xy[:, 1])
    y_ela = predict_raw_clipped(tree, composition_to_ilr(grid))
    y_rf = rf.predict(grid)

    # Evaluate on a separate low-discrepancy sample, not RF training rows.
    x_test = sample_simplex_sobol(3, 4096, seed=args.sample_seed + 1)
    y_test = predict_raw_clipped(tree, composition_to_ilr(x_test))
    y_test_rf = rf.predict(x_test)
    metrics = {
        "r2": float(r2_score(y_test, y_test_rf)),
        "rmse": float(root_mean_squared_error(y_test, y_test_rf)),
        "mae": float(mean_absolute_error(y_test, y_test_rf)),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1))

    draw_ternary_frame(axes[0])
    campaign_xy = comp_to_xy(x_campaign)
    measured = axes[0].scatter(
        campaign_xy[:, 0],
        campaign_xy[:, 1],
        c=y_campaign,
        cmap="viridis",
        s=18,
        edgecolors="white",
        linewidths=0.25,
        alpha=0.9,
        rasterized=True,
    )
    axes[0].set_title(f"Measured 3D runs\nn = {len(y_campaign)}", fontsize=11)
    fig.colorbar(measured, ax=axes[0], fraction=0.046, pad=0.02, label="Objective")

    ela_lo, ela_hi = float(min(y_ela.min(), y_rf.min())), float(max(y_ela.max(), y_rf.max()))
    for ax, values, title in (
        (axes[1], y_ela, f"ELA 18430082\nGeneration {args.generation}"),
        (
            axes[2],
            y_rf,
            f"RF surrogate from ELA\nn = {args.n_rf_samples}, test R² = {metrics['r2']:.3f}",
        ),
    ):
        draw_ternary_frame(ax)
        surface = ax.tripcolor(
            triangulation,
            values,
            cmap="viridis",
            vmin=ela_lo,
            vmax=ela_hi,
            shading="gouraud",
            rasterized=True,
        )
        ax.set_title(title, fontsize=11)
        fig.colorbar(surface, ax=ax, fraction=0.046, pad=0.02, label="Landscape value")

    train_xy = comp_to_xy(x_train)
    axes[2].scatter(
        train_xy[:, 0],
        train_xy[:, 1],
        s=2,
        c="black",
        alpha=0.16,
        linewidths=0,
        rasterized=True,
    )

    fig.suptitle(
        "Measured campaign, evolved ELA landscape, and sampled RF approximation",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()

    output = args.output
    if output is None:
        output = run_dir / "viz" / f"gen{args.generation}_data_rf{args.n_rf_samples}.png"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    metadata = {
        "output": str(output),
        "ela_run": str(run_dir),
        "generation": args.generation,
        "campaign_db": str(args.campaign_db.resolve()),
        "n_campaign": int(len(y_campaign)),
        "n_rf_samples": args.n_rf_samples,
        "sample_seed": args.sample_seed,
        "n_estimators": args.n_estimators,
        "rf_seed": args.rf_seed,
        "test_sample_size": int(len(y_test)),
        "test_metrics": metrics,
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Plot: {output}")
    print(f"Metadata: {metadata_path}")
    print(
        f"RF held-out fidelity: R²={metrics['r2']:.4f}, "
        f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
