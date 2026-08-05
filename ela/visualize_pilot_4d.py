#!/usr/bin/env python3
"""Visualize 4D S1 pilot runs as tetrahedron (4-simplex) landscapes.

Writes interactive Plotly HTML (rotatable) and a static matplotlib PNG into
``<run>/viz/``:

  * ``tetra_campaign_rf.html`` / ``.png`` — campaign RF surrogate
  * ``tetra_expression.html`` / ``.png`` — evolved symbolic ``g(z)``
  * ``tetra_rf_g.html`` / ``.png`` — RF(g) surface (MOBO objective when
    ``rf_transform_features`` is on)
  * ``tetra_triptych.html`` — three side-by-side scenes

Works from ``best/expression.json`` or the latest ``evolution/snapshots/gen_*.json``
(so incomplete runs that never wrote ``best/`` still visualize).

Usage:
  python ela/visualize_pilot_4d.py --run ela/runs/ela_4d_19448443
  python ela/visualize_pilot_4d.py --run ela/runs/ela_4d_19448443 --grid-n 36
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20_000))

from ela.compile_rf_surrogate_gallery import load_landscape_source  # noqa: E402
from ela.evolve_context import load_context_from_run  # noqa: E402
from ela.features import (  # noqa: E402
    composition_to_ilr,
    rf_transform_predict,
    train_rf_surrogate,
)
import synthetic_data.plot_ackley as pc4  # noqa: E402

DEFAULT_GRID_N = 36
DEFAULT_CORNER_LABELS = ("Comp1", "Comp2", "Comp3", "Comp4")


def _corner_labels(run_dir: Path, dim: int) -> tuple[str, ...]:
    meta = ROOT / "data" / "run_9dfe_rf_meta.json"
    if dim == 4 and meta.is_file():
        cols = json.loads(meta.read_text()).get("composition_columns")
        if isinstance(cols, list) and len(cols) == 4:
            return tuple(str(c) for c in cols)
    cfg = run_dir / "pilot_config.source.json"
    if cfg.is_file():
        raw = json.loads(cfg.read_text())
        db = str((raw.get("data") or {}).get("db", ""))
        if "run_9dfe" in db:
            return DEFAULT_CORNER_LABELS
    if dim == 4:
        return DEFAULT_CORNER_LABELS
    return tuple(f"x{i + 1}" for i in range(dim))


def _vertex_labels_trace(labels: tuple[str, ...]):
    import plotly.graph_objects as go

    pos = pc4.TETRA_VERTICES * 1.12
    return go.Scatter3d(
        x=pos[:, 0],
        y=pos[:, 1],
        z=pos[:, 2],
        mode="text",
        text=list(labels),
        textfont=dict(size=16, color="black"),
        name="vertices",
        hoverinfo="skip",
        showlegend=False,
    )


def _cloud_trace(
    comp: np.ndarray,
    values: np.ndarray,
    *,
    name: str,
    colorscale: str = "Viridis",
    cmin: float | None = None,
    cmax: float | None = None,
    colorbar_title: str = "Objective",
):
    import plotly.graph_objects as go

    xyz = pc4.to_3d(comp)
    vmin = float(np.min(values) if cmin is None else cmin)
    vmax = float(np.max(values) if cmax is None else cmax)
    hover = [
        f"x=[{a:.3f}, {b:.3f}, {c:.3f}, {d:.3f}]<br>{name}={v:.4f}"
        for (a, b, c, d), v in zip(comp, values)
    ]
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        name=name,
        text=hover,
        hoverinfo="text",
        marker=dict(
            color=values,
            colorscale=colorscale,
            cmin=vmin,
            cmax=vmax,
            size=pc4.MARKER_SIZE,
            opacity=pc4.MARKER_OPACITY,
            showscale=True,
            colorbar=dict(title=colorbar_title, len=0.65),
        ),
    )


def _write_html(
    out_path: Path,
    comp: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    labels: tuple[str, ...],
    cmin: float | None = None,
    cmax: float | None = None,
    colorbar_title: str = "Objective",
) -> Path:
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            _cloud_trace(
                comp,
                values,
                name="objective",
                cmin=cmin,
                cmax=cmax,
                colorbar_title=colorbar_title,
            ),
            pc4.tetra_edges_trace(),
            _vertex_labels_trace(labels),
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        legend=dict(x=0.0, y=1.0),
        width=pc4.FIG_W,
        height=pc4.FIG_H,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", auto_open=False)
    return out_path


def _write_png(
    out_path: Path,
    comp: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    labels: tuple[str, ...],
    vmin: float,
    vmax: float,
    top_frac: float = 0.0,
) -> Path:
    """Static tetrahedron scatter (optionally keep only the top quantile)."""
    fig = plt.figure(figsize=(8.2, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        ax.plot(
            *zip(pc4.TETRA_VERTICES[i], pc4.TETRA_VERTICES[j]),
            color="black",
            lw=1.0,
        )
    centroid = pc4.TETRA_VERTICES.mean(axis=0)
    for v, name in zip(pc4.TETRA_VERTICES, labels):
        p = v + 0.14 * (v - centroid)
        ax.text(p[0], p[1], p[2], name, ha="center", va="center", fontsize=10)

    keep = np.ones(len(values), dtype=bool)
    if top_frac > 0:
        keep = values >= np.percentile(values, 100.0 * (1.0 - top_frac))
    xyz = pc4.to_3d(comp[keep])
    mappable = ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=values[keep],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=8,
        alpha=0.22,
        linewidths=0,
        depthshade=False,
    )
    fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02, label="Objective")
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=18, azim=35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_triptych_html(
    out_path: Path,
    comp: np.ndarray,
    panels: list[tuple[str, np.ndarray]],
    *,
    title: str,
    labels: tuple[str, ...],
    vmin: float,
    vmax: float,
) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=len(panels),
        specs=[[{"type": "scene"}] * len(panels)],
        subplot_titles=[name for name, _ in panels],
        horizontal_spacing=0.02,
    )
    for col, (name, values) in enumerate(panels, start=1):
        xyz = pc4.to_3d(comp)
        showscale = col == len(panels)
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                name=name,
                marker=dict(
                    color=values,
                    colorscale="Viridis",
                    cmin=vmin,
                    cmax=vmax,
                    size=2.8,
                    opacity=0.18,
                    showscale=showscale,
                    colorbar=dict(title="Objective", len=0.65) if showscale else None,
                ),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )
        # edges
        xs, ys, zs = [], [], []
        for i in range(4):
            for j in range(i + 1, 4):
                xs += [pc4.TETRA_VERTICES[i, 0], pc4.TETRA_VERTICES[j, 0], None]
                ys += [pc4.TETRA_VERTICES[i, 1], pc4.TETRA_VERTICES[j, 1], None]
                zs += [pc4.TETRA_VERTICES[i, 2], pc4.TETRA_VERTICES[j, 2], None]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color="rgba(60,60,60,0.55)", width=2),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        pos = pc4.TETRA_VERTICES * 1.12
        fig.add_trace(
            go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                mode="text",
                text=list(labels),
                textfont=dict(size=12, color="black"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=col,
        )

    scene_kwargs = dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
    )
    layout_scenes = {f"scene{i if i > 1 else ''}": scene_kwargs for i in range(1, len(panels) + 1)}
    # plotly uses scene, scene2, scene3
    layout_scenes = {"scene": scene_kwargs}
    for i in range(2, len(panels) + 1):
        layout_scenes[f"scene{i}"] = scene_kwargs

    fig.update_layout(
        title=title,
        width=420 * len(panels),
        height=520,
        margin=dict(l=0, r=0, t=60, b=0),
        **layout_scenes,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", auto_open=False)
    return out_path


def visualize_run(
    run_dir: str | Path,
    *,
    grid_n: int = DEFAULT_GRID_N,
    out_subdir: str = "viz",
) -> Path:
    """Generate 4-simplex landscape visualizations for a pilot run."""
    run_dir = Path(run_dir).resolve()
    viz_dir = run_dir / out_subdir
    viz_dir.mkdir(parents=True, exist_ok=True)

    ctx = load_context_from_run(run_dir)
    if int(ctx.dim) != 4:
        raise ValueError(f"{run_dir.name}: expected dim=4, got dim={ctx.dim}")

    source = load_landscape_source(run_dir)
    labels = _corner_labels(run_dir, ctx.dim)
    comp = pc4.build_simplex_lattice(int(grid_n))
    z_grid = composition_to_ilr(comp)

    campaign_rf = train_rf_surrogate(ctx.x_campaign, ctx.y_campaign)
    y_campaign = np.asarray(campaign_rf.predict(comp), dtype=float).ravel()
    y_expr = np.asarray(source.predict(z_grid), dtype=float).ravel()

    rf_transform = bool(ctx.metadata.get("rf_transform_features", False))
    y_rf_g: np.ndarray | None = None
    if rf_transform and ctx.x_rf_train is not None and ctx.z_rf_train is not None:
        y_train = np.asarray(source.predict(ctx.z_rf_train), dtype=float).ravel()
        n_est = int(ctx.metadata.get("rf_transform_n_estimators", 500))
        rf_seed = int(ctx.metadata.get("rf_transform_seed", 42))
        y_rf_g = rf_transform_predict(
            ctx.x_rf_train,
            y_train,
            comp,
            n_estimators=n_est,
            random_state=rf_seed,
        )

    arrays = [y_campaign, y_expr] + ([y_rf_g] if y_rf_g is not None else [])
    vmin = float(min(float(a.min()) for a in arrays))
    vmax = float(max(float(a.max()) for a in arrays))
    run_name = run_dir.name
    gen_tag = f"gen{source.generation}" if source.generation >= 0 else "best"
    src_note = source.source

    meta = {
        "run": run_name,
        "expression_source": src_note,
        "generation": source.generation,
        "grid_n": int(grid_n),
        "n_lattice": int(comp.shape[0]),
        "rf_transform_features": rf_transform,
        "shared_colorbar": [vmin, vmax],
        "y_campaign_range": [float(y_campaign.min()), float(y_campaign.max())],
        "y_expression_range": [float(y_expr.min()), float(y_expr.max())],
    }
    if y_rf_g is not None:
        meta["y_rf_g_range"] = [float(y_rf_g.min()), float(y_rf_g.max())]

    panels: list[tuple[str, np.ndarray]] = [
        ("campaign RF", y_campaign),
        ("expression g(z)", y_expr),
    ]
    written: list[Path] = []

    for stem, title, values in (
        (
            "tetra_campaign_rf",
            f"{run_name} — campaign RF ({gen_tag})",
            y_campaign,
        ),
        (
            "tetra_expression",
            f"{run_name} — expression g(z) ({gen_tag}, from {src_note})",
            y_expr,
        ),
    ):
        written.append(
            _write_html(
                viz_dir / f"{stem}.html",
                comp,
                values,
                title=title,
                labels=labels,
                cmin=vmin,
                cmax=vmax,
                colorbar_title=stem.replace("tetra_", ""),
            )
        )
        written.append(
            _write_png(
                viz_dir / f"{stem}.png",
                comp,
                values,
                title=title,
                labels=labels,
                vmin=vmin,
                vmax=vmax,
            )
        )

    if y_rf_g is not None:
        panels.append(("RF(g)", y_rf_g))
        title = f"{run_name} — RF(g) ({gen_tag})"
        written.append(
            _write_html(
                viz_dir / "tetra_rf_g.html",
                comp,
                y_rf_g,
                title=title,
                labels=labels,
                cmin=vmin,
                cmax=vmax,
                colorbar_title="RF(g)",
            )
        )
        written.append(
            _write_png(
                viz_dir / "tetra_rf_g.png",
                comp,
                y_rf_g,
                title=title,
                labels=labels,
                vmin=vmin,
                vmax=vmax,
            )
        )
        # High-value basins only (easier to read on a static PNG).
        written.append(
            _write_png(
                viz_dir / "tetra_rf_g_top25.png",
                comp,
                y_rf_g,
                title=f"{title} (top 25%)",
                labels=labels,
                vmin=vmin,
                vmax=vmax,
                top_frac=0.25,
            )
        )

    written.append(
        _write_triptych_html(
            viz_dir / "tetra_triptych.html",
            comp,
            panels,
            title=f"{run_name} — campaign RF | expression | RF(g)"
            if y_rf_g is not None
            else f"{run_name} — campaign RF | expression",
            labels=labels,
            vmin=vmin,
            vmax=vmax,
        )
    )

    meta_path = viz_dir / "tetra_viz_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    written.append(meta_path)

    # Evolution fitness curve (dim-agnostic).
    history = run_dir / "evolution" / "history.csv"
    if history.is_file():
        import csv

        rows = list(csv.DictReader(history.open(encoding="utf-8")))
        if rows:
            gens = [int(float(r["generation"])) for r in rows]
            fit = [float(r["best_fitness"]) for r in rows]
            fig, ax = plt.subplots(figsize=(7.0, 3.6))
            ax.plot(gens, fit, color="#1f77b4", lw=1.6)
            ax.set_xlabel("generation")
            ax.set_ylabel("best fitness")
            ax.set_title(f"{run_name} — evolution")
            ax.grid(True, alpha=0.3)
            out = viz_dir / "evolution_fitness.png"
            fig.savefig(out, dpi=140, bbox_inches="tight")
            plt.close(fig)
            written.append(out)

    print(f"Wrote {len(written)} artifacts under {viz_dir}")
    for path in written:
        print(f"  {path.relative_to(run_dir)}")
    return viz_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="4D ELA pilot tetrahedron visualization")
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="ELA run directory (e.g. ela/runs/ela_4d_19448443)",
    )
    parser.add_argument(
        "--grid-n",
        type=int,
        default=DEFAULT_GRID_N,
        help=f"4-simplex lattice resolution (default {DEFAULT_GRID_N})",
    )
    parser.add_argument("--out-subdir", default="viz")
    args = parser.parse_args(argv)

    run_dir = args.run
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run directory not found: {run_dir}")

    visualize_run(run_dir, grid_n=args.grid_n, out_subdir=args.out_subdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
