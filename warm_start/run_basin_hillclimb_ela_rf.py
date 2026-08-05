#!/usr/bin/env python3
"""Hill-climb basin finding on RF(ELA) landscapes with RF transforms — RF(g).

For each ``ela/runs/ela_3d_*`` run with ``rf_transform_features``:

  1. Build the RF(g) surface (same recipe as
     :mod:`warm_start.run_greedy_optima_ela_rf`).
  2. Evaluate on dense Sobol starts ∪ a ternary lattice.
  3. Greedy **graph hill-climb** on the KNN graph of those starts (RF has
     near-zero gradients almost everywhere, so L-BFGS is useless here).
  4. Merge attractors within ``eps`` (default 0.05) and rank basins by
     catchment mass.

Writes under ``warm_start/basin_finder_ela_rf/``:

  * ``<run>_basins.json`` / ``.npz`` — full catchment table
  * ``<run>_basins.png`` — landscape + top basins
  * ``<run>_top20.json`` — top-20 centers by catchment (MOBO needle candidates)
  * ``summary.json``

Example
-------
  conda run -n zombi-hop-linebo python warm_start/run_basin_hillclimb_ela_rf.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20_000))

from optimize.mobo_landscapes import ternary_grid  # noqa: E402
from visualization.needle_overlay import comp_to_xy  # noqa: E402
from warm_start.run_greedy_optima_ela_rf import (  # noqa: E402
    build_rf_g_objective,
    list_rf_transform_runs,
)

OUT_DIR = _REPO / "warm_start" / "basin_finder_ela_rf"
EPS_DEFAULT = 0.05
EPS_LIST = [0.02, 0.05, 0.08, 0.10, 0.15]
KNN_K = 12
GRID_STARTS = 60
TOP_N = 20
_SQRT3_2 = np.sqrt(3) / 2
CORNER_LABELS = ("FA", "MA", "Br")


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def make_knn_idx(pts: np.ndarray, k: int) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k + 1, len(pts)))
    if idx.ndim == 1:
        return idx.reshape(-1, 1)[:, 1:]
    return idx[:, 1:].astype(np.int32)


def graph_ascent(
    start_idx: int,
    vals: np.ndarray,
    nn_idx: np.ndarray,
    *,
    max_steps: int = 500,
) -> int:
    """Greedy ascent on the KNN graph to a local maximum index."""
    cur = int(start_idx)
    cur_v = float(vals[cur])
    seen = {cur}
    for _ in range(max_steps):
        best_nb = None
        best_v = cur_v
        for nb in nn_idx[cur]:
            nb = int(nb)
            if nb in seen:
                continue
            v = float(vals[nb])
            if v > best_v:
                best_v = v
                best_nb = nb
        if best_nb is None:
            break
        seen.add(best_nb)
        cur = best_nb
        cur_v = best_v
    return cur


def merge_attractors(
    attractors: np.ndarray,
    vals: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy merge by descending value; return centers, center_ys, labels."""
    order = np.argsort(vals)[::-1]
    centers: list[np.ndarray] = []
    center_ys: list[float] = []
    labels = np.full(len(attractors), -1, dtype=np.int32)
    for idx in order:
        a = attractors[idx]
        assigned = None
        for j, c in enumerate(centers):
            if float(np.linalg.norm(a - c)) < eps:
                assigned = j
                break
        if assigned is None:
            assigned = len(centers)
            centers.append(a.copy())
            center_ys.append(float(vals[idx]))
        labels[idx] = assigned
    return np.asarray(centers), np.asarray(center_ys), labels


def build_starts(run_dir: Path) -> np.ndarray:
    dense = run_dir / "X_dense.npy"
    parts = []
    if dense.is_file():
        parts.append(np.load(dense))
    parts.append(ternary_grid(GRID_STARTS))
    return np.vstack(parts)


def run_hillclimbs(
    objective: Callable[[np.ndarray], np.ndarray],
    starts: np.ndarray,
    *,
    knn_k: int = KNN_K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(starts, attractors, vals_at_attractors)``."""
    vals_start = np.asarray(objective(starts), dtype=float).ravel()
    nn_idx = make_knn_idx(starts, knn_k)
    attractor_idx = np.empty(len(starts), dtype=np.int32)
    for i in range(len(starts)):
        attractor_idx[i] = graph_ascent(i, vals_start, nn_idx)
    attractors = starts[attractor_idx]
    vals = vals_start[attractor_idx]
    return starts, attractors, vals


def basin_table(
    attractors: np.ndarray,
    vals: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    center_ys: np.ndarray,
) -> list[dict]:
    basins = []
    n = len(attractors)
    for j, (c, y) in enumerate(zip(centers, center_ys)):
        mask = labels == j
        n_starts = int(mask.sum())
        basins.append({
            "id": j + 1,
            "center": c.tolist(),
            "y": float(y),
            "n_starts": n_starts,
            "frac_starts": float(n_starts / n) if n else 0.0,
            "min_coord": float(np.min(c)),
        })
    basins.sort(key=lambda b: (-b["n_starts"], -b["y"]))
    # re-id by catchment rank for readability
    for i, b in enumerate(basins, 1):
        b["id"] = i
    return basins


def plot_basins(
    objective: Callable[[np.ndarray], np.ndarray],
    starts: np.ndarray,
    attractors: np.ndarray,
    labels: np.ndarray,
    basins: list[dict],
    out_png: Path,
    *,
    title: str,
    top_n: int = TOP_N,
    grid_n: int = 120,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = ternary_grid(grid_n)
    gvals = objective(grid)
    xy = comp_to_xy(grid)

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    ax.tripcolor(xy[:, 0], xy[:, 1], gvals, shading="gouraud", cmap="viridis")
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.2)

    # faint start→attractor segments for a subsample
    rng = np.random.default_rng(0)
    show = rng.choice(len(starts), size=min(800, len(starts)), replace=False)
    sxy = comp_to_xy(starts[show])
    axy = comp_to_xy(attractors[show])
    for i in range(len(show)):
        ax.plot(
            [sxy[i, 0], axy[i, 0]], [sxy[i, 1], axy[i, 1]],
            color="white", alpha=0.08, lw=0.4, zorder=2,
        )

    top = basins[:top_n]
    if top:
        centers = np.asarray([b["center"] for b in top], dtype=float)
        cxy = comp_to_xy(centers)
        sizes = 40 + 180 * np.asarray([b["frac_starts"] for b in top])
        ax.scatter(
            cxy[:, 0], cxy[:, 1],
            s=sizes, c="red", marker="o", edgecolors="k", linewidths=0.6,
            zorder=5, label=f"top-{len(top)} basins (size∝catchment)",
        )
        for i, (u, v) in enumerate(cxy):
            ax.text(u, v, str(i + 1), color="white", fontsize=7,
                    ha="center", va="center", zorder=6)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.text(-0.03, -0.03, CORNER_LABELS[0], ha="right", va="top", fontsize=9)
    ax.text(1.03, -0.03, CORNER_LABELS[1], ha="left", va="top", fontsize=9)
    ax.text(0.5, _SQRT3_2 + 0.04, CORNER_LABELS[2], ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def process_run(
    run_dir: Path,
    *,
    eps: float,
    top_n: int,
    plot: bool,
    edge_min: float = 0.0,
    out_dir: Path | None = None,
) -> dict:
    t0 = time.perf_counter()
    objective, meta = build_rf_g_objective(run_dir)
    starts = build_starts(run_dir)
    starts, attractors, vals = run_hillclimbs(objective, starts)

    # Drop climbs that landed on the simplex boundary.
    n_starts_all = int(len(starts))
    if edge_min > 0.0:
        interior = np.all(attractors >= float(edge_min), axis=1)
        starts = starts[interior]
        attractors = attractors[interior]
        vals = vals[interior]
        if len(starts) == 0:
            raise ValueError(f"no interior attractors with min_coord >= {edge_min}")

    eps_sweep = {}
    eps_sweep_detail = {}
    for e in EPS_LIST:
        centers_e, ys_e, _ = merge_attractors(attractors, vals, e)
        eps_sweep[str(e)] = int(len(centers_e))
        # RF(g) scale varies; report count with y ≥ 95th percentile of attractors
        thr = float(np.percentile(vals, 95))
        eps_sweep_detail[str(e)] = {
            "n": int(len(centers_e)),
            "n_y_ge_p95": int(np.sum(ys_e >= thr)),
            "y_p95": thr,
        }

    centers, center_ys, labels = merge_attractors(attractors, vals, eps)
    basins = basin_table(attractors, vals, labels, centers, center_ys)
    top = basins[:top_n]
    elapsed = time.perf_counter() - t0

    out = Path(out_dir) if out_dir is not None else OUT_DIR
    payload = {
        **meta,
        "method": "knn_graph_hillclimb",
        "knn_k": KNN_K,
        "n_starts": int(len(starts)),
        "n_starts_all": n_starts_all,
        "eps": eps,
        "edge_filter": float(edge_min) if edge_min > 0 else None,
        "n_basins": len(basins),
        "eps_sweep": eps_sweep,
        "eps_sweep_detail": eps_sweep_detail,
        "basins": basins,
        "elapsed_s": round(elapsed, 3),
        "ok": True,
    }

    out.mkdir(parents=True, exist_ok=True)
    stem = run_dir.name
    with (out / f"{stem}_basins.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    np.savez_compressed(
        out / f"{stem}_basins.npz",
        starts=starts,
        attractors=attractors,
        vals=vals,
        labels=labels,
        centers=np.asarray([b["center"] for b in basins], dtype=float),
        center_ys=np.asarray([b["y"] for b in basins], dtype=float),
        basin_counts=np.asarray([b["n_starts"] for b in basins], dtype=np.int32),
    )

    top_payload = {
        "run": stem,
        "objective": "RF(g)",
        "source": meta["source"],
        "generation": meta["generation"],
        "eps": eps,
        "edge_min": float(edge_min) if edge_min > 0 else None,
        "method": "knn_graph_hillclimb",
        "n_optima": len(top),
        "selection": "top catchment mass",
        "true_optima": [b["center"] for b in top],
        "y_optima": [b["y"] for b in top],
        "n_starts": [b["n_starts"] for b in top],
        "_note": (
            f"Top-{len(top)} RF(g) basins by catchment mass "
            f"(eps={eps}, knn graph hill-climb on {len(starts)} starts)."
        ),
    }
    with (out / f"{stem}_top{top_n}.json").open("w", encoding="utf-8") as f:
        json.dump(top_payload, f, indent=2)
        f.write("\n")

    if plot:
        src_tag = "best" if meta.get("has_expression") else f"gen{meta.get('generation')}"
        plot_basins(
            objective, starts, attractors, labels, basins,
            out / f"{stem}_basins.png",
            title=(
                f"{stem}  RF(g) basins [{src_tag}]  "
                f"eps={eps}  n={len(basins)}  top{top_n} marked"
                + (f"  edge_min={edge_min}" if edge_min > 0 else "")
            ),
            top_n=top_n,
        )

    return {
        "run": stem,
        "ok": True,
        "source": meta["source"],
        "generation": meta["generation"],
        "n_starts": len(starts),
        "n_basins": len(basins),
        "eps_sweep": eps_sweep,
        "top_y_max": top[0]["y"] if top else None,
        "top_n_starts": top[0]["n_starts"] if top else None,
        "elapsed_s": round(elapsed, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, default=EPS_DEFAULT)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument(
        "--edge-min", type=float, default=0.0,
        help="Ignore attractors with any coord < edge_min (e.g. 0.05)",
    )
    parser.add_argument("--runs-root", type=Path, default=_REPO / "ela" / "runs")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = OUT_DIR if args.edge_min <= 0 else (_REPO / "warm_start" / "basin_finder_ela_rf_interior")

    runs = list_rf_transform_runs(args.runs_root)
    if not runs:
        print("No RF-transform ela_3d_* runs found.", file=sys.stderr)
        return 1

    print(
        f"Found {len(runs)} RF-transform run(s); "
        f"eps={args.eps} top_n={args.top_n} edge_min={args.edge_min} "
        f"method=knn_graph_hillclimb → {out_dir}"
    )
    summary: list[dict] = []
    for run_dir in runs:
        print(f"\n=== {run_dir.name} ===")
        try:
            row = process_run(
                run_dir,
                eps=args.eps,
                top_n=args.top_n,
                plot=not args.no_plot,
                edge_min=args.edge_min,
                out_dir=out_dir,
            )
            print(
                f"  source={row['source']}  basins={row['n_basins']}  "
                f"eps_sweep={row['eps_sweep']}  "
                f"top1 n={row['top_n_starts']} y={row['top_y_max']:.4f}  "
                f"{row['elapsed_s']:.1f}s"
            )
            summary.append(row)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            summary.append({"run": run_dir.name, "ok": False, "error": str(exc)})

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "objective": "RF(g)",
                "method": "knn_graph_hillclimb",
                "eps": args.eps,
                "top_n": args.top_n,
                "edge_min": args.edge_min,
                "runs": summary,
            },
            f,
            indent=2,
        )
        f.write("\n")

    n_ok = sum(1 for r in summary if r.get("ok"))
    print(f"\nDone: {n_ok}/{len(summary)} ok → {out_dir}")
    return 0 if n_ok == len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
