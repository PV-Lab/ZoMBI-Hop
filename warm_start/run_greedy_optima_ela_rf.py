#!/usr/bin/env python3
"""Greedy optima on RF(ELA) landscapes with RF transforms — RF(g).

For each recent ``ela/runs/ela_3d_*`` run with ``rf_transform_features``:

  1. Load the evolved expression from ``best/expression.json`` when present,
     else the latest ``evolution/snapshots/gen_*.json``.
  2. Rebuild the ELA(RF_g) surface: evaluate ``g`` on the run's fixed
     ``x_rf_train``, fit an RF, then use ``RF.predict`` as the objective
     (same recipe as :mod:`ela.visualize_pilot_3d` / fitness).
  3. Run :func:`warm_start.greedy_optima.find_optima` (default n=20) on a
     large free Sobol design (not hardware lines).

Results land under ``warm_start/optima_finder_ela_rf/``.

Example
-------
  conda run -n zombi-hop-linebo python warm_start/run_greedy_optima_ela_rf.py
  conda run -n zombi-hop-linebo python warm_start/run_greedy_optima_ela_rf.py \\
      --design dirichlet --n-design 16384
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

from ela.compile_rf_surrogate_gallery import load_landscape_source  # noqa: E402
from ela.evolve_context import load_context_from_run  # noqa: E402
from warm_start.greedy_optima import (  # noqa: E402
    CANDIDATE_MULTIPLIER,
    DEFAULT_DESIGN,
    DESIGN_POOL_FLOOR,
    TOP_FRACTION,
    find_optima,
    n_design_for_optima,
)

OUT_DIR = _REPO / "warm_start" / "optima_finder_ela_rf"
_SQRT3_2 = np.sqrt(3) / 2
CORNER_LABELS = ("FA", "MA", "Br")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def list_rf_transform_runs(runs_root: Path) -> list[Path]:
    """Most-recent ``ela_3d_*`` dirs with ``rf_transform_features`` enabled."""
    cands: list[tuple[float, Path]] = []
    for p in runs_root.iterdir():
        if not p.is_dir() or not re.fullmatch(r"ela_3d_\d+", p.name):
            continue
        cfg_path = p / "config.json"
        if not cfg_path.is_file():
            continue
        cfg = _read_json(cfg_path)
        if not bool(cfg.get("rf_transform_features", False)):
            continue
        cands.append((p.stat().st_mtime, p))
    cands.sort(reverse=True)
    return [p for _, p in cands]


def build_rf_g_objective(run_dir: Path) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    """Fit RF(g) once; return batch predictor ``(N,3) -> (N,)`` plus meta."""
    ctx = load_context_from_run(run_dir)
    if ctx.x_rf_train is None or ctx.z_rf_train is None:
        raise ValueError(
            f"{run_dir.name}: rf_transform enabled but x_rf_train missing in samples.npz"
        )

    landscape = load_landscape_source(run_dir)
    n_est = int(ctx.metadata.get("rf_transform_n_estimators", 500))
    rf_seed = int(ctx.metadata.get("rf_transform_seed", 42))

    y_train = np.asarray(landscape.predict(ctx.z_rf_train), dtype=float).ravel()

    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(
        n_estimators=n_est,
        n_jobs=1,
        random_state=rf_seed,
        bootstrap=True,
    )
    rf.fit(ctx.x_rf_train, y_train)

    def objective(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.asarray(rf.predict(x), dtype=float).ravel()

    meta = {
        "run": run_dir.name,
        "source": landscape.source,
        "generation": landscape.generation,
        "has_oracle": (run_dir / "best" / "oracle.py").is_file(),
        "has_expression": (run_dir / "best" / "expression.json").is_file(),
        "rf_transform_n_estimators": n_est,
        "rf_transform_seed": rf_seed,
        "n_rf_train": int(ctx.x_rf_train.shape[0]),
        "objective": "RF(g)",
    }
    return objective, meta


def comp_to_xy(comp: np.ndarray) -> np.ndarray:
    p = np.asarray(comp, float)
    s = p.sum(-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def ternary_grid(steps: int) -> np.ndarray:
    a, b = np.meshgrid(np.arange(steps + 1), np.arange(steps + 1), indexing="ij")
    a, b = a.ravel(), b.ravel()
    keep = a + b <= steps
    a, b = a[keep], b[keep]
    return np.column_stack([a, b, steps - a - b]) / float(steps)


def plot_optima(
    objective: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    out_png: Path,
    *,
    title: str,
    grid_n: int = 120,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = ternary_grid(grid_n)
    vals = objective(grid)
    xy = comp_to_xy(grid)
    found_xy = comp_to_xy(X)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.tripcolor(xy[:, 0], xy[:, 1], vals, shading="gouraud", cmap="viridis")
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.2)
    ax.scatter(
        found_xy[:, 0], found_xy[:, 1],
        c="red", marker="x", s=48, linewidths=1.6, zorder=5,
        label=f"greedy RF(g) optima ({len(X)})",
    )
    for i, (u, v) in enumerate(found_xy):
        ax.text(u, v, str(i), color="white", fontsize=6, ha="center", va="bottom",
                zorder=6, path_effects=None)
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


def run_one(
    run_dir: Path,
    *,
    n_optima: int,
    seed: int,
    plot: bool,
    design: str = DEFAULT_DESIGN,
    n_design: int | None = None,
    edge_min: float = 0.0,
    out_dir: Path | None = None,
) -> dict:
    t0 = time.perf_counter()
    objective, meta = build_rf_g_objective(run_dir)
    if n_design is None:
        n_design = n_design_for_optima(n_optima)
    X, y = find_optima(
        objective, dim=3, n=n_optima, seed=seed,
        design=design,  # type: ignore[arg-type]
        n_design=n_design,
        edge_min=edge_min,
    )
    elapsed = time.perf_counter() - t0

    # Pairwise nearest-neighbour separation among reported optima.
    dmat = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    nn = dmat.min(axis=1)

    payload = {
        **meta,
        "n_optima": int(len(X)),
        "seed": seed,
        "design": design,
        "n_design": int(n_design),
        "edge_min": float(edge_min) if edge_min > 0 else None,
        "top_frac": TOP_FRACTION,
        "candidate_multiplier": CANDIDATE_MULTIPLIER,
        "true_optima": X.tolist(),
        "y_optima": y.tolist(),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "y_mean": float(y.mean()),
        "min_sep": float(nn.min()),
        "med_sep": float(np.median(nn)),
        "elapsed_s": round(elapsed, 3),
        "ok": True,
    }
    out = Path(out_dir) if out_dir is not None else OUT_DIR
    out_json = out / f"{run_dir.name}_optima.json"
    out.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    if plot:
        src_tag = "best" if meta["has_expression"] else f"gen{meta['generation']}"
        plot_optima(
            objective, X, y,
            out / f"{run_dir.name}_optima.png",
            title=(
                f"{run_dir.name}  RF(g)  [{src_tag}]  "
                f"n={len(X)} {design}={n_design}  "
                f"y∈[{y.min():.3f},{y.max():.3f}]  "
                f"min_sep={nn.min():.3f}"
                + (f"  edge_min={edge_min}" if edge_min > 0 else "")
            ),
        )
    return payload


def plot_from_saved(path: Path) -> None:
    data = _read_json(path)
    run_dir = _REPO / "ela" / "runs" / data["run"]
    objective, _ = build_rf_g_objective(run_dir)
    X = np.asarray(data["true_optima"], dtype=float)
    y = np.asarray(data["y_optima"], dtype=float)
    src_tag = "best" if data.get("has_expression") else f"gen{data.get('generation')}"
    plot_optima(
        objective, X, y,
        OUT_DIR / f"{data['run']}_optima.png",
        title=(
            f"{data['run']}  RF(g)  [{src_tag}]  "
            f"n={len(X)}  y∈[{y.min():.3f},{y.max():.3f}]"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-optima", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--design",
        choices=("sobol", "dirichlet", "lines"),
        default=DEFAULT_DESIGN,
        help="Stage-1 design (default sobol; lines only for hardware-faithful runs)",
    )
    parser.add_argument(
        "--n-design",
        type=int,
        default=None,
        help=f"Free-point design size (default max({DESIGN_POOL_FLOOR}, 10*n/top_frac))",
    )
    parser.add_argument(
        "--edge-min", type=float, default=0.0,
        help="Ignore design points with any coord < edge_min (e.g. 0.05)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=_REPO / "ela" / "runs")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Replot from existing JSON under optima_finder_ela_rf/",
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = OUT_DIR if args.edge_min <= 0 else (_REPO / "warm_start" / "optima_finder_ela_rf_interior")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        paths = sorted(out_dir.glob("ela_3d_*_optima.json"))
        if not paths:
            print(f"No JSON in {out_dir}", file=sys.stderr)
            return 1
        for p in paths:
            print(f"plot {p.name} …")
            plot_from_saved(p)
        print(f"Wrote {len(paths)} PNGs → {out_dir}")
        return 0

    runs = list_rf_transform_runs(args.runs_root)
    if not runs:
        print("No RF-transform ela_3d_* runs found.", file=sys.stderr)
        return 1

    n_design = args.n_design or n_design_for_optima(args.n_optima)
    print(
        f"Found {len(runs)} RF-transform run(s); n_optima={args.n_optima} "
        f"seed={args.seed} design={args.design} n_design={n_design} "
        f"edge_min={args.edge_min} top_frac={TOP_FRACTION} "
        f"cand_mult={CANDIDATE_MULTIPLIER} → {out_dir}"
    )
    summary: list[dict] = []
    for run_dir in runs:
        print(f"\n=== {run_dir.name} ===")
        try:
            row = run_one(
                run_dir,
                n_optima=args.n_optima,
                seed=args.seed,
                plot=not args.no_plot,
                design=args.design,
                n_design=n_design,
                edge_min=args.edge_min,
                out_dir=out_dir,
            )
            print(
                f"  source={row['source']}  design={row['design']}={row['n_design']}  "
                f"y∈[{row['y_min']:.4f},{row['y_max']:.4f}]  "
                f"min_sep={row['min_sep']:.4f} med_sep={row['med_sep']:.4f}  "
                f"{row['elapsed_s']:.2f}s"
            )
            summary.append({k: row[k] for k in (
                "run", "ok", "source", "generation", "has_oracle", "has_expression",
                "n_optima", "design", "n_design", "edge_min", "top_frac", "candidate_multiplier",
                "y_min", "y_max", "y_mean", "min_sep", "med_sep", "elapsed_s",
                "objective",
            )})
        except Exception as exc:
            print(f"  FAILED: {exc}")
            summary.append({"run": run_dir.name, "ok": False, "error": str(exc)})

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_optima": args.n_optima,
                "seed": args.seed,
                "design": args.design,
                "n_design": n_design,
                "edge_min": args.edge_min,
                "top_frac": TOP_FRACTION,
                "candidate_multiplier": CANDIDATE_MULTIPLIER,
                "objective": "RF(g)",
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
