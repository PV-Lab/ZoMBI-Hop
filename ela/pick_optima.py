#!/usr/bin/env python3
"""Interactively click true maxima on an ELA twin and write a MOBO batch config.

Usage (local machine with a display)::

  python ela/pick_optima.py --job-id 18299175
  python ela/pick_optima.py --ela-run ela/runs/ela_3d_18299175 --out \\
      optimize/mobo_batch_configs/ela_3d_18299175.json

Controls
--------
  Left-click   pick near a peak (L-BFGS-B refined on the simplex by default)
  u / z        undo last pick
  Enter / q    finish and write the batch JSON
  Esc          abort without writing

Then submit continuous hparam training on ORCD::

  sbatch slurm/run_mobo_ela_18299175.sbatch
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from optimize.mobo_landscapes import (  # noqa: E402
    _ela_scalar_fn,
    _refine_callable_extremum,
    load_ela_oracle_module,
    resolve_ela_run_dir,
    ternary_grid,
)
from visualization.needle_overlay import comp_to_xy  # noqa: E402

_SQRT3_2 = np.sqrt(3) / 2
CORNER_LABELS = ("FA", "MA", "Br")
PICKER_GRID_N = 120


def xy_to_comp(x: float, y: float) -> np.ndarray:
    """Inverse of ``comp_to_xy`` (FA bottom-left, MA bottom-right, Br top)."""
    c2 = y / _SQRT3_2
    c1 = x - 0.5 * c2
    return np.array([1.0 - c1 - c2, c1, c2], dtype=float)


def draw_ternary_frame(ax, pad: float = 0.04) -> None:
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.axis("off")
    ax.text(-pad, -pad, CORNER_LABELS[0], ha="right", va="top", fontsize=9)
    ax.text(1 + pad, -pad, CORNER_LABELS[1], ha="left", va="top", fontsize=9)
    ax.text(0.5, _SQRT3_2 + pad, CORNER_LABELS[2], ha="center", va="bottom", fontsize=9)


class ElaOptimaPicker:
    """Click maxima on an ELA oracle ternary; optional L-BFGS-B refinement."""

    def __init__(
        self,
        fn,
        grid_pts: np.ndarray,
        grid_vals: np.ndarray,
        *,
        maximize: bool = True,
        refine: bool = True,
        title: str = "ELA twin",
    ):
        self.fn = fn
        self.grid_pts = grid_pts
        self.grid_vals = grid_vals
        self.maximize = maximize
        self.refine = refine
        self.title = title
        self.extrema: list[tuple[np.ndarray, float]] = []
        self._markers: list = []
        self._fig = self._ax = None
        self._done = False
        self._abort = False

    def _on_click(self, event) -> None:
        if event.inaxes is not self._ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        comp = xy_to_comp(float(event.xdata), float(event.ydata))
        if np.any(comp < -0.05):
            return
        comp = np.clip(comp, 0, None)
        comp = comp / comp.sum()
        if self.refine:
            x_ref, y_ref = _refine_callable_extremum(
                self.fn, comp, maximize=self.maximize, max_l1=0.15,
            )
        else:
            x_ref, y_ref = comp, float(self.fn(comp))
        tag = "maximum" if self.maximize else "minimum"
        print(
            f"  → #{len(self.extrema) + 1} {tag}: "
            f"{np.round(x_ref, 4).tolist()}  y={y_ref:.5f}"
        )
        self.extrema.append((x_ref, y_ref))
        xy = comp_to_xy(x_ref.reshape(1, 3))
        m = self._ax.scatter(
            xy[0, 0], xy[0, 1], marker="*", s=340, c="gold",
            zorder=12, edgecolors="black", linewidths=1.0,
        )
        t = self._ax.annotate(
            str(len(self.extrema)),
            (xy[0, 0], xy[0, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            fontweight="bold",
            color="white",
            zorder=13,
        )
        self._markers.append((m, t))
        self._fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key in ("enter", "q"):
            self._done = True
        elif key == "escape":
            self._abort = True
            self._done = True
        elif key in ("u", "z") and self.extrema:
            self.extrema.pop()
            m, t = self._markers.pop()
            m.remove()
            t.remove()
            print(f"  ← undo (now {len(self.extrema)} picks)")
            self._fig.canvas.draw_idle()

    def run(self) -> list[tuple[np.ndarray, float]] | None:
        self._done = False
        self._abort = False
        fig, ax = plt.subplots(figsize=(8.0, 7.2))
        self._fig, self._ax = fig, ax
        draw_ternary_frame(ax)
        goal = "maximum" if self.maximize else "minimum"
        ax.set_title(
            f"{self.title} — click near {goal}s, then Enter / Q\n"
            "(u/z = undo, Esc = abort)",
            fontsize=10,
        )
        gxy = comp_to_xy(self.grid_pts)
        sc = ax.scatter(
            gxy[:, 0], gxy[:, 1], c=self.grid_vals, cmap="viridis",
            s=8, alpha=0.85, zorder=2, rasterized=True,
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="oracle g(x)")
        ax.text(
            0.5, -0.08,
            "Left-click extrema.  Enter / Q when done.",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic",
        )
        cid1 = fig.canvas.mpl_connect("button_press_event", self._on_click)
        cid2 = fig.canvas.mpl_connect("key_press_event", self._on_key)
        plt.tight_layout()
        fig.canvas.draw()
        plt.show(block=False)
        while not self._done:
            try:
                fig.canvas.flush_events()
            except Exception:
                break
            time.sleep(0.05)
        fig.canvas.mpl_disconnect(cid1)
        fig.canvas.mpl_disconnect(cid2)
        plt.close(fig)
        if self._abort:
            print("Aborted — no config written.")
            return None
        return self.extrema


def write_batch_config(
    out_path: Path,
    *,
    job_id: int | None,
    ela_run: Path,
    true_optima: list[np.ndarray],
    ys: list[float],
    maximize: bool = True,
    time_limit_hours: float = 0.2,
    n_init_trials: int = 8,
) -> dict:
    name = out_path.stem
    try:
        ela_rel = str(ela_run.resolve().relative_to(_REPO.resolve()))
    except ValueError:
        ela_rel = str(ela_run)
    cfg = {
        "name": name,
        "landscape": "ela",
        "job_id": job_id,
        "ela_run": ela_rel,
        "maximize": maximize,
        "true_optima": [np.asarray(t, dtype=float).ravel().tolist() for t in true_optima],
        "time_limit_hours": time_limit_hours,
        "n_init_trials": n_init_trials,
        "_note": (
            f"Hand-picked optima via ela/pick_optima.py "
            f"(n={len(true_optima)}, y∈[{min(ys):.4f},{max(ys):.4f}])."
        ),
        "_picked_y": [float(y) for y in ys],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Click true maxima on an ELA twin → MOBO batch JSON.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--job-id", type=int, help="ELA Slurm job id (ela_3d_<id>).")
    g.add_argument("--ela-run", type=Path, help="Path to ela/runs/ela_3d_* directory.")
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output batch JSON (default: optimize/mobo_batch_configs/ela_3d_<id>.json).",
    )
    p.add_argument("--grid-n", type=int, default=PICKER_GRID_N, help="Ternary render density.")
    p.add_argument("--no-refine", action="store_true", help="Skip L-BFGS refinement.")
    p.add_argument("--minimize", action="store_true", help="Pick minima instead of maxima.")
    p.add_argument("--time-limit-hours", type=float, default=0.2)
    p.add_argument("--n-init-trials", type=int, default=8)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = resolve_ela_run_dir(
        ela_run=str(args.ela_run) if args.ela_run else None,
        job_id=args.job_id,
        repo_root=str(_REPO),
    )
    if not (run_dir / "best" / "oracle.py").is_file():
        print(f"ERROR: missing oracle at {run_dir / 'best' / 'oracle.py'}", file=sys.stderr)
        return 1

    job_id = args.job_id
    if job_id is None:
        # ela_3d_18299175 → 18299175
        name = run_dir.name
        if name.startswith("ela_3d_"):
            try:
                job_id = int(name.split("_")[-1])
            except ValueError:
                job_id = None

    out = args.out
    if out is None:
        tag = str(job_id) if job_id is not None else run_dir.name
        out = _REPO / "optimize" / "mobo_batch_configs" / f"ela_3d_{tag}.json"

    print(f"Loading oracle: {run_dir / 'best' / 'oracle.py'}")
    mod = load_ela_oracle_module(run_dir)
    fn = _ela_scalar_fn(mod.predict_composition)
    grid_pts = ternary_grid(args.grid_n)
    print(f"Evaluating ternary grid (n={args.grid_n}) …")
    grid_vals = np.asarray(mod.predict_composition(grid_pts), dtype=float).ravel()

    picker = ElaOptimaPicker(
        fn, grid_pts, grid_vals,
        maximize=not args.minimize,
        refine=not args.no_refine,
        title=run_dir.name,
    )
    picks = picker.run()
    if picks is None:
        return 2
    if not picks:
        print("No picks — nothing written.")
        return 1

    optima = [p for p, _ in picks]
    ys = [float(y) for _, y in picks]
    cfg = write_batch_config(
        out,
        job_id=job_id,
        ela_run=run_dir,
        true_optima=optima,
        ys=ys,
        maximize=not args.minimize,
        time_limit_hours=args.time_limit_hours,
        n_init_trials=args.n_init_trials,
    )
    print(f"\nWrote {len(optima)} needles → {out}")
    for i, (t, y) in enumerate(zip(cfg["true_optima"], ys), 1):
        print(f"  #{i:2d}  y={y:.5f}  {np.round(t, 4).tolist()}")
    print(
        "\nNext (on ORCD, after syncing this JSON + oracle):\n"
        f"  sbatch slurm/run_mobo_ela_{job_id}.sbatch"
        if job_id is not None
        else "  sbatch slurm/run_mobo_ela_<jobid>.sbatch  # point --config at the JSON above"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
