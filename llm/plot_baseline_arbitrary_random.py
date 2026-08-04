"""Convergence comparison for a subset of groups: baseline + arbitrary + the
uniform-random baseline only (drops perturbed and the LLM group).

Reuses the exact loading / normalization / uniform-random logic from
sweep_convergence.py so the resulting figure is directly comparable to
convergence_comparison.png, just with fewer series.

    python llm/plot_baseline_arbitrary_random.py <sweep_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- constants mirrored from sweep_convergence.py --------------------------
N_RANDOM_SEARCHES = 500
GROUPS = ["baseline", "arbitrary"]        # subset to plot (+ uniform random)
_GROUP_COLORS = {"baseline": "steelblue", "arbitrary": "darkorange"}
_GROUP_LABELS = {"baseline": "Tuned", "arbitrary": "Arbitrary"}
_RANDOM_COLOR = "slategray"


def _group_label(group: str) -> str:
    return _GROUP_LABELS.get(group, group.capitalize())


def _norm_curve(rb: np.ndarray, y_floor: float, y_opt: float) -> np.ndarray:
    span = (y_opt - y_floor) or 1.0
    return (np.asarray(rb, float) - y_floor) / span


def _ci95(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = stack.mean(axis=0)
    n = stack.shape[0]
    sem = stack.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    half = 1.96 * sem
    return mean, mean - half, mean + half


def random_running_best(sample_vals: np.ndarray, n: int, *, maximize: bool,
                        rng: np.random.Generator, n_searches: int) -> np.ndarray:
    draws = sample_vals[rng.integers(0, sample_vals.size, size=(n_searches, n))]
    accum = np.maximum.accumulate if maximize else np.minimum.accumulate
    return accum(draws, axis=1)


def _load_group_curves(gdir: Path) -> List[np.ndarray]:
    curves: List[np.ndarray] = []
    for rep in sorted(gdir.glob("rep*")):
        mp = rep / "metrics.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        rb = m.get("Y_all_running_best") or []
        y_floor = m.get("landscape_y_floor")
        y_opt = m.get("landscape_y_opt")
        if rb and y_floor is not None and y_opt is not None:
            curves.append(_norm_curve(np.asarray(rb, float), y_floor, y_opt))
    return curves


def _load_landscape_samples(sweep_dir: Path) -> List[Tuple[np.ndarray, float, float]]:
    seen: Dict[str, Tuple[np.ndarray, float, float]] = {}
    for gdir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        for rep in sorted(gdir.glob("rep*")):
            if rep.name in seen:
                continue
            mp = rep / "metrics.json"
            if not mp.exists():
                continue
            try:
                m = json.loads(mp.read_text())
            except Exception:
                continue
            sv = m.get("landscape_sample_vals")
            y_floor = m.get("landscape_y_floor")
            y_opt = m.get("landscape_y_opt")
            if sv and y_floor is not None and y_opt is not None:
                seen[rep.name] = (np.asarray(sv, float), float(y_floor), float(y_opt))
    return list(seen.values())


def _random_baseline_stack(sweep_dir: Path, L: int) -> Optional[np.ndarray]:
    reps = _load_landscape_samples(sweep_dir)
    if not reps:
        return None
    rows = []
    for i, (vals, y_floor, y_opt) in enumerate(reps):
        rng = np.random.default_rng(700_000 + i)
        curves = random_running_best(vals, L, maximize=True, rng=rng,
                                     n_searches=N_RANDOM_SEARCHES)
        rows.append(_norm_curve(curves, y_floor, y_opt))
    return np.vstack(rows)


def main(sweep_dir: Path) -> None:
    sweep_dir = Path(sweep_dir)
    out_png = sweep_dir / "convergence_baseline_arbitrary_random.png"

    group_curves = {g: _load_group_curves(sweep_dir / g) for g in GROUPS}
    group_curves = {g: c for g, c in group_curves.items() if c}
    if not group_curves:
        raise SystemExit(f"no usable running-best curves under {sweep_dir}")
    L = min(c.size for curves in group_curves.values() for c in curves)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    idx = np.arange(L)
    for group in GROUPS:
        curves = group_curves.get(group)
        if not curves:
            continue
        stack = np.vstack([c[:L] for c in curves])
        mean, lo, hi = _ci95(stack)
        color = _GROUP_COLORS[group]
        ax.fill_between(idx, lo, hi, color=color, alpha=0.2, lw=0, zorder=2)
        ax.plot(idx, mean, color=color, lw=2.0, zorder=4,
                label=f"{_group_label(group)} (mean ± 95% CI, n={stack.shape[0]})")

    rand = _random_baseline_stack(sweep_dir, L)
    if rand is not None and rand.size:
        rlo = np.percentile(rand, 5, axis=0)
        rhi = np.percentile(rand, 95, axis=0)
        ax.fill_between(idx, rlo, rhi, color=_RANDOM_COLOR, alpha=0.15, lw=0, zorder=1,
                        label="Uniform random (5–95%)")
        ax.plot(idx, rand.mean(axis=0), color=_RANDOM_COLOR, lw=1.6, ls="--", zorder=3,
                label=f"Uniform random (mean of {N_RANDOM_SEARCHES})")

    ax.set_xlabel("Objective evaluations (sample index)")
    ax.set_ylabel("Best found (fraction of landscape optimum)")
    ax.set_title("Convergence: tuned vs arbitrary vs uniform random "
                 "(mean ± 95% CI, per-landscape normalized)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    default = (Path(__file__).resolve().parent / "results"
               / "sweep_convergence_20260714_001905")
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
