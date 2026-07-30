"""
optimize/regen_convergence.py
=============================
Regenerate ``convergence.png`` for already-finished eval runs — WITHOUT re-running
MOBO — by rebuilding the plot from each run's saved ``points.csv`` / ``needles.csv``.

Every run dir already persists everything ``run_mobo.plot_convergence`` needs:
``points.csv`` has the per-sample ``Y``, ``penalized`` flag and ``activation`` id,
and ``needles.csv`` has the needle locations (whose nearest sample index reproduces
``dh.needle_indices``, exactly as DataHandler.add_needle computes it). We wrap those
columns in a tiny dh-shim and call the REAL ``plot_convergence`` so the rendered plot
is byte-for-byte the same logic future runs use — in particular the running-best line
is drawn fully disconnected per activation.

Usage
-----
    python optimize/regen_convergence.py --base optimize/runs/deploy_3d_L40_job123
    python optimize/regen_convergence.py --base <dir1> <dir2> ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_mobo as rm  # noqa: E402


class _DHShim:
    """Minimal stand-in exposing only what plot_convergence reads."""

    def __init__(self, Y: torch.Tensor, penalty: torch.Tensor | None,
                 needle_indices: torch.Tensor | None):
        self.Y_all = Y
        self._penalty = penalty
        self.needle_indices = needle_indices

    def get_penalty_mask(self, X=None):
        return self._penalty


def _needle_indices(points: pd.DataFrame, needles_csv: Path,
                    xcols: list[str]) -> torch.Tensor | None:
    """Nearest-sample index per needle — mirrors DataHandler.add_needle's argmin."""
    if not needles_csv.exists():
        return None
    nd = pd.read_csv(needles_csv)
    if nd.empty:
        return None
    P = points[xcols].to_numpy(dtype=float)
    idxs = []
    for _, r in nd.iterrows():
        needle = np.array([r[c] for c in xcols], dtype=float)
        idxs.append(int(np.linalg.norm(P - needle[None, :], axis=1).argmin()))
    return torch.tensor(idxs, dtype=torch.int64).reshape(-1, 1)


def regen_run(run_dir: Path) -> bool:
    """Rebuild convergence.png in run_dir from its CSVs. Returns True on success."""
    points_csv = run_dir / "points.csv"
    if not points_csv.exists():
        return False
    df = pd.read_csv(points_csv)
    if df.empty or "Y" not in df.columns:
        return False
    xcols = [c for c in df.columns if c.startswith("x") and c[1:].isdigit()]

    Y = torch.tensor(df["Y"].to_numpy(dtype=float)).reshape(-1, 1)
    penalty = (torch.tensor(df["penalized"].to_numpy().astype(bool))
               if "penalized" in df.columns else None)
    activations = (df["activation"].to_numpy() if "activation" in df.columns else None)
    needle_idx = _needle_indices(df, run_dir / "needles.csv", xcols)

    dh = _DHShim(Y, penalty, needle_idx)
    rm.plot_convergence(str(run_dir / "convergence.png"), dh, maximize=True,
                        activations=activations)
    return True


def iter_run_dirs(base: Path):
    """Yield every run dir (has points.csv) beneath base, bounded depth."""
    # eval/<method>/<landscape>/trial_*/run_*  →  depth 5 under base.
    for pc in base.glob("eval/*/*/trial_*/run_*/points.csv"):
        yield pc.parent
    # also support being pointed straight at an eval dir or a single run dir.
    for pc in base.glob("*/*/trial_*/run_*/points.csv"):
        yield pc.parent
    if (base / "points.csv").exists():
        yield base


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate convergence.png from CSVs.")
    ap.add_argument("--base", nargs="+", required=True,
                    help="Run root(s), eval dir(s), or single run dir(s).")
    args = ap.parse_args()

    n_ok = 0
    n_fail = 0
    seen: set[Path] = set()
    for b in args.base:
        base = Path(b)
        for run_dir in iter_run_dirs(base):
            run_dir = run_dir.resolve()
            if run_dir in seen:
                continue
            seen.add(run_dir)
            try:
                if regen_run(run_dir):
                    n_ok += 1
                else:
                    n_fail += 1
                    print(f"  [regen] skipped (no usable points.csv): {run_dir}")
            except Exception as exc:
                n_fail += 1
                print(f"  [regen] FAILED {run_dir}: {exc}")
    print(f"  regenerated {n_ok} convergence.png ({n_fail} skipped/failed)")


if __name__ == "__main__":
    main()
