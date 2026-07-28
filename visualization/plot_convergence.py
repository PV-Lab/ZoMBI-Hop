"""
visualization/plot_convergence.py
=================================
Static-PNG export of the ZoMBI-Hop convergence plot for a *real* run directory.

This is the standalone/offline twin of the GUI's ``ConvergencePlotFrame`` in
``interface/app.py``: objective Y vs. sample index, the running-best envelope,
penalized-vs-valid point colouring, and a dashed marker at every sample index
where a needle was declared. It reconstructs the dataset from the run's delta
snapshots (``reconstruct_snapshot_tensors``) — the same source the GUI and
``visualization/plot_run.py`` use — so no live Tk session is needed.

Usage
-----
  conda activate zombi-hop
  python visualization/plot_convergence.py runs/run_7eb9 runs/run_9dfe
  python visualization/plot_convergence.py run_7eb9 --out my_conv.png
  python visualization/plot_convergence.py run_9dfe --snapshot 0030_act9_z0_i0

Flags
-----
  RUN [RUN ...]     One or more run directories (or bare run names under runs/).
  --snapshot NAME   Snapshot to reconstruct up to (default: latest.txt).
                    Only meaningful when a single run is given.
  --out PATH        Output PNG path (single run only; default:
                    <run_dir>/convergence.png).
  --show            Display each figure as well as saving it.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

# ── project root on sys.path so `src` imports resolve ──────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.utils.datahandler import reconstruct_snapshot_tensors  # noqa: E402

_ACT_RE = re.compile(r"_act(\d+)")

ROOT = _HERE.parent
RUNS_DIR = ROOT / "runs"


# ── run/snapshot resolution (mirrors plot_run.py) ──────────────────────────────

def _resolve_run_dir(run_arg: str) -> Path:
    """Accept a full path, or a bare run name resolved against runs/."""
    p = Path(run_arg)
    if p.is_dir():
        return p.resolve()
    candidate = RUNS_DIR / run_arg
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"Run directory not found: {run_arg}")


def _default_snapshot(run_dir: Path) -> str:
    """The snapshot named in latest.txt, or the last snapshot directory."""
    latest = run_dir / "latest.txt"
    if latest.exists():
        name = latest.read_text().strip()
        if name:
            return name
    snaps = sorted(s.name for s in (run_dir / "snapshots").iterdir() if s.is_dir())
    if not snaps:
        raise FileNotFoundError(f"No snapshots found under {run_dir}")
    return snaps[-1]


# ── data loading ───────────────────────────────────────────────────────────────

class _ConvData:
    """The minimal slice of a run needed to draw the convergence plot."""

    def __init__(self, run_id: str, snapshot: str):
        self.run_id = run_id
        self.snapshot = snapshot
        self.Y: np.ndarray | None = None           # (n,)
        self.penalty_mask: np.ndarray | None = None  # (n,) bool
        self.needle_indices: np.ndarray | None = None  # (k,) int
        self.activation_starts: np.ndarray | None = None  # (a,) int sample idx
        self.n_points = 0
        self.n_needles = 0


def load_activation_starts(run_dir: Path, snapshot: str) -> np.ndarray:
    """Sample indices at which each *new* activation's points begin.

    Replays the delta snapshots up to ``snapshot`` (the same traversal
    ``reconstruct_snapshot_tensors`` uses) and, each time the ``actN`` number in
    the snapshot-dir name changes, records the cumulative point count as that
    activation's first sample index. The leading 0 (init / activation 0) is
    dropped so callers get only the interior reset boundaries.
    Empty for legacy full-copy runs with no per-snapshot deltas."""
    snap_dir = run_dir / "snapshots"
    if not snap_dir.is_dir():
        return np.array([], dtype=int)

    starts: list[int] = []
    n_seen = 0
    cur_act: int | None = None
    for sdir in sorted(s for s in snap_dir.iterdir() if s.is_dir()):
        delta_path = sdir / "delta.pt"
        if delta_path.exists():
            d = torch.load(str(delta_path), map_location="cpu", weights_only=False)
            x_new = d.get("X_new")
            n_new = int(x_new.shape[0]) if isinstance(x_new, torch.Tensor) else 0
        else:
            n_new = 0
        m = _ACT_RE.search(sdir.name)
        act = int(m.group(1)) if m else cur_act
        if act is not None and act != cur_act and n_new > 0:
            starts.append(n_seen)
            cur_act = act
        n_seen += n_new
        if sdir.name == snapshot:
            break

    starts = [s for s in starts if s > 0]
    return np.array(sorted(set(starts)), dtype=int)


def load_conv_data(run_dir: Path, snapshot: str | None) -> _ConvData:
    """Reconstruct Y / penalty-mask / needle-indices for a run at ``snapshot``."""
    snapshot = snapshot or _default_snapshot(run_dir)
    s = reconstruct_snapshot_tensors(run_dir, snapshot, device="cpu")

    cd = _ConvData(run_dir.name, snapshot)

    y = s.get("Y_all")
    if y is None or y.numel() == 0:
        raise RuntimeError(f"No datapoints reconstructed from {run_dir}/{snapshot}")
    cd.Y = y.float().numpy().ravel()
    cd.n_points = len(cd.Y)

    pm = s.get("penalty_mask")
    if pm is not None:
        cd.penalty_mask = pm.bool().numpy().ravel()

    ni = s.get("needle_indices")
    if ni is not None and ni.numel() > 0:
        cd.needle_indices = ni.long().numpy().ravel()
        cd.n_needles = len(cd.needle_indices)

    cd.activation_starts = load_activation_starts(run_dir, snapshot)

    return cd


# ── plotting (mirrors interface/app.py ConvergencePlotFrame.update) ────────────

def build_convergence_figure(cd: _ConvData, plt):
    """Draw the convergence plot for one run; returns the matplotlib Figure."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    Y = cd.Y
    idx = np.arange(len(Y))

    ax.scatter(idx, Y, s=10, alpha=0.65, color="steelblue", label="obs", zorder=2)

    # Running best, reset at every activation: each activation is a fresh ZoMBI
    # search phase, so the envelope accumulates only within the current phase
    # rather than over the whole run. Segment boundaries are the activations'
    # first sample indices; within each [start, end) slice the running best is an
    # independent cumulative max, drawn as its own orange segment.
    act_starts = np.array([], dtype=int)
    if cd.activation_starts is not None and len(cd.activation_starts) > 0:
        act_starts = cd.activation_starts[
            (cd.activation_starts > 0) & (cd.activation_starts < len(Y))
        ]
    bounds = np.concatenate(([0], act_starts))
    ends = np.concatenate((act_starts, [len(Y)]))
    labeled = False
    for start, end in zip(bounds, ends):
        if end <= start:
            continue
        seg_best = np.maximum.accumulate(Y[start:end])
        kw = dict(color="darkorange", lw=1.8, zorder=4)
        if not labeled:
            kw["label"] = "running best (reset per activation)"
            labeled = True
        ax.plot(idx[start:end], seg_best, **kw)

    if cd.needle_indices is not None and len(cd.needle_indices) > 0:
        labeled = False
        for ni in cd.needle_indices:
            if 0 <= ni < len(Y):
                kw = dict(color="crimson", alpha=0.55, lw=0.9, ls="--")
                if not labeled:
                    kw["label"] = "needle found"
                    labeled = True
                ax.axvline(float(ni), **kw)

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Objective Y")
    ax.set_title(f"{cd.run_id} — Convergence  "
                 f"(snap: {cd.snapshot},  {cd.n_points} pts, {cd.n_needles} needles)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def export_convergence(run_dir: Path, snapshot: str | None, out: Path | None,
                       show: bool, plt) -> Path:
    """Render and save one run's convergence PNG; returns the output path."""
    cd = load_conv_data(run_dir, snapshot)
    fig = build_convergence_figure(cd, plt)
    out = out or (run_dir / "convergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"{cd.run_id}: {cd.n_points} pts, {cd.n_needles} needles "
          f"(snap {cd.snapshot})  ->  {out}")
    if not show:
        plt.close(fig)
    return out


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static-PNG export of a ZoMBI-Hop run's convergence plot "
                    "(the offline twin of the GUI's ConvergencePlotFrame)."
    )
    parser.add_argument("runs", nargs="+",
                        help="One or more run directories or bare run names.")
    parser.add_argument("--snapshot", default=None,
                        help="Snapshot to reconstruct up to (default: latest.txt). "
                             "Only used when a single run is given.")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (single run only; default: "
                             "<run_dir>/convergence.png).")
    parser.add_argument("--show", action="store_true",
                        help="Display each figure as well as saving it.")
    args = parser.parse_args()

    if len(args.runs) > 1 and (args.out or args.snapshot):
        parser.error("--out / --snapshot only apply when a single run is given.")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for run_arg in args.runs:
        run_dir = _resolve_run_dir(run_arg)
        out = Path(args.out) if args.out else None
        export_convergence(run_dir, args.snapshot, out, args.show, plt)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
