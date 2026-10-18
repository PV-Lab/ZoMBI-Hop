"""
visualization/plot_convergence.py
=================================
Static-PNG export of the ZoMBI-Hop convergence plot for a *real* run directory.

This is the standalone/offline twin of the GUI's ``ConvergencePlotFrame`` in
``interface/app.py``: objective Y vs. sample index, the running-best envelope,
penalized-vs-valid point colouring, and a lollipop-and-leash marker per needle
(``draw_needle_leashes``): a numbered star on the sampled point, joined by a
horizontal leash to a diamond at the sample index where it was declared. It reconstructs the dataset from the run's delta
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
        self.declared_samples: np.ndarray | None = None   # (k,) sample idx per needle declaration
        self.n_points = 0
        self.n_needles = 0


def load_activation_timeline(run_dir: Path, snapshot: str) -> tuple[np.ndarray, np.ndarray]:
    """Activation start indices and needle-declaration sample indices.

    Replays the delta snapshots up to ``snapshot`` (the same traversal
    ``reconstruct_snapshot_tensors`` uses); a port of the GUI's
    ``interface/app.py:_activation_timeline``.

    ``starts``: the cumulative point count each time the ``actN`` number in the
    snapshot-dir name changes. The leading 0 (init / activation 0) is dropped so
    callers get only the interior reset boundaries.

    ``declared``: one entry per needle, in needle-stack order (so entry k pairs
    with ``needle_indices[k]``) — the last sample index measured before the
    snapshot whose needle stack first grew. When the stack is rebuilt rather than
    appended to, earlier declarations are dropped and every rebuilt needle counts
    as declared at that snapshot.

    Both empty for legacy full-copy runs with no per-snapshot deltas."""
    snap_dir = run_dir / "snapshots"
    if not snap_dir.is_dir():
        return np.array([], dtype=int), np.array([], dtype=int)

    starts: list[int] = []
    declared: list[int] = []
    n_seen = 0
    stack: list[int] = []   # needle_indices of the current needle stack
    cur_act: int | None = None
    for sdir in sorted(s for s in snap_dir.iterdir() if s.is_dir()):
        delta_path = sdir / "delta.pt"
        if delta_path.exists():
            d = torch.load(str(delta_path), map_location="cpu", weights_only=False)
            x_new = d.get("X_new")
            n_new = int(x_new.shape[0]) if isinstance(x_new, torch.Tensor) else 0
            ni = d.get("needle_indices")
            new_stack = (ni.long().ravel().tolist() if isinstance(ni, torch.Tensor)
                         else stack)
        else:
            n_new, new_stack = 0, stack
        m = _ACT_RE.search(sdir.name)
        act = int(m.group(1)) if m else cur_act
        if act is not None and act != cur_act and n_new > 0:
            starts.append(n_seen)
            cur_act = act
        n_seen += n_new
        if new_stack != stack:
            if new_stack[:len(stack)] != stack:   # rebuilt, not appended to
                declared = []
            declared.extend([n_seen - 1] * (len(new_stack) - len(declared)))
            stack = new_stack
        if sdir.name == snapshot:
            break

    return (np.array(sorted({s for s in starts if s > 0}), dtype=int),
            np.array(declared, dtype=int))


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

    cd.activation_starts, cd.declared_samples = load_activation_timeline(run_dir, snapshot)

    return cd


# ── plotting (mirrors interface/app.py ConvergencePlotFrame.update) ────────────

def draw_needle_leashes(ax, Y, needle_indices, declared_samples=None,
                        activation_starts=None) -> None:
    """Lollipop-and-leash needle markers, shared with the GUI's convergence plot.

    Needle k gets a numbered crimson star on its sampled point
    ``(needle_indices[k], Y[needle_indices[k]])`` and, when its declaration is
    known, a horizontal leash at that height out to a green diamond at
    ``declared_samples[k] + 0.5`` (+0.5: the declaration happens after the last
    sample of its line). The leash length is how long the point sat measured
    before the search confirmed it. The leash is lifted a few points so it stays
    visible where it coincides with the running-best envelope (the needle is
    usually its activation's best point). A needle declared in a later activation
    than the one that sampled it (a retro declaration on resume, or a rebuilt
    stack) gets a dashed leash and a hollow diamond."""
    if needle_indices is None or len(needle_indices) == 0:
        return
    from matplotlib.transforms import offset_copy

    lifted = offset_copy(ax.transData, fig=ax.figure, y=3, units="points")
    starts = (np.asarray(activation_starts) if activation_starts is not None
              else np.array([], dtype=int))
    n_decl = len(declared_samples) if declared_samples is not None else 0

    labeled: set[str] = set()

    def _label(name):
        if name in labeled:
            return None
        labeled.add(name)
        return name

    for k, ni in enumerate(needle_indices):
        if not 0 <= ni < len(Y):
            continue
        y = float(Y[ni])
        ax.scatter([ni], [y], marker="*", s=110, color="crimson", edgecolors="black",
                   linewidths=0.4, zorder=7, label=_label("needle point sampled"))
        ax.annotate(f"N{k + 1}", (ni, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color="crimson", zorder=8)
        if k >= n_decl:
            continue
        ds = int(declared_samples[k])
        retro = (np.searchsorted(starts, ds, side="right")
                 != np.searchsorted(starts, ni, side="right"))
        x_end = ds + 0.5
        ax.plot([ni, x_end], [y, y], transform=lifted, color="crimson", lw=1.1,
                ls="--" if retro else "-", alpha=0.8, zorder=6)
        ax.scatter([x_end], [y], transform=lifted, marker="D", s=26, zorder=7,
                   facecolors="none" if retro else "seagreen", edgecolors="seagreen",
                   linewidths=1.1,
                   label=_label("declared in a later activation" if retro
                                else "needle declared"))


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

    draw_needle_leashes(ax, Y, cd.needle_indices, cd.declared_samples, act_starts)

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
