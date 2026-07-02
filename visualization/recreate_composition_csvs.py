"""
visualization/recreate_composition_csvs.py
==========================================
Recreate the *sent* (requested/expected) compositions from a saved synthetic run
directory (``runs/run_*`` produced through interface/app.py) as a single pickled
NumPy array.

Each optimizer iteration sends one gradient *line* of exactly ``POINTS_PER_LINE``
(24) samples — ``np.linspace(start, end, 24)`` in composition space (see
interface/app.py). Every point-adding snapshot delta contributes one such line
(needle snapshots add no points and are skipped).

The output is a pickled ``numpy.ndarray`` of shape ``(24, 3, n)``:

* axis 0 — the 24 points along each line;
* axis 1 — the 3 composition ratios (the optimizer's ``d`` components);
* axis 2 — the ``n`` lines, in optimizer order.

Sent lines are expected to be exactly 24 points each (they are not deduped); the
script errors and fails if any line has a different length.

Usage
-----
  conda activate zombi-hop
  python visualization/recreate_composition_csvs.py runs/run_7eb9
  python visualization/recreate_composition_csvs.py runs/run_7eb9 --out-dir data \
      --out-name sent_compositions.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# Every sent gradient line is np.linspace(start, end, NUM_EXPERIMENTS) — see
# interface/app.py (NUM_EXPERIMENTS = 24).
POINTS_PER_LINE = 24


# ── run parsing ────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _snapshot_label(snap_dir: Path) -> str:
    """Human label for a snapshot: summary.json `label`, else the dir name minus
    its numeric ``NNNN_`` prefix."""
    summ = _load_json(snap_dir / "summary.json")
    if summ.get("label"):
        return str(summ["label"])
    name = snap_dir.name
    return name.split("_", 1)[1] if "_" in name and name.split("_", 1)[0].isdigit() else name


def collect_line_groups(run_dir: Path) -> list[dict]:
    """Replay snapshot deltas in order, returning one group per point-adding
    snapshot: {label, expected (N,d) tensor}."""
    snap_dir = run_dir / "snapshots"
    if not snap_dir.exists():
        raise FileNotFoundError(f"No snapshots/ directory under {run_dir}")

    groups: list[dict] = []
    for sdir in sorted(p for p in snap_dir.iterdir() if p.is_dir()):
        delta_path = sdir / "delta.pt"
        if not delta_path.exists():
            continue
        d = torch.load(str(delta_path), map_location="cpu", weights_only=False)
        x_new = d.get("X_new")
        if not isinstance(x_new, torch.Tensor) or x_new.shape[0] == 0:
            continue  # needle / no-op snapshot
        label = _snapshot_label(sdir)
        if label.lower().startswith("init"):
            continue  # seed scatter, not a swept 24-point line
        x_exp = d.get("X_exp_new", x_new)
        if not isinstance(x_exp, torch.Tensor) or x_exp.shape != x_new.shape:
            x_exp = x_new
        groups.append({
            "label":    label,
            "expected": x_exp.to(torch.float64),
        })
    if not groups:
        raise RuntimeError(
            f"No delta snapshots with points found under {run_dir}. "
            "(Legacy full-copy runs without delta.pt are not supported.)")
    return groups


def build_sent_array(groups: list[dict], points_per_line: int) -> np.ndarray:
    """Stack the sent lines into a ``(points_per_line, d, n)`` array.

    Every line must have exactly ``points_per_line`` points; otherwise raise.
    """
    d = groups[0]["expected"].shape[1]
    lines: list[np.ndarray] = []
    for g in groups:
        arr = g["expected"].cpu().numpy()          # (N, d)
        if arr.shape[0] != points_per_line:
            raise ValueError(
                f"Line '{g['label']}' has {arr.shape[0]} points, expected "
                f"{points_per_line}. Sent lines must not be deduped.")
        if arr.shape[1] != d:
            raise ValueError(
                f"Line '{g['label']}' has d={arr.shape[1]} components, "
                f"expected d={d}.")
        lines.append(arr)
    # (n, points_per_line, d) → (points_per_line, d, n)
    return np.stack(lines, axis=0).transpose(1, 2, 0)


# ── main ────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Run directory, e.g. runs/run_7eb9")
    ap.add_argument("--out-dir", type=Path, default=Path("data"),
                    help="Directory to write the pickle into (default: data/)")
    ap.add_argument("--out-name", default="sent_compositions.pkl",
                    help="Output pickle filename (default: sent_compositions.pkl)")
    ap.add_argument("--points-per-line", type=int, default=POINTS_PER_LINE,
                    help=f"Required points per line (default: {POINTS_PER_LINE})")
    args = ap.parse_args(argv)

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 1

    groups = collect_line_groups(run_dir)
    d = groups[0]["expected"].shape[1]
    cfg_d = _load_json(run_dir / "config.json").get("d")
    if cfg_d is not None and int(cfg_d) != d:
        print(f"Warning: config.json d={cfg_d} but tensors have d={d}; using {d}.",
              file=sys.stderr)

    array = build_sent_array(groups, args.points_per_line)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name
    with open(out_path, "wb") as f:
        pickle.dump(array, f)

    print(f"Run          : {run_dir}")
    print(f"d            : {d}   points/line: {args.points_per_line}")
    print(f"Line groups  : {len(groups)}  ({', '.join(g['label'] for g in groups[:6])}"
          f"{' ...' if len(groups) > 6 else ''})")
    print(f"Wrote array  : {out_path}  shape {array.shape}  dtype {array.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
