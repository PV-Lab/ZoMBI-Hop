"""
visualization/recreate_composition_csvs.py
==========================================
Recreate the ``sent_compositions.csv`` / ``actual_compositions.csv`` pair from a
saved synthetic run directory (``runs/run_*`` produced through interface/app.py).

The two files mirror the ones consumed by ``visualization/discrepancy.py``:

* ``sent_compositions.csv``   — the *requested* (expected) line the optimizer sent.
  Columns: ``line, c0, c1, …, c{d-1}``; one row per point, grouped by ``line``
  label (``init`` for the seed lines, then ``line1_<snap>``, ``line2_<snap>`` …).

* ``actual_compositions.csv`` — the *measured* (actual) compositions, embedded in
  the fixed hardware channel space (n × HW_WIDTH, default 10; see
  scripts/communication.py). The d optimizer components are placed at the channel
  indices named in the run's ``hw_config.json`` (``dims``, e.g. ``0,8,9``); all
  other channels are zero. Columns: a leading row index, then ``0 … HW_WIDTH-1``.

Rows are aligned 1:1 between the two files, in optimizer order. Each snapshot
delta contributes one line (needle snapshots add no points and are skipped).

Usage
-----
  conda activate zombi-hop
  python visualization/recreate_composition_csvs.py runs/run_7eb9
  python visualization/recreate_composition_csvs.py runs/run_7eb9 --out-dir data \
      --dims 0,8,9 --hw-width 10
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

# Fixed hardware channel count: scripts/communication.py requires n × 10 arrays.
DEFAULT_HW_WIDTH = 10


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
    snapshot: {label, expected (N,d) tensor, actual (N,d) tensor}."""
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
        x_exp = d.get("X_exp_new", x_new)
        if not isinstance(x_exp, torch.Tensor) or x_exp.shape != x_new.shape:
            x_exp = x_new
        groups.append({
            "label":    _snapshot_label(sdir),
            "expected": x_exp.to(torch.float64),
            "actual":   x_new.to(torch.float64),
        })
    if not groups:
        raise RuntimeError(
            f"No delta snapshots with points found under {run_dir}. "
            "(Legacy full-copy runs without delta.pt are not supported.)")
    return groups


def resolve_dims(run_dir: Path, d: int, override: str | None) -> list[int]:
    """Channel indices the d optimizer components occupy in hardware space."""
    if override:
        dims = [int(x) for x in override.split(",") if x.strip() != ""]
    else:
        hw = _load_json(run_dir / "hw_config.json")
        raw = hw.get("dims")
        if isinstance(raw, str) and raw.strip():
            dims = [int(x) for x in raw.split(",") if x.strip() != ""]
        elif isinstance(raw, (list, tuple)):
            dims = [int(x) for x in raw]
        else:
            dims = list(range(d))  # sensible fallback: first d channels
    if len(dims) != d:
        raise ValueError(f"dims {dims} has {len(dims)} entries but run has d={d}.")
    return dims


# ── writers ────────────────────────────────────────────────────────────────────

def write_sent_csv(path: Path, groups: list[dict], d: int) -> int:
    """Write requested compositions, one labelled `line` group per snapshot."""
    n_written, line_no = 0, 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line"] + [f"c{i}" for i in range(d)])
        for g in groups:
            is_init = g["label"].lower().startswith("init")
            if is_init:
                label = "init"
            else:
                line_no += 1
                label = f"line{line_no}_{g['label']}"
            for row in g["expected"].tolist():
                w.writerow([label] + [f"{v:.6f}" for v in row])
                n_written += 1
    return n_written


def write_actual_csv(path: Path, groups: list[dict], dims: list[int], hw_width: int) -> int:
    """Write measured compositions embedded in the hardware channel space."""
    idx = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [str(j) for j in range(hw_width)])
        for g in groups:
            for row in g["actual"].tolist():
                wide = [0.0] * hw_width
                for comp, ch in enumerate(dims):
                    wide[ch] = row[comp]
                w.writerow([idx] + [repr(float(v)) for v in wide])
                idx += 1
    return idx


# ── main ────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Run directory, e.g. runs/run_7eb9")
    ap.add_argument("--out-dir", type=Path, default=Path("data"),
                    help="Directory to write the CSVs into (default: data/)")
    ap.add_argument("--sent-name", default="sent_compositions.csv")
    ap.add_argument("--actual-name", default="actual_compositions.csv")
    ap.add_argument("--dims", default=None,
                    help="Override hardware channel mapping, e.g. '0,8,9' "
                         "(default: read from the run's hw_config.json)")
    ap.add_argument("--hw-width", type=int, default=None,
                    help=f"Hardware channel count (default: {DEFAULT_HW_WIDTH}, "
                         "auto-expanded if a dim index exceeds it)")
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

    dims = resolve_dims(run_dir, d, args.dims)
    hw_width = args.hw_width if args.hw_width is not None else DEFAULT_HW_WIDTH
    hw_width = max(hw_width, max(dims) + 1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sent_path   = args.out_dir / args.sent_name
    actual_path = args.out_dir / args.actual_name

    n_sent   = write_sent_csv(sent_path, groups, d)
    n_actual = write_actual_csv(actual_path, groups, dims, hw_width)

    print(f"Run          : {run_dir}")
    print(f"d            : {d}   hardware channels: {hw_width}   dims: {dims}")
    print(f"Line groups  : {len(groups)}  ({', '.join(g['label'] for g in groups[:6])}"
          f"{' ...' if len(groups) > 6 else ''})")
    print(f"Wrote sent   : {sent_path}  ({n_sent} rows)")
    print(f"Wrote actual : {actual_path}  ({n_actual} rows)")
    if n_sent != n_actual:
        print("Warning: sent and actual row counts differ — they should match 1:1.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
