"""
Apply corrected objective scores to a ZoMBI-Hop run.

Re-ingests a *corrected* DiSCO results database into an existing run whose raw
data was re-analysed after collection. The compositions, activations, zoom
structure and sample order are unchanged — only some objective scores moved —
so this replays those corrected scores onto the already-stored measured points,
re-derives the needles under the corrected surface, and leaves a run that the
GUI can display and the hardware loop can resume.

By default it works on a *corrected copy* of the run (a new UUID under the
checkpoint dir), so the original measured record is never touched.

Two phases:
  * ``--dry-run`` (default is APPLY): parse the DB, match every stored point to
    its corrected objective, and print how many scores change and where each
    needle would move — writing nothing.
  * apply: clone the run, rewrite the persisted Y across the snapshot deltas,
    then (via the resume-style headless ZoMBIHop) re-declare each converged
    activation's needle at its corrected best point and snapshot the result.

Usage:
    python scripts/apply_correction.py --uuid 39af \\
        --corrected-db /path/to/results-...-corrected.db \\
        [--checkpoint-dir runs] [--new-uuid abcd] [--device cpu] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The optimizer's logging emits some non-ASCII (e.g. arrows); a Windows cp1252
# console would otherwise crash a direct terminal run. The GUI already forces
# PYTHONIOENCODING=utf-8 for its subprocesses; this covers the standalone case.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch

from src import ZoMBIHop
from src.core import correction
from src.default_hparams import DEFAULT_HPARAMS, DEFAULT_INPUT_NOISE
from src.utils.datahandler import reconstruct_snapshot_tensors

# Constructor hyperparameters an operator can set — mirrors the list in
# scripts/retro_needles.py (kept in lock-step so both offline tools construct
# ZoMBIHop identically to the live resume).
_HPARAM_KEYS = (
    "max_zooms", "max_iterations", "top_m_points", "n_restarts", "raw",
    "input_noise_threshold_mult", "output_noise_threshold_mult",
    "n_consecutive_converged", "max_gp_points", "repulsion_lambda",
    "acquisition_type", "ucb_beta", "nat_grad_step", "nat_grad_max_steps",
    "ellipsoid_drop_fraction", "ellipsoid_eigenvalue_floor", "max_penalty_radius",
    "paring_spatial_halfnoise", "paring_y_noise_multiplier", "input_noise",
    "needle_shrink_factor", "needle_stop_noise_multiplier",
    "zoom_jaccard_threshold", "bounds_shrink_factor", "min_axis_noise_mult",
    "jaccard_window", "jaccard_threshold",
)


def _load_hparams(run_dir: Path) -> dict:
    """In-force hyperparameters, matching the real resume (see retro_needles.py):
    hardware defaults overlaid with hparams_effective.json; load_state re-applies
    config.json's subset during construction."""
    hparams = dict(DEFAULT_HPARAMS)
    hparams.setdefault("input_noise", DEFAULT_INPUT_NOISE)
    eff_path = run_dir / "hparams_effective.json"
    if eff_path.exists():
        try:
            eff = json.loads(eff_path.read_text(encoding="utf-8-sig"))
            hparams.update({k: v for k, v in eff.items() if k in _HPARAM_KEYS})
        except Exception as e:
            print(f"[correction] Could not read hparams_effective.json ({e}); "
                  f"using hardware defaults.")
    return {k: v for k, v in hparams.items() if k in _HPARAM_KEYS}


def _load_bounds(run_dir: Path, d: int, device: str, dtype: torch.dtype):
    """Restore the per-dim search box from hw_config.json (same parsing as the
    resume branch); None ⇒ ZoMBIHop's default [0,1]^d box."""
    hw_path = run_dir / "hw_config.json"
    if not hw_path.exists():
        return None
    try:
        cfg = json.loads(hw_path.read_text())
        bounds_lo = ([float(x) for x in str(cfg["bounds_lo"]).split(",")]
                     if cfg.get("bounds_lo") else None)
        bounds_hi = ([float(x) for x in str(cfg["bounds_hi"]).split(",")]
                     if cfg.get("bounds_hi") else None)
    except Exception as e:
        print(f"[correction] Could not restore bounds from hw_config.json: {e}")
        return None
    if bounds_lo is None and bounds_hi is None:
        return None
    bounds = torch.zeros((2, d), device=device, dtype=dtype)
    bounds[0] = torch.tensor(bounds_lo, device=device, dtype=dtype) if bounds_lo else 0.0
    bounds[1] = torch.tensor(bounds_hi, device=device, dtype=dtype) if bounds_hi else 1.0
    return bounds


def _fmt_comp(x) -> str:
    if not x:
        return "-"
    return "[" + ", ".join(f"{v:.3f}" for v in x) + "]"


def _build_optimizer(run_dir: Path, uuid: str, ckpt_path: Path, d: int,
                     device: str, dtype: torch.dtype) -> ZoMBIHop:
    """Resume-style headless construction (identical to retro_needles.py)."""
    hparams = _load_hparams(run_dir)
    _dummy = torch.zeros(0, d, device=device, dtype=dtype)

    def _objective(*_a, **_k):
        raise RuntimeError("apply_correction must never call the objective")

    return ZoMBIHop(
        objective=_objective,
        X_init_actual=_dummy,
        X_init_expected=_dummy,
        Y_init=torch.zeros(0, 1, device=device, dtype=dtype),
        device=device,
        dtype=dtype,
        bounds=_load_bounds(run_dir, d, device, dtype),
        run_uuid=uuid,
        checkpoint_dir=str(ckpt_path),
        verbose=True,
        **hparams,
    )


def _print_match_report(report: correction.MatchReport, corr_map: correction.CorrectionMap):
    print()
    print("=" * 90)
    print("OBJECTIVE CORRECTION — MATCH REPORT")
    print("=" * 90)
    print(f"corrected DB: {corr_map.n_rows} row(s), {corr_map.n_valid} scored, "
          f"{corr_map.n_null} null; optimizing dims {corr_map.optimizing_dims} "
          f"-> columns {corr_map.active_columns}")
    if corr_map.n_conflicting_keys:
        print(f"  note: {corr_map.n_conflicting_keys} composition(s) recur in the DB "
              f"with differing scores (last value used).")
    print(report.summary())
    if report.unmatched_examples:
        print("  unmatched examples (kept old Y):")
        for e in report.unmatched_examples[:5]:
            print(f"    idx {e['index']:>5}  {_fmt_comp(e['comp'])}  old_y={e['old_y']:.4f}")
    print("=" * 90)


def main():
    ap = argparse.ArgumentParser(
        description="Apply corrected objective scores to a ZoMBI-Hop run "
                    "(default: create a corrected copy).")
    ap.add_argument("--uuid", required=True, help="Run UUID to correct (runs/run_<uuid>).")
    ap.add_argument("--corrected-db", required=True, help="Corrected DiSCO results .db.")
    ap.add_argument("--checkpoint-dir", default="runs", help="Base checkpoint dir (default: runs).")
    ap.add_argument("--new-uuid", default=None,
                    help="UUID for the corrected copy (default: auto 4-char).")
    ap.add_argument("--in-place", action="store_true",
                    help="Correct the existing run in place instead of a copy "
                         "(DESTRUCTIVE — overwrites the original snapshots).")
    ap.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report changes and needle moves; write nothing.")
    ap.add_argument("--round-decimals", type=int, default=6,
                    help="Composition rounding for matching (default: 6).")
    ap.add_argument("--nn-tol", type=float, default=1e-4,
                    help="Nearest-neighbour L-inf tolerance for float-drift matches.")
    ap.add_argument("--min-match-fraction", type=float, default=0.5,
                    help="Abort apply if fewer than this fraction of points match "
                         "(guards against the wrong DB/run).")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint_dir)
    src_run_dir = ckpt_path / f"run_{args.uuid}"
    if not src_run_dir.exists():
        sys.exit(f"Run directory not found: {src_run_dir}")
    config_path = src_run_dir / "config.json"
    if not config_path.exists():
        sys.exit(f"config.json not found in {src_run_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    d = int(config["d"])
    dtype = torch.float64

    dims = correction.read_optimizing_dims(src_run_dir)
    if not dims:
        sys.exit(f"Could not read optimizing dims (hw_config.json/config.json 'dims') "
                 f"from {src_run_dir}")

    # Parse the corrected DB.
    try:
        corr_map = correction.load_corrected_objectives(
            args.corrected_db, dims, round_decimals=args.round_decimals)
    except Exception as e:
        sys.exit(f"[correction] Could not read corrected DB: {e}")

    # Reconstruct the stored points and match them to corrected scores.
    latest = (src_run_dir / "latest.txt").read_text().strip()
    s = reconstruct_snapshot_tensors(src_run_dir, latest, device="cpu")
    X = s.get("X_all_actual")
    Y = s.get("Y_all")
    if X is None or Y is None or X.shape[0] == 0:
        sys.exit(f"[correction] Run {args.uuid} has no measured points to correct.")
    X_np = X.float().numpy()
    Y_np = Y.float().numpy().reshape(-1)
    corrected_Y, report = correction.match_corrected_Y(
        X_np, Y_np, corr_map, nn_tol=args.nn_tol)
    _print_match_report(report, corr_map)

    # torch default device alignment (zombihop sets cuda at import when available)
    torch.set_default_device(args.device)

    if args.dry_run:
        opt = _build_optimizer(src_run_dir, args.uuid, ckpt_path, d, args.device, dtype)
        opt.data_handler.Y_all = torch.as_tensor(
            corrected_Y, device=args.device, dtype=dtype).reshape(-1, 1)
        res = opt.redeclare_needles_after_correction(dry_run=True)
        print()
        print("=" * 90)
        print("NEEDLE MOVE PREVIEW (dry run — nothing was changed)")
        print("=" * 90)
        for e in res.get("needles", []):
            act = e.get("activation")
            if "skipped_reason" in e:
                print(f"  activation {act}: {e['skipped_reason']}")
                continue
            print(f"  activation {act}: {_fmt_comp(e.get('old_x'))} "
                  f"-> {_fmt_comp(e.get('new_x'))}  (corrected Y={e.get('new_y', float('nan')):.4f})")
        print("Note: at apply time a new ellipsoid can cover a later activation's best "
              "point, so the final needle count may be lower.")
        print("=" * 90)
        return

    # --- Apply ---
    if report.match_fraction < args.min_match_fraction:
        sys.exit(f"[correction] Only {100 * report.match_fraction:.1f}% of points matched "
                 f"the corrected DB (< {100 * args.min_match_fraction:.0f}% required). "
                 f"This is likely the wrong database or run — aborting.")

    if args.in_place:
        target_uuid = args.uuid
        target_run_dir = src_run_dir
        print(f"[correction] Applying IN PLACE to run {target_uuid} (original overwritten).")
    else:
        target_uuid = args.new_uuid or str(uuid4())[:4]
        target_run_dir = ckpt_path / f"run_{target_uuid}"
        while target_run_dir.exists():
            target_uuid = str(uuid4())[:4]
            target_run_dir = ckpt_path / f"run_{target_uuid}"
        print(f"[correction] Cloning run {args.uuid} → {target_uuid} ...")
        correction.clone_run(src_run_dir, target_run_dir, target_uuid)

    # Rewrite persisted Y across the (target) snapshot deltas.
    n_deltas = correction.rewrite_snapshot_Y(target_run_dir, corrected_Y)
    print(f"[correction] Rewrote corrected Y across {n_deltas} snapshot delta(s).")

    # Resume-style construction now loads the corrected Y; re-derive needles.
    opt = _build_optimizer(target_run_dir, target_uuid, ckpt_path, d, args.device, dtype)
    # Sanity: the reconstructed Y must equal the corrected vector we wrote.
    loaded_Y = opt.data_handler.Y_all.float().cpu().numpy().reshape(-1)
    if loaded_Y.shape[0] != corrected_Y.shape[0] or \
            float((abs(loaded_Y - corrected_Y)).max()) > 1e-6:
        sys.exit("[correction] Internal error: reconstructed Y does not match the "
                 "corrected vector after rewrite — aborting before touching needles.")

    res = opt.redeclare_needles_after_correction(dry_run=False)
    if res.get("error"):
        sys.exit(f"[correction] Needle re-derivation failed: {res['error']}")

    print()
    print("=" * 90)
    n_final = opt.data_handler.needles.shape[0]
    print(f"CORRECTION APPLIED: run {target_uuid} now has {n_final} needle(s) "
          f"(re-declared {res.get('n_declared', 0)} from corrected scores).")
    latest_new = (target_run_dir / "latest.txt").read_text().strip()
    print(f"Snapshot: {latest_new}  (resume position: activation "
          f"{res.get('new_activation')}, zoom 0, iter 0)")
    print(f"NEW_RUN_UUID: {target_uuid}")
    print(f"NEW_RUN_DIR: {target_run_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
