"""
Offline preview/apply CLI for retroactive needle declaration.

Constructs the run's ZoMBIHop headlessly (exactly like the resume branch of
run_zombi_main: dummy objective + empty init tensors + run_uuid), which
replays config.json and the latest snapshot WITHOUT writing anything to the
run directory, then calls ``retro_declare_needles``.

Default is a dry run: prints which past activations would declare a needle
under the run's CURRENT criteria (config.json as it is on disk right now)
and leaves the run directory byte-identical. ``--apply`` performs the real
pass — the identical code path the resume hook uses — declaring the needles
and persisting a permanent "retro_needles" snapshot.

Usage:
    python scripts/retro_needles.py --uuid 39af [--checkpoint-dir runs]
                                    [--device cpu] [--apply]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src import ZoMBIHop
from src.default_hparams import DEFAULT_HPARAMS, DEFAULT_INPUT_NOISE

# Constructor hyperparameters an operator can set (mirrors _VALID_HPARAM_KEYS in
# scripts/run_zombi_main.py, which is not imported here to avoid its module-level
# hardware/DB side effects).
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
    """In-force hyperparameters, matching the real resume: hardware defaults
    overlaid with the run's published hparams_effective.json. load_state()
    re-applies config.json's subset on top during construction, exactly as a
    resume does. Without this, keys outside config.json's restored subset
    (e.g. max_penalty_radius) would silently fall back to constructor defaults."""
    hparams = dict(DEFAULT_HPARAMS)
    hparams.setdefault("input_noise", DEFAULT_INPUT_NOISE)
    eff_path = run_dir / "hparams_effective.json"
    if eff_path.exists():
        try:
            eff = json.loads(eff_path.read_text(encoding="utf-8-sig"))
            hparams.update({k: v for k, v in eff.items() if k in _HPARAM_KEYS})
        except Exception as e:
            print(f"[retro-cli] Could not read hparams_effective.json ({e}); "
                  f"using hardware defaults.")
    return {k: v for k, v in hparams.items() if k in _HPARAM_KEYS}


def _load_bounds(run_dir: Path, d: int, device: str, dtype: torch.dtype) -> torch.Tensor | None:
    """Restore the per-dim search box from hw_config.json (same parsing as the
    resume branch of run_zombi_main); None ⇒ ZoMBIHop's default [0,1]^d box."""
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
        print(f"[retro-cli] Could not restore bounds from hw_config.json: {e}")
        return None
    if bounds_lo is None and bounds_hi is None:
        return None
    bounds = torch.zeros((2, d), device=device, dtype=dtype)
    bounds[0] = torch.tensor(bounds_lo, device=device, dtype=dtype) if bounds_lo else 0.0
    bounds[1] = torch.tensor(bounds_hi, device=device, dtype=dtype) if bounds_hi else 1.0
    print(f"[retro-cli] Search box: lo={bounds[0].tolist()} hi={bounds[1].tolist()}")
    return bounds


def _fmt_comp(x: list | None) -> str:
    if not x:
        return "-"
    return "[" + ", ".join(f"{v:.3f}" for v in x) + "]"


def _fmt_dists(dists: list | None) -> str:
    if not dists:
        return "-"
    return ", ".join(f"{v:.3f}" for v in dists)


def _print_dry_run_table(result: dict, n_consecutive: int):
    candidates = result.get("candidates", [])
    print()
    print("=" * 100)
    print(f"RETRO NEEDLE PREVIEW (dry run — nothing was changed)  "
          f"criteria: n_consecutive={n_consecutive}")
    print("=" * 100)
    if not candidates:
        print("No past activation satisfies the needle criteria.")
        print("=" * 100)
        return
    header = (f"{'act':>4}  {'trigger':>14}  {'Y':>8}  {'skip':>8}  "
              f"{'dist to earlier candidates':<28}  candidate composition")
    print(header)
    print("-" * 100)
    for c in candidates:
        trig = f"z{c['zoom']}/i{c['iteration']} n={c['counter']}"
        y = f"{c['y']:.4f}" if "y" in c else "-"
        skip = c.get("skipped_reason", "-")
        print(f"{c['activation']:>4}  {trig:>14}  {y:>8}  {skip:>8}  "
              f"{_fmt_dists(c.get('dist_to_earlier_candidates')):<28}  "
              f"{_fmt_comp(c.get('x'))}")
    n_would = len([c for c in candidates if "skipped_reason" not in c])
    print("-" * 100)
    print(f"{n_would} of {len(candidates)} candidate(s) would declare a needle.")
    print("Note: at apply time each new penalty ellipsoid can cover later candidates,")
    print("collapsing nearby candidates into fewer needles than listed here.")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Preview (default) or apply retroactive needle declaration "
                    "for an existing ZoMBI-Hop run.")
    parser.add_argument("--uuid", required=True,
                        help="Run UUID (run directory is <checkpoint-dir>/run_<uuid>).")
    parser.add_argument("--checkpoint-dir", default="runs",
                        help="Base checkpoint directory (default: runs).")
    parser.add_argument("--apply", action="store_true",
                        help="Actually declare the needles and snapshot "
                             "(default: dry run, no changes).")
    parser.add_argument("--device", default="cpu",
                        help="Torch device (default: cpu).")
    parser.add_argument("--max-penalty-radius", type=float, default=None,
                        help="Override max_penalty_radius for the needles this "
                             "pass declares. The Hessian at a plateau optimum is "
                             "near-flat, so its ellipsoid is sized by this cap "
                             "rather than by curvature; declaring several at the "
                             "default cap can exclude most of the search box.")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint_dir)
    run_dir = ckpt_path / f"run_{args.uuid}"
    if not run_dir.exists():
        sys.exit(f"Run directory not found: {run_dir}")
    config_path = run_dir / "config.json"
    if not config_path.exists():
        sys.exit(f"config.json not found in {run_dir}")
    # utf-8-sig: tolerate a BOM from hand-edited configs (Windows editors)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    d = int(config["d"])
    dtype = torch.float64

    # src.core.zombihop sets the torch default device to cuda at import when
    # CUDA is available; align it with the requested device so replayed
    # tensors do not straddle devices.
    torch.set_default_device(args.device)

    def _objective(*_a, **_k):
        raise RuntimeError("offline retro CLI must never call the objective")

    # Resume-style headless construction: dummy init tensors are ignored when
    # run_uuid is set; construction loads config.json + the latest snapshot
    # and writes nothing to the run directory.
    hparams = _load_hparams(run_dir)
    if args.max_penalty_radius is not None:
        print(f"[retro-cli] max_penalty_radius "
              f"{hparams.get('max_penalty_radius')} → {args.max_penalty_radius}")
        hparams["max_penalty_radius"] = args.max_penalty_radius

    _dummy = torch.zeros(0, d, device=args.device, dtype=dtype)
    optimizer = ZoMBIHop(
        objective=_objective,
        X_init_actual=_dummy,
        X_init_expected=_dummy,
        Y_init=torch.zeros(0, 1, device=args.device, dtype=dtype),
        device=args.device,
        dtype=dtype,
        bounds=_load_bounds(run_dir, d, args.device, dtype),
        run_uuid=args.uuid,
        checkpoint_dir=str(ckpt_path),
        verbose=True,
        **hparams,
    )

    result = optimizer.retro_declare_needles(dry_run=not args.apply)
    if result.get("error"):
        sys.exit(f"[retro-cli] {result['error']}")

    n_consecutive = int(optimizer.data_handler.n_consecutive_converged)
    if not args.apply:
        _print_dry_run_table(result, n_consecutive)
        return

    print()
    print("=" * 100)
    if result.get("applied"):
        latest = (run_dir / "latest.txt").read_text().strip()
        print(f"RETRO NEEDLES APPLIED: declared {result.get('n_declared')} "
              f"needle(s); run now has "
              f"{optimizer.data_handler.needles.shape[0]} needle(s) total.")
        print(f"Snapshot: {latest}  (resume position: activation "
              f"{result.get('new_activation')}, zoom 0, iter 0)")
    else:
        print("RETRO NEEDLES: nothing declared "
              "(no triggers, or all candidates covered/empty).")
    print("=" * 100)


if __name__ == "__main__":
    main()
