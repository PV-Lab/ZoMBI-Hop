#!/usr/bin/env python3
"""Pilot: 3D S1 landscape recreation via GP (Muñoz Strategy S1)."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ela.evolve import EvolutionConfig, run_evolution
from ela.evolve_context import build_context
from ela.visualize_pilot_3d import visualize_run

DEFAULT_DB = ROOT / "data" / "2nd_real_run.db"
DEFAULT_TARGET = ROOT / "data" / "2nd_real_run_ela_full.json"
DEFAULT_RUNS = ROOT / "ela" / "runs"


def _configure_logging(level: str, log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="3D S1 GP landscape recreation pilot")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Campaign SQLite DB")
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="Tier-1 target JSON (ela_full output)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Run output directory")
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=3.0, help="Subspace RMSE weight (default: 3)")
    parser.add_argument("--beta", type=float, default=0.001, help="Tree complexity weight")
    parser.add_argument("--tier1-gamma", type=float, default=5.0, help="Tier-1 ELA loss weight (default: 5)")
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--n-dense", type=int, default=None, help="Dense sample size (default 4096)")
    parser.add_argument("--quick", action="store_true", help="Smoke test: small pop/gens/sample")
    parser.add_argument(
        "--early-reject-mult",
        type=float,
        default=None,
        help="Skip Tier-1 when RMSE > mult×threshold (default: 0=off, 3.0 for --quick)",
    )
    parser.add_argument("--no-landscape-viz", action="store_true", help="Skip per-generation ternary plots")
    parser.add_argument(
        "--landscape-every",
        type=int,
        default=1,
        help="Plot best landscape every N generations (default: 1)",
    )
    parser.add_argument(
        "--landscape-grid-n",
        type=int,
        default=100,
        help="Ternary resolution for per-generation plots (default: 100)",
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip post-run summary visualization")
    parser.add_argument("--grid-n", type=int, default=200, help="Ternary resolution for final viz/ folder")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    if args.quick:
        population = args.population if args.population is not None else 24
        generations = args.generations if args.generations is not None else 8
        n_dense = args.n_dense if args.n_dense is not None else 512
        snapshot_every = min(args.snapshot_every, 2)
        early_reject_mult = 3.0
    else:
        population = args.population if args.population is not None else 120
        generations = args.generations if args.generations is not None else 60
        n_dense = args.n_dense
        snapshot_every = args.snapshot_every
        early_reject_mult = 0.0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir or (DEFAULT_RUNS / f"pilot_3d_{stamp}")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    _configure_logging(args.log_level, run_dir / "pilot.log")

    log = logging.getLogger("ela.pilot_3d")
    log.info("Building evolution context from %s", args.db)
    log.info("Target fingerprint: %s", args.target)

    ctx = build_context(
        db_path=args.db,
        target_json=args.target,
        n_dense=n_dense,
        alpha_subspace=args.alpha,
        beta_complexity=args.beta,
    )
    ctx.metadata["alpha_subspace"] = args.alpha
    ctx.metadata["beta_complexity"] = args.beta
    ctx.metadata["tier1_gamma"] = args.tier1_gamma

    cfg = EvolutionConfig(
        population=population,
        generations=generations,
        alpha_subspace=args.alpha,
        beta_complexity=args.beta,
        tier1_gamma=args.tier1_gamma,
        snapshot_every=snapshot_every,
        seed=args.seed,
        early_reject_subspace_mult=(
            args.early_reject_mult if args.early_reject_mult is not None else early_reject_mult
        ),
        landscape_viz=not args.no_landscape_viz,
        landscape_viz_every=max(1, args.landscape_every),
        landscape_grid_n=args.landscape_grid_n,
    )

    log.info(
        "Starting evolution: pop=%d gens=%d n_dense=%d seed=%d -> %s",
        cfg.population,
        cfg.generations,
        ctx.n_dense,
        cfg.seed,
        run_dir,
    )

    best = run_evolution(ctx, run_dir, cfg, target_source=args.target)
    log.info(
        "Done. best fitness=%.4f rmse=%.5f accepted=%s",
        best.fitness,
        best.subspace_rmse,
        best.accepted,
    )
    log.info("Artifacts: %s", run_dir)

    if not args.no_viz:
        try:
            viz_dir = visualize_run(run_dir, grid_n=args.grid_n)
            log.info("Visualization: %s", viz_dir)
        except Exception:
            log.exception("Visualization failed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
