#!/usr/bin/env python3
"""Pilot: 3D S1 landscape recreation via GP (Muñoz Strategy S1)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ela.evolve import EvolutionConfig, run_evolution
from ela.evolve_context import build_context
from ela.run_naming import default_runs_root, pilot_prefix, unique_run_dir
from ela.visualize_pilot_3d import visualize_run

DEFAULT_DB = ROOT / "data" / "2nd_real_run.db"
DEFAULT_TARGET = ROOT / "data" / "2nd_real_run_ela_full.json"
DEFAULT_RUNS = default_runs_root(ROOT)


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
    parser = argparse.ArgumentParser(
        description="3D S1 GP landscape recreation (Muñoz S1; RF λ_T target)",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Campaign SQLite DB")
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="Tier-1 target JSON (ela_full output)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS,
        help="Parent directory for auto-named runs (default: ela/runs)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Explicit run directory (Slurm: use this)")
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--campaign-mode",
        action="store_true",
        help="ZoMBI extensions: RMSE anchor, linear calibration, 10-feature loss",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Subspace RMSE weight (default: 0 paper / 3 campaign)",
    )
    parser.add_argument("--beta", type=float, default=0.001, help="Tree complexity weight")
    parser.add_argument(
        "--tier1-gamma",
        type=float,
        default=None,
        help="Tier-1 ELA loss scale (default: 1 paper / 5 campaign)",
    )
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--n-dense", type=int, default=None, help="Dense sample size (default 4096)")
    parser.add_argument("--quick", action="store_true", help="Smoke test: small pop/gens/sample")
    parser.add_argument(
        "--early-reject-mult",
        type=float,
        default=None,
        help="Campaign mode only: skip Tier-1 when RMSE > mult×threshold",
    )
    parser.add_argument("--no-landscape-viz", action="store_true", help="Skip per-generation ternary plots")
    parser.add_argument("--landscape-every", type=int, default=1)
    parser.add_argument("--landscape-grid-n", type=int, default=100)
    parser.add_argument("--no-viz", action="store_true", help="Skip post-run summary visualization")
    parser.add_argument("--grid-n", type=int, default=200)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    paper_mode = not args.campaign_mode

    if args.quick:
        population = args.population if args.population is not None else 24
        generations = args.generations if args.generations is not None else 8
        n_dense = args.n_dense if args.n_dense is not None else 512
        snapshot_every = min(args.snapshot_every, 2)
        early_reject_mult = 0.0 if paper_mode else 3.0
    else:
        population = args.population if args.population is not None else (200 if paper_mode else 120)
        generations = args.generations if args.generations is not None else (100 if paper_mode else 60)
        n_dense = args.n_dense
        snapshot_every = args.snapshot_every
        early_reject_mult = 0.0

    if args.alpha is not None:
        alpha = args.alpha
    else:
        alpha = 0.0 if paper_mode else 3.0

    if args.tier1_gamma is not None:
        tier1_gamma = args.tier1_gamma
    else:
        tier1_gamma = 1.0 if paper_mode else 5.0

    linearity_penalty = 0.0 if paper_mode else 3.0

    if args.out_dir is not None:
        run_dir = args.out_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        prefix = pilot_prefix(seed=args.seed, quick=args.quick)
        run_dir = unique_run_dir(args.runs_dir.resolve(), prefix)

    _configure_logging(args.log_level, run_dir / "pilot.log")

    log = logging.getLogger("ela.pilot_3d")
    mode_label = "paper (Muñoz S1)" if paper_mode else "campaign-twin"
    log.info("Mode: %s", mode_label)
    log.info("Building evolution context from %s", args.db)
    log.info("Target fingerprint: %s", args.target)

    ctx = build_context(
        db_path=args.db,
        target_json=args.target,
        n_dense=n_dense,
        alpha_subspace=alpha,
        beta_complexity=args.beta,
        paper_mode=paper_mode,
    )
    ctx.metadata["alpha_subspace"] = alpha
    ctx.metadata["beta_complexity"] = args.beta
    ctx.metadata["tier1_gamma"] = tier1_gamma
    ctx.metadata["paper_mode"] = paper_mode
    ctx.metadata["linearity_penalty_gamma"] = linearity_penalty

    cfg = EvolutionConfig(
        population=population,
        generations=generations,
        alpha_subspace=alpha,
        beta_complexity=args.beta,
        tier1_gamma=tier1_gamma,
        linearity_penalty_gamma=linearity_penalty,
        paper_mode=paper_mode,
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
        "Starting evolution: mode=%s pop=%d gens=%d n_dense=%d features=%s -> %s",
        mode_label,
        cfg.population,
        cfg.generations,
        ctx.n_dense,
        list(ctx.fitness_feature_names),
        run_dir,
    )

    best = run_evolution(ctx, run_dir, cfg, target_source=args.target)
    log.info(
        "Done. best fitness=%.4f tier1=%.4f rmse=%.5f accepted=%s",
        best.fitness,
        best.tier1_loss,
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
