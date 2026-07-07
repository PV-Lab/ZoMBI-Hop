#!/usr/bin/env python3
"""Pilot: 3D S1 landscape recreation via GP (Muñoz Strategy S1)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ela.evolve import EvolutionConfig, _elitism_count, run_evolution
from ela.evolve_context import build_context
from ela.run_manifest import (
    append_manifest,
    build_finish_manifest,
    build_start_manifest,
    resolve_gp_seed,
    write_manifest,
)
from ela.run_naming import default_runs_root, pilot_prefix, unique_run_dir
from ela.visualize_pilot_3d import visualize_run

DEFAULT_DB = ROOT / "data" / "2nd_real_run.db"
DEFAULT_TARGET = ROOT / "data" / "2nd_real_run_ela_full.json"
DEFAULT_RUNS = default_runs_root(ROOT)


def _default_eval_workers() -> int:
    """Process count for parallel fitness eval (population batch)."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        cpus = max(1, int(slurm_cpus))
        return max(1, min(cpus, cpus // 4))
    cpus = os.cpu_count() or 4
    return max(1, min(8, cpus // 4))


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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="GP evolution seed (default: SLURM_JOB_ID on cluster, else random)",
    )
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
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=None,
        help="Parallel fitness eval processes (default: SLURM_CPUS/4 or local cpus/4)",
    )
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
        population = args.population if args.population is not None else (400 if paper_mode else 120)
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
    eval_workers = (
        args.eval_workers if args.eval_workers is not None else _default_eval_workers()
    )
    eval_workers = max(1, int(eval_workers))
    gp_seed, gp_seed_source = resolve_gp_seed(args.seed)

    if args.out_dir is not None:
        run_dir = args.out_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        prefix = pilot_prefix(seed=gp_seed, quick=args.quick)
        run_dir = unique_run_dir(args.runs_dir.resolve(), prefix)

    manifest_path = run_dir / "run_manifest.json"
    _configure_logging(args.log_level, run_dir / "pilot.log")

    log = logging.getLogger("ela.pilot_3d")
    mode_label = "paper (Muñoz S1)" if paper_mode else "campaign-twin"
    log.info("Mode: %s", mode_label)
    log.info("GP seed: %d (source=%s)", gp_seed, gp_seed_source)
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
    log.info(
        "λ_T sample_seed=%d (fixed) n_dense=%d eval_workers=%d",
        ctx.sample_seed,
        ctx.n_dense,
        eval_workers,
    )

    ctx.metadata["gp_seed"] = gp_seed
    ctx.metadata["gp_seed_source"] = gp_seed_source
    ctx.metadata["alpha_subspace"] = alpha
    ctx.metadata["beta_complexity"] = args.beta
    ctx.metadata["tier1_gamma"] = tier1_gamma
    ctx.metadata["paper_mode"] = paper_mode
    ctx.metadata["linearity_penalty_gamma"] = linearity_penalty
    ctx.metadata["eval_workers"] = eval_workers
    ctx.metadata["population"] = population
    ctx.metadata["generations"] = generations

    manifest = build_start_manifest(
        gp_seed=gp_seed,
        gp_seed_source=gp_seed_source,
        run_dir=run_dir,
        args=args,
        paper_mode=paper_mode,
        population=population,
        generations=generations,
        n_dense=n_dense,
        eval_workers=eval_workers,
        alpha=alpha,
        tier1_gamma=tier1_gamma,
        linearity_penalty=linearity_penalty,
        sample_seed=ctx.sample_seed,
    )

    cfg = EvolutionConfig(
        population=population,
        generations=generations,
        alpha_subspace=alpha,
        beta_complexity=args.beta,
        tier1_gamma=tier1_gamma,
        linearity_penalty_gamma=linearity_penalty,
        paper_mode=paper_mode,
        snapshot_every=snapshot_every,
        seed=gp_seed,
        early_reject_subspace_mult=(
            args.early_reject_mult if args.early_reject_mult is not None else early_reject_mult
        ),
        landscape_viz=not args.no_landscape_viz,
        landscape_viz_every=max(1, args.landscape_every),
        landscape_grid_n=args.landscape_grid_n,
        eval_workers=eval_workers,
        **(
            {}
            if paper_mode
            else dict(
                tournament_k=5,
                crossover_prob=0.7,
                mutation_prob=0.25,
                direct_transfer_prob=0.0,
                elitism=2,
                elitism_frac=0.0,
                max_tree_depth=7,
                fitness_stop_threshold=0.0,
            )
        ),
    )

    manifest["hyperparameters"]["ga"] = {
        "tournament_k": cfg.tournament_k,
        "crossover_prob": cfg.crossover_prob,
        "mutation_prob": cfg.mutation_prob,
        "direct_transfer_prob": cfg.direct_transfer_prob,
        "elitism": _elitism_count(cfg),
        "max_tree_depth": cfg.max_tree_depth,
        "fitness_stop_threshold": cfg.fitness_stop_threshold,
    }
    write_manifest(manifest_path, manifest)
    log.info("Run manifest: %s", manifest_path)
    log.info("Manifest: %s", json.dumps(manifest, indent=2, default=str))

    log.info(
        "Starting evolution: mode=%s seed=%d (%s) pop=%d gens=%d n_dense=%d "
        "eval_workers=%d tournament_k=%d elite=%d depth=%d stop=%.0e -> %s",
        mode_label,
        gp_seed,
        gp_seed_source,
        cfg.population,
        cfg.generations,
        ctx.n_dense,
        cfg.eval_workers,
        cfg.tournament_k,
        _elitism_count(cfg),
        cfg.max_tree_depth,
        cfg.fitness_stop_threshold,
        run_dir,
    )

    t0 = time.monotonic()
    best = run_evolution(ctx, run_dir, cfg, target_source=args.target)
    wall_s = time.monotonic() - t0
    append_manifest(
        manifest_path,
        build_finish_manifest(
            best_fitness=best.fitness,
            best_tier1_loss=best.tier1_loss,
            best_subspace_rmse=best.subspace_rmse,
            accepted=best.accepted,
            wall_clock_s=wall_s,
        ),
    )
    log.info(
        "Done. seed=%d fitness=%.4f tier1=%.4f rmse=%.5f accepted=%s wall=%.1fs",
        gp_seed,
        best.fitness,
        best.tier1_loss,
        best.subspace_rmse,
        best.accepted,
        wall_s,
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
