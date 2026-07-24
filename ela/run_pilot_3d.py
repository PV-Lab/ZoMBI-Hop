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
from ela.evolve_context import attach_rf_transform_samples, build_context
from ela.pilot_config import (
    DEFAULT_CONFIG_PATH,
    ResolvedPilotConfig,
    load_pilot_config,
    resolve_pilot_config,
    write_resolved_config,
)
from ela.run_manifest import (
    append_manifest,
    build_finish_manifest,
    build_start_manifest,
    resolve_gp_seed,
    write_manifest,
)
from ela.run_naming import default_runs_root, pilot_prefix, unique_run_dir
from ela.visualize_pilot_3d import visualize_run

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


def _cli_overrides(args: argparse.Namespace) -> dict:
    return {
        "db": args.db,
        "target": args.target,
        "mode": None,
        "campaign_mode": args.campaign_mode,
        "pure_paper": args.pure_paper,
        "no_calibrate": args.no_calibrate,
        "no_require_subspace_rmse": args.no_require_subspace_rmse,
        "alpha": args.alpha,
        "beta": args.beta,
        "tier1_gamma": args.tier1_gamma,
        "linearity_penalty": args.linearity_penalty,
        "population": args.population,
        "generations": args.generations,
        "n_dense": args.n_dense,
        "quick": args.quick,
        "early_reject_mult": args.early_reject_mult,
        "no_landscape_viz": args.no_landscape_viz,
        "no_viz": args.no_viz,
        "eval_workers": args.eval_workers,
        "seed": args.seed,
        "log_level": args.log_level,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="3D S1 GP landscape recreation (Muñoz S1; RF λ_T target)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Pilot config JSON (default: {DEFAULT_CONFIG_PATH.relative_to(ROOT)})",
    )
    parser.add_argument("--db", type=Path, default=None, help="Campaign SQLite DB")
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
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
        help="Override config mode → campaign-twin",
    )
    parser.add_argument(
        "--pure-paper",
        action="store_true",
        help="Override config mode → Muñoz S1 ablation",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable linear a·g+b calibration (raw g(z) evaluation)",
    )
    parser.add_argument(
        "--no-require-subspace-rmse",
        action="store_true",
        help="Accept on Tier-1 only (skip subspace RMSE gate)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Subspace RMSE weight (overrides config)",
    )
    parser.add_argument("--beta", type=float, default=None, help="Tree complexity weight")
    parser.add_argument(
        "--tier1-gamma",
        type=float,
        default=None,
        help="Tier-1 ELA loss scale (overrides config)",
    )
    parser.add_argument(
        "--linearity-penalty",
        type=float,
        default=None,
        help="Penalty for near-linear trees (overrides config)",
    )
    parser.add_argument("--snapshot-every", type=int, default=None)
    parser.add_argument("--n-dense", type=int, default=None, help="Dense sample size (default 4096)")
    parser.add_argument("--quick", action="store_true", help="Smoke test (overrides config runtime.quick)")
    parser.add_argument(
        "--early-reject-mult",
        type=float,
        default=None,
        help="Skip Tier-1 when RMSE > mult×threshold",
    )
    parser.add_argument("--no-landscape-viz", action="store_true", help="Skip per-generation ternary plots")
    parser.add_argument("--landscape-every", type=int, default=None)
    parser.add_argument("--landscape-grid-n", type=int, default=None)
    parser.add_argument("--no-viz", action="store_true", help="Skip post-run summary visualization")
    parser.add_argument("--grid-n", type=int, default=None)
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=None,
        help="Parallel fitness eval processes (default: SLURM_CPUS/4 or local cpus/4)",
    )
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    if args.campaign_mode and args.pure_paper:
        parser.error("use at most one of --campaign-mode and --pure-paper")

    config_path = args.config or DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    raw = load_pilot_config(config_path)
    cfg_resolved = resolve_pilot_config(raw, repo_root=ROOT, cli=_cli_overrides(args))

    if args.snapshot_every is not None:
        cfg_resolved.snapshot_every = args.snapshot_every
    if args.landscape_every is not None:
        cfg_resolved.landscape_every = max(1, args.landscape_every)
    if args.landscape_grid_n is not None:
        cfg_resolved.landscape_grid_n = args.landscape_grid_n
    if args.grid_n is not None:
        cfg_resolved.grid_n = args.grid_n

    eval_workers = (
        cfg_resolved.eval_workers
        if cfg_resolved.eval_workers is not None
        else _default_eval_workers()
    )
    eval_workers = max(1, int(eval_workers))
    gp_seed, gp_seed_source = resolve_gp_seed(cfg_resolved.seed)

    if args.out_dir is not None:
        run_dir = args.out_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        prefix = pilot_prefix(seed=gp_seed, quick=cfg_resolved.quick)
        run_dir = unique_run_dir(args.runs_dir.resolve(), prefix)

    manifest_path = run_dir / "run_manifest.json"
    _configure_logging(cfg_resolved.log_level, run_dir / "pilot.log")

    log = logging.getLogger("ela.pilot_3d")
    resolved_path = write_resolved_config(run_dir, cfg_resolved)
    log.info("Config: %s", config_path)
    log.info("Resolved config: %s", resolved_path)
    log.info("Mode: %s (%s)", cfg_resolved.mode_label, cfg_resolved.mode)
    log.info("GP seed: %d (source=%s)", gp_seed, gp_seed_source)
    log.info("Building evolution context from %s", cfg_resolved.db)
    log.info("Target fingerprint: %s", cfg_resolved.target)
    log.info("Pilot parameters:\n%s", json.dumps(cfg_resolved.to_log_dict(), indent=2))

    ctx = build_context(
        db_path=cfg_resolved.db,
        target_json=cfg_resolved.target,
        n_dense=cfg_resolved.n_dense,
        alpha_subspace=cfg_resolved.alpha_subspace,
        beta_complexity=cfg_resolved.beta_complexity,
        subspace_rmse_frac=cfg_resolved.subspace_rmse_frac,
        munoz_8_fitness=cfg_resolved.munoz_8_fitness,
        linear_calibration=cfg_resolved.linear_calibration,
        paper_ga=cfg_resolved.paper_ga,
        fitness_feature_names=cfg_resolved.fitness_feature_names,
        tier1_weights=cfg_resolved.tier1_weights,
    )
    if cfg_resolved.rf_transform_features:
        attach_rf_transform_samples(
            ctx,
            n_samples=cfg_resolved.rf_transform_n_samples,
        )
        # Perfect surface match target for RF(g) vs campaign RF R².
        if "rf_vs_campaign_r2" not in ctx.tier1_target:
            ctx.tier1_target["rf_vs_campaign_r2"] = 1.0
        log.info(
            "ELA(RF_g): n_train=%d trees=%d seed=%d",
            cfg_resolved.rf_transform_n_samples,
            cfg_resolved.rf_transform_n_estimators,
            cfg_resolved.rf_transform_seed,
        )
    ctx.metadata["allow_rbf"] = cfg_resolved.allow_rbf
    ctx.metadata["oscillatory_bias"] = cfg_resolved.oscillatory_bias
    ctx.metadata["rbf_upweight"] = cfg_resolved.rbf_upweight
    ctx.metadata["rbf_additive_only"] = cfg_resolved.rbf_additive_only
    ctx.metadata["rbf_min_bumps"] = cfg_resolved.rbf_min_bumps
    ctx.metadata["rbf_max_bumps"] = cfg_resolved.rbf_max_bumps
    ctx.metadata["rf_transform_features"] = cfg_resolved.rf_transform_features
    ctx.metadata["rf_transform_n_samples"] = cfg_resolved.rf_transform_n_samples
    ctx.metadata["rf_transform_n_estimators"] = cfg_resolved.rf_transform_n_estimators
    ctx.metadata["rf_transform_seed"] = cfg_resolved.rf_transform_seed
    log.info(
        "λ_T sample_seed=%d (fixed) n_dense=%d eval_workers=%d subspace_threshold=%.5f",
        ctx.sample_seed,
        ctx.n_dense,
        eval_workers,
        ctx.subspace_rmse_threshold,
    )

    ctx.metadata["gp_seed"] = gp_seed
    ctx.metadata["gp_seed_source"] = gp_seed_source
    ctx.metadata["pilot_config_source"] = str(config_path)
    ctx.metadata["pilot_config_resolved"] = str(resolved_path)
    ctx.metadata["tier1_gamma"] = cfg_resolved.tier1_gamma
    ctx.metadata["linearity_penalty_gamma"] = cfg_resolved.linearity_penalty_gamma
    ctx.metadata["tier1_acceptance_median_rel"] = cfg_resolved.tier1_acceptance_median_rel
    ctx.metadata["require_subspace_rmse"] = cfg_resolved.require_subspace_rmse
    ctx.metadata["eval_workers"] = eval_workers
    ctx.metadata["population"] = cfg_resolved.population
    ctx.metadata["generations"] = cfg_resolved.generations

    manifest = build_start_manifest(
        gp_seed=gp_seed,
        gp_seed_source=gp_seed_source,
        run_dir=run_dir,
        pilot_config=cfg_resolved,
        population=cfg_resolved.population,
        generations=cfg_resolved.generations,
        n_dense=cfg_resolved.n_dense,
        eval_workers=eval_workers,
        sample_seed=ctx.sample_seed,
        config_path=config_path,
        resolved_config_path=resolved_path,
    )

    evo_cfg = EvolutionConfig(
        population=cfg_resolved.population,
        generations=cfg_resolved.generations,
        alpha_subspace=cfg_resolved.alpha_subspace,
        beta_complexity=cfg_resolved.beta_complexity,
        tier1_gamma=cfg_resolved.tier1_gamma,
        tier1_acceptance_median_rel=cfg_resolved.tier1_acceptance_median_rel,
        linearity_penalty_gamma=cfg_resolved.linearity_penalty_gamma,
        linear_calibration=cfg_resolved.linear_calibration,
        require_subspace_rmse=cfg_resolved.require_subspace_rmse,
        paper_ga=cfg_resolved.paper_ga,
        paper_mode=cfg_resolved.paper_ga,
        allow_rbf=cfg_resolved.allow_rbf,
        oscillatory_bias=cfg_resolved.oscillatory_bias,
        rbf_upweight=cfg_resolved.rbf_upweight,
        rbf_additive_only=cfg_resolved.rbf_additive_only,
        rbf_min_bumps=cfg_resolved.rbf_min_bumps,
        rbf_max_bumps=cfg_resolved.rbf_max_bumps,
        snapshot_every=cfg_resolved.snapshot_every,
        seed=gp_seed,
        early_reject_subspace_mult=cfg_resolved.early_reject_subspace_mult,
        landscape_viz=cfg_resolved.landscape_viz,
        landscape_viz_every=cfg_resolved.landscape_every,
        landscape_grid_n=cfg_resolved.landscape_grid_n,
        eval_workers=eval_workers,
        rf_transform_features=cfg_resolved.rf_transform_features,
        rf_transform_n_estimators=cfg_resolved.rf_transform_n_estimators,
        rf_transform_seed=cfg_resolved.rf_transform_seed,
        tournament_k=cfg_resolved.tournament_k,
        crossover_prob=cfg_resolved.crossover_prob,
        mutation_prob=cfg_resolved.mutation_prob,
        direct_transfer_prob=cfg_resolved.direct_transfer_prob,
        elitism=cfg_resolved.elitism,
        elitism_frac=cfg_resolved.elitism_frac,
        max_tree_depth=cfg_resolved.max_tree_depth,
        fitness_stop_threshold=cfg_resolved.fitness_stop_threshold,
    )

    manifest["hyperparameters"]["ga"] = {
        "tournament_k": evo_cfg.tournament_k,
        "crossover_prob": evo_cfg.crossover_prob,
        "mutation_prob": evo_cfg.mutation_prob,
        "direct_transfer_prob": evo_cfg.direct_transfer_prob,
        "elitism": _elitism_count(evo_cfg),
        "max_tree_depth": evo_cfg.max_tree_depth,
        "fitness_stop_threshold": evo_cfg.fitness_stop_threshold,
        "allow_rbf": evo_cfg.allow_rbf,
        "oscillatory_bias": evo_cfg.oscillatory_bias,
        "rbf_upweight": evo_cfg.rbf_upweight,
        "rbf_additive_only": evo_cfg.rbf_additive_only,
        "rbf_min_bumps": evo_cfg.rbf_min_bumps,
        "rbf_max_bumps": evo_cfg.rbf_max_bumps,
    }
    write_manifest(manifest_path, manifest)
    log.info("Run manifest: %s", manifest_path)
    log.info("Manifest: %s", json.dumps(manifest, indent=2, default=str))

    log.info(
        "Starting evolution: mode=%s seed=%d (%s) pop=%d gens=%d n_dense=%d "
        "features=%s calibrate=%s require_rmse=%s eval_workers=%d "
        "α=%.3f β=%.4f γ=%.2f linearity=%.2f "
        "tournament_k=%d elite=%d depth=%d stop=%.0e -> %s",
        cfg_resolved.mode_label,
        gp_seed,
        gp_seed_source,
        evo_cfg.population,
        evo_cfg.generations,
        ctx.n_dense,
        list(ctx.fitness_feature_names),
        evo_cfg.linear_calibration,
        evo_cfg.require_subspace_rmse,
        evo_cfg.eval_workers,
        evo_cfg.alpha_subspace,
        evo_cfg.beta_complexity,
        evo_cfg.tier1_gamma,
        evo_cfg.linearity_penalty_gamma,
        evo_cfg.tournament_k,
        _elitism_count(evo_cfg),
        evo_cfg.max_tree_depth,
        evo_cfg.fitness_stop_threshold,
        run_dir,
    )

    t0 = time.monotonic()
    best = run_evolution(ctx, run_dir, evo_cfg, target_source=cfg_resolved.target)
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

    if cfg_resolved.post_viz:
        try:
            viz_dir = visualize_run(run_dir, grid_n=cfg_resolved.grid_n)
            log.info("Visualization: %s", viz_dir)
        except Exception:
            log.exception("Visualization failed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
