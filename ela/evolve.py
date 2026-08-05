"""Genetic programming evolution for S1 landscape recreation."""
from __future__ import annotations

import concurrent.futures
import copy
import csv
import json
import logging
import multiprocessing
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ela.evolve_context import EvolutionContext, export_run_artifacts
from ela.gp_tree import (
    Node,
    crossover,
    dump_expression,
    evaluate_raw,
    mutate,
    predict_calibrated,
    predict_raw_clipped,
    random_tree,
    raw_linearity_r2,
    tree_depth,
    tree_has_nonlinearity,
    tree_size,
    tree_to_string,
)
from sklearn.metrics import r2_score

from ela.features import rf_transform_predict
from ela.tier1 import ALLOWED_FITNESS_NAMES, TIER1_NAMES, compute_tier1, weighted_feature_loss

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    tree: Node
    fitness: float = float("inf")
    tier1_loss: float = float("inf")
    subspace_rmse: float = float("inf")
    complexity: int = 0
    calib_a: float = 1.0
    calib_b: float = 0.0
    tier1: dict[str, float] = field(default_factory=dict)
    tier1_rel_err: dict[str, float] = field(default_factory=dict)
    accepted: bool = False


@dataclass
class EvolutionConfig:
    population: int = 400
    generations: int = 100
    tournament_k: int = 10
    crossover_prob: float = 0.6
    mutation_prob: float = 0.3
    direct_transfer_prob: float = 0.1
    elitism: int = 0  # if >0: fixed count; else use elitism_frac (paper: 10%)
    elitism_frac: float = 0.1
    max_tree_depth: int = 10
    fitness_stop_threshold: float = 1e-3  # paper early stop; 0 = disabled
    linear_calibration: bool = True
    require_subspace_rmse: bool = True
    paper_ga: bool = True  # Muñoz GP operators + 60/30/10 offspring
    paper_mode: bool = True  # legacy alias for paper_ga (gp_tree / viz)
    allow_rbf: bool = False  # localized ILR Gaussian bump primitive
    oscillatory_bias: bool = False  # upweight sin/cos + additive stacking
    rbf_upweight: bool = True  # high vs mild RBF sampling rates when allow_rbf
    rbf_additive_only: bool = False  # trees are only sums of RBF bumps (+ optional const)
    rbf_min_bumps: int = 1  # additive-only: floor on number of RBF leaves
    rbf_max_bumps: int | None = None  # additive-only: ceiling (defaults to max_tree_depth)
    alpha_subspace: float = 3.0
    beta_complexity: float = 0.001
    tier1_gamma: float = 1.0
    tier1_acceptance_median_rel: float = 0.10
    linearity_penalty_gamma: float = 0.0
    snapshot_every: int = 5
    seed: int = 0
    early_reject_subspace_mult: float = 0.0  # 0 = disabled; set e.g. 3.0 only for --quick
    landscape_viz: bool = True
    landscape_viz_every: int = 1
    landscape_grid_n: int = 100
    eval_workers: int = 1
    # ELA(RF_g): features from RF fit on fixed samples of g, not raw g(z).
    rf_transform_features: bool = False
    rf_transform_n_estimators: int = 500
    rf_transform_seed: int = 42


def _elitism_count(cfg: EvolutionConfig) -> int:
    if cfg.elitism > 0:
        return min(cfg.elitism, cfg.population)
    return max(1, int(round(cfg.population * cfg.elitism_frac)))


def _spawn_offspring_trees(
    rng: random.Random,
    pop: list[Individual],
    cfg: EvolutionConfig,
    *,
    n_vars: int,
) -> list[Node]:
    """Create 1–2 child trees. Paper: 60% crossover, 30% mutation, 10% direct transfer."""
    allow_rbf = bool(cfg.allow_rbf) or bool(cfg.rbf_additive_only)
    oscillatory = bool(cfg.oscillatory_bias) and not cfg.paper_ga and not cfg.rbf_additive_only
    rbf_up = bool(cfg.rbf_upweight) and allow_rbf
    rbf_add = bool(cfg.rbf_additive_only)
    mut_kw = dict(
        n_vars=n_vars,
        max_depth=cfg.max_tree_depth,
        paper_mode=cfg.paper_ga and not rbf_add,
        allow_rbf=allow_rbf,
        oscillatory_bias=oscillatory,
        rbf_upweight=rbf_up,
        rbf_additive_only=rbf_add,
        rbf_min_bumps=int(cfg.rbf_min_bumps),
        rbf_max_bumps=cfg.rbf_max_bumps,
    )
    if cfg.paper_ga:
        r = rng.random()
        if r < cfg.crossover_prob:
            p1 = _tournament_select(rng, pop, cfg.tournament_k)
            p2 = _tournament_select(rng, pop, cfg.tournament_k)
            return list(
                crossover(
                    rng,
                    p1.tree,
                    p2.tree,
                    max_depth=cfg.max_tree_depth,
                    rbf_additive_only=rbf_add,
                    rbf_min_bumps=int(cfg.rbf_min_bumps),
                    rbf_max_bumps=cfg.rbf_max_bumps,
                )
            )
        if r < cfg.crossover_prob + cfg.mutation_prob:
            parent = _tournament_select(rng, pop, cfg.tournament_k)
            return [mutate(rng, parent.tree, **mut_kw)]
        parent = _tournament_select(rng, pop, cfg.tournament_k)
        return [copy.deepcopy(parent.tree)]

    if rng.random() < cfg.crossover_prob:
        p1 = _tournament_select(rng, pop, cfg.tournament_k)
        p2 = _tournament_select(rng, pop, cfg.tournament_k)
        child_trees = list(
            crossover(
                rng,
                p1.tree,
                p2.tree,
                max_depth=cfg.max_tree_depth,
                rbf_additive_only=rbf_add,
                rbf_min_bumps=int(cfg.rbf_min_bumps),
                rbf_max_bumps=cfg.rbf_max_bumps,
            )
        )
    else:
        parent = _tournament_select(rng, pop, cfg.tournament_k)
        child_trees = [parent.tree]

    out: list[Node] = []
    for tree in child_trees:
        if rng.random() < cfg.mutation_prob:
            tree = mutate(rng, tree, **mut_kw)
        out.append(tree)
    return out


_EVAL_WORKER_CTX: EvolutionContext | None = None
_EVAL_WORKER_CFG: EvolutionConfig | None = None


def subspace_rmse(y_pred: np.ndarray, y_ref: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_ref = np.asarray(y_ref, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_pred - y_ref) ** 2)))


def _eval_objective(
    tree: Node,
    z: np.ndarray,
    ctx: EvolutionContext,
    cfg: EvolutionConfig,
    *,
    calib: tuple[float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    if not cfg.linear_calibration:
        return predict_raw_clipped(tree, z), (1.0, 0.0)
    if calib is not None:
        y, coeffs = predict_calibrated(tree, z, calib=calib)
        return y, coeffs
    return predict_calibrated(tree, z, y_ref=ctx.y_target)


def _feature_y_dense(
    tree: Node,
    y_dense: np.ndarray,
    ctx: EvolutionContext,
    cfg: EvolutionConfig,
    *,
    calib: tuple[float, float],
) -> tuple[np.ndarray, float | None, float | None]:
    """Dense y for ELA features — optionally via RF(g).

    Returns ``(y_features, rf_oob_r2, rf_vs_campaign_r2)``. OOB / vs-campaign
    R² are set only when ``rf_transform_features`` is enabled.
    """
    if not cfg.rf_transform_features:
        return y_dense, None, None
    if ctx.x_rf_train is None or ctx.z_rf_train is None:
        raise RuntimeError(
            "rf_transform_features enabled but x_rf_train/z_rf_train missing on context"
        )
    y_train, _ = _eval_objective(tree, ctx.z_rf_train, ctx, cfg, calib=calib)
    y_features, oob = rf_transform_predict(
        ctx.x_rf_train,
        y_train,
        ctx.x_dense,
        n_estimators=cfg.rf_transform_n_estimators,
        random_state=cfg.rf_transform_seed,
        return_oob=True,
    )
    vs_campaign = float(r2_score(ctx.y_target, y_features))
    return y_features, float(oob), vs_campaign


def evaluate_individual(
    ind: Individual,
    ctx: EvolutionContext,
    cfg: EvolutionConfig,
) -> Individual:
    y_dense, (a, b) = _eval_objective(ind.tree, ctx.z_dense, ctx, cfg)
    ind.calib_a, ind.calib_b = a, b
    rmse = subspace_rmse(y_dense, ctx.y_target)
    ind.subspace_rmse = rmse
    ind.complexity = tree_size(ind.tree)

    if (
        cfg.alpha_subspace > 0
        and cfg.early_reject_subspace_mult > 0
        and rmse > cfg.early_reject_subspace_mult * ctx.subspace_rmse_threshold
    ):
        ind.tier1_loss = 1.0
        ind.tier1 = {}
        ind.tier1_rel_err = {n: 1.0 for n in TIER1_NAMES}
        ind.fitness = (
            cfg.tier1_gamma * 1.0
            + cfg.alpha_subspace * rmse / max(ctx.y_range, 1e-9)
            + cfg.beta_complexity * ind.complexity
        )
        ind.accepted = False
        return ind

    if float(np.std(y_dense)) < 1e-5:
        ind.tier1_loss = 2.0
        ind.tier1 = {}
        ind.tier1_rel_err = {n: 1.0 for n in TIER1_NAMES}
        ind.fitness = cfg.tier1_gamma * 2.0 + cfg.beta_complexity * ind.complexity
        ind.accepted = False
        return ind

    y_features, rf_oob, rf_vs_campaign = _feature_y_dense(
        ind.tree, y_dense, ctx, cfg, calib=(a, b),
    )
    if float(np.std(y_features)) < 1e-5:
        ind.tier1_loss = 2.0
        ind.tier1 = {}
        ind.tier1_rel_err = {n: 1.0 for n in TIER1_NAMES}
        ind.fitness = cfg.tier1_gamma * 2.0 + cfg.beta_complexity * ind.complexity
        ind.accepted = False
        return ind

    y_campaign, _ = _eval_objective(ind.tree, ctx.z_campaign, ctx, cfg, calib=(a, b))
    tier1 = compute_tier1(
        ctx.z_dense,
        y_features,
        ctx.x_dense,
        x_campaign=ctx.x_campaign,
        y_campaign=ctx.y_campaign,
        y_campaign_pred=y_campaign if cfg.linear_calibration else None,
        maximize=ctx.maximize,
        seed=ctx.sample_seed,
    )
    if rf_oob is not None:
        tier1["oob_r2"] = float(rf_oob)
    if rf_vs_campaign is not None:
        tier1["rf_vs_campaign_r2"] = float(rf_vs_campaign)
    loss, rel = weighted_feature_loss(
        tier1,
        ctx.tier1_target,
        ctx.tier1_weights,
        feature_names=ctx.fitness_feature_names,
    )
    ind.tier1 = tier1
    ind.tier1_rel_err = rel
    ind.tier1_loss = loss

    fitness = cfg.tier1_gamma * loss + cfg.beta_complexity * ind.complexity
    if cfg.alpha_subspace > 0:
        fitness += cfg.alpha_subspace * rmse / max(ctx.y_range, 1e-9)
    if cfg.linearity_penalty_gamma > 0:
        lin_r2 = raw_linearity_r2(ctx.z_dense, evaluate_raw(ind.tree, ctx.z_dense))
        linear_penalty = max(0.0, lin_r2 - 0.75)
        if not tree_has_nonlinearity(ind.tree):
            linear_penalty += 0.35
        fitness += cfg.linearity_penalty_gamma * linear_penalty
    ind.fitness = fitness

    fit_errs = [rel[n] for n in ctx.fitness_feature_names]
    median_rel = float(np.median(fit_errs))
    tier1_ok = median_rel < cfg.tier1_acceptance_median_rel
    if cfg.require_subspace_rmse:
        ind.accepted = rmse < ctx.subspace_rmse_threshold and tier1_ok
    else:
        ind.accepted = tier1_ok
    return ind


def _copy_individual_eval(src: Individual, dst: Individual) -> None:
    dst.fitness = src.fitness
    dst.tier1_loss = src.tier1_loss
    dst.subspace_rmse = src.subspace_rmse
    dst.complexity = src.complexity
    dst.calib_a = src.calib_a
    dst.calib_b = src.calib_b
    dst.tier1 = dict(src.tier1)
    dst.tier1_rel_err = dict(src.tier1_rel_err)
    dst.accepted = src.accepted


def _eval_worker_init(
    ctx: EvolutionContext,
    cfg: EvolutionConfig,
    omp_threads: int,
) -> None:
    global _EVAL_WORKER_CTX, _EVAL_WORKER_CFG
    omp = str(max(1, int(omp_threads)))
    os.environ["OMP_NUM_THREADS"] = omp
    os.environ["MKL_NUM_THREADS"] = omp
    os.environ["OPENBLAS_NUM_THREADS"] = omp
    _EVAL_WORKER_CTX = ctx
    _EVAL_WORKER_CFG = cfg


def _eval_tree_worker(tree: Node) -> Individual:
    if _EVAL_WORKER_CTX is None or _EVAL_WORKER_CFG is None:
        raise RuntimeError("eval worker not initialized")
    return evaluate_individual(Individual(tree=tree), _EVAL_WORKER_CTX, _EVAL_WORKER_CFG)


def _evaluate_population(
    individuals: list[Individual],
    ctx: EvolutionContext,
    cfg: EvolutionConfig,
    *,
    progress_label: str | None = None,
) -> None:
    if not individuals:
        return
    n = len(individuals)
    workers = max(1, int(cfg.eval_workers))
    log_step = max(1, n // 5)

    if workers == 1:
        for i, ind in enumerate(individuals):
            evaluate_individual(ind, ctx, cfg)
            if progress_label and (i + 1) % log_step == 0:
                logger.info("  %s %d/%d", progress_label, i + 1, n)
        return

    workers = min(workers, n)
    omp_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
    logger.info(
        "Evaluating %d individuals with ProcessPoolExecutor "
        "(max_workers=%d, OMP=%d/worker)",
        n,
        workers,
        omp_threads,
    )
    mp_ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_ctx,
        initializer=_eval_worker_init,
        initargs=(ctx, cfg, omp_threads),
    ) as pool:
        results = pool.map(
            _eval_tree_worker,
            [ind.tree for ind in individuals],
            chunksize=max(1, n // (workers * 4)),
        )
    for ind, ev in zip(individuals, results):
        _copy_individual_eval(ev, ind)
    if progress_label:
        logger.info("  %s %d/%d", progress_label, n, n)


def _tournament_select(rng: random.Random, pop: list[Individual], k: int) -> Individual:
    contenders = rng.sample(pop, k=min(k, len(pop)))
    return min(contenders, key=lambda x: x.fitness)


def _init_population(rng: random.Random, ctx: EvolutionContext, cfg: EvolutionConfig) -> list[Individual]:
    pop: list[Individual] = []
    init_depth = cfg.max_tree_depth if cfg.paper_ga else cfg.max_tree_depth - 1
    allow_rbf = bool(cfg.allow_rbf) or bool(cfg.rbf_additive_only)
    oscillatory = bool(cfg.oscillatory_bias) and not cfg.paper_ga and not cfg.rbf_additive_only
    rbf_up = bool(cfg.rbf_upweight) and allow_rbf
    rbf_add = bool(cfg.rbf_additive_only)
    for _ in range(cfg.population):
        tree = random_tree(
            rng,
            n_vars=ctx.n_vars,
            max_depth=init_depth,
            paper_mode=cfg.paper_ga and not rbf_add,
            allow_rbf=allow_rbf,
            oscillatory_bias=oscillatory,
            rbf_upweight=rbf_up,
            rbf_additive_only=rbf_add,
            rbf_min_bumps=int(cfg.rbf_min_bumps),
            rbf_max_bumps=cfg.rbf_max_bumps,
        )
        pop.append(Individual(tree=tree))
    return pop


def _log_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")


def _write_snapshot_json(
    run_dir: Path,
    gen: int,
    best: Individual,
    *,
    ctx: EvolutionContext,
) -> None:
    snap_dir = run_dir / "evolution" / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generation": gen,
        "fitness": best.fitness,
        "tier1_loss": best.tier1_loss,
        "subspace_rmse": best.subspace_rmse,
        "complexity": best.complexity,
        "calib_a": best.calib_a,
        "calib_b": best.calib_b,
        "accepted": best.accepted,
        "expression": tree_to_string(best.tree),
        "tier1": best.tier1,
        "tier1_rel_err": best.tier1_rel_err,
        "tier1_target": ctx.tier1_target,
    }
    path = snap_dir / f"gen_{gen:03d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _plot_generation_landscape(
    run_dir: Path,
    gen: int,
    best: Individual,
    landscape_cache: Any,
    cfg: EvolutionConfig,
) -> None:
    land_dir = run_dir / "evolution" / "landscapes"
    png_path = land_dir / f"gen_{gen:03d}.png"
    landscape_cache.plot_generation(
        best.tree,
        png_path,
        generation=gen,
        fitness=best.fitness,
        tier1_loss=best.tier1_loss,
        subspace_rmse=best.subspace_rmse,
        accepted=best.accepted,
        paper_mode=not cfg.linear_calibration,
        calib=None if not cfg.linear_calibration else (best.calib_a, best.calib_b),
        rf_transform=cfg.rf_transform_features,
    )
    latest = land_dir / "latest.png"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(png_path.name)
    except OSError:
        import shutil

        shutil.copy2(png_path, latest)


def write_landscape_index(run_dir: Path) -> Path | None:
    """Simple HTML gallery of per-generation landscape PNGs."""
    land_dir = run_dir / "evolution" / "landscapes"
    if not land_dir.is_dir():
        return None
    images = sorted(land_dir.glob("gen_*.png"))
    if not images:
        return None
    rows = "\n".join(
        f'<div class="card"><img src="{p.name}" /><p>{p.stem.replace("_", " ")}</p></div>'
        for p in images
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Landscape progress — {run_dir.name}</title>
<style>
body {{ font-family: sans-serif; margin: 1rem; background: #111; color: #eee; }}
h1 {{ font-size: 1.1rem; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 0.75rem; }}
.card {{ background: #222; padding: 0.5rem; border-radius: 6px; max-width: 420px; }}
.card img {{ width: 100%; height: auto; display: block; }}
.card p {{ margin: 0.35rem 0 0; font-size: 0.85rem; text-align: center; }}
</style></head><body>
<h1>Best landscape per generation — {run_dir.name}</h1>
<p>Latest: <a href="latest.png">latest.png</a> · {len(images)} frames</p>
<div class="grid">{rows}</div>
</body></html>"""
    index_path = land_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def _write_history_row(writer: csv.DictWriter, gen: int, pop: list[Individual], best: Individual, dt: float) -> None:
    fitnesses = [p.fitness for p in pop]
    writer.writerow(
        {
            "generation": gen,
            "best_fitness": best.fitness,
            "mean_fitness": float(np.mean(fitnesses)),
            "best_tier1_loss": best.tier1_loss,
            "best_subspace_rmse": best.subspace_rmse,
            "best_complexity": best.complexity,
            "best_accepted": int(best.accepted),
            "elapsed_s": round(dt, 3),
        }
    )


def run_evolution(
    ctx: EvolutionContext,
    run_dir: str | Path,
    cfg: EvolutionConfig,
    *,
    target_source: str | Path | None = None,
) -> Individual:
    run_dir = Path(run_dir)
    export_run_artifacts(run_dir, ctx, target_source=target_source)

    evo_dir = run_dir / "evolution"
    evo_dir.mkdir(parents=True, exist_ok=True)
    events_path = evo_dir / "events.jsonl"
    history_path = evo_dir / "history.csv"

    rng = random.Random(cfg.seed)
    cfg.alpha_subspace = float(ctx.metadata.get("alpha_subspace", cfg.alpha_subspace))
    cfg.beta_complexity = float(ctx.metadata.get("beta_complexity", cfg.beta_complexity))
    cfg.tier1_gamma = float(ctx.metadata.get("tier1_gamma", cfg.tier1_gamma))
    cfg.linear_calibration = bool(
        ctx.metadata.get("linear_calibration", cfg.linear_calibration)
    )
    cfg.require_subspace_rmse = bool(
        ctx.metadata.get("require_subspace_rmse", cfg.require_subspace_rmse)
    )
    cfg.paper_ga = bool(ctx.metadata.get("paper_ga", cfg.paper_ga))
    cfg.paper_mode = cfg.paper_ga
    cfg.allow_rbf = bool(ctx.metadata.get("allow_rbf", cfg.allow_rbf))
    cfg.oscillatory_bias = bool(ctx.metadata.get("oscillatory_bias", cfg.oscillatory_bias))
    cfg.rbf_upweight = bool(ctx.metadata.get("rbf_upweight", cfg.rbf_upweight))
    cfg.rbf_additive_only = bool(
        ctx.metadata.get("rbf_additive_only", cfg.rbf_additive_only)
    )
    cfg.rbf_min_bumps = int(ctx.metadata.get("rbf_min_bumps", cfg.rbf_min_bumps))
    raw_max_b = ctx.metadata.get("rbf_max_bumps", cfg.rbf_max_bumps)
    cfg.rbf_max_bumps = int(raw_max_b) if raw_max_b is not None else None
    if cfg.rbf_additive_only:
        cfg.allow_rbf = True
    cfg.linearity_penalty_gamma = float(
        ctx.metadata.get("linearity_penalty_gamma", cfg.linearity_penalty_gamma)
    )
    cfg.eval_workers = int(ctx.metadata.get("eval_workers", cfg.eval_workers))

    _log_event(
        events_path,
        {
            "event": "start",
            "paper_mode": cfg.paper_ga,
            "linear_calibration": cfg.linear_calibration,
            "require_subspace_rmse": cfg.require_subspace_rmse,
            "paper_ga": cfg.paper_ga,
            "allow_rbf": bool(cfg.allow_rbf),
            "oscillatory_bias": bool(cfg.oscillatory_bias) and not cfg.paper_ga,
            "rbf_upweight": bool(cfg.rbf_upweight) and bool(cfg.allow_rbf),
            "rbf_additive_only": bool(cfg.rbf_additive_only),
            "rbf_min_bumps": int(cfg.rbf_min_bumps),
            "rbf_max_bumps": cfg.rbf_max_bumps,
            "gp_seed": cfg.seed,
            "gp_seed_source": ctx.metadata.get("gp_seed_source"),
            "sample_seed": ctx.sample_seed,
            "fitness_features": list(ctx.fitness_feature_names),
            "population": cfg.population,
            "generations": cfg.generations,
            "seed": cfg.seed,
            "n_dense": ctx.n_dense,
            "eval_workers": cfg.eval_workers,
            "tournament_k": cfg.tournament_k,
            "crossover_prob": cfg.crossover_prob,
            "mutation_prob": cfg.mutation_prob,
            "direct_transfer_prob": cfg.direct_transfer_prob,
            "elitism": _elitism_count(cfg),
            "max_tree_depth": cfg.max_tree_depth,
            "fitness_stop_threshold": cfg.fitness_stop_threshold,
            "subspace_rmse_threshold": ctx.subspace_rmse_threshold,
        },
    )

    pop = _init_population(rng, ctx, cfg)
    landscape_cache = None
    if cfg.landscape_viz and int(ctx.dim) != 3:
        logger.warning(
            "landscape_viz requested but dim=%d; ternary plots only support 3D — skipping",
            ctx.dim,
        )
    elif cfg.landscape_viz:
        from ela.visualize_pilot_3d import LandscapePlotCache

        logger.info("Building landscape plot cache (grid_n=%d)", cfg.landscape_grid_n)
        landscape_cache = LandscapePlotCache.build(
            ctx,
            grid_n=cfg.landscape_grid_n,
            rf_transform=cfg.rf_transform_features,
            rf_n_estimators=cfg.rf_transform_n_estimators,
            rf_seed=cfg.rf_transform_seed,
        )

    logger.info("Evaluating initial population (%d)", len(pop))
    _evaluate_population(pop, ctx, cfg, progress_label="init eval")

    pop.sort(key=lambda x: x.fitness)
    best = pop[0]
    t0 = time.monotonic()

    with open(history_path, "w", newline="", encoding="utf-8") as hf:
        fields = [
            "generation",
            "best_fitness",
            "mean_fitness",
            "best_tier1_loss",
            "best_subspace_rmse",
            "best_complexity",
            "best_accepted",
            "elapsed_s",
        ]
        writer = csv.DictWriter(hf, fieldnames=fields)
        writer.writeheader()
        _write_history_row(writer, 0, pop, best, time.monotonic() - t0)
        if 0 % max(cfg.snapshot_every, 1) == 0:
            _write_snapshot_json(run_dir, 0, best, ctx=ctx)
        if cfg.landscape_viz and landscape_cache is not None:
            _plot_generation_landscape(run_dir, 0, best, landscape_cache, cfg)

        for gen in range(1, cfg.generations + 1):
            next_pop: list[Individual] = [
                copy.deepcopy(p) for p in pop[: _elitism_count(cfg)]
            ]
            new_children: list[Individual] = []

            while len(next_pop) + len(new_children) < cfg.population:
                for tree in _spawn_offspring_trees(
                    rng, pop, cfg, n_vars=ctx.n_vars
                ):
                    if len(next_pop) + len(new_children) >= cfg.population:
                        break
                    new_children.append(Individual(tree=tree))

            _evaluate_population(new_children, ctx, cfg)
            next_pop.extend(new_children)

            pop = sorted(next_pop, key=lambda x: x.fitness)
            best = pop[0]
            elapsed = time.monotonic() - t0
            _write_history_row(writer, gen, pop, best, elapsed)
            hf.flush()

            if gen % cfg.snapshot_every == 0 or gen == cfg.generations:
                _write_snapshot_json(run_dir, gen, best, ctx=ctx)
            if (
                cfg.landscape_viz
                and landscape_cache is not None
                and gen % cfg.landscape_viz_every == 0
            ):
                _plot_generation_landscape(run_dir, gen, best, landscape_cache, cfg)

            _log_event(
                events_path,
                {
                    "event": "generation",
                    "generation": gen,
                    "best_fitness": best.fitness,
                    "best_subspace_rmse": best.subspace_rmse,
                    "best_tier1_loss": best.tier1_loss,
                    "best_complexity": best.complexity,
                    "accepted": best.accepted,
                    "elapsed_s": elapsed,
                },
            )
            logger.info(
                "gen %3d | fitness=%.4f tier1=%.4f rmse=%.5f size=%d accepted=%s",
                gen,
                best.fitness,
                best.tier1_loss,
                best.subspace_rmse,
                best.complexity,
                best.accepted,
            )

            if (
                cfg.fitness_stop_threshold > 0
                and best.tier1_loss < cfg.fitness_stop_threshold
            ):
                logger.info(
                    "Early stop at gen %d: tier1_loss %.6f < %.6f",
                    gen,
                    best.tier1_loss,
                    cfg.fitness_stop_threshold,
                )
                break

    _finalize_best(run_dir, best, ctx, cfg)
    if cfg.landscape_viz:
        index = write_landscape_index(run_dir)
        if index is not None:
            logger.info("Landscape gallery: %s", index)
    return best


def _finalize_best(
    run_dir: Path,
    best: Individual,
    ctx: EvolutionContext,
    cfg: EvolutionConfig,
) -> None:
    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    y_dense, (a, b) = _eval_objective(best.tree, ctx.z_dense, ctx, cfg)
    y_features, rf_oob, rf_vs_campaign = _feature_y_dense(
        best.tree, y_dense, ctx, cfg, calib=(a, b),
    )
    y_campaign, _ = _eval_objective(best.tree, ctx.z_campaign, ctx, cfg, calib=(a, b))
    tier1 = compute_tier1(
        ctx.z_dense,
        y_features,
        ctx.x_dense,
        x_campaign=ctx.x_campaign,
        y_campaign=ctx.y_campaign,
        y_campaign_pred=y_campaign if cfg.linear_calibration else None,
        maximize=ctx.maximize,
        seed=ctx.sample_seed,
    )
    if rf_oob is not None:
        tier1["oob_r2"] = float(rf_oob)
    if rf_vs_campaign is not None:
        tier1["rf_vs_campaign_r2"] = float(rf_vs_campaign)
    loss, rel = weighted_feature_loss(
        tier1,
        ctx.tier1_target,
        ctx.tier1_weights,
        feature_names=ctx.fitness_feature_names,
    )
    rmse = subspace_rmse(y_dense, ctx.y_target)
    fit_errs = [rel[n] for n in ctx.fitness_feature_names]
    median_rel = float(np.median(fit_errs))

    metrics = {
        "fitness": best.fitness,
        "linear_calibration": cfg.linear_calibration,
        "require_subspace_rmse": cfg.require_subspace_rmse,
        "paper_ga": cfg.paper_ga,
        "rf_transform_features": cfg.rf_transform_features,
        "rf_transform_n_estimators": cfg.rf_transform_n_estimators,
        "fitness_feature_names": list(ctx.fitness_feature_names),
        "tier1_loss": loss,
        "subspace_rmse": rmse,
        "subspace_rmse_threshold": ctx.subspace_rmse_threshold,
        "subspace_rmse_frac_of_range": rmse / max(ctx.y_range, 1e-9),
        "complexity": tree_size(best.tree),
        "depth": tree_depth(best.tree),
        "median_tier1_rel_err": median_rel,
        "tier1_rel_err": rel,
        "tier1_achieved": tier1,
        "tier1_target": ctx.tier1_target,
        "accepted_subspace": (
            rmse < ctx.subspace_rmse_threshold if cfg.require_subspace_rmse else None
        ),
        "accepted_tier1": median_rel < 0.10,
        "accepted": (
            rmse < ctx.subspace_rmse_threshold and median_rel < 0.10
            if cfg.require_subspace_rmse
            else median_rel < 0.10
        ),
        "y_dense_range_achieved": [float(y_dense.min()), float(y_dense.max())],
        "y_features_range": [float(y_features.min()), float(y_features.max())],
        "campaign_r2": tier1["oob_r2"],
        "rf_oob_r2": None if rf_oob is None else float(rf_oob),
        "rf_vs_campaign_r2": None if rf_vs_campaign is None else float(rf_vs_campaign),
    }
    if cfg.linear_calibration:
        metrics["calibration"] = {"a": a, "b": b}
    else:
        metrics["evaluation"] = "raw_g(z)"
    if cfg.linearity_penalty_gamma > 0:
        metrics["raw_linearity_r2"] = raw_linearity_r2(
            ctx.z_dense, evaluate_raw(best.tree, ctx.z_dense)
        )
        metrics["has_nonlinearity"] = tree_has_nonlinearity(best.tree)
    with open(best_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    recovery_names = tuple(
        name for name in ALLOWED_FITNESS_NAMES
        if name in tier1 and name in ctx.tier1_target
    )
    _, rel_all = weighted_feature_loss(
        tier1,
        ctx.tier1_target,
        {name: 1.0 for name in recovery_names},
        feature_names=recovery_names,
    )
    recovery = {
        name: {
            "target": ctx.tier1_target[name],
            "achieved": tier1[name],
            "rel_err": rel_all[name],
            "in_fitness": name in ctx.fitness_feature_names,
        }
        for name in recovery_names
    }
    with open(best_dir / "recovery.json", "w", encoding="utf-8") as f:
        json.dump(recovery, f, indent=2)
        f.write("\n")

    expr_meta: dict[str, Any] = {
        "string": tree_to_string(best.tree),
        "linear_calibration_enabled": cfg.linear_calibration,
        "paper_ga": cfg.paper_ga,
    }
    if cfg.linear_calibration:
        expr_meta["calibration"] = {"a": a, "b": b}
    if cfg.linearity_penalty_gamma > 0:
        expr_meta["raw_linearity_r2"] = raw_linearity_r2(
            ctx.z_dense, evaluate_raw(best.tree, ctx.z_dense)
        )
        expr_meta["has_nonlinearity"] = tree_has_nonlinearity(best.tree)
    dump_expression(
        best_dir / "expression.json",
        best.tree,
        metadata=expr_meta,
    )
    _write_oracle_py(best_dir / "oracle.py", linear_calibration=cfg.linear_calibration)
    logger.info("Best saved to %s (accepted=%s)", best_dir, metrics["accepted"])


def _write_oracle_py(path: Path, *, linear_calibration: bool) -> None:
    if not linear_calibration:
        body = '''"""Evolved 3D S1 landscape oracle (Muñoz S1 paper mode — raw g(z))."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ela.features import composition_to_ilr
from ela.gp_tree import predict_raw_clipped, tree_from_jsonable

_EXPR_PATH = Path(__file__).with_name("expression.json")
with _EXPR_PATH.open(encoding="utf-8") as _f:
    _EXPRESSION = tree_from_jsonable(json.load(_f)["expression"])


def predict_composition(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return predict_ilr(composition_to_ilr(x))


def predict_ilr(z: np.ndarray) -> np.ndarray:
    return predict_raw_clipped(_EXPRESSION, z)


if __name__ == "__main__":
    print("Paper-mode oracle: predict_composition(x) with shape (n, 3).")
'''
    else:
        body = '''"""Evolved 3D S1 landscape oracle (campaign-twin mode)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ela.features import composition_to_ilr
from ela.gp_tree import apply_calibration, evaluate_raw, tree_from_jsonable

_EXPR_PATH = Path(__file__).with_name("expression.json")
with _EXPR_PATH.open(encoding="utf-8") as _f:
    _META = json.load(_f)
_EXPRESSION = tree_from_jsonable(_META["expression"])
_cal = _META.get("calibration")
if not isinstance(_cal, dict):
    _legacy = _META.get("linear_calibration", {})
    _cal = _legacy if isinstance(_legacy, dict) else {}
_CALIB_A = float(_cal.get("a", 1.0))
_CALIB_B = float(_cal.get("b", 0.0))


def predict_composition(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return predict_ilr(composition_to_ilr(x))


def predict_ilr(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    return apply_calibration(evaluate_raw(_EXPRESSION, z), _CALIB_A, _CALIB_B)


if __name__ == "__main__":
    print("Campaign-mode oracle: predict_composition(x) with shape (n, 3).")
'''
    path.write_text(body, encoding="utf-8")
