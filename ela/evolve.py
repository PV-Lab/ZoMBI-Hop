"""Genetic programming evolution for S1 landscape recreation."""
from __future__ import annotations

import copy
import csv
import json
import logging
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
    random_tree,
    raw_linearity_r2,
    tree_depth,
    tree_has_nonlinearity,
    tree_size,
    tree_to_string,
)
from ela.tier1 import TIER1_NAMES, compute_tier1, weighted_feature_loss

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
    population: int = 120
    generations: int = 60
    tournament_k: int = 5
    crossover_prob: float = 0.7
    mutation_prob: float = 0.25
    elitism: int = 2
    max_tree_depth: int = 8
    alpha_subspace: float = 3.0
    beta_complexity: float = 0.001
    tier1_gamma: float = 5.0
    linearity_penalty_gamma: float = 3.0
    snapshot_every: int = 5
    seed: int = 0
    early_reject_subspace_mult: float = 0.0  # 0 = disabled; set e.g. 3.0 only for --quick
    landscape_viz: bool = True
    landscape_viz_every: int = 1
    landscape_grid_n: int = 100


def subspace_rmse(y_pred: np.ndarray, y_ref: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_ref = np.asarray(y_ref, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_pred - y_ref) ** 2)))


def evaluate_individual(
    ind: Individual,
    ctx: EvolutionContext,
    cfg: EvolutionConfig,
) -> Individual:
    raw = evaluate_raw(ind.tree, ctx.z_dense)
    lin_r2 = raw_linearity_r2(ctx.z_dense, raw)
    y_dense, (a, b) = predict_calibrated(
        ind.tree, ctx.z_dense, y_ref=ctx.y_target,
    )
    ind.calib_a, ind.calib_b = a, b
    rmse = subspace_rmse(y_dense, ctx.y_target)
    ind.subspace_rmse = rmse
    ind.complexity = tree_size(ind.tree)

    reject_bound = (
        cfg.early_reject_subspace_mult * ctx.subspace_rmse_threshold
        if cfg.early_reject_subspace_mult > 0
        else float("inf")
    )
    if rmse > reject_bound:
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

    y_campaign, _ = predict_calibrated(
        ind.tree, ctx.z_campaign, calib=(a, b),
    )
    tier1 = compute_tier1(
        ctx.z_dense,
        y_dense,
        ctx.x_dense,
        x_campaign=ctx.x_campaign,
        y_campaign=ctx.y_campaign,
        y_campaign_pred=y_campaign,
        maximize=ctx.maximize,
        seed=ctx.sample_seed,
    )
    loss, rel = weighted_feature_loss(tier1, ctx.tier1_target, ctx.tier1_weights)
    ind.tier1 = tier1
    ind.tier1_rel_err = rel
    ind.tier1_loss = loss
    linear_penalty = max(0.0, lin_r2 - 0.75)
    if not tree_has_nonlinearity(ind.tree):
        linear_penalty += 0.35
    ind.fitness = (
        cfg.tier1_gamma * loss
        + cfg.alpha_subspace * rmse / max(ctx.y_range, 1e-9)
        + cfg.beta_complexity * ind.complexity
        + cfg.linearity_penalty_gamma * linear_penalty
    )
    ind.accepted = rmse < ctx.subspace_rmse_threshold and float(np.median(list(rel.values()))) < 0.10
    return ind


def _tournament_select(rng: random.Random, pop: list[Individual], k: int) -> Individual:
    contenders = rng.sample(pop, k=min(k, len(pop)))
    return min(contenders, key=lambda x: x.fitness)


def _init_population(rng: random.Random, ctx: EvolutionContext, cfg: EvolutionConfig) -> list[Individual]:
    pop: list[Individual] = []
    for _ in range(cfg.population):
        tree = random_tree(
            rng,
            n_vars=ctx.n_vars,
            max_depth=cfg.max_tree_depth - 1,
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
        calib=(best.calib_a, best.calib_b),
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

    _log_event(
        events_path,
        {
            "event": "start",
            "population": cfg.population,
            "generations": cfg.generations,
            "seed": cfg.seed,
            "n_dense": ctx.n_dense,
            "subspace_rmse_threshold": ctx.subspace_rmse_threshold,
        },
    )

    pop = _init_population(rng, ctx, cfg)
    landscape_cache = None
    if cfg.landscape_viz:
        from ela.visualize_pilot_3d import LandscapePlotCache

        logger.info("Building landscape plot cache (grid_n=%d)", cfg.landscape_grid_n)
        landscape_cache = LandscapePlotCache.build(ctx, grid_n=cfg.landscape_grid_n)

    logger.info("Evaluating initial population (%d)", len(pop))
    for i, ind in enumerate(pop):
        evaluate_individual(ind, ctx, cfg)
        if (i + 1) % max(1, len(pop) // 5) == 0:
            logger.info("  init eval %d/%d", i + 1, len(pop))

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
            _plot_generation_landscape(run_dir, 0, best, landscape_cache)

        for gen in range(1, cfg.generations + 1):
            next_pop: list[Individual] = [copy.deepcopy(p) for p in pop[: cfg.elitism]]

            while len(next_pop) < cfg.population:
                if rng.random() < cfg.crossover_prob:
                    p1 = _tournament_select(rng, pop, cfg.tournament_k)
                    p2 = _tournament_select(rng, pop, cfg.tournament_k)
                    c1, c2 = crossover(
                        rng,
                        p1.tree,
                        p2.tree,
                        max_depth=cfg.max_tree_depth,
                    )
                    child_trees = [c1, c2]
                else:
                    parent = _tournament_select(rng, pop, cfg.tournament_k)
                    child_trees = [parent.tree]

                for tree in child_trees:
                    if len(next_pop) >= cfg.population:
                        break
                    if rng.random() < cfg.mutation_prob:
                        tree = mutate(
                            rng,
                            tree,
                            n_vars=ctx.n_vars,
                            max_depth=cfg.max_tree_depth,
                        )
                    child = Individual(tree=tree)
                    evaluate_individual(child, ctx, cfg)
                    next_pop.append(child)

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
                _plot_generation_landscape(run_dir, gen, best, landscape_cache)

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
                "gen %3d | fitness=%.4f tier1=%.4f rmse=%.5f lin=%.2f size=%d accepted=%s",
                gen,
                best.fitness,
                best.tier1_loss,
                best.subspace_rmse,
                raw_linearity_r2(ctx.z_dense, evaluate_raw(best.tree, ctx.z_dense)),
                best.complexity,
                best.accepted,
            )

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

    y_dense, (a, b) = predict_calibrated(best.tree, ctx.z_dense, y_ref=ctx.y_target)
    y_campaign, _ = predict_calibrated(best.tree, ctx.z_campaign, calib=(a, b))
    tier1 = compute_tier1(
        ctx.z_dense,
        y_dense,
        ctx.x_dense,
        x_campaign=ctx.x_campaign,
        y_campaign=ctx.y_campaign,
        y_campaign_pred=y_campaign,
        maximize=ctx.maximize,
        seed=ctx.sample_seed,
    )
    loss, rel = weighted_feature_loss(tier1, ctx.tier1_target, ctx.tier1_weights)
    rmse = subspace_rmse(y_dense, ctx.y_target)
    median_rel = float(np.median(list(rel.values())))

    metrics = {
        "fitness": best.fitness,
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
        "accepted_subspace": rmse < ctx.subspace_rmse_threshold,
        "accepted_tier1": median_rel < 0.10,
        "accepted": rmse < ctx.subspace_rmse_threshold and median_rel < 0.10,
        "y_dense_range_achieved": [float(y_dense.min()), float(y_dense.max())],
        "campaign_r2": tier1["oob_r2"],
        "linear_calibration": {"a": a, "b": b},
        "raw_linearity_r2": raw_linearity_r2(ctx.z_dense, evaluate_raw(best.tree, ctx.z_dense)),
        "has_nonlinearity": tree_has_nonlinearity(best.tree),
    }
    with open(best_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    recovery = {
        name: {
            "target": ctx.tier1_target[name],
            "achieved": tier1[name],
            "rel_err": rel[name],
        }
        for name in TIER1_NAMES
    }
    with open(best_dir / "recovery.json", "w", encoding="utf-8") as f:
        json.dump(recovery, f, indent=2)
        f.write("\n")

    dump_expression(
        best_dir / "expression.json",
        best.tree,
        metadata={
            "string": tree_to_string(best.tree),
            "linear_calibration": {"a": a, "b": b},
            "raw_linearity_r2": raw_linearity_r2(ctx.z_dense, evaluate_raw(best.tree, ctx.z_dense)),
            "has_nonlinearity": tree_has_nonlinearity(best.tree),
        },
    )
    _write_oracle_py(best_dir / "oracle.py", ctx)
    logger.info("Best saved to %s (accepted=%s)", best_dir, metrics["accepted"])


def _write_oracle_py(path: Path, ctx: EvolutionContext) -> None:
    code = '''"""Evolved 3D S1 landscape oracle (auto-generated)."""
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
_cal = _META.get("linear_calibration", {})
_CALIB_A = float(_cal.get("a", 1.0))
_CALIB_B = float(_cal.get("b", 0.0))


def predict_composition(x: np.ndarray) -> np.ndarray:
    """Predict objective for composition(s) on the 3-simplex."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    z = composition_to_ilr(x)
    return predict_ilr(z)


def predict_ilr(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    raw = evaluate_raw(_EXPRESSION, z)
    return apply_calibration(raw, _CALIB_A, _CALIB_B)


if __name__ == "__main__":
    print("Evolved oracle ready. Use predict_composition(x) with shape (n, 3).")
'''
    path.write_text(code, encoding="utf-8")
