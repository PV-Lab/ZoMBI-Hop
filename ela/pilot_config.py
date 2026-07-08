"""Load and resolve ELA S1 pilot configuration from JSON."""
from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ela.tier1 import CAMPAIGN_WEIGHTS, MUNOZ_8_NAMES, PAPER_WEIGHTS, TIER1_NAMES

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "pilot_config.json"
PROBE_CONFIG_PATH = Path(__file__).resolve().parent / "pilot_config_probe.json"

MODE_PRESETS: dict[str, dict[str, Any]] = {
    "anchored": {
        "label": "anchored (Muñoz 8)",
        "munoz_8_fitness": True,
        "linear_calibration": True,
        "require_subspace_rmse": True,
        "paper_ga": True,
        "alpha_subspace": 3.0,
        "tier1_gamma": 1.0,
        "linearity_penalty_gamma": 0.0,
    },
    "pure_paper": {
        "label": "pure paper (Muñoz S1)",
        "munoz_8_fitness": True,
        "linear_calibration": False,
        "require_subspace_rmse": False,
        "paper_ga": True,
        "alpha_subspace": 0.0,
        "tier1_gamma": 1.0,
        "linearity_penalty_gamma": 0.0,
    },
    "campaign": {
        "label": "campaign-twin",
        "munoz_8_fitness": False,
        "linear_calibration": True,
        "require_subspace_rmse": True,
        "paper_ga": False,
        "alpha_subspace": 3.0,
        "tier1_gamma": 5.0,
        "linearity_penalty_gamma": 3.0,
    },
}

CAMPAIGN_GA_OVERRIDES: dict[str, Any] = {
    "tournament_k": 5,
    "crossover_prob": 0.7,
    "mutation_prob": 0.25,
    "direct_transfer_prob": 0.0,
    "elitism": 2,
    "elitism_frac": 0.0,
    "max_tree_depth": 7,
    "fitness_stop_threshold": 0.0,
}


@dataclass
class ResolvedPilotConfig:
    name: str
    description: str
    mode: str
    mode_label: str
    source_path: Path

    db: Path
    target: Path

    munoz_8_fitness: bool
    linear_calibration: bool
    require_subspace_rmse: bool
    paper_ga: bool

    alpha_subspace: float
    beta_complexity: float
    tier1_gamma: float
    linearity_penalty_gamma: float
    subspace_rmse_frac: float
    tier1_acceptance_median_rel: float
    tier1_weights: dict[str, float] | None

    population: int
    generations: int
    tournament_k: int
    crossover_prob: float
    mutation_prob: float
    direct_transfer_prob: float
    elitism: int
    elitism_frac: float
    max_tree_depth: int
    fitness_stop_threshold: float
    early_reject_subspace_mult: float

    n_dense: int | None
    snapshot_every: int
    landscape_viz: bool
    landscape_every: int
    landscape_grid_n: int
    post_viz: bool
    grid_n: int

    eval_workers: int | None
    log_level: str
    seed: int | None
    quick: bool

    raw: dict[str, Any] = field(repr=False)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "mode": self.mode,
            "mode_label": self.mode_label,
            "source_path": str(self.source_path.resolve()),
            "data": {"db": str(self.db), "target": str(self.target)},
            "fitness": {
                "alpha_subspace": self.alpha_subspace,
                "beta_complexity": self.beta_complexity,
                "tier1_gamma": self.tier1_gamma,
                "linearity_penalty_gamma": self.linearity_penalty_gamma,
                "subspace_rmse_frac": self.subspace_rmse_frac,
                "tier1_acceptance_median_rel": self.tier1_acceptance_median_rel,
                "linear_calibration": self.linear_calibration,
                "require_subspace_rmse": self.require_subspace_rmse,
                "munoz_8_fitness": self.munoz_8_fitness,
                "tier1_weights": self.tier1_weights,
            },
            "ga": {
                "paper_ga": self.paper_ga,
                "population": self.population,
                "generations": self.generations,
                "tournament_k": self.tournament_k,
                "crossover_prob": self.crossover_prob,
                "mutation_prob": self.mutation_prob,
                "direct_transfer_prob": self.direct_transfer_prob,
                "elitism": self.elitism,
                "elitism_frac": self.elitism_frac,
                "max_tree_depth": self.max_tree_depth,
                "fitness_stop_threshold": self.fitness_stop_threshold,
                "early_reject_subspace_mult": self.early_reject_subspace_mult,
            },
            "sampling": {"n_dense": self.n_dense},
            "viz": {
                "snapshot_every": self.snapshot_every,
                "landscape_viz": self.landscape_viz,
                "landscape_every": self.landscape_every,
                "landscape_grid_n": self.landscape_grid_n,
                "post_viz": self.post_viz,
                "grid_n": self.grid_n,
            },
            "runtime": {
                "eval_workers": self.eval_workers,
                "log_level": self.log_level,
                "seed": self.seed,
                "quick": self.quick,
            },
        }


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    val = raw.get(key, {})
    if val is None:
        return {}
    if not isinstance(val, dict):
        raise ValueError(f"config[{key!r}] must be an object")
    return val


def load_pilot_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be a JSON object")
    data = deepcopy(data)
    data["_source_path"] = str(path.resolve())
    return data


def _coalesce(preset: dict[str, Any], section: dict[str, Any], key: str, default: Any) -> Any:
    if key in section and section[key] is not None:
        return section[key]
    if key in preset and preset[key] is not None:
        return preset[key]
    return default


def _normalize_tier1_weights(
    weights: dict[str, float] | None,
    *,
    munoz_8_fitness: bool,
) -> dict[str, float] | None:
    if weights is None:
        return None
    names = MUNOZ_8_NAMES if munoz_8_fitness else TIER1_NAMES
    out = {name: float(weights.get(name, 1.0)) for name in names}
    return out


def resolve_pilot_config(
    raw: dict[str, Any],
    *,
    repo_root: Path,
    cli: dict[str, Any] | None = None,
) -> ResolvedPilotConfig:
    cli = cli or {}
    source_path = Path(raw.get("_source_path", DEFAULT_CONFIG_PATH))

    mode = str(cli.get("mode") or raw.get("mode", "anchored"))
    if mode not in MODE_PRESETS:
        raise ValueError(
            f"unknown mode {mode!r}; choose from {sorted(MODE_PRESETS)}"
        )
    preset = MODE_PRESETS[mode]

    data = _section(raw, "data")
    fitness = _section(raw, "fitness")
    ga = _section(raw, "ga")
    sampling = _section(raw, "sampling")
    viz = _section(raw, "viz")
    runtime = _section(raw, "runtime")

    db = Path(cli.get("db") or data.get("db", "data/2nd_real_run.db"))
    target = Path(cli.get("target") or data.get("target", "data/2nd_real_run_ela_full.json"))
    if not db.is_absolute():
        db = (repo_root / db).resolve()
    if not target.is_absolute():
        target = (repo_root / target).resolve()

    munoz_8 = bool(_coalesce(preset, fitness, "munoz_8_fitness", True))
    linear_cal = bool(_coalesce(preset, fitness, "linear_calibration", True))
    require_rmse = bool(_coalesce(preset, fitness, "require_subspace_rmse", True))
    paper_ga = bool(_coalesce(preset, ga, "paper_ga", preset.get("paper_ga", True)))

    if cli.get("no_calibrate"):
        linear_cal = False
    if cli.get("no_require_subspace_rmse"):
        require_rmse = False
    if cli.get("campaign_mode"):
        mode = "campaign"
        preset = MODE_PRESETS["campaign"]
        munoz_8 = False
        linear_cal = True
        require_rmse = True
        paper_ga = False
    if cli.get("pure_paper"):
        mode = "pure_paper"
        preset = MODE_PRESETS["pure_paper"]
        munoz_8 = True
        linear_cal = False
        require_rmse = False
        paper_ga = True

    alpha = float(cli.get("alpha") if cli.get("alpha") is not None else _coalesce(preset, fitness, "alpha_subspace", 3.0))
    if not require_rmse and cli.get("alpha") is None and fitness.get("alpha_subspace") is None:
        alpha = 0.0

    beta = float(cli.get("beta") if cli.get("beta") is not None else fitness.get("beta_complexity", 0.001))
    tier1_gamma = float(
        cli.get("tier1_gamma")
        if cli.get("tier1_gamma") is not None
        else _coalesce(preset, fitness, "tier1_gamma", 1.0)
    )
    linearity = float(
        cli.get("linearity_penalty")
        if cli.get("linearity_penalty") is not None
        else _coalesce(preset, fitness, "linearity_penalty_gamma", 0.0)
    )
    subspace_frac = float(fitness.get("subspace_rmse_frac", 0.02))
    tier1_accept = float(fitness.get("tier1_acceptance_median_rel", 0.10))

    tier1_weights_raw = fitness.get("tier1_weights")
    if tier1_weights_raw is None and not munoz_8:
        tier1_weights_raw = dict(CAMPAIGN_WEIGHTS)
    elif tier1_weights_raw is None and munoz_8:
        tier1_weights_raw = None
    tier1_weights = _normalize_tier1_weights(tier1_weights_raw, munoz_8_fitness=munoz_8)

    quick = bool(cli.get("quick") or runtime.get("quick", False))

    if paper_ga:
        default_pop, default_gens = 400, 100
    else:
        default_pop, default_gens = 120, 60

    population = int(cli.get("population") if cli.get("population") is not None else ga.get("population", default_pop))
    generations = int(
        cli.get("generations") if cli.get("generations") is not None else ga.get("generations", default_gens)
    )
    if quick:
        population = int(cli.get("population") if cli.get("population") is not None else ga.get("population", 24))
        generations = int(cli.get("generations") if cli.get("generations") is not None else ga.get("generations", 8))

    ga_defaults = {
        "tournament_k": 10,
        "crossover_prob": 0.6,
        "mutation_prob": 0.3,
        "direct_transfer_prob": 0.1,
        "elitism": 0,
        "elitism_frac": 0.1,
        "max_tree_depth": 10,
        "fitness_stop_threshold": 1e-3,
        "early_reject_subspace_mult": 0.0,
    }
    if not paper_ga:
        ga_defaults.update(CAMPAIGN_GA_OVERRIDES)

    if quick and require_rmse and ga.get("early_reject_subspace_mult") is None:
        ga_defaults["early_reject_subspace_mult"] = 3.0

    tournament_k = int(ga.get("tournament_k", ga_defaults["tournament_k"]))
    crossover_prob = float(ga.get("crossover_prob", ga_defaults["crossover_prob"]))
    mutation_prob = float(ga.get("mutation_prob", ga_defaults["mutation_prob"]))
    direct_transfer = float(ga.get("direct_transfer_prob", ga_defaults["direct_transfer_prob"]))
    elitism = int(ga.get("elitism", ga_defaults["elitism"]))
    elitism_frac = float(ga.get("elitism_frac", ga_defaults["elitism_frac"]))
    max_tree_depth = int(ga.get("max_tree_depth", ga_defaults["max_tree_depth"]))
    fitness_stop = float(ga.get("fitness_stop_threshold", ga_defaults["fitness_stop_threshold"]))
    early_reject = float(
        cli.get("early_reject_mult")
        if cli.get("early_reject_mult") is not None
        else ga.get("early_reject_subspace_mult", ga_defaults["early_reject_subspace_mult"])
    )

    n_dense = cli.get("n_dense") if cli.get("n_dense") is not None else sampling.get("n_dense")
    n_dense = int(n_dense) if n_dense is not None else None

    snapshot_every = int(viz.get("snapshot_every", 5))
    if quick:
        snapshot_every = min(snapshot_every, 2)

    landscape_viz = bool(viz.get("landscape_viz", True))
    if cli.get("no_landscape_viz"):
        landscape_viz = False
    post_viz = bool(viz.get("post_viz", True))
    if cli.get("no_viz"):
        post_viz = False

    eval_workers = cli.get("eval_workers") if cli.get("eval_workers") is not None else runtime.get("eval_workers")
    eval_workers = int(eval_workers) if eval_workers is not None else None

    seed = cli.get("seed") if cli.get("seed") is not None else runtime.get("seed")
    seed = int(seed) if seed is not None else None

    return ResolvedPilotConfig(
        name=str(raw.get("name", mode)),
        description=str(raw.get("description", "")),
        mode=mode,
        mode_label=str(preset["label"]),
        source_path=source_path,
        db=db,
        target=target,
        munoz_8_fitness=munoz_8,
        linear_calibration=linear_cal,
        require_subspace_rmse=require_rmse,
        paper_ga=paper_ga,
        alpha_subspace=alpha,
        beta_complexity=beta,
        tier1_gamma=tier1_gamma,
        linearity_penalty_gamma=linearity,
        subspace_rmse_frac=subspace_frac,
        tier1_acceptance_median_rel=tier1_accept,
        tier1_weights=tier1_weights,
        population=population,
        generations=generations,
        tournament_k=tournament_k,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        direct_transfer_prob=direct_transfer,
        elitism=elitism,
        elitism_frac=elitism_frac,
        max_tree_depth=max_tree_depth,
        fitness_stop_threshold=fitness_stop,
        early_reject_subspace_mult=early_reject,
        n_dense=n_dense,
        snapshot_every=snapshot_every,
        landscape_viz=landscape_viz,
        landscape_every=max(1, int(viz.get("landscape_every", 1))),
        landscape_grid_n=int(viz.get("landscape_grid_n", 100)),
        post_viz=post_viz,
        grid_n=int(viz.get("grid_n", 200)),
        eval_workers=eval_workers,
        log_level=str(cli.get("log_level") or runtime.get("log_level", "INFO")),
        seed=seed,
        quick=quick,
        raw=raw,
    )


def write_resolved_config(run_dir: Path, resolved: ResolvedPilotConfig) -> Path:
    """Copy source JSON and write fully resolved snapshot into the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    src_copy = run_dir / "pilot_config.source.json"
    shutil.copy2(resolved.source_path, src_copy)
    out = run_dir / "pilot_config.resolved.json"
    payload = resolved.to_log_dict()
    payload["resolved_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return out


def default_tier1_weights(munoz_8_fitness: bool) -> dict[str, float]:
    return dict(PAPER_WEIGHTS) if munoz_8_fitness else dict(CAMPAIGN_WEIGHTS)
