"""Fixed samples and targets for S1 landscape evolution."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ela.features import (
    composition_to_ilr,
    load_campaign_rows,
    sample_simplex_sobol,
    train_rf_surrogate,
)
from ela.tier1 import (
    DEFAULT_WEIGHTS,
    TIER1_NAMES,
    compute_tier1,
    extract_tier1_from_characterize,
    load_target_json,
    save_tier1_target,
)


@dataclass
class EvolutionContext:
    dim: int
    maximize: bool
    sample_seed: int
    n_dense: int
    y_min: float
    y_max: float
    y_range: float
    subspace_rmse_threshold: float
    tier1_target: dict[str, float]
    tier1_weights: dict[str, float]
    x_dense: np.ndarray
    z_dense: np.ndarray
    y_target: np.ndarray
    x_campaign: np.ndarray
    y_campaign: np.ndarray
    z_campaign: np.ndarray
    n_vars: int
    metadata: dict[str, Any]

    @property
    def ilr_dim(self) -> int:
        return self.n_vars


def _dense_n(dim: int, n_samples: int | None) -> int:
    target_n = dim * 1000
    return n_samples if n_samples is not None else (1 << (target_n - 1).bit_length())


def build_context(
    *,
    db_path: str | Path,
    target_json: str | Path | None = None,
    objective_column: str = "Objective",
    maximize: bool = True,
    sample_seed: int = 42,
    n_dense: int | None = None,
    alpha_subspace: float = 10.0,
    beta_complexity: float = 0.001,
    subspace_rmse_frac: float = 0.02,
) -> EvolutionContext:
    db_path = Path(db_path)
    x_campaign, y_campaign = load_campaign_rows(db_path, objective_column=objective_column)
    dim = int(x_campaign.shape[1])
    explicit_n_dense = n_dense

    if target_json is not None:
        tgt = load_target_json(target_json)
        tier1_target = tgt["tier1"]
        maximize = bool(tgt.get("maximize", maximize))
        sample_seed = int(tgt.get("sample_seed", sample_seed))
        if explicit_n_dense is None:
            n_dense = int(tgt.get("n_dense_sample", _dense_n(dim, None)))
        else:
            n_dense = explicit_n_dense
        y_dense_range = tgt.get("y_dense_range")
    else:
        tier1_target = None
        y_dense_range = None
        n_dense = _dense_n(dim, explicit_n_dense)

    rf = train_rf_surrogate(x_campaign, y_campaign)
    x_dense = sample_simplex_sobol(dim, n_dense, seed=sample_seed)
    y_target = rf.predict(x_dense)
    z_dense = composition_to_ilr(x_dense)
    z_campaign = composition_to_ilr(x_campaign)

    y_min = float(y_target.min())
    y_max = float(y_target.max())
    y_range = y_max - y_min
    if y_dense_range is not None:
        y_min, y_max = float(y_dense_range[0]), float(y_dense_range[1])
        y_range = y_max - y_min

    if tier1_target is None:
        tier1_target = compute_tier1(
            z_dense,
            y_target,
            x_dense,
            x_campaign=x_campaign,
            y_campaign=y_campaign,
            maximize=maximize,
            seed=sample_seed,
        )

    subspace_rmse_threshold = subspace_rmse_frac * max(y_range, 1e-9)

    metadata: dict[str, Any] = {
        "db_path": str(db_path.resolve()),
        "objective_column": objective_column,
        "alpha_subspace": alpha_subspace,
        "beta_complexity": beta_complexity,
        "subspace_rmse_frac": subspace_rmse_frac,
        "y_campaign_range": [float(y_campaign.min()), float(y_campaign.max())],
        "y_dense_range": [y_min, y_max],
        "n_campaign": int(x_campaign.shape[0]),
    }

    return EvolutionContext(
        dim=dim,
        maximize=maximize,
        sample_seed=sample_seed,
        n_dense=n_dense,
        y_min=y_min,
        y_max=y_max,
        y_range=y_range,
        subspace_rmse_threshold=subspace_rmse_threshold,
        tier1_target=tier1_target,
        tier1_weights=dict(DEFAULT_WEIGHTS),
        x_dense=x_dense,
        z_dense=z_dense,
        y_target=y_target,
        x_campaign=x_campaign,
        y_campaign=y_campaign,
        z_campaign=z_campaign,
        n_vars=z_dense.shape[1],
        metadata=metadata,
    )


def export_run_artifacts(
    run_dir: Path,
    ctx: EvolutionContext,
    *,
    target_source: str | Path | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "X_dense.npy", ctx.x_dense)
    np.save(run_dir / "Z_dense.npy", ctx.z_dense)
    samples_path = run_dir / "samples.npz"
    np.savez_compressed(
        samples_path,
        x_dense=ctx.x_dense,
        z_dense=ctx.z_dense,
        y_target=ctx.y_target,
        x_campaign=ctx.x_campaign,
        y_campaign=ctx.y_campaign,
        z_campaign=ctx.z_campaign,
    )

    target_payload = {
        "source_path": str(Path(target_source).resolve()) if target_source else None,
        "maximize": ctx.maximize,
        "dim": ctx.dim,
        "sample_seed": ctx.sample_seed,
        "n_dense_sample": ctx.n_dense,
        "y_dense_range": [ctx.y_min, ctx.y_max],
        "tier1": ctx.tier1_target,
        "weights": ctx.tier1_weights,
        "tier1_names": list(TIER1_NAMES),
        "subspace_rmse_threshold": ctx.subspace_rmse_threshold,
    }
    with open(run_dir / "target.json", "w", encoding="utf-8") as f:
        json.dump(target_payload, f, indent=2)
        f.write("\n")

    save_tier1_target(run_dir / "tier1_target.json", target_payload)

    config = {
        **ctx.metadata,
        "maximize": ctx.maximize,
        "dim": ctx.dim,
        "sample_seed": ctx.sample_seed,
        "n_dense_sample": ctx.n_dense,
        "n_vars": ctx.n_vars,
        "tier1_weights": ctx.tier1_weights,
        "subspace_rmse_threshold": ctx.subspace_rmse_threshold,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def load_context_from_run(run_dir: str | Path) -> EvolutionContext:
    run_dir = Path(run_dir)
    with open(run_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    with open(run_dir / "target.json", encoding="utf-8") as f:
        target = json.load(f)
    data = np.load(run_dir / "samples.npz")
    y_min, y_max = target["y_dense_range"]
    y_range = y_max - y_min
    return EvolutionContext(
        dim=int(config["dim"]),
        maximize=bool(config.get("maximize", True)),
        sample_seed=int(config["sample_seed"]),
        n_dense=int(config["n_dense_sample"]),
        y_min=float(y_min),
        y_max=float(y_max),
        y_range=float(y_range),
        subspace_rmse_threshold=float(target["subspace_rmse_threshold"]),
        tier1_target={k: float(target["tier1"][k]) for k in TIER1_NAMES},
        tier1_weights={k: float(target["weights"].get(k, 1.0)) for k in TIER1_NAMES},
        x_dense=data["x_dense"],
        z_dense=data["z_dense"],
        y_target=data["y_target"],
        x_campaign=data["x_campaign"],
        y_campaign=data["y_campaign"],
        z_campaign=data["z_campaign"],
        n_vars=int(data["z_dense"].shape[1]),
        metadata=config,
    )


def tier1_from_ela_full_json(path: str | Path) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return extract_tier1_from_characterize(data)
