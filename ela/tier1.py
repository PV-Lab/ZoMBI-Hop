"""Tier-1 ELA target vector (10 features) for S1 digital-twin evolution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sklearn.metrics import r2_score

from ela.features import feature_median_lipschitz, feature_oob_r2
from ela.pflacco_port import (
    calculate_ela_distribution,
    calculate_ela_level,
    calculate_ela_meta,
    calculate_entropy_y,
    entropic_significance,
)

TIER1_NAMES: tuple[str, ...] = (
    "R2_Q",
    "CN",
    "H_Y",
    "xi_1",
    "gamma_Y",
    "EL25",
    "LQ25",
    "PKS",
    "oob_r2",
    "median_lipschitz",
)

# Feature weights for fitness (see DIGITAL_TWIN_S1.md).
DEFAULT_WEIGHTS: dict[str, float] = {
    "R2_Q": 1.0,
    "CN": 1.0,
    "H_Y": 1.0,
    "xi_1": 0.5,
    "gamma_Y": 1.0,
    "EL25": 1.0,
    "LQ25": 1.0,
    "PKS": 1.5,  # multimodality — do not down-weight for haystack targets
    "oob_r2": 0.0,  # RF OOB target ≠ campaign R² of evolved g; exclude from fitness
    "median_lipschitz": 1.0,
}


def feature_campaign_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """In-sample R² of evolved landscape on measured campaign rows."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) < 3 or np.std(y_true) < 1e-12:
        return 0.0
    return float(r2_score(y_true, y_pred))


def compute_tier1(
    z: np.ndarray,
    y: np.ndarray,
    x_comp: np.ndarray,
    *,
    x_campaign: np.ndarray,
    y_campaign: np.ndarray,
    y_campaign_pred: np.ndarray | None = None,
    maximize: bool = True,
    seed: int = 42,
) -> dict[str, float]:
    """Compute Tier-1 features using the same definitions as ``ela_full`` output."""
    meta = calculate_ela_meta(z, y)
    level = calculate_ela_level(z, y, quantiles=[0.25], maximize=maximize)
    distr = calculate_ela_distribution(y)
    ent = entropic_significance(z, y, seed=seed)
    el25 = level["ela_level.mcva_lda_25"]
    eq25 = level["ela_level.mcva_qda_25"]
    if y_campaign_pred is not None:
        oob = feature_campaign_r2(y_campaign, y_campaign_pred)
    else:
        oob = feature_oob_r2(x_campaign, y_campaign)
    return {
        "R2_Q": float(meta["ela_meta.quad_simple.adj_r2"]),
        "CN": float(meta["ela_meta.quad_simple.cond"]),
        "H_Y": float(calculate_entropy_y(y)),
        "xi_1": float(ent["entropic.xi_1"]),
        "gamma_Y": float(distr["ela_distr.skewness"]),
        "EL25": float(el25),
        "LQ25": float(el25 / eq25) if eq25 > 1e-12 else float("nan"),
        "PKS": float(distr["ela_distr.number_of_peaks"]),
        "oob_r2": float(oob),
        "median_lipschitz": float(feature_median_lipschitz(x_comp, y)),
    }


def tier1_vector(features: dict[str, float]) -> np.ndarray:
    return np.array([float(features[n]) for n in TIER1_NAMES], dtype=float)


def weighted_feature_loss(
    achieved: dict[str, float],
    target: dict[str, float],
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Weighted RMS error per feature; returns (loss, per_feature_relative_error)."""
    w = weights or DEFAULT_WEIGHTS
    errs: dict[str, float] = {}
    sq = 0.0
    wsum = 0.0
    for name in TIER1_NAMES:
        t = float(target[name])
        a = float(achieved[name])
        if not np.isfinite(a) or not np.isfinite(t):
            rel = 1.0
        else:
            scale = max(abs(t), 0.1)
            rel = min(abs(a - t) / scale, 2.0)
        wt = float(w.get(name, 1.0))
        errs[name] = rel
        sq += wt * rel * rel
        wsum += wt
    loss = float(np.sqrt(sq / max(wsum, 1e-12)))
    return loss, errs


def extract_tier1_from_characterize(result: dict[str, Any]) -> dict[str, float]:
    """Pull Tier-1 dict from ``characterize_campaign_surrogate`` or ``ela_full`` JSON."""
    if "feature_groups" in result:
        m = result["feature_groups"]["munoz_33"]
        z = result["feature_groups"]["zombi"]
        return {
            "R2_Q": m["R2_Q"],
            "CN": m["CN"],
            "H_Y": m["H_Y"],
            "xi_1": m["xi_1"],
            "gamma_Y": m["gamma_Y"],
            "EL25": m["EL25"],
            "LQ25": m["LQ25"],
            "PKS": m["PKS"],
            "oob_r2": z["oob_r2"],
            "median_lipschitz": z["median_lipschitz"],
        }
    feats = result.get("features", result)
    return {k: float(feats[k]) for k in TIER1_NAMES}


def load_target_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    target = extract_tier1_from_characterize(data)
    return {
        "source_path": str(path.resolve()),
        "maximize": bool(data.get("maximize", True)),
        "dim": int(data.get("dim", 3)),
        "sample_seed": int(data.get("sample_seed", 42)),
        "n_dense_sample": int(data.get("n_dense_sample", 4096)),
        "y_dense_range": data.get("y_dense_range"),
        "tier1": target,
    }


def save_tier1_target(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path
