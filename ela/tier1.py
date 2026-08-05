"""Tier-1 ELA target vector (10 features) for S1 digital-twin evolution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sklearn.metrics import r2_score

from ela.features import compute_spatial_lipschitz_features, feature_oob_r2
from ela.pflacco_port import (
    calculate_nbc,
    munoz_table1_features,
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

# Muñoz & Smith-Miles (2019) Strategy S1 — 8 clustered ELA features for fitness.
MUNOZ_8_NAMES: tuple[str, ...] = TIER1_NAMES[:8]

PAPER_WEIGHTS: dict[str, float] = {name: 1.0 for name in MUNOZ_8_NAMES}

# Muñoz et al. (2019) Table 1 — all 33 named features (order matches munoz_table1_features).
MUNOZ_33_NAMES: tuple[str, ...] = (
    "FDC",
    "DISP1pct",
    "R2_L",
    "R2_LI",
    "R2_Q",
    "R2_QI",
    "beta_min",
    "beta_max",
    "CN",
    "EL10",
    "EQ10",
    "LQ10",
    "ET10",
    "EL25",
    "EQ25",
    "LQ25",
    "ET25",
    "EL50",
    "EQ50",
    "LQ50",
    "ET50",
    "gamma_Y",
    "H_Y",
    "kappa_Y",
    "PKS",
    "Hmax",
    "eps_S",
    "M0",
    "xi_D",
    "xi_1",
    "xi_2",
    "sigma_1",
    "sigma_2",
)

# ZoMBI campaign-twin extensions (--campaign-mode).
CAMPAIGN_WEIGHTS: dict[str, float] = {
    "R2_Q": 1.0,
    "CN": 1.0,
    "H_Y": 1.0,
    "xi_1": 0.5,
    "gamma_Y": 1.0,
    "EL25": 1.0,
    "LQ25": 1.0,
    "PKS": 1.5,
    "oob_r2": 0.0,
    "median_lipschitz": 0.0,
}

# Backward-compatible alias.
DEFAULT_WEIGHTS = CAMPAIGN_WEIGHTS

# Mersmann / Jones literature extras (Muñoz-33; optional fitness extensions).
LITERATURE_EXTRA_NAMES: tuple[str, ...] = ("R2_QI", "FDC")

# Spatial roughness extras — kill edge-localized / heavy-tailed Lipschitz cheats.
SPATIAL_ROUGHNESS_NAMES: tuple[str, ...] = (
    "interior_median_lipschitz",
    "lipschitz_p20",
    "lipschitz_p80",
    "edge_interior_lip_ratio",
    "ma_region_median_lipschitz",
    "fa_region_median_lipschitz",
    "br_region_median_lipschitz",
    "local_lipschitz_cv",
    "tile_min_median_lipschitz",
)

# Nearest-Better Clustering (composition-L2); short names ↔ flacco keys.
NBC_FLACCO_KEYS: dict[str, str] = {
    "NBC_mean_ratio": "nbc.nn_nb.mean_ratio",
    "NBC_sd_ratio": "nbc.nn_nb.sd_ratio",
    "NBC_cor": "nbc.nn_nb.cor",
    "NBC_dist_cv": "nbc.dist_ratio.coeff_var",
    "NBC_fitness_cor": "nbc.nb_fitness.cor",
}
NBC_NAMES: tuple[str, ...] = tuple(NBC_FLACCO_KEYS.keys())

# Dense-sample R² between RF(g) (or g) and the campaign RF target surface.
RF_SURFACE_R2_NAMES: tuple[str, ...] = ("rf_vs_campaign_r2",)

# Deduped: TIER1 / literature extras overlap Muñoz-33; keep Muñoz-33 first.
ALLOWED_FITNESS_NAMES: tuple[str, ...] = tuple(
    dict.fromkeys(
        MUNOZ_33_NAMES
        + TIER1_NAMES
        + LITERATURE_EXTRA_NAMES
        + SPATIAL_ROUGHNESS_NAMES
        + NBC_NAMES
        + RF_SURFACE_R2_NAMES
    )
)

# Set B — campaign twin fingerprint (Muñoz 8 + FDC + R²_QI + lipschitz; no ξ₁).
SET_B_NAMES: tuple[str, ...] = (
    "R2_Q",
    "R2_QI",
    "CN",
    "H_Y",
    "gamma_Y",
    "EL25",
    "LQ25",
    "PKS",
    "FDC",
    "median_lipschitz",
)

SET_B_WEIGHTS: dict[str, float] = {
    "R2_Q": 1.0,
    "R2_QI": 0.75,
    "CN": 0.5,
    "H_Y": 1.0,
    "gamma_Y": 1.0,
    "EL25": 1.0,
    "LQ25": 1.0,
    "PKS": 1.0,
    "FDC": 1.0,
    "median_lipschitz": 1.0,
}

# Set C — ~12 features: structure + composition FDC + Lipschitz + NBC (spatial multimodal).
# Drops ξ₁ / CN / R²_QI / oob; de-emphasizes PKS (Y-histogram ≠ spatial peaks).
SET_C_NAMES: tuple[str, ...] = (
    "R2_Q",
    "H_Y",
    "gamma_Y",
    "EL25",
    "LQ25",
    "FDC",
    "PKS",
    "median_lipschitz",
    "tile_min_median_lipschitz",
    "NBC_mean_ratio",
    "NBC_sd_ratio",
    "NBC_dist_cv",
)

SET_C_WEIGHTS: dict[str, float] = {name: 1.0 for name in SET_C_NAMES}


def validate_fitness_features(names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Validate and dedupe fitness feature names (order preserved)."""
    if not names:
        raise ValueError(
            f"fitness_features must not be empty; choose from {list(ALLOWED_FITNESS_NAMES)}"
        )
    unknown = [n for n in names if n not in ALLOWED_FITNESS_NAMES]
    if unknown:
        raise ValueError(
            f"unknown fitness features {unknown}; allowed: {list(ALLOWED_FITNESS_NAMES)}"
        )
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def resolve_fitness_weights(
    feature_names: tuple[str, ...],
    weights: dict[str, float] | str | None,
    *,
    fallback: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Build per-feature weights for the active fitness subset.

  ``None`` or ``"uniform"`` → weight 1.0 on every selected feature.
    """
    if weights in (None, "uniform"):
        fb = fallback or {}
        return {name: float(fb.get(name, 1.0)) for name in feature_names}
    return {
        name: float(weights.get(name, (fallback or {}).get(name, 1.0)))
        for name in feature_names
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
    oob_r2: float | None = None,
    maximize: bool = True,
    seed: int = 42,
) -> dict[str, float]:
    """Compute fitness features using the same definitions as ``ela_full`` output.

    Always includes the full Muñoz-33 block plus ZoMBI Lipschitz extras, NBC, and
    ``oob_r2`` so configs can select any ``ALLOWED_FITNESS_NAMES`` subset.

    ``oob_r2`` override: when set (e.g. OOB of RF(g) under ELA(RF_g)), use it
    instead of campaign-row OOB / campaign R².
    """
    x_comp = np.asarray(x_comp, dtype=float)
    dim = int(x_comp.shape[1])
    m33 = munoz_table1_features(z, y, x_comp, maximize=maximize, dim=dim)
    if oob_r2 is not None:
        oob = float(oob_r2)
    elif y_campaign_pred is not None:
        oob = feature_campaign_r2(y_campaign, y_campaign_pred)
    else:
        oob = feature_oob_r2(x_campaign, y_campaign)
    lip = compute_spatial_lipschitz_features(x_comp, y)
    nbc = calculate_nbc(x_comp, y, maximize=maximize)
    out: dict[str, float] = {
        **{k: float(m33[k]) for k in MUNOZ_33_NAMES},
        "oob_r2": float(oob),
        **{k: float(v) for k, v in lip.items()},
    }
    for short, flacco_key in NBC_FLACCO_KEYS.items():
        out[short] = float(nbc[flacco_key])
    return out


def tier1_vector(features: dict[str, float]) -> np.ndarray:
    return np.array([float(features[n]) for n in TIER1_NAMES], dtype=float)


def weighted_feature_loss(
    achieved: dict[str, float],
    target: dict[str, float],
    weights: dict[str, float] | None = None,
    *,
    feature_names: tuple[str, ...] | None = None,
) -> tuple[float, dict[str, float]]:
    """Weighted RMS error per feature; returns (loss, per_feature_relative_error)."""
    names = feature_names or TIER1_NAMES
    w = weights or CAMPAIGN_WEIGHTS
    errs: dict[str, float] = {}
    sq = 0.0
    wsum = 0.0
    for name in names:
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


def _json_feature_float(value: Any) -> float | None:
    """Coerce a JSON feature value to float; ``null``/non-numeric → None.

    ``save_lambda_target`` writes NaN/Inf as JSON null, so loaders must tolerate
    None (common for dim-dependent roughness extras on higher simplices).
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def extract_tier1_from_characterize(result: dict[str, Any]) -> dict[str, float]:
    """Pull fitness-feature dict from ``characterize_campaign_surrogate`` / ``ela_full``."""
    if "feature_groups" in result:
        m = result["feature_groups"]["munoz_33"]
        z = result["feature_groups"]["zombi"]
        out: dict[str, float] = {}
        for name in MUNOZ_33_NAMES:
            if name not in m:
                continue
            val = _json_feature_float(m[name])
            if val is not None:
                out[name] = val
        if "oob_r2" in z:
            val = _json_feature_float(z["oob_r2"])
            if val is not None:
                out["oob_r2"] = val
        # Spatial Lipschitz extras (median + interior / tile / regions / …).
        for name in SPATIAL_ROUGHNESS_NAMES + ("median_lipschitz",):
            raw = z[name] if name in z else m.get(name)
            val = _json_feature_float(raw)
            if val is not None:
                out[name] = val
        nbc_grp = result["feature_groups"].get("flacco_nbc") or {}
        for short, flacco_key in NBC_FLACCO_KEYS.items():
            raw = nbc_grp[flacco_key] if flacco_key in nbc_grp else m.get(short)
            val = _json_feature_float(raw)
            if val is not None:
                out[short] = val
        return out
    feats = result.get("features", result)
    out = {}
    for k in ALLOWED_FITNESS_NAMES:
        if k not in feats:
            continue
        val = _json_feature_float(feats[k])
        if val is not None:
            out[k] = val
    return out


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
