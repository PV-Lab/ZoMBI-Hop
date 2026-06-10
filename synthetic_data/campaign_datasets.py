"""
Generate campaign-style synthetic datasets (3D–10D+) and sidecar metadata for MOBO.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from synthetic_data.oracles import ORACLE_CHOICES, build_oracle, normalize_rows

OBJECTIVE_COL = "Objective"
TARGET_DATASET_SIZE = 700
LOCAL_FRACTION = 0.25
LOCAL_DIRICHLET_ALPHA = 30.0
DATASET_NOISE_STD = 0.07
OUTLIER_FRAC = 0.06
OUTLIER_NOISE_MULT = 5.0
CAMPAIGN_LINE_FRACTION = 0.60
CAMPAIGN_LOCAL_FRACTION = 0.30
CAMPAIGN_UNIFORM_FRACTION = 0.10
CAMPAIGN_PTS_PER_LINE = 8
RF_N_ESTIMATORS = 500
DEFAULT_LAYOUT = "2"


def composition_column_names(dim: int) -> list[str]:
    return [f"Comp{i + 1}" for i in range(dim)]


def dataset_sizes(
    n_peaks: int,
    *,
    target: int = TARGET_DATASET_SIZE,
    local_fraction: float = LOCAL_FRACTION,
) -> tuple[int, int]:
    if n_peaks < 1:
        raise ValueError("n_peaks must be >= 1")
    n_local_total = int(round(target * local_fraction))
    n_local_per_peak = max(1, n_local_total // n_peaks)
    n_uniform = target - n_local_per_peak * n_peaks
    if n_uniform < 1:
        raise ValueError(
            f"target={target} too small for {n_peaks} peaks "
            f"at local_fraction={local_fraction}"
        )
    return n_uniform, n_local_per_peak


def _append_sample(
    rows: list[tuple[float, ...]],
    oracle,
    x: np.ndarray,
    rng: np.random.Generator,
    noise_std: float,
    *,
    outlier_frac: float = OUTLIER_FRAC,
) -> None:
    sigma = noise_std
    if outlier_frac > 0 and rng.random() < outlier_frac:
        sigma *= OUTLIER_NOISE_MULT
    y = float(oracle(x)) + float(rng.normal(0.0, sigma))
    rows.append((*np.asarray(x, dtype=float), y))


def generate_dataset(
    oracle,
    centers: list[np.ndarray],
    *,
    dim: int,
    n_uniform: int,
    n_local_per_peak: int,
    local_alpha: float,
    noise_std: float,
    seed: int,
    outlier_frac: float = OUTLIER_FRAC,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[tuple[float, ...]] = []
    comp_cols = composition_column_names(dim)

    for x in rng.dirichlet(np.ones(dim), size=n_uniform):
        _append_sample(rows, oracle, x, rng, noise_std, outlier_frac=outlier_frac)

    for center in centers:
        for x in rng.dirichlet(local_alpha * center, size=n_local_per_peak):
            _append_sample(rows, oracle, x, rng, noise_std, outlier_frac=outlier_frac)

    return pd.DataFrame(rows, columns=comp_cols + [OBJECTIVE_COL])


def generate_campaign_dataset(
    oracle,
    centers: list[np.ndarray],
    *,
    dim: int,
    target: int,
    local_alpha: float,
    noise_std: float,
    seed: int,
    pts_per_line: int = CAMPAIGN_PTS_PER_LINE,
    outlier_frac: float = OUTLIER_FRAC,
) -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(seed)
    rows: list[tuple[float, ...]] = []
    comp_cols = composition_column_names(dim)

    n_line_pts = int(round(target * CAMPAIGN_LINE_FRACTION))
    n_local_total = int(round(target * CAMPAIGN_LOCAL_FRACTION))
    n_uniform = max(0, target - n_line_pts - n_local_total)

    n_lines = max(1, n_line_pts // pts_per_line)
    pts_per_line = max(2, n_line_pts // n_lines)

    for _ in range(n_lines):
        x0 = rng.dirichlet(np.ones(dim))
        x1 = rng.dirichlet(np.ones(dim))
        for t in np.linspace(0.0, 1.0, pts_per_line):
            x = (1.0 - t) * x0 + t * x1
            _append_sample(rows, oracle, x, rng, noise_std, outlier_frac=outlier_frac)

    n_local_per_peak = max(1, n_local_total // max(1, len(centers)))
    for center in centers:
        for x in rng.dirichlet(local_alpha * center, size=n_local_per_peak):
            _append_sample(rows, oracle, x, rng, noise_std, outlier_frac=outlier_frac)

    for x in rng.dirichlet(np.ones(dim), size=n_uniform):
        _append_sample(rows, oracle, x, rng, noise_std, outlier_frac=outlier_frac)

    if len(rows) > target:
        rows = rows[:target]
    while len(rows) < target:
        x = rng.dirichlet(np.ones(dim))
        _append_sample(rows, oracle, x, rng, noise_std, outlier_frac=outlier_frac)

    desc = (
        f"{n_lines} lines × {pts_per_line} pts + "
        f"{n_local_per_peak} local × {len(centers)} peaks + "
        f"{n_uniform} uniform"
    )
    return pd.DataFrame(rows, columns=comp_cols + [OBJECTIVE_COL]), desc


def train_rf(X: np.ndarray, y: np.ndarray, *, n_estimators: int = RF_N_ESTIMATORS) -> RandomForestRegressor:
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    return rf


def metadata_path_for_csv(csv_path: str) -> str:
    base, _ = os.path.splitext(csv_path)
    return f"{base}_meta.json"


def write_metadata(path: str, meta: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def load_metadata(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_metadata_path(csv_path: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    candidate = metadata_path_for_csv(csv_path)
    return candidate if os.path.isfile(candidate) else None


def build_metadata(
    *,
    oracle: str,
    dim: int,
    layout: str,
    seed: int,
    centers: list[np.ndarray],
    maximize: bool,
    sampling: str,
    sampling_desc: str,
    n_samples: int,
    noise_std: float,
    outlier_frac: float,
    csv_path: str,
) -> dict[str, Any]:
    comp_cols = composition_column_names(dim)
    return {
        "oracle": oracle,
        "dim": dim,
        "layout": layout,
        "seed": seed,
        "maximize": maximize,
        "true_optima": [list(map(float, np.asarray(c).ravel())) for c in centers],
        "composition_columns": comp_cols,
        "objective_column": OBJECTIVE_COL,
        "n_samples": n_samples,
        "sampling": sampling,
        "sampling_desc": sampling_desc,
        "noise_std": noise_std,
        "outlier_frac": outlier_frac,
        "csv_path": os.path.abspath(csv_path),
    }


def generate_campaign_files(
    *,
    output_csv: str,
    oracle: str = "messy",
    dim: int = 3,
    layout: str = DEFAULT_LAYOUT,
    seed: int = 42,
    n_samples: int = TARGET_DATASET_SIZE,
    sampling: str = "campaign",
    noise_std: float = DATASET_NOISE_STD,
    outlier_frac: float = OUTLIER_FRAC,
    metadata_path: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate CSV + metadata sidecar; return (dataframe, metadata dict)."""
    if oracle not in ORACLE_CHOICES:
        raise ValueError(f"Unknown oracle {oracle!r}; choose from {ORACLE_CHOICES}")

    obj, centers, oracle_label = build_oracle(oracle, dim, layout, seed=seed)
    sampling_desc = ""

    if sampling == "campaign":
        df, sampling_desc = generate_campaign_dataset(
            obj, centers,
            dim=dim,
            target=n_samples,
            local_alpha=LOCAL_DIRICHLET_ALPHA,
            noise_std=noise_std,
            seed=seed,
            outlier_frac=outlier_frac,
        )
    elif sampling == "uniform":
        n_uniform, n_local_per_peak = dataset_sizes(len(centers), target=n_samples)
        sampling_desc = (
            f"{n_uniform} uniform + {n_local_per_peak} local × {len(centers)} peaks"
        )
        df = generate_dataset(
            obj, centers,
            dim=dim,
            n_uniform=n_uniform,
            n_local_per_peak=n_local_per_peak,
            local_alpha=LOCAL_DIRICHLET_ALPHA,
            noise_std=noise_std,
            seed=seed,
            outlier_frac=outlier_frac,
        )
    else:
        raise ValueError(f"Unknown sampling {sampling!r}; use 'campaign' or 'uniform'.")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)

    meta = build_metadata(
        oracle=oracle,
        dim=dim,
        layout=layout,
        seed=seed,
        centers=centers,
        maximize=bool(getattr(obj, "maximize", True)),
        sampling=sampling,
        sampling_desc=sampling_desc,
        n_samples=len(df),
        noise_std=noise_std,
        outlier_frac=outlier_frac,
        csv_path=output_csv,
    )
    meta["oracle_label"] = oracle_label

    meta_out = metadata_path or metadata_path_for_csv(output_csv)
    write_metadata(meta_out, meta)
    return df, meta


# Presets for 3D MOBO benchmarking (same layout/oracle mix as team meeting notes).
SYNTHETIC_3D_PRESETS: dict[str, dict[str, Any]] = {
    "messy": {"oracle": "messy", "maximize": True},
    "ackley": {"oracle": "ackley", "maximize": True},
    "gaussian": {"oracle": "gaussian", "maximize": True},
    "planted_bumps": {"oracle": "planted_bumps", "maximize": True},
    "rastrigin_ilr": {"oracle": "rastrigin_ilr", "maximize": True},
}
