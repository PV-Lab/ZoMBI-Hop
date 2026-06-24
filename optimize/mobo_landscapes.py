"""
Landscape definitions for optimize/run_mobo.py.

Supports:
  • ``rf``        — Random-Forest surrogate (campaign1a or optional RF-on-CSV comparison)
  • ``synthetic`` — Direct analytic oracle from synthetic_data/oracles.py (MOBO default)
                    or ``synthetic_data.ackley.Ackley`` (realistic variant)
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def resolve_surrogate_csv_path(csv_path: str | None, repo_root: str | None = None) -> str:
    """Resolve a surrogate CSV path, tolerating stale absolute paths from other machines.

    If ``csv_path`` is a stale path, looks for the same basename under ``data/``,
    ``interactive_testing/``, and the repo root (relative to ``repo_root``).
    If ``csv_path`` is missing entirely (None/empty), scans ``data/`` (then
    ``interactive_testing/``) for a CSV and uses it when the choice is unambiguous.
    """
    root = os.path.abspath(repo_root or _REPO)

    if not csv_path:
        for rel in ("data", "interactive_testing"):
            search_dir = os.path.join(root, rel)
            if not os.path.isdir(search_dir):
                continue
            csvs = sorted(glob.glob(os.path.join(search_dir, "*.csv")))
            if len(csvs) == 1:
                print(f"  [csv] no csv_path in config; using {csvs[0]}")
                return os.path.abspath(csvs[0])
            if len(csvs) > 1:
                raise FileNotFoundError(
                    f"No csv_path in config and multiple CSVs found in {rel}/: "
                    f"{[os.path.basename(c) for c in csvs]}. "
                    f"Set 'csv_path' in the source run config to disambiguate."
                )
        raise FileNotFoundError(
            "No csv_path in config and no CSV found under data/ or "
            "interactive_testing/. Set 'csv_path' in the source run config."
        )

    if os.path.isfile(csv_path):
        return os.path.abspath(csv_path)

    basename = os.path.basename(csv_path.replace("\\", "/"))
    for rel in (os.path.join("data", basename),
                os.path.join("interactive_testing", basename),
                basename):
        candidate = os.path.join(root, rel)
        if os.path.isfile(candidate):
            print(f"  [csv] resolved stale path -> {candidate}")
            return candidate

    raise FileNotFoundError(
        f"Surrogate CSV not found: {csv_path!r} "
        f"(also checked data/, interactive_testing/, repo root for {basename!r})"
    )

# ─── Composition column naming ───────────────────────────────────────────────────

CAMPAIGN_COMPOSITION_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]


def composition_column_names(dim: int) -> list[str]:
    """CSV column names for simplex compositions (FA/MA/Br when d=3)."""
    if dim == 3:
        return ["FA", "MA", "Br"]
    return [f"x{i}" for i in range(dim)]


def infer_composition_columns(df, *, explicit: list[str] | None = None) -> list[str]:
    """Pick composition columns from config, metadata-style names, or campaign CSV."""
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise ValueError(f"composition_columns missing from CSV: {missing}")
        return list(explicit)
    for candidate in (CAMPAIGN_COMPOSITION_COLS, [f"Comp{i + 1}" for i in range(20)]):
        cols = [c for c in candidate if c in df.columns]
        if len(cols) >= 2 and all(c in df.columns for c in cols):
            # Comp* columns must be contiguous from Comp1.
            if cols[0].startswith("Comp"):
                while len(cols) < 20 and f"Comp{len(cols) + 1}" in df.columns:
                    cols.append(f"Comp{len(cols) + 1}")
            return cols
    raise ValueError(
        "Could not infer composition columns; set 'composition_columns' in the batch JSON "
        f"or use {CAMPAIGN_COMPOSITION_COLS} / Comp1..CompN."
    )


# ─── Landscape spec (passed through the MOBO loop) ───────────────────────────────

ObjectiveFn = Callable[[np.ndarray], float]


@dataclass
class LandscapeSpec:
    landscape: str
    dim: int
    maximize: bool
    true_optima: list[np.ndarray]
    fn_callable: ObjectiveFn
    grid_pts: np.ndarray | None = None
    grid_vals: np.ndarray | None = None
    time_limit_hours: float | None = 0.4
    max_activations: float | None = None
    ackley_layout: str | None = None
    synthetic_seed: int | None = None
    csv_path: str | None = None
    objective_column: str = "Objective"
    composition_columns: list[str] | None = None
    oracle: str | None = None
    metadata_path: str | None = None

    @property
    def render_ternary(self) -> bool:
        return (
            self.landscape in ("rf", "synthetic")
            and self.dim == 3
            and self.grid_pts is not None
            and self.grid_vals is not None
        )

    @property
    def label(self) -> str:
        if self.landscape == "synthetic" and self.oracle == "ackley":
            return f"Ackley-{self.dim}D"
        if self.landscape == "synthetic" and self.oracle:
            return f"Synthetic-{self.oracle}-{self.dim}D-L{self.ackley_layout or '?'}"
        if self.landscape == "rf" and self.oracle:
            return f"RF surrogate ({self.oracle}, {self.dim}D)"
        if self.landscape == "rf":
            return f"RF surrogate ({self.dim}D)"
        return self.landscape


def ternary_grid(n: int = 80) -> np.ndarray:
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append([i / n, j / n, (n - i - j) / n])
    return np.array(pts, dtype=float)


def build_synthetic_landscape(
    oracle: str,
    dim: int,
    layout: str,
    *,
    seed: int = 42,
    time_limit_hours: float | None = 0.4,
    grid_n: int = 80,
    variant: str = "layout",
    n_peaks: int | None = None,
    sigma: float | None = None,
    sigma_var: float | None = None,
    noise_freq: float | None = None,
    noise_amp: float | None = None,
) -> LandscapeSpec:
    """Direct analytic oracle — no RF CSV required (RF is comparison-only)."""
    from synthetic_data.oracles import ORACLE_CHOICES, build_oracle

    if oracle not in ORACLE_CHOICES:
        raise ValueError(f"Unknown synthetic oracle {oracle!r}; choose from {ORACLE_CHOICES}")

    fn, optima, _label = build_oracle(
        oracle, dim, layout, seed=seed, variant=variant,
        n_peaks=n_peaks, sigma=sigma, sigma_var=sigma_var,
        noise_freq=noise_freq, noise_amp=noise_amp,
    )
    grid_pts = grid_vals = None
    if dim == 3:
        grid_pts = ternary_grid(grid_n)
        grid_vals = np.array([float(fn(x)) for x in grid_pts], dtype=float)

    return LandscapeSpec(
        landscape="synthetic",
        dim=dim,
        maximize=True,
        true_optima=optima,
        fn_callable=fn,
        grid_pts=grid_pts,
        grid_vals=grid_vals,
        time_limit_hours=time_limit_hours,
        max_activations=float("inf"),
        oracle=oracle,
        ackley_layout=layout,
        synthetic_seed=seed,
    )


def parse_synthetic_batch_fields(cfg: dict) -> dict:
    """Extract synthetic-oracle fields from a batch JSON config."""
    from synthetic_data.oracles import ORACLE_CHOICES

    oracle = str(cfg.get("oracle", "messy"))
    if oracle not in ORACLE_CHOICES:
        raise ValueError(f"oracle must be one of {ORACLE_CHOICES}, got {oracle!r}")
    dim = int(cfg.get("dim", 3))
    layout = str(cfg.get("layout", "2"))
    seed = int(cfg.get("seed", 42))
    variant = str(cfg.get("variant", "layout"))
    if dim < 2 or dim > 20:
        raise ValueError(f"synthetic dim must be in [2, 20], got {dim}")
    if layout == "3" and dim < 5:
        raise ValueError("synthetic layout 3 requires dim >= 5")
    if layout == "2" and dim < 3:
        raise ValueError("synthetic layout 2 requires dim >= 3")
    out = {
        "oracle": oracle, "dim": dim, "layout": layout, "seed": seed, "variant": variant,
    }
    for key in ("n_peaks", "sigma", "sigma_var", "noise_freq", "noise_amp"):
        if key in cfg:
            out[key] = cfg[key]
    return out


def build_ackley_oracle_landscape(
    dim: int,
    *,
    variant: str = "realistic",
    peak_seed: int = 0,
    time_limit_hours: float | None = 0.4,
) -> LandscapeSpec:
    """Build a ``synthetic_data.ackley.Ackley`` landscape (replaces layout-based Multi-Ackley)."""
    from synthetic_data.ackley import Ackley

    fn = Ackley(variant, dim=dim, peak_seed=peak_seed)
    true_optima = [np.asarray(c, dtype=float) for c in fn.centers]
    grid_pts = grid_vals = None
    if dim == 3:
        grid_pts = ternary_grid()
        grid_vals = fn.predict(grid_pts)
    return LandscapeSpec(
        landscape="synthetic",
        dim=dim,
        maximize=True,
        true_optima=true_optima,
        fn_callable=fn,
        grid_pts=grid_pts,
        grid_vals=grid_vals,
        time_limit_hours=time_limit_hours,
        max_activations=float("inf"),
        oracle="ackley",
        synthetic_seed=peak_seed,
    )


def build_rf_landscape(
    rf_fn: ObjectiveFn,
    true_optima: list[np.ndarray],
    grid_pts: np.ndarray | None,
    grid_vals: np.ndarray | None,
    *,
    maximize: bool,
    csv_path: str,
    objective_column: str = "Objective",
    composition_columns: list[str] | None = None,
    dim: int | None = None,
    oracle: str | None = None,
    metadata_path: str | None = None,
    time_limit_hours: float | None = 0.4,
) -> LandscapeSpec:
    resolved_dim = dim
    if resolved_dim is None and composition_columns:
        resolved_dim = len(composition_columns)
    if resolved_dim is None and true_optima:
        resolved_dim = int(np.asarray(true_optima[0]).size)
    if resolved_dim is None:
        resolved_dim = 3
    tl = 0.4 if time_limit_hours is None else time_limit_hours
    return LandscapeSpec(
        landscape="rf",
        dim=resolved_dim,
        maximize=maximize,
        true_optima=true_optima,
        fn_callable=rf_fn,
        grid_pts=grid_pts,
        grid_vals=grid_vals,
        time_limit_hours=tl,
        max_activations=float("inf"),
        csv_path=os.path.abspath(csv_path),
        objective_column=objective_column,
        composition_columns=composition_columns,
        oracle=oracle,
        metadata_path=metadata_path,
    )


def landscape_from_run_config(cfg: dict, *, build_rf_and_grid) -> LandscapeSpec:
    """Rebuild a LandscapeSpec from a persisted run_config.json."""
    landscape = cfg.get("landscape", "rf")
    time_limit = cfg.get("time_limit_hours")

    if landscape == "ackley":
        dim = int(cfg.get("dim", 10))
        variant = str(cfg.get("ackley_variant", "realistic"))
        seed = int(cfg.get("seed", 42))
        print(
            "  [config] landscape 'ackley' (layout-based Multi-Ackley) is discontinued; "
            f"rebuilding as Ackley('{variant}', dim={dim}, peak_seed={seed})",
        )
        return build_ackley_oracle_landscape(
            dim, variant=variant, peak_seed=seed, time_limit_hours=time_limit,
        )

    if landscape == "synthetic":
        syn = parse_synthetic_batch_fields(cfg)
        return build_synthetic_landscape(
            syn["oracle"], syn["dim"], syn["layout"],
            seed=syn["seed"],
            time_limit_hours=time_limit,
        )

    csv_path = resolve_surrogate_csv_path(cfg.get("csv_path"))
    obj_col = cfg.get("objective_column", "Objective")
    comp_cols = cfg.get("composition_columns")
    maximize = bool(cfg.get("maximize", False))
    true_optima = [np.asarray(t, dtype=float) for t in cfg["true_optima"]]
    _, rf_fn, grid_pts, grid_vals, resolved_cols, resolved_dim = build_rf_and_grid(
        csv_path,
        objective_column=obj_col,
        composition_columns=comp_cols,
    )
    return build_rf_landscape(
        rf_fn, true_optima, grid_pts, grid_vals,
        maximize=maximize,
        csv_path=csv_path,
        objective_column=obj_col,
        composition_columns=resolved_cols,
        dim=resolved_dim,
        oracle=cfg.get("oracle"),
        metadata_path=cfg.get("metadata_path"),
        time_limit_hours=time_limit,
    )


