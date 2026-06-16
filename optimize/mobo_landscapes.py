"""
Landscape definitions for optimize/run_mobo.py.

Supports:
  • ``rf``        — Random-Forest surrogate (campaign1a or optional RF-on-CSV comparison)
  • ``synthetic`` — Direct analytic oracle from synthetic_data/oracles.py (MOBO default)
  • ``ackley``    — Multi-Ackley sum on the d-dimensional probability simplex
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def resolve_surrogate_csv_path(csv_path: str, repo_root: str | None = None) -> str:
    """Resolve a surrogate CSV path, tolerating stale absolute paths from other machines.

    If ``csv_path`` is missing, looks for the same basename under ``data/``,
    ``interactive_testing/``, and the repo root (relative to ``repo_root``).
    """
    if os.path.isfile(csv_path):
        return os.path.abspath(csv_path)

    root = os.path.abspath(repo_root or _REPO)
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


# ─── Multi-Ackley (composition space) ─────────────────────────────────────────

_ACKLEY_A = 20.0
_ACKLEY_B = 0.2
ACKLEY_B_SKINNY = 1.2
_ACKLEY_C = 2.0 * math.pi
_ACKLEY_SCALE = 30.0

_LAYOUT_LABELS = {
    "1": "3D-analog trimodal (centroid + v0 + edge 0–1)",
    "2": "5-peak (+ v1, v2)",
    "3": "7-peak (+ v3, v4; needs d ≥ 5)",
}


def simplex_vertex(d: int, i: int) -> np.ndarray:
    p = np.zeros(d, dtype=float)
    p[i] = 1.0
    return p


def centroid_composition(d: int) -> np.ndarray:
    return np.ones(d, dtype=float) / d


def edge_midpoint(d: int, i: int, j: int) -> np.ndarray:
    p = np.zeros(d, dtype=float)
    p[i] = 0.5
    p[j] = 0.5
    return p


def _ackley_negated(
    x: np.ndarray,
    center: np.ndarray,
    *,
    a: float = _ACKLEY_A,
    b: float = _ACKLEY_B,
    c: float = _ACKLEY_C,
    scale: float = _ACKLEY_SCALE,
) -> float:
    x = np.asarray(x, dtype=float)
    center = np.asarray(center, dtype=float)
    d = x.shape[0]
    delta = x - center
    t1 = -a * math.exp(-b * math.sqrt(np.sum(delta ** 2) / d))
    t2 = -math.exp(float(np.sum(np.cos(c * delta)) / d))
    return scale * (t1 + t2 + a + math.e)


class MultiAckleyND:
    """Sum of negated Ackley bumps on Δ^d (same as run_zombi_test multimodal_ackley)."""

    maximize = True

    def __init__(
        self,
        centers: list[np.ndarray],
        *,
        b: float = ACKLEY_B_SKINNY,
        layout_name: str = "custom",
    ):
        self.centers = [np.asarray(c, dtype=float).copy() for c in centers]
        self.b = b
        self.layout_name = layout_name

    def __call__(self, x: np.ndarray) -> float:
        return float(sum(_ackley_negated(x, c, b=self.b) for c in self.centers))

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [c.copy() for c in self.centers]


def ackley_centers_for_layout(d: int, layout: str) -> list[np.ndarray]:
    if d < 2:
        raise ValueError("Multi-Ackley requires d >= 2")

    centers: list[np.ndarray] = [
        centroid_composition(d),
        simplex_vertex(d, 0),
        edge_midpoint(d, 0, 1),
    ]

    if layout in ("2", "3"):
        if d < 3:
            raise ValueError(f"Layout {layout} requires d >= 3")
        centers.append(simplex_vertex(d, 1))
        centers.append(simplex_vertex(d, 2))

    if layout == "3":
        if d < 5:
            raise ValueError("Layout 3 requires d >= 5")
        centers.append(simplex_vertex(d, 3))
        centers.append(simplex_vertex(d, 4))

    if layout not in ("1", "2", "3"):
        raise ValueError(f"Unknown layout {layout!r}; use '1', '2', or '3'.")

    return centers


def build_multi_ackley(
    d: int,
    layout: str,
    *,
    b: float = ACKLEY_B_SKINNY,
) -> tuple[MultiAckleyND, list[np.ndarray], int]:
    """Return (callable, true_optima, max_activations)."""
    centers = ackley_centers_for_layout(d, layout)
    ack = MultiAckleyND(centers, b=b, layout_name=f"layout-{layout}")
    n_act = max(2, 2 * len(ack.true_optima))
    return ack, ack.true_optima, n_act


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
    ackley_b: float | None = None
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
        if self.landscape == "ackley":
            return f"MultiAckley-{self.dim}D-L{self.ackley_layout}"
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


def build_ackley_landscape(
    dim: int,
    layout: str,
    *,
    b: float = ACKLEY_B_SKINNY,
    time_limit_hours: float | None = None,
    max_activations: float | None = None,
) -> LandscapeSpec:
    fn, optima, default_n_act = build_multi_ackley(dim, layout, b=b)
    return LandscapeSpec(
        landscape="ackley",
        dim=dim,
        maximize=True,
        true_optima=optima,
        fn_callable=fn,
        time_limit_hours=time_limit_hours,
        max_activations=max_activations if max_activations is not None else float(default_n_act),
        ackley_layout=layout,
        ackley_b=b,
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
    return LandscapeSpec(
        landscape="rf",
        dim=resolved_dim,
        maximize=maximize,
        true_optima=true_optima,
        fn_callable=rf_fn,
        grid_pts=grid_pts,
        grid_vals=grid_vals,
        time_limit_hours=time_limit_hours,
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
        layout = str(cfg.get("ackley_layout", "1"))
        b = float(cfg.get("ackley_b", ACKLEY_B_SKINNY))
        max_act = cfg.get("max_activations")
        return build_ackley_landscape(
            dim, layout, b=b,
            time_limit_hours=time_limit,
            max_activations=float(max_act) if max_act is not None else None,
        )

    if landscape == "synthetic":
        syn = parse_synthetic_batch_fields(cfg)
        return build_synthetic_landscape(
            syn["oracle"], syn["dim"], syn["layout"],
            seed=syn["seed"],
            time_limit_hours=time_limit,
        )

    csv_path = resolve_surrogate_csv_path(cfg["csv_path"])
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


def parse_ackley_batch_fields(cfg: dict) -> dict:
    """Extract Ackley fields from a batch JSON config."""
    dim = int(cfg.get("dim", 10))
    layout = str(cfg.get("layout", "1"))
    b = float(cfg.get("ackley_b", ACKLEY_B_SKINNY))
    if dim < 2 or dim > 20:
        raise ValueError(f"Ackley dim must be in [2, 20], got {dim}")
    if layout == "3" and dim < 5:
        raise ValueError("Ackley layout 3 requires dim >= 5")
    if layout == "2" and dim < 3:
        raise ValueError("Ackley layout 2 requires dim >= 3")
    max_act = cfg.get("max_activations")
    return {
        "dim": dim,
        "layout": layout,
        "ackley_b": b,
        "max_activations": float(max_act) if max_act is not None else None,
    }


# ─── Interactive Ackley startup ────────────────────────────────────────────────

def _prompt_int(default: int, label: str, lo: int, hi: int) -> int:
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        print(f"  Invalid integer; using default {default}.")
        return default
    if not lo <= val <= hi:
        print(f"  Out of range [{lo}, {hi}]; using default {default}.")
        return default
    return val


def _prompt_layout(default: str = "1") -> str:
    print("\nSelect Multi-Ackley peak layout (Enter = 1):")
    for key, desc in _LAYOUT_LABELS.items():
        print(f"  {key}. {desc}")
    raw = input("> ").strip() or default
    if raw not in _LAYOUT_LABELS:
        print(f"  Unknown choice '{raw}'; using layout {default}.")
        return default
    return raw


def _prompt_ackley_b(default: float = ACKLEY_B_SKINNY) -> float:
    print("\nAckley peak width b (Enter = skinny 1.2):")
    print("  1.2 — skinny peaks, well-separated (matches run_zombi_test multimodal)")
    print("  0.2 — standard Ackley width")
    raw = input("> ").strip().lower()
    if raw in ("", "1.2", "skinny", "s"):
        return ACKLEY_B_SKINNY
    if raw in ("0.2", "standard", "std"):
        return _ACKLEY_B
    try:
        return float(raw)
    except ValueError:
        print(f"  Invalid; using b={default}.")
        return default


def interactive_ackley_startup() -> LandscapeSpec:
    """Prompt for dimension, peak layout, and Ackley b."""
    print("=" * 70)
    print("ZoMBI-Hop MOBO — Multi-Ackley sum on Δ^d")
    print("=" * 70)
    d = _prompt_int(10, "Simplex dimension d", lo=2, hi=20)
    layout = _prompt_layout("1")
    if layout == "3" and d < 5:
        print("  Layout 3 needs d ≥ 5; falling back to layout 2.")
        layout = "2"
    if layout == "2" and d < 3:
        print("  Layout 2 needs d ≥ 3; falling back to layout 1.")
        layout = "1"
    b = _prompt_ackley_b()
    return build_ackley_landscape(d, layout, b=b, time_limit_hours=None)
