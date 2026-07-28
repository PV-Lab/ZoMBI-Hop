"""
Landscape definitions for optimize/run_mobo.py.

Supports:
  • ``rf``        — Random-Forest surrogate (campaign1a or optional RF-on-CSV comparison)
  • ``synthetic`` — Direct analytic oracle from synthetic_data/oracles.py (MOBO default)
                    or ``synthetic_data.ackley.Ackley`` (realistic variant)
  • ``ela``       — Fixed evolved ELA twin from ``ela/runs/ela_3d_<jobid>/best/oracle.py``
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
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
    ela_run: str | None = None

    @property
    def render_ternary(self) -> bool:
        return (
            self.landscape in ("rf", "synthetic", "ela")
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
        if self.landscape == "ela":
            tag = self.oracle or (Path(self.ela_run).name if self.ela_run else "twin")
            return f"ELA twin ({tag}, {self.dim}D)"
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


def resolve_ela_run_dir(
    *,
    ela_run: str | None = None,
    job_id: int | str | None = None,
    oracle_path: str | None = None,
    repo_root: str | None = None,
) -> Path:
    """Resolve an ELA pilot run directory containing ``best/oracle.py``."""
    root = Path(repo_root or _REPO).resolve()
    if ela_run:
        p = Path(ela_run)
        if not p.is_absolute():
            p = root / p
        return p.resolve()
    if oracle_path:
        p = Path(oracle_path)
        if not p.is_absolute():
            p = root / p
        p = p.resolve()
        # Accept .../best/oracle.py or .../oracle.py
        if p.name == "oracle.py":
            return p.parent.parent if p.parent.name == "best" else p.parent
        return p
    if job_id is not None:
        return (root / "ela" / "runs" / f"ela_3d_{job_id}").resolve()
    raise ValueError(
        "ELA landscape requires 'ela_run', 'job_id', or 'oracle_path' in the batch JSON."
    )


def load_ela_oracle_module(run_dir: Path):
    """Import ``best/oracle.py`` from an ELA run (adds repo root for ``ela.*`` imports)."""
    oracle_py = run_dir / "best" / "oracle.py"
    if not oracle_py.is_file():
        raise FileNotFoundError(f"ELA oracle not found: {oracle_py}")
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    mod_name = f"_ela_oracle_{run_dir.name}"
    spec = importlib.util.spec_from_file_location(mod_name, oracle_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load ELA oracle from {oracle_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "predict_composition"):
        raise AttributeError(f"{oracle_py} has no predict_composition()")
    return mod


def build_ela_rf_g_predict(run_dir: Path):
    """Fit RF(g) for an ELA(RF_g) run; return ``predict_composition(X)->(N,)``.

    Matches the fitness / viz recipe: evaluate the evolved expression on the
    run's fixed ``x_rf_train``, fit an RF, predict on query compositions.
    Requires ``samples.npz`` with ``x_rf_train`` / ``z_rf_train`` and a
    recoverable expression (``best/expression.json`` or latest snapshot).
    """
    from sklearn.ensemble import RandomForestRegressor

    from ela.compile_rf_surrogate_gallery import load_landscape_source
    from ela.evolve_context import load_context_from_run

    run_dir = Path(run_dir)
    ctx = load_context_from_run(run_dir)
    if ctx.x_rf_train is None or ctx.z_rf_train is None:
        raise ValueError(
            f"{run_dir.name}: rf_transform / RF(g) requested but x_rf_train "
            "missing in samples.npz"
        )
    landscape = load_landscape_source(run_dir)
    n_est = int(ctx.metadata.get("rf_transform_n_estimators", 500))
    rf_seed = int(ctx.metadata.get("rf_transform_seed", 42))
    y_train = np.asarray(landscape.predict(ctx.z_rf_train), dtype=float).ravel()
    rf = RandomForestRegressor(
        n_estimators=n_est,
        n_jobs=1,
        random_state=rf_seed,
        bootstrap=True,
    )
    rf.fit(ctx.x_rf_train, y_train)

    def predict_composition(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.asarray(rf.predict(x), dtype=float).ravel()

    return predict_composition


def _ela_scalar_fn(predict_composition) -> ObjectiveFn:
    def fn(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).ravel()
        y = predict_composition(x.reshape(1, -1))
        return float(np.asarray(y, dtype=float).ravel()[0])

    return fn


def _log_softmax(log_x: np.ndarray) -> np.ndarray:
    z = log_x - np.max(log_x)
    x = np.exp(z)
    s = float(x.sum())
    return x / (s if s > 0 else 1.0)


def _refine_callable_extremum(
    fn: ObjectiveFn,
    x0: np.ndarray,
    *,
    maximize: bool,
    max_l1: float = 0.10,
) -> tuple[np.ndarray, float]:
    """L-BFGS-B refinement of a scalar objective on the simplex (log-softmax coords)."""
    from scipy.optimize import minimize as sp_minimize

    x0 = np.clip(np.asarray(x0, dtype=float).ravel(), 1e-12, None)
    x0 = x0 / x0.sum()
    log_x0 = np.log(np.maximum(x0, 1e-300))

    def obj(log_x: np.ndarray) -> float:
        val = float(fn(_log_softmax(log_x)))
        return -val if maximize else val

    res = sp_minimize(obj, log_x0, method="L-BFGS-B", options={"maxiter": 400, "ftol": 1e-9})
    x_opt = _log_softmax(res.x)
    if float(np.abs(x_opt - x0).sum()) > max_l1:
        return x0, float(fn(x0))
    return x_opt, float(fn(x_opt))


def auto_detect_callable_optima(
    fn: ObjectiveFn,
    grid_pts: np.ndarray,
    grid_vals: np.ndarray,
    *,
    maximize: bool,
    n_peaks: int = 3,
    min_sep: float = 0.15,
) -> list[np.ndarray]:
    """Greedy top-grid peak picking + L-BFGS-B refinement for any scalar callable."""
    order = np.argsort(grid_vals)
    if maximize:
        order = order[::-1]
    chosen: list[np.ndarray] = []
    for idx in order:
        if len(chosen) >= n_peaks:
            break
        pt = np.asarray(grid_pts[idx], dtype=float)
        if any(float(np.linalg.norm(pt - c)) < min_sep for c in chosen):
            continue
        x_ref, _ = _refine_callable_extremum(fn, pt, maximize=maximize)
        if any(float(np.linalg.norm(x_ref - c)) < min_sep for c in chosen):
            continue
        chosen.append(x_ref)
    if not chosen:
        dim = int(np.asarray(grid_pts[0]).size)
        chosen = [np.full(dim, 1.0 / dim, dtype=float)]
    return chosen


def build_ela_landscape(
    ela_run: str | Path,
    *,
    maximize: bool = True,
    true_optima: list[np.ndarray] | None = None,
    n_peaks: int = 3,
    min_sep: float = 0.15,
    grid_n: int = 80,
    time_limit_hours: float | None = 0.4,
    repo_root: str | None = None,
    use_rf_g: bool = False,
) -> LandscapeSpec:
    """Load a fixed ELA twin oracle and build a MOBO ``LandscapeSpec``.

    Peaks are taken from ``true_optima`` when provided; otherwise greedily
    detected on the run's dense Sobol sample (``X_dense.npy``) or a ternary grid.

    When ``use_rf_g`` is True, the objective is the ELA(RF_g) surface (RF fit on
    the evolved expression at the run's fixed RF-train sample), not raw ``g(z)``
    from ``best/oracle.py``.
    """
    run_dir = resolve_ela_run_dir(ela_run=str(ela_run), repo_root=repo_root)
    if use_rf_g:
        predict = build_ela_rf_g_predict(run_dir)
        fn = _ela_scalar_fn(predict)
        oracle_label = f"{run_dir.name}:RF(g)"
    else:
        mod = load_ela_oracle_module(run_dir)
        predict = mod.predict_composition
        fn = _ela_scalar_fn(predict)
        oracle_label = run_dir.name

    dense_x = run_dir / "X_dense.npy"
    if dense_x.is_file():
        sample_pts = np.load(dense_x)
        sample_vals = np.asarray(predict(sample_pts), dtype=float).ravel()
        dim = int(sample_pts.shape[1])
    else:
        dim = 3
        sample_pts = ternary_grid(grid_n) if dim == 3 else None
        sample_vals = None
        if sample_pts is not None:
            sample_vals = np.array([fn(x) for x in sample_pts], dtype=float)

    if true_optima:
        optima = [np.asarray(t, dtype=float).ravel() for t in true_optima]
        dim = int(optima[0].size)
    else:
        if sample_pts is None or sample_vals is None:
            raise ValueError(
                f"ELA run {run_dir} has no X_dense.npy and no true_optima; "
                "cannot auto-detect peaks."
            )
        optima = auto_detect_callable_optima(
            fn, sample_pts, sample_vals,
            maximize=maximize, n_peaks=n_peaks, min_sep=min_sep,
        )
        dim = int(optima[0].size)

    grid_pts = grid_vals = None
    if dim == 3:
        grid_pts = ternary_grid(grid_n)
        grid_vals = np.asarray(predict(grid_pts), dtype=float).ravel()

    return LandscapeSpec(
        landscape="ela",
        dim=dim,
        maximize=maximize,
        true_optima=optima,
        fn_callable=fn,
        grid_pts=grid_pts,
        grid_vals=grid_vals,
        time_limit_hours=0.4 if time_limit_hours is None else time_limit_hours,
        max_activations=float("inf"),
        oracle=oracle_label,
        ela_run=str(run_dir),
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

    if landscape == "ela":
        true_optima = None
        if cfg.get("true_optima"):
            true_optima = [np.asarray(t, dtype=float) for t in cfg["true_optima"]]
        return build_ela_landscape(
            cfg.get("ela_run") or resolve_ela_run_dir(
                job_id=cfg.get("job_id"),
                oracle_path=cfg.get("oracle_path"),
            ),
            maximize=bool(cfg.get("maximize", True)),
            true_optima=true_optima,
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


