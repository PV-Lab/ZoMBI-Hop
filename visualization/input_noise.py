"""
visualization/input_noise.py
============================
Estimate the **average input noise** of a real ZoMBI-Hop run — the discrepancy
between the compositions we *ask* the printer to deposit ("sent" / expected) and
the compositions that are actually printed and read back ("real" / actual).

For every collected sample we form the 3-vector ``sent - real`` in composition
space (FAPbI3, MAPbI3, MAPbBr3) and report the mean Euclidean magnitude across
the whole run, plus a per-component breakdown.

Two logging regimes are handled, detected automatically:

  * **Correctly-logged runs** store the sent composition alongside the real one,
    so the noise is read off directly. This is the case for
      - run directories, where ``X_all_expected`` (sent) and ``X_all_actual``
        (real) are reconstructed from the snapshots, and
      - data files that carry explicit sent columns (e.g. ``FAPbI3_sent`` …).

  * **Legacy / incorrectly-logged runs** only ever stored the real composition
    that came back — the sent composition was never written. There the sent
    composition is *reconstructed* per print line: a straight line is fit through
    the first and last sample of the line (in real-composition space) and the
    line's N samples are spread evenly along it. Those evenly-spaced points are
    treated as the sent compositions. (By construction the first and last sample
    of every line then carry zero noise.)

    This reconstruction is a *lower bound*, and a badly biased one: fitting the
    line through the realised endpoints absorbs the whole systematic
    requested-vs-realised offset into the fit, leaving only the scatter about it.
    On run_39af it under-reports the noise by roughly 4x. Prefer a composition
    log whenever one exists.

The authoritative source is a run's ``composition_log.jsonl``, written per
objective call with both the compositions the optimiser *asked* for (``sent``)
and the ones the printer *realised* (``measured``). Nothing is reconstructed and
no endpoint is forced to zero error, so this is the number that should calibrate
``input_noise``. ``--run`` picks it up automatically when the file is present;
``--comp-log`` points at one directly.

A "line" is one printed row of up to 24 samples. In a data file the line index
is the ``Iteration`` column; in a run directory each acquisition batch added to
the data handler is one line.

Usage
-----
  conda activate zombi-hop
  python visualization/input_noise.py --db data/2nd_real_run.db
  python visualization/input_noise.py --db data/2nd_real_run.db --per-line
  python visualization/input_noise.py --run runs/run_7eb9
  python visualization/input_noise.py --comp-log runs/run_39af
  python visualization/input_noise.py --db data/2nd_real_run.db --plot noise.png
  python visualization/input_noise.py --ternary
  python visualization/input_noise.py --run runs/run_7eb9 --ternary
  python visualization/input_noise.py --db data/2nd_real_run.db --ternary --input-noise 0.05

Flags
-----
  --db PATH        Data file (.db results table or .csv campaign) to analyse.
  --run PATH       Run directory (or bare run name under runs/) to analyse. Uses
                   the run's composition_log.jsonl when present, else snapshots.
  --comp-log PATH  A composition_log.jsonl (or the run directory holding one).
  --rail NAME      Restrict a composition log to one rail (default: all rails).
  --snapshot N     Snapshot to reconstruct up to for --run (default: latest.txt).
  --force-legacy   Ignore any logged sent compositions and always reconstruct
                   them from the per-line endpoints (useful for sanity checks).
  --per-line       Also print the mean noise of each individual line.
  --plot PATH      Save a histogram of per-sample noise magnitudes to PATH.
  --ternary        Show an interactive ternary plot of the compositions with the
                   GP minimum lengthscale drawn as a reference circle. If no
                   data source is given, random demo points are generated.
  --input-noise V  Override the input noise / GP min lengthscale shown on the
                   ternary (composition L2 units; default: 0.128 or read from
                   run config.json).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

# ── project root on sys.path so `src` / sibling imports resolve ────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.utils.datahandler import reconstruct_snapshot_tensors  # noqa: E402
from visualization.plot_run import (  # noqa: E402
    DB_COMP_COLS,
    _default_snapshot,
    _is_csv,
    _resolve_db_path,
    _resolve_run_dir,
)

_DEFAULT_INPUT_NOISE = 0.128  # composition L2; DataHandler default

# Column-name patterns tried, per component, when looking for an explicitly
# logged "sent" composition in a data file (correct-logging case).
_SENT_PATTERNS = ("{c}_sent", "sent_{c}", "{c}_expected", "expected_{c}", "{c}_target")

# Two real simplex points spaced closer than this (L2) make a degenerate line we
# cannot meaningfully interpolate along, so its interior noise is left at 0.
_DEGENERATE_LINE_EPS = 1e-9


# ── line container ─────────────────────────────────────────────────────────────

class Line:
    """One printed line: real compositions and (sent) compositions, both (n, d)."""

    def __init__(self, real: np.ndarray, sent: np.ndarray | None, key):
        self.key = key
        self.real = np.asarray(real, dtype=float)
        self.sent = None if sent is None else np.asarray(sent, dtype=float)

    def resolved_sent(self) -> np.ndarray:
        """Sent compositions, reconstructing them from the endpoints if absent."""
        if self.sent is not None:
            return self.sent
        return reconstruct_sent_from_endpoints(self.real)


def reconstruct_sent_from_endpoints(real: np.ndarray) -> np.ndarray:
    """Spread ``n`` points evenly along the straight line through real[0]→real[-1].

    The first and last reconstructed points coincide with the real endpoints, so
    the endpoints carry zero noise; the interior points are the model of what we
    *intended* to print, evenly spaced between them.
    """
    real = np.asarray(real, dtype=float)
    n = real.shape[0]
    if n < 2:
        return real.copy()
    first, last = real[0], real[-1]
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return first[None, :] * (1.0 - t) + last[None, :] * t


# ── data loading: data file (.db / .csv) ───────────────────────────────────────

def _detect_sent_columns(have: set[str]) -> list[str] | None:
    """Return matching sent-composition column names (one per component) or None."""
    cols: list[str] = []
    for c in DB_COMP_COLS:
        match = next((p.format(c=c) for p in _SENT_PATTERNS if p.format(c=c) in have), None)
        if match is None:
            return None
        cols.append(match)
    return cols


def load_db_lines(db_path: Path, force_legacy: bool) -> tuple[list[Line], bool]:
    """Load per-line real/sent compositions from a .db results table or .csv.

    Returns ``(lines, correct_logging)``. Lines are grouped by the ``Iteration``
    column, in row order so each line's first/last row are its print endpoints.
    """
    if _is_csv(db_path):
        import pandas as pd

        df = pd.read_csv(db_path)
        have = set(df.columns)
    else:
        con = sqlite3.connect(str(db_path))
        try:
            have = {r[1] for r in con.execute("PRAGMA table_info(results)")}
        finally:
            con.close()

    missing = [c for c in DB_COMP_COLS + ["Iteration"] if c not in have]
    if missing:
        raise RuntimeError(f"{db_path.name} is missing required columns: {missing}")

    sent_cols = None if force_legacy else _detect_sent_columns(have)
    real_cols = DB_COMP_COLS
    want = ["Iteration", *real_cols, *(sent_cols or [])]

    if _is_csv(db_path):
        sub = df[want].apply(pd.to_numeric, errors="coerce")
        sub = sub.dropna(subset=["Iteration", *real_cols])
        rows = sub.to_numpy(dtype=float)
    else:
        con = sqlite3.connect(str(db_path))
        try:
            sel = ", ".join(f'"{c}"' for c in want)
            where = " AND ".join(f'"{c}" IS NOT NULL' for c in ["Iteration", *real_cols])
            rows = np.asarray(
                con.execute(f"SELECT {sel} FROM results WHERE {where} ORDER BY rowid").fetchall(),
                dtype=float,
            )
        finally:
            con.close()

    if rows.shape[0] == 0:
        raise RuntimeError(f"No usable rows in {db_path.name}")

    iters = rows[:, 0]
    real = rows[:, 1:4]
    sent_all = rows[:, 4:7] if sent_cols else None

    # If sent columns exist but are entirely null/NaN, fall back to legacy.
    correct = sent_cols is not None and sent_all is not None and not np.isnan(sent_all).all()

    lines: list[Line] = []
    for it in _ordered_unique(iters):
        mask = iters == it
        lines.append(Line(real[mask], sent_all[mask] if correct else None, key=int(it)))
    return lines, correct


def _ordered_unique(values: np.ndarray) -> list:
    """Distinct values in first-seen order (preserves print/iteration order)."""
    seen: list = []
    seen_set: set = set()
    for v in values.tolist():
        if v not in seen_set:
            seen_set.add(v)
            seen.append(v)
    return seen


# ── data loading: composition log ──────────────────────────────────────────────

COMP_LOG_NAME = "composition_log.jsonl"


def resolve_comp_log(path_like: str | Path) -> Path:
    """Resolve a --comp-log argument to an actual composition_log.jsonl file.

    Accepts the file itself, a run directory containing one, or a bare run name
    under runs/ (the same spellings --run accepts).
    """
    p = Path(path_like)
    if p.is_file():
        return p
    for cand in (p / COMP_LOG_NAME,):
        if cand.is_file():
            return cand
    run_dir = _resolve_run_dir(str(path_like))
    cand = run_dir / COMP_LOG_NAME
    if cand.is_file():
        return cand
    raise RuntimeError(f"No {COMP_LOG_NAME} found at or under {path_like}")


def load_comp_log_lines(
    log_path: Path, force_legacy: bool, rail: str | None = None
) -> tuple[list[Line], bool]:
    """Load per-line sent/measured compositions from a ``composition_log.jsonl``.

    One :class:`Line` per (objective call, rail): ``sent`` is what the optimiser
    asked for and ``measured`` is what came back, both full ``d``-dim simplex
    points. This is the only source where the sent composition is recorded rather
    than inferred, so it needs no reconstruction and no regime detection —
    ``force_legacy`` is still honoured for A/B comparison against the line-fit
    estimate, since that comparison is the whole reason to distrust the latter.

    Rows whose sent and measured blocks disagree in length, or that carry fewer
    than one sample, are skipped: a call can be logged with an empty result when
    the objective returned nothing.
    """
    import json

    lines: list[Line] = []
    with log_path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{log_path.name}:{lineno} is not valid JSON: {exc}") from exc
            call = rec.get("call", lineno)
            for r in rec.get("rails", []):
                name = r.get("name", "?")
                if rail is not None and name != rail:
                    continue
                sent = np.asarray(r.get("sent") or [], dtype=float)
                meas = np.asarray(r.get("measured") or [], dtype=float)
                if meas.ndim != 2 or meas.shape[0] == 0:
                    continue
                if sent.shape != meas.shape:
                    continue
                lines.append(Line(meas, None if force_legacy else sent, key=f"{call}/{name}"))
    if not lines:
        raise RuntimeError(f"No usable rows in {log_path}" + (f" for rail {rail!r}" if rail else ""))
    return lines, not force_legacy


# ── data loading: run directory ────────────────────────────────────────────────

def load_run_lines(
    run_dir: Path, snapshot: str | None, force_legacy: bool, rail: str | None = None
) -> tuple[list[Line], bool]:
    """Load real/sent compositions from a run directory.

    Prefers the run's ``composition_log.jsonl`` when present: it records the sent
    composition per objective call, so it needs no reconstruction and is not
    limited to d=3. Falls back to the snapshots otherwise, where
    ``X_all_expected`` is the sent composition and ``X_all_actual`` the real one.
    When those are identical the run predates correct logging; the whole run is
    then treated as a single line for endpoint reconstruction (snapshots do not
    record per-line batch boundaries).
    """
    comp_log = run_dir / COMP_LOG_NAME
    if comp_log.is_file():
        return load_comp_log_lines(comp_log, force_legacy, rail)

    snapshot = snapshot or _default_snapshot(run_dir)
    tensors = reconstruct_snapshot_tensors(run_dir, snapshot, device="cpu")
    actual = tensors.get("X_all_actual")
    expected = tensors.get("X_all_expected")
    if actual is None or actual.shape[0] == 0:
        raise RuntimeError(f"No datapoints reconstructed from {run_dir}/{snapshot}")
    real = actual.detach().cpu().numpy().astype(float)
    sent = None if expected is None else expected.detach().cpu().numpy().astype(float)
    if real.shape[1] != 3:
        raise ValueError(f"Only d=3 runs are supported (got d={real.shape[1]}).")

    correct = (not force_legacy) and sent is not None and not np.allclose(sent, real)
    if correct:
        return [Line(real, sent, key=0)], True
    # Legacy: no per-line info in snapshots → reconstruct over the full run.
    return [Line(real, None, key=0)], False


# ── noise computation ──────────────────────────────────────────────────────────

def compute_noise(lines: list[Line]) -> dict:
    """Aggregate per-sample input noise (sent − real) across all lines.

    ``pooled_std`` is the statistic that calibrates the ``input_noise``
    hyperparameter: the per-component standard deviation of the sent−real
    residual, pooled over every component and sample. The optimiser's
    ``input_noise`` is a single isotropic per-component sigma, so that is the
    like-for-like quantity — not the mean L2 magnitude, which is larger by
    roughly sqrt(d) and is reported here only for continuity with earlier
    analyses.
    """
    diffs = []          # (n, d) signed per-component differences
    per_line = []       # (key, n, mean magnitude) per line
    for ln in lines:
        sent = ln.resolved_sent()
        d = sent - ln.real
        diffs.append(d)
        mag = np.linalg.norm(d, axis=1)
        per_line.append((ln.key, ln.real.shape[0], float(mag.mean()) if mag.size else 0.0))

    dim = diffs[0].shape[1] if diffs else len(DB_COMP_COLS)
    D = np.concatenate(diffs, axis=0) if diffs else np.zeros((0, dim))
    mag = np.linalg.norm(D, axis=1)
    return {
        "n_points": int(D.shape[0]),
        "n_lines": len(lines),
        "dim": int(dim),
        "mean_magnitude": float(mag.mean()) if mag.size else 0.0,
        "median_magnitude": float(np.median(mag)) if mag.size else 0.0,
        "max_magnitude": float(mag.max()) if mag.size else 0.0,
        "per_component_mae": np.abs(D).mean(axis=0).tolist() if D.size else [0.0] * dim,
        "per_component_bias": D.mean(axis=0).tolist() if D.size else [0.0] * dim,
        "per_component_std": D.std(axis=0).tolist() if D.size else [0.0] * dim,
        "pooled_std": float(D.reshape(-1).std()) if D.size else 0.0,
        "magnitudes": mag,
        "per_line": per_line,
    }


# ── reporting ──────────────────────────────────────────────────────────────────

def _print_report(stats: dict, correct: bool, source: str, per_line: bool) -> None:
    regime = "correctly-logged (sent read directly)" if correct else (
        "legacy (sent reconstructed from per-line endpoints) - LOWER BOUND")
    print(f"Source        : {source}")
    print(f"Logging regime: {regime}")
    print(f"Lines         : {stats['n_lines']}")
    print(f"Samples       : {stats['n_points']}")
    print()
    print(f"input_noise (pooled per-component std): {stats['pooled_std']:.5f}   <-- calibration value")
    print()
    print(f"Average input noise (||sent - real||) : {stats['mean_magnitude']:.5f}")
    print(f"  median                              : {stats['median_magnitude']:.5f}")
    print(f"  max                                 : {stats['max_magnitude']:.5f}")
    dim = stats.get("dim", len(DB_COMP_COLS))
    names = DB_COMP_COLS if dim == len(DB_COMP_COLS) else [f"x{i}" for i in range(dim)]
    print()
    print(f"  {'component':>10}  {'MAE':>9}  {'bias':>9}  {'std':>9}")
    for nm, a, b, c in zip(names, stats["per_component_mae"],
                           stats["per_component_bias"], stats["per_component_std"]):
        print(f"  {nm:>10}  {a:9.5f}  {b:9.5f}  {c:9.5f}")
    if not correct:
        print()
        print("  Note: line endpoints carry zero noise by construction, so this")
        print("        average is a lower bound on the true input noise.")
    if per_line:
        print()
        print("Per-line mean noise:")
        print(f"  {'line':>6}  {'n':>3}  mean||sent-real||")
        for key, n, m in stats["per_line"]:
            print(f"  {key:>6}  {n:>3}  {m:.5f}")


def _save_histogram(mag: np.ndarray, out: Path, mean: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(mag, bins=40, color="#4878CF", edgecolor="black", linewidth=0.5)
    ax.axvline(mean, color="crimson", linestyle="--", linewidth=1.5,
               label=f"mean = {mean:.4f}")
    ax.set_xlabel("input noise  ||sent - real||")
    ax.set_ylabel("samples")
    ax.set_title("Per-sample input noise")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved histogram -> {out}")


# ── ternary plot with lengthscale reference ────────────────────────────────────

_SQRT3_2 = np.sqrt(3) / 2

def _random_simplex_points(n: int = 100, seed: int = 42) -> np.ndarray:
    """Sample n uniform points from the 3-simplex (Dirichlet(1,1,1))."""
    rng = np.random.default_rng(seed)
    raw = rng.exponential(1.0, size=(n, 3))
    return raw / raw.sum(axis=1, keepdims=True)


def _read_run_input_noise(run_dir: Path) -> float | None:
    """Try to read input_noise from a run's config.json; return None if absent."""
    import json
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
        v = cfg.get("input_noise")
        if v is not None:
            return float(v)
        v_ilr = cfg.get("input_noise_ilr")
        if v_ilr is not None:
            return float(v_ilr) / 3.0
    except Exception:
        pass
    return None


def _comp_to_xy(comp: np.ndarray) -> np.ndarray:
    """(N, 3) simplex compositions → (N, 2) Cartesian ternary coords.

    Corner map: comp[:,0]→(0,0) bottom-left, comp[:,1]→(1,0) bottom-right,
    comp[:,2]→(0.5, √3/2) top.
    """
    p = np.asarray(comp, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def show_ternary_lengthscale(
    lines: list["Line"],
    input_noise: float,
    source: str,
    labels: tuple[str, str, str] = ("FAPbI3", "MAPbI3", "MAPbBr3"),
    ensemble_seed: int = 0,
    grid_n: int = 300,
) -> None:
    """Show an interactive ternary plot with a GP lengthscale reference circle.

    The background is a random ``Ensemble`` landscape (dim=3) so there are
    real features to compare the lengthscale against.  Real data points from
    ``lines`` are overlaid; if ``lines`` is empty, 80 random simplex points
    are scattered instead.

    The circle radius equals ``input_noise / sqrt(2)`` in 2D Cartesian ternary
    coordinates, which corresponds to a ball of radius ``input_noise`` in
    3D composition L2 space (the ternary map is an isometry scaled by sqrt(2)).
    """
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    from synthetic_data.ensemble import Ensemble, random_ensemble_config
    from visualization.plot_run import ternary_grid

    # ── ensemble background ─────────────────────────────────────────────────
    cfg = random_ensemble_config(dim=3, index=0, seed=ensemble_seed)
    fn = Ensemble(**cfg)

    grid_pts = ternary_grid(grid_n)
    grid_vals = fn.predict(grid_pts)
    grid_xy = _comp_to_xy(grid_pts)

    # ── overlay points (real data only; nothing shown in demo mode) ─────────
    real_parts = [ln.real for ln in lines if ln.real.shape[0] > 0]
    pts = np.concatenate(real_parts, axis=0) if real_parts else None
    if pts is not None:
        pt_vals = fn.predict(pts)
        pts_xy = _comp_to_xy(pts)

    # ── figure ──────────────────────────────────────────────────────────────
    vmin, vmax = float(grid_vals.min()), float(grid_vals.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(8.4, 7.6))
    ax.set_aspect("equal")
    ax.axis("off")

    # Background heatmap
    ax.scatter(grid_xy[:, 0], grid_xy[:, 1], c=grid_vals, cmap=cmap,
               norm=norm, s=6, alpha=0.85, zorder=1, linewidths=0, rasterized=True)

    # Triangle outline
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], color="black", lw=1.5, zorder=2)

    # Corner labels
    pad = 0.06
    ax.text(-pad, -pad * 0.5, labels[0], ha="right", va="top", fontsize=11)
    ax.text(1 + pad, -pad * 0.5, labels[1], ha="left", va="top", fontsize=11)
    ax.text(0.5, _SQRT3_2 + pad, labels[2], ha="center", va="bottom", fontsize=11)

    # Data points (only if real data was loaded)
    if pts is not None:
        ax.scatter(pts_xy[:, 0], pts_xy[:, 1], c=pt_vals, cmap=cmap, norm=norm,
                   s=40, alpha=0.95, edgecolors="black", linewidths=0.8, zorder=3,
                   label=f"{source} ({pts.shape[0]} pts)")

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Ensemble objective", fraction=0.035, pad=0.02)

    # ── GP min lengthscale reference circle ─────────────────────────────────
    # In composition L2:  ||Δcomp|| = sqrt(2) * ||Δxy||_Cartesian
    # so a ball of radius input_noise in composition space → radius / sqrt(2) in Cartesian.
    r_cart = input_noise / np.sqrt(2)

    # Place near the lower-left interior so it doesn't clip the triangle edge.
    cx, cy = 0.18, _SQRT3_2 * 0.28

    circle = plt.Circle((cx, cy), r_cart, fill=False, color="crimson",
                         linewidth=2.2, linestyle="-", zorder=5)
    ax.add_patch(circle)

    ax.annotate(
        "",
        xy=(cx - r_cart, cy), xytext=(cx + r_cart, cy),
        arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.5),
        zorder=6,
    )
    ax.text(
        cx, cy - r_cart - 0.025,
        f"GP min ℓ = {input_noise:.4f}  (comp. L₂)\n"
        f"Cartesian r = {r_cart:.4f}",
        ha="center", va="top", fontsize=8.5, color="crimson", zorder=6,
    )

    if pts is not None:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_title(
        f"Ensemble landscape (seed={ensemble_seed}) + GP min lengthscale\n{source}",
        fontsize=11, pad=10,
    )
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-0.22, _SQRT3_2 + 0.22)
    fig.tight_layout()
    plt.show()


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average input noise (sent vs. real composition) of a ZoMBI-Hop run."
    )
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--db", help="Data file (.db results table or .csv campaign).")
    src.add_argument("--run", help="Run directory or bare run name under runs/.")
    src.add_argument("--comp-log", dest="comp_log",
                     help="A composition_log.jsonl, or a run directory / run name "
                          "containing one. This is the authoritative source: it "
                          "records the sent composition directly.")
    parser.add_argument("--rail", default=None,
                        help="Restrict a composition log to one rail (e.g. main); "
                             "default: all rails.")
    parser.add_argument("--snapshot", default=None,
                        help="Snapshot to reconstruct up to for --run (default: latest.txt).")
    parser.add_argument("--force-legacy", action="store_true",
                        help="Always reconstruct sent compositions from line endpoints.")
    parser.add_argument("--per-line", action="store_true",
                        help="Also print each line's mean noise.")
    parser.add_argument("--plot", default=None,
                        help="Save a histogram of per-sample noise to this PNG path.")
    parser.add_argument("--ternary", action="store_true",
                        help="Show an interactive ternary plot with an Ensemble "
                             "landscape background and the GP minimum lengthscale "
                             "drawn as a reference circle. If no data source is "
                             "given, random demo points are generated.")
    parser.add_argument("--input-noise", type=float, default=None, dest="input_noise",
                        help="Override the input noise / GP min lengthscale shown on "
                             "the ternary (composition L2; default: read from run "
                             f"config.json or {_DEFAULT_INPUT_NOISE}).")
    parser.add_argument("--ensemble-seed", type=int, default=0, dest="ensemble_seed",
                        help="Seed for the random Ensemble landscape background "
                             "(default: 0; try different values to change the landscape).")
    parser.add_argument("--grid-n", type=int, default=300, dest="grid_n",
                        help="Ternary grid resolution for the Ensemble background "
                             "(default: 300).")
    args = parser.parse_args()

    # Require a data source unless --ternary is used in demo mode.
    if not args.db and not args.run and not args.comp_log and not args.ternary:
        parser.error("one of --db, --run or --comp-log is required "
                     "(or use --ternary alone for a demo)")

    lines: list[Line] = []
    correct: bool = False
    source: str = "demo"
    run_dir: Path | None = None

    if args.db:
        db_path = _resolve_db_path(args.db)
        lines, correct = load_db_lines(db_path, args.force_legacy)
        source = db_path.name
    elif args.run:
        run_dir = _resolve_run_dir(args.run)
        lines, correct = load_run_lines(run_dir, args.snapshot, args.force_legacy, args.rail)
        source = run_dir.name
    elif args.comp_log:
        log_path = resolve_comp_log(args.comp_log)
        run_dir = log_path.parent
        lines, correct = load_comp_log_lines(log_path, args.force_legacy, args.rail)
        source = f"{run_dir.name}/{log_path.name}"

    if lines:
        stats = compute_noise(lines)
        _print_report(stats, correct, source, args.per_line)

        if args.plot:
            _save_histogram(stats["magnitudes"], Path(args.plot), stats["mean_magnitude"])

    if args.ternary:
        # Resolve GP min lengthscale: CLI override → run config.json → default.
        ls_val = args.input_noise
        if ls_val is None and run_dir is not None:
            ls_val = _read_run_input_noise(run_dir)
        if ls_val is None:
            ls_val = _DEFAULT_INPUT_NOISE
        show_ternary_lengthscale(
            lines, ls_val, source,
            ensemble_seed=args.ensemble_seed,
            grid_n=args.grid_n,
        )


if __name__ == "__main__":
    main()
