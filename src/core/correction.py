"""
Objective-correction replay for ZoMBI-Hop
=========================================

Re-ingests a *corrected* hardware results database into an existing run: the
same compositions were measured (activations, zoom structure and sample order
are unchanged), but some objective scores changed after the raw data was
re-analysed. This module supplies the pure data operations —

  * parse the corrected results DB into a composition→objective map,
  * match each already-stored measured point to its corrected objective,
  * rewrite the persisted ``Y`` values inside the snapshot deltas, and
  * clone a run directory so corrections never touch the original record —

while the needle re-derivation and re-snapshot live in
``ZoMBIHop.redeclare_needles_after_correction`` (which needs the optimizer's GP
machinery). ``scripts/apply_correction.py`` and the GUI orchestrate the two.

Matching is by composition, not row order: the corrected DB and the run store
the *same* compositions bit-for-bit (both come from the one hardware run), so a
rounded-composition key aligns them robustly even though the DB carries extra
rows the optimizer never ingested (initial references, cache-rail droplets,
assay failures whose ``Objective`` is NULL). Points with no corrected score
(unmatched, or a now-NULL objective) keep their original ``Y`` and are reported.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# The optimizer composition space is the FIRST 10 columns after ``Iteration``
# in a DiSCO ``results`` table (FAPbI3 … MAPbBr3); the trailing ``X`` column and
# the stage/assay columns are not part of it. ``optimizing_dims`` index into
# these 10. See scripts/initialize_databases.py (10-wide col_0..col_9) and the
# active-dim reduction in scripts/run_zombi_main.get_y_measurements.
_N_OPT_COLUMNS = 10


@dataclass
class CorrectionMap:
    """A corrected results DB reduced to the run's optimizing dims."""
    by_comp: Dict[Tuple[float, ...], Optional[float]]  # rounded comp → objective (None = NULL)
    comp_array: np.ndarray                              # (M, d) active-dim compositions (valid-objective rows)
    obj_array: np.ndarray                               # (M,) objectives aligned with comp_array
    optimizing_dims: List[int]
    active_columns: List[str]
    round_decimals: int
    n_rows: int = 0            # total rows in the DB
    n_valid: int = 0           # rows with a finite Objective
    n_null: int = 0            # rows with NULL/non-finite Objective
    n_conflicting_keys: int = 0  # comp keys seen with >1 distinct finite objective


@dataclass
class MatchReport:
    """Outcome of aligning a run's stored points to a CorrectionMap."""
    n_points: int = 0
    n_matched: int = 0          # got a corrected finite objective
    n_changed: int = 0          # matched AND objective differs beyond tol
    n_unchanged: int = 0        # matched but objective identical
    n_unmatched: int = 0        # no composition in the DB (old Y kept)
    n_null_objective: int = 0   # matched a row whose corrected objective is NULL (old Y kept)
    n_nn_recovered: int = 0     # matched via nearest-neighbour fallback
    max_abs_delta: float = 0.0
    mean_abs_delta: float = 0.0
    changes: List[dict] = field(default_factory=list)  # per-changed-point detail
    unmatched_examples: List[dict] = field(default_factory=list)

    @property
    def match_fraction(self) -> float:
        return self.n_matched / self.n_points if self.n_points else 0.0

    def summary(self) -> str:
        lines = [
            f"points in run:        {self.n_points}",
            f"matched (corrected):  {self.n_matched}  ({100 * self.match_fraction:.1f}%)"
            + (f"  [{self.n_nn_recovered} via nearest-neighbour]" if self.n_nn_recovered else ""),
            f"  objective changed:  {self.n_changed}",
            f"  objective same:     {self.n_unchanged}",
            f"unmatched (kept old): {self.n_unmatched}",
            f"null objective (kept):{self.n_null_objective}",
        ]
        if self.n_changed:
            lines.append(f"objective delta:      max |Δ|={self.max_abs_delta:.4f}  "
                         f"mean |Δ|={self.mean_abs_delta:.4f}")
        return "\n".join(lines)


def _comp_key(row: np.ndarray, round_decimals: int) -> Tuple[float, ...]:
    # 0.0 + rounding collapses "-0.0" and any tiny signed noise to one key.
    return tuple(float(round(float(v), round_decimals)) + 0.0 for v in row)


def load_corrected_objectives(
    db_path: Union[str, Path],
    optimizing_dims: List[int],
    *,
    table: str = "results",
    objective_column: str = "Objective",
    round_decimals: int = 6,
    simplex_sum_tol: float = 5e-2,
) -> CorrectionMap:
    """Parse a corrected DiSCO results DB into a CorrectionMap.

    ``optimizing_dims`` index into the run's 10-column composition space (the 10
    columns after ``Iteration``). Rows with a NULL/non-finite objective are
    recorded (``by_comp`` value ``None``) but kept out of the nearest-neighbour
    arrays. Raises ``ValueError`` if the schema does not look like a results DB
    (missing objective column, too few composition columns, or compositions that
    do not sum to ≈1 over the 10-column space — a sign the column layout is not
    what we assume).
    """
    optimizing_dims = [int(d) for d in optimizing_dims]
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"corrected results DB not found: {db_path}")

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        cols = [r[1] for r in cur.execute(f'PRAGMA table_info("{table}")').fetchall()]
        if not cols:
            raise ValueError(f'table "{table}" not found or empty in {db_path.name}')
        if objective_column not in cols:
            raise ValueError(
                f'no "{objective_column}" column in {db_path.name} (have: {cols[:6]}…)')
        # 10 composition columns = the 10 immediately after the first column
        # (Iteration). Guard the layout assumption.
        if len(cols) < 1 + _N_OPT_COLUMNS + 1:
            raise ValueError(
                f"{db_path.name} has too few columns ({len(cols)}) to be a results DB")
        comp_columns = cols[1:1 + _N_OPT_COLUMNS]
        max_dim = max(optimizing_dims) if optimizing_dims else -1
        if max_dim >= _N_OPT_COLUMNS:
            raise ValueError(
                f"optimizing_dims {optimizing_dims} exceed the {_N_OPT_COLUMNS}-column "
                f"composition space")
        active_columns = [comp_columns[i] for i in optimizing_dims]

        sel = ", ".join(f'"{c}"' for c in comp_columns + [objective_column])
        rows = cur.execute(f'SELECT {sel} FROM "{table}"').fetchall()
    finally:
        con.close()

    by_comp: Dict[Tuple[float, ...], Optional[float]] = {}
    conflict_keys: Dict[Tuple[float, ...], float] = {}
    n_conflict = 0
    comp_list: List[List[float]] = []
    obj_list: List[float] = []
    n_valid = n_null = 0
    checked_simplex = 0
    bad_simplex = 0

    for r in rows:
        comp10 = r[:_N_OPT_COLUMNS]
        obj = r[_N_OPT_COLUMNS]
        # Simplex sanity on the first handful of finite rows: the 10-col
        # composition must sum to ≈1, else our column slice is wrong.
        if obj is not None and checked_simplex < 25:
            try:
                s = sum(float(v) for v in comp10)
                checked_simplex += 1
                if abs(s - 1.0) > simplex_sum_tol:
                    bad_simplex += 1
            except (TypeError, ValueError):
                pass
        active = [comp10[i] for i in optimizing_dims]
        try:
            active_f = [float(v) for v in active]
        except (TypeError, ValueError):
            continue  # unparseable composition row
        key = _comp_key(np.asarray(active_f), round_decimals)
        if obj is None or not np.isfinite(obj):
            n_null += 1
            by_comp.setdefault(key, None)  # do not overwrite a finite value
            continue
        obj_f = float(obj)
        n_valid += 1
        if key in conflict_keys and abs(conflict_keys[key] - obj_f) > 10 ** (-round_decimals):
            n_conflict += 1
        conflict_keys[key] = obj_f
        by_comp[key] = obj_f  # last finite value wins
        comp_list.append(active_f)
        obj_list.append(obj_f)

    if checked_simplex and bad_simplex > checked_simplex // 2:
        raise ValueError(
            f"{db_path.name}: the 10 assumed composition columns "
            f"({comp_columns}) do not sum to ~1 (checked {checked_simplex} rows, "
            f"{bad_simplex} off) — the column layout is not what apply_correction "
            f"expects; verify this is a DiSCO results database")

    comp_array = (np.asarray(comp_list, dtype=float)
                  if comp_list else np.empty((0, len(optimizing_dims)), dtype=float))
    obj_array = np.asarray(obj_list, dtype=float) if obj_list else np.empty((0,), dtype=float)
    return CorrectionMap(
        by_comp=by_comp, comp_array=comp_array, obj_array=obj_array,
        optimizing_dims=optimizing_dims, active_columns=active_columns,
        round_decimals=round_decimals, n_rows=len(rows), n_valid=n_valid,
        n_null=n_null, n_conflicting_keys=n_conflict,
    )


def match_corrected_Y(
    X_all_actual: np.ndarray,
    Y_all: np.ndarray,
    corr_map: CorrectionMap,
    *,
    change_tol: float = 1e-6,
    nn_tol: float = 1e-4,
) -> Tuple[np.ndarray, MatchReport]:
    """Return a corrected ``Y`` vector aligned to ``X_all_actual`` plus a report.

    Each stored point is looked up by rounded composition. Points with no
    corrected finite objective (composition absent, or a NULL objective) keep
    their original ``Y``. Points that miss the exact-key lookup get one
    nearest-neighbour attempt within ``nn_tol`` (L∞) against the DB's
    finite-objective compositions — this rescues tiny float drift without
    inventing matches.
    """
    X = np.asarray(X_all_actual, dtype=float)
    Y = np.asarray(Y_all, dtype=float).reshape(-1)
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError(f"X ({n}) and Y ({Y.shape[0]}) row counts differ")

    corrected = Y.copy()
    rep = MatchReport(n_points=n)
    deltas: List[float] = []
    rd = corr_map.round_decimals
    have_nn = corr_map.comp_array.shape[0] > 0

    for i in range(n):
        key = _comp_key(X[i], rd)
        obj = corr_map.by_comp.get(key, "MISS")
        recovered_nn = False
        if obj == "MISS" and have_nn:
            d = np.max(np.abs(corr_map.comp_array - X[i][None, :]), axis=1)
            j = int(np.argmin(d))
            if d[j] <= nn_tol:
                obj = float(corr_map.obj_array[j])
                recovered_nn = True
        if obj == "MISS":
            rep.n_unmatched += 1
            if len(rep.unmatched_examples) < 8:
                rep.unmatched_examples.append(
                    {"index": i, "comp": [round(float(v), 4) for v in X[i]],
                     "old_y": float(Y[i])})
            continue
        if obj is None:
            rep.n_null_objective += 1
            continue
        rep.n_matched += 1
        if recovered_nn:
            rep.n_nn_recovered += 1
        new_y = float(obj)
        old_y = float(Y[i])
        if abs(new_y - old_y) > change_tol:
            corrected[i] = new_y
            rep.n_changed += 1
            deltas.append(abs(new_y - old_y))
            if len(rep.changes) < 5000:
                rep.changes.append(
                    {"index": i, "old_y": old_y, "new_y": new_y, "delta": new_y - old_y})
        else:
            rep.n_unchanged += 1

    if deltas:
        rep.max_abs_delta = float(max(deltas))
        rep.mean_abs_delta = float(sum(deltas) / len(deltas))
    return corrected, rep


def rewrite_snapshot_Y(run_dir: Union[str, Path], Y_corrected: np.ndarray) -> int:
    """Rewrite the ``Y_new`` slices across a run's snapshot deltas in place.

    ``Y`` is partitioned across ``delta.pt`` files (each measured point's value
    lives in exactly one delta's ``Y_new``); replaying them in sorted snapshot
    order concatenates back to the full ``Y_all``. We walk the deltas in that
    same order, tracking a running row offset, and overwrite each non-empty
    ``Y_new`` with the matching slice of ``Y_corrected`` (dtype/shape/device
    preserved). Returns the number of deltas rewritten.

    Raises ``ValueError`` if the deltas' cumulative ``Y_new`` count does not
    equal ``len(Y_corrected)`` — a guard against a mismatched vector silently
    corrupting history.
    """
    run_dir = Path(run_dir)
    snap_dir = run_dir / "snapshots"
    Yc = np.asarray(Y_corrected, dtype=float).reshape(-1)
    names = sorted(s.name for s in snap_dir.iterdir() if s.is_dir()) if snap_dir.exists() else []

    # First pass: validate total count before touching any file.
    total = 0
    deltas: List[Tuple[Path, dict, int]] = []  # (path, dict, y_new_len)
    for name in names:
        p = snap_dir / name / "delta.pt"
        if not p.exists():
            continue
        d = torch.load(str(p), map_location="cpu", weights_only=False)
        yn = d.get("Y_new")
        n_new = int(yn.shape[0]) if isinstance(yn, torch.Tensor) else 0
        if n_new > 0:
            deltas.append((p, d, n_new))
            total += n_new
    if total != Yc.shape[0]:
        raise ValueError(
            f"corrected Y length ({Yc.shape[0]}) does not match the "
            f"snapshots' cumulative measured-point count ({total}); refusing to "
            f"rewrite to avoid corrupting history")

    # Second pass: overwrite Y_new slices and persist.
    offset = 0
    n_written = 0
    for p, d, n_new in deltas:
        yn: torch.Tensor = d["Y_new"]
        seg = Yc[offset:offset + n_new]
        new_yn = torch.as_tensor(seg, dtype=yn.dtype, device=yn.device).reshape(yn.shape)
        d["Y_new"] = new_yn
        torch.save(d, str(p))
        offset += n_new
        n_written += 1
    return n_written


def clone_run(src_run_dir: Union[str, Path], dst_run_dir: Union[str, Path],
              new_uuid: str) -> Path:
    """Copy an entire run directory to ``dst_run_dir`` and stamp the new UUID.

    Copies every file (snapshots, config.json, hw_config.json,
    convergence_history.jsonl, composition_log.jsonl, hparams_effective.json,
    run.log, latest.txt, …) so the corrected copy is a self-contained run, then
    updates ``config.json``'s ``run_uuid`` to ``new_uuid``. The original run is
    left untouched. Raises if the destination already exists.
    """
    src = Path(src_run_dir)
    dst = Path(dst_run_dir)
    if not src.exists():
        raise FileNotFoundError(f"source run directory not found: {src}")
    if dst.exists():
        raise FileExistsError(f"destination run directory already exists: {dst}")
    shutil.copytree(src, dst)

    cfg_path = dst / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
            cfg["run_uuid"] = new_uuid
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        except Exception:
            pass  # config is metadata; a parse hiccup must not abort the clone
    return dst


def read_optimizing_dims(run_dir: Union[str, Path]) -> Optional[List[int]]:
    """Optimizing-dim indices for a run, from hw_config.json then config.json.

    Mirrors the resume path (``run_zombi_main`` / ``main.py``): ``hw_config.json``
    ``"dims"`` (e.g. ``"0,2,3,4,8,9"``) wins, else ``config.json`` ``"dims"``.
    Returns ``None`` if neither is present.
    """
    run_dir = Path(run_dir)
    for fname in ("hw_config.json", "config.json"):
        p = run_dir / fname
        if not p.exists():
            continue
        try:
            cfg = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        dims = cfg.get("dims")
        if isinstance(dims, str) and dims.strip():
            try:
                return [int(x) for x in dims.split(",")]
            except ValueError:
                continue
        if isinstance(dims, (list, tuple)) and dims:
            try:
                return [int(x) for x in dims]
            except (ValueError, TypeError):
                continue
    return None
