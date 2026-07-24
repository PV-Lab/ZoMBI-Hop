"""
llm/sweep_convergence.py
========================
A four-group convergence sweep on **re-randomized 10-simplex Ensemble
landscapes** (``synthetic_data/ensemble.py``), combining the two convergence
experiments this repo already has:

    * ``llm/sweep_catastrophic.py`` — baseline vs a catastrophically PERTURBED
      hyperparameter set, plus an LLM group that starts from the perturbed set
      and injects a hyperparameter edit every ``INJECT_INTERVAL`` iterations
      (checkpoint/resume machinery, offline-history prompt, per-trial artifacts).
    * ``optimize/compare_arb_tuned.py`` — a re-randomized-per-seed Ensemble
      landscape and an ARBITRARY hyperparameter set scored against a tuned one,
      with the running-best curves normalized per landscape so different draws
      are commensurable.

The four groups, all scored on the SAME landscape within a rep (and with common
random numbers, so trajectories diverge only through the hyperparameters / the
LLM's edits):

    1. baseline    — the tuned 10-simplex hyperparameters from
                     ``optimize/hparams/10d_ensemble.json``.
    2. perturbed   — those baseline hyperparameters with ``PERTURB`` applied
                     (the same catastrophic edit as ``sweep_catastrophic.py``).
    3. arbitrary   — the arbitrary hyperparameters from
                     ``optimize/hparams/3d_llm_chosen.json``.
    4. llm_every_K — START from the perturbed hyperparameters, then let the LLM
                     inject a hyperparameter change every ``INJECT_INTERVAL``
                     iterations, exactly as ``sweep_catastrophic.py``'s LLM
                     group (offline history reused from a 4D ensemble MOBO run —
                     ZoMBI-Hop hyperparameters are dimension-independent, so the
                     Pareto history is a valid prior at any simplex dimension).

Re-randomized landscape per rep
-------------------------------
Unlike ``sweep_catastrophic.py`` (one FROZEN landscape for the whole sweep),
each rep ``r`` draws a DIFFERENT ``Ensemble`` landscape via
``random_ensemble_config(dim, index=r, seed=ENSEMBLE_SEED)`` — the
``compare_arb_tuned`` "different landscape per seed" design. All four groups run
on that one landscape within the rep (paired), and the next rep is a fresh draw.
Because the landscapes differ, the headline convergence plot normalizes each
rep's running-best to that landscape's own ``[y_floor, y_opt]`` (estimated from a
dense Dirichlet sample + the true optima), so ``1.0`` == the landscape's global
optimum and the reps average sensibly.

Artifacts
---------
We reuse ``sweep_catastrophic.finalize_trial``, which at ``dim == 10`` writes the
dimension-general per-trial artifacts (the ternary coverage / needles-on-landscape
plots are 3D/4D-only and simply skipped — impossible in 10D):

  <group>/rep<r>/needles.csv               declared needles (run_mobo schema)
  <group>/rep<r>/points.csv                every sampled point
  <group>/rep<r>/metrics_over_time.csv     per-iteration metric trace
  <group>/rep<r>/convergence.png           run_mobo single-envelope convergence
  <group>/rep<r>/convergence_segments.png  running-best restarting per activation
  <group>/rep<r>/metrics.json              every number behind the plots, including
                                           the three per-run metrics run_mobo reports:
                                           dist_to_needles, dup_fraction, avg_time_per_iter_s
  (llm group also: injections.json + injections/inj_*/ prompt+decision)

Sweep-level:
  convergence_comparison.png   the headline: all four groups on ONE convergence
                               figure, each a different color with a shaded 95% CI
                               band (per-landscape normalized, averaged over reps),
                               plus a uniform-random baseline (mean + 5–95% band)
                               drawn on the same landscapes, à la compare_arb_tuned.
  slopegraph_final_best.png    one line per rep connecting its four normalized
                               final-bests (paired across groups), plus the mean.
  sweep_summary.json           per-group aggregate metrics.

Usage
-----
  # repo-root uv venv (see MEMORY.md), NOT `conda activate zombi-hop`
  python llm/sweep_convergence.py             # full 4-group sweep
  python llm/sweep_convergence.py --no-llm    # baseline + perturbed + arbitrary
  python llm/sweep_convergence.py --resume    # continue newest sweep dir, skipping
                                              # reps that already have metrics.json
  python llm/sweep_convergence.py --replot [DIR]  # regenerate the sweep plots only
"""
from __future__ import annotations

# ═══ HARDCODED CONFIG ════════════════════════════════════════════════════════
# Baseline (tuned) 10-simplex hyperparameters. Repo-relative paths resolve
# against the repo root. Any JSON evaluate accepts — a flat hparam dict or a
# trial.json with an "hparams" key.
BASELINE_HPARAMS: str = "optimize/hparams/10d_ensemble.json"

# Arbitrary hyperparameters (the third group).
ARB_HPARAMS: str = "optimize/hparams/3d_llm_chosen.json"

# The catastrophic perturbation applied to the baseline for BOTH the "perturbed"
# group and the LLM group's starting point (identical to sweep_catastrophic.py's
# "extra exploitative" PERTURB). Every name must be a valid ZoMBI-Hop hyperparameter.
PERTURB: dict = {
    "ucb_beta": 0.01,
    "max_zooms": 10,
    "n_consecutive_converged": 5,
    "max_penalty_radius": 0.1,
}

DIM: int = 10                # simplex dimensionality (Ensemble landscape)
ENSEMBLE_SEED: int = 1       # base seed; rep r draws landscape index=r from it
ENSEMBLE_OPTIMA_MARGIN: float = 0.2  # optima/background gap (run_mobo default)

MAX_ITERS: int = 218         # ZoMBI-Hop objective iterations (== LineBO lines) per
                             # trial; +N_INIT_LINES (2) init lines → 220 lines total,
                             # × NUM_EXPERIMENTS (24 pts/line) = 5280 sampled points
N_REPEATS: int = 3           # reps per group; each rep is a fresh landscape draw
INJECT_INTERVAL: int = 10    # LLM injects every k iterations (llm group only)

# Multiplicative output/objective noise seen by the algorithm at every sampled
# point (see sweep_catastrophic.py). None leaves run_mobo.OUTPUT_NOISE_FRAC alone.
# 0.045 (~4.5%) is run_mobo's real-hardware default.
OUTPUT_NOISE_FRAC: "float | None" = 0.045

RESULTS_ROOT: str = "llm/results"   # sweep dir created under here (repo-root relative)

# Uniform-random baseline (à la optimize/compare_arb_tuned.py): each landscape's
# value distribution is sampled densely; RANDOM_BASELINE_SAMPLE of those values are
# stored with each rep so N_RANDOM_SEARCHES independent uniform searches can be
# replayed at plot time (also under --replot) for the mean + 5–95% baseline band.
RANDOM_BASELINE_SAMPLE: int = 5000
N_RANDOM_SEARCHES: int = 500
# ═════════════════════════════════════════════════════════════════════════════

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# sweep_catastrophic wires up sys.path (optimize/ + repo root) via evaluate_llm and
# holds all the ZoMBI/LLM plumbing we reuse: run_plain_trial, run_llm_trial,
# finalize_trial, the injection prompt, and the offline-history block. Importing it
# has no side effects beyond those imports (its main() only runs under __main__).
import sweep_catastrophic as SC  # noqa: E402

R = SC.R                                   # run_mobo module
E = SC.E                                   # evaluate_llm module
Ensemble = SC.Ensemble
random_ensemble_config = SC.random_ensemble_config
composition_column_names = SC.composition_column_names
HPARAM_NAMES = R.HPARAM_NAMES
_REPO_ROOT = SC._REPO_ROOT

# The LLM group's offline-history prompt is drawn from this MOBO run's Pareto front.
# sweep_catastrophic.TRIAL_JSON already points at the 4D ensemble run; ZoMBI-Hop
# hyperparameters are dimension-independent, so its history is a valid prior for the
# 10-simplex runs here. (We keep SC.TRIAL_JSON as-is for exactly this reason.)

# One distinct colour per group; the LLM group name carries INJECT_INTERVAL.
_GROUP_COLORS: Dict[str, str] = {
    "baseline": "steelblue",
    "perturbed": "firebrick",
    "arbitrary": "darkorange",
}
_LLM_COLOR = "mediumorchid"
_RANDOM_COLOR = "slategray"


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (_REPO_ROOT / q)


def _group_color(group: str) -> str:
    return _LLM_COLOR if group.startswith("llm") else _GROUP_COLORS.get(group, "gray")


def _group_label(group: str) -> str:
    if group.startswith("llm_every_"):
        return f"LLM (inject every {group.rsplit('_', 1)[-1]} iters)"
    return group.capitalize()


# ═══════════════════════════════════════════════════════════════════════════
# Hyperparameter + landscape helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_hparams(path: str, label: str) -> Dict[str, Any]:
    """Load a ZoMBI-Hop hparam dict from a flat JSON or a trial.json ("hparams" key)."""
    obj = json.loads(_resolve(path).read_text())
    hp_raw = obj.get("hparams", obj) if isinstance(obj, dict) else {}
    hp = {k: hp_raw[k] for k in HPARAM_NAMES if k in hp_raw}
    missing = [k for k in HPARAM_NAMES if k not in hp]
    if missing:
        raise SystemExit(f"[{label}] {path} missing hyperparameters: {missing}")
    print(f"  [{label}] {len(hp)} hyperparameters from {_resolve(path)}")
    return hp


def build_ensemble_for_rep(dim: int, rep: int) -> Tuple[Ensemble, List[np.ndarray]]:
    """The rep's landscape: a fresh Ensemble draw (index=rep off ENSEMBLE_SEED)."""
    cfg = random_ensemble_config(dim, index=rep, total=N_REPEATS, seed=ENSEMBLE_SEED,
                                 optima_margin=ENSEMBLE_OPTIMA_MARGIN)
    ens = Ensemble(**cfg)
    true_optima = [np.asarray(c, float) for c in ens.centers]
    print(f"  [rep {rep}] {ens!r}")
    print(f"  [rep {rep}] {len(true_optima)} true optima "
          f"(from {len(ens.peak_centers)} placed basins)")
    return ens, true_optima


def landscape_range(ens: Ensemble, dim: int, seed: int,
                    n_samples: int = 50_000) -> Tuple[float, float, np.ndarray]:
    """Estimate ``(y_floor, y_opt, sample_vals)`` for a landscape from a dense
    Dirichlet sample plus the true optima, so a rep's running-best can be normalized
    to ``(y - y_floor) / (y_opt - y_floor)`` (1.0 == the landscape's global optimum).
    ``sample_vals`` is the raw Dirichlet-sampled objective distribution, from which
    the uniform-random baseline draws (uniform search over the simplex). Every group
    in a rep shares all three, so the groups stay directly comparable."""
    rng = np.random.default_rng(seed)
    samples = rng.dirichlet(np.ones(dim), size=n_samples)
    vals = np.asarray(ens.predict(samples), float)
    y_floor = float(vals.min())
    y_opt = float(vals.max())
    if len(ens.centers):
        y_opt = max(y_opt, float(np.asarray(ens.predict(np.asarray(ens.centers))).max()))
    return y_floor, y_opt, vals


def _subsample(vals: np.ndarray, n: int, seed: int) -> np.ndarray:
    """A deterministic size-``n`` subsample of ``vals`` (all of it if already small)."""
    vals = np.asarray(vals, float)
    if vals.size <= n:
        return vals
    rng = np.random.default_rng(seed)
    return vals[rng.choice(vals.size, size=n, replace=False)]


# ═══════════════════════════════════════════════════════════════════════════
# Sweep-level plots (per-landscape normalized so reps are commensurable)
# ═══════════════════════════════════════════════════════════════════════════

def _norm_curve(rb: np.ndarray, y_floor: float, y_opt: float) -> np.ndarray:
    span = (y_opt - y_floor) or 1.0
    return (np.asarray(rb, float) - y_floor) / span


def _ci95(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and 95% CI (mean ± 1.96·SEM) down axis 0 (reps)."""
    mean = stack.mean(axis=0)
    n = stack.shape[0]
    sem = stack.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    half = 1.96 * sem
    return mean, mean - half, mean + half


def random_running_best(sample_vals: np.ndarray, n: int, *, maximize: bool,
                        rng: np.random.Generator, n_searches: int) -> np.ndarray:
    """(n_searches, n) running-best curves from uniform draws over ``sample_vals``.

    Mirrors optimize/compare_arb_tuned.random_running_best: each search draws ``n``
    values uniformly at random from the landscape's value distribution and tracks
    its running best — the honest "what would random search with the same budget
    get?" line."""
    draws = sample_vals[rng.integers(0, sample_vals.size, size=(n_searches, n))]
    accum = np.maximum.accumulate if maximize else np.minimum.accumulate
    return accum(draws, axis=1)


def _load_landscape_samples(sweep_dir: Path) -> List[Tuple[np.ndarray, float, float]]:
    """One ``(sample_vals, y_floor, y_opt)`` per rep (landscape) — the uniform-random
    baseline inputs — read from whichever group first logged them for that rep."""
    seen: Dict[str, Tuple[np.ndarray, float, float]] = {}
    for gdir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        for rep in sorted(gdir.glob("rep*")):
            if rep.name in seen:
                continue
            mp = rep / "metrics.json"
            if not mp.exists():
                continue
            try:
                m = json.loads(mp.read_text())
            except Exception:
                continue
            sv = m.get("landscape_sample_vals")
            y_floor = m.get("landscape_y_floor")
            y_opt = m.get("landscape_y_opt")
            if sv and y_floor is not None and y_opt is not None:
                seen[rep.name] = (np.asarray(sv, float), float(y_floor), float(y_opt))
    return list(seen.values())


def _random_baseline_stack(sweep_dir: Path, L: int) -> Optional[np.ndarray]:
    """Per-landscape normalized uniform-random running-best curves, stacked over all
    reps × N_RANDOM_SEARCHES searches (rows), truncated to ``L`` — the input to the
    baseline's mean + 5–95% band. None if no rep logged a value sample."""
    reps = _load_landscape_samples(sweep_dir)
    if not reps:
        return None
    rows = []
    for i, (vals, y_floor, y_opt) in enumerate(reps):
        rng = np.random.default_rng(700_000 + i)
        curves = random_running_best(vals, L, maximize=True, rng=rng,
                                     n_searches=N_RANDOM_SEARCHES)
        rows.append(_norm_curve(curves, y_floor, y_opt))
    return np.vstack(rows)


def _load_group_curves(gdir: Path) -> List[np.ndarray]:
    """Normalized per-rep running-best curves from a group dir's rep*/metrics.json."""
    curves: List[np.ndarray] = []
    for rep in sorted(gdir.glob("rep*")):
        mp = rep / "metrics.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        rb = m.get("Y_all_running_best") or []
        y_floor = m.get("landscape_y_floor")
        y_opt = m.get("landscape_y_opt")
        if rb and y_floor is not None and y_opt is not None:
            curves.append(_norm_curve(np.asarray(rb, float), y_floor, y_opt))
    return curves


def _discover_groups(sweep_dir: Path) -> List[str]:
    """Group dirs that hold at least one rep with metrics.json, in plot order."""
    def order(g: str) -> int:
        return {"baseline": 0, "perturbed": 1, "arbitrary": 2}.get(g, 3)
    groups = [d.name for d in sweep_dir.iterdir()
              if d.is_dir() and any((d / f"rep{r}" / "metrics.json").exists()
                                    for r in range(N_REPEATS))]
    return sorted(groups, key=lambda g: (order(g), g))


def plot_group_convergence(sweep_dir: Path,
                           out_png: Optional[Path] = None) -> Optional[Path]:
    """Headline artifact: every group on ONE convergence figure, each a different
    colour with a shaded 95% CI band (mean ± 1.96·SEM), no per-rep lines — à la
    compare_arb_tuned.plot_summary / sweep_catastrophic.plot_group_convergence.
    Curves are per-landscape normalized (fraction of that landscape's optimum) so
    the differently-drawn reps average sensibly. Truncated to the shortest curve."""
    sweep_dir = Path(sweep_dir)
    out_png = Path(out_png) if out_png is not None else sweep_dir / "convergence_comparison.png"
    groups = _discover_groups(sweep_dir)
    if not groups:
        print(f"      [convergence] no groups with metrics.json under {sweep_dir}")
        return None

    group_curves = {g: _load_group_curves(sweep_dir / g) for g in groups}
    group_curves = {g: c for g, c in group_curves.items() if c}
    if not group_curves:
        print(f"      [convergence] no usable running-best curves under {sweep_dir}")
        return None
    L = min(c.size for curves in group_curves.values() for c in curves)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    idx = np.arange(L)
    for group in groups:
        curves = group_curves.get(group)
        if not curves:
            continue
        stack = np.vstack([c[:L] for c in curves])  # (n_reps, L)
        mean, lo, hi = _ci95(stack)
        color = _group_color(group)
        ax.fill_between(idx, lo, hi, color=color, alpha=0.2, lw=0, zorder=2)
        ax.plot(idx, mean, color=color, lw=2.0, zorder=4,
                label=f"{_group_label(group)} (mean ± 95% CI, n={stack.shape[0]})")

    # Uniform-random baseline on the same landscapes (compare_arb_tuned style):
    # mean + 5–95% spread, per-landscape normalized like the group curves.
    rand = _random_baseline_stack(sweep_dir, L)
    if rand is not None and rand.size:
        rlo = np.percentile(rand, 5, axis=0)
        rhi = np.percentile(rand, 95, axis=0)
        ax.fill_between(idx, rlo, rhi, color=_RANDOM_COLOR, alpha=0.15, lw=0, zorder=1,
                        label="Uniform random (5–95%)")
        ax.plot(idx, rand.mean(axis=0), color=_RANDOM_COLOR, lw=1.6, ls="--", zorder=3,
                label=f"Uniform random (mean of {N_RANDOM_SEARCHES})")

    ax.set_xlabel("Objective evaluations (sample index)")
    ax.set_ylabel("Best found (fraction of landscape optimum)")
    ax.set_title(f"Convergence on re-randomized {DIM}-simplex Ensemble landscapes "
                 f"(mean ± 95% CI over {N_REPEATS} reps, per-landscape normalized)",
                 fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")
    return out_png


def plot_slopegraph_final_best(sweep_dir: Path,
                               out_png: Optional[Path] = None) -> Optional[Path]:
    """Slopegraph: one line per rep connecting its four groups' normalized
    final-bests (paired across groups on that rep's landscape), plus the group
    means as a heavy black line — the compare_arb_tuned slopegraph generalized to
    four groups."""
    sweep_dir = Path(sweep_dir)
    out_png = Path(out_png) if out_png is not None else sweep_dir / "slopegraph_final_best.png"
    groups = _discover_groups(sweep_dir)
    if not groups:
        return None

    # finals[rep][group] = normalized final-best on that rep's landscape.
    finals: Dict[int, Dict[str, float]] = {}
    for gi, group in enumerate(groups):
        for rep in sorted((sweep_dir / group).glob("rep*")):
            try:
                m = json.loads((rep / "metrics.json").read_text())
            except Exception:
                continue
            rb = m.get("Y_all_running_best") or []
            y_floor, y_opt = m.get("landscape_y_floor"), m.get("landscape_y_opt")
            if not rb or y_floor is None or y_opt is None:
                continue
            r = int(rep.name[3:])
            finals.setdefault(r, {})[group] = float(_norm_curve(rb, y_floor, y_opt)[-1])
    if not finals:
        return None

    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(sorted(finals)):
        ys = [finals[r].get(g, np.nan) for g in groups]
        ax.plot(x, ys, color=cmap(i % 10), lw=1.5, marker="o", ms=6, alpha=0.9,
                label=f"rep {r}", zorder=3)
    means = [float(np.nanmean([finals[r].get(g, np.nan) for r in finals])) for g in groups]
    ax.plot(x, means, color="black", lw=2.6, marker="s", ms=8, label="mean", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([_group_label(g) for g in groups], fontsize=8)
    ax.set_ylabel("Final best (fraction of landscape optimum)")
    ax.set_title(f"Final best per group  (n={len(finals)} reps, paired per landscape)",
                 fontsize=10)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")
    return out_png


def replot(sweep_dir: Path) -> None:
    plot_group_convergence(sweep_dir)
    plot_slopegraph_final_best(sweep_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════════

def _agg(samples: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    vals = np.array([float(s.get(key, np.nan)) for s in samples], float)
    fin = vals[np.isfinite(vals)]
    return {"mean": float(fin.mean()) if fin.size else float("nan"),
            "std": float(fin.std(ddof=1)) if fin.size > 1 else 0.0,
            "n": int(fin.size)}


def _find_latest_sweep_dir(prefix: str) -> Optional[Path]:
    root = _resolve(RESULTS_ROOT)
    if not root.is_dir():
        return None
    cands = sorted(d for d in root.glob(f"{prefix}_*") if d.is_dir())
    return cands[-1] if cands else None


def main(no_llm: bool = False, resume: bool = False) -> None:
    prefix = "sweep_convergence"
    dim = DIM
    base_hp = load_hparams(BASELINE_HPARAMS, "baseline")
    arb_hp = load_hparams(ARB_HPARAMS, "arbitrary")
    bad_keys = [k for k in PERTURB if k not in HPARAM_NAMES]
    if bad_keys:
        raise SystemExit(f"PERTURB has invalid hyperparameter names: {bad_keys}")
    perturbed_hp = dict(base_hp)
    perturbed_hp.update(PERTURB)
    comp_cols = composition_column_names(dim)
    llm_group = f"llm_every_{INJECT_INTERVAL}"

    # sweep_catastrophic's runners read these module globals at call time; align
    # them with our budget (defaults already match, but be explicit).
    SC.MAX_ITERS = MAX_ITERS
    SC.INJECT_INTERVAL = INJECT_INTERVAL
    SC.N_REPEATS = N_REPEATS
    SC.MAXIMIZE = True
    if OUTPUT_NOISE_FRAC is not None:
        print(f"[convergence] output noise: run_mobo.OUTPUT_NOISE_FRAC "
              f"{R.OUTPUT_NOISE_FRAC} → {OUTPUT_NOISE_FRAC}")
        R.OUTPUT_NOISE_FRAC = OUTPUT_NOISE_FRAC

    print(f"[convergence] dim={dim}  budget={MAX_ITERS}  reps={N_REPEATS}  "
          f"inject_every={INJECT_INTERVAL}")
    print(f"[convergence] PERTURB={PERTURB}")
    print(f"[convergence] offline history ← {_resolve(SC.TRIAL_JSON).parent.parent.name} "
          f"(dimension-independent hparams)")

    sweep_dir = _find_latest_sweep_dir(prefix) if resume else None
    if sweep_dir is not None:
        print(f"[convergence] --resume: continuing {sweep_dir} "
              f"(reps with metrics.json are skipped)")
    else:
        if resume:
            print(f"[convergence] --resume: no existing {prefix}_* dir; fresh sweep")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = _resolve(RESULTS_ROOT) / f"{prefix}_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"[convergence] output → {sweep_dir}  device={R.DEVICE}")

    (sweep_dir / "run_config.json").write_text(json.dumps({
        "dataset": "ensemble", "dim": dim, "maximize": True, "landscape": "synthetic",
        "hparam_names": HPARAM_NAMES, "ensemble_seed": ENSEMBLE_SEED,
        "ensemble_optima_margin": ENSEMBLE_OPTIMA_MARGIN,
        "baseline_hparams": base_hp, "arbitrary_hparams": arb_hp,
        "perturb": PERTURB, "perturbed_hparams": perturbed_hp,
        "max_iters": MAX_ITERS, "n_repeats": N_REPEATS,
        "inject_interval": INJECT_INTERVAL, "output_noise_frac": OUTPUT_NOISE_FRAC,
        "offline_history_trial_json": str(_resolve(SC.TRIAL_JSON)),
    }, indent=2))

    # (group, kind, starting hparams). "plain" = whole budget cold-started, no LLM;
    # "llm" = start from perturbed then inject every INJECT_INTERVAL iters.
    groups: List[Tuple[str, str, Dict[str, Any]]] = [
        ("baseline", "plain", base_hp),
        ("perturbed", "plain", perturbed_hp),
        ("arbitrary", "plain", arb_hp),
        (llm_group, "llm", perturbed_hp),
    ]
    if no_llm:
        groups = [g for g in groups if g[1] != "llm"]
        print("[convergence] --no-llm: baseline + perturbed + arbitrary only")

    per_group_samples: Dict[str, List[Dict[str, Any]]] = {g[0]: [] for g in groups}

    # Reps OUTER (each is its own landscape, built once and shared by all groups),
    # groups INNER. Within a rep every group shares the landscape and seed (paired /
    # common random numbers); different reps are different landscapes.
    for r in range(N_REPEATS):
        print(f"\n===== rep {r} / landscape draw index={r} =====")
        ens, true_optima = build_ensemble_for_rep(dim, r)
        y_floor, y_opt, land_vals = landscape_range(ens, dim, seed=900_000 + r)
        # Compact per-landscape value sample stored with each rep's metrics so the
        # uniform-random baseline can be reconstructed at plot time (and --replot).
        land_sample = [round(float(v), 6)
                       for v in _subsample(land_vals, RANDOM_BASELINE_SAMPLE, 900_000 + r)]
        seed = 1000 + r  # common random numbers across groups within this rep
        print(f"  [rep {r}] landscape range: y_floor={y_floor:.4f}  y_opt={y_opt:.4f}")

        for group, kind, hp in groups:
            rep_dir = sweep_dir / group / f"rep{r}"
            metrics_path = rep_dir / "metrics.json"
            if metrics_path.exists():
                try:
                    m = json.loads(metrics_path.read_text())
                    per_group_samples[group].append(m)
                    print(f"  --- {group} / rep{r}: SKIP (done: "
                          f"best_obj={m.get('best_objective', float('nan')):.4f}) ---")
                    continue
                except Exception as e:
                    print(f"  --- {group} / rep{r}: metrics.json unreadable ({e}); rerun ---")
            print(f"  --- {group} / rep{r} (seed={seed}) ---")
            try:
                t0 = time.time()
                if kind == "plain":
                    m = SC.run_plain_trial(ens, true_optima, hp, dim, seed, rep_dir, group)
                else:
                    m = SC.run_llm_trial(ens, true_optima, hp, dim, seed, rep_dir, comp_cols)
                runtime = time.time() - t0
                # avg_time_per_iter mirrors run_mobo's ZoMBI-compute-per-iteration: for
                # the LLM group subtract the injection latency (reported separately) so
                # the timing is comparable to the plain groups.
                n_iters = int(m.get("n_iters") or 0)
                compute_time = runtime - float(m.get("llm_total_latency_s") or 0.0)
                avg_tpi = compute_time / n_iters if n_iters > 0 else 0.0
                m["rep"] = r
                m["group"] = group
                m["landscape_y_floor"] = y_floor
                m["landscape_y_opt"] = y_opt
                m["landscape_sample_vals"] = land_sample
                # run_mobo-style metric names (dist_to_true_optima == dist_to_needles).
                m["dist_to_needles"] = m.get("dist_to_true_optima")
                m["runtime_s"] = round(runtime, 3)
                m["avg_time_per_iter_s"] = round(avg_tpi, 4)
                rep_dir.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text(json.dumps(m, indent=2))
                per_group_samples[group].append(m)
                # Per-run report, à la run_mobo._run_zombi_trial.
                print(f"      [run]  iters={n_iters}  dist={m['dist_to_needles']:.4f}  "
                      f"dup={m['dup_fraction']:.4f}  t/iter={avg_tpi:.3f}s  "
                      f"(total {runtime:.1f}s)  needles={m['n_needles']}/{len(true_optima)}  "
                      f"points={m['n_points_total']}", flush=True)
            except Exception as e:
                import traceback
                print(f"      [rep] FAILED: {e}")
                traceback.print_exc()

    # ── Aggregate + sweep-level plots ────────────────────────────────────────
    summary: Dict[str, Any] = {}
    for group, _, _ in groups:
        s = per_group_samples[group]
        summary[group] = {
            "n_repeats": len(s),
            "best_objective": _agg(s, "best_objective"),
            "n_needles": _agg(s, "n_needles"),
            # The three MOBO objectives run_mobo reports per trial.
            "dist_to_needles": _agg(s, "dist_to_needles"),
            "dup_fraction": _agg(s, "dup_fraction"),
            "avg_time_per_iter_s": _agg(s, "avg_time_per_iter_s"),
            "mean_pairwise_needle_dist": _agg(s, "mean_pairwise_needle_dist"),
        }
    (sweep_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))

    try:
        plot_group_convergence(sweep_dir)
    except Exception as e:
        print(f"[convergence] group-convergence plot failed: {e}")
    try:
        plot_slopegraph_final_best(sweep_dir)
    except Exception as e:
        print(f"[convergence] slopegraph failed: {e}")

    print(f"\n[convergence] done → {sweep_dir}")
    for group, _, _ in groups:
        s = summary.get(group, {})
        bo = s.get("best_objective", {})
        print(f"  {group:16s}  best_obj={bo.get('mean', float('nan')):.4f} "
              f"± {bo.get('std', 0.0):.4f}  "
              f"dist={s.get('dist_to_needles', {}).get('mean', float('nan')):.3f}  "
              f"dup={s.get('dup_fraction', {}).get('mean', float('nan')):.3f}  "
              f"t/iter={s.get('avg_time_per_iter_s', {}).get('mean', float('nan')):.3f}s  "
              f"needles={s.get('n_needles', {}).get('mean', float('nan')):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Four-group convergence sweep on re-randomized 10-simplex Ensemble landscapes.")
    parser.add_argument("--no-llm", action="store_true",
                        help="baseline + perturbed + arbitrary only (no LLM calls)")
    parser.add_argument("--resume", action="store_true",
                        help="continue the newest sweep dir, skipping reps with metrics.json")
    parser.add_argument("--replot", nargs="?", const="", default=None, metavar="DIR",
                        help="regenerate the sweep plots for DIR (or the newest sweep dir) and exit")
    args = parser.parse_args()

    if args.replot is not None:
        target = _resolve(args.replot) if args.replot else _find_latest_sweep_dir("sweep_convergence")
        if target is None or not Path(target).is_dir():
            raise SystemExit(f"--replot: no sweep dir to plot (got {target!r})")
        print(f"[convergence] --replot: regenerating plots for {target}")
        replot(Path(target))
    else:
        main(no_llm=args.no_llm, resume=args.resume)
