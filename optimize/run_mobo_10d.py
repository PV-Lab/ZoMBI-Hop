"""
optimize/run_mobo_10d.py
========================
MOBO of ZoMBI-Hop hyperparameters on a **Multi-Ackley sum** synthetic objective
on the d-dimensional probability simplex (default d=10).

Each peak is a negated Ackley bump in **composition space** (same formulation as
``scripts/run_zombi_test.py`` ``multimodal_ackley``).  The landscape is

    f(x) = Σ_k ackley_negated(x; center=c_k, b)

ZoMBI maximises f; known true optima are the centres c_k.

Three MOBO objectives (all minimised), same as ``run_mobo.py``:
  1. dist_to_needles  – greedy distance to planted Ackley centres
  2. dup_fraction     – near-duplicate sample fraction
  3. runtime          – wall-clock seconds per ZoMBI trial

Usage (interactive — choose dimension + peak layout at startup)
-----
  conda activate zombi-hop-linebo
  python optimize/run_mobo_10d.py

Non-interactive / Slurm
-----
  MPLBACKEND=Agg python optimize/run_mobo_10d.py --batch --no-show

Results: ``mobo_10d_progress.json``, ``mobo_10d_results.pt``, ``mobo_10d_pareto.png``
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
import warnings

import numpy as np
import torch

from botorch.exceptions import InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement as qLogNEHVI,
)
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

warnings.filterwarnings("ignore", category=InputDataWarning)

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _REPO)

from src import ZoMBIHop, LineBO
from src.core.linebo import line_simplex_segment, zero_sum_dirs
from src.utils.simplex import composition_to_ilr, ilr_to_composition, proj_simplex

# Shared MOBO config/metrics (no matplotlib — headless-safe)
from optimize.mobo_common import (
    HPARAM_NAMES,
    N_HPARAMS,
    NOISE_LEVEL,
    NOISE_LEVEL_ILR,
    NUM_EXPERIMENTS,
    NUM_LINES,
    N_INIT_LINES,
    UNMATCHED_PENALTY,
    ZOMBI_FIXED,
    metric_dist_to_needles,
    metric_dup_fraction,
    norm_to_hparams,
    save_running_summary,
)

# ─── Defaults (override via CLI) ───────────────────────────────────────────────

DEFAULT_DIM = 10
DEFAULT_N_INIT = 8
DEFAULT_N_MOBO = 20
DEFAULT_N_MOBO_RESTARTS = 10
DEFAULT_N_MOBO_SAMPLES = 512

_ACKLEY_A = 20.0
_ACKLEY_B = 0.2
_ACKLEY_B_SKINNY = 1.2   # skinnier peaks — matches run_zombi_test multimodal_ackley
_ACKLEY_C = 2.0 * math.pi
_ACKLEY_SCALE = 30.0

# TestFn: (callable, true_optima, n_activations, maximize, name)
TestFn = tuple


# ─── Simplex helpers ───────────────────────────────────────────────────────────

def simplex_vertex(d: int, i: int) -> np.ndarray:
    """Exact simplex vertex e_i (analog of [1,0,0] in 3D)."""
    p = np.zeros(d, dtype=float)
    p[i] = 1.0
    return p


def centroid_composition(d: int) -> np.ndarray:
    """Uniform composition (analog of [1/3, 1/3, 1/3] in 3D)."""
    return np.ones(d, dtype=float) / d


def edge_midpoint(d: int, i: int, j: int) -> np.ndarray:
    """Mass 0.5 on coordinates i and j (analog of [0.5, 0.5, 0] in 3D)."""
    p = np.zeros(d, dtype=float)
    p[i] = 0.5
    p[j] = 0.5
    return p


# ─── Multi-Ackley (composition space) ─────────────────────────────────────────

def _ackley_negated(
    x: np.ndarray,
    center: np.ndarray,
    *,
    a: float = _ACKLEY_A,
    b: float = _ACKLEY_B,
    c: float = _ACKLEY_C,
    scale: float = _ACKLEY_SCALE,
) -> float:
    """
    Negated Ackley centred at ``center`` on the simplex.

    Maximum at x == center (value ≈ 0); negative away from centre.
    Same formula as ``scripts/run_zombi_test._ackley_negated``.
    """
    x = np.asarray(x, dtype=float)
    center = np.asarray(center, dtype=float)
    d = x.shape[0]
    delta = x - center
    t1 = -a * math.exp(-b * math.sqrt(np.sum(delta ** 2) / d))
    t2 = -math.exp(float(np.sum(np.cos(c * delta)) / d))
    return scale * (t1 + t2 + a + math.e)


class MultiAckleyND:
    """
    Sum of negated Ackley bumps — direct d-dimensional generalisation of
    ``multimodal_ackley`` in ``run_zombi_test.py``.
    """

    maximize = True

    def __init__(
        self,
        centers: list[np.ndarray],
        *,
        b: float = _ACKLEY_B_SKINNY,
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
    """
    Planted Ackley peak locations on Δ^d.

    Layouts
    -------
    ``1`` — 3D-analog trimodal (default for benchmarking):
            centroid, vertex 0, edge midpoint (0,1).
            Mirrors [1/3,1/3,1/3], [1,0,0], [0.5,0.5,0] when d=3.
    ``2`` — 5-peak: layout 1 + vertices 1 and 2.
    ``3`` — 7-peak: layout 2 + vertices 3 and 4 (needs d ≥ 5).
    """
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


def build_multi_ackley_test_fn(
    d: int,
    layout: str,
    *,
    b: float = _ACKLEY_B_SKINNY,
) -> TestFn:
    """Return a single TestFn tuple for Multi-Ackley MOBO."""
    centers = ackley_centers_for_layout(d, layout)
    ack = MultiAckleyND(centers, b=b, layout_name=f"layout-{layout}")
    n_act = max(2, 2 * len(ack.true_optima))
    name = f"MultiAckley-{d}D-L{layout}"
    return (ack, ack.true_optima, n_act, ack.maximize, name)


# ─── Interactive startup ──────────────────────────────────────────────────────

_LAYOUT_LABELS = {
    "1": "3D-analog trimodal (centroid + v0 + edge 0–1)",
    "2": "5-peak (+ v1, v2)",
    "3": "7-peak (+ v3, v4; needs d ≥ 5)",
}


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


def _prompt_ackley_b(default: float = _ACKLEY_B_SKINNY) -> float:
    print("\nAckley peak width b (Enter = skinny 1.2):")
    print("  1.2 — skinny peaks, well-separated (matches run_zombi_test multimodal)")
    print("  0.2 — standard Ackley width")
    raw = input("> ").strip().lower()
    if raw in ("", "1.2", "skinny", "s"):
        return _ACKLEY_B_SKINNY
    if raw in ("0.2", "standard", "std"):
        return _ACKLEY_B
    try:
        return float(raw)
    except ValueError:
        print(f"  Invalid; using b={default}.")
        return default


def interactive_startup() -> tuple[int, str, float]:
    """Prompt for dimension, peak layout, and Ackley b."""
    print("=" * 70)
    print("ZoMBI-Hop MOBO — Multi-Ackley sum on Δ^d")
    print("=" * 70)
    d = _prompt_int(DEFAULT_DIM, "Simplex dimension d", lo=2, hi=20)
    layout = _prompt_layout("1")
    # Downgrade layout if d too small
    if layout == "3" and d < 5:
        print("  Layout 3 needs d ≥ 5; falling back to layout 2.")
        layout = "2"
    if layout == "2" and d < 3:
        print("  Layout 2 needs d ≥ 3; falling back to layout 1.")
        layout = "1"
    b = _prompt_ackley_b()
    return d, layout, b


# ─── ZoMBI simulation wrappers (dimension-aware) ───────────────────────────────

def make_sim_obj(fn_callable, dim: int, device, dtype, *, maximize: bool):
    def sim_objective(endpoints: torch.Tensor):
        left = endpoints[0, 0].to(torch.float64)
        right = endpoints[0, 1].to(torch.float64)
        t = torch.linspace(0.0, 1.0, NUM_EXPERIMENTS,
                           dtype=torch.float64, device=left.device)
        pts_t = left.unsqueeze(0) + t.unsqueeze(1) * (right - left).unsqueeze(0)
        z = composition_to_ilr(pts_t)
        z = z + torch.randn_like(z) * NOISE_LEVEL_ILR
        pts_t = ilr_to_composition(z, d=dim)
        pts_np = pts_t.detach().cpu().numpy()
        raw = np.array([fn_callable(x) for x in pts_np], dtype=float)
        y = torch.tensor(raw if maximize else -raw, dtype=dtype, device=device)
        y = y + torch.randn_like(y) * NOISE_LEVEL
        return pts_t.to(dtype=dtype, device=device), y

    return sim_objective


def make_linebo_wrapper(sim_obj, dim: int, num_lines: int, device, dtype):
    linebo = LineBO(
        sim_obj, dim,
        num_points_per_line=100, num_lines=num_lines, device=str(device),
    )

    def wrapper(x_tell, bounds, acq_fn):
        x_left_r, x_right_r = linebo.ranked_line_endpoints(x_tell, bounds, acq_fn)
        endpoints = torch.stack([x_left_r, x_right_r], dim=1)
        x_actual, y = sim_obj(endpoints)
        x_actual = x_actual.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype).ravel()
        if x_actual.shape[0] > 1:
            xc = x_actual - x_actual.mean(dim=0, keepdim=True)
            _, _, Vt = torch.linalg.svd(xc, full_matrices=False)
            direction = Vt[0]
            projs = xc @ direction
            t_vals = torch.linspace(
                projs.min().item(), projs.max().item(),
                x_actual.shape[0], device=device, dtype=dtype,
            )
            x_requested = (
                x_actual.mean(dim=0).unsqueeze(0)
                + t_vals.unsqueeze(1) * direction.unsqueeze(0)
            )
            x_requested = proj_simplex(x_requested)
        else:
            x_requested = x_actual.clone()
        return x_requested, x_actual, y

    return wrapper


def _gen_init_data(fn_callable, dim: int, device, dtype, *, maximize: bool):
    x_a_list, x_e_list, y_list, all_X = [], [], [], []
    x0 = torch.full((dim,), 1.0 / dim, device=device, dtype=dtype)
    for _ in range(N_INIT_LINES):
        dir_ = zero_sum_dirs(1, dim, device=device, dtype=dtype).squeeze(0)
        seg = line_simplex_segment(x0, dir_)
        if seg is None:
            continue
        _, _, x_left, x_right = seg
        t = torch.linspace(0.0, 1.0, NUM_EXPERIMENTS, dtype=torch.float64, device=device)
        pts_t = (
            x_left.to(torch.float64).unsqueeze(0)
            + t.unsqueeze(1) * (x_right - x_left).to(torch.float64).unsqueeze(0)
        )
        z = composition_to_ilr(pts_t)
        z = z + torch.randn_like(z) * NOISE_LEVEL_ILR
        pts_t = ilr_to_composition(z, d=dim)
        pts_np = pts_t.detach().cpu().numpy()
        raw = np.array([fn_callable(x) for x in pts_np], dtype=float)
        y_t = torch.tensor(raw if maximize else -raw, dtype=dtype, device=device)
        y_t = y_t + torch.randn_like(y_t) * NOISE_LEVEL
        pts_out = pts_t.to(dtype=dtype, device=device)
        x_a_list.append(pts_out)
        x_e_list.append(pts_out)
        y_list.append(y_t)
        all_X.extend(pts_np)
    if not x_a_list:
        raise RuntimeError("Could not generate any initial simplex lines.")
    return (
        torch.cat(x_a_list, dim=0),
        torch.cat(x_e_list, dim=0),
        torch.cat(y_list, dim=0).reshape(-1, 1),
        np.array(all_X),
    )


def _run_single_zombi(
    hparams: dict,
    fn_callable,
    true_optima: list[np.ndarray],
    n_activations: int,
    maximize: bool,
    label: str,
    *,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, float]:
    import shutil

    try:
        X_init_a, X_init_e, Y_init, init_X_np = _gen_init_data(
            fn_callable, dim, device, dtype, maximize=maximize,
        )
    except RuntimeError as exc:
        print(f"    [{label}] init failed: {exc}")
        return UNMATCHED_PENALTY, 1.0, 600.0

    all_X_run: list[np.ndarray] = list(init_X_np)
    sim_obj = make_sim_obj(fn_callable, dim, device, dtype, maximize=maximize)
    base_obj = make_linebo_wrapper(sim_obj, dim, NUM_LINES, device, dtype)

    def objective_with_accum(x_tell, bounds, acq_fn):
        x_req, x_act, y = base_obj(x_tell, bounds, acq_fn)
        all_X_run.extend(x_act.detach().cpu().numpy())
        return x_req, x_act, y

    ckpt_dir = tempfile.mkdtemp(prefix="zombi_mobo_10d_")
    t0 = time.time()
    try:
        hp = dict(hparams)
        if "top_m_points" not in hp or hp.get("top_m_points") is None:
            hp["top_m_points"] = max(dim + 1, 4)
        optimizer = ZoMBIHop(
            objective=objective_with_accum,
            X_init_actual=X_init_a,
            X_init_expected=X_init_e,
            Y_init=Y_init,
            **ZOMBI_FIXED,
            **hp,
            device=str(device),
            dtype=dtype,
            run_uuid=None,
            checkpoint_dir=ckpt_dir,
            num_iterations_saved=3,
        )
        optimizer.run(max_activations=n_activations, time_limit_hours=None)
        runtime = time.time() - t0
    except Exception as exc:
        print(f"    [{label}] ZoMBI crashed: {exc}")
        return UNMATCHED_PENALTY, 1.0, time.time() - t0
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    dh = optimizer.data_handler
    needle_t = dh.get_all_needle_locations()
    discovered = (
        needle_t.detach().cpu().numpy()
        if needle_t.numel() > 0
        else np.empty((0, dim))
    )
    X_sampled = np.array(all_X_run) if all_X_run else np.empty((0, dim))

    dist = metric_dist_to_needles(discovered, true_optima)
    dup = metric_dup_fraction(X_sampled, NOISE_LEVEL / 2.0)
    print(
        f"    [{label}]  dist={dist:.4f}  dup={dup:.4f}  t={runtime:.1f}s"
        f"  needles={len(discovered)}/{len(true_optima)}"
    )
    return dist, dup, runtime


def evaluate_trial(
    x_norm: torch.Tensor,
    test_fns: list[TestFn],
    trial_idx: int,
    *,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, float]:
    hparams = norm_to_hparams(x_norm)
    hp_str = "  ".join(
        f"{k}={round(v, 4) if isinstance(v, float) else v}"
        for k, v in hparams.items()
    )
    print(f"\n  [trial {trial_idx:>3}]  {hp_str}")

    dists, dups, runtimes = [], [], []
    for fn, optima, n_act, maximize, name in test_fns:
        d, dup, t = _run_single_zombi(
            hparams, fn, optima, n_act, maximize, name,
            dim=dim, device=device, dtype=dtype,
        )
        dists.append(d)
        dups.append(dup)
        runtimes.append(t)

    mean_dist = float(np.mean(dists))
    mean_dup = float(np.mean(dups))
    total_t = float(sum(runtimes))
    print(
        f"  [trial {trial_idx:>3}]  MEAN dist={mean_dist:.4f}  dup={mean_dup:.4f}"
        f"  total_t={total_t:.1f}s  ({len(test_fns)} functions)"
    )
    return mean_dist, mean_dup, total_t


def run_mobo(
    test_fns: list[TestFn],
    save_dir: str,
    *,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    n_init_trials: int,
    n_mobo_trials: int,
    n_mobo_restarts: int,
    n_mobo_samples: int,
    progress_json: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    bounds = torch.zeros(2, N_HPARAMS, dtype=dtype, device=device)
    bounds[1] = 1.0

    X_obs: list[torch.Tensor] = []
    Y_obs: list[torch.Tensor] = []

    fn_names = [tf[4] for tf in test_fns]
    print(f"\n{'=' * 70}")
    print(f"Multi-Ackley MOBO  |  d={dim}  |  {n_init_trials} Sobol + {n_mobo_trials} BO trials")
    print(f"Hyperparameters ({N_HPARAMS}): {HPARAM_NAMES}")
    print(f"Test functions ({len(test_fns)}): {fn_names}")
    print(f"Device: {device}")
    print(f"{'=' * 70}")

    X_sobol = draw_sobol_samples(bounds=bounds, n=n_init_trials, q=1).squeeze(1)
    for i, x in enumerate(X_sobol):
        d, dup, t = evaluate_trial(
            x, test_fns, trial_idx=i, dim=dim, device=device, dtype=dtype,
        )
        X_obs.append(x.cpu())
        Y_obs.append(torch.tensor([-d, -dup, -t], dtype=dtype))
        save_running_summary(
            X_obs, Y_obs, progress_json, n_init_trials=n_init_trials,
        )

    for trial in range(n_mobo_trials):
        X_t = torch.stack(X_obs).to(device)
        Y_t = torch.stack(Y_obs).to(device)

        span = (Y_t.max(dim=0).values - Y_t.min(dim=0).values).clamp(min=1e-6)
        ref_point = (Y_t.min(dim=0).values - 0.1 * span).tolist()

        model = SingleTaskGP(
            X_t, Y_t,
            input_transform=Normalize(d=N_HPARAMS),
            outcome_transform=Standardize(m=3),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        acq = qLogNEHVI(model=model, ref_point=ref_point, X_baseline=X_t)
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds.to(device),
            q=1,
            num_restarts=n_mobo_restarts,
            raw_samples=n_mobo_samples,
        )
        x_new = candidate.squeeze(0).detach()

        d, dup, t = evaluate_trial(
            x_new, test_fns, trial_idx=n_init_trials + trial,
            dim=dim, device=device, dtype=dtype,
        )
        X_obs.append(x_new.cpu())
        Y_obs.append(torch.tensor([-d, -dup, -t], dtype=dtype))
        save_running_summary(
            X_obs, Y_obs, progress_json, n_init_trials=n_init_trials,
        )

    return torch.stack(X_obs), torch.stack(Y_obs)


def show_pareto(
    X_obs: torch.Tensor,
    Y_obs: torch.Tensor,
    save_dir: str,
    *,
    show: bool,
) -> torch.Tensor:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pareto_mask = is_non_dominated(Y_obs)
    n_par = int(pareto_mask.sum().item())
    print(f"\n{'=' * 70}")
    print(f"Pareto front: {n_par} / {len(Y_obs)} trials")
    print(f"{'rank':>4}  {'dist':>8}  {'dup%':>8}  {'time(s)':>9}  hparams")
    print("-" * 70)
    pareto_idx = torch.where(pareto_mask)[0]
    order = pareto_idx[Y_obs[pareto_idx, 0].argsort(descending=True)]
    for rank, idx in enumerate(order):
        y = Y_obs[idx]
        hp = norm_to_hparams(X_obs[idx])
        hp_str = "  ".join(
            f"{k}={round(v, 4) if isinstance(v, float) else v}"
            for k, v in hp.items()
        )
        print(
            f"{rank + 1:>4}  {-y[0].item():>8.4f}  {-y[1].item():>8.4f}"
            f"  {-y[2].item():>9.1f}  {hp_str}"
        )

    Y_np = (-Y_obs).cpu().numpy()
    pm_np = pareto_mask.numpy()
    pairs = [
        (0, 2, "dist_to_needles", "runtime (s)"),
        (0, 1, "dist_to_needles", "dup_fraction"),
        (1, 2, "dup_fraction", "runtime (s)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Multi-Ackley MOBO Pareto front  (★ = Pareto-optimal)", fontsize=12)
    for ax, (ix, iy, xl, yl) in zip(axes, pairs):
        ax.scatter(
            Y_np[~pm_np, ix], Y_np[~pm_np, iy],
            c="steelblue", alpha=0.6, edgecolors="k", linewidths=0.3, label="dominated",
        )
        ax.scatter(
            Y_np[pm_np, ix], Y_np[pm_np, iy],
            marker="*", s=220, c="gold", zorder=5,
            edgecolors="k", linewidths=0.5, label="Pareto",
        )
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "mobo_10d_pareto.png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"\nPareto plot saved to {fig_path}")
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)
    return order


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="MOBO ZoMBI hyperparameters on Multi-Ackley sum (Δ^d).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--batch", action="store_true",
        help="Skip prompts: d=10, layout 1, b=1.2 (for Slurm).",
    )
    ap.add_argument("--dim", type=int, default=None,
                    help="With --batch: simplex dimension (default 10).")
    ap.add_argument("--layout", type=str, default=None, choices=["1", "2", "3"],
                    help="With --batch: peak layout 1/2/3.")
    ap.add_argument("--ackley-b", type=float, default=None,
                    help="With --batch: Ackley b width (default 1.2).")
    ap.add_argument("--n-init-trials", type=int, default=DEFAULT_N_INIT)
    ap.add_argument("--n-mobo-trials", type=int, default=DEFAULT_N_MOBO)
    ap.add_argument("--device", type=str, default=None,
                    help="cuda or cpu (default: cuda if available).")
    ap.add_argument("--no-show", action="store_true",
                    help="Save Pareto PNG only; do not open a plot window.")
    ap.add_argument("--save-dir", type=str, default=script_dir,
                    help="Directory for progress JSON, .pt, and Pareto PNG.")
    args = ap.parse_args()

    if args.batch:
        dim = args.dim if args.dim is not None else DEFAULT_DIM
        layout = args.layout if args.layout is not None else "1"
        ackley_b = args.ackley_b if args.ackley_b is not None else _ACKLEY_B_SKINNY
    else:
        dim, layout, ackley_b = interactive_startup()

    if dim < 2 or dim > 20:
        sys.exit("Simplex dimension must be between 2 and 20.")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype = torch.float64
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    progress_json = os.path.join(save_dir, "mobo_10d_progress.json")

    test_fn = build_multi_ackley_test_fn(dim, layout, b=ackley_b)
    test_fns = [test_fn]
    _fn, optima, n_act, _maxm, name = test_fn

    print(f"\n{'=' * 70}")
    print(f"Multi-Ackley MOBO  |  d={dim}  layout={layout}  b={ackley_b}")
    print(f"  {_LAYOUT_LABELS[layout]}")
    print(f"  Device: {device}")
    print(f"  {len(optima)} planted peaks  |  max_activations={n_act}")
    print(f"{'=' * 70}")
    for i, p in enumerate(optima):
        print(f"  peak {i + 1}: {np.round(p, 4).tolist()}")

    X_obs, Y_obs = run_mobo(
        test_fns,
        save_dir,
        dim=dim,
        device=device,
        dtype=dtype,
        n_init_trials=args.n_init_trials,
        n_mobo_trials=args.n_mobo_trials,
        n_mobo_restarts=DEFAULT_N_MOBO_RESTARTS,
        n_mobo_samples=DEFAULT_N_MOBO_SAMPLES,
        progress_json=progress_json,
    )

    pareto_order = show_pareto(
        X_obs, Y_obs, save_dir, show=not args.no_show,
    )

    results_path = os.path.join(save_dir, "mobo_10d_results.pt")
    torch.save(
        {
            "X_obs": X_obs,
            "Y_obs": Y_obs,
            "hparam_names": HPARAM_NAMES,
            "dim": dim,
            "layout": layout,
            "ackley_b": ackley_b,
            "objective": "MultiAckleyND",
        },
        results_path,
    )
    print(f"\nResults saved to {results_path}")
    print(f"Progress JSON: {progress_json}")

    if len(pareto_order) > 0:
        best_idx = pareto_order[0]
        best_hp = norm_to_hparams(X_obs[best_idx])
        print("\nBest config (min dist_to_needles on Pareto front):")
        for k, v in best_hp.items():
            print(f"  {k}: {round(v, 6) if isinstance(v, float) else v}")


if __name__ == "__main__":
    main()
