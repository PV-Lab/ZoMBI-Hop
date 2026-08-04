"""
One ZoMBI-Hop trial, warm-started or cold, under a fixed measurement budget.

This mirrors ``optimize/run_mobo.py: run_single_trial`` — same objective plumbing,
same per-trial artifacts — and adds the two things a warm-start comparison needs
and that function cannot express:

**A measurement budget instead of a wall-clock budget.**  ``run_single_trial``
bounds a trial with ``time_limit_hours``.  That is the wrong control here: a
warm-started run carries ~50 extra GP points from its first iteration, so its GP
fits are slower, so it would complete *fewer* iterations in the same wall time —
the arms would differ for a reason that has nothing to do with the warm start.
:data:`MEASUREMENT_BUDGET` caps total measured compositions instead, so both arms
spend exactly the same number of experiments, which is also the quantity that
costs money on real hardware.

**Seed injection.**  The warm arm's initial data is a line-constrained
space-filling design (``warm_start.warm_start.greedy_lines``) scored on only part
of the objective, registered with the GP as high-noise observations
(``warm_start.seed_prior``).  The cold arm gets the stock random-line init.

Everything else — landscape, hyperparameters, noise model, RNG discipline — is held
identical between the arms so the only difference is the warm start.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import traceback

import sys

import numpy as np
import torch

# run_mobo does a bare ``from eval_metrics import ...``, so optimize/ must be on the
# path as well as the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from optimize import run_mobo as rm
from optimize.composition_prediction import physics_simulate_line
from src import ZoMBIHop

from warm_start import seed_prior
from warm_start.warm_start import POINTS_PER_LINE, greedy_lines, n_lines

# Total measured compositions per trial, both arms.  Matches the 3d run config's
# 600-point budget (warm_start.warm_start.BUDGETS).
MEASUREMENT_BUDGET = 600

#: Hyperparameters held fixed across both arms and all repetitions.  Taken from
#: ``optimize/runs/archived_runs/mobo_3d_05_06_15_32/trial_112`` — the archived 3d
#: seed trial that ``optimize/scripts/ensemble_mobo_3d.sbatch`` also starts from.
#: A controlled comparison needs *one* hyperparameter setting; re-tuning per arm
#: would confound the warm start with a hyperparameter search.
REFERENCE_HPARAMS: dict = {
    "nat_grad_step": 0.00100187,
    "nat_grad_max_steps": 54,
    "n_restarts": 285,
    "raw": 200,
    "ucb_beta": 1.45791911,
    "max_zooms": 3,
    "max_iterations": 8,
    "top_m_points": 8,
    "n_consecutive_converged": 1,
    "input_noise_threshold_mult": 3.98353261,
    "output_noise_threshold_mult": 0.52251166,
    "max_penalty_radius": 4.56475059,
    "needle_shrink_factor": 0.637987,
    "needle_stop_noise_multiplier": 2.81839283,
    "paring_spatial_halfnoise": 1.21375814,
    "paring_y_noise_multiplier": 4.19507699,
}

ARMS = ("cold", "warm")


class _BudgetExhausted(Exception):
    """Raised inside the objective once the measurement budget is spent."""


# ---------------------------------------------------------------------------
# Initial designs
# ---------------------------------------------------------------------------

def _measure_lines(endpoints: np.ndarray, fn_callable, maximize: bool, rng):
    """Measure a set of line segments exactly as the campaign hardware would.

    ``endpoints`` is ``(L, 2, dim)``.  Each line yields ``POINTS_PER_LINE`` evenly
    spaced *requested* compositions and the physics-simulated *actual* ones the
    printer lays down, plus multiplicative output noise on the response — the same
    model ``run_mobo._gen_init_data`` uses, so the two arms' data are directly
    comparable.

    Returns ``(X_actual, X_expected, Y)`` as torch tensors.
    """
    x_a, x_e, ys = [], [], []
    for p, q in endpoints:
        left = torch.as_tensor(p, device=rm.DEVICE, dtype=rm.DTYPE)
        right = torch.as_tensor(q, device=rm.DEVICE, dtype=rm.DTYPE)
        t = torch.linspace(0.0, 1.0, POINTS_PER_LINE,
                           device=rm.DEVICE, dtype=torch.float64)
        clean = (left.to(torch.float64).unsqueeze(0)
                 + t.unsqueeze(1) * (right - left).to(torch.float64).unsqueeze(0))
        actual = physics_simulate_line(left, right, num_points=POINTS_PER_LINE,
                                       device=rm.DEVICE, dtype=torch.float64)
        raw = np.array([fn_callable(x) for x in actual.detach().cpu().numpy()],
                       dtype=float)
        y = torch.tensor(raw if maximize else -raw, device=rm.DEVICE, dtype=rm.DTYPE)
        # Multiplicative output noise, drawn from `rng` rather than global torch
        # state so a repetition's data is reproducible from its seed alone.
        frac = torch.as_tensor(
            rng.normal(size=y.shape[0]), device=rm.DEVICE, dtype=rm.DTYPE)
        y = y + frac * (rm.OUTPUT_NOISE_FRAC * y.abs())
        x_a.append(actual.to(dtype=rm.DTYPE))
        x_e.append(clean.to(dtype=rm.DTYPE))
        ys.append(y)
    return (torch.cat(x_a), torch.cat(x_e), torch.cat(ys).reshape(-1, 1))


def build_warm_start_init(fn_callable, dim: int, maximize: bool, seed: int,
                          y_std: float):
    """Warm-start initial data: a line design scored on a *partial* objective.

    The design is :func:`warm_start.warm_start.greedy_lines` — the hardware-realistic
    one, whole 24-point segments rather than free points.

    Each seed's response is then degraded to stand in for a partially-scored
    measurement: on a synthetic landscape there is no literal stability third to
    omit, so we add an independent perturbation of the size that third would have
    had (``seed_prior.MISSING_STD_FRAC`` of the landscape's own output std).  The
    result is a value that is *honestly* off by about what a real partial score is
    off by, which is exactly what the inflated seed variance then tells the GP.

    Returns ``(X_actual, X_expected, Y_partial, seed_var, real_var)``.
    """
    rng = np.random.default_rng(seed)
    L = n_lines(dim)
    endpoints, _ = greedy_lines(L, dim, seed=seed)
    X_a, X_e, Y_true = _measure_lines(endpoints, fn_callable, maximize, rng)

    seed_var, real_var = seed_prior.seed_noise_for_scale(y_std)
    missing_std = seed_prior.MISSING_STD_FRAC * y_std
    perturb = torch.as_tensor(rng.normal(scale=missing_std, size=Y_true.shape[0]),
                              device=rm.DEVICE, dtype=rm.DTYPE).reshape(-1, 1)
    return X_a, X_e, Y_true + perturb, seed_var, real_var


def build_cold_init(fn_callable, dim: int, maximize: bool, seed: int):
    """Cold-start initial data: the stock ``N_INIT_LINES`` random simplex lines.

    Reimplemented here rather than calling ``run_mobo._gen_init_data`` so the line
    directions come from an explicit seeded RNG; the stock version draws from
    global torch state and would not be reproducible per repetition.
    """
    rng = np.random.default_rng(seed)
    x0 = np.full(dim, 1.0 / dim)
    endpoints = []
    while len(endpoints) < rm.N_INIT_LINES:
        d = rng.normal(size=dim)
        d -= d.mean()                      # zero-sum → stays on the simplex
        n = np.linalg.norm(d)
        if n < 1e-12:
            continue
        seg = rm.line_simplex_segment(
            torch.as_tensor(x0, device=rm.DEVICE, dtype=rm.DTYPE),
            torch.as_tensor(d / n, device=rm.DEVICE, dtype=rm.DTYPE),
        )
        if seg is None:
            continue
        _, _, left, right = seg
        endpoints.append(np.stack([rm.as_numpy(left), rm.as_numpy(right)]))
    X_a, X_e, Y = _measure_lines(np.stack(endpoints), fn_callable, maximize, rng)
    return X_a, X_e, Y


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------

def run_trial(arm: str, rep: int, landscape, trial_dir: str, *,
              ensemble_config: dict | None = None,
              hparams: dict | None = None,
              budget: int = MEASUREMENT_BUDGET) -> dict:
    """Run one trial of `arm` ("cold"/"warm") and write the full artifact set.

    Both arms spend `budget` measured compositions in total, initial design
    included — so the warm arm pays for its 96 seed points out of the same budget
    the cold arm spends entirely on adaptive sampling.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")

    fn_callable = landscape.fn_callable
    true_optima = list(landscape.true_optima)
    grid_vals = landscape.grid_vals
    dim, maximize = landscape.dim, landscape.maximize

    if ensemble_config is not None:
        fn_callable, true_optima, ens_grid_vals = rm.reseed_ensemble(
            landscape, ensemble_config)
        if ens_grid_vals is not None:
            grid_vals = ens_grid_vals

    if os.path.isdir(trial_dir):
        shutil.rmtree(trial_dir, ignore_errors=True)
    os.makedirs(trial_dir, exist_ok=True)
    if ensemble_config is not None:
        with open(os.path.join(trial_dir, "ensemble_config.json"), "w") as f:
            json.dump(ensemble_config, f, indent=2)

    # The landscape's own output spread sets the scale of the seed prior.  Measured
    # from the render grid when there is one, else from a uniform simplex sample.
    if grid_vals is not None:
        y_std = float(np.std(np.asarray(grid_vals, dtype=float)))
    else:
        probe = np.random.default_rng(0).dirichlet(np.ones(dim), size=4000)
        y_std = float(np.std([fn_callable(x) for x in probe]))

    plot_state: dict = {"line_0": None, "line_1": None}
    payloads: list[dict] = []
    snap_records: list[tuple] = []
    call_counter = [0]
    dh_ref: list = [None]

    sim_obj = rm.make_sim_obj(fn_callable, rm.DEVICE, rm.DTYPE, maximize=maximize)
    inner = rm.make_linebo_wrapper(sim_obj, dim, rm.NUM_LINES, rm.DEVICE, rm.DTYPE,
                                   plot_state)
    keep_heavy = landscape.render_ternary
    HEAVY = ("pared_X", "pared_Y", "needle_M_list", "needle_B", "bounds", "gp_grid_vals")

    def obj_wrapper(x_tell, bounds, acq_fn):
        dh = dh_ref[0]
        n_so_far = dh.X_all_actual.shape[0] if dh.X_all_actual is not None else 0
        # Stop *before* measuring a line that would overrun the budget, so neither
        # arm is credited with experiments it was not allowed to run.
        if n_so_far + rm.NUM_EXPERIMENTS > budget:
            raise _BudgetExhausted(f"{n_so_far}/{budget} points measured")

        x_req, x_act, y = inner(x_tell, bounds, acq_fn)
        call_counter[0] += 1
        needles = dh.needles
        payload = dict(
            iter_num=call_counter[0],
            needles=(rm.as_numpy(needles)
                     if needles is not None and needles.shape[0] > 0 else None),
            needle_vals=(rm.as_numpy(dh.needle_vals).ravel()
                         if dh.needle_vals is not None and dh.needle_vals.shape[0] > 0
                         else None),
            line_0=plot_state.get("line_0"),
            line_1=plot_state.get("line_1"),
            n_points_before=n_so_far,
        )
        if payloads and keep_heavy:
            for k in HEAVY:
                payloads[-1].pop(k, None)
        payloads.append(payload)
        return x_req, x_act, y

    # --- initial design --------------------------------------------------------
    init_seed = 10_000 * rep + (0 if arm == "cold" else 1)
    seed_X = None
    seed_var = real_var = None
    if arm == "warm":
        X_a, X_e, Y, seed_var, real_var = build_warm_start_init(
            fn_callable, dim, maximize, init_seed, y_std)
        seed_X = X_a.clone()
    else:
        X_a, X_e, Y = build_cold_init(fn_callable, dim, maximize, init_seed)
    n_init = int(X_a.shape[0])

    hp = dict(REFERENCE_HPARAMS if hparams is None else hparams)
    optimizer = ZoMBIHop(
        objective=obj_wrapper,
        X_init_actual=X_a, X_init_expected=X_e, Y_init=Y,
        **rm.ZOMBI_FIXED, **hp,
        device=str(rm.DEVICE), dtype=rm.DTYPE,
        run_uuid=None, checkpoint_dir=None,
    )
    dh = optimizer.data_handler
    dh_ref[0] = dh

    if arm == "warm":
        # Registered *after* construction so the seeds' stored coordinates (what the
        # data handler holds, and what the GP will later be asked to fit) are what
        # the registry keys on.
        optimizer.gp_handler.set_seed_prior(seed_X, seed_var, real_var)

    orig_snap = dh.take_snapshot

    def snap_wrap(*a, **k):
        orig_snap(*a, **k)
        if dh.X_all_actual is not None:
            czb = dh.current_zoom_bounds if dh.current_zoom_bounds is not None else dh.bounds
            zoom_size = rm.zoom_size_fraction(czb) if czb is not None else 1.0
            snap_records.append((dh.X_all_actual.shape[0], dh.current_activation,
                                 dh.current_zoom, zoom_size))
    dh.take_snapshot = snap_wrap

    t0 = time.time()
    budget_hit = False
    try:
        optimizer.run(max_activations=float("inf"), time_limit_hours=None)
    except _BudgetExhausted as exc:
        budget_hit = True
        print(f"    [{arm} rep{rep}] budget reached: {exc}")
    except Exception as exc:
        print(f"    [{arm} rep{rep}] ZoMBI crashed: {exc}")
        traceback.print_exc()
    runtime = time.time() - t0

    n_iters = call_counter[0]
    needle_t = dh.get_all_needle_locations()
    discovered = (rm.as_numpy(needle_t) if needle_t.numel() > 0
                  else np.empty((0, dim)))
    X_all_np = (rm.as_numpy(dh.X_all_actual) if dh.X_all_actual is not None
                else np.empty((0, dim)))
    dist = rm.metric_dist_to_needles(discovered, true_optima, dim=dim)
    zoom_sizes = rm._zoom_size_per_point(X_all_np.shape[0], snap_records)
    dup = rm.metric_dup_fraction(X_all_np, dim=dim, zoom_sizes=zoom_sizes)
    best_y = float(dh.Y_all.max().item()) if dh.Y_all is not None and dh.Y_all.numel() else float("nan")

    metrics = {
        "arm": arm,
        "rep": rep,
        "dist": float(dist),
        "dup": float(dup),
        "best_y": best_y,
        "n_needles": int(len(discovered)),
        "n_true_optima": int(len(true_optima)),
        "n_points": int(X_all_np.shape[0]),
        "n_init": n_init,
        "n_iters": int(n_iters),
        "runtime": float(runtime),
        "avg_time_per_iter": float(runtime / n_iters) if n_iters else 0.0,
        "budget": int(budget),
        "budget_hit": bool(budget_hit),
        "y_std": y_std,
        "seed_var": seed_var,
        "real_var": real_var,
    }
    print(f"    [{arm} rep{rep}] dist={dist:.4f} dup={dup:.4f} best_y={best_y:.4f} "
          f"needles={len(discovered)}/{len(true_optima)} pts={X_all_np.shape[0]} "
          f"iters={n_iters} ({runtime:.1f}s)", flush=True)

    _write_artifacts(trial_dir, dh, snap_records, payloads, X_all_np, true_optima,
                     landscape, grid_vals, metrics, hp, dim, maximize,
                     ensemble_config)
    return metrics


def _write_artifacts(trial_dir, dh, snap_records, payloads, X_all_np, true_optima,
                     landscape, grid_vals, metrics, hp, dim, maximize,
                     ensemble_config) -> None:
    """Write the same per-trial artifact set ``run_single_trial`` produces.

    Each artifact is guarded independently: a failure in one plot must not cost us
    the trial's data, which is expensive to regenerate.
    """
    def _try(label, fn, *a, **k):
        # SystemExit is caught alongside Exception because some best-effort plot
        # helpers (e.g. optimize/coverage_plot._find_config) call sys.exit() on a
        # missing input rather than raising. sys.exit raises SystemExit, which is a
        # BaseException, so a plain ``except Exception`` would let it abort the whole
        # trial — defeating this guard's purpose of isolating per-artifact failures.
        # The Ensemble-landscape coverage plot has no run_config.json and is not
        # applicable here, so its exit must degrade to a skipped artifact.
        try:
            fn(*a, **k)
        except (Exception, SystemExit) as exc:
            print(f"    [artifact] {label} failed: {exc}")

    _try("points.csv", rm.write_points_csv,
         os.path.join(trial_dir, "points.csv"), dh, snap_records, dim=dim)
    _try("needles.csv", rm.write_needles_csv,
         os.path.join(trial_dir, "needles.csv"), dh, dim=dim)
    _try("metrics_over_time.csv", rm.write_metrics_over_time_csv,
         os.path.join(trial_dir, "metrics_over_time.csv"), payloads, X_all_np,
         true_optima, dim=dim)
    _try("dist_from_centre.png", rm.plot_dist_from_centre,
         os.path.join(trial_dir, "dist_from_centre.png"), dh, maximize)
    _try("line_length_hist.png", rm.plot_line_length_hist,
         os.path.join(trial_dir, "line_length_hist.png"), payloads)
    _try("convergence.png", rm.plot_convergence,
         os.path.join(trial_dir, "convergence.png"), dh, maximize)

    with open(os.path.join(trial_dir, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "hparams": hp,
                   "ensemble_config": ensemble_config}, f, indent=2)

    if landscape.render_ternary and payloads:
        plots_dir = os.path.join(trial_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        _try("final frame", rm.render_frame, payloads[-1], landscape.grid_pts,
             grid_vals, true_optima, maximize,
             os.path.join(plots_dir, f"iter_{payloads[-1]['iter_num'] - 1:04d}.png"))
    _try("auto plots", rm._auto_generate_plots, trial_dir, dim)
    _try("conet", rm._render_conet_artifacts, trial_dir)
