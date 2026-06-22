"""
run_zombi_test_v2_parallel.py
=============================

V2 benchmark + MOBO with **benchmark-level parallelism**: the 32–64 ZoMBI
sub-runs inside each MOBO hyperparameter evaluation can run concurrently.

This is complementary to ``--workers`` (MOBO batch parallelism across configs).
Recommended CPU layout:

    --device cpu --workers 1 --benchmark-workers 16

Avoid stacking ``--workers > 1`` and ``--benchmark-workers > 1`` on the same
node unless you have many cores (each MOBO worker spawns its own sub-run pool).

Usage
-----
    # Parallel benchmark only (no MOBO):
    python scripts/run_zombi_test_v2_parallel.py \\
        --regions scripts/max_min_regions.json \\
        --benchmark-workers 16 --device cpu

    # MOBO with parallel sub-runs per eval:
    python scripts/run_zombi_test_v2_parallel.py --mobo \\
        --regions scripts/max_min_regions.json \\
        --mobo-init 8 --mobo-iters 10 --batch 4 \\
        --workers 1 --benchmark-workers 16 --device cpu
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import multiprocessing
import os
import sys
import warnings
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.run_zombi_test import (
    CSV_ELEMENT_TRIPLE,
    CSV_OBJECTIVES,
    CSV_PEROVSKITE_PATH,
    RF_CACHE_DIR,
    ackley_edge,
    ackley_equal,
    ackley_vertex,
    build_csv_rf_objectives,
    multimodal_ackley,
    ACKLEY_CENTER_EDGE,
    ACKLEY_CENTER_EQUAL,
    ACKLEY_CENTER_VERTEX,
    MULTIMODAL_CENTERS,
)
from scripts.run_zombi_test_v2 import (
    RF_CACHE_DIR as _V2_RF_CACHE_DIR,
    _NEEDLE_VOL_PENALTY,
    load_regions,
    mobo_tune_zombi_v2,
    run_zombi_on_objective_v2,
    run_zombi_test_v2,
)

_ACKLEY_BY_NAME: Dict[str, Callable[[np.ndarray], float]] = {
    "Ackley-Centroid": ackley_equal,
    "Ackley-Edge": ackley_edge,
    "Ackley-Vertex": ackley_vertex,
    "Ackley-Multi-modal": multimodal_ackley,
}

_ACKLEY_ANALYTIC_MAX: Dict[str, List[np.ndarray]] = {
    "Ackley-Centroid": [ACKLEY_CENTER_EQUAL],
    "Ackley-Edge": [ACKLEY_CENTER_EDGE],
    "Ackley-Vertex": [ACKLEY_CENTER_VERTEX],
    "Ackley-Multi-modal": MULTIMODAL_CENTERS,
}

_NOISE_COMBOS: List[Tuple[float, float]] = [
    (0.01, 0.01),
    (0.01, 0.001),
    (0.001, 0.01),
    (0.001, 0.001),
]


def _set_thread_env(n: int) -> None:
    n = max(1, int(n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["BLAS_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)


@lru_cache(maxsize=None)
def _cached_csv_rf(
    csv_path: str,
    obj_col: str,
    rf_global_samples: int,
    rf_cache_dir: Optional[str],
):
    data = build_csv_rf_objectives(
        csv_path=csv_path,
        objectives=CSV_OBJECTIVES,
        rf_global_samples=rf_global_samples,
        cache_dir=rf_cache_dir,
    )
    return data[obj_col]["rf"]


def _resolve_subrun_fn(task: Dict[str, Any]) -> Callable[[np.ndarray], float]:
    kind = task["fn_kind"]
    if kind.startswith("ackley:"):
        name = kind.split(":", 1)[1]
        return _ACKLEY_BY_NAME[name]
    if kind.startswith("csv_rf:"):
        obj_col = kind.split(":", 1)[1]
        rf = _cached_csv_rf(
            task["csv_path"],
            obj_col,
            int(task["rf_global_samples"]),
            task.get("rf_cache_dir"),
        )

        def _fn(x: np.ndarray, _rf=rf) -> float:
            return float(_rf.predict(x.reshape(1, -1))[0])

        return _fn
    raise ValueError(f"Unknown fn_kind: {kind!r}")


def _benchmark_subrun_worker(task: Dict[str, Any]) -> Dict:
    """Picklable worker: one ZoMBI sub-run (objective × noise combo)."""
    _set_thread_env(int(task["threads_per_worker"]))
    fn = _resolve_subrun_fn(task)
    known_extrema = [np.asarray(km, dtype=np.float64) for km in task["known_extrema"]]
    zombi_kw = dict(task["zombi_kw"])
    return run_zombi_on_objective_v2(
        fn=fn,
        known_extrema=known_extrema,
        name=task["full_name"],
        obj_name_for_region=task["region_key"],
        regions_data=task["regions_data"],
        mode=task["mode"],
        L=int(task["L"]),
        input_noise=float(task["inp_noise"]),
        output_noise=float(task["out_noise"]),
        seed=int(task["seed"]),
        **zombi_kw,
    )


def _get_top_seeds(
    regions_data: Optional[Dict],
    obj_name: str,
    mode: str,
    L_max: int = 10,
) -> Tuple[List[np.ndarray], int]:
    from scripts.run_zombi_test_v2 import _get_top_seeds as _v2_get_top_seeds

    return _v2_get_top_seeds(regions_data, obj_name, mode, L_max=L_max)


def _build_subrun_tasks(
    *,
    regions_data: Optional[Dict],
    csv_rf_objectives: Dict,
    zombi_kw: Dict[str, Any],
    csv_path: str,
    rf_global_samples: int,
    rf_cache_dir: Optional[str],
    verbose: bool,
) -> List[Dict[str, Any]]:
    """Build ordered picklable task dicts for every benchmark sub-run."""
    objectives: List[Tuple[str, str, str, List[np.ndarray], str, int]] = []

    for obj_col, data in csv_rf_objectives.items():
        base_name = f"CSV-RF-{obj_col} ({'/'.join(CSV_ELEMENT_TRIPLE)})"
        region_key = f"RF-{obj_col}"
        for mode in ("max", "min"):
            top_seeds, L = _get_top_seeds(regions_data, region_key, mode)
            if not top_seeds:
                if mode == "max":
                    top_seeds = [data["global_max_x"]]
                else:
                    gmin = data.get("global_min_x")
                    top_seeds = [gmin] if gmin is not None else []
                L = max(len(top_seeds), 1)
            objectives.append(
                (base_name, region_key, f"csv_rf:{obj_col}", top_seeds, mode, L),
            )

    for aname in _ACKLEY_BY_NAME:
        region_key = aname
        analytic_max = _ACKLEY_ANALYTIC_MAX[aname]
        for mode in ("max", "min"):
            top_seeds, L = _get_top_seeds(regions_data, region_key, mode)
            if not top_seeds:
                if mode == "max":
                    top_seeds = [np.asarray(km) for km in analytic_max]
                else:
                    top_seeds = []
                L = max(len(top_seeds), 1)
            objectives.append(
                (aname, region_key, f"ackley:{aname}", top_seeds, mode, L),
            )

    tasks: List[Dict[str, Any]] = []
    total_runs = len(objectives) * len(_NOISE_COMBOS)
    run_idx = 0
    for base_name, region_key, fn_kind, known_extrema, mode, L in objectives:
        for inp_noise, out_noise in _NOISE_COMBOS:
            run_idx += 1
            noise_tag = f"in={inp_noise:.3f}/out={out_noise:.3f}"
            full_name = f"{base_name} [{mode}] [{noise_tag}]"
            if verbose:
                print(
                    f"\n  [{run_idx}/{total_runs}] {full_name}  "
                    f"L={L}  activations={math.ceil(1.2 * L)}",
                )
            tasks.append(
                {
                    "full_name": full_name,
                    "region_key": region_key,
                    "fn_kind": fn_kind,
                    "known_extrema": [np.asarray(km).tolist() for km in known_extrema],
                    "mode": mode,
                    "L": L,
                    "inp_noise": inp_noise,
                    "out_noise": out_noise,
                    "run_idx": run_idx,
                    "total_runs": total_runs,
                    "seed": run_idx,
                    "regions_data": regions_data,
                    "csv_path": csv_path,
                    "rf_global_samples": rf_global_samples,
                    "rf_cache_dir": rf_cache_dir,
                    "zombi_kw": zombi_kw,
                },
            )
    return tasks


def run_zombi_test_v2_parallel(
    *,
    regions_path: Optional[str] = None,
    benchmark_workers: int = 1,
    threads_per_worker: Optional[int] = None,
    epsilon_frac: float = 0.2,
    max_zooms: int = 3,
    max_iterations: int = 10,
    n_restarts: int = 30,
    raw_samples: int = 500,
    top_m_points: Optional[int] = None,
    penalization_threshold: float = 1e-3,
    penalty_max_radius: float = 0.3,
    convergence_pi_threshold: float = 0.01,
    n_consecutive_converged: int = 2,
    max_gp_points: int = 3000,
    ucb_beta: float = 0.1,
    repulsion_lambda: Optional[float] = None,
    nat_grad_step: float = 0.02,
    nat_grad_max_steps: int = 50,
    num_points_per_line: int = 100,
    num_lines: int = 30,
    num_init_data: int = 4,
    rf_global_samples: int = 10_000_000,
    rf_cache_dir: Optional[str] = _V2_RF_CACHE_DIR,
    csv_path: str = CSV_PEROVSKITE_PATH,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    show_plot: bool = False,
) -> List[Dict]:
    """
    Same benchmark as ``run_zombi_test_v2``, but sub-runs execute in parallel
    when ``benchmark_workers > 1``.
    """
    from scripts.run_zombi_test_v2 import _resolve_device

    if show_plot:
        warnings.warn("show_plot is ignored in parallel benchmark mode.", stacklevel=2)

    benchmark_workers = max(1, int(benchmark_workers))
    device = _resolve_device(device)
    regions_data = load_regions(regions_path)
    if regions_data is not None:
        print(
            f"  Loaded regions from: {regions_path}  "
            f"({len(regions_data.get('objectives', {}))} objectives)",
        )
    else:
        print("  No regions file — using V1 point-based distance (L=1 fallback).")

    zombi_kw = dict(
        num_init_data=num_init_data,
        max_zooms=max_zooms,
        max_iterations=max_iterations,
        n_restarts=n_restarts,
        raw_samples=raw_samples,
        top_m_points=top_m_points,
        penalization_threshold=penalization_threshold,
        penalty_max_radius=penalty_max_radius,
        convergence_pi_threshold=convergence_pi_threshold,
        n_consecutive_converged=n_consecutive_converged,
        max_gp_points=max_gp_points,
        ucb_beta=ucb_beta,
        repulsion_lambda=repulsion_lambda,
        nat_grad_step=nat_grad_step,
        nat_grad_max_steps=nat_grad_max_steps,
        num_points_per_line=num_points_per_line,
        num_lines=num_lines,
        device=device,
        dtype=dtype,
        verbose=verbose,
        epsilon_frac=epsilon_frac,
    )

    csv_rf_objectives: Dict = {}
    if os.path.isfile(csv_path):
        try:
            csv_rf_objectives = build_csv_rf_objectives(
                csv_path=csv_path,
                objectives=CSV_OBJECTIVES,
                rf_global_samples=rf_global_samples,
                cache_dir=rf_cache_dir,
            )
        except Exception as exc:
            warnings.warn(f"CSV RF build failed: {exc}", stacklevel=2)
    else:
        warnings.warn(f"CSV not found: {csv_path}", stacklevel=2)

    tasks = _build_subrun_tasks(
        regions_data=regions_data,
        csv_rf_objectives=csv_rf_objectives,
        zombi_kw=zombi_kw,
        csv_path=csv_path,
        rf_global_samples=rf_global_samples,
        rf_cache_dir=rf_cache_dir,
        verbose=verbose and benchmark_workers == 1,
    )

    if benchmark_workers == 1 or len(tasks) <= 1:
        return run_zombi_test_v2(
            regions_path=regions_path,
            epsilon_frac=epsilon_frac,
            max_zooms=max_zooms,
            max_iterations=max_iterations,
            n_restarts=n_restarts,
            raw_samples=raw_samples,
            top_m_points=top_m_points,
            penalization_threshold=penalization_threshold,
            penalty_max_radius=penalty_max_radius,
            convergence_pi_threshold=convergence_pi_threshold,
            n_consecutive_converged=n_consecutive_converged,
            max_gp_points=max_gp_points,
            ucb_beta=ucb_beta,
            repulsion_lambda=repulsion_lambda,
            nat_grad_step=nat_grad_step,
            nat_grad_max_steps=nat_grad_max_steps,
            num_points_per_line=num_points_per_line,
            num_lines=num_lines,
            num_init_data=num_init_data,
            rf_global_samples=rf_global_samples,
            rf_cache_dir=rf_cache_dir,
            csv_path=csv_path,
            device=device,
            dtype=dtype,
            verbose=verbose,
            show_plot=False,
        )

    cpu_count = os.cpu_count() or 1
    if threads_per_worker is None:
        threads_per_worker = max(1, cpu_count // benchmark_workers)
    for task in tasks:
        task["threads_per_worker"] = threads_per_worker

    print(
        f"\n  Running {len(tasks)} benchmark sub-run(s) in parallel "
        f"(benchmark_workers={benchmark_workers}, "
        f"threads/worker={threads_per_worker}) …",
    )

    mp_ctx = multiprocessing.get_context("spawn")
    results: List[Optional[Dict]] = [None] * len(tasks)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(benchmark_workers, len(tasks)),
        mp_context=mp_ctx,
    ) as pool:
        futures = {
            pool.submit(_benchmark_subrun_worker, task): i
            for i, task in enumerate(tasks)
        }
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            done += 1
            try:
                results[idx] = fut.result()
            except Exception as exc:
                warnings.warn(
                    f"  [sub-run {idx + 1}/{len(tasks)}] failed: {exc}",
                    stacklevel=2,
                )
                results[idx] = {
                    "name": tasks[idx]["full_name"],
                    "avg_region_dist": 10.0,
                    "n_redundant": 0,
                    "n_total_points": 1,
                    "total_penalty_ball_volume": float(_NEEDLE_VOL_PENALTY),
                }
            if verbose:
                res = results[idx]
                print(
                    f"  [sub-run {done}/{len(tasks)}] "
                    f"{tasks[idx]['full_name']}  "
                    f"region_dist={res['avg_region_dist']:.5f}",
                )

    return [r for r in results if r is not None]


def _aggregate_mobo_objectives(results: List[Dict]) -> Tuple[float, float, float]:
    dists = [
        r["avg_region_dist"]
        for r in results
        if np.isfinite(r["avg_region_dist"])
    ]
    avg_dist = float(np.mean(dists)) if dists else 100.0

    dup_fracs = [
        r["n_redundant"] / max(r["n_total_points"], 1)
        for r in results
    ]
    avg_dup = float(np.mean(dup_fracs)) if dup_fracs else 1.0

    vols = [
        float(v)
        for r in results
        for v in [r.get("total_penalty_ball_volume")]
        if v is not None and np.isfinite(v)
    ]
    avg_vol = float(np.mean(vols)) if vols else float(_NEEDLE_VOL_PENALTY)
    return avg_dist, avg_dup, avg_vol


def _evaluate_config_v2_parallel(
    config: Dict,
    *,
    fixed_kw: Dict,
    benchmark_workers: int,
    threads_per_worker: Optional[int],
) -> Tuple[float, float, float]:
    merged = {**fixed_kw, **config}
    merged.pop("benchmark_workers", None)
    merged.pop("threads_per_worker", None)
    results = run_zombi_test_v2_parallel(
        **merged,
        benchmark_workers=benchmark_workers,
        threads_per_worker=threads_per_worker,
    )
    return _aggregate_mobo_objectives(results)


def _run_batch_sequential_v2_parallel(
    configs: List[Dict],
    x_unit_batch: List[np.ndarray],
    fixed_kw: Dict,
    label: str,
    benchmark_workers: int,
    threads_per_worker: Optional[int],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
    dev = fixed_kw.get("device", "cpu")
    print(
        f"\n  Evaluating {len(configs)} config(s) sequentially on {dev} "
        f"(benchmark_workers={benchmark_workers}) …",
    )

    new_X: List[torch.Tensor] = []
    new_Y: List[torch.Tensor] = []
    for i, cfg in enumerate(configs):
        try:
            dist, dup, vol = _evaluate_config_v2_parallel(
                cfg,
                fixed_kw=fixed_kw,
                benchmark_workers=benchmark_workers,
                threads_per_worker=threads_per_worker,
            )
        except Exception as exc:
            warnings.warn(f"  [{label} {i + 1}/{len(configs)}] failed: {exc}", stacklevel=2)
            dist, dup, vol = (1.0, 1.0, float(_NEEDLE_VOL_PENALTY))
        print(
            f"  [{label} {i + 1}/{len(configs)}] "
            f"region_dist={dist:.5f}  dup_frac={dup:.4f}  "
            f"total_penalty_vol={vol:.6e}  config={cfg}",
        )
        new_X.append(torch.tensor(x_unit_batch[i], dtype=torch.float64))
        new_Y.append(torch.tensor([dist, dup, vol], dtype=torch.float64))
        if dev == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return new_X, new_Y, configs


def _mobo_worker_v2_parallel(task: Tuple[Dict, Dict, int, Optional[int]]) -> Tuple[float, float, float]:
    config, fixed_kw, benchmark_workers, threads_per_worker = task
    return _evaluate_config_v2_parallel(
        config,
        fixed_kw=fixed_kw,
        benchmark_workers=benchmark_workers,
        threads_per_worker=threads_per_worker,
    )


def _run_batch_mobo_v2_parallel(
    configs: List[Dict],
    x_unit_batch: List[np.ndarray],
    fixed_kw: Dict,
    label: str,
    mobo_workers: int,
    benchmark_workers: int,
    threads_per_worker: Optional[int],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
    mobo_workers = max(1, int(mobo_workers))
    ncfg = len(configs)
    if mobo_workers == 1 or ncfg == 1:
        return _run_batch_sequential_v2_parallel(
            configs,
            x_unit_batch,
            fixed_kw,
            label,
            benchmark_workers,
            threads_per_worker,
        )

    workers = min(mobo_workers, ncfg)
    tasks = [
        (cfg, fixed_kw, benchmark_workers, threads_per_worker)
        for cfg in configs
    ]
    print(
        f"\n  Evaluating {ncfg} config(s) with ProcessPoolExecutor "
        f"(mobo_workers={workers}, benchmark_workers={benchmark_workers}) …",
    )

    mp_ctx = multiprocessing.get_context("spawn")
    results_xyz: List[Optional[Tuple[float, float, float]]] = [None] * len(tasks)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_ctx,
    ) as pool:
        futures = {pool.submit(_mobo_worker_v2_parallel, t): i for i, t in enumerate(tasks)}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                results_xyz[idx] = fut.result()
            except Exception as exc:
                warnings.warn(f"  [{label} worker {idx}] failed: {exc}", stacklevel=2)
                results_xyz[idx] = (1.0, 1.0, float(_NEEDLE_VOL_PENALTY))

    new_X: List[torch.Tensor] = []
    new_Y: List[torch.Tensor] = []
    for i, triple in enumerate(results_xyz):
        dist, dup, vol = triple  # type: ignore[misc]
        print(
            f"  [{label} {i + 1}/{len(tasks)}] "
            f"region_dist={dist:.5f}  dup_frac={dup:.4f}  "
            f"total_penalty_vol={vol:.6e}  config={configs[i]}",
        )
        new_X.append(torch.tensor(x_unit_batch[i], dtype=torch.float64))
        new_Y.append(torch.tensor([dist, dup, vol], dtype=torch.float64))

    return new_X, new_Y, configs


def mobo_tune_zombi_v2_parallel(
    *,
    benchmark_workers: int = 1,
    threads_per_worker: Optional[int] = None,
    **kwargs: Any,
) -> Dict:
    """
    Same as ``mobo_tune_zombi_v2`` with parallel benchmark sub-runs per eval.

    Patches the V2 MOBO batch helpers for the duration of the run.
    """
    import scripts.run_zombi_test_v2 as v2

    benchmark_workers = max(1, int(benchmark_workers))
    cpu_count = os.cpu_count() or 1
    if threads_per_worker is None and benchmark_workers > 1:
        threads_per_worker = max(1, cpu_count // benchmark_workers)

    orig_batch = v2._run_batch_mobo_v2
    orig_worker = v2._mobo_worker_v2
    orig_eval = v2._evaluate_config_v2

    def _patched_eval(config: Dict, *, fixed_kw: Dict) -> Tuple[float, float, float]:
        return _evaluate_config_v2_parallel(
            config,
            fixed_kw=fixed_kw,
            benchmark_workers=benchmark_workers,
            threads_per_worker=threads_per_worker,
        )

    def _patched_worker(task: Tuple[Dict, Dict]) -> Tuple[float, float, float]:
        config, fixed_kw = task
        return _patched_eval(config, fixed_kw=fixed_kw)

    def _patched_batch(
        configs: List[Dict],
        x_unit_batch: List[np.ndarray],
        fixed_kw: Dict,
        label: str,
        mobo_workers: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
        return _run_batch_mobo_v2_parallel(
            configs,
            x_unit_batch,
            fixed_kw,
            label,
            mobo_workers,
            benchmark_workers,
            threads_per_worker,
        )

    v2._evaluate_config_v2 = _patched_eval
    v2._mobo_worker_v2 = _patched_worker
    v2._run_batch_mobo_v2 = _patched_batch

    n_parallel = kwargs.get("n_parallel", kwargs.get("batch", 4))
    n_initial = kwargs.get("n_initial", kwargs.get("mobo_init", 8))
    n_mobo_iterations = kwargs.get("n_mobo_iterations", kwargs.get("mobo_iters", 20))
    mobo_workers = kwargs.get("mobo_workers", kwargs.get("workers", 4))
    n_total = int(np.ceil(n_initial / n_parallel) * n_parallel) + n_mobo_iterations * n_parallel

    print("=" * 70)
    print("MOBO V2 PARALLEL  (benchmark_workers per eval)")
    print(
        f"  {n_total} total evals  |  mobo_workers={mobo_workers}  |  "
        f"benchmark_workers={benchmark_workers}  |  "
        f"threads/worker={threads_per_worker}",
    )
    print("=" * 70)

    try:
        return mobo_tune_zombi_v2(**kwargs)
    finally:
        v2._evaluate_config_v2 = orig_eval
        v2._mobo_worker_v2 = orig_worker
        v2._run_batch_mobo_v2 = orig_batch


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ZoMBI-Hop V2 with parallel benchmark sub-runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mobo", action="store_true", help="Run MOBO hyperparameter tuning.")
    ap.add_argument(
        "--regions",
        type=str,
        default=None,
        help="Path to max_min_regions.json from interactive_maxima_selector.py.",
    )
    ap.add_argument("--mobo-init", type=int, default=8)
    ap.add_argument("--mobo-iters", type=int, default=20)
    ap.add_argument(
        "--batch",
        "--parallel",
        dest="batch",
        type=int,
        default=4,
        help="MOBO: acquisition batch size (q).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="MOBO: parallel processes per batch (use 1 when benchmark-workers > 1).",
    )
    ap.add_argument(
        "--benchmark-workers",
        type=int,
        default=1,
        help="Parallel ZoMBI sub-runs inside each MOBO eval (objective × noise).",
    )
    ap.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help="BLAS/torch threads per benchmark worker (default: cpus // benchmark-workers).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="ZoMBI device: cuda or cpu. Omit to auto-select.",
    )
    ap.add_argument("--results-json", type=str, default="hyperparam_results_v2_parallel.json")
    args = ap.parse_args()

    if args.device is not None and args.device not in ("cuda", "cpu"):
        ap.error("--device must be cuda or cpu")

    dev_kw: Dict[str, Optional[str]] = {}
    if args.device is not None:
        dev_kw["device"] = args.device

    if args.mobo:
        mobo_tune_zombi_v2_parallel(
            regions_path=args.regions,
            n_initial=args.mobo_init,
            n_mobo_iterations=args.mobo_iters,
            n_parallel=args.batch,
            mobo_workers=args.workers,
            benchmark_workers=args.benchmark_workers,
            threads_per_worker=args.threads_per_worker,
            csv_path=CSV_PEROVSKITE_PATH,
            verbose_zombi=False,
            results_json=args.results_json,
            **dev_kw,
        )
    else:
        results = run_zombi_test_v2_parallel(
            regions_path=args.regions,
            benchmark_workers=args.benchmark_workers,
            threads_per_worker=args.threads_per_worker,
            epsilon_frac=0.2,
            max_zooms=3,
            max_iterations=10,
            n_restarts=30,
            raw_samples=500,
            convergence_pi_threshold=0.01,
            n_consecutive_converged=2,
            ucb_beta=0.1,
            nat_grad_step=0.02,
            nat_grad_max_steps=50,
            num_points_per_line=100,
            num_lines=30,
            num_init_data=4,
            rf_global_samples=10_000_000,
            csv_path=CSV_PEROVSKITE_PATH,
            verbose=True,
            **dev_kw,
        )
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY (V2 parallel)")
        print("=" * 70)
        header = (
            f"{'Objective':<50}  {'Ndl':>4}  {'RegDist':>9}"
            f"  {'Tot':>6}  {'Redund':>6}  {'Wst%':>5}"
        )
        print(header)
        print("-" * len(header))
        for res in results:
            n = len(res["needles"])
            d = res["avg_region_dist"]
            d_s = f"{d:.5f}" if np.isfinite(d) else "    n/a"
            nt = res["n_total_points"]
            nr = res["n_redundant"]
            wp = 100.0 * nr / nt if nt > 0 else 0.0
            print(
                f"  {res['name']:<48}  {n:>4}  {d_s:>9}"
                f"  {nt:>6}  {nr:>6}  {wp:>4.1f}%",
            )
        print("=" * 70)


if __name__ == "__main__":
    main()
