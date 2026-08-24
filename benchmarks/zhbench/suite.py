"""Run a config grid and aggregate.

A suite config is YAML:

    name: smoke
    protocol: {n_samples: 240, batch_size: 24, noise: hardware}
    seeds: [0]
    objectives:
      - {kind: ensemble, dim: 3, n_optima: 5, landscape: 0}
    optimizers:
      - {name: random}
      - {name: gp_qucb}
      - {name: zombihop, hparams: smoke}

Every (objective, optimizer, seed) cell is one run directory. ``aggregate.csv``
gets one row per run; ``summary.md`` reports mean +/- std per optimizer x objective,
plus lift over random, which is the number to read across dimensions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
from datetime import datetime

import numpy as np
import yaml

from ._repo import REPO_ROOT as _REPO_ROOT
from .protocol import Protocol
from .runner import run_one

_CURVE_KEYS = ("reached_curve_t", "reached_curve_ratio",
               "pr_curve_k", "pr_curve_peak_ratio", "pr_curve_precision",
               "needles_curve_t", "needles_curve_n", "by_n")


def _obj_label(spec: dict) -> str:
    parts = [str(spec.get("kind"))]
    for k in ("dim", "n_optima", "landscape", "basin_width"):
        if spec.get(k) is not None:
            parts.append(f"{k}={spec[k]}")
    return " ".join(parts)


def _configure_worker_env() -> None:
    """Environment a worker process needs, set before anything heavy is imported.

    **Matplotlib backend.** This is the one that actually mattered.
    ``optimize/run_mobo.py`` imports ``pyplot`` at module scope and never calls
    ``matplotlib.use``, and matplotlib's default here resolves to ``tkagg``.
    Initialising Tk inside a spawned worker crashes the process outright, which
    surfaces as ``BrokenProcessPool`` with no traceback -- exactly what killed the
    first two s1_real launches. Nothing in a benchmark run draws anything, so force
    ``Agg``.

    **Thread pinning.** Six torch processes each defaulting to one OpenMP thread
    per core is 144 threads on a 24-core box. The work is already parallel at the
    cell level, so each cell wants exactly one thread.

    ``setdefault`` throughout, so an explicit setting from the caller wins.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    # Two OpenMP runtimes end up in one process here: sklearn ships
    # .libs/vcomp140.dll (Microsoft) and torch ships lib/libiomp5md.dll (Intel),
    # and numpy and scipy each bundle a separate OpenBLAS on top. A GP cell touches
    # sklearn (the surrogate objective) and torch (BoTorch) in the same process,
    # which is the documented recipe for an access violation -- observed as exit
    # 0xC0000005 on real_gp_dim3/gp_qucb. This tells Intel's runtime to tolerate the
    # duplicate. It is a mitigation, not a cure: the real fix is one OpenMP runtime
    # in the environment, and cells stay isolated in subprocesses so a residual
    # crash costs one cell.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")


def _run_cell_inprocess(job: dict) -> tuple[str, dict]:
    """Run a cell in this process. Used for ``--workers 1`` and as the fallback."""
    _configure_worker_env()
    try:
        res = run_one(job["objective"], job["optimizer"], job["seed"],
                      Protocol(**job["protocol"]), out_dir=job["run_dir"],
                      value_tol=job["value_tol"])
    except Exception as exc:
        traceback.print_exc()
        res = {"objective": _obj_label(job["objective"]),
               "optimizer": job["optimizer"]["name"], "seed": job["seed"],
               "error": f"{type(exc).__name__}: {exc}"}
    return job["cell"], res


def _run_cell(job: dict) -> tuple[str, dict]:
    """Run one cell in its own OS process, via the runner CLI.

    A ``ProcessPoolExecutor`` was the obvious choice and the wrong one: a worker
    that dies takes the entire pool with it, so one bad cell out of 180 discards
    every cell still in flight. Two s1_real launches died that way with a bare
    ``BrokenProcessPool`` and no traceback to work from.

    A plain subprocess per cell is fully isolated. A crash, a segfault or a hang
    becomes a non-zero exit or a timeout for *that* cell, recorded as its error and
    nothing else. The cell writes its own ``metrics.json``, so the parent just
    reads it back. Threads (not processes) drive these, since they only wait on
    subprocesses.
    """
    import subprocess
    import sys

    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env.setdefault(var, "1")

    cmd = [sys.executable, "-m", "benchmarks.zhbench.runner",
           "--objective", json.dumps(job["objective"]),
           "--optimizer", json.dumps(job["optimizer"]),
           "--seed", str(job["seed"]),
           "--protocol", json.dumps(job["protocol"]),
           "--value-tol", str(job["value_tol"]),
           "--out", job["run_dir"], "--quiet"]
    err = ""
    try:
        # encoding is explicit: text=True decodes with the locale codec, which is
        # GBK on this machine and dies on the child's UTF-8 output.
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, capture_output=True,
                              encoding="utf-8", errors="replace",
                              timeout=job.get("timeout_s", 7200))
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            err = f"exit {proc.returncode}: " + " | ".join(tail)
    except subprocess.TimeoutExpired:
        err = f"timeout after {job.get('timeout_s', 7200)}s"

    done = os.path.join(job["run_dir"], "metrics.json")
    if os.path.exists(done):
        with open(done, encoding="utf-8") as fh:
            return job["cell"], json.load(fh)
    return job["cell"], {"objective": _obj_label(job["objective"]),
                         "optimizer": job["optimizer"]["name"],
                         "seed": job["seed"],
                         "error": err or "no metrics.json written"}


def run_suite(config: dict, out_root: str, resume: bool = True,
              workers: int = 1) -> str:
    name = config.get("name", "suite")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = os.path.join(out_root, f"{name}_{stamp}")
    os.makedirs(suite_dir, exist_ok=True)
    with open(os.path.join(suite_dir, "config.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    protocol_kwargs = dict(config.get("protocol", {}))
    seeds = config.get("seeds", [0])
    value_tol = float(config.get("value_tol", 0.25))

    rows: list[dict] = []
    curves: dict[str, list] = {}
    jobs: list[dict] = []
    for obj_spec in config["objectives"]:
        for opt_spec in config["optimizers"]:
            for seed in seeds:
                cell = (f"{_obj_label(obj_spec).replace(' ', '_').replace('=', '')}"
                        f"__{opt_spec['name']}__s{seed}")
                run_dir = os.path.join(suite_dir, cell)
                done = os.path.join(run_dir, "metrics.json")
                if resume and os.path.exists(done):
                    with open(done, encoding="utf-8") as fh:
                        res = json.load(fh)
                    print(f"[skip] {cell}")
                    curves[cell] = {"objective": res.get("objective"),
                                "optimizer": res.get("optimizer"),
                                **{k: res.get(k) for k in _CURVE_KEYS}}
                    rows.append({k: v for k, v in res.items() if k not in _CURVE_KEYS})
                    continue
                jobs.append({"cell": cell, "run_dir": run_dir, "objective": obj_spec,
                             "optimizer": opt_spec, "seed": seed,
                             "protocol": protocol_kwargs, "value_tol": value_tol})

    def _absorb(cell: str, res: dict) -> None:
        curves[cell] = {"objective": res.get("objective"),
                        "optimizer": res.get("optimizer"),
                        **{k: res.get(k) for k in _CURVE_KEYS}}
        rows.append({k: v for k, v in res.items() if k not in _CURVE_KEYS})

    done_cells: set[str] = set()
    if workers > 1 and len(jobs) > 1:
        # Threads, because each one only waits on an isolated subprocess.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"running {len(jobs)} cells on {workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_cell, j): j["cell"] for j in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                cell, res = fut.result()
                done_cells.add(cell)
                flag = "  !! " + res["error"][:80] if res.get("error") else ""
                print(f"[{i}/{len(jobs)}] {cell}{flag}", flush=True)
                _absorb(cell, res)
    else:
        for j in jobs:
            print(f"[run ] {j['cell']}", flush=True)
            cell, res = _run_cell_inprocess(j)
            done_cells.add(cell)
            _absorb(cell, res)

    _write_aggregate(suite_dir, rows)
    with open(os.path.join(suite_dir, "curves.json"), "w", encoding="utf-8") as fh:
        json.dump(curves, fh)
    _write_summary(suite_dir, rows, note=config.get("note"))
    print(f"\nsuite written to {suite_dir}")
    return suite_dir


def _write_aggregate(suite_dir: str, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(suite_dir, "aggregate.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


_HEADLINE = ["peak_ratio", "precision", "f1", "dist_to_needles", "n_declared",
             "reached_ratio_final", "t_first_optimum", "best_y", "input_cost",
             "dup_fraction", "wall_s"]


def _write_summary(suite_dir: str, rows: list[dict], note: str | None = None) -> None:
    objs = sorted({r.get("objective", "?") for r in rows})
    lines = ["# Suite summary", ""]
    if note:
        lines += [f"> **{note}**", ""]
    lines += [
             "`peak_ratio` = fraction of true optima matched one-to-one by the "
             "method's DECLARED optima (post-hoc extracted for methods that declare "
             "none). `precision` = fraction of declared optima that are real. "
             "`input_cost` = SnAKe composition distance travelled -- the physical "
             "price of a scattered batch. `lift` = peak_ratio / peak_ratio(random), "
             "the quantity that stays comparable across dimensions.", ""]
    for obj in objs:
        sub = [r for r in rows if r.get("objective") == obj]
        opts = sorted({r["optimizer"] for r in sub})
        rand = [r for r in sub if r["optimizer"] == "random" and "peak_ratio" in r]
        rand_pr = float(np.mean([r["peak_ratio"] for r in rand])) if rand else float("nan")
        lines += [f"## {obj}", "",
                  "| optimizer | " + " | ".join(_HEADLINE) + " | lift |",
                  "|" + "---|" * (len(_HEADLINE) + 2)]
        for opt in opts:
            cells = [r for r in sub if r["optimizer"] == opt]
            vals = []
            for key in _HEADLINE:
                v = [r[key] for r in cells if isinstance(r.get(key), (int, float))
                     and np.isfinite(r[key])]
                vals.append(f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "-")
            pr = [r["peak_ratio"] for r in cells if isinstance(r.get("peak_ratio"), (int, float))]
            lift = (f"{np.mean(pr) / rand_pr:.2f}x"
                    if pr and np.isfinite(rand_pr) and rand_pr > 0 else "-")
            lines.append(f"| {opt} | " + " | ".join(vals) + f" | {lift} |")
        errs = [r for r in sub if r.get("error")]
        if errs:
            lines += ["", "**errors:**"] + [f"- `{r['optimizer']}` seed {r.get('seed')}: "
                                            f"{r['error']}" for r in errs]
        lines.append("")
    with open(os.path.join(suite_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="run a benchmark suite")
    ap.add_argument("config", help="path to a suite YAML (or a name under configs/)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(here), "runs"))
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes; each cell is independent")
    args = ap.parse_args(argv)

    path = args.config
    if not os.path.exists(path):
        path = os.path.join(here, "configs", f"{args.config}.yaml")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    run_suite(cfg, args.out, resume=not args.no_resume, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
