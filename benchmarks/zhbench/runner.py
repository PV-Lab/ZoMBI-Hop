"""Run one (objective, optimizer, seed) and write its artifacts.

Two loops, because there are two kinds of method:

  * Batch methods are driven from here: ``suggest(q)`` -> realize + measure ->
    ``observe``. One iteration per decision, ``protocol.n_decisions`` of them.
  * ZoMBI-Hop drives itself. It is handed the same ``ObjectiveRun`` and stops on
    the same ``BudgetExhausted``, so the two paths are charged identically.

Both start from the identical initial design for a given seed.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict

import numpy as np

from . import metrics as M
from . import objectives as O
from . import optimizers as OPT
from .protocol import BudgetExhausted, ObjectiveRun, Protocol, gen_init_design
from .seeding import set_global_seed


def git_state() -> dict:
    def _run(*args):
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL,
                                           text=True).strip()
        except Exception:
            return ""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return {
        "commit": _run("git", "-C", root, "rev-parse", "HEAD"),
        "branch": _run("git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_run("git", "-C", root, "status", "--porcelain")),
    }


def _write_csv(path: str, header: list[str], rows) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def run_one(objective_spec: dict, optimizer_spec: dict, seed: int,
            protocol: Protocol, out_dir: str | None = None,
            value_tol: float = 0.25) -> dict:
    set_global_seed(seed)
    objective = O.build(objective_spec)
    optimizer = OPT.build(optimizer_spec)

    run = ObjectiveRun(fn=objective.fn, dim=objective.dim, protocol=protocol,
                       seed=seed, maximize=objective.maximize)

    t0 = time.time()
    error = ""
    try:
        if getattr(optimizer, "self_driving", False):
            optimizer.run(objective, run, protocol, seed)
        else:
            X_req0, X_act0, y0 = gen_init_design(run, protocol, seed)
            optimizer.initialize(X_act0, y0, objective, seed)
            for _ in range(protocol.n_decisions):
                X_req = np.atleast_2d(np.asarray(optimizer.suggest(protocol.batch_size),
                                                 dtype=float))
                X_act, y = run.evaluate_batch(X_req)
                optimizer.observe(X_act, y)
    except BudgetExhausted:
        pass
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        wall = time.time() - t0

    declared = optimizer.declared_optima()
    result = M.compute_all(run, objective, declared=declared, wall_s=wall,
                           value_tol=value_tol)
    result.update({
        "objective": objective.name,
        "optimizer": optimizer.name,
        "seed": int(seed),
        "dim": int(objective.dim),
        "error": error,
    })

    if out_dir:
        _write_run_dir(out_dir, run, objective, optimizer, protocol, result,
                       objective_spec, optimizer_spec, seed)
    return result


def _write_run_dir(out_dir, run, objective, optimizer, protocol, result,
                   objective_spec, optimizer_spec, seed) -> None:
    os.makedirs(out_dir, exist_ok=True)
    h = run.stacked()
    d = objective.dim

    _write_csv(
        os.path.join(out_dir, "points.csv"),
        (["sample_idx", "batch_idx"]
         + [f"x_req_{i}" for i in range(d)]
         + [f"x_act_{i}" for i in range(d)]
         + ["y_observed", "y_true"]),
        ([i, int(h["batch"][i])] + list(h["X_requested"][i]) + list(h["X_actual"][i])
         + [h["y_observed"][i], h["y_true"][i]]
         for i in range(h["X_actual"].shape[0])),
    )

    declared = optimizer.declared_optima()
    if declared is not None and len(declared):
        _write_csv(os.path.join(out_dir, "declared_optima.csv"),
                   [f"x_{i}" for i in range(d)], (list(p) for p in declared))
    log = getattr(optimizer, "needle_log", None)
    if log:
        keys = list(log[0])
        _write_csv(os.path.join(out_dir, "needles.csv"), keys,
                   ([row[k] for k in keys] for row in log))

    T, tv = M.merge_true_optima(objective.true_optima, objective.true_values)
    _write_csv(os.path.join(out_dir, "true_optima.csv"),
               [f"x_{i}" for i in range(d)] + ["value"],
               (list(T[i]) + [tv[i]] for i in range(T.shape[0])))

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=float)
    with open(os.path.join(out_dir, "config_resolved.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "objective_spec": objective_spec,
            "objective_meta": objective.meta,
            "optimizer_spec": optimizer_spec,
            "optimizer_state": optimizer.state(),
            "protocol": protocol.to_dict(),
            "seed": int(seed),
            "git": git_state(),
        }, fh, indent=2, default=str)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="run one benchmark trial")
    ap.add_argument("--objective", required=True,
                    help='JSON, e.g. \'{"kind":"ensemble","dim":3,"n_optima":5}\'')
    ap.add_argument("--optimizer", required=True, help='JSON, e.g. \'{"name":"random"}\'')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    res = run_one(json.loads(args.objective), json.loads(args.optimizer), args.seed,
                  Protocol(n_samples=args.n_samples, batch_size=args.batch),
                  out_dir=args.out)
    print(json.dumps({k: v for k, v in res.items()
                      if not k.startswith("reached_curve")}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
