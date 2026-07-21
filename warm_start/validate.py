"""
Pre-flight check for the warm-start comparison.

Runs one repetition of both arms with cheap acquisition settings and a small
budget, then asserts the invariants that would *silently* invalidate the real
experiment rather than crash it — the failure mode that matters, because a
comparison that runs to completion with a broken control looks exactly like a
successful one.

Checked:
  * equal budget — both arms measure the same number of compositions, so neither
    is credited with experiments it was not allowed to run;
  * initial designs — warm gets 4 lines (96 pts), cold gets 2 (48 pts);
  * the seed prior is actually in force (``seed_var > real_var > 0``) and reaches
    the GP, rather than being silently dropped;
  * both arms saw the same landscape (the pairing the analysis relies on);
  * every per-trial artifact is present.

Run via ``warm_start/scripts/warm_start_validate.sbatch`` — it needs a GPU node.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from warm_start import compare as C
from warm_start import trial as T
from warm_start.warm_start import POINTS_PER_LINE, n_lines

#: Cheap stand-ins for the acquisition-optimisation hyperparameters, so the check
#: costs minutes rather than hours.  Everything that shapes the *comparison*
#: (paring, penalty, zoom, convergence) is left at the reference values.
FAST_OVERRIDES = dict(n_restarts=5, raw=10, nat_grad_max_steps=5, max_iterations=2)

VALIDATE_BUDGET = 240

REQUIRED_ARTIFACTS = [
    "points.csv", "needles.csv", "metrics_over_time.csv", "metrics.json",
    "convergence.png", "dist_from_centre.png", "line_length_hist.png",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run-dir", default="warm_start/runs/validate")
    p.add_argument("--budget", type=int, default=VALIDATE_BUDGET)
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    hp = dict(T.REFERENCE_HPARAMS)
    hp.update(FAST_OVERRIDES)

    landscape, ens_cfg = C.landscape_for_rep(3, 0, 0)
    results = {}
    for arm in T.ARMS:
        print(f"\n=== {arm} ===", flush=True)
        results[arm] = T.run_trial(arm, 0, landscape, os.path.join(run_dir, arm),
                                   ensemble_config=ens_cfg, hparams=hp,
                                   budget=args.budget)

    failures: list[str] = []

    def check(ok: bool, msg: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            failures.append(msg)

    cold, warm = results["cold"], results["warm"]
    print("\n=== invariants ===")

    check(cold["n_init"] == T.rm.N_INIT_LINES * POINTS_PER_LINE,
          f"cold init is {T.rm.N_INIT_LINES} lines "
          f"({T.rm.N_INIT_LINES * POINTS_PER_LINE} pts): got {cold['n_init']}")
    expected_warm_init = n_lines(3) * POINTS_PER_LINE
    check(warm["n_init"] == expected_warm_init,
          f"warm init is {n_lines(3)} lines ({expected_warm_init} pts): "
          f"got {warm['n_init']}")

    check(cold["n_points"] == warm["n_points"],
          f"equal budget spent: cold={cold['n_points']} warm={warm['n_points']}")
    check(max(cold["n_points"], warm["n_points"]) <= args.budget,
          f"neither arm exceeds the {args.budget}-point budget")

    check(cold["seed_var"] is None and cold["real_var"] is None,
          "cold arm carries no seed prior")
    check(warm["seed_var"] is not None and warm["real_var"] is not None
          and warm["seed_var"] > warm["real_var"] > 0,
          f"warm seed prior active and inflated: seed_var={warm['seed_var']} "
          f"real_var={warm['real_var']}")

    check(abs(cold["y_std"] - warm["y_std"]) < 1e-12,
          "both arms saw the same landscape (identical y_std)")

    for arm in T.ARMS:
        missing = [a for a in REQUIRED_ARTIFACTS
                   if not os.path.isfile(os.path.join(run_dir, arm, a))]
        check(not missing, f"{arm} artifacts complete"
                           + (f" (missing: {', '.join(missing)})" if missing else ""))

    print()
    if failures:
        print(f"VALIDATION FAILED — {len(failures)} check(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
