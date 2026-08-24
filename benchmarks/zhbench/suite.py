"""Run a config grid and aggregate.

A suite config is YAML:

    name: smoke
    protocol: {n_samples: 240, batch_size: 24, input_noise: empirical}
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

from .protocol import Protocol
from .runner import run_one

_CURVE_KEYS = ("reached_curve_t", "reached_curve_ratio")


def _obj_label(spec: dict) -> str:
    parts = [str(spec.get("kind"))]
    for k in ("dim", "n_optima", "landscape", "basin_width"):
        if spec.get(k) is not None:
            parts.append(f"{k}={spec[k]}")
    return " ".join(parts)


def run_suite(config: dict, out_root: str, resume: bool = True) -> str:
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
                else:
                    print(f"[run ] {cell}", flush=True)
                    try:
                        res = run_one(obj_spec, opt_spec, seed,
                                      Protocol(**protocol_kwargs), out_dir=run_dir,
                                      value_tol=value_tol)
                    except Exception as exc:
                        traceback.print_exc()
                        res = {"objective": _obj_label(obj_spec),
                               "optimizer": opt_spec["name"], "seed": seed,
                               "error": f"{type(exc).__name__}: {exc}"}
                curves[cell] = {k: res.get(k) for k in _CURVE_KEYS}
                rows.append({k: v for k, v in res.items() if k not in _CURVE_KEYS})

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
    args = ap.parse_args(argv)

    path = args.config
    if not os.path.exists(path):
        path = os.path.join(here, "configs", f"{args.config}.yaml")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    run_suite(cfg, args.out, resume=not args.no_resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
