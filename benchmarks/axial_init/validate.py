"""
benchmarks/axial_init/validate.py
=================================
Does the axial-trace initial design (src/utils/axial_init.py) beat the random seed
lines it replaced in scripts/run_zombi_main.py?

Three seed designs, every printed line pushed through the hardware print model
(optimize/composition_prediction.physics_simulate_line) with 4.5 % multiplicative
output noise, exactly as the synthetic campaigns do:

  legacy  2 random lines in the search box (the old hardware init: two objective
          calls, main rail kept)
  random  as many random lines as the axial design has traces (same cost as axial;
          isolates the design from the count)
  axial   the axial design, n traces rounded down to even

Configurations: 3d / 4d / 6d / 10d on the full simplex, and 6d_cap = the box the
6-dim hardware campaigns run (optimizer dim 3, hardware dim 4, capped at 0.3).

Usage
-----
    python -m benchmarks.axial_init.validate coverage  [--seeds 40]
    python -m benchmarks.axial_init.validate surrogate [--seeds 20]
    python -m benchmarks.axial_init.validate optimize  --config 6d_cap --seeds 5
            [--budget 60] [--arms legacy,random,axial] [--out benchmarks/axial_init/results]
    python -m benchmarks.axial_init.validate summarize [--out benchmarks/axial_init/results]

``coverage`` and ``surrogate`` print tables and write JSON under --out;
``optimize`` writes one JSON per (config, arm, seed) so runs can be split across
processes and resumed; ``summarize`` aggregates them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from benchmarks.ablations._paths import ensure_paths

ensure_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from optimize.composition_prediction import physics_simulate_line  # noqa: E402
from src.utils.axial_init import (  # noqa: E402
    axial_design, feasible_box, point_coverage, uniform_feasible,
)

NUM_EXPERIMENTS = 24
OUTPUT_NOISE_FRAC = 0.045
INPUT_NOISE = 0.128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
OUT_DEFAULT = Path(__file__).resolve().parent / "results"

CONFIGS = {
    "3d": dict(d=3, hi=None),
    "4d": dict(d=4, hi=None),
    "6d": dict(d=6, hi=None),
    "6d_cap": dict(d=6, hi=[1, 1, 1, 0.3, 1, 1]),
    "10d": dict(d=10, hi=None),
}
ARMS = ("legacy", "random", "axial")


def box(cfg):
    d = cfg["d"]
    lo = np.zeros(d)
    hi = np.ones(d) if cfg["hi"] is None else np.asarray(cfg["hi"], float)
    return lo, hi


# ── seed designs ─────────────────────────────────────────────────────────────

def _chord(s, d, lo, hi):
    t_min, t_max = -np.inf, np.inf
    for si, di, l, h in zip(s, d, lo, hi):
        if abs(di) < 1e-15:
            continue
        a, b = (l - si) / di, (h - si) / di
        t_min, t_max = max(t_min, min(a, b)), min(t_max, max(a, b))
    return t_min, t_max


def random_box_lines(n_lines, lo, hi, rng):
    """The old hardware seed lines: a uniform point in the box, a uniform zero-sum
    direction, extended to the box boundary both ways."""
    d = len(lo)
    pts = uniform_feasible(d, n_lines, lo, hi, seed=int(rng.integers(1 << 31)))
    out = []
    for x0 in pts:
        v = rng.normal(size=d)
        v -= v.mean()
        v /= np.linalg.norm(v)
        t0, t1 = _chord(x0, v, lo, hi)
        out.append((x0 + t0 * v, x0 + t1 * v))
    return out


def seed_lines(arm, cfg, rng):
    lo, hi = box(cfg)
    des = axial_design(cfg["d"], lower=lo, upper=hi)
    if arm in ("axial", "axial_p2f"):
        # hardware prints face -> pure (run_zombi_main.INIT_PRINT_PURE_LAST);
        # axial_p2f prints pure -> face, to show what the print direction buys
        lines = [(t.end, t.start) if arm == "axial" else (t.start, t.end) for t in des.traces]
        order = rng.permutation(len(lines))
        return [lines[i] for i in order]
    if arm == "random":
        return random_box_lines(len(des.traces), lo, hi, rng)
    if arm == "legacy":
        return random_box_lines(2, lo, hi, rng)
    raise ValueError(arm)


def print_line(left, right):
    pts = physics_simulate_line(np.asarray(left, float), np.asarray(right, float),
                                num_points=NUM_EXPERIMENTS, device="cpu", dtype=DTYPE)
    return np.asarray(pts.detach().cpu().numpy() if torch.is_tensor(pts) else pts, float)


def measure(fn, X, rng):
    y = np.array([fn(x) for x in X], float)
    return y + rng.normal(size=len(y)) * OUTPUT_NOISE_FRAC * np.abs(y)


# ── part A: coverage ─────────────────────────────────────────────────────────

def coverage(args):
    rows = {}
    for name, cfg in CONFIGS.items():
        lo, hi = box(cfg)
        lo_e, hi_e = feasible_box(lo, hi)
        P = uniform_feasible(cfg["d"], 20_000, lo, hi, seed=123)
        for arm in ARMS + ("axial_p2f",):
            n_seeds = 1 if arm.startswith("axial") else args.seeds
            ms, ps, spans, n_lines = [], [], [], 0
            for s in range(n_seeds):
                rng = np.random.default_rng(1000 + s)
                lines = seed_lines(arm, cfg, rng)
                n_lines = len(lines)
                X = np.vstack([print_line(a, b) for a, b in lines])
                m, p = point_coverage(X, P)
                ms.append(m); ps.append(p)
                span = (X.max(0) - X.min(0)) / np.maximum(hi_e - lo_e, 1e-12)
                spans.append(float(span.min()))
            rows[(name, arm)] = dict(
                config=name, arm=arm, n_lines=n_lines, n_points=n_lines * NUM_EXPERIMENTS,
                cov_mean=float(np.mean(ms)), cov_mean_sd=float(np.std(ms)),
                cov_p95=float(np.mean(ps)), cov_p95_sd=float(np.std(ps)),
                worst_span=float(np.mean(spans)), worst_span_sd=float(np.std(spans)))
    print("\nA. Space coverage of the MEASURED seed points (print model applied)")
    print("   coverage = distance from a uniform random feasible composition to the")
    print("   nearest measured seed point (lower is better); worst span = smallest")
    print("   fraction of any component's feasible range the seed data spans.\n")
    print(f"{'config':8s} {'arm':7s} {'lines':>5s} {'cov mean':>15s} {'cov p95':>15s} {'worst span':>15s}")
    for r in rows.values():
        print(f"{r['config']:8s} {r['arm']:7s} {r['n_lines']:5d} "
              f"{r['cov_mean']:7.3f} ±{r['cov_mean_sd']:5.3f}  {r['cov_p95']:7.3f} ±{r['cov_p95_sd']:5.3f}  "
              f"{r['worst_span']:7.3f} ±{r['worst_span_sd']:5.3f}")
    _dump(args.out, "coverage.json", list(rows.values()))


# ── landscapes ───────────────────────────────────────────────────────────────

def landscape(cfg, seed, n_needles=3, basin_width=6.0):
    """The sweep benchmark's planted-needle landscape (plain 0.75, peaks 1.0), drawn
    until every optimum is inside the configuration's search box."""
    from benchmarks.sweeps.needles import build_landscape
    lo, hi = box(cfg)
    for k in range(200):
        built = build_landscape(cfg["d"], n_needles, basin_width, seed * 1000 + k)
        C = np.asarray(built["centers"], float)
        if ((C >= lo - 1e-9) & (C <= hi + 1e-9)).all():
            return built["fn"], C, basin_width
    raise RuntimeError("no landscape with all optima inside the box")


HIT_LEVEL = 0.9   # a measured point counts as on a peak at y >= this (peaks 1.0, plain 0.75)


def core_radius(d, b, level=HIT_LEVEL):
    # y >= level  <=>  exp(-b r / sqrt(d)) >= 2*level - 1
    return math.sqrt(d) * math.log(1.0 / (2 * level - 1)) / b


# ── part B: surrogate quality from the seed data alone ───────────────────────

def _fit_gp(X, y):
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    Xt = torch.tensor(X[:, :-1], dtype=DTYPE)       # drop one collinear component
    yt = torch.tensor(y, dtype=DTYPE).unsqueeze(-1)
    gp = SingleTaskGP(Xt, yt, input_transform=Normalize(Xt.shape[-1]),
                      outcome_transform=Standardize(1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp


def surrogate(args):
    from scipy.stats import spearmanr
    out = []
    for name, cfg in CONFIGS.items():
        lo, hi = box(cfg)
        P = uniform_feasible(cfg["d"], 4000, lo, hi, seed=7)
        for b in (2.2, 6.0):
            for s in range(args.seeds):
                fn, C, _ = landscape(cfg, s, n_needles=3, basin_width=b)
                f_true = np.array([fn(x) for x in P])
                # also score at the optima neighbourhoods, where it matters
                for arm in ARMS:
                    rng = np.random.default_rng(10_000 + s)
                    lines = seed_lines(arm, cfg, rng)
                    X = np.vstack([print_line(a, c) for a, c in lines])
                    y = measure(fn, X, rng)
                    try:
                        gp = _fit_gp(X, y)
                        with torch.no_grad():
                            # importing src sets the default torch device to CUDA
                            mu = gp.posterior(torch.tensor(P[:, :-1], dtype=DTYPE)).mean.squeeze(-1).cpu().numpy()
                    except Exception as exc:  # noqa: BLE001
                        print(f"  gp fit failed ({name} {arm} s{s}): {exc}")
                        continue
                    rho = float(spearmanr(mu, f_true).correlation)
                    r2 = float(1 - ((mu - f_true) ** 2).sum() / ((f_true - f_true.mean()) ** 2).sum())
                    top = np.argsort(-mu)[:20]   # where the first prediction would look
                    out.append(dict(config=name, basin_width=b, seed=s, arm=arm,
                                    n_lines=len(lines), spearman=rho, r2=r2,
                                    best_of_top20=float(f_true[top].max()),
                                    max_true=float(f_true.max())))
    print("\nB. Surrogate fitted on the seed data alone, scored on 4000 uniform")
    print("   feasible points of 3-needle landscapes (b=2.2 broad, b=6 narrow).")
    print("   spearman = rank agreement with the truth; top20 = best true value")
    print("   among the 20 points the surrogate ranks highest (1.0 = on a peak).\n")
    print(f"{'config':8s} {'b':>4s} {'arm':7s} {'spearman':>15s} {'R2':>15s} {'top20':>15s}")
    for name in CONFIGS:
        for b in (2.2, 6.0):
            for arm in ARMS:
                r = [o for o in out if o["config"] == name and o["arm"] == arm and o["basin_width"] == b]
                if not r:
                    continue
                f = lambda k: (np.mean([o[k] for o in r]), np.std([o[k] for o in r]))  # noqa: E731
                (a1, a2), (b1, b2), (c1, c2) = f("spearman"), f("r2"), f("best_of_top20")
                print(f"{name:8s} {b:4.1f} {arm:7s} {a1:7.3f} ±{a2:5.3f}  {b1:7.3f} ±{b2:5.3f}  {c1:7.3f} ±{c2:5.3f}")
    _dump(args.out, "surrogate.json", out)


# ── part C: full ZoMBI-Hop runs under a line budget ──────────────────────────

class _Budget(BaseException):
    pass


def optimize_one(cfg_name, arm, seed, budget, out_dir, verbose=False):
    from src import LineBO, ZoMBIHop
    from src.default_hparams import DEFAULT_HPARAMS
    from src.utils.simplex import proj_simplex

    path = Path(out_dir) / "optimize" / f"{cfg_name}__{arm}__s{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    cfg = CONFIGS[cfg_name]
    d = cfg["d"]
    lo, hi = box(cfg)
    fn, C, b = landscape(cfg, seed)
    r_core = core_radius(d, b)
    rng = np.random.default_rng(50_000 + seed)
    torch.manual_seed(seed)

    lines = seed_lines(arm, cfg, rng)
    xa, xe, ys = [], [], []
    history = []       # per measured line: best true value so far, optima hit so far
    hit = np.zeros(len(C), bool)
    best = -np.inf

    def _record(X):
        nonlocal best
        ft = np.array([fn(x) for x in X])
        best = max(best, float(ft.max()))
        dist = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        hit[:] |= (dist <= r_core).any(axis=0)
        history.append((best, int(hit.sum())))

    for a, c in lines:
        X = print_line(a, c)
        y = measure(fn, X, rng)
        t = np.linspace(0, 1, NUM_EXPERIMENTS)[:, None]
        xa.append(X); xe.append(a + t * (c - a)); ys.append(y)
        _record(X)

    n_adapt = budget - len(lines)
    count = [0]
    bounds = torch.tensor(np.stack([lo, hi]), device=DEVICE, dtype=DTYPE)
    linebo = LineBO(lambda e: None, d, num_points_per_line=100, num_lines=10, device=DEVICE)

    def wrapper(x_tell, bnds, acq_fn):
        if count[0] >= n_adapt:
            raise _Budget()
        xl, xr = linebo.ranked_line_endpoints(x_tell, bnds, acq_fn)
        X = print_line(xl[0].cpu().numpy(), xr[0].cpu().numpy())
        y = measure(fn, X, rng)
        _record(X)
        count[0] += 1
        x_act = torch.tensor(X, device=DEVICE, dtype=DTYPE)
        xc = x_act - x_act.mean(0, keepdim=True)
        _, _, Vt = torch.linalg.svd(xc, full_matrices=False)
        p = xc @ Vt[0]
        tv = torch.linspace(p.min().item(), p.max().item(), len(X), device=DEVICE, dtype=DTYPE)
        x_req = proj_simplex(x_act.mean(0, keepdim=True) + tv[:, None] * Vt[0][None, :])
        return x_req, x_act, torch.tensor(y, device=DEVICE, dtype=DTYPE)

    hp = dict(DEFAULT_HPARAMS)
    hp["top_m_points"] = max(hp.get("top_m_points", 0), d + 1)
    t0 = time.time()
    opt = ZoMBIHop(
        objective=wrapper,
        X_init_actual=torch.tensor(np.vstack(xa), device=DEVICE, dtype=DTYPE),
        X_init_expected=torch.tensor(np.vstack(xe), device=DEVICE, dtype=DTYPE),
        Y_init=torch.tensor(np.concatenate(ys), device=DEVICE, dtype=DTYPE).reshape(-1, 1),
        bounds=bounds, input_noise=INPUT_NOISE, max_gp_points=3000, acquisition_type="ucb",
        device=DEVICE, dtype=DTYPE, run_uuid=None, checkpoint_dir=None, verbose=verbose,
        **hp)
    try:
        opt.run(max_activations=float("inf"), time_limit_hours=3.0, never_terminate=True)
    except _Budget:
        pass
    from eval_metrics import metric_dist_to_needles
    dh = opt.data_handler
    needles = dh.get_all_needle_locations()
    disc = needles.detach().cpu().numpy() if needles.numel() else np.empty((0, d))
    first95 = next((k + 1 for k, (bv, _) in enumerate(history) if bv >= HIT_LEVEL), None)
    res = dict(config=cfg_name, arm=arm, seed=seed, budget=budget,
               n_init_lines=len(lines), n_lines=len(history),
               best_true=best, optima_hit=int(hit.sum()), n_optima=len(C),
               first_line_on_peak=first95,
               n_needles=int(len(disc)),
               dist_to_needles=float(metric_dist_to_needles(disc, list(C), dim=d)),
               history=history, wall_s=round(time.time() - t0, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    print(f"[opt] {cfg_name} {arm:6s} s{seed}: lines={res['n_lines']} best={best:.3f} "
          f"hit={res['optima_hit']}/{len(C)} first_peak={first95} needles={res['n_needles']} "
          f"dist={res['dist_to_needles']:.3f} ({res['wall_s']}s)", flush=True)
    return res


def optimize(args):
    arms = args.arms.split(",")
    for s in range(args.seed0, args.seed0 + args.seeds):
        for arm in arms:
            optimize_one(args.config, arm, s, args.budget, args.out, verbose=args.verbose)


def summarize(args):
    files = sorted((Path(args.out) / "optimize").glob("*.json"))
    res = [json.loads(f.read_text()) for f in files]
    if not res:
        print("no optimize results")
        return
    print("\nC. Full ZoMBI-Hop runs (hardware default hyperparameters, print model,")
    print("   4.5% output noise), same total line budget INCLUDING the seed lines.")
    print("   hit = fraction of true optima with a measured point at y>=0.9; best = best")
    print("   TRUE value measured (peaks 1.0, plain 0.75); peak@ = median first line with")
    print("   a point at y>=0.9 (budget+1 if never); curve = mean over lines of the")
    print("   best-so-far rescaled 0 (plain) .. 1 (peak), higher = found sooner; dist =")
    print("   declared needles vs true optima (0.5 = nothing declared).\n")
    print(f"{'config':8s} {'arm':7s} {'n':>2s} {'init':>4s} {'hit':>11s} {'best':>11s} "
          f"{'peak@':>6s} {'curve':>11s} {'dist':>11s}")
    out = []
    for cfg in CONFIGS:
        for arm in ARMS:
            r = [x for x in res if x["config"] == cfg and x["arm"] == arm]
            if not r:
                continue
            hits = np.array([x["optima_hit"] / x["n_optima"] for x in r])
            best = np.array([x["best_true"] for x in r])
            peak = [x["first_line_on_peak"] if x["first_line_on_peak"] else x["budget"] + 1 for x in r]
            curve = np.array([np.mean([(min(b, 1.0) - 0.75) / 0.25 for b, _ in x["history"]])
                              for x in r])
            dist = np.array([x["dist_to_needles"] for x in r])
            row = dict(config=cfg, arm=arm, n=len(r), n_init=r[0]["n_init_lines"],
                       hit=float(hits.mean()), hit_sd=float(hits.std()),
                       best=float(best.mean()), best_sd=float(best.std()),
                       peak_median=float(np.median(peak)), curve=float(curve.mean()),
                       curve_sd=float(curve.std()), dist=float(dist.mean()), dist_sd=float(dist.std()))
            out.append(row)
            print(f"{cfg:8s} {arm:7s} {len(r):2d} {row['n_init']:4d} {row['hit']:5.2f} ±{row['hit_sd']:4.2f} "
                  f"{row['best']:5.3f} ±{row['best_sd']:4.3f} {row['peak_median']:6.1f} "
                  f"{row['curve']:5.2f} ±{row['curve_sd']:4.2f} {row['dist']:5.3f} ±{row['dist_sd']:4.3f}")
    _dump(args.out, "optimize_summary.json", out)


def _dump(out, name, obj):
    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    (p / name).write_text(json.dumps(obj, indent=1, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("part", choices=["coverage", "surrogate", "optimize", "summarize"])
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--config", default="6d_cap", choices=list(CONFIGS))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.seeds is None:
        args.seeds = {"coverage": 40, "surrogate": 20, "optimize": 5}.get(args.part, 5)
    {"coverage": coverage, "surrogate": surrogate, "optimize": optimize,
     "summarize": summarize}[args.part](args)


if __name__ == "__main__":
    main()
