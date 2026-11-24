"""
optimize/lanes_eval.py
======================
Simulated lanes run for ``evaluate.py --optimizer lanes`` (and so ``showdown.py
--optimizer lanes``): the two-lane engine of ``src/lanes`` -- the hardware default
optimizer (scripts/lanes_hw.py) -- on a synthetic landscape.

The engine is built exactly as the hardware builds it (``build_lane_engine``,
preset "leader", the run's core hyperparameters after ``apply_dim_rules``). Only the
DiSCO side is simulated: two rails ("front", "back"), each holding at most the line
it is measuring plus the next one it has read (lag 1), measured in strict
alternation, 25 simulated minutes per line. That is the call order of the lanes
test driver ``run_fake`` (tests/test_lanes_engine_smoke.py, itself a port of the
sandbox self-test the leader was validated with): every result goes to BOTH lanes
(the store ingests it once), the owner lane then decides its next main + backup,
and the other lane refreshes its still-unread main with the new data.

A line is measured by run_mobo's physics print model (``make_sim_obj``: 24 points,
ramp lag / diffusion mixing, output noise), the same as a single-loop run.

Budget: ``max_lines`` is the TOTAL number of lines measured by both lanes, so a lanes
run measures the same number of points as a single-loop run with the same
``--max-lines``. ``top_k`` and the time limit stop it as they stop a single-loop run.
No torch.manual_seed: like the single-loop runs, repeats differ by the process RNG;
the engine's cover-rule seed is drawn fresh and recorded.

Artifacts: the engine's shared DataHandler (both lanes' rows and every needle /
exclusion zone) goes through ``evaluate._finalize_eval``, so the run dir gets the
usual CSVs, plots, zones.pt and metrics.json. ``lanes.json`` adds the engine
summary (per-lane lines, activations, needles, decision timings). A point's
activation in the CSVs is the lanes activation id (unique across both lanes).
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections import deque

import numpy as np
import torch

LANES = ("front", "back")
LINE_MINUTES = 25.0          # simulated time per measured line (run_fake)


def run_lanes_eval(ev, hparams: dict, ds: dict, dataset: str, out_dir: str,
                   time_limit_min: float, *, top_k: int | None = None,
                   max_lines: int | None = None, **finalize_kw) -> dict:
    """``ev`` is the evaluate module (passed in to avoid a circular import)."""
    from src.hparams import apply_dim_rules, check_conflicts
    from src.lanes import build_lane_engine
    from src.lanes.geometry import DEV, DT

    rm = ev.rm
    os.makedirs(out_dir, exist_ok=True)
    dim, fn, maximize = ds["dim"], ds["fn"], ds["maximize"]
    cols = ev.coord_cols(dim, dataset)
    grid_pts = ds.get("grid_pts")

    try:
        X_a, X_e, Y = ev.gen_init_data(fn, maximize, dim)
    except RuntimeError as exc:
        print(f"      [run] init failed: {exc}")
        return {"dist": rm.UNMATCHED_PENALTY, "dup": 1.0, "nn_spacing": 0.0, "runtime": 0.0}

    # Core hyperparameters as the single-loop run gets them (dim rules, the
    # simulator's input_noise unless the config pins it); 'verbose' is the engine's
    # own argument, not a ZoMBIHop knob here.
    hp = apply_dim_rules(hparams, dim)
    check_conflicts(hp, context="lanes_eval")
    core = {k: v for k, v in rm.ZOMBI_FIXED.items() if k != "verbose"}
    core.update({k: v for k, v in hp.items() if k != "verbose"})

    seed = int(np.random.default_rng().integers(0, 2 ** 31 - 1))
    sign = 1.0 if maximize else -1.0

    def predict(X):            # noise-free landscape, optimizer sign (engine logging only)
        X = np.asarray(X, float).reshape(-1, dim)
        return sign * np.array([float(fn(x)) for x in X])

    engine = build_lane_engine(
        X_a, X_e, Y, np.stack([np.zeros(dim), np.ones(dim)]), preset="leader",
        hparams=core, seed=seed, checkpoint_dir=None, predict=predict, verbose=False)
    sim_obj = rm.make_sim_obj(fn, rm.DEVICE, rm.DTYPE, maximize=maximize)
    coverage_X = torch.as_tensor(
        np.random.default_rng(0).dirichlet(np.ones(dim), size=ev.PENALTY_COVERAGE_SAMPLES),
        dtype=rm.DTYPE, device=rm.DEVICE)

    dh = engine.dh
    lanes = {n: engine.lanes[n] for n in LANES}
    payloads: list[dict] = []
    zone_states: list[dict] = []
    snap_records: list[tuple] = [(int(dh.X_all_actual.shape[0]), 0, 0, 1.0)]
    lines_per_lane = {n: 0 for n in LANES}
    log = dict(refresh_calls=0, refresh_new=0)
    outcome = "time limit"

    # ── the simulated rails (run_fake) ──────────────────────────────────────────
    rails = {n: deque() for n in LANES}       # [line being measured, line read next]
    slots = {n: None for n in LANES}          # (main, backup) last written
    unread = {n: False for n in LANES}        # main not yet read by its rail
    now = 0.0

    def pick(n):
        if slots[n] is None or len(rails[n]) >= 2:
            return
        main, backup = slots[n]
        if unread[n]:
            rails[n].append(main)
            lanes[n].on_pickup(main.tags["rid"], now)
            unread[n] = False
        else:                                 # main already read: the rail takes the backup
            rails[n].append(backup)
            lanes[n].on_pickup(backup.tags["rid"], now)
            slots[n] = None

    def decide(n):
        m, b, _ = lanes[n].decide(now)
        slots[n], unread[n] = (m, b), True

    def record_before(line, step: int) -> None:
        """The per-line record evaluate's single-loop run writes (state before ingest)."""
        rec = engine.line_by_rid.get(line.tags["rid"])
        box = (torch.as_tensor(rec.box, dtype=rm.DTYPE) if rec is not None and rec.box is not None
               else dh.bounds.detach().cpu().clone())
        xp, yp = dh.X_pared, dh.Y_pared
        if xp is not None and xp.shape[0] > 0:
            pared_X, pared_Y = xp.detach().cpu().numpy(), yp.detach().cpu().numpy().ravel()
            if not maximize:
                pared_Y = -pared_Y
        else:
            pared_X = pared_Y = None
        needles = dh.needles
        pen_any, pen_needle = dh.penalty_coverage(coverage_X)
        zs = ev.zone_state(dh)
        zs["bounds"] = box.clone()            # the measured line's own zoom box (plot_run outlines)
        zone_states.append(zs)
        gp_vals = None
        if dim == 3 and grid_pts is not None:
            try:
                gp_vals = rm.gp_landscape_vals(engine.gp, grid_pts, maximize)
            except Exception:
                gp_vals = None
        ends = np.asarray(line.ends, float)
        payloads.append(dict(
            iter_num=step,
            pared_X=pared_X, pared_Y=pared_Y,
            needles=(needles.detach().cpu().numpy()
                     if needles is not None and needles.shape[0] > 0 else None),
            needle_vals=(dh.needle_vals.detach().cpu().numpy().ravel()
                         if dh.needle_vals is not None and dh.needle_vals.shape[0] > 0 else None),
            needle_M_list=[m.detach().cpu().clone() if m is not None else None
                           for m in dh.needle_M_list],
            needle_B=(dh.needle_B.detach().cpu().clone() if dh.needle_B is not None else None),
            bounds=box,
            line_0=(ends[0], ends[1]), line_1=None,
            n_points_before=int(dh.X_all_actual.shape[0]),
            activation=int(line.tags["act"]), lane=line.tags["lane"],
            pct_space_penalized=100.0 * pen_any,
            pct_space_penalized_needles=100.0 * pen_needle,
            gp_grid_vals=gp_vals,
        ))
        return box

    t0 = time.time()
    deadline = t0 + time_limit_min * 60.0 if time_limit_min else None
    interrupted = False
    n_lines = 0
    try:
        # start: each lane decides twice (line 1 read at once, line 2 queued and read)
        for _ in range(2):
            for n in LANES:
                decide(n)
                pick(n)
        step = 0
        while True:
            if max_lines is not None and n_lines >= max_lines:
                outcome = f"line budget ({n_lines} lines)"
                break
            if deadline is not None and time.time() >= deadline:
                outcome = f"time limit ({n_lines} lines)"
                break
            X = LANES[step % 2]
            Yn = LANES[(step + 1) % 2]
            step += 1
            now += LINE_MINUTES * 60.0
            if not rails[X]:
                pick(X)
            if not rails[X]:                  # nothing left to read: the lane decides again
                decide(X)
                pick(X)
            line = rails[X].popleft()
            box = record_before(line, n_lines + 1)
            L = torch.as_tensor(line.ends[0], device=rm.DEVICE, dtype=rm.DTYPE)
            R = torch.as_tensor(line.ends[1], device=rm.DEVICE, dtype=rm.DTYPE)
            x_act, y = sim_obj(torch.stack([L, R]).unsqueeze(0))
            pkt = dict(x=x_act.to(DEV, DT), y=y.to(DEV, DT).reshape(-1), t=now)
            for ln in lanes.values():         # both lanes see every packet; ingested once
                ln.on_result(pkt, line.tags, now)
            n_lines += 1
            lines_per_lane[X] += 1
            snap_records.append((int(dh.X_all_actual.shape[0]), int(line.tags["act"]),
                                 int(line.tags["zoom"]), float(rm.zoom_size_fraction(box))))
            n_needles = int(dh.needles.shape[0]) if dh.needles is not None else 0
            if top_k is not None and n_needles >= top_k:
                outcome = f"top-k ({n_needles} needles)"
                break
            decide(X)
            if slots[Yn] is not None and unread[Yn]:
                log["refresh_calls"] += 1
                r = lanes[Yn].refresh(now)
                if r is not None:
                    log["refresh_new"] += 1
                    slots[Yn] = (r[0], r[1])
            pick(Yn)
    except KeyboardInterrupt:
        interrupted = True
        outcome = "interrupted"
        print("\n      [run] interrupted — writing artifacts before exit …")
    except Exception as exc:
        outcome = f"crashed: {exc!r}"
        print(f"      [lanes] engine crashed after {n_lines} line(s): {exc!r}")
        traceback.print_exc()
    runtime = time.time() - t0
    print(f"      [lanes] stopped: {outcome}; lines front={lines_per_lane['front']} "
          f"back={lines_per_lane['back']}")

    try:
        summary = engine.summary()
        summary.update(outcome=outcome, seed=seed, n_lines=n_lines, lines_per_lane=lines_per_lane,
                       runtime_s=round(runtime, 3), driver="alternating lag-1 rails (run_fake)",
                       build_warnings=engine.build_info.get("warnings"), **log)
        with open(os.path.join(out_dir, "lanes.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
    except Exception as exc:
        print(f"      [lanes] lanes.json write failed: {exc!r}")

    try:
        result = ev._finalize_eval(
            dh=dh, payloads=payloads, snap_records=snap_records, zone_states=zone_states,
            n_lines=n_lines, runtime=runtime, hparams=hparams, ds=ds, dataset=dataset,
            out_dir=out_dir, cols=cols, **finalize_kw)
    finally:
        engine.close()                        # releases the seeded random_simplex (Fix 4)
    if interrupted:
        raise KeyboardInterrupt
    return result
