"""
optimize/two_head_eval.py
=========================
Simulated two-head run for ``evaluate.py --two-head``: the synthetic counterpart of
the hardware two-head mode (scripts/run_zombi_main.py two_head=True).

Two independent ZoMBI-Hop heads start from ONE shared initial design and run on
their own threads against the same landscape. They are linked by the same
``TwoHeadHub`` the hardware path uses (src/core/two_head.py): every line a head
measures becomes GP data for the other at its next measurement; penalty regions
(needles, exclusion zones) cross only at the receiving head's activation boundary;
duplicate needles found by both heads are kept. There is no divergence mechanism.

Budget: ``max_lines`` is the TOTAL optimizer line budget, split between the heads
(head0 gets the odd line), so a two-head run measures the same number of points as
a single-head run with the same ``--max-lines``. As in the single-head run, the line
that hits a head's budget is not counted (ZoMBI-Hop never stores it).

Artifacts: one merged run, built after both heads stop: the shared initial design,
then every counted line from either head in the order it was measured, and the
union of both heads' needles in declaration order. That merged view goes through
``evaluate._finalize_eval``, so the run dir gets the usual CSVs, plots, zones.pt and
metrics.json (dist/dup/nn over the merged run). ``two_head.json`` adds the per-head
breakdown. In the merged CSVs the activation of a point is ``1000 * head +
activation`` so the two heads' activations stay distinct.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import traceback

import numpy as np
import torch

HEAD_ACT_OFFSET = 1000


def run_two_head_eval(ev, hparams: dict, ds: dict, dataset: str, out_dir: str,
                      time_limit_min: float, *, top_k: int | None = None,
                      max_lines: int | None = None, **finalize_kw) -> dict:
    """``ev`` is the evaluate module (passed in to avoid a circular import)."""
    from src.core.two_head import TwoHeadHub
    from src.hparams import apply_dim_rules, check_conflicts

    rm = ev.rm
    os.makedirs(out_dir, exist_ok=True)
    dim, fn, maximize = ds["dim"], ds["fn"], ds["maximize"]
    cols = ev.coord_cols(dim, dataset)

    try:
        X_a, X_e, Y = ev.gen_init_data(fn, maximize, dim)
    except RuntimeError as exc:
        print(f"      [run] init failed: {exc}")
        return {"dist": rm.UNMATCHED_PENALTY, "dup": 1.0, "nn_spacing": 0.0, "runtime": 0.0}

    hp = apply_dim_rules(hparams, dim)
    check_conflicts(hp, context="two_head_eval")
    zombi_fixed = dict(rm.ZOMBI_FIXED)
    for k in ev._ZOMBI_OVERRIDE_KEYS:
        if k in hp:
            zombi_fixed[k] = hp.pop(k)

    if max_lines is not None:
        quotas = [int(math.ceil(max_lines / 2)), int(max_lines // 2)]
    else:
        quotas = [None, None]
    print(f"      [two-head] line budget per head: {quotas} (total {max_lines})")

    coverage_X = torch.as_tensor(
        np.random.default_rng(0).dirichlet(np.ones(dim), size=ev.PENALTY_COVERAGE_SAMPLES),
        dtype=rm.DTYPE, device=rm.DEVICE)

    hub = TwoHeadHub(log=print)
    lock = threading.Lock()
    events: list[dict] = []         # every measured line, both heads, in measurement order
    counters = [0, 0]
    heads: list = [None, None]

    def make_wrapper(h: int):
        plot_state: dict = {"line_0": None, "line_1": None}
        sim_obj = rm.make_sim_obj(fn, rm.DEVICE, rm.DTYPE, maximize=maximize)
        inner = rm.make_linebo_wrapper(sim_obj, dim, rm.NUM_LINES, rm.DEVICE, rm.DTYPE,
                                       plot_state)

        def obj_wrapper(x_tell, bounds, acq_fn):
            x_req, x_act, y = inner(x_tell, bounds, acq_fn)
            dh = heads[h].data_handler
            counters[h] += 1
            czb = dh.current_zoom_bounds if dh.current_zoom_bounds is not None else dh.bounds
            pen_any, pen_needle = dh.penalty_coverage(coverage_X)
            final = quotas[h] is not None and counters[h] >= quotas[h]
            with lock:
                n_needles = sum(r["kind"] == "needle" for _, r in hub.regions())
                events.append(dict(
                    head=h, line=counters[h], final=final, t=time.time(),
                    x_act=x_act.detach().cpu().clone(), x_req=x_req.detach().cpu().clone(),
                    y=y.detach().cpu().clone().reshape(-1),
                    activation=int(dh.current_activation), zoom=int(dh.current_zoom),
                    zoom_size=float(rm.zoom_size_fraction(czb)) if czb is not None else 1.0,
                    line_0=plot_state.get("line_0"), line_1=plot_state.get("line_1"),
                    bounds=dh.bounds.detach().cpu().clone() if dh.bounds is not None else None,
                    n_regions=len(hub.regions()),
                    pct_any=100.0 * pen_any, pct_needle=100.0 * pen_needle,
                    zone_state=ev.zone_state(dh)))
            if top_k is not None and n_needles >= top_k:
                raise ev._TopKReached(n_needles)
            if final:
                raise ev._LineBudgetReached(counters[h])
            return x_req, x_act, y

        return obj_wrapper

    for h in (0, 1):
        heads[h] = rm.ZoMBIHop(
            objective=make_wrapper(h),
            X_init_actual=X_a.clone(), X_init_expected=X_e.clone(), Y_init=Y.clone(),
            **zombi_fixed, **dict(hp), device=str(rm.DEVICE), dtype=rm.DTYPE)
    for h in (0, 1):
        hub.attach(h, heads[h])

    outcome = ["", ""]

    def target(h: int) -> None:
        try:
            heads[h].run(max_activations=float("inf"),
                         time_limit_hours=time_limit_min / 60.0, never_terminate=True)
            outcome[h] = "time limit"
        except ev._LineBudgetReached as stop:
            outcome[h] = f"line budget ({int(stop.args[0])} lines)"
        except ev._TopKReached as stop:
            outcome[h] = f"top-k ({int(stop.args[0])} needles)"
        except Exception as exc:          # one head crashing must not lose the other
            outcome[h] = f"crashed: {exc!r}"
            traceback.print_exc()
        print(f"      [two-head] head{h} stopped: {outcome[h]}")

    t0 = time.time()
    threads = [threading.Thread(target=target, args=(h,), name=f"head{h}", daemon=True)
               for h in (0, 1)]
    for t in threads:
        t.start()
    interrupted = False
    try:
        for t in threads:
            while t.is_alive():
                t.join(timeout=1.0)
    except KeyboardInterrupt:
        interrupted = True
        print("\n      [run] interrupted — writing artifacts before exit …")
    runtime = time.time() - t0

    # ── merged view of the whole run ────────────────────────────────────────────
    merged = rm.ZoMBIHop(
        objective=None, X_init_actual=X_a.clone(), X_init_expected=X_e.clone(),
        Y_init=Y.clone(), **zombi_fixed, **dict(hp), device=str(rm.DEVICE), dtype=rm.DTYPE)
    mdh = merged.data_handler
    dev = dict(device=mdh.device, dtype=mdh.dtype)
    regions = hub.regions()
    n_init = int(mdh.X_all_actual.shape[0])
    snap_records: list[tuple] = [(n_init, 0, 0, 1.0)]
    payloads: list[dict] = []
    zone_states: list[dict] = []
    with lock:
        counted = [e for e in events if not e["final"]]

    def needles_upto(k: int):
        recs = [r for _, r in regions[:k] if r["kind"] == "needle"]
        if not recs:
            return None, None, [], None
        return (np.stack([r["center"].numpy() for r in recs]),
                np.array([r.get("value", np.nan) for r in recs], float),
                [r.get("M") for r in recs], recs[-1].get("B"))

    for i, e in enumerate(counted):
        act = HEAD_ACT_OFFSET * e["head"] + e["activation"]
        nd, nv, nM, nB = needles_upto(e["n_regions"])
        payloads.append(dict(
            iter_num=i + 1, pared_X=None, pared_Y=None,
            needles=nd, needle_vals=nv, needle_M_list=nM, needle_B=nB,
            bounds=e["bounds"], line_0=e["line_0"], line_1=e["line_1"],
            n_points_before=int(mdh.X_all_actual.shape[0]),
            activation=act, head=e["head"],
            pct_space_penalized=e["pct_any"], pct_space_penalized_needles=e["pct_needle"],
            gp_grid_vals=None))
        zone_states.append(e["zone_state"])
        x_act = merged.proj_fn(e["x_act"].to(**dev))
        mdh.add_all_points(x_act, e["x_req"].to(**dev), e["y"].to(**dev).reshape(-1, 1))
        snap_records.append((int(mdh.X_all_actual.shape[0]), act, e["zoom"], e["zoom_size"]))
    for origin, rec in regions:
        mdh.import_region(rec, origin)

    # Per-head breakdown.
    per_head = []
    for h in (0, 1):
        dh = heads[h].data_handler
        own = int((~dh.foreign_mask()).sum().item()) - n_init
        link = heads[h].two_head_link
        per_head.append(dict(
            head=h, outcome=outcome[h],
            lines_counted=sum(e["head"] == h for e in counted),
            own_points=own,
            foreign_points_taken_in=int(link.n_foreign_points),
            regions_adopted_from_other=int(link.n_imported_regions),
            needles_declared=sum(1 for o, r in regions if o == h and r["kind"] == "needle"),
            exclusions_declared=sum(1 for o, r in regions if o == h and r["kind"] != "needle"),
            activations=int(dh.current_activation) + 1))
    summary = dict(two_head=True, line_quotas=quotas, n_init_points=n_init,
                   n_points_merged=int(mdh.X_all_actual.shape[0]),
                   n_needles_merged=sum(r["kind"] == "needle" for _, r in regions),
                   runtime_s=round(runtime, 3), heads=per_head)
    with open(os.path.join(out_dir, "two_head.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for ph in per_head:
        print(f"      [two-head] head{ph['head']}: {ph['lines_counted']} line(s), "
              f"{ph['own_points']} own pt(s), took in {ph['foreign_points_taken_in']}, "
              f"{ph['needles_declared']} needle(s) declared, "
              f"{ph['regions_adopted_from_other']} region(s) adopted, "
              f"{ph['activations']} activation(s) — {ph['outcome']}")

    result = ev._finalize_eval(
        dh=mdh, payloads=payloads, snap_records=snap_records, zone_states=zone_states,
        n_lines=len(counted), runtime=runtime, hparams=hparams, ds=ds, dataset=dataset,
        out_dir=out_dir, cols=cols, **finalize_kw)
    if interrupted:
        raise KeyboardInterrupt
    return result
