"""
benchmarks/axial_init/disco_sim.py
==================================
Discrete-event simulation of DiSCO's two transporters feeding on ZoMBI's initial
design, to check the init print queue (src/utils/init_queue.py) never deadlocks and
gets the whole design printed under the conditions the real system produces.

DiSCO model (Full_System_Integration/archerfish.py, run.py, communication.py):
  * each transporter loops: reach the printer -> wait until the printer is free and a
    composition is unread (main first, then cache) -> print -> anneal / hyperspectral /
    photoconductivity / arm -> come back. With stability on, the objective for that
    slide is produced later by the Prism PC, so the transporter does not wait for it.
  * a ZoMBI write replaces BOTH slots; anything unread is lost.
  * objective_sender drops NaN rows and never sends an empty packet, so a failed slide
    produces no result at all ("silent").

Policies compared:
  queue   the new init: every result sends the next queued pair (InitQueue)
  pairs   the previous rule (and the Manual Ctrl queue's): send a pair, then wait for
          two results before sending the next

Usage
-----
    python -m benchmarks.axial_init.disco_sim [--reps 200]
"""

from __future__ import annotations

import argparse
import heapq
import itertools

import numpy as np

from benchmarks.ablations._paths import ensure_paths

ensure_paths()

from src.utils.axial_init import axial_design, sample_trace  # noqa: E402
from src.utils.init_queue import InitQueue  # noqa: E402

MIN = 60.0


def simulate(n_dims, *, policy="queue", print_min=10, forward_min=20, return_min=5,
             stability_min=45, jitter_min=0.0, silent_p=0.0, nan_row_p=0.0,
             transporters=("front", "back"), stale_results=0, seed=0,
             first_timeout_min=240, floor_min=45, mult=3, max_sends=3, noise=0.03,
             min_write_gap_s=5.0, min_latency_min=15.0,
             horizon_h=72):
    rng = np.random.default_rng(seed)
    traces = axial_design(n_dims).traces
    q = InitQueue(traces, max_sends=max_sends, min_latency=min_latency_min * MIN)
    tie = itertools.count()
    ev = []                                   # (time_s, tie, kind, payload)

    def push(t, kind, payload=None):
        heapq.heappush(ev, (t, next(tie), kind, payload))

    db = {"main": None, "cache": None}        # slot -> [trace index, unread?]
    printer_free = 0.0
    prints = []                               # trace indices actually printed
    stats = {"stalls": 0, "writes": 0, "lost_slots": 0}
    pairs_state = {"pending": 0}

    def write(now):
        pair = q.next_pair(now)
        if pair is None:
            return
        for slot in ("main", "cache"):
            if db[slot] is not None and db[slot][1]:
                stats["lost_slots"] += 1      # unread slot overwritten
        db["main"], db["cache"] = [pair[0], True], [pair[1], True]
        stats["writes"] += 1
        pairs_state["pending"] = 2
        arm_stall(now)

    stall = {"deadline": None}
    pending_send = {"at": None}

    def arm_stall(now):
        if policy != "queue":
            return
        to = q.stall_timeout(first_timeout_min * MIN, floor_min * MIN, mult)
        stall["deadline"] = (q.last_activity_t or now) + to
        push(stall["deadline"], "stall_check", stall["deadline"])

    # ZoMBI starts
    write(0.0)
    for k in range(stale_results):           # leftovers from a previous run
        push(1.0 + k, "result", ("stale", None))
    for tr in transporters:
        push(0.0, "arrive", tr)

    t_done = None
    while ev:
        now, _, kind, payload = heapq.heappop(ev)
        if now > horizon_h * 3600:
            break
        if kind == "arrive":
            tr = payload
            slot = "main" if (db["main"] and db["main"][1]) else \
                   "cache" if (db["cache"] and db["cache"][1]) else None
            if now < printer_free or slot is None:
                push(max(now + 60.0, printer_free), "arrive", tr)   # DiSCO polls
                continue
            idx = db[slot][0]
            db[slot][1] = False
            printer_free = now + print_min * MIN
            prints.append(idx)
            placed = printer_free + forward_min * MIN
            if rng.random() >= silent_p:
                lag = stability_min * MIN + rng.uniform(-jitter_min, jitter_min) * MIN
                push(placed + max(lag, 0.0), "result", ("slide", idx))
            push(placed + return_min * MIN, "arrive", tr)
        elif kind == "result":
            src, idx = payload
            if src == "stale":
                # a regular-loop line from the previous run: random chord, 24 points
                x0, v = rng.dirichlet(np.ones(n_dims)), rng.normal(size=n_dims)
                v -= v.mean()
                v *= rng.uniform(0.2, 0.7) / np.linalg.norm(v)
                x = np.clip(x0 + np.linspace(-0.5, 0.5, 24)[:, None] * v, 0, None)
                x /= x.sum(1, keepdims=True)
            else:
                x = sample_trace(traces[idx], 24) + rng.normal(0, noise, (24, n_dims))
                x = np.clip(x, 0, None)
                x /= x.sum(1, keepdims=True)
            y = np.ones(24)
            y[rng.random(24) < nan_row_p] = np.nan
            if policy == "queue":                # mirrors run_axial_initialization
                r = q.on_result(x, y, now)
                if r["kind"] != "foreign" and q.queue:
                    last = q.writes[-1]["t"] if q.writes else -1e18
                    if now - last >= min_write_gap_s:
                        write(now)
                    elif not pending_send["at"]:
                        pending_send["at"] = last + min_write_gap_s
                        push(pending_send["at"], "send_due")
                    else:
                        arm_stall(now)
                else:
                    arm_stall(now)
            else:                                # previous rule: two results per pair
                r = q.on_result(x, y, now)
                pairs_state["pending"] -= 1
                if pairs_state["pending"] <= 0 and q.queue:
                    write(now)
            if q.complete() and t_done is None:
                t_done = now
        elif kind == "send_due":
            pending_send["at"] = None
            if q.queue:
                write(now)
        elif kind == "stall_check":
            if policy == "queue" and payload == stall["deadline"] and not q.complete():
                stall["deadline"] = None
                stats["stalls"] += 1
                q.on_stall(now)
                if q.queue:
                    write(now)
                else:
                    arm_stall(now)
                if q.complete() and t_done is None:
                    t_done = now
        if t_done is not None:
            break

    n = len(traces)
    with_data = sum(1 for i in range(n) if any(len(b["y"]) for b in q.data[i]))
    return {
        "complete": t_done is not None,
        "hours": (t_done / 3600) if t_done is not None else float("nan"),
        "traces": n,
        "with_data": with_data,
        "prints": len(prints),
        "duplicates": len(prints) - len(set(prints)),
        "abandoned": sum(s == "abandoned" for s in q.status),
        **stats,
    }


SCENARIOS = {
    "hardware (stability on)": dict(),
    "no stability (result at end of cycle)": dict(stability_min=0),
    "10% slides fail silently": dict(silent_p=0.10),
    "30% slides fail silently": dict(silent_p=0.30),
    "partial NaN rows (30%)": dict(nan_row_p=0.30),
    "results out of order (±30 min)": dict(jitter_min=30),
    "back transporter disabled": dict(transporters=("front",)),
    "2 stale results at start": dict(stale_results=2),
    "fast results + 20% silent": dict(stability_min=0, silent_p=0.20),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args(argv)
    print(f"{'scenario':40s} {'dims':>4s} {'policy':>6s} {'complete':>8s} {'hours':>6s} "
          f"{'data/traces':>11s} {'prints':>6s} {'dups':>5s} {'stalls':>6s}")
    for name, kw in SCENARIOS.items():
        for d in (3, 4, 6, 10):
            for policy in ("queue", "pairs"):
                rs = [simulate(d, policy=policy, seed=s, **kw) for s in range(args.reps)]
                comp = np.mean([r["complete"] for r in rs])
                hrs = np.nanmedian([r["hours"] for r in rs]) if comp > 0 else float("nan")
                print(f"{name:40s} {d:4d} {policy:>6s} {comp:8.0%} {hrs:6.1f} "
                      f"{np.mean([r['with_data'] for r in rs]):5.1f}/{rs[0]['traces']:<5d} "
                      f"{np.mean([r['prints'] for r in rs]):6.1f} "
                      f"{np.mean([r['duplicates'] for r in rs]):5.2f} "
                      f"{np.mean([r['stalls'] for r in rs]):6.2f}")


if __name__ == "__main__":
    main()
