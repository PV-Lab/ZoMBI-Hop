"""Independent auditor of a lanes engine (DESIGN.md 1.2): ``audit(engine) -> dict``.

Moved verbatim from the sandbox self-test ``benchmarks/zoom_fix/lanes/lane_selftest.py`` (ST 250-431); only
the imports changed (``LZ.COUNTED`` / ``LZ.check_nested`` / ``LZ.DEV`` / ``LZ.DT`` -> ``src.lanes.geometry``).
It reads only the engine's logs (packets, row tags, audit records, events) and checks:
  A1 zoom-in only after >= min_iters_per_zoom OWN counted results of that zoom (recounted from
     the packet log, not the lane's counters)
  A2 every activation-scoped decision used only rows of the deciding lane's current activation
  A3 every box is nested in its parent, never wider, strictly smaller, rho-capped unless the floor
     of its zoom binds (graded floor in the lanes core)
  A4 needles only from own zoom-2 rows inside that activation's zoom-2 box: candidate, repeats,
     declared point, local-median rows (each audit record carries the box)
  X2 no zoom-in commits a box covered >= cross_cover_frac by the other lane's needles / zones
  A5 verdict rows of a zoom >= 1 packet lie inside the box in force; counted packets are own, current
     activation
  A6 bridge lines (none in the port: bridges are not ported; the check stays and finds none)
  A7 nothing measured is dropped (rows = init + packet rows, values at their rows, tags aligned)
The confirm-and-revise / needle-refinement / bridge sections stay (they read event kinds the port never
emits, DESIGN N4) so the report keeps the sandbox's shape. No side effects; no RNG.
"""
from __future__ import annotations

import numpy as np
import torch

from src.lanes.geometry import COUNTED, DEV, DT, check_nested

__all__ = ["audit"]


def audit(store, injected=None) -> dict:
    R = store.rows.a
    hp = store.lanes["front"]._hp()
    opts = store.opts
    X = store.dh.X_all_actual.detach().cpu().numpy()
    out = {}
    fails = []

    def fail(code, msg):
        fails.append(f"{code}: {msg}")

    # A1
    zi = [e for e in store.events if e["kind"] == "zoom_in"]
    for e in zi:
        n = sum(1 for p in store.packets if p.get("lane") == e["lane"] and p.get("act") == e["act"]
                and p.get("counted_zoom") == e["from_zoom"] and p.get("processed_seq") is not None
                and p["processed_seq"] <= e["seq"])
        if n < hp["min_iters"]:
            fail("A1", f"zoom-in {e['lane']} a{e['act']} z{e['from_zoom']}->{e['to_zoom']} after {n} own results")
        if n != e["results"]:
            fail("A1", f"lane counter {e['results']} != recount {n} ({e['lane']} a{e['act']})")
    out["A1_zoom_ins"] = len(zi)

    # A2 (+ needle / box zoom checks)
    n_rows = 0
    for a in store.audit:
        code = store.lanes[a["lane"]].code
        rows = np.asarray(a["rows"], dtype=np.int64)
        n_rows += rows.size
        if rows.size == 0:
            continue
        if not (np.all(R["lane"][rows] == code) and np.all(R["act"][rows] == a["act"])):
            fail("A2", f"{a['kind']} of {a['lane']} a{a['act']} used foreign rows "
                       f"{[(int(R['lane'][r]), int(R['act'][r])) for r in rows if R['lane'][r] != code or R['act'][r] != a['act']][:5]}")
        if a["kind"] in ("needle_cand", "needle_repeats", "needle_pick", "needle_median_rows", "needle_refine_rows"):
            if not np.all(R["eff"][rows] == 2):
                fail("A4", f"{a['kind']} used rows not counted for zoom 2: eff {sorted(set(R['eff'][rows].tolist()))}")
            if a.get("box") is None:
                fail("A4", f"{a['kind']} of {a['lane']} a{a['act']} logged without its zoom-2 box")
            else:
                bx = np.asarray(a["box"])
                out_ = ~np.all((X[rows] >= bx[0] - 1e-9) & (X[rows] <= bx[1] + 1e-9), axis=1)
                if out_.any():
                    d_ = float(np.max(np.maximum(bx[0] - X[rows][out_], X[rows][out_] - bx[1]).clip(min=0)))
                    fail("A4", f"{a['kind']} of {a['lane']} a{a['act']}: {int(out_.sum())} rows outside the zoom-2 box "
                               f"(up to {d_:.4f})")
        if a["kind"] == "confirm_rows" and not np.all(R["zoom"][rows] == a["to_zoom"] - 1):
            fail("A2", f"confirm rows not sent at zoom {a['to_zoom'] - 1}")
        if a["kind"] == "verdict_rows" and a.get("box") is not None:
            bx = np.asarray(a["box"])
            if a["zoom"] >= 1 and not np.all((X[rows] >= bx[0] - 1e-9) & (X[rows] <= bx[1] + 1e-9)):
                fail("A5", f"verdict rows outside the box (pid {a.get('pid')})")
    out["A2_audited_decisions"] = len(store.audit)
    out["A2_audited_rows"] = int(n_rows)
    out["audit_kinds"] = {k: sum(1 for a in store.audit if a["kind"] == k)
                          for k in sorted({a["kind"] for a in store.audit})}

    # A3 (the floor of the new box's zoom: graded by zoom in the lanes core)
    dev, dt = DEV, DT
    floor = store.lanes["front"].floor
    n_box, n_over, n_eq = 0, 0, 0
    checks = [(e["box"], e["parent"], e["to_zoom"], "zoom-in") for e in zi]
    checks += [(e["final"], e["parent"], e["zoom"], "revised") for e in store.events
               if e["kind"] == "box_confirm" and e.get("outcome") == "revised"]
    for child, parent, to_zoom, what in checks:
        try:
            c = check_nested(torch.tensor(child, device=dev, dtype=dt), torch.tensor(parent, device=dev, dtype=dt),
                             opts.rho, floor(int(to_zoom)))
            n_box += 1
            n_over += int(c["over_rho"] > 0)
            n_eq += int(c["eq_parent"] > 0)
        except AssertionError as exc:
            fail("A3", f"{what} box: {exc}")
    out["A3_boxes_checked"] = n_box
    out["A3_boxes_axis_over_rho"] = n_over          # allowed only where the floor binds
    out["A3_boxes_axis_eq_parent"] = n_eq

    # A4 needles
    # (as ZoMBIHop._select_needle_point: the candidate is the best own zoom-2 row INSIDE the box;
    #  the declared point is the median-closest of its own zoom-2 repeats within the radius)
    nd = [e for e in store.events if e["kind"] == "needle"]
    for e in nd:
        p, c = e["pick"], e["cand"]
        for r in (p, c):
            if not (R["lane"][r] == store.lanes[e["lane"]].code and R["act"][r] == e["act"] and R["eff"][r] == 2):
                fail("A4", f"needle {e['needle']} row {r} is not an own zoom-2 row")
        bx = np.asarray(e["box"])
        if not np.all((X[c] >= bx[0] - 1e-9) & (X[c] <= bx[1] + 1e-9)):
            fail("A4", f"needle {e['needle']} candidate outside its zoom-2 box")
        if not np.all((X[p] >= bx[0] - 1e-9) & (X[p] <= bx[1] + 1e-9)):
            fail("A4", f"needle {e['needle']} declared point outside its zoom-2 box")
        if float(np.linalg.norm(X[p] - X[c])) > e["radius"] + 1e-9:
            fail("A4", f"needle {e['needle']} pick is not a repeat of the candidate")
    for e in store.events:                              # refined needles: the new pick in the same box
        if e["kind"] == "needle_refine" and e.get("pick") is not None and e.get("box") is not None:
            bx, p = np.asarray(e["box"]), int(e["pick"])
            if not np.all((X[p] >= bx[0] - 1e-9) & (X[p] <= bx[1] + 1e-9)):
                fail("A4", f"refined needle {e['needle']} outside its zoom-2 box")
    out["A4_needles"] = len(nd)

    # A5 counted packets: own lane, the lane's current activation at processing, never another class
    vpid = {a["pid"]: a for a in store.audit if a["kind"] == "verdict_rows"}
    for p in store.packets:
        if p.get("counted_zoom") is not None:
            a = vpid.get(p["pid"])
            if a is None or a["lane"] != p["lane"] or a["act"] != p["act"] or a["zoom"] != p["counted_zoom"]:
                fail("A5", f"counted packet {p['pid']} ({p['lane']} a{p['act']}) not credited to its own stage")
    for p in store.packets:
        if p.get("counted_zoom") is not None:
            if p["cls"] not in COUNTED:
                fail("A5", f"packet {p['pid']} counted with class {p['cls']}")
            if p.get("in_box_frac", 1.0) < opts.count_frac and p["counted_zoom"] >= 1:
                fail("A5", f"packet {p['pid']} counted with in-box share {p['in_box_frac']:.2f}")
        elif p["cls"] in COUNTED:
            fail("A5", f"packet {p['pid']} class {p['cls']} without a counted zoom")
    out["A5_classes"] = {c: sum(1 for p in store.packets if p["cls"] == c)
                         for c in sorted({str(p["cls"]) for p in store.packets})}
    fr = [p["in_box_frac"] for p in store.packets if "in_box_frac" in p
          and ((p.get("counted_zoom") or 0) >= 1 or p["cls"] in ("own_outside", "own_bridge_outside"))]
    if fr:
        out["zoom_ge1_in_box_share"] = dict(n=len(fr), mean=round(float(np.mean(fr)), 3),
                                            p10=round(float(np.percentile(fr, 10)), 3),
                                            below_count_frac=sum(1 for v in fr if v < opts.count_frac))

    # A6 bridges
    br = [e for e in store.events if e["kind"] == "bridge"]
    n_counted_bridge = 0
    for e in br:
        ends, lb = np.asarray(e["ends"]), np.asarray(e["line_box"])
        pred, cur = torch.tensor(e["pred"], device=dev, dtype=dt), torch.tensor(e["box"], device=dev, dtype=dt)
        if not np.all((ends >= lb[0] - 1e-9) & (ends <= lb[1] + 1e-9)):
            fail("A6", f"bridge {e['rid']} endpoints outside the predicted line box")
        if not np.all((lb[0] >= np.asarray(e["pred"])[0] - 1e-12) & (lb[1] <= np.asarray(e["pred"])[1] + 1e-12)):
            fail("A6", f"bridge {e['rid']} line box not inside the predicted box")
        try:
            check_nested(pred, cur, opts.rho, floor(int(e["zoom"]) + 1))
        except AssertionError as exc:
            fail("A6", f"bridge {e['rid']} predicted box: {exc}")
        pk = [p for p in store.packets if p.get("rid") == e["rid"]]
        for p in pk:
            if p["cls"] == "own_current_bridge":
                n_counted_bridge += 1
                if p["counted_zoom"] != e["zoom"] + 1:
                    fail("A6", f"bridge {e['rid']} counted for zoom {p['counted_zoom']} (sent at {e['zoom']})")
    out["A6_bridges"] = len(br)
    out["A6_bridges_counted_next_zoom"] = n_counted_bridge
    out["A6_bridge_packets"] = {c: sum(1 for p in store.packets if p.get("bridge") and p["cls"] == c)
                                for c in sorted({str(p["cls"]) for p in store.packets if p.get("bridge")})}

    # A7
    cons = store.conservation()
    if not cons["ok"]:
        fail("A7", f"conservation {cons}")
    out["A7_conservation"] = cons

    # confirm-and-revise
    bc = [e for e in store.events if e["kind"] == "box_confirm"]
    out["confirm"] = {o: sum(1 for e in bc if e.get("outcome") == o) for o in sorted({str(e.get("outcome")) for e in bc})}
    nr = [e for e in store.events if e["kind"] == "needle_refine"]
    out["needle_refine"] = {o: sum(1 for e in nr if e.get("outcome") == o) for o in sorted({str(e.get("outcome")) for e in nr})}
    if opts.confirm_revise:
        prov = [e for e in zi if e.get("provisional")]
        out["provisional_zoom_ins"] = len(prov)
    # cross-lane
    de = [e for e in store.events if e["kind"] == "dive_end"]
    out["dive_ends"] = {r: sum(1 for e in de if e["reason"] == r) for r in sorted({e["reason"] for e in de})}
    # X2: a committed box (zoom-in) was covered by less than the cover share of the other lane's zones
    for e in zi:
        if e.get("cover_other") is not None and e["cover_other"] >= opts.cross_cover_frac:
            fail("X2", f"zoom-in {e['lane']} a{e['act']} committed a box {e['cover_other']:.2f} covered by the other lane")
    out["X2_zoom_in_cover_max"] = max([float(e.get("cover_other") or 0.0) for e in zi], default=0.0)
    vd = [p.get("verdict") or {} for p in store.packets]
    out["verdicts"] = {w: sum(1 for v in vd if v.get("why") == w) for w in sorted({str(v.get("why")) for v in vd if v})}
    if injected is not None:
        hit = [e for e in de if e["lane"] == injected["victim"] and e["act"] == injected["act"]
               and e["reason"] == "covered_by_other"]
        if not hit:
            fail("X1", f"injected cover of {injected} did not end the victim's dive")
        out["X1_injected_cover_ended_dive"] = bool(hit)
    out["fails"] = fails
    out["ok"] = not fails
    return out
