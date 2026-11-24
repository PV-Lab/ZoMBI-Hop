"""Lanes wire codec and recipe-id bookkeeping (src/lanes/wire.py; DESIGN 6.2, reviewer corrections R2 / R4).

* codec round trip on real packets (fixtures copied from benchmarks/zoom_fix/lanes/result_id/fixtures into
  tests/fixtures/lanes_wire/);
* byte identity with the sandbox reference wire.py for the same inputs, and a differential router run
  (skipped when the sandbox folder is absent);
* byte identity with the SHARED BLOCK of the hardware cable test (result_id/hw_test/*_side_test.py and the
  user's tests/zombi_side_test.py copy), so the real-cable GREEN transfers to the production encoders;
* R2 id ranges (run 0..899, Manual Control 900..999, router class ``manual``), R4 persistence hooks, the
  non-raising ``try_issue`` (DESIGN 6.12), import hygiene (F12).

F9: an autouse fixture restores the CPU / CUDA torch RNG states and the three ``random_simplex`` attributes;
files are written only under ``tmp_path``. Run: PYTHONPATH=. python -B -m pytest -q -p no:cacheprovider
tests/test_lanes_wire.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import src.lanes.wire as W

REPO = Path(__file__).resolve().parents[1]
FIX = Path(__file__).resolve().parent / "fixtures" / "lanes_wire"
RESULT_ID = REPO / "benchmarks" / "zoom_fix" / "lanes" / "result_id"
SANDBOX_WIRE = RESULT_ID / "wire.py"
HW_SOURCES = [RESULT_ID / "hw_test" / "zombi_side_test.py",
              RESULT_ID / "hw_test" / "disco_side_test.py",
              REPO / "tests" / "zombi_side_test.py"]
BLOCK_BEGIN = "# ============================ SHARED BLOCK BEGIN ============================"
BLOCK_END = "# ============================= SHARED BLOCK END ============================="
M = 60.0


# ── F9 isolation ─────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _restore_global_state():
    import torch
    import src.core.linebo as _lb
    import src.core.zombihop as _zh
    import src.utils.domain as _dom
    mods = (_dom, _zh, _lb)
    saved_rs = {m: m.__dict__.get("random_simplex") for m in mods}
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        for m, f in saved_rs.items():
            if f is not None:
                m.random_simplex = f


# ── helpers ──────────────────────────────────────────────────────────────────────────
def _jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="module")
def obj_packets():
    return _jsonl(FIX / "objective_packets_96a9.jsonl")


@pytest.fixture(scope="module")
def comp_lines():
    return _jsonl(FIX / "composition_lines.jsonl")


def _endpoint10(x, dims):
    """The _write_line_pair rule (RZM 862-867): round 3 dp -> normalise in the searched dims -> pad to 10."""
    a = np.round(np.asarray(x, float), 3)
    s = a.sum()
    a = a / (s if s != 0 else 1.0)
    out = np.zeros(10)
    out[list(dims)] = a
    return [float(v) for v in out]


def _milli10(x, dims):
    a = np.zeros(10)
    a[list(dims)] = np.asarray(x, float)
    return W.milli_vector(a)


def _four_slots(lines, i, rid0=417):
    ln, nx = lines[i], lines[(i + 1) % len(lines)]
    return {0: (rid0, _endpoint10(ln["main"][0], ln["dims"]), _endpoint10(ln["main"][1], ln["dims"])),
            1: (rid0 + 1, _endpoint10(ln["cache"][0], ln["dims"]), _endpoint10(ln["cache"][1], ln["dims"])),
            2: (rid0 + 2, _endpoint10(nx["main"][0], nx["dims"]), _endpoint10(nx["main"][1], nx["dims"])),
            3: (rid0 + 3, _endpoint10(nx["cache"][0], nx["dims"]), _endpoint10(nx["cache"][1], nx["dims"]))}


def _load_by_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                       # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sandbox_wire():
    if not SANDBOX_WIRE.exists():
        pytest.skip("sandbox result_id/wire.py not present")
    return _load_by_path(SANDBOX_WIRE, "_lanes_sandbox_wire_ref")


def _shared_block(path: Path) -> dict:
    src = path.read_text(encoding="utf-8")
    b, e = src.index(BLOCK_BEGIN), src.index(BLOCK_END)
    ns = {"__name__": "_hw_shared_" + path.stem}
    exec(compile(src[b:e], f"{path.name}:shared_block", "exec"), ns)
    return ns


def _rows(v):
    return dict(values=[v, v + 0.01], comps=[[0.1] * 10, [0.2] * 10])


# ── 1. codec round trip on real packets ──────────────────────────────────────────────
def test_objective_fixture_bytes_and_round_trip(obj_packets):
    assert len(obj_packets) == 47
    for i, p in enumerate(obj_packets):
        q = p["packet"]
        b0 = W.encode_objective(q["values"], q["comps"], q["ts"])
        assert len(b0) == p["rebuilt_bytes"]
        assert abs(len(b0) - p["logged_bytes"]) <= 2                  # DiSCO's logged size (ts digits unknown)
        assert W.legacy_objective_view(b0) == dict(values=q["values"], comps=q["comps"], ts=q["ts"])
        d0 = W.decode_objective(b0)
        assert d0["rid"] is None and d0["last_read"] is None and d0["values"] == q["values"]
        rid = (417 + i) % 1000
        k = ((rid + 1) % 1000, None if i % 3 == 0 else (rid + 290) % 1000)
        b1 = W.encode_objective(q["values"], q["comps"], q["ts"], rid=rid, last_read=k)
        d1 = W.decode_objective(b1)
        assert (d1["rid"], d1["last_read"], d1["values"], d1["comps"], d1["ts"]) == (rid, k, q["values"], q["comps"], q["ts"])
        # today's receiver view is unchanged by the new keys
        assert W.legacy_objective_view(b1) == W.legacy_objective_view(b0)
        assert len(W.encode_objective(q["values"], q["comps"], q["ts"], rid=rid)) - len(b0) == 10
    # NULL notice: today's receiver ignores it, the new one decodes it
    null = W.encode_objective([], [], 1.5, rid=905, last_read=(905, None))
    assert W.legacy_objective_view(null) is None
    assert W.decode_objective(null)["values"] == [] and W.decode_objective(null)["rid"] == 905
    # a malformed id never blocks the data
    bad = W.dumps_line({"type": "objective", "values": [0.5], "comps": [[0.1] * 10], "ts": 2.0, "r": "x7", "k": [1]})
    assert W.decode_objective(bad) == dict(values=[0.5], comps=[[0.1] * 10], ts=2.0, rid=None, last_read=None)
    assert W.decode_objective(b"garbage\n") is None and W.decode_objective(b"") is None


def test_four_slot_fixture_round_trip(comp_lines):
    assert len(comp_lines) == 240
    sizes = {}
    for i, ln in enumerate(comp_lines):
        slots = _four_slots(comp_lines, i)
        full = W.encode_slots(1793817832.321631, slots)
        sizes.setdefault(len(ln["dims"]), []).append(len(full))
        d = W.decode_slots(full)
        assert d["full"] is True and d["milli"] is False and d["ts"] == 1793817832.321631
        assert d["slots"] == {k: (r, a, b) for k, (r, a, b) in slots.items()}     # exact float round trip
        nx = comp_lines[(i + 1) % len(comp_lines)]
        sm = {0: (417, _milli10(ln["main"][0], ln["dims"]), _milli10(ln["main"][1], ln["dims"])),
              2: (419, _milli10(nx["main"][0], nx["dims"]), _milli10(nx["main"][1], nx["dims"]))}
        dm = W.decode_slots(W.encode_slots(1.0, sm, milli=True))
        assert dm["milli"] is True and sorted(dm["slots"]) == [0, 2]
        for k in (0, 2):
            for got, ref in ((dm["slots"][k][1], slots[k][1]), (dm["slots"][k][2], slots[k][2])):
                assert max(abs(x - y) for x, y in zip(got, ref)) < 1e-12
        delta = W.decode_slots(W.encode_slots(2.0, {1: slots[1]}, full=False))
        assert delta["full"] is False and list(delta["slots"]) == [1]
    # RESULT_ID.md:234-239: median 714 B (4-dim) and 918-938 B (6-dim); every packet far inside the 1 Hz budget
    med = {d: sorted(v)[len(v) // 2] for d, v in sizes.items()}
    assert med[4] < 800 and med[6] < 1000, med
    assert max(W.wire_seconds(b) for v in sizes.values() for b in v) < 0.15


def test_decode_slots_rejects_malformed():
    v = [0.1] * 10
    good = {"type": "composition", "ts": 1.0, "s": [[0, 5, v, v]]}
    assert W.decode_slots(W.dumps_line(good))["slots"] == {0: (5, v, v)}
    bad = [dict(good, ts=None), dict(good, ts=True), dict(good, s=5), dict(good, s=[[0, 5, v]]),
           dict(good, s=[[4, 5, v, v]]), dict(good, s=[[0, 5, v, v], [0, 6, v, v]]),
           dict(good, s=[[True, 5, v, v]]), dict(good, s=[[0, 1000, v, v]]), dict(good, s=[[0, 5, v[:9], v]]),
           dict(good, s=[[0, 5, [-0.1] + v[1:], v]]), dict(good, s=[[0, 5, [True] + v[1:], v]]),
           dict(good, m=1)]                           # milli with float values
    for pkt in bad:
        with pytest.raises(ValueError):
            W.decode_slots(W.dumps_line(pkt))
    assert W.decode_slots(W.dumps_line({"type": "objective", "values": []})) is None
    with pytest.raises(ValueError):
        W.encode_slots(1.0, {4: (1, v, v)})
    with pytest.raises(ValueError):
        W.encode_slots(1.0, {0: (1000, v, v)})


def test_disco_slots_reference_rules():
    S = W.DiscoSlots()
    v = [0.1] * 10
    pk = lambda ts, s: W.decode_slots(W.encode_slots(ts, s))           # noqa: E731
    assert list(S.apply(pk(1.0, {0: (1, v, v), 1: (2, v, v), 2: (3, v, v), 3: (4, v, v)})).values()) == ["replace"] * 4
    assert set(S.apply(pk(1.0, {0: (1, v, v)})).values()) == {"same_ts"}
    assert S.read("front")[:2] == (0, 1)
    a = S.apply(pk(2.0, {0: (5, v, v), 1: (2, v, v), 2: (3, v, v), 3: (4, v, v)}))
    assert a[0] == "keep_read" and a[1] == "keep"
    S.clear("front")
    assert S.slots[0].rid == 5 and S.slots[0].state == "UNREAD"
    a = S.apply(pk(3.0, {0: (5, v, v), 2: (3, v, v), 3: (4, v, v)}))
    assert a[1] == "withdrawn" and S.slots[1].state == "EMPTY"


def test_grid24_reference():
    start = W.from_milli([0, 0, 245, 0, 704, 0, 0, 0, 49, 2])
    end = W.from_milli([465, 0, 232, 0, 0, 0, 0, 0, 0, 303])
    g = np.asarray(W.grid24(start, end))
    assert g.shape == (24, 10)
    assert np.allclose(g.sum(1), 1.0, atol=1e-12)
    for row, ep in ((g[0], start), (g[-1], end)):
        r = np.round(np.asarray(ep), 3)
        assert np.array_equal(row, r / r.sum())
    a, b = np.asarray(start), np.asarray(end)                          # _write_line_pair arithmetic (RZM 858-866)
    x = np.round(np.array([a + t * (b - a) for t in np.linspace(0, 1, 24)]), 3)
    assert np.array_equal(g, x / x.sum(-1, keepdims=True))
    assert W.grid24([0.0] * 10, [0.0] * 10)[0] == [0.0] * 10           # zero rows stay finite (sum guard)


# ── 2. byte identity with the sandbox reference ─────────────────────────────────────
def test_byte_identity_with_sandbox_codec(sandbox_wire, obj_packets, comp_lines):
    S = sandbox_wire
    for i, p in enumerate(obj_packets):
        q = p["packet"]
        for rid, k in ((None, None), (417, None), (0, (999, None)), (899, (None, None)), (i, (i + 1, 900 + i))):
            bp = W.encode_objective(q["values"], q["comps"], q["ts"], rid=rid, last_read=k)
            bs = S.encode_objective(q["values"], q["comps"], q["ts"], rid=rid, last_read=k)
            assert bp == bs
            assert W.decode_objective(bp) == S.decode_objective(bs)
            assert W.legacy_objective_view(bp) == S.legacy_objective_view(bs)
    for i in range(0, len(comp_lines), 7):
        slots = _four_slots(comp_lines, i, rid0=896)
        for kw in (dict(), dict(full=False)):
            for sub in (slots, {k: slots[k] for k in (0, 3)}, {}):
                bp, bs = W.encode_slots(12.5, sub, **kw), S.encode_slots(12.5, sub, **kw)
                assert bp == bs
                assert W.decode_slots(bp) == S.decode_slots(bs)
        ln = comp_lines[i]
        sm = {0: (5, _milli10(ln["main"][0], ln["dims"]), _milli10(ln["main"][1], ln["dims"]))}
        assert W.encode_slots(3.0, sm, milli=True) == S.encode_slots(3.0, sm, milli=True)
        assert W.decode_slots(W.encode_slots(3.0, sm, milli=True)) == S.decode_slots(S.encode_slots(3.0, sm, milli=True))
        assert W.from_milli(sm[0][1]) == S.from_milli(sm[0][1])
        a, b = slots[0][1], slots[0][2]
        grid = [a] * 24
        for ids in (None, (417, 418)):
            bp = W.encode_composition_legacy(7.0, a, b, grid, a, b, grid, ids=ids)
            assert bp == S.encode_composition_legacy(7.0, a, b, grid, a, b, grid, ids=ids)
            assert W.decode_composition_legacy(bp) == S.decode_composition_legacy(bp)
            assert W.legacy_disco_accepts(bp) == S.legacy_disco_accepts(bp) is True
    assert W.milli_vector([0.1234, 0.5, 0.3766]) == S.milli_vector([0.1234, 0.5, 0.3766])
    assert W.from_milli([0] * 10) == S.from_milli([0] * 10)
    assert W._payload_hash([0.5, 1], [[0.1] * 10, [1] * 10]) == S._payload_hash([0.5, 1], [[0.1] * 10, [1] * 10])
    assert W.added_bytes_objective() == S.added_bytes_objective() == 27
    assert (W.ID_MODULUS, W.REUSE_HORIZON_S, W.LOST_AFTER_READ_S, W.MIN_READ_TO_RESULT_S, W.SLOT_RAIL, W.RAILS) == \
           (S.ID_MODULUS, S.REUSE_HORIZON_S, S.LOST_AFTER_READ_S, S.MIN_READ_TO_RESULT_S, S.SLOT_RAIL, S.RAILS)


def _scenario(seed):
    """A V12-style two-rail schedule plus duplicates, reprints, unknown / legacy packets, supersedes,
    timeouts and one DiSCO restart; ids stay below 900 (the R2 manual class is tested separately)."""
    rng = random.Random(seed)
    ev = []
    for rail in W.RAILS:
        tr = rng.uniform(0, 40) * M
        for k in range(10):
            tr += rng.uniform(45, 53) * M
            ev.append((tr - rng.uniform(5, 40) * M, 0, "issue", (rail, k)))
            if rng.random() < 0.3:
                ev.append((tr - rng.uniform(1, 4) * M, 1, "backup", (rail, k)))
            ev.append((tr, 2, "read", (rail, k)))
            res_t = tr + rng.uniform(50, 78) * M
            u = rng.random()
            kind = "null" if u < 0.08 else ("none" if u < 0.15 else "result")
            ev.append((res_t, 3, kind, (rail, k)))
            if kind == "result" and rng.random() < 0.15:
                ev.append((res_t + 2.0, 4, "dup", (rail, k)))
            if kind == "result" and rng.random() < 0.05:
                ev.append((res_t + 30.0, 4, "reprint", (rail, k)))
    for _ in range(3):
        ev.append((rng.uniform(0, 600) * M, 5, "unknown", rng.randrange(0, 900)))
        ev.append((rng.uniform(0, 600) * M, 5, "legacy", None))
    ev.append((rng.uniform(300, 500) * M, 6, "restart", None))
    ev.sort(key=lambda e: (e[0], e[1]))
    return rng.randrange(900), [(t, kind, p) for t, _, kind, p in ev], rng


def _drive(mod, seed):
    start, events, rng = _scenario(seed)
    ids = mod.RecipeIds(start=start, modulus=900)
    R = mod.ResultRouter(ids)
    out, rid_of, last = [], {}, {"front": None, "back": None}
    gaps = [(rng.uniform(0, 600) * M, rng.uniform(20, 90) * M) for _ in range(3)]
    for t, kind, p in events:
        gated = any(g0 <= t <= g0 + g1 for g0, g1 in gaps)
        k = None if gated else (last["front"], last["back"])
        if kind == "issue":
            rid_of[p] = ids.issue(0 if p[0] == "front" else 2, list(p), t)
            out.append(("issue", rid_of[p]))
        elif kind == "backup":
            b = ids.issue(1 if p[0] == "front" else 3, [p[0], p[1], "b"], t)
            R.supersede(b, t + 60.0)
            out.append(("backup", b))
        elif kind == "read":
            last[p[0]] = rid_of[p]
            if rng.random() < 0.5 and not gated:
                R.on_read_ack(k, t + 1.0)
        elif kind in ("result", "dup", "reprint", "null"):
            v = 0.1 + 0.001 * p[1] + (0.5 if kind == "reprint" else 0.0)
            pk = dict(rid=rid_of[p], last_read=k, **(_rows(v) if kind != "null" else dict(values=[], comps=[])))
            out.append((kind, R.on_objective(pk, t)))
        elif kind == "unknown":
            out.append((kind, R.on_objective(dict(rid=p, last_read=k, **_rows(0.9)), t)))
        elif kind == "legacy":
            out.append((kind, R.on_objective(dict(rid=None, last_read=k, **_rows(0.8)), t)))
        elif kind == "restart":
            out.append((kind, R.on_disco_restart(t)))
        out.append(("expire", R.expire(t)))
    state = {rid: (r.status, r.rail, r.seq_ack, r.seq_superseded, r.payload_hashes) for rid, r in ids.by_id.items()}
    js = ids.to_json(events[-1][0] + 1.0)
    common = {k: js[k] for k in ("next", "modulus", "horizon", "clock", "history", "recipes")}
    return out, state, common, dict(R.last_read)


def test_router_differential_against_sandbox(sandbox_wire):
    n_cls = {}
    for seed in range(40):
        prod, sand = _drive(W, seed), _drive(sandbox_wire, seed)
        assert prod == sand, f"seed {seed}"
        for kind, res in prod[0]:
            if isinstance(res, dict):
                n_cls[res["cls"]] = n_cls.get(res["cls"], 0) + 1
    # the scenarios exercise every sandbox class
    assert {"matched", "late", "duplicate", "reprint", "unknown", "legacy", "null"} <= set(n_cls), n_cls


# ── 3. identity with the hardware cable test's shared block ─────────────────────────
def _hw_params():
    out = [pytest.param(p, id=f"{p.parent.name}/{p.name}") for p in HW_SOURCES if p.exists()]
    return out or [pytest.param(None, id="absent")]


@pytest.mark.parametrize("hw_path", _hw_params())
def test_packets_byte_identical_to_hw_test_shared_block(hw_path):
    if hw_path is None:
        pytest.skip("no hw_test script present")
    hw = _shared_block(hw_path)
    assert (hw["BAUD"], hw["ID_MODULUS"], tuple(hw["SLOT_RAIL"]), tuple(hw["RAILS"])) == \
           (W.BAUD, W.ID_MODULUS, W.SLOT_RAIL, W.RAILS)
    assert hw["MAIN_SLOT"] == W.MAIN_SLOT and hw["BACKUP_SLOT"] == W.BACKUP_SLOT
    n_slot = n_obj = 0
    for seed in (4711, 1, 20261113):
        for milli in (0, 1):
            plan = hw["build_plan"](seed, 20, 12, milli)
            # (a) the plan's content fingerprint, recomputed with the PRODUCTION encoders
            probe = []
            for x in plan["issues"][:4]:
                a, b = hw["recipe_wire"](seed, x["label"], milli)
                sa, sb = hw["decode_endpoints"](a, b, milli)
                vals, comps = hw["result_rows"](sa, sb, 3)
                probe.append([W.encode_slots(1.5, {0: (x["rid"], a, b)}, milli=bool(milli)).decode("utf-8"),
                              W.encode_objective(vals, comps, 2.5, rid=x["rid"], last_read=(1, 2)).decode("utf-8"),
                              hw["today_packet_bytes"](seed, x["label"], x["label"], 1.5)])
            assert hashlib.sha1(json.dumps(probe).encode("utf-8")).hexdigest()[:12] == plan["content"]
            # (b) every recipe of the plan, alone and in four-slot packets with gaps
            wires = {x["rid"]: hw["recipe_wire"](seed, x["label"], milli) for x in plan["issues"]}
            rids = [x["rid"] for x in plan["issues"]]
            for j in range(0, len(rids), 3):
                slots4 = [None] * 4
                for k, rid in enumerate(rids[j:j + 4]):
                    if (j + k) % 5 != 4:
                        slots4[k] = (rid,) + tuple(wires[rid])
                ts = 1793817832.321631 + j
                bh = hw["encode_slots"](ts, slots4, milli)
                bp = W.encode_slots(ts, {k: e for k, e in enumerate(slots4) if e is not None}, milli=bool(milli))
                assert bp == bh
                ts_h, dec_h = hw["decode_slots"](hw["loads_line"](bh))
                dec_p = W.decode_slots(bp)
                assert dec_p["ts"] == ts_h and dec_p["slots"] == {k: (r, list(a), list(b)) for k, (r, a, b) in dec_h.items()}
                assert dec_p["milli"] == bool(milli) and dec_p["full"] is True
                n_slot += 1
            for rid, (a, b) in wires.items():
                W.check_id(rid)                                    # the cable test's ids 9xx and the wrap are wire-valid
                if milli:
                    assert W.from_milli(a) == hw["from_milli"](a) and W.from_milli(b) == hw["from_milli"](b)
            # (c) every result packet of the plan, with and without r / k, and NULL notices
            acks = plan["acks"]
            for n, arr in enumerate(plan["arrivals"]):
                sa, sb = hw["recipe_endpoints"](seed, arr["label"], milli)
                vals, comps = hw["result_rows"](sa, sb, arr["n_rows"])
                ts = 1793817831.5912442 + arr["t"]
                kf = acks["front"][n % len(acks["front"])] if acks["front"] else None
                kb = acks["back"][n % len(acks["back"])] if acks["back"] else None
                for rid, k in ((arr["rid"], (kf, kb)), (arr["rid"], None), (None, None), (arr["rid"], (None, kb))):
                    bh = hw["encode_objective"](vals, comps, ts, rid, k)
                    bp = W.encode_objective(vals, comps, ts, rid=rid, last_read=k)
                    assert bp == bh
                    d = W.decode_objective(bh)
                    assert (d["values"], d["comps"], d["ts"], d["rid"]) == (vals, comps, ts, rid)
                    assert d["last_read"] == (None if k is None else tuple(k))
                    n_obj += 1
                assert W.encode_objective([], [], ts, rid=arr["rid"], last_read=(kf, kb)) == \
                    hw["encode_objective"]([], [], ts, arr["rid"], (kf, kb))
    assert n_slot > 50 and n_obj > 400


@pytest.mark.parametrize("hw_path", _hw_params())
def test_decoders_and_slot_model_agree_with_hw_test(hw_path):
    if hw_path is None:
        pytest.skip("no hw_test script present")
    hw = _shared_block(hw_path)
    v = [0.1] * 10
    good = {"type": "composition", "ts": 1.0, "s": [[0, 5, v, v]]}
    for pkt in (dict(good, ts=None), dict(good, s=[[0, 5, v]]), dict(good, s=[[4, 5, v, v]]),
                dict(good, s=[[0, 5, v, v], [0, 6, v, v]]), dict(good, s=[[0, 1000, v, v]]),
                dict(good, s=[[0, 5, v[:9], v]]), dict(good, s=[[0, 5, [-0.1] + v[1:], v]]), dict(good, m=1)):
        with pytest.raises(ValueError):
            hw["decode_slots"](pkt)
        with pytest.raises(ValueError):
            W.decode_slots(W.dumps_line(pkt))
    # the two reference slot models end in the same (id, state, pending id) after a scripted sequence
    P, H = W.DiscoSlots(), hw["Slots"]()
    rng = random.Random(7)
    nid, cur = 0, [None] * 4
    for step in range(400):
        u = rng.random()
        if u < 0.45:
            for k in range(4):
                if cur[k] is None or rng.random() < 0.4:
                    cur[k] = nid
                    nid += 1
                if rng.random() < 0.1:
                    cur[k] = None
            slots = {k: (cur[k], v, v) for k in range(4) if cur[k] is not None}
            P.apply(W.decode_slots(W.encode_slots(float(step), slots)))
            H.apply({k: (r, a, b) for k, (r, a, b) in slots.items()})
        elif u < 0.75:
            rail = rng.choice(W.RAILS)
            assert (P.read(rail) or (None,))[:2] == (H.read(rail) or (None,))[:2]
        else:
            rail = rng.choice(W.RAILS)
            P.clear(rail)
            H.clear(rail)
        for k in range(4):
            p, h = P.slots[k], H.s[k]
            assert (p.rid, p.state, p.pending and p.pending[0]) == (h["rid"], h["state"], h["pending"] and h["pending"][0])


# ── 4. R2: id ranges and the manual class ────────────────────────────────────────────
def test_run_and_manual_id_ranges():
    ids = W.RecipeIds(horizon_s=0.0)
    got = []
    for i in range(1800):
        rid = ids.issue(i % 4, i, float(i))
        ids.by_id[rid].status = "resulted"
        got.append(rid)
    assert got[:900] == list(range(900)) and got[900:] == list(range(900)) and max(got) == 899
    r = W.RecipeIds(start=898)
    assert [r.issue(0, x, 0.0) for x in "abc"] == [898, 899, 0]
    r = W.RecipeIds(start=995)                     # deliberate R2 difference to the sandbox (995, 996)
    assert [r.issue(0, x, 0.0) for x in "ab"] == [95, 96]
    m = W.RecipeIds.for_manual(start=998)
    assert [m.issue(0, x, 0.0) for x in "abc"] == [998, 999, 900] and m.id_range == (900, 1000)
    assert W.RecipeIds.for_run().id_range == (0, 900) == W.RUN_ID_RANGE
    for bad in (dict(base=-1), dict(modulus=0), dict(base=900, modulus=101), dict(base=0, modulus=1001)):
        with pytest.raises(ValueError):
            W.RecipeIds(**bad)
    with pytest.raises(ValueError, match="Manual Control"):
        W.ResultRouter(W.RecipeIds(modulus=1000))
    with pytest.raises(ValueError):
        W.ResultRouter(W.RecipeIds.for_manual())
    W.ResultRouter(W.RecipeIds(modulus=1000), manual_ids=None)            # explicit opt-out is allowed


def test_manual_ids_are_classed_before_lookup_and_never_routed():
    ids = W.RecipeIds(start=10)
    R = W.ResultRouter(ids)
    a = ids.issue(0, W.make_line("lane", lane="front", rid="F1", hrid="F1", role="main", act=0, zoom=0), 0.0)
    ids.by_id[905] = W.Recipe(rid=905, slot=0, rail="front", line="planted", t_issue=0.0)   # cannot happen; proves order
    rev0 = R.rev
    res = R.on_objective(dict(rid=905, last_read=None, **_rows(0.5)), 70 * M)
    assert res == dict(cls="manual", rid=905, lost=[]) and R.rev == rev0
    assert ids.by_id[905].status == "issued" and ids.by_id[905].payload_hashes == []
    assert R.on_objective(dict(rid=950, last_read=None, values=[], comps=[]), 71 * M)["cls"] == "manual"
    # the read acks a manual packet carries are still processed
    res = R.on_objective(dict(rid=999, last_read=(a, None), **_rows(0.6)), 72 * M)
    assert res["cls"] == "manual" and ids.by_id[a].status == "read" and R.last_read["front"] == a
    assert R.counts["manual"] == 3
    assert W.disposition(res) == W.LOG
    # Manual Control's own router routes its ids normally
    mids = W.RecipeIds.for_manual()
    MR = W.ResultRouter(mids, manual_ids=None)
    q = mids.issue(0, "pair-1-main", 0.0)
    assert q == 900 and MR.on_objective(dict(rid=q, last_read=None, **_rows(0.4)), 70 * M)["cls"] == "matched"
    assert MR.on_objective(dict(rid=3, last_read=None, **_rows(0.4)), 71 * M)["cls"] == "unknown"


def test_disposition_rules():
    lane = W.make_line("lane", lane="back", rid="B3", hrid="B2b", role="backup", act=1, zoom=2, trace=None, send=4)
    init = W.make_line("init", trace=5)
    assert list(lane) == list(W.LINE_FIELDS)
    for cls in ("matched", "late", "reprint"):
        assert W.disposition(dict(cls=cls, line=lane)) == W.LANE
        assert W.disposition(dict(cls=cls, line=init)) == W.INIT
    for cls in ("duplicate", "null", "manual"):
        assert W.disposition(dict(cls=cls, line=lane), ingest_unattributed=True) == W.LOG
    for cls in ("unknown", "legacy"):
        assert W.disposition(dict(cls=cls)) == W.LOG                                # R2 default: not ingested
        assert W.disposition(dict(cls=cls), ingest_unattributed=True) == W.SHARED
        assert W.disposition(dict(cls=cls), ingest_unattributed=True, t_recv=99.0, run_start=100.0) == W.LOG
        assert W.disposition(dict(cls=cls), ingest_unattributed=True, t_recv=101.0, run_start=100.0,
                             ts=5.0, first_write_ts=5.0) == W.LOG
        assert W.disposition(dict(cls=cls), ingest_unattributed=True, t_recv=101.0, run_start=100.0,
                             ts=6.0, first_write_ts=5.0) == W.SHARED
    with pytest.raises(ValueError):
        W.make_line("bridge")
    with pytest.raises(ValueError):
        W.make_line("lane", lane="left")


def test_sandbox_v2_file_with_modulus_1000_is_clamped(caplog):
    v2 = dict(version=2, ids=dict(version=2, next=950, modulus=1000, horizon=86400.0, clock=3,
                                  history={"949": 10.0}, recipes=[]),
              last_read={"front": 949, "back": None}, lost_after=5400.0)
    with caplog.at_level(logging.WARNING, logger="src.lanes.wire"):
        R = W.ResultRouter.from_json(v2)
    assert R.ids.id_range == (0, 900) and R.ids.next == 50 and R.manual_ids == W.MANUAL_ID_RANGE
    assert R.last_read == {"front": 949, "back": None}
    assert any("overlap" in r.getMessage() for r in caplog.records)


# ── 5. R4: persistence hooks ────────────────────────────────────────────────────────
def test_issue_hook_runs_before_the_id_is_returned(tmp_path):
    ids = W.RecipeIds()
    R = W.ResultRouter(ids)
    seen = []

    def hook(reason, now):
        seen.append((reason, now, sorted(int(r["rid"]) for r in R.to_json()["ids"]["recipes"])))
    R.on_change = hook
    rid = R.try_issue(0, W.make_line("init", trace=0), 5.0)
    assert seen == [("issue", 5.0, [rid])]                  # saved with the new id, before the caller writes it
    rid2 = ids.try_issue(2, W.make_line("init", trace=1), 6.0)   # via the allocator directly: same hook
    assert seen[-1] == ("issue", 6.0, sorted([rid, rid2])) and R.rev == 2


def test_hook_fires_once_per_state_change_and_not_otherwise():
    ids = W.RecipeIds(start=100)
    R = W.ResultRouter(ids)
    calls = []
    R.on_change = lambda reason, now: calls.append(reason)
    f1 = ids.try_issue(0, "F1", 0.0)
    f1b = ids.try_issue(1, "F1b", 0.0)
    R.on_read_ack((f1, None), 10.0)
    R.on_read_ack((f1, None), 11.0)                          # unchanged ack: no save
    f2 = ids.try_issue(0, "F2", 20.0)
    R.supersede(f1b, 20.0)
    R.supersede(f1b, 21.0)                                   # already superseded: no save
    R.on_read_ack((f2, None), 3000.0)                        # read + unprinted closure: ONE save
    assert calls == ["issue", "issue", "ack", "issue", "supersede", "ack"]
    calls.clear()
    assert R.on_objective(dict(rid=f1, last_read=(f2, None), **_rows(0.5)), 3500.0)["cls"] == "matched"
    assert R.on_objective(dict(rid=f1, last_read=(f2, None), **_rows(0.5)), 3501.0)["cls"] == "duplicate"
    assert R.on_objective(dict(rid=7, last_read=None, **_rows(0.5)), 3502.0)["cls"] == "unknown"
    assert R.on_objective(dict(rid=None, last_read=None, **_rows(0.5)), 3503.0)["cls"] == "legacy"
    assert R.on_objective(dict(rid=950, last_read=None, **_rows(0.5)), 3504.0)["cls"] == "manual"
    assert R.on_objective(dict(rid=f1, last_read=None, **_rows(0.7)), 3505.0)["cls"] == "reprint"
    assert calls == ["matched", "reprint"]
    f3 = ids.try_issue(0, "F3", 3600.0)
    R.on_read_ack((f3, None), 6000.0)
    calls.clear()
    res = R.on_objective(dict(rid=f3, last_read=(f3, None), **_rows(0.9)), 9500.0)
    assert res["lost"] == [f2] and calls == ["matched"]      # FIFO closure inside the same single save
    assert R.expire(9600.0) == [] and calls == ["matched"]   # nothing expired: no save
    f4 = ids.try_issue(0, "F4", 9700.0)
    R.on_read_ack((f4, None), 9800.0)
    assert R.expire(9800.0 + 91 * M) == [f4] and calls[-1] == "expire"
    assert R.on_objective(dict(rid=f4, last_read=None, **_rows(0.2)), 9800.0 + 95 * M)["cls"] == "late"
    assert calls[-1] == "late"
    f5 = ids.try_issue(0, "F5", 20000.0)
    assert R.on_objective(dict(rid=f5, last_read=None, values=[], comps=[]), 20000.0 + 40 * M)["cls"] == "null"
    assert calls[-1] == "null"
    R.on_disco_restart(30000.0)
    assert calls[-1] == "disco_restart"
    assert R.counts == {"matched": 2, "duplicate": 1, "unknown": 1, "legacy": 1, "manual": 1, "reprint": 1,
                        "late": 1, "null": 1}


def test_persist_to_file_round_trip_and_resume(tmp_path):
    path = tmp_path / "lanes" / "ids.json"
    path.parent.mkdir()
    ids = W.RecipeIds(start=0)
    R = W.ResultRouter(ids)
    R.persist_to(path)
    a = R.try_issue(0, W.make_line("lane", lane="front", rid="F0", hrid="F0", role="main", act=0, zoom=0), 0.0)
    b = R.try_issue(1, W.make_line("lane", lane="front", rid="F1", hrid="F0b", role="backup", act=0, zoom=0), 0.0)
    assert json.loads(path.read_text())["ids"]["next"] == 2       # saved at the issue, before any slot write
    R.on_read_ack((a, None), 60.0)
    R.on_objective(dict(rid=a, last_read=(a, None), **_rows(0.5)), 70 * M)
    disk = json.loads(path.read_text())
    assert disk == json.loads(json.dumps(R.to_json(70 * M)))
    R2 = W.ResultRouter.load(path)
    assert R2.on_objective(dict(rid=a, last_read=(a, None), **_rows(0.5)), 71 * M)["cls"] == "duplicate"
    rb = R2.on_objective(dict(rid=b, last_read=(b, None), **_rows(0.6)), 80 * M)
    assert rb["cls"] == "matched" and rb["line"]["hrid"] == "F0b" and W.line_kind(rb["line"]) == "lane"
    assert R2.try_issue(0, "next", 81 * M) == 2 and R2.rev > R.rev
    assert R2.events[-1] == ("read", "front", b, 80 * M) and all(isinstance(e, tuple) for e in R2.events)
    assert len(R2.to_json()["events"]) <= W.EVENTS_PERSIST
    # a custom saver (the Phase-2 driver passes persist.save_json) is used instead of the in-place writer
    got = []
    R2.persist_to(tmp_path / "x.json", saver=lambda p, obj: got.append((p, obj["rev"])) or True)
    R2.try_issue(2, "x", 82 * M)
    assert got and got[-1] == (tmp_path / "x.json", R2.rev)


def test_pick_newer_and_hook_errors(caplog):
    ids = W.RecipeIds()
    R = W.ResultRouter(ids)
    s0 = R.to_json()
    ids.try_issue(0, "a", 0.0)
    s1 = R.to_json()
    R.on_objective(dict(rid=0, last_read=None, **_rows(0.5)), 70 * M)   # a result: rev moves, clock does not
    s2 = R.to_json()
    assert s2["ids"]["clock"] == s1["ids"]["clock"] and s2["rev"] > s1["rev"]
    assert W.ResultRouter.pick_newer(s1, s2) is s2 and W.ResultRouter.pick_newer(s2, s0, s1) is s2
    assert W.ResultRouter.pick_newer(None, s1, None) is s1 and W.ResultRouter.pick_newer(None) is None
    assert W.ResultRouter.pick_newer(s1, json.loads(json.dumps(s1))) is s1        # tie: the first (ids.json)

    def boom(reason, now):
        raise OSError("dropbox lock")
    R.on_change = boom
    with caplog.at_level(logging.ERROR, logger="src.lanes.wire"):
        assert R.try_issue(0, "b", 100.0) == 1                                  # never raises into the driver
    assert R.hook_errors == 1 and any("hook failed" in r.getMessage() for r in caplog.records)


def test_recover_from_sent_ids():
    ids = W.RecipeIds.recover({5: 100.0, 7: 200.0, 950: 210.0}, 300.0, last=7)
    assert ids.next == 8 and ids.history == {5: 100.0, 7: 200.0}             # manual ids ignored
    got = [ids.issue(0, i, 400.0) for i in range(4)]
    assert got == [8, 9, 10, 11]
    ids = W.RecipeIds.recover([898, 3, 899], 50.0)                           # no times, no last: max + 1, wrapped
    assert ids.next == 0 and ids.history == {898: 50.0, 3: 50.0, 899: 50.0}
    taken = [ids.issue(0, i, 60.0) for i in range(5)]
    assert taken == [0, 1, 2, 4, 5]                                          # 3 is not reused within 24 h
    ids = W.RecipeIds.recover([], 0.0)
    assert ids.next == 0 and ids.history == {}


# ── 6. exhaustion never raises into the driver (DESIGN 6.12) ────────────────────────
def test_try_issue_exhaustion_and_fallbacks(caplog):
    ids = W.RecipeIds(modulus=4)
    for i in range(4):
        ids.issue(0, i, 0.0)
    with caplog.at_level(logging.WARNING, logger="src.lanes.wire"):
        assert ids.try_issue(0, "x", 1.0) is None                           # all open: None, no exception
    assert ids.n_exhausted == 1 and any(r.levelno == logging.CRITICAL for r in caplog.records)
    with pytest.raises(RuntimeError):
        ids.issue(0, "x", 1.0)                                              # sandbox semantics kept for issue()

    ids = W.RecipeIds(modulus=4)                                            # all closed 13 h ago: 12 h fallback
    for i in range(4):
        ids.by_id[ids.issue(0, i, 0.0)].status = "resulted"
    assert ids.try_issue(0, "y", 11 * 3600.0) is None
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.lanes.wire"):
        assert ids.try_issue(0, "y", 13 * 3600.0) == 0
    assert ids.n_fallback == 1 and any("fallback" in r.getMessage() for r in caplog.records)

    ids = W.RecipeIds(modulus=4)                                            # superseded, never acked as read
    R = W.ResultRouter(ids, manual_ids=None)
    rs = [ids.issue(1, i, 0.0) for i in range(4)]
    for r in rs[:3]:
        R.supersede(r, 600.0)
    R.on_read_ack((None, rs[1]), 700.0)                                     # rs[1] was read: not closed by age
    assert ids.try_issue(1, "z", 10 * 3600.0) is None
    got = ids.try_issue(1, "z", 13 * 3600.0)
    assert got in (rs[0], rs[2]) and ids.n_stale_closed == 2
    assert ids.by_id[rs[1]].status == "read" and ids.by_id[rs[3]].status == "issued"


def test_events_are_bounded():
    ids = W.RecipeIds(modulus=2)
    R = W.ResultRouter(ids, manual_ids=None)
    for i in range(W.EVENTS_KEEP + 50):
        R.on_read_ack((None, i % 2 + 500), float(i))                        # unknown ids: one event each
    assert len(R.events) == W.EVENTS_KEEP


# ── 7. import hygiene (REVIEW F12) ──────────────────────────────────────────────────
def test_module_is_stdlib_only_when_loaded_by_path():
    path = REPO / "src" / "lanes" / "wire.py"
    code = ("import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('lanes_wire_standalone', r'{path}')\n"
            "m = importlib.util.module_from_spec(spec); sys.modules['lanes_wire_standalone'] = m\n"
            "spec.loader.exec_module(m)\n"
            "print(int('torch' in sys.modules), int('numpy' in sys.modules), int('src' in sys.modules))\n"
            "m.grid24([0.5] * 2 + [0.0] * 8, [0.0] * 8 + [0.5] * 2)\n"
            "print(int('numpy' in sys.modules))\n")
    out = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["0", "0", "0", "1"]


def test_import_has_no_side_effects_on_random_simplex():
    import src.core.linebo as _lb
    import src.core.zombihop as _zh
    import src.utils.domain as _dom
    before = (_dom.random_simplex, _zh.random_simplex, _lb.random_simplex)
    fresh = _load_by_path(REPO / "src" / "lanes" / "wire.py", "_lanes_wire_fresh_copy")
    assert fresh.encode_slots(1.0, {}) == W.encode_slots(1.0, {})
    assert (_dom.random_simplex, _zh.random_simplex, _lb.random_simplex) == before
