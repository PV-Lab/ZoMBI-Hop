#!/usr/bin/env python3
"""
zombi_side_test.py - run this on the ZoMBI PC.

Hardware test of the UPDATED ZoMBI <-> DiSCO serial protocol ("lanes"): four recipe slots (front
main / front backup / back main / back backup), a rolling recipe id "r" echoed in every result
packet, and the optional read ack "k". Its partner is disco_side_test.py on the DiSCO PC.

It uses the SAME cable, port settings, framing and 1 Hz re-send rhythm as production, but the new
packet format, and runs a scripted plan (20 rounds by default, about 5 minutes, nothing is printed):
both rails read their own slots in the real cross-wire order, results come back with their id, and
the edge cases are injected on purpose (failed slide, dropped result, duplicate, out-of-order
result, cut frames, a packet without k, id wrap past 999).

Standalone: Python 3 standard library + pyserial. It imports no ZoMBI or DiSCO code and never opens
compositions.db, objective.db or any handshake file; the only file it writes is its log (temp
folder, or --log).

    python zombi_side_test.py                        # COM5, 20 rounds x 12 s, random seed
    python zombi_side_test.py --port COM5 --seed 4711

Every check prints [PASS] or [FAIL]; the run ends with GREEN LIGHT / RED LIGHT and exit code 0 / 1.
See README_HW_TEST.md.
"""
# ===================== EDIT HERE: serial port on THIS PC (the ZoMBI PC) =====================
SERIAL_PORT = "COM5"   # the ZoMBI <-> DiSCO cable, as in production (ZoMBI scripts/main.py:39)
# =============================================================================================

# ============================ SHARED BLOCK BEGIN ============================
# This block is byte-identical in zombi_side_test.py and disco_side_test.py.
# Both sides build the same scripted test plan from it, and the hello handshake
# compares a digest of that plan, so two different versions of the scripts
# refuse to run against each other. Edit both files together (run_dry_run.py
# checks that the two copies are identical).
import hashlib
import heapq
import json
import os
import queue
import random
import sys
import tempfile
import textwrap
import threading
import time

PROTO = "lanes-4slot-rid/1"
TEST_VERSION = "hwtest-4 (2026-11-13)"

# --- the physical link: the SAME settings as production ----------------------
BAUD = 115200                 # DiSCO main.py:66 zombi_baud; ZoMBI scripts/main.py:77
BYTESIZE, PARITY, STOPBITS = 8, "N", 1   # pyserial defaults, unchanged on both sides
#                               (DiSCO communication.py:5474, ZoMBI scripts/communication.py:1102)
RX_RESYNC_BYTES = 262144      # bytes without a newline before a receiver resyncs
#                               (DiSCO communication.py:149/2530, ZoMBI scripts/communication.py:1162)
SEND_PERIOD_S = 1.0           # both senders re-send their current packet every second
#                               (DiSCO main.py:68, ZoMBI scripts/main.py:79)

# --- the protocol under test (benchmarks/zoom_fix/lanes/result_id/RESULT_ID.md) ---
ID_MODULUS = 1000             # rolling recipe id 0..999
SLOT_NAMES = ("front main", "front backup", "back main", "back backup")
SLOT_RAIL = ("front", "front", "back", "back")
RAILS = ("front", "back")
RAIL_LETTER = {"front": "F", "back": "B"}
MAIN_SLOT = {"front": 0, "back": 2}
BACKUP_SLOT = {"front": 1, "back": 3}
ACTIVE_DIMS = (0, 2, 3, 4, 8, 9)   # a 6-component search, like run_69af / run_96a9
TODAY_GRID_ROWS = 24          # today's packet carries two 24-row grids

# --- the scripted plan (seconds; one "round" stands for one ~50 min rail cycle) ---
MIN_ROUNDS = 20
MIN_PERIOD = 12
LEAD_S = 2                    # result of read j is written 1.5 s AFTER the rail read j+1 (cross-wire)
PRINT_S = 1.0
LONG_PRINT_S = 4.2            # ZoMBI's next recipe lands while this print is still running
LOST_AFTER_ROUNDS = 2.5       # production: 90 min after the read ack (wire.LOST_AFTER_READ_S)
MIN_RESULT_ROUNDS = 0.3       # production: 30 min stale-id guard (wire.MIN_READ_TO_RESULT_S)
SPECIAL = {
    "long_print": (("front", 3), ("back", 11)),   # reply arrives while the slot is still printing
    "null": (("front", 5), ("back", 13)),         # failed slide: no packet at all
    "ooo": ("front", 9),                          # result sent after the back rail's later result
    "dup": ("back", 7),                           # same result re-delivered with a new ts
    "corrupt_obj": ("front", 11),                 # DiSCO cuts this frame once; 1 Hz re-send recovers
    "corrupt_comp": ("back", 16),                 # ZoMBI cuts its reply frame once
    "no_k": ("front", 17),                        # this result packet carries no read ack "k"
}                                                 # + "drop": the back rail's LAST result never comes

# Today's sizes, measured on real runs (result_id/measured_sizes.json, RESULT_ID.md sections 2 and 6)
TODAY_REF = {"comp_6d_B": 5675, "comp_6d_B_69af": 5782, "comp_10d_B": 10235,
             "obj_B": 3233, "obj_min_B": 1861, "obj_max_B": 3901, "lean4_6d_B": 918}


class PlanError(Exception):
    pass


# --- small helpers ------------------------------------------------------------
CHECK_TIMES = {"PASS": [], "FAIL": []}


def say(msg=""):
    if msg.startswith("[PASS]"):
        CHECK_TIMES["PASS"].append(time.monotonic())
    elif msg.startswith("[FAIL]"):
        CHECK_TIMES["FAIL"].append(time.monotonic())
    print(msg, flush=True)


def abort_criteria(hello_detail, reason, round_reached, rounds, last_rx, peer):
    """Verdict after the link was lost mid-run: name the root cause, and judge only what happened
    while the link was still up (the later checks cannot be judged)."""
    t_ok = (last_rx or 0.0) + 2.0
    n_pass = sum(1 for t in CHECK_TIMES["PASS"] if t <= t_ok)
    n_fail = sum(1 for t in CHECK_TIMES["FAIL"] if t <= t_ok)
    return [("link + versions", True, hello_detail),
            ("link stayed up", False, "%s; the run stopped at about round %d of %d, so the remaining checks could "
                                      "not be judged" % (reason, round_reached, rounds)),
            ("checks before that", n_fail == 0, "%d checks passed and %d failed while the link was up"
             % (n_pass, n_fail)),
            ("%s side" % peer, False, "its report never arrived (the link was lost)")]


def rail_txt(rail):
    return "%-5s" % rail


def median(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def wire_s(nbytes):
    """Seconds on the wire at 115200 baud 8N1 (10 bit-times per byte)."""
    return nbytes * 10.0 / BAUD


def _rng(seed, *parts):
    h = hashlib.sha1(("%s|" % (seed,) + "|".join(str(p) for p in parts)).encode("utf-8")).hexdigest()
    return random.Random(int(h[:15], 16))      # only .random() is used: identical on every Python 3


def check_id(x):
    if isinstance(x, bool) or not isinstance(x, int) or not (0 <= x < ID_MODULUS):
        raise ValueError("recipe id must be an integer 0..%d, got %r" % (ID_MODULUS - 1, x))
    return x


# --- framing: exactly today's (one JSON object per line) ---------------------
def dumps_line(pkt):
    return (json.dumps(pkt) + "\n").encode("utf-8")


def loads_line(raw):
    try:
        txt = bytes(raw).decode("utf-8", "ignore").strip()
        if not txt:
            return None
        pkt = json.loads(txt)
    except ValueError:
        return None
    return pkt if isinstance(pkt, dict) else None


# --- recipe content (both sides compute it, so each can check the other) -------
def plain_sum(xs):
    """Left-to-right float sum. Python 3.12 made sum() of floats compensated (Neumaier), so sum()
    can differ in the last digit between the DiSCO PC (Python 3.11) and the ZoMBI PC (Python 3.13).
    Both sides must build identical recipes and test plans, so neither uses the built-in sum() on floats."""
    tot = 0.0
    for x in xs:
        tot += x
    return tot


def recipe_raw(seed, label):
    """start/end of a recipe line as ZoMBI writes them: compositions of the 6 searched components (of
    10) on the simplex, rounded to 3 decimals (run_zombi_main._write_line_pair). This gives the same
    packet sizes as real lines (4-slot median ~930 B vs 918-938 B measured)."""
    rng = _rng(seed, "recipe", label)
    out = []
    for _ in range(2):
        w = [0.0] * 10
        for d in ACTIVE_DIMS:
            w[d] = 0.02 + 0.96 * rng.random()
        tot = plain_sum(w)
        out.append([round(x / tot, 3) for x in w])
    return out[0], out[1]


def normalize(raw):
    tot = plain_sum(raw)
    return [x / tot for x in raw]


def to_milli(raw):
    return [int(round(x * 1000.0)) for x in raw]


def from_milli(m):
    tot = sum(int(v) for v in m)
    if tot <= 0:
        return [0.1] * len(m)
    return [int(v) / tot for v in m]


def recipe_wire(seed, label, milli):
    """What ZoMBI puts on the wire for a recipe: normalised floats, or integer thousandths ("m": 1)."""
    a, b = recipe_raw(seed, label)
    if milli:
        return to_milli(a), to_milli(b)
    return normalize(a), normalize(b)


def decode_endpoints(a, b, milli):
    if milli:
        return from_milli(a), from_milli(b)
    return [float(x) for x in a], [float(x) for x in b]


def recipe_endpoints(seed, label, milli):
    """The start/end DiSCO must end up with for this recipe (what reaches the printer)."""
    a, b = recipe_wire(seed, label, milli)
    return decode_endpoints(a, b, milli)


def result_rows(start, end, n_rows):
    """The 'measured' rows DiSCO sends for a slide printed from the line start->end. They lie on
    that line, so ZoMBI can check that a result really came from the recipe its id names."""
    vals, comps = [], []
    for i in range(n_rows):
        f = (i + 0.5) / n_rows
        c = [s + f * (e - s) for s, e in zip(start, end)]
        vals.append(0.25 + 0.6 * (4.0 * f * (1.0 - f)) * (0.5 + 0.5 * c[0]) + 0.05 * c[2])
        comps.append(c)
    return vals, comps


def rows_hash(vals, comps):
    return hashlib.sha1(json.dumps([vals, comps]).encode("utf-8")).hexdigest()[:16]


# --- packets ------------------------------------------------------------------
def encode_slots(ts, slots4, milli):
    """ZoMBI -> DiSCO, four-slot full-state packet: {"type":"composition","ts":T,"s":[[slot,id,start,end],..]}."""
    s = [[k, int(e[0]), list(e[1]), list(e[2])] for k, e in enumerate(slots4) if e is not None]
    pkt = {"type": "composition", "ts": ts, "s": s}
    if milli:
        pkt["m"] = 1
    return dumps_line(pkt)


def decode_slots(pkt):
    """-> (ts, {slot: (id, start10, end10)}). Raises ValueError on a malformed packet."""
    ts = pkt.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        raise ValueError("missing or bad ts")
    milli = pkt.get("m") == 1
    ents = pkt.get("s")
    if not isinstance(ents, list):
        raise ValueError("missing slot list 's'")
    out = {}
    for ent in ents:
        if not (isinstance(ent, list) and len(ent) == 4):
            raise ValueError("slot entry is not [slot, id, start, end]")
        k, rid, a, b = ent
        if isinstance(k, bool) or k not in (0, 1, 2, 3) or k in out:
            raise ValueError("bad slot number %r" % (k,))
        rid = check_id(rid)
        if not (isinstance(a, list) and isinstance(b, list) and len(a) == 10 and len(b) == 10):
            raise ValueError("start/end must have 10 values")
        for v in a + b:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or (milli and not isinstance(v, int)):
                raise ValueError("bad composition value %r" % (v,))
        sa, sb = decode_endpoints(a, b, milli)
        out[k] = (rid, sa, sb)
    return ts, out


def encode_objective(vals, comps, ts, rid, k):
    """DiSCO -> ZoMBI: today's objective packet (communication.py:2936) + "r" (recipe id) + optional "k"
    ([last id read by the front rail, last id read by the back rail]); both appended last."""
    pkt = {"type": "objective", "values": vals, "comps": comps, "ts": ts}
    if rid is not None:
        pkt["r"] = rid
    if k is not None:
        pkt["k"] = [k[0], k[1]]
    return dumps_line(pkt)


def today_packet_bytes(seed, label_main, label_cache, ts):
    """Size of TODAY's two-slot composition packet (scripts/communication.py:888-897) for two recipes,
    built the way run_zombi_main._write_line_pair builds it (24-row grids, 3 dp, normalised)."""
    def part(label):
        a, b = recipe_raw(seed, label)
        grid = []
        for i in range(TODAY_GRID_ROWS):
            t = i / (TODAY_GRID_ROWS - 1.0)
            grid.append(normalize([round(x + t * (y - x), 3) for x, y in zip(a, b)]))
        return normalize(a), normalize(b), grid
    s, e, g = part(label_main)
    sc, ec, gc = part(label_cache)
    pkt = {"type": "composition", "ts": ts, "start": s, "end": e, "comps": g,
           "start_cache": sc, "end_cache": ec, "compositions_cache": gc}
    return len(dumps_line(pkt))


# --- DiSCO's four slots (reference rules of the lanes design, wire.DiscoSlots) ----
class Slots(object):
    """Replace a slot only when its id changed; a READ (printing) slot is never replaced, the new
    recipe is held until the print clears; a printed (CLEARED) slot keeps its id, so the 1 Hz
    re-sends cannot resurrect it; an UNREAD slot missing from a full-state packet is withdrawn.
    A rail reads its OWN main, else its OWN backup; never the other rail's slots."""

    def __init__(self):
        self.s = [self._empty() for _ in range(4)]

    @staticmethod
    def _empty(rid=None):
        return {"rid": rid, "a": None, "b": None, "state": "EMPTY", "pending": None}

    def apply(self, new_slots):
        acts = []
        for k in range(4):
            cur = self.s[k]
            new = new_slots.get(k)
            if new is None:
                if cur["state"] == "UNREAD":
                    self.s[k] = self._empty(cur["rid"])
                    acts.append((k, "withdrawn", cur["rid"]))
                continue
            rid, a, b = new
            if cur["rid"] == rid:
                acts.append((k, "keep_printed" if cur["state"] == "CLEARED" else "keep", rid))
            elif cur["state"] == "READ":
                if cur["pending"] is None or cur["pending"][0] != rid:
                    cur["pending"] = (rid, a, b)
                    acts.append((k, "held", rid))
            else:
                self.s[k] = {"rid": rid, "a": a, "b": b, "state": "UNREAD", "pending": None}
                acts.append((k, "replace", rid))
        return acts

    def read(self, rail):
        for k in (MAIN_SLOT[rail], BACKUP_SLOT[rail]):
            s = self.s[k]
            if s["state"] == "UNREAD":
                s["state"] = "READ"
                return k, s["rid"], s["a"], s["b"]
        return None

    def clear(self, rail):
        acts = []
        for k in (MAIN_SLOT[rail], BACKUP_SLOT[rail]):
            s = self.s[k]
            if s["state"] == "READ":
                if s["pending"] is not None:
                    rid, a, b = s["pending"]
                    self.s[k] = {"rid": rid, "a": a, "b": b, "state": "UNREAD", "pending": None}
                    acts.append((k, "released", rid))
                else:
                    s["state"] = "CLEARED"
                    acts.append((k, "cleared", s["rid"]))
        return acts


# --- ZoMBI's lane policy for the test (one line ahead per rail) -----------------
class LaneLogic(object):
    """start: fill all four slots. A routed result of rail X -> X's next main line. A read ack
    (or a result) showing that X's BACKUP recipe was read -> a fresh backup for X."""

    def __init__(self):
        self.slot = [None] * 4

    def start(self):
        return [(0, "init"), (1, "init"), (2, "init"), (3, "init")]

    def placed(self, slot, rid):
        self.slot[slot] = rid

    def on_read(self, rail, rid):
        b = BACKUP_SLOT[rail]
        if rid is not None and rid == self.slot[b]:
            return [(b, "backup refill")]
        return []

    def on_result(self, rail, rid):
        return self.on_read(rail, rid) + [(MAIN_SLOT[rail], "next main")]


# --- the scripted plan, built identically on both PCs from (seed, rounds, period, milli) ---
def _label(rail, rnd):
    return "%s-%02d" % (RAIL_LETTER[rail], rnd)


def build_plan(seed, rounds, period, milli):
    """Simulates the whole test (DiSCO rails + ZoMBI lanes) and returns what each side must see:
    every recipe ZoMBI issues (id, slot, label), every rail read, every result packet, the
    missing results and how ZoMBI must close them, the expected read acks, and the timing."""
    try:
        seed, N, P, milli = int(seed), int(rounds), int(period), 1 if milli else 0
    except (TypeError, ValueError):
        raise PlanError("seed / rounds / period must be integers")
    if N < MIN_ROUNDS:
        raise PlanError("--rounds must be at least %d (the edge cases need them)" % MIN_ROUNDS)
    if P < MIN_PERIOD or P % 2:
        raise PlanError("--period must be an even number of seconds, at least %d" % MIN_PERIOD)
    off = {"front": 0, "back": P // 2}

    def t_read(rail, j):
        return (j - 1) * P + off[rail] + 0.5

    def t_res(rail, j):
        return j * P + off[rail] + LEAD_S

    null = set(SPECIAL["null"])
    drop = ("back", N)
    long_print = set(SPECIAL["long_print"])
    writes = []
    for rail in RAILS:
        for j in range(1, N + 1):
            if (rail, j) in null or (rail, j) == drop:
                continue
            t = t_res(rail, j)
            if (rail, j) == SPECIAL["ooo"]:
                t = t_res("back", j) + 1
            writes.append({"t": float(t), "rail": rail, "round": j, "kind": "result",
                           "has_k": (rail, j) != SPECIAL["no_k"], "corrupt": (rail, j) == SPECIAL["corrupt_obj"],
                           "ooo": (rail, j) == SPECIAL["ooo"]})
            if (rail, j) == SPECIAL["dup"]:
                writes.append({"t": float(t + 2), "rail": rail, "round": j, "kind": "dup", "has_k": True,
                               "corrupt": False, "ooo": False})
    writes.sort(key=lambda w: w["t"])
    for a, b in zip(writes, writes[1:]):
        if b["t"] - a["t"] < (2 if a["corrupt"] else 1):
            raise PlanError("internal: two result packets less than a send period apart")

    heap = []
    cnt = [0]

    def push(t, order, kind, data):
        cnt[0] += 1
        heapq.heappush(heap, (t, order, cnt[0], kind, data))

    for rail in RAILS:
        for j in range(1, N + 1):
            push(t_read(rail, j), 1, "read", (rail, j))
    for w in writes:
        push(w["t"], 0, "write", w)
    t_last = writes[-1]["t"]
    for s in range(0, int(t_last + LOST_AFTER_ROUNDS * P) + 3):
        push(float(s), 2, "tick", None)

    slots, lane = Slots(), LaneLogic()
    zslot = [None] * 4
    issues, reads, arrivals = [], [], []
    read_of, reading = {}, {}
    replies = {}
    last_read = {"front": None, "back": None}
    zk = {"front": None, "back": None}
    acks = {"front": [], "back": []}
    table = [None]

    def zombi_issue(reqs, t, trigger):
        new = []
        for slot, why in reqs:
            i = len(issues)
            issues.append({"i": i, "slot": slot, "rail": SLOT_RAIL[slot], "why": why, "t": t, "trigger": trigger})
            lane.placed(slot, i)
            zslot[slot] = i
            new.append(i)
        if reqs:
            slots.apply({k: (zslot[k], None, None) for k in range(4) if zslot[k] is not None})
        return new

    zombi_issue(lane.start(), -1.0, "init")
    while heap:
        t, order, _, kind, data = heapq.heappop(heap)
        if kind == "read":
            rail, j = data
            got = slots.read(rail)
            if got is None:
                raise PlanError("internal: the %s rail has nothing to read in round %d" % (rail, j))
            ps = LONG_PRINT_S if (rail, j) in long_print else PRINT_S
            rd = {"rail": rail, "round": j, "t": t, "slot": got[0], "i": got[1], "print_s": ps, "held": None}
            reads.append(rd)
            read_of[(rail, j)] = rd
            reading[rail] = rd
            last_read[rail] = got[1]
            push(t + ps, 3, "clear", rail)
        elif kind == "clear":
            for k, act, i in slots.clear(data):
                if act == "released":
                    reading[data]["held"] = i
        elif kind == "write":
            rd = read_of[(data["rail"], data["round"])]
            table[0] = dict(data, i=rd["i"], first_tick=t, delivered=False)
        elif kind == "tick":
            tb = table[0]
            if tb is None or (tb["corrupt"] and t == tb["first_tick"]):
                continue
            if tb["has_k"]:
                for rail in RAILS:
                    if last_read[rail] is not None and last_read[rail] != zk[rail]:
                        zk[rail] = last_read[rail]
                        acks[rail].append(zk[rail])
                        zombi_issue(lane.on_read(rail, zk[rail]), t, "k")
            if not tb["delivered"]:
                tb["delivered"] = True
                arrivals.append(dict((x, tb[x]) for x in ("rail", "round", "kind", "i", "has_k", "corrupt", "ooo")))
                arrivals[-1]["t"] = t
                if tb["kind"] == "result":
                    new = zombi_issue(lane.on_result(tb["rail"], tb["i"]), t, "result")
                    mains = [i for i in new if issues[i]["slot"] == MAIN_SLOT[tb["rail"]]]
                    replies["%s:%d" % (tb["rail"], tb["round"])] = mains[0]

    # margins: every recipe must reach DiSCO well before the read that takes it
    for rd in reads:
        x = issues[rd["i"]]
        need = 3.0 if x["trigger"] == "result" else 2.0
        if x["t"] >= 0 and rd["t"] - x["t"] < need:
            raise PlanError("internal: recipe for %s round %d issued too late" % (rd["rail"], rd["round"]))

    label_of = {}
    for rd in reads:
        label_of[rd["i"]] = _label(rd["rail"], rd["round"])
    unused = {"front": 0, "back": 0}
    for x in issues:
        if x["i"] not in label_of:
            unused[x["rail"]] += 1
            label_of[x["i"]] = "%s-u%d" % (RAIL_LETTER[x["rail"]], unused[x["rail"]])
    id_start = (ID_MODULUS - len(issues) // 2) % ID_MODULUS

    def rid(i):
        return (id_start + i) % ID_MODULUS

    def n_rows(rail, j):
        return 11 + int(_rng(seed, "rows", rail, j).random() * 13)

    missing = []
    for rail, j in sorted(null) + [drop]:
        rd = read_of[(rail, j)]
        m = {"rail": rail, "round": j, "rid": rid(rd["i"]), "label": label_of[rd["i"]], "t": float(t_res(rail, j))}
        if (rail, j) == drop:
            m.update(why="dropped", closed_by="timeout", when=None)
        else:
            m.update(why="failed slide", closed_by="rail order", when="%s:%d" % (rail, j + 1))
        missing.append(m)

    plan = {
        "proto": PROTO, "test": TEST_VERSION, "seed": seed, "rounds": N, "period": P, "milli": milli,
        "id_start": id_start, "lost_after_s": LOST_AFTER_ROUNDS * P, "min_result_s": MIN_RESULT_ROUNDS * P,
        "t_last_write": t_last, "ooo_normal_t": float(t_res(*SPECIAL["ooo"])),
        "corrupt_comp_after": "%s:%d" % SPECIAL["corrupt_comp"],
        "issues": [{"n": x["i"], "rid": rid(x["i"]), "slot": x["slot"], "rail": x["rail"],
                    "label": label_of[x["i"]], "why": x["why"]} for x in issues],
        "reads": [{"rail": r["rail"], "round": r["round"], "t": r["t"], "slot": r["slot"], "rid": rid(r["i"]),
                   "label": label_of[r["i"]], "print_s": r["print_s"],
                   "held_rid": (rid(r["held"]) if (r["held"] is not None and r["print_s"] == LONG_PRINT_S) else None)}
                  for r in reads],
        "writes": [dict(w, rid=rid(read_of[(w["rail"], w["round"])]["i"]),
                        label=label_of[read_of[(w["rail"], w["round"])]["i"]],
                        n_rows=n_rows(w["rail"], w["round"])) for w in writes],
        "arrivals": [],
        "missing": missing,
        "replies": dict((k, rid(v)) for k, v in replies.items()),
        "acks": dict((rl, [rid(i) for i in v]) for rl, v in acks.items()),
    }
    for a in arrivals:
        d = dict((x, a[x]) for x in ("t", "rail", "round", "kind", "has_k", "corrupt", "ooo"))
        d.update(rid=rid(a["i"]), label=label_of[a["i"]], n_rows=n_rows(a["rail"], a["round"]))
        plan["arrivals"].append(d)
    # fingerprint of the content code too (recipe values, result rows, packet encoders), so two script
    # versions that differ only there also refuse to run against each other
    probe = []
    for x in plan["issues"][:4]:
        a, b = recipe_wire(seed, x["label"], milli)
        sa, sb = decode_endpoints(a, b, milli)
        probe.append([encode_slots(1.5, [(x["rid"], a, b)], milli).decode("utf-8"),
                      encode_objective(*(result_rows(sa, sb, 3) + (2.5, x["rid"], (1, 2)))).decode("utf-8"),
                      today_packet_bytes(seed, x["label"], x["label"], 1.5)])
    plan["content"] = hashlib.sha1(json.dumps(probe).encode("utf-8")).hexdigest()[:12]
    plan["digest"] = hashlib.sha1(json.dumps(plan, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return plan


def plan_overview(plan):
    P, N = plan["period"], plan["rounds"]
    sp = SPECIAL
    return [
        "Plan: %d rounds x %d s (one round = one ~50 min rail cycle), seed %d, plan id %s"
        % (N, P, plan["seed"], plan["digest"]),
        "  every round: each rail reads its own slot, prints, and sends the PREVIOUS slide's result",
        "  1.5 s after it read the next recipe (the cross-wire order of the real line)",
        "  edge cases: long print (reply held) F%02d/B%02d; failed slides F%02d/B%02d -> backup used;"
        % (sp["long_print"][0][1], sp["long_print"][1][1], sp["null"][0][1], sp["null"][1][1]),
        "  out-of-order F%02d; duplicate B%02d; cut frames F%02d (DiSCO) / B%02d reply (ZoMBI);"
        % (sp["ooo"][1], sp["dup"][1], sp["corrupt_obj"][1], sp["corrupt_comp"][1]),
        "  no 'k' on F%02d; id wrap 999->0 near round %d; back result B%02d dropped (timeout)."
        % (sp["no_k"][1], _wrap_round(plan), N),
        "  Expected duration: about %.0f min after both sides connect."
        % ((plan["t_last_write"] + plan["lost_after_s"] + 15) / 60.0),
    ]


def _wrap_round(plan):
    for x in plan["issues"]:
        if x["rid"] == 0:
            for r in plan["reads"]:
                if r["rid"] == 0:
                    return r["round"]
    return 0


# --- the serial port -----------------------------------------------------------
def list_ports_text():
    try:
        from serial.tools import list_ports
        ports = list(list_ports.comports())
    except Exception:
        return "(could not list the serial ports)"
    if not ports:
        return "(no serial ports found on this PC)"
    return ", ".join("%s (%s)" % (p.device, p.description) for p in sorted(ports, key=lambda p: p.device))


def open_port(port, timeout, write_timeout):
    """Open the port with production's settings. -> (ser, None) or (None, (kind, message))."""
    try:
        import serial
    except ImportError:
        return None, ("nopyserial", "pyserial is not installed for this Python (%s)" % sys.executable)
    try:
        ser = serial.serial_for_url(port, baudrate=BAUD, bytesize=BYTESIZE, parity=PARITY, stopbits=STOPBITS,
                                    timeout=timeout, write_timeout=write_timeout,
                                    xonxoff=False, rtscts=False, dsrdtr=False)
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if "permissionerror" in low or "access is denied" in low or "busy" in low or "in use" in low:
            kind = "busy"
        elif port.lower().startswith("socket://"):
            kind = "socket"
        elif ("filenotfounderror" in low or "cannot find" in low or "no such file" in low
              or "does not exist" in low):
            kind = "missing"
        else:
            kind = "other"
        return None, (kind, msg)
    try:
        ser.reset_input_buffer()
    except Exception:
        pass
    return ser, None


def close_port(ser, reader):
    if reader is not None:
        reader.stop.set()
    try:
        if hasattr(ser, "cancel_read"):
            ser.cancel_read()
    except Exception:
        pass
    try:
        ser.close()
    except Exception:
        pass


class LineReader(threading.Thread):
    """Production-style frame reassembly on '\\n'. style 'disco': drain in_waiting up to 64 KiB, else
    block for 1 byte (DiSCO communication.py:2497-2502); style 'zombi': read(256) with a 0.1 s
    timeout (ZoMBI scripts/communication.py:1152). Each line is queued with the time its first
    and last byte were read, so the receive time of every packet can be measured."""

    def __init__(self, ser, style):
        threading.Thread.__init__(self)
        self.daemon = True
        self.ser, self.style = ser, style
        self.q = queue.Queue()
        self.stop = threading.Event()
        self.rx_bytes = 0
        self.resyncs = 0
        self.errors = 0
        self.dead = False
        self.last_error = None
        self.last_rx = None

    def run(self):
        buf = bytearray()
        t_first = None
        consecutive = 0
        while not self.stop.is_set():
            try:
                if self.style == "disco":
                    w = self.ser.in_waiting
                    chunk = self.ser.read(min(w, 65536) if w > 0 else 1)
                else:
                    chunk = self.ser.read(256)
                consecutive = 0
            except Exception as e:
                if self.stop.is_set():
                    break
                self.errors += 1
                consecutive += 1
                self.last_error = repr(e)
                if consecutive >= 10:
                    self.dead = True
                    break
                time.sleep(0.2)
                continue
            if not chunk:
                if self.style == "zombi":
                    time.sleep(0.01)
                continue
            now = time.monotonic()
            self.rx_bytes += len(chunk)
            self.last_rx = now
            if not buf:
                t_first = now
            buf.extend(chunk)
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line = bytes(buf[:nl])
                del buf[:nl + 1]
                self.q.put((line, t_first if t_first is not None else now, now))
                t_first = now if buf else None
            if len(buf) > RX_RESYNC_BYTES:
                buf = bytearray()
                t_first = None
                self.resyncs += 1


class FrameWriter(object):
    """Production write paths. style 'disco' = ZombiLink.write_frame (communication.py:2122-2149): write
    timeout scaled to the frame, bounded flush, and a frame after a cut one starts with '\\n'.
    style 'zombi' = composition_sender (scripts/communication.py:903-904): write + flush."""

    def __init__(self, ser, style):
        self.ser, self.style = ser, style
        self.lock = threading.Lock()
        self.dangling = False
        self.errors = 0

    def frame(self, payload):
        with self.lock:
            data = b"\n" + payload if self.dangling else payload
            try:
                if self.style == "disco":
                    wt = max(len(data) * 10.0 / BAUD * 1.5, 1.0)
                    cur = self.ser.write_timeout
                    if cur is None or cur < wt:
                        self.ser.write_timeout = wt
                    self.ser.write(data)
                    deadline = time.monotonic() + max(wt, 1.0)
                    try:
                        while self.ser.out_waiting > 0:
                            if time.monotonic() > deadline:
                                raise IOError("flush: bytes still queued after %.1f s" % wt)
                            time.sleep(0.01)
                    except AttributeError:
                        pass                      # socket:// test link has no out_waiting
                else:
                    self.ser.write(data)
                    self.ser.flush()
            except Exception:
                self.dangling = True
                self.errors += 1
                raise
            self.dangling = False
            return len(data)

    def cut(self, payload):
        """Deliberately send only the first half of a frame (fault injection)."""
        with self.lock:
            self.ser.write(payload[:len(payload) // 2])
            try:
                self.ser.flush()
            except Exception:
                pass
            self.dangling = True


# --- console + log file ---------------------------------------------------------
class Tee(object):
    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "a", encoding="utf-8")
        self.at_line_start = True

    def write(self, s):
        try:
            self.stream.write(s)
        except Exception:
            pass
        try:
            for part in s.splitlines(True):
                if self.at_line_start:
                    self.fh.write(time.strftime("%H:%M:%S ") + "%.3f " % (time.time() % 1))
                self.fh.write(part)
                self.at_line_start = part.endswith("\n")
        except Exception:
            pass
        return len(s)

    def flush(self):
        for f in (self.stream, self.fh):
            try:
                f.flush()
            except Exception:
                pass

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


def default_log_path(side):
    return os.path.join(tempfile.gettempdir(), "%s_side_test_%s.log" % (side, time.strftime("%Y%m%d_%H%M%S")))


def banner(side, port, plan_lines, log_path):
    line = "=" * 78
    say(line)
    say(" ZoMBI <-> DiSCO serial protocol test - %s side          %s" % (side, TEST_VERSION))
    say(line)
    say(" Stop the DiSCO control program and the ZoMBI app first; this test only")
    say(" uses the serial cable and does not print.")
    say("")
    say(" Port %s at %d baud, %d%s%d, no flow control (the production settings)" % (port, BAUD, BYTESIZE, PARITY, STOPBITS))
    say(" Protocol under test: %s - four recipe slots (front main/backup," % PROTO)
    say("   back main/backup), a recipe id 'r' echoed in every result, read ack 'k'.")
    for x in plan_lines:
        say(" " + x)
    say(" Log file: %s" % log_path)
    say(" Each check prints [PASS] or [FAIL]; [INFO] lines only say what is happening.")
    say(" The last line says GREEN LIGHT (protocol works) or RED LIGHT (and what failed).")
    say(line)


def print_summary(side, plan, criteria, extra, failing_hint):
    """criteria: list of (title, ok, detail). Returns True for GREEN."""
    line = "=" * 78
    say("")
    say(line)
    desc = ("seed %d, %d rounds x %d s, plan %s" % (plan["seed"], plan["rounds"], plan["period"], plan["digest"])
            if plan else "no test plan (the hello handshake did not complete)")
    say("SUMMARY - %s side   (%s)" % (side, desc))
    say("-" * 78)
    for n, (title, ok, detail) in enumerate(criteria, 1):
        head = "[%s] %2d %-21s " % ("PASS" if ok else "FAIL", n, title)
        body = textwrap.wrap(detail, 78 - len(head)) or [""]
        say(head + body[0])
        for more in body[1:]:
            say(" " * len(head) + more)
    if extra:
        say("-" * 78)
        for x in extra:
            say(x)
    say(line)
    failing = ["%d %s" % (n, t) for n, (t, ok, _) in enumerate(criteria, 1) if not ok]
    if not failing and criteria:
        say("GREEN LIGHT: protocol works on this link (all %d checks passed on this side," % len(criteria))
        say("             including the other side's own report)")
    else:
        say("RED LIGHT: protocol test FAILED - failing: %s" % ("; ".join(failing) or "no checks ran"))
        if failing_hint:
            say("           " + failing_hint)
    say(line)
    return not failing and bool(criteria)


def explain_port_error(err, port, side):
    kind, msg = err
    say("[FAIL] could not open %s: %s" % (port, msg))
    if kind == "nopyserial":
        say("       Install it with:  \"%s\" -m pip install pyserial" % sys.executable)
    elif kind == "busy":
        say("       %s is in use by another program." % port)
        if side == "zombi":
            say("       1. Close the ZoMBI GUI (and any Manual Control / serial_only.py window).")
            say("       2. Free the port from the ZoMBI-Hop repo root:  python -m scripts.com_port %s" % port)
            say("          (add --status to only see which process holds it).")
            say("       3. Run this test again.")
        else:
            say("       1. Stop the DiSCO control program (Ctrl+C in its window; wait until it exits).")
            say("       2. Close any serial terminal that has %s open." % port)
            say("       3. If it is still busy: end the leftover python.exe in Task Manager, or unplug")
            say("          and replug the USB-serial adapter. Then run this test again.")
    elif kind == "missing":
        say("       %s does not exist on this PC (wrong port, or the USB-serial adapter is unplugged)." % port)
        say("       Ports on this PC: %s" % list_ports_text())
        say("       Run again with --port <the port of the ZoMBI<->DiSCO cable>.")
    elif kind == "socket":
        say("       This is the one-PC dry run: start loopback_relay.py first (or use run_dry_run.py).")
    else:
        say("       %s could not be opened (see the error above). Unplug and replug the USB-serial adapter," % port)
        say("       check the port name, then run this test again.")
        say("       Ports on this PC: %s" % list_ports_text())


def no_hello_diagnosis(reader, n_bad, seen_types, peer, waited):
    """What to tell the user when the other side never answered."""
    prod = [t for t in seen_types if t in ("composition", "objective", "selftest_ping", "selftest_pong")]
    out = []
    if reader.dead:
        out.append("The port stopped working (%s): the USB-serial adapter was unplugged or reset." % reader.last_error)
    elif reader.rx_bytes == 0:
        out.append("Nothing at all was received in %.0f s. Check that: (1) %s_side_test.py is running on the"
                   " other PC and shows 'waiting for the other side'; (2) the serial cable is plugged in at"
                   " both ends; (3) the port is right (ports on this PC: %s)." % (waited, peer, list_ports_text()))
    elif prod:
        out.append("The other side is running the PRODUCTION program (received %s packets). Stop it and"
                   " start %s_side_test.py instead." % ("/".join(sorted(prod)), peer))
    elif peer == "disco" and "hwt_hello" in seen_types:
        out.append("Only ZoMBI-side hellos arrived: the other PC is ALSO running zombi_side_test.py (run"
                   " disco_side_test.py there instead), or this port echoes back what it sends.")
    elif not seen_types:
        out.append("%d B arrived but no readable packet: the other side uses a different baud rate, or"
                   " another device is on this port." % reader.rx_bytes)
    else:
        out.append("Packets arrived (%s) but no answer to our hello: the other side does not hear us"
                   " (one-way link: check the cable's transmit wire / adapter)." % ", ".join(sorted(seen_types)))
    return out
# ============================= SHARED BLOCK END =============================


# ============================ ZoMBI SIDE ======================================
import argparse


class Router(object):
    """Standalone copy of the ZoMBI-side bookkeeping in result_id/wire.py (ResultRouter + RecipeIds,
    revised 2026-11-12, VERIFY.md M1): routes each result by its id "r"; orders a rail's reads only
    on evidence (the "k" acks); closes a missing result by rail order (FIFO) or by timeout from
    its ack; recognises a re-delivered table as a duplicate by (id, rows). Times are scaled."""

    def __init__(self, lost_after_s, min_result_s):
        self.lost_after, self.min_result = lost_after_s, min_result_s
        self.by_id = {}
        self.clock = 0
        self.last_read = {"front": None, "back": None}

    def _tick(self):
        self.clock += 1
        return self.clock

    def add(self, rid, slot, label, now):
        self.by_id[rid] = {"rid": rid, "slot": slot, "rail": SLOT_RAIL[slot], "label": label, "t_issue": now,
                           "seq_issue": self._tick(), "t_read": None, "seq_ack": None, "seq_sup": None,
                           "status": "issued", "hashes": [], "closed_by": None, "t_closed": None}

    def supersede(self, rid):
        r = self.by_id.get(rid)
        if r is not None and r["seq_sup"] is None:
            r["seq_sup"] = self._tick()

    def on_read_ack(self, k, now):
        if k is None:
            return
        for rail, rid in zip(RAILS, k):
            if rid is None or rid == self.last_read[rail]:
                continue
            self.last_read[rail] = rid
            r = self.by_id.get(rid)
            if r is None or r["seq_ack"] is not None:
                continue
            r["t_read"], r["seq_ack"] = now, self._tick()
            r["rail"] = rail
            if r["status"] in ("issued", "unprinted"):
                r["status"] = "read"
            for q in self.by_id.values():          # overwritten before any ack showed it read
                if (q["rail"] == rail and q["status"] == "issued" and q["seq_sup"] is not None
                        and r["seq_issue"] > q["seq_issue"] and q["seq_sup"] < r["seq_ack"]):
                    q["status"], q["closed_by"], q["t_closed"] = "unprinted", "unprinted", now

    @staticmethod
    def _before(q, r):
        if q["seq_ack"] is None:
            return False
        if r["seq_ack"] is not None:
            return q["seq_ack"] < r["seq_ack"]
        return q["seq_ack"] < r["seq_issue"]

    def on_objective(self, rid, vals, comps, k, now):
        self.on_read_ack(k, now)
        if rid is None:
            return {"cls": "legacy", "lost": []}
        r = self.by_id.get(rid)
        if r is None:
            return {"cls": "unknown", "lost": [], "why": "id never issued"}
        if len(vals) == 0:
            if r["status"] in ("issued", "read"):
                r["status"] = "null"
            return {"cls": "null", "rec": r, "lost": self._fifo_close(r, now)}
        h = rows_hash(vals, comps)
        if h in r["hashes"]:
            return {"cls": "duplicate", "rec": r, "lost": []}
        if r["hashes"]:
            r["hashes"].append(h)
            return {"cls": "reprint", "rec": r, "lost": []}
        if now - r["t_issue"] < self.min_result:
            return {"cls": "unknown", "lost": [], "why": "too soon after issue"}
        late = r["status"] in ("lost", "unprinted", "null")
        r["status"] = "resulted"
        r["hashes"].append(h)
        return {"cls": "late" if late else "matched", "rec": r, "lost": self._fifo_close(r, now)}

    def _fifo_close(self, r, now):
        lost = []
        for q in self.by_id.values():
            if q is r or q["rail"] != r["rail"] or q["status"] != "read":
                continue
            if self._before(q, r):
                q["status"], q["closed_by"], q["t_closed"] = "lost", "rail order", now
                lost.append(q)
        return [q["rid"] for q in sorted(lost, key=lambda q: q["seq_ack"])]

    def expire(self, now):
        lost = []
        for q in self.by_id.values():
            if q["status"] == "read" and q["t_read"] is not None and now - q["t_read"] > self.lost_after:
                q["status"], q["closed_by"], q["t_closed"] = "lost", "timeout", now
                lost.append(q["rid"])
        return lost


def cut_detail(ok, n_bad, resyncs, recovered, ok_text):
    """Summary text of the cut-frame check, naming the cause when it failed."""
    base = "unreadable frames: %d (1 injected); resyncs: %d" % (n_bad, resyncs)
    if ok:
        return "%s; %s" % (ok_text, base)
    why = []
    if n_bad > 1:
        why.append("%d frame(s) were CORRUPTED ON THE CABLE beyond the 1 injected cut (line noise, loose connector, "
                   "ground problem)" % (n_bad - 1))
    if n_bad == 0:
        why.append("the injected cut frame never arrived as an unreadable frame")
    if resyncs:
        why.append("the receiver had to drop %d runs of bytes without a newline" % resyncs)
    if not recovered:
        why.append("the packet after the cut frame was not recovered by the 1 Hz re-send")
    return "%s; %s" % ("; ".join(why) or "failed", base)


def parse_k(k):
    if not (isinstance(k, list) and len(k) == 2):
        raise ValueError("k must be [front_id, back_id]")
    return tuple(None if x is None else check_id(x) for x in k)


class ZombiTest(object):
    def __init__(self, ser, plan, args):
        self.ser, self.plan, self.args = ser, plan, args
        self.seed, self.milli = plan["seed"], plan["milli"]
        self.reader = LineReader(ser, "zombi")
        self.writer = FrameWriter(ser, "zombi")
        self.t0 = time.monotonic()
        self.issue_plan = plan["issues"]
        self.read_by_rid = dict((r["rid"], r) for r in plan["reads"])
        self.label_by_rid = dict((x["rid"], x["label"]) for x in plan["issues"])
        self.missing = dict((m["rid"], dict(m, closed=None)) for m in plan["missing"])
        # hello
        self.hello_ok = False
        self.hello_detail = "the hello handshake did not complete"
        self.hello_s = None
        # rounds state
        self.slots4 = [None] * 4
        self.comp_ts = None
        self.comp_frame = None
        self.cut_next = False
        self.cut_done = False
        self.lane = LaneLogic()
        self.router = Router(plan["lost_after_s"], plan["min_result_s"])
        self.next_id = plan["id_start"]
        self.n_issued = 0
        self.recipes = {}
        self.unplanned_issues = []
        self.wrapped_at = None
        self.wrap_noted = set()
        self.ai = 0
        self.arr_status = [None] * len(plan["arrivals"])
        self.extra_results = []
        self.classes = {}
        self.unexpected_closures = []
        self.k_last = {"front": None, "back": None}
        self.ack_seq = {"front": [], "back": []}
        self.bad_k = 0
        self.refills = []            # (rail, read rid, new rid, how)
        self.last_obj_ts = None
        self.obj_resends = 0
        self.bad_lines = 0
        self.bad_line_sizes = []
        self.unexpected_types = {}
        self.malformed = 0
        self.comp_sizes, self.today_sizes, self.today4_sizes = [], [], []
        self.comp_sends = 0
        self.obj_sizes, self.obj_rx, self.obj_r, self.obj_k = [], [], [], []
        self.peer_report = None
        self.got_bye = False
        self.aborted = None
        self.t_first_comp = None

    # ---------------------------------------------------------------- helpers
    def el(self):
        return time.monotonic() - self.t0

    def lab(self, rid):
        return self.label_by_rid.get(rid, "unplanned")

    def send_ctrl(self, pkt):
        try:
            self.writer.frame(dumps_line(pkt))
        except Exception as e:
            say("[INFO] write error on %s: %r" % (self.args.port, e))

    # ---------------------------------------------------------------- hello
    def hello(self):
        p = self.plan
        nonce = "%06x" % random.SystemRandom().getrandbits(24)
        pkt = {"type": "hwt_hello", "from": "zombi", "proto": PROTO, "test": TEST_VERSION, "seed": p["seed"],
               "rounds": p["rounds"], "period": p["period"], "milli": p["milli"], "plan": p["digest"],
               "nonce": nonce}
        say("[INFO] waiting for the DiSCO side (start disco_side_test.py on the DiSCO PC now) ...")
        t_start = time.monotonic()
        deadline = t_start + self.args.hello_timeout
        last_send, last_note = 0.0, t_start
        seen, n_bad = set(), 0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if self.reader.dead:
                break
            if now - last_send >= 0.5:
                self.send_ctrl(pkt)
                last_send = now
            if now - last_note >= 15.0:
                say("[INFO]   ... still waiting for the DiSCO side (%.0f s, %d B received so far)"
                    % (now - t_start, self.reader.rx_bytes))
                last_note = now
            try:
                raw, _, _ = self.reader.q.get(timeout=0.05)
            except queue.Empty:
                continue
            if not raw.strip():
                continue
            m = loads_line(raw)
            if m is None:
                n_bad += 1
                continue
            seen.add(str(m.get("type")))
            if m.get("type") != "hwt_hello_ack" or m.get("from") != "disco" or m.get("ack") != nonce:
                continue
            if not m.get("ok"):
                why = m.get("why") or "the DiSCO side refused the test"
                self.hello_detail = "VERSION MISMATCH: %s" % why
                say("[FAIL] the DiSCO side refused the test: %s" % why)
                say("       Copy the SAME version of both scripts (this folder) to both PCs.")
                return False
            for _ in range(3):
                self.send_ctrl({"type": "hwt_go", "from": "zombi", "ack": m.get("nonce")})
                time.sleep(0.15)
            self.hello_s = time.monotonic() - t_start
            self.hello_ok = True
            self.hello_detail = ("hello answered both ways after %.1f s; same protocol (%s), test version and "
                                 "test plan (%s) on both PCs" % (self.hello_s, PROTO, p["digest"]))
            say("[PASS] link up: the DiSCO side answered after %.1f s; same protocol %s, same test plan %s"
                % (self.hello_s, PROTO, p["digest"]))
            return True
        for x in no_hello_diagnosis(self.reader, n_bad, seen, "disco", time.monotonic() - t_start):
            say("[FAIL] " + x)
        self.hello_detail = "no answer from the DiSCO side within %.0f s" % self.args.hello_timeout
        return False

    # ---------------------------------------------------------------- issuing recipes
    def issue(self, slot, why, now):
        exp = self.issue_plan[self.n_issued] if self.n_issued < len(self.issue_plan) else None
        rid = self.next_id
        self.next_id = (rid + 1) % ID_MODULUS
        self.n_issued += 1
        if exp is not None and exp["rid"] == rid and exp["slot"] == slot and exp["why"] == why:
            label = exp["label"]
        else:
            label = "%s-x%d" % (RAIL_LETTER[SLOT_RAIL[slot]], rid)
            self.unplanned_issues.append(rid)
            say("[FAIL] ZoMBI issued r=%d for the %s slot (%s), but the plan expected %s"
                % (rid, SLOT_NAMES[slot], why, ("r=%d for the %s slot (%s)" % (exp["rid"], SLOT_NAMES[exp["slot"]],
                                                                                  exp["why"])) if exp else "no more recipes"))
        self.label_by_rid.setdefault(rid, label)
        a, b = recipe_wire(self.seed, label, self.milli)
        start, end = decode_endpoints(a, b, self.milli)
        old = self.slots4[slot]
        if old is not None:
            self.router.supersede(old[0])
        self.slots4[slot] = (rid, a, b)
        self.lane.placed(slot, rid)
        self.recipes[rid] = {"label": label, "slot": slot, "start": start, "end": end}
        self.router.add(rid, slot, label, now)
        self.comp_ts = time.time()
        self.comp_frame = None
        if rid == 0 and self.n_issued > 1 and self.wrapped_at is None:
            self.wrapped_at = self.n_issued
            say("[INFO] id wrap: the rolling recipe id went 999 -> 0 (recipe %s, %s slot)" % (label, SLOT_NAMES[slot]))
        return rid

    def send_composition(self):
        if self.comp_frame is None:
            self.comp_frame = encode_slots(self.comp_ts, self.slots4, self.milli)
            self.comp_sizes.append(len(self.comp_frame))
            labs = [self.lab(e[0]) if e else None for e in self.slots4]
            if labs[0] and labs[1]:
                tf = today_packet_bytes(self.seed, labs[0], labs[1], self.comp_ts)
                self.today_sizes.append(tf)
                if labs[2] and labs[3]:
                    self.today4_sizes.append(tf + today_packet_bytes(self.seed, labs[2], labs[3], self.comp_ts))
        if self.cut_next:
            self.cut_next = False
            self.cut_done = True
            try:
                self.writer.cut(self.comp_frame)
            except Exception:
                pass
            say("[INFO] round %02d %s: fault injection: ZoMBI's reply frame is CUT IN HALF once; DiSCO must drop it"
                " and take the full frame on the next 1 Hz send" % (int(self.plan["corrupt_comp_after"].split(":")[1]),
                                                                     rail_txt(self.plan["corrupt_comp_after"].split(":")[0])))
            return
        try:
            self.writer.frame(self.comp_frame)
            self.comp_sends += 1
        except Exception as e:
            say("[INFO] write error (composition packet): %r" % (e,))

    # ---------------------------------------------------------------- results
    def handle_k(self, k, now):
        changed = []
        for rail, rid in zip(RAILS, k):
            if rid is not None and rid != self.k_last[rail]:
                self.k_last[rail] = rid
                self.ack_seq[rail].append(rid)
                changed.append((rail, rid))
        self.router.on_read_ack(k, now)
        for rail, rid in changed:
            for slot, why in self.lane.on_read(rail, rid):
                new = self.issue(slot, why, now)
                self.refills.append((rail, rid, new, "k"))
                rd = self.read_by_rid.get(rid)
                planned_backup = rd is not None and rd["slot"] == BACKUP_SLOT[rail]
                say("[%s] round %02d %s: read ack k: the %s rail took its BACKUP r=%d (%s) -> refilled the backup"
                    " slot with r=%d (%s)" % ("PASS" if planned_backup else "FAIL", rd["round"] if rd else 0,
                                              rail_txt(rail), rail, rid, self.lab(rid), new, self.lab(new)))

    def on_objective(self, pkt, size, rx_s):
        now = time.monotonic()
        k = None
        if "k" in pkt:
            try:
                k = parse_k(pkt["k"])
            except ValueError as e:
                self.bad_k += 1
                say("[FAIL] malformed read ack k=%r: %s" % (pkt.get("k"), e))
        if k is not None:
            self.handle_k(k, now)                      # before the ts de-dup (k changes between re-sends)
        ts = pkt.get("ts")
        if ts == self.last_obj_ts:
            self.obj_resends += 1
            return
        self.last_obj_ts = ts
        vals, comps, rid = pkt.get("values"), pkt.get("comps"), pkt.get("r")
        if not (isinstance(vals, list) and isinstance(comps, list) and len(vals) == len(comps)):
            self.malformed += 1
            say("[FAIL] malformed result packet (%d B): values/comps missing or of different length" % size)
            return
        if rid is not None:
            try:
                check_id(rid)
            except ValueError as e:
                self.malformed += 1
                say("[FAIL] result packet with a malformed id: %s" % e)
                rid = None
        res = self.router.on_objective(rid, vals, comps, k, now)
        cls = res["cls"]
        self.classes[cls] = self.classes.get(cls, 0) + 1
        rec = res.get("rec")
        rail = rec["rail"] if rec else None
        new_main = None
        if cls in ("matched", "late"):
            reqs = self.lane.on_result(rail, rid)
            for slot, why in reqs:
                new = self.issue(slot, why, now)
                if why == "backup refill":
                    self.refills.append((rail, rid, new, "r"))
                else:
                    new_main = new
        self.check_arrival(cls, rid, rail, vals, comps, size, rx_s, k is not None, new_main)
        if cls == "matched":
            self.obj_sizes.append(size)
            self.obj_rx.append(rx_s)
            base = len(encode_objective(vals, comps, ts, None, None))
            r_only = len(encode_objective(vals, comps, ts, rid, None))
            self.obj_r.append(r_only - base)
            if k is not None:
                self.obj_k.append(size - r_only)
        for lost in res.get("lost", []):
            self.note_closure(lost, "rail order", rid)

    def check_arrival(self, cls, rid, rail, vals, comps, size, rx_s, has_k, new_main):
        arr = self.plan["arrivals"]
        want = "dup" if cls == "duplicate" else "result"
        exp = arr[self.ai] if self.ai < len(arr) else None
        idx = None
        if exp is not None and exp["rid"] == rid and exp["kind"] == want:
            idx = self.ai
        else:
            for j in range(self.ai, len(arr)):
                if arr[j]["rid"] == rid and arr[j]["kind"] == want:
                    idx = j
                    break
            if exp is not None:
                say("[FAIL] round %02d %s: expected r=%d (%s), got r=%s (%s) [%s]"
                    % (exp["round"], rail_txt(exp["rail"]), exp["rid"], exp["label"], rid,
                       self.lab(rid) if rid is not None else "no id", cls))
            else:
                say("[FAIL] unexpected extra result r=%s (%s) [%s]" % (rid, self.lab(rid), cls))
            if idx is None:
                self.extra_results.append((rid, cls))
                return
            for j in range(self.ai, idx):
                if self.arr_status[j] is None:
                    self.arr_status[j] = "missing"
        a = arr[idx]
        self.ai = idx + 1
        if a["kind"] == "dup":
            ok = cls == "duplicate"
            self.arr_status[idx] = "ok" if ok else "bad"
            say("[%s] round %02d %s: the same result re-delivered with a new ts (r=%d) -> %s"
                % ("PASS" if ok else "FAIL", a["round"], rail_txt(a["rail"]), rid,
                   "recognised as a duplicate, not counted twice" if ok else "classed %s" % cls))
            return
        notes = []
        ok = cls == "matched"
        if not ok:
            notes.append("router class %s" % cls)
        r = self.recipes.get(rid)
        if r is not None:
            ev, ec = result_rows(r["start"], r["end"], a["n_rows"])
            if not (ev == vals and ec == comps):
                ok = False
                notes.append("the rows do NOT lie on recipe %s's line (DiSCO printed another recipe?)" % r["label"])
        rd = self.read_by_rid.get(rid)
        if rd is not None and rd["slot"] == BACKUP_SLOT[a["rail"]]:
            notes.append("printed from the backup slot")
        if a["ooo"]:
            notes.append("arrived after back round %02d's result: out of read order" % a["round"])
        if not has_k:
            notes.append("this packet carried no k (k is optional)")
        if a["corrupt"]:
            notes.append("full frame after a cut one (recovered by the 1 Hz re-send)")
        if self.wrapped_at is not None and rid < self.plan["id_start"] and a["rail"] not in self.wrap_noted:
            self.wrap_noted.add(a["rail"])
            notes.append("first %s result with an id after the 999->0 wrap" % a["rail"])
        self.arr_status[idx] = "ok" if ok else "bad"
        say("[%s] round %02d %s: result r=%d matched recipe %s (%d B, rx %.2f s)%s"
            % ("PASS" if ok else "FAIL", a["round"], rail_txt(a["rail"]), rid, a["label"], size, rx_s,
               "".join(" - " + n for n in notes)))
        key = "%s:%d" % (a["rail"], a["round"])
        if new_main is not None and key == self.plan["corrupt_comp_after"]:
            self.cut_next = True

    def note_closure(self, rid, how, by_rid):
        rec = self.router.by_id.get(rid)
        m = self.missing.get(rid)
        if m is None:
            self.unexpected_closures.append((rid, how))
            say("[FAIL] r=%d (%s) was closed as lost by %s, but its result was still expected"
                % (rid, self.lab(rid), how))
            return
        m["closed"] = how
        if how == "rail order":
            ok = m["closed_by"] == how and by_rid is not None and m["when"] == "%s:%d" % (
                m["rail"], self.read_by_rid.get(by_rid, {}).get("round", -1))
            say("[%s] round %02d %s: no result for r=%d (%s: %s, no packet) -> closed as lost by rail order when"
                " r=%d (%s) arrived" % ("PASS" if ok else "FAIL", m["round"], rail_txt(m["rail"]), rid, m["label"],
                                         m["why"], by_rid, self.lab(by_rid)))
        else:
            ok = m["closed_by"] == how
            waited = (rec["t_closed"] - rec["t_read"]) if rec and rec["t_read"] else float("nan")
            say("[%s] round %02d %s: no result for r=%d (%s: %s) -> closed as lost by timeout, %.1f s after its"
                " read was acknowledged" % ("PASS" if ok else "FAIL", m["round"], rail_txt(m["rail"]), rid,
                                            m["label"], m["why"], waited))
        m["ok"] = ok

    # ---------------------------------------------------------------- main loop
    def handle_line(self, raw, t_first, t_end):
        if not raw.strip():
            return
        pkt = loads_line(raw)
        if pkt is None:
            self.bad_lines += 1
            self.bad_line_sizes.append(len(raw))
            if self.bad_lines == 1:
                say("[INFO] dropped an unreadable frame (%d B, cut or corrupted)" % len(raw))
            else:
                say("[FAIL] dropped an unreadable frame (%d B): %d so far, but only 1 cut frame is injected on purpose"
                    " -> a frame was CORRUPTED ON THE CABLE (line noise, loose connector, ground problem)"
                    % (len(raw), self.bad_lines))
            return
        typ = pkt.get("type")
        if typ == "objective":
            self.on_objective(pkt, len(raw) + 1, t_end - t_first)
        elif typ == "hwt_report" and pkt.get("from") == "disco":
            if self.peer_report is None:
                say("[INFO] the DiSCO side finished its schedule and sent its report")
            self.peer_report = pkt
        elif typ == "hwt_bye":
            self.got_bye = True
        elif typ in ("hwt_hello_ack", "hwt_go", "hwt_hello"):
            pass
        else:
            self.unexpected_types[str(typ)] = self.unexpected_types.get(str(typ), 0) + 1

    def all_resolved(self):
        return (self.ai >= len(self.plan["arrivals"])
                and all(m["closed"] is not None for m in self.missing.values()))

    def rounds(self):
        p = self.plan
        now = time.monotonic()
        for slot, why in self.lane.start():
            self.issue(slot, why, now)
        say("[INFO] initial recipes written to all four slots: %s"
            % ", ".join("%s r=%d (%s)" % (SLOT_NAMES[k], e[0], self.lab(e[0])) for k, e in enumerate(self.slots4)))
        self.t_first_comp = time.monotonic()
        deadline = self.t_first_comp + p["t_last_write"] + p["lost_after_s"] + 60
        next_tick = time.monotonic()
        next_expire = 0.0
        last_rx = time.monotonic()
        rx_seen = self.reader.rx_bytes
        while True:
            now = time.monotonic()
            while True:
                try:
                    raw, t1, t2 = self.reader.q.get_nowait()
                except queue.Empty:
                    break
                self.handle_line(raw, t1, t2)
            now = time.monotonic()
            if now >= next_tick:
                self.send_composition()
                next_tick += SEND_PERIOD_S
                if next_tick < now - 3:
                    next_tick = now + SEND_PERIOD_S
            if now >= next_expire:
                next_expire = now + 0.5
                for rid in self.router.expire(now):
                    self.note_closure(rid, "timeout", None)
            if self.reader.rx_bytes != rx_seen:
                rx_seen = self.reader.rx_bytes
                last_rx = now
            if self.reader.dead:
                self.aborted = "the serial port stopped working (%s): adapter unplugged?" % self.reader.last_error
                break
            live_limit = 10.0 if self.obj_resends or self.classes else max(25.0, p["period"] + 15.0)
            if now - last_rx > live_limit:
                self.aborted = ("link lost: nothing received from the DiSCO side for %.0f s (cable unplugged,"
                                " or disco_side_test.py stopped)" % (now - last_rx))
                break
            if self.peer_report is not None and self.all_resolved():
                break
            if now > deadline:
                break
            time.sleep(0.005)
        if self.aborted:
            say("[FAIL] " + self.aborted)

    # ---------------------------------------------------------------- verdict
    def criteria(self):
        p = self.plan
        arr = p["arrivals"]
        if not self.hello_ok:
            return [("link + versions", False, self.hello_detail)]
        if self.aborted:
            t0 = (self.t_first_comp or time.monotonic()) + 1.0
            rnd = int(((self.reader.last_rx or t0) - t0) / p["period"]) + 1
            return abort_criteria(self.hello_detail, self.aborted, max(1, min(rnd, p["rounds"])), p["rounds"],
                                  self.reader.last_rx, "DiSCO")
        C = []
        C.append(("link + versions", self.hello_ok, self.hello_detail))
        res_idx = [i for i, a in enumerate(arr) if a["kind"] == "result"]
        n_ok = sum(1 for i in res_idx if self.arr_status[i] == "ok")
        C.append(("results routed", n_ok == len(res_idx) and not self.extra_results,
                  "%d/%d results matched to the recipe their id names, rows on that recipe's line%s"
                  % (n_ok, len(res_idx), "; %d unexpected results" % len(self.extra_results)
                     if self.extra_results else "")))
        ooo = [i for i, a in enumerate(arr) if a["ooo"]]
        C.append(("out-of-order result", all(self.arr_status[i] == "ok" for i in ooo),
                  "front round %02d's result arrived after back round %02d's (out of read order) and was routed "
                  "by its id" % (SPECIAL["ooo"][1], SPECIAL["ooo"][1])
                  if all(self.arr_status[i] == "ok" for i in ooo) else "the out-of-order result was not routed"))
        fs = [m for m in self.missing.values() if m["why"] == "failed slide"]
        fs_ok = all(m.get("ok") for m in fs)
        bk_reads = [r for r in p["reads"] if r["slot"] in (1, 3) and r["round"] > 2]
        refill_rids = set(x[1] for x in self.refills)
        bk_ok = all(r["rid"] in refill_rids for r in bk_reads)
        bk_res = [i for i, a in enumerate(arr) if a["kind"] == "result" and any(
            a["rid"] == r["rid"] for r in bk_reads)]
        bk_res_ok = all(self.arr_status[i] == "ok" for i in bk_res)
        C.append(("failed slide + backup", fs_ok and bk_ok and bk_res_ok,
                  "%d/%d failed slides (no packet) closed as lost by rail order; the rail then printed its backup "
                  "(%d/%d backups refilled, %d/%d backup results routed)"
                  % (sum(1 for m in fs if m.get("ok")), len(fs), sum(1 for r in bk_reads if r["rid"] in refill_rids),
                     len(bk_reads), sum(1 for i in bk_res if self.arr_status[i] == "ok"), len(bk_res))))
        dr = [m for m in self.missing.values() if m["why"] == "dropped"]
        C.append(("dropped result", all(m.get("ok") for m in dr),
                  "back round %02d's result never came; closed as lost by the timeout (%.0f s after its read ack)"
                  % (p["rounds"], p["lost_after_s"]) if all(m.get("ok") for m in dr)
                  else "the dropped result was not closed by the timeout"))
        du = [i for i, a in enumerate(arr) if a["kind"] == "dup"]
        du_ok = all(self.arr_status[i] == "ok" for i in du)
        C.append(("duplicate result", du_ok, "the re-delivered result (new ts, same id and rows) was recognised"
                  " as a duplicate and not counted twice" if du_ok else "the duplicate was not recognised"))
        post = [i for i in res_idx if arr[i]["rid"] < p["id_start"]]
        wrap_ok = self.wrapped_at is not None and all(self.arr_status[i] == "ok" for i in post) and post
        C.append(("id wrap 999 -> 0", bool(wrap_ok),
                  "ids wrapped past 999; %d/%d results with wrapped ids routed"
                  % (sum(1 for i in post if self.arr_status[i] == "ok"), len(post))))
        ack_ok = all(self.ack_seq[r] == p["acks"][r] for r in RAILS) and self.bad_k == 0
        nok = [i for i, a in enumerate(arr) if not a["has_k"]]
        nok_ok = all(self.arr_status[i] == "ok" for i in nok)
        C.append(("read acks (k)", ack_ok and nok_ok,
                  "k listed every read in order (front %d/%d, back %d/%d acks as planned); a packet without k was "
                  "still routed" % (len(self.ack_seq["front"]), len(p["acks"]["front"]),
                                    len(self.ack_seq["back"]), len(p["acks"]["back"]))
                  if ack_ok and nok_ok else "acks seen front %s / back %s, planned front %s / back %s"
                  % (self.ack_seq["front"], self.ack_seq["back"], p["acks"]["front"], p["acks"]["back"])))
        cut = [i for i, a in enumerate(arr) if a["corrupt"]]
        cut_ok = self.bad_lines == 1 and all(self.arr_status[i] == "ok" for i in cut) and self.reader.resyncs == 0
        C.append(("cut frame recovery", cut_ok, cut_detail(
            cut_ok, self.bad_lines, self.reader.resyncs, all(self.arr_status[i] == "ok" for i in cut),
            "DiSCO's deliberately cut frame was dropped and the full packet arrived on the next 1 Hz send")))
        odd = dict((c, n) for c, n in self.classes.items() if c not in ("matched", "duplicate"))
        nothing = (not odd and not self.unplanned_issues and not self.unexpected_closures and not self.unexpected_types
                   and not self.malformed and self.writer.errors == 0 and self.n_issued == len(p["issues"]))
        C.append(("nothing unexpected", nothing,
                  "issued %d/%d recipes exactly as planned; no unknown/late/legacy results, no wrong closures, no "
                  "stray packet types, no write errors" % (self.n_issued, len(p["issues"])) if nothing else
                  "odd classes %s, unplanned issues %s, wrong closures %s, stray types %s, malformed %d, write "
                  "errors %d, issued %d/%d" % (odd, self.unplanned_issues, self.unexpected_closures,
                                               self.unexpected_types, self.malformed, self.writer.errors,
                                               self.n_issued, len(p["issues"]))))
        comp_max = max(self.comp_sizes) if self.comp_sizes else 0
        rx_max = max(self.obj_rx) if self.obj_rx else 0.0
        t_ok = bool(self.comp_sizes) and wire_s(comp_max) < 0.5 and rx_max < 1.0
        C.append(("packet size + timing", t_ok,
                  "4-slot recipe packet max %d B = %.0f ms on the wire per 1 s period; result packets received in at"
                  " most %.2f s" % (comp_max, 1000 * wire_s(comp_max), rx_max)))
        if self.peer_report is None:
            C.append(("DiSCO side", False, "DiSCO's report never arrived (link lost, or the DiSCO script stopped)"))
        else:
            pc = self.peer_report.get("criteria") or []
            bad = [c[0] for c in pc if not c[1]]
            C.append(("DiSCO side", bool(pc) and not bad,
                      "all %d DiSCO-side checks passed" % len(pc) if pc and not bad
                      else "DiSCO-side checks FAILED: %s (see the DiSCO window)" % ", ".join(bad)))
        return C

    def stats_lines(self):
        out = ["Packet sizes on this link (bytes; wire time at 115200 baud 8N1):"]
        if self.comp_sizes:
            cs = self.comp_sizes
            out.append("  4-slot recipe packet ZoMBI->DiSCO: median %d B (min %d, max %d) = %.0f ms; %d sends"
                       % (median(cs), min(cs), max(cs), 1000 * wire_s(median(cs)), self.comp_sends))
        if self.today_sizes:
            tm = median(self.today_sizes)
            out.append("    today's 2-slot packet for the same front lines: median %d B (%.0f ms); new packet is %.0f %%"
                       " smaller" % (tm, 1000 * wire_s(tm), 100.0 * (1 - median(self.comp_sizes) / tm)))
        if self.today4_sizes:
            t4 = median(self.today4_sizes)
            out.append("    four slots in today's layout would be ~%d B (%.2f s on the wire per 1 s period)"
                       % (t4, wire_s(t4)))
        out.append("    measured on real runs: today %d B (run_96a9) / %d B (run_69af), 10 dims %d B; lean 4-slot %d B"
                   % (TODAY_REF["comp_6d_B"], TODAY_REF["comp_6d_B_69af"], TODAY_REF["comp_10d_B"],
                      TODAY_REF["lean4_6d_B"]))
        if self.obj_sizes:
            os_ = self.obj_sizes
            out.append("  result packet DiSCO->ZoMBI: median %d B (min %d, max %d) = %.0f ms; r adds %d-%d B, k adds %s B"
                       % (median(os_), min(os_), max(os_), 1000 * wire_s(median(os_)), min(self.obj_r),
                          max(self.obj_r), ("%d-%d" % (min(self.obj_k), max(self.obj_k))) if self.obj_k else "0"))
            out.append("    today's result packet (no r/k), real runs: median %d B (%d-%d)"
                       % (TODAY_REF["obj_B"], TODAY_REF["obj_min_B"], TODAY_REF["obj_max_B"]))
            out.append("  receive time per result packet (first byte -> end, incl. the 0.1 s read timeout):"
                       " median %.2f s, max %.2f s" % (median(self.obj_rx), max(self.obj_rx)))
        if self.peer_report is not None:
            st = self.peer_report.get("stats") or {}
            if st.get("rtt"):
                out.append("  round trip measured by DiSCO (result sent -> ZoMBI's next recipe back, incl. up to 1 s"
                           " waiting for ZoMBI's 1 Hz send): median %.2f s, max %.2f s"
                           % (median(st["rtt"]), max(st["rtt"])))
            if st.get("comp_rx_med") is not None:
                out.append("  receive time per recipe packet at DiSCO: median %.3f s, max %.3f s"
                           % (st["comp_rx_med"], st["comp_rx_max"]))
        out.append("  re-sends: %d recipe packets sent, %d result re-sends received (ignored by ts, as today)"
                   % (self.comp_sends, self.obj_resends))
        return out

    def finish(self):
        C = self.criteria()
        report = {"type": "hwt_report", "from": "zombi", "ok": all(c[1] for c in C),
                  "criteria": [[c[0], bool(c[1]), c[2][:300]] for c in C]}
        if self.hello_ok and not self.aborted:
            t_end = time.monotonic() + 10.0
            last = 0.0
            while time.monotonic() < t_end and not self.got_bye:
                if time.monotonic() - last >= 1.0:
                    self.send_ctrl(report)
                    last = time.monotonic()
                try:
                    raw, t1, t2 = self.reader.q.get(timeout=0.05)
                    pkt = loads_line(raw)
                    if pkt is not None and pkt.get("type") == "hwt_bye":
                        self.got_bye = True
                    elif pkt is not None and pkt.get("type") == "hwt_report" and pkt.get("from") == "disco":
                        self.peer_report = pkt
                except queue.Empty:
                    pass
            if not self.got_bye:
                say("[INFO] the DiSCO side did not confirm it received this side's report; check its window")
        hint = ("Send back both log files (this one and the DiSCO one)." if self.hello_ok
                else "Fix the link first (see the [FAIL] lines above), then run both scripts again.")
        if not self.hello_ok and self.hello_detail.startswith("VERSION MISMATCH"):
            hint = "Copy the SAME version of both scripts (one folder) to both PCs, then run both again."
        if self.aborted:
            hint = ("The link dropped mid-run: check the cable/adapter and that neither script was stopped, then run"
                    " both again. Send back both logs.")
        green = print_summary("ZoMBI", self.plan, C, self.stats_lines() if self.hello_ok else [], hint)
        return 0 if green else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="ZoMBI-side hardware test of the lanes serial protocol "
                                             "(four recipe slots + recipe id). Run disco_side_test.py on the DiSCO PC.")
    ap.add_argument("--port", default=SERIAL_PORT, help="serial port of the ZoMBI<->DiSCO cable (default %s, "
                                                        "set by SERIAL_PORT at the top of this file)" % SERIAL_PORT)
    ap.add_argument("--rounds", type=int, default=20, help="rounds (rail cycles), at least 20 (default 20)")
    ap.add_argument("--period", type=int, default=12, help="seconds per round, even, at least 12 (default 12)")
    ap.add_argument("--seed", type=int, default=None, help="test-plan seed (default: random, printed)")
    ap.add_argument("--milli", action="store_true", help="send start/end as integer thousandths (\"m\": 1)")
    ap.add_argument("--hello-timeout", type=float, default=300.0, help="seconds to wait for the DiSCO side (300)")
    ap.add_argument("--log", default=None, help="log file (default: a new file in the temp folder)")
    args = ap.parse_args(argv)
    seed = args.seed if args.seed is not None else random.SystemRandom().randint(1, 99999)
    log_path = args.log or default_log_path("zombi")
    old_stdout = sys.stdout
    tee = Tee(old_stdout, log_path)
    sys.stdout = tee
    try:
        try:
            plan = build_plan(seed, args.rounds, args.period, args.milli)
        except PlanError as e:
            say("[FAIL] %s" % e)
            say("RED LIGHT: the test could not start (bad options)")
            return 1
        banner("ZoMBI", args.port, plan_overview(plan), log_path)
        ser, err = open_port(args.port, timeout=0.1, write_timeout=5.0)   # scripts/communication.py:1102
        if err:
            explain_port_error(err, args.port, "zombi")
            print_summary("ZoMBI", plan, [("open the port", False, err[1])], [],
                          "Fix the port problem above, then run the test again.")
            return 1
        t = ZombiTest(ser, plan, args)
        t.reader.start()
        rc = 1
        try:
            if t.hello():
                t.rounds()
            rc = t.finish()
        except KeyboardInterrupt:
            say("[FAIL] interrupted with Ctrl+C")
            t.aborted = "interrupted"
            C = [("test completed", False, "interrupted with Ctrl+C before the end")]
            print_summary("ZoMBI", plan, C, [], "Run both scripts again and let them finish (about 5 min).")
            rc = 1
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stdout)
            say("[FAIL] the test script itself crashed (traceback above): %r" % (e,))
            print_summary("ZoMBI", plan, [("test completed", False, "the test script crashed: %r" % (e,))], [],
                          "A bug in the TEST script, not a protocol verdict. Send back both log files.")
            rc = 1
        finally:
            close_port(ser, t.reader)
        say("Log file: %s" % log_path)
        return rc
    finally:
        sys.stdout = old_stdout
        tee.close()


if __name__ == "__main__":
    sys.exit(main())
