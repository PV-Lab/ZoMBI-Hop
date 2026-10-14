"""Initialization print queue (src/utils/init_queue.py): bookkeeping rules, and a
no-deadlock check against the discrete-event DiSCO simulation
(benchmarks/axial_init/disco_sim.py) across dimensions and failure modes."""

import json

import numpy as np
import pytest

from src.utils.axial_init import axial_design, sample_trace
from src.utils.init_queue import InitQueue


def _print(q, i, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = sample_trace(q.traces[i], 24) + rng.normal(0, noise, (24, len(q.traces[i].start)))
    x = np.clip(x, 0, None)
    return x / x.sum(1, keepdims=True)


@pytest.fixture()
def q6():
    return InitQueue(axial_design(6).traces, max_sends=3)


def test_pairs_come_out_in_queue_order(q6):
    assert q6.next_pair(0) == (0, 1)
    assert q6.next_pair(1) == (2, 3)
    assert q6.outstanding() == [0, 1, 2, 3]
    assert list(q6.queue) == [4, 5]


def test_result_matches_its_trace_and_completes(q6):
    q6.next_pair(0)
    ev = q6.on_result(_print(q6, 0, 0.02), np.ones(24), 10)
    assert ev["kind"] == "matched" and ev["trace"] == 0 and ev["distance"] < 0.05
    assert q6.status[0] == "done" and q6.status[1] == "sent" and not ev["requeued"]


def test_overwritten_cache_is_presumed_lost_and_requeued(q6):
    q6.next_pair(0)            # write 0: (0, 1)
    q6.next_pair(1)            # write 1: (2, 3) - suppose it replaced an unread 1
    q6.on_result(_print(q6, 0), np.ones(24), 10)
    ev = q6.on_result(_print(q6, 2), np.ones(24), 11)
    assert ev["requeued"] == [1]
    assert q6.queue[0] == 1 and q6.status[1] == "queued"
    assert q6.next_pair(12)[0] == 1          # re-sent first


def test_cache_result_implies_main_was_printed(q6):
    q6.next_pair(0)
    ev = q6.on_result(_print(q6, 1), np.ones(24), 10)   # cache reports, main silent
    assert ev["requeued"] == [0]


def test_stall_requeues_then_abandons_after_max_sends(q6):
    q = InitQueue(axial_design(4).traces, max_sends=2)
    q.next_pair(0)
    q.next_pair(0)
    assert q.on_stall(5)["requeued"] == [0, 1, 2, 3]
    q.next_pair(6)
    q.next_pair(6)
    ev = q.on_stall(9)
    assert sorted(ev["abandoned"]) == [0, 1, 2, 3] and q.complete()


def test_foreign_results_are_ignored(q6):
    q6.next_pair(0)
    far = np.tile([0.05, 0.05, 0.05, 0.05, 0.4, 0.4], (24, 1))
    assert q6.on_result(far, np.ones(24), 5)["kind"] == "foreign"
    # lies on a trace that has not been sent yet: cannot be one of ours
    assert q6.on_result(_print(q6, 4), np.ones(24), 6)["kind"] == "foreign"
    assert q6.status[:2] == ["sent", "sent"] and not q6.result_times


def test_unevenly_spaced_partial_print_still_matches(q6):
    """Print lag bunches a gradient's first rows together; a result that kept only
    those rows covers a sliver of the trace but is still that trace."""
    q6.next_pair(0)
    t = q6.traces[1]
    tt = np.array([0, 0, 0, .01, .02, .02, .03, .08, .15, .22])     # print-model spacing
    x = t.end + tt[:, None] * (t.start - t.end)
    ev = q6.on_result(x, np.ones(len(x)), 10)
    assert ev["kind"] == "matched" and ev["trace"] == 1


@pytest.mark.parametrize("rows", [slice(0, 10), slice(14, 24), slice(7, 17), slice(0, 5)])
def test_partial_result_with_failed_rows_still_matches(q6, rows):
    """DiSCO drops failed rows: a contiguous stretch of a trace is still that trace."""
    q6.next_pair(0)
    ev = q6.on_result(_print(q6, 0, 0.02)[rows], np.ones(24)[rows], 10)
    assert ev["kind"] == "matched" and ev["trace"] == 0
    assert len(q6.data[0][0]["y"]) == rows.stop - rows.start


def test_result_too_soon_after_sending_is_not_ours():
    q = InitQueue(axial_design(6).traces, min_latency=900)
    q.next_pair(0)
    assert q.on_result(_print(q, 0), np.ones(24), 60)["kind"] == "foreign"   # 1 min
    assert q.status[0] == "sent"
    assert q.on_result(_print(q, 0), np.ones(24), 4500)["kind"] == "matched"  # 75 min


def test_empty_result_completes_oldest_outstanding(q6):
    q6.next_pair(0)
    ev = q6.on_result(np.zeros((0, 6)), np.zeros(0), 5)
    assert ev["kind"] == "empty" and ev["trace"] == 0 and q6.status[0] == "done"
    ev = q6.on_result(_print(q6, 1), np.full(24, np.nan), 6)   # all NaN rows
    assert ev["kind"] == "empty"


def test_late_result_of_requeued_trace_counts(q6):
    q6.next_pair(0)
    q6.on_stall(5)                         # both presumed lost, back in the queue
    ev = q6.on_result(_print(q6, 0), np.ones(24), 6)
    assert ev["kind"] == "matched" and q6.status[0] == "done" and 0 not in q6.queue


def test_lone_last_trace_fills_both_slots():
    q = InitQueue(axial_design(3).traces[:1])
    assert q.next_pair(0) == (0, 0)


def test_state_round_trip_and_resume(q6):
    q6.next_pair(0)
    q6.on_result(_print(q6, 0), np.ones(24), 10)
    q6.status[5] = "abandoned"
    st = json.loads(json.dumps(q6.to_state()))
    r = InitQueue.from_state(st, resume=True)
    assert r.status[0] == "done" and r.status[1] == "sent"
    assert r.status[5] == "queued" and r.sends[5] == 0 and 5 in r.queue
    assert len(r.measured()) == 1


def test_stall_timeout_tracks_send_to_result_latency(q6):
    assert q6.stall_timeout(240 * 60, 45 * 60, 3) == 240 * 60
    q6.next_pair(0)
    q6.on_result(_print(q6, 0), np.ones(24), 80 * 60)
    assert q6.stall_timeout(240 * 60, 45 * 60, 3) == pytest.approx(240 * 60)   # 3 x 80 min
    q6.next_pair(80 * 60)
    q6.on_result(_print(q6, 1), np.ones(24), 81 * 60)
    assert q6.stall_timeout(240 * 60, 45 * 60, 3) == pytest.approx(243 * 60)   # max latency 81


# ── never deadlocks, whatever DiSCO does ─────────────────────────────────────

SCENARIOS = [
    dict(),
    dict(stability_min=0),
    dict(silent_p=0.3),
    dict(nan_row_p=0.3),
    dict(jitter_min=30),
    dict(transporters=("front",)),
    dict(stale_results=3),
    dict(stability_min=0, silent_p=0.3, jitter_min=15),
]


@pytest.mark.parametrize("dims", [3, 4, 5, 6, 10])
@pytest.mark.parametrize("kw", SCENARIOS, ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()) or "hardware")
def test_simulated_campaign_always_completes(dims, kw):
    from benchmarks.axial_init.disco_sim import simulate
    for seed in range(25):
        r = simulate(dims, policy="queue", seed=seed, **kw)
        assert r["complete"], (dims, kw, seed, r)
        if not kw.get("silent_p"):
            assert r["with_data"] == r["traces"], (dims, kw, seed, r)
