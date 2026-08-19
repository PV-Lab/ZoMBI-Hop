"""
Tests for retroactive needle declaration (src/core/retro.py +
ZoMBIHop.retro_declare_needles).

The parser fixtures are assembled from REAL excerpts of runs/run_39af/run.log
(copied here as literals — tests never touch the live run directory), stitched
into a compact log that exercises every block type the parser must handle:
duplicated plain/timestamped line pairs (both orders), normal iteration
blocks, "No unpenalized Y" blocks, candidate-None blocks, zoom re-entries,
and kill/resume seams.
"""

import hashlib
import json
from pathlib import Path

import pytest

from src.core import retro


# =============================================================================
# run.log fixtures (real excerpts from runs/run_39af/run.log)
# =============================================================================

# Activation 1 header + zoom entry + two measured iterations (run.log:78-195,
# long composition dumps trimmed; second iteration follows the same real
# pattern with the in-sequence digits).
_EXCERPT_ACT1 = r"""==================================================
[05:26:01]
==================================================
ACTIVATION 1/inf
[05:26:01] ACTIVATION 1/inf
==================================================
[05:26:01] ==================================================
──────────────────────────────────────────────────
[05:26:01]
──────────────────────────────────────────────────
--- Zoom 1/3 ---
[05:26:01] --- Zoom 1/3 ---
Search bounds: [[0. 0. 0. 0. 0. 0.]] – [[1.  1.  1.  0.3 1.  1. ]]
[05:26:01] Search bounds: [[0. 0. 0. 0. 0. 0.]] – [[1.  1.  1.  0.3 1.  1. ]]
[05:26:01] GP data points: 30 (best_f_local=0.9750)  data_fetch=0.03s
GP data points: 30 (best_f_local=0.9750)  data_fetch=0.03s
[05:26:01]   fitting GP …
  fitting GP …
C:\Users\Public\Anaconda_2024\envs\zombi-hop-linebo\Lib\site-packages\botorch\models\utils\assorted.py:270: InputDataWarning: Data (input features) is not contained to the unit cube. Please consider min-max scaling the input data.
  check_min_max_scaling(
  [GP.fit] MLL: 1.42s  (30 pts)
  GP fitted.  1.42s
[05:26:02]   GP fitted.  1.42s
  · iter 1/2  (activation lines 0/30)
[05:26:02]
  · iter 1/2  (activation lines 0/30)
  [GP.acq] repulsion_lambda=100.00  (0.07s)
  [ZoMBIHop] candidate search done.  2.01s
[05:26:04]   [ZoMBIHop] candidate search done.  2.01s
  [ZoMBIHop] Calling objective (LineBO samples lines through this candidate)...
[05:26:04]   [ZoMBIHop] Calling objective (LineBO samples lines through this candidate)...
============================================================
[ITERATION LOG]
  candidate (x_tell): [0.20955851939228765, 0.14556806594303046, 0.025417090779956054, 0.2643563357812935, 0.14556805810492, 0.20953192999851217]
============================================================
  [DH.add] paring: 0.01s  n_pared=31  n_total=49
  [ZoMBIHop] Objective returned 17 points, Y in [0.8365, 0.9275]  2882.09s
[06:14:06]   [ZoMBIHop] Objective returned 17 points, Y in [0.8365, 0.9275]  2882.09s
  [DH.snap] delta torch.save: 0.00s  (+17 pts)
  [time] snapshot: 0.00s
[06:14:06]   [time] snapshot: 0.00s
  [GP.fit] MLL: 0.58s  (31 pts)
  [time] post-obj GP refit: 0.58s  (31 pts)
[06:14:07]   [time] post-obj GP refit: 0.58s  (31 pts)
Converged: EI=4.81e-03, improvement=-4.75e-02, input_dist=4.67e-01, logEI=1.02
[06:14:07] Converged: EI=4.81e-03, improvement=-4.75e-02, input_dist=4.67e-01, logEI=1.02
[A1/Z1/I1] Candidate: [0.20955852 0.14556807 0.02541709 0.26435634 0.14556806 0.20953193] | EI=4.81e-03
Convergence count: 1/5
[06:14:07] Convergence count: 1/5
Current max Y: 0.9750 | Overall max: 0.9750
[06:14:07] Current max Y: 0.9750 | Overall max: 0.9750
  · iter 2/2  (activation lines 1/30)
[06:14:07]
  · iter 2/2  (activation lines 1/30)
  [ZoMBIHop] Objective returned 12 points, Y in [0.8365, 0.9275]  2415.31s
[06:58:01]   [ZoMBIHop] Objective returned 12 points, Y in [0.8365, 0.9275]  2415.31s
  [time] snapshot: 0.00s
[06:58:01]   [time] snapshot: 0.00s
Converged: EI=3.20e-03, improvement=-1.20e-02, input_dist=2.10e-01, logEI=0.95
[06:58:01] Converged: EI=3.20e-03, improvement=-1.20e-02, input_dist=2.10e-01, logEI=0.95
[A1/Z1/I2] Candidate: [0.20955852 0.14556807 0.02541709 0.26435634 0.14556806 0.20953193] | EI=3.20e-03
Convergence count: 2/5
[06:58:01] Convergence count: 2/5
"""

# Zoom advance within activation 1 — the timestamped copy of the zoom header
# deliberately precedes the plain copy (both orders occur in real logs
# depending on tee buffering; run.log:92-93 shows the ts-first form).
_EXCERPT_ZOOM2_TS_FIRST = r"""──────────────────────────────────────────────────
[06:58:02]
──────────────────────────────────────────────────
[06:58:02] --- Zoom 2/3 ---
--- Zoom 2/3 ---
[06:58:02] Search bounds: [[0.05296444 0.02988345 0.19548818 0.00442266 0.0057502  0.00193639]] – [[0.14140184 0.56983021 0.73911182 0.14419074 0.0955211  0.13520262]]
Search bounds: [[0.05296444 0.02988345 0.19548818 0.00442266 0.0057502  0.00193639]] – [[0.14140184 0.56983021 0.73911182 0.14419074 0.0955211  0.13520262]]
"""

# Candidate-None failure iteration + zoom re-entry (run.log:783-824). The
# status line is print-only: it appears once, never timestamped.
_EXCERPT_CANDIDATE_NONE = r"""==================================================
[08:33:40]
==================================================
ACTIVATION 2/inf
[08:33:40] ACTIVATION 2/inf
==================================================
[08:33:40] ==================================================
──────────────────────────────────────────────────
[08:33:40]
──────────────────────────────────────────────────
--- Zoom 1/3 ---
[08:33:40] --- Zoom 1/3 ---
  · iter 1/2  (activation lines 0/30)
[08:33:43]
  · iter 1/2  (activation lines 0/30)
  [GP.acq] repulsion_lambda=100.00  (0.06s)
  [GP.cand] sample+eval: 0.06s
  [GP.cand] penalty_mask: 0.00s  (82/300 unpenalized)
  [GP] running nat_grad: 300 restarts × 400 steps …
  [GP] nat_grad done (300 restarts × 400 steps)  2.26s
  [GP] get_candidate: best_candidate = [0.08274424 0.36750701 0.04152544 0.08958983 0.20947982 0.20915366], best_acq_value = -139.903997  total=2.62s
  [ZoMBIHop] candidate search done.  2.62s
[08:33:46]   [ZoMBIHop] candidate search done.  2.62s
No valid candidate found (all in penalized regions)
[08:33:46] No valid candidate found (all in penalized regions)
[A2/Z1/I1] Candidate: None
  [failure] first failure — recomputing ellipsoids with clean local GP ...
[08:33:46]   [failure] first failure — recomputing ellipsoids with clean local GP ...
  [failure] ellipsoids recomputed for 1 needle(s).
[08:33:46]   [failure] ellipsoids recomputed for 1 needle(s).
──────────────────────────────────────────────────
[08:33:47]
──────────────────────────────────────────────────
--- Zoom 1/3 ---
[08:33:47] --- Zoom 1/3 ---
Search bounds: [[0. 0. 0. 0. 0. 0.]] – [[1.  1.  1.  0.3 1.  1. ]]
[08:33:47] Search bounds: [[0. 0. 0. 0. 0. 0.]] – [[1.  1.  1.  0.3 1.  1. ]]
GP data points: 15 (best_f_local=0.7041)  data_fetch=0.00s
[08:33:47] GP data points: 15 (best_f_local=0.7041)  data_fetch=0.00s
"""

# "No unpenalized Y" block — measured (snapshot taken), but no status line and
# no counter line — followed by a failure-retry zoom re-entry (run.log:2444-2471).
_EXCERPT_ALL_PENALIZED = r"""  · iter 1/2  (activation lines 1/30)
[14:27:00]
  · iter 1/2  (activation lines 1/30)
============================================================
  [DH.add] paring: 0.00s  n_pared=5  n_total=424
  [ZoMBIHop] Objective returned 0 points, Y=[] (empty)  1672.90s
[14:27:23]   [ZoMBIHop] Objective returned 0 points, Y=[] (empty)  1672.90s
  [DH.snap] delta torch.save: 0.00s  (+22 pts)
  [time] snapshot: 0.00s
[14:27:23]   [time] snapshot: 0.00s
  [GP.fit] MLL: 0.42s  (5 pts)
  [time] post-obj GP refit: 0.42s  (5 pts)
[14:27:24]   [time] post-obj GP refit: 0.42s  (5 pts)
No unpenalized Y values, breaking — every point in this batch lies inside at least one needle penalty ball.
[14:27:24] No unpenalized Y values, breaking — every point in this batch lies inside at least one needle penalty ball.
  [failure] subsequent failure with new data — recomputing zoom bounds (Jaccard-aware) ...
[14:27:24]   [failure] subsequent failure with new data — recomputing zoom bounds (Jaccard-aware) ...
  [failure] new bounds: [[0.05296444 0.02988345 0.19548818 0.00442266 0.0057502  0.00193639]] – [[0.14140184 0.56983021 0.73911182 0.14419074 0.0955211  0.13520262]]
[14:27:24]   [failure] new bounds: [[0.05296444 0.02988345 0.19548818 0.00442266 0.0057502  0.00193639]] – [[0.14140184 0.56983021 0.73911182 0.14419074 0.0955211  0.13520262]]
──────────────────────────────────────────────────
[14:27:24]
──────────────────────────────────────────────────
--- Zoom 1/3 ---
[14:27:24] --- Zoom 1/3 ---
GP data points: 21 (best_f_local=0.9275)  data_fetch=0.00s
[14:27:24] GP data points: 21 (best_f_local=0.9275)  data_fetch=0.00s
"""

# Kill/resume seam (run.log:3563-3646, trimmed): an iteration killed
# mid-objective-call (no "Objective returned", no record), the launcher exit /
# relaunch seams, the print-only "Loaded state:" line, the repeated activation
# header, and a resumed iteration starting at iter 2/2.
_EXCERPT_SEAM = r"""  · iter 2/2  (activation lines 3/30)
[18:34:30]
  · iter 2/2  (activation lines 3/30)
  [ZoMBIHop] Calling objective (LineBO samples lines through this candidate)...
[18:34:34]   [ZoMBIHop] Calling objective (LineBO samples lines through this candidate)...
[SerialIO] Signal 21 — closing serial path…
[Main] Signal 21 — stopping ZoMBI, then releasing COM5…
[Machine2] PORT_CLOSED COM5
[Machine2] Serial port released; cleanup complete
===== process exited (rc=1) 2026-08-12 18:48:11 =====

===== hardware run launched 2026-08-12 19:20:00 =====
$ C:\Users\Public\Anaconda_2024\envs\zombi-hop-linebo\python.exe scripts\main.py 39af --dims 0,2,3,4,8,9
[ZoMBI Process] Resuming ZoMBI-Hop v2 with UUID: 39af...
Initialized ZoMBIHop on CUDA device: NVIDIA GeForce RTX 4090
Resuming from saved run: 39af
Loaded state: activation=4, zoom=0, iteration=1
✅ Resumed from activation=4, zoom=0, iteration=1
================================================================================
STARTING OPTIMIZATION
================================================================================
  [no-stop] never_terminate=True — max_activations set to ∞.
[19:20:15]   [no-stop] never_terminate=True — max_activations set to ∞.
==================================================
[19:20:15]
==================================================
ACTIVATION 5/inf
[19:20:15] ACTIVATION 5/inf
==================================================
[19:20:15] ==================================================
──────────────────────────────────────────────────
[19:20:15]
──────────────────────────────────────────────────
--- Zoom 1/3 ---
[19:20:15] --- Zoom 1/3 ---
Search bounds: [[0.05296444 0.00961117 0.21202845 0.00442266 0.0142963  0.00480355]] – [[0.09747479 0.56983021 0.73911182 0.12951127 0.14744477 0.13520262]]
[19:20:15] Search bounds: [[0.05296444 0.00961117 0.21202845 0.00442266 0.0142963  0.00480355]] – [[0.09747479 0.56983021 0.73911182 0.12951127 0.14744477 0.13520262]]
  · iter 2/2  (activation lines 0/30)
[19:20:16]
  · iter 2/2  (activation lines 0/30)
  [ZoMBIHop] Objective returned 20 points, Y in [0.7000, 0.9134]  1500.00s
[19:45:00]   [ZoMBIHop] Objective returned 20 points, Y in [0.7000, 0.9134]  1500.00s
Converged: EI=2.00e-03, improvement=-1.00e-02, input_dist=1.00e-01, logEI=0.90
[19:45:00] Converged: EI=2.00e-03, improvement=-1.00e-02, input_dist=1.00e-01, logEI=0.90
[A5/Z1/I2] Candidate: [0.07096857 0.11667719 0.28410298 0.17353585 0.17355355 0.18116186] | EI=2.00e-03
Convergence count: 1/5
[19:45:00] Convergence count: 1/5
"""

# Trailing started-but-empty activation ending in a process kill — one
# candidate went out, nothing came back (run_39af ends this way).
_EXCERPT_TRAILING = r"""==================================================
[08:20:00]
==================================================
ACTIVATION 6/inf
[08:20:00] ACTIVATION 6/inf
==================================================
[08:20:00] ==================================================
──────────────────────────────────────────────────
[08:20:00]
──────────────────────────────────────────────────
--- Zoom 1/3 ---
[08:20:00] --- Zoom 1/3 ---
  · iter 1/2  (activation lines 0/30)
[08:20:01]
  · iter 1/2  (activation lines 0/30)
  [ZoMBIHop] Calling objective (LineBO samples lines through this candidate)...
[08:20:05]   [ZoMBIHop] Calling objective (LineBO samples lines through this candidate)...
===== process exited (rc=1) 2026-08-13 08:21:17 =====
"""

_FIXTURE_FULL = (_EXCERPT_ACT1 + _EXCERPT_ZOOM2_TS_FIRST + _EXCERPT_CANDIDATE_NONE
                 + _EXCERPT_ALL_PENALIZED + _EXCERPT_SEAM + _EXCERPT_TRAILING)

# Measured line whose status line was lost to a kill: snapshot taken, process
# died before the convergence check logged anything.
_EXCERPT_KILLED_AFTER_MEASURE = r"""  · iter 1/2  (activation lines 4/30)
[18:40:00]
  · iter 1/2  (activation lines 4/30)
  [ZoMBIHop] Objective returned 9 points, Y in [0.8000, 0.9000]  900.00s
[18:47:55]   [ZoMBIHop] Objective returned 9 points, Y in [0.8000, 0.9000]  900.00s
  [time] snapshot: 0.00s
[18:47:55]   [time] snapshot: 0.00s
===== process exited (rc=1) 2026-08-12 18:48:11 =====
"""

# Measured line with a status line but NO "Convergence count" line ⇒ the
# check did not converge (counter stays 0; the counter line is only logged
# when the counter is positive).
_EXCERPT_NOT_CONVERGED = r"""  · iter 1/2  (activation lines 0/30)
[10:00:00]
  · iter 1/2  (activation lines 0/30)
  [ZoMBIHop] Objective returned 5 points, Y in [0.5000, 0.6000]  100.00s
[10:05:00]   [ZoMBIHop] Objective returned 5 points, Y in [0.5000, 0.6000]  100.00s
[A3/Z2/I1] Candidate: [0.1 0.2 0.7] | EI=5.00e-01
Current max Y: 0.9000 | Overall max: 0.9750
[10:05:00] Current max Y: 0.9000 | Overall max: 0.9750
  · iter 2/2  (activation lines 1/30)
[10:05:00]
  · iter 2/2  (activation lines 1/30)
"""


def _write_log(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "run.log"
    p.write_text(text, encoding="utf-8")
    return p


# =============================================================================
# Parser
# =============================================================================

def test_parse_full_fixture(tmp_path):
    records = retro.parse_run_log(_write_log(tmp_path, _FIXTURE_FULL))

    expected = [
        {"activation": 0, "zoom": 0, "event": "zoom_entry"},
        {"activation": 0, "zoom": 0, "iteration": 0, "measured": True,
         "converged": True, "counter": 1},
        {"activation": 0, "zoom": 0, "iteration": 1, "measured": True,
         "converged": True, "counter": 2},
        {"activation": 0, "zoom": 1, "event": "zoom_entry"},          # ts-first pair
        {"activation": 1, "zoom": 0, "event": "zoom_entry"},
        {"activation": 1, "zoom": 0, "iteration": 0, "measured": False,
         "event": "candidate_none"},
        {"activation": 1, "zoom": 0, "event": "zoom_entry"},          # failure re-entry
        {"activation": 1, "zoom": 0, "iteration": 0, "measured": True,
         "event": "all_penalized"},
        {"activation": 1, "zoom": 0, "event": "zoom_entry"},          # failure re-entry
        {"activation": 4, "zoom": 0, "event": "zoom_entry"},          # after resume seam
        {"activation": 4, "zoom": 0, "iteration": 1, "measured": True,
         "converged": True, "counter": 1},
        {"activation": 5, "zoom": 0, "event": "zoom_entry"},          # trailing activation
    ]
    assert len(records) == len(expected), records
    for got, want in zip(records, expected):
        for key, val in want.items():
            assert got.get(key) == val, (got, want)

    # The two iterations killed mid-objective-call (seam + trailing) must
    # produce no record at all: 4 measured-type records total.
    assert sum(1 for r in records if r.get("measured")) == 4


def test_dedupe_is_order_agnostic_and_drops_bare_prefix_lines(tmp_path):
    log = (
        "--- Zoom 1/3 ---\n"
        "[05:26:01] --- Zoom 1/3 ---\n"     # plain-first pair
        "[06:58:02] --- Zoom 2/3 ---\n"
        "--- Zoom 2/3 ---\n"                # ts-first pair
        "[06:14:07] \n"                     # tee copy of a "\n"-prefixed message
    )
    records = retro.parse_run_log(_write_log(tmp_path, log))
    assert [r["zoom"] for r in records] == [0, 1]
    assert all(r["event"] == "zoom_entry" for r in records)


def test_parse_measured_killed_before_status(tmp_path):
    records = retro.parse_run_log(_write_log(tmp_path, _EXCERPT_KILLED_AFTER_MEASURE))
    assert len(records) == 1
    rec = records[0]
    assert rec["measured"] is True
    assert rec["counter"] == 0
    assert rec["converged"] is False
    assert rec["iteration"] == 0


def test_parse_measured_not_converged_has_counter_zero(tmp_path):
    records = retro.parse_run_log(_write_log(tmp_path, _EXCERPT_NOT_CONVERGED))
    assert len(records) == 1
    rec = records[0]
    assert (rec["activation"], rec["zoom"], rec["iteration"]) == (2, 1, 0)
    assert rec["measured"] is True
    assert rec["converged"] is False
    assert rec["counter"] == 0


def test_find_triggers_on_parsed_fixture(tmp_path):
    records = retro.parse_run_log(_write_log(tmp_path, _FIXTURE_FULL))
    # n=1 at any depth: acts 0 (counter 1 at z0/i0) and 4 (counter 1 at z0/i1)
    # qualify; the all-penalized line of act 1 carries no counter.
    triggers = retro.find_retro_triggers(records, 1, 0, 1)
    assert [(t["activation"], t["zoom"], t["iteration"]) for t in triggers] == \
        [(0, 0, 0), (4, 0, 1)]
    # n=2: only act 0 ever reaches a recorded counter of 2.
    triggers = retro.find_retro_triggers(records, 2, 0, 1)
    assert [(t["activation"], t["counter"]) for t in triggers] == [(0, 2)]


# =============================================================================
# Trigger logic (synthetic record streams)
# =============================================================================

def _measured(act, zoom, it, counter):
    return {"activation": act, "zoom": zoom, "iteration": it, "measured": True,
            "converged": counter > 0, "counter": counter}


def _zoom_entry(act, zoom):
    return {"activation": act, "zoom": zoom, "event": "zoom_entry"}


def test_trigger_requires_deep_zoom():
    records = []
    for zoom in (0, 1, 2):
        records.append(_zoom_entry(0, zoom))
        records.append(_measured(0, zoom, 0, 1))
        records.append(_measured(0, zoom, 1, 2))
    triggers = retro.find_retro_triggers(records, 2, 2, 2)
    assert triggers == [{"activation": 0, "zoom": 2, "iteration": 1, "counter": 2}]
    # Shallow zooms alone never fire.
    assert retro.find_retro_triggers(records[:6], 2, 2, 2) == []


def test_trigger_requires_min_iters_this_zoom():
    records = [
        _zoom_entry(0, 2),
        _measured(0, 2, 0, 5),   # counter high but only 1 line at this zoom
    ]
    assert retro.find_retro_triggers(records, 2, 2, 2) == []
    records.append(_measured(0, 2, 1, 6))
    triggers = retro.find_retro_triggers(records, 2, 2, 2)
    assert triggers == [{"activation": 0, "zoom": 2, "iteration": 1, "counter": 6}]


def test_iters_this_zoom_resets_on_zoom_entry():
    records = [
        _zoom_entry(0, 2),
        _measured(0, 2, 0, 3),
        _measured(0, 2, 1, 4),
        _zoom_entry(0, 2),       # failure-retry re-entry resets the line count
        _measured(0, 2, 0, 5),
    ]
    # Without the re-entry the third measured line would fire; with it,
    # iters_this_zoom is 1 there — but the second line already fired.
    triggers = retro.find_retro_triggers(records, 2, 2, 2)
    assert triggers == [{"activation": 0, "zoom": 2, "iteration": 1, "counter": 4}]
    # Drop the qualifying second line: nothing after the re-entry may fire.
    assert retro.find_retro_triggers(records[:2] + records[3:], 2, 2, 2) == []


def test_one_trigger_per_activation_and_skip_activations():
    records = [
        _zoom_entry(0, 2),
        _measured(0, 2, 0, 2),
        _measured(0, 2, 1, 3),   # would fire again — must not
        _zoom_entry(1, 2),
        _measured(1, 2, 0, 2),
        _measured(1, 2, 1, 3),
    ]
    triggers = retro.find_retro_triggers(records, 2, 2, 1)
    assert [(t["activation"], t["iteration"]) for t in triggers] == [(0, 0), (1, 0)]
    triggers = retro.find_retro_triggers(records, 2, 2, 1, skip_activations={0})
    assert [t["activation"] for t in triggers] == [1]


def _records_39af_shaped():
    """Compact synthetic stream shaped like run_39af: acts 0 and 2 ended at
    counter 6 (needles declared), acts 1,3,4,5,6 ended at z2/i1 with counter 2."""
    records = []
    for act in range(7):
        final = 6 if act in (0, 2) else 2
        records.append(_zoom_entry(act, 2))
        records.append(_measured(act, 2, 0, final - 1))
        records.append(_measured(act, 2, 1, final))
    return records


def test_39af_shaped_stream_n5_and_n2():
    records = _records_39af_shaped()
    # Original criteria (n=5): only the two needle activations qualify, and
    # with those skipped nothing triggers at all.
    assert [t["activation"] for t in
            retro.find_retro_triggers(records, 5, 2, 2)] == [0, 2]
    assert retro.find_retro_triggers(records, 5, 2, 2, skip_activations={0, 2}) == []
    # Loosened criteria (n=2): acts 1,3,4,5,6 all trigger at z2/i1.
    triggers = retro.find_retro_triggers(records, 2, 2, 2, skip_activations={0, 2})
    assert [(t["activation"], t["zoom"], t["iteration"], t["counter"])
            for t in triggers] == [(a, 2, 1, 2) for a in (1, 3, 4, 5, 6)]


# =============================================================================
# Sidecar loading (load_convergence_history)
# =============================================================================

def test_load_history_merge_keeps_the_better_source_per_activation(tmp_path):
    """Per activation, whichever source carries more measured lines wins —
    so neither a thin sidecar nor a thin log can erase the other's history."""
    log_records = retro.parse_run_log(_write_log(tmp_path, _FIXTURE_FULL)
                                      or (tmp_path / "run.log"))
    n_measured_log_act0 = sum(1 for r in log_records
                              if r.get("activation") == 0 and r.get("measured"))
    assert n_measured_log_act0 >= 1

    # Sidecar thinner than the log for activation 0 ⇒ the log's records survive.
    (tmp_path / "convergence_history.jsonl").write_text(
        json.dumps(_zoom_entry(0, 0)) + "\n", encoding="utf-8")
    records, source = retro.load_convergence_history(tmp_path)
    assert sum(1 for r in records
               if r.get("activation") == 0 and r.get("measured")) == n_measured_log_act0

    # Sidecar at least as rich ⇒ the structured records win for that activation.
    rich = [_zoom_entry(0, 0)] + [_measured(0, 0, i, i + 1)
                                  for i in range(n_measured_log_act0 + 1)]
    (tmp_path / "convergence_history.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rich) + "\n", encoding="utf-8")
    records, source = retro.load_convergence_history(tmp_path)
    assert "convergence_history.jsonl" in source
    assert sum(1 for r in records
               if r.get("activation") == 0 and r.get("measured")) == n_measured_log_act0 + 1


def test_partial_sidecar_does_not_shadow_logged_history(tmp_path):
    """Regression: a run predating the sidecar accumulates a stub one on its
    first resume (a couple of zoom_entry rows and nothing else). Preferring the
    file wholesale erased every logged activation, so the retro pass replayed a
    2-record history and found nothing."""
    (tmp_path / "convergence_history.jsonl").write_text(
        json.dumps(_zoom_entry(6, 2)) + "\n" + json.dumps(_zoom_entry(6, 2)) + "\n",
        encoding="utf-8")
    _write_log(tmp_path, _FIXTURE_FULL)
    records, source = retro.load_convergence_history(tmp_path)
    log_only, _ = retro.load_convergence_history_from_log_only(tmp_path) \
        if hasattr(retro, "load_convergence_history_from_log_only") \
        else (retro.parse_run_log(tmp_path / "run.log"), "")
    # Every activation the log knows about survives the merge.
    assert {r["activation"] for r in log_only} <= {r["activation"] for r in records}
    assert len(records) >= len(log_only)
    assert sum(1 for r in records if r.get("measured")) == \
        sum(1 for r in log_only if r.get("measured"))


def test_load_history_tolerates_torn_final_line(tmp_path):
    (tmp_path / "convergence_history.jsonl").write_text(
        json.dumps(_measured(0, 2, 0, 2)) + "\n" + '{"activation": 1, "zo',
        encoding="utf-8")
    records, source = retro.load_convergence_history(tmp_path)
    assert source == "convergence_history.jsonl"
    assert len(records) == 1


def test_load_history_falls_back_to_log(tmp_path):
    # Empty sidecar must not mask the log.
    (tmp_path / "convergence_history.jsonl").write_text("", encoding="utf-8")
    _write_log(tmp_path, _FIXTURE_FULL)
    records, source = retro.load_convergence_history(tmp_path)
    assert source == "run.log"
    assert len(records) == 12


def test_load_history_neither_source(tmp_path):
    records, source = retro.load_convergence_history(tmp_path)
    assert records == []
    assert "no convergence_history" in source


# =============================================================================
# End-to-end: tiny CPU campaign → retro declaration
# =============================================================================

@pytest.fixture()
def _cpu_default_device(torch):
    """src.core.zombihop sets the torch default device to cuda at import when
    CUDA is available; the tiny campaign runs on CPU (gpytorch otherwise
    creates constraint tensors on cuda and the CPU fit fails). Flip it for the
    duration and restore afterwards."""
    import src.core.zombihop  # noqa: F401 — trigger the import side effect first
    if torch.cuda.is_available():
        torch.set_default_device("cpu")
        yield
        torch.set_default_device("cuda")
    else:
        yield


def _run_tiny_campaign(torch, base_dir: Path):
    """Real (non-mocked) ZoMBIHop campaign on CPU: 1 activation, 1 zoom,
    2 iterations, criteria too strict to declare a needle (n=10), convergence
    check patched to always converge so the recorded counter climbs to 2."""
    from src.core.zombihop import ZoMBIHop

    dtype = torch.float64
    d = 3
    peak = torch.tensor([0.6, 0.3, 0.1], dtype=dtype, device="cpu")

    g = torch.Generator().manual_seed(0)
    X_init = torch.rand(6, d, generator=g, dtype=dtype, device="cpu")
    X_init = X_init / X_init.sum(dim=1, keepdim=True)
    Y_init = 1.0 - ((X_init - peak) ** 2).sum(dim=1, keepdim=True)

    calls = {"n": 0}

    def objective(X, bounds, acq):
        # (X_expected, X_actual, Y); varies per call so the near-duplicate
        # paring in add_all_points keeps every point.
        calls["n"] += 1
        k = calls["n"]
        base = torch.tensor(
            [[0.60, 0.30, 0.10],
             [0.55, 0.35, 0.10],
             [0.50, 0.30, 0.20],
             [0.45, 0.35, 0.20]], dtype=dtype, device="cpu")
        jitter = 0.013 * k * torch.tensor(
            [[1.0, -1.0, 0.0],
             [0.0, 1.0, -1.0],
             [-1.0, 0.0, 1.0],
             [1.0, 0.0, -1.0]], dtype=dtype, device="cpu")
        Xp = (base + jitter).clamp(min=0.001)
        Xp = Xp / Xp.sum(dim=1, keepdim=True)
        Y = 1.0 - ((Xp - peak) ** 2).sum(dim=1)
        return Xp.clone(), Xp.clone(), Y

    z = ZoMBIHop(
        objective=objective,
        X_init_actual=X_init.clone(),
        X_init_expected=X_init.clone(),
        Y_init=Y_init.clone(),
        device="cpu",
        dtype=dtype,
        max_zooms=1,
        max_iterations=2,
        n_consecutive_converged=10,   # never declare during the run
        n_restarts=2,
        raw=32,
        nat_grad_max_steps=5,
        resume=False,
        checkpoint_dir=str(base_dir),
        verbose=False,
        min_zoom_for_needle=0,
        min_iters_per_zoom=2,
        input_noise=0.05,
    )
    z._check_convergence_to_needle = lambda *a, **k: (True, 1e-6, -13.8)
    z.run(max_activations=1, time_limit_hours=None)
    return z


def _dir_state(root: Path):
    state = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            state[str(p.relative_to(root))] = (
                st.st_size, st.st_mtime_ns,
                hashlib.md5(p.read_bytes()).hexdigest(),
            )
    return state


def test_e2e_retro_lifecycle(torch, tmp_path, _cpu_default_device):
    from src.core.zombihop import ZoMBIHop

    base_dir = tmp_path / "runs"
    z = _run_tiny_campaign(torch, base_dir)
    dh = z.data_handler
    run_dir = dh.run_dir
    assert dh.needles.shape[0] == 0  # high n: nothing declared during the run

    # --- Sidecar written by the run loop, and preferred over run.log ---
    assert (run_dir / "convergence_history.jsonl").exists()
    assert (run_dir / "run.log").exists()
    records, source = retro.load_convergence_history(run_dir)
    assert "convergence_history.jsonl" in source
    kinds = [(r.get("event"), r.get("counter")) for r in records]
    assert kinds == [("zoom_entry", None), (None, 1), (None, 2)]
    assert all("ts" in r for r in records)

    # --- Operator loosens the criteria (as a config.json edit would on resume) ---
    dh.n_consecutive_converged = 2

    # --- Dry run: correct preview, run dir byte-identical ---
    before = _dir_state(run_dir)
    res = z.retro_declare_needles(dry_run=True)
    assert _dir_state(run_dir) == before
    assert res["applied"] is False and "error" not in res, res
    assert res["triggers"] == [
        {"activation": 0, "zoom": 0, "iteration": 1, "counter": 2}]
    (cand,) = res["candidates"]
    assert "skipped_reason" not in cand
    assert len(cand["x"]) == 3 and cand["y"] > 0.9
    assert cand["dist_to_earlier_candidates"] == []

    # --- Apply ---
    res2 = z.retro_declare_needles(dry_run=False)
    assert res2["applied"] is True and res2["n_declared"] == 1, res2
    assert dh.needles.shape[0] == 1
    # A retroactive needle is a convergence needle recognised late: it must be
    # indistinguishable from a live one to every display and export path.
    assert dh.needles_results[-1]["reason"] == "EI convergence"
    assert not any("retro" in str(v).lower()
                   for r in dh.needles_results for v in r.values())
    # Needle center is inside its own penalty ellipsoid.
    assert not dh.get_penalty_mask(dh.needles).any()
    # Permanent snapshot + latest.txt.
    latest = (run_dir / "latest.txt").read_text().strip()
    assert latest.endswith("_retro_needles")
    assert (run_dir / "snapshots" / latest / "permanent").exists()
    # Resume position advanced to a fresh activation on the full box.
    assert res2["new_activation"] == 1
    assert (z.current_activation, z.current_zoom, z.current_iteration) == (1, 0, 0)
    assert torch.equal(dh.current_zoom_bounds, z.full_bounds)
    assert torch.equal(z.bounds, z.full_bounds)

    # --- Idempotence: second apply declares nothing ---
    res3 = z.retro_declare_needles(dry_run=False)
    assert res3["applied"] is False and res3.get("n_declared", 0) == 0, res3
    assert all(c.get("skipped_reason") in ("covered", "empty", "already declared")
               for c in res3["candidates"])
    assert dh.needles.shape[0] == 1

    # --- Cross-process resume sees the retro state ---
    def _dummy_objective(*_a, **_k):
        raise RuntimeError("resume construction must not call the objective")

    _d0 = torch.zeros(0, 3, device="cpu", dtype=torch.float64)
    z2 = ZoMBIHop(
        objective=_dummy_objective,
        X_init_actual=_d0, X_init_expected=_d0,
        Y_init=torch.zeros(0, 1, device="cpu", dtype=torch.float64),
        device="cpu", dtype=torch.float64,
        run_uuid=z.run_uuid, checkpoint_dir=str(base_dir), verbose=False,
    )
    assert (z2.current_activation, z2.current_zoom, z2.current_iteration) == (1, 0, 0)
    assert z2.data_handler.needles.shape[0] == 1
    assert torch.equal(z2.data_handler.current_zoom_bounds, z2.full_bounds)
