"""The lanes convergence plot (``visualization/lanes_timeline.py``, the GUI's Convergence tab for lanes runs,
and ``visualization/render_lanes_convergence.py``).

A tiny synthetic lanes run folder (``write_lanes_run``) in the layout ``scripts/lanes_hw.py`` writes:
``lanes/init.pt``, ``rows.jsonl``, ``journal.jsonl``, ``engine_ckpt.pt``, ``meta.json`` (+ optional
``run.log``). Checks: the measured top-m per (dive, SENT zoom) and the 0.01 dip rule against the nearest
earlier zoom with rows; sent-zoom labelling (a late zoom-0 result stays in zoom 0's band); block order
F1 / B1 / F2 / B2 by activation start, and a restarted ZoMBI process's own activation numbers; needles and
dive ends from the checkpoint or run.log; loader robustness (missing files, a partially written last line);
late init results in the init band; a started dive without rows as an empty block; the init view of a run
still in its axial initialization (``write_init_run``: a synthetic ``init_state.json``); the PNG CLI never
writes into the run folder and exits 0 in every phase; a legacy run folder (no ``lanes/``, or init_state.json
+ snapshots) still takes the app's snapshot path; nothing initialises CUDA. When present, the gui_plot
fixtures (``benchmarks/zoom_fix/lanes/report/review/impl/gui_plot``) are checked against fixture.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

lt = pytest.importorskip("visualization.lanes_timeline")

T0 = 1_763_000_000.0          # wall clock of the lanes start
D = 3

# Arrival order. qid, wid, lane, act, SENT zoom, measured y. Global row = 4 init rows + position.
INIT_Y = [0.50, 0.60, 0.40, 0.55]
PACKETS = [
    (1, 101, "front", 0, 0, [0.70, 0.72, 0.71]),   # rows 0-2   (global 4-6)
    (2, 102, "back", 1, 0, [0.60, 0.62]),          # rows 3-4   (global 7-8)
    (3, 103, "front", 0, 1, [0.75, 0.76]),         # rows 5-6   (global 9-10)
    (4, 104, "back", 1, 1, [0.50, 0.52]),          # rows 7-8   (global 11-12)  dip vs zoom 0
    (5, 105, "front", 0, 0, [0.73]),               # row 9      (global 13)     LATE zoom-0 result
    (6, 106, "front", 0, 2, [0.77, 0.78]),         # rows 10-11 (global 14-15)  needle pick = global 15
    (7, 107, "front", 2, 0, [0.65]),               # row 12     (global 16)
    (8, 108, "back", 1, 2, [0.40]),                # row 13     (global 17)     back act 1 ends: zone
    (9, 109, "back", 3, 0, [0.66]),                # row 14     (global 18)
]
CKPT_EVENTS = [
    dict(kind="activation", lane="front", act=0, prev_act=None, reason="start", n_rows=4, seq=0),
    dict(kind="activation", lane="back", act=1, prev_act=None, reason="start", n_rows=4, seq=1),
    dict(kind="needle", lane="front", act=0, zoom=2, needle=0, pick=15, value=0.78, n_rows=16, seq=5),
    dict(kind="activation", lane="front", act=2, prev_act=0, reason="needle", n_rows=16, seq=6),
    dict(kind="zone", lane="back", act=1, zoom=2, reason="zooms_exhausted", n_rows=18, seq=8),
    dict(kind="activation", lane="back", act=3, prev_act=1, reason="zooms_exhausted", n_rows=18, seq=9),
]
CKPT_ZONES = [dict(seq=1, kind="needle", lane="front", act=0, index=0, pick=15),
              dict(seq=2, kind="zone", lane="back", act=1, index=0)]


def _jsonl(path: Path, recs) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def write_lanes_run(torch, run_dir: Path, *, packets=PACKETS, init_y=INIT_Y, starts=(T0,), issue_t=None,
                    ckpt=True, journal=True, init_pt=True, run_log=None, events=CKPT_EVENTS, zones=CKPT_ZONES):
    """A lanes run folder as scripts/lanes_hw.py writes it. ``issue_t``: {wid: issue time} (default: 60 s
    per packet after the first start); arrival = issue + 30 s."""
    lanes = run_dir / "lanes"
    lanes.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_init = len(init_y)
    Xi = rng.dirichlet(np.ones(D), n_init)
    if init_pt:
        torch.save(dict(X_actual=torch.tensor(Xi), X_expected=torch.tensor(Xi),
                        Y=torch.tensor(init_y, dtype=torch.float64).reshape(-1, 1),
                        bounds=torch.tensor([[0.0] * D, [1.0] * D]), opt_dims=list(range(D))), lanes / "init.pt")
    (lanes / "meta.json").write_text(json.dumps(dict(epoch=starts[0], preset="leader", n_starts=len(starts))))
    rows, jr, ys = [], [], list(init_y)
    for k, s in enumerate(starts):
        jr.append(dict(t=s, ev="start", resumed=k > 0, n_init=n_init, n_prior_rows=0, epoch=starts[0]))
    for j, (qid, wid, lane, act, zoom, yv) in enumerate(packets):
        ti = (issue_t or {}).get(wid, starts[0] + 60.0 * (j + 1))
        ta = ti + 30.0
        jr.append(dict(t=ti, ev="decision", lane=lane, kind="regular", act=act, zoom=zoom))
        jr.append(dict(t=ti, ev="issue", lane=lane, role="main", slot=0, wid=wid, act=act, zoom=zoom))
        jr.append(dict(t=ta, ev="ingested", qid=qid, wid=wid, lane=lane, act=act, zoom=zoom, n=len(yv)))
        for v in yv:
            x = rng.dirichlet(np.ones(D)).tolist()
            rows.append(dict(t=ta, ev="row", qid=qid, wid=wid, lane=lane, act=act, zoom_sent=zoom,
                             engine_rid=f"{lane[0].upper()}{qid}", pid=j, x_actual=x, x_expected=x, y_opt=v))
            ys.append(v)
    _jsonl(lanes / "rows.jsonl", rows)
    if journal:
        _jsonl(lanes / "journal.jsonl", jr)
    if ckpt:
        torch.save(dict(Y=torch.tensor(ys, dtype=torch.float64).reshape(-1, 1), packets=[], zones=list(zones),
                        events=list(events), decisions={}, t=starts[-1] + 10_000.0,
                        lane_state=dict(front=dict(act=2, zoom=0), back=dict(act=3, zoom=0))),
                   lanes / "engine_ckpt.pt")
    if run_log is not None:
        (run_dir / "run.log").write_text(run_log, encoding="utf-8")
    return run_dir


# ── the pure statistic ──────────────────────────────────────────────────────────────────────────

def test_topm_level_is_mean_of_the_m_best_per_dive_and_sent_zoom():
    y = [0.1, 0.9, 0.8, 0.7, 0.2, 0.95, 0.3]
    key = ["a", "a", "a", "a", "b", "b", None]
    zoom = [0, 0, 0, 1, 0, 0, 0]
    out = lt.zoom_topm_levels(y, key, zoom, m=2)
    got = {(r["key"], r["zoom"]): (round(r["level"], 6), r["n"], r["m"]) for r in out}
    assert got == {("a", 0): (0.85, 3, 2), ("a", 1): (0.7, 1, 1), ("b", 0): (0.575, 2, 2)}
    assert [(r["key"], r["zoom"]) for r in out] == [("a", 0), ("a", 1), ("b", 0)]   # first appearance, zoom
    assert out[0]["rows"].tolist() == [0, 1, 2]
    assert out[1]["dip"] is True                    # 0.70 is more than 0.01 below 0.85


@pytest.mark.parametrize("prev,cur,dip", [
    (0.80, 0.79, False),          # exactly 0.01 below: within tolerance (float slack)
    (0.80, 0.7899, True),         # more than 0.01 below
    (0.80, 0.80, False),
    (0.80, 0.95, False),
    (0.30, 0.2900001, False),
])
def test_dip_flag_at_the_001_tolerance(prev, cur, dip):
    out = lt.zoom_topm_levels([prev, cur], [7, 7], [0, 1], m=8)
    assert out[1]["prev"] == pytest.approx(prev)
    assert out[1]["dip"] is dip
    assert out[0]["dip"] is False                   # zoom 0 has no previous zoom


def test_dip_compares_with_the_nearest_earlier_zoom_of_the_same_dive():
    # 'a' has no zoom-1 rows: its zoom 2 compares with zoom 0 (the nearest earlier zoom WITH rows) and dips;
    # another dive's zoom 0 is never a reference ('b' zoom 1 is b's first zoom with rows: no flag)
    out = lt.zoom_topm_levels([0.9, 0.1, 0.5], ["a", "a", "b"], [0, 2, 1], m=8)
    assert [(r["key"], r["zoom"], r["dip"], r["prev_zoom"]) for r in out] == [
        ("a", 0, False, None), ("a", 2, True, 0), ("b", 1, False, None)]
    assert out[1]["prev"] == pytest.approx(0.9)
    assert lt.top_m_for_dim(6) == 8 and lt.top_m_for_dim(10) == 11 and lt.top_m_for_dim(0) == 8


def test_dip_reference_is_zoom_k_minus_1_when_it_has_rows_and_skips_a_non_finite_zoom():
    # zoom 1 has rows: zoom 2 compares with zoom 1 (0.50 -> 0.495 is within 0.01), not with zoom 0 (0.90)
    out = lt.zoom_topm_levels([0.9, 0.5, 0.495], [1, 1, 1], [0, 1, 2], m=8)
    assert [(r["dip"], r["prev_zoom"]) for r in out] == [(False, None), (True, 0), (False, 1)]
    # zoom 1's rows are all NaN (no level): zoom 2 falls back to zoom 0
    out = lt.zoom_topm_levels([0.9, np.nan, 0.7], [1, 1, 1], [0, 1, 2], m=8)
    assert [(r["dip"], r["prev_zoom"]) for r in out] == [(False, None), (False, 0), (True, 0)]


def test_topm_skips_non_finite_values_and_rejects_mismatched_lengths():
    out = lt.zoom_topm_levels([np.nan, 0.4], [1, 1], [0, 0], m=8)
    assert out[0]["level"] == pytest.approx(0.4) and out[0]["m"] == 1
    with pytest.raises(ValueError):
        lt.zoom_topm_levels([0.1, 0.2], [1], [0, 0])


# ── loading + layout ────────────────────────────────────────────────────────────────────────────

def test_blocks_in_start_order_rows_by_sent_zoom(torch, tmp_path):
    R = lt.load_lanes_run(write_lanes_run(torch, tmp_path / "run_t1"))
    assert R.n_init == 4 and len(R.y) == 15 and R.d == D and not R.warnings
    L = lt.dive_layout(R)
    assert [b["label"] for b in L.blocks] == ["F1", "B1", "F2", "B2"]
    assert [(b["start"], b["end"]) for b in L.blocks] == [(4, 12), (12, 17), (17, 18), (18, 19)]
    # F1: zoom 0 rows (incl. the LATE zoom-0 result, global 13) before zoom 1, then zoom 2
    assert L.order[4:12].tolist() == [4, 5, 6, 13, 9, 10, 14, 15]
    assert L.zoom[4:12].tolist() == [0, 0, 0, 0, 1, 1, 2, 2]
    assert L.zoom_starts.tolist() == [4, 8, 10, 12, 14, 16, 17, 18]
    assert L.zoom_levels.tolist() == [0, 1, 2, 0, 1, 2, 0, 0]
    # packets: one per (dive, zoom, qid) stretch (the late packet is its own)
    assert L.packet_starts.tolist() == [4, 7, 8, 10, 12, 14, 16, 17, 18]
    reasons = {b["label"]: (b["reason"], b["open"], b["needle"]) for b in L.blocks}
    assert reasons == {"F1": ("needle", False, True), "B1": ("zooms_exhausted", False, False),
                       "F2": (None, True, False), "B2": (None, True, False)}


def test_topm_bars_and_dips_follow_the_sent_zoom(torch, tmp_path):
    L = lt.dive_layout(lt.load_lanes_run(write_lanes_run(torch, tmp_path / "run_t2")))
    assert L.m == 8
    bars = {(lv["key"], lv["zoom"]): (round(lv["level"], 6), lv["dip"], lv["start"], lv["end"]) for lv in L.levels}
    assert bars[((0, 0), 0)] == (0.715, False, 4, 8)          # the late 0.73 counts in zoom 0
    assert bars[((0, 0), 1)] == (0.755, False, 8, 10)
    assert bars[((0, 0), 2)] == (0.775, False, 10, 12)
    assert bars[((0, 1), 0)] == (0.61, False, 12, 14)
    assert bars[((0, 1), 1)] == (0.51, True, 14, 16)
    assert bars[((0, 1), 2)] == (0.4, True, 16, 17)


def test_needle_star_and_declaration_from_the_checkpoint(torch, tmp_path):
    R = lt.load_lanes_run(write_lanes_run(torch, tmp_path / "run_t3"))
    assert len(R.needles) == 1 and R.needles[0]["row"] == 15 and not R.needles[0]["approx"]
    L = lt.dive_layout(R)
    nd = L.needles[0]
    assert (nd["label"], nd["block"], nd["star"], nd["declared"]) == ("N1", "F1", 11, 11)


def test_needles_and_dive_ends_from_run_log_without_checkpoint(torch, tmp_path):
    log = ("===== hardware run launched =====\n[Main] Optimizer: lanes\n"
           "STARTING LANES OPTIMIZATION (preset=leader; four-slot protocol)\n"
           "[lanes] front act 0: zoom_in zoom 1 from_zoom=0 to_zoom=1\n"
           "[lanes] front act 0: needle zoom 2\n"
           "[lanes] front act 2: activation reason=needle\n"
           "[lanes] back act 1: zone zoom 2 reason=zooms_exhausted\n")
    R = lt.load_lanes_run(write_lanes_run(torch, tmp_path / "run_t4", ckpt=False, run_log=log))
    assert len(R.needles) == 1
    nd = R.needles[0]
    assert nd["approx"] and nd["row"] == 15                   # the dive's best zoom-2 row (0.78)
    assert R.acts[(0, 1)]["reason"] == "zooms_exhausted"
    L = lt.dive_layout(R)
    assert [b["reason"] for b in L.blocks] == ["needle", "zooms_exhausted", None, None]


def test_restarted_process_numbers_its_own_activations(torch, tmp_path):
    t1 = T0 + 5000.0
    packets = [(1, 201, "front", 0, 0, [0.5]), (2, 202, "back", 1, 0, [0.6]),
               (3, 203, "front", 0, 0, [0.7]),     # after the restart: a NEW front act 0
               (4, 204, "back", 1, 1, [0.4])]      # late: issued BEFORE the restart -> the old back act 1
    issue = {201: T0 + 60, 202: T0 + 120, 203: t1 + 60, 204: T0 + 4000}
    R = lt.load_lanes_run(write_lanes_run(torch, tmp_path / "run_t5", packets=packets, starts=(T0, t1),
                                          issue_t=issue, ckpt=False))
    assert R.n_sessions == 2
    assert R.session.tolist() == [0, 0, 1, 0]
    L = lt.dive_layout(R)
    assert [b["label"] for b in L.blocks] == ["F1", "B1", "F2"]
    assert [b["key"] for b in L.blocks] == [(0, 0), (0, 1), (1, 0)]
    assert L.order[L.blocks[1]["start"]:L.blocks[1]["end"]].tolist() == [5, 7]     # the late row joins B1
    # dives of the earlier process ended with it (no needle known): grey 'restart'
    assert [b["reason"] for b in L.blocks] == ["restart", "restart", None]


def test_loader_survives_missing_files_and_a_partial_last_line(torch, tmp_path):
    run = write_lanes_run(torch, tmp_path / "run_t6", ckpt=False, journal=False, init_pt=False)
    with open(run / "lanes" / "rows.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"t": 1.0, "ev": "row", "qid": 10, "wid": 1')          # torn: still being written
    R = lt.load_lanes_run(run)
    assert R.n_init == 0 and R.init_source is None                     # no init.pt, no init_state.json
    assert len(R.y) == 15 and R.sources["rows_bad_lines"] == 1
    assert any("partial last line" in w for w in R.warnings)
    assert R.needles == [] and R.n_sessions == 1
    L = lt.dive_layout(R)
    assert [b["label"] for b in L.blocks] == ["F1", "B1", "F2", "B2"]
    # earlier activations of a lane are over (no needle known), the last ones are running
    assert [b["reason"] for b in L.blocks] == ["no_needle", "no_needle", None, None]


def test_init_falls_back_to_init_state_and_a_corrupt_checkpoint_is_a_warning(torch, tmp_path):
    run = write_lanes_run(torch, tmp_path / "run_t7", init_pt=False)
    data = [[{"x": [[0.2, 0.3, 0.5]] * 2, "y": [0.1, 0.2], "t": 20.0}], [{"x": [[0.2, 0.3, 0.5]], "y": [0.3], "t": 10.0}]]
    (run / "init_state.json").write_text(json.dumps({"queue": {"data": data}}))
    (run / "lanes" / "engine_ckpt.pt").write_bytes(b"not a torch file")
    R = lt.load_lanes_run(run)
    assert R.init_source == "init_state.json" and R.y_init.tolist() == [0.3, 0.1, 0.2]
    assert any("engine_ckpt.pt unreadable" in w for w in R.warnings)


def test_lanes_dir_with_init_only_and_legacy_run_detection(torch, tmp_path):
    legacy = tmp_path / "run_legacy"
    (legacy / "snapshots").mkdir(parents=True)
    assert not lt.is_lanes_run(legacy) and lt.lanes_signature(legacy) is None
    assert not lt.is_lanes_run(None)
    run = tmp_path / "run_t8"
    (run / "lanes").mkdir(parents=True)
    assert not lt.is_lanes_run(run)
    torch.save(dict(Y=torch.tensor([[0.1], [0.2]]), X_actual=torch.zeros(2, D)), run / "lanes" / "init.pt")
    assert lt.is_lanes_run(run)
    R = lt.load_lanes_run(run)
    assert R.n_init == 2 and len(R.y) == 0
    L = lt.dive_layout(R)
    assert L.blocks == [] and L.levels == []


# ── drawing ─────────────────────────────────────────────────────────────────────────────────────

def test_strip_clock_marks_dives_started_on_a_later_day():
    from datetime import datetime
    t0 = datetime(2026, 11, 13, 23, 50).timestamp()
    assert lt._clock(t0, t0) == "23:50"
    assert lt._clock(t0 + 2 * 3600, t0) == "+1d 01:50"
    assert lt._clock(t0 + 3600) == "00:50"
    assert lt._clock(float("nan"), t0) == ""


def _fig(size=(9, 4)):
    pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=size, dpi=96)
    FigureCanvasAgg(fig)
    return fig


def test_draw_one_axes_both_lanes_bars_needles_strip(torch, tmp_path):
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection
    fig = _fig()
    chk = lt.draw_lanes_convergence(fig, lt.load_lanes_run(write_lanes_run(torch, tmp_path / "run_t9")))
    fig.canvas.draw()
    assert len(fig.axes) == 1                                        # ONE axes, both lanes
    ax = fig.axes[0]
    bars = [c for c in ax.collections if isinstance(c, LineCollection)]
    red = [c for c in bars if np.allclose(c.get_color()[0], mcolors.to_rgba(lt.DIP_C))]
    black = [c for c in bars if np.allclose(c.get_color()[0], mcolors.to_rgba(lt.LEVEL_C))]
    assert (len(red), len(black)) == (2, 6) and chk["red_bars"] == 2
    texts = {t.get_text() for t in ax.texts}
    assert {"F1", "B1", "init", "N1"} <= texts
    assert {"F2 …", "B2 …"} <= texts                                 # running dives too narrow for the time
    labels = ax.get_legend().get_texts()
    lab = [t.get_text() for t in labels]
    assert "measured top-8 of the zoom" in lab and "top-8 > 0.01 below previous zoom" in lab
    assert not any("landscape" in s or "noise-free" in s for s in lab)    # unknown on hardware
    assert "no needle: zooms exhausted (exclusion zone)" in lab
    assert (chk["finished"], chk["clean"]) == (2, 1)
    assert "clean: 1 of 2 finished dives" in getattr(ax, "lanes_summary", "").replace("\n", " ")


def test_draw_handles_init_only_and_empty_runs(torch, tmp_path):
    run = tmp_path / "run_t10"
    (run / "lanes").mkdir(parents=True)
    torch.save(dict(Y=torch.tensor([[0.1], [0.2], [0.15]]), X_actual=torch.zeros(3, D)), run / "lanes" / "init.pt")
    fig = _fig()
    chk = lt.draw_lanes_convergence(fig, lt.load_lanes_run(run))
    fig.canvas.draw()
    assert chk["n_init"] == 3 and chk["n_rows"] == 0
    fig2 = _fig()
    chk2 = lt.draw_lanes_convergence(fig2, lt.LanesRun(run_dir=run, run_id="run_x"))
    fig2.canvas.draw()
    assert chk2["n_rows"] == 0


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────────

def test_cli_writes_png_outside_the_run_and_refuses_inside(torch, tmp_path, capsys):
    from visualization import render_lanes_convergence as cli
    run = write_lanes_run(torch, tmp_path / "run_t11")
    before = sorted(p.relative_to(run).as_posix() for p in run.rglob("*"))
    out = tmp_path / "out" / "conv.png"
    assert cli.main(["--run", str(run), "--out", str(out), "--dpi", "60"]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    line = capsys.readouterr().out.strip().splitlines()
    assert len(line) == 1 and line[0].startswith("wrote ") and "2 front + 2 back dives" in line[0]
    assert cli.main(["--run", str(run), "--out", str(out), "--dpi", "60"]) == 0          # overwrite in place
    assert cli.main(["--run", str(run), "--out", str(run / "lanes" / "x.png")]) == 2
    assert sorted(p.relative_to(run).as_posix() for p in run.rglob("*")) == before        # run untouched
    legacy = tmp_path / "run_legacy"
    legacy.mkdir()
    assert cli.main(["--run", str(legacy), "--out", str(tmp_path / "l.png")]) == 2


# ── the GUI hook ────────────────────────────────────────────────────────────────────────────────

def _app_module():
    try:
        import matplotlib
        import interface.app as app
        matplotlib.use("Agg", force=True)
    except Exception as e:  # noqa: BLE001 - the app needs Tk / the full stack
        pytest.skip(f"interface.app not importable here: {e!r}")
    return app


def test_gui_hook_draws_lanes_runs_and_leaves_legacy_runs_to_the_snapshot_plot(torch, tmp_path):
    app = _app_module()
    run = write_lanes_run(torch, tmp_path / "run_t12")
    frame = SimpleNamespace(fig=_fig(), draw=lambda: None)
    assert app._draw_lanes_convergence(frame, SimpleNamespace(run_dir=run)) is True
    assert len(frame.fig.axes) == 1 and "both lanes" in getattr(frame.fig.axes[0], "lanes_summary", "")
    assert app._lanes_changed(SimpleNamespace(run_dir=run)) is False
    with open(run / "lanes" / "rows.jsonl", "a", encoding="utf-8") as fh:              # a new result
        fh.write(json.dumps(dict(t=T0 + 999, ev="row", qid=10, wid=110, lane="back", act=3, zoom_sent=0,
                                 y_opt=0.7)) + "\n")
    assert app._lanes_changed(SimpleNamespace(run_dir=run)) is True

    # legacy run folder (no lanes/): the hook declines and ConvergencePlotFrame.update draws as before
    legacy = tmp_path / "run_legacy"
    (legacy / "snapshots").mkdir(parents=True)
    rd = SimpleNamespace(run_dir=legacy, Y_all=np.array([0.1, 0.3, 0.2, 0.5]), packet_starts=np.array([], int),
                         penalty_mask=None, needle_indices=None, n_init=1, zoom_starts=np.array([], int),
                         zoom_levels=np.array([], int), activation_starts=np.array([], int),
                         declared_samples=np.array([], int), run_id="run_legacy", snapshot_name="0002_x",
                         n_points=4, n_needles=0)
    frame2 = SimpleNamespace(fig=_fig(), draw=lambda: None)
    assert app._draw_lanes_convergence(frame2, rd) is False
    assert app._lanes_changed(rd) is False
    app.ConvergencePlotFrame.update(frame2, rd)
    ax = frame2.fig.axes[0]
    assert "both lanes" not in ax.get_title() and "Convergence" in ax.get_title()
    assert ax.get_xlabel() == "Sample index"
    # fakes without run_dir (benchmarks' showcase calls update this way) also take the snapshot path
    assert app._draw_lanes_convergence(frame2, SimpleNamespace(Y_all=None)) is False


def test_gui_hook_falls_back_on_a_lanes_plot_error(torch, tmp_path, monkeypatch):
    app = _app_module()
    run = write_lanes_run(torch, tmp_path / "run_t13")
    msgs = []
    frame = SimpleNamespace(fig=_fig(), draw=lambda: None,
                            winfo_toplevel=lambda: SimpleNamespace(set_status=msgs.append))

    def boom(*a, **k):
        raise RuntimeError("bad lanes data")
    monkeypatch.setattr(lt, "draw_lanes_convergence", boom)
    assert app._draw_lanes_convergence(frame, SimpleNamespace(run_dir=run)) is False
    assert msgs and "Lanes convergence plot failed" in msgs[0] and "snapshot plot" in msgs[0]


# ── late init results, empty dives ──────────────────────────────────────────────────────────────

def _append_jsonl(path: Path, recs) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in recs))


def test_late_init_rows_join_the_init_band_and_log_only_packets_stay_out(torch, tmp_path):
    run = write_lanes_run(torch, tmp_path / "run_t14")
    lanes = run / "lanes"
    t = T0 + 2000.0
    x = [0.2, 0.3, 0.5]
    # qid 20: a late init result (router disposition 'init', line without a lane): 2 rows
    # qid 21: a stale legacy packet the journal only logs: no rows
    # qid 22: unattributed shared data: 1 row, stays in 'other'
    _append_jsonl(lanes / "journal.jsonl", [
        dict(t=t, ev="route", qid=20, cls="late", wid=300, disposition="init", lane=None),
        dict(t=t, ev="ingested", qid=20, wid=300, pid=20, cls="init", n=2, lane=None, act=None, zoom=None),
        dict(t=t, ev="route", qid=21, cls="legacy", wid=None, disposition="log", lane=None),
        dict(t=t, ev="route", qid=22, cls="unknown", wid=412, disposition="shared", lane=None),
        dict(t=t, ev="ingested", qid=22, wid=412, pid=21, cls="shared", n=1, lane=None, act=None, zoom=None)])
    _append_jsonl(lanes / "rows.jsonl", [
        dict(t=t, ev="row", qid=20, wid=300, lane=None, act=None, zoom_sent=None, pid=20, x_actual=x, y_opt=0.31),
        dict(t=t, ev="row", qid=20, wid=300, lane=None, act=None, zoom_sent=None, pid=20, x_actual=x, y_opt=0.33),
        dict(t=t, ev="row", qid=22, wid=412, lane=None, act=None, zoom_sent=None, pid=21, x_actual=x, y_opt=0.2)])
    R = lt.load_lanes_run(run)
    assert R.late_init.tolist() == [False] * 15 + [True, True, False]
    L = lt.dive_layout(R)
    assert (L.n_init, L.n_late, L.n_band) == (4, 2, 6)
    assert L.order[:6].tolist() == [0, 1, 2, 3, 19, 20]                  # init.pt rows, then the late ones
    assert [b["label"] for b in L.blocks] == ["F1", "B1", "F2", "B2", "other"]
    assert L.blocks[0]["start"] == 6 and L.blocks[-1]["n"] == 1
    assert L.order[L.blocks[-1]["start"]:].tolist() == [21]               # only the shared row is 'other'
    assert 4 in L.packet_starts.tolist() and L.zoom_starts[0] == 6        # own packet; no band before F1
    nd = L.needles[0]
    assert (nd["star"], nd["declared"]) == (13, 13)                       # moved with the 2 late rows
    fig = _fig()
    chk = lt.draw_lanes_convergence(fig, R)
    fig.canvas.draw()
    assert chk["n_late_init"] == 2 and chk["n_rows"] == 18
    title = getattr(fig.axes[0], "lanes_summary", "").replace("\n", " ")
    assert "4 init + 2 late init + 16 measured pts" in title
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "initialization (6 pts)" in labels


def test_a_started_dive_without_rows_is_an_empty_labelled_block(torch, tmp_path):
    run = write_lanes_run(torch, tmp_path / "run_t15")
    # the front lane starts act 4 (its 3rd dive): a decision + issue, nothing back yet
    _append_jsonl(run / "lanes" / "journal.jsonl", [
        dict(t=T0 + 900.0, ev="decision", lane="front", kind="regular", act=4, zoom=0),
        dict(t=T0 + 900.0, ev="issue", lane="front", role="main", slot=0, wid=150, act=4, zoom=0)])
    R = lt.load_lanes_run(run)
    L = lt.dive_layout(R, gap=5)
    assert [b["label"] for b in L.blocks] == ["F1", "B1", "F2", "B2", "F3"]
    b = L.blocks[-1]
    assert b["empty"] and b["open"] and b["n"] == 0 and b["key"] == (0, 4) and b["end"] - b["start"] == 5
    assert L.order[b["start"]:b["end"]].tolist() == [-1] * 5 and np.isnan(L.y[b["start"]:]).all()
    assert b["start"] in L.packet_starts.tolist() and b["start"] in L.zoom_starts.tolist()
    assert all(lv["key"] != (0, 4) for lv in L.levels)
    fig = _fig()
    chk = lt.draw_lanes_convergence(fig, R)
    fig.canvas.draw()
    ax = fig.axes[0]
    texts = {t.get_text() for t in ax.texts}
    assert "F3 …" in texts and "no results yet" in texts
    title = getattr(ax, "lanes_summary", "").replace("\n", " ")
    assert "3 front + 2 back dives, 2 running" in title                   # B2 and the empty F3 (F2 is over)
    assert [b["open"] for b in chk["blocks"]] == [False, False, False, True, True]
    assert "dive started, no results yet" in [t.get_text() for t in ax.get_legend().get_texts()]
    assert chk["blocks"][-1]["empty"] and chk["labels_hidden"] == 0


# ── the init view: a run still in its axial initialization ──────────────────────────────────────

OPT_DIMS = [0, 2, 3, 4, 8, 9]
COMPS = [4, 5, 2, 1, 3, 0]                     # trace i varies optimizing dim COMPS[i] = real dim 8, 9, 3, 2, 4, 0


def _xs(comp, fracs):
    """Measured x (optimizing dims) along an axial trace: ``comp`` at each fraction, the rest shared equally."""
    out = []
    for f in fracs:
        row = [(1.0 - f) / 5.0] * 6
        row[comp] = f
        out.append(row)
    return out


def write_init_run(run_dir: Path, *, status=("done", "done", "sent", "sent", "queued", "queued"), data=None,
                   complete=False, sent_in=([0], [0], [1], [1], [], []), opt_dims=OPT_DIMS, run_log=None):
    """A run folder in its axial initialization: init_state.json as scripts/run_zombi_main.py writes it, no
    lanes/ folder, no snapshots. Default: traces 0 (dim 8) and 1 (dim 9) measured - trace 1's block arrived
    first - traces 2, 3 sent, 4, 5 still queued. Dim 8's points were measured at own fractions 0.9, 0.1, 0.5
    (not in order), dim 9's at 0.0, 0.3."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if data is None:
        data = [[{"x": _xs(4, [0.9, 0.1, 0.5]), "y": [0.8, 0.7, 0.75], "t": T0 + 3600.0}],   # dim 8
                [{"x": _xs(5, [0.0, 0.3]), "y": [0.5, 0.6], "t": T0 + 1800.0}],              # dim 9: first
                [], [], [], []]
    traces = [dict(component=c, start=[0.0] * 6, end=[0.2] * 6) for c in COMPS]
    st = dict(version=2, design="axial", optimizing_dims=opt_dims, complete=complete,
              queue=dict(traces=traces, max_sends=3, points_per_trace=24, status=list(status),
                         sends=[len(s) for s in sent_in], sent_in=[list(s) for s in sent_in], data=data, queue=[],
                         writes=[dict(no=0, main=0, cache=1, t=T0), dict(no=1, main=2, cache=3, t=T0 + 1800.0)],
                         result_times=[b["t"] for tr in data for b in tr], first_write_t=T0))
    if opt_dims is None:
        del st["optimizing_dims"]
    (run_dir / "init_state.json").write_text(json.dumps(st), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(dict(status="running", source="hardware")))
    if run_log is not None:
        (run_dir / "run.log").write_text(run_log, encoding="utf-8")
    return run_dir


def test_run_kind_init_lanes_legacy_and_nothing(torch, tmp_path):
    init = write_init_run(tmp_path / "run_i1")
    assert lt.run_kind(init) == "init" and not lt.is_lanes_run(init)
    sig = lt.lanes_signature(init)
    assert sig is not None and ("init_state.json",) == tuple(s[0] for s in sig if s[1] is not None)
    legacy = write_init_run(tmp_path / "run_i2")
    (legacy / "snapshots" / "0001_init").mkdir(parents=True)
    assert lt.run_kind(legacy) == "legacy_init" and lt.lanes_signature(legacy) is None
    legacy2 = write_init_run(tmp_path / "run_i3")
    (legacy2 / "latest.txt").write_text("0001_init")
    assert lt.run_kind(legacy2) == "legacy_init"
    lanes = write_lanes_run(torch, tmp_path / "run_i4")
    write_init_run(lanes)
    assert lt.run_kind(lanes) == "lanes"                       # lanes data wins over init_state.json
    empty = tmp_path / "run_i5"
    (empty / "lanes").mkdir(parents=True)
    assert lt.run_kind(empty) is None and lt.run_kind(None) is None


def test_init_state_traces_by_real_dimension_in_arrival_order(tmp_path):
    S = lt.load_init_state(write_init_run(tmp_path / "run_i6"))
    assert S["y"].tolist() == [0.5, 0.6, 0.8, 0.7, 0.75] and S["trace"].tolist() == [1, 1, 0, 0, 0]
    assert S["sizes"] == [2, 3] and S["block_trace"] == [1, 0] and S["d"] == 6
    assert [t["label"] for t in S["traces"]] == ["dim 8", "dim 9", "dim 3", "dim 2", "dim 4", "dim 0"]
    assert [t["status"] for t in S["traces"]] == ["done", "done", "sent", "sent", "queued", "queued"]
    assert [t["n_points"] for t in S["traces"]] == [3, 2, 0, 0, 0, 0] and S["complete"] is False
    assert np.allclose(S["own"], [0.0, 0.3, 0.9, 0.1, 0.5])            # each point's own-dim fraction
    assert [t["requeued"] for t in S["traces"]] == [False] * 6
    R = lt.load_lanes_run(tmp_path / "run_i6")
    assert R.kind == "init" and R.init_source == "init_state.json" and R.n_init == 5 and len(R.y) == 0
    assert R.init_sizes == [2, 3] and R.init_state is not None and not R.warnings


def test_init_state_side_file_with_more_progress_wins(tmp_path):
    # a save Dropbox blocked leaves init_state.json.<x>.tmp; the run resumes from the one with more progress
    run = write_init_run(tmp_path / "run_i16")
    st = json.loads((run / "init_state.json").read_text(encoding="utf-8"))
    st["queue"]["data"][2] = [{"x": [[1 / 6] * 6], "y": [0.9], "t": T0 + 7200.0}]
    st["queue"]["status"][2] = "done"
    st["queue"]["result_times"].append(T0 + 7200.0)
    (run / "init_state.json.123.tmp").write_text(json.dumps(st), encoding="utf-8")
    S = lt.load_init_state(run)
    assert S["y"].tolist()[-1] == 0.9 and S["traces"][2]["status"] == "done"
    assert any(s[0] == "init_state.json.123.tmp" for s in lt.lanes_signature(run))


def test_init_view_draws_traces_running_best_and_pending(tmp_path):
    import matplotlib.colors as mcolors
    fig = _fig((11, 6.2))
    chk = lt.draw_lanes_convergence(fig, lt.load_lanes_run(write_init_run(tmp_path / "run_i7")))
    fig.canvas.draw()
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    title = getattr(ax, "lanes_summary", "").replace("\n", " ")
    assert "run_i7 — initialization: 2 of 6 traces measured (5 points); lanes not started" in title
    assert chk["kind"] == "init" and (chk["n_done"], chk["n_traces"], chk["n_init"]) == (2, 6, 5)
    assert chk["pending"] == ["dim 3", "dim 2", "dim 4", "dim 0"]          # sent first, then queued
    # points coloured per trace (by trace index), packets in arrival order; each packet's points by the trace's
    # own-dim fraction, 0 -> 1 left to right (dim 8 measured at 0.9, 0.1, 0.5: drawn 0.1, 0.5, 0.9)
    pts = {mcolors.to_hex(c.get_facecolor()[0]): c.get_offsets() for c in ax.collections if len(c.get_offsets())}
    d8, d9 = np.asarray(pts[lt.TRACE_C[0]]), np.asarray(pts[lt.TRACE_C[1]])
    assert d9.tolist() == [[0, 0.5], [1, 0.6]] and d8.tolist() == [[2, 0.7], [3, 0.75], [4, 0.8]]
    rb = [ln for ln in ax.get_lines() if ln.get_label() == "running best"]
    assert len(rb) == 1 and np.allclose(rb[0].get_ydata(), [0.5, 0.6, 0.7, 0.75, 0.8])
    assert "own dimension's fraction 0 → 1" in ax.get_xlabel()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    for lab in ("dim 8 trace: 3 pts", "dim 9 trace: 2 pts", "running best", "initialization (5 pts)",
                "dim 3 trace: sent, awaiting result", "dim 0 trace: queued, not sent yet", "trace awaiting its result"):
        assert lab in labels, lab
    texts = {t.get_text() for t in ax.texts}
    assert {"dim 9", "dim 8", "dim 3", "dim 0", "awaiting result", "queued"} <= texts
    assert chk["labels_hidden"] == 0
    assert ax.get_xlim()[1] > 5 + 4 * 2                                  # room for the 4 pending slots


def test_init_view_before_any_result_and_when_complete(tmp_path):
    fig = _fig()
    run = write_init_run(tmp_path / "run_i8", status=("sent", "sent", "queued", "queued", "queued", "queued"),
                         data=[[], [], [], [], [], []])
    chk = lt.draw_lanes_convergence(fig, lt.load_lanes_run(run))
    fig.canvas.draw()
    ax = fig.axes[0]
    assert "0 of 6 traces measured (0 points)" in getattr(ax, "lanes_summary", "").replace("\n", " ")
    assert "waiting for the first init result" in {t.get_text() for t in ax.texts}
    assert chk["n_init"] == 0 and len(chk["pending"]) == 6
    pt = [[1 / 6] * 6]
    data = [[{"x": pt, "y": [0.1 * (i + 1)], "t": T0 + 100.0 * i}] for i in range(6)]
    fig2 = _fig()
    run2 = write_init_run(tmp_path / "run_i9", status=("done",) * 6, data=data, complete=True)
    chk2 = lt.draw_lanes_convergence(fig2, lt.load_lanes_run(run2))
    fig2.canvas.draw()
    assert "initialization complete: 6 of 6 traces measured (6 points); lanes not started" in \
        getattr(fig2.axes[0], "lanes_summary", "").replace("\n", " ")
    assert chk2["pending"] == []


def test_cli_exits_0_on_an_initialising_run_and_2_when_there_is_nothing_to_draw(tmp_path, capsys):
    from visualization import render_lanes_convergence as cli
    run = write_init_run(tmp_path / "run_i10")
    before = sorted((p.relative_to(run).as_posix(), p.stat().st_mtime_ns) for p in run.rglob("*"))
    out = tmp_path / "out" / "init.png"
    assert cli.main(["--run", str(run), "--out", str(out), "--dpi", "60"]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    line = capsys.readouterr().out.strip()
    assert "initialization, 2 of 6 traces measured (5 points)" in line and "pending: dim 3 dim 2 dim 4 dim 0" in line
    assert sorted((p.relative_to(run).as_posix(), p.stat().st_mtime_ns) for p in run.rglob("*")) == before
    legacy = write_init_run(tmp_path / "run_i11")
    (legacy / "snapshots" / "0001_init").mkdir(parents=True)
    # a snapshot-optimizer run past init: the PNG is only its (stale) init view -> its own exit code, 3
    assert cli.main(["--run", str(legacy), "--out", str(tmp_path / "legacy.png"), "--dpi", "60"]) == 3
    assert (tmp_path / "legacy.png").is_file() and "snapshot run past init" in capsys.readouterr().out
    empty = tmp_path / "run_i12"
    empty.mkdir()
    assert cli.main(["--run", str(empty), "--out", str(tmp_path / "e.png")]) == 2
    assert "nothing to draw" in capsys.readouterr().err
    assert not (tmp_path / "e.png").exists()


def test_gui_hook_draws_an_initialising_run_and_redraws_when_init_state_changes(tmp_path):
    app = _app_module()
    run = write_init_run(tmp_path / "run_i13")
    frame = SimpleNamespace(fig=_fig(), draw=lambda: None)
    assert app._draw_lanes_convergence(frame, SimpleNamespace(run_dir=run)) is True
    assert "initialization: 2 of 6 traces measured" in getattr(frame.fig.axes[0], "lanes_summary", "").replace("\n", " ")
    assert app._lanes_changed(SimpleNamespace(run_dir=run)) is False
    st = json.loads((run / "init_state.json").read_text(encoding="utf-8"))
    st["queue"]["data"][2].append({"x": [[1 / 6] * 6], "y": [0.9], "t": T0 + 7200.0})
    st["queue"]["status"][2] = "done"
    (run / "init_state.json").write_text(json.dumps(st) + " ", encoding="utf-8")
    assert app._lanes_changed(SimpleNamespace(run_dir=run)) is True
    assert app._draw_lanes_convergence(frame, SimpleNamespace(run_dir=run)) is True
    assert "3 of 6 traces measured (6 points)" in getattr(frame.fig.axes[0], "lanes_summary", "").replace("\n", " ")
    # an init_state.json + snapshots run (snapshot optimizer past init) keeps the snapshot plot
    legacy = write_init_run(tmp_path / "run_i14")
    (legacy / "snapshots" / "0001_init").mkdir(parents=True)
    assert app._draw_lanes_convergence(frame, SimpleNamespace(run_dir=legacy)) is False
    assert app._lanes_changed(SimpleNamespace(run_dir=legacy)) is False


def test_gui_hook_renders_inside_its_try_and_falls_back_on_a_draw_error(torch, tmp_path):
    app = _app_module()
    msgs = []

    def boom():
        raise RuntimeError("render failed")
    for run in (write_lanes_run(torch, tmp_path / "run_t16"), write_init_run(tmp_path / "run_i15")):
        frame = SimpleNamespace(fig=_fig(), draw=lambda: None, canvas=SimpleNamespace(draw=boom),
                                winfo_toplevel=lambda: SimpleNamespace(set_status=msgs.append))
        assert app._draw_lanes_convergence(frame, SimpleNamespace(run_dir=run)) is False
        assert frame.fig.axes == []                                       # cleared for the snapshot plot
    assert len(msgs) == 2 and all("render failed" in m and "snapshot plot" in m for m in msgs)


# ── review fixes: init view wording, CLI exit codes, running-dive labels ───────────────────────

def _legend(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def test_init_view_requeued_and_empty_traces_are_not_called_queued_or_measured(tmp_path):
    # trace 3 (dim 2) was sent in write 1, presumed lost and queued again; trace 2 (dim 3) reported an EMPTY
    # packet (InitQueue 'empty': a block with no points)
    data = [[{"x": _xs(4, [0.1, 0.5, 0.9]), "y": [0.7, 0.75, 0.8], "t": T0 + 3600.0}],
            [{"x": _xs(5, [0.0, 0.3]), "y": [0.5, 0.6], "t": T0 + 1800.0}],
            [{"x": [], "y": [], "t": T0 + 5000.0}], [], [], []]
    run = write_init_run(tmp_path / "run_r1", status=("done", "done", "done", "queued", "sent", "queued"),
                         sent_in=([0], [0], [1], [1], [2], []), data=data)
    S = lt.load_init_state(run)
    assert [(t["sends"], t["requeued"]) for t in S["traces"]][3:] == [(1, True), (1, False), (0, False)]
    fig = _fig((11, 6.2))
    chk = lt.draw_lanes_convergence(fig, lt.load_lanes_run(run))
    fig.canvas.draw()
    ax = fig.axes[0]
    title = getattr(ax, "lanes_summary", "").replace("\n", " ")
    assert "initialization: 3 of 6 traces reported (1 empty, 5 points); lanes not started" in title
    assert "hatched: 1 trace awaiting results, 1 re-queued (presumed lost), 1 queued" in title
    assert chk["pending"] == ["dim 4", "dim 2", "dim 0"]                  # sent, re-queued, queued
    assert (chk["n_empty"], chk["n_sent"], chk["n_requeued"], chk["n_queued"]) == (1, 1, 1, 1)
    lab = _legend(ax)
    for want in ("dim 2 trace: re-queued (sent 1×, presumed lost)", "dim 4 trace: sent, awaiting result",
                 "dim 0 trace: queued, not sent yet", "dim 3 trace: empty result"):
        assert want in lab, want
    assert "dim 2 trace: queued, not sent yet" not in lab
    assert "re-queued" in {t.get_text() for t in ax.texts}
    from visualization import render_lanes_convergence as cli
    chk["out"] = "x.png"
    line = cli.summary_line(chk)
    assert "3 of 6 traces reported (1 empty, 5 points)" in line and "(1 re-queued)" in line


def test_init_view_labels_without_optimizing_dims_and_a_non_lanes_optimizer(tmp_path):
    S = lt.load_init_state(write_init_run(tmp_path / "run_r2", opt_dims=None))
    assert [t["label"] for t in S["traces"]][:2] == ["component 4", "component 5"]   # never 'dim 4'
    assert S["traces"][0]["dim"] is None
    run = write_init_run(tmp_path / "run_r3", run_log="===== hardware run launched =====\n"
                                                      "[Main] Optimizer: zombihop\n")
    fig = _fig()
    chk = lt.draw_lanes_convergence(fig, lt.load_lanes_run(run))
    title = getattr(fig.axes[0], "lanes_summary", "").replace("\n", " ")
    assert "optimizer 'zombihop': its snapshot plot follows the init" in title and "lanes not started" not in title
    assert chk["optimizer"] == "zombihop"
    run_l = write_init_run(tmp_path / "run_r4", run_log="[Main] Optimizer: lanes\n")
    fig = _fig()
    lt.draw_lanes_convergence(fig, lt.load_lanes_run(run_l))
    assert "lanes not started" in getattr(fig.axes[0], "lanes_summary", "").replace("\n", " ")


def test_init_view_says_unreadable_when_init_state_is_locked(tmp_path, monkeypatch):
    run = write_init_run(tmp_path / "run_r5")
    real = lt._read_bytes
    monkeypatch.setattr(lt, "_read_bytes", lambda p: None if Path(p).name == "init_state.json" else real(p))
    R = lt.load_lanes_run(run)
    assert R.init_state is None and any("could not be read" in w for w in R.warnings)
    fig = _fig()
    lt.draw_lanes_convergence(fig, R)
    assert any("init_state.json unreadable" in t.get_text() for t in fig.axes[0].texts)


def test_trace_palette_has_no_orange_or_amber_and_repeats_with_a_dark_edge():
    import matplotlib.colors as mcolors
    assert len(set(lt.TRACE_C)) == len(lt.TRACE_C) >= 6
    for c in lt.TRACE_C:                                  # nothing that reads as the darkorange running best
        h, s, v = mcolors.rgb_to_hsv(mcolors.to_rgb(c))
        assert not (0.04 <= h <= 0.17 and s > 0.5), c
    styles = [lt._trace_style(i) for i in range(10)]     # a 10-D init: no two traces share colour + marker + edge
    assert len(set(styles)) == 10
    assert styles[0][2] == "white" and styles[len(lt.TRACE_C)][2] != "white"


def test_running_dive_is_padded_to_a_minimum_width_without_touching_its_rows(torch, tmp_path):
    R = lt.load_lanes_run(write_lanes_run(torch, tmp_path / "run_r6"))
    L0 = lt.dive_layout(R)
    L = lt.dive_layout(R, min_open=4)
    f2, b2 = L.blocks[2], L.blocks[3]                      # F2, B2: running, 1 row each
    assert (f2["label"], f2["open"], f2["n"], f2["pad"]) == ("F2", True, 1, 3)
    assert f2["end"] - f2["start"] == 4 and L.order[f2["start"]:f2["end"]].tolist() == [16, -1, -1, -1]
    assert np.isnan(L.y[f2["start"] + 1:f2["end"]]).all() and b2["start"] == f2["end"]
    assert [b["pad"] for b in L.blocks[:2]] == [0, 0]       # finished dives are never padded
    # same bars (rows, levels), only shifted; the padding is an unshaded stretch (zoom level 0)
    assert [(lv["n"], round(lv["level"], 6), lv["dip"]) for lv in L.levels] == \
        [(lv["n"], round(lv["level"], 6), lv["dip"]) for lv in L0.levels]
    assert f2["start"] + 1 in L.zoom_starts.tolist()
    assert L.zoom_levels[L.zoom_starts.tolist().index(f2["start"] + 1)] == 0
    fig = _fig()
    chk = lt.draw_lanes_convergence(fig, R)
    fig.canvas.draw()
    rb = [ln for ln in fig.axes[0].get_lines() if ln.get_color() == "darkorange"]
    assert all(np.isfinite(ln.get_ydata()).all() for ln in rb)
    assert max(len(ln.get_xdata()) for ln in rb[-2:]) == 1  # F2 / B2 running best: their one row, no padding
    assert chk["labels_hidden"] == 0


def test_fixture_running_dives_keep_their_labels_at_gui_size(torch):
    R = lt.load_lanes_run(_fixture("fixture_run"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(7, 4), dpi=96, tight_layout=True)        # interface/app.py PlotFrame's figure
    FigureCanvasAgg(fig)
    chk = lt.draw_lanes_convergence(fig, R)
    fig.canvas.draw()
    texts = {t.get_text() for t in fig.axes[0].texts}
    assert chk["labels_hidden"] == 0 and ({"F4 …", "B4 …"} <= texts or {"F4", "B4"} <= texts)
    assert [b["pad"] > 0 for b in chk["blocks"] if b["label"] in ("F4", "B4")] == [True, True]
    assert chk["red_bars"] == 2 and chk["clean"] == 4 and chk["finished"] == 6       # unchanged by the padding
    title = getattr(fig.axes[0], "lanes_summary", "")
    assert len(title.splitlines()) <= 4                                              # short note at GUI size


def test_legend_lists_only_the_bars_that_are_drawn(torch, tmp_path):
    # the lanes phase just started: init.pt and the first decisions, no rows yet -> no bars at all
    run = tmp_path / "run_r7"
    (run / "lanes").mkdir(parents=True)
    torch.save(dict(Y=torch.tensor([[0.1], [0.2], [0.15]]), X_actual=torch.zeros(3, D)), run / "lanes" / "init.pt")
    _jsonl(run / "lanes" / "journal.jsonl", [
        dict(t=T0, ev="start", n_init=3, n_prior_rows=0, epoch=T0),
        dict(t=T0 + 10, ev="decision", lane="front", kind="regular", act=0, zoom=0),
        dict(t=T0 + 10, ev="decision", lane="back", kind="regular", act=1, zoom=0)])
    fig = _fig()
    chk = lt.draw_lanes_convergence(fig, lt.load_lanes_run(run))
    fig.canvas.draw()
    lab = _legend(fig.axes[0])
    assert chk["bars"] == [] and not any("top-8" in s for s in lab)
    assert "dive started, no results yet" in lab
    # with bars but no dip: the black entry only
    packets = [p for p in PACKETS if p[3] != 1]                          # drop the dipping back dive
    fig = _fig()
    lt.draw_lanes_convergence(fig, lt.load_lanes_run(write_lanes_run(torch, tmp_path / "run_r8", packets=packets)))
    lab = _legend(fig.axes[0])
    assert "measured top-8 of the zoom" in lab and "top-8 > 0.01 below previous zoom" not in lab


def test_corrupt_rows_are_not_called_a_partial_last_line(torch, tmp_path):
    run = write_lanes_run(torch, tmp_path / "run_r9")
    p = run / "lanes" / "rows.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    p.write_text("".join(lines[:3]) + "garbage\n{bad\n" + "".join(lines[3:]), encoding="utf-8")
    R = lt.load_lanes_run(run)
    assert any("2 unreadable line(s) skipped (a corrupt file?)" in w for w in R.warnings)
    assert not any("partial last line" in w for w in R.warnings)


def test_title_wrap_is_balanced():
    text = "run_x — Convergence, both lanes  (176 init + 1621 measured pts; 4 front + 4 back dives)"
    for width in (40, 60, 80):
        got = lt._wrap(text, width)
        assert len(got) == len(__import__("textwrap").wrap(text, width))
        assert all(len(s) <= width for s in got)
        if len(got) > 1:
            assert min(len(s) for s in got) >= 0.5 * max(len(s) for s in got)   # no orphan word


def test_cli_exit_codes_summary_and_out_checks(torch, tmp_path, capsys, monkeypatch):
    import time as _time
    from visualization import render_lanes_convergence as cli
    run = write_lanes_run(torch, tmp_path / "run_r10")
    t = T0 + 2000.0
    _append_jsonl(run / "lanes" / "journal.jsonl", [
        dict(t=t, ev="route", qid=20, cls="late", wid=300, disposition="init", lane=None)])
    _append_jsonl(run / "lanes" / "rows.jsonl", [
        dict(t=t, ev="row", qid=20, wid=300, lane=None, act=None, zoom_sent=None, y_opt=0.31),
        dict(t=t, ev="row", qid=20, wid=300, lane=None, act=None, zoom_sent=None, y_opt=0.33)])
    out = tmp_path / "o" / "c.png"
    assert cli.main(["--run", str(run), "--out", str(out), "--dpi", "50"]) == 0
    line = capsys.readouterr().out
    assert "4 init (+2 late) + 15 rows" in line                          # late init rows counted once
    # --out is a directory: refused at once (exit 2), no retry loop
    t0 = _time.monotonic()
    assert cli.main(["--run", str(run), "--out", str(tmp_path / "o")]) == 2
    assert _time.monotonic() - t0 < 5.0 and "is a directory" in capsys.readouterr().err
    # an error while drawing (even a ValueError) is exit 1, not 'nothing to draw'
    def boom(*a, **k):
        raise ValueError("Singular matrix")
    monkeypatch.setattr(lt, "draw_lanes_convergence", boom)
    assert cli.main(["--run", str(run), "--out", str(tmp_path / "o" / "d.png")]) == 1
    assert "failed" in capsys.readouterr().err and not (tmp_path / "o" / "d.png").exists()


# ── CPU only ────────────────────────────────────────────────────────────────────────────────────

def test_import_and_draw_never_initialise_cuda(torch, tmp_path):
    # A fresh process (no src / plot_convergence loaded yet, as the PNG tool) that SEES a GPU: is_available()
    # is stubbed True and CUDA initialisation is stubbed to record and fail, so nothing reaches a real device
    # (CUDA_VISIBLE_DEVICES=-1 as a second guard; an empty value does not hide the GPU on Windows). Importing
    # lanes_timeline imports no torch; loading and drawing a lanes run and an initialising run never
    # initialises CUDA, although plot_convergence imports src (src/core/zombihop.py claims the GPU at import).
    import os
    import subprocess
    run = write_lanes_run(torch, tmp_path / "run_t17")
    init = write_init_run(tmp_path / "run_i17")
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import visualization.lanes_timeline as lt\n"
        "imported_torch = 'torch' in sys.modules\n"
        "import torch\n"
        "calls = []\n"
        "def no_init(*a, **k):\n"
        "    calls.append(1)\n"
        "    raise RuntimeError('CUDA initialisation attempted')\n"
        "torch.cuda._lazy_init = no_init\n"
        "torch.cuda.is_available = lambda: True\n"
        "from matplotlib.backends.backend_agg import FigureCanvasAgg\n"
        "from matplotlib.figure import Figure\n"
        f"for run in ({str(run)!r}, {str(init)!r}):\n"
        "    fig = Figure(figsize=(8, 4)); FigureCanvasAgg(fig)\n"
        "    lt.draw_lanes_convergence(fig, lt.load_lanes_run(run)); fig.canvas.draw()\n"
        "print(json.dumps(dict(imported_torch=imported_torch, calls=len(calls),"
        " initialized=torch.cuda.is_initialized())))\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", MPLBACKEND="Agg")
    res = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), env=env, capture_output=True,
                         text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
    got = json.loads(res.stdout.strip().splitlines()[-1])
    assert got == dict(imported_torch=False, calls=0, initialized=False)


# ── the gui_plot fixtures (real lanes_hw output; skipped when absent) ───────────────────────────

GUI_PLOT = ROOT / "benchmarks" / "zoom_fix" / "lanes" / "report" / "review" / "impl" / "gui_plot"


def _fixture(name):
    p = GUI_PLOT / name
    if not (p / "lanes" / "rows.jsonl").is_file():
        pytest.skip(f"{p} not present")
    return p


def test_fixture_run_matches_fixture_md(torch):
    R = lt.load_lanes_run(_fixture("fixture_run"))
    assert R.kind == "lanes" and R.init_source == "lanes/init.pt"
    assert R.n_init == 176 and R.init_sizes == [23, 20, 24, 24, 18, 23, 23, 21] and len(R.y) == 1621
    assert int(R.late_init.sum()) == 0                    # its init / legacy packets are log-only
    L = lt.dive_layout(R)
    assert L.n_band == 176 and [b["label"] for b in L.blocks] == ["F1", "B1", "F2", "B2", "F3", "B3", "F4", "B4"]
    rows = {(b["label"], z): sum(1 for p in range(b["start"], b["end"]) if L.zoom[p] == z)
            for b in L.blocks for z in (0, 1, 2)}
    assert [rows[("F1", z)] for z in (0, 1, 2)] == [87, 70, 91]
    assert [rows[("B3", z)] for z in (0, 1, 2)] == [89, 108, 24]
    dips = [(next(b["label"] for b in L.blocks if b["key"] == lv["key"]), lv["zoom"]) for lv in L.levels if lv["dip"]]
    assert dips == [("B1", 2), ("B3", 2)]
    assert [round(float(nd["value"]), 4) for nd in R.needles] == [0.3986, 0.61, 0.5427, 0.5328, 0.5404]
    assert [b["reason"] for b in L.blocks][5:] == ["no_own_unpen", None, None]


def test_fixture_run_small_has_two_running_dives(torch):
    R = lt.load_lanes_run(_fixture("fixture_run_small"))
    fig = _fig((11, 6.2))
    chk = lt.draw_lanes_convergence(fig, R)
    fig.canvas.draw()
    dives = [b for b in chk["blocks"] if b["key"] is not None]
    assert [(b["label"], b["open"], b["empty"]) for b in dives] == [
        ("F1", False, False), ("B1", False, False), ("F2", True, False), ("B2", True, True)]
    assert "2 front + 2 back dives, 2 running" in getattr(fig.axes[0], "lanes_summary", "").replace("\n", " ")
    assert "B2 …" in {t.get_text() for t in fig.axes[0].texts}


def test_no_visible_title_and_blue_lanes(torch, tmp_path):
    """User, 14 Nov: no title on the plot; the lanes' strip blocks are pale / darker blue."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from visualization import lanes_timeline as lt
    assert lt.LANE_C["front"].lower() != lt.LANE_C["back"].lower()
    for c in lt.LANE_C.values():                      # blue: the blue channel dominates
        r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
        assert b > r and b >= g
    run = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "zoom_fix", "lanes", "report", "review",
                       "impl", "gui_plot", "fixture_run")
    if not os.path.isdir(run):
        pytest.skip("fixture_run not present")
    fig = Figure(figsize=(9, 4), dpi=96)
    FigureCanvasAgg(fig)
    chk = lt.draw_lanes_convergence(fig, lt.load_lanes_run(run))
    fig.canvas.draw()
    assert fig.axes[0].get_title().strip() == ""
    assert "both lanes" in chk["summary"] and chk["summary"] == fig.axes[0].lanes_summary


def test_dive_exact_resume_continues_the_dives(torch, tmp_path):
    """A ZoMBI restart that resumed dive-exact (journal 'resume_dives') keeps the same dives open: no
    'cut off by a ZoMBI restart' end and no new empty F#/B# blocks."""
    import os
    import shutil
    import time as _t
    from visualization import lanes_timeline as lt
    src = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "zoom_fix", "lanes", "report", "review",
                       "impl", "gui_plot", "fixture_run_small")
    if not os.path.isdir(src):
        pytest.skip("fixture_run_small not present")
    run = tmp_path / "run_fx"
    shutil.copytree(src, run)
    before = lt.load_lanes_run(str(run))
    open_before = {k: v for k, v in before.acts.items() if v.get("open")}
    assert open_before, "fixture must have running dives"
    lanes = {v["lane"]: dict(act=int(k[1]), zoom=1) for k, v in open_before.items()}
    t = _t.time() + 10
    with open(run / "lanes" / "journal.jsonl", "a", encoding="utf-8") as fh:
        fh.write("\n" + json.dumps({"t": t, "ev": "start", "resumed": True, "n_init": 1, "n_prior_rows": 1}) + "\n")
        fh.write(json.dumps({"t": t + 0.1, "ev": "resume_dives", "lanes": lanes}) + "\n")
        for lane, st in lanes.items():
            fh.write(json.dumps({"t": t + 1, "ev": "decision", "lane": lane, "kind": "start", "act": st["act"],
                                 "zoom": 1, "main": {}, "backup": {}}) + "\n")
    log = run / "run.log"                       # a real restart appends its own banner to run.log
    if log.exists():
        with open(log, "a", encoding="utf-8") as fh:
            fh.write("\nSTARTING LANES OPTIMIZATION (preset=leader; four-slot protocol)\n")
    after = lt.load_lanes_run(str(run))
    assert after.n_sessions == before.n_sessions + 1
    assert not any(v.get("reason") == "restart" for v in after.acts.values())
    assert sorted(after.acts) == sorted(before.acts)            # same dives, no new (session, act) keys
    assert all(after.acts[k].get("open") for k in open_before)
