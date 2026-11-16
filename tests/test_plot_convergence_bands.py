"""Search-phase timeline, bands and packet orientation on the convergence plot
(``visualization/plot_convergence.py``, shared with the GUI's Convergence tab).

Builds a synthetic snapshot history named the way ZoMBIHop names it — an
untagged ``init`` snapshot, ``actN_zM_iK`` snapshots, untagged ``final`` — and
checks the replayed init size, activation / zoom boundaries and result-packet
starts, that only the init design and zoomed-in stretches are shaded (zoom 0 is
the full field), and that each packet is drawn with its higher-Y end last while
per-point attributes follow their points.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pc = pytest.importorskip("visualization.plot_convergence")


def _write_snapshots(torch, run_dir: Path, snaps) -> None:
    """``snaps``: (label, n_new) per snapshot, in order -> NNNN_label/delta.pt."""
    root = run_dir / "snapshots"
    root.mkdir(parents=True)
    for i, (label, n_new) in enumerate(snaps, start=1):
        d = root / f"{i:04d}_{label}"
        d.mkdir()
        torch.save({"X_new": torch.zeros(n_new, 3, dtype=torch.float64),
                    "Y_new": torch.zeros(n_new, 1, dtype=torch.float64)},
                   d / "delta.pt")


# 4 init samples; activation 0 searches the full field, then zooms in twice;
# activation 1 starts back on the full field. Snapshots with no new points
# never open a stretch.
_HISTORY = [
    ("init", 4),                                   # init:       samples 0-3
    ("act0_z0_i0", 3), ("act0_z0_i1", 2),          # full field: 4-8
    ("act0_z1_i0", 2),                             # 1st zoom:   9-10
    ("act0_z2_i0", 2), ("act0_z2_i0_needle", 0),   # 2nd zoom:   11-12
    ("act1_z0_i0", 3),                             # full field: 13-15
    ("act1_z1_i0", 0),                             # (empty)
    ("act1_z1_i1", 2),                             # 1st zoom:   16-17
    ("final", 0),
]


def _x_extent(patch) -> tuple[float, float]:
    """Data-x extent of an axvspan patch (a Rectangle or Polygon, by mpl version)."""
    xs = patch.get_patch_transform().transform(patch.get_path().vertices)[:, 0]
    return float(xs.min()), float(xs.max())


def test_timeline_init_activation_and_zoom_boundaries(torch, tmp_path):
    _write_snapshots(torch, tmp_path, _HISTORY)
    starts, declared, n_init, z_starts, z_levels, packets = pc.load_activation_timeline(
        tmp_path, "0010_final")
    assert n_init == 4
    assert starts.tolist() == [13]   # activation 1 only; activation 0 starts at n_init
    assert z_starts.tolist() == [4, 9, 11, 13, 16]
    assert z_levels.tolist() == [0, 1, 2, 0, 1]
    assert declared.tolist() == []
    # One packet per snapshot that adds points. Without init_state.json the
    # init design cannot be split, so it is not listed and keeps its order.
    assert packets.tolist() == [4, 7, 9, 11, 13, 16]


def test_timeline_stops_at_requested_snapshot(torch, tmp_path):
    _write_snapshots(torch, tmp_path, _HISTORY)
    _, _, n_init, z_starts, z_levels, packets = pc.load_activation_timeline(
        tmp_path, "0004_act0_z1_i0")
    assert n_init == 4
    assert z_starts.tolist() == [4, 9]
    assert z_levels.tolist() == [0, 1]
    assert packets.tolist() == [4, 7, 9]


def test_timeline_run_still_in_init_is_all_init(torch, tmp_path):
    _write_snapshots(torch, tmp_path, [("init", 6)])
    starts, _, n_init, z_starts, _, packets = pc.load_activation_timeline(
        tmp_path, "0001_init")
    assert n_init == 6
    assert starts.size == 0 and z_starts.size == 0 and packets.size == 0


def _write_init_state(run_dir: Path, traces) -> None:
    """``traces``: per axial trace, a list of (n_points, arrival_time) blocks."""
    data = [[{"x": [[0.0] * 3] * n, "y": [0.5] * n, "t": t} for n, t in blocks]
            for blocks in traces]
    (run_dir / "init_state.json").write_text(json.dumps({"queue": {"data": data}}))


def test_init_packets_come_from_init_state_in_arrival_order(torch, tmp_path):
    _write_snapshots(torch, tmp_path, _HISTORY)
    # Trace 0's block arrived after trace 1's, and trace 2 never reported (an
    # empty block): the init rows were stacked as [trace 1 (1 pt), trace 0 (3 pts)].
    _write_init_state(tmp_path, [[(3, 20.0)], [(1, 10.0)], [(0, 30.0)]])
    assert pc.init_packet_sizes(tmp_path, 4) == [1, 3]
    *_, packets = pc.load_activation_timeline(tmp_path, "0010_final")
    assert packets.tolist() == [0, 1, 4, 7, 9, 11, 13, 16]


def test_init_packets_ignored_when_they_do_not_add_up(torch, tmp_path):
    _write_snapshots(torch, tmp_path, _HISTORY)
    _write_init_state(tmp_path, [[(3, 20.0)], [(2, 10.0)]])   # 5 points, init has 4
    assert pc.init_packet_sizes(tmp_path, 4) is None
    *_, packets = pc.load_activation_timeline(tmp_path, "0010_final")
    assert packets.tolist() == [4, 7, 9, 11, 13, 16]


def test_orient_packets_puts_each_packets_higher_end_last():
    #        row 0 | packet 1-3 | packet 4-5 | packet 6-7
    Y = np.array([5.0, 9.0, 2.0, 1.0, 3.0, 3.0, 7.0, 4.0])
    pm = np.array([True, False, True, True, True, False, True, False])
    needles = np.array([1, 7, 4, 99])
    Y_in, pm_in, needles_in = Y.copy(), pm.copy(), needles.copy()

    Yd, pmd, nd = pc.orient_packets(Y, [1, 4, 6], pm, needles)

    # 1-3 starts higher than it ends -> reversed; 4-5 is a tie -> kept;
    # 6-7 reversed; row 0 precedes the first packet -> left alone.
    assert Yd.tolist() == [5.0, 1.0, 2.0, 9.0, 3.0, 3.0, 4.0, 7.0]
    order = [0, 3, 2, 1, 4, 5, 7, 6]
    assert pmd.tolist() == pm[order].tolist()
    # Needles are re-pointed at where their rows are drawn; 99 is out of range.
    assert nd.tolist() == [3, 6, 4, 99]
    assert all(Yd[d] == Y[o] for d, o in zip(nd[:3], needles[:3]))
    # Display only: the inputs are untouched.
    assert np.array_equal(Y, Y_in) and np.array_equal(pm, pm_in)
    assert np.array_equal(needles, needles_in)


def test_orient_packets_without_packets_is_identity():
    Y = np.array([3.0, 1.0, 2.0])
    Yd, pmd, nd = pc.orient_packets(Y, [])
    assert Yd.tolist() == Y.tolist() and pmd is None and nd is None
    assert pc.sample_axis_label([]) == "Sample index"
    assert pc.sample_axis_label([0]) != "Sample index"


def test_bands_shade_init_and_zoom_ins_not_full_field():
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(111)
    pc.draw_search_bands(ax, 18, n_init=4, zoom_starts=[4, 9, 11, 13, 16],
                         zoom_levels=[0, 1, 2, 0, 1])

    bands = [(_x_extent(p), p.get_alpha()) for p in ax.patches]
    assert bands == [((-0.5, 3.5), pc.INIT_BAND_ALPHA),
                     ((8.5, 10.5), pc.ZOOM_BAND_ALPHAS[0]),
                     ((10.5, 12.5), pc.ZOOM_BAND_ALPHAS[1]),
                     ((15.5, 17.5), pc.ZOOM_BAND_ALPHAS[0])]
    assert ax.get_legend_handles_labels()[1] == [
        "initialization (4 pts)", "1st zoom", "2nd zoom"]


def test_bands_past_the_ramp_reuse_its_darkest_shade():
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(111)
    pc.draw_search_bands(ax, 10, n_init=0, zoom_starts=[0, 5], zoom_levels=[5, 11])
    assert [p.get_alpha() for p in ax.patches] == [pc.ZOOM_BAND_ALPHAS[-1]] * 2
    assert ax.get_legend_handles_labels()[1] == ["5th zoom", "11th zoom"]
