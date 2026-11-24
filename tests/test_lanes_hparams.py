"""src/lanes/hparams.py + src/lanes/options.py (DESIGN.md 2.2, N3; reviewer R2 item 5). Part 1 (task T1):

* registry names unique, all ``lanes_*``, disjoint from ``src.hparams.KNOWN_NAMES``;
* ``src.hparams.ALL_DEFAULTS`` / ``KNOWN_NAMES`` / ``ui_categories()`` (and the ``random_simplex``
  attributes and the core classes) are unchanged by ``import src.lanes`` and every T1 module: frozen
  before / after in a fresh interpreter (CUDA hidden; ~5 s);
* defaults == the leader values of DESIGN 2.2 (+ R2's ``lanes_ingest_unattributed``);
* ``lanes_conflicts`` refuses z2_min > z2_max + ext_cap, z2_min = 1, and the other listed rules; warnings;
* ``LaneOptions`` defaults == the leader == ``from_preset("leader")``; not-ported switches refused.

Part 2 (task T3, engine-dependent): ``engine_hparams(d)`` vs the sandbox recipe through
``hparams_in_force()`` for d = 6, 10. Not in this file.

REVIEW F9 autouse fixture; writes nothing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import src.hparams as H
from src.lanes import hparams as LH
from src.lanes.options import PRESETS, LaneOptions

REPO = Path(__file__).resolve().parents[1]

# DESIGN.md 2.2 (the leader) + reviewer correction R2 item 5.
LEADER = {
    "lanes_rho": 0.7, "lanes_floor_rule": "graded", "lanes_inner_margin": 0.02, "lanes_inner_margin_frac": 0.25,
    "lanes_count_frac": 0.25, "lanes_cross_cover_frac": 1.01, "lanes_cross_overlap_frac": 0.5, "lanes_ext_cap": 3,
    "lanes_incumbent_lines": True, "lanes_num_lines": 10, "lanes_refresh_relevance": True,
    "lanes_refresh_full": False,
    "lanes_z0_max": 0, "lanes_z1_max": 0, "lanes_z2_max": 0, "lanes_z0_min": 0, "lanes_z1_min": 0,
    "lanes_z2_min": 3, "lanes_z0_nc": 0, "lanes_z1_nc": 0, "lanes_z2_nc": 0,
    "lanes_catchup_zooms": "12", "lanes_catchup_eps": 0.01, "lanes_catchup_max": 2, "lanes_catchup_top_m": 0,
    "lanes_refresh": True, "lanes_refresh_margin_s": 240.0, "lanes_lead_s": 1200.0, "lanes_period0_s": 3000.0,
    "lanes_replenish_delay_s": 300.0, "lanes_max_replenish": 4, "lanes_pickup_source": "infer",
    "lanes_ingest_unattributed": False,
}
LIVE = {"lanes_refresh_margin_s", "lanes_lead_s", "lanes_period0_s", "lanes_replenish_delay_s",
        "lanes_max_replenish"}


@pytest.fixture(autouse=True)
def _restore_rng_and_simplex():
    """REVIEW F9: restore CPU / CUDA RNG states and the three random_simplex attributes."""
    from src.core import linebo as _lb
    from src.core import zombihop as _zh
    from src.utils import domain as _dom
    saved = {m: m.__dict__.get("random_simplex") for m in (_dom, _zh, _lb)}
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        for m, f in saved.items():
            setattr(m, "random_simplex", f)


# ── registry ─────────────────────────────────────────────────────────────────────────────────────

def test_names_unique_prefixed_and_disjoint():
    names = [p.name for p in LH._REGISTRY]
    assert len(names) == len(set(names))
    assert all(n.startswith("lanes_") for n in names)
    assert not (set(names) & H.KNOWN_NAMES)
    assert not any(k.startswith("lanes_") for k in H.ALL_DEFAULTS)
    assert set(names) == LH.LANES_KNOWN_NAMES == set(LH.LANES_HPARAMS)


def test_defaults_are_the_leader():
    d = LH.lanes_defaults()
    assert d == LEADER
    assert all(type(d[k]) is type(v) for k, v in LEADER.items()), {k: type(d[k]) for k in d}
    assert LH.LANES_LIVE_EDITABLE == LIVE
    assert LH.lanes_conflicts() == [] and LH.lanes_conflicts({}, LEADER) == []
    assert LH.lanes_warnings() == [] and LH.lanes_warnings(H.ALL_DEFAULTS, LEADER) == []
    assert LH.lanes_lines_per_activation() == 24
    assert all(p.tune is None for p in LH._REGISTRY)
    assert {p.targets[0][0] for p in LH._REGISTRY} == {"lanes", "controller"}
    bools = [p for p in LH._REGISTRY if isinstance(p.default, bool)]
    assert bools and all(p.kind == "str" for p in bools)


def test_registry_conventions_and_views():
    cats = LH.lanes_ui_categories()
    flat = [row for rows in cats.values() for row in rows]
    assert [r[0] for r in flat] == [p.name for p in LH._REGISTRY]
    assert all(len(r) == 5 and r[1] in ("int", "float", "str") and r[2] == LEADER[r[0]] for r in flat)
    assert set(LH.lanes_option_fields()) == {f.name for f in LaneOptions.__dataclass_fields__.values()
                                            if f.name in LH.FIELD_TO_NAME}
    ctl = LH.controller_fields()
    assert ctl == dict(refresh=True, refresh_margin_s=240.0, lead_s=1200.0, period0_s=3000.0,
                       replenish_delay_s=300.0, max_replenish=4, pickup_source="infer", ingest_unattributed=False)
    assert LH.core_stock_hp() == dict(max_zooms=3, max_iter=5, min_iters=2, n_consec=2, min_zoom_needle=2)
    assert LH.zbudget_params() == LaneOptions().zbudget_params()


def test_casts():
    c = LH.lanes_coerce
    assert c("lanes_incumbent_lines", "false") is False and c("lanes_incumbent_lines", " TRUE ") is True
    for bad in (1, 0, "yes", "1", None):
        with pytest.raises(ValueError):
            c("lanes_refresh", bad)
    assert c("lanes_catchup_zooms", "21") == "12" and c("lanes_catchup_zooms", 1) == "1"
    for bad in ("", "3", "0", "12x"):
        with pytest.raises(ValueError):
            c("lanes_catchup_zooms", bad)
    assert c("lanes_floor_rule", "Volume") == "volume"
    with pytest.raises(ValueError):
        c("lanes_pickup_source", "acks")
    with pytest.raises(ValueError):
        c("lanes_rho", 1.0)
    with pytest.raises(ValueError):
        c("lanes_inner_margin_frac", 0.5)
    with pytest.raises(ValueError):
        c("lanes_count_frac", 0.0)
    with pytest.raises(ValueError):
        c("lanes_num_lines", 1)
    with pytest.raises(ValueError):
        c("lanes_z0_max", 1)
    with pytest.raises(ValueError):
        c("lanes_ext_cap", 2.5)
    with pytest.raises(ValueError):
        c("lanes_nope", 1)
    assert c("lanes_ext_cap", 3.0) == 3 and c("lanes_lead_s", "900") == 900.0


# ── conflicts / warnings ─────────────────────────────────────────────────────────────────────────

def test_conflicts_refuse_unreachable_zoom2_floor():
    p = LH.lanes_conflicts({}, {"lanes_z2_min": 9})                    # 9 > 5 + 3
    assert p and any("floor" in s and "zoom 2" in s for s in p), p
    p = LH.lanes_conflicts({}, {"lanes_z2_max": 4, "lanes_ext_cap": 0, "lanes_z2_min": 5})
    assert any("floor" in s for s in p), p
    with pytest.raises(H.HyperparameterConflict):
        LH.check_lanes_conflicts({}, {"lanes_z2_min": 9}, context="test")


def test_conflicts_refuse_z2_min_1_and_below_core_min():
    p = LH.lanes_conflicts({}, {"lanes_z2_min": 1})
    assert p and "lanes_z2_min" in p[0], p
    p = LH.lanes_conflicts({"min_iters_per_zoom": 3, "max_iterations": 5}, {"lanes_z2_min": 2})
    assert any("below max(2, min_iters_per_zoom=3)" in s for s in p), p


def test_conflicts_other_rules():
    assert any("would need 6 results from 5 sends" in s for s in LH.lanes_conflicts({}, {"lanes_z2_min": 6}))
    assert LH.lanes_conflicts({}, {"lanes_z2_min": 6, "lanes_z2_max": 6}) == []
    assert any("lanes_catchup_zooms" in s for s in LH.lanes_conflicts({"max_zooms": 2, "min_zoom_for_needle": 1},
                                                                      {"lanes_catchup_zooms": "12"}))
    assert any("lanes_lead_s" in s for s in LH.lanes_conflicts({}, {"lanes_lead_s": 3000.0}))
    assert any("unknown" in s for s in LH.lanes_conflicts({}, {"lanes_bridge": True}))
    assert any("lanes_rho" in s for s in LH.lanes_conflicts({}, {"lanes_rho": "x"}))


def test_warnings():
    w = LH.lanes_warnings({"max_iterations": 7, "n_consecutive_converged": 3})
    assert any(s.startswith("max_iterations=7") for s in w) and any("n_consecutive_converged" in s for s in w)
    w = LH.lanes_warnings({}, {"lanes_rho": 0.6, "lanes_lead_s": 900.0})
    assert len(w) == 1 and w[0].startswith("lanes_rho=0.6")                # live timing knobs: no warning


def test_extract_lanes_hparams():
    assert LH.extract_lanes_hparams({"lanes_rho": 0.6}) == {"lanes_rho": 0.6}
    assert LH.extract_lanes_hparams({"schema": 1, "lanes_hparams": {"lanes_ext_cap": 2}}) == {"lanes_ext_cap": 2}
    with pytest.raises(ValueError):
        LH.extract_lanes_hparams({"lanes_rho": 0.6, "max_iterations": 5})
    with pytest.raises(ValueError):
        LH.extract_lanes_hparams([1])


# ── LaneOptions ──────────────────────────────────────────────────────────────────────────────────

def test_options_defaults_are_the_leader():
    o = LaneOptions()
    assert PRESETS == {"leader": {}}
    assert o == LaneOptions.from_preset("leader") == LaneOptions.from_hparams(LEADER) == LaneOptions.from_hparams()
    assert o.is_leader() and o.label() == "leader"
    assert (o.incmean, o.cross_cover_frac, o.z2_min, o.catchup_zooms, o.catchup_eps, o.catchup_max) == \
        (True, 1.01, 3, "12", 0.01, 2)
    assert (o.bridge, o.confirm_revise, o.noface, o.gp_box, o.gate, o.input_noise) == \
        (False, False, False, False, False, None)
    assert o.snapshots is True
    assert o.to_hparams() == {k: v for k, v in LEADER.items() if k in LH.LANES_OPTION_FIELDS}
    d = o.to_dict()
    assert d["z2_min"] == 3 and "bridge_predict" not in d and "gate_zooms" not in d
    assert LaneOptions(**d) == o
    assert hash(o) == hash(LaneOptions())


def test_options_overrides_and_validation():
    o = LaneOptions.from_preset("leader", {"lanes_rho": "0.6", "lanes_incumbent_lines": "false"})
    assert o.rho == 0.6 and o.incmean is False
    assert o.label() == "leader+incmean=False+rho=0.6"
    assert LaneOptions.from_hparams({"lanes_lead_s": 900.0}) == LaneOptions()      # controller knob ignored
    assert LaneOptions.from_hparams(snapshots=False).snapshots is False
    assert LaneOptions(ext_cap=2.0).ext_cap == 2 and type(LaneOptions(ext_cap=2.0).ext_cap) is int
    assert LaneOptions().replace(z2_min=4).z2_min == 4
    for f in ("bridge", "confirm_revise", "noface", "gp_box", "gate"):
        with pytest.raises(ValueError, match="not ported"):
            LaneOptions(**{f: True})
    with pytest.raises(ValueError, match="not ported"):
        LaneOptions(input_noise=0.06)
    with pytest.raises(ValueError):
        LaneOptions(rho=1.5)
    with pytest.raises(ValueError):
        LaneOptions(floor_rule="other")
    with pytest.raises(ValueError):
        LaneOptions.from_hparams({"lanes_nope": 1})
    with pytest.raises(ValueError):
        LaneOptions.from_hparams(rho=0.6)                  # registry fields go through lanes_* names
    with pytest.raises(ValueError):
        LaneOptions.from_preset("best")
    with pytest.raises(Exception):
        LaneOptions().rho = 0.5                            # frozen


# ── import leaves src.hparams (and everything else) untouched ────────────────────────────────────

_PROBE = r"""
import json, sys
import src.hparams as H
from src.utils import domain as d
from src.core import zombihop as zh, linebo as lb
from src.utils.datahandler import DataHandler
from src.utils.gp_simplex import GPSimplex
from src.core.zombihop import ZoMBIHop
from src.core.linebo import LineBO

def snap():
    return dict(defaults=sorted((k, repr(v)) for k, v in H.ALL_DEFAULTS.items()),
                known=sorted(H.KNOWN_NAMES),
                ui={k: [list(map(repr, t)) for t in v] for k, v in H.ui_categories().items()},
                live=sorted(H.LIVE_EDITABLE_NAMES), owned={o: H.owned_by(o) for o in ("dh", "gp", "zombi")},
                rs=[id(m.random_simplex) for m in (d, zh, lb)],
                cls=[sorted(vars(c)) for c in (DataHandler, GPSimplex, ZoMBIHop, LineBO)])

before = snap()
import src.lanes
from src.lanes import hparams, options, budget, events, labels, geometry, lines, rng, instrument
after = snap()
print(json.dumps(dict(same=before == after, diff=[k for k in before if before[k] != after[k]],
                      engine_loaded="src.lanes.engine" in sys.modules,
                      n_known=len(before["known"]))))
"""


def test_import_src_lanes_changes_nothing_in_a_fresh_interpreter():
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-B", "-c", _PROBE], cwd=str(REPO), env=env, capture_output=True,
                       text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["same"], out["diff"]
    assert out["engine_loaded"] is False                   # build_lane_engine is lazy
    assert out["n_known"] == len(H.KNOWN_NAMES)
