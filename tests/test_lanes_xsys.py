"""GAP 4 (critic G6): ZoMBI's REAL serial layer + lanes driver + engine against the REAL DiSCO chain.

Wraps benchmarks/zoom_fix/lanes/disco_patch/archerfish_path/xsys/run_xsys.py. Each scenario runs in its own
process (virtual clock, the archerfish harness's file/sqlite/serial guards), with the DiSCO code from
disco_patch/staging and the harness's fake Archerfish / Prism / arm; ZoMBI runs its production serial session
(scripts/communication.py) and scripts/lanes_hw.run_lanes_hardware with the real engine (6 dims, small hparams).
See xsys/scenarios_x.py for X1..X5 (+ X4b) and xsys/xcheck.py for the assertions: every ingested result's lane /
line / wire id = the recipe DiSCO printed for that slide, its rows = DiSCO's comps / objective values, every
DiSCO table journaled, nothing ingested twice, plus each scenario's own checks.

The six scenarios run in parallel once per session (about 1.5-2 min wall, one CUDA context each when CUDA is
available); set ZOMBI_XSYS=0 to skip, ZOMBI_XSYS_JOBS to change the parallelism.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
XSYS = REPO / "benchmarks" / "zoom_fix" / "lanes" / "disco_patch" / "archerfish_path" / "xsys"
STAGING = REPO / "benchmarks" / "zoom_fix" / "lanes" / "disco_patch" / "staging"

pytestmark = pytest.mark.skipif(
    os.environ.get("ZOMBI_XSYS", "1") == "0" or not (XSYS / "run_xsys.py").exists() or not STAGING.exists(),
    reason="cross-system DiSCO<->ZoMBI run disabled (ZOMBI_XSYS=0) or the disco_patch folders are missing")

SCENARIOS = ["X1", "X2", "X3", "X4", "X4b", "X5"]


@pytest.fixture(scope="module")
def xsys_reports(tmp_path_factory):
    sys.path.insert(0, str(XSYS))
    try:
        import run_xsys
    finally:
        sys.path.remove(str(XSYS))
    root = tmp_path_factory.mktemp("xsys")
    jobs = int(os.environ.get("ZOMBI_XSYS_JOBS", len(SCENARIOS)))
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {n: ex.submit(run_xsys.run_one, n, code=str(STAGING), tmp_root=str(root)) for n in SCENARIOS}
        return {n: f.result() for n, f in futs.items()}


@pytest.mark.parametrize("name", SCENARIOS)
def test_xsys_scenario(name, xsys_reports):
    rep = xsys_reports[name]
    bad = [c for c in rep.get("checks") or [] if not c["ok"]]
    msg = json.dumps({"error": (rep.get("error") or "")[-2500:],
                      "failed": [(c["name"], c.get("detail")) for c in bad[:8]],
                      "summary": rep.get("summary")}, default=str, indent=1)
    assert rep.get("checks"), msg
    assert rep.get("ok"), msg
