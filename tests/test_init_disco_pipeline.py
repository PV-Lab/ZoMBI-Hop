"""Initialization against DiSCO's own result path, end to end except the cable.

Every slide result goes through the real code on both machines:
  DiSCO  write_objective -> objective.db -> objective_sender (drops NaN rows, never sends
         an empty table)  -> serial bytes
  ZoMBI  -> objective_receiver -> objective.db + handshake -> run_axial_initialization
The serial port is a fake that hands DiSCO's written bytes to ZoMBI's receiver queue.
DiSCO's transporters/printer are emulated as in test_axial_init_hardware.FakeDiSCO.

Skipped when the DiSCO code is not on this machine.
"""

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import scripts.run_zombi_main as rzm
from scripts import communication as zcomm
from src.utils.axial_init import AxialTrace
from src.utils.init_queue import _segment_distance
from test_axial_init_hardware import FakeDiSCO, _bounds, _state, _trace_of_rows, hw_env  # noqa: F401

DISCO_DIR = Path(r"C:/Users/AMLS at MIT/MIT Dropbox/Buonassisi-Group/Projects - Active/"
                 r"DiSCO/13_Codes/DiSCO_Control/Full_System_Integration")

pytestmark = pytest.mark.skipif(not (DISCO_DIR / "communication.py").exists(),
                                reason="DiSCO control code not on this machine")


@pytest.fixture(scope="module")
def disco():
    """DiSCO's communication module, loaded without leaving its folder on sys.path."""
    before = set(sys.modules)
    sys.path.insert(0, str(DISCO_DIR))
    try:
        spec = importlib.util.spec_from_file_location("disco_communication", DISCO_DIR / "communication.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(DISCO_DIR))
        for name in set(sys.modules) - before:          # its top-level helpers
            if name in ("communication", "stability", "initialize_databases"):
                del sys.modules[name]
    return mod


class FakeSerial:
    """What DiSCO's objective_sender writes goes, line by line, to ZoMBI's receiver queue."""
    baudrate = 115200

    def __init__(self):
        self.timeout = 1.0
        self.write_timeout = 1.0
        self.out_waiting = 0
        self.in_waiting = 0
        self.closed = False
        self.lines = []

    def write(self, data):
        if not self.closed:
            for line in bytes(data).split(b"\n"):
                if line.strip():
                    self.lines.append(line)
                    zcomm._read_queue.put(line)
        return len(data)

    def reset_input_buffer(self):
        pass

    def reset_output_buffer(self):
        pass


class DiscoPathDiSCO(FakeDiSCO):
    """Emulated transporters; each result is written with DiSCO's write_objective, and
    a slide with no usable row is never written at all (DiSCO's objective() skips it)."""

    def __init__(self, dims, disco_mod, obj_db, fail_rows=None, nan_comp_rows=None, **kw):
        super().__init__(dims, **kw)
        self.disco = disco_mod
        self.obj_db = str(obj_db)
        self.fail_rows = fail_rows or (lambda k: np.zeros(24, bool))   # rows whose y is NaN
        self.nan_comp_rows = nan_comp_rows or (lambda k: np.zeros(24, bool))
        self.reported = {}                                               # print k -> usable rows

    def _emit(self, k, rows):
        meas = np.clip(rows + self.rng.normal(0, self.noise, rows.shape) * (rows > 0), 0, None)
        meas /= meas.sum(axis=1, keepdims=True)
        y = 0.5 + 0.5 * np.exp(-((meas[:, self.dims] - 1 / len(self.dims)) ** 2).sum(1) / 0.05)
        bad_y, bad_c = self.fail_rows(k), self.nan_comp_rows(k)
        usable = ~(bad_y | bad_c)
        self.reported[k] = int(usable.sum())
        if not usable.any():
            self.results.append((time.time(), k))          # measured, nothing sent
            return
        vals = [None if bad_y[j] else float(y[j]) for j in range(len(y))]
        comps = [[None] * 10 if bad_c[j] else meas[j].tolist() for j in range(len(meas))]
        self.disco.write_objective(self.obj_db, np.asarray(vals, dtype=object),
                                   compositions=np.asarray(comps, dtype=object))
        self.results.append((time.time(), k))


@pytest.fixture()
def pipeline(hw_env, disco):
    obj_db = hw_env / "disco_objective.db"
    results_db = hw_env / "disco_results.db"
    disco.write_objective(str(obj_db), np.asarray([], dtype=object))   # as DiSCO starts: empty table
    ser = FakeSerial()
    zcomm._shutdown_flag.clear()
    zcomm._last_objective_ts = zcomm._last_objective_vals = zcomm._last_objective_comps = None
    while not zcomm._read_queue.empty():
        zcomm._read_queue.get_nowait()
    receiver = threading.Thread(target=zcomm.objective_receiver,
                                args=(10, "./sql/objective.db", "./sql/objective_memory.db"),
                                daemon=True)
    receiver.start()
    sender = threading.Thread(target=disco.objective_sender,
                              kwargs=dict(ser=ser, hz=5, db_path=str(obj_db),
                                          results_db_path=str(results_db)), daemon=True)
    sender.start()
    yield {"obj_db": obj_db, "ser": ser}
    ser.closed = True                       # the sender thread lingers but writes nowhere
    zcomm._shutdown_flag.set()
    receiver.join(timeout=5)


def test_partial_failed_and_clean_slides_through_disco_code(pipeline, disco, monkeypatch, torch):
    dims, hi = [0, 2, 3, 4, 8, 9], [1, 1, 1, 0.3, 1, 1]
    monkeypatch.setattr(rzm, "OPTIMIZING_DIMS", dims)

    def fail_rows(k):
        m = np.zeros(24, bool)
        if k == 0:
            m[:14] = True                    # face end of the gradient failed: 10 rows left
        elif k == 1:
            m[10:] = True                    # pure end failed: 10 rows left
        elif k == 2:
            m[:] = True                      # whole slide failed: DiSCO sends nothing
        elif k == 4:
            m[np.random.default_rng(4).random(24) < 0.3] = True   # scattered
        elif k == 5:
            m[2:] = True                     # only 2 rows survived
        return m

    def nan_comp_rows(k):
        m = np.zeros(24, bool)
        if k == 3:
            m[[3, 9, 15, 20, 22]] = True     # composition readback failed on 5 spots
        return m

    dd = DiscoPathDiSCO(dims, disco, pipeline["obj_db"], fail_rows=fail_rows,
                        nan_comp_rows=nan_comp_rows, print_s=0.4, lag_s=0.8, noise=0.01)
    dd.start()
    run_dir = Path("run_pipe")
    run_dir.mkdir()
    rzm.set_composition_log_dir(run_dir)
    t0 = time.time()
    try:
        Xa, Xe, Y = rzm.run_axial_initialization(run_dir, "pipe", _bounds(torch, 6, hi), "cpu")
    finally:
        dd.stop()
    wall = time.time() - t0
    st = _state(run_dir)
    q = st["queue"]
    traces = [AxialTrace.from_dict(t) for t in q["traces"]]

    assert st["complete"] and wall < 180                       # never held
    assert all(s == "done" for s in q["status"])
    assert np.isfinite(Xa.numpy()).all() and np.isfinite(Y.numpy()).all()
    # every init point lies on an init trace: nothing foreign got in
    X = Xa.numpy()
    assert max(min(_segment_distance(X[j:j + 1], t.start, t.end)[0] for t in traces)
               for j in range(len(X))) < 0.1

    by_trace = {k: _trace_of_rows(st, rows, dims) for k, (_, _, _, rows) in enumerate(dd.prints)}
    # partial results were kept as the trace's data, with exactly the rows DiSCO sent,
    # and the trace was NOT printed again
    for k in (0, 1, 3, 4):
        i = by_trace[k]
        got = sorted(len(b["y"]) for b in q["data"][i] if len(b["y"]))
        assert dd.reported[k] in got, (k, dd.reported[k], got)
        assert q["sends"][i] == 1, (k, q["sends"][i])
    # the slide that failed completely was re-sent and measured on its second print
    i2 = by_trace[2]
    assert q["sends"][i2] >= 2 and any(len(b["y"]) for b in q["data"][i2])
    # the composition log recorded each matched result with its usable rows only
    log = [json.loads(l) for l in (run_dir / rzm.COMPOSITION_LOG_NAME).read_text().splitlines()]
    sizes = sorted(len(r["measured"]) for rec in log for r in rec["rails"] if r["measured"])
    assert dd.reported[0] in sizes and dd.reported[3] in sizes
    # DiSCO's sender never put an empty or all-NaN table on the wire
    for line in pipeline["ser"].lines:
        pkt = json.loads(line)
        if pkt.get("type") == "objective":
            assert pkt["values"] and all(v is not None for v in pkt["values"])


def test_regular_loop_reads_a_partial_disco_result(pipeline, disco, monkeypatch, torch):
    """After initialization the optimisation loop takes the same path: a result with
    failed rows reaches objective() as just its usable rows."""
    dims = [0, 8, 9]
    monkeypatch.setattr(rzm, "OPTIMIZING_DIMS", dims)

    def fail_rows(k):
        m = np.zeros(24, bool)
        m[:9] = True
        return m

    dd = DiscoPathDiSCO(dims, disco, pipeline["obj_db"], fail_rows=fail_rows, lag_s=0.5)
    dd.start()
    try:
        ep = torch.tensor([[[1, 0, 0], [0, .5, .5]], [[0, 1, 0], [.5, 0, .5]]], dtype=torch.float64)
        x, y = rzm.objective(ep, ready_for_objectives=True)
    finally:
        dd.stop()
    assert x.shape == (15, 3) and y.shape == (15,) and np.isfinite(y.cpu().numpy()).all()
