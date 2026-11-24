"""init_state.json must survive another program holding it open (hardware run 56e6).

Run directories live in Dropbox. Its uploader opens a freshly written file, and while
it holds it Windows refuses to rename over it (PermissionError, WinError 5) - which
stopped run_56e6 two seconds into initialization. These tests hold the file open the
same way (a plain open() shares read/write but not delete, like the uploader).
"""

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import scripts.run_zombi_main as rzm
from test_axial_init_hardware import FakeDiSCO, _bounds, hw_env  # noqa: F401


def _state(n_writes, n_results=0):
    return {"version": 2, "optimizing_dims": [0, 8, 9],
            "queue": {"writes": [{"no": k} for k in range(n_writes)],
                      "result_times": [float(k) for k in range(n_results)]}}


def test_save_while_another_program_holds_the_file(tmp_path):
    path = tmp_path / rzm.INIT_STATE_NAME
    path.write_text(json.dumps(_state(0)))
    held = open(path, "r")                 # rename over it is now refused by Windows
    try:
        with pytest.raises(PermissionError):
            probe = tmp_path / "probe.tmp"
            probe.write_text("x")
            os.replace(probe, path)        # the failure run_56e6 hit
        assert rzm._save_init_state(tmp_path, _state(1)) is True
    finally:
        held.close()
    assert len(json.loads(path.read_text())["queue"]["writes"]) == 1
    assert not list(tmp_path.glob(rzm.INIT_STATE_NAME + "*.tmp"))


def test_save_retries_until_a_brief_hold_is_released(tmp_path):
    path = tmp_path / rzm.INIT_STATE_NAME
    path.write_text(json.dumps(_state(0)))
    held = open(path, "r")
    threading.Timer(0.6, held.close).start()
    t0 = time.time()
    assert rzm._save_init_state(tmp_path, _state(2)) is True
    assert time.time() - t0 < 5.5
    assert len(json.loads(path.read_text())["queue"]["writes"]) == 2
    assert not list(tmp_path.glob(rzm.INIT_STATE_NAME + "*.tmp"))


def test_save_never_raises_and_keeps_the_newest_state(tmp_path, monkeypatch):
    path = tmp_path / rzm.INIT_STATE_NAME
    path.write_text(json.dumps(_state(0)))
    monkeypatch.setattr(rzm, "INIT_STATE_REPLACE_TRIES", 2)

    def refuse(*a, **k):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(rzm.os, "replace", refuse)
    real_write = Path.write_text

    def write_text(self, *a, **k):
        if self == path:
            raise PermissionError(13, "Access is denied")
        return real_write(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", write_text)
    assert rzm._save_init_state(tmp_path, _state(3, 1)) is False      # warned, no raise
    monkeypatch.undo()
    st = rzm._load_init_state(tmp_path)
    assert len(st["queue"]["writes"]) == 3                             # from the side file


def test_load_prefers_the_copy_with_more_progress(tmp_path):
    """run_56e6 on disk: init_state.json = the plan, init_state.json.tmp = plan + the
    first pair it sent before the save failed."""
    (tmp_path / rzm.INIT_STATE_NAME).write_text(json.dumps(_state(0)))
    (tmp_path / (rzm.INIT_STATE_NAME + ".tmp")).write_text(json.dumps(_state(1)))
    assert len(rzm._load_init_state(tmp_path)["queue"]["writes"]) == 1
    # equal progress: the main file wins; unreadable side files are skipped
    (tmp_path / (rzm.INIT_STATE_NAME + ".tmp")).write_text(json.dumps(_state(0)))
    (tmp_path / (rzm.INIT_STATE_NAME + ".99.tmp")).write_text("{truncated")
    assert rzm._load_init_state(tmp_path) == _state(0)


def test_initialization_runs_while_the_file_keeps_being_grabbed(hw_env, monkeypatch, torch):
    """A full init with something opening init_state.json for a moment after every
    write, as the uploader does."""
    dims = [0, 2, 8, 9]
    monkeypatch.setattr(rzm, "OPTIMIZING_DIMS", dims)
    run_dir = hw_env / "run_grab"
    run_dir.mkdir()
    path = run_dir / rzm.INIT_STATE_NAME
    stop = threading.Event()
    grabs = [0]

    def grabber():
        last = None
        while not stop.is_set():
            try:
                m = path.stat().st_mtime_ns
                if m != last:
                    last = m
                    with open(path, "r"):
                        grabs[0] += 1
                        time.sleep(1.2)        # longer than the first retries
            except OSError:
                pass
            time.sleep(0.01)

    g = threading.Thread(target=grabber, daemon=True)
    g.start()
    disco = FakeDiSCO(dims)
    disco.start()
    try:
        Xa, _, Y = rzm.run_axial_initialization(run_dir, "grab", _bounds(torch, 4), "cpu")
    finally:
        disco.stop()
        stop.set()
        g.join(timeout=5)
    st = json.loads(path.read_text())
    assert grabs[0] >= 3 and st["complete"]
    assert all(s == "done" for s in st["queue"]["status"])
    assert np.isfinite(Y.numpy()).all() and Xa.shape[0] >= 4 * 20
    assert not list(run_dir.glob(rzm.INIT_STATE_NAME + "*.tmp"))
