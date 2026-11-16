import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _ensure_project_on_path():
    """
    Ensure `import src...` and `import scripts...` resolve when running pytest
    from the repository root or from within `zombi_replace/`.
    """
    here = Path(__file__).resolve()
    ellipsoid_pkg = here.parents[1]          # …/zombihop_ellipsoid
    zombi_replace_root = ellipsoid_pkg.parent  # …/zombi_replace (contains sibling src + scripts/)
    # Own package first — tests must exercise ``zombihop_ellipsoid/src``, not sibling ``zombi_replace/src``.
    if str(ellipsoid_pkg) not in sys.path:
        sys.path.insert(0, str(ellipsoid_pkg))
    # ``scripts.*`` imports for handshake tests resolve from the parent package root.
    if str(zombi_replace_root) not in sys.path:
        sys.path.append(str(zombi_replace_root))


@pytest.fixture(scope="session", autouse=True)
def _require_cuda(torch):
    """
    This project is intended to run on CUDA. Fail fast in tests if CUDA
    is not available, rather than silently running on CPU.
    """
    if not torch.cuda.is_available():
        pytest.fail("CUDA is required for this test suite (torch.cuda.is_available() is False).")


@pytest.fixture(scope="session")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture()
def tmp_run_dir(tmp_path: Path) -> Path:
    # DataHandler expects a directory it can create inside.
    d = tmp_path / "run"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch):
    # Make it easy to detect tests in logs / downstream behavior.
    monkeypatch.setenv("ZOMBIHOP_TESTING", "1")


# The needles.db DiSCO's visualization.py reads during a hardware campaign.
_LIVE_NEEDLES_DB = Path(__file__).resolve().parents[1] / "sql" / "needles.db"


def _same_file(a, b) -> bool:
    """Same file by any spelling (case, 8.3 short name, junction): os.path.samefile when both exist, else the
    normalised real paths. An error counts as the same file (fail closed: the write is refused)."""
    try:
        a, b = os.fspath(a), os.fspath(b)
        if os.path.exists(a) and os.path.exists(b):
            return os.path.samefile(a, b)
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except Exception:  # noqa: BLE001
        return True


def _needles_db_guard(real, ndb):
    """``real`` (write_needles / reset_needles_db) that raises instead of touching ``guarded.live``."""
    import inspect
    sig = inspect.signature(real)

    def guarded(*a, **k):
        try:
            path = sig.bind(*a, **k).arguments.get("path")
        except TypeError:
            path = None
        target = path if path is not None else ndb.DEFAULT_NEEDLES_DB_PATH
        if _same_file(target, guarded.live):
            raise RuntimeError(f"tests/conftest.py: refusing to write the live {target} from a test")
        return real(*a, **k)

    guarded.live = _LIVE_NEEDLES_DB
    guarded.__wrapped__ = real
    guarded.__name__ = real.__name__
    return guarded


def _install_needles_db_isolation(mp: pytest.MonkeyPatch, path, when: str) -> bool:
    """src.utils.needles_db's default path -> ``path``, and write_needles / reset_needles_db refuse the live file
    whatever path they are given (undone by ``mp``). False, with a warning (never silently), when
    src.utils.needles_db cannot be imported."""
    try:
        import src.utils.needles_db as ndb
    except Exception as e:  # noqa: BLE001
        import warnings
        warnings.warn(f"tests/conftest.py: needles.db isolation ({when}) NOT installed: src.utils.needles_db "
                      f"cannot be imported ({e!r}); only ZOMBI_NEEDLES_DB points at a temp file")
        return False
    mp.setattr(ndb, "DEFAULT_NEEDLES_DB_PATH", Path(path))
    for name in ("write_needles", "reset_needles_db"):
        real = getattr(ndb, name)
        real = getattr(real, "__wrapped__", real)
        mp.setattr(ndb, name, _needles_db_guard(real, ndb))
    return True


def pytest_configure(config):
    """Session-wide needles.db isolation, in place before collection and before every fixture (session- and
    module-scoped ones too, which the per-test fixture below does not cover): ZOMBIHOP_TESTING=1 and
    ZOMBI_NEEDLES_DB -> a session temp file. src.utils.needles_db itself is patched by the session fixture."""
    import tempfile
    d = tempfile.mkdtemp(prefix="zombihop_needles_db_")
    mp = pytest.MonkeyPatch()
    mp.setenv("ZOMBIHOP_TESTING", "1")
    mp.setenv("ZOMBI_NEEDLES_DB", str(Path(d) / "needles.db"))
    config._zombihop_needles_db = (mp, d)


def pytest_unconfigure(config):
    st = getattr(config, "_zombihop_needles_db", None)
    if st is None:
        return
    st[0].undo()
    import shutil
    shutil.rmtree(st[1], ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _isolate_needles_db_session(request, _ensure_project_on_path):
    """Session-scoped (autouse, so set up before any module- or session-scoped fixture of a test module):
    src.utils.needles_db's default path -> the session temp file, plus the live-file-refusing wrappers."""
    st = getattr(request.config, "_zombihop_needles_db", None)
    if st is None:                                      # pytest_configure did not run for this conftest
        import tempfile
        st = (None, tempfile.mkdtemp(prefix="zombihop_needles_db_"))
    mp = pytest.MonkeyPatch()
    mp.setenv("ZOMBIHOP_TESTING", "1")
    mp.setenv("ZOMBI_NEEDLES_DB", str(Path(st[1]) / "needles.db"))
    _install_needles_db_isolation(mp, Path(st[1]) / "needles.db", "session")
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _isolate_needles_db(tmp_path_factory, monkeypatch: pytest.MonkeyPatch):
    """No test may write the DiSCO-facing <repo>/sql/needles.db (2026-11-15: two lanes driver tests emptied the
    running campaign's needle table). The default path and ZOMBI_NEEDLES_DB point at a per-test temp file, and
    write_needles / reset_needles_db refuse the live file whatever path they are given."""
    p = tmp_path_factory.mktemp("needles_db") / "needles.db"
    monkeypatch.setenv("ZOMBI_NEEDLES_DB", str(p))
    _install_needles_db_isolation(monkeypatch, p, "test")

