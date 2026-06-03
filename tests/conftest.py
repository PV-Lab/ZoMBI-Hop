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

