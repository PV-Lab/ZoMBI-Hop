import json

import numpy as np

from benchmarks.zombihop_benchmark.io import write_json
from benchmarks.zombihop_benchmark.types import BatchObservation


def test_write_json_serializes_batch_observation(tmp_path):
    obs = BatchObservation(
        X_expected=np.array([[0.2, 0.3, 0.5]]),
        X_actual=np.array([[0.2, 0.3, 0.5]]),
        y=np.array([0.7]),
        metadata={"source": "unit"},
    )
    path = tmp_path / "obs.json"

    write_json(path, {"observation": obs})

    data = json.loads(path.read_text())
    assert data["observation"]["X_expected"] == [[0.2, 0.3, 0.5]]
    assert data["observation"]["X_actual"] == [[0.2, 0.3, 0.5]]
    assert data["observation"]["y"] == [0.7]
    assert data["observation"]["metadata"] == {"source": "unit"}


def test_write_json_serializes_nested_torch_tensor(tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    path = tmp_path / "tensor.json"

    write_json(path, {"nested": {"tensor": torch.tensor([1.0, 2.0])}})

    data = json.loads(path.read_text())
    assert data["nested"]["tensor"] == [1.0, 2.0]
