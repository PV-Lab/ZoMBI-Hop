import numpy as np

from benchmarks.zombihop_benchmark.optimizers.random_simplex import RandomSimplexOptimizer
from benchmarks.zombihop_benchmark.spaces import validate_simplex
from benchmarks.zombihop_benchmark.types import ObjectiveInfo


def test_random_optimizer_reproducible():
    info = ObjectiveInfo(name="obj", n_components=3)
    opt_a = RandomSimplexOptimizer()
    opt_b = RandomSimplexOptimizer()
    opt_a.initialize(np.empty((0, 3)), np.empty(0), info, seed=7)
    opt_b.initialize(np.empty((0, 3)), np.empty(0), info, seed=7)
    Xa = np.vstack([opt_a.suggest(2), opt_a.suggest(2)])
    Xb = np.vstack([opt_b.suggest(2), opt_b.suggest(2)])
    assert np.allclose(Xa, Xb)
    assert validate_simplex(Xa)

