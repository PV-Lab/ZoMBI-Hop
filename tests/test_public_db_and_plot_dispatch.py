"""Public-dataset loading and the plot_run diagram dispatch.

Two things are pinned here:

  * ``benchmarks/public_db`` reads the cached Olympus datasets with the shape,
    constraint and goal they actually have — in particular that the ``photo_*``
    sets are 4-component *simplices* (so they draw as a tetrahedron, not a
    ternary) and that ``hplc`` and ``crossed_barrel`` are non-simplex boxes.
    ``crossed_barrel`` shares ``d=4`` with the ``photo_*`` pair and lands on a
    different diagram, which is the case that would break first if the
    constraint were ever inferred from dimensionality.
  * ``plot_run`` routes a source to a diagram on the pair (constraint,
    dimensionality), and each of those diagrams actually builds.

These run offline against the cached files and skip if they were never fetched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "visualization"))

pub = pytest.importorskip("benchmarks.public_db")
plot_run = pytest.importorskip("plot_run")


@pytest.fixture(scope="session", autouse=True)
def _require_cuda():
    """Override ``conftest._require_cuda`` — this module is CPU-only by nature.

    That gate exists so the *optimizer* tests fail loudly instead of silently
    running on CPU. Nothing here touches torch: these are dataset parsing and
    matplotlib/plotly figure-construction checks, and they are worth running on
    any machine. Narrowly overridden here rather than relaxed in ``conftest.py``,
    so every other test keeps the gate.
    """
    return None


def _require(name: str):
    if name not in pub.available():
        pytest.skip(f"{name} not cached; run olympus.py --fetch all")
    return pub.load(name)


# -- the datasets themselves --------------------------------------------------

@pytest.mark.parametrize(
    "name,d,simplex,goal",
    [("photo_pce10", 4, True, "minimize"),
     ("photo_wf3", 4, True, "minimize"),
     ("hplc", 6, False, "maximize"),
     ("crossed_barrel", 4, False, "maximize")],
)
def test_shape_constraint_and_goal(name, d, simplex, goal):
    ds = _require(name)
    assert ds.d == d
    assert ds.simplex is simplex
    assert ds.goal == goal
    assert ds.X.shape[0] == ds.Y.shape[0] == ds.n
    assert ds.bounds.shape == (d, 2)
    assert len(ds.labels) == d


def test_photo_sets_are_quaternary_simplices_not_ternary():
    """The 'obvious' reading — 3 components on a triangle — is wrong for these.

    Rows sum to 1 across all *four* columns and the centred design matrix has
    rank 3, so the data is a 3-simplex embedded in 4 columns: a tetrahedron.
    """
    ds = _require("photo_pce10")
    assert np.allclose(ds.X.sum(axis=1), 1.0, atol=1e-9)
    rank = np.linalg.matrix_rank(ds.X - ds.X.mean(axis=0), tol=1e-9)
    assert rank == 3
    # Every column carries real signal, so none of the four can be dropped to
    # make this a ternary.
    assert (ds.X > 0).any(axis=0).all()


def test_photo_pair_shares_one_design_grid():
    a, b = _require("photo_pce10"), _require("photo_wf3")
    assert np.allclose(a.X, b.X)
    assert not np.allclose(a.Y, b.Y)


def test_unit_X_makes_hplc_columns_commensurate():
    """Raw hplc units are dominated by one column; unit_X removes that.

    This is the property the CoNet depends on: row-normalised raw units make
    ``push_speed`` ~94% of every row, so an embedding built on them describes
    that column alone.
    """
    ds = _require("hplc")

    def rownorm(X):
        s = X.sum(axis=1, keepdims=True)
        return X / np.where(s == 0, 1.0, s)

    raw_share = rownorm(ds.X).mean(axis=0)
    unit_share = rownorm(ds.unit_X).mean(axis=0)
    assert raw_share.max() > 0.9                      # one column swamps the rest
    assert unit_share.max() < 2.0 / ds.d              # none dominates after scaling
    assert ds.unit_X.min() >= 0.0 and ds.unit_X.max() <= 1.0


def test_crossed_barrel_is_a_four_dim_box_not_a_simplex():
    """Same d as the photo_* pair, no sum-to-one — so a different diagram.

    Pinned because ``d=4`` alone is exactly the ambiguous case: it means
    tetrahedron for a composition and scatter-plot matrix for a box, and nothing
    but the constraint separates them.
    """
    ds = _require("crossed_barrel")
    assert ds.simplex is False
    sums = ds.X.sum(axis=1)
    assert not np.allclose(sums, 1.0)
    # Full-rank once centred: four genuinely independent geometry parameters,
    # unlike the photo_* design matrices, which are rank 3 in 4 columns.
    assert np.linalg.matrix_rank(ds.X - ds.X.mean(axis=0), tol=1e-9) == 4
    # The pretty labels replace the paper's bare symbols on the SPLOM axes.
    assert ds.param_names == ("n", "theta", "r", "t")
    assert ds.labels != ds.param_names and len(ds.labels) == 4


def test_crossed_barrel_is_a_lattice_subset_without_replicates():
    """Declared ``continuous``, but only a handful of levels per column.

    Worth pinning: a surrogate fitted over this box is interpolating a coarse
    grid rather than a continuum, and the rows are a partial factorial — about
    half of the 4x9x11x3 lattice — with every row distinct.
    """
    ds = _require("crossed_barrel")
    levels = [len(np.unique(ds.X[:, i])) for i in range(ds.d)]
    assert levels == [4, 9, 11, 3]
    full = int(np.prod(levels))
    assert ds.n < full                                  # a subset, not the full grid
    assert len(np.unique(ds.X, axis=0)) == ds.n         # and no replicates


# -- the plot_run bridge ------------------------------------------------------

@pytest.mark.parametrize(
    "name,expect",
    [("photo_pce10", "tetrahedron"), ("photo_wf3", "tetrahedron"),
     ("hplc", "CoNet"), ("crossed_barrel", "scatter-plot matrix")],
)
def test_diagram_dispatch(name, expect):
    _require(name)
    ds = plot_run.load_public_dataset(name)
    assert expect in plot_run._diagram_name(ds)


def test_embed_X_is_unit_scaled_only_for_non_simplex():
    _require("hplc")
    _require("photo_pce10")
    box = plot_run.load_public_dataset("hplc")
    simp = plot_run.load_public_dataset("photo_pce10")
    assert np.allclose(box.embed_X, box.unit_X)
    assert not np.allclose(box.embed_X, box.X)
    assert np.allclose(simp.embed_X, simp.X)


def _synthetic(d: int, simplex: bool, n: int = 120):
    r = np.random.default_rng(0)
    hi = np.array([1.0, 10.0, 100.0, 0.05])[:d]
    X = r.random((n, d))
    if simplex:
        X = X / X.sum(axis=1, keepdims=True)
        bounds = None
    else:
        X = X * hi
        bounds = np.column_stack([np.zeros(d), hi])
    return plot_run.Dataset(
        X=X, Y=X.sum(axis=1), labels=tuple(f"p{i}" for i in range(d)),
        title="synthetic", lines=np.arange(n) // 24, value_name="obj",
        simplex=simplex, bounds=bounds, goal="maximize",
    )


@pytest.mark.parametrize(
    "d,simplex,trace",
    [(3, True, "Scatterternary"), (4, True, "Scatter3d"),
     (2, False, "Heatmap"), (3, False, "Scatter3d"), (4, False, "Splom")],
)
def test_every_diagram_builds(d, simplex, trace):
    ds = _synthetic(d, simplex)
    fig = plot_run.build_figure(
        ds.X, ds.Y, ds.labels, simplex=ds.simplex, bounds=ds.axis_bounds,
        grid_n=12, n_estimators=10, title="t", value_name="obj",
        background="rf", show_points=True, gp_length_scale=0.3, scale=1.0,
        plot_size=0.8, color_limits=None, highlight=None,
    )
    assert any(type(t).__name__ == trace for t in fig.data)


def test_non_simplex_requires_bounds():
    ds = _synthetic(2, simplex=False)
    with pytest.raises(ValueError, match="bounds"):
        plot_run.build_figure(
            ds.X, ds.Y, ds.labels, simplex=False, bounds=None,
            grid_n=12, n_estimators=10, title="t", value_name="obj",
            background="none", show_points=True, gp_length_scale=0.3,
            scale=1.0, plot_size=0.8, color_limits=None, highlight=None,
        )


def test_box2d_background_is_oriented_x_then_y():
    """z[row][col] must index [y][x]; a transpose slip here is invisible by eye.

    Y depends only on column 0, so the fitted field must vary along x and be
    flat along y.
    """
    r = np.random.default_rng(0)
    hi = np.array([1.0, 100.0])
    X = r.random((400, 2)) * hi
    fig = plot_run.build_box2d_figure(
        X, X[:, 0] / hi[0], ("p0", "p1"),
        bounds=np.column_stack([np.zeros(2), hi]),
        grid_n=24, n_estimators=30, title="t", value_name="obj",
        background="rf", show_points=False, gp_length_scale=0.3, scale=1.0,
        plot_size=0.8, color_limits=None, highlight=None,
    )
    z = np.asarray(fig.data[0].z)
    assert z.std(axis=1).mean() > 10 * z.std(axis=0).mean()
    assert z[:, 0].mean() < z[:, -1].mean()


def test_crossed_barrel_builds_a_real_splom():
    """The SPLOM branch on real data, not just on the synthetic fixture.

    ``crossed_barrel`` is the only cached dataset that reaches it, and at 600
    points it is also the only one past ``SPLOM_DENSE_N``.
    """
    _require("crossed_barrel")
    ds = plot_run.load_public_dataset("crossed_barrel")
    assert len(ds.X) > plot_run.SPLOM_DENSE_N
    fig = plot_run.build_figure(
        ds.X, ds.Y, ds.labels, simplex=ds.simplex, bounds=ds.axis_bounds,
        grid_n=12, n_estimators=10, title=ds.title, value_name=ds.value_name,
        background="rf", show_points=True, gp_length_scale=0.3, scale=1.0,
        plot_size=0.8, color_limits=None, highlight=None,
    )
    splom = next(t for t in fig.data if type(t).__name__ == "Splom")
    assert len(splom.dimensions) == ds.d
    assert tuple(dim.label for dim in splom.dimensions) == ds.labels


@pytest.mark.parametrize("name,decreasing",
                         [("photo_pce10", True), ("hplc", False),
                          ("crossed_barrel", False)])
def test_convergence_envelope_follows_goal(name, decreasing):
    _require(name)
    ds = plot_run.load_public_dataset(name)
    fig = plot_run.build_convergence_figure(ds)
    env = next(t for t in fig.data if "best" in (t.name or ""))
    y = np.asarray(env.y, dtype=float)
    if decreasing:
        assert np.all(np.diff(y) <= 1e-12)
    else:
        assert np.all(np.diff(y) >= -1e-12)
