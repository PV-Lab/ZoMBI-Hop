"""
Axial trace initialization for simplex-constrained search spaces.
=================================================================

Replaces random seed lines with a deterministic design whose size scales with the
number of components. Follows "Axial trace initialization for simplex-constrained
search spaces" (Oct 2026):

* One trace per component. Trace i is a straight composition gradient from the
  point where component i is largest (pure C_i on the bare simplex) through a
  common reference point to the opposite face, where C_i is smallest and the other
  components share the remainder.
* On the bare simplex the reference point is the centroid, so trace i runs from
  e_i to (1 - e_i)/(n - 1) -- exactly the reference design in the doc.
* Every component is swept over its full range, and every trace crosses every
  other trace at the reference point.

Hardware constraint: the printer runs traces in (front, back) pairs, so the number
of traces is rounded DOWN to an even number. When the component count is odd one
trace is dropped -- the floor in the doc, n - 1 traces, which still spans the
space. The dropped component is the one with the narrowest feasible range (the
doc's advice: drop a component that is kept low anyway), ties going to the last
component.

Search boxes
------------
ZoMBI-Hop runs inside a per-dim box ``lo <= x <= hi`` on the simplex. The doc
covers lower bounds with L-pseudocomponents and upper bounds by clipping, which
cuts traces short and loses the common crossing when an upper bound is tight.
This module instead generalises the construction so both properties survive any
feasible box:

1. Tighten the box to the bounds the simplex actually implies (``feasible_box``).
2. Pick a reference point ``s`` inside it (``reference_point``): the
   L-pseudocomponent centroid when it works -- always on the bare simplex and with
   lower bounds only, so the doc's design is reproduced exactly -- otherwise the
   point nearest to it on the way to the box-proportional centre at which every
   trace still sweeps its component over the full feasible range.
3. Trace i is the full chord through ``s`` along the Cox direction of component i
   in L-pseudocomponent space, ``V_i - s`` (``V_i`` = component i at its maximum,
   every other component at its lower bound), ending where a bound becomes active
   in each direction. Both ends therefore sit on the boundary of the search space.

Usage
-----
    from src.utils.axial_init import axial_design, pair_traces, sample_trace

    design = axial_design(6, lower=[0]*6, upper=[1, 1, 1, 0.3, 1, 1])
    for main, cache in pair_traces(design.traces, seed=0):
        X_main = sample_trace(main, 24)     # (24, 6), main[0] = start end
        ...

Run ``python -m src.utils.axial_init`` for the doc's self-test table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

_TOL = 1e-9


@dataclass
class AxialTrace:
    """One straight composition gradient.

    ``start`` is the end where ``component`` is at its maximum (pure C_i on the bare
    simplex); ``end`` is the opposite face, where it is at its minimum.
    """

    component: int
    start: np.ndarray
    end: np.ndarray

    def to_dict(self) -> dict:
        return {"component": int(self.component),
                "start": self.start.tolist(), "end": self.end.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "AxialTrace":
        return cls(int(d["component"]), np.asarray(d["start"], float),
                   np.asarray(d["end"], float))


@dataclass
class AxialDesign:
    traces: List[AxialTrace]
    reference: np.ndarray            # common crossing point s
    lower: np.ndarray                # effective (simplex-implied) lower bounds
    upper: np.ndarray                # effective upper bounds
    dropped: List[int] = field(default_factory=list)   # free components without a trace
    fixed: List[int] = field(default_factory=list)     # components the box pins to one value
    tau: float = 0.0                 # 0 = pseudocomponent centroid, 1 = proportional centre

    @property
    def n(self) -> int:
        return int(self.reference.shape[0])


def feasible_box(lower: Sequence[float], upper: Sequence[float],
                 tol: float = _TOL) -> Tuple[np.ndarray, np.ndarray]:
    """Tighten ``lower <= x <= upper`` to the bounds the simplex implies.

    On {lo <= x <= hi, sum(x) = 1} the exact range of x_i is
    [max(lo_i, 1 - sum_{j!=i} hi_j), min(hi_i, 1 - sum_{j!=i} lo_j)], because the
    other components can be chosen independently inside their own boxes.
    Raises ValueError if the box does not intersect the simplex.
    """
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    if lo.shape != hi.shape:
        raise ValueError(f"lower {lo.shape} and upper {hi.shape} bounds differ in length")
    lo = np.clip(lo, 0.0, 1.0)
    hi = np.clip(hi, 0.0, 1.0)
    if (hi < lo - tol).any():
        raise ValueError(f"upper bound below lower bound: lo={lo.tolist()} hi={hi.tolist()}")
    if lo.sum() > 1.0 + tol or hi.sum() < 1.0 - tol:
        raise ValueError(f"search box does not intersect the simplex "
                         f"(sum lo={lo.sum():.4f}, sum hi={hi.sum():.4f})")
    lo_eff = np.maximum(lo, 1.0 - (hi.sum() - hi))
    hi_eff = np.minimum(hi, 1.0 - (lo.sum() - lo))
    hi_eff = np.maximum(hi_eff, lo_eff)
    return lo_eff, hi_eff


def reference_point(lo: np.ndarray, hi: np.ndarray,
                    tol: float = _TOL) -> Tuple[np.ndarray, float]:
    """Common crossing point ``s`` for a box already tightened by ``feasible_box``.

    With u = s - lo, R = 1 - sum(lo) and w = hi - lo, trace i sweeps x_i over its
    full range [lo_i, hi_i] exactly when, for every j != i,

        u_j * R + w_j * u_i <= w_j * R

    (no other component reaches its upper bound before x_i reaches its lower).
    That holds for the L-pseudocomponent centroid u = R/n whenever every
    w_j >= R/(n - 1), and always for the box-proportional centre u = R * w / sum(w).
    The constraints are linear in u, so the admissible points on the segment
    between them form an interval; return the point closest to the centroid.
    Components with w = 0 are pinned and get u = 0. Returns (s, tau), tau = 0 at
    the centroid and 1 at the proportional centre.
    """
    w = hi - lo
    R = 1.0 - lo.sum()
    free = w > tol
    n_free = int(free.sum())
    if n_free < 2 or R <= tol:
        raise ValueError("search box pins the composition to a single point; "
                         "there is nothing to initialise")
    u_uni = np.where(free, R / n_free, 0.0)
    u_prop = R * np.where(free, w, 0.0) / w[free].sum()

    def g(u):  # g[i, j] = u_j R + w_j u_i - w_j R, must be <= 0 for i != j
        G = u[None, :] * R + w[None, :] * u[:, None] - w[None, :] * R
        np.fill_diagonal(G, -np.inf)
        return G

    g0, g1 = g(u_uni), g(u_prop)
    viol = g0 > tol * R
    tau = 0.0
    if viol.any():
        tau = float(np.max(g0[viol] / (g0[viol] - np.minimum(g1[viol], 0.0) + 1e-300)))
        tau = min(max(tau, 0.0), 1.0)
    u = (1.0 - tau) * u_uni + tau * u_prop
    return lo + u, tau


def _chord(s: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray,
           tol: float = _TOL) -> Tuple[float, float]:
    """Feasible parameter interval [t_min, t_max] of s + t*d inside the box."""
    t_min, t_max = -np.inf, np.inf
    for si, di, l, h in zip(s, d, lo, hi):
        if abs(di) <= tol * 1e-3:
            continue
        a, b = (l - si) / di, (h - si) / di
        t_min, t_max = max(t_min, min(a, b)), min(t_max, max(a, b))
    return t_min, t_max


def _clean(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Remove float dust: clip to the box and restore sum = 1."""
    x = np.clip(x, lo, hi)
    x[np.abs(x) < 1e-13] = 0.0
    return x / x.sum()


def n_traces_for(n_free: int, even: bool = True) -> int:
    """Number of traces for ``n_free`` free components: n, rounded down to even."""
    if n_free < 2:
        return 0
    return n_free - (n_free % 2) if even else n_free


def axial_design(
    n: Optional[int] = None,
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
    drop: Union[str, int, None] = "auto",
    even: bool = True,
) -> AxialDesign:
    """Build the axial trace design.

    Parameters
    ----------
    n : number of components (inferred from ``lower``/``upper`` if omitted).
    lower, upper : per-component search box (default 0 / 1).
    drop : which component loses its trace when ``even`` forces one out.
        "auto" -> narrowest feasible range, ties to the last component;
        an int -> that component; None -> keep every trace (ignores ``even``).
    even : round the trace count down to an even number (front/back rails).
    """
    if n is None:
        ref = lower if lower is not None else upper
        if ref is None:
            raise ValueError("give n or bounds")
        n = len(ref)
    n = int(n)
    lo_in = np.zeros(n) if lower is None else np.asarray(lower, float)
    hi_in = np.ones(n) if upper is None else np.asarray(upper, float)
    if lo_in.shape != (n,) or hi_in.shape != (n,):
        raise ValueError(f"bounds must have length {n}")

    lo, hi = feasible_box(lo_in, hi_in)
    s, tau = reference_point(lo, hi)
    w = hi - lo
    free = [i for i in range(n) if w[i] > _TOL]
    fixed = [i for i in range(n) if w[i] <= _TOL]

    keep = list(free)
    dropped: List[int] = []
    if drop is not None and even and len(keep) % 2 == 1:
        if drop == "auto":
            # narrowest range first; among ties the LAST index (stable, documented)
            victim = min(keep, key=lambda i: (round(w[i], 9), -i))
        else:
            victim = int(drop)
            if victim not in keep:
                raise ValueError(f"drop={victim} is not a free component {keep}")
        keep.remove(victim)
        dropped.append(victim)

    traces: List[AxialTrace] = []
    for i in keep:
        V = lo.copy()
        V[i] = 1.0 - (lo.sum() - lo[i])          # component i at max, others at lower bound
        d = V - s
        t_min, t_max = _chord(s, d, lo, hi)
        if not (t_max > 0 > t_min):
            raise RuntimeError(f"degenerate trace for component {i} (t in [{t_min}, {t_max}])")
        start = _clean(s + t_max * d, lo, hi)
        end = _clean(s + t_min * d, lo, hi)
        traces.append(AxialTrace(i, start, end))

    return AxialDesign(traces=traces, reference=s, lower=lo, upper=hi,
                       dropped=dropped, fixed=fixed, tau=tau)


def sample_trace(trace: AxialTrace, k: int) -> np.ndarray:
    """k evenly spaced compositions from ``trace.start`` to ``trace.end``."""
    t = np.linspace(0.0, 1.0, int(k))[:, None]
    return trace.start[None, :] + t * (trace.end - trace.start)[None, :]


def pair_traces(traces: Sequence[AxialTrace],
                seed: Optional[int] = None) -> List[Tuple[AxialTrace, AxialTrace]]:
    """Group traces into (main, cache) pairs for the two hardware rails.

    The doc recommends randomising run order so no component is confounded with
    time/drift; ``seed`` makes the order reproducible (None keeps the given order).
    """
    order = list(range(len(traces)))
    if seed is not None:
        np.random.default_rng(int(seed)).shuffle(order)
    if len(order) % 2:
        raise ValueError(f"need an even number of traces for two rails, got {len(order)}")
    return [(traces[order[k]], traces[order[k + 1]]) for k in range(0, len(order), 2)]


# ── evaluation helpers (validation / benchmarks) ─────────────────────────────

def uniform_feasible(n: int, n_samples: int, lower=None, upper=None,
                     seed: int = 0, max_rounds: int = 200) -> np.ndarray:
    """Uniform samples from the simplex restricted to a box (rejection)."""
    rng = np.random.default_rng(seed)
    lo = np.zeros(n) if lower is None else np.asarray(lower, float)
    hi = np.ones(n) if upper is None else np.asarray(upper, float)
    out, got = [], 0
    for _ in range(max_rounds):
        P = rng.dirichlet(np.ones(n), max(n_samples, 10_000))
        P = P[((P >= lo - 1e-12) & (P <= hi + 1e-12)).all(axis=1)]
        out.append(P)
        got += len(P)
        if got >= n_samples:
            break
    P = np.vstack(out)[:n_samples]
    if len(P) < n_samples:
        raise RuntimeError("box too small for rejection sampling")
    return P


def segment_coverage(segments: Sequence[Tuple[np.ndarray, np.ndarray]], P: np.ndarray
                     ) -> Tuple[float, float]:
    """Doc's coverage metric: mean / p95 distance from points P to the nearest segment."""
    best = np.full(len(P), np.inf)
    for a, b in segments:
        a = np.asarray(a, float)
        d = np.asarray(b, float) - a
        dd = float(d @ d)
        s = np.zeros(len(P)) if dd == 0 else np.clip((P - a) @ d / dd, 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(P - (a + s[:, None] * d), axis=1))
    return float(best.mean()), float(np.percentile(best, 95))


def point_coverage(X: np.ndarray, P: np.ndarray, chunk: int = 20_000) -> Tuple[float, float]:
    """Mean / p95 distance from points P to the nearest sampled point in X."""
    X = np.asarray(X, float)
    best = []
    for k in range(0, len(P), chunk):
        Q = P[k:k + chunk]
        d2 = (Q * Q).sum(1)[:, None] - 2 * Q @ X.T + (X * X).sum(1)[None, :]
        best.append(np.sqrt(np.maximum(d2.min(axis=1), 0.0)))
    b = np.concatenate(best)
    return float(b.mean()), float(np.percentile(b, 95))


if __name__ == "__main__":
    # Doc self-test: unique compositions and coverage on the bare simplex.
    for n, k in ((3, 7), (5, 11), (10, 11)):
        des = axial_design(n, drop=None)
        X = np.vstack([sample_trace(t, k) for t in des.traces])
        n_unique = len(np.unique(X.round(12), axis=0))
        assert n_unique == n * (k - 1) + 1, (n, k, n_unique)
        P = uniform_feasible(n, 100_000, seed=0)
        m, p95 = segment_coverage([(t.start, t.end) for t in des.traces], P)
        print(f"n={n:2d} k={k:2d} rows={len(X):3d} unique={n_unique:3d} "
              f"coverage mean={m:.3f} p95={p95:.3f}")
    print()
    print(" n  traces  dropped  coverage(mean/p95)")
    for n in range(2, 11):
        des = axial_design(n)
        P = uniform_feasible(n, 100_000, seed=0)
        m, p95 = segment_coverage([(t.start, t.end) for t in des.traces], P)
        print(f"{n:2d}  {len(des.traces):6d}  {str(des.dropped):7s}  {m:.3f} / {p95:.3f}")
