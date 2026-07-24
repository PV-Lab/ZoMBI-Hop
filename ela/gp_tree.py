"""Lightweight symbolic GP trees evaluated on ILR coordinates."""
from __future__ import annotations

import copy
import json
import math
import random
from typing import Any

import numpy as np

Node = tuple | float | int

UNARY_OPS: tuple[str, ...] = ("neg", "sin", "cos", "tanh", "sqr", "exp", "exp_neg", "abs")
BINARY_OPS: tuple[str, ...] = ("add", "sub", "mul", "div")
# Localized Gaussian bump in ILR: ("rbf", c0, c1, amp, length) → amp * exp(-‖z-c‖²/ℓ²).
SPECIAL_OPS: tuple[str, ...] = ("rbf",)
ALL_OPS: tuple[str, ...] = UNARY_OPS + BINARY_OPS + SPECIAL_OPS

DEFAULT_UNARY_WEIGHTS: dict[str, float] = {
    "neg": 0.8,
    "sin": 2.0,
    "cos": 2.0,
    "tanh": 1.2,
    "sqr": 1.8,
    "exp": 1.0,
    "exp_neg": 1.0,
    "abs": 1.2,
}
DEFAULT_BINARY_WEIGHTS: dict[str, float] = {
    "add": 0.8,
    "sub": 0.9,
    "mul": 2.5,
    "div": 1.0,
}

# Peak-stacking bias: prefer oscillatory unary + additive combination.
OSCILLATORY_UNARY_WEIGHTS: dict[str, float] = {
    "neg": 0.6,
    "sin": 3.5,
    "cos": 3.5,
    "tanh": 1.0,
    "sqr": 1.2,
    "exp": 0.8,
    "exp_neg": 0.8,
    "abs": 0.6,
}
OSCILLATORY_BINARY_WEIGHTS: dict[str, float] = {
    "add": 2.5,
    "sub": 1.2,
    "mul": 1.5,
    "div": 0.5,
}

# Muñoz-style GP operator mix (paper-faithful mode).
PAPER_UNARY_WEIGHTS: dict[str, float] = {
    "neg": 1.0,
    "sin": 1.2,
    "cos": 1.2,
    "tanh": 1.0,
    "sqr": 1.0,
    "exp": 0.8,
    "exp_neg": 0.8,
    "abs": 0.5,
}
PAPER_BINARY_WEIGHTS: dict[str, float] = {
    "add": 1.2,
    "sub": 1.0,
    "mul": 1.2,
    "div": 0.5,
}

# When allow_rbf: prefer additive stacking so bumps accumulate across the simplex.
RBF_BINARY_WEIGHTS: dict[str, float] = {
    "add": 3.5,
    "sub": 1.0,
    "mul": 1.0,
    "div": 0.4,
}

# RBF sampling rates. ``normal`` = mild bump prior; ``upweighted`` = bump-sweep rates.
RBF_RATE_PRESETS: dict[str, dict[str, float]] = {
    "normal": {
        "terminal": 0.22,
        "midgrow": 0.12,
        "jitter": 0.25,
        "graft": 0.15,
    },
    "upweighted": {
        "terminal": 0.45,
        "midgrow": 0.28,
        "jitter": 0.35,
        "graft": 0.30,
    },
}


def _rbf_rates(*, rbf_upweight: bool = True) -> dict[str, float]:
    return RBF_RATE_PRESETS["upweighted" if rbf_upweight else "normal"]


# Back-compat aliases (upweighted defaults).
RBF_TERMINAL_P = RBF_RATE_PRESETS["upweighted"]["terminal"]
RBF_MIDGROW_P = RBF_RATE_PRESETS["upweighted"]["midgrow"]
RBF_JITTER_P = RBF_RATE_PRESETS["upweighted"]["jitter"]
RBF_GRAFT_P = RBF_RATE_PRESETS["upweighted"]["graft"]


# Sentinel used when config sets ``max_tree_depth`` to null/0 ("no depth cap").
UNLIMITED_TREE_DEPTH = 10_000


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Use a signed floor so tiny negative divisors cannot cancel to zero.
    denom = np.where(np.abs(b) < 1e-6, np.copysign(1e-6, b + 1e-30), b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(a, denom, out=np.full_like(a, np.nan, dtype=float), where=denom != 0)


def _apply_unary(op: str, x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    if op == "neg":
        return -x
    if op == "sin":
        return np.sin(x)
    if op == "cos":
        return np.cos(x)
    if op == "tanh":
        return np.tanh(x)
    if op == "sqr":
        return x * x
    if op == "exp":
        return np.exp(np.clip(x, -10.0, 10.0))
    if op == "exp_neg":
        return np.exp(-np.clip(x, -10.0, 10.0))
    if op == "abs":
        return np.abs(x)
    raise ValueError(f"unknown unary op: {op}")


def _apply_binary(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.clip(a, -20.0, 20.0)
    b = np.clip(b, -20.0, 20.0)
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        return _safe_div(a, b)
    raise ValueError(f"unknown binary op: {op}")


def evaluate_tree(node: Node, z: np.ndarray) -> np.ndarray:
    """Evaluate symbolic tree on ILR design matrix ``z`` (n, ilr_dim)."""
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(1, -1)

    if isinstance(node, (float, int)):
        return np.full(z.shape[0], float(node), dtype=float)

    if not isinstance(node, tuple) or len(node) == 0:
        raise ValueError(f"invalid node: {node!r}")

    tag = node[0]
    if tag == "const":
        return np.full(z.shape[0], float(node[1]), dtype=float)
    if tag == "var":
        j = int(node[1])
        return z[:, j].copy()
    if tag == "rbf":
        # amp * exp(-((z0-c0)^2 + (z1-c1)^2) / length^2); uses first two ILR dims.
        c0 = float(node[1])
        c1 = float(node[2])
        amp = float(node[3])
        length = max(abs(float(node[4])), 1e-3)
        d2 = (z[:, 0] - c0) ** 2 + (z[:, 1] - c1) ** 2
        return amp * np.exp(-d2 / (length * length))
    if tag in UNARY_OPS:
        return _apply_unary(tag, evaluate_tree(node[1], z))
    if tag in BINARY_OPS:
        return _apply_binary(tag, evaluate_tree(node[1], z), evaluate_tree(node[2], z))
    raise ValueError(f"unknown node tag: {tag}")


def tree_depth(node: Node) -> int:
    if isinstance(node, (float, int)):
        return 0
    if not isinstance(node, tuple):
        return 0
    tag = node[0]
    if tag in ("const", "var", "rbf"):
        return 1
    if tag in UNARY_OPS:
        return 1 + tree_depth(node[1])
    if tag in BINARY_OPS:
        return 1 + max(tree_depth(node[1]), tree_depth(node[2]))
    return 0


def tree_size(node: Node) -> int:
    if isinstance(node, (float, int)):
        return 1
    if not isinstance(node, tuple):
        return 1
    tag = node[0]
    if tag in ("const", "var"):
        return 1
    if tag == "rbf":
        return 5  # count center/amp/length as parameters
    if tag in UNARY_OPS:
        return 1 + tree_size(node[1])
    if tag in BINARY_OPS:
        return 1 + tree_size(node[1]) + tree_size(node[2])
    return 1


NONLINEAR_OPS: frozenset[str] = frozenset(
    {"sin", "cos", "tanh", "sqr", "exp", "exp_neg", "abs", "mul", "div", "rbf"}
)


def tree_has_nonlinearity(node: Node) -> bool:
    """True if tree uses any op beyond affine combinations of vars."""
    if isinstance(node, (float, int)):
        return False
    if not isinstance(node, tuple):
        return False
    tag = node[0]
    if tag in NONLINEAR_OPS:
        return True
    if tag in UNARY_OPS:
        return tree_has_nonlinearity(node[1])
    if tag in BINARY_OPS:
        return tree_has_nonlinearity(node[1]) or tree_has_nonlinearity(node[2])
    return False


def raw_linearity_r2(z: np.ndarray, raw: np.ndarray) -> float:
    """R² of linear ILR fit to raw tree output; 1.0 ≈ smooth gradient / plane."""
    from sklearn.linear_model import LinearRegression

    z = np.asarray(z, dtype=float)
    raw = np.nan_to_num(np.asarray(raw, dtype=float).ravel(), nan=0.0)
    if np.std(raw) < 1e-12:
        return 1.0
    return float(LinearRegression().fit(z, raw).score(z, raw))


def tree_to_string(node: Node) -> str:
    if isinstance(node, (float, int)):
        return f"{float(node):.4g}"
    if not isinstance(node, tuple):
        return str(node)
    tag = node[0]
    if tag == "const":
        return f"{float(node[1]):.4g}"
    if tag == "var":
        return f"z{int(node[1])}"
    if tag == "rbf":
        return (
            f"rbf(c=({float(node[1]):.4g},{float(node[2]):.4g}),"
            f"a={float(node[3]):.4g},l={float(node[4]):.4g})"
        )
    if tag in UNARY_OPS:
        return f"{tag}({tree_to_string(node[1])})"
    if tag in BINARY_OPS:
        sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[tag]
        return f"({tree_to_string(node[1])} {sym} {tree_to_string(node[2])})"
    return repr(node)


def tree_to_jsonable(node: Node) -> Any:
    if isinstance(node, (float, int)):
        return float(node)
    if isinstance(node, tuple) and node and node[0] == "rbf":
        return ["rbf", float(node[1]), float(node[2]), float(node[3]), float(node[4])]
    return list(node[:1]) + [tree_to_jsonable(c) for c in node[1:]]


def tree_from_jsonable(data: Any) -> Node:
    if isinstance(data, list):
        if not data:
            raise ValueError("empty node list")
        tag = data[0]
        if tag == "var":
            return ("var", int(data[1]))
        if tag == "const":
            return ("const", float(data[1]))
        if tag == "rbf":
            return (
                "rbf",
                float(data[1]),
                float(data[2]),
                float(data[3]),
                float(data[4]),
            )
        if tag in UNARY_OPS:
            return (tag, tree_from_jsonable(data[1]))
        if tag in BINARY_OPS:
            return (tag, tree_from_jsonable(data[1]), tree_from_jsonable(data[2]))
        return tuple(tree_from_jsonable(x) for x in data)
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        return float(data)
    raise TypeError(f"cannot decode node from {type(data)}")


def _random_const(rng: random.Random) -> Node:
    if rng.random() < 0.3:
        return ("const", round(rng.uniform(-3.0, 3.0), 4))
    return ("const", round(rng.uniform(-1.0, 1.0), 4))


def _random_rbf(rng: random.Random) -> Node:
    """Random localized ILR Gaussian bump."""
    return (
        "rbf",
        round(rng.uniform(-2.5, 2.5), 4),
        round(rng.uniform(-2.5, 2.5), 4),
        round(rng.uniform(-2.0, 2.0), 4),
        round(rng.uniform(0.3, 2.5), 4),
    )


def _jitter_rbf(rng: random.Random, node: Node) -> Node:
    assert isinstance(node, tuple) and node[0] == "rbf"
    c0 = float(node[1]) + rng.uniform(-0.35, 0.35)
    c1 = float(node[2]) + rng.uniform(-0.35, 0.35)
    amp = float(node[3]) * rng.uniform(0.7, 1.3)
    length = max(0.15, abs(float(node[4])) * rng.uniform(0.7, 1.35))
    return (
        "rbf",
        round(c0, 4),
        round(c1, 4),
        round(amp, 4),
        round(length, 4),
    )


def _random_terminal(
    rng: random.Random,
    n_vars: int,
    *,
    allow_rbf: bool = False,
    rbf_upweight: bool = True,
) -> Node:
    rates = _rbf_rates(rbf_upweight=rbf_upweight)
    if allow_rbf and rng.random() < rates["terminal"]:
        return _random_rbf(rng)
    if rng.random() < 0.55:
        return ("var", rng.randrange(n_vars))
    return _random_const(rng)


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    vals = [weights[k] for k in keys]
    return rng.choices(keys, weights=vals, k=1)[0]


def random_additive_rbf_tree(
    rng: random.Random,
    *,
    max_depth: int = 8,
    allow_const: bool = True,
    min_bumps: int = 1,
    max_bumps: int | None = None,
) -> Node:
    """Forest of localized bumps: ``rbf + rbf + …`` (optional constant offset).

    Nesting depth ≈ number of bumps (left-associated adds). When ``max_bumps`` is
    None and depth is uncapped, init uses exactly ``min_bumps`` (mutation/crossover
    repair may grow further).
    """
    lo = max(1, int(min_bumps))
    if max_bumps is None:
        if max_depth >= UNLIMITED_TREE_DEPTH // 2:
            hi = lo
        else:
            hi = max(lo, int(max_depth))
    else:
        hi = max(lo, int(max_bumps))
    hi = max(lo, min(hi, max_depth))
    n_bumps = rng.randint(lo, hi)
    tree: Node = _random_rbf(rng)
    for _ in range(n_bumps - 1):
        tree = ("add", tree, _random_rbf(rng))
    # Only attach a const if there is depth headroom left.
    if allow_const and tree_depth(tree) < max_depth and rng.random() < 0.2:
        tree = ("add", tree, _random_const(rng))
    return tree


def count_rbf_bumps(node: Node) -> int:
    return sum(1 for n in _node_list(node) if isinstance(n, tuple) and n[0] == "rbf")


def random_tree(
    rng: random.Random,
    *,
    n_vars: int,
    max_depth: int = 5,
    method: str = "grow",
    paper_mode: bool = True,
    allow_rbf: bool = False,
    oscillatory_bias: bool = False,
    rbf_upweight: bool = True,
    rbf_additive_only: bool = False,
    rbf_min_bumps: int = 1,
    rbf_max_bumps: int | None = None,
) -> Node:
    """Ramped half-and-half style tree generation."""
    if rbf_additive_only:
        return random_additive_rbf_tree(
            rng,
            max_depth=max_depth,
            min_bumps=rbf_min_bumps,
            max_bumps=rbf_max_bumps,
        )

    if paper_mode:
        unary_w = PAPER_UNARY_WEIGHTS
        binary_w = PAPER_BINARY_WEIGHTS
        oscillatory_bias = False
    elif oscillatory_bias:
        unary_w = OSCILLATORY_UNARY_WEIGHTS
        binary_w = OSCILLATORY_BINARY_WEIGHTS
    elif allow_rbf:
        unary_w = DEFAULT_UNARY_WEIGHTS
        binary_w = RBF_BINARY_WEIGHTS
    else:
        unary_w = DEFAULT_UNARY_WEIGHTS
        binary_w = DEFAULT_BINARY_WEIGHTS

    rates = _rbf_rates(rbf_upweight=rbf_upweight)

    def grow(depth: int) -> Node:
        if depth <= 0 or (method == "grow" and rng.random() < 0.35):
            return _random_terminal(
                rng, n_vars, allow_rbf=allow_rbf, rbf_upweight=rbf_upweight,
            )
        if allow_rbf and rng.random() < rates["midgrow"]:
            # Treat RBF as a leaf-like motif that can still sit under add/mul.
            return _random_rbf(rng)
        if rng.random() < 0.45:
            op = _weighted_choice(rng, unary_w)
            return (op, grow(depth - 1))
        op = _weighted_choice(rng, binary_w)
        return (op, grow(depth - 1), grow(depth - 1))

    depth = rng.randint(2 if paper_mode else 3, max_depth)
    if paper_mode:
        return grow(depth)
    for _ in range(12):
        tree = grow(depth)
        if tree_has_nonlinearity(tree):
            return tree
    if allow_rbf:
        return _random_rbf(rng)
    op = _weighted_choice(rng, unary_w)
    return (
        op,
        _random_terminal(rng, n_vars, allow_rbf=allow_rbf, rbf_upweight=rbf_upweight),
    )


def _node_list(node: Node, acc: list[Node] | None = None) -> list[Node]:
    acc = acc if acc is not None else []
    acc.append(node)
    if isinstance(node, tuple) and len(node) > 0:
        tag = node[0]
        if tag in UNARY_OPS:
            _node_list(node[1], acc)
        elif tag in BINARY_OPS:
            _node_list(node[1], acc)
            _node_list(node[2], acc)
        # rbf / const / var: leaf-like, no children to walk
    return acc


def _replace_at(node: Node, target: Node, replacement: Node) -> Node:
    if node is target:
        return copy.deepcopy(replacement)
    if isinstance(node, tuple) and len(node) > 0:
        tag = node[0]
        if tag in UNARY_OPS:
            return (tag, _replace_at(node[1], target, replacement))
        if tag in BINARY_OPS:
            return (
                tag,
                _replace_at(node[1], target, replacement),
                _replace_at(node[2], target, replacement),
            )
    return node


def crossover(
    rng: random.Random,
    a: Node,
    b: Node,
    *,
    max_depth: int = 8,
    rbf_additive_only: bool = False,
    rbf_min_bumps: int = 1,
    rbf_max_bumps: int | None = None,
) -> tuple[Node, Node]:
    """Subtree crossover. Additive-RBF mode prefers swapping bump leaves + repairs mins."""
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)

    def _pick(nodes: list[Node], *, prefer_rbf: bool) -> Node:
        if prefer_rbf:
            rbfs = [n for n in nodes if isinstance(n, tuple) and n[0] == "rbf"]
            if rbfs:
                return rng.choice(rbfs)
        return rng.choice(nodes)

    prefer = bool(rbf_additive_only)
    nodes_a = _node_list(a)
    nodes_b = _node_list(b)
    pa = _pick(nodes_a, prefer_rbf=prefer)
    pb = _pick(nodes_b, prefer_rbf=prefer)
    child_a = _replace_at(a, pa, copy.deepcopy(pb))
    child_b = _replace_at(b, pb, copy.deepcopy(pa))
    for _ in range(6):
        if tree_depth(child_a) <= max_depth and tree_depth(child_b) <= max_depth:
            break
        child_a = copy.deepcopy(a)
        child_b = copy.deepcopy(b)
        pa = _pick(_node_list(child_a), prefer_rbf=prefer)
        pb = _pick(_node_list(child_b), prefer_rbf=prefer)
        child_a = _replace_at(child_a, pa, copy.deepcopy(pb))
        child_b = _replace_at(child_b, pb, copy.deepcopy(pa))

    if rbf_additive_only:
        child_a = repair_additive_rbf_forest(
            rng,
            child_a,
            min_bumps=rbf_min_bumps,
            max_bumps=rbf_max_bumps,
            max_depth=max_depth,
        )
        child_b = repair_additive_rbf_forest(
            rng,
            child_b,
            min_bumps=rbf_min_bumps,
            max_bumps=rbf_max_bumps,
            max_depth=max_depth,
        )
    return child_a, child_b


def _graft_rbf(rng: random.Random, node: Node) -> Node:
    """Plant a new bump: replace a random subtree with ``add(subtree, rbf)``."""
    nodes = _node_list(node)
    target = rng.choice(nodes)
    return _replace_at(node, target, ("add", copy.deepcopy(target), _random_rbf(rng)))


def _drop_one_rbf(rng: random.Random, node: Node) -> Node:
    """Remove one RBF leaf by replacing ``add(x, rbf)`` / ``add(rbf, x)`` with ``x``."""
    if not isinstance(node, tuple) or not node:
        return node
    if node[0] == "rbf":
        return _random_rbf(rng)
    if node[0] != "add":
        return node
    left, right = node[1], node[2]
    # Prefer collapsing an add whose one side is a bare rbf.
    if isinstance(right, tuple) and right[0] == "rbf" and rng.random() < 0.5:
        return copy.deepcopy(left)
    if isinstance(left, tuple) and left[0] == "rbf":
        return copy.deepcopy(right)
    if isinstance(right, tuple) and right[0] == "rbf":
        return copy.deepcopy(left)
    # Recurse into a random child that still has bumps.
    if count_rbf_bumps(left) >= count_rbf_bumps(right):
        return ("add", _drop_one_rbf(rng, left), copy.deepcopy(right))
    return ("add", copy.deepcopy(left), _drop_one_rbf(rng, right))


def repair_additive_rbf_forest(
    rng: random.Random,
    node: Node,
    *,
    min_bumps: int = 1,
    max_bumps: int | None = None,
    max_depth: int = 10_000,
) -> Node:
    """Enforce additive-RBF family + bump-count floor (and optional ceiling)."""
    if not _is_additive_rbf_forest(node):
        return random_additive_rbf_tree(
            rng,
            max_depth=max_depth,
            min_bumps=min_bumps,
            max_bumps=max_bumps,
        )
    node = copy.deepcopy(node)
    min_b = max(1, int(min_bumps))
    # Grow up to min bumps (and depth headroom).
    guard = 0
    while count_rbf_bumps(node) < min_b and tree_depth(node) < max_depth and guard < 256:
        node = _graft_rbf(rng, node)
        guard += 1
    if count_rbf_bumps(node) < min_b:
        return random_additive_rbf_tree(
            rng,
            max_depth=max_depth,
            min_bumps=min_b,
            max_bumps=max_bumps if max_bumps is not None else min_b,
        )
    if max_bumps is not None:
        max_b = max(min_b, int(max_bumps))
        guard = 0
        while count_rbf_bumps(node) > max_b and guard < 256:
            node = _drop_one_rbf(rng, node)
            guard += 1
        if count_rbf_bumps(node) > max_b or count_rbf_bumps(node) < min_b:
            return random_additive_rbf_tree(
                rng,
                max_depth=max_depth,
                min_bumps=min_b,
                max_bumps=max_b,
            )
    if tree_depth(node) > max_depth:
        return random_additive_rbf_tree(
            rng,
            max_depth=max_depth,
            min_bumps=min_b,
            max_bumps=max_bumps,
        )
    return node


def _is_additive_rbf_forest(node: Node) -> bool:
    """True if tree uses only ``add`` / ``rbf`` / ``const`` (no vars or other ops)."""
    if isinstance(node, (float, int)):
        return True
    if not isinstance(node, tuple) or not node:
        return False
    tag = node[0]
    if tag == "rbf":
        return True
    if tag == "const":
        return True
    if tag == "add":
        return _is_additive_rbf_forest(node[1]) and _is_additive_rbf_forest(node[2])
    return False


def mutate(
    rng: random.Random,
    node: Node,
    *,
    n_vars: int,
    max_depth: int = 8,
    p_subtree: float = 0.2,
    p_const: float = 0.1,
    paper_mode: bool = True,
    allow_rbf: bool = False,
    oscillatory_bias: bool = False,
    rbf_upweight: bool = True,
    rbf_additive_only: bool = False,
    rbf_min_bumps: int = 1,
    rbf_max_bumps: int | None = None,
) -> Node:
    if paper_mode:
        oscillatory_bias = False
    if rbf_additive_only:
        allow_rbf = True
    node = copy.deepcopy(node)
    rates = _rbf_rates(rbf_upweight=rbf_upweight)
    rbfs = [n for n in _node_list(node) if isinstance(n, tuple) and n[0] == "rbf"]
    n_bumps = len(rbfs)
    max_b = max_depth if rbf_max_bumps is None else int(rbf_max_bumps)
    max_b = max(1, min(max_b, max_depth))
    min_b = max(1, min(int(rbf_min_bumps), max_b))

    if rbf_additive_only:
        # Stay inside the additive-bump family: graft / jitter / replace / const.
        # Prefer grafting until min_bumps; block grafts past max_bumps.
        r = rng.random()
        if n_bumps < min_b or ((r < 0.45 or not rbfs) and n_bumps < max_b):
            node = _graft_rbf(rng, node)
        elif r < 0.75 and rbfs:
            target = rng.choice(rbfs)
            node = _replace_at(node, target, _jitter_rbf(rng, target))
        elif r < 0.92 and rbfs:
            target = rng.choice(rbfs)
            node = _replace_at(node, target, _random_rbf(rng))
        else:
            consts = [n for n in _node_list(node) if isinstance(n, tuple) and n[0] == "const"]
            if consts:
                node = _replace_at(node, rng.choice(consts), _random_const(rng))
            elif n_bumps < max_b:
                node = ("add", node, _random_const(rng))
            elif rbfs:
                node = _replace_at(node, rng.choice(rbfs), _jitter_rbf(rng, rng.choice(rbfs)))
        return repair_additive_rbf_forest(
            rng,
            node,
            min_bumps=min_b,
            max_bumps=None if rbf_max_bumps is None else max_b,
            max_depth=max_depth,
        )

    if allow_rbf and rng.random() < rates["graft"]:
        # Prefer planting a new localized bump over other mutations.
        node = _graft_rbf(rng, node)
    elif allow_rbf and rbfs and rng.random() < rates["jitter"]:
        target = rng.choice(rbfs)
        node = _replace_at(node, target, _jitter_rbf(rng, target))
    elif rng.random() < p_subtree:
        nodes = _node_list(node)
        target = rng.choice(nodes)
        repl = random_tree(
            rng,
            n_vars=n_vars,
            max_depth=max(2, max_depth - 2),
            paper_mode=paper_mode,
            allow_rbf=allow_rbf,
            oscillatory_bias=oscillatory_bias,
            rbf_upweight=rbf_upweight,
            rbf_additive_only=rbf_additive_only,
            rbf_min_bumps=rbf_min_bumps,
            rbf_max_bumps=rbf_max_bumps,
        )
        node = _replace_at(node, target, repl)
    elif rng.random() < p_const:
        nodes = [n for n in _node_list(node) if isinstance(n, tuple) and n[0] == "const"]
        if nodes:
            target = rng.choice(nodes)
            node = _replace_at(node, target, _random_const(rng))
    else:
        nodes = [n for n in _node_list(node) if isinstance(n, tuple) and n[0] == "var"]
        if nodes and rng.random() < 0.5:
            target = rng.choice(nodes)
            node = _replace_at(node, target, ("var", rng.randrange(n_vars)))
        else:
            nodes = _node_list(node)
            target = rng.choice(nodes)
            node = _replace_at(
                node,
                target,
                _random_terminal(
                    rng, n_vars, allow_rbf=allow_rbf, rbf_upweight=rbf_upweight,
                ),
            )
    if tree_depth(node) > max_depth:
        return random_tree(
            rng,
            n_vars=n_vars,
            max_depth=max_depth,
            paper_mode=paper_mode,
            allow_rbf=allow_rbf,
            oscillatory_bias=oscillatory_bias,
            rbf_upweight=rbf_upweight,
            rbf_additive_only=rbf_additive_only,
            rbf_min_bumps=rbf_min_bumps,
            rbf_max_bumps=rbf_max_bumps,
        )
    return node


def affine_rescale(y: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    """Map raw tree output to ``[y_min, y_max]`` via robust affine fit."""
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    lo, hi = float(np.percentile(y, 2)), float(np.percentile(y, 98))
    if hi - lo < 1e-9:
        return np.full_like(y, 0.5 * (y_min + y_max))
    scaled = (y - lo) / (hi - lo)
    return y_min + scaled * (y_max - y_min)


def fit_linear_calibration(
    y_raw: np.ndarray,
    y_ref: np.ndarray,
) -> tuple[float, float]:
    """Least-squares ``y_ref ≈ a * y_raw + b``; returns ``(a, b)``."""
    y_raw = np.nan_to_num(np.asarray(y_raw, dtype=float).ravel(), nan=0.0)
    y_ref = np.asarray(y_ref, dtype=float).ravel()
    if np.std(y_raw) < 1e-12:
        return 0.0, float(np.mean(y_ref))
    design = np.column_stack([y_raw, np.ones_like(y_raw)])
    coeffs, _, _, _ = np.linalg.lstsq(design, y_ref, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def apply_calibration(y_raw: np.ndarray, a: float, b: float) -> np.ndarray:
    y_raw = np.nan_to_num(np.asarray(y_raw, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    return a * y_raw + b


def evaluate_raw(node: Node, z: np.ndarray) -> np.ndarray:
    return evaluate_tree(node, z)


def predict_raw_clipped(node: Node, z: np.ndarray) -> np.ndarray:
    """Paper S1: evaluate ``g(z)`` directly (no post-hoc calibration)."""
    raw = evaluate_raw(node, z)
    return np.clip(
        np.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6),
        -50.0,
        50.0,
    )


def predict_calibrated(
    node: Node,
    z: np.ndarray,
    *,
    y_ref: np.ndarray | None = None,
    calib: tuple[float, float] | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Map tree output to objective values.

    If ``y_ref`` or ``calib`` is given, use linear calibration (best for λ_T matching).
    Otherwise fall back to percentile rescale into ``[y_min, y_max]``.
    """
    raw = evaluate_raw(node, z)
    if calib is not None:
        a, b = calib
    elif y_ref is not None:
        a, b = fit_linear_calibration(raw, y_ref)
    else:
        if y_min is None or y_max is None:
            raise ValueError("y_min and y_max required without calibration reference")
        return affine_rescale(raw, y_min, y_max), (1.0, 0.0)
    return apply_calibration(raw, a, b), (a, b)


def predict_tree(
    node: Node,
    z: np.ndarray,
    *,
    y_min: float,
    y_max: float,
    y_ref: np.ndarray | None = None,
    calib: tuple[float, float] | None = None,
) -> np.ndarray:
    y, _ = predict_calibrated(
        node, z, y_ref=y_ref, calib=calib, y_min=y_min, y_max=y_max,
    )
    return y


def parse_expression_calibration(
    metadata: dict[str, Any],
    *,
    linear_calibration: bool | None = None,
) -> tuple[float, float]:
    """Read ``(a, b)`` from ``expression.json`` metadata (handles bool legacy field)."""
    if linear_calibration is False:
        return 1.0, 0.0
    for key in ("calibration", "linear_calibration"):
        cal = metadata.get(key)
        if isinstance(cal, dict):
            return float(cal.get("a", 1.0)), float(cal.get("b", 0.0))
    enabled = metadata.get("linear_calibration_enabled")
    if enabled is None:
        legacy = metadata.get("linear_calibration")
        if isinstance(legacy, bool):
            enabled = legacy
    if enabled is False:
        return 1.0, 0.0
    return 1.0, 0.0


def dump_expression(path: str, node: Node, *, metadata: dict[str, Any] | None = None) -> None:
    payload = {
        "expression": tree_to_jsonable(node),
        "string": tree_to_string(node),
        "depth": tree_depth(node),
        "size": tree_size(node),
    }
    if metadata:
        payload.update(metadata)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
