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
ALL_OPS: tuple[str, ...] = UNARY_OPS + BINARY_OPS

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
    if tag in ("const", "var"):
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
    if tag in UNARY_OPS:
        return 1 + tree_size(node[1])
    if tag in BINARY_OPS:
        return 1 + tree_size(node[1]) + tree_size(node[2])
    return 1


NONLINEAR_OPS: frozenset[str] = frozenset(
    {"sin", "cos", "tanh", "sqr", "exp", "exp_neg", "abs", "mul", "div"}
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
    if tag in UNARY_OPS:
        return f"{tag}({tree_to_string(node[1])})"
    if tag in BINARY_OPS:
        sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[tag]
        return f"({tree_to_string(node[1])} {sym} {tree_to_string(node[2])})"
    return repr(node)


def tree_to_jsonable(node: Node) -> Any:
    if isinstance(node, (float, int)):
        return float(node)
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


def _random_terminal(rng: random.Random, n_vars: int) -> Node:
    if rng.random() < 0.55:
        return ("var", rng.randrange(n_vars))
    return _random_const(rng)


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    vals = [weights[k] for k in keys]
    return rng.choices(keys, weights=vals, k=1)[0]


def random_tree(
    rng: random.Random,
    *,
    n_vars: int,
    max_depth: int = 5,
    method: str = "grow",
    paper_mode: bool = True,
) -> Node:
    """Ramped half-and-half style tree generation."""
    unary_w = PAPER_UNARY_WEIGHTS if paper_mode else DEFAULT_UNARY_WEIGHTS
    binary_w = PAPER_BINARY_WEIGHTS if paper_mode else DEFAULT_BINARY_WEIGHTS

    def grow(depth: int) -> Node:
        if depth <= 0 or (method == "grow" and rng.random() < 0.35):
            return _random_terminal(rng, n_vars)
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
    op = _weighted_choice(rng, DEFAULT_UNARY_WEIGHTS)
    return (op, _random_terminal(rng, n_vars))


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


def crossover(rng: random.Random, a: Node, b: Node, *, max_depth: int = 8) -> tuple[Node, Node]:
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    nodes_a = _node_list(a)
    nodes_b = _node_list(b)
    pa = rng.choice(nodes_a)
    pb = rng.choice(nodes_b)
    child_a = _replace_at(a, pa, copy.deepcopy(pb))
    child_b = _replace_at(b, pb, copy.deepcopy(pa))
    for _ in range(4):
        if tree_depth(child_a) <= max_depth and tree_depth(child_b) <= max_depth:
            break
        child_a = copy.deepcopy(a)
        child_b = copy.deepcopy(b)
        pa = rng.choice(_node_list(child_a))
        pb = rng.choice(_node_list(child_b))
        child_a = _replace_at(child_a, pa, copy.deepcopy(pb))
        child_b = _replace_at(child_b, pb, copy.deepcopy(pa))
    return child_a, child_b


def mutate(
    rng: random.Random,
    node: Node,
    *,
    n_vars: int,
    max_depth: int = 8,
    p_subtree: float = 0.2,
    p_const: float = 0.1,
) -> Node:
    node = copy.deepcopy(node)
    if rng.random() < p_subtree:
        nodes = _node_list(node)
        target = rng.choice(nodes)
        repl = random_tree(rng, n_vars=n_vars, max_depth=max(2, max_depth - 2))
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
            node = _replace_at(node, target, _random_terminal(rng, n_vars))
    if tree_depth(node) > max_depth:
        return random_tree(rng, n_vars=n_vars, max_depth=max_depth)
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
