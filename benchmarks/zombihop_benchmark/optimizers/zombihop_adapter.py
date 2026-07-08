from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..line_audit import audit_line_endpoints
from ..seeding import set_global_seed
from ..spaces import composition_to_ilr_np, project_simplex
from ..types import BatchObservation, ObjectiveInfo


class ZoMBIHopAdapter:
    name = "zombihop"
    supports_point = False
    supports_line = True

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float64",
        max_activations: int = 1,
        num_points_per_line: int = 5,
        num_lines: int = 5,
        checkpoint_subdir: str = "checkpoints",
        **zombihop_params: Any,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.max_activations = int(max_activations)
        self.num_points_per_line = int(num_points_per_line)
        self.num_lines = int(num_lines)
        self.checkpoint_subdir = checkpoint_subdir
        self.zombihop_params = zombihop_params
        self._state: dict[str, Any] = {}

    def initialize(self, X: np.ndarray, y: np.ndarray, objective_info: ObjectiveInfo, seed: int) -> None:
        self._state = {
            "n_init": int(len(y)),
            "objective": objective_info.name,
            "seed": int(seed),
        }

    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        raise NotImplementedError("ZoMBIHopAdapter uses run_full_trial().")

    def observe(self, obs: BatchObservation) -> None:
        self._state["last_observation_count"] = int(len(obs.y))

    def get_state(self) -> dict[str, Any]:
        return dict(self._state)

    def run_full_trial(
        self,
        objective,
        X_init_actual: np.ndarray,
        X_init_expected: np.ndarray,
        Y_init: np.ndarray,
        run_dir: str | Path,
        seed: int,
        n_line_budget: int | None = None,
        points_per_line: int | None = None,
    ) -> dict[str, Any]:
        """Run ZoMBI-Hop with one benchmark objective call per evaluated LineBO line."""
        set_global_seed(seed)
        try:
            import torch
            from src.core.linebo import LineBO
            from src.core.zombihop import ZoMBIHop
        except ImportError as exc:
            raise RuntimeError(
                "ZoMBI-Hop smoke run requires torch, BoTorch/GPyTorch, and the repo's src package."
            ) from exc

        torch_dtype = getattr(torch, self.dtype)
        device = torch.device(self.device)
        n_components = objective.info.n_components
        points_per_line = int(points_per_line or self.num_points_per_line)
        line_budget = None if n_line_budget is None else int(n_line_budget)
        if line_budget is not None and line_budget <= 0:
            raise ValueError("ZoMBI-Hop line mode requires a positive n_line_budget")

        run_dir = Path(run_dir)
        checkpoint_dir = run_dir / self.checkpoint_subdir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        call_counter = {"n": 0}
        run_started = {"t": None}
        line_observations: list[BatchObservation] = []
        line_records: list[dict[str, Any]] = []

        def evaluate_ranked_line(endpoints_ranked):
            call_counter["n"] += 1
            line_index = call_counter["n"]
            line_start = time.time()
            endpoints_np = endpoints_ranked.detach().cpu().numpy()
            selected_endpoints = project_simplex(endpoints_np[0])
            obs_raw = objective.evaluate_line(
                selected_endpoints,
                n_points=points_per_line,
                seed=seed + 1000 + line_index,
            )
            selected_endpoints = _line_endpoints_from_observation(obs_raw, selected_endpoints)
            line_length_l2, line_length_ilr = _line_lengths(selected_endpoints)
            line_audit = audit_line_endpoints(selected_endpoints)
            runtime_s_line = time.time() - line_start
            line_id = f"zombihop_seed{seed}_line{line_index}"
            line_metadata = {
                "line_id": line_id,
                "candidate_index": 0,
                "endpoints": selected_endpoints.tolist(),
                "line_num_points": int(len(obs_raw.y)),
                "line_score": "",
                "line_score_method": "internal_linebo_mean_acq",
                "line_length_l2": line_length_l2,
                "line_length_ilr": line_length_ilr,
                "zombihop_internal_linebo": True,
                "n_ranked_candidate_lines": int(endpoints_ranked.shape[0]),
                "line_adapter": "zombihop_internal_linebo",
                "line_adapter_caveat": "ZoMBI-Hop chooses an internal candidate anchor and LineBO returns ranked printable lines.",
                **line_audit,
            }
            obs = BatchObservation(
                X_expected=obs_raw.X_expected,
                X_actual=obs_raw.X_actual,
                y=obs_raw.y,
                metadata={**dict(obs_raw.metadata), "line": line_metadata},
            )
            line_observations.append(obs)
            line_records.append(
                {
                    "line_index": line_index,
                    "optimizer": self.name,
                    "seed": seed,
                    "n_points": int(len(obs.y)),
                    "line_id": line_id,
                    "line_score": "",
                    "line_score_method": "internal_linebo_mean_acq",
                    "line_best_y": float(np.max(obs.y)),
                    "line_mean_y": float(np.mean(obs.y)),
                    "line_min_y": float(np.min(obs.y)),
                    "line_std_y": float(np.std(obs.y)),
                    "line_length_l2": line_length_l2,
                    "line_length_ilr": line_length_ilr,
                    **line_audit,
                    "line_adapter": "zombihop_internal_linebo",
                    "line_adapter_caveat": "ZoMBI-Hop chooses an internal candidate anchor and LineBO returns ranked printable lines.",
                    "runtime_s_line": runtime_s_line,
                    "selected_left": _json_list(selected_endpoints[0]),
                    "selected_right": _json_list(selected_endpoints[1]),
                    "n_ranked_candidate_lines": int(endpoints_ranked.shape[0]),
                    "zombihop_internal_linebo": True,
                    "activation": "",
                    "zoom": "",
                    "iteration": "",
                    "global_iteration": "",
                    "candidate_anchor": "",
                    "bounds_lower": "",
                    "bounds_upper": "",
                    "line_budget_reached": False,
                    "runtime_s_cumulative": "",
                }
            )
            x_expected = torch.tensor(obs.X_expected, device=device, dtype=torch_dtype)
            x_actual = torch.tensor(obs.X_actual, device=device, dtype=torch_dtype)
            y = torch.tensor(obs.y.reshape(-1), device=device, dtype=torch_dtype)
            return x_expected, x_actual, y

        linebo = LineBO(
            lambda endpoints_ranked: evaluate_ranked_line(endpoints_ranked)[1:],
            dimensions=n_components,
            num_points_per_line=points_per_line,
            num_lines=self.num_lines,
            device=str(device),
        )

        def zombi_objective(x_candidate, bounds, acquisition_function):
            x_left_ranked, x_right_ranked = linebo.ranked_line_endpoints(
                x_candidate, bounds, acquisition_function
            )
            endpoints_ranked = torch.stack([x_left_ranked, x_right_ranked], dim=1)
            return evaluate_ranked_line(endpoints_ranked)

        def objective_call_callback(context: dict[str, Any]) -> None:
            line_idx = int(context["objective_calls"]) - 1
            if line_idx < 0 or line_idx >= len(line_records):
                return
            bounds_list = _to_numpy_or_list(context["bounds"])
            line_record = line_records[line_idx]
            line_record.update(
                {
                    "activation": int(context["activation"]),
                    "zoom": int(context["zoom"]),
                    "iteration": int(context["iteration"]),
                    "global_iteration": int(context["global_iteration"]),
                    "candidate_anchor": _json_list(context["candidate"]),
                    "bounds_lower": _json_list(bounds_list[0]),
                    "bounds_upper": _json_list(bounds_list[1]),
                    "line_budget_reached": bool(context["line_budget_reached"]),
                    "runtime_s_cumulative": (
                        "" if run_started["t"] is None else float(time.time() - run_started["t"])
                    ),
                }
            )
            line_meta = line_observations[line_idx].metadata.get("line", {})
            line_meta.update(
                {
                    "activation": line_record["activation"],
                    "zoom": line_record["zoom"],
                    "iteration": line_record["iteration"],
                    "global_iteration": line_record["global_iteration"],
                    "candidate_anchor": line_record["candidate_anchor"],
                    "bounds_lower": line_record["bounds_lower"],
                    "bounds_upper": line_record["bounds_upper"],
                    "line_budget_reached": line_record["line_budget_reached"],
                }
            )

        kwargs = dict(self.zombihop_params)
        kwargs.setdefault("max_zooms", 1)
        kwargs.setdefault("max_iterations", 2)
        kwargs.setdefault("n_restarts", 3)
        kwargs.setdefault("raw", 32)
        kwargs.setdefault("max_gp_points", 128)
        kwargs.setdefault("verbose", False)

        optimizer = ZoMBIHop(
            objective=zombi_objective,
            X_init_actual=torch.tensor(X_init_actual, device=device, dtype=torch_dtype),
            X_init_expected=torch.tensor(X_init_expected, device=device, dtype=torch_dtype),
            Y_init=torch.tensor(Y_init.reshape(-1, 1), device=device, dtype=torch_dtype),
            device=str(device),
            dtype=torch_dtype,
            checkpoint_dir=str(checkpoint_dir),
            resume=False,
            **kwargs,
        )
        start = time.time()
        run_started["t"] = start
        needles_results, needles, needle_vals, X_all_actual, Y_all = optimizer.run(
            max_activations=self.max_activations,
            max_objective_calls=line_budget,
            objective_call_callback=objective_call_callback,
        )
        runtime_s = time.time() - start
        X_actual, X_expected, Y_all_full = optimizer.data_handler.get_all_points()
        obs = BatchObservation(
            X_expected=X_expected.detach().cpu().numpy(),
            X_actual=X_actual.detach().cpu().numpy(),
            y=Y_all_full.detach().cpu().numpy().reshape(-1),
            metadata={"kind": "zombihop_full_trial", "line_calls": call_counter["n"]},
        )
        line_budget_reached = line_budget is None or call_counter["n"] >= line_budget
        self._state = {
            "run_uuid": optimizer.run_uuid,
            "runtime_s": runtime_s,
            "line_calls": call_counter["n"],
            "line_budget_requested": line_budget,
            "line_budget_reached": line_budget_reached,
            "points_per_line": points_per_line,
            "zombihop_internal_linebo": True,
            "n_points": int(len(obs.y)),
        }
        return {
            "observation": obs,
            "line_observations": line_observations,
            "line_records": line_records,
            "runtime_s": runtime_s,
            "line_budget_requested": line_budget,
            "line_budget_reached": line_budget_reached,
            "points_per_line": points_per_line,
            "needles_results": needles_results,
            "needles": _to_numpy_or_list(needles),
            "needle_vals": _to_numpy_or_list(needle_vals),
            "X_all_actual": _to_numpy_or_list(X_all_actual),
            "Y_all": _to_numpy_or_list(Y_all),
            "state": self.get_state(),
        }


def _to_numpy_or_list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _json_list(value: Any) -> str:
    return json.dumps(_to_numpy_or_list(value))


def _line_endpoints_from_observation(obs: BatchObservation, fallback: np.ndarray) -> np.ndarray:
    metadata = obs.metadata or {}
    if "left" in metadata and "right" in metadata:
        return project_simplex(np.asarray([metadata["left"], metadata["right"]], dtype=float))
    return project_simplex(np.asarray(fallback, dtype=float))


def _line_lengths(endpoints: np.ndarray) -> tuple[float, float]:
    endpoints = project_simplex(endpoints)
    length_l2 = float(np.linalg.norm(endpoints[1] - endpoints[0]))
    endpoint_ilr = composition_to_ilr_np(endpoints)
    length_ilr = float(np.linalg.norm(endpoint_ilr[1] - endpoint_ilr[0]))
    return length_l2, length_ilr
