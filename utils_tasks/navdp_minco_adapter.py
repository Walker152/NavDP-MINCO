import time

import numpy as np

from experiments.analyzers.metrics import compute_geometric_metrics, compute_minco_temporal_profile
from utils_tasks.esdf_query_utils import EsdfGridView


def preprocess_guide_path(candidate):
    """Sanitize only; native MINCO owns local extraction and path processing."""
    path = np.asarray(candidate, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] < 2:
        return {"valid": False, "reason": "INVALID_PATH_SHAPE", "path": None}
    if not np.all(np.isfinite(path)):
        return {"valid": False, "reason": "NONFINITE_PATH", "path": None}
    path = path[:, :3] if path.shape[1] >= 3 else np.column_stack((path[:, :2], np.zeros(len(path))))
    path[:, 2] = 0.0
    keep = np.r_[True, np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1) > 1e-4]
    path = path[keep]
    if len(path) < 2:
        return {"valid": False, "reason": "PATH_TOO_SHORT", "path": None}
    return {"valid": True, "reason": "NONE", "path": path}


class NavDPMincoAdapter:
    def __init__(
        self,
        esdf: dict,
        optimization_safe_dist=0.45,
        validation_safe_dist=0.35,
        initial_top_k=2,
        max_top_k=4,
        candidate_time_budget_ms=1500.0,
        start_validation_exemption_radius=0.35,
        sample_dt=0.05,
        speed=0.5,
        max_vel=None,
        max_acc=4.0,
        max_iterations=64,
        max_yaw_rate=0.5,
        penalty_weight_pos=1000.0,
        penalty_weight_vel=1000.0,
        penalty_weight_acc=10000.0,
        penalty_weight_attractor=20.0,
        time_weight=0.01,
        time_barrier_weight=100.0,
        warm_start_mode="gated",
        enable=True,
    ):
        self.esdf = esdf
        self._esdf_grid = EsdfGridView.from_mapping(esdf)
        self.optimization_safe_dist = float(optimization_safe_dist)
        self.validation_safe_dist = float(validation_safe_dist)
        if self.optimization_safe_dist < self.validation_safe_dist:
            raise ValueError("optimization_safe_dist must be >= validation_safe_dist")
        self.initial_top_k = max(1, int(initial_top_k))
        self.max_top_k = max(self.initial_top_k, int(max_top_k))
        self.candidate_time_budget_ms = float(candidate_time_budget_ms)
        self.start_validation_exemption_radius = max(
            0.0, float(start_validation_exemption_radius)
        )
        self.sample_dt = float(sample_dt)
        self.speed = float(speed)
        self.max_vel = float(self.speed if max_vel is None else max_vel)
        self.max_acc = float(max_acc)
        self.max_iterations = int(max_iterations)
        self.max_yaw_rate = float(max_yaw_rate)
        self.penalty_weights = np.asarray(
            [penalty_weight_pos, penalty_weight_vel, penalty_weight_acc,
             penalty_weight_attractor, time_barrier_weight], dtype=np.float64
        )
        self.time_weight = float(time_weight)
        if not np.isfinite(self.max_yaw_rate) or self.max_yaw_rate <= 0.0:
            raise ValueError("max_yaw_rate must be finite and positive")
        if not np.all(np.isfinite(self.penalty_weights)) or np.any(self.penalty_weights < 0.0):
            raise ValueError("MINCO penalty weights must be finite and non-negative")
        if not np.isfinite(self.time_weight) or self.time_weight < 0.0:
            raise ValueError("MINCO time weight must be finite and non-negative")
        self.enabled = bool(enable)
        if warm_start_mode not in {"cold", "gated"}:
            raise ValueError("warm_start_mode must be cold or gated")
        self.warm_start_mode = warm_start_mode
        self.processor = None
        if self.enabled:
            import minco_processor

            self.processor = minco_processor.MincoProcessor()
            if hasattr(self.processor, "configure"):
                self.processor.configure(
                    max_vel=self.max_vel,
                    max_acc=self.max_acc,
                    optimization_safe_dist=self.optimization_safe_dist,
                    validation_safe_dist=self.validation_safe_dist,
                    start_validation_exemption_radius=self.start_validation_exemption_radius,
                    sample_dt=self.sample_dt,
                    max_iterations=self.max_iterations,
                    max_yaw_rate=self.max_yaw_rate,
                    penalty_weight_pos=float(self.penalty_weights[0]),
                    penalty_weight_vel=float(self.penalty_weights[1]),
                    penalty_weight_acc=float(self.penalty_weights[2]),
                    penalty_weight_attractor=float(self.penalty_weights[3]),
                    time_weight=self.time_weight,
                    time_barrier_weight=float(self.penalty_weights[4]),
                )
            self.processor.set_static_esdf_2d(
                distance=self._esdf_grid.distance,
                free=np.asarray(esdf["free"], dtype=np.uint8),
                origin=self._esdf_grid.origin,
                resolution=self._esdf_grid.resolution,
            )

    def optimize_candidates(
        self,
        candidates_world,
        critic_values,
        states,
        raw_top1_world,
        terminal_goals_world=None,
    ):
        if self.warm_start_mode == "cold" and hasattr(self.processor, "reset_history"):
            self.processor.reset_history()
        candidates_world = np.asarray(candidates_world, dtype=object)
        critic_values = np.asarray(critic_values)
        raw_top1_world = np.asarray(raw_top1_world, dtype=object)
        batch_size = len(candidates_world)
        if terminal_goals_world is None:
            terminal_goals_world = [None for _ in range(batch_size)]
        results = []
        for env_idx in range(batch_size):
            start_time = time.perf_counter()
            best = None
            failures = []
            candidate_timings = []
            state = states[env_idx]
            zero_state = np.zeros(3, dtype=np.float64)
            position = np.asarray(state.get("position", zero_state), dtype=np.float64)
            velocity = np.asarray(state.get("velocity", zero_state), dtype=np.float64)
            acceleration = np.asarray(state.get("acceleration", zero_state), dtype=np.float64)
            yaw = float(state.get("yaw", 0.0))
            yaw_rate = float(state.get("yaw_rate", 0.0))
            terminal_goal = self._as_terminal_goal(terminal_goals_world[env_idx])
            candidate_screen_start = time.perf_counter()
            order, candidate_evaluations = self.rank_candidates(
                candidates_world[env_idx],
                critic_values[env_idx],
                self._esdf_grid,
                self.validation_safe_dist,
                self.start_validation_exemption_radius,
                position,
            )
            candidate_screen_ms = (
                time.perf_counter() - candidate_screen_start
            ) * 1000.0
            candidate_indices = order[:min(self.max_top_k, len(order))]
            for rank, selected_idx in enumerate(candidate_indices):
                if rank >= self.initial_top_k and (time.perf_counter() - start_time) * 1000.0 >= self.candidate_time_budget_ms:
                    failures.append("CANDIDATE_TIME_BUDGET_EXHAUSTED")
                    break
                screened = preprocess_guide_path(candidates_world[env_idx][selected_idx])
                candidate = screened["path"]
                if not screened["valid"]:
                    reason = screened["reason"]
                    failures.append(f"idx={selected_idx}: {reason}")
                    candidate_timings.append({
                        "candidate_rank": int(rank),
                        "selected_index": int(selected_idx),
                        "python_call_ms": 0.0,
                        "cpp_optimize_time_ms": float("nan"),
                        "success": False,
                        "objective": float("inf"),
                        "min_esdf": candidate_evaluations[selected_idx]["min_esdf"],
                        "screen_safe": False,
                        "screen_reason": candidate_evaluations[selected_idx]["screen_reason"],
                        "failure_reason": reason,
                    })
                    continue
                candidate_call_start = time.perf_counter()
                try:
                    optimize_method = self.processor.optimize_preview if hasattr(self.processor, "optimize_preview") else self.processor.optimize
                    result = optimize_method(
                        guide_path=candidate,
                        position=position,
                        velocity=velocity,
                        acceleration=acceleration,
                        yaw=yaw,
                        yaw_rate=yaw_rate,
                        terminal_goal=terminal_goal,
                    )
                except Exception as exc:
                    candidate_call_ms = (time.perf_counter() - candidate_call_start) * 1000.0
                    candidate_timings.append({
                        "candidate_rank": int(rank),
                        "selected_index": int(selected_idx),
                        "python_call_ms": float(candidate_call_ms),
                        "cpp_optimize_time_ms": float("nan"),
                        "success": False,
                        "objective": float("inf"),
                        "min_esdf": candidate_evaluations[selected_idx]["min_esdf"],
                        "screen_safe": candidate_evaluations[selected_idx]["screen_safe"],
                        "screen_reason": candidate_evaluations[selected_idx]["screen_reason"],
                        "failure_reason": str(exc),
                    })
                    failures.append(f"idx={selected_idx}: {exc}")
                    continue
                candidate_call_ms = (time.perf_counter() - candidate_call_start) * 1000.0
                cpp_ms = float(result.get("cpp_optimize_time_ms", result.get("duration", 0.0) * 1000.0))
                candidate_timings.append({
                    "candidate_rank": int(rank),
                    "selected_index": int(selected_idx),
                    "python_call_ms": float(candidate_call_ms),
                    "cpp_optimize_time_ms": float(cpp_ms),
                    "timing_ms": dict(result.get("timing_ms", {})),
                    "dense_path_size": int(result.get("dense_path_size", 0)),
                    "sparse_waypoint_size": int(result.get("sparse_waypoint_size", 0)),
                    "optimizer_iteration_count": int(result.get("optimizer_iteration_count", 0)),
                    "optimizer_return_code": int(result.get("optimizer_return_code", 0)),
                    "mandatory_corner_count": int(result.get("mandatory_corner_count", 0)),
                    "planning_state": str(result.get("planning_state", "")),
                    "local_end_is_goal": bool(result.get("local_end_is_goal", False)),
                    "success": bool(result.get("success", False)),
                    "optimizer_success": bool(np.isfinite(float(result.get("objective", np.inf)))),
                    "objective": float(result.get("objective", np.inf)),
                    "min_esdf": float(result.get("min_esdf", np.nan)),
                    "screen_safe": candidate_evaluations[selected_idx]["screen_safe"],
                    "screen_reason": candidate_evaluations[selected_idx]["screen_reason"],
                    "screen_min_esdf": candidate_evaluations[selected_idx]["min_esdf"],
                    "failure_reason": str(result.get("failure_reason", "")),
                    "validation_min_clearance": float(result.get("validation_min_clearance", np.nan)),
                    "validation_oob_count": int(result.get("validation_oob_count", 0)),
                    "validation_start_exempt_count": int(result.get("validation_start_exempt_count", 0)),
                    "validation_failure_reason": str(result.get("validation_failure_reason", "")),
                    "cpp_validation_success": bool(result.get("success", False)),
                    "python_validation_success": "",
                })
                if not result.get("success", False):
                    if result.get("proposal_id") is not None and hasattr(self.processor, "discard_proposal"):
                        self.processor.discard_proposal(int(result["proposal_id"]))
                    failures.append(f"idx={selected_idx}: {result.get('failure_reason', 'FAILED')}")
                    continue
                waypoints = np.asarray(result.get("waypoints", []), dtype=np.float64)
                if waypoints.ndim != 2 or waypoints.shape[0] < 2 or waypoints.shape[1] < 2:
                    failures.append(f"idx={selected_idx}: invalid_waypoints")
                    candidate_timings[-1]["success"] = False
                    candidate_timings[-1]["failure_reason"] = "invalid_waypoints"
                    continue
                validation_start = time.perf_counter()
                py_report = self._inspect_esdf(
                    waypoints,
                    self.validation_safe_dist,
                    self.start_validation_exemption_radius,
                )
                python_validation_ms = (time.perf_counter() - validation_start) * 1000.0
                py_min_esdf = float(py_report["min_clearance"])
                candidate_timings[-1]["python_validation_ms"] = python_validation_ms
                candidate_timings[-1]["python_min_esdf"] = py_min_esdf
                candidate_timings[-1]["python_validation_reason"] = py_report["reason"]
                candidate_timings[-1]["python_validation_success"] = bool(py_report["safe"])
                if not py_report["safe"]:
                    if result.get("proposal_id") is not None and hasattr(self.processor, "discard_proposal"):
                        self.processor.discard_proposal(int(result["proposal_id"]))
                    unsafe_reason = (
                        f"PY_ESDF_{py_report['reason']} py_min={py_min_esdf:.3f} "
                        f"safe={self.validation_safe_dist:.3f}"
                    )
                    failures.append(f"idx={selected_idx}: {unsafe_reason}")
                    candidate_timings[-1]["success"] = False
                    candidate_timings[-1]["failure_reason"] = unsafe_reason
                    continue
                scored = dict(result)
                scored["selected_index"] = int(selected_idx)
                scored["selected_candidate_rank"] = int(rank)
                scored["python_call_ms"] = candidate_call_ms
                scored["cpp_optimize_time_ms"] = cpp_ms
                scored["py_min_esdf"] = py_min_esdf
                scored["python_validation_ms"] = python_validation_ms
                scored["python_validation_report"] = py_report
                best = scored
                break

            pre_metrics_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if best is not None:
                result_metrics_start = time.perf_counter()
                samples = best.get("samples")
                samples_np = None
                speed_profile = None
                if samples is not None:
                    samples_np = np.asarray(samples, dtype=np.float64)
                    if samples_np.ndim == 2 and samples_np.shape[1] >= 6:
                        speed_profile = np.linalg.norm(samples_np[:, 4:6], axis=1)

                sparse_waypoints = best.get("sparse_waypoints", None)
                if sparse_waypoints is not None:
                    sparse_waypoints = np.asarray(sparse_waypoints, dtype=np.float64)
                    if sparse_waypoints.ndim == 2 and sparse_waypoints.shape[1] >= 2:
                        sparse_waypoints = sparse_waypoints[:, :2]
                    else:
                        sparse_waypoints = None

                selected_candidate = np.asarray(
                    candidates_world[env_idx][best["selected_index"]],
                    dtype=np.float64,
                )
                if selected_candidate.ndim == 2 and selected_candidate.shape[1] >= 2:
                    selected_candidate = selected_candidate[:, :2]
                else:
                    selected_candidate = None

                raw_top1 = np.asarray(raw_top1_world[env_idx], dtype=np.float64)
                if raw_top1.ndim == 2 and raw_top1.shape[1] >= 2:
                    raw_top1 = raw_top1[:, :2]
                else:
                    raw_top1 = None
                raw_geometry, _ = compute_geometric_metrics(raw_top1)
                raw_safety = self._inspect_esdf(
                    raw_top1,
                    self.validation_safe_dist,
                    self.start_validation_exemption_radius,
                )
                minco_waypoints = np.asarray(best["waypoints"], dtype=np.float64)[:, :2]
                minco_geometry, _ = compute_geometric_metrics(minco_waypoints)
                minco_safety = self._inspect_esdf(
                    minco_waypoints,
                    self.validation_safe_dist,
                    self.start_validation_exemption_radius,
                )
                temporal_metrics, _ = compute_minco_temporal_profile(samples_np)

                result = {
                    "success": True,
                    "waypoints": minco_waypoints,
                    "samples": samples_np,
                    "selected_index": int(best["selected_index"]),
                    "configured_top_k": int(self.max_top_k),
                    "attempted_candidate_count": int(len(candidate_timings)),
                    "attempted_candidate_indices": [
                        int(item["selected_index"]) for item in candidate_timings
                    ],
                    "selected_candidate_rank": int(best["selected_candidate_rank"]),
                    "objective": float(best.get("objective", np.inf)),
                    "min_esdf": float(best.get("min_esdf", np.nan)),
                    "py_min_esdf": float(best.get("py_min_esdf", np.nan)),
                    "optimization_safe_dist": self.optimization_safe_dist,
                    "validation_safe_dist": self.validation_safe_dist,
                    "validation_failure_reason": str(best.get("validation_failure_reason", "NONE")),
                    "failure_reason": best.get("failure_reason", "NONE"),
                    "fallback": False,
                    "fallback_mode": "NONE",
                    "status": "MINCO_OK",
                    "time_ms": pre_metrics_elapsed_ms,
                    "sparse_waypoints": sparse_waypoints,
                    "selected_candidate": selected_candidate,
                    "raw_top1": raw_top1,
                    "speed_profile": speed_profile,
                    "adapter_total_ms": pre_metrics_elapsed_ms,
                    "candidate_timings": candidate_timings,
                    "candidate_evaluations": candidate_evaluations,
                    "selected_cpp_optimize_time_ms": float(best.get("cpp_optimize_time_ms", best.get("duration", 0.0) * 1000.0)),
                    "selected_python_call_ms": float(best.get("python_call_ms", np.nan)),
                    "timing_ms": dict(best.get("timing_ms", {})),
                    "dense_path_size": int(best.get("dense_path_size", 0)),
                    "sparse_waypoint_size": int(best.get("sparse_waypoint_size", 0)),
                    "mandatory_corner_count": int(best.get("mandatory_corner_count", 0)),
                    "planning_state": str(best.get("planning_state", "")),
                    "local_end_is_goal": bool(best.get("local_end_is_goal", False)),
                    "optimizer_iteration_count": int(best.get("optimizer_iteration_count", 0)),
                    "optimizer_return_code": int(best.get("optimizer_return_code", 0)),
                    "optimizer_success": True,
                    "validation_min_clearance": float(best.get("validation_min_clearance", np.nan)),
                    "validation_oob_count": int(best.get("validation_oob_count", 0)),
                    "validation_start_exempt_count": int(best.get("validation_start_exempt_count", 0)),
                    "python_validation_ms": float(best.get("python_validation_ms", np.nan)),
                    "cpp_validation_success": True,
                    "python_validation_success": True,
                    "raw_min_clearance_m": raw_safety["min_clearance"],
                    "raw_unsafe_ratio": raw_safety["unsafe_ratio"],
                    "raw_esdf_oob_ratio": raw_safety["oob_ratio"],
                    "raw_path_length_m": raw_geometry["path_length_m"],
                    "raw_curvature_abs_p95_1pm": raw_geometry["curvature_abs_p95_1pm"],
                    "raw_curvature_tv_1pm": raw_geometry["curvature_tv_1pm"],
                    "minco_min_clearance_m": minco_safety["min_clearance"],
                    "minco_unsafe_ratio": minco_safety["unsafe_ratio"],
                    "minco_path_length_m": minco_geometry["path_length_m"],
                    "actual_speed_mean_mps": temporal_metrics["actual_speed_mean_mps"],
                    "actual_acc_rms_mps2": temporal_metrics["actual_acc_rms_mps2"],
                    "actual_jerk_rms_mps3": temporal_metrics["actual_jerk_rms_mps3"],
                    "actual_yaw_rate_rms_radps": temporal_metrics["actual_yaw_rate_rms_radps"],
                    "proposal_id": best.get("proposal_id"),
                    "hot_start_accepted": bool(best.get("hot_start_accepted", False)),
                    "hot_reject_reason": str(best.get("hot_reject_reason", "")),
                    "history_plan_uid": best.get("history_plan_uid"),
                    "history_age_s": float(best.get("history_age_s", np.nan)),
                    "position_error": float(best.get("position_error", np.nan)),
                    "velocity_error": float(best.get("velocity_error", np.nan)),
                    "direction_dot": float(best.get("direction_dot", np.nan)),
                    "remaining_duration": float(best.get("remaining_duration", np.nan)),
                    "history_min_clearance": float(best.get("history_min_clearance", np.nan)),
                    "shifted_seed_valid": bool(best.get("shifted_seed_valid", False)),
                    "copied_waypoints": int(best.get("copied_waypoints", 0)),
                    "copied_durations": int(best.get("copied_durations", 0)),
                }
                result_metrics_ms = (
                    time.perf_counter() - result_metrics_start
                ) * 1000.0
                elapsed_ms = pre_metrics_elapsed_ms + result_metrics_ms
                adapter_timing = self.aggregate_timing(
                    adapter_total_ms=elapsed_ms,
                    candidate_screen_ms=candidate_screen_ms,
                    candidate_timings=candidate_timings,
                    result_metrics_ms=result_metrics_ms,
                )
                result["adapter_total_ms"] = elapsed_ms
                result["time_ms"] = elapsed_ms
                result["adapter_timing_ms"] = adapter_timing
                result["timing_ms"].update(adapter_timing)
                print(
                    "[NavDP-Minco] "
                    f"env={env_idx} status=MINCO_OK selected_idx={result['selected_index']} "
                    f"selected_rank={result['selected_candidate_rank']} "
                    f"attempted={result['attempted_candidate_count']}/{result['configured_top_k']} "
                    f"planning_state={result['planning_state']} local_end_is_goal={int(result['local_end_is_goal'])} "
                    f"objective={result['objective']:.4f} min_esdf={result['min_esdf']:.4f} "
                    f"py_esdf={result['py_min_esdf']:.4f} "
                    f"raw_path_size={len(raw_top1) if raw_top1 is not None else 0} "
                    f"sparse_waypoint_size={result['sparse_waypoint_size']} "
                    f"mandatory_corner_count={result['mandatory_corner_count']} "
                    f"trajectory_sample_size={len(samples_np) if samples_np is not None else 0} "
                    f"adapter_ms={elapsed_ms:.2f} cpp_ms={result['selected_cpp_optimize_time_ms']:.2f}"
                )
            else:
                reason = "; ".join(failures) if failures else "NO_VALID_CANDIDATE"
                result = self._fallback_result(
                    env_idx,
                    raw_top1_world[env_idx],
                    reason,
                    pre_metrics_elapsed_ms,
                    candidate_timings,
                    candidate_evaluations,
                    candidate_screen_ms,
                )
            results.append(result)
        return results

    def commit_selected(self, result, applied_time):
        proposal_id = result.get("proposal_id") if isinstance(result, dict) else None
        if proposal_id is None or not result.get("success", False):
            return False
        if not hasattr(self.processor, "commit_history"):
            return self.warm_start_mode == "cold"
        return bool(self.processor.commit_history(int(proposal_id), float(applied_time)))

    def discard_proposal(self, result):
        proposal_id = result.get("proposal_id") if isinstance(result, dict) else None
        if proposal_id is not None and hasattr(self.processor, "discard_proposal"):
            self.processor.discard_proposal(int(proposal_id))

    def reset_history(self):
        if self.processor is not None and hasattr(self.processor, "reset_history"):
            self.processor.reset_history()

    def _fallback_result(
        self,
        env_idx,
        raw_top1_world,
        reason,
        elapsed_ms,
        candidate_timings=None,
        candidate_evaluations=None,
        candidate_screen_ms=0.0,
    ):
        result_metrics_start = time.perf_counter()
        waypoints = np.asarray(raw_top1_world, dtype=np.float64)
        raw_xy = waypoints[:, :2] if waypoints.ndim == 2 and waypoints.shape[1] >= 2 else None
        raw_geometry, _ = compute_geometric_metrics(raw_xy)
        raw_safety = self._inspect_esdf(
            raw_xy,
            self.validation_safe_dist,
            self.start_validation_exemption_radius,
        )
        attempted = list(candidate_timings or [])
        last_attempt = attempted[-1] if attempted else {}
        result_metrics_ms = (
            time.perf_counter() - result_metrics_start
        ) * 1000.0
        elapsed_ms = float(elapsed_ms) + result_metrics_ms
        adapter_timing = self.aggregate_timing(
            adapter_total_ms=elapsed_ms,
            candidate_screen_ms=candidate_screen_ms,
            candidate_timings=attempted,
            result_metrics_ms=result_metrics_ms,
        )
        result = {
            "success": False,
            "waypoints": None,
            "samples": None,
            "selected_index": -1,
            "configured_top_k": int(self.max_top_k),
            "attempted_candidate_count": int(len(candidate_timings or [])),
            "attempted_candidate_indices": [
                int(item["selected_index"]) for item in (candidate_timings or [])
            ],
            "selected_candidate_rank": -1,
            "objective": last_attempt.get("objective", float("nan")),
            "min_esdf": float("nan"),
            "py_min_esdf": float("nan"),
            "optimization_safe_dist": self.optimization_safe_dist,
            "validation_safe_dist": self.validation_safe_dist,
            "validation_failure_reason": last_attempt.get(
                "validation_failure_reason", reason
            ),
            "failure_reason": reason,
            "fallback": False,
            "fallback_mode": "HOLD_LAST_OR_STOP",
            "status": "MINCO_FAIL",
            "time_ms": elapsed_ms,
            "sparse_waypoints": None,
            "selected_candidate": None,
            "raw_top1": raw_xy,
            "speed_profile": None,
            "adapter_total_ms": elapsed_ms,
            "candidate_timings": candidate_timings or [],
            "candidate_evaluations": candidate_evaluations or [],
            "selected_cpp_optimize_time_ms": float("nan"),
            "selected_python_call_ms": float("nan"),
            "timing_ms": dict(adapter_timing),
            "adapter_timing_ms": adapter_timing,
            "dense_path_size": 0,
            "sparse_waypoint_size": 0,
            "mandatory_corner_count": 0,
            "planning_state": "COLD_START",
            "local_end_is_goal": False,
            "optimizer_iteration_count": last_attempt.get(
                "optimizer_iteration_count", 0
            ),
            "optimizer_return_code": last_attempt.get("optimizer_return_code", ""),
            "optimizer_success": any(
                bool(item.get("optimizer_success", False)) for item in attempted
            ),
            "validation_min_clearance": last_attempt.get("validation_min_clearance", ""),
            "validation_oob_count": last_attempt.get("validation_oob_count", ""),
            "validation_start_exempt_count": last_attempt.get("validation_start_exempt_count", ""),
            "python_validation_ms": last_attempt.get("python_validation_ms", ""),
            "cpp_validation_success": last_attempt.get(
                "cpp_validation_success", False
            ),
            "python_validation_success": last_attempt.get(
                "python_validation_success", False
            ),
            "raw_min_clearance_m": raw_safety["min_clearance"],
            "raw_unsafe_ratio": raw_safety["unsafe_ratio"],
            "raw_esdf_oob_ratio": raw_safety["oob_ratio"],
            "raw_path_length_m": raw_geometry["path_length_m"],
            "raw_curvature_abs_p95_1pm": raw_geometry["curvature_abs_p95_1pm"],
            "raw_curvature_tv_1pm": raw_geometry["curvature_tv_1pm"],
        }
        print(
            f"[NavDP-Minco] env={env_idx} status=MINCO_FAIL fallback_mode=HOLD_LAST_OR_STOP "
            f"attempted={result['attempted_candidate_count']}/{result['configured_top_k']} "
            f"adapter_ms={elapsed_ms:.2f} reason={reason}"
        )
        return result

    @staticmethod
    def aggregate_timing(
        adapter_total_ms,
        candidate_screen_ms,
        candidate_timings,
        result_metrics_ms=0.0,
    ):
        def finite_value(value, default=0.0):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return float(default)
            return number if np.isfinite(number) else float(default)

        def finite_sum(field):
            return float(sum(
                finite_value(record.get(field, 0.0))
                for record in candidate_timings
            ))

        total_ms = finite_value(adapter_total_ms)
        screen_ms = finite_value(candidate_screen_ms)
        attempt_ms = finite_sum("python_call_ms")
        cpp_ms = finite_sum("cpp_optimize_time_ms")
        validation_ms = finite_sum("python_validation_ms")
        metrics_ms = finite_value(result_metrics_ms)
        accounted_ms = screen_ms + attempt_ms + validation_ms + metrics_ms
        return {
            "candidate_screen_ms": screen_ms,
            "candidate_attempt_total_ms": attempt_ms,
            "candidate_cpp_total_ms": cpp_ms,
            "python_validation_total_ms": validation_ms,
            "result_metrics_ms": metrics_ms,
            "adapter_overhead_ms": max(0.0, total_ms - accounted_ms),
        }

    def _query_min_esdf(self, points):
        return self._esdf_grid.query_polyline(points)

    @staticmethod
    def rank_candidates(
        candidates,
        critic_values,
        esdf_grid,
        validation_safe_dist,
        start_exemption_radius=0.0,
        start_position=None,
    ):
        candidates = list(candidates)
        critic_values = np.asarray(critic_values, dtype=np.float64).reshape(-1)
        diagnostics = []
        for index, candidate in enumerate(candidates):
            screened = preprocess_guide_path(candidate)
            critic = (
                float(critic_values[index])
                if index < len(critic_values) and np.isfinite(critic_values[index])
                else float("-inf")
            )
            if not screened["valid"]:
                diagnostics.append({
                    "candidate_index": index,
                    "critic_value": critic,
                    "screen_valid": False,
                    "screen_safe": False,
                    "screen_reason": screened["reason"],
                    "min_esdf": float("nan"),
                    "unsafe_ratio": 1.0,
                    "oob_ratio": 1.0,
                    "path_length_m": float("nan"),
                    "safety_tier": 4,
                })
                continue
            path = screened["path"]
            if start_position is not None:
                start = np.asarray(start_position, dtype=np.float64).reshape(-1)
                if start.size >= 2 and np.all(np.isfinite(start[:2])):
                    start_xyz = np.array([start[0], start[1], 0.0], dtype=np.float64)
                    if np.linalg.norm(path[0, :2] - start_xyz[:2]) > 1e-4:
                        path = np.vstack((start_xyz, path))
            report = esdf_grid.inspect_polyline(
                path[:, :2],
                safe_dist=validation_safe_dist,
                start_exemption_radius=start_exemption_radius,
            )
            length = float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())
            geometry, _ = compute_geometric_metrics(path[:, :2])
            min_esdf = float(report["min_clearance"])
            if not report["valid"]:
                tier = 3
            elif np.isfinite(min_esdf) and min_esdf < 0.0:
                tier = 2
            elif not report["safe"]:
                tier = 1
            else:
                tier = 0
            diagnostics.append({
                "candidate_index": index,
                "critic_value": critic,
                "screen_valid": True,
                "screen_safe": bool(report["safe"]),
                "screen_reason": report["reason"],
                "min_esdf": min_esdf,
                "unsafe_ratio": float(report["unsafe_ratio"]),
                "oob_ratio": float(report["oob_ratio"]),
                "path_length_m": length,
                "curvature_tv_1pm": geometry["curvature_tv_1pm"],
                "safety_tier": tier,
            })

        def sort_key(index):
            item = diagnostics[index]
            critic = item["critic_value"]
            clearance = item["min_esdf"]
            if not np.isfinite(clearance):
                clearance = float("-inf")
            if item["safety_tier"] == 0:
                return (0, -critic, -clearance, index)
            return (item["safety_tier"], -clearance, -critic, index)

        order = sorted(range(len(diagnostics)), key=sort_key)
        for rank, index in enumerate(order):
            diagnostics[index]["screen_rank"] = rank
        return order, diagnostics

    def _inspect_esdf(self, points, safe_dist=None, start_exemption_radius=None):
        return self._esdf_grid.inspect_polyline(
            points,
            safe_dist=self.validation_safe_dist if safe_dist is None else safe_dist,
            start_exemption_radius=(
                self.start_validation_exemption_radius
                if start_exemption_radius is None
                else start_exemption_radius
            ),
        )

    @staticmethod
    def _as_guide_path(candidate):
        candidate = np.asarray(candidate, dtype=np.float64)
        if candidate.ndim != 2 or candidate.shape[0] < 2 or candidate.shape[1] < 2:
            return None
        if not np.all(np.isfinite(candidate)):
            return None
        if candidate.shape[1] == 2:
            zeros = np.zeros((candidate.shape[0], 1), dtype=np.float64)
            candidate = np.concatenate([candidate, zeros], axis=1)
        else:
            candidate = candidate[:, :3]
        candidate[:, 2] = 0.0
        length = np.linalg.norm(np.diff(candidate[:, :2], axis=0), axis=1).sum()
        if not np.isfinite(length) or length < 1e-3:
            return None
        return candidate

    @staticmethod
    def _as_terminal_goal(goal):
        if goal is None:
            return None
        goal = np.asarray(goal, dtype=np.float64).reshape(-1)
        if goal.size < 3 or not np.all(np.isfinite(goal[:3])):
            return None
        out = goal[:3].copy()
        out[2] = 0.0
        return out

    @staticmethod
    def _is_better(candidate, current):
        candidate_score = (float(candidate.get("objective", np.inf)), -float(candidate.get("min_esdf", -np.inf)))
        current_score = (float(current.get("objective", np.inf)), -float(current.get("min_esdf", -np.inf)))
        return candidate_score < current_score
