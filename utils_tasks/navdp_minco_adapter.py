import time

import numpy as np

from utils_tasks.esdf_query_utils import EsdfGridView


class NavDPMincoAdapter:
    def __init__(
        self,
        esdf: dict,
        safe_dist=0.30,
        top_k=4,
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
        self.safe_dist = float(safe_dist)
        self.top_k = max(1, int(top_k))
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
                    safe_dist=self.safe_dist,
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
            order = self._candidate_order(critic_values[env_idx], len(candidates_world[env_idx]))
            candidate_indices = order[:min(self.top_k, len(order))]
            for rank, selected_idx in enumerate(candidate_indices):
                candidate = self._as_guide_path(candidates_world[env_idx][selected_idx])
                if candidate is None:
                    failures.append(f"idx={selected_idx}: invalid_candidate")
                    candidate_timings.append({
                        "candidate_rank": int(rank),
                        "selected_index": int(selected_idx),
                        "python_call_ms": 0.0,
                        "cpp_optimize_time_ms": float("nan"),
                        "success": False,
                        "objective": float("inf"),
                        "min_esdf": float("nan"),
                        "failure_reason": "invalid_candidate",
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
                        "min_esdf": float("nan"),
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
                    "mandatory_corner_count": int(result.get("mandatory_corner_count", 0)),
                    "planning_state": str(result.get("planning_state", "")),
                    "local_end_is_goal": bool(result.get("local_end_is_goal", False)),
                    "success": bool(result.get("success", False)),
                    "objective": float(result.get("objective", np.inf)),
                    "min_esdf": float(result.get("min_esdf", np.nan)),
                    "failure_reason": str(result.get("failure_reason", "")),
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
                py_min_esdf = self._query_min_esdf(waypoints)
                if not np.isfinite(py_min_esdf) or py_min_esdf <= self.safe_dist:
                    if result.get("proposal_id") is not None and hasattr(self.processor, "discard_proposal"):
                        self.processor.discard_proposal(int(result["proposal_id"]))
                    unsafe_reason = (
                        f"PY_ESDF_UNSAFE py_min={py_min_esdf:.3f} safe={self.safe_dist:.3f}"
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
                best = scored
                break

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if best is not None:
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

                result = {
                    "success": True,
                    "waypoints": np.asarray(best["waypoints"], dtype=np.float64)[:, :2],
                    "samples": samples_np,
                    "selected_index": int(best["selected_index"]),
                    "configured_top_k": int(self.top_k),
                    "attempted_candidate_count": int(len(candidate_timings)),
                    "attempted_candidate_indices": [
                        int(item["selected_index"]) for item in candidate_timings
                    ],
                    "selected_candidate_rank": int(best["selected_candidate_rank"]),
                    "objective": float(best.get("objective", np.inf)),
                    "min_esdf": float(best.get("min_esdf", np.nan)),
                    "py_min_esdf": float(best.get("py_min_esdf", np.nan)),
                    "safe_dist": self.safe_dist,
                    "failure_reason": best.get("failure_reason", "NONE"),
                    "fallback": False,
                    "fallback_mode": "NONE",
                    "status": "MINCO_OK",
                    "time_ms": elapsed_ms,
                    "sparse_waypoints": sparse_waypoints,
                    "selected_candidate": selected_candidate,
                    "raw_top1": raw_top1,
                    "speed_profile": speed_profile,
                    "adapter_total_ms": elapsed_ms,
                    "candidate_timings": candidate_timings,
                    "selected_cpp_optimize_time_ms": float(best.get("cpp_optimize_time_ms", best.get("duration", 0.0) * 1000.0)),
                    "selected_python_call_ms": float(best.get("python_call_ms", np.nan)),
                    "timing_ms": dict(best.get("timing_ms", {})),
                    "dense_path_size": int(best.get("dense_path_size", 0)),
                    "sparse_waypoint_size": int(best.get("sparse_waypoint_size", 0)),
                    "mandatory_corner_count": int(best.get("mandatory_corner_count", 0)),
                    "planning_state": str(best.get("planning_state", "")),
                    "local_end_is_goal": bool(best.get("local_end_is_goal", False)),
                    "optimizer_iteration_count": int(best.get("optimizer_iteration_count", 0)),
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
                result = self._fallback_result(env_idx, raw_top1_world[env_idx], reason, elapsed_ms, candidate_timings)
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

    def _fallback_result(self, env_idx, raw_top1_world, reason, elapsed_ms, candidate_timings=None):
        waypoints = np.asarray(raw_top1_world, dtype=np.float64)
        result = {
            "success": False,
            "waypoints": None,
            "samples": None,
            "selected_index": -1,
            "configured_top_k": int(self.top_k),
            "attempted_candidate_count": int(len(candidate_timings or [])),
            "attempted_candidate_indices": [
                int(item["selected_index"]) for item in (candidate_timings or [])
            ],
            "selected_candidate_rank": -1,
            "objective": float("inf"),
            "min_esdf": float("nan"),
            "py_min_esdf": float("nan"),
            "safe_dist": self.safe_dist,
            "failure_reason": reason,
            "fallback": False,
            "fallback_mode": "HOLD_LAST_OR_STOP",
            "status": "MINCO_FAIL",
            "time_ms": elapsed_ms,
            "sparse_waypoints": None,
            "selected_candidate": None,
            "raw_top1": waypoints[:, :2] if waypoints.ndim == 2 and waypoints.shape[1] >= 2 else None,
            "speed_profile": None,
            "adapter_total_ms": elapsed_ms,
            "candidate_timings": candidate_timings or [],
            "selected_cpp_optimize_time_ms": float("nan"),
            "selected_python_call_ms": float("nan"),
            "timing_ms": {},
            "dense_path_size": 0,
            "sparse_waypoint_size": 0,
            "mandatory_corner_count": 0,
            "planning_state": "COLD_START",
            "local_end_is_goal": False,
            "optimizer_iteration_count": 0,
        }
        print(
            f"[NavDP-Minco] env={env_idx} status=MINCO_FAIL fallback_mode=HOLD_LAST_OR_STOP "
            f"attempted={result['attempted_candidate_count']}/{result['configured_top_k']} "
            f"adapter_ms={elapsed_ms:.2f} reason={reason}"
        )
        return result

    def _query_min_esdf(self, points):
        return self._esdf_grid.query_polyline(points)

    @staticmethod
    def _candidate_order(values, candidate_count):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        count = min(candidate_count, values.size)
        if count == 0:
            return []
        safe_values = values[:count].copy()
        safe_values[~np.isfinite(safe_values)] = -np.inf
        return list(np.argsort(-safe_values))

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
