from __future__ import annotations

import time
import math


class ExperimentHookBridge:
    """Fail-small monitor bridge; it observes events and never changes commands."""
    def __init__(self, sink, identity):
        self.sink = sink; self.identity = dict(identity); self.episode_uid = None; self.generation = -1; self._cycle = 0; self._frame = 0
        self._cycle_rows = []; self._tracking_errors = []; self._episode_started = None

    def _base(self): return {**self.identity, "episode_uid":self.episode_uid, "data_source":"REAL"}

    def start_episode(self, episode_uid, generation):
        self.episode_uid = str(episode_uid); self.generation = int(generation); self._cycle = 0; self._frame = 0
        self._cycle_rows = []; self._tracking_errors = []; self._episode_started = time.monotonic()
        self.sink.submit_csv("events", {**self._base(), "timestamp_monotonic_s":time.monotonic(), "frame_idx":0, "plan_uid":"", "event_type":"EPISODE_START", "severity":"INFO", "primary_reason":"", "secondary_reason":"", "message":f"generation={generation}"})

    def record_planning_cycle(self, cycle_uid, published, stale, fallback_mode, failure_reason="", **fields):
        row = {**self._base(), "episode_generation":self.generation, "planning_cycle_uid":cycle_uid, "trigger_timestamp_s":time.monotonic(), "raw_available":fields.get("raw_available", True), "candidate_count":fields.get("candidate_count", 0), "screened_candidate_count":fields.get("screened_candidate_count", 0), "rejected_candidate_count":fields.get("rejected_candidate_count", 0), "attempted_candidate_count":fields.get("attempted_candidate_count", 0), "selected_candidate_index":fields.get("selected_candidate_index", ""), "optimizer_success":fields.get("optimizer_success", ""), "cpp_validation_success":fields.get("cpp_validation_success", ""), "python_validation_success":fields.get("python_validation_success", ""), "validation_failure_reason":fields.get("validation_failure_reason", ""), "stale":bool(stale), "published":bool(published), "fallback_mode":fallback_mode, "failure_reason":failure_reason, "navdp_ms":fields.get("navdp_ms", ""), "minco_ms":fields.get("minco_ms", ""), "validation_ms":fields.get("validation_ms", ""), "planning_total_ms":fields.get("planning_total_ms", ""), "plan_age_when_applied_ms":fields.get("plan_age_when_applied_ms", "")}
        self.sink.submit_csv("planning_cycles", row)
        self._cycle_rows.append({**row, "planning_state":fields.get("planning_state", ""), "hot_start_accepted":fields.get("hot_start_accepted", False)})
        self._cycle += 1

    def record_plan(self, plan_uid, published, stale, **fields):
        if not published or stale: return False
        columns = (
            "raw_safety_class", "turn_class", "temporal_class",
            "raw_min_clearance_m", "raw_unsafe_ratio", "raw_esdf_oob_ratio",
            "raw_path_length_m", "raw_curvature_abs_p95_1pm", "raw_curvature_tv_1pm",
            "raw_interplan_position_rmse_m", "raw_initial_tangent_jump_rad",
            "minco_min_clearance_m", "minco_unsafe_ratio", "minco_path_length_m",
            "actual_speed_mean_mps", "actual_acc_rms_mps2", "actual_jerk_rms_mps3",
            "actual_yaw_rate_rms_radps", "planning_state", "hot_start_accepted",
            "hot_reject_reason", "optimizer_success", "python_validation_success",
            "failure_reason", "optimizer_iteration_count", "planning_total_ms",
        )
        row = {
            **self._base(), "plan_uid":plan_uid,
            "timestamp_monotonic_s":time.monotonic(),
            "plan_status":fields.get("plan_status", "PUBLISHED"),
            "fallback_mode":fields.get("fallback_mode", "NONE"),
        }
        row.update({column: fields.get(column, "") for column in columns})
        self.sink.submit_csv("plan_metrics", row)
        return True

    def record_candidates(self, plan_uid, critic_values, selected_index=-1):
        for index, value in enumerate(critic_values):
            self.sink.submit_csv("candidate_metrics", {
                **self._base(), "plan_uid":plan_uid, "candidate_index":index,
                "candidate_rank":index + 1, "critic_value":float(value),
                "selected":index == int(selected_index),
            })

    def record_timing(self, metric_name, duration_ms, plan_uid="", frame_idx=-1, status="OK", event_type="STAGE"):
        self.sink.submit_csv("timing_samples", {
            **self._base(), "event_type":event_type, "plan_uid":plan_uid,
            "frame_idx":frame_idx, "timestamp_monotonic_s":time.monotonic(),
            "metric_name":metric_name, "duration_ms":float(duration_ms), "status":status,
        })

    def record_control_step(self, frame_idx, plan_uid, cmd_v, cmd_w, **fields):
        error = fields.get("time_aligned_position_error_m", "")
        try:
            error_value = float(error)
            if math.isfinite(error_value): self._tracking_errors.append(error_value)
        except (TypeError, ValueError): pass
        self.sink.submit_csv("control_samples", {
            **self._base(), "frame_idx":frame_idx, "plan_uid":plan_uid,
            "timestamp_monotonic_s":time.monotonic(), "control_state":fields.get("control_state", "TRACK"),
            "robot_x_m":fields.get("robot_x_m", ""), "robot_y_m":fields.get("robot_y_m", ""),
            "robot_yaw_rad":fields.get("robot_yaw_rad", ""), "reference_x_m":fields.get("reference_x_m", ""),
            "reference_y_m":fields.get("reference_y_m", ""), "planned_v_mps":fields.get("planned_v_mps", ""),
            "planned_w_radps":fields.get("planned_w_radps", ""), "cmd_v_mps":cmd_v, "cmd_w_radps":cmd_w,
            "zero_command_reason":fields.get("zero_command_reason", ""),
            "expected_motion_zero":fields.get("expected_motion_zero", False),
            "expected_motion_zero_streak":fields.get("expected_motion_zero_streak", 0),
            "mpc_solver_status":fields.get("mpc_solver_status", ""),
            "mpc_recovery_action":fields.get("mpc_recovery_action", ""),
            "cross_track_error_m":fields.get("cross_track_error_m", ""),
            "time_aligned_position_error_m":error, "mpc_success":fields.get("mpc_success", ""),
            "mpc_solve_ms":fields.get("mpc_solve_ms", ""), "reference_age_ms":fields.get("reference_age_ms", ""),
        }); self._frame += 1

    def end_episode(self, success=False, **fields):
        tracking_rmse = math.sqrt(sum(value * value for value in self._tracking_errors) / len(self._tracking_errors)) if self._tracking_errors else ""
        duration = fields.get("episode_duration_s", time.monotonic() - self._episode_started if self._episode_started is not None else "")
        self.sink.submit_csv("episode_metrics", {
            **self._base(), "episode_index":fields.get("episode_index", 0), "success":bool(success),
            "collision":fields.get("collision", False), "timeout":fields.get("timeout", False),
            "done_reason":fields.get("done_reason", "GOAL_REACHED" if success else "UNKNOWN"),
            "episode_duration_s":duration, "actual_path_length_m":fields.get("actual_path_length_m", ""),
            "repository_spl":fields.get("repository_spl", ""), "tracking_error_rmse_m":tracking_rmse,
            "minimum_executed_clearance_m":fields.get("minimum_executed_clearance_m", ""),
            "planning_count":self._cycle,
            "minco_ok_count":sum(bool(row.get("published")) for row in self._cycle_rows) if self.identity.get("variant") != "raw" else 0,
            "validation_failure_count":sum(str(row.get("python_validation_success")).lower() == "false" for row in self._cycle_rows),
            "hold_count":sum(row.get("fallback_mode") == "HOLD_LAST" for row in self._cycle_rows),
            "stop_count":sum(row.get("fallback_mode") == "STOP" for row in self._cycle_rows),
            "hot_start_count":sum(bool(row.get("hot_start_accepted")) for row in self._cycle_rows),
            "cold_start_count":sum(row.get("planning_state") == "COLD_START" for row in self._cycle_rows),
        })

    def reset(self, generation):
        self.episode_uid = None; self.generation = int(generation); self._cycle = 0; self._frame = 0
        self._cycle_rows = []; self._tracking_errors = []; self._episode_started = None
