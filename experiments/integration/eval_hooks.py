from __future__ import annotations

import time
import math
import json
import numpy as np


def _finite_or_blank(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


class ExperimentHookBridge:
    """Fail-small monitor bridge; it observes events and never changes commands."""
    def __init__(self, sink, identity):
        self.sink = sink; self.identity = dict(identity); self.episode_uid = None; self.generation = -1; self._cycle = 0; self._frame = 0
        self._cycle_rows = []; self._tracking_errors = []; self._episode_started = None

    def _base(self): return {**self.identity, "episode_uid":self.episode_uid, "data_source":"REAL"}

    def start_episode(self, episode_uid, generation, initial_goal_distance_m=None):
        self.episode_uid = str(episode_uid); self.generation = int(generation); self._cycle = 0; self._frame = 0
        self._cycle_rows = []; self._tracking_errors = []; self._episode_started = time.monotonic()
        self._initial_goal_distance_m = float(initial_goal_distance_m) if initial_goal_distance_m is not None else None
        self.sink.submit_csv("events", {**self._base(), "timestamp_monotonic_s":time.monotonic(), "frame_idx":0, "plan_uid":"", "event_type":"EPISODE_START", "severity":"INFO", "primary_reason":"", "secondary_reason":"", "message":f"generation={generation}"})

    def record_planning_cycle(self, cycle_uid, published, stale, fallback_mode, failure_reason="", **fields):
        attempted_indices = fields.get("attempted_candidate_indices", [])
        if not isinstance(attempted_indices, str):
            attempted_indices = json.dumps([int(value) for value in attempted_indices])
        observation_timestamp = fields.get("observation_timestamp_s", "")
        row = {**self._base(), "episode_generation":self.generation, "planning_cycle_uid":cycle_uid, "trigger_timestamp_s":fields.get("trigger_timestamp_s", observation_timestamp if observation_timestamp != "" else time.monotonic()), "observation_sequence":fields.get("observation_sequence", ""), "observation_timestamp_s":observation_timestamp, "planning_started_timestamp_s":fields.get("planning_started_timestamp_s", ""), "published_timestamp_s":fields.get("published_timestamp_s", ""), "observation_to_plan_ms":fields.get("observation_to_plan_ms", ""), "planning_deadline_ms":fields.get("planning_deadline_ms", ""), "planning_deadline_miss":fields.get("planning_deadline_miss", ""), "input_snapshot_ms":fields.get("input_snapshot_ms", ""), "raw_diagnostics_ms":fields.get("raw_diagnostics_ms", ""), "candidate_transform_ms":fields.get("candidate_transform_ms", ""), "threshold_profile_id":fields.get("threshold_profile_id", ""), "raw_available":fields.get("raw_available", True), "candidate_count":fields.get("candidate_count", 0), "screened_candidate_count":fields.get("screened_candidate_count", 0), "rejected_candidate_count":fields.get("rejected_candidate_count", 0), "attempted_candidate_count":fields.get("attempted_candidate_count", 0), "attempted_candidate_indices":attempted_indices, "selected_candidate_index":fields.get("selected_candidate_index", ""), "optimizer_success":fields.get("optimizer_success", ""), "optimizer_return_code":fields.get("optimizer_return_code", ""), "optimizer_iteration_count":fields.get("optimizer_iteration_count", ""), "objective":fields.get("objective", ""), "cpp_validation_success":fields.get("cpp_validation_success", ""), "cpp_validation_min_clearance_m":fields.get("cpp_validation_min_clearance_m", ""), "python_validation_success":fields.get("python_validation_success", ""), "python_validation_min_clearance_m":fields.get("python_validation_min_clearance_m", ""), "validation_start_exempt_count":fields.get("validation_start_exempt_count", ""), "validation_oob_count":fields.get("validation_oob_count", ""), "validation_failure_reason":fields.get("validation_failure_reason", ""), "candidate_screen_ms":fields.get("candidate_screen_ms", ""), "candidate_attempt_total_ms":fields.get("candidate_attempt_total_ms", ""), "candidate_cpp_total_ms":fields.get("candidate_cpp_total_ms", ""), "python_validation_total_ms":fields.get("python_validation_total_ms", ""), "adapter_overhead_ms":fields.get("adapter_overhead_ms", ""), "stale":bool(stale), "published":bool(published), "fallback_mode":fallback_mode, "failure_reason":failure_reason, "navdp_ms":fields.get("navdp_ms", ""), "minco_ms":fields.get("minco_ms", ""), "validation_ms":fields.get("validation_ms", ""), "planning_total_ms":fields.get("planning_total_ms", ""), "plan_age_when_applied_ms":fields.get("plan_age_when_applied_ms", "")}
        for field in (
            "optimizer_return_code", "optimizer_iteration_count", "objective",
            "cpp_validation_min_clearance_m", "python_validation_min_clearance_m",
            "validation_start_exempt_count", "validation_oob_count", "navdp_ms",
            "candidate_screen_ms", "candidate_attempt_total_ms",
            "candidate_cpp_total_ms", "python_validation_total_ms",
            "adapter_overhead_ms", "minco_ms", "validation_ms", "planning_total_ms",
            "plan_age_when_applied_ms", "observation_sequence",
            "observation_timestamp_s", "planning_started_timestamp_s",
            "published_timestamp_s", "observation_to_plan_ms",
            "planning_deadline_ms", "input_snapshot_ms", "raw_diagnostics_ms",
            "candidate_transform_ms",
        ):
            row[field] = _finite_or_blank(row[field])
        self.sink.submit_csv("planning_cycles", row)
        self._cycle_rows.append({**row, "planning_state":fields.get("planning_state", ""), "hot_start_accepted":fields.get("hot_start_accepted", False)})
        self._cycle += 1

    def record_plan(self, plan_uid, published, stale, **fields):
        if not published or stale: return False
        columns = (
            "threshold_profile_id", "hot_wrong_accept",
            "raw_safety_class", "turn_class", "temporal_class",
            "raw_min_clearance_m", "raw_unsafe_ratio", "raw_esdf_oob_ratio",
            "raw_path_length_m", "raw_curvature_abs_p95_1pm", "raw_curvature_tv_1pm",
            "raw_curvature_rate_rms_1pm2",
            "raw_interplan_position_rmse_m", "raw_initial_tangent_jump_rad",
            "minco_min_clearance_m", "minco_unsafe_ratio", "minco_path_length_m",
            "actual_speed_mean_mps", "actual_speed_p95_mps", "actual_speed_max_mps",
            "actual_acc_rms_mps2", "actual_acc_p95_mps2", "actual_acc_max_mps2",
            "actual_jerk_rms_mps3", "actual_jerk_p95_mps3", "actual_jerk_max_mps3",
            "actual_yaw_rate_rms_radps", "actual_yaw_rate_max_radps", "trajectory_duration_s",
            "planning_state", "hot_start_accepted",
            "hot_reject_reason", "history_age_s", "position_error_m",
            "velocity_error_mps", "direction_dot", "remaining_duration_s",
            "history_min_clearance_m", "shifted_seed_valid", "copied_waypoints",
            "copied_durations", "optimizer_success", "python_validation_success",
            "optimizer_return_code", "optimizer_iteration_count", "objective",
            "cpp_validation_success", "cpp_validation_min_clearance_m",
            "python_validation_min_clearance_m", "validation_start_exempt_count",
            "validation_oob_count", "failure_reason", "planning_total_ms",
        )
        row = {
            **self._base(), "plan_uid":plan_uid,
            "timestamp_monotonic_s":time.monotonic(),
            "plan_status":fields.get("plan_status", "PUBLISHED"),
            "fallback_mode":fields.get("fallback_mode", "NONE"),
        }
        row.update({column: fields.get(column, "") for column in columns})
        for column in (
            "raw_min_clearance_m", "raw_unsafe_ratio", "raw_esdf_oob_ratio",
            "raw_path_length_m", "raw_curvature_abs_p95_1pm", "raw_curvature_tv_1pm",
            "raw_curvature_rate_rms_1pm2",
            "raw_interplan_position_rmse_m", "raw_initial_tangent_jump_rad",
            "minco_min_clearance_m", "minco_unsafe_ratio", "minco_path_length_m",
            "actual_speed_mean_mps", "actual_speed_p95_mps", "actual_speed_max_mps",
            "actual_acc_rms_mps2", "actual_acc_p95_mps2", "actual_acc_max_mps2",
            "actual_jerk_rms_mps3", "actual_jerk_p95_mps3", "actual_jerk_max_mps3",
            "actual_yaw_rate_rms_radps", "actual_yaw_rate_max_radps", "trajectory_duration_s",
            "optimizer_return_code",
            "optimizer_iteration_count", "objective", "cpp_validation_min_clearance_m",
            "python_validation_min_clearance_m", "validation_start_exempt_count",
            "validation_oob_count", "planning_total_ms",
            "history_age_s", "position_error_m", "velocity_error_mps",
            "direction_dot", "remaining_duration_s", "history_min_clearance_m",
            "copied_waypoints", "copied_durations",
        ):
            row[column] = _finite_or_blank(row[column])
        self.sink.submit_csv("plan_metrics", row)
        return True

    def record_candidates(
        self,
        planning_cycle_uid,
        plan_uid,
        critic_values,
        selected_index=-1,
        candidate_records=None,
        attempted_records=None,
    ):
        critic_values = list(critic_values)
        critic_order = sorted(
            range(len(critic_values)),
            key=lambda index: (
                -float(critic_values[index])
                if math.isfinite(float(critic_values[index]))
                else float("inf"),
                index,
            ),
        )
        critic_ranks = {index: rank for rank, index in enumerate(critic_order)}
        candidate_by_index = {
            int(record.get("candidate_index", index)): record
            for index, record in enumerate(candidate_records or [])
        }
        attempt_by_index = {
            int(record["selected_index"]): record
            for record in (attempted_records or [])
            if "selected_index" in record
        }
        for index, value in enumerate(critic_values):
            candidate = candidate_by_index.get(index, {})
            attempt = attempt_by_index.get(index, {})
            attempt_stages = attempt.get("timing_ms", {}) or {}
            self.sink.submit_csv("candidate_metrics", {
                **self._base(), "planning_cycle_uid":planning_cycle_uid,
                "plan_uid":plan_uid, "candidate_index":index,
                "candidate_rank":candidate.get("screen_rank", critic_ranks[index]),
                "critic_rank":critic_ranks[index],
                "screen_rank":candidate.get("screen_rank", ""),
                "attempted_rank":attempt.get("candidate_rank", ""),
                "critic_value":_finite_or_blank(value),
                "screen_valid":candidate.get("screen_valid", ""),
                "screen_safe":candidate.get("screen_safe", ""),
                "screen_reason":candidate.get("screen_reason", ""),
                "attempted":index in attempt_by_index,
                "optimizer_success":attempt.get(
                    "optimizer_success", attempt.get("success", "")
                ),
                "optimizer_return_code":attempt.get("optimizer_return_code", ""),
                "optimizer_iteration_count":attempt.get("optimizer_iteration_count", ""),
                "objective":_finite_or_blank(attempt.get("objective", "")),
                "failure_reason":attempt.get("failure_reason", ""),
                "candidate_call_ms":_finite_or_blank(attempt.get("python_call_ms", "")),
                "cpp_pipeline_ms":_finite_or_blank(attempt.get("cpp_optimize_time_ms", "")),
                "optimizer_ms":_finite_or_blank(attempt_stages.get("optimizer_ms", "")),
                "cpp_validation_ms":_finite_or_blank(attempt_stages.get("validate_ms", "")),
                "python_validation_ms":_finite_or_blank(attempt.get("python_validation_ms", "")),
                "path_length_m":_finite_or_blank(candidate.get("path_length_m", "")),
                "min_clearance_m":_finite_or_blank(candidate.get("min_esdf", "")),
                "python_min_clearance_m":_finite_or_blank(attempt.get("python_min_esdf", "")),
                "unsafe_ratio":_finite_or_blank(candidate.get("unsafe_ratio", "")),
                "esdf_oob_ratio":_finite_or_blank(candidate.get("oob_ratio", "")),
                "curvature_tv_1pm":_finite_or_blank(candidate.get("curvature_tv_1pm", "")),
                "selected":index == int(selected_index),
            })

    def record_event(self, event_type, severity="INFO", primary_reason="", secondary_reason="", message="", plan_uid="", frame_idx=-1):
        self.sink.submit_csv("events", {
            **self._base(), "timestamp_monotonic_s":time.monotonic(),
            "frame_idx":frame_idx, "plan_uid":plan_uid, "event_type":event_type,
            "severity":severity, "primary_reason":primary_reason,
            "secondary_reason":secondary_reason, "message":message,
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
            "observation_sequence":fields.get("observation_sequence", ""),
            "observation_timestamp_s":_finite_or_blank(fields.get("observation_timestamp_s", "")),
            "plan_published_timestamp_s":_finite_or_blank(fields.get("plan_published_timestamp_s", "")),
            "observation_to_command_ms":_finite_or_blank(fields.get("observation_to_command_ms", "")),
            "control_loop_ms":_finite_or_blank(fields.get("control_loop_ms", "")),
            "video_write_ms":_finite_or_blank(fields.get("video_write_ms", "")),
            "control_deadline_ms":_finite_or_blank(fields.get("control_deadline_ms", "")),
            "control_deadline_miss":fields.get("control_deadline_miss", ""),
            "robot_x_m":fields.get("robot_x_m", ""), "robot_y_m":fields.get("robot_y_m", ""),
            "robot_yaw_rad":fields.get("robot_yaw_rad", ""), "actual_v_mps":fields.get("actual_v_mps", ""),
            "actual_w_radps":fields.get("actual_w_radps", ""), "reference_x_m":fields.get("reference_x_m", ""),
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
            "termination_term_raw":fields.get("termination_term_raw", ""),
            "failure_stage":fields.get("failure_stage", ""),
            "failure_reason":fields.get("failure_reason", ""),
            "episode_duration_s":duration, "actual_path_length_m":fields.get("actual_path_length_m", ""),
            "repository_spl":fields.get("repository_spl", ""),
            "tracking_error_rmse_m":tracking_rmse,
            "tracking_error_p95_m": float(np.percentile(self._tracking_errors, 95)) if self._tracking_errors else "",
            "initial_goal_distance_m": getattr(self, '_initial_goal_distance_m', None) or "",
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
