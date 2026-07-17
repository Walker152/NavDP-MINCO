from __future__ import annotations

import math
import numpy as np

from experiments.analyzers.metrics import compute_geometric_metrics
from experiments.recorders.trace_writer import PlanningTraceWriter


def _identity(run, episode):
    return {"suite_id": run.suite_id, "experiment_id": run.experiment_id, "run_id": run.run_id, "variant": run.variant, "scene_label": run.scene_label, "scene_id": run.scene_id, "seed": run.seed, "episode_uid": episode.episode_uid}


class MockBackend:
    """Deterministic, CPU-only contract simulator for tooling tests."""
    name = "mock"

    def run(self, run, episodes, writer):
        trace_writer = PlanningTraceWriter(writer.output_dir / "traces")
        for episode in episodes:
            base = _identity(run, episode); start = np.asarray(episode.start_pose[:2], float); goal = np.asarray(episode.goal_pose[:2], float)
            alpha = np.linspace(0, 1, 21); path = start + alpha[:, None] * (goal - start)
            if episode.scenario_id in {"SCN-02", "SCN-03", "SCN-08"}: path[:, 1] += 0.3 * np.sin(alpha * math.pi)
            geometry, _ = compute_geometric_metrics(path, 0.05)
            plan_uid = f"{episode.episode_uid}_{run.variant}_p000"
            cycle_uid = f"{episode.episode_uid}_{run.variant}_c000"
            raw_clearance = 0.42 if episode.scenario_id in {"SCN-03", "SCN-06"} else (0.7 if run.scene_label == "DENSE" else 1.2)
            minco_clearance = raw_clearance + (0.25 if run.variant != "raw" else 0)
            failure = episode.scenario_id == "SCN-06" and run.variant != "raw"
            fallback = "HOLD_LAST" if failure and run.variant == "minco-hot" else "STOP" if failure else "NONE"
            published = not failure
            attempted_count = (
                0 if run.variant == "raw"
                else 1 + int(episode.episode_index) % 2
            )
            candidate_screen_ms = (
                "" if run.variant == "raw"
                else 0.8 + 0.05 * int(episode.episode_index)
            )
            candidate_attempt_ms = (
                "" if run.variant == "raw" else 2.0 + attempted_count
            )
            candidate_cpp_ms = (
                "" if run.variant == "raw" else 1.5 + attempted_count
            )
            python_validation_ms = (
                "" if run.variant == "raw"
                else 0.25 + 0.02 * int(episode.episode_index)
            )
            adapter_overhead_ms = "" if run.variant == "raw" else 0.4
            hot_accepted = run.variant == "minco-hot" and episode.scenario_id == "SCN-07"
            hot_reject = "HOT_REJECT_DIRECTION" if run.variant == "minco-hot" and episode.scenario_id == "SCN-08" else ("" if hot_accepted else "COLD_FORCED")
            writer.submit_csv("planning_cycles", {**base, "episode_generation":0, "planning_cycle_uid":cycle_uid, "trigger_timestamp_s":.1, "raw_available":True, "candidate_count":2, "attempted_candidate_count":attempted_count, "selected_candidate_index":0 if published else "", "optimizer_success":False if failure else ("" if run.variant == "raw" else True), "cpp_validation_success":False if failure else ("" if run.variant == "raw" else True), "python_validation_success":False if failure else ("" if run.variant == "raw" else True), "candidate_screen_ms":candidate_screen_ms, "candidate_attempt_total_ms":candidate_attempt_ms, "candidate_cpp_total_ms":candidate_cpp_ms, "python_validation_total_ms":python_validation_ms, "adapter_overhead_ms":adapter_overhead_ms, "stale":False, "published":published, "fallback_mode":fallback, "failure_reason":"MINCO_VALIDATION_FAILED" if failure else "", "navdp_ms":8.0, "minco_ms":0.0 if run.variant == "raw" else 6.0 + attempted_count, "validation_ms":0.5, "planning_total_ms":12.0 if run.variant == "raw" else 18.0 + attempted_count, "plan_age_when_applied_ms":5.0 if published else "", "data_source":"SIMULATED"})
            if published:
                writer.submit_csv("plan_metrics", {**base, "plan_uid": plan_uid, "timestamp_monotonic_s": 0.1, "plan_status": "RAW_OK" if run.variant == "raw" else "MINCO_OK", "fallback_mode": "NONE", "raw_safety_class": "RAW_UNSAFE" if raw_clearance <= .6 else "RAW_SAFE", "turn_class": "HIGH_TURN" if episode.scenario_id in {"SCN-02", "SCN-04"} else "LOW_TURN", "temporal_class": "STABLE_INPUT" if episode.scenario_id == "SCN-07" else "JUMP_INPUT" if episode.scenario_id == "SCN-08" else "NO_HISTORY", "raw_min_clearance_m": raw_clearance, "raw_unsafe_ratio": float(raw_clearance <= .6), "raw_esdf_oob_ratio": 0.0, "raw_path_length_m": geometry["path_length_m"], "raw_curvature_abs_p95_1pm": geometry["curvature_abs_p95_1pm"], "raw_curvature_tv_1pm": geometry["curvature_tv_1pm"], "raw_interplan_position_rmse_m":.05 if episode.scenario_id == "SCN-07" else .8 if episode.scenario_id == "SCN-08" else "", "raw_initial_tangent_jump_rad":.05 if episode.scenario_id == "SCN-07" else .7 if episode.scenario_id == "SCN-08" else "", "minco_min_clearance_m": "" if run.variant == "raw" else minco_clearance, "minco_unsafe_ratio": "" if run.variant == "raw" else float(minco_clearance <= .6), "minco_path_length_m": "" if run.variant == "raw" else geometry["path_length_m"], "actual_speed_mean_mps": "" if run.variant == "raw" else .5, "actual_acc_rms_mps2": "" if run.variant == "raw" else .1, "actual_jerk_rms_mps3": "" if run.variant == "raw" else .05, "actual_yaw_rate_rms_radps": "" if run.variant == "raw" else .02, "planning_state": "HOT_START" if hot_accepted else "COLD_START", "hot_start_accepted": hot_accepted, "hot_reject_reason": hot_reject, "optimizer_success": "" if run.variant == "raw" else True, "python_validation_success": "" if run.variant == "raw" else True, "failure_reason": "", "optimizer_iteration_count": "" if run.variant == "raw" else 8, "planning_total_ms": 12.0 if run.variant == "raw" else 18.0, "data_source": "SIMULATED"})
            writer.submit_csv("candidate_metrics", {**base, "planning_cycle_uid":cycle_uid, "plan_uid": plan_uid, "candidate_index": 0, "candidate_rank": 0, "critic_value": .9, "attempted":run.variant != "raw", "optimizer_success":False if failure else ("" if run.variant == "raw" else True), "candidate_call_ms":"" if run.variant == "raw" else candidate_attempt_ms, "cpp_pipeline_ms":"" if run.variant == "raw" else candidate_cpp_ms, "optimizer_ms":"" if run.variant == "raw" else 1.5 + 0.1 * attempted_count, "cpp_validation_ms":"" if run.variant == "raw" else 0.3, "python_validation_ms":"" if run.variant == "raw" else python_validation_ms, "path_length_m": geometry["path_length_m"], "min_clearance_m": raw_clearance, "unsafe_ratio": float(raw_clearance <= .6), "curvature_tv_1pm": geometry["curvature_tv_1pm"], "selected": published, "data_source": "SIMULATED"})
            for frame, xy in enumerate(path):
                writer.submit_csv("control_samples", {**base, "frame_idx": frame, "plan_uid": plan_uid, "timestamp_monotonic_s": frame * .1, "control_state": "TRACK", "robot_x_m": xy[0], "robot_y_m": xy[1], "robot_yaw_rad": 0, "reference_x_m": xy[0], "reference_y_m": xy[1], "cmd_v_mps": .5, "cmd_w_radps": 0, "cross_track_error_m": .01, "time_aligned_position_error_m": .01, "mpc_success": True, "mpc_solve_ms": 2.0, "reference_age_ms": 10, "data_source": "SIMULATED"})
            timing_rows = [
                ("PLANNING", "planning_total_ms", 12.0 if run.variant == "raw" else 18.0 + attempted_count),
                ("PLANNING", "navdp_step_ms", 8.0),
            ]
            if run.variant != "raw":
                timing_rows.extend([
                    ("MINCO_ADAPTER_STAGE", "candidate_screen_ms", candidate_screen_ms),
                    ("MINCO_ADAPTER_STAGE", "candidate_attempt_total_ms", candidate_attempt_ms),
                    ("MINCO_ADAPTER_STAGE", "candidate_cpp_total_ms", candidate_cpp_ms),
                    ("MINCO_ADAPTER_STAGE", "python_validation_total_ms", python_validation_ms),
                    ("MINCO_ADAPTER_STAGE", "adapter_overhead_ms", adapter_overhead_ms),
                    ("MINCO_CPP_STAGE", "optimizer_ms", 1.5 + 0.1 * attempted_count),
                    ("MINCO_CPP_STAGE", "validate_ms", 0.3),
                ])
            for event_type, metric_name, duration_ms in timing_rows:
                writer.submit_csv("timing_samples", {**base, "event_type":event_type, "plan_uid":cycle_uid, "frame_idx":0, "timestamp_monotonic_s":.1, "metric_name":metric_name, "duration_ms":duration_ms, "status":"OK" if published else "FAILED", "data_source":"SIMULATED"})
            writer.submit_csv("events", {**base, "timestamp_monotonic_s": 0.0, "frame_idx": 0, "plan_uid": plan_uid, "event_type": "EPISODE_START", "severity": "INFO", "primary_reason": "", "secondary_reason": "", "message": "mock episode", "data_source": "SIMULATED"})
            collision = episode.scenario_id == "SCN-06"; timeout = episode.scenario_id == "SCN-08"; success = not collision and not timeout and not failure
            distance = float(np.linalg.norm(goal - start)); writer.submit_csv("episode_metrics", {**base, "episode_index": episode.episode_index, "success": success, "collision": collision, "timeout": timeout, "done_reason": "COLLISION" if collision else "TIMEOUT" if timeout else "GOAL_REACHED", "episode_duration_s": max(distance / .5, .1), "actual_path_length_m": geometry["path_length_m"], "repository_spl": 1.0 if success else 0.0, "tracking_error_rmse_m": .01 if success else .3, "minimum_executed_clearance_m": minco_clearance, "planning_count": 1, "minco_ok_count": int(run.variant != "raw" and published), "validation_failure_count": int(failure), "hold_count": int(fallback == "HOLD_LAST"), "stop_count": int(fallback == "STOP"), "hot_start_count": int(hot_accepted), "cold_start_count": int(not hot_accepted), "data_source": "SIMULATED"})
            trace_writer.write(cycle_uid, {"raw_path_xy":path, "topk_candidates_xy":np.stack([path, path + np.array([0,.1])]), "critic_values":np.array([.9,.8]), "selected_candidate_xy":path if published else np.empty((0,2)), "robot_state":np.array([*episode.start_pose[:2], episode.start_pose[2], 0., 0.]), "goal":goal, "esdf_distance":np.ones((8,8)), "esdf_origin":np.array([-1.,-1.]), "esdf_resolution":np.array(.25)})
