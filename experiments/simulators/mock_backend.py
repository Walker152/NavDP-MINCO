from __future__ import annotations

import math
import numpy as np

from experiments.analyzers.metrics import compute_geometric_metrics
from experiments.core.schemas import IDENTITY


def _identity(run, episode):
    return {"suite_id": run.suite_id, "experiment_id": run.experiment_id, "run_id": run.run_id, "variant": run.variant, "scene_label": run.scene_label, "scene_id": run.scene_id, "seed": run.seed, "episode_uid": episode.episode_uid}


class MockBackend:
    """Deterministic, CPU-only contract simulator for tooling tests."""
    name = "mock"

    def run(self, run, episodes, writer):
        for episode in episodes:
            base = _identity(run, episode); start = np.asarray(episode.start_pose[:2], float); goal = np.asarray(episode.goal_pose[:2], float)
            alpha = np.linspace(0, 1, 21); path = start + alpha[:, None] * (goal - start)
            if run.scene_label == "DENSE": path[:, 1] += 0.15 * np.sin(alpha * math.pi)
            geometry, _ = compute_geometric_metrics(path, 0.05)
            plan_uid = f"{episode.episode_uid}_{run.variant}_p000"
            raw_clearance = 0.45 if run.scene_label == "DENSE" else 1.2
            minco_clearance = raw_clearance + (0.25 if run.variant != "raw" else 0)
            writer.submit_csv("plan_metrics", {**base, "plan_uid": plan_uid, "timestamp_monotonic_s": 0.1, "plan_status": "RAW_OK" if run.variant == "raw" else "MINCO_OK", "fallback_mode": "NONE", "raw_safety_class": "RAW_UNSAFE" if raw_clearance <= .6 else "RAW_SAFE", "turn_class": "LOW_TURN", "temporal_class": "NO_HISTORY", "raw_min_clearance_m": raw_clearance, "raw_unsafe_ratio": float(raw_clearance <= .6), "raw_esdf_oob_ratio": 0.0, "raw_path_length_m": geometry["path_length_m"], "raw_curvature_abs_p95_1pm": geometry["curvature_abs_p95_1pm"], "raw_curvature_tv_1pm": geometry["curvature_tv_1pm"], "minco_min_clearance_m": "" if run.variant == "raw" else minco_clearance, "minco_unsafe_ratio": "" if run.variant == "raw" else float(minco_clearance <= .6), "minco_path_length_m": "" if run.variant == "raw" else geometry["path_length_m"], "actual_speed_mean_mps": "" if run.variant == "raw" else .5, "actual_acc_rms_mps2": "" if run.variant == "raw" else .1, "actual_jerk_rms_mps3": "" if run.variant == "raw" else .05, "actual_yaw_rate_rms_radps": "" if run.variant == "raw" else .02, "planning_state": "HOT_START" if run.variant == "minco-hot" else "COLD_START", "hot_start_accepted": run.variant == "minco-hot", "hot_reject_reason": "" if run.variant == "minco-hot" else "COLD_FORCED", "optimizer_success": "" if run.variant == "raw" else True, "python_validation_success": "" if run.variant == "raw" else True, "failure_reason": "", "optimizer_iteration_count": "" if run.variant == "raw" else 8, "planning_total_ms": 12.0 if run.variant == "raw" else 18.0, "data_source": "SIMULATED"})
            writer.submit_csv("candidate_metrics", {**base, "plan_uid": plan_uid, "candidate_index": 0, "candidate_rank": 0, "critic_value": .9, "path_length_m": geometry["path_length_m"], "min_clearance_m": raw_clearance, "unsafe_ratio": float(raw_clearance <= .6), "curvature_tv_1pm": geometry["curvature_tv_1pm"], "selected": True, "data_source": "SIMULATED"})
            for frame, xy in enumerate(path):
                writer.submit_csv("control_samples", {**base, "frame_idx": frame, "plan_uid": plan_uid, "timestamp_monotonic_s": frame * .1, "control_state": "TRACK", "robot_x_m": xy[0], "robot_y_m": xy[1], "robot_yaw_rad": 0, "reference_x_m": xy[0], "reference_y_m": xy[1], "cmd_v_mps": .5, "cmd_w_radps": 0, "cross_track_error_m": .01, "time_aligned_position_error_m": .01, "mpc_success": True, "mpc_solve_ms": 2.0, "reference_age_ms": 10, "data_source": "SIMULATED"})
            writer.submit_csv("timing_samples", {**base, "event_type": "planning", "plan_uid": plan_uid, "frame_idx": 0, "timestamp_monotonic_s": .1, "metric_name": "planning_total_ms", "duration_ms": 12 if run.variant == "raw" else 18, "status": "OK", "data_source": "SIMULATED"})
            writer.submit_csv("events", {**base, "timestamp_monotonic_s": 0.0, "frame_idx": 0, "plan_uid": plan_uid, "event_type": "EPISODE_START", "severity": "INFO", "primary_reason": "", "secondary_reason": "", "message": "mock episode", "data_source": "SIMULATED"})
            distance = float(np.linalg.norm(goal - start)); writer.submit_csv("episode_metrics", {**base, "episode_index": episode.episode_index, "success": True, "collision": False, "timeout": False, "done_reason": "GOAL_REACHED", "episode_duration_s": max(distance / .5, .1), "actual_path_length_m": geometry["path_length_m"], "repository_spl": 1.0, "tracking_error_rmse_m": .01, "minimum_executed_clearance_m": minco_clearance, "planning_count": 1, "minco_ok_count": int(run.variant != "raw"), "validation_failure_count": 0, "hold_count": 0, "stop_count": 0, "hot_start_count": int(run.variant == "minco-hot"), "cold_start_count": int(run.variant != "minco-hot"), "data_source": "SIMULATED"})
            writer.submit_npz(f"traces/planning_trace_{episode.episode_uid}.npz", {"raw_top1_world": path, "selected_index": np.array([0])})
