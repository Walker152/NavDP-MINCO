from __future__ import annotations

from pathlib import Path

from experiments.baselines.raw_navdp.original_mpc import ORIGINAL_MPC_SPEC
from experiments.calibration.profile import load_robot_calibration


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CALIBRATION_PATH = _REPO_ROOT / "configs/robots/dingo_calibration_v1.json"
_CALIBRATION = load_robot_calibration(_CALIBRATION_PATH)


EFFECTIVE_PARAMETERS = {
    "minco": {
        "initial_top_k": 2,
        "max_top_k": 4,
        "candidate_time_budget_ms": 1500.0,
        "optimization_safe_distance_m": _CALIBRATION.optimization_safe_dist_m,
        "validation_safe_distance_m": _CALIBRATION.validation_safe_dist_m,
        "start_validation_exemption_radius_m": 0.35,
        "sample_dt_s": 0.05,
        "max_velocity_mps": 1.0,
        "max_acceleration_mps2": 1.0,
        "max_yaw_rate_radps": 0.5,
        "max_iterations": 64,
        "penalty_weight_pos": 10000.0,
        "penalty_weight_vel": 1000.0,
        "penalty_weight_acc": 1000.0,
        "penalty_weight_attractor": 20.0,
        "time_weight": 0.1,
        "time_barrier_weight": 10.0,
        "constraint_profile": "safe_corridor_v1",
        "guide_corridor_weight": 2000.0,
        "corridor_max_radius_m": 0.45,
        "corridor_min_radius_m": 0.04,
        "corridor_sample_step_m": 0.025,
        "adaptive_max_spatial_step_m": 0.025,
        "adaptive_near_clearance_m": 0.05,
        "adaptive_max_depth": 14,
        "adaptive_sample_budget": 20000,
        "max_jerk_mps3": 20.0,
        "wheel_radius_m": _CALIBRATION.wheel_radius_m,
        "wheel_base_m": _CALIBRATION.wheel_base_m,
        "max_wheel_speed_radps": _CALIBRATION.max_wheel_speed_radps,
    },
    "esdf": {
        "resolution_m": 0.05,
        "padding_m": 1.0,
        "cache_name": "esdf_2d.npz",
        "force_rebuild": False,
        "obstacle_min_height_m": 0.08,
        "obstacle_max_height_m": 1.50,
        "fill_footprint": True,
        "footprint_inflate_cells": 1,
        "robot_circumscribed_radius_m": _CALIBRATION.circumscribed_radius_m,
    },
    "raw_mpc": {
        **{key: value for key, value in ORIGINAL_MPC_SPEC.items() if key not in {"Q", "R"}},
        "Q": list(ORIGINAL_MPC_SPEC["Q"]),
        "R": list(ORIGINAL_MPC_SPEC["R"]),
    },
    "minco_mpc": {
        "N": 15, "T_source": "isaac_env_step_dt", "desired_v_mps": 0.5,
        "v_max_mps": 0.5, "w_max_radps": 0.5, "ref_gap": 3,
        "max_acceleration_mps2": 1.0, "max_yaw_acceleration_radps2": 1.0,
        "wheel_radius_m": _CALIBRATION.wheel_radius_m,
        "wheel_base_m": _CALIBRATION.wheel_base_m,
        "wheel_radius_source": "robot_calibration",
        "wheel_base_source": "robot_calibration",
        "max_wheel_speed_radps": _CALIBRATION.max_wheel_speed_radps,
        "q_xy": 20.0, "q_yaw": 4.0, "q_v": 4.0, "q_w": 1.5,
        "r_dv": 3.0, "r_dw": 1.0, "terminal_xy_scale": 1.5,
        "terminal_yaw_scale": 1.0, "ipopt.max_iter": 100,
        "ipopt.acceptable_tol": 1e-8, "ipopt.acceptable_obj_change_tol": 1e-6,
    },
    "video": {
        "enabled": True, "fps": 10, "crf": 23, "scale": 1.0,
        "codec": "libx264", "pixel_format": "yuv420p", "macro_block_size": 16,
    },
    "scene": {"scale": 1.0},
    "robot_calibration": {
        "path": str(_CALIBRATION_PATH),
        "sha256": _CALIBRATION.calibration_sha256,
        "robot_model_id": _CALIBRATION.robot_model_id,
        "status": _CALIBRATION.status,
        "wheel_radius_m": _CALIBRATION.wheel_radius_m,
        "wheel_base_m": _CALIBRATION.wheel_base_m,
        "max_wheel_speed_radps": _CALIBRATION.max_wheel_speed_radps,
        "circumscribed_radius_m": _CALIBRATION.circumscribed_radius_m,
        "validation_safe_dist_m": _CALIBRATION.validation_safe_dist_m,
        "optimization_safe_dist_m": _CALIBRATION.optimization_safe_dist_m,
    },
    "runtime_diagnostics": {
        "threshold_profile_id": "full-real-v2",
        "high_turn_curvature_p95_1pm": 2.0,
        "high_turn_curvature_tv_1pm": 100.0,
        "jump_position_rmse_m": 0.3,
        "jump_tangent_rad": 0.3,
        "planning_deadline_ms": 100.0,
    },
}


def effective_parameters(video_enabled=True, overrides=None):
    # Import lazily so the compatibility module can continue exporting the
    # immutable default snapshot without creating an import cycle.
    from experiments.core.parameter_receipt import resolve_parameter_receipt

    return resolve_parameter_receipt(
        video_enabled=video_enabled,
        overrides=overrides,
    )["effective"]
