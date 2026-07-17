from __future__ import annotations

from copy import deepcopy

from experiments.baselines.raw_navdp.original_mpc import ORIGINAL_MPC_SPEC


EFFECTIVE_PARAMETERS = {
    "minco": {
        "initial_top_k": 2,
        "max_top_k": 4,
        "candidate_time_budget_ms": 1500.0,
        "optimization_safe_distance_m": 0.45,
        "validation_safe_distance_m": 0.35,
        "start_validation_exemption_radius_m": 0.35,
        "sample_dt_s": 0.05,
        "max_velocity_mps": 1.0,
        "max_acceleration_mps2": 1.0,
        "max_iterations": 64,
        "penalty_weight_pos": 10000.0,
        "penalty_weight_vel": 1000.0,
        "penalty_weight_acc": 1000.0,
        "penalty_weight_attractor": 20.0,
        "time_weight": 0.1,
        "time_barrier_weight": 10.0,
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
        "wheel_radius_source": "DINGO_WHEEL_RADIUS",
        "wheel_base_source": "DINGO_WHEEL_BASE", "max_wheel_speed_radps": None,
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
}


def effective_parameters(video_enabled=True, overrides=None):
    snapshot = deepcopy(EFFECTIVE_PARAMETERS)
    for section, values in (overrides or {}).items():
        if section not in snapshot or not isinstance(values, dict):
            raise ValueError(f"unknown parameter section: {section}")
        unknown = set(values) - set(snapshot[section])
        if unknown: raise ValueError(f"unknown {section} parameters: {sorted(unknown)}")
        snapshot[section].update(values)
    snapshot["video"]["enabled"] = bool(video_enabled)
    return snapshot
