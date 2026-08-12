from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def _vector(mapping: Mapping[str, Any], key: str, length: int = 3) -> np.ndarray:
    value = np.asarray(mapping.get(key), dtype=np.float64)
    if value.shape != (length,) or not np.all(np.isfinite(value)):
        raise RuntimeError(f"dynamic sanity invalid {key}")
    return value


def _require_hash(label: str, expected: Any, observed: Any) -> None:
    if not str(expected) or str(observed) != str(expected):
        raise RuntimeError(f"dynamic {label} hash mismatch")


def validate_dynamic_receipts(
    spec: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    case_sha256: str,
    scene_sha256: str,
    calibration_sha256: str,
) -> dict[str, Any]:
    """Fail closed before planning when frozen inputs or reset state differ."""
    _require_hash("case", spec.get("case_hash"), case_sha256)
    _require_hash("scene", spec.get("scene_sha256"), scene_sha256)
    _require_hash(
        "calibration", spec.get("calibration_sha256"), calibration_sha256
    )
    requirements = spec.get("frame_sanity_requirements", {})
    requested_pose = _vector(spec, "start_pose_xy_yaw")
    observed_pose = _vector(observed, "start_pose_xy_yaw")
    pose_tolerance = float(requirements.get("pose_match_tolerance_m", 0.02))
    yaw_tolerance = float(requirements.get("yaw_match_tolerance_rad", 0.02))
    if not np.allclose(
        observed_pose[:2], requested_pose[:2], rtol=0.0, atol=pose_tolerance
    ) or not math.isclose(
        float(observed_pose[2]), float(requested_pose[2]),
        rel_tol=0.0, abs_tol=yaw_tolerance,
    ):
        raise RuntimeError("dynamic start pose mismatch")
    requested_linear = _vector(spec, "initial_linear_velocity_xyz_mps")
    observed_linear = _vector(observed, "linear_velocity_xyz_mps")
    requested_angular = _vector(spec, "initial_angular_velocity_xyz_radps")
    observed_angular = _vector(observed, "angular_velocity_xyz_radps")
    velocity_tolerance = float(
        requirements.get("velocity_match_tolerance_mps", 0.02)
    )
    yaw_rate_tolerance = float(
        requirements.get("yaw_rate_match_tolerance_radps", 0.02)
    )
    if not np.allclose(
        observed_linear, requested_linear, rtol=0.0, atol=velocity_tolerance
    ) or not math.isclose(
        float(observed_angular[2]), float(requested_angular[2]),
        rel_tol=0.0, abs_tol=yaw_rate_tolerance,
    ):
        raise RuntimeError("dynamic initial velocity mismatch")
    clearance = float(observed.get("initial_esdf_clearance_m", math.nan))
    expected_clearance = float(
        observed.get("expected_initial_esdf_clearance_m", math.nan)
    )
    minimum_clearance = float(
        requirements.get("minimum_initial_clearance_m", 0.0)
    )
    clearance_tolerance = float(
        requirements.get("esdf_clearance_match_tolerance_m", 0.05)
    )
    if not math.isfinite(clearance) or clearance < minimum_clearance:
        raise RuntimeError("dynamic initial ESDF clearance unsafe")
    if math.isfinite(expected_clearance) and not math.isclose(
        clearance, expected_clearance, rel_tol=0.0, abs_tol=clearance_tolerance
    ):
        raise RuntimeError("dynamic initial ESDF clearance mismatch")
    return {
        "schema_version": 1,
        "status": "VALIDATED",
        "case_uid": str(spec.get("case_uid", "")),
        "case_hash": str(case_sha256),
        "scene_sha256": str(scene_sha256),
        "calibration_sha256": str(calibration_sha256),
        "observed": dict(observed),
    }
