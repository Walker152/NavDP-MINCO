from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import numpy as np

from experiments.static.case_schema import StaticCase


@dataclass(frozen=True)
class StaticRunResult:
    case_uid: str
    case_hash: str
    mode: str
    status: str
    engine: str
    native_extension_path: str
    native_extension_sha256: str
    diagnostics: Mapping[str, Any]
    samples: np.ndarray
    waypoints: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_legacy_profile(config_path: Path | str) -> dict[str, Any]:
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid legacy profile: {error}") from error
    minco = config.get("minco")
    if not isinstance(minco, dict):
        raise ValueError("legacy profile is missing minco settings")
    return dict(minco)


def native_environment_diagnostics() -> dict[str, Any]:
    remediation = (
        "cmake -S minco_processor -B minco_processor/build && "
        "cmake --build minco_processor/build --target minco_processor_py"
    )
    diagnostics: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_abi": getattr(sys, "abiflags", ""),
        "remediation": remediation,
    }
    try:
        extension = importlib.import_module("_minco_processor")
    except ImportError:
        try:
            importlib.import_module("minco_processor")
            extension = importlib.import_module("_minco_processor")
        except ImportError as error:
            diagnostics["import_error"] = str(error)
            raise RuntimeError(
                "native MINCO extension unavailable; run: " + remediation
            ) from error
    extension_path = Path(extension.__file__).resolve()
    diagnostics.update(
        {
            "extension_path": str(extension_path),
            "extension_sha256": _sha256(extension_path),
        }
    )
    return diagnostics


def _normalise_native_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _normalise_native_value(v) for k, v in value.items()}
    return str(value)


def run_static_case(
    case: StaticCase,
    profile: Mapping[str, Any],
    mode: str,
) -> StaticRunResult:
    if mode not in {"inspect-only", "recompute"}:
        raise ValueError("mode must be inspect-only or recompute")
    if mode == "inspect-only":
        samples = np.asarray(
            case.auxiliary_arrays.get(
                "historical_minco_samples", np.empty((0, 15))
            ),
            dtype=np.float64,
        )
        if samples.ndim != 2 or samples.shape[1] != 15:
            raise ValueError("historical_minco_samples must have shape (N, 15)")
        return StaticRunResult(
            case_uid=case.case_uid,
            case_hash=case.case_hash,
            mode=mode,
            status="INSPECTED",
            engine="historical-trace",
            native_extension_path="",
            native_extension_sha256="",
            diagnostics={
                "success": None,
                "failure_reason": "INSPECT_ONLY",
                "state_availability": dict(case.state_availability),
            },
            samples=samples.copy(),
            waypoints=samples[:, 1:4].copy(),
        )
    if not case.esdf_available:
        raise ValueError("recompute requires an explicitly compatible ESDF")

    environment = native_environment_diagnostics()
    from minco_processor import MincoProcessor

    processor = MincoProcessor()
    processor.configure(
        float(profile["max_velocity_mps"]),
        float(profile["max_acceleration_mps2"]),
        float(profile["optimization_safe_distance_m"]),
        float(profile["validation_safe_distance_m"]),
        float(profile["start_validation_exemption_radius_m"]),
        float(profile["sample_dt_s"]),
        int(profile["max_iterations"]),
        float(profile["max_yaw_rate_radps"]),
        float(profile["penalty_weight_pos"]),
        float(profile["penalty_weight_vel"]),
        float(profile["penalty_weight_acc"]),
        float(profile["penalty_weight_attractor"]),
        float(profile["time_weight"]),
        float(profile["time_barrier_weight"]),
    )
    constraint_profile = str(profile.get("constraint_profile", "legacy"))
    if constraint_profile == "safe_corridor_v1":
        processor.configure_safety_profile(
            constraint_profile,
            float(profile["guide_corridor_weight"]),
            float(profile["corridor_max_radius_m"]),
            float(profile["corridor_min_radius_m"]),
            float(profile["corridor_sample_step_m"]),
            float(profile["adaptive_max_spatial_step_m"]),
            float(profile["adaptive_near_clearance_m"]),
            int(profile["adaptive_max_depth"]),
            int(profile["adaptive_sample_budget"]),
            float(profile["max_jerk_mps3"]),
            float(profile["wheel_radius_m"]),
            float(profile["wheel_base_m"]),
            float(profile["max_wheel_speed_radps"]),
        )
    processor.set_static_esdf_2d(
        np.asarray(case.esdf_distance, dtype=np.float64),
        np.asarray(~case.occupancy, dtype=np.uint8),
        np.asarray(case.esdf_origin, dtype=np.float64),
        case.esdf_resolution,
    )
    processor.reset_history()
    raw = processor.optimize_preview(
        np.asarray(case.guide_path_xyz, dtype=np.float64),
        np.asarray(case.start_position, dtype=np.float64),
        np.asarray(case.start_velocity, dtype=np.float64),
        np.asarray(case.start_acceleration, dtype=np.float64),
        case.start_yaw,
        case.start_yaw_rate,
        None
        if case.terminal_goal is None
        else np.asarray(case.terminal_goal, dtype=np.float64),
    )
    diagnostics = {
        str(key): _normalise_native_value(value)
        for key, value in raw.items()
        if key not in {"samples", "waypoints", "sparse_waypoints"}
    }
    samples = np.asarray(raw["samples"], dtype=np.float64)
    waypoints = np.asarray(raw["waypoints"], dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 15:
        raise RuntimeError("native MINCO returned invalid samples shape")
    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise RuntimeError("native MINCO returned invalid waypoints shape")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(waypoints)):
        raise RuntimeError("native MINCO returned non-finite trajectory output")
    return StaticRunResult(
        case_uid=case.case_uid,
        case_hash=case.case_hash,
        mode=mode,
        status="SUCCEEDED" if bool(raw["success"]) else "FAILED",
        engine="minco_processor.MincoProcessor",
        native_extension_path=environment["extension_path"],
        native_extension_sha256=environment["extension_sha256"],
        diagnostics=diagnostics,
        samples=samples.copy(),
        waypoints=waypoints.copy(),
    )
