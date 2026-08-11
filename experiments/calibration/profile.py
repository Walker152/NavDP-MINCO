from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from experiments.calibration.transforms import invert_transform, validate_transform


OVERRIDABLE_FIELDS = frozenset(
    {
        "max_wheel_speed_radps",
        "max_linear_speed_mps",
        "max_yaw_rate_radps",
        "max_linear_acc_mps2",
        "max_yaw_acc_radps2",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _polygon_radii(polygon: np.ndarray) -> tuple[float, float]:
    circumscribed = float(np.max(np.linalg.norm(polygon, axis=1)))
    distances = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length <= 1e-12:
            raise ValueError("footprint has repeated vertices")
        distances.append(abs(float(np.cross(edge, -start))) / length)
    return float(min(distances)), circumscribed


@dataclass(frozen=True)
class RobotCalibration:
    path: Path
    calibration_sha256: str
    robot_model_id: str
    base_frame: str
    camera_frame: str
    T_base_camera: np.ndarray
    T_camera_base: np.ndarray
    footprint_polygon_xy_m: np.ndarray
    inscribed_radius_m: float
    circumscribed_radius_m: float
    wheel_radius_m: float
    wheel_base_m: float
    max_wheel_speed_radps: float
    max_linear_speed_mps: float
    max_yaw_rate_radps: float
    max_linear_acc_mps2: float
    max_yaw_acc_radps2: float
    command_delay_s: float | None
    validation_safe_dist_m: float
    optimization_safe_dist_m: float
    optimization_buffer_m: float
    status: str
    source_sha256: Mapping[str, str]
    overrides: Mapping[str, float]
    raw: Mapping[str, Any]


def load_robot_calibration(
    path: Path | str,
    *,
    overrides: Mapping[str, float] | None = None,
) -> RobotCalibration:
    path = Path(path).resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid robot calibration: {error}") from error
    if data.get("schema_version") != 1:
        raise ValueError("robot calibration schema_version mismatch")
    override_values = dict(overrides or {})
    unknown = set(override_values) - OVERRIDABLE_FIELDS
    if unknown:
        raise ValueError(f"unknown calibration override: {sorted(unknown)}")
    for key, value in override_values.items():
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"calibration override {key} must be positive")
        data[key] = number
        override_values[key] = number

    t_base_camera = validate_transform(data["T_base_camera"])
    t_camera_base = validate_transform(data["T_camera_base"])
    if not np.allclose(
        t_camera_base, invert_transform(t_base_camera), atol=1e-10
    ):
        raise ValueError("camera/base transforms are not inverses")
    polygon = np.asarray(data["footprint_polygon_xy_m"], dtype=np.float64)
    if (
        polygon.ndim != 2
        or polygon.shape[0] < 3
        or polygon.shape[1] != 2
        or not np.all(np.isfinite(polygon))
    ):
        raise ValueError("footprint_polygon_xy_m must be finite shape (N>=3,2)")
    inscribed, circumscribed = _polygon_radii(polygon)
    declared_inscribed = float(data["inscribed_radius_m"])
    declared_circumscribed = float(data["circumscribed_radius_m"])
    if not math.isclose(inscribed, declared_inscribed, rel_tol=0, abs_tol=1e-8):
        raise ValueError("declared inscribed radius does not match footprint")
    if not math.isclose(circumscribed, declared_circumscribed, rel_tol=0, abs_tol=1e-8):
        raise ValueError("declared circumscribed radius does not match footprint")
    safety = data["safety_distance"]
    components = safety["validation_components_m"]
    validation = float(sum(float(value) for value in components.values()))
    if not math.isclose(
        validation, float(safety["validation_safe_dist_m"]), abs_tol=1e-10
    ):
        raise ValueError("validation safety distance components do not sum")
    buffer = float(safety["optimization_buffer_m"])
    optimization = validation + buffer
    if not math.isclose(
        optimization, float(safety["optimization_safe_dist_m"]), abs_tol=1e-10
    ):
        raise ValueError("optimization safety distance formula mismatch")
    positive = (
        "wheel_radius_m",
        "wheel_base_m",
        "max_wheel_speed_radps",
        "max_linear_speed_mps",
        "max_yaw_rate_radps",
        "max_linear_acc_mps2",
        "max_yaw_acc_radps2",
    )
    if any(not math.isfinite(float(data[key])) or float(data[key]) <= 0 for key in positive):
        raise ValueError("robot drive limits must be finite and positive")
    return RobotCalibration(
        path=path,
        calibration_sha256=_sha256(path),
        robot_model_id=str(data["robot_model_id"]),
        base_frame=str(data["base_frame"]),
        camera_frame=str(data["camera_frame"]),
        T_base_camera=t_base_camera,
        T_camera_base=t_camera_base,
        footprint_polygon_xy_m=polygon,
        inscribed_radius_m=inscribed,
        circumscribed_radius_m=circumscribed,
        wheel_radius_m=float(data["wheel_radius_m"]),
        wheel_base_m=float(data["wheel_base_m"]),
        max_wheel_speed_radps=float(data["max_wheel_speed_radps"]),
        max_linear_speed_mps=float(data["max_linear_speed_mps"]),
        max_yaw_rate_radps=float(data["max_yaw_rate_radps"]),
        max_linear_acc_mps2=float(data["max_linear_acc_mps2"]),
        max_yaw_acc_radps2=float(data["max_yaw_acc_radps2"]),
        command_delay_s=None
        if data.get("command_delay_s") is None
        else float(data["command_delay_s"]),
        validation_safe_dist_m=validation,
        optimization_safe_dist_m=optimization,
        optimization_buffer_m=buffer,
        status=str(data["status"]),
        source_sha256=MappingProxyType(dict(data["source_sha256"])),
        overrides=MappingProxyType(override_values),
        raw=MappingProxyType(data),
    )
