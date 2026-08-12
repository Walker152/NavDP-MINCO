from __future__ import annotations

from copy import deepcopy
import math
from typing import Mapping

from experiments.core.effective_parameters import EFFECTIVE_PARAMETERS


def calibrated_defaults() -> dict[str, dict[str, object]]:
    """Return an independent snapshot of the calibrated experiment defaults."""

    return deepcopy(EFFECTIVE_PARAMETERS)


def normalize_overrides(
    overrides: Mapping[str, Mapping[str, object]] | None,
    *,
    defaults: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    normalized: dict[str, dict[str, object]] = {}
    for section, values in (overrides or {}).items():
        if section not in defaults or not isinstance(values, Mapping):
            raise ValueError(f"unknown parameter section: {section}")
        unknown = set(values) - set(defaults[section])
        if unknown:
            raise ValueError(f"unknown {section} parameters: {sorted(unknown)}")
        if section == "robot_calibration":
            changed = [
                key
                for key, value in values.items()
                if value != defaults[section][key]
            ]
            if changed:
                raise ValueError(
                    "robot_calibration override conflicts with calibration truth: "
                    f"{sorted(changed)}"
                )
        normalized[section] = deepcopy(dict(values))
    return normalized


def merge_sections(
    defaults: Mapping[str, Mapping[str, object]],
    overrides: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    effective = deepcopy(dict(defaults))
    for section, values in overrides.items():
        effective[section].update(deepcopy(dict(values)))
    return effective


def _positive_number(section: Mapping[str, object], key: str, label: str) -> float:
    value = section.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite and positive") from error
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def validate_effective_parameters(
    effective: Mapping[str, Mapping[str, object]],
) -> None:
    calibration = effective["robot_calibration"]
    minco = effective["minco"]
    minco_mpc = effective["minco_mpc"]
    esdf = effective["esdf"]

    for section_name, section in (
        ("robot_calibration", calibration),
        ("minco", minco),
        ("minco_mpc", minco_mpc),
    ):
        for key in ("wheel_radius_m", "wheel_base_m", "max_wheel_speed_radps"):
            _positive_number(section, key, f"{section_name}.{key}")

    circumscribed = _positive_number(
        calibration,
        "circumscribed_radius_m",
        "robot_calibration.circumscribed_radius_m",
    )
    esdf_radius = _positive_number(
        esdf,
        "robot_circumscribed_radius_m",
        "esdf.robot_circumscribed_radius_m",
    )
    if not math.isclose(esdf_radius, circumscribed, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "esdf.robot_circumscribed_radius_m must match calibrated "
            "circumscribed_radius_m"
        )

    validation = _positive_number(
        minco,
        "validation_safe_distance_m",
        "minco.validation_safe_distance_m",
    )
    optimization = _positive_number(
        minco,
        "optimization_safe_distance_m",
        "minco.optimization_safe_distance_m",
    )
    profile = str(minco.get("constraint_profile", ""))
    if profile not in {"legacy", "safe_corridor_v1"}:
        raise ValueError(f"unsupported minco.constraint_profile: {profile!r}")
    if profile == "safe_corridor_v1":
        if validation < circumscribed:
            raise ValueError(
                "minco.validation_safe_distance_m must be greater than or equal "
                "to the calibrated circumscribed radius"
            )
        if optimization < validation:
            raise ValueError(
                "minco.optimization_safe_distance_m must be greater than or "
                "equal to minco.validation_safe_distance_m"
            )


def resolve_parameter_receipt(
    *,
    video_enabled: bool,
    overrides: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, object]:
    defaults = calibrated_defaults()
    normalized_overrides = normalize_overrides(overrides, defaults=defaults)
    effective = merge_sections(defaults, normalized_overrides)
    effective["video"]["enabled"] = bool(video_enabled)
    validate_effective_parameters(effective)
    return {
        "schema_version": 1,
        "defaults": defaults,
        "overrides": normalized_overrides,
        "effective": effective,
        "calibration_sha256": effective["robot_calibration"]["sha256"],
    }
