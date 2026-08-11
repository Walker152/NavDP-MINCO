from __future__ import annotations

import math


def _validate(wheel_radius_m: float, wheel_base_m: float) -> tuple[float, float]:
    radius, base = float(wheel_radius_m), float(wheel_base_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("wheel_radius_m must be finite and positive")
    if not math.isfinite(base) or base <= 0.0:
        raise ValueError("wheel_base_m must be finite and positive")
    return radius, base


def body_to_wheels(
    linear_speed_mps: float,
    yaw_rate_radps: float,
    wheel_radius_m: float,
    wheel_base_m: float,
) -> tuple[float, float]:
    radius, base = _validate(wheel_radius_m, wheel_base_m)
    left = (2.0 * float(linear_speed_mps) - float(yaw_rate_radps) * base) / (2.0 * radius)
    right = (2.0 * float(linear_speed_mps) + float(yaw_rate_radps) * base) / (2.0 * radius)
    return left, right


def wheels_to_body(
    left_wheel_radps: float,
    right_wheel_radps: float,
    wheel_radius_m: float,
    wheel_base_m: float,
) -> tuple[float, float]:
    radius, base = _validate(wheel_radius_m, wheel_base_m)
    linear = radius * (float(left_wheel_radps) + float(right_wheel_radps)) / 2.0
    yaw_rate = radius * (float(right_wheel_radps) - float(left_wheel_radps)) / base
    return linear, yaw_rate
