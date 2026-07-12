from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DifferentialDriveLimits:
    wheel_radius: Optional[float] = None
    wheel_base: Optional[float] = None
    max_wheel_speed: Optional[float] = None

    @classmethod
    def create(cls, wheel_radius=None, wheel_base=None, max_wheel_speed=None):
        values = (wheel_radius, wheel_base, max_wheel_speed)
        if not all(
            value is not None and np.isfinite(value) and float(value) > 0.0
            for value in values
        ):
            return cls()
        return cls(*(float(value) for value in values))

    @property
    def enabled(self) -> bool:
        return self.wheel_radius is not None

    def wheel_speeds(self, linear_speed, angular_speed) -> Tuple[object, object]:
        if not self.enabled:
            raise ValueError("wheel speeds require enabled differential-drive limits")
        denominator = 2.0 * self.wheel_radius
        left = (2.0 * linear_speed - angular_speed * self.wheel_base) / denominator
        right = (2.0 * linear_speed + angular_speed * self.wheel_base) / denominator
        return left, right

    def project(self, command, v_max: float, w_max: float) -> np.ndarray:
        command = np.asarray(command, dtype=np.float64).reshape(2)
        v = float(np.clip(command[0], 0.0, v_max))
        w = float(np.clip(command[1], -w_max, w_max))
        if not self.enabled:
            return np.array([v, w], dtype=np.float64)
        v = min(v, self.wheel_radius * self.max_wheel_speed)
        wheel_w_limit = max(
            0.0,
            2.0 * (self.wheel_radius * self.max_wheel_speed - v) / self.wheel_base,
        )
        w_limit = min(w_max, wheel_w_limit)
        return np.array([v, np.clip(w, -w_limit, w_limit)], dtype=np.float64)
