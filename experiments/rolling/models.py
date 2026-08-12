from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


ROLLOUT_STATUSES = frozenset(
    {
        "GOAL_REACHED",
        "COLLISION",
        "OPTIMIZATION_FAILED",
        "STALLED",
        "MAX_CYCLES",
        "TIMEOUT",
    }
)
ROLLOUT_METHODS = frozenset({"legacy", "safe_corridor_v1"})
MINCO_SAMPLE_COLUMNS = 15
HOT_START_STATE_TOLERANCE = 5e-4


def readonly_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    min_rows: int | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if min_rows is not None and (array.ndim == 0 or array.shape[0] < min_rows):
        raise ValueError(f"{name} must contain at least {min_rows} rows")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def safe_uid(value: object, name: str) -> str:
    result = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", result):
        raise ValueError(f"{name} must contain only safe identifier characters")
    return result


def finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def positive_float(value: object, name: str) -> float:
    result = finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _freeze(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        output = {str(key): _freeze(item, f"{name}.{key}") for key, item in value.items()}
        if any(not key for key in output):
            raise ValueError(f"{name} keys must be nonempty")
        return MappingProxyType(output)
    if isinstance(value, np.ndarray):
        return readonly_array(value, name=name)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item, f"{name}[]") for item in value)
    if isinstance(value, np.generic):
        return _freeze(value.item(), name)
    if isinstance(value, float):
        return finite_float(value, name)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"{name} contains unsupported value {type(value).__name__}")


def _mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    output = {str(key): _freeze(item, f"{name}.{key}") for key, item in value.items()}
    if any(not key for key in output):
        raise ValueError(f"{name} keys must be nonempty")
    return MappingProxyType(output)


@dataclass(frozen=True, eq=False)
class RobotState:
    position_xyz: np.ndarray
    velocity_xyz_mps: np.ndarray
    acceleration_xyz_mps2: np.ndarray
    yaw_rad: float
    yaw_rate_radps: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_xyz", readonly_array(self.position_xyz, name="position_xyz", shape=(3,)))
        object.__setattr__(self, "velocity_xyz_mps", readonly_array(self.velocity_xyz_mps, name="velocity_xyz_mps", shape=(3,)))
        object.__setattr__(self, "acceleration_xyz_mps2", readonly_array(self.acceleration_xyz_mps2, name="acceleration_xyz_mps2", shape=(3,)))
        object.__setattr__(self, "yaw_rad", finite_float(self.yaw_rad, "yaw_rad"))
        object.__setattr__(self, "yaw_rate_radps", finite_float(self.yaw_rate_radps, "yaw_rate_radps"))

    @classmethod
    def from_minco_sample(cls, sample: Sequence[float]) -> "RobotState":
        row = np.asarray(sample, dtype=np.float64)
        if row.shape != (MINCO_SAMPLE_COLUMNS,) or not np.all(np.isfinite(row)):
            raise ValueError(f"MINCO sample must be finite shape ({MINCO_SAMPLE_COLUMNS},)")
        return cls(row[1:4], row[4:7], row[7:10], row[13], row[14])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RobotState) and (
            np.array_equal(self.position_xyz, other.position_xyz)
            and np.array_equal(self.velocity_xyz_mps, other.velocity_xyz_mps)
            and np.array_equal(self.acceleration_xyz_mps2, other.acceleration_xyz_mps2)
            and self.yaw_rad == other.yaw_rad
            and self.yaw_rate_radps == other.yaw_rate_radps
        )


@dataclass(frozen=True, eq=False)
class ObstacleState:
    obstacle_uid: str
    center_xy: np.ndarray
    radius_m: float
    velocity_xy_mps: np.ndarray = field(default_factory=lambda: np.zeros(2))
    dynamic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "obstacle_uid", safe_uid(self.obstacle_uid, "obstacle_uid"))
        object.__setattr__(self, "center_xy", readonly_array(self.center_xy, name="center_xy", shape=(2,)))
        object.__setattr__(self, "velocity_xy_mps", readonly_array(self.velocity_xy_mps, name="velocity_xy_mps", shape=(2,)))
        object.__setattr__(self, "radius_m", positive_float(self.radius_m, "radius_m"))
        object.__setattr__(self, "dynamic", bool(self.dynamic))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ObstacleState) and (
            self.obstacle_uid == other.obstacle_uid
            and np.array_equal(self.center_xy, other.center_xy)
            and self.radius_m == other.radius_m
            and np.array_equal(self.velocity_xy_mps, other.velocity_xy_mps)
            and self.dynamic == other.dynamic
        )


@dataclass(frozen=True)
class RolloutConfig:
    planning_period_s: float = 0.5
    execute_duration_s: float = 0.5
    local_horizon_m: float = 2.0
    max_cycles: int = 40
    max_time_s: float = 20.0
    stall_window_cycles: int = 4
    stall_distance_m: float = 0.02
    collision_distance_m: float = 0.0
    goal_tolerance_m: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "planning_period_s", "execute_duration_s", "local_horizon_m",
            "max_time_s", "stall_distance_m", "goal_tolerance_m",
        ):
            object.__setattr__(self, name, positive_float(getattr(self, name), name))
        collision = finite_float(self.collision_distance_m, "collision_distance_m")
        if collision < 0.0:
            raise ValueError("collision_distance_m must be non-negative")
        object.__setattr__(self, "collision_distance_m", collision)
        for name in ("max_cycles", "stall_window_cycles"):
            raw = getattr(self, name)
            try:
                value = int(raw)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name} must be a positive integer") from error
            if isinstance(raw, bool) or value <= 0 or float(raw) != float(value):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, value)
        if self.execute_duration_s > self.planning_period_s:
            raise ValueError("execute_duration_s cannot exceed planning_period_s")


@dataclass(frozen=True)
class RolloutCycle:
    cycle_index: int
    time_s: float
    input_state: RobotState
    local_guide_xyz: np.ndarray
    local_goal_xyz: np.ndarray
    candidate_samples: np.ndarray
    executed_samples: np.ndarray
    corridor_segments: np.ndarray
    obstacle_states: tuple[ObstacleState, ...]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        index = int(self.cycle_index)
        if index < 0 or index != self.cycle_index:
            raise ValueError("cycle_index must be a non-negative integer")
        object.__setattr__(self, "cycle_index", index)
        time_s = finite_float(self.time_s, "time_s")
        if time_s < 0.0:
            raise ValueError("time_s must be non-negative")
        object.__setattr__(self, "time_s", time_s)
        if not isinstance(self.input_state, RobotState):
            raise TypeError("input_state must be RobotState")
        guide = readonly_array(self.local_guide_xyz, name="local_guide_xyz", ndim=2, min_rows=2)
        if guide.shape[1] != 3:
            raise ValueError("local_guide_xyz must have shape (N>=2, 3)")
        object.__setattr__(self, "local_guide_xyz", guide)
        object.__setattr__(self, "local_goal_xyz", readonly_array(self.local_goal_xyz, name="local_goal_xyz", shape=(3,)))
        candidate = readonly_array(self.candidate_samples, name="candidate_samples", ndim=2, min_rows=1)
        executed = readonly_array(self.executed_samples, name="executed_samples", ndim=2, min_rows=1)
        if candidate.shape[1] != MINCO_SAMPLE_COLUMNS or executed.shape[1] != MINCO_SAMPLE_COLUMNS:
            raise ValueError(f"trajectory samples must have {MINCO_SAMPLE_COLUMNS} columns")
        object.__setattr__(self, "candidate_samples", candidate)
        object.__setattr__(self, "executed_samples", executed)
        corridor = readonly_array(self.corridor_segments, name="corridor_segments", ndim=2)
        if corridor.shape[1] != 5:
            raise ValueError("corridor_segments must have shape (N, 5)")
        if corridor.size and np.any(corridor[:, 4] <= 0.0):
            raise ValueError("corridor radii must be positive")
        object.__setattr__(self, "corridor_segments", corridor)
        obstacles = tuple(self.obstacle_states)
        if not all(isinstance(row, ObstacleState) for row in obstacles):
            raise TypeError("obstacle_states must contain ObstacleState")
        if len({row.obstacle_uid for row in obstacles}) != len(obstacles):
            raise ValueError("obstacle_states contains duplicate UIDs")
        object.__setattr__(self, "obstacle_states", obstacles)
        object.__setattr__(self, "diagnostics", _mapping(self.diagnostics, "diagnostics"))


def validate_cycle_sequence(
    cycles: Sequence[RolloutCycle], tolerance: float = 1e-8
) -> list[str]:
    tolerance = positive_float(tolerance, "tolerance")
    errors: list[str] = []
    for expected_index, cycle in enumerate(cycles):
        if cycle.cycle_index != expected_index:
            errors.append(f"cycle index discontinuity at {expected_index}")
        if np.any(np.diff(cycle.candidate_samples[:, 0]) <= 0.0):
            errors.append(f"cycle {cycle.cycle_index} candidate sample time is not strictly increasing")
        if np.any(np.diff(cycle.executed_samples[:, 0]) <= 0.0):
            errors.append(f"cycle {cycle.cycle_index} executed sample time is not strictly increasing")
        if cycle.executed_samples.shape[0] > cycle.candidate_samples.shape[0] or not np.allclose(
            cycle.executed_samples,
            cycle.candidate_samples[: cycle.executed_samples.shape[0]],
            rtol=0.0,
            atol=tolerance,
        ):
            errors.append(f"cycle {cycle.cycle_index} executed samples are not a candidate prefix")
        first = cycle.candidate_samples[0]
        state_values = np.concatenate(
            [
                cycle.input_state.position_xyz,
                cycle.input_state.velocity_xyz_mps,
                cycle.input_state.acceleration_xyz_mps2,
                [cycle.input_state.yaw_rad, cycle.input_state.yaw_rate_radps],
            ]
        )
        sample_values = np.concatenate([first[1:10], first[13:15]])
        state_tolerance = (
            HOT_START_STATE_TOLERANCE
            if cycle.diagnostics.get("planning_state") == "HOT_START"
            else tolerance
        )
        if not np.allclose(
            state_values, sample_values, rtol=0.0, atol=state_tolerance
        ):
            errors.append(f"cycle {cycle.cycle_index} candidate start state discontinuity")
        if expected_index:
            previous = cycles[expected_index - 1]
            if cycle.time_s <= previous.time_s:
                errors.append(f"cycle {cycle.cycle_index} time is not monotonic")
            previous_state = RobotState.from_minco_sample(previous.executed_samples[-1])
            if not (
                np.allclose(cycle.input_state.position_xyz, previous_state.position_xyz, rtol=0.0, atol=tolerance)
                and np.allclose(cycle.input_state.velocity_xyz_mps, previous_state.velocity_xyz_mps, rtol=0.0, atol=tolerance)
                and np.allclose(cycle.input_state.acceleration_xyz_mps2, previous_state.acceleration_xyz_mps2, rtol=0.0, atol=tolerance)
                and abs(cycle.input_state.yaw_rad - previous_state.yaw_rad) <= tolerance
                and abs(cycle.input_state.yaw_rate_radps - previous_state.yaw_rate_radps) <= tolerance
            ):
                errors.append(f"cycle {cycle.cycle_index} input state discontinuity")
    return errors


@dataclass(frozen=True)
class RolloutResult:
    scenario_uid: str
    method: str
    status: str
    cycles: tuple[RolloutCycle, ...]
    executed_samples: np.ndarray
    final_goal_xyz: np.ndarray
    metrics: Mapping[str, object]
    goal_tolerance_m: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_uid", safe_uid(self.scenario_uid, "scenario_uid"))
        if self.method not in ROLLOUT_METHODS:
            raise ValueError(f"unsupported rollout method: {self.method}")
        if self.status not in ROLLOUT_STATUSES:
            raise ValueError(f"unsupported rollout status: {self.status}")
        cycles = tuple(self.cycles)
        if not cycles or not all(isinstance(cycle, RolloutCycle) for cycle in cycles):
            raise ValueError("cycles must contain at least one RolloutCycle")
        object.__setattr__(self, "cycles", cycles)
        samples = readonly_array(self.executed_samples, name="executed_samples", ndim=2, min_rows=1)
        if samples.shape[1] != MINCO_SAMPLE_COLUMNS:
            raise ValueError(f"executed_samples must have {MINCO_SAMPLE_COLUMNS} columns")
        object.__setattr__(self, "executed_samples", samples)
        object.__setattr__(self, "final_goal_xyz", readonly_array(self.final_goal_xyz, name="final_goal_xyz", shape=(3,)))
        object.__setattr__(self, "metrics", _mapping(self.metrics, "metrics"))
        object.__setattr__(self, "goal_tolerance_m", positive_float(self.goal_tolerance_m, "goal_tolerance_m"))

    def validate(self) -> list[str]:
        errors = validate_cycle_sequence(self.cycles)
        expected = np.concatenate(
            [
                self.cycles[0].executed_samples,
                *[cycle.executed_samples[1:] for cycle in self.cycles[1:]],
            ]
        )
        if not np.array_equal(self.executed_samples, expected):
            errors.append("result executed_samples do not match accumulated cycle prefixes")
        if np.any(np.diff(self.executed_samples[:, 0]) <= 0.0):
            errors.append("result executed sample time is not strictly increasing")
        final_error = float(np.linalg.norm(self.executed_samples[-1, 1:4] - self.final_goal_xyz))
        if self.status == "GOAL_REACHED" and final_error > self.goal_tolerance_m:
            errors.append(
                f"GOAL_REACHED final error {final_error:.9g} exceeds goal tolerance {self.goal_tolerance_m:.9g}"
            )
        metric_error = self.metrics.get("final_error_m")
        if metric_error is not None:
            try:
                value = finite_float(metric_error, "metrics.final_error_m")
            except ValueError as error:
                errors.append(str(error))
            else:
                if abs(value - final_error) > 1e-8:
                    errors.append("metrics.final_error_m does not match recorded trajectory")
        return errors
