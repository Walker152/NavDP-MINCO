from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from experiments.rolling.models import (
    ObstacleState,
    RobotState,
    finite_float,
    positive_float,
    readonly_array,
    safe_uid,
)
from experiments.static.esdf import signed_distance_from_occupancy


REQUIRED_SCENARIO_FAMILIES = frozenset(
    {
        "unobstructed",
        "static_sparse",
        "static_dense",
        "narrow_passage",
        "malformed_detour",
        "dynamic_crossing",
        "dynamic_head_on",
        "dynamic_sudden_appearance",
    }
)
STATE_FACTORS = (
    "position_xyz",
    "velocity_xyz_mps",
    "acceleration_xyz_mps2",
    "yaw_rad",
    "yaw_rate_radps",
)


@dataclass(frozen=True)
class StaticRectangle:
    obstacle_uid: str
    bounds_xy: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "obstacle_uid", safe_uid(self.obstacle_uid, "obstacle_uid"))
        bounds = readonly_array(self.bounds_xy, name="bounds_xy", shape=(4,))
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError("rectangle bounds must have positive area")
        object.__setattr__(self, "bounds_xy", bounds)


@dataclass(frozen=True)
class MovingDisc:
    obstacle_uid: str
    radius_m: float
    keyframes: tuple[tuple[float, float, float], ...]
    active_from_s: float = 0.0
    active_until_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "obstacle_uid", safe_uid(self.obstacle_uid, "obstacle_uid"))
        object.__setattr__(self, "radius_m", positive_float(self.radius_m, "radius_m"))
        rows = tuple(tuple(finite_float(item, "keyframe") for item in row) for row in self.keyframes)
        if len(rows) < 2 or any(len(row) != 3 for row in rows):
            raise ValueError("moving obstacle keyframes require at least two (time,x,y) rows")
        if any(right[0] <= left[0] for left, right in zip(rows, rows[1:])):
            raise ValueError("moving obstacle keyframes must be strictly monotonic")
        active_from = finite_float(self.active_from_s, "active_from_s")
        active_until = rows[-1][0] if self.active_until_s is None else finite_float(self.active_until_s, "active_until_s")
        if active_from < rows[0][0] or active_until > rows[-1][0] or active_until < active_from:
            raise ValueError("moving obstacle active interval must lie within keyframes")
        object.__setattr__(self, "keyframes", rows)
        object.__setattr__(self, "active_from_s", active_from)
        object.__setattr__(self, "active_until_s", active_until)

    def state_at(self, time_s: float) -> ObstacleState:
        time_s = finite_float(time_s, "time_s")
        rows = self.keyframes
        if time_s <= rows[0][0]:
            left, right = rows[0], rows[1]
            center = left[1:3]
        elif time_s >= rows[-1][0]:
            left, right = rows[-2], rows[-1]
            center = right[1:3]
        else:
            left, right = next(
                (left, right)
                for left, right in zip(rows, rows[1:])
                if left[0] <= time_s <= right[0]
            )
            ratio = (time_s - left[0]) / (right[0] - left[0])
            center = (
                left[1] + ratio * (right[1] - left[1]),
                left[2] + ratio * (right[2] - left[2]),
            )
        velocity = (
            (right[1] - left[1]) / (right[0] - left[0]),
            (right[2] - left[2]) / (right[0] - left[0]),
        )
        return ObstacleState(
            self.obstacle_uid,
            center,
            self.radius_m,
            velocity,
            True,
        )

    def active_at(self, time_s: float) -> bool:
        return self.active_from_s <= time_s <= float(self.active_until_s)


@dataclass(frozen=True)
class RollingScenario:
    scenario_uid: str
    family: str
    world_bounds_xy: np.ndarray
    resolution_m: float
    guide_path_xyz: np.ndarray
    final_goal_xyz: np.ndarray
    initial_state: RobotState
    static_rectangles: tuple[StaticRectangle, ...] = ()
    moving_discs: tuple[MovingDisc, ...] = ()
    max_experiment_time_s: float = 20.0
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_uid", safe_uid(self.scenario_uid, "scenario_uid"))
        family = safe_uid(self.family, "family")
        if family not in REQUIRED_SCENARIO_FAMILIES:
            raise ValueError(f"unsupported rolling scene family: {family}")
        object.__setattr__(self, "family", family)
        bounds = readonly_array(self.world_bounds_xy, name="world_bounds_xy", shape=(4,))
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError("world bounds must have positive area")
        object.__setattr__(self, "world_bounds_xy", bounds)
        object.__setattr__(self, "resolution_m", positive_float(self.resolution_m, "resolution_m"))
        guide = readonly_array(self.guide_path_xyz, name="guide_path_xyz", ndim=2, min_rows=2)
        if guide.shape[1] != 3:
            raise ValueError("guide_path_xyz must have shape (N>=2, 3)")
        object.__setattr__(self, "guide_path_xyz", guide)
        goal = readonly_array(self.final_goal_xyz, name="final_goal_xyz", shape=(3,))
        object.__setattr__(self, "final_goal_xyz", goal)
        if not np.allclose(goal, guide[-1], rtol=0.0, atol=1e-12):
            raise ValueError("final goal must equal the final guide point")
        if not isinstance(self.initial_state, RobotState):
            raise TypeError("initial_state must be RobotState")
        rectangles = tuple(self.static_rectangles)
        moving = tuple(self.moving_discs)
        uids = [row.obstacle_uid for row in rectangles] + [row.obstacle_uid for row in moving]
        if len(set(uids)) != len(uids):
            raise ValueError("duplicate obstacle UIDs are forbidden")
        object.__setattr__(self, "static_rectangles", rectangles)
        object.__setattr__(self, "moving_discs", moving)
        maximum = positive_float(self.max_experiment_time_s, "max_experiment_time_s")
        object.__setattr__(self, "max_experiment_time_s", maximum)
        object.__setattr__(self, "seed", int(self.seed))
        xmin, ymin, xmax, ymax = bounds
        if np.any(guide[:, 0] < xmin) or np.any(guide[:, 0] > xmax) or np.any(guide[:, 1] < ymin) or np.any(guide[:, 1] > ymax):
            raise ValueError("guide lies outside declared world")
        if not (xmin <= self.initial_state.position_xyz[0] <= xmax and ymin <= self.initial_state.position_xyz[1] <= ymax):
            raise ValueError("initial state lies outside declared world")
        for rectangle in rectangles:
            x0, y0, x1, y1 = rectangle.bounds_xy
            if x0 < xmin or y0 < ymin or x1 > xmax or y1 > ymax:
                raise ValueError(f"rectangle {rectangle.obstacle_uid} lies outside declared world")
        for disc in moving:
            if disc.keyframes[0][0] < 0.0 or disc.keyframes[-1][0] > maximum:
                raise ValueError(f"dynamic motion {disc.obstacle_uid} lies outside experiment time range")
            for _, x, y in disc.keyframes:
                if x - disc.radius_m < xmin or x + disc.radius_m > xmax or y - disc.radius_m < ymin or y + disc.radius_m > ymax:
                    raise ValueError(f"dynamic obstacle {disc.obstacle_uid} lies outside declared world")

    def obstacles_at(self, time_s: float) -> tuple[ObstacleState, ...]:
        time_s = finite_float(time_s, "time_s")
        if time_s < 0.0 or time_s > self.max_experiment_time_s:
            raise ValueError("obstacle query time lies outside experiment range")
        return tuple(disc.state_at(time_s) for disc in self.moving_discs if disc.active_at(time_s))


@dataclass(frozen=True)
class WorldSnapshot:
    occupancy: np.ndarray
    esdf_distance: np.ndarray
    origin_xy: np.ndarray
    resolution_m: float
    obstacles: tuple[ObstacleState, ...]

    def __post_init__(self) -> None:
        occupancy = np.asarray(self.occupancy, dtype=np.bool_)
        if occupancy.ndim != 2 or occupancy.size == 0:
            raise ValueError("occupancy must be a nonempty 2D array")
        occupancy = np.ascontiguousarray(occupancy).copy()
        occupancy.setflags(write=False)
        distance = readonly_array(self.esdf_distance, name="esdf_distance", ndim=2)
        if distance.shape != occupancy.shape:
            raise ValueError("ESDF shape must match occupancy")
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "esdf_distance", distance)
        object.__setattr__(self, "origin_xy", readonly_array(self.origin_xy, name="origin_xy", shape=(2,)))
        object.__setattr__(self, "resolution_m", positive_float(self.resolution_m, "resolution_m"))
        object.__setattr__(self, "obstacles", tuple(self.obstacles))


@dataclass(frozen=True)
class InitialStateSweep:
    sweep_uid: str
    factor: str
    baseline: RobotState
    variant: RobotState

    def __post_init__(self) -> None:
        object.__setattr__(self, "sweep_uid", safe_uid(self.sweep_uid, "sweep_uid"))
        if self.factor not in STATE_FACTORS:
            raise ValueError(f"unsupported state sweep factor: {self.factor}")
        if changed_fields(self.baseline, self.variant) != {self.factor}:
            raise ValueError("initial-state sweep must change exactly one declared factor")


def _robot_state(payload: Mapping[str, object]) -> RobotState:
    return RobotState(
        payload["position_xyz"],
        payload["velocity_xyz_mps"],
        payload["acceleration_xyz_mps2"],
        payload["yaw_rad"],
        payload["yaw_rate_radps"],
    )


def changed_fields(left: RobotState, right: RobotState) -> set[str]:
    changed = set()
    for field in STATE_FACTORS:
        a, b = getattr(left, field), getattr(right, field)
        if isinstance(a, np.ndarray):
            if not np.array_equal(a, b):
                changed.add(field)
        elif a != b:
            changed.add(field)
    return changed


def _scenario(payload: Mapping[str, object]) -> RollingScenario:
    rectangles = tuple(
        StaticRectangle(row["obstacle_uid"], row["bounds_xy"])
        for row in payload.get("static_rectangles", [])
    )
    moving = tuple(
        MovingDisc(
            obstacle_uid=row["obstacle_uid"],
            radius_m=row["radius_m"],
            keyframes=tuple(tuple(value) for value in row["keyframes"]),
            active_from_s=row.get("active_from_s", 0.0),
            active_until_s=row.get("active_until_s"),
        )
        for row in payload.get("moving_discs", [])
    )
    guide = payload.get("guide_path_xyz")
    if guide is None:
        raise ValueError("rolling scenario is missing guide_path_xyz")
    goal = payload.get("final_goal_xyz")
    if goal is None:
        raise ValueError("rolling scenario is missing final goal")
    return RollingScenario(
        scenario_uid=payload["scenario_uid"],
        family=payload["family"],
        world_bounds_xy=payload["world_bounds_xy"],
        resolution_m=payload["resolution_m"],
        guide_path_xyz=guide,
        final_goal_xyz=goal,
        initial_state=_robot_state(payload["initial_state"]),
        static_rectangles=rectangles,
        moving_discs=moving,
        max_experiment_time_s=payload["max_experiment_time_s"],
        seed=payload.get("seed", 0),
    )


def load_showcase_config(path: Path | str) -> dict[str, object]:
    path = Path(path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable rolling showcase config {path}: {error}") from error
    if payload.get("schema_version") != 1:
        raise ValueError("rolling showcase schema_version must be 1")
    scenarios = tuple(_scenario(row) for row in payload.get("scenarios", []))
    if {scenario.family for scenario in scenarios} != REQUIRED_SCENARIO_FAMILIES:
        raise ValueError("rolling showcase must define exactly the eight required families")
    if len({scenario.scenario_uid for scenario in scenarios}) != len(scenarios):
        raise ValueError("rolling showcase contains duplicate scenario UIDs")
    rollout = payload.get("rollout", {})
    from experiments.rolling.models import RolloutConfig

    RolloutConfig(**rollout)
    initial_state_sweeps(payload)
    return payload


def load_scenarios(path: Path | str) -> dict[str, RollingScenario]:
    payload = load_showcase_config(path)
    scenarios = tuple(_scenario(row) for row in payload["scenarios"])
    return {scenario.scenario_uid: scenario for scenario in scenarios}


def initial_state_sweeps(config: Mapping[str, object]) -> tuple[InitialStateSweep, ...]:
    sweeps = []
    for group in config.get("initial_state_sweeps", []):
        baseline = _robot_state(group["baseline"])
        for index, variant_payload in enumerate(group.get("variants", [])):
            sweeps.append(
                InitialStateSweep(
                    sweep_uid=str(group.get("sweep_uid", group["factor"])) + f"_{index:02d}",
                    factor=group["factor"],
                    baseline=baseline,
                    variant=_robot_state(variant_payload),
                )
            )
    if {row.factor for row in sweeps} != set(STATE_FACTORS):
        raise ValueError("initial-state sweeps must cover all five state factors")
    return tuple(sweeps)


def materialize_world(scenario: RollingScenario, time_s: float) -> WorldSnapshot:
    obstacles = scenario.obstacles_at(time_s)
    xmin, ymin, xmax, ymax = scenario.world_bounds_xy
    width = int(math.ceil((xmax - xmin) / scenario.resolution_m))
    height = int(math.ceil((ymax - ymin) / scenario.resolution_m))
    xs = xmin + (np.arange(width) + 0.5) * scenario.resolution_m
    ys = ymin + (np.arange(height) + 0.5) * scenario.resolution_m
    grid_x, grid_y = np.meshgrid(xs, ys)
    occupancy = np.zeros((height, width), dtype=bool)
    occupancy[[0, -1], :] = True
    occupancy[:, [0, -1]] = True
    for rectangle in scenario.static_rectangles:
        x0, y0, x1, y1 = rectangle.bounds_xy
        occupancy |= (grid_x >= x0) & (grid_x <= x1) & (grid_y >= y0) & (grid_y <= y1)
    for obstacle in obstacles:
        occupancy |= (
            (grid_x - obstacle.center_xy[0]) ** 2
            + (grid_y - obstacle.center_xy[1]) ** 2
            <= obstacle.radius_m**2
        )
    esdf = signed_distance_from_occupancy(occupancy, scenario.resolution_m)
    return WorldSnapshot(occupancy, esdf, [xmin, ymin], scenario.resolution_m, obstacles)


__all__ = [
    "InitialStateSweep",
    "MovingDisc",
    "REQUIRED_SCENARIO_FAMILIES",
    "RollingScenario",
    "StaticRectangle",
    "WorldSnapshot",
    "changed_fields",
    "initial_state_sweeps",
    "load_scenarios",
    "load_showcase_config",
    "materialize_world",
]
