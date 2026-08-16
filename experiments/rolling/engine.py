"""Deterministic full-route rolling local-planning execution."""

from __future__ import annotations

import math
from typing import Callable, Mapping

import numpy as np

from experiments.rolling.models import (
    MINCO_SAMPLE_COLUMNS,
    RobotState,
    RolloutConfig,
    RolloutCycle,
    RolloutResult,
)
try:
    from experiments.rolling.scenarios import materialize_world
except ImportError:  # Allows model/engine tests while the scenario module lands.
    def materialize_world(*_args, **_kwargs):
        raise RuntimeError("experiments.rolling.scenarios is unavailable")


Planner = Callable[..., object]
PLANNED_STATE_TOLERANCE = 1e-3


def _normalise_diagnostic(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _normalise_diagnostic(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        if np.all(np.isfinite(value)):
            return value
        return tuple(_normalise_diagnostic(item) for item in value.tolist())
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.generic):
        return _normalise_diagnostic(value.item())
    if isinstance(value, (list, tuple)):
        return tuple(_normalise_diagnostic(item) for item in value)
    return value


def _planned_state_error(samples: np.ndarray, state: RobotState) -> float:
    first = samples[0]
    requested = np.concatenate(
        (
            state.position_xyz,
            state.velocity_xyz_mps,
            state.acceleration_xyz_mps2,
            [state.yaw_rad, state.yaw_rate_radps],
        )
    )
    planned = np.concatenate((first[1:10], first[13:15]))
    return float(np.max(np.abs(planned - requested)))


def _guide_deviation_max(path_xyz: np.ndarray, guide_xyz: np.ndarray) -> float:
    maximum = 0.0
    for point in path_xyz:
        best = math.inf
        for start, end in zip(guide_xyz, guide_xyz[1:]):
            delta = end - start
            denominator = float(np.dot(delta, delta))
            ratio = 0.0 if denominator <= 1e-12 else float(
                np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0)
            )
            best = min(best, float(np.linalg.norm(point - (start + ratio * delta))))
        maximum = max(maximum, best)
    return maximum


def _polyline_arclength(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    deltas = np.diff(path, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return lengths, cumulative


def _point_at_progress(
    path: np.ndarray,
    lengths: np.ndarray,
    cumulative: np.ndarray,
    progress_s: float,
) -> np.ndarray:
    progress = min(max(0.0, float(progress_s)), float(cumulative[-1]))
    segment = min(
        int(np.searchsorted(cumulative, progress, side="right") - 1),
        len(lengths) - 1,
    )
    if lengths[segment] <= 1e-12:
        return path[segment + 1].copy()
    ratio = (progress - cumulative[segment]) / lengths[segment]
    return path[segment] + ratio * (path[segment + 1] - path[segment])


def _closest_forward_progress(
    path: np.ndarray,
    position: np.ndarray,
    minimum_progress_s: float,
) -> float:
    lengths, cumulative = _polyline_arclength(path)
    best_distance = math.inf
    best_progress = min(max(0.0, minimum_progress_s), float(cumulative[-1]))
    for index, length in enumerate(lengths):
        if length <= 1e-12:
            continue
        delta = path[index + 1] - path[index]
        ratio = float(np.dot(position - path[index], delta) / (length * length))
        ratio = min(1.0, max(0.0, ratio))
        progress = float(cumulative[index] + ratio * length)
        if progress + 1e-12 < minimum_progress_s:
            continue
        projected = path[index] + ratio * delta
        distance = float(np.linalg.norm(projected - position))
        if distance < best_distance:
            best_distance = distance
            best_progress = progress
    return max(float(minimum_progress_s), best_progress)


def select_local_guide(
    guide_path_xyz: object,
    current_position_xyz: object,
    progress_s: float,
    horizon_m: float,
) -> tuple[np.ndarray, float]:
    """Select a forward-only guide slice capped by arclength horizon."""

    guide = np.asarray(guide_path_xyz, dtype=np.float64)
    position = np.asarray(current_position_xyz, dtype=np.float64)
    if guide.ndim != 2 or guide.shape[0] < 2 or guide.shape[1] != 3:
        raise ValueError("guide_path_xyz must have shape (N>=2, 3)")
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("current_position_xyz must be finite shape (3,)")
    if not np.all(np.isfinite(guide)):
        raise ValueError("guide_path_xyz must be finite")
    horizon = float(horizon_m)
    minimum_progress = float(progress_s)
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon_m must be finite and positive")
    if not math.isfinite(minimum_progress) or minimum_progress < 0.0:
        raise ValueError("progress_s must be finite and non-negative")

    lengths, cumulative = _polyline_arclength(guide)
    if cumulative[-1] <= 1e-12:
        raise ValueError("guide_path_xyz must have positive arclength")
    progress = _closest_forward_progress(guide, position, minimum_progress)
    end_progress = min(float(cumulative[-1]), progress + horizon)
    points = [_point_at_progress(guide, lengths, cumulative, progress)]
    for index, vertex_progress in enumerate(cumulative[1:-1], start=1):
        if progress + 1e-12 < vertex_progress < end_progress - 1e-12:
            points.append(guide[index].copy())
    points.append(_point_at_progress(guide, lengths, cumulative, end_progress))
    local = np.asarray(points, dtype=np.float64)
    if len(local) < 2 or np.linalg.norm(local[-1] - local[0]) <= 1e-12:
        local = np.vstack([local[0], guide[-1]])
    return local, progress


def select_execution_prefix(
    samples: object,
    execute_duration_s: float,
    final_goal_xyz: object,
    goal_tolerance_m: float,
) -> np.ndarray:
    """Return an exact candidate prefix through the execution time boundary."""

    trajectory = np.asarray(samples, dtype=np.float64)
    goal = np.asarray(final_goal_xyz, dtype=np.float64)
    duration = float(execute_duration_s)
    tolerance = float(goal_tolerance_m)
    if (
        trajectory.ndim != 2
        or trajectory.shape[0] < 1
        or trajectory.shape[1] != MINCO_SAMPLE_COLUMNS
        or not np.all(np.isfinite(trajectory))
    ):
        raise ValueError(f"samples must be finite shape (N>=1, {MINCO_SAMPLE_COLUMNS})")
    if goal.shape != (3,) or not np.all(np.isfinite(goal)):
        raise ValueError("final_goal_xyz must be finite shape (3,)")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("execute_duration_s must be finite and positive")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("goal_tolerance_m must be finite and positive")
    if np.any(np.diff(trajectory[:, 0]) <= 0.0):
        raise ValueError("candidate sample times must be strictly increasing")

    goal_rows = np.flatnonzero(
        np.linalg.norm(trajectory[:, 1:4] - goal, axis=1) <= tolerance
    )
    time_rows = np.flatnonzero(
        trajectory[:, 0] <= trajectory[0, 0] + duration + 1e-12
    )
    end_index = int(time_rows[-1]) if len(time_rows) else 0
    if len(goal_rows) and goal_rows[0] <= end_index + 1:
        end_index = int(goal_rows[0])
    if end_index == 0 and len(trajectory) > 1:
        end_index = 1
    return trajectory[: end_index + 1].copy()


def _corridor_segments(diagnostics: Mapping[str, object]) -> np.ndarray:
    raw = diagnostics.get("corridor_segments")
    if raw is None:
        return np.empty((0, 5), dtype=np.float64)
    segments = np.asarray(raw, dtype=np.float64)
    if segments.size == 0:
        return np.empty((0, 5), dtype=np.float64)
    if segments.ndim != 2 or segments.shape[1] not in {5, 8}:
        raise ValueError("corridor_segments must have shape (N, 5) or native (N, 8)")
    if segments.shape[1] == 8:
        segments = segments[:, [0, 1, 3, 4, 6]]
    if not np.all(np.isfinite(segments)) or np.any(segments[:, 4] <= 0.0):
        raise ValueError("corridor_segments must be finite with positive radii")
    return segments.copy()


def _collides(
    executed: np.ndarray,
    world: object,
    collision_distance_m: float,
) -> bool:
    for obstacle in tuple(world.obstacles):
        threshold = float(obstacle.radius_m) + collision_distance_m
        distances = np.linalg.norm(
            executed[:, 1:3] - np.asarray(obstacle.center_xy), axis=1
        )
        if np.any(distances <= threshold):
            return True
    distance = np.asarray(world.esdf_distance, dtype=np.float64)
    origin = np.asarray(world.origin_xy, dtype=np.float64)
    resolution = float(world.resolution_m)
    cells_xy = np.floor((executed[:, 1:3] - origin) / resolution).astype(int)
    outside = (
        (cells_xy[:, 0] < 0)
        | (cells_xy[:, 0] >= distance.shape[1])
        | (cells_xy[:, 1] < 0)
        | (cells_xy[:, 1] >= distance.shape[0])
    )
    if np.any(outside):
        return True
    clearances = distance[cells_xy[:, 1], cells_xy[:, 0]]
    if np.any(clearances <= collision_distance_m):
        return True
    return False


def run_rollout(
    scenario: object,
    *,
    method: str,
    profile: Mapping[str, object],
    config: RolloutConfig,
    planner: Planner | None = None,
    reset_history_each_cycle: bool = False,
) -> RolloutResult:
    """Execute fixed-prefix local plans until one fixed terminal class applies.

    ``reset_history_each_cycle`` is intentionally only a reproducibility mode
    for static full-route figures.  It prevents native wall-clock hot-start
    age from affecting a recorded route; real closed-loop simulation keeps the
    default history-reuse behaviour.
    """

    if method not in {"legacy", "superplanner_sfc_v1", "safe_corridor_v1"}:
        raise ValueError(f"unsupported rollout method: {method}")
    if not isinstance(config, RolloutConfig):
        raise TypeError("config must be RolloutConfig")
    if planner is None:
        from experiments.static.runner import create_native_planner

        planner = create_native_planner(profile)
    state = scenario.initial_state
    if not isinstance(state, RobotState):
        raise TypeError("scenario.initial_state must be RobotState")
    final_goal = np.asarray(scenario.final_goal_xyz, dtype=np.float64)
    guide = np.asarray(scenario.guide_path_xyz, dtype=np.float64)
    _, guide_cumulative = _polyline_arclength(guide)
    progress_s = 0.0
    time_s = 0.0
    cycles: list[RolloutCycle] = []
    status = "MAX_CYCLES"
    executed_ends: list[np.ndarray] = []
    replanning_failures = 0

    for cycle_index in range(config.max_cycles):
        world = materialize_world(scenario, time_s)
        local_guide, progress_s = select_local_guide(
            guide, state.position_xyz, progress_s, config.local_horizon_m
        )
        local_goal = local_guide[-1].copy()
        final_inside_horizon = (
            float(guide_cumulative[-1]) - progress_s
            <= config.local_horizon_m + 1e-9
        )
        terminal_goal = final_goal if final_inside_horizon else None
        planner_arguments = dict(
            guide_path_xyz=local_guide,
            world=world,
            state=state,
            terminal_goal_xyz=terminal_goal,
            profile=profile,
            reset_history=reset_history_each_cycle or cycle_index == 0,
        )
        plan = planner(**planner_arguments)
        initial_samples = np.asarray(
            getattr(plan, "samples", np.empty((0, MINCO_SAMPLE_COLUMNS)))
        )
        planned_start_error = math.inf
        if (
            initial_samples.ndim == 2
            and len(initial_samples)
            and (planned_start_error := _planned_state_error(initial_samples, state))
            > PLANNED_STATE_TOLERANCE
        ):
            reset = getattr(planner, "reset_execution_history", None)
            if callable(reset):
                reset()
                # The native planner must also receive the reset request.  A
                # bare Python-side reset is insufficient when its processor
                # retains a stale proposal: it can otherwise return a cold
                # plan whose first state is not the executed state.  This is
                # especially visible late in long, full-route rollouts.
                planner_arguments["reset_history"] = True
                plan = planner(**planner_arguments)
                initial_samples = np.asarray(
                    getattr(plan, "samples", np.empty((0, MINCO_SAMPLE_COLUMNS)))
                )
                planned_start_error = (
                    _planned_state_error(initial_samples, state)
                    if initial_samples.ndim == 2 and len(initial_samples)
                    else math.inf
                )
        diagnostics = {
            str(key): _normalise_diagnostic(value)
            for key, value in dict(getattr(plan, "diagnostics", {})).items()
        }
        plan_status = str(getattr(plan, "status", "FAILED"))
        samples = np.asarray(getattr(plan, "samples", np.empty((0, 15))))
        success = plan_status in {"SUCCEEDED", "SUCCESS"} and bool(
            diagnostics.get("success", True)
        )
        if success and planned_start_error > PLANNED_STATE_TOLERANCE:
            success = False
            diagnostics.update(
                {
                    "success": False,
                    "failure_reason": "PLANNED_START_STATE_DISCONTINUITY",
                    "planned_start_state_error": planned_start_error,
                }
            )
        if not success or samples.size == 0:
            replanning_failures += 1
            anchor = np.zeros((1, MINCO_SAMPLE_COLUMNS), dtype=np.float64)
            anchor[0, 0] = time_s
            anchor[0, 1:4] = state.position_xyz
            anchor[0, 4:7] = state.velocity_xyz_mps
            anchor[0, 7:10] = state.acceleration_xyz_mps2
            anchor[0, 13] = state.yaw_rad
            anchor[0, 14] = state.yaw_rate_radps
            samples = anchor
            executed = anchor.copy()
            status = "OPTIMIZATION_FAILED"
        else:
            samples = samples.copy()
            samples[:, 0] = time_s + (samples[:, 0] - samples[0, 0])
            executed = select_execution_prefix(
                samples,
                config.execute_duration_s,
                final_goal,
                config.goal_tolerance_m,
            )
            commit = getattr(planner, "commit_execution", None)
            if callable(commit):
                commit(plan, float(executed[-1, 0] - samples[0, 0]))
        cycle = RolloutCycle(
            cycle_index=cycle_index,
            time_s=time_s,
            input_state=state,
            local_guide_xyz=local_guide,
            local_goal_xyz=local_goal,
            candidate_samples=samples,
            executed_samples=executed,
            corridor_segments=_corridor_segments(diagnostics),
            obstacle_states=tuple(world.obstacles),
            diagnostics={**diagnostics, "planner_status": plan_status},
        )
        cycles.append(cycle)
        executed_ends.append(executed[-1, 1:4].copy())

        if status == "OPTIMIZATION_FAILED":
            break
        state = RobotState.from_minco_sample(executed[-1])
        if _collides(executed, world, config.collision_distance_m):
            status = "COLLISION"
            break
        if np.linalg.norm(state.position_xyz - final_goal) <= config.goal_tolerance_m:
            status = "GOAL_REACHED"
            break
        if len(executed_ends) >= config.stall_window_cycles:
            window = executed_ends[-config.stall_window_cycles :]
            if np.linalg.norm(window[-1] - window[0]) <= config.stall_distance_m:
                status = "STALLED"
                break
        time_s += config.planning_period_s
        if time_s >= config.max_time_s - 1e-12:
            status = "TIMEOUT"
            break
    else:
        status = "MAX_CYCLES"

    accumulated = np.concatenate(
        [cycles[0].executed_samples]
        + [cycle.executed_samples[1:] for cycle in cycles[1:]],
        axis=0,
    )
    final_error = float(np.linalg.norm(accumulated[-1, 1:4] - final_goal))
    clearance_values = [
        float(value)
        for cycle in cycles
        for value in (cycle.diagnostics.get("validation_min_clearance"),)
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    result = RolloutResult(
        scenario_uid=scenario.scenario_uid,
        method=method,
        status=status,
        cycles=tuple(cycles),
        executed_samples=accumulated,
        final_goal_xyz=final_goal,
        metrics={
            "termination_reason": status,
            "final_error_m": final_error,
            "cycle_count": len(cycles),
            "replanning_failure_count": replanning_failures,
            "simulated_time_s": cycles[-1].time_s,
            "guide_deviation_max_m": _guide_deviation_max(
                accumulated[:, 1:4], guide
            ),
            "min_clearance_m": min(clearance_values) if clearance_values else None,
        },
        goal_tolerance_m=config.goal_tolerance_m,
    )
    errors = result.validate()
    if errors:
        raise RuntimeError("rolling result validation failed: " + "; ".join(errors))
    return result


__all__ = ["run_rollout", "select_execution_prefix", "select_local_guide"]
