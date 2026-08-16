from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.rolling.models import (
    ObstacleState,
    RobotState,
    RolloutConfig,
)


def _state(x: float = 0.0) -> RobotState:
    return RobotState(
        position_xyz=[x, 0.0, 0.0],
        velocity_xyz_mps=[0.0, 0.0, 0.0],
        acceleration_xyz_mps2=[0.0, 0.0, 0.0],
        yaw_rad=0.0,
        yaw_rate_radps=0.0,
    )


@dataclass(frozen=True)
class _Scenario:
    scenario_uid: str
    guide_path_xyz: np.ndarray
    final_goal_xyz: np.ndarray
    initial_state: RobotState


def _scenario(length: float = 4.0) -> _Scenario:
    return _Scenario(
        scenario_uid="rolling-straight",
        guide_path_xyz=np.asarray(
            [[0.0, 0.0, 0.0], [length / 2.0, 0.0, 0.0], [length, 0.0, 0.0]]
        ),
        final_goal_xyz=np.asarray([length, 0.0, 0.0]),
        initial_state=_state(),
    )


def _world(*, obstacles=(), clearance: float = 100.0):
    return SimpleNamespace(
        occupancy=np.zeros((32, 32), dtype=bool),
        esdf_distance=np.full((32, 32), clearance, dtype=float),
        origin_xy=np.asarray([-8.0, -8.0]),
        resolution_m=0.5,
        obstacles=tuple(obstacles),
    )


class _LinearPlanner:
    def __init__(self, *, fail_at: int | None = None, stationary: bool = False, nan_diagnostic: bool = False):
        self.calls = []
        self.fail_at = fail_at
        self.stationary = stationary
        self.nan_diagnostic = nan_diagnostic

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_at is not None and len(self.calls) - 1 == self.fail_at:
            return SimpleNamespace(
                status="FAILED",
                diagnostics={"failure_reason": "FIXTURE_OPTIMIZATION_FAILED"},
                samples=np.empty((0, 15)),
                waypoints=np.empty((0, 3)),
            )
        state = kwargs["state"]
        target = np.asarray(kwargs["guide_path_xyz"][-1], dtype=float)
        start = state.position_xyz
        if self.stationary:
            target = start
        samples = np.zeros((3, 15), dtype=float)
        samples[:, 0] = [0.0, 0.25, 0.5]
        samples[:, 1:4] = np.linspace(start, target, 3)
        velocity = (target - start) / 0.5
        samples[:, 4:7] = velocity
        samples[0, 4:7] = state.velocity_xyz_mps
        samples[0, 7:10] = state.acceleration_xyz_mps2
        samples[:, 13] = state.yaw_rad
        samples[:, 14] = state.yaw_rate_radps
        corridor = np.asarray(
            [[start[0], start[1], target[0], target[1], 0.35]], dtype=float
        )
        return SimpleNamespace(
            status="SUCCEEDED",
            diagnostics={
                "success": True,
                "failure_reason": "NONE",
                "corridor_segments": corridor,
                "optimizer_iteration_count": 4,
                "history_age_s": float("nan") if self.nan_diagnostic else None,
            },
            samples=samples,
            waypoints=np.asarray([start, target]),
        )


class _MismatchedStartPlanner(_LinearPlanner):
    """Fixture for a native proposal that cannot start from executed state."""

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        result.samples[0, 1] += 0.01
        return result


def _config(**overrides) -> RolloutConfig:
    values = {
        "planning_period_s": 0.5,
        "execute_duration_s": 0.5,
        "local_horizon_m": 1.0,
        "max_cycles": 20,
        "max_time_s": 20.0,
        "stall_window_cycles": 3,
        "stall_distance_m": 0.01,
        "collision_distance_m": 0.0,
        "goal_tolerance_m": 0.01,
    }
    values.update(overrides)
    return RolloutConfig(**values)


class RollingEngineTests(unittest.TestCase):
    def test_rollout_fails_closed_when_native_candidate_start_is_discontinuous(self):
        from experiments.rolling.engine import run_rollout

        with patch("experiments.rolling.engine.materialize_world", return_value=_world()):
            result = run_rollout(
                _scenario(), method="legacy", profile={}, config=_config(),
                planner=_MismatchedStartPlanner(),
            )

        self.assertEqual(result.status, "OPTIMIZATION_FAILED")
        self.assertEqual(result.cycles[0].diagnostics["failure_reason"], "PLANNED_START_STATE_DISCONTINUITY")
        self.assertEqual(result.validate(), [])

    def test_rollout_executes_prefixes_until_full_guide_goal(self):
        from experiments.rolling.engine import run_rollout

        planner = _LinearPlanner()
        with patch("experiments.rolling.engine.materialize_world", return_value=_world()):
            result = run_rollout(
                _scenario(), method="legacy", profile={}, config=_config(), planner=planner
            )

        self.assertEqual(result.status, "GOAL_REACHED")
        self.assertGreater(len(result.cycles), 1)
        np.testing.assert_allclose(result.executed_samples[-1, 1:4], [4, 0, 0])
        self.assertTrue(np.all(np.diff(result.executed_samples[:, 0]) > 0.0))
        self.assertTrue(
            all(
                cycle.candidate_samples[0, 0] == cycle.time_s
                for cycle in result.cycles
            )
        )
        self.assertEqual(result.validate(), [])
        self.assertIsNone(planner.calls[0]["terminal_goal_xyz"])
        np.testing.assert_allclose(planner.calls[-1]["terminal_goal_xyz"], [4, 0, 0])

    def test_next_cycle_receives_previous_executed_end_state(self):
        from experiments.rolling.engine import run_rollout

        planner = _LinearPlanner()
        with patch("experiments.rolling.engine.materialize_world", return_value=_world()):
            result = run_rollout(
                _scenario(2.5), method="legacy", profile={}, config=_config(), planner=planner
            )

        self.assertTrue(planner.calls[0]["reset_history"])
        self.assertTrue(all(not call["reset_history"] for call in planner.calls[1:]))
        for left, right in zip(result.cycles, result.cycles[1:]):
            end = left.executed_samples[-1]
            np.testing.assert_allclose(right.input_state.position_xyz, end[1:4])
            np.testing.assert_allclose(right.input_state.velocity_xyz_mps, end[4:7])
            np.testing.assert_allclose(right.input_state.acceleration_xyz_mps2, end[7:10])
            self.assertAlmostEqual(right.input_state.yaw_rad, end[13])
            self.assertAlmostEqual(right.input_state.yaw_rate_radps, end[14])

    def test_cold_replan_mode_resets_native_history_on_every_cycle(self):
        from experiments.rolling.engine import run_rollout

        planner = _LinearPlanner()
        with patch("experiments.rolling.engine.materialize_world", return_value=_world()):
            result = run_rollout(
                _scenario(2.5),
                method="legacy",
                profile={},
                config=_config(),
                planner=planner,
                reset_history_each_cycle=True,
            )

        self.assertEqual(result.status, "GOAL_REACHED")
        self.assertGreater(len(planner.calls), 1)
        self.assertTrue(all(call["reset_history"] for call in planner.calls))

    def test_local_guide_progress_never_backtracks(self):
        from experiments.rolling.engine import select_local_guide

        guide = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
        first, progress = select_local_guide(guide, [0.8, 0.2, 0], 0.0, 1.0)
        second, next_progress = select_local_guide(guide, [0.5, 0, 0], progress, 1.0)

        self.assertGreaterEqual(next_progress, progress)
        self.assertGreaterEqual(second[0, 0], first[0, 0] - 1e-12)
        self.assertLessEqual(np.linalg.norm(np.diff(first, axis=0), axis=1).sum(), 1.0 + 1e-9)

    def test_dynamic_world_rebuilds_and_real_corridor_is_captured(self):
        from experiments.rolling.engine import run_rollout

        planner = _LinearPlanner()
        sampled_times = []

        def moving_world(_scenario, time_s):
            sampled_times.append(time_s)
            obstacle = ObstacleState(
                "moving", [time_s, 3.0], 0.2, [1.0, 0.0], True
            )
            return _world(obstacles=(obstacle,))

        with patch("experiments.rolling.engine.materialize_world", side_effect=moving_world):
            result = run_rollout(
                _scenario(2.0),
                method="safe_corridor_v1",
                profile={"constraint_profile": "safe_corridor_v1"},
                config=_config(),
                planner=planner,
            )

        self.assertEqual(sampled_times, [cycle.time_s for cycle in result.cycles])
        self.assertNotEqual(result.cycles[0].obstacle_states, result.cycles[-1].obstacle_states)
        self.assertTrue(all(cycle.corridor_segments.shape == (1, 5) for cycle in result.cycles))
        self.assertTrue(all(cycle.corridor_segments[0, 4] == 0.35 for cycle in result.cycles))

    def test_fixed_termination_classification(self):
        from experiments.rolling.engine import run_rollout

        cases = []
        cases.append(("OPTIMIZATION_FAILED", _LinearPlanner(fail_at=0), _world(), _config()))
        collision_obstacle = ObstacleState("hit", [0.5, 0.0], 0.2)
        cases.append(("COLLISION", _LinearPlanner(), _world(obstacles=(collision_obstacle,)), _config()))
        cases.append(("COLLISION", _LinearPlanner(), _world(clearance=-0.1), _config()))
        cases.append(("STALLED", _LinearPlanner(stationary=True), _world(), _config()))
        cases.append(("MAX_CYCLES", _LinearPlanner(), _world(), _config(max_cycles=1)))
        cases.append(("TIMEOUT", _LinearPlanner(), _world(), _config(max_time_s=0.5)))

        for expected, planner, world, config in cases:
            with self.subTest(expected=expected):
                with patch("experiments.rolling.engine.materialize_world", return_value=world):
                    result = run_rollout(
                        _scenario(), method="legacy", profile={}, config=config, planner=planner
                    )
                self.assertEqual(result.status, expected)
                self.assertEqual(result.metrics["termination_reason"], expected)

    def test_native_cycle_planner_reuses_history_and_rebuilds_esdf(self):
        from experiments.static.runner import run_native_plan

        class Processor:
            def __init__(self):
                self.reset_count = 0
                self.world_count = 0
                self.starts = []

            def set_static_esdf_2d(self, distance, free, origin, resolution):
                self.world_count += 1
                self.last_resolution = resolution

            def reset_history(self):
                self.reset_count += 1

            def optimize_preview(self, guide, position, velocity, acceleration, yaw, yaw_rate, goal):
                self.starts.append(np.asarray(position).copy())
                samples = np.zeros((2, 15), dtype=float)
                samples[:, 0] = [0.0, 0.5]
                samples[0, 1:4] = position
                samples[1, 1:4] = guide[-1]
                samples[0, 4:7] = velocity
                samples[0, 7:10] = acceleration
                samples[0, 13:15] = [yaw, yaw_rate]
                return {
                    "success": True,
                    "failure_reason": "NONE",
                    "samples": samples,
                    "waypoints": np.asarray(guide),
                    "corridor_segments": np.asarray([[0, 0, 0, 1, 0, 0, 0.3, 0.4]]),
                }

        processor = Processor()
        first = run_native_plan(
            guide_path_xyz=np.asarray([[0, 0, 0], [1, 0, 0]]),
            world=_world(clearance=2.0),
            state=_state(0.0),
            terminal_goal_xyz=None,
            profile={},
            reset_history=True,
            processor=processor,
        )
        second = run_native_plan(
            guide_path_xyz=np.asarray([[1, 0, 0], [2, 0, 0]]),
            world=_world(clearance=1.0),
            state=_state(1.0),
            terminal_goal_xyz=[2, 0, 0],
            profile={},
            reset_history=False,
            processor=processor,
        )

        self.assertEqual(processor.reset_count, 1)
        self.assertEqual(processor.world_count, 2)
        self.assertEqual(first.status, "SUCCEEDED")
        self.assertEqual(second.status, "SUCCEEDED")
        self.assertEqual(second.diagnostics["corridor_segments"].shape, (1, 8))
        np.testing.assert_allclose(processor.starts, [[0, 0, 0], [1, 0, 0]])

    def test_default_native_planner_instance_is_shared_across_rollout_cycles(self):
        from experiments.rolling.engine import run_rollout

        planner = _LinearPlanner()
        with patch(
            "experiments.static.runner.create_native_planner",
            return_value=planner,
        ) as factory, patch(
            "experiments.rolling.engine.materialize_world",
            return_value=_world(),
        ):
            result = run_rollout(
                _scenario(2.5), method="legacy", profile={"marker": 7}, config=_config()
            )

        factory.assert_called_once_with({"marker": 7})
        self.assertGreater(len(result.cycles), 1)
        self.assertTrue(planner.calls[0]["reset_history"])
        self.assertTrue(all(not row["reset_history"] for row in planner.calls[1:]))

    def test_native_nonfinite_optional_diagnostic_becomes_explicit_null(self):
        from experiments.rolling.engine import run_rollout

        with patch(
            "experiments.rolling.engine.materialize_world", return_value=_world()
        ):
            result = run_rollout(
                _scenario(1.0),
                method="legacy",
                profile={},
                config=_config(),
                planner=_LinearPlanner(nan_diagnostic=True),
            )

        self.assertIsNone(result.cycles[0].diagnostics["history_age_s"])

    def test_successful_execution_prefix_is_committed_to_native_history(self):
        from experiments.rolling.engine import run_rollout

        planner = _LinearPlanner()
        planner.commits = []
        planner.commit_execution = lambda plan, applied_time: planner.commits.append(
            (plan.diagnostics["proposal_id"], applied_time)
        )
        original_call = planner.__class__.__call__

        def call_with_proposal(instance, **kwargs):
            result = original_call(instance, **kwargs)
            result.diagnostics["proposal_id"] = len(instance.calls)
            return result

        with patch.object(planner.__class__, "__call__", call_with_proposal), patch(
            "experiments.rolling.engine.materialize_world", return_value=_world()
        ):
            result = run_rollout(
                _scenario(2.0),
                method="legacy",
                profile={},
                config=_config(),
                planner=planner,
            )

        self.assertEqual(len(planner.commits), len(result.cycles))
        self.assertTrue(all(applied_time > 0.0 for _, applied_time in planner.commits))

    def test_hot_start_state_mismatch_replans_cold_before_recording(self):
        from experiments.rolling.engine import run_rollout

        class MismatchPlanner(_LinearPlanner):
            def __init__(self):
                super().__init__()
                self.reset_execution_count = 0

            def __call__(self, **kwargs):
                result = super().__call__(**kwargs)
                if len(self.calls) == 2 and not kwargs["reset_history"]:
                    result.samples[0, 4] += 0.01
                    result.diagnostics["planning_state"] = "HOT_START"
                else:
                    result.diagnostics["planning_state"] = "COLD_START"
                return result

            def reset_execution_history(self):
                self.reset_execution_count += 1

        planner = MismatchPlanner()
        with patch(
            "experiments.rolling.engine.materialize_world", return_value=_world()
        ):
            result = run_rollout(
                _scenario(2.0), method="legacy", profile={}, config=_config(), planner=planner
            )

        self.assertEqual(planner.reset_execution_count, 1)
        self.assertGreater(len(planner.calls), len(result.cycles))
        self.assertEqual(result.validate(), [])


if __name__ == "__main__":
    unittest.main()
