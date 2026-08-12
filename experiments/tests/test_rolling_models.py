from __future__ import annotations

from types import MappingProxyType
import unittest

import numpy as np


def make_state(x: float = 0.0):
    from experiments.rolling.models import RobotState

    return RobotState(
        position_xyz=[x, 0.0, 0.0],
        velocity_xyz_mps=[0.5, 0.0, 0.0],
        acceleration_xyz_mps2=[0.0, 0.0, 0.0],
        yaw_rad=0.0,
        yaw_rate_radps=0.0,
    )


def make_cycle(index: int = 0, input_x: float = 0.0, executed_end_x: float = 0.25):
    from experiments.rolling.models import RolloutCycle

    candidate = np.zeros((3, 15), dtype=float)
    candidate[:, 0] = [index, index + 0.25, index + 0.5]
    candidate[:, 1] = [input_x, executed_end_x, input_x + 0.5]
    candidate[:, 4] = 0.5
    executed = candidate[:2].copy()
    return RolloutCycle(
        cycle_index=index,
        time_s=float(index),
        input_state=make_state(input_x),
        local_guide_xyz=[[input_x, 0, 0], [input_x + 1, 0, 0]],
        local_goal_xyz=[input_x + 1, 0, 0],
        candidate_samples=candidate,
        executed_samples=executed,
        corridor_segments=np.empty((0, 5)),
        obstacle_states=(),
        diagnostics={"planner_status": "SUCCEEDED"},
    )


class RollingModelTests(unittest.TestCase):
    def test_robot_state_is_finite_immutable_and_builds_from_minco_sample(self):
        from experiments.rolling.models import RobotState

        with self.assertRaisesRegex(ValueError, "finite"):
            RobotState([0, 0, 0], [float("nan"), 0, 0], [0, 0, 0], 0, 0)
        state = make_state()
        with self.assertRaises(ValueError):
            state.position_xyz[0] = 9.0
        sample = np.arange(15, dtype=float)
        reconstructed = RobotState.from_minco_sample(sample)
        np.testing.assert_array_equal(reconstructed.position_xyz, sample[1:4])
        np.testing.assert_array_equal(reconstructed.velocity_xyz_mps, sample[4:7])
        np.testing.assert_array_equal(reconstructed.acceleration_xyz_mps2, sample[7:10])
        self.assertEqual(reconstructed.yaw_rad, sample[13])
        self.assertEqual(reconstructed.yaw_rate_radps, sample[14])

    def test_cycle_record_requires_exact_state_continuity_and_prefix(self):
        from experiments.rolling.models import RolloutCycle, validate_cycle_sequence

        cycle = make_cycle(input_x=0.0, executed_end_x=0.25)
        following = make_cycle(index=1, input_x=0.2)
        self.assertTrue(
            any("state discontinuity" in error for error in validate_cycle_sequence([cycle, following]))
        )
        invalid_prefix = RolloutCycle(
            **{
                **cycle.__dict__,
                "executed_samples": np.asarray(cycle.executed_samples).copy() + 0.01,
            }
        )
        self.assertTrue(
            any("prefix" in error for error in validate_cycle_sequence([invalid_prefix]))
        )
        duplicate_time = np.asarray(cycle.candidate_samples).copy()
        duplicate_time[1, 0] = duplicate_time[0, 0]
        invalid_time = RolloutCycle(
            **{
                **cycle.__dict__,
                "candidate_samples": duplicate_time,
                "executed_samples": duplicate_time[:2],
            }
        )
        self.assertTrue(
            any("sample time" in error for error in validate_cycle_sequence([invalid_time]))
        )

    def test_success_requires_final_goal_tolerance_and_metrics_are_immutable(self):
        from experiments.rolling.models import RolloutResult

        cycle = make_cycle(executed_end_x=1.0)
        result = RolloutResult(
            scenario_uid="scenario-a",
            method="legacy",
            status="GOAL_REACHED",
            cycles=(cycle,),
            executed_samples=cycle.executed_samples,
            final_goal_xyz=[3.0, 0.0, 0.0],
            metrics={"final_error_m": 2.0},
            goal_tolerance_m=0.1,
        )
        self.assertTrue(any("goal tolerance" in error for error in result.validate()))
        self.assertIsInstance(result.metrics, MappingProxyType)

    def test_rollout_config_rejects_unsafe_or_nonfinite_limits(self):
        from experiments.rolling.models import RolloutConfig

        with self.assertRaisesRegex(ValueError, "positive"):
            RolloutConfig(execute_duration_s=0.0)
        with self.assertRaisesRegex(ValueError, "integer"):
            RolloutConfig(max_cycles=2.5)
        config = RolloutConfig()
        self.assertGreater(config.max_cycles, 0)
        self.assertLessEqual(config.execute_duration_s, config.planning_period_s)

    def test_nested_diagnostics_are_deeply_immutable(self):
        from experiments.rolling.models import RolloutCycle

        cycle = make_cycle()
        protected = RolloutCycle(
            **{
                **cycle.__dict__,
                "diagnostics": {"penalties": {"terms": [1.0, 2.0]}},
            }
        )
        self.assertIsInstance(protected.diagnostics["penalties"], MappingProxyType)
        self.assertEqual(protected.diagnostics["penalties"]["terms"], (1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
