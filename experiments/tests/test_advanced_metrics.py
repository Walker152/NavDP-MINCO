import unittest

import numpy as np

from experiments.analyzers.metrics import (
    compare_trajectory_prefixes,
    compute_command_deltas,
    compute_deadline_metrics,
    wrap_angle,
)
from experiments.analyzers.statistics import bootstrap_ci, mcnemar_test, proportion_ci, sample_rule, wilcoxon_paired


def trajectory(times, offset=0.0, yaw=0.0):
    samples = np.zeros((len(times), 15))
    samples[:, 0] = times
    samples[:, 1] = np.asarray(times) + offset
    samples[:, 4] = 1.0
    samples[:, 13] = yaw
    return samples


class AdvancedMetricTests(unittest.TestCase):
    def test_trajectory_comparison_shifts_old_plan_and_resamples(self):
        previous = trajectory(np.arange(0.0, 4.1, 0.2))
        current = trajectory(np.arange(0.0, 3.1, 0.1), offset=1.0)
        metrics, _ = compare_trajectory_prefixes(previous, 10.0, current, 11.0, 0.1)
        self.assertLess(metrics["interplan_position_rmse_m"], 1e-9)
        self.assertAlmostEqual(metrics["common_duration_s"], 3.0)

    def test_trajectory_comparison_wraps_yaw(self):
        previous = trajectory(np.array([0.0, 1.0]), yaw=np.pi - 0.01)
        current = trajectory(np.array([0.0, 1.0]), yaw=-np.pi + 0.01)
        metrics, _ = compare_trajectory_prefixes(previous, 0.0, current, 0.0, 0.1)
        self.assertLess(metrics["interplan_yaw_rmse_rad"], 0.03)
        self.assertAlmostEqual(wrap_angle(3 * np.pi), -np.pi)

    def test_finished_old_trajectory_returns_nan_summary(self):
        metrics, _ = compare_trajectory_prefixes(trajectory(np.array([0.0, 1.0])), 0.0, trajectory(np.array([0.0, 1.0])), 2.0, 0.1)
        self.assertTrue(np.isnan(metrics["interplan_position_rmse_m"]))

    def test_command_delta_and_deadline_metrics(self):
        delta = compute_command_deltas(np.array([[1.0, 0.2], [1.5, -0.1], [1.0, 0.4]]))
        self.assertAlmostEqual(delta["command_delta_w_abs_mean_radps"], 0.4)
        deadline = compute_deadline_metrics(np.array([5.0, 10.0, 15.0]), deadline_ms=10.0)
        self.assertEqual(deadline["deadline_miss_ratio"], 1 / 3)

    def test_statistics_report_methods_and_small_sample_rule(self):
        low, high = bootstrap_ci(np.arange(10.0), seed=7)
        self.assertLess(low, high)
        self.assertEqual(sample_rule(4), "CASES_ONLY")
        self.assertEqual(sample_rule(7), "DESCRIPTIVE_BOOTSTRAP")
        self.assertEqual(sample_rule(10), "PAIRED_TEST_ALLOWED")
        self.assertEqual(proportion_ci(5, 10)["method"], "wilson")
        self.assertEqual(mcnemar_test(3, 1)["method"], "exact_binomial")
        self.assertIn(wilcoxon_paired(np.arange(10.0), np.arange(10.0) + 1.0)["method"], {"scipy_wilcoxon", "normal_approximation"})
