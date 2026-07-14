import numpy as np
import unittest

from experiments.analyzers.metrics import (
    compute_geometric_metrics,
    compute_minco_temporal_profile,
    compute_safety_metrics,
    distance_point_to_polyline,
)
from experiments.analyzers.situations import SituationThresholds, classify_plan_situation


class MetricTests(unittest.TestCase):
 def test_straight_line_has_near_zero_curvature_and_duplicate_points_are_safe(self):
    points = np.array([[0, 0], [0, 0], [1, 0], [2, 0], [np.nan, 1]])
    metrics, profile = compute_geometric_metrics(points, sample_ds=0.05)
    self.assertEqual(metrics["path_length_m"], 2.0)
    self.assertLess(np.nanmax(np.abs(profile["curvature_1pm"])), 1e-8)


 def test_esdf_bilinear_and_oob_are_unsafe(self):
    grid = np.array([[0.0, 1.0], [1.0, 2.0]])
    metrics, profile = compute_safety_metrics(
        np.array([[0.5, 0.5], [2.0, 2.0]]),
        {"distance": grid, "origin": np.array([0.0, 0.0]), "resolution": 1.0},
        safe_dist=0.6,
        sample_ds=10.0,
    )
    self.assertEqual(profile["esdf_m"][0], 1.0)
    self.assertEqual(metrics["unsafe_ratio"], 0.5)
    self.assertEqual(metrics["esdf_oob_ratio"], 0.5)


 def test_point_to_polyline_distance(self):
    distance, segment, ratio = distance_point_to_polyline(np.array([1.0, 1.0]), np.array([[0, 0], [2, 0]]))
    self.assertEqual((distance, segment, ratio), (1.0, 0, 0.5))


 def test_minco_temporal_profile_uses_time_weighted_rms(self):
    samples = np.zeros((3, 15))
    samples[:, 0] = [0.0, 1.0, 3.0]
    samples[:, 4] = [1.0, 1.0, 3.0]
    summary, _ = compute_minco_temporal_profile(samples)
    self.assertEqual(summary["trajectory_duration_s"], 3.0)
    self.assertGreater(summary["actual_speed_mean_mps"], 1.0)


 def test_situation_thresholds_treat_oob_as_unsafe_and_boundary_as_high_turn(self):
    thresholds = SituationThresholds(0.6, 1.0, 2.0, 0.5, 0.4)
    labels = classify_plan_situation({
        "raw_min_clearance_m": 1.0,
        "raw_unsafe_ratio": 0.0,
        "raw_esdf_oob_ratio": 0.1,
        "raw_curvature_abs_p95_1pm": 1.0,
        "raw_curvature_tv_1pm": 0.0,
        "raw_interplan_position_rmse_m": np.nan,
        "raw_initial_tangent_jump_rad": np.nan,
    }, thresholds)
    self.assertEqual(labels, {"raw_safety_class": "RAW_UNSAFE", "turn_class": "HIGH_TURN", "temporal_class": "NO_HISTORY"})
