from pathlib import Path
import unittest

import numpy as np

from experiments.baselines.raw_navdp.controller import RAW_MPC_DEFAULTS, RawReferenceGenerator
from experiments.baselines.raw_navdp.equivalence import verify_provenance
from experiments.baselines.raw_navdp.metrics_adapter import compute_raw_episode_metrics
from experiments.baselines.raw_navdp.trajectory_adapter import camera_top1_to_world


class RawBaselineTests(unittest.TestCase):
    def test_provenance_hashes_match_read_only_sources(self):
        self.assertEqual(verify_provenance(Path(".")), [])

    def test_raw_mpc_defaults_match_source(self):
        self.assertEqual(RAW_MPC_DEFAULTS, {"N":15, "desired_v":.5, "v_max":.5, "w_max":.5, "ref_gap":3, "T":.1, "dense_ratio":50})

    def test_reference_generator_matches_original_arc_length_selection(self):
        path = np.column_stack([np.linspace(0, 2, 101), np.zeros(101)])
        reference = RawReferenceGenerator(path, desired_v=.5, ref_gap=3, horizon=15, dt=.1).find_reference(np.array([.2, 0., 0.]))
        self.assertEqual(reference.shape, (6, 2))
        self.assertTrue(np.all(np.diff(reference[:, 0]) >= 0))
        self.assertGreaterEqual(reference[1, 0] - reference[0, 0], .14)

    def test_camera_top1_transform_matches_source_formula(self):
        camera_path = np.array([[1., 0.], [0., 1.]])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        result = camera_top1_to_world(camera_path, np.array([2., 3., 1.]), rotation)
        np.testing.assert_allclose(result, [[2., 4.], [1., 3.]])

    def test_raw_spl_formula_matches_original(self):
        metrics = compute_raw_episode_metrics(initial_distance=4., trajectory_length=5., success=True)
        self.assertEqual(metrics, {"success":1.0, "spl":.8, "distance":4.0})
