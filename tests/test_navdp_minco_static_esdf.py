import importlib
import os
import tempfile
import unittest
from unittest import mock

import numpy as np


class NavDPMincoAdapterTest(unittest.TestCase):
    def test_selects_best_successful_topk_result_and_falls_back_on_failure(self):
        fake_module = mock.Mock()
        fake_processor = mock.Mock()
        fake_processor.optimize.side_effect = [
            {
                "success": False,
                "failure_reason": "blocked",
                "objective": 10.0,
                "min_esdf": -0.1,
                "samples": np.zeros((0, 15)),
                "waypoints": np.zeros((0, 3)),
            },
            {
                "success": True,
                "failure_reason": "NONE",
                "objective": 3.0,
                "min_esdf": 0.6,
                "samples": np.zeros((2, 15)),
                "waypoints": np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            },
        ]
        fake_module.MincoProcessor.return_value = fake_processor

        with mock.patch.dict("sys.modules", {"minco_processor": fake_module}):
            adapter_module = importlib.import_module("utils_tasks.navdp_minco_adapter")
            adapter = adapter_module.NavDPMincoAdapter(
                esdf={
                    "distance": np.ones((4, 4)),
                    "free": np.ones((4, 4), dtype=bool),
                    "origin": np.zeros(2),
                    "resolution": 0.1,
                },
                top_k=3,
                enable=True,
            )

        candidates = np.array(
            [[
                [[0.0, 0.0], [0.3, 0.0], [0.6, 0.0]],
                [[0.0, 0.0], [2.0, 0.0], [2.5, 0.0]],
                [[np.nan, 0.0], [1.0, 0.0], [2.0, 0.0]],
            ]]
        )
        values = np.array([[0.9, 0.8, 1.0]])
        raw_top1 = np.array([[[9.0, 9.0], [10.0, 10.0]]])
        states = [{
            "position": np.zeros(3),
            "velocity": np.zeros(3),
            "acceleration": np.zeros(3),
            "yaw": 0.0,
            "yaw_rate": 0.0,
        }]

        results = adapter.optimize_candidates(candidates, values, states, raw_top1)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["success"])
        self.assertFalse(results[0]["fallback"])
        self.assertEqual(results[0]["selected_index"], 1)
        np.testing.assert_allclose(results[0]["waypoints"], [[0.0, 0.0], [2.0, 0.0]])

        fake_processor.optimize.side_effect = [{
            "success": False,
            "failure_reason": "still blocked",
            "objective": 10.0,
            "min_esdf": -0.1,
            "samples": np.zeros((0, 15)),
            "waypoints": np.zeros((0, 3)),
        }]
        fallback = adapter.optimize_candidates(candidates[:, :1], values[:, :1], states, raw_top1)
        self.assertFalse(fallback[0]["success"])
        self.assertTrue(fallback[0]["fallback"])
        np.testing.assert_allclose(fallback[0]["waypoints"], raw_top1[0])


class SimEsdfBuilderTest(unittest.TestCase):
    def test_loads_cache_when_metadata_matches_and_query_reports_distance(self):
        from utils_tasks.sim_esdf_builder import SimEsdfBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = os.path.join(tmpdir, "esdf_2d.npz")
            np.savez(
                cache,
                distance=np.array([[1.0, -0.1], [0.5, 0.25]]),
                occupied=np.array([[False, True], [False, False]]),
                free=np.array([[True, False], [True, True]]),
                origin=np.array([-1.0, -2.0]),
                resolution=np.array(0.5),
                ground_z=np.array(0.1),
                scene_scale=np.array(1.0),
            )
            builder = SimEsdfBuilder(resolution=0.5, cache_name="esdf_2d.npz")
            esdf = builder.build_or_load_from_stage(None, tmpdir, scene_scale=1.0)

        self.assertEqual(esdf["distance"].shape, (2, 2))
        ok, dist = builder.query_grid(esdf, np.array([-0.75, -1.75]))
        self.assertTrue(ok)
        self.assertAlmostEqual(dist, 1.0)


if __name__ == "__main__":
    unittest.main()
