from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class RollingScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = (
            Path(__file__).parents[1] / "configs" / "rolling_showcase_v1.json"
        )

    def test_config_defines_all_required_families_and_valid_worlds(self):
        from experiments.rolling.scenarios import load_scenarios, materialize_world

        scenarios = load_scenarios(self.config_path)
        self.assertEqual(
            set(scenarios),
            {
                "unobstructed",
                "static_sparse",
                "static_dense",
                "narrow_passage",
                "malformed_detour",
                "dynamic_crossing",
                "dynamic_head_on",
                "dynamic_sudden_appearance",
            },
        )
        for scenario in scenarios.values():
            self.assertEqual(scenario.guide_path_xyz.shape[1], 3)
            self.assertTrue(np.allclose(scenario.final_goal_xyz, scenario.guide_path_xyz[-1]))
            world = materialize_world(scenario, 0.0)
            self.assertEqual(world.occupancy.shape, world.esdf_distance.shape)
            self.assertFalse(world.occupancy.flags.writeable)

    def test_dynamic_obstacles_are_deterministic_and_time_varying(self):
        from experiments.rolling.scenarios import load_scenarios

        scenario = load_scenarios(self.config_path)["dynamic_crossing"]
        self.assertEqual(scenario.obstacles_at(1.25), scenario.obstacles_at(1.25))
        self.assertNotEqual(scenario.obstacles_at(0.0), scenario.obstacles_at(2.0))

    def test_dense_scene_has_more_occupied_area_than_sparse_scene(self):
        from experiments.rolling.scenarios import load_scenarios, materialize_world

        scenarios = load_scenarios(self.config_path)
        sparse = materialize_world(scenarios["static_sparse"], 0.0)
        dense = materialize_world(scenarios["static_dense"], 0.0)
        self.assertGreater(dense.occupancy.sum(), sparse.occupancy.sum())

    def test_initial_state_sweeps_change_exactly_one_factor(self):
        from experiments.rolling.scenarios import (
            changed_fields,
            initial_state_sweeps,
            load_showcase_config,
        )

        groups = initial_state_sweeps(load_showcase_config(self.config_path))
        self.assertEqual(
            {group.factor for group in groups},
            {"position_xyz", "velocity_xyz_mps", "acceleration_xyz_mps2", "yaw_rad", "yaw_rate_radps"},
        )
        for group in groups:
            self.assertEqual(changed_fields(group.baseline, group.variant), {group.factor})

    def test_invalid_keyframes_duplicate_uids_and_out_of_world_are_rejected(self):
        from experiments.rolling.scenarios import load_showcase_config

        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        dynamic = next(row for row in payload["scenarios"] if row["scenario_uid"] == "dynamic_crossing")
        dynamic["moving_discs"][0]["keyframes"] = [[1.0, 2.0, 1.0], [0.0, 2.0, 2.0]]
        duplicate = dict(dynamic["moving_discs"][0])
        dynamic["moving_discs"].append(duplicate)
        dynamic["static_rectangles"] = [{"obstacle_uid": duplicate["obstacle_uid"], "bounds_xy": [99, 99, 100, 100]}]
        path = Path(tempfile.mkdtemp()) / "invalid.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "keyframes|duplicate|world"):
            load_showcase_config(path)


if __name__ == "__main__":
    unittest.main()
