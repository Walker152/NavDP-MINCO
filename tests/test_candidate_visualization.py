import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from utils_tasks.visualization_utils import (
    RGB_CANDIDATE_OTHER,
    RGB_CYAN,
    VisualizationManager,
)


class CandidateVisualizationTests(unittest.TestCase):
    def test_selected_candidate_has_distinct_style(self):
        manager = VisualizationManager(show_all_candidates=True)
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        trajectories = [
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 0.0], [0.0, 1.0]]),
        ]

        with patch.object(manager, "draw_polyline_world") as draw:
            manager._draw_candidate_trajectories(
                image, trajectories, selected_index=1, robot_pose=np.zeros(3)
            )

        self.assertEqual(draw.call_count, 3)
        self.assertEqual(draw.call_args_list[0].kwargs["color"], RGB_CANDIDATE_OTHER)
        self.assertEqual(draw.call_args_list[0].kwargs["thickness"], 1)
        self.assertEqual(draw.call_args_list[1].kwargs["color"], RGB_CANDIDATE_OTHER)
        self.assertEqual(draw.call_args_list[2].kwargs["color"], RGB_CYAN)
        self.assertEqual(draw.call_args_list[2].kwargs["thickness"], 3)

    def test_compose_panels_uses_same_height_horizontal_layout(self):
        manager = VisualizationManager(show_all_candidates=True)
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        main = np.zeros((200, 200, 3), dtype=np.uint8)
        candidates = np.zeros((200, 200, 3), dtype=np.uint8)

        three_panel = manager._compose_panels(rgb, main, candidates)
        two_panel = manager._compose_panels(rgb, main, None)

        self.assertEqual(three_panel.shape, (120, 400, 3))
        self.assertEqual(two_panel.shape, (120, 280, 3))

    def test_pointgoal_enables_candidates_and_passes_selected_index(self):
        source = Path("eval_pointgoal_wheeled.py").read_text(encoding="utf-8")
        self.assertIn(
            "VisualizationManager(history_size=5, show_all_candidates=True)", source
        )
        self.assertIn("selected_candidate_index =", source)
        self.assertIn('current_minco_info[i].get("selected_index", -1)', source)
        self.assertIn("selected_candidate_index=selected_candidate_index", source)

    def test_visualize_keeps_three_panel_video_shape_even_without_candidates(self):
        manager = VisualizationManager(show_all_candidates=True)
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        depth = np.zeros((120, 160, 1), dtype=np.float32)
        occupancy = np.zeros((4, 4), dtype=np.uint8)
        intrinsic = np.eye(3)
        with patch.object(
            manager, "build_occupancy_grid", return_value=(occupancy, np.zeros(2))
        ):
            output = manager.visualize_trajectory(
                rgb,
                depth,
                intrinsic,
                trajectory_points=None,
                robot_pose=np.zeros(3),
                all_trajectories_points=None,
            )
        self.assertEqual(output.shape, (120, 400, 3))


if __name__ == "__main__":
    unittest.main()
