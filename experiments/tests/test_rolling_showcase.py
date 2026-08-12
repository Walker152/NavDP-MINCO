from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import imageio.v2 as imageio
import numpy as np


def _samples(y_offset: float, yaw_offset: float = 0.0) -> np.ndarray:
    samples = np.zeros((9, 15), dtype=np.float64)
    samples[:, 0] = np.linspace(0.0, 2.0, len(samples))
    samples[:, 1] = np.linspace(0.0, 4.0, len(samples))
    samples[:, 2] = y_offset + 0.25 * np.sin(samples[:, 1])
    samples[:, 4] = 0.8
    samples[:, 13] = yaw_offset + np.linspace(0.0, 0.45, len(samples))
    samples[:, 14] = 0.18
    return samples


def make_paired_results(*, corridor: bool = True, goal_yaw: float | None = None):
    corridor_segments = (
        np.array(
            [
                [0.0, 0.0, 0.0, 2.0, 0.10, 0.0, 0.38],
                [2.0, 0.10, 0.0, 4.0, -0.10, 0.0, 0.30],
            ],
            dtype=np.float64,
        )
        if corridor
        else np.empty((0, 7), dtype=np.float64)
    )
    obstacles = (
        {
            "obstacle_uid": "moving-disc",
            "position_xy_m": [2.0, 1.0],
            "velocity_xy_mps": [0.0, -0.35],
            "radius_m": 0.28,
        },
    )
    cycles = (
        {
            "cycle_index": 0,
            "time_s": 0.0,
            "input_state": {
                "position_xyz": [0.0, 0.0, 0.0],
                "velocity_xyz_mps": [0.8, 0.0, 0.0],
                "yaw_rad": 0.0,
            },
            "candidate_samples": _samples(0.0),
            "executed_samples": _samples(0.0)[:5],
            "corridor_segments": corridor_segments,
            "obstacle_states": obstacles,
            "diagnostics": {"failure_reason": ""},
        },
        {
            "cycle_index": 1,
            "time_s": 1.0,
            "input_state": {
                "position_xyz": [2.0, 0.1, 0.0],
                "velocity_xyz_mps": [0.7, 0.1, 0.0],
                "yaw_rad": 0.25,
            },
            "candidate_samples": _samples(0.1, 0.2),
            "executed_samples": _samples(0.1, 0.2)[4:],
            "corridor_segments": corridor_segments,
            "obstacle_states": obstacles,
            "diagnostics": {"failure_reason": ""},
        },
    )
    return {
        "scenario_uid": "dynamic_crossing_fixture",
        "guide_path_xyz": np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        ),
        "final_goal_xyz": np.array([4.0, 0.0, 0.0]),
        "goal_yaw_rad": goal_yaw,
        "initial_state": {
            "position_xyz": [0.0, 0.0, 0.0],
            "velocity_xyz_mps": [0.8, 0.0, 0.0],
            "yaw_rad": 0.0,
        },
        "legacy": {
            "method": "legacy",
            "status": "GOAL_REACHED",
            "cycles": cycles,
            "executed_samples": _samples(0.18),
            "metrics": {"min_clearance_m": 0.31, "runtime_ms": 8.2},
        },
        "safe_corridor_v1": {
            "method": "safe_corridor_v1",
            "status": "GOAL_REACHED",
            "cycles": cycles,
            "executed_samples": _samples(0.05),
            "metrics": {"min_clearance_m": 0.44, "runtime_ms": 9.1},
        },
    }


class RollingShowcaseRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_scene_package_contains_all_paper_outputs_and_valid_receipts(self):
        from experiments.visualizers.rolling_showcase import (
            render_scene_package,
            validate_scene_package,
        )

        package = render_scene_package(make_paired_results(), self.root / "scene")

        required = {
            "three_panel.png",
            "three_panel.pdf",
            "overlay.png",
            "overlay.pdf",
            "safe_corridor.png",
            "safe_corridor.pdf",
            "three_way.gif",
            "figure_data.csv",
            "caption.md",
            "caption_zh.md",
            "scene_manifest.json",
            "artifact_receipt.json",
            "validation.json",
        }
        self.assertTrue(required.issubset({path.name for path in package.files}))
        self.assertEqual(validate_scene_package(package.output_dir), [])
        validation = json.loads(package.validation_path.read_text(encoding="utf-8"))
        self.assertTrue(validation["valid"])
        receipt = json.loads(package.receipt_path.read_text(encoding="utf-8"))
        recorded = {row["path"] for row in receipt["artifacts"]}
        actual = {
            path.name
            for path in package.output_dir.iterdir()
            if path.is_file() and path != package.receipt_path
        }
        self.assertEqual(recorded, actual)
        with (package.output_dir / "figure_data.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            record_types = {row["record_type"] for row in csv.DictReader(stream)}
        self.assertTrue(
            {
                "INITIAL_STATE",
                "GOAL_STATE",
                "EXECUTED_SAMPLE",
                "CORRIDOR_SEGMENT",
                "OBSTACLE_STATE",
                "ANIMATION_FRAME",
            }.issubset(record_types)
        )

    def test_visual_contract_records_arrows_equal_axes_and_na_goal_yaw(self):
        from experiments.visualizers.rolling_showcase import render_scene_package

        package = render_scene_package(make_paired_results(goal_yaw=None), self.root / "scene")
        manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
        contract = manifest["visual_contract"]

        self.assertEqual(contract["axis_aspect"], "equal")
        self.assertEqual(contract["shared_xy_limits"], contract["panel_xy_limits"][0])
        self.assertTrue(
            all(limits == contract["shared_xy_limits"] for limits in contract["panel_xy_limits"])
        )
        self.assertEqual(contract["robot_heading"], "ARROW")
        self.assertEqual(contract["initial_yaw"], "ARROW")
        self.assertEqual(contract["goal_yaw"], "N/A")
        self.assertEqual(contract["velocity"], "ARROW")
        self.assertEqual(contract["dynamic_obstacle_velocity"], "ARROW")
        self.assertGreater(contract["sampled_heading_arrow_count"], 1)
        self.assertGreater(contract["corridor_capsule_count"], 0)

    def test_safe_corridor_requires_real_recorded_corridor_segments(self):
        from experiments.visualizers.rolling_showcase import render_scene_package

        with self.assertRaisesRegex(ValueError, "corridor evidence"):
            render_scene_package(
                make_paired_results(corridor=False), self.root / "scene"
            )

    def test_gif_is_synchronized_and_captions_are_bilingual_scientific(self):
        from experiments.visualizers.rolling_showcase import render_scene_package

        package = render_scene_package(
            make_paired_results(goal_yaw=0.4), self.root / "scene"
        )
        manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
        frames = imageio.mimread(package.output_dir / "three_way.gif")
        with (package.output_dir / "figure_data.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        animation_rows = [row for row in rows if row["record_type"] == "ANIMATION_FRAME"]

        self.assertEqual(len(frames), len(animation_rows))
        self.assertEqual(len(frames), manifest["animation"]["frame_count"])
        self.assertGreaterEqual(manifest["animation"]["terminal_hold_frames"], 3)
        self.assertEqual(manifest["animation"]["panels"], ["guide", "legacy", "safe_corridor_v1"])
        self.assertTrue(manifest["animation"]["shows_current_state"])
        self.assertTrue(manifest["animation"]["shows_final_goal"])
        self.assertTrue(manifest["animation"]["shows_safe_corridor"])
        self.assertEqual(
            manifest["animation"]["state_box_fields"],
            [
                "cycle",
                "time_s",
                "x_m",
                "y_m",
                "yaw_rad",
                "yaw_deg",
                "speed_mps",
                "acceleration_mps2",
                "yaw_rate_radps",
                "local_goal_xy_m",
                "final_goal_xy_m",
                "status",
            ],
        )
        for caption_name, labels in (
            (
                "caption.md",
                ("Source:", "Paired key:", "Denominator:", "Units:", "Sample size:", "Missing data:", "Interpretation:", "Limitations:"),
            ),
            (
                "caption_zh.md",
                ("数据来源：", "配对键：", "分母：", "单位：", "样本量：", "缺失数据：", "解读：", "局限性："),
            ),
        ):
            text = (package.output_dir / caption_name).read_text(encoding="utf-8")
            for label in labels:
                self.assertIn(label, text)

    def test_tampered_figure_data_is_detected(self):
        from experiments.visualizers.rolling_showcase import (
            render_scene_package,
            validate_scene_package,
        )

        package = render_scene_package(make_paired_results(), self.root / "scene")
        with (package.output_dir / "figure_data.csv").open("a", encoding="utf-8") as stream:
            stream.write("tampered,row\n")

        errors = validate_scene_package(package.output_dir)

        self.assertTrue(any("figure_data.csv hash mismatch" in error for error in errors), errors)

    def test_malformed_visual_contract_fails_closed_without_validator_crash(self):
        from experiments.visualizers.rolling_showcase import (
            render_scene_package,
            validate_scene_package,
        )

        package = render_scene_package(make_paired_results(), self.root / "scene")
        manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
        manifest["visual_contract"]["panel_xy_limits"] = None
        manifest["visual_contract"]["corridor_capsule_count"] = "not-an-integer"
        package.manifest_path.write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        errors = validate_scene_package(package.output_dir)

        self.assertTrue(any("panel XY limits" in error for error in errors), errors)
        self.assertTrue(
            any("recorded capsule evidence" in error for error in errors), errors
        )

    def test_static_rectangles_are_rendered_from_scenario_evidence(self):
        from experiments.visualizers.rolling_showcase import render_scene_package

        paired = make_paired_results()
        paired["static_rectangles"] = (
            {"obstacle_uid": "wall", "bounds_xy": [1.0, -0.4, 1.5, 0.4]},
        )

        package = render_scene_package(paired, self.root / "scene")
        manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["visual_contract"]["static_rectangle_count"], 1)

    def test_renderer_consumes_production_rollout_dataclasses(self):
        from experiments.rolling.models import (
            ObstacleState,
            RobotState,
            RolloutCycle,
            RolloutResult,
        )
        from experiments.visualizers.rolling_showcase import render_scene_package

        samples = _samples(0.0)
        state = RobotState.from_minco_sample(samples[0])
        obstacle = ObstacleState(
            obstacle_uid="recorded-disc",
            center_xy=np.array([2.0, 0.8]),
            radius_m=0.25,
            velocity_xy_mps=np.array([0.0, -0.3]),
            dynamic=True,
        )
        cycle = RolloutCycle(
            cycle_index=0,
            time_s=0.0,
            input_state=state,
            local_guide_xyz=np.array(
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
            ),
            local_goal_xyz=np.array([4.0, 0.0, 0.0]),
            candidate_samples=samples,
            executed_samples=samples,
            corridor_segments=np.array(
                [[0.0, 0.0, 2.0, 0.1, 0.35], [2.0, 0.1, 4.0, 0.0, 0.3]]
            ),
            obstacle_states=(obstacle,),
            diagnostics={},
        )

        def result(method: str) -> RolloutResult:
            return RolloutResult(
                scenario_uid="production_contract",
                method=method,
                status="MAX_CYCLES",
                cycles=(cycle,),
                executed_samples=samples,
                final_goal_xyz=np.array([4.0, 0.0, 0.0]),
                metrics={},
            )

        package = render_scene_package(
            (result("legacy"), result("safe_corridor_v1")),
            self.root / "production",
        )
        manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["scenario_uid"], "production_contract")
        self.assertEqual(
            manifest["visual_contract"]["dynamic_obstacle_velocity"], "ARROW"
        )


if __name__ == "__main__":
    unittest.main()
