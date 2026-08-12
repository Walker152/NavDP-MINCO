from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest


class RollingShowcasePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Path(__file__).parents[2]
        self.config = (
            self.repo / "experiments" / "configs" / "rolling_showcase_v1.json"
        )
        self.output = Path(tempfile.mkdtemp()) / "paper_showcase"

    def test_required_showcase_directories_are_declared(self) -> None:
        from experiments.rolling.showcase import SHOWCASE_DIRECTORIES

        self.assertEqual(
            SHOWCASE_DIRECTORIES,
            (
                "01_trajectory_optimization",
                "02_safe_corridor",
                "03_initial_state",
                "04_dynamic_obstacles",
                "05_extreme_cases",
                "06_aggregate_figures",
            ),
        )

    def test_extreme_selection_rejects_dense_short_as_corridor_effect(self) -> None:
        from experiments.rolling.showcase import select_extreme_cases

        rows = [
            {
                "scenario_uid": "syn_dense_short",
                "scene_family": "sampling_boundary",
                "distortion_improvement": 99.0,
                "clearance_improvement_m": 99.0,
                "failure_improvement": 1,
            },
            {
                "scenario_uid": "dense_detour",
                "scene_family": "dense_static",
                "distortion_improvement": 1.0,
                "clearance_improvement_m": 0.2,
                "failure_improvement": 0,
            },
        ]

        selected = select_extreme_cases(rows, count=1)

        self.assertEqual(selected, ("dense_detour",))

    def test_index_contains_direct_links_and_scientific_meaning(self) -> None:
        from experiments.rolling.showcase import write_showcase_index

        manifest = {
            "scenes": [
                {
                    "scenario_uid": "dense_detour",
                    "scene_family": "dense_static",
                    "package_path": (
                        "02_safe_corridor/dense_obstacles/dense_detour"
                    ),
                    "headline_figure": "three_panel.png",
                    "conclusion_zh": "展示约束前后的完整路线差异。",
                    "limitation_zh": "本地确定性规划实验。",
                }
            ]
        }

        path = write_showcase_index(manifest, self.output)
        text = path.read_text(encoding="utf-8")

        self.assertIn("轨迹优化", text)
        self.assertIn("安全走廊", text)
        self.assertIn("不同初值", text)
        self.assertIn("动态障碍", text)
        self.assertIn("极端案例", text)
        self.assertIn("看什么", text)
        self.assertIn("证明什么", text)
        self.assertIn("限制", text)
        self.assertIn(
            "02_safe_corridor/dense_obstacles/dense_detour/three_panel.png",
            text,
        )

    def test_manifest_validation_fails_when_scene_methods_are_unfair(self) -> None:
        from experiments.rolling.showcase import validate_showcase_manifest

        manifest = {
            "schema_version": 1,
            "paired_key": ["scenario_uid", "seed", "initial_state_hash"],
            "scenes": [
                {
                    "scenario_uid": "dense_detour",
                    "methods": ["guide_reference", "safe_corridor_v1"],
                    "method_pair_receipts": {
                        "guide_reference": {"seed": 7, "initial_state_hash": "a"},
                        "safe_corridor_v1": {"seed": 7, "initial_state_hash": "a"},
                    },
                }
            ],
        }

        errors = validate_showcase_manifest(manifest)

        self.assertTrue(any("methods" in error for error in errors), errors)

    def test_config_covers_required_scene_families_and_initial_factors(self) -> None:
        payload = json.loads(self.config.read_text(encoding="utf-8"))

        families = {row["family"] for row in payload["scenarios"]}
        self.assertTrue(
            {
                "unobstructed",
                "static_sparse",
                "static_dense",
                "narrow_passage",
                "malformed_detour",
                "dynamic_crossing",
                "dynamic_head_on",
                "dynamic_sudden_appearance",
            }.issubset(families)
        )
        factors = {row["factor"] for row in payload["initial_state_sweeps"]}
        self.assertEqual(
            factors,
            {
                "position_xyz",
                "velocity_xyz_mps",
                "acceleration_xyz_mps2",
                "yaw_rad",
                "yaw_rate_radps",
            },
        )

    def test_run_showcase_executes_both_methods_and_renders_each_scenario(self) -> None:
        from experiments.rolling.showcase import run_rolling_showcase

        calls: list[tuple[str, str]] = []
        rendered: list[str] = []

        class Result:
            def __init__(self, scenario_uid: str, method: str) -> None:
                self.scenario_uid = scenario_uid
                self.method = method
                self.status = "GOAL_REACHED"
                self.metrics = {"final_error_m": 0.0}

        def runner(scenario, *, method, **_kwargs):
            calls.append((scenario.scenario_uid, method))
            return Result(scenario.scenario_uid, method)

        class Package:
            def __init__(self, output_dir: Path) -> None:
                self.output_dir = output_dir

        def renderer(paired, output_dir):
            rendered.append(paired["scenario_uid"])
            output = Path(output_dir)
            output.mkdir(parents=True)
            (output / "three_panel.png").write_bytes(b"figure")
            return Package(output)

        serialized: list[tuple[str, str]] = []

        def serializer(result, output_dir):
            serialized.append((result.scenario_uid, result.method))
            output = Path(output_dir)
            output.mkdir(parents=True)
            (output / "run_manifest.json").write_text("{}\n")

        manifest = run_rolling_showcase(
            self.config,
            self.output,
            rollout_runner=runner,
            scene_renderer=renderer,
            rollout_serializer=serializer,
            write_receipts=False,
        )

        payload = json.loads(self.config.read_text())
        scenario_count = len(payload["scenarios"])
        variant_count = sum(
            len(group["variants"]) for group in payload["initial_state_sweeps"]
        )
        total = scenario_count + variant_count
        self.assertEqual(len(calls), 2 * total)
        self.assertEqual(len(rendered), total)
        self.assertEqual(len(serialized), 2 * total)
        self.assertEqual(len(manifest["scenes"]), total)
        for scene in manifest["scenes"]:
            self.assertEqual(scene["methods"], list(("guide_reference", "legacy", "safe_corridor_v1")))
            self.assertEqual(scene["paired_key"], "scenario_uid+seed+initial_state_hash")

    def test_run_showcase_includes_every_initial_state_variant(self) -> None:
        from experiments.rolling.showcase import run_rolling_showcase

        class Result:
            status = "GOAL_REACHED"
            metrics = {"final_error_m": 0.0}

        class Package:
            def __init__(self, output_dir: Path) -> None:
                self.output_dir = output_dir

        def renderer(_paired, output_dir):
            output = Path(output_dir)
            output.mkdir(parents=True)
            (output / "three_panel.png").write_bytes(b"figure")
            return Package(output)

        manifest = run_rolling_showcase(
            self.config,
            self.output,
            rollout_runner=lambda *_args, **_kwargs: Result(),
            scene_renderer=renderer,
            rollout_serializer=lambda *_args, **_kwargs: None,
            write_receipts=False,
        )

        factors = {
            row["initial_factor"]
            for row in manifest["scenes"]
            if row.get("experiment_group") == "initial_state"
        }
        self.assertEqual(
            factors,
            {
                "position_xyz",
                "velocity_xyz_mps",
                "acceleration_xyz_mps2",
                "yaw_rad",
                "yaw_rate_radps",
            },
        )

    def test_aggregate_and_extreme_directories_receive_real_outputs(self) -> None:
        from experiments.rolling.showcase import run_rolling_showcase

        class Result:
            status = "GOAL_REACHED"
            metrics = {"final_error_m": 0.0, "cycle_count": 2}

        class Package:
            def __init__(self, output_dir: Path) -> None:
                self.output_dir = output_dir

        def renderer(_paired, output_dir):
            output = Path(output_dir)
            output.mkdir(parents=True)
            (output / "three_panel.png").write_bytes(b"figure")
            return Package(output)

        run_rolling_showcase(
            self.config,
            self.output,
            rollout_runner=lambda *_args, **_kwargs: Result(),
            scene_renderer=renderer,
            rollout_serializer=lambda *_args, **_kwargs: None,
            write_receipts=False,
        )

        self.assertTrue(
            (self.output / "05_extreme_cases" / "selection.csv").is_file()
        )
        self.assertTrue(
            (self.output / "06_aggregate_figures" / "all_scenarios.csv").is_file()
        )
        self.assertTrue(
            (self.output / "06_aggregate_figures" / "status_comparison.png").is_file()
        )
        self.assertTrue(
            (self.output / "06_aggregate_figures" / "status_comparison.pdf").is_file()
        )

    def test_extreme_scores_use_recorded_distortion_clearance_and_status(self) -> None:
        from experiments.rolling.showcase import _extreme_evidence_row

        class Result:
            def __init__(self, status, distortion, clearance):
                self.status = status
                self.metrics = {
                    "guide_deviation_max_m": distortion,
                    "min_clearance_m": clearance,
                }

        row = _extreme_evidence_row(
            "dense_case",
            "static_dense",
            Result("OPTIMIZATION_FAILED", 1.4, 0.12),
            Result("GOAL_REACHED", 0.3, 0.42),
        )

        self.assertAlmostEqual(row["distortion_improvement"], 1.1)
        self.assertAlmostEqual(row["clearance_improvement_m"], 0.3)
        self.assertEqual(row["failure_improvement"], 1)
        self.assertAlmostEqual(row["score"], 12.6)


if __name__ == "__main__":
    unittest.main()
