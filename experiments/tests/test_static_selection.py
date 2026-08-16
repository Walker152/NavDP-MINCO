import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class StaticSelectionPolicyTests(unittest.TestCase):
    def test_factor_grid_gif_uses_reviewable_playback_cadence(self):
        from experiments.visualizers.static_benchmark import (
            GIF_FRAME_DURATION_S,
            GRID_GIF_FRAME_DURATION_S,
            GRID_GIF_MIN_FRAMES,
        )

        self.assertEqual(GRID_GIF_FRAME_DURATION_S, 0.20)
        self.assertGreaterEqual(GRID_GIF_MIN_FRAMES, 24)
        self.assertGreater(GRID_GIF_FRAME_DURATION_S, GIF_FRAME_DURATION_S)

    def test_full_route_showcase_covers_initial_state_and_obstacle_extremes(self):
        from experiments.static.selection import FULL_ROUTE_SHOWCASE_CASES

        self.assertEqual(
            FULL_ROUTE_SHOWCASE_CASES,
            (
                "syn_straight",
                "state_v_reverse",
                "state_a_along",
                "state_yaw_reverse",
                "syn_s_curve_sparse",
                "syn_l90_dense",
            ),
        )

    def test_corridor_showcase_cases_include_success_and_fail_closed_evidence(self):
        from experiments.static.selection import CORRIDOR_SHOWCASE_CASES

        self.assertEqual(
            CORRIDOR_SHOWCASE_CASES,
            ("syn_l90", "syn_s_curve_sparse", "syn_l90_dense"),
        )

    def _rows(self):
        base = {
            "profile": "safe_corridor_v1",
            "dynamic_replayable": True,
            "case_hash": "",
            "min_normalized_margin": 0.5,
            "guide_deviation_p95_m": 0.1,
            "path_length_ratio": 1.0,
            "actual_jerk_p95_mps3": 1.0,
            "runtime_ms": 10.0,
            "violation_margin": 0.0,
        }
        rows = []
        for uid, category, classification, reason, margin in (
            ("best_a", "straight", "SAFE_FEASIBLE", "NONE", 0.9),
            ("best_b", "straight", "SAFE_FEASIBLE", "NONE", 0.8),
            ("best_c", "s_curve", "SAFE_FEASIBLE", "NONE", 0.7),
            ("degraded", "u_return", "SAFE_BUT_DEGRADED", "NONE", 0.2),
            (
                "closed",
                "through_obstacle",
                "FAIL_CLOSED_EXPECTED",
                "CORRIDOR_GUIDE_UNSAFE",
                -1.0,
            ),
            ("excluded", "l90", "VALIDATION_FAILED", "VALIDATION_JERK", -0.3),
        ):
            row = copy.deepcopy(base)
            row.update(
                {
                    "case_uid": uid,
                    "case_hash": "hash_" + uid,
                    "expected_category": category,
                    "classification": classification,
                    "failure_reason": reason,
                    "min_normalized_margin": margin,
                }
            )
            rows.append(row)
        rows[-1]["dynamic_replayable"] = False
        return rows

    def test_selection_is_stable_diverse_and_excludes_nonreplayable(self):
        from experiments.static.selection import default_selection_policy, select_cases

        policy = default_selection_policy()
        first = select_cases(self._rows(), policy)
        second = select_cases(list(reversed(self._rows())), policy)

        self.assertEqual(first, second)
        self.assertEqual(first["best"][:2], ["best_a", "best_c"])
        self.assertEqual(first["worst"][:2], ["closed", "degraded"])
        self.assertNotIn("excluded", first["eligible_case_uids"])

    def test_duplicate_hash_uses_uid_tiebreak_and_is_receipted(self):
        from experiments.static.selection import default_selection_policy, select_cases

        rows = self._rows()
        duplicate = copy.deepcopy(rows[0])
        duplicate["case_uid"] = "z_duplicate"
        rows.append(duplicate)
        selected = select_cases(rows, default_selection_policy())

        self.assertNotIn("z_duplicate", selected["eligible_case_uids"])
        self.assertEqual(
            selected["exclusions"]["z_duplicate"], "DUPLICATE_CASE_HASH"
        )

    def test_boundary_config_declares_real_single_and_two_factor_scans(self):
        from experiments.static.selection import boundary_factor_metadata

        config_path = (
            Path(__file__).parents[1]
            / "configs"
            / "static_boundary_selection_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metadata = boundary_factor_metadata(config)

        self.assertTrue(
            any(row["factor_name"] == "initial_speed_mps" for row in metadata.values())
        )
        grid_rows = [
            row for row in metadata.values() if row.get("factor_name_secondary")
        ]
        self.assertGreaterEqual(len(grid_rows), 9)
        expected_pairs = {
            ("initial_speed_mps", "yaw_error_rad"),
            ("initial_yaw_rate_radps", "initial_speed_mps"),
            ("initial_acceleration_x_mps2", "yaw_error_rad"),
        }
        actual_pairs = {
            (row["factor_name"], row["factor_name_secondary"]) for row in grid_rows
        }
        self.assertTrue(
            expected_pairs.issubset(actual_pairs),
            f"Missing grid pairs: {expected_pairs - actual_pairs}",
        )

    def test_lateral_offset_variants_are_pure_lateral(self):
        from experiments.static.selection import generate_boundary_cases

        config_path = (
            Path(__file__).parents[1]
            / "configs"
            / "static_boundary_selection_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        base = Path(config["base_case_config"])
        config["base_case_config"] = str((config_path.parent / base).resolve())
        cases = {case.case_uid: case for case in generate_boundary_cases(config)}

        # syn_straight runs along y=1 with zero lateral normal component in x,
        # so a lateral offset must move the start purely in y with NO
        # longitudinal (x) displacement.
        guide_start = cases["syn_straight"].guide_path_xyz[0]
        expected = {
            "state_offset_small": guide_start + np.array([0.0, 0.15, 0.0]),
            "state_offset_large": guide_start + np.array([0.0, 0.35, 0.0]),
        }
        for uid, target in expected.items():
            np.testing.assert_allclose(
                cases[uid].start_position, target, atol=1e-12
            )
            self.assertEqual(
                cases[uid].start_position[0], guide_start[0],
                f"{uid} must have zero longitudinal displacement",
            )
        # The lateral normal is perpendicular to the guide start direction
        # ((3,0) for syn_straight), so the offset stays on the guide start
        # position, not 1.0m ahead of it.
        np.testing.assert_allclose(
            cases["state_offset_small"].start_position[:2], [0.0, 1.15]
        )
        np.testing.assert_allclose(
            cases["state_offset_large"].start_position[:2], [0.0, 1.35]
        )

    def test_semantically_duplicate_initial_states_share_selection_hash(self):
        from experiments.static.selection import (
            boundary_case_content_hash,
            generate_boundary_cases,
        )

        config_path = (
            Path(__file__).parents[1]
            / "configs"
            / "static_boundary_selection_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        base = Path(config["base_case_config"])
        config["base_case_config"] = str((config_path.parent / base).resolve())
        cases = {case.case_uid: case for case in generate_boundary_cases(config)}

        # A grid cell with zero speed and zero yaw on syn_s_curve
        # should be semantically identical to the base syn_s_curve case.
        base_s_curve = cases["syn_s_curve"]
        grid_zero = cases["grid_speed_yaw_on_s_curve_x00_y00"]
        self.assertNotEqual(
            base_s_curve.case_hash,
            grid_zero.case_hash,
        )
        self.assertEqual(
            boundary_case_content_hash(base_s_curve),
            boundary_case_content_hash(grid_zero),
        )

    def test_acceleration_grid_keeps_the_declared_matched_velocity(self):
        from experiments.static.selection import generate_boundary_cases

        config_path = Path(__file__).parents[1] / "configs" / "static_boundary_selection_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["base_case_config"] = str(
            (config_path.parent / config["base_case_config"]).resolve()
        )
        cases = {
            case.case_uid: case
            for case in generate_boundary_cases(config)
            if case.case_uid.startswith("grid_accel_yaw_on_l90_")
        }
        self.assertTrue(cases)
        for case in cases.values():
            np.testing.assert_allclose(case.start_velocity, [0.5, 0.0, 0.0])

    def test_capability_comparison_pairs_legacy_with_superplanner_sfc(self):
        from experiments.analyzers.sweep_comparison import generate_sweep_comparison

        rows = []
        for profile in ("legacy", "superplanner_sfc_v1"):
            rows.append(
                {
                    "case_uid": "extreme_case",
                    "profile": profile,
                    "status": "SUCCEEDED",
                    "expected_category": "l90",
                    "failure_reason": "NONE",
                    "min_clearance_m": "0.5",
                    "unsafe_ratio": "0.0",
                    "path_length_ratio": "1.0",
                    "guide_deviation_p95_m": "0.1",
                    "yaw_rate_violation_ratio": "0.0",
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (root / "sweep_runs.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary = generate_sweep_comparison(root)
            self.assertEqual(summary["paired_cases"], 1)
            self.assertEqual(
                [row["profile"] for row in summary["profiles"]],
                ["legacy", "superplanner_sfc_v1"],
            )


class StaticSelectedGifEvidenceTests(unittest.TestCase):
    def test_selected_paired_gif_has_complete_chinese_evidence_package(self):
        from dataclasses import replace

        from experiments.static.metrics import compute_static_case_metrics
        from experiments.static.runner import StaticRunResult
        from experiments.static.selection import _render_selected_artifacts
        from experiments.tests.test_static_benchmark import StaticCaseSchemaTests
        from experiments.visualizers.video_evidence import FRAME_METRICS_FIELDS

        case = StaticCaseSchemaTests().make_case()
        samples = np.zeros((4, 15), dtype=np.float64)
        samples[:, 0] = np.linspace(0.0, 0.3, 4)
        samples[:, 1:4] = np.linspace(
            case.guide_path_xyz[0], case.guide_path_xyz[-1], 4
        )
        samples[:, 4] = 0.2
        limits = {
            "sample_ds_m": 0.1,
            "safe_distance_m": 0.15,
            "max_velocity_mps": 1.0,
            "max_acceleration_mps2": 1.5,
            "max_jerk_mps3": 20.0,
            "max_yaw_rate_radps": 0.5,
        }
        per_case = {}
        frozen = {"cases": {case.case_uid: {}}}
        for profile in ("legacy", "superplanner_sfc_v1"):
            profile_case = replace(case, constraint_profile=profile)
            result = StaticRunResult(
                case_uid=case.case_uid,
                case_hash=profile_case.case_hash,
                mode="recompute",
                status="SUCCEEDED",
                engine="test-native-fixture",
                native_extension_path="",
                native_extension_sha256="",
                diagnostics={
                    "success": True,
                    "failure_reason": "",
                    "constraint_profile": profile,
                    "sfc_cells": ([{"vertices": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], "halfspaces": [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [-1.0, 0.0, 0.0]]}] if profile == "superplanner_sfc_v1" else []),
                },
                samples=samples,
                waypoints=samples[:, 1:4],
            )
            metrics, detail = compute_static_case_metrics(
                profile_case, result, limits
            )
            metrics["safe_dist_m"] = limits["safe_distance_m"]
            per_case[(case.case_uid, profile)] = (
                profile_case,
                result,
                metrics,
                detail,
            )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            _render_selected_artifacts(
                [case.case_uid], per_case, output, frozen
            )
            gif_path = (
                output
                / "selected_artifacts"
                / case.case_uid
                / f"{case.case_uid}_legacy_vs_superplanner_sfc.gif"
            )
            package = gif_path.with_name(f"{gif_path.stem}_evidence")
            for name in (
                "caption.md",
                "caption.txt",
                "caption_zh.md",
                "frame_metrics.csv",
                "event_timeline.csv",
                "video_manifest.json",
                "evidence_manifest.json",
                "validation.json",
            ):
                self.assertTrue((package / name).is_file(), name)
            import csv

            with (package / "frame_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                frame_rows = list(csv.DictReader(stream))
            self.assertTrue(frame_rows)
            self.assertEqual(tuple(frame_rows[0]), tuple(FRAME_METRICS_FIELDS))
            validation = json.loads(
                (package / "validation.json").read_text(encoding="utf-8")
            )
            self.assertTrue(validation["valid"])
            self.assertEqual(validation["errors"], [])
            self.assertIn(
                str((package / "evidence_manifest.json").relative_to(output)),
                frozen["cases"][case.case_uid]["artifact_paths"],
            )


if __name__ == "__main__":
    unittest.main()
