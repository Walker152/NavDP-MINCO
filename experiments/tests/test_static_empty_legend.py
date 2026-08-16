from __future__ import annotations

import json
import logging
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.static.runner import StaticRunResult
from experiments.static.synthetic import (
    _build_static_case,
    generate_catalogue,
    generate_obstacle_variants,
)
from experiments.visualizers.static_benchmark import render_static_case


class StaticEmptyLegendTests(unittest.TestCase):
    def test_failed_case_without_clearance_samples_emits_no_legend_warning(self):
        config = Path(__file__).parents[1] / "configs" / "static_legacy_suite.json"
        case = generate_catalogue(config)[0]
        result = StaticRunResult(
            case_uid=case.case_uid,
            case_hash=case.case_hash,
            mode="recompute",
            status="FAILED",
            engine="test",
            native_extension_path="",
            native_extension_sha256="",
            diagnostics={"success": False},
            samples=np.empty((0, 15), dtype=np.float64),
            waypoints=np.empty((0, 3), dtype=np.float64),
        )
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        logger = logging.getLogger("matplotlib.legend")
        handler = Capture()
        logger.addHandler(handler)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                render_static_case(
                    case,
                    result,
                    {"safe_dist_m": 0.15},
                    {},
                    Path(temporary),
                )
        finally:
            logger.removeHandler(handler)

        messages = [record.getMessage() for record in records]
        self.assertFalse(
            any("No artists with labels found" in message for message in messages),
            messages,
        )


class StaticCatalogueBehaviorTests(unittest.TestCase):
    def _legacy_config_path(self) -> Path:
        return (
            Path(__file__).parents[1] / "configs" / "static_legacy_suite.json"
        )

    def test_astar_guide_rewrite_is_opt_in_and_tracked(self):
        cases = {
            case.case_uid: case
            for case in generate_catalogue(self._legacy_config_path())
        }
        through = cases["syn_through_obstacle"]

        # No opt-in flag in the legacy config: the guide must stay as
        # authored, with its original 3 waypoints running through the obstacle.
        self.assertEqual(through.references.get("guide_rewritten"), "false")
        self.assertEqual(through.guide_path_xyz.shape, (3, 3))
        np.testing.assert_allclose(
            through.guide_path_xyz,
            [[0.0, 1.0, 0.0], [4.0, 1.0, 0.0], [8.0, 1.0, 0.0]],
        )
        occupancy, resolution, origin = (
            through.occupancy,
            through.esdf_resolution,
            through.esdf_origin,
        )
        crossing_cells = []
        for i in range(len(through.guide_path_xyz) - 1):
            for point in np.linspace(
                through.guide_path_xyz[i],
                through.guide_path_xyz[i + 1],
                50,
            ):
                col = int((point[0] - origin[0]) / resolution)
                row = int((point[1] - origin[1]) / resolution)
                if 0 <= row < occupancy.shape[0] and 0 <= col < occupancy.shape[1]:
                    crossing_cells.append(bool(occupancy[row, col]))
        self.assertTrue(any(crossing_cells), "guide must go through the obstacle")

        # An explicit opt-in clone of the same case IS rewritten and tracked.
        config = json.loads(
            self._legacy_config_path().read_text(encoding="utf-8")
        )
        source = next(
            spec
            for spec in config["cases"]
            if spec["case_uid"] == "syn_through_obstacle"
        )
        opted_in = _build_static_case(
            {**source, "allow_astar_rewrite": True},
            config["grid"],
            profile=str(config["constraint_profile"]),
        )
        self.assertEqual(opted_in.references.get("guide_rewritten"), "true")
        self.assertNotEqual(
            opted_in.guide_path_xyz.shape, through.guide_path_xyz.shape
        )
        np.testing.assert_allclose(
            opted_in.guide_path_xyz[0], through.guide_path_xyz[0]
        )
        np.testing.assert_allclose(
            opted_in.guide_path_xyz[-1], through.guide_path_xyz[-1]
        )

    def test_obstacle_variants_preserve_source_and_target_clearance(self):
        cases = {
            case.case_uid: case
            for case in generate_catalogue(self._legacy_config_path())
        }
        for uid, clearance in (
            ("syn_straight_sparse", 0.60),
            ("syn_straight_dense", 0.32),
            ("syn_l90_sparse", 0.60),
            ("syn_l90_dense", 0.45),
            ("syn_s_curve_sparse", 0.60),
            ("syn_s_curve_dense", 0.32),
        ):
            variant = cases[uid]
            source = cases[uid.rsplit("_", 1)[0]]
            # Same guide path, start state and ESDF grid as the source case.
            np.testing.assert_allclose(
                variant.guide_path_xyz, source.guide_path_xyz
            )
            np.testing.assert_allclose(
                variant.start_position, source.start_position
            )
            self.assertEqual(
                variant.esdf_distance.shape, source.esdf_distance.shape
            )
            self.assertEqual(variant.esdf_resolution, source.esdf_resolution)
            np.testing.assert_allclose(variant.esdf_origin, source.esdf_origin)
            self.assertIn("OBSTACLE_VARIANT", variant.tags)
            self.assertEqual(
                variant.expected_category,
                f"{source.expected_category}_{uid.rsplit('_', 1)[1]}",
            )
            self.assertEqual(
                variant.references.get("guide_rewritten"), "false"
            )

            # Minimum passage clearance (distance from the guide to the
            # nearest obstacle rectangle edge) matches the density target.
            guide = variant.guide_path_xyz[:, :2]
            dense_guide = np.concatenate(
                [
                    np.linspace(guide[i], guide[i + 1], 200)
                    for i in range(len(guide) - 1)
                ]
            )
            rectangles = variant.auxiliary_arrays[
                "materialization_obstacle_rectangles_xyxy_m"
            ]
            min_clearance = min(
                float(
                    np.hypot(
                        dense_guide[:, 0] - np.clip(dense_guide[:, 0], x0, x1),
                        dense_guide[:, 1] - np.clip(dense_guide[:, 1], y0, y1),
                    ).min()
                )
                for x0, y0, x1, y1 in rectangles
            )
            self.assertAlmostEqual(min_clearance, clearance, delta=0.02)

    def test_generate_obstacle_variants_standalone_uses_config_schema(self):
        config = json.loads(
            self._legacy_config_path().read_text(encoding="utf-8")
        )
        variants = generate_obstacle_variants(config)
        self.assertEqual(
            {case.case_uid for case in variants},
            {
                "syn_straight_dense",
                "syn_straight_sparse",
                "syn_l90_dense",
                "syn_l90_sparse",
                "syn_s_curve_dense",
                "syn_s_curve_sparse",
            },
        )
        # Variants must never inherit an A* rewrite opt-in from a source.
        self.assertTrue(
            all(case.references.get("guide_rewritten") == "false" for case in variants)
        )


if __name__ == "__main__":
    unittest.main()
