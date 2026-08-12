"""Tests that static GIFs contain required visual elements (Plan A)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from experiments.static.case_schema import StaticCase
from experiments.static.runner import StaticRunResult


def _make_straight_case(profile: str = "legacy") -> StaticCase:
    """Minimal straight case with known geometry for rendering tests."""
    occupancy = np.zeros((8, 10), dtype=bool)
    occupancy[3:5, 6] = True
    distance = np.ones_like(occupancy, dtype=np.float64)
    distance[occupancy] = -0.1
    return StaticCase(
        case_uid="test_straight",
        case_source="STATIC_SYNTHETIC",
        guide_path_xyz=np.array([[0.5, 0.5, 0.0], [3.5, 0.5, 0.0]], dtype=np.float64),
        occupancy=occupancy,
        esdf_distance=distance,
        esdf_origin=np.array([0.0, 0.0], dtype=np.float64),
        esdf_resolution=0.5,
        start_position=np.array([0.5, 0.5, 0.0]),
        start_velocity=np.array([0.1, 0.0, 0.0]),
        start_acceleration=np.zeros(3),
        start_yaw=0.0,
        start_yaw_rate=0.05,
        terminal_goal=np.array([3.5, 0.5, 0.0]),
        constraint_profile=profile,
        expected_category="straight",
    )


def _make_result(case: StaticCase, status: str = "SUCCEEDED") -> StaticRunResult:
    """Create a StaticRunResult with synthetic samples matching the case."""
    n = 32
    samples = np.zeros((n, 15), dtype=np.float64)
    samples[:, 0] = np.linspace(0.0, 3.1, n)  # time
    x_vals = np.linspace(case.start_position[0], case.terminal_goal[0], n)
    samples[:, 1] = x_vals
    samples[:, 2] = 0.5  # y constant
    samples[:, 3] = 0.0
    samples[:, 4] = 1.0  # vx
    samples[:, 5] = 0.0
    samples[:, 6] = 0.0
    samples[:, 7] = 0.1  # ax
    samples[:, 8] = 0.0
    samples[:, 9] = 0.0
    samples[:, 10] = 0.01  # jx
    samples[:, 11] = 0.0
    samples[:, 12] = 0.0
    samples[:, 13] = 0.0  # yaw
    samples[:, 14] = 0.0  # yaw rate
    return StaticRunResult(
        case_uid=case.case_uid,
        case_hash=case.case_hash,
        mode="recompute",
        status=status,
        engine="test",
        native_extension_path="",
        native_extension_sha256="",
        diagnostics={"success": True},
        samples=samples,
        waypoints=samples[:, 1:4],
    )


def _make_detail(path: np.ndarray) -> dict:
    """Create a minimal detail dict with clearance data."""
    n = len(path)
    return {
        "clearance_xy": path[:, :2].copy(),
        "clearance_m": np.full(n, 0.5, dtype=np.float64),
        "clearance_valid": np.ones(n, dtype=bool),
        "clearance_s_m": np.linspace(0.0, 3.0, n),
        "t_s": np.linspace(0.0, 3.1, n),
        "speed_mps": np.full(n, 1.0),
        "acc_mps2": np.full(n, 0.1),
        "jerk_mps3": np.full(n, 0.01),
        "yaw_rate_radps": np.full(n, 0.0),
    }


class StaticGifContentTests(unittest.TestCase):
    """Verify that rendered static GIFs contain Plan-A required elements."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_single_gif_has_required_artifacts(self):
        """Single-case rendering produces GIF and evidence package."""
        from experiments.visualizers.static_benchmark import render_static_case

        case = _make_straight_case()
        result = _make_result(case)
        path = result.samples[:, 1:4]
        detail = _make_detail(path)
        metrics = {"safe_dist_m": 0.15, "min_clearance_m": 0.45}

        artifacts = render_static_case(case, result, metrics, detail, self.output)
        artifact_names = {Path(p).name for p in artifacts}

        self.assertIn("test_straight_animation.gif", artifact_names)
        self.assertIn("test_straight_overview.png", artifact_names)
        self.assertIn("test_straight_metrics.json", artifact_names)

    def test_gif_decodes_with_correct_frame_count(self):
        """GIF frame count matches the intended number of frames."""
        from experiments.visualizers.static_benchmark import render_static_case

        case = _make_straight_case()
        result = _make_result(case)
        path = result.samples[:, 1:4]
        detail = _make_detail(path)
        metrics = {"safe_dist_m": 0.15, "min_clearance_m": 0.45}

        artifacts = render_static_case(case, result, metrics, detail, self.output)
        gif_path = self.output / "test_straight_animation.gif"
        self.assertTrue(gif_path.is_file(), f"GIF not found at {gif_path}")

        frames = imageio.mimread(gif_path)
        self.assertGreater(len(frames), 0, "GIF has zero frames")
        # Default: min(32, len(path)) = 32 frames
        self.assertEqual(len(frames), 32, f"Expected 32 frames, got {len(frames)}")

    def test_gif_frames_are_rgb(self):
        """Every GIF frame is a valid RGB image."""
        from experiments.visualizers.static_benchmark import render_static_case

        case = _make_straight_case()
        result = _make_result(case)
        path = result.samples[:, 1:4]
        detail = _make_detail(path)
        metrics = {"safe_dist_m": 0.15, "min_clearance_m": 0.45}

        artifacts = render_static_case(case, result, metrics, detail, self.output)
        gif_path = self.output / "test_straight_animation.gif"

        frames = imageio.mimread(gif_path)
        for i, frame in enumerate(frames):
            self.assertEqual(frame.ndim, 3, f"Frame {i} is not 3D")
            self.assertEqual(frame.shape[2], 3, f"Frame {i} is not RGB")

    def test_evidence_package_exists_and_validates(self):
        """Evidence package passes validation."""
        from experiments.visualizers.static_benchmark import render_static_case

        case = _make_straight_case()
        result = _make_result(case)
        path = result.samples[:, 1:4]
        detail = _make_detail(path)
        metrics = {"safe_dist_m": 0.15, "min_clearance_m": 0.45}

        render_static_case(case, result, metrics, detail, self.output)
        evidence_dir = self.output / "test_straight_animation_evidence"
        self.assertTrue(evidence_dir.is_dir(), f"Evidence dir missing: {evidence_dir}")

        validation = json.loads((evidence_dir / "validation.json").read_text())
        self.assertTrue(validation.get("valid"), f"Validation failed: {validation}")
        self.assertEqual(validation.get("errors", ["unexpected"]), [])

    def test_frame_metrics_csv_aligns_with_gif(self):
        """frame_metrics.csv row count == decoded GIF frame count."""
        import csv
        from experiments.visualizers.static_benchmark import render_static_case

        case = _make_straight_case()
        result = _make_result(case)
        path = result.samples[:, 1:4]
        detail = _make_detail(path)
        metrics = {"safe_dist_m": 0.15, "min_clearance_m": 0.45}

        render_static_case(case, result, metrics, detail, self.output)
        gif_path = self.output / "test_straight_animation.gif"
        gif_frames = imageio.mimread(gif_path)
        evidence_dir = self.output / "test_straight_animation_evidence"

        with (evidence_dir / "frame_metrics.csv").open(newline="") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)

        self.assertEqual(
            len(csv_rows), len(gif_frames),
            f"CSV rows ({len(csv_rows)}) != GIF frames ({len(gif_frames)})"
        )

    def test_caption_contains_required_chinese_fields(self):
        """Caption files contain all required Chinese fields."""
        from experiments.visualizers.static_benchmark import render_static_case

        case = _make_straight_case()
        result = _make_result(case)
        path = result.samples[:, 1:4]
        detail = _make_detail(path)
        metrics = {"safe_dist_m": 0.15, "min_clearance_m": 0.45}

        render_static_case(case, result, metrics, detail, self.output)
        evidence_dir = self.output / "test_straight_animation_evidence"

        caption = (evidence_dir / "caption_zh.md").read_text(encoding="utf-8")
        required_fields = [
            "研究问题", "数据来源", "配对键", "分母", "局限性", "解读",
        ]
        for field in required_fields:
            self.assertIn(field, caption, f"Caption missing field: {field}")

    def test_artifact_receipt_covers_all_files(self):
        """Every file in evidence package has a matching receipt entry."""
        import hashlib
        from experiments.visualizers.static_benchmark import render_static_case

        case = _make_straight_case()
        result = _make_result(case)
        path = result.samples[:, 1:4]
        detail = _make_detail(path)
        metrics = {"safe_dist_m": 0.15, "min_clearance_m": 0.45}

        render_static_case(case, result, metrics, detail, self.output)
        evidence_dir = self.output / "test_straight_animation_evidence"

        receipt = json.loads((evidence_dir / "artifact_receipt.json").read_text())
        receipted = {str(r["path"]) for r in receipt["artifacts"]}
        actual = {
            p.relative_to(evidence_dir).as_posix()
            for p in evidence_dir.iterdir()
            if p.is_file() and p.name != "artifact_receipt.json"
        }

        missing_from_receipt = actual - receipted
        self.assertEqual(
            missing_from_receipt, set(),
            f"Files not in receipt: {missing_from_receipt}"
        )

        for r in receipt["artifacts"]:
            fp = evidence_dir / r["path"]
            self.assertTrue(fp.is_file(), f"Receipted file missing: {r['path']}")
            if fp.stat().st_size != r.get("size_bytes"):
                self.fail(f"Size mismatch for {r['path']}")
            digest = hashlib.sha256(fp.read_bytes()).hexdigest()
            self.assertEqual(digest, r["sha256"], f"Hash mismatch for {r['path']}")
