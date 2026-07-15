import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.analyzers.artifact_manifest import validate_artifact_manifest
from experiments.designers.manifest import load_manifest
from experiments.orchestrators.suite_runner import run_suite
from experiments.cli.main import build_parser


class FullPipelineTests(unittest.TestCase):
    def test_cli_accepts_full_orchestration_flags(self):
        args = build_parser().parse_args(["run-suite", "--config", "suite.json", "--backend", "mock", "--resume", "--retry-failed", "--dry-run", "--skip-video", "--analysis-only"])
        self.assertTrue(args.resume and args.retry_failed and args.dry_run and args.skip_video and args.analysis_only)

    def test_mock_manifest_covers_eight_scenarios_in_two_scenes(self):
        manifest = load_manifest("experiments/configs/mock_full_scenarios.json")
        scenario_ids = {episode.scenario_id for scene in manifest.scenes for episode in scene.episodes}
        self.assertEqual(scenario_ids, {f"SCN-{index:02d}" for index in range(1, 9)})
        self.assertEqual({scene.scene_label for scene in manifest.scenes}, {"SPARSE", "DENSE"})

    def test_full_mock_pipeline_generates_required_artifacts(self):
        root = Path(tempfile.mkdtemp())
        config = json.loads(Path("experiments/configs/mock_full_suite.json").read_text())
        config["output_root"] = str(root)
        config_path = root / "suite.json"; config_path.write_text(json.dumps(config))
        # Resolve manifest independently from temporary config location.
        config["manifest"] = str(Path("experiments/configs/mock_full_scenarios.json").resolve())
        config_path.write_text(json.dumps(config))
        result = run_suite(config_path, backend_name="mock", resume=True)
        self.assertEqual(result.completed, 6)
        suite = root / config["suite_id"]
        self.assertEqual(json.loads((suite / "suite_status.json").read_text())["status"], "COMPLETE")
        self.assertTrue((suite / "scenario_manifest.json").exists())
        cycles = list(suite.glob("experiments/*/*/*/*/*/planning_cycles.csv"))
        self.assertEqual(len(cycles), 6)
        statuses = set()
        for path in cycles:
            with path.open() as stream:
                statuses.update(row["fallback_mode"] for row in csv.DictReader(stream))
        self.assertTrue({"NONE", "HOLD_LAST", "STOP"}.issubset(statuses))
        for exp in range(9):
            report_dir = suite / "reports" / f"EXP-{exp:02d}_{['headless','raw_profile','safety','smoothness','warm_start','control','navigation','timing','failures'][exp]}"
            self.assertTrue((report_dir / "report.md").exists(), report_dir)
        for name in ("table_data_quality.csv", "table_raw_profile.csv", "table_safety_repair.csv", "table_smoothness.csv", "table_warm_start.csv", "table_control_navigation.csv", "table_timing.csv"):
            self.assertTrue((suite / "reports" / "core_tables" / name).exists(), name)
        self.assertTrue((suite / "reports" / "representative_cases.csv").exists())
        self.assertTrue((suite / "reports" / "failure_case_report.md").exists())
        self.assertEqual(validate_artifact_manifest(suite), [])
        self.assertIn("SIMULATED", (suite / "reports" / "suite_report.md").read_text())
        with (suite / "reports" / "core_tables" / "table_control_navigation.csv").open() as stream:
            control_rows = list(csv.DictReader(stream))
        self.assertIn("success_rate", {row["metric"] for row in control_rows})
        self.assertTrue(list((suite / "reports" / "paired").glob("*/*/paired_comparison.csv")))
        self.assertNotIn("mock_pair", (suite / "reports" / "EXP-06_navigation" / "paired_metrics.csv").read_text())
        for run_dir in suite.glob("experiments/*/*/*/*/*"):
            if run_dir.is_dir(): self.assertTrue((run_dir / "validation" / "validation_report.json").exists())

    def test_all_visualizer_modules_are_importable(self):
        for name in ("raw_profile", "safety", "smoothness", "warm_start", "control", "navigation", "timing", "failures"):
            module = __import__(f"experiments.visualizers.{name}", fromlist=["PLOT_FILENAMES"])
            self.assertTrue(module.PLOT_FILENAMES)
