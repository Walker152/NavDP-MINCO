from __future__ import annotations

import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class ResearchWorkflowTests(unittest.TestCase):
    def test_retry_failed_rebuilds_missing_outputs_from_completed_receipt(self):
        from experiments.orchestrators.research_workflow import (
            WorkflowOptions,
            _run_stage,
        )

        output = self.output
        target = output / "stage-output.txt"
        calls = []
        first = _run_stage(
            output_root=output,
            options=WorkflowOptions(output_root=output),
            name="recoverable",
            command=("fixture",),
            input_paths=(),
            output_paths=(target,),
            action=lambda: (
                calls.append("first"), target.parent.mkdir(parents=True, exist_ok=True),
                target.write_text("first\n", encoding="utf-8"),
            ),
        )
        self.assertEqual(first["status"], "COMPLETE")
        target.unlink()
        second = _run_stage(
            output_root=output,
            options=WorkflowOptions(
                output_root=output, resume=True, retry_failed=True
            ),
            name="recoverable",
            command=("fixture",),
            input_paths=(),
            output_paths=(target,),
            action=lambda: (
                calls.append("rebuild"), target.write_text("rebuild\n", encoding="utf-8"),
            ),
        )
        self.assertEqual(second["status"], "COMPLETE")
        self.assertEqual(calls, ["first", "rebuild"])

    def test_retry_failed_rebuilds_completed_stage_after_input_changes(self):
        from experiments.orchestrators.research_workflow import (
            WorkflowOptions,
            _run_stage,
        )

        input_path = self.root / "input.txt"
        input_path.write_text("v1\n", encoding="utf-8")
        target = self.output / "stage-output.txt"
        calls = []
        for expected in ("first", "rebuild"):
            _run_stage(
                output_root=self.output,
                options=WorkflowOptions(
                    output_root=self.output,
                    resume=bool(calls),
                    retry_failed=bool(calls),
                ),
                name="input-recoverable",
                command=("fixture",),
                input_paths=(input_path,),
                output_paths=(target,),
                action=lambda expected=expected: (
                    calls.append(expected),
                    target.write_text(expected + "\n", encoding="utf-8"),
                ),
            )
            if expected == "first":
                input_path.write_text("v2\n", encoding="utf-8")
        self.assertEqual(calls, ["first", "rebuild"])

    def setUp(self):
        self.repo = Path(__file__).parents[2].resolve()
        self.root = Path(tempfile.mkdtemp())
        self.output = self.root / "research"
        self.config_dir = self.root / "configs"
        self.config_dir.mkdir()
        for name in (
            "static_legacy_suite.json",
            "static_safe_corridor_suite.json",
            "static_superplanner_sfc_suite.json",
            "static_boundary_selection_v1.json",
        ):
            shutil.copy2(
                self.repo / "experiments" / "configs" / name,
                self.config_dir / name,
            )

    def test_static_workflow_runs_from_empty_output_and_resume_hashes_inputs(self):
        from experiments.orchestrators.research_workflow import (
            WorkflowOptions,
            run_static_workflow,
        )

        options = WorkflowOptions(
            output_root=self.output,
            legacy_config=self.config_dir / "static_legacy_suite.json",
            safe_config=self.config_dir / "static_superplanner_sfc_suite.json",
            selection_config=(
                self.config_dir / "static_boundary_selection_v1.json"
            ),
        )
        receipt = run_static_workflow(repo_root=self.repo, options=options)

        self.assertEqual(receipt["status"], "COMPLETE")
        self.assertEqual(
            receipt["stages"]["legacy_benchmark"]["status"], "COMPLETE"
        )
        self.assertEqual(
            receipt["stages"]["superplanner_sfc_benchmark"]["status"], "COMPLETE"
        )
        self.assertTrue((self.output / "paper" / "report.md").is_file())
        self.assertTrue(
            (self.output / "paper" / "paper_manifest.json").is_file()
        )
        self.assertTrue((self.output / "paper" / "static").is_dir())
        self.assertTrue((self.output / "artifact_manifest.json").is_file())

        selection = self.config_dir / "static_boundary_selection_v1.json"
        selection.write_text(
            selection.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "input hash changed"):
            run_static_workflow(
                repo_root=self.repo,
                options=WorkflowOptions(
                    output_root=self.output,
                    resume=True,
                    legacy_config=options.legacy_config,
                    safe_config=options.safe_config,
                    selection_config=options.selection_config,
                ),
            )

    def test_workflow_cli_commands_share_safe_real_simulation_gate(self):
        from experiments.cli.main import build_parser

        parser = build_parser()
        for command in (
            "run-static-workflow",
            "run-simulation-workflow",
            "run-all-workflows",
        ):
            args = parser.parse_args([command, "--output", str(self.output)])
            self.assertEqual(Path(args.output), self.output)
            self.assertFalse(args.allow_real_simulation)
            self.assertFalse(args.full_suite)
            self.assertFalse(args.skip_video)
            self.assertFalse(args.skip_rolling_showcase)
            self.assertIsNone(args.rolling_showcase_config)

    def test_workflow_cli_accepts_explicit_rolling_showcase_options(self):
        from experiments.cli.main import build_parser, _workflow_options

        config = self.root / "rolling.json"
        config.write_text("{}\n", encoding="utf-8")
        args = build_parser().parse_args(
            [
                "run-static-workflow",
                "--output",
                str(self.output),
                "--rolling-showcase-config",
                str(config),
                "--skip-rolling-showcase",
            ]
        )

        options = _workflow_options(args)

        self.assertEqual(options.rolling_showcase_config, config)
        self.assertTrue(options.skip_rolling_showcase)

    def test_showcase_stage_is_resumable_and_hash_validated(self):
        from experiments.orchestrators.research_workflow import (
            WorkflowOptions,
            _run_rolling_showcase_stage,
        )

        config = self.root / "rolling.json"
        config.write_text('{"schema_version": 1}\n', encoding="utf-8")
        options = WorkflowOptions(
            output_root=self.output,
            rolling_showcase_config=config,
        )

        with patch(
            "experiments.orchestrators.research_workflow.run_rolling_showcase"
        ) as runner:
            def generate(_config, output):
                output = Path(output)
                output.mkdir(parents=True)
                (output / "showcase_manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "paired_key": [
                                "scenario_uid",
                                "seed",
                                "initial_state_hash",
                            ],
                            "scenes": [],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (output / "README_成果索引.md").write_text(
                    "# 结果\n", encoding="utf-8"
                )
                for name in (
                    "01_trajectory_optimization",
                    "02_safe_corridor",
                    "03_initial_state",
                    "04_dynamic_obstacles",
                    "05_extreme_cases",
                    "06_aggregate_figures",
                ):
                    (output / name).mkdir()

            runner.side_effect = generate
            stage = _run_rolling_showcase_stage(
                output_root=self.output,
                options=options,
                input_paths=[config],
            )

        self.assertEqual(stage["status"], "COMPLETE")
        self.assertTrue((self.output / "paper_showcase").is_dir())
        config.write_text('{"schema_version": 2}\n', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "input hash changed"):
            _run_rolling_showcase_stage(
                output_root=self.output,
                options=WorkflowOptions(
                    output_root=self.output,
                    rolling_showcase_config=config,
                    resume=True,
                ),
                input_paths=[config],
            )

    def test_static_validation_requires_showcase_by_default(self):
        from experiments.orchestrators.research_workflow import _static_validation

        with patch(
            "experiments.orchestrators.research_workflow.validate_static_benchmark",
            return_value=[],
        ), patch(
            "experiments.orchestrators.research_workflow._validate_inventory",
            return_value=[],
        ), patch(
            "experiments.orchestrators.research_workflow.validate_paper_artifact_manifest",
            return_value=[],
        ), patch(
            "experiments.orchestrators.research_workflow.validate_showcase",
            return_value=["missing heading arrows"],
        ):
            with self.assertRaisesRegex(RuntimeError, "missing heading arrows"):
                _static_validation(self.output)

    def test_dynamic_real_stage_uses_the_prepared_suite_id(self):
        from experiments.orchestrators.research_workflow import _dynamic_suite_dir

        dynamic = self.output / "dynamic_readiness"
        dynamic.mkdir(parents=True)
        suite = dynamic / "dynamic_suite.json"
        expected = dynamic / "dry_run_results" / "task06_dynamic_sparse_dense_narrow_folded_v1"
        suite.write_text(
            json.dumps(
                {
                    "suite_id": "task06_dynamic_sparse_dense_narrow_folded_v1",
                    "output_root": str(dynamic / "dry_run_results"),
                }
            ) + "\n",
            encoding="utf-8",
        )

        self.assertEqual(_dynamic_suite_dir(suite), expected)

    def test_simulation_validation_accepts_three_method_twelve_run_plan(self):
        from experiments.orchestrators.research_workflow import _simulation_validation

        readiness = self.output / "dynamic_readiness"
        readiness.mkdir(parents=True)
        plan = readiness / "dry_run_plan.json"
        plan.write_text(json.dumps({"run_count": 12, "started_processes": 0}) + "\n")
        from experiments.core.artifact_receipt import sha256_file
        (readiness / "dynamic_readiness_receipt.json").write_text(
            json.dumps({
                "run_count": 12, "started_processes": 0,
                "dry_run_plan": str(plan), "dry_run_plan_sha256": sha256_file(plan),
            }) + "\n", encoding="utf-8"
        )
        mock = self.output / "simulation" / "mock_smoke"
        mock.mkdir(parents=True)
        (mock / "mock_smoke_receipt.json").write_text(
            json.dumps({"status": "COMPLETE", "failed_runs": 0}) + "\n",
            encoding="utf-8",
        )

        _simulation_validation(self.output)

        payload = json.loads((self.output / "validation" / "simulation_validation.json").read_text())
        self.assertTrue(payload["valid"], payload)

    def test_paper_stage_uses_unified_data_driven_report_generator(self):
        from experiments.orchestrators.research_workflow import _paper

        paper_dir = self.root / "paper"
        manifest = _paper(self.root, paper_dir)

        self.assertEqual(manifest["data_source"], "STATIC_ONLY")
        self.assertEqual(manifest["dynamic_evidence"], "UNAVAILABLE")
        self.assertTrue((paper_dir / "paper_manifest.json").is_file())
        self.assertTrue((paper_dir / "artifact_receipt.json").is_file())
        self.assertIn("UNAVAILABLE", (paper_dir / "report.md").read_text())

    def test_dynamic_paper_stage_has_an_independent_immutable_output(self):
        from experiments.orchestrators.research_workflow import _dynamic_paper

        dynamic_suite = self.root / "real_dynamic_suite"
        dynamic_suite.mkdir()
        output = self.root / "paper" / "dynamic_report"
        manifest = _dynamic_paper(dynamic_suite, output)

        self.assertEqual(manifest["data_source"], "STATIC_ONLY")
        self.assertTrue((output / "paper_manifest.json").is_file())
        self.assertTrue((output / "tables" / "data_quality.csv").is_file())

    def test_mock_smoke_stage_runs_locally_and_completes(self):
        from experiments.orchestrators.research_workflow import (
            run_mock_smoke_suite,
        )

        receipt = run_mock_smoke_suite(
            repo_root=self.repo,
            output_dir=self.root / "mock_smoke",
        )

        self.assertEqual(receipt["status"], "COMPLETE")
        self.assertEqual(receipt["completed_runs"], 6)
        self.assertTrue(Path(receipt["suite_dir"]).is_dir())
        self.assertTrue(
            (Path(receipt["suite_dir"]) / "reports" / "suite_report.md").is_file()
        )

    def test_all_workflow_writes_validated_experiment_receipt(self):
        from experiments.orchestrators.research_workflow import (
            WorkflowOptions,
            run_all_workflows,
        )

        complete = {"status": "COMPLETE", "stages": {}, "validation_errors": []}
        with patch(
            "experiments.orchestrators.research_workflow.run_static_workflow",
            return_value=complete,
        ), patch(
            "experiments.orchestrators.research_workflow.run_simulation_workflow",
            return_value=complete,
        ), patch(
            "experiments.orchestrators.research_workflow._write_artifact_manifest",
            return_value={"artifact_count": 0},
        ):
            receipt = run_all_workflows(
                repo_root=self.repo,
                options=WorkflowOptions(output_root=self.output),
            )

        self.assertEqual(receipt["status"], "COMPLETE")
        self.assertEqual(receipt["validation_errors"], [])
        stored = self.output / "experiment_receipt.json"
        self.assertTrue(stored.is_file())
        self.assertEqual(receipt, json.loads(stored.read_text()))

    def test_environment_receipt_records_static_and_simulation_interpreters(self):
        from experiments.orchestrators.research_workflow import (
            _environment_receipt,
        )

        navdp = self.root / "envs" / "navdp" / "bin" / "python"
        isaac = self.root / "envs" / "isaaclab" / "bin" / "python"
        navdp.parent.mkdir(parents=True)
        isaac.parent.mkdir(parents=True)
        navdp.write_text("navdp", encoding="utf-8")
        isaac.write_text("isaac", encoding="utf-8")

        with patch.dict(
            "os.environ",
            {
                "NAVDP_PYTHON": str(navdp),
                "ISAACLAB_PYTHON": str(isaac),
            },
            clear=False,
        ):
            receipt = _environment_receipt()

        self.assertEqual(receipt["static_analysis"]["python"], str(navdp.resolve()))
        self.assertEqual(receipt["real_simulation"]["python"], str(isaac.resolve()))
        self.assertEqual(receipt["static_analysis"]["environment"], "navdp")
        self.assertEqual(receipt["real_simulation"]["environment"], "isaaclab")

    def test_unavailable_isaac_environment_is_explicit_not_fabricated(self):
        from experiments.orchestrators.research_workflow import (
            _environment_receipt,
        )

        with patch.dict(
            "os.environ",
            {"ISAACLAB_PYTHON": str(self.root / "missing" / "python")},
            clear=False,
        ):
            receipt = _environment_receipt()

        self.assertFalse(receipt["real_simulation"]["available"])
        self.assertEqual(
            receipt["real_simulation"]["status"], "PENDING_REAL_SIMULATION"
        )

    def test_real_simulation_requires_verified_autodl_runtime(self):
        """A local Isaac interpreter must never authorize a real rollout."""
        from experiments.orchestrators.research_workflow import (
            WorkflowOptions,
            run_simulation_workflow,
        )

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "AutoDL server runtime"):
                run_simulation_workflow(
                    repo_root=self.repo,
                    options=WorkflowOptions(
                        output_root=self.output,
                        allow_real_simulation=True,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
