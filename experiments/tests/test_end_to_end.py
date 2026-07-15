import json
import tempfile
import unittest
from unittest import mock

from experiments.orchestrators.suite_runner import run_suite


class EndToEndTests(unittest.TestCase):
 def test_mock_suite_generates_validated_report_and_resumes(self):
    tmp_path = __import__('pathlib').Path(tempfile.mkdtemp())
    manifest = tmp_path / "scenarios.json"
    manifest.write_text(json.dumps({"manifest_version": 1, "manifest_id": "mock", "seed": 0, "scenes": [{
        "scene_id": "sparse_0", "scene_label": "SPARSE", "scene_path": "mock://sparse", "episodes": [
            {"scenario_id": "SCN-01", "episode_index": 0, "seed": 1, "start_pose": [0,0,0], "goal_pose": [2,0,0]}
        ]}]}))
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"suite_id": "smoke", "output_root": str(tmp_path / "results"), "manifest": str(manifest), "runs": [
        {"experiment_id": "EXP-01_raw_profile", "variant": "raw", "warm_start_mode": "cold"},
        {"experiment_id": "EXP-04_warm_start", "variant": "minco-hot", "warm_start_mode": "gated"}
    ]}))
    first = run_suite(suite, backend_name="mock", resume=True)
    second = run_suite(suite, backend_name="mock", resume=True)
    self.assertEqual(first.completed, 2)
    self.assertEqual(second.skipped, 2)
    report = tmp_path / "results" / "smoke" / "reports" / "suite_report.md"
    self.assertIn("SIMULATED", report.read_text())

 def _completed_suite(self):
    tmp_path = __import__('pathlib').Path(tempfile.mkdtemp())
    manifest = tmp_path / "scenarios.json"
    manifest.write_text(json.dumps({"manifest_version":1,"manifest_id":"resume","seed":0,"scenes":[{
        "scene_id":"s","scene_label":"SPARSE","scene_path":"mock://s","episodes":[
            {"scenario_id":"SCN-01","episode_index":0,"seed":1,"start_pose":[0,0,0],"goal_pose":[1,0,0]}
        ]}]}))
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"suite_id":"resume","output_root":str(tmp_path / "results"),"manifest":str(manifest),"runs":[
        {"experiment_id":"EXP-01_raw_profile","variant":"raw","warm_start_mode":"cold"}
    ]}))
    run_suite(suite, backend_name="mock", resume=True)
    run_dir = next((tmp_path / "results" / "resume" / "experiments").glob("*/*/*/*/*"))
    return suite, run_dir

 def test_resume_continues_from_simulation_complete_without_rerunning_backend(self):
    suite, run_dir = self._completed_suite()
    status = json.loads((run_dir / "run_status.json").read_text()); status["status"] = "SIMULATION_COMPLETE"
    (run_dir / "run_status.json").write_text(json.dumps(status))
    with mock.patch("experiments.simulators.mock_backend.MockBackend.run", side_effect=AssertionError("must not rerun")):
        result = run_suite(suite, backend_name="mock", resume=True)
    self.assertEqual(result.completed, 1)
    self.assertEqual(json.loads((run_dir / "run_status.json").read_text())["status"], "COMPLETE")

 def test_failed_run_is_not_retried_without_retry_failed(self):
    suite, run_dir = self._completed_suite()
    status = json.loads((run_dir / "run_status.json").read_text()); status["status"] = "FAILED"
    (run_dir / "run_status.json").write_text(json.dumps(status))
    with mock.patch("experiments.simulators.mock_backend.MockBackend.run", side_effect=AssertionError("failed run must not retry")):
        result = run_suite(suite, backend_name="mock", resume=True, retry_failed=False)
    self.assertEqual(result.skipped, 1)
    self.assertEqual(json.loads((run_dir / "run_status.json").read_text())["status"], "FAILED")
