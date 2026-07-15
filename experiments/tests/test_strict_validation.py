import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.analyzers.validator import validate_run
from experiments.orchestrators.suite_runner import run_suite


class StrictValidationTests(unittest.TestCase):
    def _mock_run(self):
        root = Path(tempfile.mkdtemp())
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"manifest_version":1,"manifest_id":"v","scenes":[{
            "scene_id":"s","scene_label":"SPARSE","scene_path":"mock://s","episodes":[
                {"scenario_id":"SCN-01","episode_index":0,"seed":0,"start_pose":[0,0,0],"goal_pose":[1,0,0]}
            ]}]}))
        suite = root / "suite.json"
        suite.write_text(json.dumps({"suite_id":"v","output_root":str(root),"manifest":str(manifest),"runs":[
            {"experiment_id":"EXP","variant":"raw","warm_start_mode":"cold"}
        ]}))
        run_suite(suite, backend_name="mock", resume=True)
        return next((root / "v" / "experiments").glob("*/*/*/*/*"))

    def test_validator_rejects_duplicate_primary_key(self):
        run_dir = self._mock_run(); path = run_dir / "episode_metrics.csv"
        with path.open() as stream: rows = list(csv.DictReader(stream)); fields = rows[0].keys()
        with path.open("a", newline="") as stream: csv.DictWriter(stream, fieldnames=fields).writerow(rows[0])
        errors = validate_run(run_dir, write_report=False)["errors"]
        self.assertTrue(any("duplicate primary key: episode_metrics" in error for error in errors), errors)

    def test_validator_rejects_raw_cycle_that_attempts_minco(self):
        run_dir = self._mock_run(); path = run_dir / "planning_cycles.csv"
        with path.open() as stream: rows = list(csv.DictReader(stream)); fields = rows[0].keys()
        rows[0]["attempted_candidate_count"] = "2"; rows[0]["minco_ms"] = "5"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        errors = validate_run(run_dir, write_report=False)["errors"]
        self.assertIn("raw variant attempted MINCO candidates", errors)
        self.assertIn("raw variant recorded nonzero minco_ms", errors)

    def test_validator_requires_expected_episode_video_and_trace(self):
        run_dir = self._mock_run(); config_path = run_dir / "run_config.json"
        config = json.loads(config_path.read_text()); config.update({"video_required":True,"trace_required":True})
        config_path.write_text(json.dumps(config))
        errors = validate_run(run_dir, write_report=False)["errors"]
        self.assertTrue(any("missing complete video" in error for error in errors), errors)


if __name__ == "__main__": unittest.main()
