import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.analyzers.paired import compare_runs


class PairingTests(unittest.TestCase):
    def _run(self, root: Path, name: str, safe_dist: float, values):
        path = root / name; path.mkdir()
        (path / "run_config.json").write_text(json.dumps({"scene_id": "s", "seed": 0, "safe_dist": safe_dist, "manifest_id": "m"}))
        with (path / "episode_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["episode_uid", "success", "episode_duration_s", "actual_path_length_m", "repository_spl"]); writer.writeheader()
            for uid, duration in values: writer.writerow({"episode_uid": uid, "success": True, "episode_duration_s": duration, "actual_path_length_m": duration, "repository_spl": 1})
        return path

    def test_compare_pairs_by_episode_uid(self):
        root = Path(tempfile.mkdtemp()); baseline = self._run(root, "a", .6, [("x", 3), ("y", 4)]); method = self._run(root, "b", .6, [("x", 2), ("z", 1)])
        result = compare_runs(baseline, method, root / "report")
        self.assertEqual(result["paired_count"], 1)
        self.assertEqual(result["episode_duration_s_delta_mean"], -1.0)

    def test_compare_rejects_mismatched_safety_configuration(self):
        root = Path(tempfile.mkdtemp()); baseline = self._run(root, "a", .6, [("x", 3)]); method = self._run(root, "b", .7, [("x", 2)])
        with self.assertRaisesRegex(ValueError, "safe_dist"):
            compare_runs(baseline, method, root / "report")
