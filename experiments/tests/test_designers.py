import json
import tempfile
import unittest

from experiments.designers.manifest import load_manifest
from experiments.designers.suite import load_suite


class DesignerTests(unittest.TestCase):
 def test_manifest_sorts_scenes_and_generates_stable_uid(self):
    tmp_path = __import__('pathlib').Path(tempfile.mkdtemp())
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"manifest_version": 1, "manifest_id": "m", "seed": 0, "scenes": [
        {"scene_id": "z", "scene_label": "DENSE", "scene_path": "z.usd", "episodes": [{"scenario_id": "SCN-01", "episode_index": 0, "seed": 1, "start_pose": [0,0,0], "goal_pose": [1,0,0]}]},
        {"scene_id": "a", "scene_label": "SPARSE", "scene_path": "a.usd", "episodes": [{"scenario_id": "SCN-01", "episode_index": 0, "seed": 1, "start_pose": [0,0,0], "goal_pose": [1,0,0]}]},
    ]}))
    manifest = load_manifest(path)
    self.assertEqual([scene.scene_id for scene in manifest.scenes], ["a", "z"])
    self.assertNotEqual(manifest.scenes[0].episodes[0].episode_uid, manifest.scenes[1].episodes[0].episode_uid)


 def test_suite_rejects_hot_variant_with_cold_mode(self):
    tmp_path = __import__('pathlib').Path(tempfile.mkdtemp())
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"suite_id": "s", "output_root": "results", "manifest": "m.json", "runs": [
        {"experiment_id": "EXP-04_warm_start", "variant": "minco-hot", "warm_start_mode": "cold"}
    ]}))
    with self.assertRaisesRegex(ValueError, "gated"):
        load_suite(path)
