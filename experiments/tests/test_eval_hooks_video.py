import json
from pathlib import Path
import tempfile
import unittest

import imageio.v2 as imageio
import numpy as np

from experiments.integration.eval_hooks import ExperimentHookBridge
from experiments.recorders.video_recorder import EpisodeVideoRecorder


class Sink:
    def __init__(self): self.rows = []
    def submit_csv(self, table, row): self.rows.append((table, row))


class EvalHookVideoTests(unittest.TestCase):
    def test_hook_sequence_and_stale_hold_stop_semantics(self):
        sink = Sink(); bridge = ExperimentHookBridge(sink, {"suite_id":"s", "experiment_id":"e", "run_id":"r", "variant":"minco-hot", "scene_label":"SPARSE", "scene_id":"x", "seed":0})
        bridge.start_episode("ep", generation=1)
        bridge.record_planning_cycle("c1", published=False, stale=True, fallback_mode="NONE")
        bridge.record_planning_cycle("c2", published=False, stale=False, fallback_mode="HOLD_LAST")
        bridge.record_planning_cycle("c3", published=False, stale=False, fallback_mode="STOP")
        bridge.record_plan("p-bad", published=False, stale=True)
        bridge.record_plan("p-ok", published=True, stale=False)
        bridge.record_control_step(0, "p-ok", .2, .0)
        bridge.end_episode(success=True)
        self.assertEqual([table for table, _ in sink.rows].count("planning_cycles"), 3)
        self.assertEqual([row["plan_uid"] for table, row in sink.rows if table == "plan_metrics"], ["p-ok"])
        bridge.reset(generation=2)
        self.assertIsNone(bridge.episode_uid)

    def test_synthetic_episode_video_is_readable_and_has_metadata(self):
        root = Path(tempfile.mkdtemp()); recorder = EpisodeVideoRecorder(root, fps=5, crf=28)
        recorder.start_episode("ep_stable")
        for index in range(4): recorder.write(np.full((32, 48, 3), index * 50, dtype=np.uint8))
        video = recorder.end_episode(); recorder.close()
        self.assertGreater(video.stat().st_size, 0)
        metadata = json.loads((root / "ep_stable.video_complete.json").read_text())
        self.assertEqual(metadata["frame_count"], 4); self.assertTrue(metadata["complete"])
        reader = imageio.get_reader(video); self.assertEqual(reader.get_data(0).shape[:2], (32, 48)); reader.close()

    def test_eval_exposes_static_experiment_cli_without_changing_defaults(self):
        source = Path("eval_pointgoal_wheeled.py").read_text()
        for flag in ("--experiment-config", "--experiment-run-dir", "--experiment-variant", "--scenario-manifest", "--episode-uids", "--headless", "--save-video", "--save-debug-visuals", "--video-fps", "--eval-monitor", "--save-planning-trace", "--warm-start-mode", "--seed", "--navdp-seed"):
            self.assertIn(flag, source)
        self.assertIn("AppLauncher(headless=args_cli.headless, enable_cameras=True)", source)
        self.assertIn("num_envs must be 1", source)
