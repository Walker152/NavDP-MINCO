import json
from pathlib import Path
import tempfile
import unittest

import imageio.v2 as imageio
import numpy as np
import math

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
        bridge.record_plan("p-ok", published=True, stale=False, planning_state="HOT_START", hot_start_accepted=True, hot_reject_reason="", optimizer_iteration_count=7)
        bridge.record_control_step(0, "p-ok", .2, .0)
        bridge.end_episode(success=True)
        self.assertEqual([table for table, _ in sink.rows].count("planning_cycles"), 3)
        self.assertEqual([row["plan_uid"] for table, row in sink.rows if table == "plan_metrics"], ["p-ok"])
        plan_row = next(row for table, row in sink.rows if table == "plan_metrics")
        self.assertEqual(plan_row["planning_state"], "HOT_START")
        self.assertTrue(plan_row["hot_start_accepted"])
        self.assertEqual(plan_row["optimizer_iteration_count"], 7)
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

    def test_eval_integrates_uid_video_trace_candidates_and_stale_cycles(self):
        source = Path("eval_pointgoal_wheeled.py").read_text()
        for token in (
            "EpisodeVideoRecorder(", "episode_video_recorder.start_episode(",
            "episode_video_recorder.write(", "episode_video_recorder.end_episode(",
            "PlanningTraceWriter(", "trace_writer.write(",
            "experiment_hook.record_candidates(", "failure_reason=\"STALE_RESULT\"",
            "failure_reason=\"PLANNING_EXCEPTION\"",
        ):
            self.assertIn(token, source)

    def test_candidate_and_timing_hooks_emit_unified_schema_rows(self):
        sink = Sink(); bridge = ExperimentHookBridge(sink, {"suite_id":"s", "experiment_id":"e", "run_id":"r", "variant":"raw", "scene_label":"SPARSE", "scene_id":"x", "seed":0})
        bridge.start_episode("ep", 0)
        bridge.record_candidates("p", [0.8, 0.2], selected_index=0)
        bridge.record_timing("planning", 12.5, plan_uid="p", status="OK")
        candidate_rows = [row for table, row in sink.rows if table == "candidate_metrics"]
        timing_rows = [row for table, row in sink.rows if table == "timing_samples"]
        self.assertEqual(len(candidate_rows), 2)
        self.assertTrue(candidate_rows[0]["selected"])
        self.assertEqual(timing_rows[0]["duration_ms"], 12.5)

    def test_episode_summary_aggregates_cycles_tracking_and_navigation(self):
        sink = Sink(); bridge = ExperimentHookBridge(sink, {"suite_id":"s", "experiment_id":"e", "run_id":"r", "variant":"minco-hot", "scene_label":"DENSE", "scene_id":"x", "seed":0})
        bridge.start_episode("ep", 0)
        bridge.record_planning_cycle("c1", published=True, stale=False, fallback_mode="NONE", optimizer_success=True, python_validation_success=True)
        bridge.record_planning_cycle("c2", published=False, stale=False, fallback_mode="HOLD_LAST", optimizer_success=False, python_validation_success=False)
        bridge.record_control_step(0, "p", 0.5, 0.1, time_aligned_position_error_m=0.3, mpc_success=True)
        bridge.end_episode(success=True, actual_path_length_m=5.0, repository_spl=0.8, episode_duration_s=12.0)
        row = next(row for table, row in sink.rows if table == "episode_metrics")
        self.assertEqual(row["planning_count"], 2)
        self.assertEqual(row["hold_count"], 1)
        self.assertEqual(row["minco_ok_count"], 1)
        self.assertAlmostEqual(row["tracking_error_rmse_m"], 0.3)
        self.assertEqual(row["repository_spl"], 0.8)
