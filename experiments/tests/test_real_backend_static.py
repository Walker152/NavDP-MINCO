import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from experiments.core.models import RunSpec
from experiments.designers.manifest import load_manifest
from experiments.simulators.isaac_navdp_backend import IsaacNavDPBackend
from experiments.orchestrators.suite_runner import run_suite


class RealBackendStaticTests(unittest.TestCase):
    def test_backend_import_is_lazy(self):
        self.assertFalse(any(name.startswith("omni") for name in sys.modules))

    def test_variant_commands_share_episode_inputs_but_select_adapters(self):
        backend = IsaacNavDPBackend(repo_root=Path("."))
        scene = load_manifest("experiments/configs/real_pointgoal_scenarios.json").scenes[0]
        episodes = list(scene.episodes)
        commands = {}
        for variant, mode in (("raw","cold"),("minco-cold","cold"),("minco-hot","gated")):
            run = RunSpec("s","EXP-ALL",variant,mode,scene.scene_label,scene.scene_id,0,"run")
            commands[variant] = backend.build_command(run, Path("/tmp/run"), Path("experiments/configs/real_pointgoal_scenarios.json"), scene, episodes)
        for command in commands.values():
            self.assertIn("--headless", command); self.assertIn("--save-video", command); self.assertIn("--eval-monitor", command)
            self.assertIn(episodes[0].episode_uid, command); self.assertIn(str(episodes[0].navdp_seed), command)
        self.assertIn("raw", commands["raw"]); self.assertIn("cold", commands["minco-cold"]); self.assertIn("gated", commands["minco-hot"])

    def test_run_requires_explicit_real_simulation_permission(self):
        backend = IsaacNavDPBackend(repo_root=Path("."))
        with self.assertRaises(PermissionError): backend.run(None, [], None, allow_real_simulation=False)

    def test_isaac_dry_run_does_not_start_subprocess(self):
        with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("must not spawn")):
            result = run_suite("experiments/configs/static_real_suite.json", backend_name="isaac", resume=True, dry_run=True)
        self.assertEqual(result.failed, 0)
        plan = json.loads(Path("results/navdp_minco_static_real/dry_run_plan.json").read_text())
        self.assertEqual(plan["started_processes"], 0)
        self.assertEqual(len(plan["commands"]), 6)
        self.assertEqual(len(plan["server_commands"]), 6)
        self.assertTrue(all(command[:4] == ["conda", "run", "-n", "navdp"] for command in plan["server_commands"]))

    def test_skip_video_reaches_real_dry_run_commands(self):
        run_suite("experiments/configs/static_real_suite.json", backend_name="isaac", dry_run=True, skip_video=True)
        plan = json.loads(Path("results/navdp_minco_static_real/dry_run_plan.json").read_text())
        self.assertTrue(all("--no-save-video" in command and "--save-video" not in command for command in plan["commands"]))
