import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from experiments.baselines.raw_navdp.controller_factory import create_tracking_controller, update_tracking_reference
from experiments.baselines.raw_navdp.original_mpc import ORIGINAL_MPC_SPEC, RawNavDPMPCController
from experiments.designers.manifest import load_manifest
from experiments.designers.pointgoal_import import build_default_real_manifest
from experiments.integration.episode_selection import materialize_episode_init
from experiments.simulators.isaac_navdp_backend import IsaacNavDPBackend
from experiments.simulators.process_supervisor import ProcessSupervisor


class _FakeRaw:
    def __init__(self, path, **kwargs): self.path = path; self.kwargs = kwargs


class _FakeMinco:
    def __init__(self, path, **kwargs): self.path = path; self.kwargs = kwargs


class PairedRealExperimentTests(unittest.TestCase):
    def test_raw_factory_selects_original_controller_without_minco_samples(self):
        path = np.array([[0.0, 0.0], [1.0, 0.0]])
        controller = create_tracking_controller(
            "raw", path, trajectory_samples=np.ones((2, 15)), raw_controller_cls=_FakeRaw,
            minco_controller_cls=_FakeMinco, desired_v=0.5, v_max=0.5, w_max=0.5, T=0.1,
        )
        self.assertIsInstance(controller, _FakeRaw)
        self.assertNotIn("trajectory_samples", controller.kwargs)
        self.assertEqual(controller.kwargs, {
            "N":15, "desired_v":0.5, "v_max":0.5, "w_max":0.5, "ref_gap":3, "T":0.1,
        })

    def test_raw_reference_update_cannot_change_original_speed(self):
        class Fake:
            def update_reference(self, path, desired_v=None): self.desired_v = desired_v; return True
        controller = Fake()
        self.assertTrue(update_tracking_reference(controller, "raw", np.ones((2, 2)), desired_v=9.0))
        self.assertEqual(controller.desired_v, 0.5)

    def test_raw_reference_update_resets_solver_warm_start_like_source_reconstruction(self):
        controller = RawNavDPMPCController.__new__(RawNavDPMPCController)
        controller.desired_v = 0.5; controller.progress_idx = 9
        controller.last_opt_x_states = np.ones((2, 3)); controller.last_opt_u_controls = np.ones((2, 2)); controller._current_reference = np.ones(5)
        controller.update_reference(np.array([[0.0, 0.0], [1.0, 0.0]]), desired_v=0.5)
        self.assertIsNone(controller.last_opt_x_states)
        self.assertIsNone(controller.last_opt_u_controls)
        self.assertEqual(controller.progress_idx, 0)

    def test_minco_factory_requires_temporal_samples_and_uses_current_controller(self):
        path = np.array([[0.0, 0.0], [1.0, 0.0]])
        samples = np.ones((2, 15))
        controller = create_tracking_controller(
            "minco-cold", path, trajectory_samples=samples, raw_controller_cls=_FakeRaw,
            minco_controller_cls=_FakeMinco, desired_v=0.5, v_max=0.5, w_max=0.5, T=0.1,
        )
        self.assertIsInstance(controller, _FakeMinco)
        self.assertIs(controller.kwargs["trajectory_samples"], samples)
        with self.assertRaises(ValueError):
            create_tracking_controller("minco-hot", path, trajectory_samples=None, raw_controller_cls=_FakeRaw, minco_controller_cls=_FakeMinco)

    def test_original_mpc_spec_matches_read_only_source(self):
        self.assertEqual(ORIGINAL_MPC_SPEC["N"], 15)
        self.assertEqual(ORIGINAL_MPC_SPEC["T"], 0.1)
        self.assertEqual(ORIGINAL_MPC_SPEC["Q"], (10.0, 10.0, 0.0))
        self.assertEqual(ORIGINAL_MPC_SPEC["R"], (0.02, 0.15))
        self.assertEqual(ORIGINAL_MPC_SPEC["ipopt.max_iter"], 100)
        source = Path("experiments/baselines/raw_navdp/original_mpc.py").read_text()
        self.assertIn("class RawNavDPMPCController", source)
        self.assertNotIn("from utils_tasks.tracking_utils import MPC_Controller", source)

    def test_original_reference_selection_is_numerically_identical(self):
        with mock.patch.dict("sys.modules", {"casadi": types.SimpleNamespace()}):
            from navdp_raw.utils_tasks.tracking_utils import MPC_Controller as SourceController
        source = SourceController.__new__(SourceController)
        adapted = RawNavDPMPCController.__new__(RawNavDPMPCController)
        for controller in (source, adapted):
            controller.N = 15; controller.desired_v = 0.5; controller.ref_gap = 3; controller.T = 0.1
            controller.ref_traj_len = 6
        path = np.column_stack((np.linspace(0.0, 3.0, 151), np.sin(np.linspace(0.0, 1.0, 151))))
        source_dense = source.make_ref_denser(path)
        adapted_dense = adapted.make_ref_denser(path)
        np.testing.assert_allclose(adapted_dense, source_dense, rtol=0.0, atol=0.0)
        state = np.array([0.37, 0.1, 0.2])
        np.testing.assert_allclose(
            adapted.find_reference_traj(state, adapted_dense),
            source.find_reference_traj(state, source_dense), rtol=0.0, atol=0.0,
        )

    def test_episode_selection_materializes_exact_manifest_rows(self):
        manifest = load_manifest("experiments/configs/real_pointgoal_scenarios.json")
        scene = manifest.scenes[0]
        output = Path(tempfile.mkdtemp()) / "selected.npy"
        selected = materialize_episode_init(
            "experiments/configs/real_pointgoal_scenarios.json", scene.scene_id,
            [episode.episode_uid for episode in scene.episodes], output,
        )
        expected = np.array([
            [ep.start_pose[0], ep.start_pose[1], ep.goal_pose[0], ep.goal_pose[1], ep.start_pose[2]]
            for ep in scene.episodes
        ])
        np.testing.assert_allclose(selected, expected)
        np.testing.assert_allclose(np.load(output), expected)

    def test_real_manifest_assets_and_hashes_are_valid(self):
        raw = json.loads(Path("experiments/configs/real_pointgoal_scenarios.json").read_text())
        manifest = load_manifest("experiments/configs/real_pointgoal_scenarios.json")
        self.assertEqual({scene.scene_label for scene in manifest.scenes}, {"SPARSE", "DENSE"})
        for item, scene in zip(sorted(raw["scenes"], key=lambda value: value["scene_id"]), manifest.scenes):
            path = Path(item["scene_path"])
            self.assertTrue(path.is_dir())
            usd = next(path.glob("*.usd"))
            digest = hashlib.sha256(usd.read_bytes()).hexdigest()
            self.assertEqual(item["asset_hash"], digest)
            self.assertEqual(scene.asset_hash, digest)
            source_rows = np.load(path / "pointgoal_start_goal_pairs.npy")
            for episode in item["episodes"]:
                row = source_rows[episode["source_episode_index"]]
                np.testing.assert_allclose(
                    row, [*episode["start_pose"][:2], *episode["goal_pose"][:2], episode["start_pose"][2]]
                )
            self.assertEqual([episode.source_episode_index for episode in scene.episodes], [0, 1])

    def test_real_manifest_is_reproducible_from_original_pointgoal_init(self):
        generated = build_default_real_manifest(Path("."))
        committed = json.loads(Path("experiments/configs/real_pointgoal_scenarios.json").read_text())
        self.assertEqual(generated, committed)

    def test_backend_rejects_mock_scene_and_builds_paired_real_command(self):
        manifest = load_manifest("experiments/configs/real_pointgoal_scenarios.json")
        scene = manifest.scenes[0]
        episodes = list(scene.episodes)
        backend = IsaacNavDPBackend(Path("."))
        from experiments.core.models import RunSpec, SceneSpec
        run = RunSpec("suite", "EXP-ALL", "raw", "cold", scene.scene_label, scene.scene_id, episodes[0].seed, "run_x")
        command = backend.build_command(run, Path("/tmp/run"), Path("experiments/configs/real_pointgoal_scenarios.json"), scene, episodes)
        self.assertIn("--raw-controller", command)
        self.assertEqual(command[command.index("--raw-controller") + 1], "original-navdp-mpc")
        self.assertEqual(command[command.index("--scene-path") + 1], scene.scene_path)
        self.assertEqual(command[command.index("--scene-id") + 1], scene.scene_id)
        self.assertEqual(command[command.index("--use_robot_base_frame") + 1], "0")
        mock_scene = SceneSpec("bad", "SPARSE", "mock://bad", tuple(episodes))
        with self.assertRaises(ValueError):
            backend.build_command(run, Path("/tmp/run"), Path("manifest.json"), mock_scene, episodes)

    def test_backend_builds_separate_conda_server_and_eval_commands(self):
        backend = IsaacNavDPBackend(Path("."))
        server = backend.build_server_command(Path("/tmp/run"), "run_x", seed=17, port=8888)
        self.assertEqual(server[:4], ["conda", "run", "-n", "navdp"])
        self.assertIn("baselines/navdp/checkpoints/navdp_checkpoint.ckpt", " ".join(server))
        self.assertIn("--no-save-video", server)
        manifest = load_manifest("experiments/configs/real_pointgoal_scenarios.json")
        scene = manifest.scenes[0]
        run = __import__("experiments.core.models", fromlist=["RunSpec"]).RunSpec(
            "suite", "EXP-ALL", "raw", "cold", scene.scene_label, scene.scene_id, 0, "run_x"
        )
        evaluation = backend.build_command(run, Path("/tmp/run"), Path("experiments/configs/real_pointgoal_scenarios.json"), scene, list(scene.episodes))
        self.assertEqual(evaluation[:4], ["conda", "run", "-n", "isaaclab"])

    def test_process_supervisor_escalates_cleanup_without_global_pkill(self):
        source = Path("experiments/simulators/process_supervisor.py").read_text()
        self.assertIn("os.killpg", source)
        self.assertIn("signal.SIGINT", source)
        self.assertIn("signal.SIGTERM", source)
        self.assertIn("signal.SIGKILL", source)
        self.assertNotIn("pkill", source)

    def test_eval_control_loop_uses_factory_and_selected_episode_file(self):
        source = Path("eval_pointgoal_wheeled.py").read_text()
        self.assertIn("create_tracking_controller(", source)
        self.assertIn("update_tracking_reference(", source)
        self.assertIn("materialize_episode_init(", source)
        self.assertIn("init_path = selected_episode_init_path", source)
        self.assertIn("args_cli.experiment_variant", source)
        self.assertNotIn("allow_geometric_fallback=not args_cli.enable_minco", source)
        self.assertIn("camera_top1_to_world(", source)

    def test_navdp_client_forwards_seed_and_eval_reseeds_each_episode(self):
        client = Path("utils_tasks/client_utils.py").read_text()
        evaluation = Path("eval_pointgoal_wheeled.py").read_text()
        self.assertIn("def navigator_reset(intrinsic=None,stop_threshold=-0.5,batch_size=1,port=8888,env_id=None,seed=None", client)
        self.assertIn("'seed':seed", client)
        self.assertIn("args_cli.navdp_seeds[episode_num + 1]", evaluation)

    def test_eval_monitor_identity_comes_from_run_config(self):
        source = Path("eval_pointgoal_wheeled.py").read_text()
        self.assertIn("monitor_identity.update", source)
        self.assertIn("json.load(config_stream)", source)
        self.assertNotIn('"scene_label": "UNKNOWN"', source)

    def test_dry_run_is_labeled_dry_run_not_simulated_performance(self):
        from experiments.orchestrators.suite_runner import run_suite
        run_suite("experiments/configs/static_real_suite.json", backend_name="isaac", dry_run=True)
        suite = Path("results/navdp_minco_static_real")
        self.assertEqual(json.loads((suite / "suite_config.json").read_text())["data_source"], "DRY_RUN")
        self.assertEqual(json.loads((suite / "suite_status.json").read_text())["data_source"], "DRY_RUN")
        plan = json.loads((suite / "dry_run_plan.json").read_text())
        for command in plan["commands"]:
            config_path = Path(command[command.index("--experiment-config") + 1])
            self.assertTrue(config_path.exists())
            config = json.loads(config_path.read_text())
            self.assertEqual(config["data_source"], "DRY_RUN")
            self.assertEqual(config["speed_mps"], 0.5)
            self.assertEqual(config["raw_controller"], "original-navdp-mpc" if config["variant"] == "raw" else "disabled")
            self.assertTrue(config["video_required"] and config["trace_required"])


if __name__ == "__main__":
    unittest.main()
