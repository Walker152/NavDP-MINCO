from __future__ import annotations

from pathlib import Path
import subprocess
import hashlib
import numpy as np

from experiments.simulators.process_supervisor import ProcessSupervisor


class IsaacNavDPBackend:
    """Command adapter with a fail-closed real-simulation gate and no heavy imports."""
    name = "isaac"

    def __init__(self, repo_root: Path, supervisor_factory=ProcessSupervisor):
        self.repo_root = Path(repo_root).resolve()
        self.supervisor_factory = supervisor_factory

    def validate_static_configuration(self, run, scene=None, episodes=None):
        errors = []
        if run.variant not in {"raw", "minco-cold", "minco-hot"}: errors.append("unsupported variant")
        expected = "gated" if run.variant == "minco-hot" else "cold"
        if run.warm_start_mode != expected: errors.append(f"{run.variant} requires {expected} warm start")
        if not (self.repo_root / "eval_pointgoal_wheeled.py").exists(): errors.append("missing eval_pointgoal_wheeled.py")
        if scene is not None:
            if str(scene.scene_path).startswith("mock://"):
                errors.append("isaac backend rejects mock scene paths")
            scene_path = Path(scene.scene_path)
            scene_path = scene_path if scene_path.is_absolute() else self.repo_root / scene_path
            if not scene_path.is_dir(): errors.append(f"missing scene directory: {scene_path}")
            elif not list(scene_path.glob("*.usd")): errors.append(f"scene has no USD: {scene_path}")
            else:
                usd_path = sorted(scene_path.glob("*.usd"))[0]
                digest = hashlib.sha256(usd_path.read_bytes()).hexdigest()
                if not scene.asset_hash: errors.append("scene asset_hash is required")
                elif digest != scene.asset_hash: errors.append("scene asset_hash mismatch")
            if run.scene_id != scene.scene_id: errors.append("run/scene id mismatch")
        if episodes is not None:
            if not episodes: errors.append("run has no episodes")
            for episode in episodes:
                if episode.scene_id != run.scene_id: errors.append("episode/run scene mismatch")
                if episode.seed != run.seed: errors.append("episode/run seed mismatch")
            if scene is not None and not str(scene.scene_path).startswith("mock://"):
                init_path = scene_path / "pointgoal_start_goal_pairs.npy"
                if not init_path.exists(): errors.append(f"missing pointgoal init: {init_path}")
                else:
                    source_rows = np.load(init_path, allow_pickle=False)
                    for episode in episodes:
                        index = episode.source_episode_index
                        if index is None or not 0 <= index < len(source_rows):
                            errors.append(f"invalid source_episode_index: {index}")
                            continue
                        expected = np.asarray([
                            episode.start_pose[0], episode.start_pose[1],
                            episode.goal_pose[0], episode.goal_pose[1], episode.start_pose[2],
                        ])
                        if not np.allclose(source_rows[index], expected, rtol=0.0, atol=1e-12):
                            errors.append(f"episode source row mismatch: {episode.episode_uid}")
        return errors

    def build_command(self, run, run_dir, manifest_path, scene, episodes, save_video=True, save_trace=True):
        errors = self.validate_static_configuration(run, scene, episodes)
        if errors: raise ValueError("; ".join(errors))
        episode_uids = [episode.episode_uid for episode in episodes]
        navdp_seeds = [episode.navdp_seed for episode in episodes]
        command = [
            "conda", "run", "-n", "isaaclab", "python", str(self.repo_root / "eval_pointgoal_wheeled.py"), "--experiment-config", str(Path(run_dir) / "run_config.json"),
            "--experiment-run-dir", str(run_dir), "--experiment-variant", run.variant, "--scenario-manifest", str(manifest_path),
            "--scene-path", str(scene.scene_path), "--scene-id", scene.scene_id,
            "--episode-uids", *list(episode_uids), "--headless", "--save-video" if save_video else "--no-save-video", "--save-debug-visuals", "--eval-monitor", "--save-planning-trace" if save_trace else "--no-save-planning-trace",
            "--warm-start-mode", run.warm_start_mode, "--seed", str(run.seed), "--navdp-seed", str(navdp_seeds[0]),
            "--navdp-seeds", *[str(value) for value in navdp_seeds], "--num_envs", "1", "--num_episodes", str(len(episodes)),
            "--speed", "0.5", "--mpc_max_yaw_rate", "0.5",
            "--use_robot_base_frame", "0",
        ]
        command.extend(["--raw-controller", "original-navdp-mpc" if run.variant == "raw" else "disabled"])
        command.append("--no-enable_minco" if run.variant == "raw" else "--enable_minco")
        return command

    def build_server_command(self, run_dir, run_id, seed, port=8888):
        return [
            "conda", "run", "-n", "navdp", "python", str(self.repo_root / "baselines/navdp/navdp_server.py"),
            "--port", str(port), "--checkpoint", str(self.repo_root / "baselines/navdp/checkpoints/navdp_checkpoint.ckpt"),
            "--output-dir", str(run_dir), "--run-id", str(run_id), "--seed", str(seed), "--no-save-video",
        ]

    def run(self, run, episodes, writer, allow_real_simulation=False, command=None):
        if not allow_real_simulation: raise PermissionError("real simulation requires --allow-real-simulation")
        if command is None: raise ValueError("an explicit validated command is required")
        run_dir = Path(command[command.index("--experiment-run-dir") + 1])
        server_command = self.build_server_command(run_dir, run.run_id, run.seed)
        return self.supervisor_factory().run_pair(server_command, command, run_dir, self.repo_root)
