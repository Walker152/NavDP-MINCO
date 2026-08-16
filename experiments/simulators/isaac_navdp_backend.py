from __future__ import annotations

from pathlib import Path
import json
import subprocess
import hashlib
import os
import math
import numpy as np

from experiments.simulators.process_supervisor import ProcessSupervisor
from experiments.core.effective_parameters import EFFECTIVE_PARAMETERS


class IsaacNavDPBackend:
    """Command adapter with a fail-closed real-simulation gate and no heavy imports."""
    name = "isaac"

    def __init__(self, repo_root: Path, supervisor_factory=ProcessSupervisor, isaaclab_dir=None, navdp_port=None):
        self.repo_root = Path(repo_root).resolve()
        self.evaluator_path = (
            self.repo_root / "run_scripts" / "eval_pointgoal_wheeled.py"
        ).resolve()
        self.supervisor_factory = supervisor_factory
        default_isaaclab_dir = self.repo_root.parent / "IsaacLab"
        self.isaaclab_dir = Path(
            isaaclab_dir or os.environ.get("ISAACLAB_DIR", default_isaaclab_dir)
        ).resolve()
        self.navdp_port = int(navdp_port or os.environ.get("NAVDP_PORT", "8889"))

    def validate_static_configuration(self, run, scene=None, episodes=None):
        errors = []
        if run.variant not in {"raw", "minco-cold", "minco-hot"}: errors.append("unsupported variant")
        expected = "gated" if run.variant == "minco-hot" else "cold"
        if run.warm_start_mode != expected: errors.append(f"{run.variant} requires {expected} warm start")
        if not self.evaluator_path.is_file():
            errors.append(f"missing evaluator: {self.evaluator_path}")
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

    def build_command(self, run, run_dir, manifest_path, scene, episodes, save_video=True, save_trace=True, effective=None):
        errors = self.validate_static_configuration(run, scene, episodes)
        if errors: raise ValueError("; ".join(errors))
        effective = effective or EFFECTIVE_PARAMETERS
        controller = effective["raw_mpc"] if run.variant == "raw" else effective["minco_mpc"]
        calibration = effective.get("robot_calibration", {})
        calibration_path = Path(str(calibration.get("path", "")))
        if not calibration_path.is_file():
            raise ValueError(
                f"effective robot calibration is missing: {calibration_path}"
            )
        episode_uids = [episode.episode_uid for episode in episodes]
        navdp_seeds = [episode.navdp_seed for episode in episodes]
        isaac_python = os.environ.get("ISAACLAB_PYTHON", "").strip()
        isaac_selector = (
            ["-p", str(Path(isaac_python).expanduser().resolve().parent.parent)]
            if isaac_python
            else ["-n", "isaaclab"]
        )
        conda_bin = os.environ.get("CONDA_BIN", "conda").strip() or "conda"
        command = [
            conda_bin, "run", "--no-capture-output", *isaac_selector,
            "bash", str(self.isaaclab_dir / "isaaclab.sh"), "-p", str(self.evaluator_path),
            "--experiment-config", str(Path(run_dir) / "run_config.json"),
            "--experiment-run-dir", str(run_dir), "--experiment-variant", run.variant, "--scenario-manifest", str(manifest_path),
            "--scene-path", str(scene.scene_path), "--scene-id", scene.scene_id,
            "--episode-uids", *list(episode_uids), "--headless", "--save-video" if save_video else "--no-save-video", "--save-debug-visuals", "--eval-monitor", "--save-planning-trace" if save_trace else "--no-save-planning-trace",
            "--warm-start-mode", run.warm_start_mode, "--seed", str(run.seed), "--navdp-seed", str(navdp_seeds[0]),
            "--navdp-seeds", *[str(value) for value in navdp_seeds], "--num_envs", "1", "--num_episodes", str(len(episodes)),
            "--speed", str(controller.get("desired_v_mps", controller.get("desired_v"))), "--mpc_max_yaw_rate", str(effective["minco"].get("max_yaw_rate_radps", controller.get("w_max_radps", controller.get("w_max")))),
            "--use_robot_base_frame", "0",
            "--robot-calibration", str(calibration_path),
            "--port", str(self.navdp_port),
        ]
        dynamic_case_path = Path(scene.scene_path) / "dynamic_case.json"
        if dynamic_case_path.is_file():
            dynamic_case = json.loads(dynamic_case_path.read_text(encoding="utf-8"))
            usd_path = sorted(Path(scene.scene_path).glob("*.usd"))[0]
            scene_hash = hashlib.sha256(usd_path.read_bytes()).hexdigest()
            if dynamic_case.get("scene_id") != scene.scene_id:
                raise ValueError("dynamic case/scene id mismatch")
            if dynamic_case.get("scene_sha256") != scene_hash:
                raise ValueError("dynamic case scene hash mismatch")
            if any(
                episode.scenario_id != dynamic_case.get("case_uid")
                for episode in episodes
            ):
                raise ValueError("dynamic case/episode pairing mismatch")
            if dynamic_case.get("calibration_sha256") != calibration.get("sha256"):
                raise ValueError("dynamic case calibration hash mismatch")
            command.extend([
                "--dynamic-case-spec", str(dynamic_case_path.resolve()),
                "--dynamic-case-hash", str(dynamic_case.get("case_hash", "")),
                "--dynamic-scene-sha256", scene_hash,
                "--dynamic-calibration-sha256", str(calibration.get("sha256", "")),
            ])
        command.extend(["--raw-controller", "original-navdp-mpc" if run.variant == "raw" else "disabled"])
        minco = effective["minco"]; esdf = effective["esdf"]; video = effective["video"]
        diagnostics = effective["runtime_diagnostics"]
        if not str(diagnostics.get("threshold_profile_id", "")).strip():
            raise ValueError("runtime_diagnostics.threshold_profile_id must be non-empty")
        for key in (
            "high_turn_curvature_p95_1pm",
            "high_turn_curvature_tv_1pm",
            "jump_position_rmse_m",
            "jump_tangent_rad",
            "planning_deadline_ms",
        ):
            value = float(diagnostics.get(key, float("nan")))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"runtime_diagnostics.{key} must be finite and positive")
        command.extend([
            "--minco_initial_top_k", str(minco["initial_top_k"]),
            "--minco_max_top_k", str(minco["max_top_k"]),
            "--minco_candidate_time_budget_ms", str(minco["candidate_time_budget_ms"]),
            "--minco_optimization_safe_dist", str(minco["optimization_safe_distance_m"]),
            "--minco_validation_safe_dist", str(minco["validation_safe_distance_m"]),
            "--minco_start_validation_exemption_radius", str(minco["start_validation_exemption_radius_m"]),
            "--minco_sample_dt", str(minco["sample_dt_s"]), "--minco_max_vel", str(minco["max_velocity_mps"]),
            "--minco_max_acc", str(minco["max_acceleration_mps2"]), "--minco_max_iterations", str(minco["max_iterations"]),
            "--minco_penalty_weight_pos", str(minco["penalty_weight_pos"]), "--minco_penalty_weight_vel", str(minco["penalty_weight_vel"]),
            "--minco_penalty_weight_acc", str(minco["penalty_weight_acc"]), "--minco_penalty_weight_attractor", str(minco["penalty_weight_attractor"]),
            "--minco_time_weight", str(minco["time_weight"]), "--minco_time_barrier_weight", str(minco["time_barrier_weight"]),
            "--minco_constraint_profile", str(minco["constraint_profile"]),
            "--minco_guide_corridor_weight", str(minco["guide_corridor_weight"]),
            "--minco_corridor_max_radius", str(minco["corridor_max_radius_m"]),
            "--minco_corridor_min_radius", str(minco["corridor_min_radius_m"]),
            "--minco_corridor_sample_step", str(minco["corridor_sample_step_m"]),
            "--minco_sfc_bound_distance", str(minco["sfc_bound_distance_m"]),
            "--minco_sfc_seed_line_max_length", str(minco["sfc_seed_line_max_length_m"]),
            "--minco_sfc_min_overlap_depth", str(minco["sfc_min_overlap_depth_m"]),
            "--minco_adaptive_max_spatial_step", str(minco["adaptive_max_spatial_step_m"]),
            "--minco_adaptive_near_clearance", str(minco["adaptive_near_clearance_m"]),
            "--minco_adaptive_max_depth", str(minco["adaptive_max_depth"]),
            "--minco_adaptive_sample_budget", str(minco["adaptive_sample_budget"]),
            "--minco_max_jerk", str(minco["max_jerk_mps3"]),
            "--esdf_resolution", str(esdf["resolution_m"]), "--esdf_padding", str(esdf["padding_m"]),
            "--esdf_cache_name", str(esdf["cache_name"]), "--esdf_obstacle_min_height", str(esdf["obstacle_min_height_m"]),
            "--esdf_obstacle_max_height", str(esdf["obstacle_max_height_m"]), "--esdf_fill_footprint", str(int(esdf["fill_footprint"])),
            "--esdf_footprint_inflate_cells", str(esdf["footprint_inflate_cells"]),
            "--mpc_max_yaw_acc", str(effective["minco_mpc"]["max_yaw_acceleration_radps2"]),
            "--mpc_max_wheel_speed", str(effective["minco_mpc"]["max_wheel_speed_radps"] or 0.0), "--scene_scale", str(effective["scene"]["scale"]),
            "--video-fps", str(video["fps"]), "--video-crf", str(video["crf"]), "--video-scale", str(video["scale"]),
            "--threshold-profile-id", str(diagnostics["threshold_profile_id"]),
            "--high-turn-curvature-p95", str(diagnostics["high_turn_curvature_p95_1pm"]),
            "--high-turn-curvature-tv", str(diagnostics["high_turn_curvature_tv_1pm"]),
            "--jump-position-rmse", str(diagnostics["jump_position_rmse_m"]),
            "--jump-tangent-rad", str(diagnostics["jump_tangent_rad"]),
            "--planning-deadline-ms", str(diagnostics["planning_deadline_ms"]),
        ])
        if esdf["force_rebuild"]: command.append("--esdf_force_rebuild")
        command.append("--no-enable_minco" if run.variant == "raw" else "--enable_minco")
        return command

    def build_server_command(self, run_dir, run_id, seed, port=None):
        port = self.navdp_port if port is None else int(port)
        navdp_python = os.environ.get("NAVDP_PYTHON", "").strip()
        navdp_selector = (
            ["-p", str(Path(navdp_python).expanduser().resolve().parent.parent)]
            if navdp_python
            else ["-n", "navdp"]
        )
        conda_bin = os.environ.get("CONDA_BIN", "conda").strip() or "conda"
        return [
            conda_bin, "run", "--no-capture-output", *navdp_selector, "python", "-u",
            str(self.repo_root / "baselines/navdp/navdp_server.py"),
            "--port", str(port), "--checkpoint", str(self.repo_root / "baselines/navdp/checkpoints/navdp_checkpoint.ckpt"),
            "--output-dir", str(run_dir), "--run-id", str(run_id), "--seed", str(seed), "--no-save-video",
        ]

    def run(self, run, episodes, writer, allow_real_simulation=False, command=None, progress_callback=None):
        if not allow_real_simulation: raise PermissionError("real simulation requires --allow-real-simulation")
        if command is None: raise ValueError("an explicit validated command is required")
        launcher = self.isaaclab_dir / "isaaclab.sh"
        if not launcher.is_file():
            raise FileNotFoundError(f"IsaacLab launcher not found: {launcher}")
        run_dir = Path(command[command.index("--experiment-run-dir") + 1])
        run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        timeout_s = float(run_config["timeout_s"])
        server_command = self.build_server_command(run_dir, run.run_id, run.seed)
        return self.supervisor_factory().run_pair(
            server_command, command, run_dir, self.repo_root,
            port=self.navdp_port, timeout_s=timeout_s, progress_callback=progress_callback,
        )
