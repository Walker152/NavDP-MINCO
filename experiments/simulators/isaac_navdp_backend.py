from __future__ import annotations

from pathlib import Path
import subprocess


class IsaacNavDPBackend:
    """Command adapter with a fail-closed real-simulation gate and no heavy imports."""
    name = "isaac"

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()

    def validate_static_configuration(self, run):
        errors = []
        if run.variant not in {"raw", "minco-cold", "minco-hot"}: errors.append("unsupported variant")
        expected = "gated" if run.variant == "minco-hot" else "cold"
        if run.warm_start_mode != expected: errors.append(f"{run.variant} requires {expected} warm start")
        if not (self.repo_root / "eval_pointgoal_wheeled.py").exists(): errors.append("missing eval_pointgoal_wheeled.py")
        return errors

    def build_command(self, run, run_dir, manifest_path, episode_uids, seed, navdp_seed):
        if self.validate_static_configuration(run): raise ValueError("; ".join(self.validate_static_configuration(run)))
        command = [
            "python", str(self.repo_root / "eval_pointgoal_wheeled.py"), "--experiment-config", str(Path(run_dir) / "run_config.json"),
            "--experiment-run-dir", str(run_dir), "--experiment-variant", run.variant, "--scenario-manifest", str(manifest_path),
            "--episode-uids", *list(episode_uids), "--headless", "--save-video", "--save-debug-visuals", "--eval-monitor", "--save-planning-trace",
            "--warm-start-mode", run.warm_start_mode, "--seed", str(seed), "--navdp-seed", str(navdp_seed), "--num_envs", "1",
        ]
        command.append("--no-enable_minco" if run.variant == "raw" else "--enable_minco")
        return command

    def run(self, run, episodes, writer, allow_real_simulation=False, command=None):
        if not allow_real_simulation: raise PermissionError("real simulation requires --allow-real-simulation")
        if command is None: raise ValueError("an explicit validated command is required")
        return subprocess.Popen(command, cwd=self.repo_root, start_new_session=True)
