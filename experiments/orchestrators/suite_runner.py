from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from experiments.analyzers.run_analysis import analyze_run
from experiments.analyzers.artifact_manifest import generate_artifact_manifest
from experiments.analyzers.suite_analysis import analyze_suite
from experiments.analyzers.validator import validate_run
from experiments.core.layout import ResultLayout
from experiments.core.models import SuiteResult
from experiments.core.schemas import SCHEMAS
from experiments.designers.manifest import load_manifest
from experiments.designers.suite import expand_runs, load_suite
from experiments.recorders.async_writer import AsyncRecordWriter
from experiments.recorders.run_recorder import RunLifecycle, atomic_json
from experiments.simulators.mock_backend import MockBackend
from experiments.simulators.isaac_navdp_backend import IsaacNavDPBackend


def run_suite(config_path, backend_name="mock", resume=False, retry_failed=False, dry_run=False, analysis_only=False, allow_real_simulation=False):
    if backend_name not in {"mock", "isaac"}: raise ValueError("backend must be mock or isaac")
    config = load_suite(config_path); manifest = load_manifest(config.manifest_path); layout = ResultLayout(config.output_root); backend = MockBackend() if backend_name == "mock" else IsaacNavDPBackend(Path("."))
    suite_dir = layout.suite_dir(config.suite_id); suite_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(suite_dir / "suite_config.json", {"suite_id": config.suite_id, "manifest": str(config.manifest_path), "backend": backend.name, "data_source": "SIMULATED"})
    atomic_json(suite_dir / "scenario_manifest.json", json.loads(config.manifest_path.read_text(encoding="utf-8")))
    atomic_json(suite_dir / "suite_status.json", {"status":"RUNNING", "backend":backend.name, "data_source":"SIMULATED"})
    if dry_run:
        commands = []
        scenes = {scene.scene_id:scene for scene in manifest.scenes}
        for run in expand_runs(config, manifest):
            episodes = [episode for episode in scenes[run.scene_id].episodes if episode.seed == run.seed]
            if backend_name == "isaac": commands.append(backend.build_command(run, layout.run_dir(run), config.manifest_path, [episode.episode_uid for episode in episodes], run.seed, episodes[0].navdp_seed if episodes else run.seed))
        atomic_json(suite_dir / "dry_run_plan.json", {"backend":backend_name, "run_count":len(list(expand_runs(config, manifest))), "commands":commands, "started_processes":0})
        return SuiteResult(0, 0, 0)
    if analysis_only:
        analyze_suite(suite_dir); return SuiteResult(0, 0, 0)
    if backend_name == "isaac" and not allow_real_simulation: raise PermissionError("isaac backend requires --dry-run or --allow-real-simulation")
    completed = skipped = failed = 0
    scenes = {scene.scene_id: scene for scene in manifest.scenes}
    for run in expand_runs(config, manifest):
        run_dir = layout.run_dir(run); status_path = run_dir / "run_status.json"
        if resume and status_path.exists() and json.loads(status_path.read_text()).get("status") == "COMPLETE" and validate_run(run_dir, write_report=False)["valid"]:
            skipped += 1; continue
        lifecycle = RunLifecycle(run_dir)
        if lifecycle.status in {"FAILED", "COMPLETE"}: # explicit rerun gets a fresh lifecycle record
            lifecycle.status = "CREATED"; lifecycle._write(restarted=True)
        atomic_json(run_dir / "run_config.json", {**asdict(run), "backend": backend.name, "data_source": "SIMULATED", "manifest_id": manifest.manifest_id})
        try:
            lifecycle.transition("RUNNING"); writer = AsyncRecordWriter(run_dir, SCHEMAS)
            episodes = [episode for episode in scenes[run.scene_id].episodes if episode.seed == run.seed]
            backend.run(run, episodes, writer); writer.close(); lifecycle.transition("SIMULATION_COMPLETE"); lifecycle.transition("VALIDATING")
            validation = validate_run(run_dir)
            if not validation["valid"]: raise RuntimeError("run validation failed: " + "; ".join(validation["errors"]))
            analyze_run(run_dir); lifecycle.transition("COMPLETE", validation="PASS"); completed += 1
        except BaseException as error:
            failed += 1
            if lifecycle.status not in {"FAILED", "COMPLETE"}: lifecycle.status = "FAILED"; lifecycle._write(error=repr(error))
            raise
    analyze_suite(suite_dir)
    atomic_json(suite_dir / "suite_status.json", {"status":"COMPLETE", "backend":backend.name, "data_source":"SIMULATED", "completed":completed, "skipped":skipped, "failed":failed})
    generate_artifact_manifest(suite_dir)
    return SuiteResult(completed, skipped, failed)
