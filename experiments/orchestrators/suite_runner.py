from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from experiments.analyzers.run_analysis import analyze_run
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


def run_suite(config_path, backend_name="mock", resume=False):
    if backend_name != "mock": raise ValueError("only the safe mock backend is implemented; real simulation is never started implicitly")
    config = load_suite(config_path); manifest = load_manifest(config.manifest_path); layout = ResultLayout(config.output_root); backend = MockBackend()
    suite_dir = layout.suite_dir(config.suite_id); suite_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(suite_dir / "suite_config.json", {"suite_id": config.suite_id, "manifest": str(config.manifest_path), "backend": backend.name, "data_source": "SIMULATED"})
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
    return SuiteResult(completed, skipped, failed)
