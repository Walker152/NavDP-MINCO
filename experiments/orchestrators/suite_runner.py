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
from experiments.recorders.run_manifest import write_run_manifest
from experiments.core.effective_parameters import effective_parameters
from experiments.simulators.mock_backend import MockBackend
from experiments.simulators.isaac_navdp_backend import IsaacNavDPBackend
from experiments.orchestrators.progress import SuiteProgressReporter


def _merge_parameter_overrides(suite_overrides, run_overrides):
    """Merge per-run parameter overrides on top of suite-level overrides."""
    if not run_overrides:
        return suite_overrides
    merged = {section: dict(values) for section, values in suite_overrides.items()}
    for section, values in (run_overrides or {}).items():
        merged.setdefault(section, {}).update(values)
    return merged


def _run_config(run, scene, episodes, backend_name, data_source, manifest_id, video_required=None, trace_required=None, parameter_overrides=None):
    parameters = effective_parameters(video_required if video_required is not None else backend_name == "isaac", parameter_overrides)
    controller_parameters = parameters["raw_mpc"] if run.variant == "raw" else parameters["minco_mpc"]
    episode_count = max(1, len(episodes))
    timeout_per_episode_s = 240.0 if run.variant == "raw" else 600.0
    run_timeout_s = max(1800.0, timeout_per_episode_s * episode_count)
    return {
        **asdict(run), "backend":backend_name, "data_source":data_source, "manifest_id":manifest_id,
        "scene_path":scene.scene_path, "asset_hash":scene.asset_hash,
        "episode_uids":[episode.episode_uid for episode in episodes],
        "episodes":[episode.as_dict() for episode in episodes],
        "speed_mps":controller_parameters.get("desired_v_mps", controller_parameters.get("desired_v")), "max_yaw_rate_radps":controller_parameters.get("w_max_radps", controller_parameters.get("w_max")), "optimization_safe_dist":parameters["minco"]["optimization_safe_distance_m"], "validation_safe_dist":parameters["minco"]["validation_safe_distance_m"],
        "timeout_s":run_timeout_s,
        "timeout_policy":{
            "minimum_s":1800.0,
            "per_episode_s":timeout_per_episode_s,
            "episode_count":episode_count,
        },
        "video_required":backend_name == "isaac" if video_required is None else bool(video_required),
        "trace_required":backend_name == "isaac" if trace_required is None else bool(trace_required),
        "raw_controller":"original-navdp-mpc" if backend_name == "isaac" and run.variant == "raw" else "disabled",
        "effective_parameters":parameters,
    }


def run_suite(config_path, backend_name=None, resume=False, retry_failed=False, dry_run=False, analysis_only=False, allow_real_simulation=False, skip_video=False):
    config = load_suite(config_path)
    if analysis_only:
        raise ValueError(
            "run-suite --analysis-only is disabled because it can mutate suite "
            "artifacts; use analyze-suite-readonly with an external --output"
        )
    backend_name = backend_name or config.backend
    if backend_name not in {"mock", "isaac"}: raise ValueError("backend must be mock or isaac")
    manifest = load_manifest(config.manifest_path); layout = ResultLayout(config.output_root); backend = MockBackend() if backend_name == "mock" else IsaacNavDPBackend(Path("."))
    video_required = backend_name == "isaac" and bool((config.video or {}).get("enabled", True)) and not skip_video
    trace_required = backend_name == "isaac" and bool((config.monitor or {}).get("planning_trace", True))
    suite_dir = layout.suite_dir(config.suite_id); suite_dir.mkdir(parents=True, exist_ok=True)
    data_source = "SIMULATED" if backend_name == "mock" else ("DRY_RUN" if dry_run else "REAL")
    parameter_overrides = {section:dict(values) for section, values in (config.parameters or {}).items()}
    video_overrides = parameter_overrides.setdefault("video", {})
    for key in ("fps", "crf", "scale"):
        if key in (config.video or {}): video_overrides[key] = config.video[key]
    atomic_json(suite_dir / "suite_config.json", {"suite_id": config.suite_id, "manifest": str(config.manifest_path), "backend": backend.name, "data_source": data_source})
    atomic_json(suite_dir / "scenario_manifest.json", json.loads(config.manifest_path.read_text(encoding="utf-8")))
    atomic_json(suite_dir / "suite_status.json", {"status":"DRY_RUN" if dry_run else "RUNNING", "backend":backend.name, "data_source":data_source})
    if dry_run:
        commands = []; server_commands = []
        scenes = {scene.scene_id:scene for scene in manifest.scenes}
        for run in expand_runs(config, manifest):
            episodes = [episode for episode in scenes[run.scene_id].episodes if episode.seed == run.seed]
            if backend_name == "isaac":
                run_dir = layout.run_dir(run)
                merged_overrides = _merge_parameter_overrides(parameter_overrides, run.parameter_overrides)
                atomic_json(run_dir / "run_config.json", _run_config(run, scenes[run.scene_id], episodes, backend.name, data_source, manifest.manifest_id, video_required, trace_required, merged_overrides))
                run_configuration = json.loads((run_dir / "run_config.json").read_text())
                commands.append(backend.build_command(run, run_dir, config.manifest_path, scenes[run.scene_id], episodes, save_video=video_required, save_trace=trace_required, effective=run_configuration["effective_parameters"]))
                server_commands.append(backend.build_server_command(run_dir, run.run_id, run.seed))
                write_run_manifest(run_dir / "run_manifest.json", backend.repo_root, json.loads((run_dir / "run_config.json").read_text()), commands[-1], server_commands[-1], probe_external=False)
        atomic_json(suite_dir / "dry_run_plan.json", {"backend":backend_name, "run_count":len(list(expand_runs(config, manifest))), "commands":commands, "server_commands":server_commands, "started_processes":0})
        return SuiteResult(0, 0, 0)
    if backend_name == "isaac" and not allow_real_simulation: raise PermissionError("isaac backend requires --dry-run or --allow-real-simulation")
    completed = skipped = failed = 0
    scenes = {scene.scene_id: scene for scene in manifest.scenes}
    expanded_runs = list(expand_runs(config, manifest))
    episode_counts = [len([episode for episode in scenes[run.scene_id].episodes if episode.seed == run.seed]) for run in expanded_runs]
    progress = SuiteProgressReporter(sum(episode_counts))
    for run_index, (run, expected_episode_count) in enumerate(zip(expanded_runs, episode_counts), start=1):
        run_dir = layout.run_dir(run); status_path = run_dir / "run_status.json"
        episodes = [episode for episode in scenes[run.scene_id].episodes if episode.seed == run.seed]
        progress.start_run(run_index, len(expanded_runs), run.variant, run.scene_id, expected_episode_count)
        if resume and status_path.exists() and json.loads(status_path.read_text()).get("status") == "COMPLETE" and validate_run(run_dir, write_report=False)["valid"]:
            progress.skip_completed_run(); skipped += 1; continue
        lifecycle = RunLifecycle(run_dir)
        if resume and lifecycle.status == "FAILED" and not retry_failed:
            progress.fail_run("previous FAILED run skipped; pass --retry-failed")
            skipped += 1; continue
        if resume and lifecycle.status == "RUNNING":
            lifecycle.transition("INTERRUPTED", reason="orphaned run detected during resume")
        if lifecycle.status in {"FAILED", "COMPLETE"}: # explicit rerun or --retry-failed gets a fresh record
            lifecycle.status = "CREATED"; lifecycle._write(restarted=True)
        try:
            if lifecycle.status in {"CREATED", "INTERRUPTED"}:
                merged_overrides = _merge_parameter_overrides(parameter_overrides, run.parameter_overrides)
                atomic_json(run_dir / "run_config.json", _run_config(run, scenes[run.scene_id], episodes, backend.name, data_source, manifest.manifest_id, video_required, trace_required, merged_overrides))
                lifecycle.transition("RUNNING"); writer = AsyncRecordWriter(run_dir, SCHEMAS)
                try:
                    if backend_name == "isaac":
                        run_configuration = json.loads((run_dir / "run_config.json").read_text())
                        command = backend.build_command(run, run_dir, config.manifest_path, scenes[run.scene_id], episodes, save_video=video_required, save_trace=trace_required, effective=run_configuration["effective_parameters"])
                        server_command = backend.build_server_command(run_dir, run.run_id, run.seed)
                        write_run_manifest(run_dir / "run_manifest.json", backend.repo_root, json.loads((run_dir / "run_config.json").read_text()), command, server_command)
                        backend.run(run, episodes, writer, allow_real_simulation=allow_real_simulation, command=command, progress_callback=progress.update_run)
                    else:
                        backend.run(run, episodes, writer)
                finally:
                    writer.close()
                lifecycle.transition("SIMULATION_COMPLETE")
            if lifecycle.status == "SIMULATION_COMPLETE": lifecycle.transition("VALIDATING")
            if lifecycle.status == "VALIDATING":
                validation = validate_run(run_dir)
                if not validation["valid"]: raise RuntimeError("run validation failed: " + "; ".join(validation["errors"]))
                lifecycle.transition("ANALYZING")
            if lifecycle.status == "ANALYZING":
                analyze_run(run_dir); lifecycle.transition("COMPLETE", validation="PASS")
            progress.finish_run()
            completed += 1
        except BaseException as error:
            progress.fail_run(error)
            failed += 1
            if lifecycle.status not in {"FAILED", "COMPLETE"}: lifecycle.status = "FAILED"; lifecycle._write(error=repr(error))
            raise
    analyze_suite(suite_dir)
    atomic_json(suite_dir / "suite_status.json", {"status":"COMPLETE", "backend":backend.name, "data_source":data_source, "completed":completed, "skipped":skipped, "failed":failed})
    generate_artifact_manifest(suite_dir)
    return SuiteResult(completed, skipped, failed)
