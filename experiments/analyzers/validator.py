from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.core.schemas import SCHEMAS
from experiments.core.trace_schema import validate_trace
from experiments.analyzers.data_quality import summarize_field_coverage
from experiments.recorders.run_recorder import atomic_json


PRIMARY_KEYS = {
    "planning_cycles": ("episode_uid", "planning_cycle_uid"),
    "episode_metrics": ("episode_uid",),
    "plan_metrics": ("plan_uid",),
    "candidate_metrics": ("episode_uid", "planning_cycle_uid", "candidate_index"),
    "control_samples": ("episode_uid", "frame_idx"),
    "timing_samples": ("episode_uid", "event_type", "plan_uid", "frame_idx", "metric_name"),
    "events": ("episode_uid", "timestamp_monotonic_s", "event_type"),
}

def _is_blank(value):
    return str(value).strip().lower() in {
        "", "nan", "+nan", "-nan", "inf", "+inf", "-inf",
        "infinity", "+infinity", "-infinity",
    }


def required_diagnostic_errors(table_rows, variant, data_source):
    if data_source != "REAL" or variant == "raw":
        return []
    errors = []
    required_cycle_fields = (
        "attempted_candidate_indices",
        "optimizer_return_code",
        "optimizer_iteration_count",
        "objective",
        "cpp_validation_min_clearance_m",
        "python_validation_min_clearance_m",
        "validation_start_exempt_count",
        "validation_oob_count",
    )
    required_timing_fields = (
        "candidate_screen_ms",
        "candidate_attempt_total_ms",
        "candidate_cpp_total_ms",
        "python_validation_total_ms",
        "adapter_overhead_ms",
    )
    for row in table_rows.get("planning_cycles", []):
        stale = str(row.get("stale", "")).lower() == "true"
        try:
            candidate_count = int(float(row.get("candidate_count") or 0))
        except (TypeError, ValueError):
            candidate_count = 0
        key = row.get("planning_cycle_uid", "")
        if not stale and candidate_count > 0:
            for field in required_timing_fields:
                if _is_blank(row.get(field, "")):
                    errors.append(
                        f"missing required MINCO diagnostic: "
                        f"planning_cycles.{field}: {key}"
                    )
        if str(row.get("published", "")).lower() != "true" or stale:
            continue
        for field in required_cycle_fields:
            if _is_blank(row.get(field, "")):
                errors.append(f"missing required MINCO diagnostic: planning_cycles.{field}: {key}")
    required_candidate_fields = (
        "critic_rank", "screen_rank", "screen_valid", "screen_safe",
        "screen_reason", "attempted",
    )
    for row in table_rows.get("candidate_metrics", []):
        key = f"{row.get('planning_cycle_uid', '')}/{row.get('candidate_index', '')}"
        for field in required_candidate_fields:
            if _is_blank(row.get(field, "")):
                errors.append(f"missing required MINCO diagnostic: candidate_metrics.{field}: {key}")
        if str(row.get("screen_valid", "")).lower() == "true":
            for field in ("path_length_m", "min_clearance_m", "unsafe_ratio", "esdf_oob_ratio"):
                if _is_blank(row.get(field, "")):
                    errors.append(f"missing required MINCO diagnostic: candidate_metrics.{field}: {key}")
    return errors


def validate_run(run_dir: Path | str, write_report=True):
    run_dir = Path(run_dir); errors = [] ; counts = {}
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if not config_path.exists(): errors.append("missing run_config.json")
    table_rows = {}
    for table, schema in SCHEMAS.items():
        path = run_dir / f"{table}.csv"
        if not path.exists(): errors.append(f"missing {path.name}"); continue
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream); rows = list(reader); counts[table] = len(rows); table_rows[table] = rows
            if reader.fieldnames != schema: errors.append(f"schema mismatch: {table}")
            if any(row.get("data_source") not in {"SIMULATED", "REAL"} for row in rows): errors.append(f"invalid data_source: {table}")
            expected_source = config.get("data_source")
            if expected_source in {"SIMULATED", "REAL"} and any(row.get("data_source") != expected_source for row in rows): errors.append(f"data_source mismatch: {table}")
            key_fields = PRIMARY_KEYS[table]; seen = set()
            for row in rows:
                key = tuple(row.get(field, "") for field in key_fields)
                if any(value == "" for value in key): errors.append(f"empty primary key: {table}: {key}")
                elif key in seen: errors.append(f"duplicate primary key: {table}: {key}")
                seen.add(key)
                for field, value in row.items():
                    if str(value).strip().lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
                        errors.append(f"non-finite value: {table}.{field}: {key}")
    status_path = run_dir / "run_status.json"
    if not status_path.exists(): errors.append("missing run_status.json")
    for npz_path in sorted((run_dir / "traces").glob("*.npz")) if (run_dir / "traces").exists() else []:
        metadata_path = npz_path.with_name(npz_path.stem + ".metadata.json")
        if not metadata_path.exists(): errors.append(f"missing trace metadata: {npz_path.name}")
        else: errors.extend(validate_trace(npz_path, metadata_path))
    cycles = table_rows.get("planning_cycles", [])
    if config.get("variant") == "raw":
        if any(int(float(row.get("attempted_candidate_count") or 0)) != 0 for row in cycles): errors.append("raw variant attempted MINCO candidates")
        if any(float(row.get("minco_ms") or 0.0) != 0.0 for row in cycles): errors.append("raw variant recorded nonzero minco_ms")
    errors.extend(required_diagnostic_errors(
        table_rows,
        config.get("variant", ""),
        config.get("data_source", ""),
    ))
    expected_uids = list(config.get("episode_uids", []))
    actual_uids = [row.get("episode_uid") for row in table_rows.get("episode_metrics", [])]
    if expected_uids and sorted(actual_uids) != sorted(expected_uids): errors.append("episode completion set does not match run_config")
    if config.get("data_source") == "REAL":
        if len(expected_uids) != 10 or len(set(expected_uids)) != 10:
            errors.append("REAL run requires exactly 10 distinct episode_uids")
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists(): errors.append("missing run_manifest.json")
        else:
            try:
                manifest_payload = json.loads(manifest_path.read_text())
                if not manifest_payload.get("effective_parameters"): errors.append("run_manifest missing effective_parameters")
                if not manifest_payload.get("commands", {}).get("evaluation"): errors.append("run_manifest missing evaluation command")
                if "sha256" not in manifest_payload.get("checkpoint", {}): errors.append("run_manifest missing checkpoint hash")
                host = manifest_payload.get("host", {})
                if not all(key in host for key in ("cpu", "memory_total_bytes", "gpu")): errors.append("run_manifest missing host inventory")
            except Exception as error: errors.append(f"unreadable run_manifest.json: {error}")
        samples_path = run_dir / "resource_samples.csv"
        if not samples_path.exists(): errors.append("missing resource_samples.csv")
        summary_path = run_dir / "resource_summary.json"
        if not summary_path.exists(): errors.append("missing resource_summary.json")
        else:
            try:
                summary = json.loads(summary_path.read_text())
                required = ("started_at_utc", "ended_at_utc", "duration_s", "peak_owned_rss_bytes", "peak_gpu_memory_mib")
                if not all(key in summary for key in required): errors.append("incomplete resource_summary.json")
            except Exception as error: errors.append(f"unreadable resource_summary.json: {error}")
        if config.get("variant") != "raw" and not (run_dir / "esdf_runtime.json").exists():
            errors.append("missing esdf_runtime.json")
    if config.get("trace_required") and not list((run_dir / "traces").glob("*.npz")): errors.append("missing required planning traces")
    if config.get("video_required"):
        for uid in expected_uids:
            video = run_dir / "videos" / f"{uid}.mp4"; metadata = run_dir / "videos" / f"{uid}.video_complete.json"
            if not video.exists() or not metadata.exists(): errors.append(f"missing complete video: {uid}"); continue
            try:
                payload = json.loads(metadata.read_text())
                if not payload.get("complete") or int(payload.get("frame_count", 0)) <= 0: errors.append(f"incomplete video metadata: {uid}")
            except Exception as error: errors.append(f"unreadable video metadata: {uid}: {error}")
    result = {"valid": not errors, "errors": errors, "row_counts": counts}
    summarize_field_coverage(run_dir, write_output=True)
    if write_report:
        atomic_json(run_dir / "validation_report.json", result)
        (run_dir / "validation_report.md").write_text("# Run Validation\n\n" + ("PASS" if result["valid"] else "FAIL") + "\n\n" + "\n".join(f"- {item}" for item in errors) + "\n", encoding="utf-8")
        validation_dir = run_dir / "validation"; validation_dir.mkdir(exist_ok=True)
        atomic_json(validation_dir / "validation_report.json", result)
        (validation_dir / "validation_report.md").write_text("# Run Validation\n\n" + ("PASS" if result["valid"] else "FAIL") + "\n", encoding="utf-8")
    return result
