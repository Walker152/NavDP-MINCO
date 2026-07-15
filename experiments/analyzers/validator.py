from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.core.schemas import SCHEMAS
from experiments.core.trace_schema import validate_trace
from experiments.recorders.run_recorder import atomic_json


PRIMARY_KEYS = {
    "planning_cycles": ("episode_uid", "planning_cycle_uid"),
    "episode_metrics": ("episode_uid",),
    "plan_metrics": ("plan_uid",),
    "candidate_metrics": ("plan_uid", "candidate_index"),
    "control_samples": ("episode_uid", "frame_idx"),
    "timing_samples": ("episode_uid", "event_type", "plan_uid", "frame_idx", "metric_name"),
    "events": ("episode_uid", "timestamp_monotonic_s", "event_type"),
}


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
    expected_uids = list(config.get("episode_uids", []))
    actual_uids = [row.get("episode_uid") for row in table_rows.get("episode_metrics", [])]
    if expected_uids and sorted(actual_uids) != sorted(expected_uids): errors.append("episode completion set does not match run_config")
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
    if write_report:
        atomic_json(run_dir / "validation_report.json", result)
        (run_dir / "validation_report.md").write_text("# Run Validation\n\n" + ("PASS" if result["valid"] else "FAIL") + "\n\n" + "\n".join(f"- {item}" for item in errors) + "\n", encoding="utf-8")
        validation_dir = run_dir / "validation"; validation_dir.mkdir(exist_ok=True)
        atomic_json(validation_dir / "validation_report.json", result)
        (validation_dir / "validation_report.md").write_text("# Run Validation\n\n" + ("PASS" if result["valid"] else "FAIL") + "\n", encoding="utf-8")
    return result
