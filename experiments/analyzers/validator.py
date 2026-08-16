from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.core.schemas import SCHEMAS
from experiments.core.trace_schema import validate_trace
from experiments.analyzers.data_quality import summarize_field_coverage
from experiments.recorders.run_recorder import atomic_json
from experiments.recorders.video_recorder import validate_video_receipt
from experiments.core.failure_taxonomy import classify_reason


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


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def real_machine_truth_errors(config, table_rows):
    if config.get("data_source") != "REAL":
        return []
    errors = []
    is_minco = config.get("variant") != "raw"
    for row in table_rows.get("episode_metrics", []):
        uid = row.get("episode_uid", "")
        if _is_blank(row.get("termination_term_raw", "")):
            errors.append(f"missing raw termination terms: {uid}")
        if _is_blank(row.get("termination_frame_idx", "")):
            errors.append(f"missing termination frame association: {uid}")
        if is_minco and (
            _is_blank(row.get("termination_plan_uid", ""))
            or _is_blank(row.get("termination_planning_cycle_uid", ""))
        ):
            errors.append(f"missing termination plan association: {uid}")
        if _is_blank(row.get("hold_duration_s", "")) or _is_blank(
            row.get("stop_duration_s", "")
        ):
            errors.append(f"missing recovery duration truth: {uid}")
        collision = _truthy(row.get("collision", "")) or str(
            row.get("done_reason", "")
        ) == "COLLISION"
        contact = row.get("contact_detected", "")
        if _is_blank(contact) or collision != _truthy(contact):
            errors.append(f"contact consistency mismatch: {uid}")
        for field in ("done_reason", "failure_reason"):
            reason = str(row.get(field, "")).strip()
            if reason and classify_reason(reason)["reason_source"] == "UNMAPPED":
                errors.append(f"unmapped machine reason: {uid}: {reason}")

    wheel_limit = (
        config.get("effective_parameters", {})
        .get("robot_calibration", {})
        .get("max_wheel_speed_radps")
    )
    try:
        wheel_limited = float(wheel_limit) > 0.0
    except (TypeError, ValueError):
        wheel_limited = False
    if wheel_limited:
        required = (
            "actual_left_wheel_radps", "actual_right_wheel_radps",
            "wheel_speed_limit_radps", "wheel_saturated",
        )
        for row in table_rows.get("control_samples", []):
            if any(_is_blank(row.get(field, "")) for field in required):
                errors.append(
                    f"missing wheel truth: {row.get('episode_uid', '')}/"
                    f"{row.get('frame_idx', '')}"
                )
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
            actual_fields = list(reader.fieldnames or [])
            if actual_fields != schema:
                unknown = set(actual_fields) - set(schema)
                identity_missing = set(SCHEMAS[table][:8]) - set(actual_fields)
                if unknown or identity_missing:
                    errors.append(f"schema mismatch: {table}")
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
    errors.extend(real_machine_truth_errors(config, table_rows))
    if config.get("data_source") == "REAL" and config.get("variant") != "raw":
        for row in cycles:
            published = str(row.get("published", "")).lower() == "true"
            stale = str(row.get("stale", "")).lower() == "true"
            fallback = str(row.get("fallback_mode", ""))
            key = row.get("planning_cycle_uid", "")
            if published and (
                str(row.get("cpp_validation_success", "")).lower() != "true"
                or str(row.get("python_validation_success", "")).lower() != "true"
            ):
                errors.append(
                    f"published MINCO plan lacks hard validation success: {key}"
                )
            if published and (stale or fallback in {"HOLD_LAST", "STOP"}):
                errors.append(
                    f"published plan has stale/recovery semantics: {key}"
                )
            if fallback == "HOLD_LAST" and published:
                errors.append(f"HOLD_LAST recorded as new publication: {key}")
            try:
                oob_count = int(float(row.get("validation_oob_count") or 0))
            except (TypeError, ValueError):
                oob_count = 0
            if published and oob_count > 0:
                errors.append(f"published plan contains ESDF OOB samples: {key}")
            reason = str(row.get("failure_reason", "")).strip()
            if reason and classify_reason(reason)["reason_source"] == "UNMAPPED":
                errors.append(f"unmapped failure reason: {key}: {reason}")
    expected_uids = list(config.get("episode_uids", []))
    raw_actual_uids = [
        row.get("episode_uid")
        for row in table_rows.get("episode_metrics", [])
    ]
    invalid_actual_uids = [
        value
        for value in raw_actual_uids
        if value is None or not str(value).strip()
    ]
    if invalid_actual_uids:
        errors.append("episode_metrics contains empty episode_uid values")
    actual_uids = [
        str(value)
        for value in raw_actual_uids
        if value is not None and str(value).strip()
    ]
    if expected_uids and sorted(actual_uids) != sorted(expected_uids): errors.append("episode completion set does not match run_config")
    if config.get("data_source") == "REAL":
        availability_path = run_dir / "machine_truth_availability.json"
        if not availability_path.exists():
            errors.append("missing machine_truth_availability.json")
        else:
            try:
                availability = json.loads(availability_path.read_text())
                for field in (
                    "contact_sensor", "impact_force_tensor",
                    "collision_object_identity", "wheel_joint_velocity",
                ):
                    if field not in availability:
                        errors.append(
                            f"machine truth availability missing field: {field}"
                        )
            except Exception as error:
                errors.append(f"unreadable machine_truth_availability.json: {error}")
        is_dynamic_stress = str(config.get("suite_id", "")).startswith(
            "task06_dynamic_"
        )
        expected_real_count = 1 if is_dynamic_stress else 10
        if (
            len(expected_uids) != expected_real_count
            or len(set(expected_uids)) != expected_real_count
        ):
            errors.append(
                f"REAL run requires exactly {expected_real_count} distinct episode_uids"
            )
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
        for row in table_rows.get("episode_metrics", []):
            reason = str(row.get("done_reason", "")).strip()
            raw = str(row.get("termination_term_raw", "")).strip()
            if reason in {"", "UNKNOWN"} and not raw:
                errors.append(
                    f"missing machine termination truth: {row.get('episode_uid', '')}"
                )
    if config.get("trace_required") and not list((run_dir / "traces").glob("*.npz")): errors.append("missing required planning traces")
    if config.get("video_required"):
        for uid in expected_uids:
            video = run_dir / "videos" / f"{uid}.mp4"; metadata = run_dir / "videos" / f"{uid}.video_complete.json"
            if not video.exists() or not metadata.exists(): errors.append(f"missing complete video: {uid}"); continue
            try:
                payload = json.loads(metadata.read_text())
                if not payload.get("complete") or int(payload.get("frame_count", 0)) <= 0: errors.append(f"incomplete video metadata: {uid}")
            except Exception as error: errors.append(f"unreadable video metadata: {uid}: {error}")
            else:
                errors.extend(
                    f"video receipt validation failed: {uid}: {error}"
                    for error in validate_video_receipt(video, metadata)
                )
    result = {"valid": not errors, "errors": errors, "row_counts": counts}
    summarize_field_coverage(run_dir, write_output=True)
    if write_report:
        atomic_json(run_dir / "validation_report.json", result)
        (run_dir / "validation_report.md").write_text("# Run Validation\n\n" + ("PASS" if result["valid"] else "FAIL") + "\n\n" + "\n".join(f"- {item}" for item in errors) + "\n", encoding="utf-8")
        validation_dir = run_dir / "validation"; validation_dir.mkdir(exist_ok=True)
        atomic_json(validation_dir / "validation_report.json", result)
        (validation_dir / "validation_report.md").write_text("# Run Validation\n\n" + ("PASS" if result["valid"] else "FAIL") + "\n", encoding="utf-8")
    return result
