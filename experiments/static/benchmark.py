from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

import numpy as np

from experiments.core.artifact_receipt import (
    file_receipt,
    inventory_receipts,
    validate_file_receipt,
)
from experiments.static.case_schema import (
    StaticCase,
    load_static_case,
    save_static_case,
)
from experiments.static.metrics import compute_static_case_metrics
from experiments.static.runner import (
    StaticRunResult,
    load_legacy_profile,
    native_environment_diagnostics,
    run_static_case,
)
from experiments.static.synthetic import generate_catalogue
from experiments.static.trace_import import import_trace_case
from experiments.visualizers.static_benchmark import render_static_case


@dataclass(frozen=True)
class StaticBenchmarkReceipt:
    output_dir: Path
    manifest_path: Path
    case_count: int
    deterministic: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _git_receipt(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            args,
            cwd=repo_root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return completed.stdout.strip()

    return {
        "git_head": run("git", "rev-parse", "HEAD"),
        "status_porcelain": run("git", "status", "--short"),
        "diff_stat": run("git", "diff", "--stat"),
    }


def generate_static_cases(
    config_path: Path | str,
    output_dir: Path | str,
    *,
    case_uids: Iterable[str] | None = None,
) -> list[Path]:
    selected = None if case_uids is None else set(case_uids)
    cases = [
        case
        for case in generate_catalogue(config_path)
        if selected is None or case.case_uid in selected
    ]
    if selected is not None and {case.case_uid for case in cases} != selected:
        missing = sorted(selected - {case.case_uid for case in cases})
        raise ValueError("unknown static case_uids: " + ", ".join(missing))
    return [
        save_static_case(case, output_dir).metadata_path for case in cases
    ]


def replay_static_case(
    metadata_path: Path | str,
    config_path: Path | str,
    output_dir: Path | str,
    mode: str,
) -> tuple[StaticRunResult, dict[str, Any]]:
    case = load_static_case(metadata_path)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    result = run_static_case(case, load_legacy_profile(config_path), mode)
    metrics, detail = compute_static_case_metrics(
        case, result, config["metric_limits"]
    )
    metrics["safe_dist_m"] = float(config["metric_limits"]["safe_distance_m"])
    render_static_case(case, result, metrics, detail, output_dir)
    return result, metrics


def _same_metrics(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    if first["status"] != second["status"] or first["failure_reason"] != second["failure_reason"]:
        return False
    for key in sorted(set(first) & set(second)):
        left, right = first[key], second[key]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if math.isnan(float(left)) and math.isnan(float(right)):
                continue
            if not math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-8):
                return False
    return True


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return fields


def _read_csv_case_uids(
    path: Path,
    expected_header: list[str],
    errors: list[str],
) -> set[str]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            actual_header = reader.fieldnames or []
            rows = list(reader)
    except (OSError, csv.Error) as error:
        errors.append(f"unreadable CSV {path.name}: {error}")
        return set()
    if actual_header != expected_header:
        errors.append(
            f"CSV header mismatch {path.name}: expected {expected_header}, "
            f"got {actual_header}"
        )
    if "case_uid" not in actual_header:
        errors.append(f"CSV missing case_uid {path.name}")
        return set()
    values = [str(row.get("case_uid", "")) for row in rows]
    if any(not value for value in values):
        errors.append(f"CSV contains blank case_uid {path.name}")
    if len(values) != len(set(values)):
        errors.append(f"CSV contains duplicate case_uid {path.name}")
    return set(values)


def validate_static_benchmark(result_dir: Path | str) -> list[str]:
    root = Path(result_dir).resolve()
    required = (
        "legacy_baseline_manifest.json",
        "legacy_case_metrics.csv",
        "legacy_case_index.csv",
        "legacy_report.md",
        "artifact_receipt.json",
    )
    errors = [f"missing {name}" for name in required if not (root / name).is_file()]
    manifest_path = root / "legacy_baseline_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 1:
                errors.append("manifest schema mismatch")
            if not manifest.get("cases"):
                errors.append("manifest contains no cases")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"unreadable manifest: {error}")

    if manifest:
        cases = manifest.get("cases", [])
        if not isinstance(cases, list) or not cases or any(
            not isinstance(row, Mapping) for row in cases
        ):
            errors.append("manifest cases must be a nonempty list")
            cases = []
        case_uids = [str(row.get("case_uid", "")) for row in cases]
        manifest_case_set = set(case_uids)
        if len(case_uids) != len(manifest_case_set) or "" in manifest_case_set:
            errors.append("manifest case_uid values must be nonempty and unique")
        try:
            repeat_count = int(manifest.get("repeat_count", 0))
        except (TypeError, ValueError):
            repeat_count = 0
        if repeat_count < 2:
            errors.append("benchmark repeat_count must be at least 2")
        if manifest.get("all_deterministic") is not True:
            errors.append("benchmark is not deterministic")
        if any(row.get("deterministic_repeat") is not True for row in cases):
            errors.append("one or more cases are not deterministic")

        for row in cases:
            input_receipts = row.get("input_receipts", [])
            artifact_receipts = row.get("artifact_receipts", [])
            if not isinstance(input_receipts, list) or not isinstance(
                artifact_receipts, list
            ):
                errors.append(
                    f"case evidence receipts malformed {row.get('case_uid', '')}"
                )
                continue
            receipts = input_receipts + artifact_receipts
            if any(not isinstance(receipt, Mapping) for receipt in receipts):
                errors.append(
                    f"case evidence receipts malformed {row.get('case_uid', '')}"
                )
                continue
            if len(receipts) < 3:
                errors.append(
                    f"case evidence receipts incomplete {row.get('case_uid', '')}"
                )
            for receipt in receipts:
                errors.extend(validate_file_receipt(root, receipt))
            metadata_receipt = next(
                (
                    item
                    for item in row.get("input_receipts", [])
                    if str(item.get("path", "")).endswith(".case.json")
                ),
                None,
            )
            if metadata_receipt is not None:
                try:
                    loaded = load_static_case(root / str(metadata_receipt["path"]))
                    if loaded.case_uid != row.get("case_uid"):
                        errors.append(
                            f"case metadata UID mismatch {row.get('case_uid', '')}"
                        )
                    if loaded.case_hash != row.get("case_hash"):
                        errors.append(
                            f"case metadata hash mismatch {row.get('case_uid', '')}"
                        )
                except ValueError as error:
                    errors.append(
                        f"invalid static case {row.get('case_uid', '')}: {error}"
                    )

        schemas = manifest.get("csv_schemas", {})
        if not isinstance(schemas, Mapping):
            errors.append("manifest csv_schemas must be a mapping")
            schemas = {}
        for name in (
            "legacy_case_metrics.csv",
            "legacy_case_index.csv",
            "static_case_metrics.csv",
            "static_case_index.csv",
        ):
            path = root / name
            expected_header = schemas.get(name)
            if not path.is_file():
                errors.append(f"missing {name}")
                continue
            if not isinstance(expected_header, list) or not expected_header:
                errors.append(f"missing CSV schema {name}")
                continue
            observed = _read_csv_case_uids(path, expected_header, errors)
            if observed != manifest_case_set:
                errors.append(
                    f"CSV case set mismatch {name}: expected "
                    f"{sorted(manifest_case_set)}, got {sorted(observed)}"
                )

    receipt_path = root / "artifact_receipt.json"
    if receipt_path.is_file():
        try:
            inventory = json.loads(receipt_path.read_text(encoding="utf-8"))
            if inventory.get("schema_version") != 1:
                errors.append("artifact receipt schema mismatch")
            receipts = inventory.get("artifacts", [])
            if not isinstance(receipts, list) or any(
                not isinstance(row, Mapping) for row in receipts
            ):
                errors.append("artifact receipt artifacts must be a list")
                return errors
            recorded_paths = [str(row.get("path", "")) for row in receipts]
            if len(recorded_paths) != len(set(recorded_paths)):
                errors.append("artifact receipt contains duplicate paths")
            for receipt in receipts:
                errors.extend(validate_file_receipt(root, receipt))
            actual_paths = {
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and path.resolve() != receipt_path
            }
            recorded = set(recorded_paths)
            for relative in sorted(actual_paths - recorded):
                errors.append(f"unreceipted artifact {relative}")
            for relative in sorted(recorded - actual_paths):
                errors.append(f"receipt references absent artifact {relative}")
        except (OSError, json.JSONDecodeError, TypeError) as error:
            errors.append(f"unreadable artifact receipt: {error}")
    try:
        from experiments.visualizers.video_evidence import (
            validate_video_evidence_package,
        )
    except ImportError as error:
        errors.append(f"video evidence validator unavailable: {error}")
    else:
        for gif_path in sorted(root.rglob("*.gif")):
            if any(parent.name.endswith("_evidence") for parent in gif_path.parents):
                continue
            package = gif_path.with_name(f"{gif_path.stem}_evidence")
            required_aliases = (
                "caption.md",
                "caption.txt",
                "caption_zh.md",
                "frame_metrics.csv",
                "event_timeline.csv",
                "video_manifest.json",
                "evidence_manifest.json",
                "validation.json",
            )
            for name in required_aliases:
                if not (package / name).is_file():
                    errors.append(
                        f"video evidence missing {gif_path.relative_to(root)}: {name}"
                    )
            if not package.is_dir():
                continue
            package_errors = validate_video_evidence_package(package)
            errors.extend(
                f"video evidence {gif_path.relative_to(root)}: {error}"
                for error in package_errors
            )
            validation_path = package / "validation.json"
            if validation_path.is_file():
                try:
                    validation = json.loads(
                        validation_path.read_text(encoding="utf-8")
                    )
                    if validation.get("valid") is not True or validation.get(
                        "errors"
                    ) != []:
                        errors.append(
                            f"video evidence {gif_path.relative_to(root)}: "
                            "validation.json is not VALID"
                        )
                except (OSError, json.JSONDecodeError) as error:
                    errors.append(
                        f"video evidence {gif_path.relative_to(root)}: "
                        f"unreadable validation.json: {error}"
                    )
    return errors


def run_static_benchmark(
    config_path: Path | str,
    output_dir: Path | str,
    *,
    case_uids: Iterable[str] | None = None,
    trace_limit: int | None = None,
    repeat_count: int = 2,
) -> StaticBenchmarkReceipt:
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"immutable static baseline already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = input_dir / "static_benchmark_config.json"
    config_snapshot.write_bytes(config_path.read_bytes())
    profile = load_legacy_profile(config_path)
    cases = generate_catalogue(config_path)
    if case_uids is not None:
        selected = set(case_uids)
        cases = [case for case in cases if case.case_uid in selected]
        if {case.case_uid for case in cases} != selected:
            raise ValueError("unknown static case selection")

    trace_index_value = config.get("trace_case_index")
    if trace_limit != 0 and trace_index_value:
        trace_index = Path(trace_index_value)
        if not trace_index.is_absolute():
            trace_index = (config_path.parent / trace_index).resolve()
        rows = json.loads(trace_index.read_text(encoding="utf-8"))
        limit = len(rows) if trace_limit is None else max(0, int(trace_limit))
        for row in rows[:limit]:
            cases.append(
                import_trace_case(
                    row["trace_path"],
                    case_uid=f"replay_{row['case_uid']}",
                )
            )
    case_input_dir = output_dir / "cases"
    artifacts_root = output_dir / "artifacts"
    rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    manifest_cases: list[dict[str, Any]] = []
    all_deterministic = True
    for case in cases:
        receipt = save_static_case(case, case_input_dir)
        mode = "recompute" if case.esdf_available else "inspect-only"
        results = [
            run_static_case(case, profile, mode)
            for _ in range(max(1, int(repeat_count)))
        ]
        computed = [
            compute_static_case_metrics(case, result, config["metric_limits"])
            for result in results
        ]
        metrics, detail = computed[0]
        metrics["safe_dist_m"] = float(config["metric_limits"]["safe_distance_m"])
        deterministic = all(
            _same_metrics(metrics, other[0]) for other in computed[1:]
        )
        all_deterministic &= deterministic
        artifact_dir = artifacts_root / case.case_uid
        artifacts = render_static_case(
            case,
            results[0],
            metrics,
            detail,
            artifact_dir,
            footprint_radius_m=float(config.get("visualization_footprint_radius_m", 0.2)),
        )
        row = _json_safe(
            {
                **metrics,
                "expected_category": case.expected_category,
                "mode": mode,
                "deterministic_repeat": deterministic,
            }
        )
        rows.append(row)
        relative_artifacts = [
            str(path.relative_to(output_dir)) for path in artifacts
        ]
        index_rows.append(
            {
                "case_uid": case.case_uid,
                "case_source": case.case_source,
                "expected_category": case.expected_category,
                "mode": mode,
                "status": results[0].status,
                "failure_reason": results[0].diagnostics.get("failure_reason", ""),
                "case_hash": case.case_hash,
                "input_metadata": str(receipt.metadata_path.relative_to(output_dir)),
                "artifact_dir": str(artifact_dir.relative_to(output_dir)),
            }
        )
        manifest_cases.append(
            {
                **index_rows[-1],
                "esdf_available": case.esdf_available,
                "esdf_hash": hashlib.sha256(
                    np.ascontiguousarray(case.esdf_distance).tobytes()
                ).hexdigest()
                if case.esdf_available
                else None,
                "references": dict(case.references),
                "deterministic_repeat": deterministic,
                "artifact_paths": relative_artifacts,
                "input_receipts": [
                    file_receipt(receipt.metadata_path, output_dir),
                    file_receipt(receipt.array_path, output_dir),
                ],
                "artifact_receipts": [
                    file_receipt(path, output_dir) for path in artifacts
                ],
            }
        )
    metrics_header = _write_csv(output_dir / "legacy_case_metrics.csv", rows)
    index_header = _write_csv(output_dir / "legacy_case_index.csv", index_rows)
    _write_csv(output_dir / "static_case_metrics.csv", rows)
    _write_csv(output_dir / "static_case_index.csv", index_rows)
    environment = native_environment_diagnostics()
    manifest = {
        "schema_version": 1,
        "suite_id": config["suite_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_sources": ["STATIC_SYNTHETIC", "STATIC_REPLAY_REAL_TRACE"],
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "config_receipt": file_receipt(config_snapshot, output_dir),
        "effective_minco_config": profile,
        "constraint_profile": config.get("constraint_profile", "legacy"),
        "robot_parameter_source": (
            "calibrated robot profile"
            if config.get("robot_calibration")
            else "legacy pre-calibration defaults; visualization-only footprint radius"
        ),
        "native_environment": environment,
        "git": _git_receipt(config_path.parents[2]),
        "repeat_count": max(1, int(repeat_count)),
        "all_deterministic": all_deterministic,
        "csv_schemas": {
            "legacy_case_metrics.csv": metrics_header,
            "legacy_case_index.csv": index_header,
            "static_case_metrics.csv": metrics_header,
            "static_case_index.csv": index_header,
        },
        "cases": manifest_cases,
    }
    manifest_path = output_dir / "legacy_baseline_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "static_benchmark_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    succeeded = sum(row["status"] == "SUCCEEDED" for row in index_rows)
    failed = sum(row["status"] == "FAILED" for row in index_rows)
    inspected = sum(row["status"] == "INSPECTED" for row in index_rows)
    report = (
        f"# {config.get('constraint_profile', 'legacy')} static MINCO benchmark\n\n"
        f"- Cases: {len(cases)} ({succeeded} native success, {failed} native failure, {inspected} trace inspect-only)\n"
        f"- Repeated execution deterministic: {all_deterministic}\n"
        "- Synthetic cases are labelled `STATIC_SYNTHETIC`; imported traces are labelled "
        "`STATIC_REPLAY_REAL_TRACE`.\n"
        "- Imported V3 traces without a recorded compatible ESDF are inspect-only and are not "
        "presented as native recomputations.\n"
    )
    (output_dir / "legacy_report.md").write_text(report, encoding="utf-8")
    (output_dir / "static_report.md").write_text(report, encoding="utf-8")
    artifact_receipt_path = output_dir / "artifact_receipt.json"
    artifact_receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "root": ".",
                "artifacts": inventory_receipts(
                    output_dir, exclude=(artifact_receipt_path,)
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    errors = validate_static_benchmark(output_dir)
    if errors:
        raise RuntimeError("invalid static baseline: " + "; ".join(errors))
    return StaticBenchmarkReceipt(output_dir, manifest_path, len(cases), all_deterministic)
