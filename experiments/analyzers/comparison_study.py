from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, TypeVar

from experiments.analyzers.comparison_index import (
    EpisodeGroup,
    EpisodeRecord,
    build_episode_groups,
)
from experiments.analyzers.readonly import (
    snapshot_input_evidence,
    snapshot_protected_receipts,
)
from experiments.analyzers.trace_evidence import (
    load_trace_evidence,
    render_trace_evidence,
)
from experiments.visualizers.paired_video import (
    VideoSource,
    render_paired_episode_video,
    validate_paired_video_bundle,
)


GENERATOR_VERSION = "task03-comparison-v2"
PAIRING_KEY = ("experiment_id", "scene_id", "seed", "episode_uid")
TRACE_COVERAGE_TAGS = (
    "RAW_UNSAFE",
    "HIGH_TURN",
    "JUMP_INPUT",
    "MINCO_FAIL",
    "RAW_SAFE",
    "LOW_TURN",
    "STABLE_INPUT",
)
_TraceCandidate = TypeVar("_TraceCandidate")


@dataclass(frozen=True)
class ComparisonStudyResult:
    input_suite: Path
    output_dir: Path
    manifest_path: Path
    artifact_manifest_path: Path
    episode_group_count: int
    paired_video_count: int
    rendered_trace_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _source_classification(suite_dir: Path) -> str:
    path = suite_dir / "suite_config.json"
    if not path.is_file():
        return "UNKNOWN"
    config = json.loads(path.read_text(encoding="utf-8"))
    source = str(config.get("data_source", "UNKNOWN")).upper()
    return {
        "REAL": "ISAAC_REAL_SIMULATION",
        "ISAAC_REAL_SIMULATION": "ISAAC_REAL_SIMULATION",
        "SIMULATED": "SIMULATED",
        "DRY_RUN": "DRY_RUN",
    }.get(source, "UNKNOWN")


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _meaningful_reason(*values: object) -> str:
    sentinels = {"", "NONE", "N/A", "NA", "OK", "SUCCESS"}
    for value in values:
        reason = str(value).strip()
        if reason.upper() not in sentinels:
            return reason
    return ""


def _terminal_status(record: EpisodeRecord) -> str:
    metrics = record.metrics
    reason = str(metrics.get("done_reason", "")).strip().upper()
    if reason and reason != "UNKNOWN":
        return reason
    if _truth(metrics.get("collision", False)):
        return "COLLISION"
    if _truth(metrics.get("timeout", False)):
        return "TIMEOUT"
    if _truth(metrics.get("success", False)):
        return "SUCCESS"
    return "UNKNOWN"


def _finite_or_none(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _group_id(group: EpisodeGroup) -> str:
    experiment, scene, seed, episode_uid = group.key
    return _slug(f"{experiment}_{scene}_seed{seed}_{episode_uid}")


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _paired_rows(groups: list[EpisodeGroup], data_source: str) -> list[dict]:
    rows = []
    metric_fields = (
        "success",
        "collision",
        "timeout",
        "done_reason",
        "episode_duration_s",
        "actual_path_length_m",
        "repository_spl",
        "tracking_error_rmse_m",
        "minimum_executed_clearance_m",
    )
    for group in groups:
        experiment, scene, seed, episode_uid = group.key
        row = {
            "experiment_id": experiment,
            "scene_id": scene,
            "seed": seed,
            "episode_uid": episode_uid,
            "pairing_key": "|".join(map(str, group.key)),
            "status": group.status,
            "missing_variants": "|".join(group.missing_variants),
            "data_source": data_source,
        }
        for variant, prefix in (
            ("raw", "raw"),
            ("minco-cold", "cold"),
            ("minco-hot", "hot"),
        ):
            record = group.variants.get(variant)
            row[f"{prefix}_run_id"] = record.run.run_id if record else ""
            for field in metric_fields:
                row[f"{prefix}_{field}"] = (
                    record.metrics.get(field, "") if record else ""
                )
        rows.append(row)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _trace_context(record: EpisodeRecord, planning_cycle_uid: str) -> dict[str, str]:
    cycle = next(
        (
            row
            for row in _read_csv(record.run.run_dir / "planning_cycles.csv")
            if row.get("planning_cycle_uid") == planning_cycle_uid
        ),
        {},
    )
    candidates = [
        row
        for row in _read_csv(record.run.run_dir / "candidate_metrics.csv")
        if row.get("planning_cycle_uid") == planning_cycle_uid
    ]
    selected = next(
        (row for row in candidates if _truth(row.get("selected", False))),
        candidates[0] if candidates else {},
    )
    plan_uid = selected.get("plan_uid", "")
    plan = next(
        (
            row
            for row in _read_csv(record.run.run_dir / "plan_metrics.csv")
            if plan_uid and row.get("plan_uid") == plan_uid
        ),
        {},
    )
    tags = []
    for field in ("raw_safety_class", "turn_class", "temporal_class"):
        value = str(plan.get(field, "")).strip()
        if value:
            tags.append(value)
    if _truth(plan.get("hot_wrong_accept", False)):
        tags.append("HOT_WRONG_ACCEPT")
    failure_reason = _meaningful_reason(
        selected.get("failure_reason", ""),
        cycle.get("validation_failure_reason", ""),
        cycle.get("failure_reason", ""),
        plan.get("failure_reason", ""),
    )
    if failure_reason:
        tags.append("MINCO_FAIL")
    return {
        "plan_uid": plan_uid,
        "failure_reason": failure_reason,
        "case_tags": "|".join(dict.fromkeys(tags)),
        "raw_min_clearance_m": plan.get("raw_min_clearance_m", ""),
        "raw_unsafe_ratio": plan.get("raw_unsafe_ratio", ""),
        "minco_min_clearance_m": plan.get("minco_min_clearance_m", ""),
        "minco_unsafe_ratio": plan.get("minco_unsafe_ratio", ""),
    }


def _select_stratified_trace_candidates(
    candidates: list[tuple[_TraceCandidate, tuple[str, ...]]],
    *,
    limit: int,
) -> list[tuple[_TraceCandidate, tuple[str, ...]]]:
    """Cover diagnostic tags first, preserving the caller's stable tie-break."""
    selected: list[tuple[_TraceCandidate, tuple[str, ...]]] = []
    selected_indices: set[int] = set()
    covered_tags: set[str] = set()
    for target in TRACE_COVERAGE_TAGS:
        if target in covered_tags:
            continue
        for index, candidate in enumerate(candidates):
            if index not in selected_indices and target in candidate[1]:
                selected.append(candidate)
                selected_indices.add(index)
                covered_tags.update(candidate[1])
                break
        if len(selected) >= limit:
            return selected
    for index, candidate in enumerate(candidates):
        if index not in selected_indices:
            selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _resume_compatible(
    output_dir: Path,
    input_suite: Path,
    protected_hashes: Mapping[str, str],
    consumed_inputs: list[dict[str, object]],
    max_trace_cases: int,
) -> None:
    manifest_path = output_dir / "study_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("cannot resume: study_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "input_suite": str(input_suite),
        "input_receipt_hashes": dict(protected_hashes),
        "consumed_inputs": consumed_inputs,
        "max_trace_cases": max_trace_cases,
        "generator_version": GENERATOR_VERSION,
    }
    mismatches = [
        key for key, value in expected.items() if manifest.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "cannot resume comparison study: incompatible "
            + ", ".join(mismatches)
        )


def _artifact_rows(
    output_dir: Path,
    associations: Mapping[str, dict[str, object]],
    data_source: str,
) -> list[dict[str, object]]:
    rows = []
    excluded = {"artifact_manifest.json"}
    type_by_suffix = {
        ".mp4": "paired_video",
        ".png": "figure",
        ".csv": "table",
        ".json": "metadata",
        ".md": "report",
    }
    for path in sorted(
        item
        for item in output_dir.rglob("*")
        if item.is_file() and item.name not in excluded and ".tmp" not in item.name
    ):
        relative = str(path.relative_to(output_dir))
        association = associations.get(relative, {})
        artifact_type = type_by_suffix.get(path.suffix.lower(), "file")
        if (
            artifact_type == "paired_video"
            and path.parent.name.endswith("_evidence")
        ):
            artifact_type = "evidence_media"
        rows.append(
            {
                "artifact_type": artifact_type,
                "path": relative,
                "sha256": _sha256(path),
                "size": path.stat().st_size,
                "episode_uid": association.get("episode_uid", ""),
                "case_uid": association.get("case_uid", ""),
                "variant": association.get("variant", ""),
                "complete": association.get("complete", True),
                "data_source": data_source,
            }
        )
    return rows


def generate_comparison_study(
    input_suite: Path | str,
    output_dir: Path | str,
    *,
    max_trace_cases: int = 12,
    resume: bool = False,
) -> ComparisonStudyResult:
    input_suite = Path(input_suite).resolve()
    output_dir = Path(output_dir).resolve()
    if max_trace_cases < 1:
        raise ValueError("max_trace_cases must be positive")
    if output_dir == input_suite or input_suite in output_dir.parents:
        raise ValueError("comparison output must be outside the input suite")

    protected_before = snapshot_protected_receipts(input_suite)
    consumed_before = snapshot_input_evidence(input_suite)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(
                f"comparison output already exists; pass resume: {output_dir}"
            )
        _resume_compatible(
            output_dir,
            input_suite,
            protected_before,
            consumed_before,
            max_trace_cases,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("paired_videos", "planning_cases", "tables"):
        (output_dir / directory).mkdir(exist_ok=True)

    data_source = _source_classification(input_suite)
    groups = build_episode_groups(input_suite)
    paired_rows = _paired_rows(groups, data_source)
    paired_fields = list(paired_rows[0]) if paired_rows else [
        "experiment_id",
        "scene_id",
        "seed",
        "episode_uid",
        "pairing_key",
        "status",
        "missing_variants",
        "data_source",
    ]
    _write_csv(
        output_dir / "tables" / "paired_episode_metrics.csv",
        paired_fields,
        paired_rows,
    )

    associations: dict[str, dict[str, object]] = {}
    missing_rows: list[dict[str, object]] = []
    paired_video_count = 0
    for group in groups:
        for missing_variant in group.missing_variants:
            missing_rows.append(
                {
                    "episode_uid": group.key[3],
                    "planning_cycle_uid": "",
                    "variant": missing_variant,
                    "field": "variant",
                    "affected_artifact": "paired_video",
                    "reason": "VARIANT_NOT_RECORDED",
                    "next_collection_requirement": (
                        f"record {missing_variant} with shared episode_uid"
                    ),
                }
            )
        sources = {}
        for variant, record in group.variants.items():
            if record.video_path and record.video_receipt_path:
                duration = _finite_or_none(
                    record.metrics.get("episode_duration_s", "")
                )
                sources[variant] = VideoSource(
                    variant=variant,
                    path=record.video_path,
                    receipt_path=record.video_receipt_path,
                    terminal_status=_terminal_status(record),
                    terminal_time_s=duration,
                    control_samples_path=(
                        record.run.run_dir / "control_samples.csv"
                        if (record.run.run_dir / "control_samples.csv").is_file()
                        else None
                    ),
                    control_episode_uid=group.key[3],
                )
            else:
                missing_rows.append(
                    {
                        "episode_uid": group.key[3],
                        "planning_cycle_uid": "",
                        "variant": variant,
                        "field": (
                            "video"
                            if record.video_path is None
                            else "video_receipt"
                        ),
                        "affected_artifact": "paired_video",
                        "reason": "VIDEO_EVIDENCE_MISSING",
                        "next_collection_requirement": (
                            "record episode video and video_complete receipt"
                        ),
                    }
                )
        if len(sources) >= 2:
            label = "_".join(
                name.replace("minco-", "")
                for name in ("raw", "minco-cold", "minco-hot")
                if name in ("raw", "minco-cold", "minco-hot")
                and name in sources
            )
            path = (
                output_dir
                / "paired_videos"
                / f"{_group_id(group)}_{_slug(label)}.mp4"
            )
            sidecar = path.with_suffix(".comparison.json")
            evidence_path = path.with_name(f"{path.stem}_evidence")
            if (
                resume
                and path.is_file()
                and sidecar.is_file()
                and evidence_path.is_dir()
            ):
                existing = json.loads(sidecar.read_text(encoding="utf-8"))
                expected_order = [
                    variant
                    for variant in ("raw", "minco-cold", "minco-hot")
                    if variant in sources
                ]
                if (
                    existing.get("episode_uid") != group.key[3]
                    or existing.get("panel_order") != expected_order
                ):
                    raise ValueError(
                        f"cannot resume incompatible paired video: {path}"
                    )
                bundle_errors = validate_paired_video_bundle(path)
                if bundle_errors:
                    raise ValueError(
                        f"cannot resume invalid paired video: {path}: "
                        + "; ".join(bundle_errors)
                    )
                artifact_paths = (
                    path.resolve(),
                    sidecar.resolve(),
                    *sorted(item.resolve() for item in evidence_path.iterdir()),
                )
            else:
                receipt = render_paired_episode_video(
                    sources,
                    path,
                    episode_uid=group.key[3],
                    data_source=data_source,
                )
                artifact_paths = (
                    receipt.output_path,
                    receipt.output_path.with_suffix(".comparison.json"),
                    *sorted(
                        item.resolve()
                        for item in receipt.evidence_package_path.iterdir()
                    ),
                )
            paired_video_count += 1
            for artifact in artifact_paths:
                associations[str(artifact.relative_to(output_dir))] = {
                    "episode_uid": group.key[3],
                    "complete": len(sources) == 3,
                }

    trace_candidates: list[tuple[EpisodeGroup, EpisodeRecord, Path]] = []
    variant_priority = {"minco-cold": 0, "minco-hot": 1, "raw": 2}
    for group in groups:
        records = sorted(
            group.variants.values(),
            key=lambda record: (
                variant_priority.get(record.run.variant, 9),
                record.run.variant,
            ),
        )
        for record in records:
            for trace_path in record.trace_paths:
                trace_candidates.append((group, record, trace_path))
    trace_candidates.sort(
        key=lambda item: (
            item[0].key,
            variant_priority.get(item[1].run.variant, 9),
            str(item[2]),
        )
    )

    annotated_trace_candidates = []
    for candidate in trace_candidates:
        _, record, trace_path = candidate
        planning_cycle_uid = trace_path.stem.removeprefix("planning_trace_")
        context = _trace_context(record, planning_cycle_uid)
        tags = tuple(
            tag for tag in context["case_tags"].split("|") if tag
        )
        annotated_trace_candidates.append((candidate, tags))
    selected_trace_candidates = _select_stratified_trace_candidates(
        annotated_trace_candidates,
        limit=max_trace_cases,
    )

    case_rows = []
    for (group, record, trace_path), _ in selected_trace_candidates:
        planning_cycle_uid = trace_path.stem.removeprefix("planning_trace_")
        case_uid = _slug(f"{planning_cycle_uid}_{record.run.variant}")
        evidence = load_trace_evidence(trace_path)
        receipt = render_trace_evidence(
            evidence,
            output_dir / "planning_cases",
            case_uid=case_uid,
            variant=record.run.variant,
            data_source=data_source,
        )
        context = _trace_context(record, planning_cycle_uid)
        figures = [
            str(path.relative_to(output_dir))
            for path in receipt.artifact_paths
            if path.suffix == ".png"
        ]
        video_path = (
            next(
                (
                    str(path.relative_to(output_dir))
                    for path in (output_dir / "paired_videos").glob(
                        f"{_group_id(group)}_*.mp4"
                    )
                ),
                "",
            )
        )
        case_rows.append(
            {
                "case_uid": case_uid,
                "episode_uid": group.key[3],
                "plan_uid": context["plan_uid"],
                "planning_cycle_uid": planning_cycle_uid,
                "variant": record.run.variant,
                "scene_id": record.run.scene_id,
                "speed": (
                    record.run.speed_mps
                    if record.run.speed_mps is not None
                    else ""
                ),
                "case_tags": context["case_tags"],
                "raw_min_clearance_m": context["raw_min_clearance_m"],
                "raw_unsafe_ratio": context["raw_unsafe_ratio"],
                "minco_min_clearance_m": context["minco_min_clearance_m"],
                "minco_unsafe_ratio": context["minco_unsafe_ratio"],
                "failure_reason": context["failure_reason"],
                "trace_path": str(trace_path),
                "video_path": video_path,
                "figure_paths": "|".join(figures),
                "data_source": data_source,
            }
        )
        for artifact in receipt.artifact_paths:
            associations[str(artifact.relative_to(output_dir))] = {
                "episode_uid": group.key[3],
                "case_uid": case_uid,
                "variant": record.run.variant,
                "complete": True,
            }
        for field in receipt.missing_evidence:
            missing_rows.append(
                {
                    "episode_uid": group.key[3],
                    "planning_cycle_uid": planning_cycle_uid,
                    "variant": record.run.variant,
                    "field": field,
                    "affected_artifact": case_uid,
                    "reason": "TRACE_FIELD_NOT_RECORDED",
                    "next_collection_requirement": (
                        f"record trace array {field} with schema metadata"
                    ),
                }
            )

    case_fields = [
        "case_uid",
        "episode_uid",
        "plan_uid",
        "planning_cycle_uid",
        "variant",
        "scene_id",
        "speed",
        "case_tags",
        "raw_min_clearance_m",
        "raw_unsafe_ratio",
        "minco_min_clearance_m",
        "minco_unsafe_ratio",
        "failure_reason",
        "trace_path",
        "video_path",
        "figure_paths",
        "data_source",
    ]
    _write_csv(output_dir / "tables" / "case_index.csv", case_fields, case_rows)
    (output_dir / "tables" / "case_index.json").write_text(
        json.dumps(case_rows, indent=2) + "\n", encoding="utf-8"
    )
    missing_fields = [
        "episode_uid",
        "planning_cycle_uid",
        "variant",
        "field",
        "affected_artifact",
        "reason",
        "next_collection_requirement",
    ]
    unique_missing = {
        tuple(str(row.get(field, "")) for field in missing_fields): row
        for row in missing_rows
    }
    missing_rows = [
        unique_missing[key] for key in sorted(unique_missing)
    ]
    _write_csv(
        output_dir / "tables" / "missing_evidence.csv",
        missing_fields,
        missing_rows,
    )

    study_manifest = {
        "schema_version": 1,
        "study_id": output_dir.name,
        "generator_version": GENERATOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_suite": str(input_suite),
        "input_receipt_hashes": dict(protected_before),
        "consumed_inputs": consumed_before,
        "pairing_key": list(PAIRING_KEY),
        "max_trace_cases": max_trace_cases,
        "data_source": data_source,
        "episode_group_count": len(groups),
        "complete_group_count": sum(
            group.status == "COMPLETE" for group in groups
        ),
        "incomplete_group_count": sum(
            group.status != "COMPLETE" for group in groups
        ),
        "paired_video_count": paired_video_count,
        "rendered_trace_count": len(case_rows),
        "missing_evidence_count": len(missing_rows),
    }
    study_manifest_path = output_dir / "study_manifest.json"
    study_manifest_path.write_text(
        json.dumps(study_manifest, indent=2) + "\n", encoding="utf-8"
    )

    exact_sync_count = 0
    fixed_sync_count = 0
    for sidecar in (output_dir / "paired_videos").glob("*.comparison.json"):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        methods = payload.get("sync_method_by_variant", {}).values()
        exact_sync_count += sum(
            method == "RECORDED_ABSOLUTE_TIMESTAMPS" for method in methods
        )
        fixed_sync_count += sum(
            method == "FIXED_FPS_RECONSTRUCTION" for method in methods
        )
    report = (
        "# Read-Only RAW / MINCO Comparison Study\n\n"
        f"Input suite: `{input_suite}`  \n"
        f"Data source: `{data_source}`  \n"
        f"Strict pairing key: `{PAIRING_KEY}`  \n\n"
        "## Evidence generated\n\n"
        f"- Episode groups: {len(groups)}\n"
        f"- Paired videos: {paired_video_count}\n"
        f"- Rendered planning traces: {len(case_rows)}\n"
        f"- Missing-evidence records: {len(missing_rows)}\n\n"
        "## Synchronization boundary\n\n"
        f"Absolute timestamp streams: {exact_sync_count}; fixed-FPS reconstructed "
        f"streams: {fixed_sync_count}. Only streams declaring absolute epoch time "
        "and one shared clock domain are aligned as exact wall clock; relative "
        "and fixed-FPS streams remain non-exact. Each comparison sidecar records "
        "its method and error bound. Shorter "
        "variants freeze their final frame.\n\n"
        "## Interpretation boundary\n\n"
        "Videos show recorded visual/control behavior and relative timing under "
        "the declared synchronization method. They do not prove unrecorded ESDF, "
        "collision truth, causal superiority, or generalization. Missing trace "
        "fields remain N/A and are listed in `tables/missing_evidence.csv`.\n"
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    protected_after = snapshot_protected_receipts(input_suite)
    if protected_before != protected_after:
        raise RuntimeError("comparison generation mutated protected input receipts")
    if consumed_before != snapshot_input_evidence(input_suite):
        raise RuntimeError("comparison generation mutated source evidence")

    artifact_manifest = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "input_suite": str(input_suite),
        "input_receipt_hashes": dict(protected_before),
        "consumed_inputs": consumed_before,
        "pairing_key": list(PAIRING_KEY),
        "data_source": data_source,
        "artifacts": _artifact_rows(output_dir, associations, data_source),
    }
    artifact_manifest_path = output_dir / "artifact_manifest.json"
    artifact_manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return ComparisonStudyResult(
        input_suite=input_suite,
        output_dir=output_dir,
        manifest_path=study_manifest_path,
        artifact_manifest_path=artifact_manifest_path,
        episode_group_count=len(groups),
        paired_video_count=paired_video_count,
        rendered_trace_count=len(case_rows),
    )


def validate_comparison_study(output_dir: Path | str) -> list[str]:
    output_dir = Path(output_dir).resolve()
    errors: list[str] = []
    required = (
        "study_manifest.json",
        "artifact_manifest.json",
        "report.md",
        "tables/paired_episode_metrics.csv",
        "tables/case_index.csv",
        "tables/case_index.json",
        "tables/missing_evidence.csv",
    )
    for relative in required:
        if not (output_dir / relative).is_file():
            errors.append(f"missing required artifact: {relative}")
    if errors:
        return errors

    try:
        study = json.loads(
            (output_dir / "study_manifest.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable manifest: {error}"]

    if study.get("schema_version") != 1:
        errors.append("study schema_version mismatch")
    if manifest.get("schema_version") != 1:
        errors.append("artifact schema_version mismatch")
    if manifest.get("pairing_key") != list(PAIRING_KEY):
        errors.append("pairing_key mismatch")
    if manifest.get("data_source") != study.get("data_source"):
        errors.append("study/artifact data_source mismatch")

    input_suite_text = manifest.get("input_suite")
    if not input_suite_text:
        errors.append("missing input_suite")
    else:
        input_suite = Path(input_suite_text)
        try:
            current_hashes = snapshot_protected_receipts(input_suite)
        except (FileNotFoundError, ValueError) as error:
            errors.append(f"input suite unavailable: {error}")
        else:
            if current_hashes != manifest.get("input_receipt_hashes"):
                errors.append("input receipt hashes changed")
            current_inputs = snapshot_input_evidence(input_suite)
            if current_inputs != manifest.get("consumed_inputs"):
                errors.append("consumed input evidence changed")

    seen = set()
    paired_video_count = 0
    for index, row in enumerate(manifest.get("artifacts", [])):
        key = (
            row.get("artifact_type"),
            row.get("path"),
            row.get("episode_uid", ""),
        )
        if key in seen:
            errors.append(f"duplicate artifact key: {key}")
        seen.add(key)
        relative_text = row.get("path")
        if not relative_text:
            errors.append(f"artifact {index} has no path")
            continue
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"artifact path escapes output: {relative_text}")
            continue
        artifact = (output_dir / relative).resolve()
        if artifact.parent != output_dir and output_dir not in artifact.parents:
            errors.append(f"artifact path escapes output: {relative_text}")
            continue
        if not artifact.is_file():
            errors.append(f"missing artifact: {relative_text}")
            continue
        if artifact.stat().st_size != row.get("size"):
            errors.append(f"size mismatch: {relative_text}")
        if _sha256(artifact) != row.get("sha256"):
            errors.append(f"hash mismatch: {relative_text}")
        if row.get("data_source") != manifest.get("data_source"):
            errors.append(f"data_source mismatch: {relative_text}")
        if row.get("artifact_type") == "paired_video":
            paired_video_count += 1

    if paired_video_count != study.get("paired_video_count"):
        errors.append(
            "paired_video_count mismatch: "
            f"{paired_video_count} != {study.get('paired_video_count')}"
        )
    for paired_video in sorted((output_dir / "paired_videos").glob("*.mp4")):
        errors.extend(
            f"paired evidence invalid ({paired_video.name}): {error}"
            for error in validate_paired_video_bundle(paired_video)
        )
    return errors
