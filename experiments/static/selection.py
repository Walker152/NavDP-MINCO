from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

import numpy as np

from experiments.static.case_schema import StaticCase
from experiments.static.metrics import compute_static_case_metrics
from experiments.static.runner import StaticRunResult, load_legacy_profile, run_static_case
from experiments.static.synthetic import generate_catalogue
from experiments.visualizers.static_benchmark import (
    build_paired_static_gif_evidence,
    render_static_case,
)
from experiments.core.artifact_receipt import inventory_receipts


POLICY_VERSION = "best-worst-lexicographic-v1"
CORRIDOR_SHOWCASE_CASES = (
    "syn_l90",
    "syn_s_curve_sparse",
    "syn_l90_dense",
)
# Full-route evidence is deliberately a compact, interpretable matrix rather
# than every one-shot sweep cell.  It contains the nominal baseline, two
# isolated initial-condition extremes, and sparse/dense obstacle geometries.
# Each case is rolled from start to final goal under both profiles; a failed
# hard constraint is retained with its recorded terminal reason.
FULL_ROUTE_SHOWCASE_CASES = (
    "syn_straight",
    "state_v_reverse",
    "state_a_along",
    "state_yaw_reverse",
    "syn_s_curve_sparse",
    "syn_l90_dense",
)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def boundary_case_content_hash(case: StaticCase) -> str:
    """Hash measured geometry/state content independently of the case label."""
    digest = hashlib.sha256()
    scalars = {
        "esdf_resolution": case.esdf_resolution,
        "start_yaw": case.start_yaw,
        "start_yaw_rate": case.start_yaw_rate,
        "esdf_available": case.esdf_available,
    }
    digest.update(
        json.dumps(scalars, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    arrays: dict[str, np.ndarray] = {
        "guide_path_xyz": case.guide_path_xyz,
        "occupancy": case.occupancy,
        "esdf_distance": case.esdf_distance,
        "esdf_origin": case.esdf_origin,
        "start_position": case.start_position,
        "start_velocity": case.start_velocity,
        "start_acceleration": case.start_acceleration,
    }
    if case.terminal_goal is not None:
        arrays["terminal_goal"] = case.terminal_goal
    for name, value in case.auxiliary_arrays.items():
        arrays[f"aux__{name}"] = value
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
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


def default_selection_policy() -> dict[str, Any]:
    policy = {
        "policy_version": POLICY_VERSION,
        # Library callers that do not supply a research configuration keep the
        # historical fixture default.  All formal configs explicitly set the
        # SuperPlanner SFC selection profile.
        "eligible_profile": "safe_corridor_v1",
        "best_order": [
            "hard_validation_pass",
            "min_normalized_margin_desc",
            "guide_deviation_p95_m_asc",
            "path_length_ratio_asc",
            "actual_jerk_p95_mps3_asc",
            "runtime_ms_asc",
            "case_uid_asc",
        ],
        "best_geometry_category_dedup": True,
        "worst_slots": [
            "highest_severity_fail_closed",
            "largest_shape_degradation_with_safe_output",
        ],
        "failure_severity": [
            "VALIDATION_NEGATIVE_ESDF",
            "VALIDATION_ESDF_OOB",
            "VALIDATION_CLEARANCE",
            "VALIDATION_WHEEL_SPEED",
            "VALIDATION_YAW_RATE",
            "VALIDATION_JERK",
            "VALIDATION_ACCELERATION",
            "VALIDATION_VELOCITY",
            "VALIDATION_CORRIDOR",
            "VALIDATION_SFC",
            "SFC_GUIDE_NEGATIVE_ESDF",
            "SFC_GUIDE_UNSAFE",
            "SFC_DISCONNECTED",
            "CORRIDOR_GUIDE_NEGATIVE_ESDF",
            "CORRIDOR_GUIDE_UNSAFE",
            "CORRIDOR_DISCONNECTED",
            "VALIDATION_BUDGET_EXHAUSTED",
            "OPTIMIZER_FAILED",
        ],
        "backup_count_per_side": 2,
        "duplicate_rule": (
            "retain lexicographically first case_uid per semantic content hash; "
            "case labels and profiles do not alter content identity"
        ),
    }
    policy["policy_sha256"] = _canonical_hash(policy)
    return policy


def _number(row: Mapping[str, Any], key: str, default: float) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def select_cases(
    rows: Iterable[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    profile = str(policy["eligible_profile"])
    candidates = sorted(
        (dict(row) for row in rows if str(row.get("profile")) == profile),
        key=lambda row: str(row.get("case_uid", "")),
    )
    eligible: list[dict[str, Any]] = []
    exclusions: dict[str, str] = {}
    seen_hashes: set[str] = set()
    for row in candidates:
        uid = str(row["case_uid"])
        if not bool(row.get("dynamic_replayable", False)):
            exclusions[uid] = str(
                row.get("not_replayable_reason") or "NOT_DYNAMIC_REPLAYABLE"
            )
            continue
        case_hash = str(
            row.get("selection_content_hash") or row.get("case_hash", "")
        )
        if not case_hash or case_hash in seen_hashes:
            exclusions[uid] = "DUPLICATE_CASE_HASH"
            continue
        seen_hashes.add(case_hash)
        eligible.append(row)

    safe = [
        row
        for row in eligible
        if row.get("classification") in {"SAFE_FEASIBLE", "SAFE_BUT_DEGRADED"}
    ]
    best_ranked = sorted(
        safe,
        key=lambda row: (
            -_number(row, "min_normalized_margin", -math.inf),
            _number(row, "guide_deviation_p95_m", math.inf),
            abs(_number(row, "path_length_ratio", math.inf) - 1.0),
            _number(row, "actual_jerk_p95_mps3", math.inf),
            _number(row, "runtime_ms", math.inf),
            str(row["case_uid"]),
        ),
    )
    best: list[str] = []
    used_categories: set[str] = set()
    for row in best_ranked:
        category = str(row.get("expected_category", ""))
        if len(best) < 2 and category in used_categories:
            continue
        best.append(str(row["case_uid"]))
        used_categories.add(category)
    for row in best_ranked:
        uid = str(row["case_uid"])
        if uid not in best:
            best.append(uid)

    severity = {
        reason: index
        for index, reason in enumerate(policy.get("failure_severity", []))
    }
    failed = [
        row
        for row in eligible
        if row.get("classification")
        in {"FAIL_CLOSED_EXPECTED", "VALIDATION_FAILED", "OPTIMIZER_FAILED"}
    ]
    failed_ranked = sorted(
        failed,
        key=lambda row: (
            severity.get(str(row.get("failure_reason")), len(severity)),
            -abs(_number(row, "violation_margin", 0.0)),
            str(row["case_uid"]),
        ),
    )
    degraded = [
        row for row in safe if row.get("classification") == "SAFE_BUT_DEGRADED"
    ]
    degraded_ranked = sorted(
        degraded,
        key=lambda row: (
            -_number(row, "guide_deviation_p95_m", -math.inf),
            -_number(row, "path_length_ratio", -math.inf),
            -_number(row, "actual_jerk_p95_mps3", -math.inf),
            str(row["case_uid"]),
        ),
    )
    worst: list[str] = []
    if failed_ranked:
        worst.append(str(failed_ranked[0]["case_uid"]))
    if degraded_ranked:
        uid = str(degraded_ranked[0]["case_uid"])
        if uid not in worst:
            worst.append(uid)
    for row in failed_ranked + degraded_ranked + list(reversed(best_ranked)):
        uid = str(row["case_uid"])
        if uid not in worst:
            worst.append(uid)

    def complete_rows(
        primary: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        ordered: list[dict[str, Any]] = []
        observed: set[str] = set()
        for row in list(primary) + eligible:
            uid = str(row["case_uid"])
            if uid not in observed:
                observed.add(uid)
                ordered.append(dict(row))
        return ordered

    best_complete = complete_rows(
        best_ranked
        + sorted(
            (row for row in eligible if row not in safe),
            key=lambda row: (
                severity.get(str(row.get("failure_reason")), len(severity)),
                str(row["case_uid"]),
            ),
        )
    )
    worst_complete = complete_rows(
        failed_ranked + degraded_ranked + list(reversed(best_ranked))
    )

    def ranking_records(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        keys = (
            "case_uid",
            "case_hash",
            "selection_content_hash",
            "expected_category",
            "classification",
            "failure_reason",
            "min_normalized_margin",
            "guide_deviation_p95_m",
            "path_length_ratio",
            "actual_jerk_p95_mps3",
            "runtime_ms",
            "violation_margin",
            "factor_name",
            "factor_level",
            "factor_name_secondary",
            "factor_level_secondary",
        )
        return [
            {
                "rank": rank,
                **{key: row.get(key, "") for key in keys},
            }
            for rank, row in enumerate(values, 1)
        ]

    return {
        "policy_version": policy["policy_version"],
        "policy_sha256": policy["policy_sha256"],
        "eligible_case_uids": [str(row["case_uid"]) for row in eligible],
        "exclusions": exclusions,
        "best": best,
        "worst": worst,
        "best_ranking": ranking_records(best_complete),
        "worst_ranking": ranking_records(worst_complete),
    }


def _apply_factor_field(
    specification: dict[str, Any], field: str, level: float
) -> None:
    if field == "velocity_x_mps":
        specification["velocity_xyz_mps"] = [float(level), 0.0, 0.0]
    elif field == "velocity_y_mps":
        specification["velocity_xyz_mps"] = [0.0, float(level), 0.0]
    elif field == "yaw_rad":
        specification["yaw_rad"] = float(level)
    elif field == "yaw_rate_radps":
        specification["yaw_rate_radps"] = float(level)
    elif field == "acceleration_x_mps2":
        specification["acceleration_xyz_mps2"] = [float(level), 0.0, 0.0]
    elif field == "acceleration_y_mps2":
        specification["acceleration_xyz_mps2"] = [0.0, float(level), 0.0]
    else:
        raise ValueError(f"unsupported boundary factor field: {field}")


def _expanded_variant_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    specifications = [dict(row) for row in config.get("state_variants", [])]
    for grid in config.get("factor_grids", []):
        first = grid["x_factor"]
        second = grid["y_factor"]
        for x_index, x_level in enumerate(first["levels"]):
            for y_index, y_level in enumerate(second["levels"]):
                specification: dict[str, Any] = {
                    "case_uid": (
                        f"{grid['grid_uid']}_x{x_index:02d}_y{y_index:02d}"
                    ),
                    "source_case_uid": grid["source_case_uid"],
                    "factor_name": first["name"],
                    "factor_level": x_level,
                    "factor_name_secondary": second["name"],
                    "factor_level_secondary": y_level,
                    "scan_group": grid["grid_uid"],
                    "tags": ["TWO_FACTOR_GRID", str(grid["grid_uid"])],
                    **dict(grid.get("fixed_state", {})),
                }
                _apply_factor_field(
                    specification, str(first["field"]), float(x_level)
                )
                _apply_factor_field(
                    specification, str(second["field"]), float(y_level)
                )
                specifications.append(specification)
    return specifications


def boundary_factor_metadata(
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return explicit, measured factor metadata keyed by generated case UID."""
    metadata: dict[str, dict[str, Any]] = {}
    for specification in _expanded_variant_specs(config):
        uid = str(specification["case_uid"])
        factor_name = str(specification.get("factor_name", ""))
        if not factor_name:
            raise ValueError(f"state variant lacks factor_name: {uid}")
        if "factor_level" not in specification:
            raise ValueError(f"state variant lacks factor_level: {uid}")
        metadata[uid] = {
            "factor_name": factor_name,
            "factor_level": specification["factor_level"],
            "factor_name_secondary": str(
                specification.get("factor_name_secondary", "")
            ),
            "factor_level_secondary": specification.get(
                "factor_level_secondary", ""
            ),
            "scan_group": str(
                specification.get("scan_group", factor_name)
            ),
        }
    return metadata


def _state_variant(case: StaticCase, specification: Mapping[str, Any]) -> StaticCase:
    velocity = np.asarray(
        specification.get("velocity_xyz_mps", case.start_velocity),
        dtype=np.float64,
    )
    acceleration = np.asarray(
        specification.get("acceleration_xyz_mps2", case.start_acceleration),
        dtype=np.float64,
    )
    position = case.start_position
    if "position_xyz_m" in specification:
        position = np.asarray(specification["position_xyz_m"], dtype=np.float64)
    elif "lateral_offset_m" in specification:
        # Pure lateral offset from the guide start: zero longitudinal
        # displacement along the guide direction.
        offset = float(specification["lateral_offset_m"])
        if not math.isfinite(offset):
            raise ValueError("lateral_offset_m must be finite")
        start_direction = (
            case.guide_path_xyz[1, :2] - case.guide_path_xyz[0, :2]
        )
        length = float(np.linalg.norm(start_direction))
        if length <= 1e-9:
            raise ValueError(
                "cannot compute lateral normal from zero-length guide "
                "start segment"
            )
        lateral_normal = np.array(
            [-start_direction[1], start_direction[0]], dtype=np.float64
        ) / length
        position = np.asarray(case.guide_path_xyz[0], dtype=np.float64).copy()
        position[:2] = position[:2] + lateral_normal * offset
    return replace(
        case,
        case_uid=str(specification["case_uid"]),
        start_position=position,
        start_velocity=velocity,
        start_acceleration=acceleration,
        start_yaw=float(specification.get("yaw_rad", case.start_yaw)),
        start_yaw_rate=float(
            specification.get("yaw_rate_radps", case.start_yaw_rate)
        ),
        tags=tuple(case.tags) + tuple(specification.get("tags", [])),
    )


def generate_boundary_cases(config: Mapping[str, Any]) -> list[StaticCase]:
    base_config = Path(str(config["base_case_config"])).resolve()
    base_cases = generate_catalogue(base_config)
    by_uid = {case.case_uid: case for case in base_cases}
    cases = list(base_cases)
    for specification in _expanded_variant_specs(config):
        source_uid = str(specification["source_case_uid"])
        if source_uid not in by_uid:
            raise ValueError(f"unknown source_case_uid: {source_uid}")
        cases.append(_state_variant(by_uid[source_uid], specification))
    return sorted(cases, key=lambda case: case.case_uid)


def _classify(
    status: str, reason: str, metrics: Mapping[str, Any]
) -> str:
    if status == "SUCCEEDED":
        degraded = (
            _number(metrics, "path_length_ratio", 1.0) > 1.25
            or _number(metrics, "guide_deviation_p95_m", 0.0) > 0.35
            or _number(metrics, "backtracking_ratio", 0.0) > 0.05
            or _number(metrics, "self_intersection_count", 0.0) > 0.0
        )
        return "SAFE_BUT_DEGRADED" if degraded else "SAFE_FEASIBLE"
    if reason.startswith("CORRIDOR_") or reason in {
        "VALIDATION_NEGATIVE_ESDF",
        "VALIDATION_ESDF_OOB",
        "VALIDATION_CLEARANCE",
        "VALIDATION_CORRIDOR",
    }:
        return "FAIL_CLOSED_EXPECTED"
    if reason.startswith("OPTIMIZER"):
        return "OPTIMIZER_FAILED"
    if reason.startswith("VALIDATION"):
        return "VALIDATION_FAILED"
    return "INVALID_INPUT"


def _safe_margin(
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    limits: Mapping[str, Any],
) -> float:
    values = [
        (
            _number(metrics, "min_clearance_m", -math.inf)
            - float(limits["safe_distance_m"])
        )
        / max(1e-9, float(limits["safe_distance_m"])),
        (
            float(limits["max_velocity_mps"])
            - _number(metrics, "actual_speed_max_mps", math.inf)
        )
        / max(1e-9, float(limits["max_velocity_mps"])),
        (
            float(limits["max_acceleration_mps2"])
            - _number(metrics, "actual_acc_max_mps2", math.inf)
        )
        / max(1e-9, float(limits["max_acceleration_mps2"])),
        (
            float(limits["max_jerk_mps3"])
            - _number(metrics, "actual_jerk_max_mps3", math.inf)
        )
        / max(1e-9, float(limits["max_jerk_mps3"])),
        (
            float(limits["max_yaw_rate_radps"])
            - _number(metrics, "actual_yaw_rate_max_radps", math.inf)
        )
        / max(1e-9, float(limits["max_yaw_rate_radps"])),
    ]
    corridor = _number(diagnostics, "corridor_min_overlap", math.inf)
    if math.isfinite(corridor):
        values.append(corridor / max(1e-9, float(limits["safe_distance_m"])))
    return min(values)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _boundary_result_cache_paths(
    root: Path, case_uid: str, profile_name: str
) -> tuple[Path, Path]:
    stem = root / "run_cache" / case_uid / profile_name
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def _load_cached_boundary_result(
    root: Path,
    case: StaticCase,
    profile_name: str,
    profile_hash: str,
) -> StaticRunResult | None:
    metadata_path, arrays_path = _boundary_result_cache_paths(
        root, case.case_uid, profile_name
    )
    if not metadata_path.is_file() or not arrays_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            payload.get("case_hash") != case.case_hash
            or payload.get("profile_name") != profile_name
            or payload.get("profile_sha256") != profile_hash
        ):
            return None
        with np.load(arrays_path, allow_pickle=False) as arrays:
            samples = np.asarray(arrays["samples"], dtype=np.float64)
            waypoints = np.asarray(arrays["waypoints"], dtype=np.float64)
        return StaticRunResult(
            case_uid=case.case_uid,
            case_hash=case.case_hash,
            mode=str(payload["mode"]),
            status=str(payload["status"]),
            engine=str(payload["engine"]),
            native_extension_path=str(payload["native_extension_path"]),
            native_extension_sha256=str(payload["native_extension_sha256"]),
            diagnostics=payload["diagnostics"],
            samples=samples,
            waypoints=waypoints,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _cache_boundary_result(
    root: Path,
    case: StaticCase,
    profile_name: str,
    profile_hash: str,
    result: StaticRunResult,
) -> None:
    metadata_path, arrays_path = _boundary_result_cache_paths(
        root, case.case_uid, profile_name
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        samples=np.asarray(result.samples, dtype=np.float64),
        waypoints=np.asarray(result.waypoints, dtype=np.float64),
    )
    metadata_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "schema_version": 1,
                    "case_hash": case.case_hash,
                    "profile_name": profile_name,
                    "profile_sha256": profile_hash,
                    "mode": result.mode,
                    "status": result.status,
                    "engine": result.engine,
                    "native_extension_path": result.native_extension_path,
                    "native_extension_sha256": result.native_extension_sha256,
                    "diagnostics": result.diagnostics,
                }
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def run_boundary_selection(
    config_path: Path | str, output_dir: Path | str, *, resume: bool = False
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    checkpoint_path = output_dir / "selection_checkpoint.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        expected_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if checkpoint_path.is_file():
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"unreadable selection checkpoint: {error}") from error
            if checkpoint.get("input_config_sha256") != expected_hash:
                raise ValueError("selection checkpoint config hash mismatch")
            if checkpoint.get("status") == "COMPLETE":
                errors = validate_boundary_selection(output_dir)
                if errors:
                    raise RuntimeError(
                        "completed boundary selection is invalid: " + "; ".join(errors)
                    )
                try:
                    frozen = json.loads(
                        (output_dir / "selected_dynamic_cases.json").read_text(
                            encoding="utf-8"
                        )
                    )
                except (OSError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"completed boundary selection is unreadable: {error}"
                    ) from error
                return frozen
        else:
            # One-time recovery for an interruption between the first cached
            # native result and checkpoint creation. Every cache entry embeds
            # profile/case hashes, so only a matching config cache is accepted.
            cached = list((output_dir / "run_cache").rglob("*.json"))
            if not resume or not cached:
                raise FileExistsError(
                    f"selection output already exists without a resumable checkpoint: {output_dir}"
                )
            for path in cached:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise ValueError(f"unreadable early boundary cache {path}: {error}") from error
                if not payload.get("case_hash") or not payload.get("profile_sha256"):
                    raise ValueError("early boundary cache is not hash-receipted")
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "RUNNING",
                        "input_config_sha256": expected_hash,
                        "selected_uids": [],
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("base_case_config",):
        value = Path(str(config[key]))
        config[key] = str(
            value if value.is_absolute() else (config_path.parent / value).resolve()
        )
    cases = generate_boundary_cases(config)
    factor_metadata = boundary_factor_metadata(config)
    profile_paths = {}
    for name, path in config["profiles"].items():
        value = Path(path)
        profile_paths[name] = (
            value if value.is_absolute() else (config_path.parent / value).resolve()
        )
    profile_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in profile_paths.items()
    }
    rows: list[dict[str, Any]] = []
    per_case: dict[
        tuple[str, str], tuple[StaticCase, Any, dict[str, Any], dict[str, Any]]
    ] = {}
    replayability = config.get("dynamic_replayability", {})
    for case in cases:
        # Formal experiments compare exactly the raw legacy optimizer against
        # the native 2-D SuperPlanner SFC implementation.  The retired
        # capsule profile is deliberately not recomputed here.
        for profile_name in ("legacy", "superplanner_sfc_v1"):
            profile_path = profile_paths[profile_name]
            profile_config = json.loads(profile_path.read_text(encoding="utf-8"))
            run_case = replace(case, constraint_profile=profile_name)
            result = _load_cached_boundary_result(
                output_dir, run_case, profile_name, profile_hashes[profile_name]
            )
            if result is None:
                result = run_static_case(
                    run_case, load_legacy_profile(profile_path), "recompute"
                )
                _cache_boundary_result(
                    output_dir,
                    run_case,
                    profile_name,
                    profile_hashes[profile_name],
                    result,
                )
            metrics, detail = compute_static_case_metrics(
                run_case, result, profile_config["metric_limits"]
            )
            reason = str(result.diagnostics.get("failure_reason", ""))
            replay = replayability.get(case.case_uid, {})
            dynamic_replayable = bool(
                replay.get(
                    "dynamic_replayable",
                    "non_replayable" not in case.tags
                    and "initial_acceleration_only" not in case.tags,
                )
            )
            row = {
                **metrics,
                "case_uid": case.case_uid,
                "case_hash": case.case_hash,
                "selection_content_hash": boundary_case_content_hash(case),
                "expected_category": case.expected_category,
                "profile": profile_name,
                "status": result.status,
                "failure_reason": reason,
                "classification": _classify(result.status, reason, metrics),
                "runtime_ms": _number(
                    result.diagnostics, "cpp_optimize_time_ms", math.nan
                ),
                "min_normalized_margin": _safe_margin(
                    metrics, result.diagnostics, profile_config["metric_limits"]
                )
                if result.status == "SUCCEEDED"
                else -math.inf,
                "violation_margin": abs(
                    _number(result.diagnostics, "validation_measured_value", 0.0)
                    - _number(result.diagnostics, "validation_limit_value", 0.0)
                ),
                "dynamic_replayable": dynamic_replayable,
                "not_replayable_reason": str(
                    replay.get(
                        "reason",
                        "INITIAL_ACCELERATION_NOT_DIRECTLY_INJECTABLE"
                        if not dynamic_replayable
                        else "",
                    )
                ),
                "required_scene": str(
                    replay.get("required_scene", "synthetic_obstacle_layout_v1")
                ),
                "materialization_method": str(
                    replay.get(
                        "materialization_method",
                        "occupancy rectangles + start/goal state injection",
                    )
                ),
                "required_start_pose": json.dumps(
                    [
                        *case.start_position[:2].tolist(),
                        float(case.start_yaw),
                    ],
                    separators=(",", ":"),
                ),
                "required_initial_velocity": json.dumps(
                    case.start_velocity.tolist(), separators=(",", ":")
                ),
                "required_initial_acceleration": json.dumps(
                    case.start_acceleration.tolist(), separators=(",", ":")
                ),
                "required_yaw_rate": float(case.start_yaw_rate),
                "required_goal": json.dumps(
                    (
                        case.terminal_goal
                        if case.terminal_goal is not None
                        else case.guide_path_xyz[-1]
                    ).tolist(),
                    separators=(",", ":"),
                ),
                "guide_path": json.dumps(
                    case.guide_path_xyz.tolist(), separators=(",", ":")
                ),
                "obstacle_layout": json.dumps(
                    np.asarray(
                        case.auxiliary_arrays.get(
                            "materialization_obstacle_rectangles_xyxy_m",
                            np.empty((0, 4)),
                        )
                    ).tolist(),
                    separators=(",", ":"),
                ),
                "config_sha256": profile_hashes[profile_name],
                "native_extension_path": result.native_extension_path,
                "native_extension_sha256": result.native_extension_sha256,
                **factor_metadata.get(
                    case.case_uid,
                    {
                        "factor_name": "geometry_category",
                        "factor_level": case.expected_category,
                        "factor_name_secondary": "",
                        "factor_level_secondary": "",
                        "scan_group": "geometry_catalogue",
                    },
                ),
            }
            for key in (
                "constraint_profile",
                "corridor_failure_reason",
                "corridor_segment_count",
                "corridor_min_radius",
                "corridor_min_clearance",
                "corridor_min_overlap",
                "sfc_generation_reason",
                "sfc_min_overlap",
                "sfc_min_margin",
                "sfc_cells",
                "adaptive_validation_sample_count",
                "adaptive_validation_subdivision_count",
                "validation_offending_sample_index",
                "validation_offending_time_s",
                "validation_measured_value",
                "validation_limit_value",
            ):
                row[key] = result.diagnostics.get(key)
            rows.append(row)
            per_case[(case.case_uid, profile_name)] = (
                run_case,
                result,
                metrics,
                detail,
            )

    policy = default_selection_policy()
    requested_profile = str(config.get("selection_profile", policy["eligible_profile"]))
    if requested_profile not in profile_paths:
        raise ValueError(f"selection_profile missing from profiles: {requested_profile}")
    policy["eligible_profile"] = requested_profile
    policy["factor_grid_uids"] = [
        str(grid["grid_uid"]) for grid in config.get("factor_grids", [])
    ]
    policy["policy_sha256"] = _canonical_hash({key: value for key, value in policy.items() if key != "policy_sha256"})
    selection = select_cases(rows, policy)
    backup_count = int(policy["backup_count_per_side"])
    frozen = {
        "schema_version": 1,
        "selection_version": datetime.now(timezone.utc).strftime(
            "selection_%Y%m%dT%H%M%SZ"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_version": policy["policy_version"],
        "policy_sha256": policy["policy_sha256"],
        "input_config": str(config_path),
        "input_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "best2": selection["best"][:2],
        "worst2": selection["worst"][:2],
        "best_backups": selection["best"][2 : 2 + backup_count],
        "worst_backups": selection["worst"][2 : 2 + backup_count],
        "eligible_case_uids": selection["eligible_case_uids"],
        "best_ranking": selection["best_ranking"],
        "worst_ranking": selection["worst_ranking"],
        "hot_start_evidence": "PENDING_DYNAMIC_VALIDATION",
        "exclusions": selection["exclusions"],
        "cases": {},
    }
    selected_uids = (
        frozen["best2"]
        + frozen["worst2"]
        + frozen["best_backups"]
        + frozen["worst_backups"]
    )
    row_lookup = {
        (str(row["case_uid"]), str(row["profile"])): row for row in rows
    }
    case_lookup = {case.case_uid: case for case in cases}
    for uid in selected_uids:
        case = case_lookup[uid]
        frozen["cases"][uid] = {
            "case_hash": case.case_hash,
            "selection_content_hash": boundary_case_content_hash(case),
            "expected_category": case.expected_category,
            "legacy_static": row_lookup[(uid, "legacy")],
            "superplanner_sfc_v1_static": row_lookup[
                (uid, "superplanner_sfc_v1")
            ],
            "materialization": {
                key: row_lookup[(uid, "superplanner_sfc_v1")][key]
                for key in (
                    "dynamic_replayable",
                    "required_scene",
                    "required_start_pose",
                    "required_initial_velocity",
                    "required_initial_acceleration",
                    "required_yaw_rate",
                    "required_goal",
                    "guide_path",
                    "obstacle_layout",
                    "materialization_method",
                )
            },
        }
    # Persist the frozen numerical study before any expensive rendering.  A
    # resumed run recomputes deterministic native plans, but never changes the
    # selected identities or overwrites a completed visual package.
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "RENDERING",
                "input_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "selected_uids": selected_uids,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    _render_selected_artifacts(
        frozen["best2"] + frozen["worst2"], per_case, output_dir, frozen
    )
    corridor_showcase = {
        "schema_version": 1,
        "purpose_zh": (
            "走廊硬约束实证：前两例为完成的约束正例，最后一例为 "
            "SuperPlanner 2-D SFC 拒绝不满足凸单元约束轨迹的 fail-closed 反例。"
        ),
        "case_uids": list(CORRIDOR_SHOWCASE_CASES),
        "profiles": ["legacy", "superplanner_sfc_v1"],
        "cases": {uid: {} for uid in CORRIDOR_SHOWCASE_CASES},
    }
    _render_selected_artifacts(
        list(CORRIDOR_SHOWCASE_CASES),
        per_case,
        output_dir,
        corridor_showcase,
        artifact_directory="corridor_showcase",
    )
    _render_full_route_showcase(output_dir, case_lookup, profile_paths)
    (output_dir / "corridor_showcase.json").write_text(
        json.dumps(corridor_showcase, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Render factor grid comparison GIFs for each scan group
    _render_factor_grid_gifs(config, per_case, output_dir)

    (output_dir / "selection_policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "selected_dynamic_cases.json").write_text(
        json.dumps(_json_safe(frozen), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(output_dir / "static_runs.csv", rows)
    _write_csv(
        output_dir / "case_selection.csv",
        [
            {
                "case_uid": uid,
                "selection_group": group,
                "selection_rank": rank + 1,
            }
            for group, values in (
                ("BEST", frozen["best2"] + frozen["best_backups"]),
                ("WORST", frozen["worst2"] + frozen["worst_backups"]),
            )
            for rank, uid in enumerate(values)
        ],
    )
    best_by_uid = {
        str(row["case_uid"]): row for row in frozen["best_ranking"]
    }
    worst_by_uid = {
        str(row["case_uid"]): row for row in frozen["worst_ranking"]
    }
    _write_csv(
        output_dir / "complete_case_rankings.csv",
        [
            {
                "case_uid": uid,
                "best_rank": best_by_uid[uid]["rank"],
                "worst_rank": worst_by_uid[uid]["rank"],
                "classification": best_by_uid[uid]["classification"],
                "failure_reason": best_by_uid[uid]["failure_reason"],
                "factor_name": best_by_uid[uid]["factor_name"],
                "factor_level": best_by_uid[uid]["factor_level"],
                "factor_name_secondary": best_by_uid[uid][
                    "factor_name_secondary"
                ],
                "factor_level_secondary": best_by_uid[uid][
                    "factor_level_secondary"
                ],
            }
            for uid in sorted(best_by_uid)
        ],
    )
    report = [
        "# Static capability boundary and frozen dynamic selection",
        "",
        f"- Policy: `{policy['policy_version']}` / `{policy['policy_sha256']}`",
        f"- Paired cases: {len(cases)} ({len(rows)} profile runs)",
        f"- Best2: {', '.join(frozen['best2'])}",
        f"- Worst2: {', '.join(frozen['worst2'])}",
        f"- Best backups: {', '.join(frozen['best_backups'])}",
        f"- Worst backups: {', '.join(frozen['worst_backups'])}",
        "",
        "The grid is a controlled capability scan; rows are not treated as",
        "independent population samples. Failures remain in all denominators.",
        "No dynamic simulator was started.",
        "Static hot-start history is not claimed; hot-start evidence is "
        "`PENDING_DYNAMIC_VALIDATION`.",
    ]
    (output_dir / "selection_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    from experiments.analyzers.static_comparison import (
        generate_static_paper_outputs,
    )

    generate_static_paper_outputs(output_dir, output_dir / "paper")
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "input_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "selected_uids": selected_uids,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    selection_receipt_path = output_dir / "artifact_receipt.json"
    selection_receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "root": ".",
                "artifacts": inventory_receipts(
                    output_dir, exclude=(selection_receipt_path,)
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return frozen


def validate_boundary_selection(output_dir: Path | str) -> list[str]:
    """Fail closed unless a complete, hash-receipted boundary study exists."""
    from experiments.core.artifact_receipt import validate_file_receipt

    root = Path(output_dir).resolve()
    errors: list[str] = []
    required = (
        "selection_policy.json",
        "selected_dynamic_cases.json",
        "static_runs.csv",
        "case_selection.csv",
        "complete_case_rankings.csv",
        "selection_report.md",
        "corridor_showcase.json",
        "full_route_showcase/full_route_index.json",
        "artifact_receipt.json",
    )
    for relative in required:
        if not (root / relative).is_file():
            errors.append(f"missing boundary artifact: {relative}")
    if errors:
        return errors
    try:
        selected = json.loads((root / "selected_dynamic_cases.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable selected_dynamic_cases.json: {error}"]
    selected_uids = list(selected.get("best2", [])) + list(selected.get("worst2", []))
    if len(selected_uids) != 4 or len(set(selected_uids)) != 4:
        errors.append("boundary selection does not contain four unique Best2/Worst2 cases")
    try:
        with (root / "static_runs.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as error:
        errors.append(f"unreadable static_runs.csv: {error}")
        rows = []
    profiles = {row.get("profile", "") for row in rows}
    if profiles != {"legacy", "superplanner_sfc_v1"}:
        errors.append(f"boundary profiles are incomplete or unexpected: {sorted(profiles)}")
    for grid in json.loads((root / "selection_policy.json").read_text(encoding="utf-8")).get("factor_grid_uids", []):
        if not (root / "factor_grids" / str(grid) / f"{grid}_comparison.gif").is_file():
            errors.append(f"missing factor-grid GIF: {grid}")
    for uid in selected_uids:
        paired = root / "selected_artifacts" / uid / f"{uid}_legacy_vs_superplanner_sfc.gif"
        card = root / "selected_artifacts" / uid / f"{uid}_paired_card.png"
        if not paired.is_file():
            errors.append(f"missing selected paired GIF: {uid}")
        if not card.is_file():
            errors.append(f"missing selected comparison card: {uid}")
    try:
        full_route = json.loads(
            (root / "full_route_showcase" / "full_route_index.json").read_text(
                encoding="utf-8"
            )
        )
        if tuple(full_route.get("case_uids", ())) != FULL_ROUTE_SHOWCASE_CASES:
            errors.append("full-route showcase case set is not the frozen evidence matrix")
        for uid in FULL_ROUTE_SHOWCASE_CASES:
            visual = root / "full_route_showcase" / uid / "visual"
            if not (visual / "three_way.gif").is_file():
                errors.append(f"missing full-route three-way GIF: {uid}")
            if not (visual / "superplanner_sfc.pdf").is_file():
                errors.append(f"missing full-route SFC PDF: {uid}")
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"unreadable full-route showcase: {error}")
    try:
        corridor = json.loads((root / "corridor_showcase.json").read_text(encoding="utf-8"))
        if tuple(corridor.get("case_uids", ())) != CORRIDOR_SHOWCASE_CASES:
            errors.append("corridor showcase case set is not the frozen hard-constraint evidence matrix")
        for uid in CORRIDOR_SHOWCASE_CASES:
            package = root / "corridor_showcase" / uid
            if not (package / f"{uid}_legacy_vs_superplanner_sfc.gif").is_file():
                errors.append(f"missing corridor showcase paired GIF: {uid}")
            if not (package / f"{uid}_paired_card.png").is_file():
                errors.append(f"missing corridor showcase comparison card: {uid}")
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"unreadable corridor showcase: {error}")
    try:
        receipt = json.loads((root / "artifact_receipt.json").read_text(encoding="utf-8"))
        errors.extend(
            error
            for row in receipt.get("artifacts", [])
            for error in validate_file_receipt(root, row)
        )
    except (OSError, json.JSONDecodeError, TypeError) as error:
        errors.append(f"invalid boundary artifact receipt: {error}")
    return errors


def _render_factor_grid_gifs(
    config: Mapping[str, Any],
    per_case: Mapping[
        tuple[str, str], tuple[StaticCase, Any, dict[str, Any], dict[str, Any]]
    ],
    output_dir: Path,
) -> None:
    """Render factor-grid comparison GIFs for each factor_grid in the config."""
    from experiments.visualizers.static_benchmark import render_factor_grid_gif

    for grid in config.get("factor_grids", []):
        grid_uid = str(grid["grid_uid"])
        x_factor = grid["x_factor"]
        y_factor = grid["y_factor"]
        x_levels = list(x_factor["levels"])
        y_levels = list(y_factor["levels"])

        grid_cells: list[list[dict[str, object]]] = []
        for yi, y_level in enumerate(y_levels):
            row_cells: list[dict[str, object]] = []
            for xi, x_level in enumerate(x_levels):
                case_uid = f"{grid_uid}_x{xi:02d}_y{yi:02d}"
                profile = "superplanner_sfc_v1"
                key = (case_uid, profile)
                if key not in per_case:
                    profile = "legacy"
                    key = (case_uid, profile)
                if key not in per_case:
                    raise ValueError(
                        f"factor grid case not found: {case_uid}"
                    )
                case, result, metrics, detail = per_case[key]
                row_cells.append({
                    "case": case,
                    "result": result,
                    "detail": detail,
                })
            grid_cells.append(row_cells)

        grid_dir = output_dir / "factor_grids" / grid_uid
        grid_dir.mkdir(parents=True, exist_ok=True)
        gif_path = grid_dir / f"{grid_uid}_comparison.gif"

        if not gif_path.is_file():
            render_factor_grid_gif(
                gif_path,
                grid_cells=grid_cells,
                row_factor={
                    "name": str(y_factor["name"]),
                    "levels": y_levels,
                    "label": str(y_factor["name"]),
                },
                col_factor={
                    "name": str(x_factor["name"]),
                    "levels": x_levels,
                    "label": str(x_factor["name"]),
                },
            )


def _static_rectangles_for_rollout(case: StaticCase) -> tuple[object, ...]:
    """Materialize only recorded static rectangle geometry for a rollout."""
    from experiments.rolling.scenarios import StaticRectangle

    raw = np.asarray(
        case.auxiliary_arrays.get(
            "materialization_obstacle_rectangles_xyxy_m", np.empty((0, 4))
        ),
        dtype=np.float64,
    )
    if raw.ndim != 2 or raw.shape[1] != 4:
        return ()
    return tuple(
        StaticRectangle(f"static-{index}", row)
        for index, row in enumerate(raw)
        if row[2] > row[0] and row[3] > row[1]
    )


def _render_full_route_showcase(
    output_dir: Path,
    case_lookup: Mapping[str, StaticCase],
    profile_paths: Mapping[str, Path],
) -> None:
    """Produce hash-validated full-route paired evidence for key static cases."""
    from experiments.rolling.engine import run_rollout
    from experiments.rolling.models import RobotState, RolloutConfig
    from experiments.rolling.scenarios import RollingScenario
    from experiments.rolling.serialization import load_rollout_result, write_rollout_result
    from experiments.visualizers.rolling_showcase import (
        render_scene_package,
        validate_scene_package,
    )

    root = output_dir / "full_route_showcase"
    root.mkdir(parents=True, exist_ok=True)
    profiles = {
        name: load_legacy_profile(profile_paths[name])
        for name in ("legacy", "superplanner_sfc_v1")
    }
    # A fixed local horizon is part of the evidence protocol, not a tuned
    # per-case parameter.  It makes all full-route comparisons comparable.
    rollout_config = RolloutConfig(
        planning_period_s=0.5,
        execute_duration_s=0.5,
        local_horizon_m=2.0,
        max_cycles=120,
        max_time_s=60.0,
        goal_tolerance_m=0.1,
    )
    index: list[dict[str, object]] = []
    for uid in FULL_ROUTE_SHOWCASE_CASES:
        case = case_lookup[uid]
        scene_root = root / uid
        evidence_root = scene_root / "rollout_evidence"
        reusable = (
            (scene_root / "visual").is_dir()
            and not validate_scene_package(scene_root / "visual")
            and all(
                (evidence_root / method / "run_manifest.json").is_file()
                for method in ("legacy", "superplanner_sfc_v1")
            )
        )
        if reusable:
            results = {
                method: load_rollout_result(
                    evidence_root / method / "run_manifest.json"
                )
                for method in ("legacy", "superplanner_sfc_v1")
            }
        else:
            if scene_root.exists():
                shutil.rmtree(scene_root)
        xmin, ymin = case.esdf_origin
        height, width = case.occupancy.shape
        scenario = RollingScenario(
            scenario_uid=uid,
            family="static_sparse",
            world_bounds_xy=(
                float(xmin), float(ymin),
                float(xmin + width * case.esdf_resolution),
                float(ymin + height * case.esdf_resolution),
            ),
            resolution_m=case.esdf_resolution,
            guide_path_xyz=case.guide_path_xyz,
            final_goal_xyz=(
                case.terminal_goal
                if case.terminal_goal is not None
                else case.guide_path_xyz[-1]
            ),
            initial_state=RobotState(
                case.start_position,
                case.start_velocity,
                case.start_acceleration,
                case.start_yaw,
                case.start_yaw_rate,
            ),
            static_rectangles=_static_rectangles_for_rollout(case),
            max_experiment_time_s=60.0,
        )
        if not reusable:
            results = {
                method: run_rollout(
                    scenario,
                    method=method,
                    profile=profiles[method],
                    config=rollout_config,
                    reset_history_each_cycle=True,
                )
                for method in ("legacy", "superplanner_sfc_v1")
            }
            for method, result in results.items():
                write_rollout_result(result, evidence_root / method)
            render_scene_package(
                {
                    "scenario_uid": uid,
                    "guide_path_xyz": scenario.guide_path_xyz,
                    "final_goal_xyz": scenario.final_goal_xyz,
                    "goal_yaw_rad": None,
                    "initial_state": scenario.initial_state,
                    "static_rectangles": scenario.static_rectangles,
                    **results,
                },
                scene_root / "visual",
            )
        index.append(
            {
                "case_uid": uid,
                "case_hash": case.case_hash,
                "protocol": "fixed-horizon cold-replanned full-route v1",
                "legacy_status": results["legacy"].status,
                "legacy_final_error_m": results["legacy"].metrics["final_error_m"],
                "legacy_cycles": results["legacy"].metrics["cycle_count"],
                "sfc_status": results["superplanner_sfc_v1"].status,
                "sfc_final_error_m": results["superplanner_sfc_v1"].metrics["final_error_m"],
                "sfc_cycles": results["superplanner_sfc_v1"].metrics["cycle_count"],
                "visual_package": f"{uid}/visual",
            }
        )
    (root / "full_route_index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose_zh": "从起点到终点的真实逐周期冷启动重规划对比；每周期重置原生历史，避免墙钟热启动年龄影响可复现性；失败保留为硬约束或优化拒绝证据。",
                "case_uids": list(FULL_ROUTE_SHOWCASE_CASES),
                "rows": index,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def _render_selected_artifacts(
    selected_uids: list[str],
    per_case: Mapping[
        tuple[str, str], tuple[StaticCase, Any, dict[str, Any], dict[str, Any]]
    ],
    output_dir: Path,
    frozen: dict[str, Any],
    *,
    artifact_directory: str = "selected_artifacts",
) -> None:
    from PIL import Image, ImageOps, ImageDraw

    root = output_dir / artifact_directory
    for uid in selected_uids:
        uid_root = root / uid
        paired_existing = uid_root / f"{uid}_legacy_vs_superplanner_sfc.gif"
        card_existing = uid_root / f"{uid}_paired_card.png"
        evidence_existing = paired_existing.with_name(f"{paired_existing.stem}_evidence")
        profile_outputs = (
            "overview.png", "overview.pdf", "clearance.png", "clearance.pdf",
            "dynamics.png", "dynamics.pdf", "animation.gif", "metrics.json",
        )
        complete_profiles = all(
            all((uid_root / profile / f"{uid}_{suffix}").is_file() for suffix in profile_outputs)
            for profile in ("legacy", "superplanner_sfc_v1")
        )
        complete_paired = (
            paired_existing.is_file()
            and card_existing.is_file()
            and evidence_existing.is_dir()
            and all(
                (evidence_existing / name).is_file()
                for name in ("frame_metrics.csv", "event_timeline.csv", "caption_zh.md", "validation.json")
            )
        )
        if complete_profiles and complete_paired:
            frozen["cases"][uid]["artifact_paths"] = [
                str(path.relative_to(output_dir))
                for path in sorted(uid_root.rglob("*"))
                if path.is_file()
            ]
            continue
        if uid_root.exists():
            shutil.rmtree(uid_root)
        profile_dirs: dict[str, Path] = {}
        artifacts: list[str] = []
        trajectory_rows: list[dict[str, Any]] = []
        constrained_profile = "superplanner_sfc_v1"
        for profile in ("legacy", constrained_profile):
            case, result, metrics, detail = per_case[(uid, profile)]
            profile_dir = root / uid / profile
            profile_dirs[profile] = profile_dir
            paths = render_static_case(
                case, result, metrics, detail, profile_dir
            )
            artifacts.extend(
                str(path.relative_to(output_dir)) for path in paths
            )
            trajectory = (
                result.samples[:, 1:4]
                if len(result.samples)
                else np.empty((0, 3), dtype=np.float64)
            )
            trajectory_rows.extend(
                {
                    "case_uid": uid,
                    "profile": profile,
                    "series": "trajectory",
                    "sample_index": index,
                    "x_m": float(point[0]),
                    "y_m": float(point[1]),
                }
                for index, point in enumerate(trajectory)
            )

        guide = per_case[(uid, "legacy")][0].guide_path_xyz
        trajectory_rows.extend(
            {
                "case_uid": uid,
                "profile": "guide",
                "series": "guide",
                "sample_index": index,
                "x_m": float(point[0]),
                "y_m": float(point[1]),
            }
            for index, point in enumerate(guide)
        )
        trajectory_path = root / uid / "trajectory_samples.csv"
        _write_csv(trajectory_path, trajectory_rows)
        artifacts.append(str(trajectory_path.relative_to(output_dir)))

        legacy_overview = next(profile_dirs["legacy"].glob("*_overview.png"))
        new_overview = next(
            profile_dirs[constrained_profile].glob("*_overview.png")
        )
        images = [Image.open(path).convert("RGB") for path in (legacy_overview, new_overview)]
        target_height = max(image.height for image in images)
        images = [
            ImageOps.pad(image, (image.width, target_height), color="white")
            for image in images
        ]
        card = Image.new(
            "RGB", (sum(image.width for image in images), target_height + 36), "white"
        )
        x = 0
        for label, image in zip(("legacy", constrained_profile), images):
            card.paste(image, (x, 36))
            ImageDraw.Draw(card).text((x + 12, 10), label, fill="black")
            x += image.width
        card_path = root / uid / f"{uid}_paired_card.png"
        card.save(card_path)
        artifacts.append(str(card_path.relative_to(output_dir)))

        legacy_case, legacy_result, _, legacy_detail = per_case[(uid, "legacy")]
        _, safe_result, _, safe_detail = per_case[(uid, constrained_profile)]

        suffix = "superplanner_sfc" if constrained_profile == "superplanner_sfc_v1" else "safe"
        paired_gif = root / uid / f"{uid}_legacy_vs_{suffix}.gif"
        from experiments.visualizers.static_benchmark import render_paired_static_gif
        render_paired_static_gif(
            paired_gif,
            case=legacy_case,
            legacy_result=legacy_result,
            legacy_detail=legacy_detail,
            safe_result=safe_result,
            safe_detail=safe_detail,
        )
        artifacts.append(str(paired_gif.relative_to(output_dir)))

        # render_paired_static_gif builds the evidence package internally;
        # register its files as artifacts for receipting.
        paired_evidence_dir = paired_gif.with_name(f"{paired_gif.stem}_evidence")
        artifacts.extend(
            str(path.relative_to(output_dir))
            for path in sorted(paired_evidence_dir.iterdir())
        )
        frozen["cases"][uid]["artifact_paths"] = artifacts
