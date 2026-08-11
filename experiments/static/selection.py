from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from experiments.static.case_schema import StaticCase
from experiments.static.metrics import compute_static_case_metrics
from experiments.static.runner import load_legacy_profile, run_static_case
from experiments.static.synthetic import generate_catalogue
from experiments.visualizers.static_benchmark import render_static_case


POLICY_VERSION = "best-worst-lexicographic-v1"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def default_selection_policy() -> dict[str, Any]:
    policy = {
        "policy_version": POLICY_VERSION,
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
            "CORRIDOR_GUIDE_NEGATIVE_ESDF",
            "CORRIDOR_GUIDE_UNSAFE",
            "CORRIDOR_DISCONNECTED",
            "VALIDATION_BUDGET_EXHAUSTED",
            "OPTIMIZER_FAILED",
        ],
        "backup_count_per_side": 2,
        "duplicate_rule": "retain lexicographically first case_uid per case_hash",
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
        case_hash = str(row.get("case_hash", ""))
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

    return {
        "policy_version": policy["policy_version"],
        "policy_sha256": policy["policy_sha256"],
        "eligible_case_uids": [str(row["case_uid"]) for row in eligible],
        "exclusions": exclusions,
        "best": best,
        "worst": worst,
    }


def _state_variant(case: StaticCase, specification: Mapping[str, Any]) -> StaticCase:
    velocity = np.asarray(
        specification.get("velocity_xyz_mps", case.start_velocity),
        dtype=np.float64,
    )
    acceleration = np.asarray(
        specification.get("acceleration_xyz_mps2", case.start_acceleration),
        dtype=np.float64,
    )
    return replace(
        case,
        case_uid=str(specification["case_uid"]),
        start_position=np.asarray(
            specification.get("position_xyz_m", case.start_position),
            dtype=np.float64,
        ),
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
    for specification in config.get("state_variants", []):
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


def run_boundary_selection(
    config_path: Path | str, output_dir: Path | str
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"selection output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("base_case_config",):
        value = Path(str(config[key]))
        config[key] = str(
            value if value.is_absolute() else (config_path.parent / value).resolve()
        )
    cases = generate_boundary_cases(config)
    profile_paths = {}
    for name, path in config["profiles"].items():
        value = Path(path)
        profile_paths[name] = (
            value if value.is_absolute() else (config_path.parent / value).resolve()
        )
    rows: list[dict[str, Any]] = []
    per_case: dict[
        tuple[str, str], tuple[StaticCase, Any, dict[str, Any], dict[str, Any]]
    ] = {}
    replayability = config.get("dynamic_replayability", {})
    for case in cases:
        for profile_name in ("legacy", "safe_corridor_v1"):
            profile_path = profile_paths[profile_name]
            profile_config = json.loads(profile_path.read_text(encoding="utf-8"))
            run_case = replace(case, constraint_profile=profile_name)
            result = run_static_case(
                run_case, load_legacy_profile(profile_path), "recompute"
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
                "config_sha256": hashlib.sha256(
                    profile_path.read_bytes()
                ).hexdigest(),
                "native_extension_path": result.native_extension_path,
                "native_extension_sha256": result.native_extension_sha256,
            }
            for key in (
                "constraint_profile",
                "corridor_failure_reason",
                "corridor_segment_count",
                "corridor_min_radius",
                "corridor_min_clearance",
                "corridor_min_overlap",
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
            "expected_category": case.expected_category,
            "legacy_static": row_lookup[(uid, "legacy")],
            "safe_corridor_v1_static": row_lookup[
                (uid, "safe_corridor_v1")
            ],
            "materialization": {
                key: row_lookup[(uid, "safe_corridor_v1")][key]
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
    _render_selected_artifacts(
        frozen["best2"] + frozen["worst2"], per_case, output_dir, frozen
    )

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
    _render_selection_figures(rows, frozen, output_dir)
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
    ]
    (output_dir / "selection_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return frozen


def _render_selection_figures(
    rows: list[dict[str, Any]], frozen: Mapping[str, Any], output_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    new_rows = [row for row in rows if row["profile"] == "safe_corridor_v1"]
    categories = sorted({str(row["expected_category"]) for row in new_rows})
    classes = sorted({str(row["classification"]) for row in new_rows})
    matrix = np.zeros((len(categories), len(classes)))
    for row in new_rows:
        matrix[categories.index(str(row["expected_category"])), classes.index(str(row["classification"]))] += 1
    fig, ax = plt.subplots(figsize=(max(7, len(classes) * 1.4), max(5, len(categories) * 0.35)))
    image = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(categories)), categories)
    ax.set_title(f"safe_corridor_v1 capability boundary (n={len(new_rows)})")
    fig.colorbar(image, ax=ax, label="case count")
    fig.tight_layout()
    fig.savefig(figure_dir / "capability_heatmap.png", dpi=160)
    plt.close(fig)

    reasons = sorted({str(row["failure_reason"]) for row in rows})
    profiles = ["legacy", "safe_corridor_v1"]
    counts = np.asarray(
        [[sum(row["profile"] == p and row["failure_reason"] == r for row in rows) for r in reasons] for p in profiles]
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(profiles))
    for index, reason in enumerate(reasons):
        ax.bar(profiles, counts[:, index], bottom=bottom, label=reason)
        bottom += counts[:, index]
    ax.set_ylabel("case count")
    ax.set_title(f"Failure reasons by profile (paired n={len(new_rows)})")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(figure_dir / "failure_reason_stack.png", dpi=160)
    plt.close(fig)

    successful = [row for row in rows if row["status"] == "SUCCEEDED"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for profile in profiles:
        subset = [row for row in successful if row["profile"] == profile]
        ax.scatter(
            [_number(row, "runtime_ms", math.nan) for row in subset],
            [_number(row, "guide_deviation_p95_m", math.nan) for row in subset],
            label=f"{profile} (n={len(subset)})",
            alpha=0.75,
        )
    ax.set_xlabel("native runtime (ms)")
    ax.set_ylabel("guide deviation p95 (m)")
    ax.set_title("Runtime–shape Pareto view")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "runtime_shape_pareto.png", dpi=160)
    plt.close(fig)

    lookup = {(row["case_uid"], row["profile"]): row for row in rows}
    transitions: dict[tuple[str, str], int] = {}
    for uid in sorted({row["case_uid"] for row in rows}):
        key = (
            str(lookup[(uid, "legacy")]["classification"]),
            str(lookup[(uid, "safe_corridor_v1")]["classification"]),
        )
        transitions[key] = transitions.get(key, 0) + 1
    transition_rows = [
        {"legacy": left, "safe_corridor_v1": right, "case_count": count}
        for (left, right), count in sorted(transitions.items())
    ]
    _write_csv(output_dir / "legacy_to_new_transitions.csv", transition_rows)


def _render_selected_artifacts(
    selected_uids: list[str],
    per_case: Mapping[
        tuple[str, str], tuple[StaticCase, Any, dict[str, Any], dict[str, Any]]
    ],
    output_dir: Path,
    frozen: dict[str, Any],
) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageOps, ImageDraw

    root = output_dir / "selected_artifacts"
    for uid in selected_uids:
        profile_dirs: dict[str, Path] = {}
        artifacts: list[str] = []
        for profile in ("legacy", "safe_corridor_v1"):
            case, result, metrics, detail = per_case[(uid, profile)]
            profile_dir = root / uid / profile
            profile_dirs[profile] = profile_dir
            paths = render_static_case(
                case, result, metrics, detail, profile_dir
            )
            artifacts.extend(
                str(path.relative_to(output_dir)) for path in paths
            )

        legacy_overview = next(profile_dirs["legacy"].glob("*_overview.png"))
        new_overview = next(
            profile_dirs["safe_corridor_v1"].glob("*_overview.png")
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
        for label, image in zip(("legacy", "safe_corridor_v1"), images):
            card.paste(image, (x, 36))
            ImageDraw.Draw(card).text((x + 12, 10), label, fill="black")
            x += image.width
        card_path = root / uid / f"{uid}_paired_card.png"
        card.save(card_path)
        artifacts.append(str(card_path.relative_to(output_dir)))

        gifs = [
            next(profile_dirs[profile].glob("*_animation.gif"))
            for profile in ("legacy", "safe_corridor_v1")
        ]
        frames = [imageio.mimread(path) for path in gifs]
        frame_count = max(len(sequence) for sequence in frames)
        paired_frames = []
        for index in range(frame_count):
            left = np.asarray(frames[0][min(index, len(frames[0]) - 1)])
            right = np.asarray(frames[1][min(index, len(frames[1]) - 1)])
            height = max(left.shape[0], right.shape[0])
            if left.shape[0] != height:
                left = np.pad(
                    left, ((0, height - left.shape[0]), (0, 0), (0, 0)),
                    constant_values=255,
                )
            if right.shape[0] != height:
                right = np.pad(
                    right, ((0, height - right.shape[0]), (0, 0), (0, 0)),
                    constant_values=255,
                )
            paired_frames.append(np.concatenate((left, right), axis=1))
        paired_gif = root / uid / f"{uid}_legacy_vs_safe.gif"
        imageio.mimsave(paired_gif, paired_frames, duration=0.1, loop=0)
        artifacts.append(str(paired_gif.relative_to(output_dir)))
        frozen["cases"][uid]["artifact_paths"] = artifacts
