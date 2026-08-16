from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any, Mapping

import numpy as np

from experiments.calibration.profile import load_robot_calibration
from experiments.orchestrators.suite_runner import run_suite
from experiments.visualizers.video_evidence import build_video_evidence_package
from experiments.visualizers.paired_video import (
    VideoSource,
    render_paired_episode_video,
    validate_paired_video_bundle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_pending_video_evidence(
    output_dir: Path | str, expected_runs: list[Mapping[str, object]]
) -> list[str]:
    """Materialize honest header-only evidence for videos not collected yet."""
    root = Path(output_dir).resolve() / "pending_video_evidence"
    packages = []
    for row in expected_runs:
        case_uid = str(row["case_uid"])
        episode_uid = str(row["episode_uid"])
        profile = str(row["profile"])
        package = root / case_uid / profile
        build_video_evidence_package(
            None,
            package,
            evidence_uid=f"pending-{case_uid}-{profile}-video-evidence",
            media_uid=f"pending-{episode_uid}-{profile}-video",
            data_source="UNAVAILABLE",
            status="PENDING_REAL_SIMULATION",
            case_uid=case_uid,
            episode_uid=episode_uid,
            caption_overrides={
                "scene_zh": f"动态候选案例 {case_uid}",
                "method_zh": profile,
                "conclusion_limit_zh": (
                    "必须完成真实 Isaac/NavDP 采集并通过媒体与控制数据校验后，"
                    "才能报告任何动态定量结论"
                ),
            },
        )
        packages.append(str(package))
    return packages


def _terminal_state(run_dir: Path, episode_uid: str) -> tuple[str, float | None]:
    """Read a recorded terminal state when available; never infer one."""
    for candidate in (run_dir / "episodes.csv", run_dir / "episode_summary.csv"):
        if not candidate.is_file():
            continue
        with candidate.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("episode_uid") != episode_uid:
                    continue
                status = str(
                    row.get("termination_state", row.get("status", "UNKNOWN"))
                ).strip() or "UNKNOWN"
                for key in ("terminal_time_s", "duration_s", "elapsed_s"):
                    try:
                        value = float(row.get(key, ""))
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value) and value >= 0.0:
                        return status, value
                return status, None
    return "UNKNOWN", None


def _find_dynamic_run_dir(
    suite_dir: Path, experiment_id: str
) -> Path:
    matches = []
    for config_path in suite_dir.rglob("run_config.json"):
        try:
            payload = _load_json(config_path)
        except ValueError:
            continue
        if payload.get("experiment_id") == experiment_id:
            matches.append(config_path.parent)
    if len(matches) != 1:
        raise ValueError(
            f"{experiment_id}: expected exactly one recorded run directory, found {len(matches)}"
        )
    return matches[0]


def build_dynamic_comparison_videos(
    dynamic_output: Path | str,
    *,
    output_dir: Path | str | None = None,
) -> dict[str, object]:
    """Create real three-way videos from completed Isaac run artifacts only.

    This intentionally refuses dry-run, missing-video, or missing-control-data
    inputs.  It does not draw a surrogate NavDP/MINCO/SFC comparison.
    """
    dynamic_output = Path(dynamic_output).resolve()
    readiness = _load_json(dynamic_output / "dynamic_readiness_receipt.json")
    suite = _load_json(dynamic_output / "dynamic_suite.json")
    suite_dir = Path(suite["output_root"]).resolve() / str(suite["suite_id"])
    suite_status = _load_json(suite_dir / "suite_status.json")
    if suite_status.get("status") != "COMPLETE" or suite_status.get("data_source") != "REAL":
        raise RuntimeError("dynamic comparison requires a completed REAL Isaac suite")
    target = (
        Path(output_dir).resolve()
        if output_dir is not None
        else dynamic_output / "comparison_videos"
    )
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"dynamic comparison output already exists: {target}")
    target.mkdir(parents=True, exist_ok=True)
    methods = (
        ("navdp_native", "navdp_native"),
        ("legacy", "legacy"),
        ("superplanner_sfc_v1", "superplanner_sfc_v1"),
    )
    outputs: list[dict[str, object]] = []
    for case_uid in readiness.get("case_uids", []):
        episode_uid = f"dynamic_{case_uid}"
        sources: dict[str, VideoSource] = {}
        for panel, suffix in methods:
            run_dir = _find_dynamic_run_dir(suite_dir, f"DYNAMIC-{case_uid}-{suffix}")
            config = _load_json(run_dir / "run_config.json")
            if panel == "navdp_native" and config.get("variant") != "raw":
                raise ValueError(f"{case_uid}: native NavDP source is not raw")
            if panel != "navdp_native":
                profile = (
                    config.get("effective_parameters", {})
                    .get("minco", {})
                    .get("constraint_profile")
                )
                if profile != panel:
                    raise ValueError(f"{case_uid}: {panel} profile mismatch: {profile}")
            video = run_dir / "videos" / f"{episode_uid}.mp4"
            video_receipt = run_dir / "videos" / f"{episode_uid}.video_complete.json"
            controls = run_dir / "control_samples.csv"
            if not video.is_file() or not video_receipt.is_file() or not controls.is_file():
                raise FileNotFoundError(
                    f"{case_uid}/{panel}: require recorded MP4, receipt, and control_samples.csv"
                )
            terminal_status, terminal_time = _terminal_state(run_dir, episode_uid)
            sources[panel] = VideoSource(
                variant=panel,
                path=video,
                receipt_path=video_receipt,
                control_samples_path=controls,
                control_episode_uid=episode_uid,
                terminal_status=terminal_status,
                terminal_time_s=terminal_time,
            )
        output_video = target / f"{case_uid}_navdp_vs_minco_vs_superplanner_sfc.mp4"
        comparison = render_paired_episode_video(
            sources,
            output_video,
            episode_uid=episode_uid,
            data_source="REAL_VALIDATED",
        )
        errors = validate_paired_video_bundle(comparison.output_path)
        if errors:
            raise RuntimeError("paired-video validation failed: " + "; ".join(errors))
        outputs.append(
            {
                "case_uid": case_uid,
                "episode_uid": episode_uid,
                "video": str(comparison.output_path),
                "evidence_package": str(comparison.evidence_package_path),
                "panel_order": list(comparison.panel_order),
            }
        )
    receipt = {
        "schema_version": 1,
        "status": "COMPLETE",
        "data_source": "REAL_VALIDATED",
        "methods": [item[0] for item in methods],
        "suite_dir": str(suite_dir),
        "outputs": outputs,
    }
    (target / "comparison_video_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def _decode_vector(value: Any, name: str, length: int = 3) -> list[float]:
    decoded = json.loads(value) if isinstance(value, str) else value
    array = np.asarray(decoded, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values")
    return array.tolist()


def _decode_matrix(value: Any, name: str, columns: int) -> list[list[float]]:
    decoded = json.loads(value) if isinstance(value, str) else value
    array = np.asarray(decoded, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, columns), dtype=np.float64).tolist()
    if array.ndim != 2 or array.shape[1] != columns or not np.all(
        np.isfinite(array)
    ):
        raise ValueError(f"{name} must have shape (N,{columns})")
    return array.tolist()


def _ensure_materializable_acceleration(
    acceleration: list[float], case_uid: str
) -> None:
    if np.linalg.norm(np.asarray(acceleration, dtype=np.float64)) > 1e-9:
        raise ValueError(
            f"{case_uid}: nonzero initial acceleration cannot be faithfully materialized"
        )


def _case_uid(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("case_uid", value.get("uid", ""))
    return str(value)


def _materialization_failure(case_payload: Mapping[str, Any]) -> str | None:
    materialization = case_payload.get("materialization", {})
    if not materialization.get("dynamic_replayable", False):
        return "DYNAMIC_REPLAY_DISABLED"
    try:
        acceleration = _decode_vector(
            materialization.get("required_initial_acceleration", [0.0, 0.0, 0.0]),
            "required_initial_acceleration",
        )
    except (TypeError, ValueError):
        return "INITIAL_ACCELERATION_INVALID"
    if np.linalg.norm(np.asarray(acceleration, dtype=np.float64)) > 1e-9:
        return "INITIAL_ACCELERATION_NOT_INJECTABLE"
    return None


def resolve_materializable_selection(
    selected: Mapping[str, Any], *, started_processes: int = 0
) -> tuple[list[str], list[dict[str, Any]]]:
    """Resolve the frozen four slots using only their predeclared backups."""
    cases = selected.get("cases", {})
    chosen: list[str] = []
    substitutions: list[dict[str, Any]] = []
    categories = (("best2", "best_backups"), ("worst2", "worst_backups"))
    for main_key, backup_key in categories:
        backups = [_case_uid(value) for value in selected.get(backup_key, [])]
        backup_cursor = 0
        for slot_index, raw_uid in enumerate(selected.get(main_key, [])):
            uid = _case_uid(raw_uid)
            payload = cases.get(uid, {})
            reason = _materialization_failure(payload)
            if reason is None:
                chosen.append(uid)
                continue
            if started_processes:
                raise RuntimeError(
                    "frozen substitutions are disabled after a real process starts"
                )
            replacement = ""
            while backup_cursor < len(backups):
                candidate = backups[backup_cursor]
                backup_cursor += 1
                if candidate not in chosen and _materialization_failure(
                    cases.get(candidate, {})
                ) is None:
                    replacement = candidate
                    break
            if not replacement:
                raise ValueError(
                    f"no materializable frozen backup for {main_key}[{slot_index}] {uid}"
                )
            chosen.append(replacement)
            substitutions.append({
                "slot": f"{main_key}[{slot_index}]",
                "reason": reason,
                "rejected_case_uid": uid,
                "rejected_case_hash": payload.get("case_hash", ""),
                "selected_case_uid": replacement,
                "selected_case_hash": cases[replacement].get("case_hash", ""),
            })
    if len(chosen) != 4 or len(set(chosen)) != 4:
        raise ValueError("materialized selection must contain four unique cases")
    return chosen, substitutions


SHOWCASE_CASE_UIDS = (
    "syn_s_curve_sparse",
    "syn_dense_short",
    "extreme_yaw_reverse_narrow",
    "syn_l135",
)


def resolve_showcase_selection(
    selected: Mapping[str, Any], *, started_processes: int = 0
) -> tuple[list[str], list[dict[str, Any]]]:
    """Lock dynamic coverage to sparse, dense, narrow-extreme and folded guides."""
    cases = selected.get("cases", {})
    chosen: list[str] = []
    for uid in SHOWCASE_CASE_UIDS:
        reason = _materialization_failure(cases.get(uid, {}))
        if reason is not None:
            raise ValueError(f"required dynamic showcase case {uid}: {reason}")
        chosen.append(uid)
    if started_processes or len(chosen) != 4 or len(set(chosen)) != 4:
        raise RuntimeError("dynamic showcase selection is not a valid frozen four-case matrix")
    return chosen, []


def _usda_scene(rectangles: list[list[float]]) -> str:
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        '    metersPerUnit = 1',
        '    upAxis = "Z"',
        ")",
        'def Xform "World" {',
        '    def Cube "Floor" {',
        '        prepend apiSchemas = ["PhysicsCollisionAPI"]',
        "        bool physics:collisionEnabled = true",
        "        double size = 1",
        "        double3 xformOp:scale = (20, 20, 0.1)",
        "        double3 xformOp:translate = (5, 5, -0.05)",
        '        uniform token[] xformOpOrder = ["xformOp:scale", "xformOp:translate"]',
        "    }",
    ]
    for index, (xmin, ymin, xmax, ymax) in enumerate(rectangles):
        width = xmax - xmin
        depth = ymax - ymin
        if width <= 0.0 or depth <= 0.0:
            raise ValueError("invalid obstacle rectangle")
        lines.extend(
            [
                f'    def Cube "Obstacle_{index:03d}" {{',
                '        prepend apiSchemas = ["PhysicsCollisionAPI"]',
                "        bool physics:collisionEnabled = true",
                "        double size = 1",
                f"        double3 xformOp:scale = ({width}, {depth}, 1.0)",
                f"        double3 xformOp:translate = ({0.5 * (xmin + xmax)}, {0.5 * (ymin + ymax)}, 0.5)",
                '        uniform token[] xformOpOrder = ["xformOp:scale", "xformOp:translate"]',
                "    }",
            ]
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def prepare_dynamic_pilot(
    selected_cases_path: Path | str,
    calibration_path: Path | str,
    legacy_profile_path: Path | str,
    safe_profile_path: Path | str,
    output_dir: Path | str,
    *,
    repo_root: Path | str = ".",
    execute_dry_run: bool = True,
) -> dict[str, Any]:
    selected_cases_path = Path(selected_cases_path).resolve()
    calibration_path = Path(calibration_path).resolve()
    legacy_profile_path = Path(legacy_profile_path).resolve()
    safe_profile_path = Path(safe_profile_path).resolve()
    output_dir = Path(output_dir).resolve()
    repo_root = Path(repo_root).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"dynamic pilot output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = _load_json(selected_cases_path)
    policy_path = selected_cases_path.parent / "selection_policy.json"
    policy = _load_json(policy_path)
    if selected.get("policy_sha256") != policy.get("policy_sha256"):
        raise ValueError("selection policy hash mismatch")
    calibration = load_robot_calibration(calibration_path)
    profiles = {
        "legacy": _load_json(legacy_profile_path),
        "superplanner_sfc_v1": _load_json(safe_profile_path),
    }
    case_uids, substitutions = resolve_showcase_selection(selected)
    (output_dir / "substitution_receipts.json").write_text(
        json.dumps({
            "schema_version": 1,
            "started_processes": 0,
            "substitutions": substitutions,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    scenes = []
    materialization_receipts = {}
    suite_runs = []
    for case_index, uid in enumerate(case_uids):
        case_payload = selected["cases"].get(uid)
        if not case_payload:
            raise ValueError(f"selected case missing payload: {uid}")
        materialization = case_payload["materialization"]
        if not materialization.get("dynamic_replayable"):
            raise ValueError(f"selected case is not dynamic replayable: {uid}")
        if case_payload.get("case_hash") != case_payload["superplanner_sfc_v1_static"].get(
            "case_hash"
        ):
            raise ValueError(f"case hash mismatch: {uid}")
        start_pose = _decode_vector(
            materialization["required_start_pose"], "required_start_pose"
        )
        velocity = _decode_vector(
            materialization["required_initial_velocity"],
            "required_initial_velocity",
        )
        acceleration = _decode_vector(
            materialization["required_initial_acceleration"],
            "required_initial_acceleration",
        )
        _ensure_materializable_acceleration(acceleration, uid)
        goal = _decode_vector(materialization["required_goal"], "required_goal")
        guide = _decode_matrix(materialization["guide_path"], "guide_path", 3)
        rectangles = _decode_matrix(
            materialization["obstacle_layout"], "obstacle_layout", 4
        )
        yaw_rate = float(materialization["required_yaw_rate"])
        if not np.isfinite(yaw_rate):
            raise ValueError(f"{uid}: nonfinite yaw rate")

        scene_id = f"pilot_{case_index + 1:02d}_{uid}"
        scene_dir = output_dir / "scenes" / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        usd_path = scene_dir / "scene.usd"
        usd_path.write_text(_usda_scene(rectangles), encoding="utf-8")
        np.save(
            scene_dir / "pointgoal_start_goal_pairs.npy",
            np.asarray(
                [[start_pose[0], start_pose[1], goal[0], goal[1], start_pose[2]]],
                dtype=np.float64,
            ),
            allow_pickle=False,
        )
        dynamic_case = {
            "schema_version": 1,
            "case_uid": uid,
            "case_hash": case_payload["case_hash"],
            "scene_sha256": _sha256(usd_path),
            "calibration_sha256": calibration.calibration_sha256,
            "scene_id": scene_id,
            "start_pose_xy_yaw": start_pose,
            "initial_linear_velocity_xyz_mps": velocity,
            "initial_angular_velocity_xyz_radps": [0.0, 0.0, yaw_rate],
            "initial_acceleration_xyz_mps2": acceleration,
            "goal_xyz_m": goal,
            "guide_path_xyz_m": guide,
            "obstacle_rectangles_xyxy_m": rectangles,
            "frame_sanity_requirements": {
                "initial_penetration": False,
                "transform_profile_sha256": calibration.calibration_sha256,
                "esdf_clearance_match_tolerance_m": 0.05,
                "pose_match_tolerance_m": 0.02,
                "yaw_match_tolerance_rad": 0.02,
                "minimum_initial_clearance_m": calibration.validation_safe_dist_m,
                "velocity_match_tolerance_mps": 0.02,
                "yaw_rate_match_tolerance_radps": 0.02,
            },
        }
        case_receipt_path = scene_dir / "dynamic_case.json"
        case_receipt_path.write_text(
            json.dumps(dynamic_case, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        materialization_receipts[uid] = {
            "scene_id": scene_id,
            "scene_usd": str(usd_path),
            "scene_sha256": _sha256(usd_path),
            "dynamic_case": str(case_receipt_path),
            "dynamic_case_sha256": _sha256(case_receipt_path),
        }
        episode_uid = f"dynamic_{uid}"
        scenes.append(
            {
                "scene_id": scene_id,
                "scene_label": "DYNAMIC_STRESS",
                "scene_path": str(scene_dir),
                "asset_hash": _sha256(usd_path),
                "episodes": [
                    {
                        "scenario_id": uid,
                        "episode_index": 0,
                        "source_episode_index": 0,
                        "seed": 6100 + case_index,
                        "navdp_seed": 16100 + case_index,
                        "start_pose": start_pose,
                        "goal_pose": goal,
                        "episode_uid": episode_uid,
                        "selection_reason": "frozen sparse+dense+narrow+folded showcase coverage",
                    }
                ],
            }
        )
        # Three actual controller conditions: the unmodified NavDP controller,
        # MINCO without a corridor, and MINCO constrained/validated by native
        # SuperPlanner 2-D SFC.  ``raw`` is never synthesized from a guide.
        suite_runs.append(
            {
                "experiment_id": f"DYNAMIC-{uid}-navdp_native",
                "variant": "raw",
                "warm_start_mode": "cold",
                "scene_ids": [scene_id],
                "parameter_overrides": {},
            }
        )
        for profile_name in ("legacy", "superplanner_sfc_v1"):
            minco = dict(profiles[profile_name]["minco"])
            minco.setdefault("constraint_profile", profile_name)
            suite_runs.append(
                {
                    "experiment_id": f"DYNAMIC-{uid}-{profile_name}",
                    "variant": "minco-hot",
                    "warm_start_mode": "gated",
                    "scene_ids": [scene_id],
                    "parameter_overrides": {"minco": minco},
                }
            )

    manifest = {
        "manifest_version": 1,
        "manifest_id": "task06_sparse_dense_narrow_folded_three_method_v1",
        "seed": 6100,
        "scenes": scenes,
    }
    manifest_path = output_dir / "dynamic_scenario_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    safe_minco = dict(profiles["superplanner_sfc_v1"]["minco"])
    safe_minco.setdefault("constraint_profile", "superplanner_sfc_v1")
    suite = {
        "suite_id": "task06_dynamic_sparse_dense_narrow_folded_v1",
        "backend": "isaac",
        "output_root": str(output_dir / "dry_run_results"),
        "scenario_manifest": str(manifest_path),
        "video": {"enabled": True, "fps": 10},
        "monitor": {"enabled": True, "planning_trace": True},
        "analysis": {"enabled": True, "paired": True},
        "retry": {"failed": False},
        "resume": True,
        "parameters": {
            "minco": safe_minco,
            "robot_calibration": {
                "path": str(calibration_path),
                "sha256": calibration.calibration_sha256,
                "robot_model_id": calibration.robot_model_id,
                "status": calibration.status,
                "wheel_radius_m": calibration.wheel_radius_m,
                "wheel_base_m": calibration.wheel_base_m,
                "circumscribed_radius_m": calibration.circumscribed_radius_m,
                "validation_safe_dist_m": calibration.validation_safe_dist_m,
                "optimization_safe_dist_m": calibration.optimization_safe_dist_m,
            },
        },
        "runs": suite_runs,
    }
    suite_path = output_dir / "dynamic_suite.json"
    suite_path.write_text(
        json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if execute_dry_run:
        run_suite(
            suite_path,
            backend_name="isaac",
            dry_run=True,
            allow_real_simulation=False,
        )
    dry_plan_path = (
        output_dir
        / "dry_run_results"
        / suite["suite_id"]
        / "dry_run_plan.json"
    )
    if not dry_plan_path.is_file():
        raise RuntimeError("dry-run plan was not generated")
    dry_plan = _load_json(dry_plan_path)
    expected_run_count = len(case_uids) * 3
    if dry_plan.get("started_processes") != 0 or dry_plan.get("run_count") != expected_run_count:
        raise RuntimeError("dry-run process/run count mismatch")

    real_command = [
        "python",
        "-m",
        "experiments",
        "run-suite",
        "--config",
        str(suite_path),
        "--backend",
        "isaac",
        "--allow-real-simulation",
    ]
    expected_video_runs = [
        {
            "case_uid": uid,
            "episode_uid": f"dynamic_{uid}",
            "profile": profile,
        }
        for uid in case_uids
        for profile in ("navdp_native", "legacy", "superplanner_sfc_v1")
    ]
    pending_video_evidence = _write_pending_video_evidence(
        output_dir, expected_video_runs
    )
    receipt = {
        "schema_version": 1,
        "status": "READY_FOR_REAL_RUN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_cases_path": str(selected_cases_path),
        "selected_cases_sha256": _sha256(selected_cases_path),
        "selection_policy_sha256": selected["policy_sha256"],
        "calibration_path": str(calibration_path),
        "calibration_sha256": calibration.calibration_sha256,
        "profile_hashes": {
            "legacy": _sha256(legacy_profile_path),
            "superplanner_sfc_v1": _sha256(safe_profile_path),
        },
        "case_uids": case_uids,
        "substitutions": substitutions,
        "substitution_receipts": str(output_dir / "substitution_receipts.json"),
        "run_count": expected_run_count,
        "profiles": ["navdp_native", "legacy", "superplanner_sfc_v1"],
        "warm_start_mode": "gated",
        "suite_seed": 6100,
        "navdp_seeds": [16100 + index for index in range(4)],
        "started_processes": 0,
        "pending_video_evidence": pending_video_evidence,
        "estimated_resources": {
            "isaac_processes_per_run": 1,
            "navdp_server_processes_per_run": 1,
            "runs_sequential": True,
            "video_count_expected": expected_run_count,
            "planning_trace_count_expected": expected_run_count,
        },
        "materialization_receipts": materialization_receipts,
        "dry_run_plan": str(dry_plan_path),
        "dry_run_plan_sha256": _sha256(dry_plan_path),
        "real_command_argv": real_command,
        "real_command_shell": shlex.join(real_command),
        "expected_artifacts": [
            f"{expected_run_count} episode videos",
            f"{expected_run_count} planning trace archives",
            f"{expected_run_count} machine termination receipts",
            "4 synchronized NavDP-vs-legacy-MINCO-vs-SuperPlanner-SFC videos",
            "constraint/control/timing CSV tables",
            "static-vs-dynamic case cards",
        ],
    }
    receipt_path = output_dir / "dynamic_readiness_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "RUNBOOK.md").write_text(
        "# Task 06 dynamic pilot\n\n"
        "Status: **READY_FOR_REAL_RUN** (no real simulation executed).\n\n"
        f"Dry-run processes started: `{receipt['started_processes']}`.\n\n"
        "Authorized real command:\n\n"
        f"```bash\n{receipt['real_command_shell']}\n```\n\n"
        "Do not change the selected-case, calibration, profile, scene, or seed "
        "hashes after starting. Any infrastructure fix invalidates the pilot "
        "and requires all twelve three-method runs to restart.\n",
        encoding="utf-8",
    )
    return receipt
