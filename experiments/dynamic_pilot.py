from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any, Mapping

import numpy as np

from experiments.calibration.profile import load_robot_calibration
from experiments.orchestrators.suite_runner import run_suite


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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
        "safe_corridor_v1": _load_json(safe_profile_path),
    }
    case_uids = list(selected.get("best2", [])) + list(
        selected.get("worst2", [])
    )
    if len(case_uids) != 4 or len(set(case_uids)) != 4:
        raise ValueError("selected file must contain exactly four unique cases")

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
        if case_payload.get("case_hash") != case_payload["safe_corridor_v1_static"].get(
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
                        "selection_reason": "Task05 frozen Best2/Worst2",
                    }
                ],
            }
        )
        for profile_name in ("legacy", "safe_corridor_v1"):
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
        "manifest_id": "task06_frozen_best2_worst2_v1",
        "seed": 6100,
        "scenes": scenes,
    }
    manifest_path = output_dir / "dynamic_scenario_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    safe_minco = dict(profiles["safe_corridor_v1"]["minco"])
    safe_minco.setdefault("constraint_profile", "safe_corridor_v1")
    suite = {
        "suite_id": "task06_dynamic_best2_worst2_v1",
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
    if dry_plan.get("started_processes") != 0 or dry_plan.get("run_count") != 8:
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
            "safe_corridor_v1": _sha256(safe_profile_path),
        },
        "case_uids": case_uids,
        "run_count": 8,
        "profiles": ["legacy", "safe_corridor_v1"],
        "warm_start_mode": "gated",
        "suite_seed": 6100,
        "navdp_seeds": [16100 + index for index in range(4)],
        "started_processes": 0,
        "estimated_resources": {
            "isaac_processes_per_run": 1,
            "navdp_server_processes_per_run": 1,
            "runs_sequential": True,
            "video_count_expected": 8,
            "planning_trace_count_expected": 8,
        },
        "materialization_receipts": materialization_receipts,
        "dry_run_plan": str(dry_plan_path),
        "dry_run_plan_sha256": _sha256(dry_plan_path),
        "real_command_argv": real_command,
        "real_command_shell": shlex.join(real_command),
        "expected_artifacts": [
            "8 episode videos",
            "8 planning trace archives",
            "8 machine termination receipts",
            "4 synchronized legacy-vs-safe videos",
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
        "and requires all eight paired runs to restart.\n",
        encoding="utf-8",
    )
    return receipt
