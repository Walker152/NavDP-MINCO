from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments.core.artifact_receipt import (
    inventory_receipts,
    validate_file_receipt,
)
from experiments.rolling.models import (
    ObstacleState,
    RobotState,
    RolloutCycle,
    RolloutResult,
)


SCHEMA_VERSION = 1
CANONICAL_ARTIFACTS = (
    "run_manifest.json",
    "cycle_metrics.csv",
    "executed_trajectory.csv",
    "candidate_trajectories.npz",
    "corridor_segments.csv",
    "obstacle_states.csv",
    "metrics.json",
)
CYCLE_FIELDS = (
    "schema_version", "scenario_uid", "method", "cycle_index", "time_s",
    "input_x_m", "input_y_m", "input_z_m", "input_vx_mps", "input_vy_mps",
    "input_vz_mps", "input_ax_mps2", "input_ay_mps2", "input_az_mps2",
    "input_yaw_rad", "input_yaw_rate_radps", "local_goal_x_m", "local_goal_y_m",
    "local_goal_z_m", "candidate_sample_count", "executed_sample_count",
    "corridor_segment_count", "obstacle_count", "diagnostics_json",
)
TRAJECTORY_FIELDS = (
    "schema_version", "scenario_uid", "method", "scope", "sample_index",
    "time_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps",
    "ax_mps2", "ay_mps2", "az_mps2", "jerk_x_mps3", "jerk_y_mps3",
    "jerk_z_mps3", "yaw_rad", "yaw_rate_radps",
)
CORRIDOR_FIELDS = (
    "schema_version", "scenario_uid", "method", "cycle_index", "segment_index",
    "start_x_m", "start_y_m", "end_x_m", "end_y_m", "radius_m",
)
OBSTACLE_FIELDS = (
    "schema_version", "scenario_uid", "method", "cycle_index", "obstacle_uid",
    "center_x_m", "center_y_m", "radius_m", "velocity_x_mps", "velocity_y_mps",
    "dynamic",
)


@dataclass(frozen=True)
class RolloutReceipt:
    output_dir: Path
    manifest_path: Path
    artifact_receipt_path: Path
    artifact_count: int


def _jsonable(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported rollout JSON value: {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _cycle_row(result: RolloutResult, cycle: RolloutCycle) -> dict[str, object]:
    state = cycle.input_state
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_uid": result.scenario_uid,
        "method": result.method,
        "cycle_index": cycle.cycle_index,
        "time_s": cycle.time_s,
        "input_x_m": state.position_xyz[0], "input_y_m": state.position_xyz[1], "input_z_m": state.position_xyz[2],
        "input_vx_mps": state.velocity_xyz_mps[0], "input_vy_mps": state.velocity_xyz_mps[1], "input_vz_mps": state.velocity_xyz_mps[2],
        "input_ax_mps2": state.acceleration_xyz_mps2[0], "input_ay_mps2": state.acceleration_xyz_mps2[1], "input_az_mps2": state.acceleration_xyz_mps2[2],
        "input_yaw_rad": state.yaw_rad, "input_yaw_rate_radps": state.yaw_rate_radps,
        "local_goal_x_m": cycle.local_goal_xyz[0], "local_goal_y_m": cycle.local_goal_xyz[1], "local_goal_z_m": cycle.local_goal_xyz[2],
        "candidate_sample_count": len(cycle.candidate_samples),
        "executed_sample_count": len(cycle.executed_samples),
        "corridor_segment_count": len(cycle.corridor_segments),
        "obstacle_count": len(cycle.obstacle_states),
        "diagnostics_json": json.dumps(_jsonable(cycle.diagnostics), sort_keys=True, separators=(",", ":"), allow_nan=False),
    }


def _trajectory_rows(result: RolloutResult) -> list[dict[str, object]]:
    rows = []
    for index, sample in enumerate(result.executed_samples):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION, "scenario_uid": result.scenario_uid,
                "method": result.method, "scope": "ALL_RUN", "sample_index": index,
                "time_s": sample[0], "x_m": sample[1], "y_m": sample[2], "z_m": sample[3],
                "vx_mps": sample[4], "vy_mps": sample[5], "vz_mps": sample[6],
                "ax_mps2": sample[7], "ay_mps2": sample[8], "az_mps2": sample[9],
                "jerk_x_mps3": sample[10], "jerk_y_mps3": sample[11], "jerk_z_mps3": sample[12],
                "yaw_rad": sample[13], "yaw_rate_radps": sample[14],
            }
        )
    return rows


def write_rollout_result(result: RolloutResult, output_dir: Path | str) -> RolloutReceipt:
    if not isinstance(result, RolloutResult):
        raise TypeError("result must be RolloutResult")
    validation_errors = result.validate()
    if validation_errors:
        raise ValueError("invalid rollout result: " + "; ".join(validation_errors))
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"rollout output must be absent or nonempty-free: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scenario_uid": result.scenario_uid,
        "method": result.method,
        "status": result.status,
        "cycle_count": len(result.cycles),
        "executed_sample_count": len(result.executed_samples),
        "final_goal_xyz": result.final_goal_xyz.tolist(),
        "goal_tolerance_m": result.goal_tolerance_m,
        "data_source": "STATIC_SYNTHETIC",
        "artifacts": list(CANONICAL_ARTIFACTS[1:]),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    _write_csv(output_dir / "cycle_metrics.csv", CYCLE_FIELDS, [_cycle_row(result, cycle) for cycle in result.cycles])
    _write_csv(output_dir / "executed_trajectory.csv", TRAJECTORY_FIELDS, _trajectory_rows(result))
    arrays: dict[str, np.ndarray] = {}
    for cycle in result.cycles:
        prefix = f"cycle_{cycle.cycle_index:06d}"
        arrays[f"{prefix}__candidate_samples"] = cycle.candidate_samples
        arrays[f"{prefix}__executed_samples"] = cycle.executed_samples
        arrays[f"{prefix}__local_guide_xyz"] = cycle.local_guide_xyz
    np.savez_compressed(output_dir / "candidate_trajectories.npz", **arrays)
    corridor_rows = []
    obstacle_rows = []
    for cycle in result.cycles:
        for index, row in enumerate(cycle.corridor_segments):
            corridor_rows.append({
                "schema_version": SCHEMA_VERSION, "scenario_uid": result.scenario_uid,
                "method": result.method, "cycle_index": cycle.cycle_index, "segment_index": index,
                "start_x_m": row[0], "start_y_m": row[1], "end_x_m": row[2], "end_y_m": row[3], "radius_m": row[4],
            })
        for obstacle in cycle.obstacle_states:
            obstacle_rows.append({
                "schema_version": SCHEMA_VERSION, "scenario_uid": result.scenario_uid,
                "method": result.method, "cycle_index": cycle.cycle_index,
                "obstacle_uid": obstacle.obstacle_uid, "center_x_m": obstacle.center_xy[0],
                "center_y_m": obstacle.center_xy[1], "radius_m": obstacle.radius_m,
                "velocity_x_mps": obstacle.velocity_xy_mps[0], "velocity_y_mps": obstacle.velocity_xy_mps[1],
                "dynamic": str(obstacle.dynamic).lower(),
            })
    _write_csv(output_dir / "corridor_segments.csv", CORRIDOR_FIELDS, corridor_rows)
    _write_csv(output_dir / "obstacle_states.csv", OBSTACLE_FIELDS, obstacle_rows)
    _write_json(output_dir / "metrics.json", {
        "schema_version": SCHEMA_VERSION, "scenario_uid": result.scenario_uid,
        "method": result.method, "scope": "ALL_RUN", "status": result.status,
        "metrics": result.metrics,
    })
    receipt_path = output_dir / "artifact_receipt.json"
    receipts = inventory_receipts(output_dir, exclude=(receipt_path,))
    _write_json(receipt_path, {"schema_version": SCHEMA_VERSION, "root": ".", "artifacts": receipts})
    errors = validate_rollout_result(output_dir)
    if errors:
        raise RuntimeError("generated rollout evidence is invalid: " + "; ".join(errors))
    return RolloutReceipt(output_dir, output_dir / "run_manifest.json", receipt_path, len(receipts))


def _read_csv(path: Path, fields: Sequence[str], errors: list[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                errors.append(f"{path.name} schema mismatch")
            return list(reader)
    except (OSError, csv.Error) as error:
        errors.append(f"unreadable {path.name}: {error}")
        return []


def validate_rollout_result(output_dir: Path | str) -> list[str]:
    output_dir = Path(output_dir).resolve()
    errors: list[str] = []
    try:
        manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable run_manifest.json: {error}"]
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("run_manifest schema_version mismatch")
    try:
        receipt = json.loads((output_dir / "artifact_receipt.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"unreadable artifact_receipt.json: {error}")
        receipt = {"artifacts": []}
    recorded = {str(row.get("path", "")) for row in receipt.get("artifacts", [])}
    if recorded != set(CANONICAL_ARTIFACTS):
        errors.append("artifact receipt does not cover exact canonical rollout artifacts")
    for row in receipt.get("artifacts", []):
        errors.extend(validate_file_receipt(output_dir, row))

    cycles = _read_csv(output_dir / "cycle_metrics.csv", CYCLE_FIELDS, errors)
    trajectory = _read_csv(output_dir / "executed_trajectory.csv", TRAJECTORY_FIELDS, errors)
    corridors = _read_csv(output_dir / "corridor_segments.csv", CORRIDOR_FIELDS, errors)
    obstacles = _read_csv(output_dir / "obstacle_states.csv", OBSTACLE_FIELDS, errors)
    identity = (str(manifest.get("scenario_uid", "")), str(manifest.get("method", "")))
    for name, rows in (("cycle_metrics.csv", cycles), ("executed_trajectory.csv", trajectory), ("corridor_segments.csv", corridors), ("obstacle_states.csv", obstacles)):
        for index, row in enumerate(rows):
            if (row.get("scenario_uid"), row.get("method")) != identity:
                errors.append(f"{name} row {index} identity mismatch")
            if row.get("schema_version") != str(SCHEMA_VERSION):
                errors.append(f"{name} row {index} schema mismatch")
    if len(cycles) != manifest.get("cycle_count"):
        errors.append("cycle count mismatch")
    if len(trajectory) != manifest.get("executed_sample_count"):
        errors.append("executed trajectory count mismatch")
    try:
        with np.load(output_dir / "candidate_trajectories.npz", allow_pickle=False) as archive:
            expected_keys = {
                f"cycle_{index:06d}__{kind}"
                for index in range(len(cycles))
                for kind in ("candidate_samples", "executed_samples", "local_guide_xyz")
            }
            if set(archive.files) != expected_keys:
                errors.append("candidate NPZ keys do not match cycle records")
            for key in archive.files:
                array = archive[key]
                if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
                    errors.append(f"candidate NPZ contains invalid array {key}")
    except (OSError, ValueError) as error:
        errors.append(f"unreadable candidate_trajectories.npz: {error}")
    try:
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        if (metrics.get("scenario_uid"), metrics.get("method")) != identity:
            errors.append("metrics identity mismatch")
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"unreadable metrics.json: {error}")
    return errors


def _float(row: Mapping[str, str], field: str) -> float:
    return float(row[field])


def load_rollout_result(manifest_path: Path | str) -> RolloutResult:
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    errors = validate_rollout_result(root)
    if errors:
        raise ValueError("invalid rollout evidence: " + "; ".join(errors))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with (root / "cycle_metrics.csv").open(newline="", encoding="utf-8") as stream:
        cycle_rows = list(csv.DictReader(stream))
    with (root / "corridor_segments.csv").open(newline="", encoding="utf-8") as stream:
        corridor_rows = list(csv.DictReader(stream))
    with (root / "obstacle_states.csv").open(newline="", encoding="utf-8") as stream:
        obstacle_rows = list(csv.DictReader(stream))
    cycles = []
    with np.load(root / "candidate_trajectories.npz", allow_pickle=False) as archive:
        for row in cycle_rows:
            index = int(row["cycle_index"])
            prefix = f"cycle_{index:06d}"
            corridors = np.asarray([
                [_float(item, field) for field in ("start_x_m", "start_y_m", "end_x_m", "end_y_m", "radius_m")]
                for item in corridor_rows if int(item["cycle_index"]) == index
            ], dtype=float).reshape((-1, 5))
            obstacles = tuple(
                ObstacleState(
                    item["obstacle_uid"],
                    [_float(item, "center_x_m"), _float(item, "center_y_m")],
                    _float(item, "radius_m"),
                    [_float(item, "velocity_x_mps"), _float(item, "velocity_y_mps")],
                    item["dynamic"] == "true",
                )
                for item in obstacle_rows if int(item["cycle_index"]) == index
            )
            state = RobotState(
                [_float(row, "input_x_m"), _float(row, "input_y_m"), _float(row, "input_z_m")],
                [_float(row, "input_vx_mps"), _float(row, "input_vy_mps"), _float(row, "input_vz_mps")],
                [_float(row, "input_ax_mps2"), _float(row, "input_ay_mps2"), _float(row, "input_az_mps2")],
                _float(row, "input_yaw_rad"), _float(row, "input_yaw_rate_radps"),
            )
            cycles.append(RolloutCycle(
                index, _float(row, "time_s"), state,
                archive[f"{prefix}__local_guide_xyz"],
                [_float(row, "local_goal_x_m"), _float(row, "local_goal_y_m"), _float(row, "local_goal_z_m")],
                archive[f"{prefix}__candidate_samples"], archive[f"{prefix}__executed_samples"],
                corridors, obstacles, json.loads(row["diagnostics_json"]),
            ))
    with (root / "executed_trajectory.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    executed = np.asarray([
        [_float(row, field) for field in TRAJECTORY_FIELDS[5:]] for row in rows
    ], dtype=float)
    metrics_payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    return RolloutResult(
        manifest["scenario_uid"], manifest["method"], manifest["status"], tuple(cycles),
        executed, manifest["final_goal_xyz"], metrics_payload["metrics"], manifest["goal_tolerance_m"],
    )


__all__ = ["RolloutReceipt", "load_rollout_result", "validate_rollout_result", "write_rollout_result"]
