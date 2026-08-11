from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from experiments.analyzers.trace_evidence import load_trace_evidence
from experiments.static.case_schema import StaticCase


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xyz(points: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] not in (2, 3):
        raise ValueError(f"{name} must have shape (N>=2, 2|3)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite")
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])
    return points


def _load_esdf(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    path = Path(path).resolve()
    try:
        with np.load(path, allow_pickle=False) as archive:
            distance = np.asarray(archive["distance"], dtype=np.float64)
            if "occupied" in archive:
                occupancy = np.asarray(archive["occupied"])
            elif "free" in archive:
                occupancy = ~np.asarray(archive["free"], dtype=bool)
            else:
                raise ValueError("ESDF requires occupied or free array")
            origin = np.asarray(archive["origin"], dtype=np.float64)
            resolution = float(archive["resolution"])
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"invalid compatible ESDF: {error}") from error
    return distance, occupancy.astype(bool), origin, resolution, _sha256(path)


def import_trace_case(
    trace_path: Path | str,
    *,
    case_uid: str,
    expected_trace_sha256: str | None = None,
    compatible_esdf_path: Path | str | None = None,
    expected_esdf_sha256: str | None = None,
) -> StaticCase:
    evidence = load_trace_evidence(trace_path)
    trace_hash = _sha256(evidence.path)
    if expected_trace_sha256 is not None and trace_hash != expected_trace_sha256:
        raise ValueError("trace hash mismatch")
    required = ("selected_candidate_xy", "robot_state")
    missing = [name for name in required if name not in evidence.arrays]
    if missing:
        raise ValueError("trace missing required fields: " + ", ".join(missing))
    guide = _xyz(evidence.arrays["selected_candidate_xy"], "selected candidate")
    state = np.asarray(evidence.arrays["robot_state"], dtype=np.float64)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("robot_state must be finite [x,y,yaw,vx,vy,yaw_rate]")

    references = {"trace_path": str(evidence.path), "trace_sha256": trace_hash}
    if compatible_esdf_path is None:
        occupancy = np.zeros((1, 1), dtype=bool)
        distance = np.zeros((1, 1), dtype=np.float64)
        origin = np.asarray(state[:2], dtype=np.float64)
        resolution = 1.0
        esdf_available = False
    else:
        distance, occupancy, origin, resolution, esdf_hash = _load_esdf(
            compatible_esdf_path
        )
        if expected_esdf_sha256 is not None and esdf_hash != expected_esdf_sha256:
            raise ValueError("ESDF hash mismatch")
        references.update(
            {
                "esdf_path": str(Path(compatible_esdf_path).resolve()),
                "esdf_sha256": esdf_hash,
            }
        )
        esdf_available = True

    auxiliary = {}
    if "raw_path_xy" in evidence.arrays:
        auxiliary["raw_path_xyz"] = _xyz(
            evidence.arrays["raw_path_xy"], "raw path"
        )
    if "minco_samples" in evidence.arrays:
        samples = np.asarray(evidence.arrays["minco_samples"], dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != 15 or not np.all(
            np.isfinite(samples)
        ):
            raise ValueError("minco_samples must be finite shape (N, 15)")
        auxiliary["historical_minco_samples"] = samples
    if "topk_candidates_xy" in evidence.arrays:
        topk = np.asarray(evidence.arrays["topk_candidates_xy"], dtype=np.float64)
        if topk.ndim == 3 and topk.shape[2] == 2 and np.all(np.isfinite(topk)):
            auxiliary["topk_candidates_xy"] = topk
    goal = evidence.arrays.get("goal")
    terminal = None
    if goal is not None:
        goal = np.asarray(goal, dtype=np.float64)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError("goal must be finite shape (3,)")
        terminal = goal
    return StaticCase(
        case_uid=case_uid,
        case_source="STATIC_REPLAY_REAL_TRACE",
        guide_path_xyz=guide,
        occupancy=occupancy,
        esdf_distance=distance,
        esdf_origin=origin,
        esdf_resolution=resolution,
        start_position=np.array([state[0], state[1], 0.0]),
        start_velocity=np.array([state[3], state[4], 0.0]),
        start_acceleration=np.zeros(3),
        start_yaw=float(state[2]),
        start_yaw_rate=float(state[5]),
        terminal_goal=terminal,
        constraint_profile="legacy",
        expected_category="real_trace_replay",
        esdf_available=esdf_available,
        tags=("INSPECT_ONLY",) if not esdf_available else ("RECOMPUTE_READY",),
        state_availability={
            "position": True,
            "velocity": True,
            "acceleration": False,
            "yaw": True,
            "yaw_rate": True,
        },
        references=references,
        auxiliary_arrays=auxiliary,
    )
