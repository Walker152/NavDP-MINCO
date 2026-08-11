from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from experiments.analyzers.metrics import (
    compute_geometric_metrics,
    compute_minco_temporal_profile,
    resample_polyline_by_arclength,
)
from experiments.static.case_schema import StaticCase
from experiments.static.runner import StaticRunResult
from utils_tasks.esdf_query_utils import EsdfGridView


def _distances_to_polyline(points: np.ndarray, guide: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)[:, :2]
    line = np.asarray(guide, dtype=float)[:, :2]
    starts = line[:-1]
    vectors = np.diff(line, axis=0)
    denom = np.sum(vectors * vectors, axis=1)
    valid = denom > 1e-12
    if not np.any(valid) or len(points) == 0:
        return np.full(len(points), np.nan)
    starts, vectors, denom = starts[valid], vectors[valid], denom[valid]
    delta = points[:, None, :] - starts[None, :, :]
    ratios = np.clip(
        np.sum(delta * vectors[None, :, :], axis=2) / denom[None, :],
        0.0,
        1.0,
    )
    projected = starts[None, :, :] + ratios[:, :, None] * vectors[None, :, :]
    return np.min(np.linalg.norm(points[:, None, :] - projected, axis=2), axis=1)


def _backtracking_ratio(points: np.ndarray, guide: np.ndarray) -> float:
    edges = np.diff(np.asarray(points, dtype=float)[:, :2], axis=0)
    total = float(np.sum(np.linalg.norm(edges, axis=1)))
    direction = np.asarray(guide[-1, :2] - guide[0, :2], dtype=float)
    norm = float(np.linalg.norm(direction))
    if total <= 1e-12 or norm <= 1e-12:
        return 0.0
    signed = edges @ (direction / norm)
    return float(np.sum(np.maximum(-signed, 0.0)) / total)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(np.cross(b - a, c - a))


def _proper_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    epsilon = 1e-10
    return o1 * o2 < -epsilon and o3 * o4 < -epsilon


def _self_intersection_count(points: np.ndarray) -> int:
    points = np.asarray(points, dtype=float)[:, :2]
    count = 0
    for first in range(max(0, len(points) - 1)):
        for second in range(first + 2, len(points) - 1):
            if second == first + 1:
                continue
            count += int(
                _proper_intersection(
                    points[first],
                    points[first + 1],
                    points[second],
                    points[second + 1],
                )
            )
    return count


def _ratio(values: np.ndarray, limit: float) -> float:
    return float(np.mean(values > float(limit))) if len(values) else float("nan")


def compute_static_case_metrics(
    case: StaticCase,
    result: StaticRunResult,
    limits: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = (
        result.samples[:, 1:4]
        if len(result.samples)
        else np.asarray(result.waypoints, dtype=float)
    )
    geometry, geometry_detail = compute_geometric_metrics(
        path, float(limits["sample_ds_m"])
    )
    sampled, arclength = resample_polyline_by_arclength(
        path, float(limits["sample_ds_m"])
    )
    grid = EsdfGridView(
        distance=case.esdf_distance,
        origin=case.esdf_origin,
        resolution=case.esdf_resolution,
    )
    clearance, valid = grid.query_points(sampled)
    guide_deviation = _distances_to_polyline(sampled, case.guide_path_xyz)
    path_length = float(geometry.get("path_length_m", float("nan")))
    guide_length = compute_geometric_metrics(
        case.guide_path_xyz, float(limits["sample_ds_m"])
    )[0]["path_length_m"]
    temporal, temporal_detail = compute_minco_temporal_profile(result.samples)
    speed = np.linalg.norm(result.samples[:, 4:7], axis=1)
    acceleration = np.linalg.norm(result.samples[:, 7:10], axis=1)
    jerk = np.linalg.norm(result.samples[:, 10:13], axis=1)
    yaw_rate = np.abs(result.samples[:, 14])
    unsafe = (~valid) | (
        valid & (clearance <= float(limits["safe_distance_m"]))
    )
    metrics: dict[str, Any] = {
        **geometry,
        **temporal,
        "case_uid": case.case_uid,
        "case_hash": case.case_hash,
        "case_source": case.case_source,
        "status": result.status,
        "failure_reason": str(result.diagnostics.get("failure_reason", "")),
        "min_clearance_m": float(np.min(clearance[valid])) if np.any(valid) else float("nan"),
        "clearance_p05_m": float(np.percentile(clearance[valid], 5)) if np.any(valid) else float("nan"),
        "unsafe_ratio": float(np.mean(unsafe)) if len(unsafe) else 1.0,
        "oob_ratio": float(np.mean(~valid)) if len(valid) else 1.0,
        "negative_esdf_count": int(np.sum(valid & (clearance < 0.0))),
        "guide_deviation_mean_m": float(np.nanmean(guide_deviation)) if len(guide_deviation) else float("nan"),
        "guide_deviation_p95_m": float(np.nanpercentile(guide_deviation, 95)) if len(guide_deviation) else float("nan"),
        "guide_deviation_max_m": float(np.nanmax(guide_deviation)) if len(guide_deviation) else float("nan"),
        "path_length_ratio": path_length / guide_length if guide_length > 1e-12 else float("nan"),
        "backtracking_ratio": _backtracking_ratio(path, case.guide_path_xyz),
        "self_intersection_count": _self_intersection_count(path),
        "endpoint_error_m": float(np.linalg.norm(path[-1] - case.terminal_goal))
        if len(path) and case.terminal_goal is not None
        else float("nan"),
        "velocity_violation_ratio": _ratio(speed, limits["max_velocity_mps"]),
        "acceleration_violation_ratio": _ratio(acceleration, limits["max_acceleration_mps2"]),
        "jerk_violation_ratio": _ratio(jerk, limits["max_jerk_mps3"]),
        "yaw_rate_violation_ratio": _ratio(yaw_rate, limits["max_yaw_rate_radps"]),
        "sample_count": int(len(result.samples)),
    }
    detail = {
        **geometry_detail,
        **temporal_detail,
        "clearance_s_m": arclength,
        "clearance_xy": sampled,
        "clearance_m": clearance,
        "clearance_valid": valid,
        "guide_deviation_m": guide_deviation,
    }
    return metrics, detail
