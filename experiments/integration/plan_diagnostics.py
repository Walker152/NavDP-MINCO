from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from experiments.analyzers.metrics import (
    compute_geometric_metrics,
    sanitize_polyline,
)
from experiments.analyzers.situations import (
    SituationThresholds,
    classify_plan_situation,
)


@dataclass(frozen=True)
class ThresholdProfile:
    profile_id: str
    safe_dist_m: float
    high_turn_curvature_p95_1pm: float
    high_turn_curvature_tv_1pm: float
    jump_position_rmse_m: float
    jump_tangent_rad: float
    planning_deadline_ms: float
    control_deadline_ms: float
    start_exemption_radius_m: float = 0.35

    def situation_thresholds(self) -> SituationThresholds:
        return SituationThresholds(
            safe_dist_m=self.safe_dist_m,
            high_turn_curvature_p95_1pm=self.high_turn_curvature_p95_1pm,
            high_turn_curvature_tv_1pm=self.high_turn_curvature_tv_1pm,
            jump_position_rmse_m=self.jump_position_rmse_m,
            jump_tangent_rad=self.jump_tangent_rad,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def write_threshold_profile(path, profile: ThresholdProfile) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(profile.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _normalized_arclength_samples(points, count=64):
    points = sanitize_polyline(points)
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.float64)
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if cumulative[-1] <= 1e-12:
        return np.empty((0, 2), dtype=np.float64)
    query = np.linspace(0.0, cumulative[-1], int(count))
    return np.column_stack(
        [np.interp(query, cumulative, points[:, axis]) for axis in range(2)]
    )


def compare_raw_paths(previous_path, current_path) -> dict:
    previous = _normalized_arclength_samples(previous_path)
    current = _normalized_arclength_samples(current_path)
    if len(previous) == 0 or len(current) == 0:
        return {
            "raw_interplan_position_rmse_m": float("nan"),
            "raw_initial_tangent_jump_rad": float("nan"),
        }

    delta = current - previous
    previous_tangent = previous[1] - previous[0]
    current_tangent = current[1] - current[0]
    denominator = np.linalg.norm(previous_tangent) * np.linalg.norm(current_tangent)
    tangent_jump = float("nan")
    if denominator > 1e-12:
        cosine = np.clip(
            np.dot(previous_tangent, current_tangent) / denominator,
            -1.0,
            1.0,
        )
        tangent_jump = float(math.acos(cosine))
    return {
        "raw_interplan_position_rmse_m": float(
            np.sqrt(np.mean(np.sum(delta * delta, axis=1)))
        ),
        "raw_initial_tangent_jump_rad": tangent_jump,
    }


def compute_raw_plan_diagnostics(
    path,
    esdf_grid,
    profile: ThresholdProfile,
    previous_path=None,
    safety_report=None,
) -> dict:
    geometry, _ = compute_geometric_metrics(path)
    safety = safety_report or esdf_grid.inspect_polyline(
        path,
        safe_dist=profile.safe_dist_m,
        start_exemption_radius=profile.start_exemption_radius_m,
    )
    temporal = compare_raw_paths(previous_path, path)
    metrics = {
        "threshold_profile_id": profile.profile_id,
        "raw_min_clearance_m": safety["min_clearance"],
        "raw_unsafe_ratio": safety["unsafe_ratio"],
        "raw_esdf_oob_ratio": safety["oob_ratio"],
        "raw_path_length_m": geometry["path_length_m"],
        "raw_curvature_abs_p95_1pm": geometry["curvature_abs_p95_1pm"],
        "raw_curvature_tv_1pm": geometry["curvature_tv_1pm"],
        "raw_curvature_rate_rms_1pm2": geometry["curvature_rate_rms_1pm2"],
        **temporal,
    }
    metrics.update(classify_plan_situation(metrics, profile.situation_thresholds()))
    return metrics
