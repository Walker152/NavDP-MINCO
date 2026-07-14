from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SituationThresholds:
    safe_dist_m: float
    high_turn_curvature_p95_1pm: float
    high_turn_curvature_tv_1pm: float
    jump_position_rmse_m: float
    jump_tangent_rad: float


def classify_plan_situation(raw_metrics, thresholds):
    clearance = raw_metrics.get("raw_min_clearance_m", float("nan"))
    unsafe = (not math.isfinite(clearance) or clearance <= thresholds.safe_dist_m or raw_metrics.get("raw_unsafe_ratio", 0) > 0 or raw_metrics.get("raw_esdf_oob_ratio", 0) > 0)
    high_turn = raw_metrics.get("raw_curvature_abs_p95_1pm", 0) >= thresholds.high_turn_curvature_p95_1pm or raw_metrics.get("raw_curvature_tv_1pm", 0) >= thresholds.high_turn_curvature_tv_1pm
    position = raw_metrics.get("raw_interplan_position_rmse_m", float("nan")); tangent = raw_metrics.get("raw_initial_tangent_jump_rad", float("nan"))
    if not math.isfinite(position) and not math.isfinite(tangent): temporal = "NO_HISTORY"
    else: temporal = "JUMP_INPUT" if (math.isfinite(position) and position >= thresholds.jump_position_rmse_m) or (math.isfinite(tangent) and tangent >= thresholds.jump_tangent_rad) else "STABLE_INPUT"
    return {"raw_safety_class": "RAW_UNSAFE" if unsafe else "RAW_SAFE", "turn_class": "HIGH_TURN" if high_turn else "LOW_TURN", "temporal_class": temporal}
