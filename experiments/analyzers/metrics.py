from __future__ import annotations

import math
import numpy as np

from utils_tasks.esdf_query_utils import EsdfGridView


def sanitize_polyline(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    points = points[:, :2]
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        return points
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12]
    return points[keep]


def resample_polyline_by_arclength(points: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray]:
    if ds <= 0:
        raise ValueError("ds must be positive")
    points = sanitize_polyline(points)
    if len(points) < 2:
        return np.empty((0, 2)), np.empty(0)
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if cumulative[-1] <= 1e-12:
        return np.empty((0, 2)), np.empty(0)
    samples_s = np.r_[np.arange(0.0, cumulative[-1], ds), cumulative[-1]]
    samples_s = np.unique(samples_s)
    xy = np.column_stack([np.interp(samples_s, cumulative, points[:, axis]) for axis in range(2)])
    return xy.astype(np.float64), samples_s.astype(np.float64)


def compute_geometric_metrics(points: np.ndarray, sample_ds: float = 0.025):
    xy, s = resample_polyline_by_arclength(points, sample_ds)
    if len(xy) < 2:
        nan = float("nan")
        return {k: nan for k in ("path_length_m", "heading_change_max_rad", "curvature_abs_mean_1pm", "curvature_abs_p95_1pm", "curvature_abs_max_1pm", "curvature_tv_1pm", "curvature_rate_rms_1pm2")}, {"s_m": s, "xy": xy, "curvature_1pm": np.empty(0)}
    edge_order = 2 if len(xy) >= 3 else 1
    dx = np.gradient(xy[:, 0], s, edge_order=edge_order)
    dy = np.gradient(xy[:, 1], s, edge_order=edge_order)
    ddx = np.gradient(dx, s, edge_order=edge_order)
    ddy = np.gradient(dy, s, edge_order=edge_order)
    denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-12)
    curvature = (dx * ddy - dy * ddx) / denom
    heading = np.unwrap(np.arctan2(dy, dx))
    rate = np.gradient(curvature, s, edge_order=edge_order)
    metrics = {
        "path_length_m": float(s[-1]),
        "heading_change_max_rad": float(np.ptp(heading)),
        "curvature_abs_mean_1pm": float(np.mean(np.abs(curvature))),
        "curvature_abs_p95_1pm": float(np.percentile(np.abs(curvature), 95)),
        "curvature_abs_max_1pm": float(np.max(np.abs(curvature))),
        "curvature_tv_1pm": float(np.sum(np.abs(np.diff(curvature)))),
        "curvature_rate_rms_1pm2": float(np.sqrt(np.mean(rate * rate))),
    }
    return metrics, {"s_m": s, "xy": xy, "heading_rad": heading, "curvature_1pm": curvature, "curvature_rate_1pm2": rate}


def query_esdf_bilinear(distance, origin, resolution, points_xy):
    return EsdfGridView(
        distance=np.asarray(distance, dtype=np.float64),
        origin=np.asarray(origin, dtype=np.float64),
        resolution=float(resolution),
    ).query_points(points_xy)


def compute_safety_metrics(points_xy, esdf, safe_dist, sample_ds):
    sampled, s = resample_polyline_by_arclength(points_xy, sample_ds)
    if len(sampled) == 0:
        sampled = sanitize_polyline(points_xy)
        s = np.arange(len(sampled), dtype=float)
    values, valid = query_esdf_bilinear(esdf["distance"], esdf["origin"], esdf["resolution"], sampled)
    unsafe = (~valid) | (values <= safe_dist)
    count = len(sampled)
    metrics = {"min_clearance_m": float(np.min(values[valid])) if np.any(valid) else float("nan"), "unsafe_ratio": float(np.mean(unsafe)) if count else 1.0, "esdf_oob_ratio": float(np.mean(~valid)) if count else 1.0, "unsafe_sample_count": int(np.sum(unsafe)), "sample_count": count, "safe_dist_m": float(safe_dist)}
    return metrics, {"s_m": s, "xy": sampled, "esdf_m": values, "valid_mask": valid, "unsafe_mask": unsafe}


def _weighted(values, t, rms=False):
    if len(values) < 2 or t[-1] <= t[0]: return float("nan")
    base = values * values if rms else values
    value = float(np.trapz(base, t) / (t[-1] - t[0]))
    return math.sqrt(max(value, 0.0)) if rms else value


def compute_minco_temporal_profile(samples):
    samples = np.asarray(samples, dtype=float)
    names = ["actual_speed_mean_mps", "actual_speed_p95_mps", "actual_speed_max_mps", "actual_acc_rms_mps2", "actual_acc_p95_mps2", "actual_acc_max_mps2", "actual_jerk_rms_mps3", "actual_jerk_p95_mps3", "actual_jerk_max_mps3", "actual_yaw_rate_rms_radps", "actual_yaw_rate_max_radps", "trajectory_duration_s"]
    if samples.ndim != 2 or samples.shape[1] < 15:
        return {name: float("nan") for name in names}, {}
    samples = samples[np.all(np.isfinite(samples), axis=1)]
    if len(samples) < 2 or np.any(np.diff(samples[:, 0]) < 0):
        return {name: float("nan") for name in names}, {}
    t = samples[:, 0]; speed = np.linalg.norm(samples[:, 4:7], axis=1); acc = np.linalg.norm(samples[:, 7:10], axis=1); jerk = np.linalg.norm(samples[:, 10:13], axis=1); yaw_rate = np.abs(samples[:, 14])
    summary = {"actual_speed_mean_mps": _weighted(speed, t), "actual_speed_p95_mps": float(np.percentile(speed, 95)), "actual_speed_max_mps": float(np.max(speed)), "actual_acc_rms_mps2": _weighted(acc, t, True), "actual_acc_p95_mps2": float(np.percentile(acc, 95)), "actual_acc_max_mps2": float(np.max(acc)), "actual_jerk_rms_mps3": _weighted(jerk, t, True), "actual_jerk_p95_mps3": float(np.percentile(jerk, 95)), "actual_jerk_max_mps3": float(np.max(jerk)), "actual_yaw_rate_rms_radps": _weighted(yaw_rate, t, True), "actual_yaw_rate_max_radps": float(np.max(yaw_rate)), "trajectory_duration_s": float(t[-1] - t[0])}
    return summary, {"t_s": t, "speed_mps": speed, "acc_mps2": acc, "jerk_mps3": jerk, "yaw_rate_radps": samples[:, 14]}


def distance_point_to_polyline(point_xy, polyline_xy):
    point = np.asarray(point_xy, float); line = sanitize_polyline(polyline_xy)
    if len(line) < 2: return float("nan"), -1, float("nan")
    starts, vectors = line[:-1], np.diff(line, axis=0)
    denom = np.sum(vectors * vectors, axis=1)
    ratios = np.clip(np.sum((point - starts) * vectors, axis=1) / denom, 0, 1)
    distances = np.linalg.norm(point - (starts + ratios[:, None] * vectors), axis=1)
    index = int(np.argmin(distances))
    return float(distances[index]), index, float(ratios[index])


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def _nan_trajectory_summary():
    names = ("interplan_position_rmse_m", "interplan_position_max_m", "interplan_velocity_rmse_mps", "interplan_yaw_rmse_rad", "initial_tangent_jump_rad", "initial_speed_jump_mps", "initial_yaw_rate_jump_radps", "common_duration_s")
    return {name: float("nan") for name in names}, {}


def compare_trajectory_prefixes(previous_samples, previous_published_monotonic_s, current_samples, now_monotonic_s, sample_dt):
    if sample_dt <= 0 or previous_samples is None or current_samples is None or previous_published_monotonic_s is None:
        return _nan_trajectory_summary()
    old = np.asarray(previous_samples, float); new = np.asarray(current_samples, float)
    if old.ndim != 2 or new.ndim != 2 or old.shape[1] < 15 or new.shape[1] < 15 or len(old) < 2 or len(new) < 2:
        return _nan_trajectory_summary()
    elapsed = max(0.0, now_monotonic_s - previous_published_monotonic_s)
    old_remaining = old[-1, 0] - elapsed; new_duration = new[-1, 0] - new[0, 0]
    common = min(old_remaining, new_duration)
    if common <= 0: return _nan_trajectory_summary()
    times = np.r_[np.arange(0.0, common, sample_dt), common]
    def sample(data, query):
        return np.column_stack([np.interp(query, data[:, 0], data[:, column]) for column in range(data.shape[1])])
    old_q = sample(old, times + elapsed); new_q = sample(new, times + new[0, 0])
    pos_delta = np.linalg.norm(old_q[:, 1:4] - new_q[:, 1:4], axis=1)
    vel_delta = np.linalg.norm(old_q[:, 4:7] - new_q[:, 4:7], axis=1)
    yaw_delta = wrap_angle(old_q[:, 13] - new_q[:, 13])
    old_tangent = math.atan2(old_q[0, 5], old_q[0, 4]); new_tangent = math.atan2(new_q[0, 5], new_q[0, 4])
    metrics = {
        "interplan_position_rmse_m": float(np.sqrt(np.mean(pos_delta ** 2))), "interplan_position_max_m": float(np.max(pos_delta)),
        "interplan_velocity_rmse_mps": float(np.sqrt(np.mean(vel_delta ** 2))), "interplan_yaw_rmse_rad": float(np.sqrt(np.mean(yaw_delta ** 2))),
        "initial_tangent_jump_rad": float(abs(wrap_angle(old_tangent - new_tangent))),
        "initial_speed_jump_mps": float(abs(np.linalg.norm(old_q[0, 4:7]) - np.linalg.norm(new_q[0, 4:7]))),
        "initial_yaw_rate_jump_radps": float(abs(old_q[0, 14] - new_q[0, 14])), "common_duration_s": float(common),
    }
    return metrics, {"t_s": times, "previous": old_q, "current": new_q, "position_error_m": pos_delta, "yaw_error_rad": yaw_delta}


def compute_command_deltas(commands_vw):
    commands = np.asarray(commands_vw, float)
    if commands.ndim != 2 or commands.shape[1] < 2 or len(commands) < 2:
        return {"command_delta_v_abs_mean_mps": float("nan"), "command_delta_w_abs_mean_radps": float("nan")}
    delta = np.abs(np.diff(commands[:, :2], axis=0))
    return {"command_delta_v_abs_mean_mps": float(np.mean(delta[:, 0])), "command_delta_w_abs_mean_radps": float(np.mean(delta[:, 1]))}


def compute_deadline_metrics(durations_ms, deadline_ms):
    values = np.asarray(durations_ms, float); values = values[np.isfinite(values)]
    if not len(values) or deadline_ms <= 0: return {"count": int(len(values)), "deadline_miss_ratio": float("nan")}
    return {"count": int(len(values)), "deadline_miss_ratio": float(np.mean(values > deadline_ms)), "deadline_ms": float(deadline_ms)}
