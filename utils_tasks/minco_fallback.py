import math

import numpy as np


def is_hold_trajectory_valid(
    cache,
    episode_generation,
    now,
    validation_safe_dist,
    clearance_query,
    start_exemption_radius=0.0,
):
    if cache is None or int(cache.get("episode_generation", -1)) != int(episode_generation):
        return False
    elapsed = float(now) - float(cache.get("published_time", 0.0))
    duration = float(cache.get("duration", 0.0))
    if not math.isfinite(elapsed) or not math.isfinite(duration) or elapsed < 0.0 or elapsed > duration:
        return False
    samples = np.asarray(cache.get("samples"), dtype=np.float64)
    if samples.ndim == 2 and samples.shape[0] >= 2 and samples.shape[1] >= 3:
        times = samples[:, 0]
        if np.all(np.isfinite(times)) and np.all(np.diff(times) >= 0.0):
            query_time = float(np.clip(elapsed, times[0], times[-1]))
            current = np.array([
                np.interp(query_time, times, samples[:, 1]),
                np.interp(query_time, times, samples[:, 2]),
            ])
            remaining = samples[times > query_time, 1:3]
            points = np.vstack((current, remaining))
        else:
            points = np.asarray(cache.get("waypoints"), dtype=np.float64)
    else:
        points = np.asarray(cache.get("waypoints"), dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return False
    try:
        report = clearance_query(
            points[:, :2],
            safe_dist=float(validation_safe_dist),
            start_exemption_radius=float(start_exemption_radius),
        )
    except TypeError:
        report = clearance_query(points[:, :2])
    if isinstance(report, dict):
        return bool(report.get("safe", False))
    clearance = float(report)
    return math.isfinite(clearance) and clearance > float(validation_safe_dist)
