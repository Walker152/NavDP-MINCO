import math

import numpy as np


def is_hold_trajectory_valid(
    cache,
    episode_generation,
    now,
    validation_safe_dist,
    clearance_query,
):
    if cache is None or int(cache.get("episode_generation", -1)) != int(episode_generation):
        return False
    elapsed = float(now) - float(cache.get("published_time", 0.0))
    duration = float(cache.get("duration", 0.0))
    if not math.isfinite(elapsed) or not math.isfinite(duration) or elapsed < 0.0 or elapsed > duration:
        return False
    points = np.asarray(cache.get("waypoints"), dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return False
    clearance = float(clearance_query(points[:, :2]))
    return math.isfinite(clearance) and clearance > float(validation_safe_dist)
