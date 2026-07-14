import numpy as np


def compute_raw_episode_metrics(initial_distance, trajectory_length, success):
    success_value = float(bool(success)); length = float(trajectory_length)
    spl = float(np.clip(float(initial_distance) / length, 0, 1) * success_value) if length > 0 else 0.0
    return {"success":success_value, "spl":spl, "distance":float(initial_distance)}
