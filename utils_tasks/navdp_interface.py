from __future__ import annotations

import numpy as np


def transform_local_paths(
    points,
    camera_position,
    camera_rotation,
    robot_position=None,
) -> np.ndarray:
    """Transform candidates through the declared local-frame world pose.

    ``robot_position`` is accepted only as a compatibility argument. It is
    intentionally not used: translating the rotated points a second time
    breaks the homogeneous transform chain whenever sensor and base origins
    differ.
    """
    local_xy = np.asarray(points, dtype=np.float64)
    if local_xy.ndim < 2 or local_xy.shape[-1] < 2:
        raise ValueError("points must have shape (..., points, >=2)")

    camera_position = np.asarray(camera_position, dtype=np.float64)
    camera_rotation = np.asarray(camera_rotation, dtype=np.float64)
    local_xyz = np.zeros(local_xy.shape[:-1] + (3,), dtype=np.float64)
    local_xyz[..., :2] = local_xy[..., :2]
    world = np.einsum("ij,...j->...i", camera_rotation, local_xyz)
    world += camera_position
    world[..., 2] = 0.0
    return world
