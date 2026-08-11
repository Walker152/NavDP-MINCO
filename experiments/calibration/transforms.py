from __future__ import annotations

import numpy as np


def quaternion_wxyz_to_matrix(quaternion_wxyz) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must be finite wxyz shape (4,)")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_transform(translation_xyz, quaternion_wxyz) -> np.ndarray:
    translation = np.asarray(translation_xyz, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("translation must be finite shape (3,)")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_to_matrix(quaternion_wxyz)
    transform[:3, 3] = translation
    return transform


def validate_transform(transform) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("transform must be finite shape (4,4)")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
        raise ValueError("transform has invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError("transform rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise ValueError("transform rotation must be right-handed")
    return value


def invert_transform(transform) -> np.ndarray:
    value = validate_transform(transform)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = value[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ value[:3, 3]
    return inverse


def transform_points(transform, points) -> np.ndarray:
    value = validate_transform(transform)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points must be finite shape (...,3)")
    return np.einsum("ij,...j->...i", value[:3, :3], points) + value[:3, 3]
