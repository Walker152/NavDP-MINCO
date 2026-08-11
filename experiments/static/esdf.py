from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from utils_tasks.esdf_query_utils import EsdfGridView


def signed_distance_from_occupancy(
    occupancy: np.ndarray,
    resolution: float,
) -> np.ndarray:
    """Build a cell-centred signed ESDF; free is positive, occupied negative."""
    raw = np.asarray(occupancy)
    if raw.ndim != 2 or raw.size == 0:
        raise ValueError("occupancy must be a nonempty 2D array")
    if raw.dtype != np.bool_:
        if not np.issubdtype(raw.dtype, np.number) or not np.all(
            np.isfinite(raw)
        ):
            raise ValueError("occupancy must contain finite boolean values")
        if not np.all((raw == 0) | (raw == 1)):
            raise ValueError("occupancy must contain only 0 or 1")
    occupied = raw.astype(bool, copy=False)
    resolution = float(resolution)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution must be finite and positive")
    if not np.any(occupied) or np.all(occupied):
        raise ValueError("occupancy must contain both free and occupied cells")

    free_distance = distance_transform_edt(~occupied) * resolution
    occupied_distance = distance_transform_edt(occupied) * resolution
    distance = free_distance
    distance[occupied] = -occupied_distance[occupied]
    return np.ascontiguousarray(distance, dtype=np.float64)


@dataclass(frozen=True)
class StaticEsdf:
    distance: np.ndarray
    occupancy: np.ndarray
    origin: np.ndarray
    resolution: float

    def __post_init__(self) -> None:
        distance = np.asarray(self.distance, dtype=np.float64)
        occupancy = np.asarray(self.occupancy)
        origin = np.asarray(self.origin, dtype=np.float64)
        resolution = float(self.resolution)
        if (
            distance.ndim != 2
            or distance.size == 0
            or occupancy.shape != distance.shape
        ):
            raise ValueError("distance and occupancy must be matching 2D arrays")
        if occupancy.dtype != np.bool_:
            raise ValueError("occupancy must have bool dtype")
        if not np.all(np.isfinite(distance)):
            raise ValueError("distance must be finite")
        if origin.shape != (2,) or not np.all(np.isfinite(origin)):
            raise ValueError("origin must be a finite (2,) array")
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("resolution must be finite and positive")
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "resolution", resolution)

    @property
    def free(self) -> np.ndarray:
        return np.asarray(~self.occupancy, dtype=np.uint8)

    def query_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return EsdfGridView(
            distance=self.distance,
            origin=self.origin,
            resolution=self.resolution,
        ).query_points(points)
