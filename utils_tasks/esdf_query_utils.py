from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class EsdfGridView:
    distance: np.ndarray
    origin: np.ndarray
    resolution: float

    @classmethod
    def from_mapping(cls, esdf: Mapping):
        return cls(
            distance=np.asarray(esdf["distance"], dtype=np.float64),
            origin=np.asarray(esdf["origin"], dtype=np.float64),
            resolution=float(esdf["resolution"]),
        )

    def query_polyline(self, points) -> float:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
            return float("nan")
        cells = np.floor((points[:, :2] - self.origin) / self.resolution)
        finite = np.all(np.isfinite(cells), axis=1)
        cells_safe = np.where(finite[:, None], cells, -1).astype(np.int64)
        mx = cells_safe[:, 0]
        my = cells_safe[:, 1]
        valid = (
            finite
            & (mx >= 0)
            & (mx < self.distance.shape[1])
            & (my >= 0)
            & (my < self.distance.shape[0])
        )
        if not np.any(valid):
            return float("nan")
        return float(np.min(self.distance[my[valid], mx[valid]]))


def query_esdf_polyline(esdf, points) -> float:
    if esdf is None or points is None:
        return float("nan")
    return EsdfGridView.from_mapping(esdf).query_polyline(points)
