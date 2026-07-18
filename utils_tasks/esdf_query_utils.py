from dataclasses import dataclass
import math
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

    def query_points(self, points) -> tuple[np.ndarray, np.ndarray]:
        """Query cell-centred ESDF samples with bilinear interpolation."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        count = len(points) if points.ndim == 2 else 0
        values = np.full(count, np.nan, dtype=np.float64)
        valid = np.zeros(count, dtype=bool)
        if (
            points.ndim != 2
            or points.shape[1] < 2
            or self.distance.ndim != 2
            or self.distance.size == 0
            or self.origin.size < 2
            or not np.isfinite(self.resolution)
            or self.resolution <= 0.0
        ):
            return values, valid

        raw = (points[:, :2] - self.origin[:2]) / self.resolution
        height, width = self.distance.shape
        finite = np.all(np.isfinite(raw), axis=1)
        inside = (
            finite
            & (raw[:, 0] >= 0.0)
            & (raw[:, 0] < width)
            & (raw[:, 1] >= 0.0)
            & (raw[:, 1] < height)
        )
        if not np.any(inside):
            return values, valid

        center = raw[inside] - 0.5
        x = np.clip(center[:, 0], 0.0, max(0, width - 1))
        y = np.clip(center[:, 1], 0.0, max(0, height - 1))
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        tx = x - x0
        ty = y - y0
        d00 = self.distance[y0, x0]
        d10 = self.distance[y0, x1]
        d01 = self.distance[y1, x0]
        d11 = self.distance[y1, x1]
        interpolated = (
            (1.0 - tx) * (1.0 - ty) * d00
            + tx * (1.0 - ty) * d10
            + (1.0 - tx) * ty * d01
            + tx * ty * d11
        )
        finite_values = np.isfinite(interpolated)
        indices = np.flatnonzero(inside)
        values[indices[finite_values]] = interpolated[finite_values]
        valid[indices[finite_values]] = True
        return values, valid

    def sample_polyline(self, points, max_step=None) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[0] == 0
            or points.shape[1] < 2
            or not np.all(np.isfinite(points[:, :2]))
        ):
            return np.empty((0, 2), dtype=np.float64)
        points = points[:, :2]
        if len(points) == 1:
            return points.copy()
        step = self.resolution * 0.5 if max_step is None else float(max_step)
        if not np.isfinite(step) or step <= 0.0:
            step = self.resolution * 0.5
        step = max(step, 1e-6)
        samples = [points[0]]
        for start, end in zip(points[:-1], points[1:]):
            length = float(np.linalg.norm(end - start))
            count = max(1, int(math.ceil(length / step)))
            for index in range(1, count + 1):
                samples.append(start + (end - start) * (index / count))
        return np.asarray(samples, dtype=np.float64)

    def inspect_polyline(
        self,
        points,
        safe_dist,
        start_exemption_radius=0.0,
        max_step=None,
    ) -> dict:
        samples = self.sample_polyline(points, max_step=max_step)
        if len(samples) == 0:
            return {
                "valid": False,
                "safe": False,
                "reason": "INVALID_PATH",
                "min_clearance": float("nan"),
                "unsafe_ratio": 1.0,
                "oob_ratio": 1.0,
                "oob_count": 0,
                "sample_count": 0,
                "start_exempt_count": 0,
            }
        values, valid = self.query_points(samples)
        return self._summarize_polyline(
            samples, values, valid, safe_dist, start_exemption_radius
        )

    @staticmethod
    def _summarize_polyline(
        samples,
        values,
        valid,
        safe_dist,
        start_exemption_radius,
    ) -> dict:
        oob_count = int(np.sum(~valid))
        start_radius = max(0.0, float(start_exemption_radius))
        start_exempt = np.linalg.norm(samples - samples[0], axis=1) <= start_radius + 1e-12
        negative = valid & (values < 0.0)
        clearance_failure = valid & ~start_exempt & (values <= float(safe_dist))
        unsafe = (~valid) | negative | clearance_failure
        if oob_count:
            reason = "ESDF_OOB"
        elif np.any(negative):
            reason = "NEGATIVE_ESDF"
        elif np.any(clearance_failure):
            reason = "CLEARANCE"
        else:
            reason = "NONE"
        return {
            "valid": bool(np.all(valid)),
            "safe": bool(not np.any(unsafe)),
            "reason": reason,
            "min_clearance": float(np.min(values[valid])) if np.any(valid) else float("nan"),
            "unsafe_ratio": float(np.mean(unsafe)),
            "oob_ratio": float(np.mean(~valid)),
            "oob_count": oob_count,
            "sample_count": int(len(samples)),
            "start_exempt_count": int(np.sum(start_exempt & valid)),
        }

    def inspect_polylines(
        self,
        paths,
        safe_dist,
        start_exemption_radius=0.0,
        max_step=None,
    ) -> list[dict]:
        """Inspect candidates with one ESDF query while preserving scalar semantics."""
        samples = [self.sample_polyline(path, max_step=max_step) for path in paths]
        nonempty = [sample for sample in samples if len(sample)]
        if nonempty:
            all_samples = np.concatenate(nonempty, axis=0)
            all_values, all_valid = self.query_points(all_samples)
        else:
            all_values = np.empty(0, dtype=np.float64)
            all_valid = np.empty(0, dtype=bool)

        reports = []
        offset = 0
        invalid = {
            "valid": False,
            "safe": False,
            "reason": "INVALID_PATH",
            "min_clearance": float("nan"),
            "unsafe_ratio": 1.0,
            "oob_ratio": 1.0,
            "oob_count": 0,
            "sample_count": 0,
            "start_exempt_count": 0,
        }
        for sample in samples:
            count = len(sample)
            if not count:
                reports.append(dict(invalid))
                continue
            reports.append(
                self._summarize_polyline(
                    sample,
                    all_values[offset : offset + count],
                    all_valid[offset : offset + count],
                    safe_dist,
                    start_exemption_radius,
                )
            )
            offset += count
        return reports

    def query_polyline(self, points) -> float:
        samples = self.sample_polyline(points)
        if len(samples) == 0:
            return float("nan")
        values, valid = self.query_points(samples)
        if not np.all(valid):
            return float("nan")
        return float(np.min(values))


def query_esdf_polyline(esdf, points) -> float:
    if esdf is None or points is None:
        return float("nan")
    return EsdfGridView.from_mapping(esdf).query_polyline(points)
