import os
import time

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, distance_transform_edt


class SimEsdfBuilder:
    REQUIRED_FIELDS = (
        "distance",
        "occupied",
        "free",
        "origin",
        "resolution",
        "ground_z",
        "scene_scale",
        "padding",
        "obstacle_min_height",
        "obstacle_max_height",
        "ground_quantile",
        "fill_footprint",
        "footprint_inflate_cells",
    )

    def __init__(
        self,
        resolution=0.05,
        padding=1.0,
        robot_radius=0.25,
        safe_dist=0.30,
        ground_quantile=0.05,
        obstacle_min_height=0.08,
        obstacle_max_height=1.50,
        cache_name="esdf_2d.npz",
        force_rebuild=False,
        fill_footprint=True,
        footprint_inflate_cells=1,
    ):
        self.resolution = float(resolution)
        self.padding = float(padding)
        self.robot_radius = float(robot_radius)
        self.safe_dist = float(safe_dist)
        self.ground_quantile = float(ground_quantile)
        self.obstacle_min_height = float(obstacle_min_height)
        self.obstacle_max_height = float(obstacle_max_height)
        self.cache_name = cache_name
        self.force_rebuild = bool(force_rebuild)
        self.fill_footprint = bool(fill_footprint)
        self.footprint_inflate_cells = int(footprint_inflate_cells)

    def build_or_load_from_stage(
        self,
        stage,
        scene_path: str,
        scene_scale: float = 1.0,
        env_prim_path: str = "/World/Scene/terrain",
    ) -> dict:
        total_start = time.perf_counter()
        print(f"[SimESDF] search env_prim_path={env_prim_path}")
        cache_path = os.path.join(scene_path, self.cache_name)
        cache_start = time.perf_counter()
        cached = self._load_cache(cache_path, scene_scale)
        cache_ms = (time.perf_counter() - cache_start) * 1000.0
        if cached is not None and not self.force_rebuild:
            cached["timing"] = {
                "source": "cache",
                "cache_load_ms": cache_ms,
                "total_ms": (time.perf_counter() - total_start) * 1000.0,
            }
            self._print_debug(cached, cache_path, "cache")
            print(
                f"[SimESDF][Timing] source=cache cache_load_ms={cache_ms:.2f} "
                f"total_ms={cached['timing']['total_ms']:.2f}"
            )
            return cached
        if stage is None:
            raise RuntimeError("SimEsdfBuilder requires a loaded USD stage when no valid cache exists")

        extract_start = time.perf_counter()
        points, triangles = self._extract_static_mesh(stage, env_prim_path, scene_scale)
        extract_mesh_ms = (time.perf_counter() - extract_start) * 1000.0
        if points.size == 0 or not triangles:
            raise RuntimeError(f"No static mesh triangles found under {env_prim_path}")

        bounds_start = time.perf_counter()
        xy_min = points[:, :2].min(axis=0) - self.padding
        xy_max = points[:, :2].max(axis=0) + self.padding
        origin = xy_min.astype(np.float64)
        size = np.maximum(3, np.ceil((xy_max - xy_min) / self.resolution).astype(int))
        width, height = int(size[0]), int(size[1])
        occupied = np.zeros((height, width), dtype=bool)

        ground_z = float(np.quantile(points[:, 2], self.ground_quantile))
        z_low = ground_z + self.obstacle_min_height
        z_high = ground_z + self.obstacle_max_height
        spacing = self.resolution * 0.5
        bounds_ms = (time.perf_counter() - bounds_start) * 1000.0

        raster_start = time.perf_counter()
        for tri in triangles:
            tri_z_min = float(np.min(tri[:, 2]))
            tri_z_max = float(np.max(tri[:, 2]))
            if tri_z_max < z_low or tri_z_min > z_high:
                continue
            for sample in self.sample_triangle(tri[0], tri[1], tri[2], spacing):
                if sample[2] < z_low or sample[2] > z_high:
                    continue
                mx = int(np.floor((sample[0] - origin[0]) / self.resolution))
                my = int(np.floor((sample[1] - origin[1]) / self.resolution))
                if 0 <= mx < width and 0 <= my < height:
                    occupied[my, mx] = True

        if self.fill_footprint:
            occupied = binary_closing(occupied, structure=np.ones((3, 3), dtype=bool))
            occupied = binary_fill_holes(occupied)

        if self.footprint_inflate_cells > 0:
            occupied = binary_dilation(
                occupied,
                structure=np.ones((3, 3), dtype=bool),
                iterations=int(self.footprint_inflate_cells),
            )

        occupied[0, :] = True
        occupied[-1, :] = True
        occupied[:, 0] = True
        occupied[:, -1] = True
        rasterize_ms = (time.perf_counter() - raster_start) * 1000.0

        dt_start = time.perf_counter()
        outside_dist = distance_transform_edt(~occupied) * self.resolution
        inside_dist = distance_transform_edt(occupied) * self.resolution
        distance = outside_dist.astype(np.float64)
        distance[occupied] = -inside_dist[occupied]
        free = distance > 0.0
        distance_transform_ms = (time.perf_counter() - dt_start) * 1000.0

        esdf = {
            "distance": distance,
            "occupied": occupied,
            "free": free,
            "origin": origin,
            "resolution": self.resolution,
            "ground_z": ground_z,
            "scene_scale": float(scene_scale),
            "padding": self.padding,
            "obstacle_min_height": self.obstacle_min_height,
            "obstacle_max_height": self.obstacle_max_height,
            "ground_quantile": self.ground_quantile,
            "fill_footprint": self.fill_footprint,
            "footprint_inflate_cells": self.footprint_inflate_cells,
        }
        cache_write_start = time.perf_counter()
        save_esdf = {k: v for k, v in esdf.items() if k != "timing"}
        np.savez(cache_path, **save_esdf)
        cache_write_ms = (time.perf_counter() - cache_write_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        esdf["timing"] = {
            "source": "stage",
            "cache_load_ms": cache_ms,
            "extract_mesh_ms": extract_mesh_ms,
            "bounds_ms": bounds_ms,
            "rasterize_ms": rasterize_ms,
            "distance_transform_ms": distance_transform_ms,
            "cache_write_ms": cache_write_ms,
            "total_ms": total_ms,
        }
        self._print_debug(esdf, cache_path, "stage")
        print(
            "[SimESDF][Timing] "
            f"source=stage extract={extract_mesh_ms:.2f}ms "
            f"bounds={bounds_ms:.2f}ms raster={rasterize_ms:.2f}ms "
            f"dt={distance_transform_ms:.2f}ms cache_write={cache_write_ms:.2f}ms "
            f"total={total_ms:.2f}ms"
        )
        return esdf

    def query_grid(self, esdf: dict, pos_xy: np.ndarray):
        pos_xy = np.asarray(pos_xy, dtype=np.float64).reshape(-1)
        origin = np.asarray(esdf["origin"], dtype=np.float64)
        resolution = float(esdf["resolution"])
        distance = np.asarray(esdf["distance"])
        if pos_xy.size < 2 or not np.all(np.isfinite(pos_xy[:2])):
            return False, float("nan")
        mx = int(np.floor((pos_xy[0] - origin[0]) / resolution))
        my = int(np.floor((pos_xy[1] - origin[1]) / resolution))
        if my < 0 or my >= distance.shape[0] or mx < 0 or mx >= distance.shape[1]:
            return False, float("nan")
        return True, float(distance[my, mx])

    @staticmethod
    def sample_triangle(v0, v1, v2, spacing):
        v0 = np.asarray(v0, dtype=np.float64)
        v1 = np.asarray(v1, dtype=np.float64)
        v2 = np.asarray(v2, dtype=np.float64)
        spacing = max(float(spacing), 1e-3)
        max_edge = max(np.linalg.norm(v1 - v0), np.linalg.norm(v2 - v1), np.linalg.norm(v0 - v2))
        n = max(1, int(np.ceil(max_edge / spacing)))
        samples = [v0, v1, v2]
        for i in range(n + 1):
            for j in range(n + 1 - i):
                a = i / n
                b = j / n
                c = 1.0 - a - b
                samples.append(a * v0 + b * v1 + c * v2)
        return samples

    def _load_cache(self, cache_path, scene_scale):
        if not os.path.exists(cache_path) or self.force_rebuild:
            return None
        try:
            data = np.load(cache_path, allow_pickle=False)
            if not all(field in data.files for field in self.REQUIRED_FIELDS):
                return None
            resolution = float(np.asarray(data["resolution"]))
            cached_scene_scale = float(np.asarray(data["scene_scale"]))
            cache_params = {
                "padding": self.padding,
                "obstacle_min_height": self.obstacle_min_height,
                "obstacle_max_height": self.obstacle_max_height,
                "ground_quantile": self.ground_quantile,
                "footprint_inflate_cells": self.footprint_inflate_cells,
            }
            for key, expected in cache_params.items():
                if not np.isclose(float(np.asarray(data[key])), float(expected)):
                    return None
            cached_fill_footprint = bool(np.asarray(data["fill_footprint"]))
            if (
                not np.isclose(resolution, self.resolution)
                or not np.isclose(cached_scene_scale, scene_scale)
                or cached_fill_footprint != self.fill_footprint
            ):
                return None
            return {
                "distance": np.asarray(data["distance"], dtype=np.float64),
                "occupied": np.asarray(data["occupied"]).astype(bool),
                "free": np.asarray(data["free"]).astype(bool),
                "origin": np.asarray(data["origin"], dtype=np.float64),
                "resolution": resolution,
                "ground_z": float(np.asarray(data["ground_z"])),
                "scene_scale": cached_scene_scale,
                "padding": float(np.asarray(data["padding"])),
                "obstacle_min_height": float(np.asarray(data["obstacle_min_height"])),
                "obstacle_max_height": float(np.asarray(data["obstacle_max_height"])),
                "ground_quantile": float(np.asarray(data["ground_quantile"])),
                "fill_footprint": cached_fill_footprint,
                "footprint_inflate_cells": int(np.asarray(data["footprint_inflate_cells"])),
            }
        except Exception as exc:
            print(f"[SimESDF] cache load failed: {exc}")
            return None

    def _extract_static_mesh(self, stage, env_prim_path, scene_scale):
        from pxr import UsdGeom

        xform_cache = UsdGeom.XformCache()
        all_points = []
        triangles = []
        skipped_tokens = ("robot", "camera", "goal", "marker", "light", "sensor", "contact")

        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if env_prim_path not in path:
                continue
            lower = path.lower()
            if any(token in lower for token in skipped_tokens):
                continue
            if not prim.IsA(UsdGeom.Mesh):
                continue

            mesh = UsdGeom.Mesh(prim)
            points_attr = mesh.GetPointsAttr().Get()
            counts_attr = mesh.GetFaceVertexCountsAttr().Get()
            indices_attr = mesh.GetFaceVertexIndicesAttr().Get()
            if points_attr is None or counts_attr is None or indices_attr is None:
                continue
            points = np.asarray(points_attr, dtype=np.float64)
            face_counts = np.asarray(counts_attr, dtype=np.int64)
            face_indices = np.asarray(indices_attr, dtype=np.int64)
            if points.size == 0 or face_counts.size == 0 or face_indices.size == 0:
                continue

            tf = np.array(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64).T
            points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
            points_w = (points_h @ tf.T)[:, :3]
            if scene_scale != 1.0:
                # Loaded stages are usually already scaled. Keep this disabled unless a caller explicitly
                # supplies a stage with unscaled points and expects metadata matching.
                pass

            all_points.append(points_w)
            cursor = 0
            for count in face_counts:
                face = face_indices[cursor:cursor + count]
                cursor += count
                if count < 3:
                    continue
                for k in range(1, count - 1):
                    triangles.append(points_w[[face[0], face[k], face[k + 1]], :])

        if not all_points:
            return np.zeros((0, 3), dtype=np.float64), []
        points = np.concatenate(all_points, axis=0)
        print(
            "[SimESDF] mesh bounds "
            f"x=[{points[:, 0].min():.3f},{points[:, 0].max():.3f}] "
            f"y=[{points[:, 1].min():.3f},{points[:, 1].max():.3f}] "
            f"z=[{points[:, 2].min():.3f},{points[:, 2].max():.3f}]"
        )
        return points, triangles

    def _print_debug(self, esdf, cache_path, source):
        distance = np.asarray(esdf["distance"])
        occupied = np.asarray(esdf["occupied"]).astype(bool)
        free = np.asarray(esdf["free"]).astype(bool)
        origin = np.asarray(esdf["origin"], dtype=np.float64)
        resolution = float(esdf["resolution"])
        height, width = distance.shape
        x_max = origin[0] + width * resolution
        y_max = origin[1] + height * resolution
        print(f"[SimESDF] source={source}")
        print(f"[SimESDF] cache={cache_path}")
        print(f"[SimESDF] shape=({height},{width}) resolution={resolution:.3f}")
        print(f"[SimESDF] origin=({origin[0]:.3f},{origin[1]:.3f}) ground_z={float(esdf['ground_z']):.3f}")
        print(f"[SimESDF] occupied={int(occupied.sum())} free={int(free.sum())}")
        print(f"[SimESDF] bounds x=[{origin[0]:.3f},{x_max:.3f}] y=[{origin[1]:.3f},{y_max:.3f}]")
