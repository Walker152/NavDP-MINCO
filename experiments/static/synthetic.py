from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.static.case_schema import StaticCase
from experiments.static.esdf import signed_distance_from_occupancy
from experiments.static.pathfinding import astar_path


def _load_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid static catalogue config: {error}") from error
    catalogue_reference = value.get("case_catalogue_config")
    if catalogue_reference and not isinstance(value.get("cases"), list):
        reference_path = Path(str(catalogue_reference))
        if not reference_path.is_absolute():
            reference_path = (path.parent / reference_path).resolve()
        referenced = _load_config(reference_path)
        value = {
            **referenced,
            **value,
            "grid": referenced["grid"],
            "cases": referenced["cases"],
        }
    if value.get("schema_version") != 1 or not isinstance(value.get("cases"), list):
        raise ValueError("static catalogue schema_version/cases mismatch")
    return value


def _rasterize_rectangles(
    shape: tuple[int, int],
    origin: np.ndarray,
    resolution: float,
    rectangles: list[list[float]],
) -> np.ndarray:
    occupancy = np.zeros(shape, dtype=bool)
    height, width = shape
    x_centres = origin[0] + (np.arange(width) + 0.5) * resolution
    y_centres = origin[1] + (np.arange(height) + 0.5) * resolution
    occupancy[[0, -1], :] = True
    occupancy[:, [0, -1]] = True
    for rectangle in rectangles:
        if len(rectangle) != 4:
            raise ValueError("obstacle rectangle must be [xmin,ymin,xmax,ymax]")
        xmin, ymin, xmax, ymax = map(float, rectangle)
        if not (xmin < xmax and ymin < ymax):
            raise ValueError("obstacle rectangle has invalid bounds")
        x_mask = (x_centres >= xmin) & (x_centres <= xmax)
        y_mask = (y_centres >= ymin) & (y_centres <= ymax)
        occupancy[np.ix_(y_mask, x_mask)] = True
    return occupancy


def generate_catalogue(path: Path | str) -> list[StaticCase]:
    config = _load_config(path)
    grid = config["grid"]
    resolution = float(grid["resolution_m"])
    origin = np.asarray(grid["origin_xy_m"], dtype=np.float64)
    shape = (int(grid["height_cells"]), int(grid["width_cells"]))
    profile = str(config["constraint_profile"])
    cases: list[StaticCase] = []
    seen: set[str] = set()
    for specification in config["cases"]:
        uid = str(specification["case_uid"])
        if uid in seen:
            raise ValueError(f"duplicate case_uid: {uid}")
        seen.add(uid)
        occupancy = _rasterize_rectangles(
            shape,
            origin,
            resolution,
            specification.get("obstacle_rectangles_xyxy_m", []),
        )
        esdf = signed_distance_from_occupancy(occupancy, resolution)
        guide = np.asarray(specification["guide_path_xyz_m"], dtype=np.float64)

        # If guide path crosses obstacles, use A* to find a safe path
        obstacles = specification.get("obstacle_rectangles_xyxy_m", [])
        if obstacles:
            guide_crosses = False
            for i in range(len(guide) - 1):
                samples = np.linspace(0.0, 1.0, max(2, int(
                    np.linalg.norm(guide[i + 1, :2] - guide[i, :2]) / resolution
                )))
                for t in samples:
                    pt = guide[i] + t * (guide[i + 1] - guide[i])
                    col = int((pt[0] - origin[0]) / resolution)
                    row = int((pt[1] - origin[1]) / resolution)
                    if 0 <= row < shape[0] and 0 <= col < shape[1]:
                        if occupancy[row, col]:
                            guide_crosses = True
                            break
                if guide_crosses:
                    break
            if guide_crosses:
                try:
                    safe_guide = astar_path(
                        occupancy, origin, resolution,
                        guide[0, :2], guide[-1, :2],
                    )
                    if len(safe_guide) >= 2:
                        guide = safe_guide
                except Exception:
                    pass  # keep original guide if A* fails
        state = specification.get("start_state", {})
        terminal = specification.get("terminal_goal_xyz_m", guide[-1].tolist())
        cases.append(
            StaticCase(
                case_uid=uid,
                case_source="STATIC_SYNTHETIC",
                guide_path_xyz=guide,
                occupancy=occupancy,
                esdf_distance=esdf,
                esdf_origin=origin,
                esdf_resolution=resolution,
                start_position=np.asarray(
                    state.get("position_xyz_m", guide[0]), dtype=np.float64
                ),
                start_velocity=np.asarray(
                    state.get("velocity_xyz_mps", [0.0, 0.0, 0.0]),
                    dtype=np.float64,
                ),
                start_acceleration=np.asarray(
                    state.get("acceleration_xyz_mps2", [0.0, 0.0, 0.0]),
                    dtype=np.float64,
                ),
                start_yaw=float(state.get("yaw_rad", 0.0)),
                start_yaw_rate=float(state.get("yaw_rate_radps", 0.0)),
                terminal_goal=np.asarray(terminal, dtype=np.float64),
                constraint_profile=profile,
                expected_category=str(specification["category"]),
                tags=tuple(specification.get("tags", [])),
                frame="world",
                units={"position": "m", "time": "s", "yaw": "rad"},
                auxiliary_arrays={
                    "materialization_obstacle_rectangles_xyxy_m": np.asarray(
                        specification.get("obstacle_rectangles_xyxy_m", []),
                        dtype=np.float64,
                    ).reshape(-1, 4)
                },
            )
        )
    return cases
