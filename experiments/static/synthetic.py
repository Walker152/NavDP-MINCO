from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.static.case_schema import StaticCase
from experiments.static.esdf import signed_distance_from_occupancy
from experiments.static.pathfinding import astar_path

# Default per-side passage clearance (distance from the guide line to the
# nearest obstacle edge) for obstacle-density variants, keyed by density.
OBSTACLE_VARIANT_DEFAULT_CLEARANCE_M = {
    "sparse": 0.60,
    "dense": 0.32,
}


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


def _build_static_case(
    specification: dict[str, Any],
    grid: dict[str, Any],
    *,
    profile: str,
) -> StaticCase:
    """Build a StaticCase from one catalogue specification entry.

    The guide path is only rewritten with an A* detour when the specification
    explicitly opts in via ``allow_astar_rewrite``; whether a rewrite happened
    is recorded in the case references under ``guide_rewritten``.
    """
    resolution = float(grid["resolution_m"])
    origin = np.asarray(grid["origin_xy_m"], dtype=np.float64)
    shape = (int(grid["height_cells"]), int(grid["width_cells"]))
    occupancy = _rasterize_rectangles(
        shape,
        origin,
        resolution,
        specification.get("obstacle_rectangles_xyxy_m", []),
    )
    esdf = signed_distance_from_occupancy(occupancy, resolution)
    guide = np.asarray(specification["guide_path_xyz_m"], dtype=np.float64)

    # A* guide rewriting is opt-in so expected-failure cases (e.g.
    # guide_through_obstacle) keep their original guide by default.
    obstacles = specification.get("obstacle_rectangles_xyxy_m", [])
    guide_rewritten = False
    if obstacles and specification.get("allow_astar_rewrite", False):
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
                    guide_rewritten = True
            except Exception:
                pass  # keep original guide if A* fails

    state = specification.get("start_state", {})
    terminal = specification.get("terminal_goal_xyz_m", guide[-1].tolist())
    references = {
        str(key): str(value)
        for key, value in specification.get("references", {}).items()
    }
    references["guide_rewritten"] = "true" if guide_rewritten else "false"
    return StaticCase(
        case_uid=str(specification["case_uid"]),
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
        references=references,
        auxiliary_arrays={
            "materialization_obstacle_rectangles_xyxy_m": np.asarray(
                specification.get("obstacle_rectangles_xyxy_m", []),
                dtype=np.float64,
            ).reshape(-1, 4)
        },
    )


def _passage_rectangles(
    guide_xy: np.ndarray,
    clearance_m: float,
    grid: dict[str, Any],
) -> list[list[float]]:
    """Build two axis-aligned rectangles bracketing the guide's extent.

    The rectangles' inner edges lie exactly ``clearance_m`` from the guide's
    bounding box along its narrower axis, so the minimum distance from the
    guide path to any obstacle edge is ``clearance_m`` and the passage stays
    clear of the guide by construction for arbitrarily shaped guides.
    """
    guide_xy = np.asarray(guide_xy, dtype=np.float64)
    if guide_xy.ndim != 2 or guide_xy.shape[1] < 2:
        raise ValueError("guide_xy must have shape (N, 2)")
    xmin_guide, ymin_guide = guide_xy.min(axis=0)
    xmax_guide, ymax_guide = guide_xy.max(axis=0)
    x_extent = xmax_guide - xmin_guide
    y_extent = ymax_guide - ymin_guide

    resolution = float(grid["resolution_m"])
    origin = np.asarray(grid["origin_xy_m"], dtype=np.float64)
    shape = (int(grid["height_cells"]), int(grid["width_cells"]))
    xmin, ymin = origin
    xmax = xmin + shape[1] * resolution
    ymax = ymin + shape[0] * resolution

    if x_extent >= y_extent:
        # Guide runs mostly along x: bracket its y-extent with horizontal slabs.
        rectangles = [
            [xmin, ymin, xmax, ymin_guide - clearance_m],
            [xmin, ymax_guide + clearance_m, xmax, ymax],
        ]
    else:
        # Guide runs mostly along y: bracket its x-extent with vertical slabs.
        rectangles = [
            [xmin, ymin, xmin_guide - clearance_m, ymax],
            [xmax_guide + clearance_m, ymin, xmax, ymax],
        ]
    for rectangle in rectangles:
        x0, y0, x1, y1 = map(float, rectangle)
        if not (x0 < x1 and y0 < y1):
            raise ValueError(
                "obstacle variant passage would produce a degenerate rectangle"
            )
    return rectangles


def generate_obstacle_variants(config: dict[str, Any]) -> list[StaticCase]:
    """Generate obstacle-density variants for sources listed in the config.

    Each entry in ``config["obstacle_variants"]`` names a ``source_case_uid``
    and a list of ``variants`` (``case_uid_suffix``, ``density`` and optional
    ``clearance_target_m``). The source case is cloned with new flanking
    obstacle rectangles; the guide path, start state and ESDF grid parameters
    are preserved. Generated UIDs are ``{source_uid}_{case_uid_suffix}``.
    """
    grid = config["grid"]
    profile = str(config["constraint_profile"])
    source_by_uid = {
        str(specification["case_uid"]): specification
        for specification in config.get("cases", [])
    }
    cases: list[StaticCase] = []
    for entry in config.get("obstacle_variants", []):
        source_uid = str(entry["source_case_uid"])
        if source_uid not in source_by_uid:
            raise ValueError(f"unknown obstacle variant source_case_uid: {source_uid}")
        source = source_by_uid[source_uid]
        guide_xy = np.asarray(source["guide_path_xyz_m"], dtype=np.float64)[:, :2]
        for variant in entry.get("variants", []):
            suffix = str(variant["case_uid_suffix"])
            density = str(variant["density"])
            if density not in OBSTACLE_VARIANT_DEFAULT_CLEARANCE_M:
                raise ValueError(f"unsupported obstacle variant density: {density}")
            clearance = float(
                variant.get(
                    "clearance_target_m",
                    OBSTACLE_VARIANT_DEFAULT_CLEARANCE_M[density],
                )
            )
            if not np.isfinite(clearance) or clearance <= 0.0:
                raise ValueError(
                    f"obstacle variant clearance must be finite and positive: "
                    f"{clearance}"
                )
            explicit_rectangles = variant.get("obstacle_rectangles_xyxy_m")
            rectangles = (
                explicit_rectangles
                if explicit_rectangles is not None
                else _passage_rectangles(guide_xy, clearance, grid)
            )
            variant_specification = {
                **source,
                "case_uid": f"{source_uid}_{suffix}",
                "category": f"{source['category']}_{density}",
                "obstacle_rectangles_xyxy_m": rectangles,
                "tags": list(source.get("tags", []))
                + [density.upper(), "OBSTACLE_VARIANT"],
            }
            # Variants are constructed scenes; they must never inherit an
            # opt-in that would silently rewrite the cloned guide path.
            variant_specification.pop("allow_astar_rewrite", None)
            cases.append(
                _build_static_case(variant_specification, grid, profile=profile)
            )
    return cases


def generate_catalogue(path: Path | str) -> list[StaticCase]:
    config = _load_config(path)
    grid = config["grid"]
    profile = str(config["constraint_profile"])
    cases: list[StaticCase] = []
    seen: set[str] = set()
    for specification in config["cases"]:
        uid = str(specification["case_uid"])
        if uid in seen:
            raise ValueError(f"duplicate case_uid: {uid}")
        seen.add(uid)
        cases.append(_build_static_case(specification, grid, profile=profile))
    for variant_case in generate_obstacle_variants(config):
        if variant_case.case_uid in seen:
            raise ValueError(f"duplicate case_uid: {variant_case.case_uid}")
        seen.add(variant_case.case_uid)
        cases.append(variant_case)
    return cases
