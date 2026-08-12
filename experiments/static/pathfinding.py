"""A* pathfinding on occupancy grids for static case guide-path generation."""

from __future__ import annotations

import heapq
import math
from typing import Sequence

import numpy as np


def _world_to_grid(
    xy: Sequence[float],
    origin: np.ndarray,
    resolution: float,
    shape: tuple[int, int],
) -> tuple[int, int]:
    """Convert world coordinates (x, y) to grid cell indices (row, col)."""
    col = int((float(xy[0]) - float(origin[0])) / resolution)
    row = int((float(xy[1]) - float(origin[1])) / resolution)
    height, width = shape
    return (max(0, min(height - 1, row)), max(0, min(width - 1, col)))


def _grid_to_world(
    cell: tuple[int, int],
    origin: np.ndarray,
    resolution: float,
) -> tuple[float, float]:
    """Convert grid cell (row, col) to world centre coordinates (x, y)."""
    row, col = cell
    x = float(origin[0]) + (col + 0.5) * resolution
    y = float(origin[1]) + (row + 0.5) * resolution
    return (x, y)


def _inflate_obstacles(
    occupancy: np.ndarray, radius_cells: int
) -> np.ndarray:
    """Dilate occupied cells by a given radius to create a safety margin."""
    if radius_cells <= 0:
        return occupancy.copy()
    from scipy.ndimage import binary_dilation

    structure = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
    return binary_dilation(occupancy, structure=structure)


def _simplify_path(
    waypoints: list[tuple[float, float]],
    epsilon_m: float = 0.15,
) -> list[tuple[float, float]]:
    """Remove redundant colinear waypoints (Ramer-Douglas-Peucker simplification)."""
    if len(waypoints) <= 2:
        return waypoints

    points = np.asarray(waypoints, dtype=np.float64)

    def _rdp(indices: np.ndarray) -> list[int]:
        if len(indices) <= 2:
            return list(indices)
        start, end = points[indices[0]], points[indices[-1]]
        vec = end - start
        length_sq = float(np.dot(vec, vec))
        if length_sq < 1e-12:
            distances = np.linalg.norm(points[indices] - start, axis=1)
        else:
            t = np.clip(
                np.dot(points[indices] - start, vec) / length_sq, 0.0, 1.0
            )
            projections = start + t[:, None] * vec
            distances = np.linalg.norm(points[indices] - projections, axis=1)

        farthest = int(np.argmax(distances))
        if distances[farthest] <= epsilon_m:
            return [indices[0], indices[-1]]

        left = _rdp(indices[: farthest + 1])
        right = _rdp(indices[farthest:])
        return left[:-1] + right

    kept = _rdp(np.arange(len(points)))
    return [(float(p[0]), float(p[1])) for p in points[kept]]


def astar_path(
    occupancy: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    *,
    inflation_cells: int = 2,
    simplify_epsilon_m: float = 0.15,
) -> np.ndarray:
    """Compute a collision-free path from start to goal using A* on the occupancy grid.

    Args:
        occupancy: 2D bool array (True = occupied, False = free), shape (height, width).
        origin: World coordinates (x0, y0) of the grid's bottom-left corner.
        resolution: Cell size in metres.
        start_xy: Start position in world coordinates (x, y).
        goal_xy: Goal position in world coordinates (x, y).
        inflation_cells: Number of cells to dilate obstacles by (safety margin).
        simplify_epsilon_m: Max deviation for path simplification (metres).

    Returns:
        (N, 3) array of waypoints in world coordinates (x, y, 0).
        Returns a 2-point straight line if start and goal have line-of-sight.
    """
    origin_arr = np.asarray(origin, dtype=np.float64)
    shape: tuple[int, int] = (int(occupancy.shape[0]), int(occupancy.shape[1]))
    safe_grid = _inflate_obstacles(occupancy, inflation_cells)

    start_cell = _world_to_grid(start_xy, origin_arr, resolution, shape)
    goal_cell = _world_to_grid(goal_xy, origin_arr, resolution, shape)

    # Quick check: if start or goal is inside an obstacle, return straight line anyway
    if safe_grid[start_cell] or safe_grid[goal_cell]:
        return np.array(
            [
                [float(start_xy[0]), float(start_xy[1]), 0.0],
                [float(goal_xy[0]), float(goal_xy[1]), 0.0],
            ],
            dtype=np.float64,
        )

    # 8-connected neighbours
    neighbours = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    height, width = shape
    open_set: list[tuple[float, int, int, int, int, tuple[float, ...]]] = []
    start_h = math.hypot(goal_cell[0] - start_cell[0], goal_cell[1] - start_cell[1])
    heapq.heappush(open_set, (start_h, 0, start_cell[0], start_cell[1], start_cell[0], start_cell[1], ()))

    g_scores: dict[tuple[int, int], float] = {start_cell: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    visited: set[tuple[int, int]] = set()

    found = False
    goal_node: tuple[int, int] = goal_cell

    while open_set:
        _, g, row, col, parent_r, parent_c, _ = heapq.heappop(open_set)
        current = (row, col)

        if current in visited:
            continue
        visited.add(current)
        came_from[current] = (parent_r, parent_c)

        if current == goal_cell:
            found = True
            break

        for dr, dc in neighbours:
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if safe_grid[nr, nc]:
                continue
            neighbour = (nr, nc)
            if neighbour in visited:
                continue

            move_cost = math.hypot(dr, dc)
            tentative_g = g + move_cost
            existing_g = g_scores.get(neighbour)
            if existing_g is not None and tentative_g >= existing_g:
                continue

            g_scores[neighbour] = tentative_g
            h = math.hypot(goal_cell[0] - nr, goal_cell[1] - nc)
            f = tentative_g + h
            heapq.heappush(open_set, (f, tentative_g, nr, nc, row, col, ()))

    if not found:
        # No path found — return straight line as fallback
        return np.array(
            [
                [float(start_xy[0]), float(start_xy[1]), 0.0],
                [float(goal_xy[0]), float(goal_xy[1]), 0.0],
            ],
            dtype=np.float64,
        )

    # Reconstruct path
    path_cells: list[tuple[int, int]] = []
    node = goal_cell
    safety = 0
    while node != start_cell and safety < 100000:
        path_cells.append(node)
        node = came_from.get(node, start_cell)
        safety += 1
    path_cells.append(start_cell)
    path_cells.reverse()

    # Convert to world coords and simplify
    world_waypoints = [_grid_to_world(c, origin_arr, resolution) for c in path_cells]
    simplified = _simplify_path(world_waypoints, simplify_epsilon_m)

    # Ensure start and goal are exactly at the requested positions
    simplified[0] = (float(start_xy[0]), float(start_xy[1]))
    simplified[-1] = (float(goal_xy[0]), float(goal_xy[1]))

    return np.array(
        [[x, y, 0.0] for x, y in simplified], dtype=np.float64
    )
