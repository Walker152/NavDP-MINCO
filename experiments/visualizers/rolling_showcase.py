"""Paper-grade visual packages built only from recorded rolling evidence."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np


GUIDE_STYLE = {"color": "#4B5563", "linestyle": "-.", "linewidth": 1.6}
LEGACY_STYLE = {"color": "#D55E00", "linestyle": "--", "linewidth": 2.4}
SAFE_STYLE = {"color": "#0072B2", "linestyle": "-", "linewidth": 3.0}
GOAL_COLOR = "#7E22CE"
OBSTACLE_COLOR = "#991B1B"
VELOCITY_COLOR = "#009E73"
FRAME_DURATION_S = 0.15
TERMINAL_HOLD_FRAMES = 4

FIGURE_DATA_FIELDS = (
    "record_type",
    "scenario_uid",
    "panel",
    "method",
    "frame_index",
    "time_s",
    "cycle_index",
    "sample_index",
    "x_m",
    "y_m",
    "yaw_rad",
    "velocity_x_mps",
    "velocity_y_mps",
    "segment_start_x_m",
    "segment_start_y_m",
    "segment_end_x_m",
    "segment_end_y_m",
    "corridor_radius_m",
    "obstacle_uid",
    "obstacle_radius_m",
    "status",
    "data_availability",
)


@dataclass(frozen=True)
class RollingScenePackage:
    output_dir: Path
    files: tuple[Path, ...]
    manifest_path: Path
    receipt_path: Path
    validation_path: Path


def _value(source: object, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _first(source: object, keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        value = _value(source, key, None)
        if value is not None:
            return value
    return default


def _array(value: object, *, columns: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, columns or 0), dtype=np.float64)
    if array.ndim != 2 or (columns is not None and array.shape[1] < columns):
        raise ValueError("recorded array has an incompatible shape")
    if not np.all(np.isfinite(array)):
        raise ValueError("recorded visualization evidence must be finite")
    return array


def _samples(result: object) -> np.ndarray:
    return _array(_value(result, "executed_samples", np.empty((0, 15))))


def _sample_xy(samples: np.ndarray) -> np.ndarray:
    if samples.shape[1] >= 15:
        return samples[:, 1:3]
    if samples.shape[1] >= 2:
        return samples[:, :2]
    raise ValueError("executed samples do not contain XY evidence")


def _sample_time(samples: np.ndarray) -> np.ndarray:
    if samples.shape[1] >= 15:
        return samples[:, 0]
    return np.arange(len(samples), dtype=float)


def _sample_yaw(samples: np.ndarray) -> np.ndarray:
    if samples.shape[1] >= 15:
        return samples[:, 13]
    return np.full(len(samples), np.nan)


def _sample_velocity(samples: np.ndarray) -> np.ndarray:
    if samples.shape[1] >= 15:
        return samples[:, 4:6]
    return np.full((len(samples), 2), np.nan)


def _state_xy(state: object) -> np.ndarray:
    value = _first(state, ("position_xyz", "position_xyz_m", "position"))
    array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) < 2 or not np.all(np.isfinite(array[:2])):
        raise ValueError("recorded robot state lacks finite XY")
    return array[:2]


def _state_yaw(state: object) -> float | None:
    value = _first(state, ("yaw_rad", "yaw"))
    if value is None:
        return None
    yaw = float(value)
    return yaw if math.isfinite(yaw) else None


def _state_velocity(state: object) -> np.ndarray | None:
    value = _first(
        state,
        ("velocity_xyz_mps", "velocity_xy_mps", "velocity", "linear_velocity"),
    )
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) < 2 or not np.all(np.isfinite(array[:2])):
        return None
    return array[:2]


def _cycles(result: object) -> tuple[object, ...]:
    values = _value(result, "cycles", ())
    return tuple(values or ())


def _guide(paired: object, safe: object) -> np.ndarray:
    direct = _first(
        paired,
        ("guide_path_xyz", "guide_path", "global_guide_xyz", "reference_path_xyz"),
    )
    if direct is not None:
        return _array(direct, columns=2)[:, :2]
    pieces = []
    for cycle in _cycles(safe):
        local = _first(cycle, ("local_guide_xyz", "guide_path_xyz"))
        if local is not None:
            pieces.append(_array(local, columns=2)[:, :2])
    if not pieces:
        raise ValueError("rolling evidence contains no recorded guide")
    points = np.concatenate(pieces, axis=0)
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10
    return points[keep]


def _goal(paired: object, safe: object, guide: np.ndarray) -> np.ndarray:
    value = _first(paired, ("final_goal_xyz", "goal_xyz", "final_goal"))
    if value is None:
        value = _first(safe, ("final_goal_xyz", "goal_xyz", "final_goal"))
    if value is None:
        return guide[-1].copy()
    array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) < 2 or not np.all(np.isfinite(array[:2])):
        raise ValueError("recorded final goal lacks finite XY")
    return array[:2]


def _initial_state(paired: object, legacy: object, safe: object) -> object:
    direct = _value(paired, "initial_state", None)
    if direct is not None:
        return direct
    for result in (legacy, safe):
        cycles = _cycles(result)
        if cycles:
            state = _value(cycles[0], "input_state", None)
            if state is not None:
                return state
    raise ValueError("rolling evidence contains no initial state")


def _normalise_pair(paired: object) -> tuple[str, object, object]:
    scenario_uid = str(_value(paired, "scenario_uid", "")).strip()
    legacy = _first(paired, ("legacy", "legacy_result"))
    safe = _first(paired, ("safe_corridor_v1", "safe", "safe_result"))
    if legacy is None or safe is None:
        try:
            values = tuple(paired)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("paired rolling evidence lacks legacy and safe results") from error
        if len(values) != 2:
            raise ValueError("paired rolling evidence must contain two methods")
        legacy, safe = values
    if not scenario_uid:
        scenario_uid = str(_value(legacy, "scenario_uid", "")).strip()
    if not scenario_uid:
        raise ValueError("rolling evidence lacks scenario_uid")
    return scenario_uid, legacy, safe


def _corridors(safe: object) -> list[tuple[float, float, float, float, float, int]]:
    output: list[tuple[float, float, float, float, float, int]] = []
    observed: set[tuple[float, ...]] = set()
    for fallback_index, cycle in enumerate(_cycles(safe)):
        cycle_index = int(_value(cycle, "cycle_index", fallback_index))
        values = _value(cycle, "corridor_segments", ())
        if isinstance(values, np.ndarray):
            array = _array(values)
            for row in array:
                if len(row) >= 7:
                    x0, y0, x1, y1, radius = row[0], row[1], row[3], row[4], row[6]
                elif len(row) >= 5:
                    x0, y0, x1, y1, radius = row[:5]
                else:
                    raise ValueError("recorded corridor segment must contain endpoints and radius")
                item = (float(x0), float(y0), float(x1), float(y1), float(radius), cycle_index)
                key = tuple(round(value, 10) for value in item[:-1])
                if key not in observed:
                    observed.add(key)
                    output.append(item)
            continue
        for segment in values or ():
            start = np.asarray(
                _first(segment, ("start_xyz", "start_xy", "start")), dtype=float
            ).reshape(-1)
            end = np.asarray(
                _first(segment, ("end_xyz", "end_xy", "end")), dtype=float
            ).reshape(-1)
            radius = float(_first(segment, ("radius_m", "radius")))
            if len(start) < 2 or len(end) < 2:
                raise ValueError("recorded corridor segment lacks XY endpoints")
            item = (float(start[0]), float(start[1]), float(end[0]), float(end[1]), radius, cycle_index)
            key = tuple(round(value, 10) for value in item[:-1])
            if key not in observed:
                observed.add(key)
                output.append(item)
    if not output or any(not math.isfinite(row[4]) or row[4] <= 0.0 for row in output):
        raise ValueError("safe corridor figure requires real corridor evidence")
    return output


def _obstacles(safe: object) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for fallback_index, cycle in enumerate(_cycles(safe)):
        cycle_index = int(_value(cycle, "cycle_index", fallback_index))
        for obstacle in _value(cycle, "obstacle_states", ()) or ():
            position = np.asarray(
                _first(
                    obstacle,
                    (
                        "position_xy_m",
                        "center_xy",
                        "position_xyz",
                        "position_xy",
                        "position",
                    ),
                ),
                dtype=float,
            ).reshape(-1)
            velocity_value = _first(
                obstacle,
                ("velocity_xy_mps", "velocity_xyz_mps", "velocity_xy", "velocity"),
                [0.0, 0.0],
            )
            velocity = np.asarray(velocity_value, dtype=float).reshape(-1)
            radius = float(_first(obstacle, ("radius_m", "radius"), 0.0))
            if len(position) < 2 or len(velocity) < 2 or radius <= 0.0:
                raise ValueError("recorded obstacle evidence is incomplete")
            output.append(
                {
                    "cycle_index": cycle_index,
                    "obstacle_uid": str(
                        _value(obstacle, "obstacle_uid", f"obstacle-{len(output)}")
                    ),
                    "x_m": float(position[0]),
                    "y_m": float(position[1]),
                    "vx_mps": float(velocity[0]),
                    "vy_mps": float(velocity[1]),
                    "radius_m": radius,
                }
            )
    return output


def _static_rectangles(paired: object) -> list[dict[str, object]]:
    output = []
    for index, rectangle in enumerate(_value(paired, "static_rectangles", ()) or ()):
        bounds = np.asarray(
            _first(rectangle, ("bounds_xy", "bounds_xyxy_m", "bounds")),
            dtype=float,
        ).reshape(-1)
        if len(bounds) != 4 or not np.all(np.isfinite(bounds)):
            raise ValueError("static rectangle evidence requires finite xyxy bounds")
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError("static rectangle evidence requires positive area")
        output.append(
            {
                "obstacle_uid": str(_value(rectangle, "obstacle_uid", f"wall-{index}")),
                "bounds_xy": bounds,
            }
        )
    return output


def _draw_static_rectangles(
    ax: plt.Axes, rectangles: Sequence[Mapping[str, object]]
) -> None:
    for rectangle in rectangles:
        x0, y0, x1, y1 = np.asarray(rectangle["bounds_xy"], dtype=float)
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="#374151",
                edgecolor="#111827",
                alpha=0.75,
                zorder=4,
            )
        )


def draw_heading_arrow(
    ax: plt.Axes,
    xy: Sequence[float],
    yaw_rad: float,
    *,
    color: str,
    length_m: float,
    hollow: bool = False,
    linewidth: float = 2.0,
    label: str | None = None,
) -> None:
    dx = length_m * math.cos(float(yaw_rad))
    dy = length_m * math.sin(float(yaw_rad))
    ax.arrow(
        float(xy[0]),
        float(xy[1]),
        dx,
        dy,
        facecolor="none" if hollow else color,
        edgecolor=color,
        linewidth=linewidth,
        length_includes_head=True,
        head_width=max(0.04, 0.22 * length_m),
        head_length=max(0.06, 0.30 * length_m),
        zorder=9,
        label=label,
    )


def _draw_velocity(
    ax: plt.Axes, xy: np.ndarray, velocity: np.ndarray | None, *, label: str | None = None
) -> bool:
    if velocity is None or not np.all(np.isfinite(velocity)):
        return False
    norm = float(np.linalg.norm(velocity))
    if norm <= 1e-12:
        ax.scatter(xy[0], xy[1], marker="x", color=VELOCITY_COLOR, label=label, zorder=9)
        return True
    scale = 0.55 / max(1.0, norm)
    ax.arrow(
        xy[0],
        xy[1],
        velocity[0] * scale,
        velocity[1] * scale,
        color=VELOCITY_COLOR,
        linewidth=1.0,
        head_width=0.08,
        length_includes_head=True,
        zorder=8,
        label=label,
    )
    return True


def _draw_capsule(
    ax: plt.Axes,
    segment: tuple[float, float, float, float, float, int],
    *,
    alpha: float = 0.18,
    label: str | None = None,
) -> None:
    x0, y0, x1, y1, radius, _ = segment
    vector = np.asarray([x1 - x0, y1 - y0], dtype=float)
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        ax.add_patch(
            Circle(
                (x0, y0),
                radius,
                color="#56B4E9",
                alpha=alpha,
                zorder=1,
                label=label,
            )
        )
        return
    normal = radius * np.asarray([-vector[1], vector[0]]) / length
    polygon = np.asarray(
        [
            [x0 + normal[0], y0 + normal[1]],
            [x1 + normal[0], y1 + normal[1]],
            [x1 - normal[0], y1 - normal[1]],
            [x0 - normal[0], y0 - normal[1]],
        ]
    )
    ax.add_patch(
        Polygon(
            polygon,
            closed=True,
            facecolor="#56B4E9",
            edgecolor="#0072B2",
            alpha=alpha,
            zorder=1,
            label=label,
        )
    )
    ax.add_patch(Circle((x0, y0), radius, facecolor="#56B4E9", edgecolor="#0072B2", alpha=alpha, zorder=1))
    ax.add_patch(Circle((x1, y1), radius, facecolor="#56B4E9", edgecolor="#0072B2", alpha=alpha, zorder=1))


def _draw_obstacles(ax: plt.Axes, values: Sequence[Mapping[str, object]]) -> int:
    count = 0
    for obstacle in values:
        xy = np.asarray([obstacle["x_m"], obstacle["y_m"]], dtype=float)
        radius = float(obstacle["radius_m"])
        ax.add_patch(
            Circle(xy, radius, facecolor="#FCA5A5", edgecolor=OBSTACLE_COLOR, alpha=0.75, zorder=5)
        )
        velocity = np.asarray([obstacle["vx_mps"], obstacle["vy_mps"]], dtype=float)
        if float(np.linalg.norm(velocity)) > 1e-12:
            draw_heading_arrow(
                ax,
                xy,
                math.atan2(velocity[1], velocity[0]),
                color=OBSTACLE_COLOR,
                length_m=min(0.55, max(0.18, float(np.linalg.norm(velocity)))),
                linewidth=1.0,
                label="obstacle velocity" if count == 0 else None,
            )
            count += 1
    return count


def _limits(
    guide: np.ndarray,
    legacy_xy: np.ndarray,
    safe_xy: np.ndarray,
    goal: np.ndarray,
    corridors: Sequence[tuple[float, float, float, float, float, int]],
    obstacles: Sequence[Mapping[str, object]],
    rectangles: Sequence[Mapping[str, object]] = (),
) -> list[float]:
    points = [guide, legacy_xy, safe_xy, goal.reshape(1, 2)]
    for x0, y0, x1, y1, radius, _ in corridors:
        points.append(np.asarray([[x0 - radius, y0 - radius], [x1 + radius, y1 + radius]]))
    for obstacle in obstacles:
        radius = float(obstacle["radius_m"])
        points.append(
            np.asarray(
                [
                    [float(obstacle["x_m"]) - radius, float(obstacle["y_m"]) - radius],
                    [float(obstacle["x_m"]) + radius, float(obstacle["y_m"]) + radius],
                ]
            )
        )
    for rectangle in rectangles:
        x0, y0, x1, y1 = np.asarray(rectangle["bounds_xy"], dtype=float)
        points.append(np.asarray([[x0, y0], [x1, y1]]))
    combined = np.concatenate([value for value in points if len(value)], axis=0)
    minimum = np.min(combined, axis=0)
    maximum = np.max(combined, axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    margin = max(0.4, 0.08 * float(max(span)))
    return [
        float(minimum[0] - margin),
        float(maximum[0] + margin),
        float(minimum[1] - margin),
        float(maximum[1] + margin),
    ]


def _configure_axis(ax: plt.Axes, limits: Sequence[float], title: str) -> None:
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    ax.set_title(title)
    ax.grid(alpha=0.22)


def _draw_orientation_contract(
    ax: plt.Axes,
    *,
    samples: np.ndarray,
    initial_state: object,
    goal: np.ndarray,
    goal_yaw: float | None,
    arrow_length: float,
) -> int:
    count = 0
    initial_xy = _state_xy(initial_state)
    initial_yaw = _state_yaw(initial_state)
    if initial_yaw is not None:
        draw_heading_arrow(
            ax,
            initial_xy,
            initial_yaw,
            color="#111827",
            length_m=arrow_length,
            linewidth=2.0,
            label="initial yaw",
        )
        count += 1
    xy = _sample_xy(samples)
    yaw = _sample_yaw(samples)
    if len(xy):
        arrow_indices = np.unique(np.linspace(0, len(xy) - 1, min(6, len(xy))).astype(int))
        for arrow_index, sample_index in enumerate(arrow_indices):
            if math.isfinite(float(yaw[sample_index])):
                draw_heading_arrow(
                    ax,
                    xy[sample_index],
                    float(yaw[sample_index]),
                    color="#1D4ED8",
                    length_m=0.72 * arrow_length,
                    linewidth=2.3 if sample_index == len(xy) - 1 else 1.2,
                    label="current yaw" if arrow_index == 0 else None,
                )
                count += 1
    _draw_velocity(ax, initial_xy, _state_velocity(initial_state), label="initial velocity")
    if goal_yaw is None:
        ax.annotate(
            "goal yaw: N/A",
            xy=goal,
            xytext=(8, -18),
            textcoords="offset points",
            color=GOAL_COLOR,
            fontsize=8,
            zorder=10,
        )
    else:
        draw_heading_arrow(
            ax,
            goal,
            goal_yaw,
            color=GOAL_COLOR,
            length_m=arrow_length,
            hollow=True,
            linewidth=1.8,
            label="goal yaw",
        )
    return count


def _save_figure(figure: plt.Figure, png: Path, pdf: Path) -> None:
    figure.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "NavDP rolling showcase"},
    )
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"Creator": "NavDP rolling showcase", "CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def _finite_or_blank(value: object) -> object:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def _figure_rows(
    scenario_uid: str,
    guide: np.ndarray,
    legacy: object,
    safe: object,
    corridors: Sequence[tuple[float, float, float, float, float, int]],
    obstacles: Sequence[Mapping[str, object]],
    initial_state: object,
    goal: np.ndarray,
    goal_yaw: float | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    initial_xy = _state_xy(initial_state)
    initial_velocity = _state_velocity(initial_state)
    rows.append(
        {
            "record_type": "INITIAL_STATE",
            "scenario_uid": scenario_uid,
            "panel": "all",
            "x_m": initial_xy[0],
            "y_m": initial_xy[1],
            "yaw_rad": _finite_or_blank(_state_yaw(initial_state)),
            "velocity_x_mps": (
                _finite_or_blank(initial_velocity[0])
                if initial_velocity is not None
                else ""
            ),
            "velocity_y_mps": (
                _finite_or_blank(initial_velocity[1])
                if initial_velocity is not None
                else ""
            ),
            "data_availability": "RECORDED_ROLLOUT",
        }
    )
    rows.append(
        {
            "record_type": "GOAL_STATE",
            "scenario_uid": scenario_uid,
            "panel": "all",
            "x_m": goal[0],
            "y_m": goal[1],
            "yaw_rad": _finite_or_blank(goal_yaw),
            "data_availability": (
                "RECORDED_ROLLOUT"
                if goal_yaw is not None
                else "GOAL_YAW_NOT_AVAILABLE"
            ),
        }
    )
    for sample_index, point in enumerate(guide):
        rows.append(
            {
                "record_type": "GUIDE_SAMPLE",
                "scenario_uid": scenario_uid,
                "panel": "guide",
                "method": "guide_reference",
                "sample_index": sample_index,
                "x_m": point[0],
                "y_m": point[1],
                "data_availability": "RECORDED_ROLLOUT",
            }
        )
    for method, result in (("legacy", legacy), ("safe_corridor_v1", safe)):
        samples = _samples(result)
        xy = _sample_xy(samples)
        yaw = _sample_yaw(samples)
        velocity = _sample_velocity(samples)
        times = _sample_time(samples)
        for sample_index, point in enumerate(xy):
            rows.append(
                {
                    "record_type": "EXECUTED_SAMPLE",
                    "scenario_uid": scenario_uid,
                    "panel": method,
                    "method": method,
                    "sample_index": sample_index,
                    "time_s": _finite_or_blank(times[sample_index]),
                    "x_m": point[0],
                    "y_m": point[1],
                    "yaw_rad": _finite_or_blank(yaw[sample_index]),
                    "velocity_x_mps": _finite_or_blank(velocity[sample_index, 0]),
                    "velocity_y_mps": _finite_or_blank(velocity[sample_index, 1]),
                    "status": str(_value(result, "status", "UNKNOWN")),
                    "data_availability": "RECORDED_ROLLOUT",
                }
            )
    for x0, y0, x1, y1, radius, cycle_index in corridors:
        rows.append(
            {
                "record_type": "CORRIDOR_SEGMENT",
                "scenario_uid": scenario_uid,
                "panel": "safe_corridor",
                "method": "safe_corridor_v1",
                "cycle_index": cycle_index,
                "segment_start_x_m": x0,
                "segment_start_y_m": y0,
                "segment_end_x_m": x1,
                "segment_end_y_m": y1,
                "corridor_radius_m": radius,
                "data_availability": "RECORDED_ROLLOUT",
            }
        )
    for obstacle in obstacles:
        rows.append(
            {
                "record_type": "OBSTACLE_STATE",
                "scenario_uid": scenario_uid,
                "panel": "all",
                "cycle_index": obstacle["cycle_index"],
                "x_m": obstacle["x_m"],
                "y_m": obstacle["y_m"],
                "velocity_x_mps": obstacle["vx_mps"],
                "velocity_y_mps": obstacle["vy_mps"],
                "obstacle_uid": obstacle["obstacle_uid"],
                "obstacle_radius_m": obstacle["radius_m"],
                "data_availability": "RECORDED_ROLLOUT",
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIGURE_DATA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIGURE_DATA_FIELDS})


def _status_box(
    ax: plt.Axes,
    *,
    method: str,
    status: str,
    frame: int,
    total: int,
    cycle: int | None,
    sample: np.ndarray | None,
    local_goal: np.ndarray | None,
    final_goal: np.ndarray,
) -> None:
    if sample is not None and len(sample) >= 15:
        speed = float(np.linalg.norm(sample[4:7]))
        acceleration = float(np.linalg.norm(sample[7:10]))
        state = (
            f"x/y: {sample[1]:.2f}/{sample[2]:.2f} m\n"
            f"yaw: {sample[13]:.2f} rad / {math.degrees(sample[13]):.1f} deg\n"
            f"v/a: {speed:.2f} m/s / {acceleration:.2f} m/s²\n"
            f"yaw rate: {sample[14]:.2f} rad/s\n"
        )
    else:
        state = "state: guide reference (N/A)\n"
    local_text = (
        "N/A" if local_goal is None else f"{local_goal[0]:.2f}/{local_goal[1]:.2f}"
    )
    ax.text(
        0.02,
        0.98,
        f"{method}\ncycle: {cycle if cycle is not None else 'N/A'} · "
        f"time: {frame * FRAME_DURATION_S:.2f} s\n"
        f"{state}local goal: {local_text} m\n"
        f"final goal: {final_goal[0]:.2f}/{final_goal[1]:.2f} m\n"
        f"frame {frame + 1}/{total} · status: {status}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.88, "edgecolor": "#6B7280"},
        zorder=20,
    )


def _animation(
    path: Path,
    *,
    scenario_uid: str,
    guide: np.ndarray,
    legacy: object,
    safe: object,
    initial_state: object,
    goal: np.ndarray,
    goal_yaw: float | None,
    limits: Sequence[float],
    obstacles: Sequence[Mapping[str, object]],
    corridors: Sequence[tuple[float, float, float, float, float, int]] = (),
    rectangles: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    legacy_samples = _samples(legacy)
    safe_samples = _samples(safe)
    legacy_xy, safe_xy = _sample_xy(legacy_samples), _sample_xy(safe_samples)
    base_frames = max(8, min(32, max(len(guide), len(legacy_xy), len(safe_xy))))
    source_indices = {
        "guide": np.rint(np.linspace(0, len(guide) - 1, base_frames)).astype(int),
        "legacy": np.rint(np.linspace(0, len(legacy_xy) - 1, base_frames)).astype(int),
        "safe_corridor_v1": np.rint(np.linspace(0, len(safe_xy) - 1, base_frames)).astype(int),
    }
    cycle_indices = sorted(
        {
            int(_value(cycle, "cycle_index", index))
            for result in (legacy, safe)
            for index, cycle in enumerate(_cycles(result))
        }
        | {int(row["cycle_index"]) for row in obstacles}
    )
    frames: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    total_frames = base_frames + TERMINAL_HOLD_FRAMES
    arrow_length = 0.08 * max(limits[1] - limits[0], limits[3] - limits[2])
    for encoded_index in range(total_frames):
        progress_index = min(encoded_index, base_frames - 1)
        figure, axes = plt.subplots(1, 3, figsize=(11.4, 4.0), constrained_layout=True)
        latest_cycle = (
            cycle_indices[
                min(
                    len(cycle_indices) - 1,
                    int(progress_index * len(cycle_indices) / max(1, base_frames)),
                )
            ]
            if cycle_indices
            else None
        )
        obstacle_snapshot = [
            row for row in obstacles if row["cycle_index"] == latest_cycle
        ]
        corridor_snapshot = [
            row for row in corridors if latest_cycle is None or row[5] == latest_cycle
        ]
        panels = (
            ("guide", guide, GUIDE_STYLE, "guide_reference", "REFERENCE"),
            ("legacy", legacy_xy, LEGACY_STYLE, "legacy", str(_value(legacy, "status", "UNKNOWN"))),
            (
                "safe_corridor_v1",
                safe_xy,
                SAFE_STYLE,
                "safe_corridor_v1",
                str(_value(safe, "status", "UNKNOWN")),
            ),
        )
        for axis, (panel, trajectory, style, method, status) in zip(axes, panels):
            axis.plot(guide[:, 0], guide[:, 1], label="guide", **GUIDE_STYLE)
            last = int(source_indices[panel][progress_index]) + 1
            if panel != "guide":
                axis.plot(trajectory[:last, 0], trajectory[:last, 1], label=method, **style)
            if panel == "safe_corridor_v1":
                for segment in corridor_snapshot:
                    _draw_capsule(axis, segment, alpha=0.14)
            axis.scatter(goal[0], goal[1], marker="*", s=90, color=GOAL_COLOR, label="goal", zorder=10)
            current_xy = trajectory[max(0, last - 1)]
            if panel == "legacy":
                yaw_values = _sample_yaw(legacy_samples)
            elif panel == "safe_corridor_v1":
                yaw_values = _sample_yaw(safe_samples)
            else:
                yaw_values = np.full(len(guide), np.nan)
            current_source = int(source_indices[panel][progress_index])
            sample = None
            local_goal = None
            if current_source < len(yaw_values) and math.isfinite(float(yaw_values[current_source])):
                draw_heading_arrow(
                    axis,
                    current_xy,
                    float(yaw_values[current_source]),
                    color="#1D4ED8",
                    length_m=arrow_length,
                    linewidth=2.0,
                )
            if panel == "legacy":
                sample = legacy_samples[current_source]
                cycles = _cycles(legacy)
            elif panel == "safe_corridor_v1":
                sample = safe_samples[current_source]
                cycles = _cycles(safe)
            else:
                cycles = ()
            if cycles:
                cycle_position = min(
                    len(cycles) - 1,
                    int(progress_index * len(cycles) / max(1, base_frames)),
                )
                local_value = _value(cycles[cycle_position], "local_goal_xyz", None)
                if local_value is not None:
                    local_goal = np.asarray(local_value, dtype=float).reshape(-1)[:2]
            initial_yaw = _state_yaw(initial_state)
            if initial_yaw is not None:
                draw_heading_arrow(
                    axis,
                    _state_xy(initial_state),
                    initial_yaw,
                    color="#111827",
                    length_m=arrow_length,
                    linewidth=1.5,
                )
            _draw_velocity(axis, _state_xy(initial_state), _state_velocity(initial_state))
            if goal_yaw is None:
                axis.annotate("goal yaw: N/A", goal, xytext=(5, -14), textcoords="offset points", color=GOAL_COLOR, fontsize=7)
            else:
                draw_heading_arrow(axis, goal, goal_yaw, color=GOAL_COLOR, length_m=arrow_length, hollow=True)
            _draw_obstacles(axis, obstacle_snapshot)
            _draw_static_rectangles(axis, rectangles)
            _configure_axis(axis, limits, method)
            _status_box(
                axis,
                method=method,
                status=status,
                frame=encoded_index,
                total=total_frames,
                cycle=latest_cycle,
                sample=sample,
                local_goal=local_goal,
                final_goal=goal,
            )
        figure.canvas.draw()
        frames.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
        plt.close(figure)
        rows.append(
            {
                "record_type": "ANIMATION_FRAME",
                "scenario_uid": scenario_uid,
                "panel": "guide|legacy|safe_corridor_v1",
                "frame_index": encoded_index,
                "time_s": encoded_index * FRAME_DURATION_S,
                "cycle_index": latest_cycle if latest_cycle is not None else "",
                "status": (
                    "TERMINAL_HOLD"
                    if encoded_index >= base_frames
                    else "ROLLING"
                ),
                "data_availability": "RECORDED_ROLLOUT",
            }
        )
    imageio.mimsave(path, frames, duration=FRAME_DURATION_S, loop=0)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def render_scene_package(
    paired_results: object, output_dir: Path | str
) -> RollingScenePackage:
    """Render one immutable scene package from recorded paired rollouts."""
    scenario_uid, legacy, safe = _normalise_pair(paired_results)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"immutable rolling scene package exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    guide = _guide(paired_results, safe)
    legacy_samples, safe_samples = _samples(legacy), _samples(safe)
    if len(legacy_samples) == 0 or len(safe_samples) == 0:
        raise ValueError("rolling showcase requires recorded executed trajectories")
    legacy_xy, safe_xy = _sample_xy(legacy_samples), _sample_xy(safe_samples)
    initial_state = _initial_state(paired_results, legacy, safe)
    goal = _goal(paired_results, safe, guide)
    goal_yaw_value = _first(
        paired_results, ("goal_yaw_rad", "final_goal_yaw_rad"), None
    )
    goal_yaw = None if goal_yaw_value is None else float(goal_yaw_value)
    if goal_yaw is not None and not math.isfinite(goal_yaw):
        goal_yaw = None
    corridors = _corridors(safe)
    obstacles = _obstacles(safe)
    rectangles = _static_rectangles(paired_results)
    limits = _limits(
        guide, legacy_xy, safe_xy, goal, corridors, obstacles, rectangles
    )
    arrow_length = 0.08 * max(limits[1] - limits[0], limits[3] - limits[2])
    latest_obstacles = (
        [row for row in obstacles if row["cycle_index"] == max(int(item["cycle_index"]) for item in obstacles)]
        if obstacles
        else []
    )

    heading_count = 0
    obstacle_arrow_count = 0
    three_panel = plt.figure(figsize=(11.8, 4.35), constrained_layout=True)
    axes = three_panel.subplots(1, 3)
    panels = (
        (
            axes[0],
            "Guide input",
            guide,
            GUIDE_STYLE,
            np.empty((0, 15), dtype=float),
        ),
        (axes[1], "Legacy rollout", legacy_xy, LEGACY_STYLE, legacy_samples),
        (axes[2], "Safe-corridor rollout", safe_xy, SAFE_STYLE, safe_samples),
    )
    for axis, title, trajectory, style, orientation_samples in panels:
        axis.plot(guide[:, 0], guide[:, 1], label="guide", **GUIDE_STYLE)
        if title != "Guide input":
            axis.plot(trajectory[:, 0], trajectory[:, 1], label=title, **style)
        axis.scatter(goal[0], goal[1], marker="*", s=100, color=GOAL_COLOR, label="goal", zorder=10)
        heading_count += _draw_orientation_contract(
            axis,
            samples=orientation_samples,
            initial_state=initial_state,
            goal=goal,
            goal_yaw=goal_yaw,
            arrow_length=arrow_length,
        )
        obstacle_arrow_count += _draw_obstacles(axis, latest_obstacles)
        _draw_static_rectangles(axis, rectangles)
        _configure_axis(axis, limits, title)
    handles: list[Any] = []
    labels: list[str] = []
    observed_labels: set[str] = set()
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label and label not in observed_labels:
                observed_labels.add(label)
                handles.append(handle)
                labels.append(label)
    three_panel.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=min(5, len(labels)),
        fontsize=7,
        frameon=False,
    )
    _save_figure(three_panel, output_dir / "three_panel.png", output_dir / "three_panel.pdf")

    overlay, axis = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    axis.plot(guide[:, 0], guide[:, 1], label="guide", **GUIDE_STYLE)
    axis.plot(legacy_xy[:, 0], legacy_xy[:, 1], label="legacy", **LEGACY_STYLE)
    axis.plot(safe_xy[:, 0], safe_xy[:, 1], label="safe_corridor_v1", **SAFE_STYLE)
    axis.scatter(goal[0], goal[1], marker="*", s=100, color=GOAL_COLOR, label="goal", zorder=10)
    heading_count += _draw_orientation_contract(
        axis,
        samples=safe_samples,
        initial_state=initial_state,
        goal=goal,
        goal_yaw=goal_yaw,
        arrow_length=arrow_length,
    )
    obstacle_arrow_count += _draw_obstacles(axis, latest_obstacles)
    _draw_static_rectangles(axis, rectangles)
    _configure_axis(axis, limits, "Recorded full-route overlay")
    axis.legend(loc="best", fontsize=8, frameon=False)
    _save_figure(overlay, output_dir / "overlay.png", output_dir / "overlay.pdf")

    corridor_figure, axis = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    for index, segment in enumerate(corridors):
        _draw_capsule(
            axis,
            segment,
            label="recorded corridor capsule" if index == 0 else None,
        )
    axis.plot(guide[:, 0], guide[:, 1], label="guide", **GUIDE_STYLE)
    axis.plot(safe_xy[:, 0], safe_xy[:, 1], label="safe_corridor_v1", **SAFE_STYLE)
    axis.scatter(goal[0], goal[1], marker="*", s=100, color=GOAL_COLOR, label="goal", zorder=10)
    heading_count += _draw_orientation_contract(
        axis,
        samples=safe_samples,
        initial_state=initial_state,
        goal=goal,
        goal_yaw=goal_yaw,
        arrow_length=arrow_length,
    )
    obstacle_arrow_count += _draw_obstacles(axis, latest_obstacles)
    _draw_static_rectangles(axis, rectangles)
    _configure_axis(axis, limits, "Recorded safe-corridor capsule union")
    axis.legend(loc="best", fontsize=8, frameon=False)
    _save_figure(
        corridor_figure,
        output_dir / "safe_corridor.png",
        output_dir / "safe_corridor.pdf",
    )

    rows = _figure_rows(
        scenario_uid,
        guide,
        legacy,
        safe,
        corridors,
        obstacles,
        initial_state,
        goal,
        goal_yaw,
    )
    animation_rows = _animation(
        output_dir / "three_way.gif",
        scenario_uid=scenario_uid,
        guide=guide,
        legacy=legacy,
        safe=safe,
        initial_state=initial_state,
        goal=goal,
        goal_yaw=goal_yaw,
        limits=limits,
        obstacles=obstacles,
        corridors=corridors,
        rectangles=rectangles,
    )
    rows.extend(animation_rows)
    _write_csv(output_dir / "figure_data.csv", rows)

    sample_count = len(guide) + len(legacy_xy) + len(safe_xy)
    paired_key = f"scenario_uid={scenario_uid}; methods=legacy|safe_corridor_v1"
    (output_dir / "caption.md").write_text(
        "# Rolling MINCO scene evidence\n\n"
        f"- Source: Recorded rolling cycles and executed trajectories for `{scenario_uid}`.\n"
        f"- Paired key: {paired_key}.\n"
        f"- Denominator: {len(_cycles(legacy))} legacy cycles and {len(_cycles(safe))} safe cycles; failures are retained.\n"
        "- Units: world XY and corridor radii in m; time in s; yaw in rad; velocity in m/s.\n"
        f"- Sample size: n={sample_count} plotted guide/executed samples, {len(corridors)} recorded corridor capsules.\n"
        "- Missing data: unavailable recorded values remain blank; missing goal yaw is explicitly N/A.\n"
        "- Interpretation: the three panels and overlay compare the same recorded guide and paired full-route rollouts.\n"
        "- Limitations: deterministic rolling evidence does not establish real sensor, contact, or simulator performance.\n",
        encoding="utf-8",
    )
    (output_dir / "caption_zh.md").write_text(
        "# 滚动 MINCO 场景证据\n\n"
        f"- 数据来源：`{scenario_uid}` 的已记录滚动周期、候选与执行轨迹。\n"
        f"- 配对键：{paired_key}。\n"
        f"- 分母：legacy {len(_cycles(legacy))} 个周期，safe {len(_cycles(safe))} 个周期；失败不删除。\n"
        "- 单位：世界坐标与走廊半径为 m，时间为 s，yaw 为 rad，速度为 m/s。\n"
        f"- 样本量：n={sample_count} 个 guide/执行轨迹样本，{len(corridors)} 个真实记录胶囊。\n"
        "- 缺失数据：未记录量保持空白；未提供目标 yaw 时明确标记 N/A。\n"
        "- 解读：三栏图与叠加图比较同一 guide 下的成对完整路线滚动结果。\n"
        "- 局限性：确定性滚动证据不能证明真实传感器、接触或仿真闭环性能。\n",
        encoding="utf-8",
    )

    gif_frames = imageio.mimread(output_dir / "three_way.gif")
    manifest = {
        "schema_version": 1,
        "scenario_uid": scenario_uid,
        "paired_key": ["scenario_uid", "method"],
        "methods": ["guide_reference", "legacy", "safe_corridor_v1"],
        "source": "RECORDED_ROLLING_EVIDENCE",
        "visual_contract": {
            "axis_aspect": "equal",
            "shared_xy_limits": limits,
            "panel_xy_limits": [limits, limits, limits],
            "robot_heading": "ARROW",
            "initial_yaw": "ARROW" if _state_yaw(initial_state) is not None else "N/A",
            "goal_yaw": "N/A" if goal_yaw is None else "HOLLOW_ARROW",
            "velocity": "ARROW" if _state_velocity(initial_state) is not None else "N/A",
            "dynamic_obstacle_velocity": "ARROW" if obstacle_arrow_count else "N/A",
            "sampled_heading_arrow_count": heading_count,
            "corridor_geometry": "RECORDED_CAPSULE_UNION",
            "corridor_capsule_count": len(corridors),
            "static_rectangle_count": len(rectangles),
            "unavailable_values_are_zero": False,
        },
        "animation": {
            "path": "three_way.gif",
            "panels": ["guide", "legacy", "safe_corridor_v1"],
            "frame_count": len(gif_frames),
            "frame_duration_s": FRAME_DURATION_S,
            "terminal_hold_frames": TERMINAL_HOLD_FRAMES,
            "status_box": True,
            "shows_current_state": True,
            "shows_final_goal": True,
            "shows_safe_corridor": True,
            "state_box_fields": [
                "cycle",
                "time_s",
                "x_m",
                "y_m",
                "yaw_rad",
                "yaw_deg",
                "speed_mps",
                "acceleration_mps2",
                "yaw_rate_radps",
                "local_goal_xy_m",
                "final_goal_xy_m",
                "status",
            ],
            "synchronization": "relative normalized progress; terminal frames frozen",
        },
        "figure_data": {
            "path": "figure_data.csv",
            "schema": list(FIGURE_DATA_FIELDS),
            "row_count": len(rows),
            "animation_frame_rows": len(animation_rows),
        },
    }
    manifest_path = output_dir / "scene_manifest.json"
    _write_json(manifest_path, manifest)
    validation_path = output_dir / "validation.json"
    _write_json(
        validation_path,
        {
            "schema_version": 1,
            "valid": True,
            "errors": [],
            "verified_manifest": _receipt(manifest_path, output_dir),
        },
    )
    receipt_path = output_dir / "artifact_receipt.json"
    _write_json(
        receipt_path,
        {
            "schema_version": 1,
            "root": ".",
            "artifacts": [
                _receipt(path, output_dir)
                for path in sorted(output_dir.iterdir())
                if path.is_file() and path != receipt_path
            ],
        },
    )
    errors = validate_scene_package(output_dir)
    if errors:
        raise RuntimeError("invalid rolling scene package: " + "; ".join(errors))
    return RollingScenePackage(
        output_dir=output_dir,
        files=tuple(sorted(path for path in output_dir.iterdir() if path.is_file())),
        manifest_path=manifest_path,
        receipt_path=receipt_path,
        validation_path=validation_path,
    )


def validate_scene_package(output_dir: Path | str) -> list[str]:
    root = Path(output_dir).resolve()
    errors: list[str] = []
    required = {
        "three_panel.png",
        "three_panel.pdf",
        "overlay.png",
        "overlay.pdf",
        "safe_corridor.png",
        "safe_corridor.pdf",
        "three_way.gif",
        "figure_data.csv",
        "caption.md",
        "caption_zh.md",
        "scene_manifest.json",
        "validation.json",
        "artifact_receipt.json",
    }
    for name in sorted(required):
        if not (root / name).is_file():
            errors.append(f"missing scene artifact {name}")
    try:
        manifest = json.loads((root / "scene_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [*errors, f"unreadable scene_manifest.json: {error}"]
    if manifest.get("schema_version") != 1:
        errors.append("scene manifest schema mismatch")
    contract = manifest.get("visual_contract", {})
    if not isinstance(contract, Mapping):
        errors.append("visual contract is invalid")
        contract = {}
    if contract.get("axis_aspect") != "equal":
        errors.append("XY visual contract is not equal-aspect")
    limits = contract.get("shared_xy_limits")
    if not isinstance(limits, list) or len(limits) != 4:
        errors.append("shared XY limits are invalid")
    panel_limits = contract.get("panel_xy_limits")
    if not isinstance(panel_limits, list) or len(panel_limits) != 3:
        errors.append("three-panel XY limits are invalid")
        panel_limits = []
    if any(panel != limits for panel in panel_limits):
        errors.append("three-panel XY limits are not shared")
    capsule_count = contract.get("corridor_capsule_count")
    if not isinstance(capsule_count, int) or isinstance(capsule_count, bool):
        capsule_count = 0
    if (
        contract.get("corridor_geometry") != "RECORDED_CAPSULE_UNION"
        or capsule_count <= 0
    ):
        errors.append("safe corridor lacks recorded capsule evidence")
    try:
        with (root / "figure_data.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            if tuple(reader.fieldnames or ()) != FIGURE_DATA_FIELDS:
                errors.append("figure_data.csv schema mismatch")
    except (OSError, csv.Error) as error:
        errors.append(f"unreadable figure_data.csv: {error}")
        rows = []
    if len(rows) != manifest.get("figure_data", {}).get("row_count"):
        errors.append("figure_data.csv row count mismatch")
    animation_rows = [row for row in rows if row.get("record_type") == "ANIMATION_FRAME"]
    try:
        frames = imageio.mimread(root / "three_way.gif")
    except Exception as error:
        errors.append(f"three_way.gif decode failed: {error}")
        frames = []
    if len(frames) != len(animation_rows):
        errors.append("three_way.gif frame count differs from figure data")
    if len(frames) != manifest.get("animation", {}).get("frame_count"):
        errors.append("three_way.gif frame count differs from manifest")
    caption_fields = {
        "caption.md": ("Source:", "Paired key:", "Denominator:", "Units:", "Sample size:", "Missing data:", "Interpretation:", "Limitations:"),
        "caption_zh.md": ("数据来源：", "配对键：", "分母：", "单位：", "样本量：", "缺失数据：", "解读：", "局限性："),
    }
    for name, fields in caption_fields.items():
        try:
            text = (root / name).read_text(encoding="utf-8")
        except OSError as error:
            errors.append(f"unreadable {name}: {error}")
            continue
        for field in fields:
            if field not in text:
                errors.append(f"{name} missing {field}")
    try:
        validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
        if validation.get("valid") is not True or validation.get("errors") != []:
            errors.append("validation.json does not record valid output")
        verified = validation.get("verified_manifest", {})
        manifest_path = root / "scene_manifest.json"
        if verified.get("path") != "scene_manifest.json" or verified.get("sha256") != _sha256(manifest_path):
            errors.append("validation manifest receipt mismatch")
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"unreadable validation.json: {error}")
    try:
        receipt = json.loads((root / "artifact_receipt.json").read_text(encoding="utf-8"))
        receipts = receipt.get("artifacts", [])
        recorded = {str(row.get("path", "")) for row in receipts}
        actual = {
            path.relative_to(root).as_posix()
            for path in root.iterdir()
            if path.is_file() and path.name != "artifact_receipt.json"
        }
        if recorded != actual:
            errors.append("artifact receipt inventory mismatch")
        for row in receipts:
            path = root / str(row.get("path", ""))
            if not path.is_file():
                errors.append(f"missing receipted artifact {row.get('path')}")
            else:
                if path.stat().st_size != row.get("size_bytes"):
                    errors.append(f"{row.get('path')} size mismatch")
                if _sha256(path) != row.get("sha256"):
                    errors.append(f"{row.get('path')} hash mismatch")
    except (OSError, json.JSONDecodeError, TypeError) as error:
        errors.append(f"unreadable artifact_receipt.json: {error}")
    return errors


__all__ = [
    "FIGURE_DATA_FIELDS",
    "RollingScenePackage",
    "draw_heading_arrow",
    "render_scene_package",
    "validate_scene_package",
]
