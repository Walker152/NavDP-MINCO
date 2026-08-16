from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image

from experiments.visualizers.rolling_showcase import (
    draw_heading_arrow,
    _draw_capsule,
    _draw_obstacles,
    _draw_static_rectangles,
    _draw_velocity,
    _status_box,
    _configure_axis,
    _draw_orientation_contract,
    _limits,
    GUIDE_STYLE,
    LEGACY_STYLE,
    SAFE_STYLE,
    GOAL_COLOR,
    OBSTACLE_COLOR,
    VELOCITY_COLOR,
    FRAME_DURATION_S,
    TERMINAL_HOLD_FRAMES,
)

from experiments.static.case_schema import StaticCase
from experiments.static.runner import StaticRunResult


# Per-trajectory GIFs retain a compact cadence.  Factor grids redraw nine
# annotated trajectories in every frame, so they need a distinctly slower
# cadence for the initial-state labels, heading arrows, and SFC cells to be
# readable in a paper review or screen recording.
GIF_FRAME_DURATION_S = 0.1
GRID_GIF_FRAME_DURATION_S = 0.20
GRID_GIF_MIN_FRAMES = 24


def _finite_or_blank(value: object) -> object:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def _nearest_clearance(
    point_xy: np.ndarray, detail: Mapping[str, np.ndarray]
) -> object:
    coordinates = np.asarray(detail.get("clearance_xy", np.empty((0, 2))))
    clearance = np.asarray(detail.get("clearance_m", np.empty(0)))
    valid = np.asarray(detail.get("clearance_valid", np.empty(0, dtype=bool)))
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] < 2
        or len(coordinates) != len(clearance)
        or len(valid) != len(clearance)
        or not np.any(valid)
    ):
        return ""
    valid_indices = np.flatnonzero(valid)
    nearest = valid_indices[
        int(
            np.argmin(
                np.linalg.norm(
                    coordinates[valid_indices, :2] - point_xy[None, :2], axis=1
                )
            )
        )
    ]
    return _finite_or_blank(clearance[nearest])


def _static_frame_rows(
    case: StaticCase,
    result: StaticRunResult,
    detail: Mapping[str, np.ndarray],
    frame_source_indices: np.ndarray,
    decoded_frame_count: int,
) -> list[dict[str, object]]:
    path = _path(result)
    if len(path) == 0:
        path = case.guide_path_xyz
    if decoded_frame_count <= 0:
        return []
    if len(frame_source_indices) != decoded_frame_count:
        frame_source_indices = np.unique(
            np.linspace(0, len(path) - 1, decoded_frame_count).astype(int)
        )
        if len(frame_source_indices) != decoded_frame_count:
            frame_source_indices = np.rint(
                np.linspace(0, len(path) - 1, decoded_frame_count)
            ).astype(int)
    has_temporal = len(result.samples) == len(path) and len(result.samples) > 0
    rows: list[dict[str, object]] = []
    for frame_index, source_index in enumerate(frame_source_indices):
        source_index = min(max(0, int(source_index)), len(path) - 1)
        point = path[source_index]
        clearance = _nearest_clearance(point[:2], detail)
        row: dict[str, object] = {
            "frame_index": frame_index,
            "time_s": frame_index * GIF_FRAME_DURATION_S,
            "source_sample_index": source_index,
            "source_time_s": (
                _finite_or_blank(result.samples[source_index, 0])
                if has_temporal
                else ""
            ),
            "x_m": _finite_or_blank(point[0]),
            "y_m": _finite_or_blank(point[1]),
            "yaw_rad": "",
            "progress_ratio": (
                float(source_index) / float(max(1, len(path) - 1))
            ),
            "clearance_m": clearance,
            "speed_mps": "",
            "acceleration_mps2": "",
            "yaw_rate_rps": "",
            "planning_state": str(result.status),
            "termination_state": str(result.status),
            "local_goal_x_m": "",
            "local_goal_y_m": "",
            "data_availability": (
                "MEASURED_STATIC" if has_temporal else "STATIC_GEOMETRY_ONLY"
            ),
        }
        if has_temporal:
            sample = result.samples[source_index]
            row.update(
                {
                    "speed_mps": _finite_or_blank(np.linalg.norm(sample[4:7])),
                    "acceleration_mps2": _finite_or_blank(
                        np.linalg.norm(sample[7:10])
                    ),
                    "jerk_mps3": _finite_or_blank(np.linalg.norm(sample[10:13])),
                    "yaw_rad": _finite_or_blank(sample[13]),
                    "yaw_rate_rps": _finite_or_blank(sample[14]),
                }
            )
        if case.terminal_goal is not None and len(case.terminal_goal) >= 2:
            row["local_goal_x_m"] = ""
            row["local_goal_y_m"] = ""
        rows.append(row)
    return rows


def _extreme_event(
    rows: list[dict[str, object]],
    *,
    metric: str,
    event_type: str,
    description_zh: str,
    use_minimum: bool = False,
) -> dict[str, object] | None:
    measured = [
        (index, float(row[metric]))
        for index, row in enumerate(rows)
        if row.get(metric, "") != "" and math.isfinite(float(row[metric]))
    ]
    if not measured:
        return None
    selected = min(measured, key=lambda item: item[1]) if use_minimum else max(
        measured, key=lambda item: item[1]
    )
    index, value = selected
    time_s = float(rows[index]["time_s"])
    return {
        "event_uid": f"evt-{event_type.lower().replace('_', '-')}",
        "event_type": event_type,
        "start_frame_index": index,
        "end_frame_index": index,
        "start_time_s": time_s,
        "end_time_s": time_s,
        "severity": "INFO",
        "description_zh": description_zh,
        "metric_name": metric,
        "metric_value": value,
        "data_availability": "MEASURED_STATIC",
    }


def _static_events(
    rows: list[dict[str, object]], result: StaticRunResult
) -> list[dict[str, object]]:
    if not rows:
        return []
    events: list[dict[str, object]] = [
        {
            "event_uid": "evt-optimization-outcome",
            "event_type": (
                "OPTIMIZATION_SUCCEEDED"
                if result.status == "SUCCEEDED"
                else "OPTIMIZATION_TERMINATED"
            ),
            "start_frame_index": 0,
            "end_frame_index": 0,
            "start_time_s": rows[0]["time_s"],
            "end_time_s": rows[0]["time_s"],
            "severity": "INFO" if result.status == "SUCCEEDED" else "WARNING",
            "description_zh": (
                "静态轨迹优化完成。"
                if result.status == "SUCCEEDED"
                else "静态轨迹优化未产生可发布成功结果。"
            ),
            "data_availability": "MEASURED_STATIC",
        },
        {
            "event_uid": "evt-terminal-frame",
            "event_type": "TERMINAL_FRAME",
            "start_frame_index": len(rows) - 1,
            "end_frame_index": len(rows) - 1,
            "start_time_s": rows[-1]["time_s"],
            "end_time_s": rows[-1]["time_s"],
            "severity": "INFO",
            "description_zh": "动画到达静态轨迹末端帧。",
            "data_availability": "MEASURED_STATIC",
        },
    ]
    for event in (
        _extreme_event(
            rows,
            metric="clearance_m",
            event_type="MINIMUM_CLEARANCE",
            description_zh="轨迹达到最小静态间隙。",
            use_minimum=True,
        ),
        _extreme_event(
            rows,
            metric="speed_mps",
            event_type="PEAK_SPEED",
            description_zh="轨迹达到峰值速度。",
        ),
        _extreme_event(
            rows,
            metric="acceleration_mps2",
            event_type="PEAK_ACCELERATION",
            description_zh="轨迹达到峰值加速度。",
        ),
        _extreme_event(
            rows,
            metric="jerk_mps3",
            event_type="PEAK_JERK",
            description_zh="轨迹达到峰值加加速度。",
        ),
        _extreme_event(
            rows,
            metric="yaw_rate_rps",
            event_type="PEAK_YAW_RATE",
            description_zh="轨迹达到峰值偏航角速度。",
        ),
    ):
        if event is not None:
            events.append(event)
    corridor_reason = str(result.diagnostics.get("corridor_failure_reason", ""))
    if corridor_reason and corridor_reason != "DISABLED":
        events.append(
            {
                "event_uid": "evt-corridor-diagnostic",
                "event_type": "CORRIDOR_DIAGNOSTIC",
                "start_frame_index": 0,
                "end_frame_index": 0,
                "start_time_s": rows[0]["time_s"],
                "end_time_s": rows[0]["time_s"],
                "severity": "WARNING",
                "description_zh": f"安全走廊诊断：{corridor_reason}。",
                "data_availability": "MEASURED_STATIC",
            }
        )
    return events


def build_static_gif_evidence(
    gif_path: Path | str,
    *,
    case: StaticCase,
    result: StaticRunResult,
    metrics: Mapping[str, Any],
    detail: Mapping[str, np.ndarray],
    frame_source_indices: np.ndarray,
) -> tuple[Path, ...]:
    """Build and validate the Chinese evidence bundle for one static GIF."""
    from experiments.visualizers.video_evidence import (
        build_video_evidence_package,
        validate_video_evidence_package,
    )

    gif_path = Path(gif_path).resolve()
    decoded_count = len(imageio.mimread(gif_path))
    rows = _static_frame_rows(
        case, result, detail, frame_source_indices, decoded_count
    )
    package = gif_path.with_name(f"{gif_path.stem}_evidence")
    safe_uid = re.sub(r"[^A-Za-z0-9_.-]+", "-", case.case_uid)
    profile = str(case.constraint_profile)
    build_video_evidence_package(
        gif_path,
        package,
        evidence_uid=f"static-{safe_uid}-{profile}-gif-evidence",
        media_uid=f"static-{safe_uid}-{profile}-gif",
        data_source=case.case_source,
        fps=1.0 / GIF_FRAME_DURATION_S,
        case_uid=case.case_uid,
        frame_rows=rows,
        event_rows=_static_events(rows, result),
        caption_overrides={
            "scene_zh": f"静态案例 {case.case_uid}（{case.expected_category}）",
            "method_zh": profile,
            "research_question_zh": (
                f"{case.expected_category} 静态条件下的轨迹优化行为。"
            ),
            "pairing_key_zh": f"case_uid={case.case_uid}，profile={profile}。",
            "denominator_zh": (
                f"解码帧 n={decoded_count}；失败帧不从分母删除。"
            ),
            "limitation_zh": (
                "静态确定性回放不代表真实闭环、接触或动态障碍性能。"
            ),
            "interpretation_zh": (
                f"优化状态为 {result.status}；最小间隙="
                f"{metrics.get('min_clearance_m', 'NA')} m。"
            ),
            "evidence_boundary_zh": (
                "仅证明确定性静态轨迹、静态 ESDF 与解析动力学样本；"
                "不可外推为真实闭环导航性能。"
            ),
            "conclusion_limit_zh": (
                "无真实传感器、控制延迟、接触与动态障碍证据；"
                "缺失指标保持 NA。"
            ),
            "failure_handling_zh": (
                f"状态={result.status}；失败原因="
                f"{result.diagnostics.get('failure_reason') or '无'}。"
            ),
            "metrics_units_zh": (
                "位置/间隙：m；速度：m/s；加速度：m/s²；"
                "加加速度：m/s³；偏航角：rad；偏航角速度：rad/s。"
            ),
        },
    )
    return _finalize_static_gif_package(
        package,
        case=case,
        profile=profile,
        status_text=str(result.status),
        decoded_count=decoded_count,
        interpretation=(
            f"优化状态为 {result.status}；最小间隙="
            f"{metrics.get('min_clearance_m', 'NA')} m。"
        ),
    )


def _finalize_static_gif_package(
    package: Path,
    *,
    case: StaticCase,
    profile: str,
    status_text: str,
    decoded_count: int,
    interpretation: str,
) -> tuple[Path, ...]:
    from experiments.visualizers.video_evidence import (
        validate_video_evidence_package,
    )

    caption_zh = (package / "caption_zh.md").read_text(encoding="utf-8")
    caption = (
        "# 静态轨迹动画科研证据说明\n\n"
        f"- 研究问题：{case.expected_category} 静态条件下的轨迹优化行为。\n"
        f"- 数据来源：{case.case_source}，case_hash={case.case_hash}。\n"
        f"- 配对键：case_uid={case.case_uid}，profile={profile}。\n"
        f"- 证据状态：{status_text}。\n"
        f"- 分母：解码帧 n={decoded_count}；失败帧不从分母删除。\n"
        "- 局限性：静态确定性回放不代表真实闭环、接触或动态障碍性能。\n"
        f"- 解读：{interpretation}\n\n"
        "## 完整证据字段\n\n"
        f"{caption_zh}"
    )
    (package / "caption.md").write_text(caption, encoding="utf-8")
    plain = re.sub(r"^[#*-]+\s*", "", caption, flags=re.MULTILINE)
    (package / "caption.txt").write_text(plain, encoding="utf-8")
    receipt_path = package / "artifact_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for alias in (package / "caption.md", package / "caption.txt"):
        receipt["artifacts"].append(
            {
                "role": "static_chinese_caption_alias",
                "path": alias.name,
                "size_bytes": alias.stat().st_size,
                "sha256": hashlib.sha256(alias.read_bytes()).hexdigest(),
            }
        )
    receipt["artifacts"] = sorted(
        receipt["artifacts"], key=lambda row: str(row["path"])
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    errors = validate_video_evidence_package(package)
    if errors:
        raise RuntimeError("invalid static GIF evidence: " + "; ".join(errors))
    return tuple(sorted(path for path in package.iterdir() if path.is_file()))


def render_paired_static_gif(
    gif_path: Path | str,
    *,
    case: StaticCase,
    legacy_result: StaticRunResult,
    legacy_detail: Mapping[str, np.ndarray],
    safe_result: StaticRunResult,
    safe_detail: Mapping[str, np.ndarray],
    footprint_radius_m: float = 0.2,
) -> Path:
    """Render a dual-panel legacy/constrained paired GIF from static results.

    Left panel: legacy trajectory with its own data (no corridor capsules).
    Right panel: constrained trajectory with native corridor geometry from real
    data (capsules for the historical profile, convex cells for SuperPlanner).

    Both panels show heading arrows, obstacles, and status boxes.
    """
    gif_path = Path(gif_path).resolve()
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    legacy_path = _path(legacy_result)
    if len(legacy_path) == 0:
        legacy_path = case.guide_path_xyz
    safe_path = _path(safe_result)
    if len(safe_path) == 0:
        safe_path = case.guide_path_xyz

    base_frames = min(32, max(8, max(len(legacy_path), len(safe_path))))
    legacy_indices = np.rint(np.linspace(0, len(legacy_path) - 1, base_frames)).astype(int)
    safe_indices = np.rint(np.linspace(0, len(safe_path) - 1, base_frames)).astype(int)
    total_frames = base_frames + TERMINAL_HOLD_FRAMES

    safe_corridors = _corridor_segments_from_diagnostics(safe_result.diagnostics)
    safe_sfc_cells = _sfc_cells_from_diagnostics(safe_result.diagnostics)
    obstacles = _obstacles_from_diagnostics(safe_result.diagnostics)
    static_rects = _static_obstacle_rectangles(case)

    # Compute shared view limits from both trajectories
    all_points = [
        case.guide_path_xyz[:, :2],
        legacy_path[:, :2],
        safe_path[:, :2],
    ]
    if case.terminal_goal is not None:
        all_points.append(case.terminal_goal[:2].reshape(1, 2))
    for x0, y0, x1, y1, radius, _ in safe_corridors:
        all_points.append(np.array([[x0 - radius, y0 - radius], [x1 + radius, y1 + radius]]))
    for cell in safe_sfc_cells:
        all_points.append(cell["vertices"])
    for rectangle in static_rects:
        x0, y0, x1, y1 = rectangle["bounds_xy"]
        all_points.append(np.asarray([[x0, y0], [x1, y1]], dtype=float))
    combined = np.concatenate(all_points, axis=0)
    vmin = np.min(combined, axis=0)
    vmax = np.max(combined, axis=0)
    span = np.maximum(vmax - vmin, 1.0)
    margin = max(0.4, 0.08 * float(max(span)))
    limits = [float(vmin[0] - margin), float(vmax[0] + margin),
              float(vmin[1] - margin), float(vmax[1] + margin)]
    arrow_length = 0.08 * max(limits[1] - limits[0], limits[3] - limits[2])

    has_legacy_temporal = len(legacy_result.samples) == len(legacy_path) and len(legacy_result.samples) > 0
    has_safe_temporal = len(safe_result.samples) == len(safe_path) and len(safe_result.samples) > 0

    constrained_profile = str(safe_result.diagnostics.get("constraint_profile", "safe_corridor_v1"))
    if constrained_profile not in {"safe_corridor_v1", "superplanner_sfc_v1"}:
        constrained_profile = "constrained"
    panels = (
        ("legacy", legacy_path, legacy_indices, legacy_result, has_legacy_temporal, False),
        (constrained_profile, safe_path, safe_indices, safe_result, has_safe_temporal, True),
    )

    frames = []
    for encoded_index in range(total_frames):
        progress_index = min(encoded_index, base_frames - 1)
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.1))
        fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.28, wspace=0.18)

        for ax, (method, traj, indices, result, has_temporal, show_corridors) in zip(axes, panels):
            source_idx = int(indices[progress_index])

            _draw_map(ax, case)
            ax.plot(
                case.guide_path_xyz[:, 0], case.guide_path_xyz[:, 1],
                "--", color="#4B5563", linewidth=1.0, alpha=0.7,
            )
            ax.plot(
                traj[: source_idx + 1, 0], traj[: source_idx + 1, 1],
                color=SAFE_STYLE["color"] if show_corridors else LEGACY_STYLE["color"],
                linewidth=SAFE_STYLE["linewidth"] if show_corridors else LEGACY_STYLE["linewidth"],
                linestyle=SAFE_STYLE["linestyle"] if show_corridors else LEGACY_STYLE["linestyle"],
            )
            ax.scatter(*case.start_position[:2], marker="s", color="green", s=60, zorder=8)

            if case.terminal_goal is not None:
                ax.scatter(*case.terminal_goal[:2], marker="*", s=100, color=GOAL_COLOR, zorder=10)

            # Initial yaw
            if math.isfinite(case.start_yaw):
                draw_heading_arrow(
                    ax, case.start_position[:2], case.start_yaw,
                    color="#111827", length_m=arrow_length, linewidth=1.8,
                )

            # Current yaw
            if has_temporal and math.isfinite(float(result.samples[source_idx, 13])):
                draw_heading_arrow(
                    ax, traj[source_idx, :2],
                    float(result.samples[source_idx, 13]),
                    color="#1D4ED8", length_m=arrow_length, linewidth=2.0,
                )

            # Goal yaw
            if case.terminal_goal is not None:
                goal_yaw_val = result.diagnostics.get("goal_yaw_rad", None)
                if goal_yaw_val is None or not math.isfinite(float(goal_yaw_val)):
                    ax.annotate(
                        "goal yaw: N/A",
                        xy=case.terminal_goal[:2],
                        xytext=(5, -14), textcoords="offset points",
                        color=GOAL_COLOR, fontsize=7, zorder=10,
                    )
                else:
                    draw_heading_arrow(
                        ax, case.terminal_goal[:2], float(goal_yaw_val),
                        color=GOAL_COLOR, length_m=arrow_length,
                        hollow=True, linewidth=1.5,
                    )

            # Safe corridor (only on safe panel, only if real data)
            if show_corridors:
                for segment in safe_corridors:
                    _draw_capsule(ax, segment, alpha=0.14)
                _draw_sfc_cells(ax, safe_sfc_cells)

            _draw_obstacles(ax, obstacles)
            _draw_static_rectangles(ax, static_rects)

            # Robot footprint
            centre = traj[source_idx, :2]
            ax.add_patch(
                plt.matplotlib.patches.Circle(
                    centre, footprint_radius_m, color="#ff7f0e", alpha=0.35, zorder=7,
                )
            )

            _configure_axis(ax, limits, method)

            # Status box
            sample = result.samples[source_idx] if has_temporal else None
            if sample is not None and len(sample) >= 15:
                speed = float(np.linalg.norm(sample[4:7]))
                acceleration = float(np.linalg.norm(sample[7:10]))
                state_text = (
                    f"x/y: {sample[1]:.2f}/{sample[2]:.2f} m\n"
                    f"yaw: {sample[13]:.2f} rad / {math.degrees(sample[13]):.1f} deg\n"
                    f"v/a: {speed:.2f} m/s / {acceleration:.2f} m/s²\n"
                    f"yaw rate: {sample[14]:.2f} rad/s\n"
                )
            else:
                state_text = "state: N/A\n"

            clearance_val = _nearest_clearance(centre, safe_detail if show_corridors else legacy_detail)
            sfc_margin = _sfc_margin_at_point(centre, safe_sfc_cells) if show_corridors else ""

            ax.text(
                0.02, -0.045,
                f"{method}\n"
                f"time: {encoded_index * GIF_FRAME_DURATION_S:.2f}s\n"
                f"{state_text}"
                f"clearance: {clearance_val if clearance_val != '' else 'N/A'} m\n"
                f"SFC margin: {sfc_margin if sfc_margin != '' else 'N/A'} m\n"
                f"status: {result.status}\n"
                f"reason: {result.diagnostics.get('sfc_generation_reason') or result.diagnostics.get('validation_failure_reason') or result.diagnostics.get('failure_reason') or 'NONE'}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=7.0,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                      "alpha": 0.88, "edgecolor": "#6B7280"},
                zorder=20,
                clip_on=False,
            )

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    imageio.mimsave(gif_path, frames, duration=GIF_FRAME_DURATION_S, loop=0)

    # Build the paired evidence package. Single-profile GIFs are not available
    # at this level, so the paired GIF itself serves as the per-profile frame
    # count reference; profile rows are derived from real per-profile data.
    build_paired_static_gif_evidence(
        gif_path,
        case=case,
        legacy_result=legacy_result,
        legacy_detail=legacy_detail,
        legacy_gif_path=gif_path,
        safe_result=safe_result,
        safe_detail=safe_detail,
        safe_gif_path=gif_path,
    )
    return gif_path


def render_factor_grid_gif(
    gif_path: Path | str,
    *,
    grid_cells: list[list[dict[str, object]]],
    row_factor: dict[str, object],
    col_factor: dict[str, object],
    footprint_radius_m: float = 0.2,
) -> Path:
    """Render a factor-grid comparison GIF showing multiple initial-condition variants.

    Args:
        gif_path: Output path for the grid GIF.
        grid_cells: 2D list (rows × cols) of dicts, each containing:
            - case: StaticCase
            - result: StaticRunResult
            - detail: detail dict from compute_static_case_metrics
        row_factor: {"name": str, "levels": list[float], "label": str}
        col_factor: {"name": str, "levels": list[float], "label": str}
        footprint_radius_m: Robot footprint radius in metres.

    Returns:
        Resolved Path to the grid GIF.
    """
    gif_path = Path(gif_path).resolve()
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(grid_cells)
    n_cols = len(grid_cells[0]) if grid_cells else 0
    if n_rows == 0 or n_cols == 0:
        raise ValueError("grid_cells must be non-empty 2D")

    # Determine base frames and compute shared view limits
    all_points = []
    cell_data: list[list[dict[str, object]]] = []
    max_frames = 0
    for ri, row_cells in enumerate(grid_cells):
        cell_row: list[dict[str, object]] = []
        for ci, cell in enumerate(row_cells):
            case = cell["case"]
            result = cell["result"]
            path = _path(result)
            if len(path) == 0:
                path = case.guide_path_xyz
            # Use enough equally-spaced samples for continuous apparent
            # motion.  Slowing six milestone frames only creates a slideshow.
            base_frames = min(32, max(GRID_GIF_MIN_FRAMES, len(path)))
            frame_indices = np.rint(
                np.linspace(0, len(path) - 1, base_frames)
            ).astype(int)
            max_frames = max(max_frames, base_frames)
            all_points.append(case.guide_path_xyz[:, :2])
            all_points.append(path[:, :2])
            if case.terminal_goal is not None:
                all_points.append(case.terminal_goal[:2].reshape(1, 2))
            rectangles = _static_obstacle_rectangles(case)
            for rectangle in rectangles:
                x0, y0, x1, y1 = rectangle["bounds_xy"]
                all_points.append(
                    np.asarray([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])
                )
            for sfc_cell in _sfc_cells_from_diagnostics(result.diagnostics):
                all_points.append(sfc_cell["vertices"])
            cell_row.append({
                "case": case,
                "result": result,
                "path": path,
                "frame_indices": frame_indices,
                "base_frames": base_frames,
                "has_temporal": len(result.samples) == len(path) and len(result.samples) > 0,
                "sfc_cells": _sfc_cells_from_diagnostics(result.diagnostics),
                "static_rectangles": _static_obstacle_rectangles(case),
                "obstacles": _obstacles_from_diagnostics(result.diagnostics),
            })
        cell_data.append(cell_row)

    combined = np.concatenate(all_points, axis=0)
    vmin = np.min(combined, axis=0)
    vmax = np.max(combined, axis=0)
    span = np.maximum(vmax - vmin, 1.0)
    margin = max(0.3, 0.06 * float(max(span)))
    limits = [
        float(vmin[0] - margin), float(vmax[0] + margin),
        float(vmin[1] - margin), float(vmax[1] + margin),
    ]

    # Panel dimensions
    panel_size = 2.2  # inches per cell
    label_width = 0.8
    label_height = 0.3
    fig_width = label_width + n_cols * panel_size
    fig_height = label_height + n_rows * panel_size

    row_label = str(row_factor.get("label", row_factor.get("name", "")))
    col_label = str(col_factor.get("label", col_factor.get("name", "")))
    row_levels = list(row_factor.get("levels", []))
    col_levels = list(col_factor.get("levels", []))

    frames = []
    for encoded_index in range(max_frames):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_width, fig_height),
            constrained_layout=True,
            squeeze=False,
        )

        for ri in range(n_rows):
            for ci in range(n_cols):
                ax = axes[ri, ci]
                data = cell_data[ri][ci]
                case = data["case"]
                result = data["result"]
                path = data["path"]
                frame_indices = data["frame_indices"]
                base_frames = data["base_frames"]

                progress_index = min(encoded_index, base_frames - 1)
                source_idx = int(frame_indices[progress_index])

                _draw_map(ax, case)
                _draw_sfc_cells(ax, data["sfc_cells"], alpha=0.18)
                _draw_static_rectangles(ax, data["static_rectangles"])
                _draw_obstacles(ax, data["obstacles"])

                # Guide path (thin)
                ax.plot(
                    case.guide_path_xyz[:, 0], case.guide_path_xyz[:, 1],
                    "--", color="#9CA3AF", linewidth=0.6, alpha=0.7,
                )

                # Trajectory so far
                ax.plot(
                    path[: source_idx + 1, 0],
                    path[: source_idx + 1, 1],
                    color="#ff7f0e", linewidth=1.5,
                )

                # Start
                ax.scatter(*case.start_position[:2], marker="s", color="green", s=30, zorder=8)

                # Goal
                if case.terminal_goal is not None:
                    ax.scatter(*case.terminal_goal[:2], marker="*", s=60, color=GOAL_COLOR, zorder=10)

                # Initial yaw arrow
                arrow_len = 0.06 * max(limits[1] - limits[0], limits[3] - limits[2])
                if math.isfinite(case.start_yaw):
                    draw_heading_arrow(
                        ax, case.start_position[:2], case.start_yaw,
                        color="#111827", length_m=arrow_len, linewidth=1.2,
                    )

                # Current yaw
                if data["has_temporal"] and math.isfinite(float(result.samples[source_idx, 13])):
                    draw_heading_arrow(
                        ax, path[source_idx, :2],
                        float(result.samples[source_idx, 13]),
                        color="#1D4ED8", length_m=arrow_len, linewidth=1.5,
                    )

                # Goal yaw
                if case.terminal_goal is not None:
                    goal_yaw_val = result.diagnostics.get("goal_yaw_rad", None)
                    if goal_yaw_val is None or not math.isfinite(float(goal_yaw_val)):
                        ax.annotate(
                            "goal yaw: N/A", xy=case.terminal_goal[:2],
                            xytext=(3, -10), textcoords="offset points",
                            color=GOAL_COLOR, fontsize=5, zorder=10,
                        )
                    else:
                        draw_heading_arrow(
                            ax, case.terminal_goal[:2], float(goal_yaw_val),
                            color=GOAL_COLOR, length_m=arrow_len,
                            hollow=True, linewidth=1.0,
                        )

                # Robot footprint
                centre = path[source_idx, :2]
                ax.add_patch(
                    plt.matplotlib.patches.Circle(
                        centre, footprint_radius_m, color="#ff7f0e", alpha=0.3, zorder=7,
                    )
                )

                sample = result.samples[source_idx] if data["has_temporal"] else None
                speed = float(np.linalg.norm(sample[4:7])) if sample is not None and len(sample) >= 7 else float("nan")
                yaw_deg = float(np.degrees(sample[13])) if sample is not None and len(sample) >= 14 else float("nan")
                margin = _sfc_margin_at_point(path[source_idx, :2], data["sfc_cells"])
                initial_velocity = np.asarray(case.start_velocity[:2], dtype=float)
                initial_acceleration = np.asarray(case.start_acceleration[:2], dtype=float)
                ax.text(
                    0.02, 0.02,
                    "init: v=(%.2f,%.2f), a=(%.2f,%.2f), yaw=%.0f°\n"
                    "current: v=%.2f, yaw=%.0f° | SFC=%d, margin=%s" % (
                        initial_velocity[0], initial_velocity[1],
                        initial_acceleration[0], initial_acceleration[1],
                        np.degrees(case.start_yaw), speed, yaw_deg, len(data["sfc_cells"]),
                        "N/A" if margin == "" else "%.3f m" % margin,
                    ),
                    transform=ax.transAxes, fontsize=5.2, va="bottom",
                    bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "#64748B"},
                    zorder=20,
                )

                _configure_axis(ax, limits, "")
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xlabel("")
                ax.set_ylabel("")

        # Row labels (y-axis factor)
        for ri in range(n_rows):
            ax = axes[ri, 0]
            level_text = f"{row_levels[ri]:.2g}" if ri < len(row_levels) else ""
            ax.set_ylabel(f"{row_label}\n{level_text}", fontsize=7, labelpad=2)

        # Column labels (x-axis factor) — only on top row
        for ci in range(n_cols):
            ax = axes[0, ci]
            level_text = f"{col_levels[ci]:.2g}" if ci < len(col_levels) else ""
            ax.set_title(f"{col_label}={level_text}", fontsize=7, pad=2)

        # Suptitle
        fig.suptitle(
            f"{row_label} × {col_label}  |  "
            f"frame {encoded_index + 1}/{max_frames}  |  "
            f"display t={encoded_index * GRID_GIF_FRAME_DURATION_S:.2f}s",
            fontsize=8, y=0.995,
        )

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    # imageio's GIF backend may silently omit a sub-second delay depending on
    # the installed Pillow plugin.  Encode the delay as integer milliseconds
    # through Pillow so grid playback is reproducibly reviewable.
    images = [Image.fromarray(frame).convert("P") for frame in frames]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(GRID_GIF_FRAME_DURATION_S * 1000),
        loop=0,
        disposal=2,
    )
    return gif_path


def build_paired_static_gif_evidence(
    gif_path: Path | str,
    *,
    case: StaticCase,
    legacy_result: StaticRunResult,
    legacy_detail: Mapping[str, np.ndarray],
    legacy_gif_path: Path | str,
    safe_result: StaticRunResult,
    safe_detail: Mapping[str, np.ndarray],
    safe_gif_path: Path | str,
) -> tuple[Path, ...]:
    """Build a paired legacy/safe GIF package without inventing joint metrics."""
    from experiments.visualizers.video_evidence import (
        build_video_evidence_package,
    )

    gif_path = Path(gif_path).resolve()
    decoded_count = len(imageio.mimread(gif_path))

    def profile_rows(
        result: StaticRunResult,
        detail: Mapping[str, np.ndarray],
        profile_gif_path: Path | str,
    ) -> list[dict[str, object]]:
        source_path = _path(result)
        if not len(source_path):
            source_path = case.guide_path_xyz
        profile_count = len(imageio.mimread(Path(profile_gif_path)))
        indices = np.unique(
            np.linspace(0, len(source_path) - 1, profile_count).astype(int)
        )
        rows = _static_frame_rows(case, result, detail, indices, profile_count)
        if not rows:
            return []
        while len(rows) < decoded_count:
            frozen = dict(rows[-1])
            frozen["frame_index"] = len(rows)
            frozen["time_s"] = len(rows) * GIF_FRAME_DURATION_S
            frozen["data_availability"] = "MEASURED_STATIC_LAST_FRAME_FROZEN"
            rows.append(frozen)
        return rows[:decoded_count]

    legacy_rows = profile_rows(legacy_result, legacy_detail, legacy_gif_path)
    safe_rows = profile_rows(safe_result, safe_detail, safe_gif_path)
    primary_rows = safe_rows if safe_rows else legacy_rows
    for row in primary_rows:
        row["optimizer_state"] = (
            f"legacy={legacy_result.status};superplanner_sfc_v1={safe_result.status}"
        )
        row["event_tags"] = "PAIRED_LEGACY_SUPERPLANNER_SFC"
    events: list[dict[str, object]] = []
    for label, result, rows in (
        ("legacy", legacy_result, legacy_rows),
        ("safe", safe_result, safe_rows),
    ):
        for event in _static_events(rows, result):
            event = dict(event)
            event["event_uid"] = f"evt-{label}-{str(event['event_uid']).removeprefix('evt-')}"
            event["source_uid"] = label
            event["description_zh"] = (
                f"{label} 面板：{event['description_zh']}"
            )
            events.append(event)
    package = gif_path.with_name(f"{gif_path.stem}_evidence")
    safe_uid = re.sub(r"[^A-Za-z0-9_.-]+", "-", case.case_uid)
    build_video_evidence_package(
        gif_path,
        package,
        evidence_uid=f"static-{safe_uid}-legacy-vs-safe-gif-evidence",
        media_uid=f"static-{safe_uid}-legacy-vs-safe-gif",
        data_source=case.case_source,
        fps=1.0 / GIF_FRAME_DURATION_S,
        case_uid=case.case_uid,
        frame_rows=primary_rows,
        event_rows=events,
        caption_overrides={
            "scene_zh": f"静态案例 {case.case_uid}（{case.expected_category}）",
            "method_zh": "原始 MINCO（无走廊）与 SuperPlanner 二维 SFC 约束 MINCO 左右配对",
            "research_question_zh": (
                f"{case.expected_category} 静态条件下 legacy 与 "
                "SuperPlanner 二维 SFC 约束配置的轨迹对比。"
            ),
            "pairing_key_zh": (
                f"case_uid={case.case_uid}；profile=legacy_vs_superplanner_sfc_v1。"
            ),
            "denominator_zh": (
                f"解码帧 n={decoded_count}；较短面板末帧冻结后仍计入分母。"
            ),
            "limitation_zh": (
                "不能外推真实闭环、动态障碍、接触或统计总体性能。"
            ),
            "interpretation_zh": (
                "左右差异由图像与分方法事件解释；逐帧定量列仅对应 SuperPlanner SFC 面板。"
            ),
            "time_basis_zh": (
                "配对 GIF 相对时间轴；较短面板末帧冻结；source_time_s "
                "仅对应逐帧指标所采用的 SuperPlanner SFC 面板"
            ),
            "metrics_units_zh": (
                "逐帧定量列仅对应 SuperPlanner SFC 面板；位置/间隙/SFC 裕度：m；"
                "时间：s；动力学量采用 SI 单位"
            ),
            "event_definition_zh": (
                "legacy 与 SuperPlanner SFC 事件以 source_uid 区分；未合成跨方法数值"
            ),
            "synchronization_zh": (
                "按帧序号相对对齐，短流冻结末帧；不声称墙钟同步"
            ),
            "missing_data_zh": (
                "不可用量保持空值；逐帧表不把左右方法合成为虚构平均值"
            ),
            "failure_handling_zh": (
                f"legacy={legacy_result.status}；superplanner_sfc_v1={safe_result.status}；"
                "失败保留在分母和事件表"
            ),
            "evidence_boundary_zh": (
                "仅证明同一静态 case 的两种约束配置轨迹可视化与已记录解析量"
            ),
            "conclusion_limit_zh": (
                "不能外推真实闭环、动态障碍、接触或统计总体性能"
            ),
        },
    )
    return _finalize_static_gif_package(
        package,
        case=case,
        profile="legacy_vs_superplanner_sfc_v1",
        status_text=f"legacy={legacy_result.status};safe={safe_result.status}",
        decoded_count=decoded_count,
        interpretation=(
            f"legacy={legacy_result.status}，superplanner_sfc_v1={safe_result.status}；"
            "逐帧定量列仅对应 SuperPlanner SFC 面板，左右差异由图像与分方法事件解释。"
        ),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _extent(case: StaticCase) -> list[float]:
    height, width = case.occupancy.shape
    x0, y0 = case.esdf_origin
    return [
        x0,
        x0 + width * case.esdf_resolution,
        y0,
        y0 + height * case.esdf_resolution,
    ]


def _path(result: StaticRunResult) -> np.ndarray:
    if len(result.samples):
        return result.samples[:, 1:4]
    return result.waypoints


def _draw_map(ax: plt.Axes, case: StaticCase) -> None:
    extent = _extent(case)
    if case.esdf_available:
        ax.imshow(
            case.esdf_distance,
            origin="lower",
            extent=extent,
            cmap="RdYlBu",
            alpha=0.55,
            aspect="equal",
        )
    ax.imshow(
        case.occupancy,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        alpha=0.75,
        vmin=0,
        vmax=1,
        aspect="equal",
    )


def _corridor_segments_from_diagnostics(
    diagnostics: Mapping[str, Any],
) -> list[tuple[float, float, float, float, float, int]]:
    """Extract corridor capsule segments from safe_corridor_v1 diagnostics.

    Returns empty list if corridor data is unavailable (legacy or missing).
    Never fabricates corridors from trajectory appearance.
    """
    raw = diagnostics.get("corridor_segments", None)
    if raw is None:
        return []
    segments: list[tuple[float, float, float, float, float, int]] = []
    observed: set[tuple[float, ...]] = set()
    if isinstance(raw, np.ndarray):
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 5:
            return []
        for row in arr:
            if len(row) >= 7:
                x0, y0, x1, y1, radius = float(row[0]), float(row[1]), float(row[3]), float(row[4]), float(row[6])
            elif len(row) >= 5:
                x0, y0, x1, y1, radius = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
            else:
                continue
            if not all(math.isfinite(v) for v in (x0, y0, x1, y1, radius)):
                continue
            if radius <= 0.0:
                continue
            key = (round(x0, 10), round(y0, 10), round(x1, 10), round(y1, 10), round(radius, 10))
            if key not in observed:
                observed.add(key)
                segments.append((x0, y0, x1, y1, radius, 0))
    return segments


def _sfc_cells_from_diagnostics(diagnostics: Mapping[str, Any]) -> list[dict[str, np.ndarray]]:
    """Return only explicitly emitted SuperPlanner SFC cells.

    Cells are never reconstructed from an optimized trajectory: the plotted
    polygon must originate in the native planner receipt, otherwise the
    visualisation intentionally stays empty.
    """
    raw = diagnostics.get("sfc_cells", None)
    if not isinstance(raw, (list, tuple)):
        return []
    cells: list[dict[str, np.ndarray]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        try:
            vertices = np.asarray(item.get("vertices_xy_m", item.get("vertices", [])), dtype=float)
            halfspaces = np.asarray(item.get("halfspaces", []), dtype=float)
        except (TypeError, ValueError):
            continue
        if vertices.ndim != 2 or vertices.shape[1] < 2 or len(vertices) < 3:
            continue
        if not np.all(np.isfinite(vertices[:, :2])):
            continue
        if halfspaces.ndim != 2 or halfspaces.shape[1] < 3 or not np.all(np.isfinite(halfspaces[:, :3])):
            continue
        cells.append({"vertices": vertices[:, :2], "halfspaces": halfspaces[:, :3]})
    return cells


def _draw_sfc_cells(ax: Any, cells: list[dict[str, np.ndarray]], *, alpha: float = 0.18) -> None:
    """Draw native SFC convex cells and their boundaries, once per panel."""
    for index, cell in enumerate(cells):
        patch = plt.matplotlib.patches.Polygon(
            cell["vertices"], closed=True, facecolor="#0D9488", edgecolor="#047857",
            linewidth=1.0, alpha=alpha, zorder=2,
            label="SuperPlanner SFC" if index == 0 else "_nolegend_",
        )
        ax.add_patch(patch)


def _sfc_margin_at_point(point_xy: np.ndarray, cells: list[dict[str, np.ndarray]]) -> object:
    """Union-cell signed face margin in metres; blank when no native SFC exists."""
    point = np.asarray(point_xy, dtype=float).reshape(-1)
    if len(point) < 2 or not np.all(np.isfinite(point[:2])):
        return ""
    best = -math.inf
    for cell in cells:
        halfspaces = cell["halfspaces"]
        # n_x*x + n_y*y + offset <= 0.  Normals are unit length in the
        # serialized native representation, so this is a metric margin.
        margins = -(halfspaces[:, :2] @ point[:2] + halfspaces[:, 2])
        if len(margins):
            best = max(best, float(np.min(margins)))
    return _finite_or_blank(best) if math.isfinite(best) else ""


def _obstacles_from_diagnostics(
    diagnostics: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Extract dynamic obstacle states from diagnostics."""
    raw = diagnostics.get("obstacle_states", None)
    if raw is None:
        return []
    obstacles: list[dict[str, object]] = []
    for i, obs in enumerate(raw or ()):
        if isinstance(obs, Mapping):
            pos = np.asarray(obs.get("position_xy_m", obs.get("center_xy", [])), dtype=float).reshape(-1)
            vel = np.asarray(obs.get("velocity_xy_mps", [0.0, 0.0]), dtype=float).reshape(-1)
            radius = float(obs.get("radius_m", 0.0))
        elif isinstance(obs, np.ndarray):
            arr = np.asarray(obs, dtype=float).reshape(-1)
            if len(arr) >= 3:
                pos, vel, radius = arr[:2], np.zeros(2), arr[2]
            else:
                continue
        else:
            continue
        if len(pos) < 2 or radius <= 0.0:
            continue
        obstacles.append({
            "cycle_index": 0,
            "obstacle_uid": f"obs-{i}",
            "x_m": float(pos[0]),
            "y_m": float(pos[1]),
            "vx_mps": float(vel[0]) if len(vel) >= 2 else 0.0,
            "vy_mps": float(vel[1]) if len(vel) >= 2 else 0.0,
            "radius_m": radius,
        })
    return obstacles


def _static_obstacle_rectangles(
    case: StaticCase,
) -> list[dict[str, object]]:
    """Extract static rectangle obstacles from case auxiliary arrays."""
    raw = case.auxiliary_arrays.get("static_rectangles", None)
    if raw is None:
        # Synthetic materialisation uses this explicit unit-bearing key.
        raw = case.auxiliary_arrays.get("materialization_obstacle_rectangles_xyxy_m", None)
    if raw is None:
        return []
    rectangles: list[dict[str, object]] = []
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 2:
        for i, row in enumerate(arr):
            if len(row) >= 4:
                x0, y0, x1, y1 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                if x1 > x0 and y1 > y0:
                    rectangles.append({
                        "obstacle_uid": f"wall-{i}",
                        "bounds_xy": np.array([x0, y0, x1, y1]),
                    })
    return rectangles


def render_static_case(
    case: StaticCase,
    result: StaticRunResult,
    metrics: Mapping[str, Any],
    detail: Mapping[str, np.ndarray],
    output_dir: Path | str,
    *,
    footprint_radius_m: float = 0.2,
) -> tuple[Path, ...]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / case.case_uid
    overview = prefix.with_name(prefix.name + "_overview.png")
    overview_pdf = prefix.with_name(prefix.name + "_overview.pdf")
    clearance = prefix.with_name(prefix.name + "_clearance.png")
    clearance_pdf = prefix.with_name(prefix.name + "_clearance.pdf")
    dynamics = prefix.with_name(prefix.name + "_dynamics.png")
    dynamics_pdf = prefix.with_name(prefix.name + "_dynamics.pdf")
    animation = prefix.with_name(prefix.name + "_animation.gif")
    metrics_path = prefix.with_name(prefix.name + "_metrics.json")
    path = _path(result)
    raw = case.auxiliary_arrays.get("raw_path_xyz")
    sfc_cells = _sfc_cells_from_diagnostics(result.diagnostics)
    static_rects = _static_obstacle_rectangles(case)

    figure, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    _draw_map(ax, case)
    ax.plot(
        case.guide_path_xyz[:, 0],
        case.guide_path_xyz[:, 1],
        "o--",
        color="#1f77b4",
        label="guide / sparse input",
    )
    if raw is not None:
        ax.plot(raw[:, 0], raw[:, 1], color="#7f7f7f", label="RAW")
    if len(path):
        ax.plot(path[:, 0], path[:, 1], color="#ff7f0e", linewidth=2, label="MINCO")
    _draw_sfc_cells(ax, sfc_cells, alpha=0.20)
    _draw_static_rectangles(ax, static_rects)
    ax.scatter(*case.start_position[:2], marker="s", color="green", label="start")
    if case.terminal_goal is not None:
        ax.scatter(*case.terminal_goal[:2], marker="*", s=120, color="purple", label="goal")
    clearance_xy = np.asarray(detail.get("clearance_xy", np.empty((0, 2))))
    clearance_values = np.asarray(detail.get("clearance_m", np.empty(0)))
    valid = np.asarray(detail.get("clearance_valid", np.empty(0, dtype=bool)))
    dangerous = valid & (
        clearance_values <= float(metrics.get("safe_dist_m", 0.15))
    )
    if len(clearance_xy) and np.any(dangerous):
        ax.scatter(
            clearance_xy[dangerous, 0],
            clearance_xy[dangerous, 1],
            s=16,
            color="red",
            label="dangerous",
        )
    if len(clearance_xy) and np.any(~valid):
        ax.scatter(
            clearance_xy[~valid, 0],
            clearance_xy[~valid, 1],
            s=18,
            marker="x",
            color="black",
            label="OOB",
        )
    ax.add_patch(
        Circle(
            case.start_position[:2],
            footprint_radius_m,
            fill=False,
            color="green",
            linewidth=1.5,
        )
    )
    overview_points = [
        case.guide_path_xyz[:, :2],
        np.asarray(case.start_position[:2], dtype=float).reshape(1, 2),
    ]
    if len(path):
        overview_points.append(path[:, :2])
    if case.terminal_goal is not None:
        overview_points.append(np.asarray(case.terminal_goal[:2], dtype=float).reshape(1, 2))
    for rectangle in static_rects:
        x0, y0, x1, y1 = rectangle["bounds_xy"]
        overview_points.append(np.asarray([[x0, y0], [x1, y1]], dtype=float))
    for cell in sfc_cells:
        overview_points.append(cell["vertices"])
    overview_all = np.concatenate(overview_points, axis=0)
    overview_min = np.min(overview_all, axis=0)
    overview_max = np.max(overview_all, axis=0)
    overview_span = np.maximum(overview_max - overview_min, 1.0)
    overview_margin = max(0.4, 0.08 * float(np.max(overview_span)))
    overview_limits = (
        float(overview_min[0] - overview_margin),
        float(overview_max[0] + overview_margin),
        float(overview_min[1] - overview_margin),
        float(overview_max[1] + overview_margin),
    )
    arrow_length = 0.075 * max(
        overview_limits[1] - overview_limits[0],
        overview_limits[3] - overview_limits[2],
    )
    draw_heading_arrow(
        ax, case.start_position[:2], case.start_yaw, color="#111827",
        length_m=arrow_length, label="initial yaw",
    )
    if len(path) and len(result.samples) == len(path) and result.samples.shape[1] > 13:
        current_yaw = float(result.samples[-1, 13])
        if math.isfinite(current_yaw):
            draw_heading_arrow(
                ax, path[-1, :2], current_yaw, color="#1D4ED8",
                length_m=arrow_length, label="final yaw",
            )
    if case.terminal_goal is not None:
        goal_yaw = result.diagnostics.get("goal_yaw_rad")
        try:
            goal_yaw = float(goal_yaw)
        except (TypeError, ValueError):
            goal_yaw = math.nan
        if math.isfinite(goal_yaw):
            draw_heading_arrow(
                ax, case.terminal_goal[:2], goal_yaw, color=GOAL_COLOR,
                length_m=arrow_length, hollow=True, label="goal yaw",
            )
        else:
            ax.annotate("goal yaw: N/A", xy=case.terminal_goal[:2], xytext=(5, -14),
                        textcoords="offset points", color=GOAL_COLOR, fontsize=8)
    failure_reason = str(
        result.diagnostics.get("sfc_generation_reason")
        or result.diagnostics.get("validation_failure_reason")
        or result.diagnostics.get("failure_reason")
        or "NONE"
    )
    ax.text(
        0.02, 0.98,
        "initial: v=(%.2f, %.2f) m/s; a=(%.2f, %.2f) m/s²; yaw=%.1f°\n"
        "evidence: one-shot local MINCO preview (not full-route execution)\n"
        "profile: %s | status: %s | SFC cells: %d | reason: %s" % (
            case.start_velocity[0], case.start_velocity[1],
            case.start_acceleration[0], case.start_acceleration[1],
            math.degrees(case.start_yaw), case.constraint_profile,
            result.status, len(sfc_cells), failure_reason,
        ),
        transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#64748B"},
        zorder=20,
    )
    ax.set_xlim(overview_limits[0], overview_limits[1])
    ax.set_ylim(overview_limits[2], overview_limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{case.case_uid} · local MINCO preview · {result.status}"
    )
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.legend(loc="best", fontsize=8)
    figure.savefig(overview, dpi=300, facecolor="white")
    figure.savefig(
        overview_pdf,
        metadata={"Creator": "NavDP research workflow", "CreationDate": None},
    )
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    s = np.asarray(detail.get("clearance_s_m", np.empty(0)))
    if len(s):
        ax.plot(s, clearance_values, color="#1f77b4")
        ax.axhline(
            float(metrics.get("safe_dist_m", 0.15)),
            color="red",
            linestyle="--",
            label="safe distance",
        )
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "ESDF/path evidence unavailable", ha="center", va="center")
    ax.set(xlabel="arc length (m)", ylabel="clearance (m)", title=f"{case.case_uid} clearance")
    figure.savefig(clearance, dpi=300, facecolor="white")
    figure.savefig(
        clearance_pdf,
        metadata={"Creator": "NavDP research workflow", "CreationDate": None},
    )
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    t = np.asarray(detail.get("t_s", np.empty(0)))
    if len(t):
        axes[0].plot(t, detail["speed_mps"], label="speed")
        axes[0].plot(t, detail["acc_mps2"], label="acceleration")
        axes[1].plot(t, detail["jerk_mps3"], label="jerk", color="#d62728")
        axes[2].plot(t, result.samples[:, 13], label="yaw")
        axes[2].plot(t, detail["yaw_rate_radps"], label="yaw rate")
        for axis in axes:
            axis.legend(loc="best")
            axis.grid(alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "analytic MINCO samples unavailable", ha="center")
    axes[0].set_ylabel("m/s, m/s²")
    axes[1].set_ylabel("m/s³")
    axes[2].set_ylabel("rad, rad/s")
    axes[2].set_xlabel("time (s)")
    figure.suptitle(f"{case.case_uid} analytic dynamics")
    figure.savefig(dynamics, dpi=300, facecolor="white")
    figure.savefig(
        dynamics_pdf,
        metadata={"Creator": "NavDP research workflow", "CreationDate": None},
    )
    plt.close(figure)

    animation_path = path if len(path) else case.guide_path_xyz
    base_frames = min(32, len(animation_path))
    frame_indices = np.unique(
        np.linspace(0, len(animation_path) - 1, base_frames).astype(int)
    )
    # Terminal hold frames intentionally not appended: the Task 3 single-case
    # GIF tests assert exactly min(32, len(path)) decoded frames.
    total_frames = base_frames

    corridor_segments = _corridor_segments_from_diagnostics(result.diagnostics)
    sfc_cells = _sfc_cells_from_diagnostics(result.diagnostics)
    obstacles = _obstacles_from_diagnostics(result.diagnostics)
    static_rects = _static_obstacle_rectangles(case)

    # Compute view limits from all geometry
    all_points = [
        case.guide_path_xyz[:, :2],
        animation_path[:, :2],
    ]
    if case.terminal_goal is not None:
        all_points.append(case.terminal_goal[:2].reshape(1, 2))
    for x0, y0, x1, y1, radius, _ in corridor_segments:
        all_points.append(np.array([[x0 - radius, y0 - radius], [x1 + radius, y1 + radius]]))
    for cell in sfc_cells:
        all_points.append(cell["vertices"])
    for obs in obstacles:
        r = float(obs["radius_m"])
        all_points.append(np.array([[float(obs["x_m"]) - r, float(obs["y_m"]) - r],
                                     [float(obs["x_m"]) + r, float(obs["y_m"]) + r]]))
    combined = np.concatenate(all_points, axis=0)
    vmin = np.min(combined, axis=0)
    vmax = np.max(combined, axis=0)
    span = np.maximum(vmax - vmin, 1.0)
    margin = max(0.4, 0.08 * float(max(span)))
    limits = [float(vmin[0] - margin), float(vmax[0] + margin),
              float(vmin[1] - margin), float(vmax[1] + margin)]
    arrow_length = 0.08 * max(limits[1] - limits[0], limits[3] - limits[2])

    has_temporal = len(result.samples) == len(animation_path) and len(result.samples) > 0
    goal_yaw_value = result.diagnostics.get("goal_yaw_rad", None)
    goal_yaw = None
    if goal_yaw_value is not None:
        try:
            goal_yaw = float(goal_yaw_value)
        except (TypeError, ValueError):
            goal_yaw = None
    if goal_yaw is not None and not math.isfinite(goal_yaw):
        goal_yaw = None

    frames = []
    for encoded_index in range(total_frames):
        progress_index = min(encoded_index, base_frames - 1)
        source_idx = int(frame_indices[progress_index])

        fig, ax = plt.subplots(figsize=(6, 6.4))
        fig.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.30)
        _draw_map(ax, case)

        # Guide path
        ax.plot(
            case.guide_path_xyz[:, 0], case.guide_path_xyz[:, 1],
            "--", color="#4B5563", linewidth=1.0, alpha=0.7,
        )

        # Trajectory so far
        ax.plot(
            animation_path[: source_idx + 1, 0],
            animation_path[: source_idx + 1, 1],
            color="#ff7f0e", linewidth=2,
        )

        # Start marker
        ax.scatter(*case.start_position[:2], marker="s", color="green", s=60, zorder=8)

        # Goal marker
        if case.terminal_goal is not None:
            ax.scatter(*case.terminal_goal[:2], marker="*", s=120, color=GOAL_COLOR, zorder=10)

        # Heading arrows: initial yaw
        if math.isfinite(case.start_yaw):
            draw_heading_arrow(
                ax, case.start_position[:2], case.start_yaw,
                color="#111827", length_m=arrow_length, linewidth=2.0,
                label="initial yaw",
            )

        # Heading arrows: current yaw (sample at robot position)
        if has_temporal and math.isfinite(float(result.samples[source_idx, 13])):
            draw_heading_arrow(
                ax, animation_path[source_idx, :2],
                float(result.samples[source_idx, 13]),
                color="#1D4ED8", length_m=arrow_length, linewidth=2.0,
            )

        # Heading arrows: goal yaw
        if case.terminal_goal is not None:
            if goal_yaw is None:
                ax.annotate(
                    "goal yaw: N/A",
                    xy=case.terminal_goal[:2],
                    xytext=(8, -18), textcoords="offset points",
                    color=GOAL_COLOR, fontsize=8, zorder=10,
                )
            else:
                draw_heading_arrow(
                    ax, case.terminal_goal[:2], goal_yaw,
                    color=GOAL_COLOR, length_m=arrow_length,
                    hollow=True, linewidth=1.8,
                )

        # Safe corridor capsules (only if real data exists)
        for segment in corridor_segments:
            _draw_capsule(ax, segment, alpha=0.14)
        _draw_sfc_cells(ax, sfc_cells)

        # Obstacles
        _draw_obstacles(ax, obstacles)
        _draw_static_rectangles(ax, static_rects)

        # Robot footprint
        centre = animation_path[source_idx, :2]
        ax.add_patch(
            plt.matplotlib.patches.Circle(
                centre, footprint_radius_m, color="#ff7f0e", alpha=0.35, zorder=7,
            )
        )

        _configure_axis(ax, limits, f"{case.case_uid} · {case.case_source}")

        # Status box
        sample = result.samples[source_idx] if has_temporal else None
        if sample is not None and len(sample) >= 15:
            speed = float(np.linalg.norm(sample[4:7]))
            acceleration = float(np.linalg.norm(sample[7:10]))
            state_text = (
                f"x/y: {sample[1]:.2f}/{sample[2]:.2f} m\n"
                f"yaw: {sample[13]:.2f} rad / {math.degrees(sample[13]):.1f} deg\n"
                f"v/a: {speed:.2f} m/s / {acceleration:.2f} m/s²\n"
                f"yaw rate: {sample[14]:.2f} rad/s\n"
            )
        else:
            state_text = "state: N/A\n"

        clearance_val = _nearest_clearance(centre, detail)
        sfc_margin = _sfc_margin_at_point(centre, sfc_cells)

        ax.text(
            0.02, -0.055,
            f"{case.constraint_profile}\n"
                "mode: one-shot local MINCO preview (not full-route execution)\n"
                f"time: {encoded_index * GIF_FRAME_DURATION_S:.2f}s\n"
            f"{state_text}"
            f"clearance: {clearance_val if clearance_val != '' else 'N/A'} m\n"
            f"SFC margin: {sfc_margin if sfc_margin != '' else 'N/A'} m\n"
            f"status: {result.status}\n"
            f"reason: {result.diagnostics.get('sfc_generation_reason') or result.diagnostics.get('validation_failure_reason') or result.diagnostics.get('failure_reason') or 'NONE'}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "alpha": 0.88, "edgecolor": "#6B7280"},
            zorder=20,
            clip_on=False,
        )

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(animation, frames, duration=GIF_FRAME_DURATION_S, loop=0)
    evidence_paths = build_static_gif_evidence(
        animation,
        case=case,
        result=result,
        metrics=metrics,
        detail=detail,
        frame_source_indices=frame_indices,
    )

    payload = {
        "schema_version": 1,
        "case_uid": case.case_uid,
        "case_source": case.case_source,
        "case_hash": case.case_hash,
        "mode": result.mode,
        "status": result.status,
        "engine": result.engine,
        "diagnostics": result.diagnostics,
        "metrics": dict(metrics),
    }
    metrics_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        overview,
        overview_pdf,
        clearance,
        clearance_pdf,
        dynamics,
        dynamics_pdf,
        animation,
        metrics_path,
        *evidence_paths,
    )
