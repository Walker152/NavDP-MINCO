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

from experiments.static.case_schema import StaticCase
from experiments.static.runner import StaticRunResult


GIF_FRAME_DURATION_S = 0.1


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
            "progress_ratio": (
                float(source_index) / float(max(1, len(path) - 1))
            ),
            "clearance_m": _nearest_clearance(point[:2], detail),
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
            f"legacy={legacy_result.status};safe_corridor_v1={safe_result.status}"
        )
        row["event_tags"] = "PAIRED_LEGACY_SAFE"
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
            "method_zh": "legacy 与 safe_corridor_v1 左右配对",
            "time_basis_zh": (
                "配对 GIF 相对时间轴；较短面板末帧冻结；source_time_s "
                "仅对应逐帧指标所采用的 safe 面板"
            ),
            "metrics_units_zh": (
                "逐帧定量列仅对应 safe_corridor_v1 面板；位置/间隙：m；"
                "时间：s；动力学量采用 SI 单位"
            ),
            "event_definition_zh": (
                "legacy 与 safe 事件以 source_uid 区分；未合成跨方法数值"
            ),
            "synchronization_zh": (
                "按帧序号相对对齐，短流冻结末帧；不声称墙钟同步"
            ),
            "missing_data_zh": (
                "不可用量保持空值；逐帧表不把左右方法合成为虚构平均值"
            ),
            "failure_handling_zh": (
                f"legacy={legacy_result.status}；safe_corridor_v1={safe_result.status}；"
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
        profile="legacy_vs_safe_corridor_v1",
        status_text=f"legacy={legacy_result.status};safe={safe_result.status}",
        decoded_count=decoded_count,
        interpretation=(
            f"legacy={legacy_result.status}，safe_corridor_v1={safe_result.status}；"
            "逐帧定量列仅对应 safe 面板，左右差异由图像与分方法事件解释。"
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
    ax.set_title(f"{case.case_uid} · {case.case_source} · {result.status}")
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
    frame_indices = np.unique(
        np.linspace(0, len(animation_path) - 1, min(32, len(animation_path))).astype(int)
    )
    frames = []
    for index in frame_indices:
        figure, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        _draw_map(ax, case)
        ax.plot(
            case.guide_path_xyz[:, 0],
            case.guide_path_xyz[:, 1],
            "--",
            color="#1f77b4",
        )
        ax.plot(
            animation_path[: index + 1, 0],
            animation_path[: index + 1, 1],
            color="#ff7f0e",
            linewidth=2,
        )
        centre = animation_path[index, :2]
        ax.add_patch(Circle(centre, footprint_radius_m, color="#ff7f0e", alpha=0.35))
        ax.set_title(f"{case.case_uid} · {case.case_source}")
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world y (m)")
        figure.canvas.draw()
        frame = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(figure)
    imageio.mimsave(animation, frames, duration=0.1, loop=0)
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
