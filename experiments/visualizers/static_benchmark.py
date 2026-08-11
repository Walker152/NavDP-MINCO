from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from experiments.static.case_schema import StaticCase
from experiments.static.runner import StaticRunResult


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
    clearance = prefix.with_name(prefix.name + "_clearance.png")
    dynamics = prefix.with_name(prefix.name + "_dynamics.png")
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
    figure.savefig(overview, dpi=140)
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
    else:
        ax.text(0.5, 0.5, "ESDF/path evidence unavailable", ha="center", va="center")
    ax.set(xlabel="arc length (m)", ylabel="clearance (m)", title=f"{case.case_uid} clearance")
    ax.legend(loc="best")
    figure.savefig(clearance, dpi=140)
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
    figure.savefig(dynamics, dpi=140)
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
    return overview, clearance, dynamics, animation, metrics_path
