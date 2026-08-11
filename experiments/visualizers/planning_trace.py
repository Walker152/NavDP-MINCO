from __future__ import annotations

from pathlib import Path
from typing import Mapping
import warnings

import numpy as np


def _plot_module():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    return plt


def render_trajectory_overview(
    arrays: Mapping[str, np.ndarray],
    output_path: Path,
    *,
    title: str,
    data_source: str,
) -> None:
    plt = _plot_module()
    figure, axis = plt.subplots(figsize=(8, 7))
    distance = arrays.get("esdf_distance")
    origin = arrays.get("esdf_origin")
    resolution = arrays.get("esdf_resolution")
    if (
        distance is not None
        and distance.ndim == 2
        and origin is not None
        and np.asarray(origin).size >= 2
        and resolution is not None
        and np.asarray(resolution).size >= 1
    ):
        resolution_value = float(np.asarray(resolution).reshape(-1)[0])
        origin_xy = np.asarray(origin).reshape(-1)[:2]
        extent = (
            origin_xy[0],
            origin_xy[0] + distance.shape[1] * resolution_value,
            origin_xy[1],
            origin_xy[1] + distance.shape[0] * resolution_value,
        )
        image = axis.imshow(
            distance,
            origin="lower",
            extent=extent,
            cmap="viridis",
            alpha=0.65,
        )
        figure.colorbar(image, ax=axis, label="ESDF clearance (m)")

    styles = (
        ("raw_path_xy", "RAW Top-1", "--", "#d62728"),
        ("selected_candidate_xy", "Selected candidate", ":", "#17becf"),
        ("sparse_waypoints", "Sparse guide", "-.", "#f1c40f"),
    )
    for field, label, linestyle, color in styles:
        points = arrays.get(field)
        if points is not None and points.ndim == 2 and points.shape[1] >= 2:
            axis.plot(
                points[:, 0],
                points[:, 1],
                label=label,
                linestyle=linestyle,
                color=color,
                linewidth=2,
            )

    samples = arrays.get("minco_samples")
    if samples is not None and samples.ndim == 2 and samples.shape[1] >= 3:
        axis.plot(
            samples[:, 1],
            samples[:, 2],
            label="MINCO analytic trajectory",
            color="#2ca02c",
            linewidth=2.4,
        )
        axis.scatter(
            [samples[0, 1], samples[-1, 1]],
            [samples[0, 2], samples[-1, 2]],
            marker="o",
            color=["black", "magenta"],
            zorder=5,
            label="Start / end",
        )

    axis.set_title(title)
    axis.set_xlabel("World x (m)")
    axis.set_ylabel("World y (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.text(0.01, 0.01, f"Data source: {data_source}", fontsize=8)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def render_clearance_figure(
    arrays: Mapping[str, np.ndarray],
    output_path: Path,
    *,
    title: str,
    data_source: str,
) -> bool:
    plt = _plot_module()
    figure, axis = plt.subplots(figsize=(8, 4))
    clearance = arrays.get("minco_clearance")
    sampled_s = arrays.get("sampled_s")
    available = (
        clearance is not None
        and sampled_s is not None
        and np.asarray(clearance).ndim == 1
        and np.asarray(sampled_s).ndim == 1
        and len(clearance) == len(sampled_s)
        and len(clearance) > 0
    )
    if available:
        axis.plot(sampled_s, clearance, color="#2ca02c")
        axis.set_xlabel("Arc length (m)")
        axis.set_ylabel("Clearance (m)")
        axis.grid(True, alpha=0.25)
    else:
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "N/A — ESDF/clearance samples were not recorded",
            ha="center",
            va="center",
            fontsize=14,
        )
    axis.set_title(title)
    figure.text(0.01, 0.01, f"Data source: {data_source}", fontsize=8)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return bool(available)


def render_dynamics_figure(
    samples: np.ndarray | None,
    output_path: Path,
    *,
    title: str,
    data_source: str,
) -> bool:
    plt = _plot_module()
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    available = (
        samples is not None
        and samples.ndim == 2
        and samples.shape[1] >= 15
        and len(samples) >= 2
    )
    if available:
        t = samples[:, 0]
        speed = np.linalg.norm(samples[:, 4:7], axis=1)
        acceleration = np.linalg.norm(samples[:, 7:10], axis=1)
        jerk = np.linalg.norm(samples[:, 10:13], axis=1)
        axes[0].plot(t, speed, label="speed (m/s)")
        axes[0].plot(t, acceleration, label="acceleration (m/s²)")
        axes[0].legend()
        axes[1].plot(t, jerk, label="jerk (m/s³)", color="#d62728")
        axes[1].legend()
        axes[2].plot(t, samples[:, 13], label="yaw (rad)")
        axes[2].plot(t, samples[:, 14], label="yaw rate (rad/s)")
        axes[2].legend()
        axes[2].set_xlabel("Trajectory time (s)")
        for axis in axes:
            axis.grid(True, alpha=0.25)
    else:
        for axis in axes:
            axis.axis("off")
        axes[1].text(
            0.5,
            0.5,
            "N/A — analytic MINCO samples were not recorded",
            ha="center",
            va="center",
            fontsize=14,
        )
    figure.suptitle(title)
    figure.text(0.01, 0.01, f"Data source: {data_source}", fontsize=8)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return bool(available)
