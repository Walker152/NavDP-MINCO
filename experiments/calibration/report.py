from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.calibration.profile import load_robot_calibration


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _placeholder(path: Path, title: str, protocol_id: str) -> None:
    figure, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=18)
    ax.text(
        0.5,
        0.42,
        "READY_FOR_CALIBRATION_RUN\nNo real-simulation receipt exists",
        ha="center",
        va="center",
        fontsize=13,
        color="#b22222",
    )
    ax.text(0.5, 0.2, protocol_id, ha="center", fontsize=10)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def render_calibration_report(
    calibration_path: Path | str,
    usd_evidence_path: Path | str,
    protocol_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    calibration_path = Path(calibration_path).resolve()
    usd_evidence_path = Path(usd_evidence_path).resolve()
    protocol_path = Path(protocol_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = load_robot_calibration(calibration_path)
    evidence = json.loads(usd_evidence_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("real_run_authorized") is not False:
        raise ValueError("report renderer expects an unexecuted calibration protocol")

    footprint_path = output_dir / "robot_footprint.png"
    figure, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    polygon = np.vstack(
        [profile.footprint_polygon_xy_m, profile.footprint_polygon_xy_m[0]]
    )
    ax.fill(polygon[:, 0], polygon[:, 1], color="#4c78a8", alpha=0.25)
    ax.plot(polygon[:, 0], polygon[:, 1], color="#4c78a8", label="collision hull")
    ax.add_patch(
        plt.Circle(
            (0, 0),
            profile.inscribed_radius_m,
            fill=False,
            linestyle="--",
            color="green",
            label="inscribed radius",
        )
    )
    ax.add_patch(
        plt.Circle(
            (0, 0),
            profile.circumscribed_radius_m,
            fill=False,
            linestyle=":",
            color="red",
            label="circumscribed radius",
        )
    )
    for wheel in evidence["wheels"]:
        centre = wheel["centre_base_m"]
        ax.scatter(centre[0], centre[1], marker="o", color="black")
    ax.arrow(0, 0, 0.12, 0, width=0.002, color="black")
    ax.text(0.125, 0, "base +x", va="center")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set(xlabel="base x (m)", ylabel="base y (m)", title="Dingo collision footprint from USD")
    ax.legend(loc="best")
    figure.savefig(footprint_path, dpi=150)
    plt.close(figure)

    frame_path = output_dir / "frame_tree_and_extrinsic.png"
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    axes[0].axis("off")
    axes[0].text(0.5, 0.85, "world", ha="center", bbox={"boxstyle": "round", "fc": "white"})
    axes[0].annotate("", xy=(0.5, 0.63), xytext=(0.5, 0.8), arrowprops={"arrowstyle": "->"})
    axes[0].text(0.5, 0.57, "base_link", ha="center", bbox={"boxstyle": "round", "fc": "#d9ead3"})
    axes[0].annotate("", xy=(0.5, 0.35), xytext=(0.5, 0.52), arrowprops={"arrowstyle": "->"})
    axes[0].text(0.5, 0.28, "front_cam_ros", ha="center", bbox={"boxstyle": "round", "fc": "#cfe2f3"})
    axes[0].text(0.5, 0.08, "T_base_camera: config-declared sensor offset", ha="center")
    image = axes[1].imshow(profile.T_base_camera, cmap="coolwarm", vmin=-1, vmax=1)
    for row in range(4):
        for column in range(4):
            axes[1].text(column, row, f"{profile.T_base_camera[row,column]:.2f}", ha="center", va="center")
    axes[1].set_title("T_base_camera (camera coordinates → base)")
    figure.colorbar(image, ax=axes[1], shrink=0.7)
    figure.savefig(frame_path, dpi=150)
    plt.close(figure)

    dynamic_paths = {
        "straight": output_dir / "straight_calibration.png",
        "rotation": output_dir / "rotation_calibration.png",
        "braking": output_dir / "braking_distance.png",
        "wall_clearance": output_dir / "wall_clearance.png",
    }
    for kind, path in dynamic_paths.items():
        _placeholder(path, kind.replace("_", " ").title(), protocol["protocol_id"])

    report_path = output_dir / "robot_calibration_report.md"
    safety = profile.raw["safety_distance"]
    report_path.write_text(
        "# Dingo robot calibration report\n\n"
        "Status: **STATIC_VERIFIED / READY_FOR_CALIBRATION_RUN**\n\n"
        "No Isaac application, navigation episode, wheel command, contact test, or "
        "dynamic calibration was run. Dynamic figures are explicit readiness cards.\n\n"
        "## Static model evidence\n\n"
        f"- USD: `{evidence['usd_path']}` (`{evidence['usd_sha256']}`)\n"
        f"- Collision prims: {len(evidence['collision_prims'])}\n"
        f"- Wheel radius from transformed collision geometry: "
        f"{evidence['wheel_radius_geometry_estimate_m']:.9f} m\n"
        f"- Wheel-centre separation: {evidence['wheel_base_geometry_estimate_m']:.9f} m\n"
        f"- Footprint inscribed/circumscribed radii: "
        f"{profile.inscribed_radius_m:.9f} / {profile.circumscribed_radius_m:.9f} m\n"
        "- `front_cam` is spawned by `CameraCfg`; its configured ROS-convention "
        "offset is therefore combined with USD evidence rather than invented as a USD prim.\n\n"
        "## Safety-distance derivation\n\n"
        + "\n".join(
            f"- {key}: {float(value):.9f} m"
            for key, value in safety["validation_components_m"].items()
        )
        + f"\n\nValidation distance: {profile.validation_safe_dist_m:.9f} m. "
        f"Optimization buffer: {profile.optimization_buffer_m:.9f} m. "
        f"Optimization distance: {profile.optimization_safe_dist_m:.9f} m.\n\n"
        "The tracking component is a one-validation-step conservative bound, not "
        "a measured delay/braking result. It must be replaced or confirmed by the "
        "authorized isolated protocol.\n\n"
        "## Static discrepancies\n\n"
        + "\n".join(f"- {item}" for item in profile.raw["static_discrepancies"])
        + "\n",
        encoding="utf-8",
    )
    artifacts = [
        footprint_path,
        frame_path,
        *dynamic_paths.values(),
        report_path,
    ]
    manifest = {
        "schema_version": 1,
        "status": "STATIC_VERIFIED_DYNAMIC_READY_FOR_CALIBRATION_RUN",
        "real_run_authorized": False,
        "calibration_path": str(calibration_path),
        "calibration_sha256": profile.calibration_sha256,
        "usd_evidence_path": str(usd_evidence_path),
        "usd_evidence_sha256": _sha256(usd_evidence_path),
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "artifacts": [
            {"path": path.name, "sha256": _sha256(path)} for path in artifacts
        ],
    }
    manifest_path = output_dir / "calibration_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
