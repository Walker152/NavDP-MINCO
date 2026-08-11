from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _number(value: Any) -> Any:
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return str(value)
    return scalar if math.isfinite(scalar) else str(scalar)


def _vector(value: Any) -> list[float]:
    return [float(component) for component in value]


def _matrix(value: Any) -> list[list[float]]:
    return [[float(value[row][column]) for column in range(4)] for row in range(4)]


def _corners(extent: Any, Gf: Any) -> list[Any]:
    return [
        Gf.Vec3d(x, y, z)
        for x in (extent[0][0], extent[1][0])
        for y in (extent[0][1], extent[1][1])
        for z in (extent[0][2], extent[1][2])
    ]


def _aabb(points: list[Any]) -> dict[str, list[float]]:
    return {
        "min": [min(float(point[axis]) for point in points) for axis in range(3)],
        "max": [max(float(point[axis]) for point in points) for axis in range(3)],
    }


def extract_usd_evidence(
    usd_path: Path | str,
    *,
    base_prim_path: str = "/dingo/base_link",
    wheel_link_paths: tuple[str, str] = (
        "/dingo/left_wheel_link",
        "/dingo/right_wheel_link",
    ),
    wheel_joint_paths: tuple[str, str] = (
        "/dingo/left_wheel_joint",
        "/dingo/right_wheel_joint",
    ),
) -> dict[str, Any]:
    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as error:
        raise RuntimeError(
            "OpenUSD Python bindings are unavailable. Add the bundled "
            "omni.usd.libs extension root to PYTHONPATH and its bin directory "
            "to LD_LIBRARY_PATH; Isaac/Kit does not need to be started."
        ) from error
    usd_path = Path(usd_path).resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"failed to open USD stage: {usd_path}")
    base = stage.GetPrimAtPath(base_prim_path)
    if not base.IsValid():
        raise ValueError(f"required base prim missing: {base_prim_path}")
    cache = UsdGeom.XformCache()
    base_world = cache.GetLocalToWorldTransform(base)
    base_origin_world = base_world.Transform(Gf.Vec3d(0.0))
    base_axes_world = [
        base_world.TransformDir(axis).GetNormalized()
        for axis in (
            Gf.Vec3d(1.0, 0.0, 0.0),
            Gf.Vec3d(0.0, 1.0, 0.0),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
    ]

    def world_point_to_base(point: Any) -> Any:
        delta = point - base_origin_world
        return Gf.Vec3d(
            *(Gf.Dot(delta, axis) for axis in base_axes_world)
        )
    collisions = []
    for prim in stage.Traverse():
        if "PhysicsCollisionAPI" not in prim.GetAppliedSchemas():
            continue
        path = str(prim.GetPath())
        if not path.startswith("/dingo/") or "/GroundPlane/" in path:
            continue
        extent = prim.GetAttribute("extent").Get()
        if extent is None or len(extent) != 2:
            raise ValueError(f"collision prim lacks usable extent: {path}")
        local_points = _corners(extent, Gf)
        local_to_world = cache.GetLocalToWorldTransform(prim)
        world_points = [local_to_world.Transform(point) for point in local_points]
        base_points = [world_point_to_base(point) for point in world_points]
        collisions.append(
            {
                "prim_path": path,
                "type_name": prim.GetTypeName(),
                "local_extent": [_vector(extent[0]), _vector(extent[1])],
                "local_to_world": _matrix(local_to_world),
                "world_aabb_m": _aabb(world_points),
                "base_aabb_m": _aabb(base_points),
                "radius_attribute": _number(prim.GetAttribute("radius").Get()),
                "axis_attribute": _number(prim.GetAttribute("axis").Get()),
            }
        )
    if not collisions:
        raise ValueError("no PhysicsCollisionAPI prims found below robot")

    wheels = []
    for link_path, joint_path in zip(wheel_link_paths, wheel_joint_paths):
        link = stage.GetPrimAtPath(link_path)
        joint = stage.GetPrimAtPath(joint_path)
        if not link.IsValid():
            raise ValueError(f"required wheel link missing: {link_path}")
        if not joint.IsValid():
            raise ValueError(f"required wheel joint missing: {joint_path}")
        link_world = cache.GetLocalToWorldTransform(link)
        centre_base = world_point_to_base(
            link_world.Transform(Gf.Vec3d(0.0))
        )
        wheel_collision = next(
            (
                row
                for row in collisions
                if row["prim_path"].startswith(link_path + "/")
            ),
            None,
        )
        if wheel_collision is None:
            raise ValueError(f"wheel collision prim missing below: {link_path}")
        bounds = wheel_collision["base_aabb_m"]
        half_extent = [
            (bounds["max"][axis] - bounds["min"][axis]) / 2.0
            for axis in range(3)
        ]
        axis = str(joint.GetAttribute("physics:axis").Get())
        if axis != "X":
            raise ValueError(f"unsupported wheel joint axis {axis}: {joint_path}")
        wheels.append(
            {
                "link_prim_path": link_path,
                "joint_prim_path": joint_path,
                "centre_base_m": _vector(centre_base),
                "joint_axis": axis,
                "radius_geometry_estimate_m": max(half_extent[0], half_extent[2]),
                "joint_max_velocity_attribute": _number(
                    joint.GetAttribute("physxJoint:maxJointVelocity").Get()
                ),
                "drive_target_velocity_attribute": _number(
                    joint.GetAttribute(
                        "drive:angular:physics:targetVelocity"
                    ).Get()
                ),
            }
        )
    wheel_base = abs(wheels[0]["centre_base_m"][1] - wheels[1]["centre_base_m"][1])
    wheel_radius = sum(row["radius_geometry_estimate_m"] for row in wheels) / 2.0
    footprint_points = []
    for row in collisions:
        bounds = row["base_aabb_m"]
        footprint_points.extend(
            [
                [x, y]
                for x in (bounds["min"][0], bounds["max"][0])
                for y in (bounds["min"][1], bounds["max"][1])
            ]
        )
    return {
        "schema_version": 1,
        "extractor": "experiments.calibration.usd_extract",
        "usd_path": str(usd_path),
        "usd_sha256": hashlib.sha256(usd_path.read_bytes()).hexdigest(),
        "stage_meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "default_prim": str(stage.GetDefaultPrim().GetPath()),
        "base_prim_path": base_prim_path,
        "base_local_to_world": _matrix(base_world),
        "collision_prims": collisions,
        "footprint_aabb_corner_evidence_xy_m": footprint_points,
        "wheels": wheels,
        "wheel_radius_geometry_estimate_m": wheel_radius,
        "wheel_base_geometry_estimate_m": wheel_base,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read Dingo USD evidence without starting Isaac")
    parser.add_argument("--usd", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    evidence = extract_usd_evidence(args.usd)
    text = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
