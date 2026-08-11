from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np


STATIC_CASE_SCHEMA_VERSION = 1
CASE_SOURCES = frozenset({"STATIC_SYNTHETIC", "STATIC_REPLAY_REAL_TRACE"})
CONSTRAINT_PROFILES = frozenset({"legacy", "safe_corridor_v1"})
STATE_FIELDS = (
    "position",
    "velocity",
    "acceleration",
    "yaw",
    "yaw_rate",
)


def _readonly_array(
    value: object,
    *,
    dtype: np.dtype | type,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype.hasobject:
        raise ValueError(f"{name} object arrays are forbidden")
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


def _array_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _canonical_metadata(case: "StaticCase") -> dict[str, object]:
    return {
        "schema_version": STATIC_CASE_SCHEMA_VERSION,
        "case_uid": case.case_uid,
        "case_source": case.case_source,
        "esdf_resolution": case.esdf_resolution,
        "start_yaw": case.start_yaw,
        "start_yaw_rate": case.start_yaw_rate,
        "has_terminal_goal": case.terminal_goal is not None,
        "constraint_profile": case.constraint_profile,
        "expected_category": case.expected_category,
        "esdf_available": case.esdf_available,
        "tags": list(case.tags),
        "frame": case.frame,
        "units": dict(case.units),
        "state_availability": dict(case.state_availability),
        "references": dict(case.references),
    }


def _case_arrays(case: "StaticCase") -> dict[str, np.ndarray]:
    arrays = {
        "guide_path_xyz": case.guide_path_xyz,
        "occupancy": case.occupancy,
        "esdf_distance": case.esdf_distance,
        "esdf_origin": case.esdf_origin,
        "start_position": case.start_position,
        "start_velocity": case.start_velocity,
        "start_acceleration": case.start_acceleration,
    }
    if case.terminal_goal is not None:
        arrays["terminal_goal"] = case.terminal_goal
    for name, value in sorted(case.auxiliary_arrays.items()):
        arrays[f"aux__{name}"] = value
    return arrays


def _compute_case_hash(case: "StaticCase") -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            _canonical_metadata(case),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name, array in sorted(_case_arrays(case).items()):
        digest.update(name.encode("utf-8"))
        digest.update(_array_digest(array).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class StaticCase:
    case_uid: str
    case_source: str
    guide_path_xyz: np.ndarray
    occupancy: np.ndarray
    esdf_distance: np.ndarray
    esdf_origin: np.ndarray
    esdf_resolution: float
    start_position: np.ndarray
    start_velocity: np.ndarray
    start_acceleration: np.ndarray
    start_yaw: float
    start_yaw_rate: float
    terminal_goal: np.ndarray | None
    constraint_profile: str
    expected_category: str
    esdf_available: bool = True
    tags: tuple[str, ...] = ()
    frame: str = "world"
    units: Mapping[str, str] = field(
        default_factory=lambda: {"position": "m", "time": "s", "yaw": "rad"}
    )
    state_availability: Mapping[str, bool] = field(
        default_factory=lambda: {name: True for name in STATE_FIELDS}
    )
    references: Mapping[str, str] = field(default_factory=dict)
    auxiliary_arrays: Mapping[str, np.ndarray] = field(default_factory=dict)
    case_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.case_uid):
            raise ValueError("case_uid must contain only safe filename characters")
        if self.case_source not in CASE_SOURCES:
            raise ValueError(f"unsupported case_source: {self.case_source}")
        if self.constraint_profile not in CONSTRAINT_PROFILES:
            raise ValueError(
                f"unsupported constraint_profile: {self.constraint_profile}"
            )
        if not self.frame.strip():
            raise ValueError("frame must be nonempty")
        if not self.expected_category.strip():
            raise ValueError("expected_category must be nonempty")

        guide = _readonly_array(
            self.guide_path_xyz, dtype=np.float64, name="guide_path_xyz"
        )
        if guide.ndim != 2 or guide.shape[0] < 2 or guide.shape[1] != 3:
            raise ValueError("guide_path_xyz must have shape (N>=2, 3)")
        occupancy = _readonly_array(
            self.occupancy, dtype=np.bool_, name="occupancy"
        )
        distance = _readonly_array(
            self.esdf_distance, dtype=np.float64, name="esdf_distance"
        )
        if occupancy.ndim != 2 or occupancy.size == 0:
            raise ValueError("occupancy must be a nonempty 2D array")
        if distance.shape != occupancy.shape:
            raise ValueError("esdf_distance shape must match occupancy")
        origin = _readonly_array(
            self.esdf_origin,
            dtype=np.float64,
            name="esdf_origin",
            shape=(2,),
        )
        resolution = float(self.esdf_resolution)
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("esdf_resolution must be finite and positive")

        position = _readonly_array(
            self.start_position,
            dtype=np.float64,
            name="start_position",
            shape=(3,),
        )
        velocity = _readonly_array(
            self.start_velocity,
            dtype=np.float64,
            name="start_velocity",
            shape=(3,),
        )
        acceleration = _readonly_array(
            self.start_acceleration,
            dtype=np.float64,
            name="start_acceleration",
            shape=(3,),
        )
        terminal = (
            None
            if self.terminal_goal is None
            else _readonly_array(
                self.terminal_goal,
                dtype=np.float64,
                name="terminal_goal",
                shape=(3,),
            )
        )
        yaw = float(self.start_yaw)
        yaw_rate = float(self.start_yaw_rate)
        if not math.isfinite(yaw) or not math.isfinite(yaw_rate):
            raise ValueError("start_yaw and start_yaw_rate must be finite")

        units = {str(key): str(value) for key, value in self.units.items()}
        if not units or any(not key or not value for key, value in units.items()):
            raise ValueError("units must contain nonempty string mappings")
        availability = {
            name: bool(self.state_availability.get(name, False))
            for name in STATE_FIELDS
        }
        references = {str(key): str(value) for key, value in self.references.items()}
        if any(not key or not value for key, value in references.items()):
            raise ValueError("references must contain nonempty string mappings")
        auxiliary = {}
        for name, value in self.auxiliary_arrays.items():
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(name)):
                raise ValueError(f"invalid auxiliary array name: {name}")
            auxiliary[str(name)] = _readonly_array(
                value, dtype=np.float64, name=f"auxiliary_arrays[{name}]"
            )

        object.__setattr__(self, "guide_path_xyz", guide)
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "esdf_distance", distance)
        object.__setattr__(self, "esdf_origin", origin)
        object.__setattr__(self, "esdf_resolution", resolution)
        object.__setattr__(self, "start_position", position)
        object.__setattr__(self, "start_velocity", velocity)
        object.__setattr__(self, "start_acceleration", acceleration)
        object.__setattr__(self, "terminal_goal", terminal)
        object.__setattr__(self, "start_yaw", yaw)
        object.__setattr__(self, "start_yaw_rate", yaw_rate)
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        object.__setattr__(self, "units", MappingProxyType(units))
        object.__setattr__(
            self, "state_availability", MappingProxyType(availability)
        )
        object.__setattr__(self, "esdf_available", bool(self.esdf_available))
        object.__setattr__(self, "references", MappingProxyType(references))
        object.__setattr__(
            self, "auxiliary_arrays", MappingProxyType(auxiliary)
        )
        object.__setattr__(self, "case_hash", _compute_case_hash(self))


@dataclass(frozen=True)
class StaticCaseReceipt:
    case_uid: str
    case_hash: str
    array_path: Path
    metadata_path: Path
    array_sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_static_case(
    case: StaticCase,
    output_dir: Path | str,
) -> StaticCaseReceipt:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    array_path = output_dir / f"{case.case_uid}.npz"
    metadata_path = output_dir / f"{case.case_uid}.case.json"
    if array_path.exists() or metadata_path.exists():
        if array_path.is_file() and metadata_path.is_file():
            existing = load_static_case(metadata_path)
            if existing.case_hash == case.case_hash:
                return StaticCaseReceipt(
                    case.case_uid,
                    case.case_hash,
                    array_path,
                    metadata_path,
                    _file_sha256(array_path),
                )
        raise FileExistsError(f"static case already exists: {case.case_uid}")

    arrays = _case_arrays(case)
    temp_array = array_path.with_suffix(".npz.tmp")
    with temp_array.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    array_sha256 = _file_sha256(temp_array)
    metadata = {
        **_canonical_metadata(case),
        "case_hash": case.case_hash,
        "array_file": array_path.name,
        "array_sha256": array_sha256,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_digest(value),
            }
            for name, value in sorted(arrays.items())
        },
    }
    temp_metadata = metadata_path.with_suffix(".json.tmp")
    temp_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_array.replace(array_path)
    temp_metadata.replace(metadata_path)
    return StaticCaseReceipt(
        case.case_uid,
        case.case_hash,
        array_path,
        metadata_path,
        array_sha256,
    )


def load_static_case(metadata_path: Path | str) -> StaticCase:
    metadata_path = Path(metadata_path).resolve()
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable static case metadata: {error}") from error
    if metadata.get("schema_version") != STATIC_CASE_SCHEMA_VERSION:
        raise ValueError("static case schema_version mismatch")
    array_path = metadata_path.parent / str(metadata.get("array_file", ""))
    if not array_path.is_file():
        raise ValueError(f"static case array file missing: {array_path}")
    if _file_sha256(array_path) != metadata.get("array_sha256"):
        raise ValueError("static case array file hash mismatch")

    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(array_path, allow_pickle=False) as archive:
            for name in archive.files:
                value = archive[name]
                if value.dtype.hasobject:
                    raise ValueError(f"object array is forbidden: {name}")
                arrays[name] = value
    except ValueError as error:
        raise ValueError(f"unsafe static case array file: {error}") from error
    for name, specification in metadata.get("arrays", {}).items():
        if name not in arrays:
            raise ValueError(f"static case array missing: {name}")
        if list(arrays[name].shape) != specification.get("shape"):
            raise ValueError(f"static case array shape mismatch: {name}")
        if str(arrays[name].dtype) != specification.get("dtype"):
            raise ValueError(f"static case array dtype mismatch: {name}")
        if _array_digest(arrays[name]) != specification.get("sha256"):
            raise ValueError(f"static case array hash mismatch: {name}")

    auxiliary = {
        name.removeprefix("aux__"): value
        for name, value in arrays.items()
        if name.startswith("aux__")
    }
    case = StaticCase(
        case_uid=str(metadata["case_uid"]),
        case_source=str(metadata["case_source"]),
        guide_path_xyz=arrays["guide_path_xyz"],
        occupancy=arrays["occupancy"],
        esdf_distance=arrays["esdf_distance"],
        esdf_origin=arrays["esdf_origin"],
        esdf_resolution=float(metadata["esdf_resolution"]),
        start_position=arrays["start_position"],
        start_velocity=arrays["start_velocity"],
        start_acceleration=arrays["start_acceleration"],
        start_yaw=float(metadata["start_yaw"]),
        start_yaw_rate=float(metadata["start_yaw_rate"]),
        terminal_goal=arrays.get("terminal_goal"),
        constraint_profile=str(metadata["constraint_profile"]),
        expected_category=str(metadata["expected_category"]),
        esdf_available=bool(metadata.get("esdf_available", True)),
        tags=tuple(metadata.get("tags", [])),
        frame=str(metadata["frame"]),
        units=metadata.get("units", {}),
        state_availability=metadata.get("state_availability", {}),
        references=metadata.get("references", {}),
        auxiliary_arrays=auxiliary,
    )
    if case.case_hash != metadata.get("case_hash"):
        raise ValueError("static case hash mismatch")
    return case
