from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from experiments.analyzers.metrics import (
    compute_geometric_metrics,
    compute_minco_temporal_profile,
)
from experiments.core.trace_schema import TRACE_FIELDS
from experiments.visualizers.planning_trace import (
    render_clearance_figure,
    render_dynamics_figure,
    render_trajectory_overview,
)


@dataclass(frozen=True)
class TraceEvidence:
    path: Path
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, object]
    available_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]


@dataclass(frozen=True)
class TraceRenderReceipt:
    case_uid: str
    variant: str
    artifact_paths: tuple[Path, ...]
    missing_evidence: tuple[str, ...]


def _load_metadata(trace_path: Path) -> dict[str, object]:
    metadata_path = trace_path.with_name(
        f"{trace_path.stem}.metadata.json"
    )
    if not metadata_path.is_file():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable trace metadata {metadata_path}: {error}") from error


def load_trace_evidence(trace_path: Path | str) -> TraceEvidence:
    trace_path = Path(trace_path).resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"trace does not exist: {trace_path}")
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(trace_path, allow_pickle=False) as archive:
            for field in archive.files:
                value = archive[field]
                if value.dtype.hasobject:
                    raise ValueError(f"object array is forbidden: {field}")
                arrays[field] = np.asarray(value)
    except ValueError as error:
        message = str(error)
        if "Object arrays cannot be loaded" in message:
            message = f"object/pickle trace arrays are forbidden: {message}"
        raise ValueError(f"unsafe or invalid trace {trace_path}: {message}") from error

    metadata = _load_metadata(trace_path)
    metadata_arrays = metadata.get("arrays", {})
    if isinstance(metadata_arrays, dict):
        for field, specification in metadata_arrays.items():
            if field not in arrays or not isinstance(specification, dict):
                continue
            expected_shape = specification.get("shape")
            expected_dtype = specification.get("dtype")
            if expected_shape is not None and list(arrays[field].shape) != expected_shape:
                raise ValueError(f"trace shape mismatch: {field}")
            if expected_dtype is not None and str(arrays[field].dtype) != expected_dtype:
                raise ValueError(f"trace dtype mismatch: {field}")

    available = tuple(sorted(arrays))
    missing = tuple(sorted(set(TRACE_FIELDS) - set(arrays)))
    return TraceEvidence(
        path=trace_path,
        arrays=MappingProxyType(arrays),
        metadata=MappingProxyType(metadata),
        available_fields=available,
        missing_fields=missing,
    )


def _json_finite(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def render_trace_evidence(
    evidence: TraceEvidence,
    output_dir: Path | str,
    *,
    case_uid: str,
    variant: str,
    data_source: str,
) -> TraceRenderReceipt:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / f"{case_uid}_trajectory.png"
    clearance_path = output_dir / f"{case_uid}_clearance.png"
    dynamics_path = output_dir / f"{case_uid}_dynamics.png"
    metrics_path = output_dir / f"{case_uid}_metrics.json"
    title_prefix = f"{case_uid} · {variant}"

    render_trajectory_overview(
        evidence.arrays,
        overview_path,
        title=f"{title_prefix} trajectory comparison",
        data_source=data_source,
    )
    clearance_available = render_clearance_figure(
        evidence.arrays,
        clearance_path,
        title=f"{title_prefix} clearance vs arc length",
        data_source=data_source,
    )
    samples = evidence.arrays.get("minco_samples")
    render_dynamics_figure(
        samples,
        dynamics_path,
        title=f"{title_prefix} analytic dynamics",
        data_source=data_source,
    )

    raw_geometry, _ = compute_geometric_metrics(
        evidence.arrays.get("raw_path_xy", np.empty((0, 2)))
    )
    minco_path = (
        samples[:, 1:3]
        if samples is not None
        and samples.ndim == 2
        and samples.shape[1] >= 3
        else np.empty((0, 2))
    )
    minco_geometry, _ = compute_geometric_metrics(minco_path)
    temporal, _ = compute_minco_temporal_profile(samples)
    metrics = {
        "schema_version": 1,
        "case_uid": case_uid,
        "variant": variant,
        "data_source": data_source,
        "trace_path": str(evidence.path),
        "raw_geometry": raw_geometry,
        "minco_geometry": minco_geometry,
        "temporal": temporal,
        "clearance": {
            "status": "AVAILABLE" if clearance_available else "MISSING",
            "source": "RECORDED_TRACE" if clearance_available else None,
        },
        "available_fields": list(evidence.available_fields),
        "missing_evidence": list(evidence.missing_fields),
    }
    metrics_path.write_text(
        json.dumps(_json_finite(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    artifacts = (overview_path, clearance_path, dynamics_path, metrics_path)
    return TraceRenderReceipt(
        case_uid=case_uid,
        variant=variant,
        artifact_paths=artifacts,
        missing_evidence=evidence.missing_fields,
    )
