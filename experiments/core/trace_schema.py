from __future__ import annotations

import json
from pathlib import Path
import numpy as np


TRACE_SCHEMA_VERSION = 1
TRACE_FIELDS = {
    "raw_path_xy": "m", "topk_candidates_xy": "m", "critic_values": "unitless",
    "selected_candidate_xy": "m", "minco_samples": "mixed:t,m,m/s,m/s2,m/s3,rad,rad/s",
    "previous_committed_samples": "mixed", "robot_state": "mixed", "goal": "m",
    "esdf_distance": "m", "esdf_origin": "m", "esdf_resolution": "m",
    "sampled_s": "m", "raw_clearance": "m", "minco_clearance": "m",
    "raw_curvature": "1/m", "minco_curvature": "1/m",
}


def build_metadata(arrays: dict[str, np.ndarray]) -> dict:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "unit": TRACE_FIELDS.get(name, "unknown")} for name, value in arrays.items()},
        "missing_fields": sorted(set(TRACE_FIELDS) - set(arrays)),
    }


def validate_trace(npz_path: Path, metadata_path: Path) -> list[str]:
    errors = []
    try:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if metadata.get("schema_version") != TRACE_SCHEMA_VERSION: errors.append("schema_version mismatch")
        with np.load(npz_path, allow_pickle=False) as data:
            for name, spec in metadata.get("arrays", {}).items():
                if name not in data: errors.append(f"missing array: {name}"); continue
                if list(data[name].shape) != spec.get("shape"): errors.append(f"shape mismatch: {name}")
                if str(data[name].dtype) != spec.get("dtype"): errors.append(f"dtype mismatch: {name}")
    except Exception as error:
        errors.append(f"unreadable trace: {error}")
    return errors
