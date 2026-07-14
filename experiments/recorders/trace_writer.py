from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from experiments.core.trace_schema import build_metadata


class PlanningTraceWriter:
    def __init__(self, trace_dir: Path):
        self.trace_dir = Path(trace_dir); self.trace_dir.mkdir(parents=True, exist_ok=True)

    def write(self, planning_cycle_uid: str, arrays: dict):
        normalized = {name: np.asarray(value) for name, value in arrays.items() if value is not None}
        npz_path = self.trace_dir / f"planning_trace_{planning_cycle_uid}.npz"
        metadata_path = self.trace_dir / f"planning_trace_{planning_cycle_uid}.metadata.json"
        temp_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
        temp_meta = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        with temp_npz.open("wb") as stream: np.savez_compressed(stream, **normalized)
        temp_meta.write_text(json.dumps(build_metadata(normalized), indent=2) + "\n", encoding="utf-8")
        temp_npz.replace(npz_path); temp_meta.replace(metadata_path)
        return npz_path, metadata_path
