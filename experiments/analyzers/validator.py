from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.core.schemas import SCHEMAS
from experiments.core.trace_schema import validate_trace
from experiments.recorders.run_recorder import atomic_json


def validate_run(run_dir: Path | str, write_report=True):
    run_dir = Path(run_dir); errors = [] ; counts = {}
    for table, schema in SCHEMAS.items():
        path = run_dir / f"{table}.csv"
        if not path.exists(): errors.append(f"missing {path.name}"); continue
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream); rows = list(reader); counts[table] = len(rows)
            if reader.fieldnames != schema: errors.append(f"schema mismatch: {table}")
            if any(row.get("data_source") not in {"SIMULATED", "REAL"} for row in rows): errors.append(f"invalid data_source: {table}")
    status_path = run_dir / "run_status.json"
    if not status_path.exists(): errors.append("missing run_status.json")
    for npz_path in sorted((run_dir / "traces").glob("*.npz")) if (run_dir / "traces").exists() else []:
        metadata_path = npz_path.with_name(npz_path.stem + ".metadata.json")
        if not metadata_path.exists(): errors.append(f"missing trace metadata: {npz_path.name}")
        else: errors.extend(validate_trace(npz_path, metadata_path))
    result = {"valid": not errors, "errors": errors, "row_counts": counts}
    if write_report:
        atomic_json(run_dir / "validation_report.json", result)
        (run_dir / "validation_report.md").write_text("# Run Validation\n\n" + ("PASS" if result["valid"] else "FAIL") + "\n\n" + "\n".join(f"- {item}" for item in errors) + "\n", encoding="utf-8")
        validation_dir = run_dir / "validation"; validation_dir.mkdir(exist_ok=True)
        atomic_json(validation_dir / "validation_report.json", result)
        (validation_dir / "validation_report.md").write_text("# Run Validation\n\n" + ("PASS" if result["valid"] else "FAIL") + "\n", encoding="utf-8")
    return result
