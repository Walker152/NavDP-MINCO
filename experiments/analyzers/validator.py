from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.core.schemas import SCHEMAS
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
    result = {"valid": not errors, "errors": errors, "row_counts": counts}
    if write_report:
        atomic_json(run_dir / "validation_report.json", result)
        (run_dir / "validation_report.md").write_text("# Run Validation\n\n" + ("PASS" if result["valid"] else "FAIL") + "\n\n" + "\n".join(f"- {item}" for item in errors) + "\n", encoding="utf-8")
    return result
