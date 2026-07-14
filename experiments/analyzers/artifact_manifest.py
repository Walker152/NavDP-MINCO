from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


FIELDS = ["artifact_type", "experiment", "scene", "variant", "episode_uid", "plan_uid", "path", "sha256", "size", "data_source", "description"]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def generate_artifact_manifest(suite_dir):
    suite_dir = Path(suite_dir); rows = []
    excluded = {"artifact_manifest.json", "artifact_manifest.csv"}
    for path in sorted(item for item in suite_dir.rglob("*") if item.is_file() and item.name not in excluded and ".tmp" not in item.name):
        relative = path.relative_to(suite_dir); parts = relative.parts
        experiment = parts[1] if len(parts) > 1 and parts[0] == "experiments" else (parts[1] if len(parts) > 1 and parts[0] == "reports" and parts[1].startswith("EXP-") else "")
        scene = parts[2] if len(parts) > 2 and parts[0] == "experiments" else ""; variant = parts[3] if len(parts) > 3 and parts[0] == "experiments" else ""
        suffix = path.suffix.lower(); kind = {".csv":"table", ".npz":"trace", ".png":"plot", ".svg":"plot", ".md":"report", ".mp4":"video", ".json":"metadata"}.get(suffix, "file")
        rows.append({"artifact_type":kind, "experiment":experiment, "scene":scene, "variant":variant, "episode_uid":"", "plan_uid":"", "path":str(relative), "sha256":_sha256(path), "size":path.stat().st_size, "data_source":"SIMULATED", "description":path.name})
    reports = suite_dir / "reports"; reports.mkdir(parents=True, exist_ok=True)
    (reports / "artifact_manifest.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (reports / "artifact_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    return rows


def validate_artifact_manifest(suite_dir):
    suite_dir = Path(suite_dir); path = suite_dir / "reports" / "artifact_manifest.json"
    if not path.exists(): return ["missing artifact_manifest.json"]
    errors = []
    for row in json.loads(path.read_text(encoding="utf-8")):
        artifact = suite_dir / row["path"]
        if not artifact.exists(): errors.append(f"missing: {row['path']}")
        elif artifact.stat().st_size != row["size"]: errors.append(f"size mismatch: {row['path']}")
        elif _sha256(artifact) != row["sha256"]: errors.append(f"hash mismatch: {row['path']}")
    return errors
