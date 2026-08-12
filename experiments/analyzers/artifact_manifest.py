from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from experiments.core.artifact_receipt import (
    sha256_file,
    validate_file_receipt,
)


FIELDS = ["artifact_type", "experiment", "scene", "variant", "episode_uid", "plan_uid", "path", "sha256", "size", "data_source", "description"]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def generate_artifact_manifest(suite_dir):
    suite_dir = Path(suite_dir); rows = []
    config_path = suite_dir / "suite_config.json"
    suite_source = json.loads(config_path.read_text()).get("data_source", "UNKNOWN") if config_path.exists() else "UNKNOWN"
    excluded = {"artifact_manifest.json", "artifact_manifest.csv"}
    for path in sorted(item for item in suite_dir.rglob("*") if item.is_file() and item.name not in excluded and ".tmp" not in item.name):
        relative = path.relative_to(suite_dir); parts = relative.parts
        experiment = parts[1] if len(parts) > 1 and parts[0] == "experiments" else (parts[1] if len(parts) > 1 and parts[0] == "reports" and parts[1].startswith("EXP-") else "")
        scene = parts[2] if len(parts) > 2 and parts[0] == "experiments" else ""; variant = parts[3] if len(parts) > 3 and parts[0] == "experiments" else ""
        suffix = path.suffix.lower(); kind = {".csv":"table", ".npz":"trace", ".png":"plot", ".svg":"plot", ".md":"report", ".mp4":"video", ".json":"metadata"}.get(suffix, "file")
        episode_uid = path.stem if suffix == ".mp4" and parts and "videos" in parts else ""
        plan_uid = path.stem.removeprefix("planning_trace_") if suffix == ".npz" and path.stem.startswith("planning_trace_") else ""
        rows.append({"artifact_type":kind, "experiment":experiment, "scene":scene, "variant":variant, "episode_uid":episode_uid, "plan_uid":plan_uid, "path":str(relative), "sha256":_sha256(path), "size":path.stat().st_size, "data_source":suite_source, "description":path.name})
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


def validate_paper_artifact_manifest(paper_dir):
    """Validate figure groups, external inputs, and the paper inventory."""

    paper_dir = Path(paper_dir).resolve()
    errors = []
    bundles = []
    receipt_dirs = sorted(
        {path.parent for path in paper_dir.rglob("receipts/*.json")}
    )
    for receipt_dir in receipt_dirs:
        bundle_root = receipt_dir.parent
        figure_receipts = []
        for path in sorted(receipt_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                errors.append(
                    f"invalid paper receipt {path.relative_to(paper_dir)}: {error}"
                )
                continue
            if payload.get("figure_stem"):
                figure_receipts.append((path, payload))
        bundles.append((bundle_root, figure_receipts))

    required_caption_fields = (
        "Source:",
        "Units:",
        "n=",
        "Paired key:",
        "Denominator:",
        "Missing/failed:",
        "Limitations:",
        "Interpretation:",
    )
    for bundle_root, figure_receipts in bundles:
        bundle_label = bundle_root.relative_to(paper_dir).as_posix() or "."
        figure_dir = bundle_root / "figures"
        expected_stems = (
            {path.stem for path in figure_dir.glob("*.png")}
            if figure_dir.is_dir()
            else set()
        )
        receipt_stems = {
            str(payload["figure_stem"]) for _, payload in figure_receipts
        }
        for stem in sorted(expected_stems - receipt_stems):
            errors.append(f"missing figure receipt {bundle_label}/{stem}")
        for stem in sorted(receipt_stems - expected_stems):
            errors.append(f"missing PNG for figure receipt {bundle_label}/{stem}")
        for _, payload in figure_receipts:
            stem = str(payload["figure_stem"])
            expected_outputs = {
                f"figures/{stem}.png",
                f"figures/{stem}.pdf",
                f"tables/{stem}.csv",
                f"captions/{stem}.md",
            }
            output_receipts = payload.get("outputs", [])
            recorded_outputs = {
                str(receipt.get("path", "")) for receipt in output_receipts
            }
            for relative in sorted(expected_outputs - recorded_outputs):
                errors.append(
                    f"unreceipted figure output {bundle_label}/{relative}"
                )
            for receipt in output_receipts:
                errors.extend(validate_file_receipt(bundle_root, receipt))
            caption_path = bundle_root / "captions" / f"{stem}.md"
            if caption_path.is_file():
                caption = caption_path.read_text(encoding="utf-8")
                for field in required_caption_fields:
                    if field not in caption:
                        errors.append(
                            f"caption {bundle_label}/{stem} missing {field}"
                        )
            inputs = payload.get("inputs", [])
            if not inputs:
                errors.append(f"figure {bundle_label}/{stem} has no input receipts")
            for receipt in inputs:
                path = Path(str(receipt.get("path", "")))
                if not path.is_absolute() or not path.is_file():
                    errors.append(f"missing figure input {path}")
                    continue
                if path.stat().st_size != receipt.get("size_bytes"):
                    errors.append(f"input size mismatch {path}")
                if sha256_file(path) != receipt.get("sha256"):
                    errors.append(f"input hash mismatch {path}")

    inventory_path = paper_dir / "artifact_receipt.json"
    if not inventory_path.is_file():
        errors.append("missing paper artifact_receipt.json")
        return errors
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        errors.append(f"invalid paper artifact_receipt.json: {error}")
        return errors
    artifact_receipts = inventory.get("artifacts", [])
    recorded = {str(receipt.get("path", "")) for receipt in artifact_receipts}
    actual = {
        path.relative_to(paper_dir).as_posix()
        for path in paper_dir.rglob("*")
        if path.is_file() and path != inventory_path
    }
    for relative in sorted(actual - recorded):
        errors.append(f"unreceipted paper artifact {relative}")
    for relative in sorted(recorded - actual):
        errors.append(f"missing paper artifact {relative}")
    for receipt in artifact_receipts:
        errors.extend(validate_file_receipt(paper_dir, receipt))
    return errors
