"""Capability sweep orchestrator — runs unified controlled-variable experiments."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.static.case_schema import StaticCase, save_static_case
from experiments.static.metrics import compute_static_case_metrics
from experiments.static.runner import (
    StaticRunResult,
    load_legacy_profile,
    run_static_case,
)
from experiments.static.synthetic import generate_catalogue, generate_obstacle_variants
from experiments.visualizers.static_benchmark import render_static_case


def _assemble_profile_config(
    sweep_config: Mapping[str, Any], profile_name: str
) -> dict[str, Any]:
    """Merge shared blocks with per-profile overrides to produce a flat minco config."""
    shared = {
        **sweep_config.get("dynamic_limits", {}),
        **sweep_config.get("minco", {}),
        **sweep_config.get("validation", {}),
        **sweep_config.get("metric_limits", {}),
    }
    profile = dict(sweep_config.get("profiles", {}).get(profile_name, {}))
    return {**shared, **profile}


def _assemble_metric_limits(sweep_config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(sweep_config.get("metric_limits", {}))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def run_capability_sweep(
    config_path: Path | str, output_dir: Path | str
) -> dict[str, Any]:
    """Run a full controlled-variable capability sweep.

    Loads the unified sweep config, expands cases (base + obstacle variants +
    state variants + factor grids), runs each for both legacy and
    safe_corridor_v1 profiles with shared validation, and produces
    per-case artifacts + sweep-level CSV + manifest.
    """
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"sweep output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert sweep_config.get("schema_version") == 1

    # Expand cases
    base_config = sweep_config.get("base_case_config",
                                   "experiments/configs/static_legacy_suite.json")
    base_config_path = Path(base_config)
    if not base_config_path.is_absolute():
        base_config_path = (config_path.parent / base_config_path).resolve()
    base_cases = generate_catalogue(str(base_config_path))
    case_list = list(base_cases)

    # Obstacle density variants — merge grid and cases from base config
    base_config_raw = json.loads(base_config_path.read_text(encoding="utf-8"))
    sweep_with_grid = dict(sweep_config)
    for key in ("grid", "cases", "constraint_profile"):
        if key not in sweep_with_grid and key in base_config_raw:
            sweep_with_grid[key] = base_config_raw[key]
    if sweep_with_grid.get("obstacle_variants"):
        case_list.extend(generate_obstacle_variants(sweep_with_grid))

    # State variants + factor grids (reuse boundary expansion)
    from experiments.static.selection import generate_boundary_cases
    try:
        boundary_cases = generate_boundary_cases(sweep_with_grid)
        existing = {c.case_uid for c in case_list}
        for c in boundary_cases:
            if c.case_uid not in existing:
                case_list.append(c)
                existing.add(c.case_uid)
    except Exception:
        pass

    profiles = ["legacy", "safe_corridor_v1"]
    metric_limits = _assemble_metric_limits(sweep_config)

    rows: list[dict[str, Any]] = []
    # Deduplicate by case_uid (boundary expansion may overlap with obstacle variants)
    seen_uids: set[str] = set()
    unique_cases: list[StaticCase] = []
    for case in case_list:
        if case.case_uid not in seen_uids:
            seen_uids.add(case.case_uid)
            unique_cases.append(case)
    case_list = unique_cases

    for case in case_list:
        for profile_name in profiles:
            profile_config = _assemble_profile_config(sweep_config, profile_name)
            # Force constraint_profile from assembly
            profile_config["constraint_profile"] = sweep_config.get(
                "profiles", {}
            ).get(profile_name, {}).get("constraint_profile", profile_name)

            result = run_static_case(case, profile_config, "recompute")
            metrics, detail = compute_static_case_metrics(
                case, result, metric_limits
            )
            metrics["safe_dist_m"] = float(metric_limits.get("safe_distance_m", 0.279))

            artifact_dir = output_dir / "artifacts" / case.case_uid / profile_name
            render_static_case(case, result, metrics, detail, artifact_dir)

            row = _json_safe({
                "case_uid": case.case_uid,
                "case_source": case.case_source,
                "expected_category": case.expected_category,
                "profile": profile_name,
                "status": result.status,
                "failure_reason": result.diagnostics.get("failure_reason", ""),
                "corridor_segment_count": result.diagnostics.get("corridor_segment_count", 0),
                "corridor_failure_reason": result.diagnostics.get("corridor_failure_reason", ""),
                "validation_failure_reason": result.diagnostics.get("validation_failure_reason", ""),
                **metrics,
            })
            rows.append(row)

    # Write sweep CSV
    fields = sorted({k for row in rows for k in row})
    csv_path = output_dir / "sweep_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Manifest
    manifest = {
        "schema_version": 1,
        "suite_id": sweep_config.get("suite_id", ""),
        "config_path": str(config_path),
        "case_count": len(case_list),
        "profile_count": len(profiles),
        "total_runs": len(rows),
        "calibration_status": sweep_config.get("calibration_status", {}),
    }
    (output_dir / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return manifest
