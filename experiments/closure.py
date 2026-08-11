from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from experiments.analyzers.readonly import analyze_suite_readonly
from experiments.core.failure_taxonomy import classify_reason, taxonomy_payload
from experiments.dynamic_pilot import prepare_dynamic_pilot
from experiments.orchestrators.suite_runner import run_suite
from experiments.static.benchmark import (
    run_static_benchmark,
    validate_static_benchmark,
)
from experiments.static.selection import run_boundary_selection


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _profile_rows(path: Path, profile: str) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    return [{**row, "constraint_profile": profile} for row in rows]


def _validate_selection(path: Path) -> list[str]:
    errors = []
    try:
        selected = json.loads(path.read_text(encoding="utf-8"))
        policy = json.loads(
            (path.parent / "selection_policy.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        return [f"selection unreadable: {error}"]
    if selected.get("policy_sha256") != policy.get("policy_sha256"):
        errors.append("selection policy hash mismatch")
    main = list(selected.get("best2", [])) + list(selected.get("worst2", []))
    if len(main) != 4 or len(set(main)) != 4:
        errors.append("selection does not contain four unique main cases")
    if len(selected.get("best_backups", [])) < 2:
        errors.append("selection has fewer than two Best backups")
    if len(selected.get("worst_backups", [])) < 2:
        errors.append("selection has fewer than two Worst backups")
    for uid in main:
        payload = selected.get("cases", {}).get(uid, {})
        if payload.get("case_hash") != payload.get(
            "safe_corridor_v1_static", {}
        ).get("case_hash"):
            errors.append(f"selection case hash mismatch: {uid}")
        for relative in payload.get("artifact_paths", []):
            if not (path.parent / relative).is_file():
                errors.append(f"missing selected artifact: {uid}: {relative}")
    return errors


def _validate_dynamic(path: Path) -> list[str]:
    errors = []
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        dry_plan = json.loads(Path(receipt["dry_run_plan"]).read_text())
    except (OSError, KeyError, json.JSONDecodeError) as error:
        return [f"dynamic readiness unreadable: {error}"]
    if receipt.get("status") != "READY_FOR_REAL_RUN":
        errors.append("dynamic readiness status is not READY_FOR_REAL_RUN")
    if receipt.get("run_count") != 8:
        errors.append("dynamic readiness run count is not eight")
    if receipt.get("started_processes") != 0:
        errors.append("dynamic readiness started a process")
    if dry_plan.get("started_processes") != 0:
        errors.append("dynamic dry-run started a process")
    if _sha256(Path(receipt["dry_run_plan"])) != receipt.get(
        "dry_run_plan_sha256"
    ):
        errors.append("dynamic dry-run plan hash mismatch")
    return errors


def run_codex_closure(
    *,
    repo_root: Path | str,
    output_dir: Path | str,
    static_only: bool = False,
    select_cases: bool = False,
    dynamic_dry_run: bool = False,
    allow_real_simulation: bool = False,
    analysis_only_readonly: bool = False,
    analysis_suite: Path | str | None = None,
    resume: bool = False,
    retry_failed: bool = False,
    skip_video: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"closure output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_root = repo_root / "results/navdp_minco_longterm_20260726"
    legacy_dir = result_root / "static_baseline"
    safe_dir = result_root / "static_safe_corridor_v1"
    selection_dir = result_root / "static_boundary_selection_v1"
    dynamic_dir = result_root / "dynamic_pilot_readiness"
    legacy_config = repo_root / "experiments/configs/static_legacy_suite.json"
    safe_config = repo_root / "experiments/configs/static_safe_corridor_suite.json"
    selection_config = (
        repo_root / "experiments/configs/static_boundary_selection_v1.json"
    )
    calibration = repo_root / "configs/robots/dingo_calibration_v1.json"

    actions = []
    if static_only and not safe_dir.exists():
        run_static_benchmark(safe_config, safe_dir, trace_limit=0)
        actions.append("static_safe_corridor_generated")
    if select_cases and not selection_dir.exists():
        run_boundary_selection(selection_config, selection_dir)
        actions.append("selection_generated")
    if dynamic_dry_run and not dynamic_dir.exists():
        prepare_dynamic_pilot(
            selection_dir / "selected_dynamic_cases.json",
            calibration,
            legacy_config,
            safe_config,
            dynamic_dir,
            repo_root=repo_root,
        )
        actions.append("dynamic_readiness_generated")
    if analysis_only_readonly:
        if analysis_suite is None:
            raise ValueError("--analysis-only-readonly requires --analysis-suite")
        analyze_suite_readonly(
            analysis_suite,
            output_dir / "readonly_analysis",
            resume=resume,
        )
        actions.append("readonly_analysis_generated")
    if allow_real_simulation:
        suite_path = dynamic_dir / "dynamic_suite.json"
        run_suite(
            suite_path,
            backend_name="isaac",
            resume=resume,
            retry_failed=retry_failed,
            allow_real_simulation=True,
            skip_video=skip_video,
        )
        actions.append("authorized_real_dynamic_pilot_executed")

    errors = []
    errors.extend(validate_static_benchmark(legacy_dir))
    errors.extend(validate_static_benchmark(safe_dir))
    errors.extend(
        _validate_selection(selection_dir / "selected_dynamic_cases.json")
    )
    errors.extend(
        _validate_dynamic(dynamic_dir / "dynamic_readiness_receipt.json")
    )

    tables = output_dir / "static_tables"
    legacy_metrics = _profile_rows(
        legacy_dir / "legacy_case_metrics.csv", "legacy"
    )
    safe_metrics = _profile_rows(
        safe_dir / "legacy_case_metrics.csv", "safe_corridor_v1"
    )
    static_metrics = legacy_metrics + safe_metrics
    _write_csv(tables / "static_metrics.csv", static_metrics)
    _write_csv(
        tables / "static_cases.csv",
        [
            {
                "case_uid": row.get("case_uid", ""),
                "case_hash": row.get("case_hash", ""),
                "case_source": row.get("case_source", ""),
                "expected_category": row.get("expected_category", ""),
                "constraint_profile": row.get("constraint_profile", ""),
            }
            for row in static_metrics
        ],
    )
    selection_runs = _read_csv(selection_dir / "static_runs.csv")
    _write_csv(tables / "static_runs.csv", selection_runs)
    _write_csv(
        tables / "static_events.csv",
        [
            {
                "case_uid": row.get("case_uid", ""),
                "constraint_profile": row.get("profile", ""),
                "classification": row.get("classification", ""),
                "failure_reason": row.get("failure_reason", ""),
                **classify_reason(row.get("failure_reason", "")),
            }
            for row in selection_runs
            if row.get("failure_reason") not in {"", "NONE"}
        ],
    )
    shutil.copyfile(
        selection_dir / "case_selection.csv",
        tables / "case_selection.csv",
    )
    taxonomy = taxonomy_payload()
    (output_dir / "failure_taxonomy.json").write_text(
        json.dumps(taxonomy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifact_rows = []
    for task_root in (
        result_root / "comparisons",
        legacy_dir,
        safe_dir,
        selection_dir,
        dynamic_dir,
        result_root / "mock_closure",
        result_root / "mock_closure_readonly",
        repo_root / "reports/codex_longterm",
    ):
        if not task_root.exists():
            continue
        for path in sorted(task_root.rglob("*")):
            if path.is_file():
                artifact_rows.append(
                    {
                        "task_root": str(task_root.relative_to(repo_root)),
                        "artifact_path": str(path.relative_to(repo_root)),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    _write_csv(output_dir / "artifact_index.csv", artifact_rows)
    status = "READY_FOR_REAL_RUN" if not errors else "VALIDATION_FAILED"
    summary = {
        "schema_version": 1,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "actions": actions,
        "validation_errors": errors,
        "static_metric_rows": len(static_metrics),
        "boundary_run_rows": len(selection_runs),
        "artifact_count": len(artifact_rows),
        "dynamic_real_run_authorized": allow_real_simulation,
        "dynamic_real_run_completed": "authorized_real_dynamic_pilot_executed"
        in actions,
        "video_evidence_reduced": skip_video,
        "taxonomy_sha256": taxonomy["taxonomy_sha256"],
    }
    (output_dir / "closure_receipt.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "closure_report.md").write_text(
        "# NavDP–MINCO one-click closure\n\n"
        f"Status: **{status}**\n\n"
        f"- Static metric rows: {len(static_metrics)}\n"
        f"- Paired boundary runs: {len(selection_runs)}\n"
        f"- Indexed artifacts: {len(artifact_rows)}\n"
        f"- Validation errors: {len(errors)}\n"
        f"- Real dynamic simulation completed: "
        f"{summary['dynamic_real_run_completed']}\n\n"
        "The Task06 pilot remains readiness-only unless explicit real permission "
        "was supplied. Static grid cases are a controlled capability scan and "
        "are not population-level evidence.\n",
        encoding="utf-8",
    )
    return summary
