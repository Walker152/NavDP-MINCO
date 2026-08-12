from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

from experiments.analyzers.readonly import analyze_suite_readonly
from experiments.analyzers.artifact_manifest import (
    validate_artifact_manifest,
    validate_paper_artifact_manifest,
)
from experiments.analyzers.paper_report import generate_paper_report
from experiments.calibration.report import render_calibration_report
from experiments.core.artifact_receipt import (
    sha256_file,
    validate_file_receipt,
)
from experiments.dynamic_pilot import prepare_dynamic_pilot
from experiments.orchestrators.suite_runner import run_suite
from experiments.static.benchmark import (
    run_static_benchmark,
    validate_static_benchmark,
)
from experiments.static.selection import run_boundary_selection
from experiments.rolling.showcase import (
    run_rolling_showcase,
    validate_showcase,
)


@dataclass(frozen=True)
class WorkflowOptions:
    output_root: Path
    resume: bool = False
    retry_failed: bool = False
    allow_real_simulation: bool = False
    full_suite: bool = False
    skip_video: bool = False
    skip_rolling_showcase: bool = False
    rolling_showcase_config: Path | None = None
    legacy_config: Path | None = None
    safe_config: Path | None = None
    selection_config: Path | None = None
    calibration_path: Path | None = None
    calibration_protocol: Path | None = None
    robot_usd: Path | None = None
    full_suite_config: Path | None = None


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    started_at_utc: str
    ended_at_utc: str
    inputs: tuple[dict[str, object], ...]
    outputs: tuple[dict[str, object], ...]
    command: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _python_environment_entry(
    path_value: str | Path | None, *, environment: str, pending: bool = False
) -> dict[str, object]:
    path = Path(path_value).expanduser().resolve() if path_value else None
    available = bool(path and path.is_file() and os.access(path, os.X_OK))
    version = "UNAVAILABLE"
    if available:
        try:
            completed = subprocess.run(
                [str(path), "--version"],
                text=True,
                capture_output=True,
                timeout=10,
                check=True,
            )
            version = (completed.stdout or completed.stderr).strip()
        except (OSError, subprocess.SubprocessError):
            available = False
    return {
        "environment": environment,
        "python": str(path) if path else "",
        "available": available,
        "version": version,
        "status": (
            "AVAILABLE"
            if available
            else ("PENDING_REAL_SIMULATION" if pending else "UNAVAILABLE")
        ),
    }


def _environment_receipt() -> dict[str, object]:
    navdp = os.environ.get("NAVDP_PYTHON") or sys.executable
    isaac = os.environ.get("ISAACLAB_PYTHON")
    if not isaac:
        candidate = Path("/home/alioth/miniforge3/envs/isaaclab/bin/python")
        isaac = str(candidate) if candidate.is_file() else None
    return {
        "static_analysis": _python_environment_entry(
            navdp, environment="navdp"
        ),
        "real_simulation": _python_environment_entry(
            isaac, environment="isaaclab", pending=True
        ),
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_options(repo_root: Path, options: WorkflowOptions) -> WorkflowOptions:
    config_dir = repo_root / "experiments" / "configs"
    return WorkflowOptions(
        output_root=Path(options.output_root).resolve(),
        resume=options.resume,
        retry_failed=options.retry_failed,
        allow_real_simulation=options.allow_real_simulation,
        full_suite=options.full_suite,
        skip_video=options.skip_video,
        skip_rolling_showcase=options.skip_rolling_showcase,
        rolling_showcase_config=Path(
            options.rolling_showcase_config
            or config_dir / "rolling_showcase_v1.json"
        ).resolve(),
        legacy_config=Path(
            options.legacy_config or config_dir / "static_legacy_suite.json"
        ).resolve(),
        safe_config=Path(
            options.safe_config or config_dir / "static_safe_corridor_suite.json"
        ).resolve(),
        selection_config=Path(
            options.selection_config
            or config_dir / "static_boundary_selection_v1.json"
        ).resolve(),
        calibration_path=Path(
            options.calibration_path
            or repo_root / "configs" / "robots" / "dingo_calibration_v1.json"
        ).resolve(),
        calibration_protocol=Path(
            options.calibration_protocol
            or config_dir / "dingo_isolated_calibration_protocol.json"
        ).resolve(),
        robot_usd=Path(
            options.robot_usd or repo_root / "assets" / "robots" / "dingo.usd"
        ).resolve(),
        full_suite_config=Path(
            options.full_suite_config
            or repo_root / "configs" / "experiments" / "full_suite.json"
        ).resolve(),
    )


def _input_receipts(paths: Iterable[Path]) -> tuple[dict[str, object], ...]:
    unique = sorted({Path(path).resolve() for path in paths})
    receipts = []
    for path in unique:
        if not path.is_file():
            raise FileNotFoundError(f"workflow input does not exist: {path}")
        receipts.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return tuple(receipts)


def _output_receipts(
    paths: Iterable[Path], output_root: Path
) -> tuple[dict[str, object], ...]:
    files: set[Path] = set()
    for value in paths:
        path = Path(value).resolve()
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(
                item.resolve()
                for item in path.rglob("*")
                if item.is_file() and not item.is_symlink()
            )
        else:
            raise FileNotFoundError(f"stage output does not exist: {path}")
    rows = []
    for path in sorted(files):
        try:
            label = path.relative_to(output_root).as_posix()
        except ValueError:
            label = str(path)
        rows.append(
            {
                "path": label,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return tuple(rows)


def _receipt_output_path(row: Mapping[str, object], output_root: Path) -> Path:
    path = Path(str(row.get("path", "")))
    return path if path.is_absolute() else output_root / path


def _verify_stage_outputs(
    rows: Sequence[Mapping[str, object]], output_root: Path
) -> list[str]:
    errors = []
    for row in rows:
        path = _receipt_output_path(row, output_root)
        if not path.is_file():
            errors.append(f"missing output {path}")
            continue
        if path.stat().st_size != row.get("size_bytes"):
            errors.append(f"output size changed {path}")
        if sha256_file(path) != row.get("sha256"):
            errors.append(f"output hash changed {path}")
    return errors


def _run_stage(
    *,
    output_root: Path,
    options: WorkflowOptions,
    name: str,
    command: Sequence[str],
    input_paths: Sequence[Path],
    output_paths: Sequence[Path],
    action: Callable[[], object],
) -> dict[str, object]:
    receipt_path = output_root / "stages" / f"{name}.json"
    inputs = _input_receipts(input_paths)
    command_tuple = tuple(str(value) for value in command)
    if receipt_path.is_file() and options.resume:
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        if previous.get("name") != name:
            raise RuntimeError(f"{name}: stage name changed")
        if previous.get("command") != list(command_tuple):
            raise RuntimeError(f"{name}: command changed")
        if previous.get("inputs") != list(inputs):
            raise RuntimeError(f"{name}: input hash changed")
        status = previous.get("status")
        if status == "COMPLETE":
            errors = _verify_stage_outputs(previous.get("outputs", []), output_root)
            if errors:
                raise RuntimeError(f"{name}: " + "; ".join(errors))
            return previous
        if status == "FAILED" and not options.retry_failed:
            raise RuntimeError(
                f"{name}: previous stage failed; pass --retry-failed"
            )
        if status == "FAILED" and options.retry_failed:
            for value in output_paths:
                path = Path(value).resolve()
                try:
                    path.relative_to(output_root)
                except ValueError as error:
                    raise RuntimeError(
                        f"{name}: refusing to clear output outside workflow root: {path}"
                    ) from error
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()

    started = _utc_now()
    running = StageResult(
        name=name,
        status="RUNNING",
        started_at_utc=started,
        ended_at_utc="",
        inputs=inputs,
        outputs=(),
        command=command_tuple,
    )
    _atomic_json(receipt_path, asdict(running))
    try:
        action()
        outputs = _output_receipts(output_paths, output_root)
    except Exception as error:
        failed = {
            **asdict(running),
            "status": "FAILED",
            "ended_at_utc": _utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _atomic_json(receipt_path, failed)
        raise
    complete = StageResult(
        name=name,
        status="COMPLETE",
        started_at_utc=started,
        ended_at_utc=_utc_now(),
        inputs=inputs,
        outputs=outputs,
        command=command_tuple,
    )
    payload = asdict(complete)
    _atomic_json(receipt_path, payload)
    return payload


def _run_checked(
    commands: Sequence[Sequence[str]], *, cwd: Path, log_path: Path,
    env: Mapping[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for command in commands:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        blocks.append(
            "$ " + " ".join(command) + "\n" + completed.stdout.rstrip() + "\n"
        )
        if completed.returncode:
            log_path.write_text("\n".join(blocks), encoding="utf-8")
            raise RuntimeError(
                f"command failed ({completed.returncode}): {' '.join(command)}"
            )
    log_path.write_text("\n".join(blocks), encoding="utf-8")


def _native_sources(repo_root: Path) -> list[Path]:
    root = repo_root / "minco_processor"
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name == "CMakeLists.txt"
            or path.suffix.lower() in {".cpp", ".cc", ".c", ".hpp", ".h"}
        )
        and "build" not in path.parts
    )


def _native_build(repo_root: Path, output: Path) -> None:
    build_dir = repo_root / "minco_processor" / "build"
    commands: list[list[str]] = []
    if not (build_dir / "CMakeCache.txt").is_file():
        commands.append(
            [
                "cmake", "-S", str(repo_root / "minco_processor"),
                "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
            ]
        )
    commands.append(["cmake", "--build", str(build_dir), "-j2"])
    commands.append([str(build_dir / "minco_processor_compile_test")])
    _run_checked(commands, cwd=repo_root, log_path=output / "build.log")
    extension = next(
        (
            path
            for pattern in ("minco_processor*.so", "_minco_processor*.so")
            for path in sorted(build_dir.glob(pattern))
        ),
        None,
    )
    executable = build_dir / "minco_processor_compile_test"
    if extension is None or not executable.is_file():
        raise RuntimeError("native MINCO build outputs are incomplete")
    _atomic_json(
        output / "native_build_receipt.json",
        {
            "schema_version": 1,
            "extension": str(extension),
            "extension_sha256": sha256_file(extension),
            "compile_test": str(executable),
            "compile_test_sha256": sha256_file(executable),
        },
    )


def _usd_environment(repo_root: Path) -> tuple[Path, dict[str, str]]:
    conda_exe = Path(os.environ.get("CONDA_EXE", "/home/alioth/miniforge3/bin/conda"))
    base = conda_exe.resolve().parent.parent
    isaac_prefix = base / "envs" / "isaaclab"
    python = isaac_prefix / "bin" / "python"
    candidates = sorted(
        (isaac_prefix / "lib" / "python3.10" / "site-packages" / "isaacsim" / "extscache").glob(
            "omni.usd.libs-*"
        )
    )
    if not python.is_file() or not candidates:
        raise RuntimeError(
            "IsaacLab OpenUSD runtime is required for read-only robot calibration"
        )
    usd_root = candidates[-1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(usd_root), str(repo_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(usd_root / "bin"), env.get("LD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    return python, env


def _calibration(
    repo_root: Path, options: WorkflowOptions, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / "usd_evidence.json"
    python, env = _usd_environment(repo_root)
    command = [
        str(python), "-m", "experiments.calibration.usd_extract",
        "--usd", str(options.robot_usd), "--output", str(evidence),
    ]
    _run_checked(
        [command], cwd=repo_root, log_path=output / "usd_extract.log", env=env
    )
    render_calibration_report(
        options.calibration_path,
        evidence,
        options.calibration_protocol,
        output,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _static_pair(legacy: Path, safe: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    legacy_rows = {
        row["case_uid"]: row for row in _read_csv(legacy / "legacy_case_metrics.csv")
    }
    safe_rows = {
        row["case_uid"]: row for row in _read_csv(safe / "legacy_case_metrics.csv")
    }
    fields = sorted(
        set().union(*(row.keys() for row in [*legacy_rows.values(), *safe_rows.values()]))
    )
    rows = []
    for uid in sorted(set(legacy_rows) | set(safe_rows)):
        left = legacy_rows.get(uid, {})
        right = safe_rows.get(uid, {})
        row: dict[str, object] = {
            "case_uid": uid,
            "pair_status": "COMPLETE" if left and right else "INCOMPLETE",
        }
        for field in fields:
            if field == "case_uid":
                continue
            row[f"legacy_{field}"] = left.get(field, "")
            row[f"safe_{field}"] = right.get(field, "")
        rows.append(row)
    output_csv = output / "paired_profile_metrics.csv"
    csv_fields = list(rows[0]) if rows else ["case_uid", "pair_status"]
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    _atomic_json(
        output / "comparison_receipt.json",
        {
            "schema_version": 1,
            "pairing_key": ["case_uid"],
            "case_count": len(rows),
            "complete_pair_count": sum(row["pair_status"] == "COMPLETE" for row in rows),
            "legacy_input_sha256": sha256_file(legacy / "legacy_case_metrics.csv"),
            "safe_input_sha256": sha256_file(safe / "legacy_case_metrics.csv"),
        },
    )
    (output / "report.md").write_text(
        "# Paired static profile comparison\n\n"
        f"Cases are joined exactly by `case_uid`: {len(rows)} total.\n"
        "No failed or missing case is dropped; `pair_status` records completeness.\n",
        encoding="utf-8",
    )


def _paper(input_root: Path, output: Path) -> dict[str, object]:
    return generate_paper_report(input_root, output)


def _run_rolling_showcase_stage(
    *,
    output_root: Path,
    options: WorkflowOptions,
    input_paths: Sequence[Path],
) -> dict[str, object]:
    output = output_root / "paper_showcase"

    def action() -> None:
        run_rolling_showcase(options.rolling_showcase_config, output)
        errors = validate_showcase(output)
        if errors:
            raise RuntimeError("rolling showcase validation failed: " + "; ".join(errors))

    return _run_stage(
        output_root=output_root,
        options=options,
        name="rolling_showcase",
        command=("rolling-showcase", str(options.rolling_showcase_config)),
        input_paths=input_paths,
        output_paths=(output,),
        action=action,
    )


def _validate_inventory(root: Path) -> list[str]:
    path = root / "artifact_receipt.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable artifact receipt {path}: {error}"]
    return [
        error
        for row in payload.get("artifacts", [])
        for error in validate_file_receipt(root, row)
    ]


def _static_validation(
    output_root: Path, *, require_rolling_showcase: bool = True
) -> None:
    errors = []
    errors.extend(validate_static_benchmark(output_root / "static" / "legacy"))
    errors.extend(
        validate_static_benchmark(output_root / "static" / "safe_corridor_v1")
    )
    errors.extend(_validate_inventory(output_root / "paper"))
    errors.extend(validate_paper_artifact_manifest(output_root / "paper"))
    if require_rolling_showcase:
        errors.extend(validate_showcase(output_root / "paper_showcase"))
    selection = output_root / "boundary" / "selected_dynamic_cases.json"
    try:
        frozen = json.loads(selection.read_text(encoding="utf-8"))
        main = list(frozen.get("best2", [])) + list(frozen.get("worst2", []))
        if len(main) != 4 or len(set(main)) != 4:
            errors.append("selection must contain four unique Best2/Worst2 cases")
        if frozen.get("hot_start_evidence") != "PENDING_DYNAMIC_VALIDATION":
            errors.append("static workflow cannot claim hot-start evidence")
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"selection unreadable: {error}")
    report = {
        "schema_version": 1,
        "valid": not errors,
        "errors": errors,
    }
    _atomic_json(output_root / "validation" / "static_validation.json", report)
    if errors:
        raise RuntimeError("static workflow validation failed: " + "; ".join(errors))


def _write_artifact_manifest(output_root: Path) -> dict[str, object]:
    path = output_root / "artifact_manifest.json"
    excluded_roots = {
        (output_root / "stages").resolve(),
    }
    artifacts = []
    for candidate in sorted(output_root.rglob("*")):
        resolved = candidate.resolve()
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or candidate == path
            or any(root == resolved.parent or root in resolved.parents for root in excluded_roots)
            or candidate.name.endswith("workflow_receipt.json")
        ):
            continue
        artifacts.append(
            {
                "path": candidate.relative_to(output_root).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    payload = {
        "schema_version": 1,
        "root": str(output_root),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    _atomic_json(path, payload)
    return payload


def _workflow_inputs(repo_root: Path, options: WorkflowOptions) -> list[Path]:
    paths = [
        options.legacy_config,
        options.safe_config,
        options.selection_config,
        options.calibration_path,
        options.calibration_protocol,
        options.robot_usd,
        repo_root / "configs" / "robots" / "dingo_config.py",
    ]
    if not options.skip_rolling_showcase:
        paths.append(options.rolling_showcase_config)
    return paths


def _prepare_root(output_root: Path, resume: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError(
            f"workflow output already exists; pass --resume: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)


def run_static_workflow(
    *, repo_root: Path | str, options: WorkflowOptions
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    options = _resolve_options(repo_root, options)
    output_root = options.output_root
    _prepare_root(output_root, options.resume)
    common_inputs = _workflow_inputs(repo_root, options)
    stages: dict[str, dict[str, object]] = {}

    build_output = output_root / "build"
    stages["native_build"] = _run_stage(
        output_root=output_root,
        options=options,
        name="native_build",
        command=("cmake-configure-build-test",),
        input_paths=[*common_inputs, *_native_sources(repo_root)],
        output_paths=(build_output,),
        action=lambda: _native_build(repo_root, build_output),
    )
    calibration_output = output_root / "calibration"
    stages["calibration"] = _run_stage(
        output_root=output_root,
        options=options,
        name="calibration",
        command=("read-only-usd-calibration", str(options.robot_usd)),
        input_paths=common_inputs,
        output_paths=(calibration_output,),
        action=lambda: _calibration(repo_root, options, calibration_output),
    )
    legacy_output = output_root / "static" / "legacy"
    stages["legacy_benchmark"] = _run_stage(
        output_root=output_root,
        options=options,
        name="legacy_benchmark",
        command=("static-benchmark", "legacy", "--trace-limit=0"),
        input_paths=common_inputs,
        output_paths=(legacy_output,),
        action=lambda: run_static_benchmark(
            options.legacy_config, legacy_output, trace_limit=0
        ),
    )
    safe_output = output_root / "static" / "safe_corridor_v1"
    stages["safe_benchmark"] = _run_stage(
        output_root=output_root,
        options=options,
        name="safe_benchmark",
        command=("static-benchmark", "safe_corridor_v1", "--trace-limit=0"),
        input_paths=common_inputs,
        output_paths=(safe_output,),
        action=lambda: run_static_benchmark(
            options.safe_config, safe_output, trace_limit=0
        ),
    )
    comparison_output = output_root / "static" / "comparison"
    stages["static_comparison"] = _run_stage(
        output_root=output_root,
        options=options,
        name="static_comparison",
        command=("pair-static-profiles", "case_uid"),
        input_paths=[
            *common_inputs,
            legacy_output / "legacy_case_metrics.csv",
            safe_output / "legacy_case_metrics.csv",
        ],
        output_paths=(comparison_output,),
        action=lambda: _static_pair(legacy_output, safe_output, comparison_output),
    )
    boundary_output = output_root / "boundary"
    stages["boundary_selection"] = _run_stage(
        output_root=output_root,
        options=options,
        name="boundary_selection",
        command=("static-boundary-selection", "Best2", "Worst2"),
        input_paths=common_inputs,
        output_paths=(boundary_output,),
        action=lambda: run_boundary_selection(
            options.selection_config, boundary_output
        ),
    )
    if not options.skip_rolling_showcase:
        stages["rolling_showcase"] = _run_rolling_showcase_stage(
            output_root=output_root,
            options=options,
            input_paths=common_inputs,
        )
    paper_output = output_root / "paper"
    stages["paper_report"] = _run_stage(
        output_root=output_root,
        options=options,
        name="paper_report",
        command=("generate-data-driven-paper-report",),
        input_paths=[
            *common_inputs,
            boundary_output / "static_runs.csv",
            boundary_output / "selected_dynamic_cases.json",
        ],
        output_paths=(paper_output,),
        action=lambda: _paper(output_root, paper_output),
    )
    validation_output = output_root / "validation" / "static_validation.json"
    stages["validation"] = _run_stage(
        output_root=output_root,
        options=options,
        name="validation",
        command=("validate-static-workflow",),
        input_paths=common_inputs,
        output_paths=(validation_output,),
        action=lambda: _static_validation(
            output_root,
            require_rolling_showcase=not options.skip_rolling_showcase,
        ),
    )
    manifest = _write_artifact_manifest(output_root)
    receipt = {
        "schema_version": 1,
        "workflow": "static",
        "environments": _environment_receipt(),
        "status": "COMPLETE",
        "created_at_utc": _utc_now(),
        "output_root": str(output_root),
        "options": {
            "resume": options.resume,
            "retry_failed": options.retry_failed,
            "skip_video": options.skip_video,
            "skip_rolling_showcase": options.skip_rolling_showcase,
            "rolling_showcase_config": str(options.rolling_showcase_config),
        },
        "stages": stages,
        "validation_errors": [],
        "artifact_count": manifest["artifact_count"],
    }
    _atomic_json(output_root / "static_workflow_receipt.json", receipt)
    return receipt


def _simulation_validation(output_root: Path) -> None:
    receipt_path = output_root / "dynamic_readiness" / "dynamic_readiness_receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        dry_plan = json.loads(Path(receipt["dry_run_plan"]).read_text(encoding="utf-8"))
    except (OSError, KeyError, json.JSONDecodeError) as error:
        errors = [f"dynamic readiness unreadable: {error}"]
    else:
        errors = []
        if receipt.get("run_count") != 8 or receipt.get("started_processes") != 0:
            errors.append("dynamic readiness must contain eight runs and zero processes")
        if dry_plan.get("run_count") != 8 or dry_plan.get("started_processes") != 0:
            errors.append("dynamic dry-run must contain eight runs and zero processes")
        if sha256_file(Path(receipt["dry_run_plan"])) != receipt.get("dry_run_plan_sha256"):
            errors.append("dynamic dry-run plan hash mismatch")
    mock_path = (
        output_root / "simulation" / "mock_smoke" / "mock_smoke_receipt.json"
    )
    try:
        mock = json.loads(mock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"mock smoke receipt unreadable: {error}")
    else:
        if mock.get("status") != "COMPLETE" or mock.get("failed_runs") != 0:
            errors.append("mock smoke suite is not complete")
    _atomic_json(
        output_root / "validation" / "simulation_validation.json",
        {"schema_version": 1, "valid": not errors, "errors": errors},
    )
    if errors:
        raise RuntimeError("simulation workflow validation failed: " + "; ".join(errors))


def _materialize_full_suite(options: WorkflowOptions, output: Path) -> Path:
    payload = json.loads(options.full_suite_config.read_text(encoding="utf-8"))
    manifest = Path(payload.get("scenario_manifest", payload.get("manifest", "")))
    if not manifest.is_absolute():
        manifest = (options.full_suite_config.parent / manifest).resolve()
    payload["scenario_manifest"] = str(manifest)
    payload["output_root"] = str(output / "results")
    path = output / "full_suite.json"
    _atomic_json(path, payload)
    return path


def run_mock_smoke_suite(
    *, repo_root: Path | str, output_dir: Path | str
) -> dict[str, object]:
    """Execute the local mock smoke suite and retain its validation/analysis."""
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = repo_root / "experiments" / "configs" / "smoke_suite.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    manifest = Path(payload.get("scenario_manifest", payload.get("manifest", "")))
    if not manifest.is_absolute():
        manifest = (source.parent / manifest).resolve()
    payload["backend"] = "mock"
    payload["scenario_manifest"] = str(manifest)
    payload["output_root"] = str(output_dir / "results")
    payload["analysis"] = {"enabled": True, "paired": True}
    config_path = output_dir / "mock_smoke_suite.json"
    _atomic_json(config_path, payload)
    result = run_suite(
        config_path,
        backend_name="mock",
        resume=True,
        retry_failed=False,
        analysis_enabled=True,
    )
    suite_dir = Path(payload["output_root"]) / payload["suite_id"]
    errors = validate_artifact_manifest(suite_dir)
    status_payload = json.loads(
        (suite_dir / "suite_status.json").read_text(encoding="utf-8")
    )
    if result.failed or status_payload.get("status") != "COMPLETE" or errors:
        raise RuntimeError(
            "mock smoke suite failed: "
            + "; ".join(errors or [f"failed_runs={result.failed}"])
        )
    receipt = {
        "schema_version": 1,
        "status": "COMPLETE",
        "data_source": "SIMULATED",
        "completed_runs": result.completed + result.skipped,
        "failed_runs": result.failed,
        "suite_dir": str(suite_dir),
        "artifact_validation_errors": errors,
    }
    _atomic_json(output_dir / "mock_smoke_receipt.json", receipt)
    return receipt


def run_simulation_workflow(
    *, repo_root: Path | str, options: WorkflowOptions
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    options = _resolve_options(repo_root, options)
    output_root = options.output_root
    static_receipt = output_root / "static_workflow_receipt.json"
    if not static_receipt.is_file():
        run_static_workflow(repo_root=repo_root, options=options)
        options = _resolve_options(
            repo_root, WorkflowOptions(**{**asdict(options), "resume": True})
        )
    else:
        _prepare_root(output_root, True)
    common_inputs = _workflow_inputs(repo_root, options)
    stages: dict[str, dict[str, object]] = {}
    mock_output = output_root / "simulation" / "mock_smoke"
    smoke_config = repo_root / "experiments" / "configs" / "smoke_suite.json"
    smoke_manifest = repo_root / "experiments" / "configs" / "mock_scenarios.json"
    stages["mock_smoke"] = _run_stage(
        output_root=output_root,
        options=options,
        name="mock_smoke",
        command=("run-suite", "mock", "smoke", "--analysis"),
        input_paths=[*common_inputs, smoke_config, smoke_manifest],
        output_paths=(mock_output,),
        action=lambda: run_mock_smoke_suite(
            repo_root=repo_root, output_dir=mock_output
        ),
    )
    dynamic_output = output_root / "dynamic_readiness"
    stages["dynamic_readiness"] = _run_stage(
        output_root=output_root,
        options=options,
        name="dynamic_readiness",
        command=("dynamic-prepare", "--dry-run", "8-runs"),
        input_paths=[
            *common_inputs,
            output_root / "boundary" / "selected_dynamic_cases.json",
            output_root / "boundary" / "selection_policy.json",
        ],
        output_paths=(dynamic_output,),
        action=lambda: prepare_dynamic_pilot(
            output_root / "boundary" / "selected_dynamic_cases.json",
            options.calibration_path,
            options.legacy_config,
            options.safe_config,
            dynamic_output,
            repo_root=repo_root,
        ),
    )
    if options.allow_real_simulation:
        suite_path = dynamic_output / "dynamic_suite.json"
        dynamic_suite_dir = dynamic_output / "dry_run_results" / "task06_dynamic_best2_worst2_v1"
        stages["dynamic_real"] = _run_stage(
            output_root=output_root,
            options=options,
            name="dynamic_real",
            command=("run-suite", str(suite_path), "--allow-real-simulation"),
            input_paths=[*common_inputs, suite_path],
            output_paths=(dynamic_suite_dir,),
            action=lambda: run_suite(
                suite_path,
                backend_name="isaac",
                resume=options.resume,
                retry_failed=options.retry_failed,
                allow_real_simulation=True,
                skip_video=options.skip_video,
            ),
        )
        analysis_output = output_root / "paper" / "dynamic_pilot"
        stages["dynamic_analysis"] = _run_stage(
            output_root=output_root,
            options=options,
            name="dynamic_analysis",
            command=("analyze-suite-readonly", str(dynamic_suite_dir)),
            input_paths=[*common_inputs, dynamic_suite_dir / "suite_status.json"],
            output_paths=(analysis_output,),
            action=lambda: analyze_suite_readonly(
                dynamic_suite_dir, analysis_output, resume=options.resume
            ),
        )
    if options.allow_real_simulation and options.full_suite:
        simulation_dir = output_root / "simulation"
        simulation_dir.mkdir(parents=True, exist_ok=True)
        suite_path = _materialize_full_suite(options, simulation_dir)
        suite_payload = json.loads(suite_path.read_text(encoding="utf-8"))
        suite_dir = Path(suite_payload["output_root"]) / suite_payload["suite_id"]
        stages["full_real_suite"] = _run_stage(
            output_root=output_root,
            options=options,
            name="full_real_suite",
            command=("run-suite", str(suite_path), "--allow-real-simulation"),
            input_paths=[*common_inputs, options.full_suite_config, suite_path],
            output_paths=(suite_dir,),
            action=lambda: run_suite(
                suite_path,
                backend_name="isaac",
                resume=options.resume,
                retry_failed=options.retry_failed,
                allow_real_simulation=True,
                skip_video=options.skip_video,
            ),
        )
        analysis_output = output_root / "paper" / "full_simulation"
        stages["full_real_analysis"] = _run_stage(
            output_root=output_root,
            options=options,
            name="full_real_analysis",
            command=("analyze-suite-readonly", str(suite_dir)),
            input_paths=[*common_inputs, suite_dir / "suite_status.json"],
            output_paths=(analysis_output,),
            action=lambda: analyze_suite_readonly(
                suite_dir, analysis_output, resume=options.resume
            ),
        )
    validation_path = output_root / "validation" / "simulation_validation.json"
    stages["simulation_validation"] = _run_stage(
        output_root=output_root,
        options=options,
        name="simulation_validation",
        command=("validate-simulation-workflow",),
        input_paths=[
            *common_inputs,
            dynamic_output / "dynamic_readiness_receipt.json",
        ],
        output_paths=(validation_path,),
        action=lambda: _simulation_validation(output_root),
    )
    manifest = _write_artifact_manifest(output_root)
    status = "COMPLETE" if options.allow_real_simulation else "READY_FOR_REAL_RUN"
    receipt = {
        "schema_version": 1,
        "workflow": "simulation",
        "environments": _environment_receipt(),
        "status": status,
        "created_at_utc": _utc_now(),
        "output_root": str(output_root),
        "real_simulation_authorized": options.allow_real_simulation,
        "full_suite_requested": options.full_suite,
        "stages": stages,
        "validation_errors": [],
        "artifact_count": manifest["artifact_count"],
    }
    _atomic_json(output_root / "simulation_workflow_receipt.json", receipt)
    return receipt


def run_all_workflows(
    *, repo_root: Path | str, options: WorkflowOptions
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    options = _resolve_options(repo_root, options)
    static = run_static_workflow(repo_root=repo_root, options=options)
    simulation_options = WorkflowOptions(**{**asdict(options), "resume": True})
    simulation = run_simulation_workflow(
        repo_root=repo_root, options=simulation_options
    )
    manifest = _write_artifact_manifest(options.output_root)
    receipt = {
        "schema_version": 1,
        "workflow": "all",
        "environments": _environment_receipt(),
        "status": simulation["status"],
        "created_at_utc": _utc_now(),
        "output_root": str(options.output_root),
        "static": static,
        "simulation": simulation,
        "stages": {**static["stages"], **simulation["stages"]},
        "validation_errors": [
            *static.get("validation_errors", []),
            *simulation.get("validation_errors", []),
        ],
        "artifact_count": manifest["artifact_count"],
    }
    if receipt["validation_errors"]:
        receipt["status"] = "VALIDATION_FAILED"
    _atomic_json(options.output_root / "all_workflow_receipt.json", receipt)
    _atomic_json(options.output_root / "experiment_receipt.json", receipt)
    return receipt
