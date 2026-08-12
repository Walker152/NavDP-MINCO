from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


PROTECTED_RECEIPTS = frozenset(
    {
        "suite_status.json",
        "suite_config.json",
        "scenario_manifest.json",
        "run_status.json",
        "run_config.json",
        "run_manifest.json",
    }
)

INPUT_EVIDENCE_SUFFIXES = frozenset({".csv", ".json", ".mp4", ".npz"})


@dataclass(frozen=True)
class ReadOnlyAnalysisResult:
    input_suite: Path
    output_dir: Path
    protected_hashes: Mapping[str, str]
    input_evidence: tuple[Mapping[str, object], ...] = ()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_protected_receipts(suite_dir: Path | str) -> dict[str, str]:
    suite_dir = Path(suite_dir).resolve()
    if not suite_dir.is_dir():
        raise FileNotFoundError(f"input suite does not exist: {suite_dir}")
    paths = (
        path
        for path in suite_dir.rglob("*.json")
        if path.name in PROTECTED_RECEIPTS
    )
    return {
        str(path.relative_to(suite_dir)): _sha256(path)
        for path in sorted(paths)
    }


def snapshot_input_evidence(
    suite_dir: Path | str,
) -> list[dict[str, object]]:
    """Return a stable content inventory of analysis-consumable source files."""
    suite_dir = Path(suite_dir).resolve()
    if not suite_dir.is_dir():
        raise FileNotFoundError(f"input suite does not exist: {suite_dir}")
    paths = sorted(
        path
        for path in suite_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in INPUT_EVIDENCE_SUFFIXES
    )
    return [
        {
            "path": str(path.relative_to(suite_dir)),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def analyze_suite_readonly(
    suite_dir: Path | str,
    output_dir: Path | str,
    *,
    max_trace_cases: int = 12,
    resume: bool = False,
) -> ReadOnlyAnalysisResult:
    suite_dir = Path(suite_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == suite_dir or suite_dir in output_dir.parents:
        raise ValueError("read-only analysis output must be outside the input suite")

    before = snapshot_protected_receipts(suite_dir)
    evidence_before = snapshot_input_evidence(suite_dir)
    from experiments.analyzers.comparison_study import (
        generate_comparison_study,
    )

    generate_comparison_study(
        suite_dir,
        output_dir,
        max_trace_cases=max_trace_cases,
        resume=resume,
    )
    after = snapshot_protected_receipts(suite_dir)
    if before != after:
        raise RuntimeError("read-only analysis mutated protected receipts")
    if evidence_before != snapshot_input_evidence(suite_dir):
        raise RuntimeError("read-only analysis mutated source evidence")

    return ReadOnlyAnalysisResult(
        input_suite=suite_dir,
        output_dir=output_dir,
        protected_hashes=MappingProxyType(before),
        input_evidence=tuple(
            MappingProxyType(dict(row)) for row in evidence_before
        ),
    )
