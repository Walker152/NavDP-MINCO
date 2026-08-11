from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


REQUIRED_VARIANTS = ("raw", "minco-cold", "minco-hot")


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    experiment_id: str
    scene_id: str
    scene_label: str
    seed: int
    variant: str
    run_id: str
    data_source: str
    config_hash: str
    speed_mps: float | None


@dataclass(frozen=True)
class EpisodeRecord:
    episode_uid: str
    run: RunRecord
    metrics: Mapping[str, str]
    video_path: Path | None
    video_receipt_path: Path | None
    trace_paths: tuple[Path, ...]
    control_csv: Path | None


@dataclass(frozen=True)
class EpisodeGroup:
    key: tuple[str, str, int, str]
    variants: Mapping[str, EpisodeRecord]
    missing_variants: tuple[str, ...]
    status: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(config: Mapping[str, object], field: str, path: Path) -> str:
    value = config.get(field)
    if value is None or not str(value).strip():
        raise ValueError(f"malformed run config {path}: missing {field}")
    return str(value)


def _load_run(config_path: Path) -> RunRecord:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed run config {config_path}: {error}") from error
    try:
        seed = int(config["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"malformed run config {config_path}: invalid seed"
        ) from error
    try:
        speed_mps = float(config["speed_mps"])
        if not math.isfinite(speed_mps):
            speed_mps = None
    except (KeyError, TypeError, ValueError):
        speed_mps = None
    return RunRecord(
        run_dir=config_path.parent.resolve(),
        experiment_id=_required_text(config, "experiment_id", config_path),
        scene_id=_required_text(config, "scene_id", config_path),
        scene_label=str(config.get("scene_label", "")),
        seed=seed,
        variant=_required_text(config, "variant", config_path),
        run_id=_required_text(config, "run_id", config_path),
        data_source=str(config.get("data_source", "UNKNOWN")),
        config_hash=_sha256(config_path),
        speed_mps=speed_mps,
    )


def _read_episode_rows(run: RunRecord) -> list[dict[str, str]]:
    path = run.run_dir / "episode_metrics.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for index, row in enumerate(rows, start=2):
        if not str(row.get("episode_uid", "")).strip():
            raise ValueError(f"missing episode_uid in {path}:{index}")
    return rows


def _episode_record(run: RunRecord, row: dict[str, str]) -> EpisodeRecord:
    episode_uid = str(row["episode_uid"])
    video = run.run_dir / "videos" / f"{episode_uid}.mp4"
    receipt = run.run_dir / "videos" / f"{episode_uid}.video_complete.json"
    trace_paths = tuple(
        sorted(
            path.resolve()
            for path in (run.run_dir / "traces").glob(
                f"planning_trace_{episode_uid}_*.npz"
            )
        )
    )
    control_csv = run.run_dir / "control_samples.csv"
    return EpisodeRecord(
        episode_uid=episode_uid,
        run=run,
        metrics=MappingProxyType(dict(row)),
        video_path=video.resolve() if video.is_file() else None,
        video_receipt_path=receipt.resolve() if receipt.is_file() else None,
        trace_paths=trace_paths,
        control_csv=control_csv.resolve() if control_csv.is_file() else None,
    )


def build_episode_groups(
    suite_dir: Path | str,
    required_variants: tuple[str, ...] = REQUIRED_VARIANTS,
) -> list[EpisodeGroup]:
    suite_dir = Path(suite_dir).resolve()
    if not suite_dir.is_dir():
        raise FileNotFoundError(f"input suite does not exist: {suite_dir}")

    grouped: dict[
        tuple[str, str, int, str], dict[str, EpisodeRecord]
    ] = {}
    for config_path in sorted(
        (suite_dir / "experiments").glob("*/*/*/*/*/run_config.json")
    ):
        run = _load_run(config_path)
        for row in _read_episode_rows(run):
            record = _episode_record(run, row)
            key = (
                run.experiment_id,
                run.scene_id,
                run.seed,
                record.episode_uid,
            )
            variants = grouped.setdefault(key, {})
            if run.variant in variants:
                previous = variants[run.variant].run.run_dir
                raise ValueError(
                    "duplicate variant for episode identity "
                    f"{key}: {run.variant} in {previous} and {run.run_dir}"
                )
            variants[run.variant] = record

    groups = []
    for key, variants in sorted(grouped.items()):
        missing = tuple(
            variant for variant in required_variants if variant not in variants
        )
        groups.append(
            EpisodeGroup(
                key=key,
                variants=MappingProxyType(dict(sorted(variants.items()))),
                missing_variants=missing,
                status="COMPLETE" if not missing else "INCOMPLETE",
            )
        )
    return groups
