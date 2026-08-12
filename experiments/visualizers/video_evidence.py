"""Immutable, fail-closed evidence packages for paper videos and animations."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Iterable, Mapping, Sequence

import cv2
import imageio.v2 as imageio


SCHEMA_VERSION = 1
COMPLETE = "COMPLETE"
PENDING_REAL_SIMULATION = "PENDING_REAL_SIMULATION"
NOT_APPLICABLE = "NOT_APPLICABLE"

FRAME_METRICS_FIELDS = (
    "schema_version",
    "evidence_uid",
    "media_uid",
    "case_uid",
    "episode_uid",
    "frame_index",
    "time_s",
    "source_sample_index",
    "source_time_s",
    "x_m",
    "y_m",
    "yaw_rad",
    "progress_ratio",
    "clearance_m",
    "guide_deviation_m",
    "speed_mps",
    "yaw_rate_rps",
    "acceleration_mps2",
    "yaw_acceleration_rps2",
    "jerk_mps3",
    "tracking_error_m",
    "command_linear_mps",
    "command_angular_rps",
    "optimizer_state",
    "control_state",
    "planning_state",
    "termination_state",
    "collision",
    "goal_reached",
    "local_goal_x_m",
    "local_goal_y_m",
    "event_tags",
    "data_availability",
)

EVENT_TIMELINE_FIELDS = (
    "schema_version",
    "evidence_uid",
    "media_uid",
    "case_uid",
    "episode_uid",
    "event_uid",
    "event_type",
    "start_frame_index",
    "end_frame_index",
    "start_time_s",
    "end_time_s",
    "severity",
    "source_uid",
    "metric_name",
    "metric_value",
    "metric_unit",
    "description_zh",
    "data_availability",
)

CAPTION_FIELDS = (
    "证据状态",
    "证据编号",
    "媒体编号",
    "对象编号",
    "配对键",
    "数据来源",
    "研究问题",
    "媒体类型",
    "场景与方法",
    "时间基准",
    "帧与样本",
    "分母",
    "指标与单位",
    "事件定义",
    "同步与误差",
    "缺失数据",
    "失败与终止",
    "证据边界",
    "局限性",
    "结论限制",
    "解读",
)

_CAPTION_DEFAULTS = {
    "scene_zh": "未单独声明场景",
    "method_zh": "未单独声明方法",
    "time_basis_zh": "媒体帧时间轴；以 frame_metrics.csv 为准",
    "metrics_units_zh": "位置/间隙/误差：m；时间：s；速度：m/s；角量：rad",
    "event_definition_zh": "事件来自源数据中的可验证状态或数值边界",
    "synchronization_zh": "单媒体帧索引对齐；未声称精确墙钟同步",
    "missing_data_zh": "不可用值保留为空，不以数值零替代",
    "failure_handling_zh": "失败与终止事件保留在时间轴和分母中",
    "evidence_boundary_zh": "仅支持清单所列数据源和媒体内容",
    "conclusion_limit_zh": "不得超出数据来源与同步精度作性能推断",
    "research_question_zh": "未单独声明研究问题；以证据编号与对象编号标识的研究对象为准",
    "pairing_key_zh": "未单独声明配对键；以对象编号（case_uid/episode_uid）为键",
    "denominator_zh": "分母为全部解码帧；失败与终止帧不从分母删除",
    "limitation_zh": "证据仅支持清单所列数据源与媒体内容；不得外推至未测量条件",
    "interpretation_zh": "解读需结合媒体、逐帧指标与事件时间线；单字段不构成结论",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt(path: Path, root: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: object, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _integer(value: object, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if str(value).strip() not in {str(result), f"{result}.0"}:
        raise ValueError(f"{label} must be an integer")
    return result


def _identity(value: object, label: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{label} must be non-empty")
    return result


def _decode_media(path: Path) -> dict[str, object]:
    suffix = path.suffix.lower()
    if suffix not in {".gif", ".mp4"}:
        raise ValueError("evidence media must be GIF or MP4")
    if suffix == ".mp4":
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"media cannot be decoded: {path}")
        frame_count = 0
        width = height = None
        decoder_fps = float(capture.get(cv2.CAP_PROP_FPS))
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                current_height, current_width = frame.shape[:2]
                if current_height <= 0 or current_width <= 0:
                    raise ValueError(f"decoded frame {frame_count} is empty")
                if width is None:
                    width, height = current_width, current_height
                elif (width, height) != (current_width, current_height):
                    raise ValueError("decoded media changes frame dimensions")
                frame_count += 1
        finally:
            capture.release()
        if frame_count <= 0 or width is None or height is None:
            raise ValueError(f"media contains no decodable frames: {path}")
        return {
            "media_type": "MP4",
            "frame_count": frame_count,
            "width_px": width,
            "height_px": height,
            "decoder_fps": (
                decoder_fps
                if math.isfinite(decoder_fps) and decoder_fps > 0.0
                else None
            ),
            "decode_verified": True,
        }
    try:
        reader = imageio.get_reader(path)
    except Exception as error:
        raise ValueError(f"media cannot be decoded: {path}: {error}") from error
    frame_count = 0
    width = height = None
    metadata: Mapping[str, object] = {}
    try:
        metadata = reader.get_meta_data()
        for frame in reader:
            if getattr(frame, "ndim", 0) < 2:
                raise ValueError(f"decoded frame {frame_count} has invalid shape")
            current_height, current_width = int(frame.shape[0]), int(frame.shape[1])
            if current_height <= 0 or current_width <= 0:
                raise ValueError(f"decoded frame {frame_count} is empty")
            if width is None:
                width, height = current_width, current_height
            elif (width, height) != (current_width, current_height):
                raise ValueError("decoded media changes frame dimensions")
            frame_count += 1
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"media decode failed: {path}: {error}") from error
    finally:
        reader.close()
    if frame_count <= 0 or width is None or height is None:
        raise ValueError(f"media contains no decodable frames: {path}")
    decoder_fps = metadata.get("fps")
    try:
        decoder_fps = float(decoder_fps) if decoder_fps is not None else None
    except (TypeError, ValueError):
        decoder_fps = None
    if decoder_fps is not None and not math.isfinite(decoder_fps):
        decoder_fps = None
    return {
        "media_type": suffix[1:].upper(),
        "frame_count": frame_count,
        "width_px": width,
        "height_px": height,
        "decoder_fps": decoder_fps,
        "decode_verified": True,
    }


def _normalise_rows(
    rows: Iterable[Mapping[str, object]],
    fields: Sequence[str],
    identity: Mapping[str, str],
) -> list[dict[str, object]]:
    output = []
    allowed = set(fields)
    for index, source in enumerate(rows):
        unknown = set(source) - allowed
        if unknown:
            raise ValueError(f"unsupported evidence fields: {sorted(unknown)}")
        row = {field: source.get(field, "") for field in fields}
        row["schema_version"] = SCHEMA_VERSION
        for field, expected in identity.items():
            supplied = str(source.get(field, "")).strip()
            if supplied and supplied != expected:
                raise ValueError(
                    f"row {index} {field} does not agree with package identity"
                )
            row[field] = expected
        if not str(row.get("data_availability", "")).strip():
            row["data_availability"] = "UNAVAILABLE"
        output.append(row)
    return output


def _validate_frame_rows(rows: Sequence[Mapping[str, object]]) -> list[float]:
    times = []
    for expected_index, row in enumerate(rows):
        frame_index = _integer(row.get("frame_index"), "frame_index")
        if frame_index != expected_index:
            raise ValueError("frame_index must be contiguous from zero")
        time_s = _finite(row.get("time_s"), "frame time_s")
        if time_s < 0.0:
            raise ValueError("frame time_s must be non-negative")
        times.append(time_s)
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("frame time_s must be strictly increasing")
    return times


def _validate_event_rows(
    rows: Sequence[Mapping[str, object]], frame_times: Sequence[float]
) -> None:
    event_uids: set[str] = set()
    for row in rows:
        event_uid = _identity(row.get("event_uid"), "event_uid")
        if event_uid in event_uids:
            raise ValueError(f"duplicate event_uid: {event_uid}")
        event_uids.add(event_uid)
        _identity(row.get("event_type"), "event_type")
        start_frame = _integer(row.get("start_frame_index"), "event start_frame_index")
        end_frame = _integer(row.get("end_frame_index"), "event end_frame_index")
        if (
            start_frame < 0
            or end_frame < start_frame
            or end_frame >= len(frame_times)
        ):
            raise ValueError(f"event frame bounds are invalid for {event_uid}")
        start_time = _finite(row.get("start_time_s"), "event start_time_s")
        end_time = _finite(row.get("end_time_s"), "event end_time_s")
        if (
            start_time > end_time
            or start_time < frame_times[0]
            or end_time > frame_times[-1]
        ):
            raise ValueError(f"event time bounds are invalid for {event_uid}")


def _write_csv(
    path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _caption_values(
    *,
    evidence_uid: str,
    media_uid: str,
    case_uid: str,
    episode_uid: str,
    data_source: str,
    status: str,
    media_type: str,
    frame_count: int,
    event_count: int,
    overrides: Mapping[str, object],
) -> dict[str, str]:
    values = {**_CAPTION_DEFAULTS}
    unknown = set(overrides) - set(values)
    if unknown:
        raise ValueError(f"unsupported caption overrides: {sorted(unknown)}")
    values.update({key: _identity(value, key) for key, value in overrides.items()})
    object_uid = (
        f"case_uid={case_uid}"
        if case_uid != NOT_APPLICABLE
        else f"episode_uid={episode_uid}"
    )
    if status == PENDING_REAL_SIMULATION:
        values["time_basis_zh"] = "尚未授权或完成真实仿真；无媒体时间轴"
        values["missing_data_zh"] = "尚未授权或完成真实仿真；指标表和事件表仅保留表头"
        values["failure_handling_zh"] = "PENDING_REAL_SIMULATION 不是失败，也不是性能结果"
        values["evidence_boundary_zh"] = (
            "尚未授权或完成真实仿真；不得生成或推断定量动态指标"
        )
    return {
        "证据状态": status,
        "证据编号": evidence_uid,
        "媒体编号": media_uid,
        "对象编号": object_uid,
        "配对键": values["pairing_key_zh"],
        "数据来源": data_source,
        "研究问题": values["research_question_zh"],
        "媒体类型": media_type,
        "场景与方法": f"{values['scene_zh']}；{values['method_zh']}",
        "时间基准": values["time_basis_zh"],
        "帧与样本": f"解码帧 {frame_count}；逐帧记录 {frame_count}；事件 {event_count}",
        "分母": values["denominator_zh"],
        "指标与单位": values["metrics_units_zh"],
        "事件定义": values["event_definition_zh"],
        "同步与误差": values["synchronization_zh"],
        "缺失数据": values["missing_data_zh"],
        "失败与终止": values["failure_handling_zh"],
        "证据边界": values["evidence_boundary_zh"],
        "局限性": values["limitation_zh"],
        "结论限制": values["conclusion_limit_zh"],
        "解读": values["interpretation_zh"],
    }


def _write_caption(path: Path, values: Mapping[str, str]) -> None:
    lines = ["# 视频证据说明", ""]
    lines.extend(f"- {field}：{values[field]}" for field in CAPTION_FIELDS)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_video_evidence_package(
    media_path: Path | str | None,
    output_dir: Path | str,
    *,
    evidence_uid: str,
    media_uid: str,
    data_source: str,
    fps: float | None = None,
    status: str = COMPLETE,
    case_uid: str | None = None,
    episode_uid: str | None = None,
    frame_rows: Iterable[Mapping[str, object]] = (),
    event_rows: Iterable[Mapping[str, object]] = (),
    caption_overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a self-contained GIF/MP4 evidence package and validate it."""

    evidence_uid = _identity(evidence_uid, "evidence_uid")
    media_uid = _identity(media_uid, "media_uid")
    data_source = _identity(data_source, "data_source")
    if status not in {COMPLETE, PENDING_REAL_SIMULATION}:
        raise ValueError(f"unsupported evidence status: {status}")
    case_uid = _identity(case_uid, "case_uid") if case_uid else NOT_APPLICABLE
    episode_uid = (
        _identity(episode_uid, "episode_uid") if episode_uid else NOT_APPLICABLE
    )
    if case_uid == NOT_APPLICABLE and episode_uid == NOT_APPLICABLE and status == COMPLETE:
        raise ValueError("complete evidence requires case_uid or episode_uid")
    identity = {
        "evidence_uid": evidence_uid,
        "media_uid": media_uid,
        "case_uid": case_uid,
        "episode_uid": episode_uid,
    }
    frames = _normalise_rows(frame_rows, FRAME_METRICS_FIELDS, identity)
    events = _normalise_rows(event_rows, EVENT_TIMELINE_FIELDS, identity)

    decoded: dict[str, object] | None
    source: Path | None
    declared_fps: float | None
    if status == PENDING_REAL_SIMULATION:
        if media_path is not None:
            raise ValueError("PENDING_REAL_SIMULATION cannot contain media")
        if frames or events:
            raise ValueError(
                "PENDING_REAL_SIMULATION cannot contain frame or event metrics"
            )
        if data_source != "UNAVAILABLE":
            raise ValueError("PENDING_REAL_SIMULATION data_source must be UNAVAILABLE")
        source = None
        decoded = None
        declared_fps = None
        frame_times: list[float] = []
    else:
        if media_path is None:
            raise ValueError("complete evidence requires GIF or MP4 media")
        source = Path(media_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"evidence media does not exist: {source}")
        declared_fps = _positive(fps, "fps")
        decoded = _decode_media(source)
        if len(frames) != decoded["frame_count"]:
            raise ValueError(
                "frame row count does not agree with decoded media: "
                f"{len(frames)} != {decoded['frame_count']}"
            )
        frame_times = _validate_frame_rows(frames)
        _validate_event_rows(events, frame_times)

    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"immutable evidence output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    media_output = None
    if source is not None:
        media_output = output_dir / source.name
        if source != media_output:
            shutil.copy2(source, media_output)
    frame_path = output_dir / "frame_metrics.csv"
    event_path = output_dir / "event_timeline.csv"
    caption_path = output_dir / "caption_zh.md"
    caption_text_path = output_dir / "caption_zh.txt"
    _write_csv(frame_path, FRAME_METRICS_FIELDS, frames)
    _write_csv(event_path, EVENT_TIMELINE_FIELDS, events)
    caption_values = _caption_values(
        evidence_uid=evidence_uid,
        media_uid=media_uid,
        case_uid=case_uid,
        episode_uid=episode_uid,
        data_source=data_source,
        status=status,
        media_type=str(decoded["media_type"]) if decoded else "UNAVAILABLE",
        frame_count=len(frames),
        event_count=len(events),
        overrides=caption_overrides or {},
    )
    _write_caption(caption_path, caption_values)
    shutil.copy2(caption_path, caption_text_path)
    artifacts = [
        _receipt(caption_path, output_dir, "chinese_caption"),
        _receipt(caption_text_path, output_dir, "chinese_caption_text"),
        _receipt(frame_path, output_dir, "frame_metrics"),
        _receipt(event_path, output_dir, "event_timeline"),
    ]
    if media_output is not None:
        artifacts.insert(0, _receipt(media_output, output_dir, "media"))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        **identity,
        "data_source": data_source,
        "declared_fps": declared_fps,
        "timeline_start_s": frame_times[0] if frame_times else None,
        "timeline_end_s": frame_times[-1] if frame_times else None,
        "frame_row_count": len(frames),
        "event_row_count": len(events),
        "decoded_media": decoded,
        "source_media": (
            {
                "path": str(source),
                "size_bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
            if source is not None
            else None
        ),
        "frame_metrics_schema": list(FRAME_METRICS_FIELDS),
        "event_timeline_schema": list(EVENT_TIMELINE_FIELDS),
        "caption_fields": list(CAPTION_FIELDS),
        "artifacts": artifacts,
        "claims": {
            "media_decode_verified": decoded is not None,
            "quantitative_metrics_available": status == COMPLETE,
            "real_dynamic_performance_claim_allowed": (
                status == COMPLETE and data_source == "REAL"
            ),
            "unavailable_values_are_zero": False,
        },
    }
    manifest_path = output_dir / "video_manifest.json"
    compatibility_manifest_path = output_dir / "evidence_manifest.json"
    _write_json(manifest_path, manifest)
    shutil.copy2(manifest_path, compatibility_manifest_path)
    errors = _validate_video_evidence_package(output_dir, verify_auxiliary=False)
    if errors:
        raise RuntimeError("generated video evidence is invalid: " + "; ".join(errors))
    validation_path = output_dir / "validation.json"
    _write_json(
        validation_path,
        {
            "schema_version": SCHEMA_VERSION,
            "valid": True,
            "errors": [],
            "verified_manifest": _receipt(
                manifest_path, output_dir, "video_manifest"
            ),
        },
    )
    receipt_path = output_dir / "artifact_receipt.json"
    _write_json(
        receipt_path,
        {
            "schema_version": SCHEMA_VERSION,
            "root": ".",
            "artifacts": [
                _receipt(path, output_dir, "package_artifact")
                for path in sorted(output_dir.iterdir())
                if path.is_file() and path != receipt_path
            ],
        },
    )
    errors = _validate_video_evidence_package(output_dir, verify_auxiliary=True)
    if errors:
        raise RuntimeError("generated video evidence is invalid: " + "; ".join(errors))
    return manifest


def _read_csv(
    path: Path, expected_fields: Sequence[str], errors: list[str]
) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != tuple(expected_fields):
                errors.append(f"{path.name} schema mismatch")
            return list(reader)
    except (OSError, csv.Error) as error:
        errors.append(f"unreadable {path.name}: {error}")
        return []


def _validate_video_evidence_package(
    package_dir: Path | str, *, verify_auxiliary: bool
) -> list[str]:
    package_dir = Path(package_dir).resolve()
    errors: list[str] = []
    manifest_path = package_dir / "video_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable video_manifest.json: {error}"]
    compatibility_path = package_dir / "evidence_manifest.json"
    try:
        compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"unreadable evidence_manifest.json: {error}")
    else:
        if compatibility != manifest:
            errors.append("evidence_manifest.json does not match video_manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("evidence manifest schema_version mismatch")
    status = manifest.get("status")
    if status not in {COMPLETE, PENDING_REAL_SIMULATION}:
        errors.append(f"unsupported evidence status: {status}")

    roles: dict[str, Path] = {}
    for receipt in manifest.get("artifacts", []):
        relative = Path(str(receipt.get("path", "")))
        path = package_dir / relative
        role = str(receipt.get("role", ""))
        if not relative.name or relative.is_absolute() or ".." in relative.parts:
            errors.append(f"unsafe evidence artifact path: {relative}")
            continue
        roles[role] = path
        if not path.is_file():
            errors.append(f"missing evidence artifact: {relative}")
            continue
        if path.stat().st_size != receipt.get("size_bytes"):
            errors.append(f"{relative} size mismatch")
        if _sha256(path) != receipt.get("sha256"):
            errors.append(f"{relative} hash mismatch")
    for role in (
        "chinese_caption",
        "chinese_caption_text",
        "frame_metrics",
        "event_timeline",
    ):
        if role not in roles:
            errors.append(f"missing {role} artifact receipt")

    caption_path = package_dir / "caption_zh.md"
    caption: str | None = None
    try:
        caption = caption_path.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"unreadable caption_zh.md: {error}")
    else:
        for field in CAPTION_FIELDS:
            if f"- {field}：" not in caption:
                errors.append(f"caption_zh.md missing fixed field {field}")
    caption_text_path = package_dir / "caption_zh.txt"
    try:
        caption_text = caption_text_path.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"unreadable caption_zh.txt: {error}")
    else:
        if caption is not None and caption_text != caption:
            errors.append("caption_zh.txt does not match caption_zh.md")

    frames = _read_csv(package_dir / "frame_metrics.csv", FRAME_METRICS_FIELDS, errors)
    events = _read_csv(package_dir / "event_timeline.csv", EVENT_TIMELINE_FIELDS, errors)
    identity_fields = ("evidence_uid", "media_uid", "case_uid", "episode_uid")
    for table, rows in (("frame_metrics.csv", frames), ("event_timeline.csv", events)):
        for index, row in enumerate(rows):
            for field in identity_fields:
                if row.get(field) != str(manifest.get(field, "")):
                    errors.append(f"{table} row {index} {field} mismatch")
            if row.get("schema_version") != str(SCHEMA_VERSION):
                errors.append(f"{table} row {index} schema_version mismatch")
    if len(frames) != manifest.get("frame_row_count"):
        errors.append("frame_metrics.csv row count mismatch")
    if len(events) != manifest.get("event_row_count"):
        errors.append("event_timeline.csv row count mismatch")

    frame_times: list[float] = []
    if frames:
        try:
            frame_times = _validate_frame_rows(frames)
        except ValueError as error:
            errors.append(str(error))
        if frame_times:
            if frame_times[0] != manifest.get("timeline_start_s"):
                errors.append("timeline_start_s mismatch")
            if frame_times[-1] != manifest.get("timeline_end_s"):
                errors.append("timeline_end_s mismatch")
    if events and frame_times:
        try:
            _validate_event_rows(events, frame_times)
        except ValueError as error:
            errors.append(str(error))

    decoded_manifest = manifest.get("decoded_media")
    media_path = roles.get("media")
    if status == COMPLETE:
        if media_path is None or not media_path.is_file():
            errors.append("complete evidence has no receipted media")
        else:
            try:
                decoded_actual = _decode_media(media_path)
            except ValueError as error:
                errors.append(str(error))
            else:
                for field in ("media_type", "frame_count", "width_px", "height_px"):
                    if not isinstance(decoded_manifest, Mapping) or (
                        decoded_actual[field] != decoded_manifest.get(field)
                    ):
                        errors.append(f"decoded media {field} mismatch")
                if decoded_actual["frame_count"] != len(frames):
                    errors.append("decoded media and frame_metrics.csv frame count mismatch")
        try:
            _positive(manifest.get("declared_fps"), "declared_fps")
        except ValueError as error:
            errors.append(str(error))
    elif status == PENDING_REAL_SIMULATION:
        if media_path is not None or decoded_manifest is not None:
            errors.append("PENDING_REAL_SIMULATION must not contain media")
        if frames or events:
            errors.append("PENDING_REAL_SIMULATION contains fabricated metrics")
        if manifest.get("data_source") != "UNAVAILABLE":
            errors.append("PENDING_REAL_SIMULATION data_source must be UNAVAILABLE")
        if manifest.get("declared_fps") is not None:
            errors.append("PENDING_REAL_SIMULATION must not declare fps")

    if verify_auxiliary:
        validation_path = package_dir / "validation.json"
        try:
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"unreadable validation.json: {error}")
        else:
            if validation.get("schema_version") != SCHEMA_VERSION:
                errors.append("validation.json schema_version mismatch")
            if validation.get("valid") is not True or validation.get("errors") != []:
                errors.append("validation.json does not record a valid package")
            verified = validation.get("verified_manifest", {})
            if verified.get("path") != "video_manifest.json":
                errors.append("validation.json verified manifest path mismatch")
            if verified.get("size_bytes") != manifest_path.stat().st_size:
                errors.append("verified manifest size mismatch")
            if verified.get("sha256") != _sha256(manifest_path):
                errors.append("verified manifest hash mismatch")

        receipt_path = package_dir / "artifact_receipt.json"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"unreadable artifact_receipt.json: {error}")
        else:
            required = {
                "caption_zh.md",
                "caption_zh.txt",
                "frame_metrics.csv",
                "event_timeline.csv",
                "video_manifest.json",
                "evidence_manifest.json",
                "validation.json",
            }
            if status == COMPLETE and media_path is not None:
                required.add(media_path.name)
            recorded = {
                str(row.get("path", "")) for row in receipt.get("artifacts", [])
            }
            for relative in sorted(required - recorded):
                errors.append(f"artifact_receipt.json missing {relative}")
            for row in receipt.get("artifacts", []):
                relative = Path(str(row.get("path", "")))
                if not relative.name or relative.is_absolute() or ".." in relative.parts:
                    errors.append(f"unsafe package receipt path: {relative}")
                    continue
                path = package_dir / relative
                if not path.is_file():
                    errors.append(f"missing package artifact: {relative}")
                    continue
                if path.stat().st_size != row.get("size_bytes"):
                    errors.append(f"{relative} size mismatch")
                if _sha256(path) != row.get("sha256"):
                    errors.append(f"{relative} hash mismatch")
    return errors


def validate_video_evidence_package(package_dir: Path | str) -> list[str]:
    """Return every provenance, schema, timeline, and decode validation error."""

    return _validate_video_evidence_package(package_dir, verify_auxiliary=True)


__all__ = [
    "CAPTION_FIELDS",
    "EVENT_TIMELINE_FIELDS",
    "FRAME_METRICS_FIELDS",
    "build_video_evidence_package",
    "validate_video_evidence_package",
]
