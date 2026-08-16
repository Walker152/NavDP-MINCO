from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import cv2
import imageio.v2 as imageio
import numpy as np

from experiments.recorders.video_recorder import validate_video_receipt
from experiments.visualizers.video_evidence import (
    build_video_evidence_package,
    validate_video_evidence_package,
)


# The first three identifiers preserve historical generic paired-video output.
# The latter three are the formal dynamic study panels and must remain in this
# order: unmodified NavDP, MINCO without SFC, MINCO with native SFC.
PANEL_ORDER = ("raw", "minco-cold", "minco-hot")
FORMAL_DYNAMIC_PANEL_ORDER = (
    "navdp_native",
    "legacy",
    "superplanner_sfc_v1",
)
LIVE_CURVE_FIELDS = (
    ("actual_v_mps", "speed", (80, 210, 255)),
    ("executed_clearance_m", "clearance", (80, 230, 120)),
    ("time_aligned_position_error_m", "tracking", (255, 190, 70)),
)


def _panel_order_for_sources(sources: Mapping[str, "VideoSource"]) -> tuple[str, ...]:
    formal_present = set(sources).intersection(FORMAL_DYNAMIC_PANEL_ORDER)
    expected = FORMAL_DYNAMIC_PANEL_ORDER if formal_present else PANEL_ORDER
    ordered = tuple(variant for variant in expected if variant in sources)
    return ordered + tuple(sorted(set(sources) - set(expected)))


@dataclass(frozen=True)
class VideoSource:
    variant: str
    path: Path
    receipt_path: Path
    terminal_status: str = "UNKNOWN"
    terminal_time_s: float | None = None
    control_samples_path: Path | None = None
    control_episode_uid: str | None = None


@dataclass(frozen=True)
class VideoClock:
    fps: float
    frame_count: int
    duration_s: float
    method: str
    error_bound_s: float
    exact_wall_clock: bool
    frame_timestamps_s: tuple[float, ...] = ()
    clock_domain: str | None = None


@dataclass(frozen=True)
class PairedVideoReceipt:
    episode_uid: str
    output_path: Path
    fps: float
    frame_count: int
    panel_order: tuple[str, ...]
    sync_method_by_variant: Mapping[str, str]
    sync_error_bound_s_by_variant: Mapping[str, float]
    exact_wall_clock_alignment: bool = False
    common_clock_domain: str | None = None
    evidence_package_path: Path | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded_video_truth(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    count = 0
    shape = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current = [int(value) for value in frame.shape]
            if shape is None:
                shape = current
            elif shape != current:
                raise ValueError(f"decoded frame shape changed: {path}")
            count += 1
    finally:
        capture.release()
    if count <= 0 or shape is None:
        raise ValueError(f"video contains no readable frames: {path}")
    return {"frame_count": count, "shape": shape}


def _validate_source_truth(source: VideoSource, clock: VideoClock) -> None:
    payload = json.loads(source.receipt_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 1)) >= 2:
        errors = validate_video_receipt(source.path, source.receipt_path)
        if errors:
            raise ValueError(
                f"source video receipt mismatch ({source.variant}): "
                + "; ".join(errors)
            )
    decoded = _decoded_video_truth(source.path)
    if decoded["frame_count"] != clock.frame_count:
        raise ValueError(
            f"source decoded frame_count does not match receipt "
            f"({source.variant}): {decoded['frame_count']} != {clock.frame_count}"
        )
    declared_shape = payload.get("decoded_shape", payload.get("shape"))
    if declared_shape is not None and list(declared_shape) != decoded["shape"]:
        raise ValueError(
            f"source decoded shape does not match receipt ({source.variant}): "
            f"{decoded['shape']} != {declared_shape}"
        )


def _finite_positive(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def read_video_clock(
    video_path: Path | str,
    receipt_path: Path | str,
) -> VideoClock:
    video_path = Path(video_path)
    receipt_path = Path(receipt_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video does not exist: {video_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable video receipt {receipt_path}: {error}") from error

    fps = _finite_positive(receipt.get("fps"), "video fps")
    try:
        frame_count = int(receipt["frame_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("video frame_count must be a positive integer") from error
    if frame_count <= 0:
        raise ValueError("video frame_count must be a positive integer")

    raw_timestamps = receipt.get("frame_timestamps_s")
    if raw_timestamps is not None:
        timestamps = tuple(float(value) for value in raw_timestamps)
        if (
            len(timestamps) != frame_count
            or not all(math.isfinite(value) for value in timestamps)
            or any(right <= left for left, right in zip(timestamps, timestamps[1:]))
        ):
            raise ValueError("frame_timestamps_s must be finite and strictly increasing")
        gaps = [
            right - left for left, right in zip(timestamps, timestamps[1:])
        ]
        error_bound = 0.5 * max(gaps, default=1.0 / fps)
        duration = timestamps[-1] - timestamps[0] + 1.0 / fps
        clock_domain = str(
            receipt.get("frame_timestamp_clock_domain", "")
        ).strip() or None
        absolute_epoch = (
            receipt.get("frame_timestamps_are_absolute_epoch") is True
            and clock_domain is not None
            and all(value >= 0.0 for value in timestamps)
        )
        return VideoClock(
            fps=fps,
            frame_count=frame_count,
            duration_s=duration,
            method=(
                "RECORDED_ABSOLUTE_TIMESTAMPS"
                if absolute_epoch
                else "RECORDED_RELATIVE_TIMESTAMPS"
            ),
            error_bound_s=error_bound,
            exact_wall_clock=absolute_epoch,
            frame_timestamps_s=timestamps,
            clock_domain=clock_domain,
        )

    return VideoClock(
        fps=fps,
        frame_count=frame_count,
        duration_s=frame_count / fps,
        method="FIXED_FPS_RECONSTRUCTION",
        error_bound_s=1.0 / fps,
        exact_wall_clock=False,
    )


class _FrameCursor:
    def __init__(self, source: VideoSource, clock: VideoClock):
        self.source = source
        self.clock = clock
        self.capture = cv2.VideoCapture(str(source.path))
        if not self.capture.isOpened():
            raise ValueError(f"cannot open video: {source.path}")
        self.index = -1
        self.frame: np.ndarray | None = None

    def frame_at(self, target_index: int) -> np.ndarray:
        target_index = min(max(0, target_index), self.clock.frame_count - 1)
        while self.index < target_index:
            next_index = self.index + 1
            ok, frame_bgr = self.capture.read()
            if not ok:
                if self.frame is None:
                    raise ValueError(
                        f"video ended before receipt: {self.source.path}"
                    )
                self.index = target_index
                break
            self.frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.index = next_index
        if self.frame is None:
            raise ValueError(f"video contains no readable frames: {self.source.path}")
        return self.frame

    def close(self) -> None:
        self.capture.release()


def _source_frame_index(
    clock: VideoClock,
    output_time_s: float,
    *,
    timeline_origin_s: float | None = None,
) -> int:
    if clock.frame_timestamps_s:
        target = output_time_s
        if timeline_origin_s is not None:
            target += timeline_origin_s
            timestamps = np.asarray(clock.frame_timestamps_s)
        else:
            timestamps = (
                np.asarray(clock.frame_timestamps_s)
                - clock.frame_timestamps_s[0]
            )
        return int(np.argmin(np.abs(timestamps - target)))
    return min(
        int(math.floor(output_time_s * clock.fps + 1e-9)),
        clock.frame_count - 1,
    )


def _last_native_time_s(clock: VideoClock) -> float:
    if clock.frame_timestamps_s:
        return clock.frame_timestamps_s[-1] - clock.frame_timestamps_s[0]
    return (clock.frame_count - 1) / clock.fps


def _optional_control_value(row: Mapping[str, str], field: str) -> object:
    value = str(row.get(field, "")).strip()
    if not value:
        return ""
    try:
        numeric = float(value)
    except ValueError:
        return value
    return numeric if math.isfinite(numeric) else ""


def _load_control_rows(
    source: VideoSource,
    episode_uid: str,
) -> tuple[dict[str, str], ...]:
    path = source.control_samples_path
    if path is None:
        return ()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"control_samples.csv does not exist: {path}")
    expected_uid = source.control_episode_uid or episode_uid
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row.get("episode_uid") == expected_uid
        ]
    timed = []
    for row in rows:
        try:
            timestamp = float(row.get("timestamp_monotonic_s", ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(timestamp):
            timed.append((timestamp, row))
    timed.sort(key=lambda item: item[0])
    if any(right[0] <= left[0] for left, right in zip(timed, timed[1:])):
        raise ValueError(
            f"control timestamps must be strictly increasing: {source.variant}"
        )
    return tuple(row for _, row in timed)


def _control_time_axis(rows: tuple[dict[str, str], ...]) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=np.float64)
    absolute = np.asarray(
        [float(row["timestamp_monotonic_s"]) for row in rows],
        dtype=np.float64,
    )
    return absolute - absolute[0]


def _aligned_control_fields(
    row: Mapping[str, str],
    *,
    source_index: int,
    source_time_s: float,
    alignment_error_s: float,
    variant: str,
) -> dict[str, object]:
    try:
        recorded_index = int(row.get("frame_idx", source_index))
    except (TypeError, ValueError):
        recorded_index = source_index
    result = {
        "source_sample_index": recorded_index,
        "source_time_s": source_time_s,
        "x_m": _optional_control_value(row, "robot_x_m"),
        "y_m": _optional_control_value(row, "robot_y_m"),
        "yaw_rad": _optional_control_value(row, "robot_yaw_rad"),
        "clearance_m": _optional_control_value(row, "executed_clearance_m"),
        "speed_mps": _optional_control_value(row, "actual_v_mps"),
        "yaw_rate_rps": _optional_control_value(row, "actual_w_radps"),
        "tracking_error_m": _optional_control_value(
            row, "time_aligned_position_error_m"
        ),
        "command_linear_mps": _optional_control_value(row, "cmd_v_mps"),
        "command_angular_rps": _optional_control_value(row, "cmd_w_radps"),
        "optimizer_state": str(row.get("mpc_solver_status", "")).strip(),
        "control_state": str(row.get("control_state", "")).strip(),
        "planning_state": str(row.get("planning_state", "")).strip(),
        "data_availability": (
            f"ALIGNED_CONTROL_SAMPLE:{variant};"
            f"NEAREST_RELATIVE_TIME_ERROR_S={alignment_error_s:.9g}"
        ),
    }
    return result


def _evidence_frame_rows(
    ordered: tuple[str, ...],
    sources: Mapping[str, VideoSource],
    clocks: Mapping[str, VideoClock],
    *,
    output_frames: int,
    output_fps: float,
    timeline_origin_s: float | None,
    control_rows_by_variant: Mapping[str, tuple[dict[str, str], ...]],
) -> list[dict[str, object]]:
    metric_variant = next(
        (variant for variant in ordered if control_rows_by_variant.get(variant)),
        None,
    )
    metric_controls = (
        control_rows_by_variant[metric_variant] if metric_variant else ()
    )
    metric_times = _control_time_axis(metric_controls)
    rows = []
    for output_index in range(output_frames):
        output_time = output_index / output_fps
        mappings = []
        for variant in ordered:
            source_index = _source_frame_index(
                clocks[variant],
                output_time,
                timeline_origin_s=timeline_origin_s,
            )
            frozen = output_time > _last_native_time_s(clocks[variant]) + 1e-9
            mappings.append(
                f"{variant}:{source_index}:{'FROZEN' if frozen else 'LIVE'}"
            )
        row = {
                "frame_index": output_index,
                "time_s": output_time,
                "event_tags": "|".join(mappings),
                "data_availability": "VIDEO_CLOCK_ONLY;CONTROL_SAMPLES_UNAVAILABLE",
        }
        if metric_variant is not None:
            control_index = int(np.argmin(np.abs(metric_times - output_time)))
            row.update(
                _aligned_control_fields(
                    metric_controls[control_index],
                    source_index=control_index,
                    source_time_s=float(metric_times[control_index]),
                    alignment_error_s=abs(
                        float(metric_times[control_index]) - output_time
                    ),
                    variant=metric_variant,
                )
            )
            source = sources[metric_variant]
            terminal_time = source.terminal_time_s
            if terminal_time is not None and math.isfinite(terminal_time):
                terminal_index = min(
                    output_frames - 1,
                    max(0, int(round(terminal_time * output_fps))),
                )
                if output_index == terminal_index:
                    status = str(source.terminal_status).upper()
                    if status == "COLLISION":
                        row["collision"] = True
                    if status == "GOAL_REACHED":
                        row["goal_reached"] = True
        rows.append(row)
    return rows


def _evidence_event_rows(
    ordered: tuple[str, ...],
    sources: Mapping[str, VideoSource],
    clocks: Mapping[str, VideoClock],
    *,
    output_frames: int,
    output_fps: float,
    control_rows_by_variant: Mapping[str, tuple[dict[str, str], ...]],
) -> list[dict[str, object]]:
    last_output_index = output_frames - 1
    last_output_time = last_output_index / output_fps
    rows = []
    for variant in ordered:
        clock = clocks[variant]
        source = sources[variant]
        native_end = _last_native_time_s(clock)
        first_frozen = int(math.floor(native_end * output_fps + 1e-9)) + 1
        if first_frozen <= last_output_index:
            rows.append(
                {
                    "event_uid": f"{variant}-freeze-last-frame",
                    "event_type": "FREEZE_LAST_FRAME",
                    "start_frame_index": first_frozen,
                    "end_frame_index": last_output_index,
                    "start_time_s": first_frozen / output_fps,
                    "end_time_s": last_output_time,
                    "severity": "INFO",
                    "source_uid": variant,
                    "metric_name": "synchronization_error_bound",
                    "metric_value": clock.error_bound_s,
                    "metric_unit": "s",
                    "description_zh": (
                        f"{variant} 原视频结束后保留末帧；该区间不表示机器人继续运动。"
                    ),
                    "data_availability": "DERIVED_FROM_VIDEO_CLOCK",
                }
            )
        terminal_time = source.terminal_time_s
        if terminal_time is None or not math.isfinite(terminal_time):
            terminal_time = min(native_end, last_output_time)
            availability = "TERMINAL_TIME_UNAVAILABLE"
        else:
            terminal_time = min(max(0.0, terminal_time), last_output_time)
            availability = "RECORDED_OR_CLAMPED_TO_MEDIA"
        terminal_index = min(
            last_output_index,
            max(0, int(round(terminal_time * output_fps))),
        )
        terminal_time = terminal_index / output_fps
        terminal_status = str(source.terminal_status or "UNKNOWN").upper()
        terminal_event_type = {
            "COLLISION": "COLLISION",
            "GOAL_REACHED": "GOAL_REACHED",
        }.get(terminal_status, "TERMINATION")
        rows.append(
            {
                "event_uid": f"{variant}-terminal-status",
                "event_type": terminal_event_type,
                "start_frame_index": terminal_index,
                "end_frame_index": terminal_index,
                "start_time_s": terminal_time,
                "end_time_s": terminal_time,
                "severity": "INFO",
                "source_uid": variant,
                "description_zh": (
                    f"{variant} 终止状态：{source.terminal_status or 'UNKNOWN'}。"
                ),
                "data_availability": availability,
            }
        )
        controls = control_rows_by_variant.get(variant, ())
        times = _control_time_axis(controls)
        previous_state = None
        for control_index, (control, control_time) in enumerate(
            zip(controls, times)
        ):
            frame_index = min(
                last_output_index,
                max(0, int(round(float(control_time) * output_fps))),
            )
            event_time = frame_index / output_fps
            state = str(control.get("control_state", "")).strip()
            if state and previous_state is not None and state != previous_state:
                rows.append(
                    {
                        "event_uid": f"{variant}-control-state-{control_index}",
                        "event_type": "CONTROL_STATE_CHANGE",
                        "start_frame_index": frame_index,
                        "end_frame_index": frame_index,
                        "start_time_s": event_time,
                        "end_time_s": event_time,
                        "severity": "INFO",
                        "source_uid": variant,
                        "description_zh": f"{variant} 控制状态由 {previous_state} 变为 {state}。",
                        "data_availability": "RECORDED_CONTROL_SAMPLE",
                    }
                )
            if state:
                previous_state = state
            if str(control.get("wheel_saturated", "")).strip().lower() in {
                "1", "true", "yes",
            }:
                rows.append(
                    {
                        "event_uid": f"{variant}-wheel-saturation-{control_index}",
                        "event_type": "WHEEL_SATURATION",
                        "start_frame_index": frame_index,
                        "end_frame_index": frame_index,
                        "start_time_s": event_time,
                        "end_time_s": event_time,
                        "severity": "WARNING",
                        "source_uid": variant,
                        "description_zh": f"{variant} 记录到轮速饱和。",
                        "data_availability": "RECORDED_CONTROL_SAMPLE",
                    }
                )
    return rows


def _evidence_package_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_evidence")


def _normalize_panel(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"invalid video frame shape: {frame.shape}")
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized_width = max(1, int(round(frame.shape[1] * scale)))
    resized_height = max(1, int(round(frame.shape[0] * scale)))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - resized_width) // 2
    y0 = (height - resized_height) // 2
    panel[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return panel


def _annotate_panel(
    panel: np.ndarray,
    source: VideoSource,
    episode_uid: str,
    output_time_s: float,
    control_row: Mapping[str, str] | None = None,
) -> np.ndarray:
    annotated = panel.copy()
    terminal = source.terminal_status or "UNKNOWN"
    terminal_time = (
        f"{source.terminal_time_s:.2f}s"
        if source.terminal_time_s is not None
        and math.isfinite(source.terminal_time_s)
        else "N/A"
    )
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 18), (0, 0, 0), -1)
    label = (
        f"{source.variant} {episode_uid} t={output_time_s:.2f}s "
        f"{terminal}@{terminal_time}"
        + (" state=N/A" if control_row is None else "")
    )
    cv2.putText(
        annotated,
        label,
        (2, 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.28,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    # The captured Isaac frame can contain a richer native overlay, but the
    # comparison compositor must not make that a precondition.  Draw a small,
    # panel-local box from the *same variant's* recorded control row so that a
    # viewer can read state even after three videos are resized side-by-side.
    if control_row is None:
        # Preserve the source image below the compact title when no physical
        # control trace exists.  The title explicitly marks the state as N/A;
        # drawing an invented black dashboard would hide evidence instead.
        return annotated
    else:
        def value(name: str, unit: str = "") -> str:
            raw = _optional_control_value(control_row, name)
            return "N/A" if raw == "" else f"{raw}{unit}"

        yaw_value = _optional_control_value(control_row, "robot_yaw_rad")
        if isinstance(yaw_value, (int, float)):
            yaw_text = f"{float(yaw_value):.2f}rad/{math.degrees(float(yaw_value)):.1f}deg"
        else:
            yaw_text = "N/A"
        lines = (
            "xy=(%s,%s) yaw=%s" % (
                value("robot_x_m"), value("robot_y_m"), yaw_text
            ),
            "v=%s w=%s | cmd=(%s,%s)" % (
                value("actual_v_mps"), value("actual_w_radps"),
                value("cmd_v_mps"), value("cmd_w_radps"),
            ),
            "clear=%s ctrl=%s plan=%s" % (
                value("executed_clearance_m"),
                str(control_row.get("control_state", "")).strip() or "N/A",
                str(control_row.get("planning_state", "")).strip() or "N/A",
            ),
        )
    line_height = max(10, int(round(annotated.shape[0] * 0.032)))
    box_bottom = min(annotated.shape[0] - 1, 20 + line_height * len(lines) + 4)
    cv2.rectangle(annotated, (0, 19), (annotated.shape[1] - 1, box_bottom), (0, 0, 0), -1)
    for line_index, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (2, 20 + line_height * (line_index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.23, min(0.38, annotated.shape[1] / 1500.0)),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _live_curve_strip(
    ordered: tuple[str, ...],
    control_rows_by_variant: Mapping[str, tuple[dict[str, str], ...]],
    control_times_by_variant: Mapping[str, np.ndarray],
    *,
    output_time_s: float,
    output_duration_s: float,
    panel_width: int,
    height: int,
) -> np.ndarray:
    """Render recorded per-method speed/clearance/tracking histories."""
    strip = np.full((height, panel_width * len(ordered), 3), 18, dtype=np.uint8)
    row_height = max(18, height // len(LIVE_CURVE_FIELDS))
    for panel_index, variant in enumerate(ordered):
        x0 = panel_index * panel_width
        x1 = x0 + panel_width - 1
        rows = control_rows_by_variant.get(variant, ())
        times = control_times_by_variant.get(variant, np.asarray([], dtype=np.float64))
        for field_index, (field, label, colour) in enumerate(LIVE_CURVE_FIELDS):
            y0 = field_index * row_height
            y1 = min(height - 1, y0 + row_height - 1)
            cv2.rectangle(strip, (x0, y0), (x1, y1), (32, 32, 32), 1)
            cv2.putText(
                strip, label, (x0 + 3, min(y1 - 2, y0 + 11)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, colour, 1, cv2.LINE_AA,
            )
            numeric = []
            for row_index, row in enumerate(rows):
                value = _optional_control_value(row, field)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    numeric.append((float(times[row_index]), float(value)))
            if not numeric:
                cv2.putText(
                    strip, "N/A", (max(x0 + 3, x1 - 28), min(y1 - 2, y0 + 11)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.27, (170, 170, 170), 1, cv2.LINE_AA,
                )
                continue
            values = [value for _, value in numeric]
            lower = min(0.0, min(values))
            upper = max(values)
            if upper - lower < 1e-9:
                upper = lower + 1.0
            points = []
            for timestamp, value in numeric:
                if timestamp > output_time_s + 1e-9:
                    break
                x = x0 + 2 + int(round(
                    max(0.0, min(1.0, timestamp / max(1e-9, output_duration_s))) *
                    max(1, panel_width - 5)
                ))
                normalized = (value - lower) / (upper - lower)
                y = y1 - 3 - int(round(normalized * max(1, y1 - y0 - 15)))
                points.append((x, y))
            if len(points) >= 2:
                cv2.polylines(
                    strip, [np.asarray(points, dtype=np.int32)], False, colour, 1, cv2.LINE_AA,
                )
            elif points:
                cv2.circle(strip, points[0], 1, colour, -1)
        cv2.putText(
            strip, variant, (x0 + 3, height - 3), cv2.FONT_HERSHEY_SIMPLEX,
            0.27, (235, 235, 235), 1, cv2.LINE_AA,
        )
    return strip


def render_paired_episode_video(
    sources: Mapping[str, VideoSource],
    output_path: Path | str,
    *,
    episode_uid: str,
    data_source: str = "UNKNOWN",
) -> PairedVideoReceipt:
    if len(sources) < 2:
        raise ValueError("paired video requires at least two variants")
    ordered = _panel_order_for_sources(sources)

    clocks = {
        variant: read_video_clock(source.path, source.receipt_path)
        for variant, source in sources.items()
    }
    for variant, source in sources.items():
        _validate_source_truth(source, clocks[variant])
    output_fps = max(clock.fps for clock in clocks.values())
    clock_domains = {clock.clock_domain for clock in clocks.values()}
    exact_wall_clock_alignment = (
        all(clock.exact_wall_clock for clock in clocks.values())
        and len(clock_domains) == 1
        and None not in clock_domains
    )
    timeline_origin_s = (
        min(clock.frame_timestamps_s[0] for clock in clocks.values())
        if exact_wall_clock_alignment
        else None
    )
    if exact_wall_clock_alignment:
        output_duration_s = max(
            clock.frame_timestamps_s[-1] + 1.0 / clock.fps
            for clock in clocks.values()
        ) - float(timeline_origin_s)
    else:
        output_duration_s = max(clock.duration_s for clock in clocks.values())
    output_frames = max(
        1,
        int(
            math.ceil(
                output_duration_s * output_fps - 1e-9
            )
        ),
    )
    cursors = {
        variant: _FrameCursor(sources[variant], clocks[variant])
        for variant in ordered
    }
    first_frames = {
        variant: cursor.frame_at(0) for variant, cursor in cursors.items()
    }
    panel_height = max(frame.shape[0] for frame in first_frames.values())
    panel_width = max(frame.shape[1] for frame in first_frames.values())
    curve_height = max(96, panel_height // 3)
    control_rows_by_variant = {
        variant: _load_control_rows(sources[variant], episode_uid)
        for variant in ordered
    }
    control_times_by_variant = {
        variant: _control_time_axis(control_rows_by_variant[variant])
        for variant in ordered
    }

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output_path,
        fps=output_fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=16,
        output_params=["-crf", "18", "-movflags", "faststart"],
    )
    try:
        for output_index in range(output_frames):
            output_time = output_index / output_fps
            panels = []
            for variant in ordered:
                source_index = _source_frame_index(
                    clocks[variant],
                    output_time,
                    timeline_origin_s=timeline_origin_s,
                )
                frame = cursors[variant].frame_at(source_index)
                panel = _normalize_panel(frame, panel_width, panel_height)
                control_rows = control_rows_by_variant[variant]
                control_row = None
                if control_rows:
                    control_index = int(
                        np.argmin(
                            np.abs(control_times_by_variant[variant] - output_time)
                        )
                    )
                    control_row = control_rows[control_index]
                panels.append(
                    _annotate_panel(
                        panel, sources[variant], episode_uid, output_time,
                        control_row,
                    )
                )
            panel_row = np.concatenate(panels, axis=1)
            curves = _live_curve_strip(
                ordered,
                control_rows_by_variant,
                control_times_by_variant,
                output_time_s=output_time,
                output_duration_s=output_duration_s,
                panel_width=panel_width,
                height=curve_height,
            )
            writer.append_data(np.ascontiguousarray(np.concatenate((panel_row, curves), axis=0)))
    finally:
        writer.close()
        for cursor in cursors.values():
            cursor.close()

    evidence_path = _evidence_package_path(output_path)
    metric_variant = next(
        (variant for variant in ordered if control_rows_by_variant[variant]),
        None,
    )
    metric_control_times = (
        _control_time_axis(control_rows_by_variant[metric_variant])
        if metric_variant
        else np.asarray([], dtype=np.float64)
    )
    control_alignment_error = (
        max(
            float(np.min(np.abs(metric_control_times - index / output_fps)))
            for index in range(output_frames)
        )
        if metric_variant is not None
        else None
    )
    missing_controls = [
        variant for variant in ordered if not control_rows_by_variant[variant]
    ]
    frame_rows = _evidence_frame_rows(
        ordered,
        sources,
        clocks,
        output_frames=output_frames,
        output_fps=output_fps,
        timeline_origin_s=timeline_origin_s,
        control_rows_by_variant=control_rows_by_variant,
    )
    event_rows = _evidence_event_rows(
        ordered,
        sources,
        clocks,
        output_frames=output_frames,
        output_fps=output_fps,
        control_rows_by_variant=control_rows_by_variant,
    )
    expected_variants = (
        FORMAL_DYNAMIC_PANEL_ORDER
        if set(sources).intersection(FORMAL_DYNAMIC_PANEL_ORDER)
        else PANEL_ORDER
    )
    missing_variants = [variant for variant in expected_variants if variant not in sources]
    method_summary = "；".join(
        f"{variant}={clocks[variant].method}, 误差界≤{clocks[variant].error_bound_s:.6g}s"
        for variant in ordered
    )
    build_video_evidence_package(
        output_path,
        evidence_path,
        evidence_uid=f"{episode_uid}-paired-video-evidence",
        media_uid=f"{episode_uid}-paired-video",
        data_source=data_source,
        fps=output_fps,
        episode_uid=episode_uid,
        frame_rows=frame_rows,
        event_rows=event_rows,
        caption_overrides={
            "scene_zh": f"共享 episode_uid={episode_uid} 的并排闭环回放",
            "method_zh": " / ".join(ordered),
            "metrics_units_zh": (
                "逐帧数值字段锚定方法="
                + (metric_variant or "无可用控制样本")
                + "；位置/间隙/误差：m；时间：s；速度：m/s；角量：rad"
            ),
            "time_basis_zh": (
                "共享绝对墙钟时间轴"
                if exact_wall_clock_alignment
                else "相对媒体时间轴；不声称精确墙钟同步"
            ),
            "synchronization_zh": (
                f"同步方法与同步误差：{method_summary}；"
                "较短视频采用末帧冻结，冻结段不代表继续运动；"
                + (
                    f"逐帧数值以 {metric_variant} 控制样本首时刻为相对原点做最近邻对齐，"
                    f"最大对齐误差≤{control_alignment_error:.6g}s"
                    if metric_variant is not None
                    else "无控制样本可供逐帧数值对齐"
                )
            ),
            "missing_data_zh": (
                "缺失证据/面板："
                + ("、".join(missing_variants) if missing_variants else "无")
                + "；缺失控制样本："
                + ("、".join(missing_controls) if missing_controls else "无")
                + "；未记录指标保持为空"
            ),
            "failure_handling_zh": (
                "各方法终止状态独立标注；末帧冻结事件显式记录于事件表"
            ),
            "evidence_boundary_zh": (
                "仅证明已录制图像、声明时间轴及终止状态；不替代碰撞、ESDF或因果证据"
            ),
            "conclusion_limit_zh": (
                "非共享绝对时钟的面板只能作误差界内的相对时序比较"
            ),
        },
    )

    receipt = PairedVideoReceipt(
        episode_uid=episode_uid,
        output_path=output_path,
        fps=output_fps,
        frame_count=output_frames,
        panel_order=ordered,
        sync_method_by_variant=MappingProxyType(
            {variant: clocks[variant].method for variant in ordered}
        ),
        sync_error_bound_s_by_variant=MappingProxyType(
            {variant: clocks[variant].error_bound_s for variant in ordered}
        ),
        exact_wall_clock_alignment=exact_wall_clock_alignment,
        common_clock_domain=(
            next(iter(clock_domains)) if exact_wall_clock_alignment else None
        ),
        evidence_package_path=evidence_path,
    )
    sidecar = output_path.with_suffix(".comparison.json")
    sidecar.write_text(
        json.dumps(
            {
                "episode_uid": receipt.episode_uid,
                "output_path": output_path.name,
                "fps": receipt.fps,
                "frame_count": receipt.frame_count,
                "panel_order": list(receipt.panel_order),
                "sync_method_by_variant": dict(
                    receipt.sync_method_by_variant
                ),
                "sync_error_bound_s_by_variant": dict(
                    receipt.sync_error_bound_s_by_variant
                ),
                "exact_wall_clock_alignment": (
                    receipt.exact_wall_clock_alignment
                ),
                "common_clock_domain": receipt.common_clock_domain,
                "shorter_variants_end_behavior": "FREEZE_LAST_FRAME",
                "per_panel_live_state_overlay": True,
                "per_panel_live_curves": True,
                "live_curve_fields": [field for field, _, _ in LIVE_CURVE_FIELDS],
                "live_state_fields": [
                    "x_m", "y_m", "yaw_rad", "speed_mps", "yaw_rate_rps",
                    "command_linear_mps", "command_angular_rps", "clearance_m",
                    "control_state", "planning_state",
                ],
                "video_size_bytes": output_path.stat().st_size,
                "video_sha256": _sha256(output_path),
                "evidence_package": evidence_path.name,
                "source_receipts": {
                    variant: {
                        "video_path": str(sources[variant].path.resolve()),
                        "video_sha256": _sha256(sources[variant].path),
                        "receipt_path": str(
                            sources[variant].receipt_path.resolve()
                        ),
                        "receipt_sha256": _sha256(
                            sources[variant].receipt_path
                        ),
                        **(
                            {
                                "control_samples_path": str(
                                    Path(
                                        sources[variant].control_samples_path
                                    ).resolve()
                                ),
                                "control_samples_sha256": _sha256(
                                    Path(
                                        sources[variant].control_samples_path
                                    )
                                ),
                            }
                            if sources[variant].control_samples_path is not None
                            else {}
                        ),
                    }
                    for variant in ordered
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt


def validate_paired_video_bundle(output_path: Path | str) -> list[str]:
    output_path = Path(output_path).resolve()
    errors = []
    sidecar = output_path.with_suffix(".comparison.json")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable paired video sidecar: {error}"]
    if not output_path.is_file():
        errors.append(f"missing paired video: {output_path}")
    else:
        if payload.get("video_size_bytes") != output_path.stat().st_size:
            errors.append("paired video size mismatch")
        if payload.get("video_sha256") != _sha256(output_path):
            errors.append("paired video hash mismatch")
        try:
            decoded = _decoded_video_truth(output_path)
        except ValueError as error:
            errors.append(str(error))
        else:
            if decoded["frame_count"] != payload.get("frame_count"):
                errors.append("paired video decoded frame_count mismatch")
    evidence_name = payload.get("evidence_package")
    evidence_path = output_path.parent / str(evidence_name or "")
    if not evidence_name:
        errors.append("paired video sidecar missing evidence_package")
    else:
        errors.extend(validate_video_evidence_package(evidence_path))
    for variant, source in payload.get("source_receipts", {}).items():
        for kind in ("video", "receipt"):
            path = Path(str(source.get(f"{kind}_path", "")))
            if not path.is_file():
                errors.append(f"missing {variant} source {kind}")
            elif _sha256(path) != source.get(f"{kind}_sha256"):
                errors.append(f"{variant} source {kind} hash mismatch")
        control_path_text = source.get("control_samples_path")
        if control_path_text:
            control_path = Path(str(control_path_text))
            if not control_path.is_file():
                errors.append(f"missing {variant} source control_samples")
            elif _sha256(control_path) != source.get("control_samples_sha256"):
                errors.append(f"{variant} source control_samples hash mismatch")
    return errors


__all__ = [
    "PairedVideoReceipt",
    "VideoClock",
    "VideoSource",
    "read_video_clock",
    "render_paired_episode_video",
    "validate_paired_video_bundle",
]
