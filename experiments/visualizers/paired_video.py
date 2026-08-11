from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import cv2
import imageio.v2 as imageio
import numpy as np


PANEL_ORDER = ("raw", "minco-cold", "minco-hot")


@dataclass(frozen=True)
class VideoSource:
    variant: str
    path: Path
    receipt_path: Path
    terminal_status: str = "UNKNOWN"
    terminal_time_s: float | None = None


@dataclass(frozen=True)
class VideoClock:
    fps: float
    frame_count: int
    duration_s: float
    method: str
    error_bound_s: float
    exact_wall_clock: bool
    frame_timestamps_s: tuple[float, ...] = ()


@dataclass(frozen=True)
class PairedVideoReceipt:
    episode_uid: str
    output_path: Path
    fps: float
    frame_count: int
    panel_order: tuple[str, ...]
    sync_method_by_variant: Mapping[str, str]
    sync_error_bound_s_by_variant: Mapping[str, float]


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
        return VideoClock(
            fps=fps,
            frame_count=frame_count,
            duration_s=duration,
            method="RECORDED_FRAME_TIMESTAMPS",
            error_bound_s=error_bound,
            exact_wall_clock=True,
            frame_timestamps_s=timestamps,
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


def _source_frame_index(clock: VideoClock, output_time_s: float) -> int:
    if clock.frame_timestamps_s:
        origin = clock.frame_timestamps_s[0]
        relative = np.asarray(clock.frame_timestamps_s) - origin
        return int(np.argmin(np.abs(relative - output_time_s)))
    return min(
        int(math.floor(output_time_s * clock.fps + 1e-9)),
        clock.frame_count - 1,
    )


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
    return annotated


def render_paired_episode_video(
    sources: Mapping[str, VideoSource],
    output_path: Path | str,
    *,
    episode_uid: str,
) -> PairedVideoReceipt:
    if len(sources) < 2:
        raise ValueError("paired video requires at least two variants")
    ordered = tuple(variant for variant in PANEL_ORDER if variant in sources)
    unsupported = tuple(sorted(set(sources) - set(PANEL_ORDER)))
    ordered += unsupported

    clocks = {
        variant: read_video_clock(source.path, source.receipt_path)
        for variant, source in sources.items()
    }
    output_fps = max(clock.fps for clock in clocks.values())
    output_frames = max(
        1,
        int(
            math.ceil(
                max(clock.duration_s for clock in clocks.values())
                * output_fps
                - 1e-9
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
                source_index = _source_frame_index(clocks[variant], output_time)
                frame = cursors[variant].frame_at(source_index)
                panel = _normalize_panel(frame, panel_width, panel_height)
                panels.append(
                    _annotate_panel(
                        panel, sources[variant], episode_uid, output_time
                    )
                )
            writer.append_data(np.ascontiguousarray(np.concatenate(panels, axis=1)))
    finally:
        writer.close()
        for cursor in cursors.values():
            cursor.close()

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
                "shorter_variants_end_behavior": "FREEZE_LAST_FRAME",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt
