from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import imageio.v2 as imageio
import numpy as np


CLOCK_DOMAIN = "episode_relative_monotonic"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_video(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode video: {path}")
    count = 0
    shape = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current = [int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2])]
            if shape is None:
                shape = current
            elif current != shape:
                raise ValueError(f"decoded video shape changed: {current} != {shape}")
            count += 1
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if count <= 0 or shape is None:
        raise ValueError(f"video contains no decodable frames: {path}")
    return {
        "decoded_frame_count": count,
        "decoded_shape": shape,
        "decoded_fps": fps if math.isfinite(fps) and fps > 0.0 else None,
    }


def validate_video_receipt(
    video_path: Path | str,
    receipt_path: Path | str,
) -> list[str]:
    video_path = Path(video_path)
    receipt_path = Path(receipt_path)
    errors = []
    if not video_path.is_file():
        return [f"missing video: {video_path}"]
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable video receipt: {error}"]
    if payload.get("complete") is not True:
        errors.append("video receipt is not complete")
    if payload.get("video_size_bytes") != video_path.stat().st_size:
        errors.append("video_size_bytes mismatch")
    if payload.get("video_sha256") != _sha256(video_path):
        errors.append("video_sha256 mismatch")
    try:
        decoded = _decode_video(video_path)
    except ValueError as error:
        errors.append(str(error))
        return errors
    for field in ("decoded_frame_count", "decoded_shape"):
        if payload.get(field) != decoded[field]:
            errors.append(f"{field} mismatch")
    try:
        declared_count = int(payload.get("frame_count", 0))
    except (TypeError, ValueError):
        declared_count = 0
    if declared_count != decoded["decoded_frame_count"]:
        errors.append("frame_count does not match decoded frame count")
    timestamps = payload.get("frame_timestamps_s")
    if not isinstance(timestamps, list) or len(timestamps) != declared_count:
        errors.append("frame_timestamps_s count mismatch")
    else:
        try:
            values = [float(value) for value in timestamps]
        except (TypeError, ValueError):
            errors.append("frame_timestamps_s must be numeric")
        else:
            if (
                not all(math.isfinite(value) and value >= 0.0 for value in values)
                or any(right <= left for left, right in zip(values, values[1:]))
            ):
                errors.append("frame_timestamps_s must be finite and strictly increasing")
    if payload.get("frame_timestamp_clock_domain") != CLOCK_DOMAIN:
        errors.append("frame timestamp clock domain mismatch")
    if payload.get("frame_timestamps_are_absolute_epoch") is not False:
        errors.append("relative monotonic timestamps must not claim absolute epoch")
    return errors


class EpisodeVideoRecorder:
    def __init__(self, output_dir, fps=10, crf=23, scale=1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.crf = crf
        self.scale = scale
        self.writer = None
        self.uid = None
        self.shape = None
        self.frame_count = 0
        self._episode_origin_monotonic_s = None
        self._frame_timestamps_s: list[float] = []

    def start_episode(
        self,
        episode_uid,
        expected_shape=None,
        *,
        timestamp_monotonic_s=None,
    ):
        if self.writer is not None:
            self.end_episode()
        origin = (
            time.monotonic()
            if timestamp_monotonic_s is None
            else float(timestamp_monotonic_s)
        )
        if not math.isfinite(origin):
            raise ValueError("episode monotonic origin must be finite")
        self.uid = str(episode_uid)
        self.shape = tuple(expected_shape) if expected_shape else None
        self.frame_count = 0
        self._episode_origin_monotonic_s = origin
        self._frame_timestamps_s = []
        path = self.output_dir / f"{self.uid}.mp4"
        self.writer = imageio.get_writer(
            path,
            fps=self.fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=16,
            output_params=["-crf", str(self.crf), "-movflags", "faststart"],
        )
        return path

    def write(self, frame, *, timestamp_monotonic_s=None):
        if self.writer is None:
            raise RuntimeError("start_episode must be called first")
        timestamp = (
            time.monotonic()
            if timestamp_monotonic_s is None
            else float(timestamp_monotonic_s)
        )
        if not math.isfinite(timestamp):
            raise ValueError("frame timestamp must be finite")
        relative = timestamp - float(self._episode_origin_monotonic_s)
        if relative < 0.0:
            raise ValueError("frame timestamp precedes episode monotonic origin")
        relative = round(relative, 9)
        if self._frame_timestamps_s and relative <= self._frame_timestamps_s[-1]:
            raise ValueError("frame timestamps must be strictly increasing")
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"invalid frame shape: {frame.shape}")
        if self.scale != 1.0:
            if not np.isfinite(self.scale) or self.scale <= 0.0:
                raise ValueError("video scale must be finite and positive")
            frame = cv2.resize(
                frame,
                None,
                fx=float(self.scale),
                fy=float(self.scale),
                interpolation=cv2.INTER_AREA,
            )
        if self.shape is None:
            self.shape = tuple(frame.shape)
        if tuple(frame.shape) != self.shape:
            raise ValueError(f"frame shape changed: {frame.shape} != {self.shape}")
        self.writer.append_data(np.ascontiguousarray(frame))
        self._frame_timestamps_s.append(relative)
        self.frame_count += 1

    def end_episode(self):
        if self.writer is None:
            return None
        path = self.output_dir / f"{self.uid}.mp4"
        self.writer.close()
        self.writer = None
        if self.frame_count == 0:
            path.unlink(missing_ok=True)
            result = None
            decoded = {
                "decoded_frame_count": 0,
                "decoded_shape": None,
                "decoded_fps": None,
            }
        else:
            result = path
            decoded = _decode_video(path)
        metadata = {
            "schema_version": 2,
            "episode_uid": self.uid,
            "fps": self.fps,
            "crf": self.crf,
            "scale": self.scale,
            "frame_count": self.frame_count,
            "shape": list(self.shape) if self.shape else None,
            "codec": "libx264",
            "pixel_format": "yuv420p",
            "macro_block_size": 16,
            "complete": self.frame_count > 0,
            "frame_timestamps_s": list(self._frame_timestamps_s),
            "frame_timestamp_clock_domain": CLOCK_DOMAIN,
            "frame_timestamps_are_absolute_epoch": False,
            **decoded,
            "video_size_bytes": path.stat().st_size if result else 0,
            "video_sha256": _sha256(path) if result else None,
        }
        receipt_path = self.output_dir / f"{self.uid}.video_complete.json"
        receipt_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        if result:
            errors = validate_video_receipt(path, receipt_path)
            if errors:
                raise RuntimeError("video receipt validation failed: " + "; ".join(errors))
        return result

    def close(self):
        if self.writer is not None:
            self.end_episode()
