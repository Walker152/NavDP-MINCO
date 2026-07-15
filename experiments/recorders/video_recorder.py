from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import cv2


class EpisodeVideoRecorder:
    def __init__(self, output_dir, fps=10, crf=23, scale=1.0):
        self.output_dir = Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True); self.fps=fps; self.crf=crf; self.scale=scale; self.writer=None; self.uid=None; self.shape=None; self.frame_count=0

    def start_episode(self, episode_uid, expected_shape=None):
        if self.writer is not None: self.end_episode()
        self.uid = str(episode_uid); self.shape = tuple(expected_shape) if expected_shape else None; self.frame_count = 0
        path = self.output_dir / f"{self.uid}.mp4"
        self.writer = imageio.get_writer(path, fps=self.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=16, output_params=["-crf", str(self.crf), "-movflags", "faststart"])
        return path

    def write(self, frame):
        if self.writer is None: raise RuntimeError("start_episode must be called first")
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim == 2: frame = np.repeat(frame[:,:,None], 3, axis=2)
        if frame.ndim != 3 or frame.shape[2] != 3: raise ValueError(f"invalid frame shape: {frame.shape}")
        if self.scale != 1.0:
            if not np.isfinite(self.scale) or self.scale <= 0.0: raise ValueError("video scale must be finite and positive")
            frame = cv2.resize(frame, None, fx=float(self.scale), fy=float(self.scale), interpolation=cv2.INTER_AREA)
        if self.shape is None: self.shape = tuple(frame.shape)
        if tuple(frame.shape) != self.shape: raise ValueError(f"frame shape changed: {frame.shape} != {self.shape}")
        self.writer.append_data(np.ascontiguousarray(frame)); self.frame_count += 1

    def end_episode(self):
        if self.writer is None: return None
        path = self.output_dir / f"{self.uid}.mp4"; self.writer.close(); self.writer = None
        if self.frame_count == 0:
            path.unlink(missing_ok=True); result = None
        else: result = path
        metadata = {"episode_uid":self.uid, "fps":self.fps, "crf":self.crf, "scale":self.scale, "frame_count":self.frame_count, "shape":list(self.shape) if self.shape else None, "codec":"libx264", "pixel_format":"yuv420p", "macro_block_size":16, "complete":self.frame_count > 0}
        (self.output_dir / f"{self.uid}.video_complete.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return result

    def close(self):
        if self.writer is not None: self.end_episode()
