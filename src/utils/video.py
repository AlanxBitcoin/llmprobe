from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Build optional video artifacts from generated frame images.
# - Keep codec/fallback behavior deterministic for local runs.

from pathlib import Path
from typing import Literal

import importlib.util

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def preferred_video_suffix() -> Literal[".mp4", ".gif"]:
    return ".mp4" if importlib.util.find_spec("imageio_ffmpeg") else ".gif"


def synthesize_video(frame_paths: list[str | Path], output_path: str | Path, fps: int = 2) -> None:
    if not frame_paths:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".gif":
        frames = _load_uniform_frames(frame_paths)
        duration_ms = max(100, int(1000 / max(1, fps)))
        imageio.mimsave(output, frames, duration=duration_ms / 1000.0)
        return

    frames = _load_uniform_frames(frame_paths)
    with imageio.get_writer(output, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def _load_uniform_frames(frame_paths: list[str | Path]) -> list[np.ndarray]:
    normalized: list[np.ndarray] = []
    base_size: tuple[int, int] | None = None
    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        if base_size is None:
            base_size = image.size
        elif image.size != base_size:
            image = image.resize(base_size, Image.Resampling.LANCZOS)
        normalized.append(np.asarray(image))
    return normalized
