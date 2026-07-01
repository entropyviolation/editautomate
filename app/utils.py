"""Shared utilities for the video pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

ProgressCallback = Callable[[str, float], None]


def default_progress(message: str, fraction: float) -> None:
    fraction = max(0.0, min(1.0, fraction))
    print(f"[{fraction * 100:5.1f}%] {message}", file=sys.stderr)


def run_ffmpeg(args: list[str], log: Callable[[str], None] | None = None) -> None:
    """Run ffmpeg with -y (overwrite) prepended; raise on non-zero exit."""
    cmd = ["ffmpeg", "-y", *args]
    if log:
        log("Running: " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed")


def probe_video(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def get_video_info(path: Path) -> tuple[int, int, float, float]:
    """Return width, height, fps, duration."""
    data = probe_video(path)
    video_stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1] or 1)
    duration = float(data["format"].get("duration", 0))
    return width, height, fps, duration


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def work_dir(base: Path | None = None) -> Path:
    """Default workspace for downloads, temp renders, and the song/source/edit library."""
    root = base or Path.cwd() / ".editautomate_cache"
    return ensure_dir(root)


def get_torch_device():
    """Best available PyTorch device: CUDA GPU, Apple MPS, or CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def torch_acceleration_label() -> str:
    """Human-readable label for the active PyTorch backend."""
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"CUDA ({name})"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "Apple GPU (MPS)"
    return "CPU"
