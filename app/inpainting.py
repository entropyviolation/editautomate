"""Remove on-screen text using generative inpainting (LaMa) with OpenCV fallback."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.text_detection import (
    FontStyle,
    TextRegion,
    build_text_mask,
    build_text_mask_from_frame,
    build_union_text_mask_from_video,
    detect_text_regions,
    get_reader,
)
from app.utils import ProgressCallback, default_progress, get_video_info, run_ffmpeg

_LAMA = None


def _get_lama():
    global _LAMA
    if _LAMA is None:
        try:
            from simple_lama_inpainting import SimpleLama

            _LAMA = SimpleLama()
        except Exception:
            _LAMA = False
    return _LAMA if _LAMA is not False else None


def _inpaint_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lama = _get_lama()
    if lama is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_mask = mask
        if mask.max() <= 1:
            pil_mask = (mask * 255).astype(np.uint8)
        result = lama(rgb, pil_mask)
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    # Fallback: OpenCV telea inpainting
    return cv2.inpaint(frame, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


def _merge_regions_per_frame(
    base_regions: list[TextRegion],
    frame: np.ndarray,
    reader,
) -> list[TextRegion]:
    """Re-detect text on a frame; fall back to base regions."""
    results = reader.readtext(frame)
    detected: list[TextRegion] = []
    for bbox, text, conf in results:
        if conf < 0.3:
            continue
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        polygon = tuple((int(p[0]), int(p[1])) for p in bbox)
        detected.append(
            TextRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                text=text,
                confidence=conf,
                polygon=polygon,
            )
        )
    return detected if detected else base_regions


def remove_text_from_video(
    video_path: Path,
    output_path: Path,
    style: FontStyle | None = None,
    progress: ProgressCallback = default_progress,
    use_per_frame_detection: bool = False,
) -> tuple[Path, FontStyle]:
    if style is None:
        style = detect_text_regions(video_path, progress)

    progress("Removing text with generative fill…", 0.30)

    width, height, fps, duration = get_video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_video = output_path.with_suffix(".temp_noaudio.mp4")
    writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max(1, int(duration * fps))
    reader = get_reader()

    # Pre-build a union glyph mask from sampled frames when positions are stable.
    static_mask: np.ndarray | None = None
    if not use_per_frame_detection:
        static_mask = build_union_text_mask_from_video(video_path, reader)
        if static_mask is None and style.regions:
            static_mask = build_text_mask(
                np.zeros((height, width, 3), dtype=np.uint8),
                style.regions,
            )

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if use_per_frame_detection:
            mask = build_text_mask_from_frame(frame, reader)
            if not mask.any():
                regions = _merge_regions_per_frame(style.regions, frame, reader)
                mask = build_text_mask(frame, regions)
        else:
            mask = static_mask if static_mask is not None else build_text_mask(frame, style.regions)

        if mask.any():
            frame = _inpaint_frame(frame, mask)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 5 == 0 or frame_idx == total_frames:
            progress(
                f"Inpainting frame {frame_idx}/{total_frames}…",
                0.30 + (frame_idx / total_frames) * 0.35,
            )

    cap.release()
    writer.release()

    # Strip audio and re-encode for quality
    progress("Encoding clean video (no audio)…", 0.66)
    run_ffmpeg(
        [
            "-i",
            str(temp_video),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )
    temp_video.unlink(missing_ok=True)
    progress("Text removed", 0.68)
    return output_path, style
