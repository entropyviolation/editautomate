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
from app.utils import (
    ProgressCallback,
    default_progress,
    get_torch_device,
    get_video_info,
    run_ffmpeg,
    torch_acceleration_label,
)

_LAMA = None
_LAMA_DEVICE = None
_ROI_PAD = 48
_ROI_MAX_COVERAGE = 0.72


def _get_lama():
    global _LAMA, _LAMA_DEVICE
    device = get_torch_device()
    if _LAMA is None or _LAMA_DEVICE != device:
        try:
            from simple_lama_inpainting import SimpleLama

            _LAMA = SimpleLama(device=device)
            _LAMA_DEVICE = device
        except Exception:
            _LAMA = False
            _LAMA_DEVICE = None
    return _LAMA if _LAMA is not False else None


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _roi_bounds(
    mask: np.ndarray,
    frame_shape: tuple[int, ...],
    *,
    pad: int = _ROI_PAD,
) -> tuple[int, int, int, int] | None:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    height, width = frame_shape[:2]
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(width, x1 + pad),
        min(height, y1 + pad),
    )


def _should_inpaint_roi(mask: np.ndarray, frame_shape: tuple[int, ...]) -> bool:
    bounds = _roi_bounds(mask, frame_shape)
    if bounds is None:
        return False
    x0, y0, x1, y1 = bounds
    height, width = frame_shape[:2]
    roi_area = max(1, (x1 - x0) * (y1 - y0))
    frame_area = max(1, height * width)
    return roi_area / frame_area < _ROI_MAX_COVERAGE


def _paste_inpaint_result(
    frame: np.ndarray,
    result_rgb: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    crop_h, crop_w = y1 - y0, x1 - x0
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    if result_bgr.shape[0] != crop_h or result_bgr.shape[1] != crop_w:
        result_bgr = cv2.resize(result_bgr, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    out = frame.copy()
    out[y0:y1, x0:x1] = result_bgr
    return out


def _opencv_inpaint(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if _should_inpaint_roi(mask, frame.shape):
        x0, y0, x1, y1 = _roi_bounds(mask, frame.shape)  # type: ignore[misc]
        crop = frame[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        filled = cv2.inpaint(crop, crop_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        out = frame.copy()
        out[y0:y1, x0:x1] = filled
        return out
    return cv2.inpaint(frame, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


def _lama_inpaint(frame: np.ndarray, mask: np.ndarray, lama) -> np.ndarray:
    if _should_inpaint_roi(mask, frame.shape):
        x0, y0, x1, y1 = _roi_bounds(mask, frame.shape)  # type: ignore[misc]
        crop = frame[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if crop_mask.max() <= 1:
            crop_mask = (crop_mask * 255).astype(np.uint8)
        result = lama(rgb, crop_mask)
        return _paste_inpaint_result(frame, np.array(result), (x0, y0, x1, y1))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_mask = mask
    if mask.max() <= 1:
        pil_mask = (mask * 255).astype(np.uint8)
    result = lama(rgb, pil_mask)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


def _inpaint_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lama = _get_lama()
    if lama is not None:
        return _lama_inpaint(frame, mask, lama)
    return _opencv_inpaint(frame, mask)


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

    lama = _get_lama()
    backend = torch_acceleration_label() if lama is not None else "OpenCV (LaMa unavailable)"
    progress(f"Removing text with generative fill ({backend})…", 0.30)

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
