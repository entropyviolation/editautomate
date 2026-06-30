"""Detect on-screen text and analyze font appearance."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

from app.fonts import (
    FALLBACK_FONT,
    FONT_MATCH_MIN_AGREEMENT,
    FONT_MATCH_MIN_CONFIDENCE,
    TIKTOK_FONT_CANDIDATES,
    font_available,
)
from app.utils import ProgressCallback, default_progress, get_video_info


@dataclass
class TextRegion:
    x: int
    y: int
    w: int
    h: int
    text: str = ""
    color: tuple[int, int, int] = (255, 255, 255)
    font_size: int = 48
    font_name: str = FALLBACK_FONT
    font_confidence: float = 0.0
    stroke_color: tuple[int, int, int] | None = (0, 0, 0)
    stroke_width: int = 2
    confidence: float = 0.0
    polygon: tuple[tuple[int, int], ...] | None = None

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


@dataclass
class FontStyle:
    """Captured style from original on-screen text."""

    regions: list[TextRegion] = field(default_factory=list)
    dominant_font: str = FALLBACK_FONT
    dominant_color: tuple[int, int, int] = (255, 255, 255)
    dominant_size: int = 48
    font_identified: bool = False
    has_stroke: bool = True
    stroke_color: tuple[int, int, int] = (0, 0, 0)
    stroke_width: int = 2
    vertical_anchor: str = "center"  # top | center | bottom


def _reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False, verbose=False)


_READER = None


def get_reader():
    global _READER
    if _READER is None:
        _READER = _reader()
    return _READER


def sample_frames(video_path: Path, count: int = 12) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = np.linspace(0, max(total - 1, 0), num=min(count, total), dtype=int)
    frames: list[tuple[int, np.ndarray]] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append((int(idx), frame))
    cap.release()
    return frames


def _extract_text_color(frame: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[int, int, int]:
    roi = frame[max(0, y) : y + h, max(0, x) : x + w]
    if roi.size == 0:
        return (255, 255, 255)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Text pixels tend to be very bright or very dark vs background
    bright = gray > 200
    dark = gray < 60
    mask = bright | dark
    if mask.sum() < 10:
        mask = np.ones(gray.shape, dtype=bool)

    pixels = roi[mask]
    if len(pixels) == 0:
        return (255, 255, 255)

    # Pick the most saturated / extreme luminance cluster
    luminance = pixels.mean(axis=1)
    if (luminance > 128).sum() >= (luminance <= 128).sum():
        chosen = pixels[luminance > luminance.mean()]
    else:
        chosen = pixels[luminance <= luminance.mean()]

    if len(chosen) == 0:
        chosen = pixels
    bgr = chosen.mean(axis=0)
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def _guess_font(frame: np.ndarray, region: TextRegion) -> tuple[str, float]:
    """Score candidate fonts; return best match and confidence (0–1)."""
    x, y, w, h = region.x, region.y, region.w, region.h
    roi = frame[max(0, y) : y + h, max(0, x) : x + w]
    if roi.size == 0:
        return FALLBACK_FONT, 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = edges.mean() / 255.0
    aspect = w / max(h, 1)
    char_count = max(1, len(region.text.replace(" ", "")) or int(aspect * 1.4))
    char_width = w / char_count

    scores: dict[str, float] = {name: 0.0 for name in TIKTOK_FONT_CANDIDATES}

    # Condensed lowercase lyric overlays (very common on TikTok)
    if aspect >= 3.8 and char_width <= h * 0.62:
        scores["Arial Narrow"] += 0.42
        scores["Helvetica Neue"] += 0.18
    if edge_density < 0.095 and aspect >= 3.2:
        scores["Arial Narrow"] += 0.28
        scores["Helvetica"] += 0.12
    if h <= 52 and aspect >= 4.0:
        scores["Arial Narrow"] += 0.18

    # Heavy block display type
    if edge_density > 0.165 and aspect < 5.5:
        scores["Impact"] += 0.42
    if edge_density > 0.125 and h >= 58:
        scores["Arial Black"] += 0.28
        scores["Impact"] += 0.18
    elif edge_density > 0.105 and h >= 50:
        scores["Arial Bold"] += 0.24
        scores["Arial Black"] += 0.16

    # Wide grotesk bars
    if aspect >= 7.5:
        scores["Montserrat Bold"] += 0.26
        scores["Proxima Nova"] += 0.14

    for name, score in list(scores.items()):
        if score > 0 and font_available(name):
            scores[name] += 0.06

    ranked = sorted(scores.items(), key=lambda item: -item[1])
    best_name, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_score < 0.22:
        return FALLBACK_FONT, 0.0

    margin = best_score - second_score
    confidence = min(1.0, margin * 1.35 + best_score * 0.45)
    if confidence < FONT_MATCH_MIN_CONFIDENCE:
        return FALLBACK_FONT, confidence
    return best_name, confidence


def _resolve_dominant_font(regions: list[TextRegion]) -> tuple[str, bool]:
    """Pick a font only when detections agree with sufficient confidence."""
    if not regions:
        return FALLBACK_FONT, False

    confident = [r for r in regions if r.font_confidence >= FONT_MATCH_MIN_CONFIDENCE]
    pool = confident or regions
    votes = Counter(r.font_name for r in pool if r.font_name != FALLBACK_FONT or r.font_confidence > 0)
    if not votes:
        return FALLBACK_FONT, False

    dominant_name, top_count = votes.most_common(1)[0]
    agreement = top_count / len(pool)
    matching = [r for r in pool if r.font_name == dominant_name]
    avg_conf = mean(r.font_confidence for r in matching)

    identified = (
        agreement >= FONT_MATCH_MIN_AGREEMENT
        and avg_conf >= FONT_MATCH_MIN_CONFIDENCE
        and dominant_name != FALLBACK_FONT
    )
    if identified:
        return dominant_name, True
    return FALLBACK_FONT, False


def _estimate_stroke(frame: np.ndarray, region: TextRegion) -> tuple[bool, tuple[int, int, int], int]:
    x, y, w, h = region.x, region.y, region.w, region.h
    pad = max(2, region.font_size // 20)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return True, (0, 0, 0), 2

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = edges.mean() / 255.0
    has_stroke = edge_ratio > 0.08
    stroke_color = (0, 0, 0)
    stroke_width = max(1, region.font_size // 24)
    return has_stroke, stroke_color, stroke_width


def detect_text_regions(
    video_path: Path,
    progress: ProgressCallback = default_progress,
    sample_count: int = 16,
) -> FontStyle:
    progress("Analyzing on-screen text…", 0.22)
    reader = get_reader()
    frames = sample_frames(video_path, sample_count)

    all_regions: dict[tuple[int, int, int, int], TextRegion] = {}

    for i, (frame_idx, frame) in enumerate(frames):
        progress(f"Scanning frame {i + 1}/{len(frames)} for text…", 0.22 + (i / len(frames)) * 0.06)
        results = reader.readtext(frame)

        for bbox, text, conf in results:
            if conf < 0.35 or not text.strip():
                continue
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
            if w < 8 or h < 8:
                continue

            key = (x // 20 * 20, y // 20 * 20, w // 10 * 10, h // 10 * 10)
            color = _extract_text_color(frame, x, y, w, h)
            font_size = max(18, int(h * 0.95))
            tmp = TextRegion(x=x, y=y, w=w, h=h, font_size=font_size, text=text.strip())
            font_name, font_confidence = _guess_font(frame, tmp)
            polygon = tuple((int(p[0]), int(p[1])) for p in bbox)
            region = TextRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                text=text.strip(),
                color=color,
                font_size=font_size,
                font_name=font_name,
                font_confidence=font_confidence,
                confidence=float(conf),
                polygon=polygon,
            )
            has_stroke, stroke_color, stroke_width = _estimate_stroke(frame, region)
            region.stroke_color = stroke_color if has_stroke else None
            region.stroke_width = stroke_width

            existing = all_regions.get(key)
            if existing is None or conf > existing.confidence:
                all_regions[key] = region

    regions = sorted(all_regions.values(), key=lambda r: (-r.confidence, -r.h))
    if not regions:
        # Fallback: typical TikTok lyric bar at lower-center
        _, height, _, _ = get_video_info(video_path)
        regions = [
            TextRegion(
                x=40,
                y=int(height * 0.72),
                w=100,
                h=60,
                text="",
                font_size=max(36, height // 22),
            )
        ]

    dominant = regions[0]
    dominant_font, font_identified = _resolve_dominant_font(regions)
    style = FontStyle(
        regions=regions,
        dominant_font=dominant_font,
        dominant_color=dominant.color,
        dominant_size=dominant.font_size,
        font_identified=font_identified,
        has_stroke=dominant.stroke_color is not None if font_identified else False,
        stroke_color=dominant.stroke_color or (0, 0, 0),
        stroke_width=dominant.stroke_width if font_identified else 0,
    )

    # Vertical anchor from median y-position
    _, height, _, _ = get_video_info(video_path)
    median_y = np.median([r.y + r.h / 2 for r in regions])
    if median_y < height * 0.33:
        style.vertical_anchor = "top"
    elif median_y > height * 0.66:
        style.vertical_anchor = "bottom"
    else:
        style.vertical_anchor = "center"

    progress(f"Found {len(regions)} text region(s)", 0.28)
    return style


def build_mask_from_ocr_results(
    frame_shape: tuple[int, ...],
    results: list,
    *,
    min_confidence: float = 0.3,
    dilate: int = 2,
) -> np.ndarray:
    """Mask only detected glyph polygons (not full bounding boxes)."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    for bbox, text, conf in results:
        if conf < min_confidence or not str(text).strip():
            continue
        pts = np.array([[int(p[0]), int(p[1])] for p in bbox], dtype=np.int32)
        if pts.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [pts], 255)

    if dilate > 0 and mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def build_text_mask_from_frame(
    frame: np.ndarray,
    reader,
    *,
    min_confidence: float = 0.3,
    dilate: int = 2,
) -> np.ndarray:
    results = reader.readtext(frame)
    return build_mask_from_ocr_results(frame.shape, results, min_confidence=min_confidence, dilate=dilate)


def build_text_mask(
    frame: np.ndarray,
    regions: list[TextRegion],
    dilate: int = 2,
) -> np.ndarray:
    """Build a tight mask from stored OCR polygons, with minimal fallback padding."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for region in regions:
        if region.polygon:
            pts = np.array(region.polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            continue

        x, y, w, h = region.x, region.y, region.w, region.h
        pad = max(1, min(w, h) // 20)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(frame.shape[1], x + w + pad)
        y1 = min(frame.shape[0], y + h + pad)
        mask[y0:y1, x0:x1] = 255

    if dilate > 0 and mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def build_union_text_mask_from_video(
    video_path: Path,
    reader,
    *,
    sample_count: int = 10,
    min_confidence: float = 0.3,
    dilate: int = 2,
) -> np.ndarray | None:
    """Union glyph masks from sampled frames for mostly-static on-screen text."""
    frames = sample_frames(video_path, sample_count)
    if not frames:
        return None

    mask: np.ndarray | None = None
    for _, frame in frames:
        frame_mask = build_text_mask_from_frame(
            frame,
            reader,
            min_confidence=min_confidence,
            dilate=0,
        )
        if not frame_mask.any():
            continue
        mask = frame_mask if mask is None else cv2.bitwise_or(mask, frame_mask)

    if mask is None or not mask.any():
        return None

    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask
