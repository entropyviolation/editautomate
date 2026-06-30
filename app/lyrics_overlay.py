"""Overlay lyrics using the original on-screen font style."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.audio import LyricLine
from app.fonts import (
    FALLBACK_BLEED_BLUR_RADIUS,
    FALLBACK_BLEED_OPACITY,
    FALLBACK_FONT,
    FALLBACK_HORIZONTAL_STRETCH,
    TIKTOK_FONT_CANDIDATES,
    find_font_file,
)
from app.storage import OverlayTweak
from app.text_detection import FontStyle, TextRegion
from app.utils import ProgressCallback, default_progress, get_video_info, run_ffmpeg

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _resolve_font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    key = (name, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    path = find_font_file(name)
    if path is not None:
        font = ImageFont.truetype(str(path), size=size)
        _FONT_CACHE[key] = font
        return font

    for candidate_name in TIKTOK_FONT_CANDIDATES:
        if candidate_name == name:
            continue
        path = find_font_file(candidate_name)
        if path is not None:
            font = ImageFont.truetype(str(path), size=size)
            _FONT_CACHE[key] = font
            return font

    font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _pick_anchor_region(style: FontStyle, width: int, height: int) -> TextRegion:
    if style.regions:
        return max(style.regions, key=lambda r: (r.confidence, r.h * r.w))
    return TextRegion(x=width // 10, y=int(height * 0.75), w=width * 8 // 10, h=60, font_size=height // 22)


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _apply_tweak(style: FontStyle, tweak: OverlayTweak | None) -> FontStyle:
    if tweak is None:
        return style
    from dataclasses import replace

    updated = replace(
        style,
        dominant_font=tweak.font_name or style.dominant_font,
        dominant_size=tweak.font_size or style.dominant_size,
        dominant_color=tweak.color or style.dominant_color,
        has_stroke=tweak.has_stroke if tweak.has_stroke is not None else style.has_stroke,
        stroke_color=tweak.stroke_color or style.stroke_color,
        stroke_width=tweak.stroke_width if tweak.stroke_width is not None else style.stroke_width,
    )
    if tweak.font_name:
        updated.font_identified = True
    return updated


def _optical_kern_gap(left: str, right: str) -> float:
    """Approximate optical kerning with pair-specific tightening."""
    if not left or not right:
        return 0.0
    pair = (left, right)
    tight_pairs = {
        ("T", "a"), ("T", "e"), ("T", "o"), ("T", "u"), ("T", "y"),
        ("A", "v"), ("A", "w"), ("A", "y"), ("L", "y"), ("F", "a"),
        ("P", "a"), ("P", "e"), ("V", "a"), ("W", "a"), ("W", "o"),
        ("r", "a"), ("r", "e"), ("r", "o"), ("v", "e"), ("w", "a"),
        ("f", "i"), ("f", "l"), ("t", "o"), ("t", "y"), ("y", "o"),
    }
    if pair in tight_pairs:
        return -2.0
    if left in "TAVWYFP" and right in "aeioru":
        return -1.4
    if left in "aeiou" and right in ".,!?":
        return -0.8
    return -0.55


def _measure_kerned_line(draw: ImageDraw.ImageDraw, line: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if not line:
        return 0, 0
    width = 0
    height = 0
    for i, ch in enumerate(line):
        bbox = draw.textbbox((0, 0), ch, font=font)
        width += bbox[2] - bbox[0]
        height = max(height, bbox[3] - bbox[1])
        if i + 1 < len(line):
            width += int(_optical_kern_gap(ch, line[i + 1]))
    return width, height


def _draw_kerned_line(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    line: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    *,
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int] | None = None,
) -> None:
    cx = float(x)
    for i, ch in enumerate(line):
        draw.text(
            (cx, y),
            ch,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        cx += draw.textlength(ch, font=font)
        if i + 1 < len(line):
            cx += _optical_kern_gap(ch, line[i + 1])


def _render_line_layer(
    line: str,
    font: ImageFont.ImageFont,
    color: tuple[int, int, int],
    *,
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int] | None = None,
    use_optical_kerning: bool = False,
    horizontal_stretch: float = 1.0,
    text_bleed: bool = False,
) -> Image.Image:
    pad = max(8, stroke_width * 2 + 4)
    probe = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    if use_optical_kerning:
        line_w, line_h = _measure_kerned_line(probe, line, font)
    else:
        bbox = probe.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
        line_w, line_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    layer = Image.new("RGBA", (line_w + pad * 2, line_h + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if use_optical_kerning:
        _draw_kerned_line(draw, pad, pad, line, font, color, stroke_width=stroke_width, stroke_fill=stroke_fill)
    elif stroke_width > 0 and stroke_fill is not None:
        draw.text((pad, pad), line, font=font, fill=color, stroke_width=stroke_width, stroke_fill=stroke_fill)
    else:
        draw.text((pad, pad), line, font=font, fill=color)

    if horizontal_stretch != 1.0:
        new_w = max(1, int(layer.width * horizontal_stretch))
        layer = layer.resize((new_w, layer.height), Image.Resampling.LANCZOS)

    if text_bleed:
        blurred = layer.filter(ImageFilter.GaussianBlur(radius=FALLBACK_BLEED_BLUR_RADIUS))
        r, g, b, a = blurred.split()
        a = a.point(lambda p: min(255, int(p * FALLBACK_BLEED_OPACITY)))
        blurred.putalpha(a)
        layer = Image.alpha_composite(layer, blurred)

    return layer


def _draw_lyric_on_frame(
    frame: np.ndarray,
    text: str,
    style: FontStyle,
    width: int,
    height: int,
    tweak: OverlayTweak | None = None,
) -> np.ndarray:
    style = _apply_tweak(style, tweak)
    use_fallback = not style.font_identified
    anchor = _pick_anchor_region(style, width, height)
    font_size = style.dominant_size or anchor.font_size
    font_name = FALLBACK_FONT if use_fallback else style.dominant_font
    font = _resolve_font(font_name, font_size)
    color = style.dominant_color
    stroke = style.stroke_color if style.has_stroke else None
    stroke_width = style.stroke_width if style.has_stroke else 0
    ox = tweak.offset_x if tweak else 0
    oy = tweak.offset_y if tweak else 0

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(pil)

    display_text = text.lower() if use_fallback else text.upper()
    max_width = int(width * 0.88)
    lines = _wrap_text(display_text, font, max_width, draw)

    line_layers: list[Image.Image] = []
    line_heights: list[int] = []
    line_widths: list[int] = []
    for line in lines:
        layer = _render_line_layer(
            line,
            font,
            color,
            stroke_width=0 if use_fallback else stroke_width,
            stroke_fill=None if use_fallback else stroke,
            use_optical_kerning=use_fallback,
            horizontal_stretch=FALLBACK_HORIZONTAL_STRETCH if use_fallback else 1.0,
            text_bleed=use_fallback,
        )
        line_layers.append(layer)
        line_widths.append(layer.width)
        line_heights.append(layer.height)

    line_gap = 6
    total_h = sum(line_heights) + max(0, (len(lines) - 1) * line_gap)
    anchor_cy = anchor.y + anchor.h // 2

    if style.vertical_anchor == "top":
        start_y = max(10, anchor.y)
    elif style.vertical_anchor == "bottom":
        start_y = max(10, anchor_cy - total_h // 2)
    else:
        start_y = max(10, anchor_cy - total_h // 2)

    y = start_y + oy
    for i, layer in enumerate(line_layers):
        x = (width - line_widths[i]) // 2 + ox
        pil.paste(layer, (x, y), layer)
        y += line_heights[i] + line_gap

    rgb = pil.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def _active_lyric(time_sec: float, lyrics: list[LyricLine]) -> str:
    for line in lyrics:
        if line.start <= time_sec < line.end:
            return line.text
    return ""


def overlay_lyrics(
    video_path: Path,
    lyrics: list[LyricLine],
    style: FontStyle,
    output_path: Path,
    progress: ProgressCallback = default_progress,
    tweak: OverlayTweak | None = None,
    snippet_start: float = 0.0,
) -> Path:
    progress("Rendering lyrics with matched font…", 0.80)

    width, height, fps, duration = get_video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_video = output_path.with_suffix(".temp_lyrics.mp4")
    writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max(1, int(duration * fps))
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps + snippet_start
        text = _active_lyric(t, lyrics)
        if text:
            frame = _draw_lyric_on_frame(frame, text, style, width, height, tweak=tweak)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 10 == 0 or frame_idx == total_frames:
            progress(
                f"Overlaying lyrics {frame_idx}/{total_frames}…",
                0.80 + (frame_idx / total_frames) * 0.15,
            )

    cap.release()
    writer.release()

    progress("Muxing final high-quality output…", 0.96)
    run_ffmpeg(
        [
            "-i",
            str(temp_video),
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "copy",
            "-shortest",
            str(output_path),
        ]
    )
    temp_video.unlink(missing_ok=True)
    progress("Complete", 1.0)
    return output_path


def re_render_edit(
    with_audio_path: Path,
    lyrics: list[LyricLine],
    style: FontStyle,
    output_path: Path,
    tweak: OverlayTweak | None = None,
    snippet_start: float = 0.0,
    progress: ProgressCallback = default_progress,
) -> Path:
    """Re-render lyrics overlay on an existing video (studio tweaks)."""
    return overlay_lyrics(
        with_audio_path,
        lyrics,
        style,
        output_path,
        progress=progress,
        tweak=tweak,
        snippet_start=snippet_start,
    )
