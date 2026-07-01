"""Audio replacement and lyrics transcription."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from app.utils import ProgressCallback, default_progress, get_video_info, get_whisper_device, run_ffmpeg

_WHISPER_MODEL = None
_WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")

_HALLUCINATION_RE = re.compile(
    r"(thank(s| you) for watching|please subscribe|subtitles by|"
    r"^\s*(you|the|a|an|and|or|to|in|on|at|it|is|so|oh|uh|um|ah)\s*$)",
    re.IGNORECASE,
)


@dataclass
class LyricLine:
    text: str
    start: float
    end: float


def _get_whisper(progress: ProgressCallback | None = None):
    """Load OpenAI Whisper model locally (free — no API key)."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        try:
            if next(_WHISPER_MODEL.parameters()).device.type == "mps":
                _WHISPER_MODEL = None
        except StopIteration:
            pass
    if _WHISPER_MODEL is None:
        import whisper

        if progress:
            progress(f"Loading Whisper model ({_WHISPER_MODEL_NAME})…", 0.68)
        try:
            device = get_whisper_device()
            _WHISPER_MODEL = whisper.load_model(_WHISPER_MODEL_NAME, device=str(device))
        except Exception as exc:
            from app.utils import format_user_error

            raise RuntimeError(format_user_error(exc)) from exc
    return _WHISPER_MODEL


def _segment_is_reliable(segment: dict) -> bool:
    """Drop Whisper segments that are likely silence or hallucinations."""
    if segment.get("no_speech_prob", 0.0) > 0.55:
        return False
    if segment.get("avg_logprob", 0.0) < -1.0:
        return False
    text = (segment.get("text") or "").strip()
    if not text:
        return False
    if _HALLUCINATION_RE.search(text):
        return False
    return True


def _merge_fragment_lines(lines: list[LyricLine], *, max_fragment_chars: int = 4) -> list[LyricLine]:
    """Merge orphan single-word lines into the next lyric line."""
    if len(lines) < 2:
        return lines

    merged: list[LyricLine] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        words = line.text.split()
        if (
            len(words) == 1
            and len(line.text.strip()) <= max_fragment_chars
            and i + 1 < len(lines)
            and lines[i + 1].start - line.end < 1.5
        ):
            nxt = lines[i + 1]
            merged.append(
                LyricLine(
                    text=f"{line.text.strip()} {nxt.text.strip()}",
                    start=line.start,
                    end=nxt.end,
                )
            )
            i += 2
            continue
        merged.append(line)
        i += 1
    return merged


def merge_lyrics_range(
    existing: list[LyricLine],
    new_lines: list[LyricLine],
    range_start: float,
    range_end: float | None,
) -> list[LyricLine]:
    """Replace lyrics inside a time window while keeping lines outside it."""
    window_end = range_end if range_end is not None else float("inf")
    kept = [line for line in existing if line.end <= range_start or line.start >= window_end]
    merged = [*kept, *new_lines]
    merged.sort(key=lambda line: line.start)
    return merged


def strip_audio(video_path: Path, output_path: Path) -> Path:
    run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-an",
            "-c:v",
            "copy",
            str(output_path),
        ]
    )
    return output_path


def _group_words_into_lines(
    words: list[tuple[str, float, float]],
    *,
    pause_gap: float = 0.35,
    max_chars: int = 42,
) -> list[LyricLine]:
    """Build display lines from Whisper word timestamps."""
    if not words:
        return []

    lines: list[LyricLine] = []
    chunk: list[str] = []
    chunk_start = words[0][1]
    chunk_end = words[0][2]

    for word, start, end in words:
        gap = start - chunk_end if chunk else 0.0
        trial = " ".join([*chunk, word]) if chunk else word
        if chunk and (gap > pause_gap or len(trial) > max_chars):
            lines.append(LyricLine(text=" ".join(chunk), start=chunk_start, end=chunk_end))
            chunk = [word]
            chunk_start = start
        else:
            if not chunk:
                chunk_start = start
            chunk.append(word)
        chunk_end = end

    if chunk:
        lines.append(LyricLine(text=" ".join(chunk), start=chunk_start, end=chunk_end))
    return lines


def transcribe_lyrics(
    audio_path: Path,
    progress: ProgressCallback = default_progress,
) -> list[LyricLine]:
    progress("Transcribing new song lyrics…", 0.70)
    model = _get_whisper(progress)
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language="en",
        task="transcribe",
        condition_on_previous_text=False,
        no_speech_threshold=0.45,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        temperature=(0.0, 0.2, 0.4, 0.6),
    )

    words: list[tuple[str, float, float]] = []
    for segment in result.get("segments", []):
        if not _segment_is_reliable(segment):
            continue
        for word_info in segment.get("words") or []:
            text = (word_info.get("word") or "").strip()
            if not text:
                continue
            words.append((text, float(word_info["start"]), float(word_info["end"])))

    lines = _merge_fragment_lines(_group_words_into_lines(words))
    if not lines:
        for segment in result.get("segments", []):
            if not _segment_is_reliable(segment):
                continue
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            lines.append(
                LyricLine(
                    text=text,
                    start=float(segment["start"]),
                    end=float(segment["end"]),
                )
            )
        lines = _merge_fragment_lines(lines)

    if not lines and result.get("text"):
        duration = _audio_duration(audio_path)
        lines.append(LyricLine(text=result["text"].strip(), start=0.0, end=max(duration, 1.0)))

    progress(f"Transcribed {len(lines)} lyric segment(s)", 0.75)
    return lines


def _audio_duration(path: Path) -> float:
    from app.utils import probe_video

    data = probe_video(path)
    return float(data["format"].get("duration", 0))


def replace_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    progress: ProgressCallback = default_progress,
    snippet_start: float = 0.0,
    snippet_end: float | None = None,
) -> Path:
    progress("Replacing audio track…", 0.76)
    _, _, _, video_duration = get_video_info(video_path)
    audio_duration = _audio_duration(audio_path)

    if snippet_end is not None:
        audio_duration = min(audio_duration, snippet_end) - snippet_start
    elif snippet_start > 0:
        audio_duration = max(0.0, audio_duration - snippet_start)

    duration = min(video_duration, audio_duration) if audio_duration > 0 else video_duration

    ffmpeg_args = ["-i", str(video_path)]
    if snippet_start > 0:
        ffmpeg_args.extend(["-ss", str(snippet_start)])
    if snippet_end is not None:
        ffmpeg_args.extend(["-to", str(snippet_end)])
    ffmpeg_args.extend(["-i", str(audio_path)])

    run_ffmpeg(
        [
            *ffmpeg_args,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "320k",
            "-shortest",
            "-t",
            str(duration),
            str(output_path),
        ]
    )
    progress("Audio replaced", 0.78)
    return output_path


def clip_lyrics_to_snippet(
    lyrics: list[LyricLine],
    start: float,
    end: float | None,
) -> list[LyricLine]:
    """Filter and re-base lyric timestamps for a song snippet."""
    if start <= 0 and end is None:
        return lyrics
    clipped: list[LyricLine] = []
    window_end = end if end is not None else float("inf")
    for line in lyrics:
        if line.end <= start or line.start >= window_end:
            continue
        clipped.append(
            LyricLine(
                text=line.text,
                start=max(0.0, line.start - start),
                end=min(window_end - start, line.end - start),
            )
        )
    return clipped


def extract_audio_from_file(media_path: Path, output_path: Path) -> Path:
    run_ffmpeg(
        [
            "-i",
            str(media_path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "0",
            str(output_path),
        ]
    )
    return output_path
