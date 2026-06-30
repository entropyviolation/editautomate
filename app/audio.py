"""Audio replacement and lyrics transcription."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.utils import ProgressCallback, default_progress, get_video_info, run_ffmpeg

_WHISPER_MODEL = None


@dataclass
class LyricLine:
    text: str
    start: float
    end: float


def _get_whisper():
    """Load OpenAI Whisper model locally (free — no API key)."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        import whisper

        _WHISPER_MODEL = whisper.load_model("base")
    return _WHISPER_MODEL


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
    model = _get_whisper()
    result = model.transcribe(str(audio_path), word_timestamps=True)

    words: list[tuple[str, float, float]] = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words") or []:
            text = (word_info.get("word") or "").strip()
            if not text:
                continue
            words.append((text, float(word_info["start"]), float(word_info["end"])))

    lines = _group_words_into_lines(words)
    if not lines:
        for segment in result.get("segments", []):
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
