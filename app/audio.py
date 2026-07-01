"""Audio replacement, lyrics transcription, and dialog preservation."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.utils import ProgressCallback, default_progress, get_video_info, get_whisper_device, probe_video, run_ffmpeg

_WHISPER_MODEL = None
_WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
# Whisper uses KV-cache hooks that race when transcribe() runs on multiple threads.
_WHISPER_LOCK = threading.Lock()

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


@dataclass
class SpeechRegion:
    start: float
    end: float
    text: str = ""
    is_likely_dialog: bool = True


def _merge_nearby_regions(
    regions: list[SpeechRegion],
    *,
    gap_sec: float = 0.35,
) -> list[SpeechRegion]:
    if not regions:
        return []
    ordered = sorted(regions, key=lambda r: r.start)
    merged: list[SpeechRegion] = [
        SpeechRegion(
            start=ordered[0].start,
            end=ordered[0].end,
            text=ordered[0].text,
            is_likely_dialog=ordered[0].is_likely_dialog,
        )
    ]
    for region in ordered[1:]:
        prev = merged[-1]
        if region.start - prev.end <= gap_sec:
            prev.end = max(prev.end, region.end)
            if region.text:
                prev.text = f"{prev.text} {region.text}".strip()
            prev.is_likely_dialog = prev.is_likely_dialog or region.is_likely_dialog
        else:
            merged.append(region)
    return merged


def _looks_like_sung_lyrics(text: str) -> bool:
    """Heuristic: repeated short phrases suggest sung lyrics rather than dialog."""
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < 4:
        return False
    return len(set(words)) / len(words) < 0.55


def detect_speech_regions(
    audio_path: Path,
    progress: ProgressCallback = default_progress,
    *,
    replacement_lyrics: list[LyricLine] | None = None,
) -> list[SpeechRegion]:
    """
    Detect speech/dialog in mixed audio (e.g. movie clip + background song).
    Filters segments that overlap replacement-song lyrics when provided.
    """
    progress("Scanning original audio for dialog/speech…", 0.14)
    model = _get_whisper(progress)
    result = _whisper_transcribe(
        model,
        str(audio_path),
        word_timestamps=True,
        language="en",
        task="transcribe",
        condition_on_previous_text=False,
        no_speech_threshold=0.55,
        logprob_threshold=-1.0,
    )

    lyric_windows: list[tuple[float, float]] = []
    if replacement_lyrics:
        lyric_windows = [(line.start, line.end) for line in replacement_lyrics]

    def overlaps_lyrics(start: float, end: float) -> bool:
        for ls, le in lyric_windows:
            if end > ls + 0.05 and start < le - 0.05:
                return True
        return False

    regions: list[SpeechRegion] = []
    for segment in result.get("segments") or []:
        if not segment:
            continue
        if segment.get("no_speech_prob", 1.0) > 0.62:
            continue
        if segment.get("avg_logprob", 0.0) < -1.05:
            continue
        text = (segment.get("text") or "").strip()
        if not text or _HALLUCINATION_RE.search(text):
            continue
        start_raw, end_raw = segment.get("start"), segment.get("end")
        if start_raw is None or end_raw is None:
            continue
        start = float(start_raw)
        end = float(end_raw)
        if end - start < 0.12:
            continue
        sung = _looks_like_sung_lyrics(text)
        likely_dialog = not sung and not overlaps_lyrics(start, end)
        if likely_dialog or (not sung and segment.get("no_speech_prob", 1.0) < 0.35):
            regions.append(
                SpeechRegion(start=start, end=end, text=text, is_likely_dialog=likely_dialog)
            )

    merged = _merge_nearby_regions(regions)
    dialog_count = sum(1 for r in merged if r.is_likely_dialog)
    progress(
        f"Found {len(merged)} speech region(s) ({dialog_count} likely dialog)",
        0.16,
    )
    return merged


def has_extraneous_speech(
    audio_path: Path,
    progress: ProgressCallback = default_progress,
) -> bool:
    """Quick check whether the source clip contains non-music speech."""
    regions = detect_speech_regions(audio_path, progress)
    return any(r.is_likely_dialog for r in regions)


def _load_mono_audio(path: Path, sr: int = 44100) -> tuple[np.ndarray, int]:
    import librosa

    y, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    return y, loaded_sr


def _write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    import soundfile as sf

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    sf.write(str(path), y, sr)


def isolate_dialog_from_mixed(
    original_audio_path: Path,
    speech_regions: list[SpeechRegion],
    output_path: Path,
    progress: ProgressCallback = default_progress,
    *,
    padding_sec: float = 0.18,
) -> Path:
    """
    Attenuate background music in speech regions while keeping dialog.
    Uses HPSS harmonic/percussive split and speech-band emphasis.
    """
    import librosa

    progress("Isolating dialog from background music…", 0.17)
    y, sr = _load_mono_audio(original_audio_path)
    if y.size == 0:
        raise RuntimeError("Original audio is empty")

    dialog = np.zeros_like(y)
    pad_samples = int(padding_sec * sr)

    for region in speech_regions:
        if not region.is_likely_dialog:
            continue
        s = max(0, int(region.start * sr) - pad_samples)
        e = min(len(y), int(region.end * sr) + pad_samples)
        if e - s < sr // 20:
            continue
        seg = y[s:e]
        harmonic, percussive = librosa.effects.hpss(seg, margin=2.5)
        speech_est = percussive * 0.72 + seg * 0.38 - harmonic * 0.35
        speech_est = librosa.effects.preemphasis(speech_est, coef=0.92)
        dialog[s:e] += speech_est

    if not np.any(np.abs(dialog) > 1e-6):
        progress("No dialog isolated — using speech-only gate", 0.175)
        for region in speech_regions:
            if not region.is_likely_dialog:
                continue
            s = max(0, int(region.start * sr) - pad_samples)
            e = min(len(y), int(region.end * sr) + pad_samples)
            dialog[s:e] = y[s:e]

    wav_tmp = output_path.with_suffix(".wav")
    _write_wav(wav_tmp, dialog, sr)
    run_ffmpeg(["-i", str(wav_tmp), "-c:a", "libmp3lame", "-q:a", "2", str(output_path)])
    wav_tmp.unlink(missing_ok=True)
    progress("Dialog track extracted", 0.18)
    return output_path


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_mixed_audio_track(
    new_song_path: Path,
    dialog_path: Path | None,
    speech_regions: list[SpeechRegion],
    duration: float,
    output_path: Path,
    progress: ProgressCallback = default_progress,
    *,
    snippet_start: float = 0.0,
    snippet_end: float | None = None,
    song_duck_db: float = -14.0,
    dialog_gain: float = 1.15,
) -> Path:
    """Mix replacement song with preserved dialog; duck song during speech."""
    progress("Mixing new song with preserved dialog…", 0.765)
    song, sr = _load_mono_audio(new_song_path)

    if snippet_start > 0 or snippet_end is not None:
        s0 = int(snippet_start * sr)
        s1 = int((snippet_end if snippet_end is not None else len(song) / sr) * sr)
        song = song[s0:s1]

    target_samples = max(1, int(duration * sr))
    if len(song) < target_samples:
        reps = int(np.ceil(target_samples / max(len(song), 1)))
        song = np.tile(song, reps)[:target_samples]
    else:
        song = song[:target_samples]

    mix = song.copy()
    duck_linear = 10 ** (song_duck_db / 20.0)

    if dialog_path and dialog_path.is_file():
        dialog, _ = _load_mono_audio(dialog_path, sr=sr)
        if len(dialog) < target_samples:
            dialog = np.pad(dialog, (0, target_samples - len(dialog)))
        else:
            dialog = dialog[:target_samples]

        envelope = np.zeros(target_samples, dtype=float)
        fade = int(0.12 * sr)
        for region in speech_regions:
            if not region.is_likely_dialog:
                continue
            s = max(0, int(region.start * sr))
            e = min(target_samples, int(region.end * sr))
            if e <= s:
                continue
            envelope[s:e] = 1.0
            if fade > 0:
                ramp = _smoothstep(0, fade, np.arange(min(fade, e - s), dtype=float))
                envelope[s : s + len(ramp)] = np.maximum(envelope[s : s + len(ramp)], ramp)
                tail = _smoothstep(0, fade, np.arange(min(fade, e - s), dtype=float)[::-1])
                envelope[e - len(tail) : e] = np.maximum(envelope[e - len(tail) : e], tail)

        mix = mix * (1.0 - envelope * (1.0 - duck_linear)) + dialog * envelope * dialog_gain

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        mix = mix * (0.98 / peak)

    wav_path = output_path.with_suffix(".wav")
    _write_wav(wav_path, mix, sr)
    run_ffmpeg(["-i", str(wav_path), "-c:a", "aac", "-b:a", "320k", str(output_path)])
    wav_path.unlink(missing_ok=True)
    progress("Mixed audio track ready", 0.77)
    return output_path


def render_dialog_on_output_timeline(
    isolated_dialog_path: Path,
    original_regions: list[SpeechRegion],
    output_regions: list[SpeechRegion],
    output_duration: float,
    output_path: Path,
    progress: ProgressCallback = default_progress,
) -> Path:
    """Place isolated dialog samples on the beat-synced output timeline."""
    progress("Aligning dialog to beat-synced timeline…", 0.179)
    dialog, sr = _load_mono_audio(isolated_dialog_path)
    target_samples = max(1, int(output_duration * sr))
    out = np.zeros(target_samples, dtype=float)

    orig_dialog = [r for r in original_regions if r.is_likely_dialog]
    out_dialog = [r for r in output_regions if r.is_likely_dialog]
    pair_count = min(len(orig_dialog), len(out_dialog))
    for idx in range(pair_count):
        orig = orig_dialog[idx]
        mapped = out_dialog[idx]
        src_start = max(0, int(orig.start * sr))
        src_end = min(len(dialog), int(orig.end * sr))
        if src_end - src_start < sr // 30:
            continue
        chunk = dialog[src_start:src_end]
        dst_start = max(0, int(mapped.start * sr))
        dst_end = min(target_samples, int(mapped.end * sr))
        dst_len = dst_end - dst_start
        if dst_len <= 0:
            continue
        if len(chunk) != dst_len:
            chunk = np.interp(
                np.linspace(0, len(chunk) - 1, dst_len),
                np.arange(len(chunk)),
                chunk,
            )
        out[dst_start:dst_end] += chunk

    wav_tmp = output_path.with_suffix(".wav")
    _write_wav(wav_tmp, out, sr)
    run_ffmpeg(["-i", str(wav_tmp), "-c:a", "libmp3lame", "-q:a", "2", str(output_path)])
    wav_tmp.unlink(missing_ok=True)
    return output_path


def replace_audio_with_dialog(
    video_path: Path,
    mixed_audio_path: Path,
    output_path: Path,
    progress: ProgressCallback = default_progress,
) -> Path:
    """Mux a pre-mixed audio track onto video."""
    progress("Applying mixed audio (song + dialog)…", 0.775)
    _, _, _, video_duration = get_video_info(video_path)
    audio_duration = _audio_duration(mixed_audio_path)
    duration = min(video_duration, audio_duration) if audio_duration > 0 else video_duration

    run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-i",
            str(mixed_audio_path),
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
    progress("Audio replaced (dialog preserved)", 0.78)
    return output_path


def video_has_audio(path: Path) -> bool:
    data = probe_video(path)
    return any(s.get("codec_type") == "audio" for s in data.get("streams", []))


def _get_whisper(progress: ProgressCallback | None = None):
    """Load OpenAI Whisper model locally (free — no API key)."""
    global _WHISPER_MODEL
    with _WHISPER_LOCK:
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


def _whisper_transcribe(model: Any, audio: str, **kwargs: Any) -> dict:
    """Thread-safe wrapper — Whisper is not safe for concurrent transcribe() calls."""
    with _WHISPER_LOCK:
        return model.transcribe(audio, **kwargs)


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


def dedupe_overlapping_lyrics(lines: list[LyricLine]) -> list[LyricLine]:
    """Remove overlapping timestamps and merge duplicate consecutive text."""
    if len(lines) < 2:
        return lines

    ordered = sorted(lines, key=lambda line: (line.start, line.end))
    merged: list[LyricLine] = [LyricLine(text=ordered[0].text, start=ordered[0].start, end=ordered[0].end)]

    for line in ordered[1:]:
        prev = merged[-1]
        if line.start < prev.end - 0.02:
            same_text = line.text.strip().lower() == prev.text.strip().lower()
            if same_text or line.text.strip() in prev.text or prev.text.strip() in line.text:
                prev.end = max(prev.end, line.end)
                continue
            prev.end = min(prev.end, line.start)
        merged.append(LyricLine(text=line.text, start=line.start, end=line.end))

    return merged


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
    result = _whisper_transcribe(
        model,
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

    if not result:
        raise RuntimeError("Whisper returned no transcription result")

    words: list[tuple[str, float, float]] = []
    for segment in result.get("segments") or []:
        if not segment or not _segment_is_reliable(segment):
            continue
        for word_info in segment.get("words") or []:
            if not word_info:
                continue
            text = (word_info.get("word") or "").strip()
            if not text:
                continue
            word_start, word_end = word_info.get("start"), word_info.get("end")
            if word_start is None or word_end is None:
                continue
            words.append((text, float(word_start), float(word_end)))

    lines = dedupe_overlapping_lyrics(_merge_fragment_lines(_group_words_into_lines(words)))
    if not lines:
        for segment in result.get("segments") or []:
            if not segment or not _segment_is_reliable(segment):
                continue
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            seg_start, seg_end = segment.get("start"), segment.get("end")
            if seg_start is None or seg_end is None:
                continue
            lines.append(
                LyricLine(
                    text=text,
                    start=float(seg_start),
                    end=float(seg_end),
                )
            )
        lines = dedupe_overlapping_lyrics(_merge_fragment_lines(lines))

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


def filter_lyrics_to_snippet(
    lyrics: list[LyricLine],
    start: float,
    end: float | None,
) -> list[LyricLine]:
    """Return lyric lines overlapping a snippet window (absolute timestamps preserved)."""
    if not lyrics:
        return []
    window_end = end if end is not None else float("inf")
    return [
        line
        for line in lyrics
        if not (line.end <= start or line.start >= window_end)
    ]


def clip_lyrics_to_snippet(
    lyrics: list[LyricLine],
    start: float,
    end: float | None,
) -> list[LyricLine]:
    """Filter and re-base lyric timestamps for a song snippet."""
    if start <= 0 and end is None:
        return dedupe_overlapping_lyrics(lyrics)
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
    return dedupe_overlapping_lyrics(clipped)


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
