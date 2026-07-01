"""Align video structure (cuts/beats) to a replacement song and loop when needed."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.utils import ProgressCallback, default_progress, get_video_info, probe_video, run_ffmpeg


@dataclass
class BeatAnalysis:
    bpm: float
    beats: list[float]
    duration: float


# Reuse beat grids for the same audio file across jobs (keyed by path + mtime).
_BEAT_CACHE: dict[str, BeatAnalysis] = {}


def _beat_cache_key(audio_path: Path) -> str:
    resolved = audio_path.resolve()
    try:
        mtime = resolved.stat().st_mtime_ns
    except OSError:
        mtime = 0
    return f"{resolved}:{mtime}"


def _audio_duration(path: Path) -> float:
    data = probe_video(path)
    return float(data["format"].get("duration", 0))


def analyze_audio_beats(
    audio_path: Path,
    progress: ProgressCallback = default_progress,
    *,
    use_cache: bool = True,
) -> BeatAnalysis:
    """Detect BPM and beat grid from an audio file."""
    cache_key = _beat_cache_key(audio_path)
    if use_cache and cache_key in _BEAT_CACHE:
        return _BEAT_CACHE[cache_key]

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    import librosa

    progress("Analyzing song BPM and beat grid…", 0.72)
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    if not beat_times:
        bpm = float(tempo) if tempo else 120.0
        interval = 60.0 / max(bpm, 1.0)
        beat_times = [i * interval for i in range(int(duration / interval) + 1)]

    bpm = float(tempo) if tempo else (60.0 * len(beat_times) / max(duration, 0.1))

    # Ensure beat grid covers full duration
    if beat_times[-1] < duration - 0.05:
        interval = 60.0 / max(bpm, 1.0)
        t = beat_times[-1] + interval
        while t < duration:
            beat_times.append(t)
            t += interval

    progress(f"Song: {bpm:.0f} BPM, {len(beat_times)} beats", 0.74)
    result = BeatAnalysis(bpm=bpm, beats=beat_times, duration=duration)
    if use_cache:
        _BEAT_CACHE[cache_key] = result
    return result


def detect_scene_cuts(
    video_path: Path,
    progress: ProgressCallback = default_progress,
    threshold: float = 0.38,
    min_gap_sec: float = 0.25,
) -> list[float]:
    """Return scene-change timestamps in seconds (always includes 0 and duration)."""
    progress("Detecting video scene cuts…", 0.68)
    _, _, fps, duration = get_video_info(video_path)
    if duration <= 0:
        return [0.0]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [0.0, duration]

    cuts: list[float] = [0.0]
    prev_hist: np.ndarray | None = None
    frame_idx = 0
    step = max(1, int(fps // 4))  # sample ~4 per second

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            small = cv2.resize(frame, (64, 64))
            hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                t = frame_idx / fps
                if diff >= threshold and t - cuts[-1] >= min_gap_sec:
                    cuts.append(t)
            prev_hist = hist
        frame_idx += 1

    cap.release()

    if cuts[-1] < duration - 0.05:
        cuts.append(duration)
    elif cuts[-1] > duration:
        cuts[-1] = duration

    progress(f"Found {len(cuts) - 1} scene cut(s)", 0.70)
    return cuts


def _extract_audio_from_video(video_path: Path, output_path: Path) -> Path:
    run_ffmpeg(["-i", str(video_path), "-vn", "-acodec", "libmp3lame", "-q:a", "4", str(output_path)])
    return output_path


def analyze_video_beats(
    video_path: Path,
    work_dir: Path,
    progress: ProgressCallback = default_progress,
) -> BeatAnalysis | None:
    """Try to detect beats from the original TikTok audio track."""
    data = probe_video(video_path)
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    if not has_audio:
        return None

    tmp = work_dir / "original_audio.mp3"
    try:
        _extract_audio_from_video(video_path, tmp)
        return analyze_audio_beats(tmp, progress)
    except Exception:
        return None


def _video_duration(path: Path) -> float:
    from app.utils import get_video_info

    _, _, _, duration = get_video_info(path)
    return duration


def _loop_video_ffmpeg(segments: list[tuple[Path, float, float]], output: Path) -> Path:
    """Concatenate trimmed segments via ffmpeg concat demuxer."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_path = Path(f.name)
        for seg_path, start, end in segments:
            trimmed = seg_path
            if start > 0 or end < _video_duration(seg_path):
                trimmed = output.parent / f"trim_{start:.3f}_{end:.3f}_{seg_path.name}"
                run_ffmpeg(
                    [
                        "-i",
                        str(seg_path),
                        "-ss",
                        str(start),
                        "-to",
                        str(end),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "18",
                        "-an",
                        str(trimmed),
                    ]
                )
            f.write(f"file '{trimmed.resolve()}'\n")

    run_ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output),
        ]
    )
    list_path.unlink(missing_ok=True)
    return output


def loop_video_to_duration(
    video_path: Path,
    target_duration: float,
    cut_points: list[float],
    output_path: Path,
    progress: ProgressCallback = default_progress,
) -> Path:
    """Loop video at scene-cut boundaries until target_duration is reached."""
    _, _, _, video_duration = get_video_info(video_path)
    if video_duration <= 0:
        raise RuntimeError("Video has zero duration")

    if target_duration <= video_duration + 0.05:
        run_ffmpeg(
            [
                "-i",
                str(video_path),
                "-t",
                str(target_duration),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-an",
                str(output_path),
            ]
        )
        return output_path

    progress(f"Looping video to {target_duration:.1f}s…", 0.71)

    # Prefer looping at last scene cut before end (seamless loop point)
    loop_point = video_duration
    for cut in reversed(cut_points):
        if 0.5 < cut < video_duration - 0.15:
            loop_point = cut
            break

    segments: list[tuple[Path, float, float]] = []
    accumulated = 0.0

    # First playthrough (full or to loop point)
    first_len = min(loop_point, target_duration)
    segments.append((video_path, 0.0, first_len))
    accumulated += first_len

    # Repeat loop body
    body_start = 0.0
    body_end = loop_point
    body_len = body_end - body_start
    if body_len < 0.1:
        body_start, body_end, body_len = 0.0, video_duration, video_duration

    while accumulated < target_duration - 0.05:
        take = min(body_len, target_duration - accumulated)
        segments.append((video_path, body_start, body_start + take))
        accumulated += take

    _loop_video_ffmpeg(segments, output_path)
    progress("Video looped to song length", 0.73)
    return output_path


def _beats_in_range(beats: list[float], start: float, end: float) -> list[float]:
    clipped = [b for b in beats if start <= b <= end]
    if not clipped or clipped[0] > start + 0.01:
        clipped.insert(0, start)
    if clipped[-1] < end - 0.01:
        clipped.append(end)
    return clipped


def _segment_speed_filter(speed: float) -> str:
    """Build ffmpeg setpts filter; speed > 1 = faster playback."""
    speed = max(0.25, min(4.0, speed))
    return f"setpts=PTS/{speed:.6f}"


def sync_video_to_song(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    work_dir: Path,
    progress: ProgressCallback = default_progress,
    snippet_start: float = 0.0,
    snippet_end: float | None = None,
    beat_analysis: BeatAnalysis | None = None,
) -> Path:
    """
    Map video scene cuts to song beats within a snippet window.
    Loops the video when the song snippet is longer than the source clip.
    """
    song = beat_analysis or analyze_audio_beats(audio_path, progress)
    end = snippet_end if snippet_end is not None else song.duration
    end = min(end, song.duration)
    start = max(0.0, snippet_start)
    target_duration = max(0.1, end - start)

    if target_duration <= 0:
        raise ValueError("Invalid snippet range")

    cuts = detect_scene_cuts(video_path, progress)
    _, _, _, video_duration = get_video_info(video_path)

    # Loop first if song snippet exceeds video length
    working = video_path
    looped = work_dir / "sync_looped.mp4"
    if target_duration > video_duration - 0.05:
        working = loop_video_to_duration(video_path, target_duration, cuts, looped, progress)
        cuts = detect_scene_cuts(working, progress)
        _, _, _, video_duration = get_video_info(working)

    # Trim working video to at least target duration
    if video_duration > target_duration + 0.1:
        trimmed = work_dir / "sync_trimmed.mp4"
        run_ffmpeg(
            [
                "-i",
                str(working),
                "-t",
                str(target_duration),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-an",
                str(trimmed),
            ]
        )
        working = trimmed
        cuts = detect_scene_cuts(working, progress)
        video_duration = target_duration

    song_beats = _beats_in_range(song.beats, start, end)
    # Normalize song beats to 0..target_duration timeline
    song_beats_local = [b - start for b in song_beats]

    # Align video cuts to song beats via per-segment speed adjustment
    n_seg = min(len(cuts) - 1, len(song_beats_local) - 1)
    if n_seg < 1:
        run_ffmpeg(["-i", str(working), "-c:v", "copy", "-an", str(output_path)])
        return output_path

    progress("Mapping video cuts to song beats…", 0.75)

    segment_files: list[Path] = []
    for i in range(n_seg):
        v_start, v_end = cuts[i], cuts[i + 1]
        s_start, s_end = song_beats_local[i], song_beats_local[i + 1]
        v_dur = max(0.04, v_end - v_start)
        s_dur = max(0.04, s_end - s_start)
        speed = v_dur / s_dur

        seg_out = work_dir / f"sync_seg_{i:03d}.mp4"
        vf = _segment_speed_filter(speed)
        run_ffmpeg(
            [
                "-i",
                str(working),
                "-ss",
                str(v_start),
                "-to",
                str(v_end),
                "-vf",
                vf,
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                str(seg_out),
            ]
        )
        segment_files.append(seg_out)

    # If video had extra tail beyond mapped segments, skip it (beat-aligned edit)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_path = Path(f.name)
        for seg in segment_files:
            f.write(f"file '{seg.resolve()}'\n")

    run_ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-t",
            str(target_duration),
            "-an",
            str(output_path),
        ]
    )
    list_path.unlink(missing_ok=True)
    for seg in segment_files:
        seg.unlink(missing_ok=True)

    progress("Beat-synced video ready", 0.78)
    return output_path


def extract_audio_snippet(
    audio_path: Path,
    output_path: Path,
    start: float,
    end: float | None = None,
) -> Path:
    """Extract a time range from an audio file (output timeline starts at 0)."""
    args: list[str] = []
    if start > 0:
        args.extend(["-ss", str(start)])
    args.extend(["-i", str(audio_path)])
    if end is not None:
        args.extend(["-t", str(max(0.01, end - start))])
    args.extend(["-c:a", "libmp3lame", "-q:a", "0", str(output_path)])
    run_ffmpeg(args)
    return output_path
