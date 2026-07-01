"""Align video structure (cuts/beats) to a replacement song and loop when needed."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.utils import ProgressCallback, default_progress, get_video_info, probe_video, run_ffmpeg


BeatSyncMode = str  # "standard" | "beat_drop"


@dataclass
class BeatAnalysis:
    bpm: float
    beats: list[float]
    duration: float
    beat_drop_time: float | None = None


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


def _normalize_envelope(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def detect_beat_drop_time(y: np.ndarray, sr: int, duration: float) -> float | None:
    """
    Find the primary beat drop: a sudden rise in bass, RMS, and onset energy.
    Skips the first ~10% and last ~15% of the track to avoid intro/outro false positives.
    """
    if duration < 2.0 or y.size < sr:
        return None

    import librosa

    hop = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.median)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop, fmax=8000)
    bass = mel[:24].mean(axis=0)

    times = librosa.frames_to_time(np.arange(len(oenv)), sr=sr, hop_length=hop)
    oenv_n = _normalize_envelope(oenv)
    rms_n = _normalize_envelope(rms)
    bass_n = _normalize_envelope(bass)

    min_len = min(len(oenv_n), len(rms_n), len(bass_n), len(times))
    if min_len < 8:
        return None

    oenv_n = oenv_n[:min_len]
    rms_n = rms_n[:min_len]
    bass_n = bass_n[:min_len]
    times = times[:min_len]

    combined = bass_n * 0.40 + rms_n * 0.35 + oenv_n * 0.25
    window = max(3, int(min_len * 0.025))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    smoothed = np.convolve(combined, kernel, mode="same")
    deriv = np.diff(smoothed, prepend=smoothed[0])
    drop_score = np.clip(deriv, 0.0, None) * 0.55 + combined * 0.45

    min_time = duration * 0.10
    max_time = duration * 0.85
    mask = (times >= min_time) & (times <= max_time)
    if not mask.any():
        return None

    idx = int(np.argmax(drop_score[mask]))
    valid = np.where(mask)[0]
    drop_time = float(times[valid[idx]])
    return drop_time


def _snap_to_nearest_beat(time_sec: float, beats: list[float]) -> float:
    if not beats:
        return time_sec
    return min(beats, key=lambda b: abs(b - time_sec))


def _numpy_scalar(value: object, default: float = 0.0) -> float:
    """Coerce librosa/numpy return values to a plain Python float."""
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


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
    duration = _numpy_scalar(librosa.get_duration(y=y, sr=sr))

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    tempo_bpm = _numpy_scalar(tempo)
    if not beat_times:
        bpm = tempo_bpm or 120.0
        interval = 60.0 / max(bpm, 1.0)
        beat_times = [i * interval for i in range(int(duration / interval) + 1)]

    bpm = tempo_bpm or (60.0 * len(beat_times) / max(duration, 0.1))

    # Ensure beat grid covers full duration
    if beat_times[-1] < duration - 0.05:
        interval = 60.0 / max(bpm, 1.0)
        t = beat_times[-1] + interval
        while t < duration:
            beat_times.append(t)
            t += interval

    progress(f"Song: {bpm:.0f} BPM, {len(beat_times)} beats", 0.74)

    beat_drop_time: float | None = None
    try:
        raw_drop = detect_beat_drop_time(y, sr, duration)
        if raw_drop is not None:
            beat_drop_time = _snap_to_nearest_beat(raw_drop, beat_times)
            progress(f"Beat drop detected at {beat_drop_time:.1f}s", 0.745)
    except Exception:
        beat_drop_time = None

    result = BeatAnalysis(bpm=bpm, beats=beat_times, duration=duration, beat_drop_time=beat_drop_time)
    if use_cache:
        _BEAT_CACHE[cache_key] = result
    return result


def detect_scene_cuts(
    video_path: Path,
    progress: ProgressCallback = default_progress,
    threshold: float = 0.44,
    min_gap_sec: float = 0.45,
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
    speed = max(0.2, min(3.0, speed))
    return f"setpts=PTS/{speed:.6f}"


def _segment_motion_scores(
    video_path: Path,
    cuts: list[float],
    progress: ProgressCallback = default_progress,
) -> list[float]:
    """Return a 0..1 motion score for each segment between scene cuts."""
    _, _, fps, _ = get_video_info(video_path)
    if len(cuts) < 2:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [0.5] * (len(cuts) - 1)

    scores: list[float] = []
    try:
        for i in range(len(cuts) - 1):
            v_start, v_end = cuts[i], cuts[i + 1]
            seg_dur = max(0.04, v_end - v_start)
            sample_count = max(3, min(12, int(seg_dur * 4)))
            prev_gray: np.ndarray | None = None
            diffs: list[float] = []

            for s in range(sample_count):
                t = v_start + (s / max(1, sample_count - 1)) * seg_dur
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
                ok, frame = cap.read()
                if not ok:
                    continue
                gray = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diffs.append(float(np.mean(cv2.absdiff(prev_gray, gray)) / 255.0))
                prev_gray = gray

            if diffs:
                scores.append(float(np.clip(np.mean(diffs) * 4.5, 0.0, 1.0)))
            else:
                scores.append(0.0)
    finally:
        cap.release()

    progress(
        f"Motion scores: {sum(1 for s in scores if s < 0.25)} calm, "
        f"{sum(1 for s in scores if s >= 0.55)} dynamic segment(s)",
        0.74,
    )
    return scores


def _merge_adjacent_cuts(
    cuts: list[float],
    motion_scores: list[float],
    *,
    max_segments: int,
) -> tuple[list[float], list[float]]:
    """Merge the calmest neighboring segments until segment count fits the beat budget."""
    if len(cuts) - 1 <= max_segments:
        return cuts, motion_scores

    merged_cuts = list(cuts)
    merged_scores = list(motion_scores)
    while len(merged_cuts) - 1 > max_segments and len(merged_scores) >= 2:
        merge_idx = min(
            range(len(merged_scores) - 1),
            key=lambda i: merged_scores[i] + merged_scores[i + 1],
        )
        combined = (
            merged_scores[merge_idx] * 0.45 + merged_scores[merge_idx + 1] * 0.55
        )
        merged_scores[merge_idx : merge_idx + 2] = [combined]
        merged_cuts.pop(merge_idx + 1)

    return merged_cuts, merged_scores


def _allocate_beat_counts_phase(
    motion_scores: list[float],
    v_durations: list[float],
    n_beats: int,
    *,
    slow: bool,
) -> list[int]:
    """Allocate beats across segments for one phase (pre- or post-drop)."""
    n_seg = len(motion_scores)
    if n_seg == 0:
        return []
    if n_beats <= 0:
        return [0] * n_seg
    if n_seg == 1:
        return [n_beats]

    weights: list[float] = []
    for motion, v_dur in zip(motion_scores, v_durations):
        calm = 1.0 - motion
        if slow:
            # Lingering, cinematic pacing before the drop.
            weights.append(max(0.12, v_dur * (0.35 + calm * 4.5 + motion * 0.15)))
        else:
            # Rapid cuts after the drop — dynamic shots stay on one beat.
            weights.append(max(0.06, v_dur * (0.15 + calm * 0.45 + motion * 1.8)))

    total_w = sum(weights)
    raw = [n_beats * w / total_w for w in weights]
    floor_min = 2 if slow else 1
    counts = [max(floor_min, int(math.floor(r))) for r in raw]

    remainder = n_beats - sum(counts)
    ranked = sorted(
        (
            (raw[i] - math.floor(raw[i]), (1.0 - motion_scores[i]) if slow else motion_scores[i], i)
            for i in range(n_seg)
        ),
        reverse=True,
    )
    for _, _, idx in ranked:
        if remainder <= 0:
            break
        counts[idx] += 1
        remainder -= 1

    while sum(counts) > n_beats:
        trim_idx = max(
            range(n_seg),
            key=lambda i: (
                (counts[i] if slow else -counts[i]),
                counts[i] / max(v_durations[i], 0.04),
            ),
        )
        if counts[trim_idx] <= floor_min:
            break
        counts[trim_idx] -= 1

    while sum(counts) < n_beats:
        add_idx = max(
            range(n_seg),
            key=lambda i: (
                ((1.0 - motion_scores[i]) if slow else motion_scores[i]) * v_durations[i],
                counts[i],
            ),
        )
        counts[add_idx] += 1

    return counts


def _allocate_beat_counts_beat_drop(
    motion_scores: list[float],
    v_durations: list[float],
    n_beats: int,
    drop_beat_idx: int,
) -> list[int]:
    """Slow sequence before the drop, rapid sequence after."""
    n_seg = len(motion_scores)
    if n_seg <= 1:
        return _allocate_beat_counts(motion_scores, v_durations, n_beats)

    drop_beat_idx = int(np.clip(drop_beat_idx, 1, max(1, n_beats - 1)))
    n_pre = drop_beat_idx
    n_post = max(1, n_beats - drop_beat_idx)

    drop_ratio = drop_beat_idx / max(n_beats, 1)
    pre_seg = max(1, min(n_seg - 1, round(n_seg * drop_ratio)))

    pre_counts = _allocate_beat_counts_phase(
        motion_scores[:pre_seg],
        v_durations[:pre_seg],
        n_pre,
        slow=True,
    )
    post_counts = _allocate_beat_counts_phase(
        motion_scores[pre_seg:],
        v_durations[pre_seg:],
        n_post,
        slow=False,
    )
    return pre_counts + post_counts


def _drop_beat_index(beats_local: list[float], drop_time: float | None) -> int | None:
    if drop_time is None or not beats_local:
        return None
    for i, beat in enumerate(beats_local):
        if beat >= drop_time - 0.02:
            return max(1, i)
    return max(1, len(beats_local) - 1)


def _allocate_beat_counts(
    motion_scores: list[float],
    v_durations: list[float],
    n_beats: int,
) -> list[int]:
    """Give calm/static segments more beats; keep dynamic segments snappy."""
    n_seg = len(motion_scores)
    if n_seg == 0:
        return []
    if n_beats <= 0:
        return [0] * n_seg
    if n_seg == 1:
        return [n_beats]

    weights: list[float] = []
    for motion, v_dur in zip(motion_scores, v_durations):
        calm = 1.0 - motion
        # Long, static shots linger across many beats; busy shots stay on one beat.
        weights.append(max(0.08, v_dur * (0.25 + calm * 3.0 + motion * 0.35)))

    total_w = sum(weights)
    raw = [n_beats * w / total_w for w in weights]
    counts = [max(1, int(math.floor(r))) for r in raw]

    remainder = n_beats - sum(counts)
    ranked = sorted(
        (
            (raw[i] - math.floor(raw[i]), 1.0 - motion_scores[i], i)
            for i in range(n_seg)
        ),
        reverse=True,
    )
    for _, _, idx in ranked:
        if remainder <= 0:
            break
        counts[idx] += 1
        remainder -= 1

    while sum(counts) > n_beats:
        trim_idx = max(
            range(n_seg),
            key=lambda i: (motion_scores[i], counts[i] / max(v_durations[i], 0.04)),
        )
        if counts[trim_idx] <= 1:
            break
        counts[trim_idx] -= 1

    while sum(counts) < n_beats:
        add_idx = max(
            range(n_seg),
            key=lambda i: ((1.0 - motion_scores[i]) * v_durations[i], counts[i]),
        )
        counts[add_idx] += 1

    return counts


@dataclass
class SegmentTimeMap:
    """Maps a slice of the working video timeline to the beat-synced output."""

    video_start: float
    video_end: float
    output_start: float
    output_end: float


def remap_time_through_sync(t: float, mappings: list[SegmentTimeMap]) -> float | None:
    for mapping in mappings:
        if mapping.video_start - 0.02 <= t <= mapping.video_end + 0.02:
            span = max(mapping.video_end - mapping.video_start, 1e-6)
            ratio = (t - mapping.video_start) / span
            out_span = mapping.output_end - mapping.output_start
            return mapping.output_start + ratio * out_span
    return None


def remap_speech_regions(
    regions: list,
    mappings: list[SegmentTimeMap],
    *,
    source_duration: float,
) -> list:
    """Re-time dialog regions after beat-sync warping."""
    from app.audio import SpeechRegion

    remapped: list[SpeechRegion] = []
    for region in regions:
        if region.end <= 0 or region.start >= source_duration:
            continue
        start = remap_time_through_sync(max(0.0, region.start), mappings)
        end = remap_time_through_sync(min(region.end, source_duration), mappings)
        if start is None or end is None or end - start < 0.08:
            continue
        remapped.append(
            SpeechRegion(
                start=start,
                end=end,
                text=region.text,
                is_likely_dialog=region.is_likely_dialog,
            )
        )
    return remapped


def sync_video_to_song(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    work_dir: Path,
    progress: ProgressCallback = default_progress,
    snippet_start: float = 0.0,
    snippet_end: float | None = None,
    beat_sync_mode: BeatSyncMode = "standard",
    beat_analysis: BeatAnalysis | None = None,
    time_map_out: list[SegmentTimeMap] | None = None,
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

    n_beats = max(1, len(song_beats_local) - 1)
    motion_scores = _segment_motion_scores(working, cuts, progress)
    cuts, motion_scores = _merge_adjacent_cuts(cuts, motion_scores, max_segments=n_beats)
    n_seg = len(cuts) - 1
    if n_seg < 1:
        run_ffmpeg(["-i", str(working), "-c:v", "copy", "-an", str(output_path)])
        return output_path

    v_durations = [max(0.04, cuts[i + 1] - cuts[i]) for i in range(n_seg)]

    drop_time_local: float | None = None
    if song.beat_drop_time is not None and start <= song.beat_drop_time <= end:
        drop_time_local = song.beat_drop_time - start

    if beat_sync_mode == "beat_drop" and drop_time_local is not None:
        drop_idx = _drop_beat_index(song_beats_local, drop_time_local)
        if drop_idx is not None:
            beat_counts = _allocate_beat_counts_beat_drop(
                motion_scores, v_durations, n_beats, drop_idx
            )
            progress(
                f"Beat-drop sync: slow until {drop_time_local:.1f}s, rapid after",
                0.752,
            )
        else:
            beat_counts = _allocate_beat_counts(motion_scores, v_durations, n_beats)
    else:
        beat_counts = _allocate_beat_counts(motion_scores, v_durations, n_beats)

    progress("Mapping video cuts to song beats…", 0.75)

    segment_files: list[Path] = []
    segment_maps: list[SegmentTimeMap] = []
    output_cursor = 0.0
    beat_cursor = 0
    for i in range(n_seg):
        beat_span = beat_counts[i]
        if beat_cursor + beat_span > len(song_beats_local) - 1:
            beat_span = max(1, len(song_beats_local) - 1 - beat_cursor)
        if beat_span < 1:
            break

        v_start, v_end = cuts[i], cuts[i + 1]
        s_start = song_beats_local[beat_cursor]
        s_end = song_beats_local[beat_cursor + beat_span]
        beat_cursor += beat_span

        v_dur = max(0.04, v_end - v_start)
        s_dur = max(0.04, s_end - s_start)
        speed = v_dur / s_dur

        segment_maps.append(
            SegmentTimeMap(
                video_start=v_start,
                video_end=v_end,
                output_start=output_cursor,
                output_end=output_cursor + s_dur,
            )
        )
        output_cursor += s_dur

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

    if not segment_files:
        run_ffmpeg(["-i", str(working), "-c:v", "copy", "-an", str(output_path)])
        return output_path

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

    if time_map_out is not None:
        time_map_out.clear()
        time_map_out.extend(segment_maps)

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
