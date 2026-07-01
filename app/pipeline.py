"""End-to-end processing pipeline."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from app.audio import (
    LyricLine,
    SpeechRegion,
    build_mixed_audio_track,
    clip_lyrics_to_snippet,
    detect_speech_regions,
    extract_audio_from_file,
    isolate_dialog_from_mixed,
    render_dialog_on_output_timeline,
    replace_audio,
    replace_audio_with_dialog,
    transcribe_lyrics,
    video_has_audio,
)
from app.beat_sync import (
    BeatAnalysis,
    BeatSyncMode,
    analyze_audio_beats,
    extract_audio_snippet,
    remap_speech_regions,
    sync_video_to_song,
)
from app.downloader import download_tiktok
from app.inpainting import remove_text_from_video
from app.lyrics_overlay import overlay_lyrics
from app.storage import EditRecord, Library, OverlayTweak, SongRecord, SourceRecord
from app.text_detection import FontStyle, detect_text_regions
from app.utils import ProgressCallback, default_progress, ensure_dir


@dataclass
class PipelineConfig:
    """Full Create-tab run: download/inpaint → beat-sync → audio swap → lyrics overlay."""

    tiktok_url: str
    replacement_audio: Path
    output_path: Path
    work_dir: Path
    per_frame_text_detection: bool = False  # slower OCR per frame when captions move
    skip_download: bool = False
    source_video: Path | None = None  # local file instead of TikTok URL
    snippet_start: float = 0.0
    snippet_end: float | None = None
    lyrics_override: list[LyricLine] | None = None  # user-edited timestamps from Songs tab
    overlay_tweak: OverlayTweak | None = None  # Studio tab re-render overrides
    beat_sync: bool = True
    beat_sync_mode: BeatSyncMode = "standard"
    preserve_dialog: bool = False
    library: Library | None = None
    song_id: str | None = None
    source_id: str | None = None  # reuse inpainted clip from Sources library


@dataclass
class PipelineResult:
    output_path: Path
    font_style: FontStyle
    steps_completed: list[str]
    lyrics: list[LyricLine]
    song_record: SongRecord | None = None
    source_record: SourceRecord | None = None
    edit_record: EditRecord | None = None
    with_audio_path: Path | None = None


@dataclass
class SourcePipelineConfig:
    tiktok_url: str
    work_dir: Path
    per_frame_text_detection: bool = False
    library: Library | None = None


@dataclass
class SourcePipelineResult:
    source_record: SourceRecord
    font_style: FontStyle
    steps_completed: list[str]


def run_source_pipeline(
    config: SourcePipelineConfig,
    progress: ProgressCallback = default_progress,
) -> SourcePipelineResult:
    """Download a TikTok, remove on-screen text, and save to the sources library."""
    ensure_dir(config.work_dir)
    if config.library is None:
        raise ValueError("Library is required to save a source")

    steps: list[str] = []
    source = download_tiktok(config.tiktok_url, config.work_dir / "downloads", progress)
    steps.append("download")

    style = detect_text_regions(source, progress)
    steps.append("text_detection")

    cleaned = config.work_dir / "cleaned_no_text.mp4"
    remove_text_from_video(
        source,
        cleaned,
        style=style,
        progress=progress,
        use_per_frame_detection=config.per_frame_text_detection,
    )
    steps.append("inpainting")

    source_record = config.library.add_source(
        cleaned,
        config.tiktok_url,
        style,
        title=Path(source).stem,
    )

    return SourcePipelineResult(
        source_record=source_record,
        font_style=style,
        steps_completed=steps,
    )


def run_pipeline(config: PipelineConfig, progress: ProgressCallback = default_progress) -> PipelineResult:
    """Run the full remix pipeline and persist song/source/edit records when a library is set."""
    ensure_dir(config.work_dir)
    lib = config.library
    steps: list[str] = []

    # Beat analysis overlaps with download/inpaint when beat-sync is enabled.
    beat_info: BeatAnalysis | None = None
    beat_holder: list[BeatAnalysis] = []
    beat_error: list[BaseException] = []
    beat_thread: threading.Thread | None = None
    if config.beat_sync:
        def _analyze_beats() -> None:
            try:
                beat_holder.append(analyze_audio_beats(config.replacement_audio, progress))
            except BaseException as exc:
                beat_error.append(exc)

        beat_thread = threading.Thread(target=_analyze_beats, daemon=True)
        beat_thread.start()

    source_record: SourceRecord | None = None
    if lib is not None and config.source_id:
        source_record = lib.get_source(config.source_id)

    speech_regions: list[SpeechRegion] = []
    original_speech_regions: list[SpeechRegion] = []
    dialog_track: Path | None = None
    source_duration: float = 0.0

    if source_record and Path(source_record.path).exists():
        cleaned = Path(source_record.path)
        style = source_record.font_style
        progress("Using library source (inpainted)", 0.20)
        steps.extend(["download_skip", "text_detection_skip", "inpainting_skip"])
        if config.preserve_dialog:
            progress(
                "Dialog preservation needs the original TikTok download — skipped for library source",
                0.12,
            )
    else:
        # 1. Download
        if config.skip_download and config.source_video and config.source_video.exists():
            source = config.source_video
            progress("Using existing video file", 0.15)
        else:
            source = download_tiktok(config.tiktok_url, config.work_dir / "downloads", progress)
        steps.append("download")

        # Optional: preserve dialog from original mixed audio before inpainting strips it
        if config.preserve_dialog and video_has_audio(source):
            original_audio = config.work_dir / "original_mixed_audio.mp3"
            progress("Extracting original audio for dialog detection…", 0.12)
            extract_audio_from_file(source, original_audio)
            from app.utils import get_video_info

            _, _, _, source_duration = get_video_info(source)
            speech_regions = detect_speech_regions(original_audio, progress)
            original_speech_regions = list(speech_regions)
            dialog_regions = [r for r in speech_regions if r.is_likely_dialog]
            if dialog_regions:
                dialog_track = config.work_dir / "isolated_dialog.mp3"
                isolate_dialog_from_mixed(original_audio, speech_regions, dialog_track, progress)
                progress(
                    f"Will preserve {len(dialog_regions)} dialog region(s) over new song",
                    0.19,
                )
            else:
                progress("No dialog detected — using full song replacement", 0.19)
                speech_regions = []
        elif config.preserve_dialog:
            progress("Source has no audio — dialog preservation skipped", 0.12)

        # 2. Detect text style before removal (preserve font info)
        style = detect_text_regions(source, progress)
        steps.append("text_detection")

        # 3. Remove text + audio via inpainting
        cleaned = config.work_dir / "cleaned_no_text.mp4"
        remove_text_from_video(
            source,
            cleaned,
            style=style,
            progress=progress,
            use_per_frame_detection=config.per_frame_text_detection,
        )
        steps.append("inpainting")

        if lib is not None:
            source_record = lib.add_source(
                cleaned,
                config.tiktok_url,
                style,
                title=Path(source).stem if source else None,
            )

    # 4. Beat-sync video to song snippet (loop + map cuts to beats)
    synced = cleaned
    beat_time_map: list = []
    if config.beat_sync:
        if beat_thread is not None:
            beat_thread.join()
            if beat_error:
                raise beat_error[0]
            beat_info = beat_holder[0] if beat_holder else None
        synced = config.work_dir / "beat_synced.mp4"
        sync_video_to_song(
            cleaned,
            config.replacement_audio,
            synced,
            config.work_dir,
            progress=progress,
            snippet_start=config.snippet_start,
            snippet_end=config.snippet_end,
            beat_sync_mode=config.beat_sync_mode,
            beat_analysis=beat_info,
            time_map_out=beat_time_map,
        )
        if beat_time_map and speech_regions and source_duration > 0:
            speech_regions = remap_speech_regions(
                speech_regions,
                beat_time_map,
                source_duration=source_duration,
            )
            progress("Re-timed dialog regions for beat-synced video", 0.781)
        steps.append("beat_sync")
    elif config.snippet_start > 0 or config.snippet_end is not None:
        synced = config.work_dir / "trimmed_video.mp4"
        from app.utils import run_ffmpeg, get_video_info

        _, _, _, dur = get_video_info(cleaned)
        end = config.snippet_end
        if end is None:
            from app.beat_sync import _audio_duration

            end = config.snippet_start + _audio_duration(config.replacement_audio)
        seg_dur = min(dur, (end or dur) - config.snippet_start)
        run_ffmpeg(
            [
                "-i",
                str(cleaned),
                "-t",
                str(seg_dur),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-an",
                str(synced),
            ]
        )

    # 5. Replace audio (or mix new song with preserved dialog)
    with_audio = config.work_dir / "with_new_audio.mp4"
    dialog_regions = [r for r in speech_regions if r.is_likely_dialog]
    if dialog_track and dialog_track.is_file() and dialog_regions:
        from app.utils import get_video_info

        _, _, _, video_dur = get_video_info(synced)
        aligned_dialog = dialog_track
        if beat_time_map and original_speech_regions:
            aligned_dialog = config.work_dir / "dialog_aligned.mp3"
            render_dialog_on_output_timeline(
                dialog_track,
                original_speech_regions,
                speech_regions,
                video_dur,
                aligned_dialog,
                progress,
            )
        mixed = config.work_dir / "mixed_audio.m4a"
        build_mixed_audio_track(
            config.replacement_audio,
            aligned_dialog,
            speech_regions,
            video_dur,
            mixed,
            progress,
            snippet_start=config.snippet_start,
            snippet_end=config.snippet_end,
        )
        replace_audio_with_dialog(synced, mixed, with_audio, progress)
        steps.append("audio_replace_dialog")
    else:
        replace_audio(
            synced,
            config.replacement_audio,
            with_audio,
            progress,
            snippet_start=config.snippet_start,
            snippet_end=config.snippet_end,
        )
        steps.append("audio_replace")

    # 6. Overlay lyrics — reuse saved/edited lines when available; transcribe only as fallback
    lyrics: list[LyricLine]
    if config.lyrics_override:
        lyrics = clip_lyrics_to_snippet(
            config.lyrics_override,
            config.snippet_start,
            config.snippet_end,
        )
        if lyrics:
            progress(f"Using {len(lyrics)} saved lyric line(s)", 0.75)
        else:
            progress("Saved lyrics don't cover this snippet — transcribing…", 0.70)
            audio_tmp = config.work_dir / "replacement.mp3"
            if config.snippet_start > 0 or config.snippet_end is not None:
                extract_audio_snippet(
                    config.replacement_audio,
                    audio_tmp,
                    config.snippet_start,
                    config.snippet_end,
                )
            else:
                extract_audio_from_file(config.replacement_audio, audio_tmp)
            lyrics = transcribe_lyrics(audio_tmp, progress)
    else:
        audio_tmp = config.work_dir / "replacement.mp3"
        if config.snippet_start > 0 or config.snippet_end is not None:
            extract_audio_snippet(
                config.replacement_audio,
                audio_tmp,
                config.snippet_start,
                config.snippet_end,
            )
        else:
            extract_audio_from_file(config.replacement_audio, audio_tmp)
        lyrics = transcribe_lyrics(audio_tmp, progress)

    overlay_lyrics(
        with_audio,
        lyrics,
        style,
        config.output_path,
        progress,
        tweak=config.overlay_tweak,
        snippet_start=0.0,
    )
    steps.append("lyrics_overlay")

    song_record: SongRecord | None = None
    edit_record: EditRecord | None = None
    if lib is not None:
        if beat_info is None:
            beat_info = analyze_audio_beats(config.replacement_audio, progress)
        if config.song_id:
            song_record = lib.get_song(config.song_id)
            if song_record:
                song_record.lyrics = lyrics
                song_record.bpm = beat_info.bpm
                song_record.snippet_start = config.snippet_start
                song_record.snippet_end = config.snippet_end
                lib.update_song(song_record)
        else:
            song_record = lib.add_song(
                config.replacement_audio,
                lyrics=lyrics,
                bpm=beat_info.bpm,
                title=config.replacement_audio.stem,
            )
            song_record.snippet_start = config.snippet_start
            song_record.snippet_end = config.snippet_end
            lib.update_song(song_record)

        if source_record and song_record:
            edit_record = lib.add_edit(
                config.output_path,
                with_audio,
                source_record.id,
                song_record.id,
                style,
                lyrics,
                tweak=config.overlay_tweak or OverlayTweak(),
                snippet_start=config.snippet_start,
                snippet_end=config.snippet_end,
                title=config.output_path.stem,
            )

    return PipelineResult(
        output_path=config.output_path,
        font_style=style,
        steps_completed=steps,
        lyrics=lyrics,
        song_record=song_record,
        source_record=source_record,
        edit_record=edit_record,
        with_audio_path=with_audio,
    )
