"""Modern tabbed GUI for the EditAutomate video remix pipeline."""

from __future__ import annotations

import platform
import subprocess
import threading
import uuid
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

import customtkinter as ctk

from app.audio import (
    LyricLine,
    clip_lyrics_to_snippet,
    extract_audio_from_file,
    merge_lyrics_range,
    transcribe_lyrics,
)
from app.beat_sync import analyze_audio_beats, extract_audio_snippet
from app.lyrics_overlay import re_render_edit
from app.pipeline import PipelineConfig, PipelineResult, SourcePipelineConfig, run_pipeline, run_source_pipeline
from app.storage import Library, OverlayTweak
from app.caption_timeline import CaptionTimeline
from app.studio_preview import StudioPreview
from app.video_preview import VideoPreview
from app.fonts import TIKTOK_FONT_CANDIDATES
from app.utils import ensure_dir, get_video_info, work_dir
from app.waveform import WaveformSelector


@dataclass
class QueuedEdit:
    """Background Create-tab job tracked in the queue UI."""

    id: str
    config: PipelineConfig
    title: str
    status: str = "queued"  # queued | running | done | error
    step: str = "Waiting…"
    fraction: float = 0.0
    error: str | None = None
    result: PipelineResult | None = None
    widgets: dict = field(default_factory=dict)  # row widgets for live progress updates


@dataclass
class QueuedSource:
    """Background Sources-tab download + inpaint job."""

    id: str
    config: SourcePipelineConfig
    url: str
    title: str
    status: str = "running"
    step: str = "Starting…"
    fraction: float = 0.0
    error: str | None = None
    result: object | None = None
    widgets: dict = field(default_factory=dict)

# Studio dark palette
BG = "#08080e"
SURFACE = "#12121a"
SURFACE_RAISED = "#1a1a26"
BORDER = "#2a2a3c"
ACCENT = "#00e5c0"
ACCENT_HOVER = "#00c9aa"
ACCENT_DIM = "#0d3d36"
TEXT = "#eeeef4"
TEXT_MUTED = "#8b8ba3"
TEXT_DIM = "#55556a"
LOG_BG = "#0c0c14"
SUCCESS = "#3dd68c"
WARNING = "#f0b429"


class EditAutomateApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("EditAutomate — TikTok Remix Studio")
        self.geometry("1020x860")
        self.minsize(900, 720)
        self.configure(fg_color=BG)

        self._bg_busy = False
        self._last_output: Path | None = None
        self._work = work_dir()
        self._library = Library(self._work)
        self._progress_fraction = 0.0
        self._selected_song_id: str | None = None
        self._selected_source_id: str | None = None
        self._selected_edit_id: str | None = None
        self._selected_song_lyrics: list[LyricLine] = []
        self._selected_snippet_id: str | None = None
        self._snippet_picker_values: list[str] = []
        self._jobs: dict[str, QueuedEdit] = {}
        self._job_order: list[str] = []
        self._source_jobs: dict[str, QueuedSource] = {}
        self._source_job_order: list[str] = []
        self._preview_source_id: str | None = None
        self._studio_lyrics: list[LyricLine] = []
        self._studio_duration = 30.0
        self._studio_syncing = False
        self._studio_font_style = None

        self._build_ui()
        self._refresh_songs_list()
        self._refresh_sources_list()
        self._refresh_edits_list()

    # --- Shared UI helpers ---

    def _section_label(self, parent: ctk.CTkFrame, text: str, row: int) -> None:
        ctk.CTkLabel(
            parent,
            text=text.upper(),
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT_DIM,
            anchor="w",
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=20, pady=(14, 4))

    def _field_label(self, parent: ctk.CTkFrame, text: str, row: int) -> None:
        ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=13),
            text_color=TEXT_MUTED,
            anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=20, pady=6)

    def _styled_entry(self, parent: ctk.CTkFrame, **kwargs: object) -> ctk.CTkEntry:
        return ctk.CTkEntry(
            parent,
            height=38,
            corner_radius=8,
            border_width=1,
            border_color=BORDER,
            fg_color=SURFACE_RAISED,
            text_color=TEXT,
            placeholder_text_color=TEXT_DIM,
            **kwargs,
        )

    def _outline_btn(self, parent: ctk.CTkFrame, text: str, command: object, **kwargs: object) -> ctk.CTkButton:
        height = kwargs.pop("height", 38)
        width = kwargs.pop("width", 96)
        return ctk.CTkButton(
            parent,
            text=text,
            height=height,
            width=width,
            corner_radius=8,
            fg_color="transparent",
            border_width=1,
            border_color=BORDER,
            hover_color=SURFACE_RAISED,
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=13),
            command=command,
            **kwargs,
        )

    def _accent_btn(self, parent: ctk.CTkFrame, text: str, command: object, **kwargs: object) -> ctk.CTkButton:
        height = kwargs.pop("height", 40)
        font = kwargs.pop("font", None) or ctk.CTkFont(size=13, weight="bold")
        return ctk.CTkButton(
            parent,
            text=text,
            height=height,
            corner_radius=8,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=BG,
            font=font,
            command=command,
            **kwargs,
        )

    def _pipeline_chip(self, parent: ctk.CTkFrame, text: str) -> ctk.CTkLabel:
        return ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=ACCENT,
            fg_color=ACCENT_DIM,
            corner_radius=20,
            padx=12,
            pady=4,
        )

    def _scrollable_tab(self, name: str) -> ctk.CTkScrollableFrame:
        """Full-tab vertical scroll so stacked sections stay reachable."""
        container = self.tabs.tab(name)
        container.configure(fg_color=SURFACE)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)
        scroll = ctk.CTkScrollableFrame(
            container,
            fg_color=SURFACE,
            corner_radius=0,
            border_width=0,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=TEXT_DIM,
        )
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)
        return scroll

    def _list_panel(
        self,
        parent: ctk.CTkFrame,
        *,
        fg_color: str = SURFACE,
        corner_radius: int = 8,
        border_width: int = 0,
        border_color: str = BORDER,
    ) -> ctk.CTkFrame:
        """Expandable list container (outer tab scroll handles overflow)."""
        frame = ctk.CTkFrame(
            parent,
            fg_color=fg_color,
            corner_radius=corner_radius,
            border_width=border_width,
            border_color=border_color,
        )
        frame.grid_columnconfigure(0, weight=1)
        return frame

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        accent_bar = ctk.CTkFrame(self, height=3, corner_radius=0, fg_color=ACCENT)
        accent_bar.grid(row=0, column=0, sticky="ew")

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=1, column=0, sticky="ew", padx=28, pady=(18, 4))
        header.grid_columnconfigure(0, weight=1)

        title_row = ctk.CTkFrame(header, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(title_row, text="Edit", font=ctk.CTkFont(size=28, weight="bold"), text_color=TEXT).pack(side="left")
        ctk.CTkLabel(title_row, text="Automate", font=ctk.CTkFont(size=28, weight="bold"), text_color=ACCENT).pack(side="left")

        chips = ctk.CTkFrame(header, fg_color="transparent")
        chips.grid(row=1, column=0, sticky="w", pady=(4, 0))
        for i, step in enumerate(("Download", "Beat Sync", "Inpaint", "Lyrics")):
            if i:
                ctk.CTkLabel(chips, text="→", text_color=TEXT_DIM, font=ctk.CTkFont(size=11)).pack(side="left", padx=6)
            self._pipeline_chip(chips, step).pack(side="left")

        self.tabs = ctk.CTkTabview(
            self,
            fg_color=SURFACE,
            segmented_button_fg_color=SURFACE_RAISED,
            segmented_button_selected_color=ACCENT_DIM,
            segmented_button_selected_hover_color=ACCENT_DIM,
            segmented_button_unselected_color=SURFACE_RAISED,
            segmented_button_unselected_hover_color=BORDER,
            text_color=TEXT_MUTED,
        )
        self.tabs.grid(row=2, column=0, sticky="nsew", padx=28, pady=8)
        self.tabs.add("Create")
        self.tabs.add("Songs")
        self.tabs.add("Sources")
        self.tabs.add("Studio")

        self._build_create_tab()
        self._build_songs_tab()
        self._build_sources_tab()
        self._build_studio_tab()
        self._build_progress_footer()
        self.bind("<Command-Return>", lambda _e: self._try_generate_shortcut())
        self.bind("<Control-Return>", lambda _e: self._try_generate_shortcut())

    def _build_create_tab(self) -> None:
        tab = self._scrollable_tab("Create")
        tab.grid_columnconfigure(0, weight=1)

        hero = ctk.CTkFrame(tab, fg_color=ACCENT_DIM, corner_radius=12)
        hero.grid(row=0, column=0, sticky="ew", pady=(8, 8))
        hero.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            hero,
            text="Promo edit workflow",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=ACCENT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(10, 0))
        flow = ctk.CTkFrame(hero, fg_color="transparent")
        flow.grid(row=1, column=0, sticky="w", padx=12, pady=(4, 10))
        for i, step in enumerate(("Song snippet", "TikTok clip", "Generate")):
            if i:
                ctk.CTkLabel(flow, text="→", text_color=TEXT_DIM, font=ctk.CTkFont(size=10)).pack(side="left", padx=4)
            self._pipeline_chip(flow, step).pack(side="left")
        ctk.CTkLabel(
            flow,
            text="9:16 · 15–60s",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_MUTED,
        ).pack(side="left", padx=(10, 0))

        form = ctk.CTkFrame(tab, fg_color=SURFACE_RAISED, corner_radius=12, border_width=1, border_color=BORDER)
        form.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        form.grid_columnconfigure(1, weight=1)

        self._section_label(form, "Clip source", 0)

        self._field_label(form, "TikTok URL", 1)
        self.url_entry = self._styled_entry(form, placeholder_text="https://www.tiktok.com/@user/video/…")
        self.url_entry.grid(row=1, column=1, sticky="ew", padx=(0, 20), pady=6)

        self._field_label(form, "Or library source", 2)
        src_row = ctk.CTkFrame(form, fg_color="transparent")
        src_row.grid(row=2, column=1, sticky="ew", padx=(0, 20), pady=6)
        src_row.grid_columnconfigure(0, weight=1)
        self.source_pick = ctk.CTkComboBox(
            src_row,
            values=["(Download new TikTok)"],
            height=38,
            corner_radius=8,
            border_color=BORDER,
            fg_color=SURFACE,
            button_color=BORDER,
            button_hover_color=ACCENT_DIM,
            dropdown_fg_color=SURFACE_RAISED,
            command=self._on_source_pick,
        )
        self.source_pick.set("(Download new TikTok)")
        self.source_pick.grid(row=0, column=0, sticky="ew")

        self._section_label(form, "Song & output", 3)

        self._field_label(form, "Your song", 4)
        audio_row = ctk.CTkFrame(form, fg_color="transparent")
        audio_row.grid(row=4, column=1, sticky="ew", padx=(0, 20), pady=6)
        audio_row.grid_columnconfigure(0, weight=1)
        self.audio_entry = self._styled_entry(audio_row, placeholder_text="MP3, M4A, WAV — use Songs tab for snippet + lyrics")
        self.audio_entry.grid(row=0, column=0, sticky="ew")
        self._outline_btn(audio_row, "Browse", self._browse_audio).grid(row=0, column=1, padx=(10, 0))
        self._outline_btn(audio_row, "From Library", self._pick_song_from_library, width=110).grid(row=0, column=2, padx=(10, 0))

        self._field_label(form, "Save as", 5)
        out_row = ctk.CTkFrame(form, fg_color="transparent")
        out_row.grid(row=5, column=1, sticky="ew", padx=(0, 20), pady=6)
        out_row.grid_columnconfigure(0, weight=1)
        default_out = Path.home() / "Movies" / "EditAutomate" / f"remix_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        self.output_entry = self._styled_entry(out_row)
        self.output_entry.insert(0, str(default_out))
        self.output_entry.grid(row=0, column=0, sticky="ew")
        self._outline_btn(out_row, "Browse", self._browse_output).grid(row=0, column=1, padx=(10, 0))

        opts = ctk.CTkFrame(form, fg_color=SURFACE, corner_radius=8)
        opts.grid(row=6, column=0, columnspan=2, sticky="ew", padx=20, pady=(8, 12))
        self.per_frame_var = ctk.BooleanVar(value=False)
        self.beat_sync_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            opts, text="Beat-sync cuts to song BPM (recommended for promo edits)",
            variable=self.beat_sync_var, font=ctk.CTkFont(size=12), text_color=TEXT_MUTED,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, border_color=BORDER, checkmark_color=BG,
        ).pack(anchor="w", padx=14, pady=(12, 4))
        ctk.CTkCheckBox(
            opts, text="Per-frame text detection (slower — moving captions only)",
            variable=self.per_frame_var, font=ctk.CTkFont(size=12), text_color=TEXT_MUTED,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, border_color=BORDER, checkmark_color=BG,
        ).pack(anchor="w", padx=14, pady=(4, 12))

        actions = ctk.CTkFrame(tab, fg_color=SURFACE_RAISED, corner_radius=12, border_width=1, border_color=BORDER)
        actions.grid(row=2, column=0, sticky="ew", pady=(4, 6))
        actions.grid_columnconfigure(0, weight=3)
        actions.grid_columnconfigure(1, weight=1)
        actions.grid_columnconfigure(2, weight=1)

        self.generate_btn = self._accent_btn(
            actions, "▶  Generate Edit", self._generate_edit, height=56,
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.generate_btn.grid(row=0, column=0, sticky="ew", padx=(14, 8), pady=14)
        self.queue_btn = self._outline_btn(actions, "Add to Queue", self._enqueue_edit, height=56, width=140)
        self.queue_btn.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=14)
        self.run_queue_btn = self._outline_btn(actions, "Run Queue", self._run_queue, height=56, width=120)
        self.run_queue_btn.grid(row=0, column=2, sticky="ew", padx=(0, 14), pady=14)

        out_actions = ctk.CTkFrame(tab, fg_color="transparent")
        out_actions.grid(row=3, column=0, sticky="ew", pady=(0, 4))
        self.open_btn = self._outline_btn(out_actions, "Open Output", self._open_output, height=36, state="disabled")
        self.open_btn.pack(side="left", padx=(0, 8))
        self.share_btn = self._outline_btn(out_actions, "Reveal in Finder", self._reveal_output, height=36, state="disabled")
        self.share_btn.pack(side="left")

        queue_hdr = ctk.CTkFrame(tab, fg_color="transparent")
        queue_hdr.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        ctk.CTkLabel(
            queue_hdr, text="Batch queue", font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_MUTED,
        ).pack(side="left")
        self.queue_stats_label = ctk.CTkLabel(
            queue_hdr,
            text="No jobs",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_DIM,
        )
        self.queue_stats_label.pack(side="left", padx=(10, 0))
        self._outline_btn(queue_hdr, "Clear done", self._clear_finished_jobs, width=90).pack(side="right")

        self.queue_frame = self._list_panel(
            tab, fg_color=SURFACE_RAISED, corner_radius=10, border_width=1,
        )
        self.queue_frame.grid(row=5, column=0, sticky="ew", pady=(4, 8))
        self._queue_empty_label = ctk.CTkLabel(
            self.queue_frame,
            text="Batch jobs appear here — Generate starts immediately",
            text_color=TEXT_DIM,
            font=ctk.CTkFont(size=11),
        )
        self._queue_empty_label.pack(pady=16, padx=12)
        self._update_queue_stats()

    def _build_songs_tab(self) -> None:
        tab = self._scrollable_tab("Songs")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=2)

        left = ctk.CTkFrame(tab, fg_color=SURFACE_RAISED, corner_radius=12, border_width=1, border_color=BORDER)
        left.grid(row=0, column=0, sticky="new", padx=(8, 4), pady=8)
        left.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(left, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        ctk.CTkLabel(hdr, text="Your Songs", font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT).pack(side="left")
        self._outline_btn(hdr, "+ Upload", self._upload_song, width=90).pack(side="right")

        self.songs_list = self._list_panel(left, fg_color=SURFACE)
        self.songs_list.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))

        right = ctk.CTkFrame(tab, fg_color=SURFACE_RAISED, corner_radius=12, border_width=1, border_color=BORDER)
        right.grid(row=0, column=1, sticky="new", padx=(4, 8), pady=8)
        right.grid_columnconfigure(0, weight=1)

        self.song_detail_title = ctk.CTkLabel(right, text="Select a song", font=ctk.CTkFont(size=16, weight="bold"), text_color=TEXT)
        self.song_detail_title.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 4))

        self.song_bpm_label = ctk.CTkLabel(right, text="", font=ctk.CTkFont(size=12), text_color=TEXT_DIM)
        self.song_bpm_label.grid(row=1, column=0, sticky="w", padx=16, pady=(0, 6))

        flow = ctk.CTkFrame(right, fg_color="transparent")
        flow.grid(row=2, column=0, sticky="w", padx=16, pady=(0, 8))
        for i, step in enumerate(("Pick song", "Select snippet", "Edit lyrics", "Generate")):
            if i:
                ctk.CTkLabel(flow, text="→", text_color=TEXT_DIM, font=ctk.CTkFont(size=10)).pack(side="left", padx=4)
            chip = ctk.CTkLabel(
                flow, text=step, font=ctk.CTkFont(size=10, weight="bold"),
                text_color=ACCENT if i < 2 else TEXT_DIM, fg_color=ACCENT_DIM if i < 2 else SURFACE,
                corner_radius=12, padx=8, pady=3,
            )
            chip.pack(side="left")

        ctk.CTkLabel(
            right, text="SNIPPET — drag handles on waveform", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_DIM,
        ).grid(row=3, column=0, sticky="nw", padx=16, pady=(0, 4))

        snippet_row = ctk.CTkFrame(right, fg_color="transparent")
        snippet_row.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 6))
        snippet_row.grid_columnconfigure(0, weight=1)
        self.snippet_picker = ctk.CTkOptionMenu(
            snippet_row,
            values=["(New selection)"],
            command=self._on_snippet_picked,
            fg_color=SURFACE,
            button_color=BORDER,
            button_hover_color=ACCENT_DIM,
            dropdown_fg_color=SURFACE_RAISED,
            dropdown_hover_color=BORDER,
            text_color=TEXT,
            font=ctk.CTkFont(size=12),
            width=200,
        )
        self.snippet_picker.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._outline_btn(snippet_row, "Save Snippet", self._save_named_snippet, width=100).grid(row=0, column=1, padx=(0, 4))
        self._outline_btn(snippet_row, "Delete", self._delete_snippet, width=70).grid(row=0, column=2)

        self.waveform = WaveformSelector(right, on_change=self._on_waveform_change)
        self.waveform.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 8))

        ctk.CTkLabel(
            right, text="LYRICS FOR SNIPPET (timestamps relative to selection)", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_DIM,
        ).grid(row=6, column=0, sticky="nw", padx=16, pady=(0, 4))

        self.lyrics_editor = ctk.CTkTextbox(
            right, height=180, font=ctk.CTkFont(family="Menlo", size=12), fg_color=LOG_BG,
            text_color=TEXT_MUTED, corner_radius=8, border_width=1, border_color=BORDER,
        )
        self.lyrics_editor.grid(row=7, column=0, sticky="ew", padx=16, pady=(0, 8))

        btn_row = ctk.CTkFrame(right, fg_color="transparent")
        btn_row.grid(row=8, column=0, sticky="ew", padx=16, pady=(0, 16))
        self._outline_btn(btn_row, "Transcribe Snippet", self._transcribe_selected_song).pack(side="left", padx=(0, 8))
        for secs, label in ((15, "15s"), (30, "30s"), (60, "60s")):
            self._outline_btn(
                btn_row, label, lambda s=secs: self._apply_snippet_preset(s), width=52,
            ).pack(side="left", padx=(0, 4))
        self._accent_btn(btn_row, "Use in Create →", self._use_song_in_create).pack(side="right")

    def _build_sources_tab(self) -> None:
        tab = self._scrollable_tab("Sources")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=2)

        hdr = ctk.CTkFrame(tab, fg_color="transparent")
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 0))
        ctk.CTkLabel(
            hdr,
            text="Inpainted source files — TikTok edits with text removed, ready to remix",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_MUTED,
        ).pack(side="left")
        self._outline_btn(hdr, "Clear Finished", self._clear_finished_source_jobs, width=100).pack(side="right", padx=(8, 0))
        self._outline_btn(hdr, "Refresh", self._refresh_sources_list, width=80).pack(side="right")

        add_form = ctk.CTkFrame(tab, fg_color=SURFACE_RAISED, corner_radius=12, border_width=1, border_color=BORDER)
        add_form.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        add_form.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            add_form,
            text="Add source",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(14, 4))

        url_row = ctk.CTkFrame(add_form, fg_color="transparent")
        url_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 8))
        url_row.grid_columnconfigure(0, weight=1)

        self.source_url_entry = self._styled_entry(
            url_row,
            placeholder_text="Paste TikTok URL — https://www.tiktok.com/@user/video/…",
        )
        self.source_url_entry.grid(row=0, column=0, sticky="ew")
        self.source_url_entry.bind("<Return>", lambda _e: self._add_source())
        self._accent_btn(url_row, "Download & Remove Text", self._add_source, height=38).grid(
            row=0, column=1, padx=(10, 0)
        )

        self.source_per_frame_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            add_form,
            text="Per-frame text detection (slower — moving captions only)",
            variable=self.source_per_frame_var,
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            border_color=BORDER,
            checkmark_color=BG,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 14))

        jobs_hdr = ctk.CTkFrame(tab, fg_color="transparent")
        jobs_hdr.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 4))
        ctk.CTkLabel(
            jobs_hdr,
            text="IMPORTS",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT_DIM,
        ).pack(side="left")
        self.source_jobs_stats = ctk.CTkLabel(jobs_hdr, text="", font=ctk.CTkFont(size=11), text_color=TEXT_DIM)
        self.source_jobs_stats.pack(side="right")

        self.source_jobs_frame = self._list_panel(
            tab, fg_color=SURFACE_RAISED, corner_radius=12, border_width=1,
        )
        self.source_jobs_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))
        self._source_jobs_empty_label = ctk.CTkLabel(
            self.source_jobs_frame,
            text="Paste a link and hit Download — add as many as you want in parallel.",
            text_color=TEXT_DIM,
            font=ctk.CTkFont(size=11),
        )
        self._source_jobs_empty_label.pack(pady=12, padx=12)

        library_col = ctk.CTkFrame(tab, fg_color="transparent")
        library_col.grid(row=4, column=0, sticky="new", padx=(8, 4), pady=(0, 8))
        library_col.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            library_col,
            text="LIBRARY",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT_DIM,
            anchor="w",
        ).grid(row=0, column=0, sticky="nw", padx=8, pady=(0, 4))

        self.sources_list = self._list_panel(
            library_col, fg_color=SURFACE_RAISED, corner_radius=12, border_width=1,
        )
        self.sources_list.grid(row=1, column=0, sticky="ew", padx=0, pady=(0, 0))

        preview_col = ctk.CTkFrame(tab, fg_color=SURFACE_RAISED, corner_radius=12, border_width=1, border_color=BORDER)
        preview_col.grid(row=4, column=1, sticky="new", padx=(4, 8), pady=(0, 8))
        preview_col.grid_columnconfigure(0, weight=1)

        self.source_preview_title = ctk.CTkLabel(
            preview_col, text="Select a source", font=ctk.CTkFont(size=16, weight="bold"), text_color=TEXT,
        )
        self.source_preview_title.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 4))

        self.source_preview_meta = ctk.CTkLabel(
            preview_col, text="", font=ctk.CTkFont(size=12), text_color=TEXT_DIM, wraplength=420, justify="left",
        )
        self.source_preview_meta.grid(row=1, column=0, sticky="w", padx=16, pady=(0, 8))

        self.source_video_preview = VideoPreview(preview_col)
        self.source_video_preview.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))

        preview_actions = ctk.CTkFrame(preview_col, fg_color="transparent")
        preview_actions.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 16))
        self._outline_btn(preview_actions, "Use in Create", self._use_preview_source_in_create, width=120).pack(side="left", padx=(0, 8))
        self._outline_btn(preview_actions, "Reveal in Finder", self._reveal_preview_source, width=130).pack(side="left")

    def _build_studio_tab(self) -> None:
        tab = self._scrollable_tab("Studio")
        tab.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        top.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(top, text="Studio", font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT).grid(
            row=0, column=0, sticky="w",
        )
        self.studio_title = ctk.CTkLabel(
            top, text="Select a project to edit captions", font=ctk.CTkFont(size=13), text_color=TEXT_MUTED, anchor="w",
        )
        self.studio_title.grid(row=0, column=1, sticky="ew", padx=(12, 0))
        actions = ctk.CTkFrame(top, fg_color="transparent")
        actions.grid(row=0, column=2, sticky="e")
        self._outline_btn(actions, "Open Video", self._open_studio_output, width=100).pack(side="right", padx=(8, 0))
        self._accent_btn(actions, "Export", self._rerender_edit, height=36, width=100).pack(side="right")

        body = ctk.CTkFrame(tab, fg_color=SURFACE_RAISED, corner_radius=12, border_width=1, border_color=BORDER)
        body.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        body.grid_columnconfigure(0, weight=1)

        picker = ctk.CTkFrame(body, fg_color="transparent")
        picker.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        ctk.CTkLabel(picker, text="PROJECT", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_DIM).pack(side="left")
        self.studio_edit_pick = ctk.CTkComboBox(
            picker,
            values=["(No edits yet)"],
            height=36,
            width=320,
            corner_radius=8,
            border_color=BORDER,
            fg_color=SURFACE,
            button_color=BORDER,
            dropdown_fg_color=SURFACE_RAISED,
            command=self._on_studio_edit_pick,
        )
        self.studio_edit_pick.set("(No edits yet)")
        self.studio_edit_pick.pack(side="left", padx=(10, 0))

        self.studio_preview = StudioPreview(body, on_time_change=self._on_studio_preview_seek)
        self.studio_preview.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

        self.studio_timeline = CaptionTimeline(
            body,
            on_select=self._on_studio_caption_select,
            on_change=self._on_studio_timeline_change,
            on_seek=self._on_studio_timeline_seek,
        )
        self.studio_timeline.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))

        tl_tools = ctk.CTkFrame(body, fg_color="transparent")
        tl_tools.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))
        self._outline_btn(tl_tools, "+ Add Caption", self._studio_add_caption, width=110).pack(side="left", padx=(0, 8))
        self._outline_btn(tl_tools, "Delete", self._studio_delete_caption, width=80).pack(side="left")
        ctk.CTkLabel(
            tl_tools,
            text="Drag blocks to move · drag edges to trim · click timeline or preview to scrub",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_DIM,
        ).pack(side="right")

        editor = ctk.CTkFrame(tab, fg_color=SURFACE_RAISED, corner_radius=12, border_width=1, border_color=BORDER)
        editor.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        editor.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            editor, text="SELECTED CAPTION", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_DIM,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=16, pady=(14, 8))

        ctk.CTkLabel(editor, text="Text", font=ctk.CTkFont(size=12), text_color=TEXT_MUTED).grid(
            row=1, column=0, sticky="w", padx=16, pady=6,
        )
        self.studio_caption_text = self._styled_entry(editor, placeholder_text="Caption text…")
        self.studio_caption_text.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 16), pady=6)
        self.studio_caption_text.bind("<KeyRelease>", lambda _e: self._studio_caption_text_changed())

        timing = ctk.CTkFrame(editor, fg_color=SURFACE, corner_radius=8)
        timing.grid(row=2, column=0, columnspan=3, sticky="ew", padx=16, pady=(4, 12))
        for col, (label, attr) in enumerate((("Start", "studio_start"), ("End", "studio_end"), ("Duration", "studio_dur"))):
            cell = ctk.CTkFrame(timing, fg_color="transparent")
            cell.grid(row=0, column=col, sticky="ew", padx=(14 if col == 0 else 8, 8), pady=10)
            ctk.CTkLabel(cell, text=label, font=ctk.CTkFont(size=11), text_color=TEXT_DIM).pack(anchor="w")
            entry = self._styled_entry(cell)
            entry.pack(fill="x", pady=(4, 0))
            setattr(self, attr, entry)
        timing.grid_columnconfigure(0, weight=1)
        timing.grid_columnconfigure(1, weight=1)
        timing.grid_columnconfigure(2, weight=1)
        self.studio_dur.configure(state="disabled")
        self.studio_start.bind("<KeyRelease>", lambda _e: self._studio_timing_changed())
        self.studio_end.bind("<KeyRelease>", lambda _e: self._studio_timing_changed())

        style_hdr = ctk.CTkFrame(tab, fg_color="transparent")
        style_hdr.grid(row=3, column=0, sticky="ew", padx=16, pady=(8, 4))
        ctk.CTkLabel(style_hdr, text="TEXT STYLE", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_DIM).pack(side="left")

        style = ctk.CTkFrame(tab, fg_color=SURFACE_RAISED, corner_radius=12, border_width=1, border_color=BORDER)
        style.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))
        style.grid_columnconfigure(1, weight=1)

        row = 0
        for label, attr, from_, to_, default in (
            ("Position X", "tweak_x", -200, 200, 0),
            ("Position Y", "tweak_y", -200, 200, 0),
            ("Size", "tweak_size", 18, 120, 48),
            ("Outline", "tweak_stroke", 0, 12, 2),
        ):
            ctk.CTkLabel(style, text=label, font=ctk.CTkFont(size=12), text_color=TEXT_MUTED).grid(
                row=row, column=0, sticky="w", padx=16, pady=6,
            )
            lbl = ctk.CTkLabel(style, text=str(default), font=ctk.CTkFont(size=11), text_color=TEXT_DIM, width=44)
            lbl.grid(row=row, column=2, sticky="e", padx=16)
            slider = ctk.CTkSlider(
                style,
                from_=from_,
                to=to_,
                command=lambda v, l=lbl: (l.configure(text=f"{int(float(v))}"), self._studio_style_changed()),
            )
            slider.set(default)
            slider.grid(row=row, column=1, sticky="ew", padx=8, pady=6)
            setattr(self, attr, slider)
            setattr(self, f"{attr}_label", lbl)
            row += 1

        ctk.CTkLabel(style, text="Font", font=ctk.CTkFont(size=12), text_color=TEXT_MUTED).grid(
            row=row, column=0, sticky="w", padx=16, pady=6,
        )
        self.tweak_font = ctk.CTkComboBox(
            style, values=TIKTOK_FONT_CANDIDATES, height=36, corner_radius=8,
            border_color=BORDER, fg_color=SURFACE, button_color=BORDER,
        )
        self.tweak_font.set("Arial Narrow")
        self.tweak_font.grid(row=row, column=1, columnspan=2, sticky="ew", padx=16, pady=6)
        self.tweak_font.configure(command=lambda _v: self._studio_style_changed())
        row += 1

        self.tweak_stroke_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            style, text="Text outline", variable=self.tweak_stroke_var,
            font=ctk.CTkFont(size=12), text_color=TEXT_MUTED,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, border_color=BORDER, checkmark_color=BG,
            command=self._studio_style_changed,
        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=16, pady=(4, 14))

        self.edits_list = self._list_panel(tab, fg_color=SURFACE_RAISED, corner_radius=12, border_width=1)
        self.edits_list.grid(row=5, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.edits_list.grid_remove()

    def _build_progress_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=14, border_width=1, border_color=BORDER)
        footer.grid(row=3, column=0, sticky="ew", padx=28, pady=(0, 20))
        footer.grid_columnconfigure(0, weight=1)

        status_row = ctk.CTkFrame(footer, fg_color="transparent")
        status_row.grid(row=0, column=0, sticky="ew", padx=20, pady=(14, 6))
        status_row.grid_columnconfigure(1, weight=1)

        self._status_dot = ctk.CTkLabel(status_row, text="●", font=ctk.CTkFont(size=10), text_color=SUCCESS, width=16)
        self._status_dot.grid(row=0, column=0, sticky="w")
        self.status_label = ctk.CTkLabel(status_row, text="Ready", anchor="w", font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT)
        self.status_label.grid(row=0, column=1, sticky="w")
        self.pct_label = ctk.CTkLabel(status_row, text="0%", font=ctk.CTkFont(size=12), text_color=TEXT_DIM)
        self.pct_label.grid(row=0, column=2, sticky="e")

        self.progress = ctk.CTkProgressBar(footer, height=6, corner_radius=3, fg_color=SURFACE_RAISED, progress_color=ACCENT, border_width=0)
        self.progress.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))
        self.progress.set(0)

        self.log_box = ctk.CTkTextbox(
            footer, height=72, font=ctk.CTkFont(family="Menlo", size=11),
            fg_color=LOG_BG, text_color=TEXT_MUTED, corner_radius=8, border_width=1, border_color=BORDER,
        )
        self.log_box.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 14))
        self.log_box.configure(state="disabled")
        self._log("Welcome! Songs tab → pick snippet & lyrics → Create → Generate Edit.")

    # --- Logging / progress ---

    def _set_status_dot(self, color: str) -> None:
        self._status_dot.configure(text_color=color)

    def _log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{stamp}] {message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _set_progress(self, message: str, fraction: float) -> None:
        def update() -> None:
            self._progress_fraction = max(0.0, min(1.0, fraction))
            self.status_label.configure(text=message)
            self.progress.set(self._progress_fraction)
            self.pct_label.configure(text=f"{int(self._progress_fraction * 100)}%")
            if self._has_active_work():
                self._set_status_dot(ACCENT)
            self._log(message)
        self.after(0, update)

    def _has_active_work(self) -> bool:
        return (
            self._bg_busy
            or any(j.status == "running" for j in self._jobs.values())
            or any(j.status == "running" for j in self._source_jobs.values())
        )

    # --- Songs tab ---

    def _format_snippet_label(self, name: str, start: float, end: float | None) -> str:
        end_s = f"{end:.0f}s" if end is not None else "end"
        return f"{name} ({start:.0f}s–{end_s})"

    def _refresh_snippet_picker(self, song: object | None = None) -> None:
        from app.storage import SongRecord

        values = ["(New selection)"]
        self._snippet_picker_values = values
        if song is not None:
            assert isinstance(song, SongRecord)
            for snippet in song.snippets:
                label = self._format_snippet_label(snippet.name, snippet.start, snippet.end)
                values.append(label)
                self._snippet_picker_values.append(snippet.id)
        self.snippet_picker.configure(values=values)
        if self._selected_snippet_id and song is not None:
            assert isinstance(song, SongRecord)
            snippet = song.get_snippet(self._selected_snippet_id)
            if snippet:
                label = self._format_snippet_label(snippet.name, snippet.start, snippet.end)
                self.snippet_picker.set(label)
                return
        self.snippet_picker.set("(New selection)")

    def _load_snippet_range(self, start: float, end: float | None) -> None:
        if not self._selected_song_id:
            return
        song = self._library.get_song(self._selected_song_id)
        if not song or not Path(song.path).exists():
            return
        self.waveform.load_audio(Path(song.path), start, end)
        self._refresh_snippet_lyrics()

    def _on_snippet_picked(self, choice: str) -> None:
        if choice == "(New selection)":
            self._selected_snippet_id = None
            return
        idx = self.snippet_picker.cget("values").index(choice) if choice in self.snippet_picker.cget("values") else -1
        if idx <= 0 or idx >= len(self._snippet_picker_values):
            self._selected_snippet_id = None
            return
        snippet_id = self._snippet_picker_values[idx]
        if not self._selected_song_id:
            return
        song = self._library.get_song(self._selected_song_id)
        if not song:
            return
        snippet = song.get_snippet(snippet_id)
        if not snippet:
            return
        self._selected_snippet_id = snippet.id
        self._load_snippet_range(snippet.start, snippet.end)

    def _apply_snippet_preset(self, seconds: float) -> None:
        if not self._selected_song_id:
            return
        start, _ = self.waveform.get_selection()
        self.waveform.set_selection(start, start + seconds)
        self._refresh_snippet_lyrics()

    def _refresh_songs_list(self) -> None:
        for w in self.songs_list.winfo_children():
            w.destroy()
        songs = self._library.list_songs()
        if not songs:
            ctk.CTkLabel(self.songs_list, text="No songs yet — upload one!", text_color=TEXT_DIM).pack(pady=20)
            return
        for song in songs:
            btn = ctk.CTkButton(
                self.songs_list,
                text=(
                    f"{song.title}\n{len(song.snippets)} snippet{'s' if len(song.snippets) != 1 else ''}"
                    f" · {len(song.lyrics)} lyric lines · {song.bpm:.0f} BPM"
                    if song.bpm
                    else f"{song.title}\n{len(song.snippets)} saved snippet{'s' if len(song.snippets) != 1 else ''}"
                ),
                anchor="w",
                height=52,
                fg_color=SURFACE_RAISED,
                hover_color=BORDER,
                text_color=TEXT,
                font=ctk.CTkFont(size=12),
                command=lambda s=song: self._select_song(s),
            )
            btn.pack(fill="x", pady=4)

    def _select_song(self, song: object) -> None:
        from app.storage import SongRecord

        assert isinstance(song, SongRecord)
        self._selected_song_id = song.id
        self.song_detail_title.configure(text=song.title)
        self.song_bpm_label.configure(text=f"{song.bpm:.0f} BPM" if song.bpm else "BPM not analyzed yet")

        self._selected_song_lyrics = list(song.lyrics)
        if song.snippets:
            first = song.snippets[0]
            self._selected_snippet_id = first.id
            snippet_start, snippet_end = first.start, first.end
        else:
            self._selected_snippet_id = None
            snippet_start = song.snippet_start
            snippet_end = song.snippet_end
        self._refresh_snippet_picker(song)
        if Path(song.path).exists():
            self.waveform.load_audio(Path(song.path), snippet_start, snippet_end)
        else:
            self.waveform.load_audio(Path(song.path))
        self._refresh_snippet_lyrics()

    def _on_waveform_change(self, start: float, end: float) -> None:
        self._selected_snippet_id = None
        self.snippet_picker.set("(New selection)")
        self._refresh_snippet_lyrics()

    def _refresh_snippet_lyrics(self) -> None:
        start, end = self.waveform.get_selection()
        self.lyrics_editor.delete("1.0", "end")
        if not self._selected_song_lyrics:
            self.lyrics_editor.insert("end", "(No lyrics — click Transcribe Snippet)\n")
            return
        clipped = clip_lyrics_to_snippet(self._selected_song_lyrics, start, end)
        if not clipped:
            self.lyrics_editor.insert("end", "(No lyrics in this snippet — transcribe or edit manually)\n")
            return
        for line in clipped:
            self.lyrics_editor.insert("end", f"[{line.start:.1f}s – {line.end:.1f}s] {line.text}\n")

    def _parse_lyrics_editor(self, snippet_start: float = 0.0) -> list[LyricLine]:
        lines: list[LyricLine] = []
        for raw in self.lyrics_editor.get("1.0", "end").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("("):
                continue
            if raw.startswith("[") and "]" in raw:
                header, _, text = raw.partition("]")
                text = text.strip()
                times = header.strip("[]").replace("–", "-").split("-")
                if len(times) == 2:
                    try:
                        start = float(times[0].replace("s", "").strip()) + snippet_start
                        end = float(times[1].replace("s", "").strip()) + snippet_start
                        lines.append(LyricLine(text=text, start=start, end=end))
                        continue
                    except ValueError:
                        pass
            lines.append(LyricLine(text=raw, start=snippet_start, end=snippet_start + 5.0))
        return lines

    def _upload_song(self) -> None:
        path = filedialog.askopenfilename(
            title="Upload song",
            filetypes=[("Audio", "*.mp3 *.m4a *.wav *.aac *.flac *.ogg"), ("All files", "*.*")],
        )
        if not path:
            return
        self._run_bg(f"Uploading {Path(path).name}…", lambda: self._import_song(Path(path)))

    def _import_song(self, path: Path) -> None:
        lyrics = transcribe_lyrics(path, self._set_progress)
        beat = analyze_audio_beats(path, self._set_progress)
        record = self._library.add_song(path, lyrics=lyrics, bpm=beat.bpm, title=path.stem)
        self.after(0, lambda r=record: (self._refresh_songs_list(), self._select_song(r), self._log(f"Added song: {r.title}")))

    def _transcribe_selected_song(self) -> None:
        if not self._selected_song_id:
            messagebox.showinfo("No song", "Select a song first.")
            return
        song = self._library.get_song(self._selected_song_id)
        if not song:
            return
        start, end = self.waveform.get_selection()
        self._run_bg("Transcribing…", lambda: self._do_transcribe(song, start, end))

    def _do_transcribe(self, song: object, start: float, end: float) -> None:
        from app.storage import SongRecord

        assert isinstance(song, SongRecord)
        tmp = self._work / "retranscribe.mp3"
        extract_audio_snippet(Path(song.path), tmp, start, end)
        lyrics = transcribe_lyrics(tmp, self._set_progress)
        for line in lyrics:
            line.start += start
            line.end += start
        song.lyrics = merge_lyrics_range(song.lyrics, lyrics, start, end)
        beat = analyze_audio_beats(Path(song.path), self._set_progress)
        song.bpm = beat.bpm
        self._library.update_song(song)
        self.after(0, lambda s=song: self._on_transcribe_complete(s))

    def _on_transcribe_complete(self, song: object) -> None:
        from app.storage import SongRecord

        assert isinstance(song, SongRecord)
        self._selected_song_lyrics = list(song.lyrics)
        self.song_bpm_label.configure(text=f"{song.bpm:.0f} BPM" if song.bpm else "BPM not analyzed yet")
        self._refresh_songs_list()
        self._refresh_snippet_lyrics()
        self._log(f"Transcribed snippet for {song.title}")

    def _save_named_snippet(self) -> None:
        if not self._selected_song_id:
            messagebox.showinfo("No song", "Select a song first.")
            return
        song = self._library.get_song(self._selected_song_id)
        if not song:
            return
        snippet_start, snippet_end = self.waveform.get_selection()
        edited = self._parse_lyrics_editor(snippet_start)
        song.lyrics = merge_lyrics_range(song.lyrics, edited, snippet_start, snippet_end)
        self._selected_song_lyrics = list(song.lyrics)

        existing = song.get_snippet(self._selected_snippet_id) if self._selected_snippet_id else None
        if existing:
            name = existing.name
        else:
            default_name = f"Snippet {len(song.snippets) + 1}"
            name = simpledialog.askstring("Save snippet", "Name this snippet:", initialvalue=default_name, parent=self)
            if not name or not name.strip():
                return
            name = name.strip()
            existing = None

        if existing:
            existing.start = snippet_start
            existing.end = snippet_end
        else:
            from app.storage import SongSnippet

            new_snippet = SongSnippet(id=uuid.uuid4().hex[:12], name=name, start=snippet_start, end=snippet_end)
            song.snippets.append(new_snippet)
            self._selected_snippet_id = new_snippet.id

        song.snippet_start = snippet_start
        song.snippet_end = snippet_end
        self._library.update_song(song)
        self._refresh_snippet_picker(song)
        self._refresh_snippet_lyrics()
        self._log(f"Saved snippet '{name}' for {song.title}")
        messagebox.showinfo("Saved", f"Snippet '{name}' saved with lyrics.")

    def _delete_snippet(self) -> None:
        if not self._selected_song_id or not self._selected_snippet_id:
            messagebox.showinfo("No snippet", "Select a saved snippet to delete.")
            return
        song = self._library.get_song(self._selected_song_id)
        if not song:
            return
        snippet = song.get_snippet(self._selected_snippet_id)
        if not snippet:
            return
        if not messagebox.askyesno("Delete snippet", f"Delete snippet '{snippet.name}'?"):
            return
        song.snippets = [s for s in song.snippets if s.id != snippet.id]
        self._selected_snippet_id = None
        if song.snippets:
            song.snippet_start = song.snippets[0].start
            song.snippet_end = song.snippets[0].end
        self._library.update_song(song)
        self._refresh_snippet_picker(song)
        self.snippet_picker.set("(New selection)")
        self._log(f"Deleted snippet '{snippet.name}' from {song.title}")

    def _use_song_in_create(self) -> None:
        if not self._selected_song_id:
            messagebox.showinfo("No song", "Select a song first.")
            return
        song = self._library.get_song(self._selected_song_id)
        if not song:
            return
        self.tabs.set("Create")
        self.audio_entry.delete(0, "end")
        self.audio_entry.insert(0, song.path)
        snippet_start, snippet_end = self.waveform.get_selection()
        song.snippet_start = snippet_start
        song.snippet_end = snippet_end
        self._library.update_song(song)
        end_label = f"{snippet_end:.1f}s" if snippet_end else "end"
        snippet_name = ""
        if self._selected_snippet_id:
            snip = song.get_snippet(self._selected_snippet_id)
            if snip:
                snippet_name = f" ({snip.name})"
        self._log(
            f"Song '{song.title}'{snippet_name} loaded — snippet {snippet_start:.1f}s–{end_label} · hit Generate Edit"
        )

    # --- Sources tab ---

    def _refresh_sources_list(self) -> None:
        self.source_video_preview.stop()
        for w in self.sources_list.winfo_children():
            w.destroy()
        sources = self._library.list_sources()
        self._source_options = ["(Download new TikTok)"] + [f"{s.title} ({s.id[:6]})" for s in sources]
        self.source_pick.configure(values=self._source_options)

        if not sources:
            self._preview_source_id = None
            self.source_video_preview.unload()
            self.source_preview_title.configure(text="Select a source")
            self.source_preview_meta.configure(text="")
            ctk.CTkLabel(
                self.sources_list,
                text="No sources yet — paste a TikTok link above to download and remove text.",
                text_color=TEXT_DIM,
            ).pack(pady=40)
            return

        preview_src = next((s for s in sources if s.id == self._preview_source_id), None)
        if preview_src is None:
            preview_src = sources[0]
            self._preview_source_id = preview_src.id

        for src in sources:
            selected = src.id == self._preview_source_id
            card = ctk.CTkFrame(
                self.sources_list,
                fg_color=ACCENT_DIM if selected else SURFACE,
                corner_radius=10,
                border_width=1,
                border_color=ACCENT if selected else BORDER,
                cursor="hand2",
            )
            card.pack(fill="x", padx=8, pady=6)
            card.grid_columnconfigure(0, weight=1)

            title_lbl = ctk.CTkLabel(
                card, text=src.title, font=ctk.CTkFont(size=14, weight="bold"),
                text_color=TEXT, cursor="hand2",
            )
            title_lbl.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 2))
            url_lbl = ctk.CTkLabel(
                card, text=src.tiktok_url or "Local file", font=ctk.CTkFont(size=11),
                text_color=TEXT_DIM, wraplength=280, cursor="hand2",
            )
            url_lbl.grid(row=1, column=0, sticky="w", padx=14)
            font_label = (
                f"Font: {src.font_style.dominant_font}"
                if src.font_style.font_identified
                else f"Font: {src.font_style.dominant_font} (auto fallback)"
            )
            meta_lbl = ctk.CTkLabel(
                card, text=f"Added {src.added_at[:10]} · {font_label}",
                font=ctk.CTkFont(size=11), text_color=TEXT_DIM, cursor="hand2",
            )
            meta_lbl.grid(row=2, column=0, sticky="w", padx=14, pady=(2, 12))

            btns = ctk.CTkFrame(card, fg_color="transparent")
            btns.grid(row=0, column=1, rowspan=3, padx=14, pady=12)
            self._outline_btn(btns, "Preview", lambda s=src: self._select_source_preview(s), width=80).pack(pady=2)
            self._outline_btn(btns, "Use in Create", lambda s=src: self._use_source_in_create(s), width=110).pack(pady=2)
            self._outline_btn(btns, "Reveal", lambda s=src: self._reveal_path(Path(s.path)), width=110).pack(pady=2)

            for widget in (card, title_lbl, url_lbl, meta_lbl):
                widget.bind("<Button-1>", lambda _e, s=src: self._select_source_preview(s))

        self._select_source_preview(preview_src, refresh_list=False)

    def _select_source_preview(self, src: object, *, refresh_list: bool = True) -> None:
        from app.storage import SourceRecord

        assert isinstance(src, SourceRecord)
        if refresh_list and self._preview_source_id != src.id:
            self._preview_source_id = src.id
            self._refresh_sources_list()
            return

        self._preview_source_id = src.id
        self.source_preview_title.configure(text=src.title)
        path = Path(src.path)
        font_label = (
            src.font_style.dominant_font
            if src.font_style.font_identified
            else f"{src.font_style.dominant_font} (auto fallback)"
        )
        meta_parts = [src.tiktok_url or "Local file", f"Font: {font_label}"]
        duration = 0.0
        if path.exists():
            try:
                _w, _h, _fps, duration = get_video_info(path)
                meta_parts.append(f"{duration:.1f}s · {_w}×{_h}")
            except (RuntimeError, StopIteration, KeyError, ValueError):
                meta_parts.append(path.name)
        else:
            meta_parts.append("File missing")
        self.source_preview_meta.configure(text=" · ".join(meta_parts))
        self.source_video_preview.load(path if path.exists() else None, duration=duration)

    def _use_preview_source_in_create(self) -> None:
        if not self._preview_source_id:
            return
        src = self._library.get_source(self._preview_source_id)
        if src:
            self._use_source_in_create(src)

    def _reveal_preview_source(self) -> None:
        if not self._preview_source_id:
            return
        src = self._library.get_source(self._preview_source_id)
        if src:
            self._reveal_path(Path(src.path))

    def _add_source(self) -> None:
        url = self.source_url_entry.get().strip()
        if not url or "tiktok" not in url.lower():
            messagebox.showerror("Missing URL", "Paste a valid TikTok URL.")
            return

        self.source_url_entry.delete(0, "end")

        job_id = uuid.uuid4().hex[:8]
        job_work = self._work / f"source_{job_id}"
        config = SourcePipelineConfig(
            tiktok_url=url,
            work_dir=job_work,
            per_frame_text_detection=self.source_per_frame_var.get(),
            library=self._library,
        )
        short_url = url if len(url) <= 48 else f"{url[:45]}…"
        job = QueuedSource(id=job_id, config=config, url=url, title=short_url)
        self._source_jobs[job_id] = job
        self._source_job_order.append(job_id)
        self._render_source_job_row(job)
        self._update_source_jobs_stats()
        self._set_status_dot(ACCENT)
        self._log(f"Import started: {short_url}")
        threading.Thread(target=self._run_source_job, args=(job_id,), daemon=True).start()

    def _update_source_jobs_stats(self) -> None:
        running = sum(1 for j in self._source_jobs.values() if j.status == "running")
        done = sum(1 for j in self._source_jobs.values() if j.status == "done")
        failed = sum(1 for j in self._source_jobs.values() if j.status == "error")
        parts: list[str] = []
        if running:
            parts.append(f"{running} running")
        if done:
            parts.append(f"{done} done")
        if failed:
            parts.append(f"{failed} failed")
        self.source_jobs_stats.configure(text=" · ".join(parts) if parts else "")

    def _render_source_job_row(self, job: QueuedSource) -> None:
        if self._source_jobs_empty_label.winfo_exists():
            self._source_jobs_empty_label.pack_forget()

        row = ctk.CTkFrame(self.source_jobs_frame, fg_color=SURFACE, corner_radius=8, border_width=1, border_color=BORDER)
        row.pack(fill="x", pady=3, padx=4)
        row.grid_columnconfigure(1, weight=1)

        status_colors = {"running": ACCENT, "done": SUCCESS, "error": WARNING}
        dot = ctk.CTkLabel(row, text="●", font=ctk.CTkFont(size=10), text_color=status_colors.get(job.status, TEXT_DIM), width=16)
        dot.grid(row=0, column=0, rowspan=2, padx=(10, 6), pady=6, sticky="w")
        title_lbl = ctk.CTkLabel(row, text=job.title, font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT, anchor="w")
        title_lbl.grid(row=0, column=1, sticky="ew", pady=(6, 0))
        step_lbl = ctk.CTkLabel(row, text=job.error or job.step, font=ctk.CTkFont(size=11), text_color=TEXT_DIM, anchor="w")
        step_lbl.grid(row=1, column=1, sticky="ew", pady=(0, 6))
        pct_lbl = ctk.CTkLabel(row, text="0%", font=ctk.CTkFont(size=11), text_color=TEXT_DIM, width=36)
        pct_lbl.grid(row=0, column=2, rowspan=2, padx=(4, 10), pady=6)
        bar = ctk.CTkProgressBar(row, height=4, width=80, progress_color=ACCENT, fg_color=SURFACE_RAISED)
        bar.set(job.fraction)
        bar.grid(row=0, column=3, rowspan=2, padx=(0, 10), pady=6)

        job.widgets = {"row": row, "dot": dot, "title": title_lbl, "step": step_lbl, "pct": pct_lbl, "bar": bar}

    def _update_source_job_ui(self, job_id: str) -> None:
        job = self._source_jobs.get(job_id)
        if not job or not job.widgets:
            return
        status_colors = {"running": ACCENT, "done": SUCCESS, "error": WARNING}
        job.widgets["dot"].configure(text_color=status_colors.get(job.status, TEXT_DIM))
        job.widgets["step"].configure(text=job.error or job.step)
        job.widgets["pct"].configure(text=f"{int(job.fraction * 100)}%")
        job.widgets["bar"].set(job.fraction)
        self._update_source_jobs_stats()

    def _source_job_progress(self, job_id: str, message: str, fraction: float) -> None:
        def update() -> None:
            job = self._source_jobs.get(job_id)
            if not job:
                return
            job.step = message
            job.fraction = max(0.0, min(1.0, fraction))
            job.status = "running"
            self._update_source_job_ui(job_id)
            if self._has_active_work():
                self._set_status_dot(ACCENT)
            self._log(f"[source {job_id[:6]}] {message}")
        self.after(0, update)

    def _run_source_job(self, job_id: str) -> None:
        job = self._source_jobs[job_id]
        try:
            result = run_source_pipeline(
                job.config,
                progress=lambda m, f, jid=job_id: self._source_job_progress(jid, m, f),
            )
            self.after(0, lambda r=result, jid=job_id: self._source_job_done(jid, r))
        except Exception as exc:
            self.after(0, lambda e=exc, jid=job_id: self._source_job_error(jid, e))

    def _source_job_done(self, job_id: str, result: object) -> None:
        from app.pipeline import SourcePipelineResult

        assert isinstance(result, SourcePipelineResult)
        job = self._source_jobs[job_id]
        job.status = "done"
        job.step = f"Ready — {result.source_record.title}"
        job.fraction = 1.0
        job.result = result
        self._update_source_job_ui(job_id)
        self._refresh_sources_list()
        self._log(f"Source ready: {result.source_record.title}")
        if not self._has_active_work():
            self._set_status_dot(SUCCESS)
            self.status_label.configure(text="Ready")

    def _source_job_error(self, job_id: str, exc: Exception) -> None:
        job = self._source_jobs[job_id]
        job.status = "error"
        job.error = str(exc)
        job.step = "Failed"
        self._update_source_job_ui(job_id)
        self._log(f"ERROR [source {job_id[:6]}]: {exc}")
        if not self._has_active_work():
            self._set_status_dot(WARNING)
            self.status_label.configure(text="Ready")

    def _clear_finished_source_jobs(self) -> None:
        to_remove = [jid for jid in self._source_job_order if self._source_jobs[jid].status in ("done", "error")]
        for jid in to_remove:
            job = self._source_jobs.pop(jid)
            self._source_job_order.remove(jid)
            if job.widgets.get("row"):
                job.widgets["row"].destroy()
        if not self._source_job_order and self._source_jobs_empty_label.winfo_exists():
            self._source_jobs_empty_label.pack(pady=12, padx=12)
        self._update_source_jobs_stats()

    def _on_source_pick(self, choice: str) -> None:
        if choice.startswith("("):
            self._selected_source_id = None
        else:
            for src in self._library.list_sources():
                if src.id[:6] in choice:
                    self._selected_source_id = src.id
                    break

    def _use_source_in_create(self, src: object) -> None:
        from app.storage import SourceRecord

        assert isinstance(src, SourceRecord)
        self._selected_source_id = src.id
        self.source_pick.set(f"{src.title} ({src.id[:6]})")
        if src.tiktok_url:
            self.url_entry.delete(0, "end")
            self.url_entry.insert(0, src.tiktok_url)
        self.tabs.set("Create")
        self._log(f"Using source '{src.title}' — will skip download & inpainting")

    # --- Studio tab ---

    def _refresh_edits_list(self) -> None:
        edits = self._library.list_edits()
        options = [f"{e.title} ({e.id[:6]})" for e in edits] or ["(No edits yet)"]
        self.studio_edit_pick.configure(values=options)
        if not edits:
            self.studio_edit_pick.set("(No edits yet)")
            return
        if self._selected_edit_id:
            for edit in edits:
                if edit.id == self._selected_edit_id:
                    self.studio_edit_pick.set(f"{edit.title} ({edit.id[:6]})")
                    return
        self.studio_edit_pick.set(options[0])
        self._select_edit(edits[0])

    def _on_studio_edit_pick(self, choice: str) -> None:
        if choice.startswith("("):
            return
        for edit in self._library.list_edits():
            if edit.id[:6] in choice:
                self._select_edit(edit)
                break

    def _studio_edit_duration(self, edit: object) -> float:
        from app.storage import EditRecord

        assert isinstance(edit, EditRecord)
        path = Path(edit.with_audio_path)
        if path.exists():
            try:
                return get_video_info(path)[3]
            except Exception:
                pass
        if edit.lyrics:
            return max(l.end for l in edit.lyrics)
        return 30.0

    def _select_edit(self, edit: object) -> None:
        from app.storage import EditRecord

        assert isinstance(edit, EditRecord)
        self._selected_edit_id = edit.id
        self._studio_lyrics = [LyricLine(text=l.text, start=l.start, end=l.end) for l in edit.lyrics]
        self._studio_duration = self._studio_edit_duration(edit)
        self._studio_font_style = edit.font_style
        self.studio_title.configure(text=f"{edit.title} · {len(self._studio_lyrics)} captions")

        self._studio_syncing = True
        self.studio_timeline.load(self._studio_lyrics, self._studio_duration)

        t = edit.tweak
        self.tweak_x.set(t.offset_x)
        self.tweak_y.set(t.offset_y)
        self.tweak_size.set(t.font_size or edit.font_style.dominant_size)
        self.tweak_stroke.set(t.stroke_width or edit.font_style.stroke_width)
        self.tweak_x_label.configure(text=str(t.offset_x))
        self.tweak_y_label.configure(text=str(t.offset_y))
        self.tweak_size_label.configure(text=str(t.font_size or edit.font_style.dominant_size))
        self.tweak_stroke_label.configure(text=str(t.stroke_width or edit.font_style.stroke_width))
        self.tweak_font.set(t.font_name or edit.font_style.dominant_font)
        self.tweak_stroke_var.set(t.has_stroke if t.has_stroke is not None else edit.font_style.has_stroke)

        playhead = self.studio_timeline.playhead_time()
        self.studio_preview.load(
            Path(edit.with_audio_path),
            self._studio_duration,
            self._studio_lyrics,
            edit.font_style,
            self._current_tweak(),
            time_sec=playhead,
        )
        self._studio_syncing = False

        idx = self.studio_timeline.selected_index()
        self._fill_studio_caption_fields(idx)

    def _studio_style_changed(self, *_args: object) -> None:
        self._studio_refresh_preview()

    def _studio_refresh_preview(self) -> None:
        if self._studio_font_style is None:
            return
        self.studio_preview.set_overlay(
            self._studio_lyrics,
            self._studio_font_style,
            self._current_tweak(),
        )

    def _on_studio_timeline_seek(self, t: float) -> None:
        if self._studio_syncing:
            return
        self._studio_syncing = True
        self.studio_preview.set_time(t, notify=False)
        self._studio_syncing = False

    def _on_studio_preview_seek(self, t: float) -> None:
        if self._studio_syncing:
            return
        self._studio_syncing = True
        self.studio_timeline.set_playhead(t, notify=False)
        self._studio_syncing = False

    def _fill_studio_caption_fields(self, index: int | None) -> None:
        self._studio_syncing = True
        if index is None or index >= len(self._studio_lyrics):
            self.studio_caption_text.delete(0, "end")
            self.studio_start.delete(0, "end")
            self.studio_end.delete(0, "end")
            self.studio_dur.delete(0, "end")
            self._studio_syncing = False
            return
        line = self._studio_lyrics[index]
        self.studio_caption_text.delete(0, "end")
        self.studio_caption_text.insert(0, line.text)
        self.studio_start.delete(0, "end")
        self.studio_start.insert(0, f"{line.start:.2f}")
        self.studio_end.delete(0, "end")
        self.studio_end.insert(0, f"{line.end:.2f}")
        self.studio_dur.delete(0, "end")
        self.studio_dur.insert(0, f"{line.end - line.start:.2f}")
        self._studio_syncing = False

    def _on_studio_caption_select(self, index: int | None) -> None:
        self._fill_studio_caption_fields(index)

    def _on_studio_timeline_change(self, lyrics: list[LyricLine]) -> None:
        if self._studio_syncing:
            return
        self._studio_lyrics = lyrics
        idx = self.studio_timeline.selected_index()
        self._fill_studio_caption_fields(idx)
        self._studio_refresh_preview()

    def _studio_caption_text_changed(self) -> None:
        if self._studio_syncing:
            return
        idx = self.studio_timeline.selected_index()
        if idx is None or idx >= len(self._studio_lyrics):
            return
        self._studio_lyrics[idx].text = self.studio_caption_text.get().strip()
        self.studio_timeline.set_lyrics(self._studio_lyrics, notify=False)
        self._studio_refresh_preview()

    def _studio_timing_changed(self) -> None:
        if self._studio_syncing:
            return
        idx = self.studio_timeline.selected_index()
        if idx is None or idx >= len(self._studio_lyrics):
            return
        try:
            start = float(self.studio_start.get().strip())
            end = float(self.studio_end.get().strip())
        except ValueError:
            return
        if end <= start:
            end = start + 0.25
        end = min(end, self._studio_duration)
        start = max(0.0, start)
        self._studio_lyrics[idx].start = start
        self._studio_lyrics[idx].end = end
        self.studio_timeline.set_lyrics(self._studio_lyrics, notify=False)
        self._fill_studio_caption_fields(idx)
        self._studio_refresh_preview()

    def _studio_add_caption(self) -> None:
        if not self._selected_edit_id:
            messagebox.showinfo("No project", "Create an edit first, then open it in Studio.")
            return
        idx = self.studio_timeline.selected_index()
        if idx is not None and idx < len(self._studio_lyrics):
            after = self._studio_lyrics[idx].end
        elif self._studio_lyrics:
            after = self._studio_lyrics[-1].end
        else:
            after = 0.0
        start = after
        end = min(start + 2.0, self._studio_duration)
        if end - start < 0.25:
            start = max(0.0, self._studio_duration - 2.0)
            end = self._studio_duration
        self._studio_lyrics.append(LyricLine(text="new caption", start=start, end=end))
        self.studio_timeline.set_lyrics(self._studio_lyrics)
        self.studio_timeline.select_index(len(self._studio_lyrics) - 1)

    def _studio_delete_caption(self) -> None:
        idx = self.studio_timeline.selected_index()
        if idx is None or idx >= len(self._studio_lyrics):
            messagebox.showinfo("No caption", "Select a caption block on the timeline first.")
            return
        if len(self._studio_lyrics) <= 1:
            messagebox.showinfo("Cannot delete", "Keep at least one caption, or edit the text to empty.")
            return
        del self._studio_lyrics[idx]
        self.studio_timeline.set_lyrics(self._studio_lyrics)
        if self._studio_lyrics:
            self.studio_timeline.select_index(min(idx, len(self._studio_lyrics) - 1))
        else:
            self._fill_studio_caption_fields(None)

    def _current_tweak(self) -> OverlayTweak:
        return OverlayTweak(
            offset_x=int(self.tweak_x.get()),
            offset_y=int(self.tweak_y.get()),
            font_size=int(self.tweak_size.get()),
            font_name=self.tweak_font.get(),
            has_stroke=self.tweak_stroke_var.get(),
            stroke_width=int(self.tweak_stroke.get()),
        )

    def _rerender_edit(self) -> None:
        if not self._selected_edit_id:
            messagebox.showinfo("No project", "Select a project from the dropdown.")
            return
        edit = self._library.get_edit(self._selected_edit_id)
        if not edit:
            return
        tweak = self._current_tweak()
        edit.tweak = tweak
        edit.lyrics = [LyricLine(text=l.text, start=l.start, end=l.end) for l in self._studio_lyrics]
        self._library.update_edit(edit)
        self._run_bg("Exporting edit…", lambda: self._do_rerender(edit, tweak))

    def _do_rerender(self, edit: object, tweak: OverlayTweak) -> None:
        from app.storage import EditRecord

        assert isinstance(edit, EditRecord)
        out = Path(edit.output_path)
        lyrics = [LyricLine(text=l.text, start=l.start, end=l.end) for l in edit.lyrics]
        re_render_edit(
            Path(edit.with_audio_path),
            lyrics,
            edit.font_style,
            out,
            tweak=tweak,
            snippet_start=0.0,
            progress=self._set_progress,
        )
        self._last_output = out
        self.after(0, lambda p=out: (
            self.open_btn.configure(state="normal"),
            self.share_btn.configure(state="normal"),
            messagebox.showinfo("Done", f"Re-rendered:\n{p}"),
        ))

    def _open_studio_output(self) -> None:
        if self._selected_edit_id:
            edit = self._library.get_edit(self._selected_edit_id)
            if edit and Path(edit.output_path).exists():
                self._last_output = Path(edit.output_path)
        self._open_output()

    # --- Create tab actions ---

    def _pick_song_from_library(self) -> None:
        songs = self._library.list_songs()
        if not songs:
            messagebox.showinfo("Empty library", "Upload songs in the Songs tab first.")
            return
        self.tabs.set("Songs")
        self._select_song(songs[0])

    def _browse_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose replacement song",
            filetypes=[("Audio", "*.mp3 *.m4a *.wav *.aac *.flac *.ogg"), ("All files", "*.*")],
        )
        if path:
            self.audio_entry.delete(0, "end")
            self.audio_entry.insert(0, path)

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Save output video", defaultextension=".mp4", filetypes=[("MP4 video", "*.mp4")])
        if path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)

    def _validate(self, job_work_dir: Path | None = None) -> PipelineConfig | None:
        url = self.url_entry.get().strip()
        audio_str = self.audio_entry.get().strip()
        output = Path(self.output_entry.get().strip())

        use_source = self._selected_source_id is not None
        if not use_source and (not url or "tiktok" not in url.lower()):
            messagebox.showerror("Missing URL", "Enter a valid TikTok URL or pick a library source.")
            return None
        if not audio_str:
            messagebox.showerror("Missing audio", "Choose a replacement song (Browse or Songs tab → Use in Create).")
            return None
        audio = Path(audio_str)
        if not audio.is_file():
            messagebox.showerror("Missing audio", f"Replacement song not found or not a file:\n{audio}")
            return None
        ensure_dir(output.parent)

        snippet_start = 0.0
        snippet_end: float | None = None
        lyrics_override = None
        song_id = None

        for song in self._library.list_songs():
            if Path(song.path).resolve() == audio.resolve():
                song_id = song.id
                if self._selected_song_id == song.id:
                    snippet_start, snippet_end = self.waveform.get_selection()
                else:
                    snippet_start = song.snippet_start
                    snippet_end = song.snippet_end
                if song.lyrics:
                    lyrics_override = song.lyrics
                break

        source_video = None
        if use_source:
            src = self._library.get_source(self._selected_source_id)
            if src:
                source_video = Path(src.path)

        return PipelineConfig(
            tiktok_url=url or "library://source",
            replacement_audio=audio,
            output_path=output,
            work_dir=job_work_dir or self._work,
            per_frame_text_detection=self.per_frame_var.get(),
            skip_download=use_source,
            source_video=source_video,
            snippet_start=snippet_start,
            snippet_end=snippet_end,
            lyrics_override=lyrics_override,
            beat_sync=self.beat_sync_var.get(),
            library=self._library,
            song_id=song_id,
            source_id=self._selected_source_id,
        )

    # --- Job queue ---

    def _update_queue_stats(self) -> None:
        running = sum(1 for j in self._jobs.values() if j.status == "running")
        queued = sum(1 for j in self._jobs.values() if j.status == "queued")
        done = sum(1 for j in self._jobs.values() if j.status == "done")
        failed = sum(1 for j in self._jobs.values() if j.status == "error")
        parts: list[str] = []
        if running:
            parts.append(f"{running} running")
        if queued:
            parts.append(f"{queued} queued")
        if done:
            parts.append(f"{done} done")
        if failed:
            parts.append(f"{failed} failed")
        self.queue_stats_label.configure(text=" · ".join(parts) if parts else "No jobs")
        self.run_queue_btn.configure(state="normal" if queued else "disabled")

    def _create_job(self, *, start: bool) -> QueuedEdit | None:
        job_id = uuid.uuid4().hex[:8]
        job_work = self._work / "jobs" / job_id
        config = self._validate(job_work_dir=job_work)
        if config is None:
            return None

        title = config.output_path.stem or f"Edit {job_id}"
        job = QueuedEdit(
            id=job_id,
            config=config,
            title=title,
            status="queued",
            step="Queued — click Run Queue" if not start else "Queued",
        )
        self._jobs[job_id] = job
        self._job_order.append(job_id)
        self._render_job_row(job)
        self._update_queue_stats()

        default_out = Path.home() / "Movies" / "EditAutomate" / f"remix_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        self.output_entry.delete(0, "end")
        self.output_entry.insert(0, str(default_out))

        if start:
            self._log(f"Generating: {title}")
            self._submit_job(job)
        else:
            self._log(f"Queued (not started): {title}")
        return job

    def _try_generate_shortcut(self) -> None:
        if str(self.tabs.get()) == "Create":
            self._generate_edit()

    def _generate_edit(self) -> None:
        self._create_job(start=True)

    def _enqueue_edit(self) -> None:
        self._create_job(start=False)

    def _run_queue(self) -> None:
        started = 0
        for jid in self._job_order:
            job = self._jobs.get(jid)
            if job and job.status == "queued":
                self._submit_job(job)
                started += 1
        if started:
            self._log(f"Started {started} queued job(s)")
        else:
            self._log("No queued jobs to run")
        self._update_queue_stats()

    def _render_job_row(self, job: QueuedEdit) -> None:
        if self._queue_empty_label.winfo_exists():
            self._queue_empty_label.pack_forget()

        row = ctk.CTkFrame(self.queue_frame, fg_color=SURFACE, corner_radius=8, border_width=1, border_color=BORDER)
        row.pack(fill="x", pady=3, padx=4)
        row.grid_columnconfigure(1, weight=1)

        status_colors = {"queued": TEXT_DIM, "running": ACCENT, "done": SUCCESS, "error": WARNING}
        status_labels = {"queued": "QUEUED", "running": "RUN", "done": "DONE", "error": "FAIL"}
        dot = ctk.CTkLabel(row, text="●", font=ctk.CTkFont(size=10), text_color=status_colors.get(job.status, TEXT_DIM), width=16)
        dot.grid(row=0, column=0, rowspan=2, padx=(10, 6), pady=6, sticky="w")
        title_lbl = ctk.CTkLabel(row, text=job.title, font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT, anchor="w")
        title_lbl.grid(row=0, column=1, sticky="ew", pady=(6, 0))
        step_lbl = ctk.CTkLabel(row, text=job.error or job.step, font=ctk.CTkFont(size=11), text_color=TEXT_DIM, anchor="w")
        step_lbl.grid(row=1, column=1, sticky="ew", pady=(0, 6))
        badge = ctk.CTkLabel(
            row,
            text=status_labels.get(job.status, job.status.upper()),
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=status_colors.get(job.status, TEXT_DIM),
            width=44,
        )
        badge.grid(row=0, column=2, padx=(4, 4), pady=(6, 0))
        pct_lbl = ctk.CTkLabel(row, text="0%", font=ctk.CTkFont(size=11), text_color=TEXT_DIM, width=36)
        pct_lbl.grid(row=1, column=2, padx=(4, 4), pady=(0, 6))
        bar = ctk.CTkProgressBar(row, height=4, width=100, progress_color=ACCENT, fg_color=SURFACE_RAISED)
        bar.set(job.fraction)
        bar.grid(row=0, column=3, rowspan=2, padx=(0, 10), pady=6)

        job.widgets = {"row": row, "dot": dot, "title": title_lbl, "step": step_lbl, "pct": pct_lbl, "bar": bar, "badge": badge}

    def _update_job_ui(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if not job or not job.widgets:
            return
        status_colors = {"queued": TEXT_DIM, "running": ACCENT, "done": SUCCESS, "error": WARNING}
        status_labels = {"queued": "QUEUED", "running": "RUN", "done": "DONE", "error": "FAIL"}
        job.widgets["dot"].configure(text_color=status_colors.get(job.status, TEXT_DIM))
        job.widgets["step"].configure(text=job.error or job.step)
        job.widgets["pct"].configure(text=f"{int(job.fraction * 100)}%")
        job.widgets["bar"].set(job.fraction)
        if "badge" in job.widgets:
            job.widgets["badge"].configure(
                text=status_labels.get(job.status, job.status.upper()),
                text_color=status_colors.get(job.status, TEXT_DIM),
            )
        self._update_queue_stats()

    def _job_progress(self, job_id: str, message: str, fraction: float) -> None:
        def update() -> None:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.step = message
            job.fraction = max(0.0, min(1.0, fraction))
            job.status = "running"
            self._update_job_ui(job_id)
            self._set_status_dot(ACCENT)
            self.status_label.configure(text=f"{job.title}: {message}")
            self.progress.set(job.fraction)
            self.pct_label.configure(text=f"{int(job.fraction * 100)}%")
            self._log(f"[{job.title}] {message}")
        self.after(0, update)

    def _submit_job(self, job: QueuedEdit) -> None:
        if job.status != "queued":
            return
        job.status = "running"
        job.step = "Starting…"
        self._update_job_ui(job.id)
        self._set_status_dot(ACCENT)
        threading.Thread(target=self._run_job, args=(job.id,), daemon=True).start()

    def _run_job(self, job_id: str) -> None:
        job = self._jobs[job_id]
        try:
            result = run_pipeline(job.config, progress=lambda m, f, jid=job_id: self._job_progress(jid, m, f))
            self.after(0, lambda r=result, jid=job_id: self._job_done(jid, r))
        except Exception as exc:
            self.after(0, lambda e=exc, jid=job_id: self._job_error(jid, e))

    def _job_done(self, job_id: str, result: PipelineResult) -> None:
        job = self._jobs[job_id]
        job.status = "done"
        job.step = "Complete"
        job.fraction = 1.0
        job.result = result
        self._update_job_ui(job_id)
        self._last_output = result.output_path
        self.open_btn.configure(state="normal")
        self.share_btn.configure(state="normal")
        self._set_status_dot(SUCCESS)
        self.status_label.configure(text=f"Done: {job.title}")
        self._log(f"Saved: {result.output_path}")
        self._refresh_songs_list()
        self._refresh_sources_list()
        self._refresh_edits_list()
        self._update_queue_stats()

    def _job_error(self, job_id: str, exc: Exception) -> None:
        job = self._jobs[job_id]
        job.status = "error"
        job.error = str(exc)
        job.step = "Failed"
        self._update_job_ui(job_id)
        self._set_status_dot(WARNING)
        self.status_label.configure(text=f"Error: {job.title}")
        self._log(f"ERROR [{job.title}]: {exc}")
        messagebox.showerror("Processing failed", f"{job.title}\n\n{exc}")
        self._update_queue_stats()

    def _clear_finished_jobs(self) -> None:
        to_remove = [jid for jid in self._job_order if self._jobs[jid].status in ("done", "error")]
        for jid in to_remove:
            job = self._jobs.pop(jid)
            self._job_order.remove(jid)
            if job.widgets.get("row"):
                job.widgets["row"].destroy()
        if not self._job_order and self._queue_empty_label.winfo_exists():
            self._queue_empty_label.pack(pady=20, padx=12)
        self._update_queue_stats()

    def _run_bg(self, status: str, fn: object) -> None:
        self._bg_busy = True
        self._set_status_dot(ACCENT)
        self.status_label.configure(text=status)

        def worker() -> None:
            try:
                fn()
            except Exception as exc:
                err_msg = str(exc)
                self.after(0, lambda msg=err_msg: messagebox.showerror("Error", msg))
            finally:
                self.after(0, lambda: (
                    setattr(self, "_bg_busy", False),
                    self._set_status_dot(SUCCESS),
                    self.status_label.configure(text="Ready"),
                ))

        threading.Thread(target=worker, daemon=True).start()

    def _open_output(self) -> None:
        if self._last_output and self._last_output.exists():
            if platform.system() == "Darwin":
                subprocess.run(["open", str(self._last_output)], check=False)
            elif platform.system() == "Windows":
                subprocess.run(["start", "", str(self._last_output)], shell=True, check=False)
            else:
                webbrowser.open(self._last_output.as_uri())

    def _reveal_output(self) -> None:
        if self._last_output and self._last_output.exists():
            self._reveal_path(self._last_output)

    def _reveal_path(self, path: Path) -> None:
        if platform.system() == "Darwin":
            subprocess.run(["open", "-R", str(path)], check=False)
        elif platform.system() == "Windows":
            subprocess.run(["explorer", "/select,", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path.parent)], check=False)


def launch() -> None:
    app = EditAutomateApp()
    app.mainloop()
