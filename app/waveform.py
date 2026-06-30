"""Interactive audio waveform selector with draggable snippet handles."""

from __future__ import annotations

import platform
import subprocess
import tempfile
import tkinter as tk
from pathlib import Path
from typing import Callable

import customtkinter as ctk
import numpy as np

from app.beat_sync import _audio_duration, extract_audio_snippet

# Match studio palette (avoid importing gui — circular)
_BG = "#0c0c14"
_ACCENT = "#00e5c0"
_ACCENT_DIM = "#0d3d36"
_WAVE = "#55556a"
_WAVE_SEL = "#00e5c0"
_HANDLE = "#00e5c0"
_TEXT = "#eeeef4"
_TEXT_DIM = "#8b8ba3"

# Reuse decoded peaks for the same file (path + mtime).
_PEAK_CACHE: dict[str, np.ndarray] = {}


def _peak_cache_key(path: Path) -> str:
    resolved = path.resolve()
    try:
        mtime = resolved.stat().st_mtime_ns
    except OSError:
        mtime = 0
    return f"{resolved}:{mtime}"


def _fmt_time(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


class WaveformSelector(ctk.CTkFrame):
    """TikTok/Instagram-style draggable snippet selector on an audio waveform."""

    HANDLE_W = 10
    MIN_SELECTION = 1.0

    def __init__(
        self,
        master: object,
        on_change: Callable[[float, float], None] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(master, fg_color=_BG, corner_radius=8, **kwargs)
        self._on_change = on_change
        self._audio_path: Path | None = None
        self._duration = 0.0
        self._start = 0.0
        self._end = 30.0
        self._peaks: np.ndarray | None = None
        self._drag_mode: str | None = None
        self._drag_anchor = 0.0
        self._preview_proc: subprocess.Popen | None = None

        self.canvas = tk.Canvas(self, height=96, bg=_BG, highlightthickness=0, cursor="hand2")
        self.canvas.pack(fill="x", padx=10, pady=(10, 6))
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        info = ctk.CTkFrame(self, fg_color="transparent")
        info.pack(fill="x", padx=10, pady=(0, 10))
        self.range_label = ctk.CTkLabel(
            info,
            text="Select a song to load waveform",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=_TEXT_DIM,
            anchor="w",
        )
        self.range_label.pack(side="left")
        self.preview_btn = ctk.CTkButton(
            info,
            text="▶  Preview",
            width=96,
            height=30,
            corner_radius=8,
            fg_color=_ACCENT_DIM,
            hover_color="#14554a",
            text_color=_ACCENT,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._play_preview,
            state="disabled",
        )
        self.preview_btn.pack(side="right")

    def load_audio(self, path: Path, start: float = 0.0, end: float | None = None) -> None:
        self._audio_path = path
        if path.exists():
            try:
                self._duration = _audio_duration(path)
            except Exception:
                self._duration = 0.0
        else:
            self._duration = 0.0

        self._peaks = None
        if path.exists() and self._duration > 0:
            cache_key = _peak_cache_key(path)
            cached = _PEAK_CACHE.get(cache_key)
            if cached is not None:
                self._peaks = cached
            else:
                try:
                    import librosa

                    y, _sr = librosa.load(str(path), sr=8000, mono=True, duration=self._duration)
                    n_bins = 500
                    chunk = max(1, len(y) // n_bins)
                    peaks = [
                        float(np.max(np.abs(y[i * chunk : (i + 1) * chunk])))
                        if y[i * chunk : (i + 1) * chunk].size
                        else 0.0
                        for i in range(n_bins)
                    ]
                    arr = np.array(peaks, dtype=float)
                    peak = float(np.max(arr)) if arr.size else 1.0
                    self._peaks = arr / peak if peak > 0 else arr
                    _PEAK_CACHE[cache_key] = self._peaks
                except Exception:
                    self._peaks = None

        self._start = max(0.0, start)
        default_end = end if end is not None else min(30.0, self._duration or 30.0)
        self._end = min(default_end, self._duration) if self._duration > 0 else default_end
        if self._end <= self._start:
            self._end = min(self._start + self.MIN_SELECTION, self._duration or self._start + 30.0)

        self.preview_btn.configure(state="normal" if self._audio_path else "disabled")
        self._update_label()
        self._redraw()

    def get_selection(self) -> tuple[float, float]:
        return self._start, self._end

    def set_selection(self, start: float, end: float) -> None:
        self._start = max(0.0, start)
        self._end = end
        if self._duration > 0:
            self._end = min(self._end, self._duration)
        if self._end <= self._start:
            self._end = self._start + self.MIN_SELECTION
        self._update_label()
        self._redraw()
        if self._on_change:
            self._on_change(self._start, self._end)

    def _update_label(self) -> None:
        dur = self._end - self._start
        self.range_label.configure(
            text=f"{_fmt_time(self._start)} – {_fmt_time(self._end)}  ·  {dur:.0f}s selected",
            text_color=_ACCENT if self._audio_path else _TEXT_DIM,
        )

    def _x_to_time(self, x: float) -> float:
        w = max(1, self.canvas.winfo_width())
        if self._duration <= 0:
            return 0.0
        return max(0.0, min(self._duration, x / w * self._duration))

    def _time_to_x(self, t: float) -> float:
        w = max(1, self.canvas.winfo_width())
        if self._duration <= 0:
            return 0.0
        return t / self._duration * w

    def _hit_zone(self, x: float) -> str:
        sx = self._time_to_x(self._start)
        ex = self._time_to_x(self._end)
        hw = self.HANDLE_W
        if abs(x - sx) <= hw:
            return "start"
        if abs(x - ex) <= hw:
            return "end"
        if sx < x < ex:
            return "move"
        return "seek"

    def _on_press(self, event: tk.Event) -> None:
        if self._duration <= 0:
            return
        x = event.x
        zone = self._hit_zone(x)
        if zone == "seek":
            width = self._end - self._start
            center = self._x_to_time(x)
            self._start = max(0.0, center - width / 2)
            self._end = self._start + width
            if self._end > self._duration:
                self._end = self._duration
                self._start = max(0.0, self._end - width)
            zone = "move"
        self._drag_mode = zone
        self._drag_anchor = self._x_to_time(x)

    def _on_drag(self, event: tk.Event) -> None:
        if not self._drag_mode or self._duration <= 0:
            return
        t = self._x_to_time(event.x)
        if self._drag_mode == "start":
            self._start = min(t, self._end - self.MIN_SELECTION)
            self._start = max(0.0, self._start)
        elif self._drag_mode == "end":
            self._end = max(t, self._start + self.MIN_SELECTION)
            self._end = min(self._duration, self._end)
        elif self._drag_mode == "move":
            dt = t - self._drag_anchor
            width = self._end - self._start
            new_start = self._start + dt
            new_end = new_start + width
            if new_start < 0:
                new_start = 0.0
                new_end = width
            if new_end > self._duration:
                new_end = self._duration
                new_start = self._duration - width
            self._start = new_start
            self._end = new_end
            self._drag_anchor = t
        self._update_label()
        self._redraw()

    def _on_release(self, _event: tk.Event) -> None:
        if self._drag_mode and self._on_change:
            self._on_change(self._start, self._end)
        self._drag_mode = None

    def _redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        w = max(1, c.winfo_width())
        h = max(1, c.winfo_height())
        mid = h // 2

        if self._peaks is not None and len(self._peaks):
            n = len(self._peaks)
            bar_w = max(1, w // n)
            for i, amp in enumerate(self._peaks):
                bh = int(amp * (h * 0.42))
                x0 = i * w // n
                c.create_rectangle(x0, mid - bh, x0 + bar_w, mid + bh, fill=_WAVE, outline="")

        if self._duration <= 0:
            c.create_text(w // 2, mid, text="No audio loaded", fill=_TEXT_DIM, font=("Menlo", 11))
            return

        sx = self._time_to_x(self._start)
        ex = self._time_to_x(self._end)
        c.create_rectangle(sx, 2, ex, h - 2, fill=_ACCENT_DIM, outline=_ACCENT, width=1)

        if self._peaks is not None:
            n = len(self._peaks)
            for i, amp in enumerate(self._peaks):
                t = i / n * self._duration
                if self._start <= t <= self._end:
                    bh = int(amp * (h * 0.42))
                    x0 = i * w // n
                    bar_w = max(1, w // n)
                    c.create_rectangle(x0, mid - bh, x0 + bar_w, mid + bh, fill=_WAVE_SEL, outline="")

        hw = self.HANDLE_W // 2
        for x in (sx, ex):
            c.create_rectangle(x - hw, 0, x + hw, h, fill=_HANDLE, outline=_TEXT)
            c.create_text(x, h - 10, text="◆", fill=_TEXT, font=("Arial", 8))

    def _play_preview(self) -> None:
        if not self._audio_path or not self._audio_path.exists():
            return
        if self._preview_proc and self._preview_proc.poll() is None:
            self._preview_proc.terminate()
            self._preview_proc = None
            return

        tmp = Path(tempfile.gettempdir()) / "editautomate_preview.mp3"
        extract_audio_snippet(self._audio_path, tmp, self._start, self._end)
        if platform.system() == "Darwin":
            self._preview_proc = subprocess.Popen(["afplay", str(tmp)])
        else:
            self._preview_proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", str(tmp)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
