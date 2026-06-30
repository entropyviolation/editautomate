"""CapCut-style caption track — drag blocks to move or trim when text appears."""

from __future__ import annotations

import tkinter as tk
from typing import Callable

import customtkinter as ctk

from app.audio import LyricLine

_BG = "#0c0c14"
_SURFACE = "#12121a"
_SURFACE_RAISED = "#1a1a26"
_BORDER = "#2a2a3c"
_ACCENT = "#00e5c0"
_ACCENT_DIM = "#0d3d36"
_TEXT = "#eeeef4"
_TEXT_MUTED = "#8b8ba3"
_TEXT_DIM = "#55556a"
_BLOCK = "#2a3550"
_BLOCK_SEL = "#3d5a80"
_HANDLE = "#00e5c0"

MIN_DURATION = 0.25
HANDLE_W = 8
TRACK_TOP = 28
TRACK_PAD = 6


def _fmt_time(seconds: float) -> str:
    s = max(0.0, seconds)
    if s >= 60:
        return f"{int(s // 60)}:{int(s % 60):02d}.{int((s % 1) * 10)}"
    return f"{s:.1f}s"


class CaptionTimeline(ctk.CTkFrame):
    """Horizontal caption track with draggable, resizable text blocks."""

    def __init__(
        self,
        master: object,
        on_select: Callable[[int | None], None] | None = None,
        on_change: Callable[[list[LyricLine]], None] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(master, fg_color=_SURFACE_RAISED, corner_radius=10, border_width=1, border_color=_BORDER, **kwargs)
        self._on_select = on_select
        self._on_change = on_change
        self._lyrics: list[LyricLine] = []
        self._duration = 30.0
        self._selected: int | None = None
        self._playhead = 0.0
        self._drag_mode: str | None = None
        self._drag_index: int | None = None
        self._drag_anchor_t = 0.0
        self._drag_orig_start = 0.0
        self._drag_orig_end = 0.0
        self._suppress_notify = False

        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(
            hdr,
            text="CAPTION TRACK",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_TEXT_DIM,
        ).pack(side="left")
        self.duration_label = ctk.CTkLabel(hdr, text="0:00", font=ctk.CTkFont(size=11), text_color=_TEXT_MUTED)
        self.duration_label.pack(side="right")

        self.canvas = tk.Canvas(self, height=108, bg=_BG, highlightthickness=0, cursor="hand2")
        self.canvas.pack(fill="x", padx=10, pady=(0, 10))
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    def load(self, lyrics: list[LyricLine], duration: float) -> None:
        self._lyrics = [LyricLine(text=l.text, start=l.start, end=l.end) for l in lyrics]
        self._duration = max(MIN_DURATION, duration, max((l.end for l in self._lyrics), default=0.0))
        self._selected = 0 if self._lyrics else None
        self._playhead = self._lyrics[0].start if self._lyrics else 0.0
        self.duration_label.configure(text=f"Total {_fmt_time(self._duration)}")
        self._redraw()
        if self._on_select and self._selected is not None:
            self._on_select(self._selected)

    def get_lyrics(self) -> list[LyricLine]:
        return [LyricLine(text=l.text, start=l.start, end=l.end) for l in self._lyrics]

    def set_lyrics(self, lyrics: list[LyricLine], *, notify: bool = True) -> None:
        self._suppress_notify = not notify
        self._lyrics = [LyricLine(text=l.text, start=l.start, end=l.end) for l in lyrics]
        if self._selected is not None and self._selected >= len(self._lyrics):
            self._selected = len(self._lyrics) - 1 if self._lyrics else None
        self._duration = max(self._duration, max((l.end for l in self._lyrics), default=0.0))
        self._redraw()
        self._suppress_notify = False
        if notify and self._on_change:
            self._on_change(self.get_lyrics())

    def select_index(self, index: int | None) -> None:
        if index is not None and (index < 0 or index >= len(self._lyrics)):
            return
        self._selected = index
        if index is not None:
            self._playhead = self._lyrics[index].start
        self._redraw()
        if self._on_select:
            self._on_select(index)

    def selected_index(self) -> int | None:
        return self._selected

    def set_playhead(self, t: float) -> None:
        self._playhead = max(0.0, min(self._duration, t))
        self._redraw()

    def _time_to_x(self, t: float) -> float:
        w = max(1, self.canvas.winfo_width() - TRACK_PAD * 2)
        return TRACK_PAD + t / self._duration * w

    def _x_to_time(self, x: float) -> float:
        w = max(1, self.canvas.winfo_width() - TRACK_PAD * 2)
        return max(0.0, min(self._duration, (x - TRACK_PAD) / w * self._duration))

    def _block_rect(self, index: int) -> tuple[float, float, float, float]:
        line = self._lyrics[index]
        x0 = self._time_to_x(line.start)
        x1 = self._time_to_x(line.end)
        h = self.canvas.winfo_height()
        y0 = TRACK_TOP
        y1 = max(y0 + 20, h - 12)
        return x0, y0, x1, y1

    def _hit_test(self, x: float, y: float) -> tuple[str, int | None]:
        h = self.canvas.winfo_height()
        if y < TRACK_TOP - 4 or y > h:
            return "ruler", None
        for i in range(len(self._lyrics) - 1, -1, -1):
            x0, y0, x1, y1 = self._block_rect(i)
            if y0 <= y <= y1 and x0 - HANDLE_W <= x <= x1 + HANDLE_W:
                if abs(x - x0) <= HANDLE_W:
                    return "resize_start", i
                if abs(x - x1) <= HANDLE_W:
                    return "resize_end", i
                if x0 <= x <= x1:
                    return "move", i
        return "seek", None

    def _on_press(self, event: tk.Event) -> None:
        if not self._lyrics:
            self._playhead = self._x_to_time(event.x)
            self._redraw()
            return
        mode, idx = self._hit_test(event.x, event.y)
        t = self._x_to_time(event.x)
        if mode == "seek" or idx is None:
            self._playhead = t
            self._redraw()
            return
        self._drag_mode = mode
        self._drag_index = idx
        self._drag_anchor_t = t
        line = self._lyrics[idx]
        self._drag_orig_start = line.start
        self._drag_orig_end = line.end
        if self._selected != idx:
            self.select_index(idx)

    def _clamp_block(self, index: int, start: float, end: float) -> tuple[float, float]:
        start = max(0.0, start)
        end = min(self._duration, end)
        if end - start < MIN_DURATION:
            if self._drag_mode == "resize_start":
                start = end - MIN_DURATION
            else:
                end = start + MIN_DURATION
        # Avoid overlap with neighbors (simple push)
        if index > 0:
            start = max(start, self._lyrics[index - 1].end)
        if index < len(self._lyrics) - 1:
            end = min(end, self._lyrics[index + 1].start)
        if end - start < MIN_DURATION:
            end = min(self._duration, start + MIN_DURATION)
        return start, end

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_mode is None or self._drag_index is None:
            return
        idx = self._drag_index
        line = self._lyrics[idx]
        t = self._x_to_time(event.x)
        dt = t - self._drag_anchor_t

        if self._drag_mode == "move":
            dur = self._drag_orig_end - self._drag_orig_start
            start = self._drag_orig_start + dt
            end = start + dur
            if start < 0:
                start, end = 0.0, dur
            if end > self._duration:
                end, start = self._duration, self._duration - dur
            line.start, line.end = start, end
        elif self._drag_mode == "resize_start":
            start, end = self._clamp_block(idx, t, line.end)
            line.start, line.end = start, end
        elif self._drag_mode == "resize_end":
            start, end = self._clamp_block(idx, line.start, t)
            line.start, line.end = start, end

        self._playhead = line.start
        self._redraw()

    def _on_release(self, _event: tk.Event) -> None:
        if self._drag_mode and not self._suppress_notify and self._on_change:
            self._on_change(self.get_lyrics())
        self._drag_mode = None
        self._drag_index = None

    def _redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        w = max(1, c.winfo_width())
        h = max(1, c.winfo_height())

        # Time ruler
        c.create_rectangle(0, 0, w, TRACK_TOP - 2, fill=_SURFACE, outline="")
        tick_count = max(4, min(12, int(self._duration // 2) + 1))
        for i in range(tick_count + 1):
            t = i / tick_count * self._duration
            x = self._time_to_x(t)
            c.create_line(x, 4, x, TRACK_TOP - 4, fill=_BORDER)
            c.create_text(x, 14, text=_fmt_time(t), fill=_TEXT_DIM, font=("Menlo", 9))

        # Track lane
        c.create_rectangle(TRACK_PAD, TRACK_TOP, w - TRACK_PAD, h - 8, fill=_SURFACE, outline=_BORDER)

        if not self._lyrics:
            c.create_text(w // 2, (TRACK_TOP + h) // 2, text="No captions — add one below", fill=_TEXT_DIM, font=("Menlo", 11))
        else:
            for i, line in enumerate(self._lyrics):
                x0, y0, x1, y1 = self._block_rect(i)
                selected = i == self._selected
                fill = _BLOCK_SEL if selected else _BLOCK
                outline = _ACCENT if selected else _BORDER
                c.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline, width=2 if selected else 1)
                if selected:
                    for hx in (x0, x1):
                        c.create_rectangle(hx - HANDLE_W // 2, y0, hx + HANDLE_W // 2, y1, fill=_HANDLE, outline=_TEXT)
                label = line.text.replace("\n", " ")
                if len(label) > 22:
                    label = label[:20] + "…"
                c.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=label, fill=_TEXT, font=("Arial", 10, "bold"))

        # Playhead
        px = self._time_to_x(self._playhead)
        c.create_line(px, 0, px, h, fill=_ACCENT, width=2)
        c.create_polygon(px - 5, 0, px + 5, 0, px, 8, fill=_ACCENT, outline="")
