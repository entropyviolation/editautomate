"""Plain video preview — frame display with play and scrub controls."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

_BG = "#0c0c14"
_SURFACE = "#12121a"
_SURFACE_RAISED = "#1a1a26"
_BORDER = "#2a2a3c"
_TEXT = "#eeeef4"
_TEXT_MUTED = "#8b8ba3"
_TEXT_DIM = "#55556a"


def _fmt_time(seconds: float) -> str:
    s = max(0.0, seconds)
    if s >= 60:
        return f"{int(s // 60)}:{int(s % 60):02d}.{int((s % 1) * 10)}"
    return f"{s:.1f}s"


class VideoPreview(ctk.CTkFrame):
    """Shows a video frame with play/pause and click-to-scrub."""

    def __init__(self, master: object, **kwargs: object) -> None:
        super().__init__(master, fg_color=_SURFACE_RAISED, corner_radius=10, border_width=1, border_color=_BORDER, **kwargs)
        self._video_path: Path | None = None
        self._duration = 0.0
        self._time = 0.0
        self._fps = 30.0
        self._render_gen = 0
        self._refresh_after_id: str | None = None
        self._play_after_id: str | None = None
        self._playing = False
        self._photo: ImageTk.PhotoImage | None = None

        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=12, pady=(10, 6))
        ctk.CTkLabel(
            hdr,
            text="PREVIEW",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_TEXT_DIM,
        ).pack(side="left")
        self.time_label = ctk.CTkLabel(hdr, text="0.0s / 0.0s", font=ctk.CTkFont(size=11), text_color=_TEXT_MUTED)
        self.time_label.pack(side="left", padx=(12, 0))
        self.play_btn = ctk.CTkButton(
            hdr,
            text="▶  Play",
            width=72,
            height=28,
            corner_radius=8,
            fg_color=_SURFACE,
            hover_color=_BORDER,
            border_width=1,
            border_color=_BORDER,
            text_color=_TEXT,
            font=ctk.CTkFont(size=12),
            command=self._toggle_play,
            state="disabled",
        )
        self.play_btn.pack(side="right")

        self._frame_host = ctk.CTkFrame(self, fg_color=_BG, corner_radius=8, border_width=1, border_color=_BORDER)
        self._frame_host.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self._frame_host.pack_propagate(False)
        self._frame_host.configure(height=360)

        self.canvas = tk.Canvas(self._frame_host, bg=_BG, highlightthickness=0, cursor="hand2")
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)
        self.canvas.bind("<Configure>", lambda _e: self._place_frame_image())
        self.canvas.bind("<ButtonPress-1>", self._on_scrub_press)
        self.canvas.bind("<B1-Motion>", self._on_scrub_drag)

        self._placeholder_id = self.canvas.create_text(
            0, 0, text="Select a source to preview", fill=_TEXT_DIM, font=("Menlo", 12),
        )
        self._image_id: int | None = None

    def load(self, video_path: Path | None, *, duration: float | None = None, time_sec: float = 0.0) -> None:
        self.stop()
        self._video_path = video_path if video_path and video_path.exists() else None
        self._duration = max(0.0, duration or 0.0)
        self._time = 0.0
        self._fps = 30.0

        if self._video_path:
            cap = cv2.VideoCapture(str(self._video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps and fps > 1:
                    self._fps = fps
                if not self._duration:
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if frame_count and fps:
                        self._duration = frame_count / fps
                cap.release()
            else:
                self._video_path = None

        self._time = max(0.0, min(self._duration, time_sec))
        enabled = self._video_path is not None
        self.play_btn.configure(state="normal" if enabled else "disabled")
        self._update_time_label()
        self.refresh(immediate=True)

    def unload(self) -> None:
        self.stop()
        self._video_path = None
        self._duration = 0.0
        self._time = 0.0
        self.play_btn.configure(state="disabled")
        self._update_time_label()
        self._clear_image()
        self.canvas.itemconfigure(self._placeholder_id, text="Select a source to preview")
        self._place_frame_image()

    def stop(self) -> None:
        if self._play_after_id is not None:
            self.after_cancel(self._play_after_id)
            self._play_after_id = None
        self._playing = False
        self.play_btn.configure(text="▶  Play")

    def refresh(self, *, immediate: bool = False) -> None:
        if self._refresh_after_id is not None:
            self.after_cancel(self._refresh_after_id)
            self._refresh_after_id = None
        delay = 0 if immediate else 60
        self._refresh_after_id = self.after(delay, self._do_refresh)

    def _update_time_label(self) -> None:
        self.time_label.configure(text=f"{_fmt_time(self._time)} / {_fmt_time(self._duration)}")

    def _do_refresh(self) -> None:
        self._refresh_after_id = None
        if not self._video_path:
            self._clear_image()
            self.canvas.itemconfigure(self._placeholder_id, text="Select a source to preview")
            self._place_frame_image()
            return

        self._render_gen += 1
        gen = self._render_gen
        t = self._time
        video_path = self._video_path
        fps = self._fps

        def work() -> None:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                cap.release()
                self.after(0, lambda: self._show_error(gen))
                return
            frame_idx = max(0, int(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            cap.release()
            if not ok:
                self.after(0, lambda: self._show_error(gen))
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            self.after(0, lambda p=pil, g=gen: self._show_frame(p, g))

        threading.Thread(target=work, daemon=True).start()

    def _show_error(self, gen: int) -> None:
        if gen != self._render_gen:
            return
        self._clear_image()
        self.canvas.itemconfigure(self._placeholder_id, text="Could not read video frame")
        self._place_frame_image()

    def _show_frame(self, pil: Image.Image, gen: int) -> None:
        if gen != self._render_gen:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        fw, fh = pil.size
        scale = min(cw / fw, ch / fh)
        dw = max(1, int(fw * scale))
        dh = max(1, int(fh * scale))
        if (dw, dh) != pil.size:
            pil = pil.resize((dw, dh), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)
        self.canvas.itemconfigure(self._placeholder_id, state="hidden")
        if self._image_id is None:
            self._image_id = self.canvas.create_image(0, 0, anchor="center", image=self._photo)
        else:
            self.canvas.itemconfigure(self._image_id, image=self._photo)
        self._place_frame_image()

    def _clear_image(self) -> None:
        self._photo = None
        if self._image_id is not None:
            self.canvas.delete(self._image_id)
            self._image_id = None
        self.canvas.itemconfigure(self._placeholder_id, state="normal")

    def _place_frame_image(self) -> None:
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        cx, cy = cw // 2, ch // 2
        self.canvas.coords(self._placeholder_id, cx, cy)
        if self._image_id is not None:
            self.canvas.coords(self._image_id, cx, cy)

    def _set_time(self, t: float) -> None:
        t = max(0.0, min(self._duration, t))
        if abs(t - self._time) < 1e-4:
            return
        self._time = t
        self._update_time_label()
        self.refresh()

    def _canvas_to_time(self, x: float) -> float:
        cw = max(1, self.canvas.winfo_width())
        return max(0.0, min(self._duration, x / cw * self._duration))

    def _on_scrub_press(self, event: tk.Event) -> None:
        if not self._video_path:
            return
        self.stop()
        self._set_time(self._canvas_to_time(event.x))

    def _on_scrub_drag(self, event: tk.Event) -> None:
        if not self._video_path:
            return
        self._set_time(self._canvas_to_time(event.x))

    def _toggle_play(self) -> None:
        if not self._video_path:
            return
        if self._playing:
            self.stop()
            return
        if self._time >= self._duration - 0.05:
            self._set_time(0.0)
        self._playing = True
        self.play_btn.configure(text="⏸  Pause")
        self._play_tick()

    def _play_tick(self) -> None:
        if not self._playing:
            return
        step = 1.0 / self._fps
        next_t = self._time + step
        if next_t >= self._duration:
            self._set_time(self._duration)
            self.stop()
            return
        self._set_time(next_t)
        self._play_after_id = self.after(max(16, int(1000 / self._fps)), self._play_tick)
