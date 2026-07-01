"""Persistent library for songs, inpainted sources, and finished edits."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.audio import LyricLine
from app.text_detection import FontStyle, TextRegion


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _lyric_to_dict(line: LyricLine) -> dict:
    return {"text": line.text, "start": line.start, "end": line.end}


def _lyric_from_dict(d: dict) -> LyricLine:
    return LyricLine(text=d["text"], start=float(d["start"]), end=float(d["end"]))


def _region_to_dict(r: TextRegion) -> dict:
    return {
        "x": r.x,
        "y": r.y,
        "w": r.w,
        "h": r.h,
        "text": r.text,
        "color": list(r.color),
        "font_size": r.font_size,
        "font_name": r.font_name,
        "font_confidence": r.font_confidence,
        "stroke_color": list(r.stroke_color) if r.stroke_color else None,
        "stroke_width": r.stroke_width,
        "confidence": r.confidence,
    }


def _region_from_dict(d: dict) -> TextRegion:
    sc = d.get("stroke_color")
    return TextRegion(
        x=d["x"],
        y=d["y"],
        w=d["w"],
        h=d["h"],
        text=d.get("text", ""),
        color=tuple(d.get("color", [255, 255, 255])),
        font_size=d.get("font_size", 48),
        font_name=d.get("font_name", "Arial Narrow"),
        font_confidence=d.get("font_confidence", 0.0),
        stroke_color=tuple(sc) if sc else None,
        stroke_width=d.get("stroke_width", 2),
        confidence=d.get("confidence", 0.0),
    )


def font_style_to_dict(style: FontStyle) -> dict:
    return {
        "regions": [_region_to_dict(r) for r in style.regions],
        "dominant_font": style.dominant_font,
        "dominant_color": list(style.dominant_color),
        "dominant_size": style.dominant_size,
        "font_identified": style.font_identified,
        "has_stroke": style.has_stroke,
        "stroke_color": list(style.stroke_color),
        "stroke_width": style.stroke_width,
        "vertical_anchor": style.vertical_anchor,
    }


def font_style_from_dict(d: dict) -> FontStyle:
    return FontStyle(
        regions=[_region_from_dict(r) for r in d.get("regions", [])],
        dominant_font=d.get("dominant_font", "Arial Narrow"),
        dominant_color=tuple(d.get("dominant_color", [255, 255, 255])),
        dominant_size=d.get("dominant_size", 48),
        font_identified=d.get("font_identified", False),
        has_stroke=d.get("has_stroke", True),
        stroke_color=tuple(d.get("stroke_color", [0, 0, 0])),
        stroke_width=d.get("stroke_width", 2),
        vertical_anchor=d.get("vertical_anchor", "center"),
    )


@dataclass
class OverlayTweak:
    offset_x: int = 0
    offset_y: int = 0
    font_size: int | None = None
    font_name: str | None = None
    color: tuple[int, int, int] | None = None
    has_stroke: bool | None = None
    stroke_color: tuple[int, int, int] | None = None
    stroke_width: int | None = None

    def to_dict(self) -> dict:
        return {
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "font_size": self.font_size,
            "font_name": self.font_name,
            "color": list(self.color) if self.color else None,
            "has_stroke": self.has_stroke,
            "stroke_color": list(self.stroke_color) if self.stroke_color else None,
            "stroke_width": self.stroke_width,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> OverlayTweak:
        if not d:
            return cls()
        c = d.get("color")
        sc = d.get("stroke_color")
        return cls(
            offset_x=d.get("offset_x", 0),
            offset_y=d.get("offset_y", 0),
            font_size=d.get("font_size"),
            font_name=d.get("font_name"),
            color=tuple(c) if c else None,
            has_stroke=d.get("has_stroke"),
            stroke_color=tuple(sc) if sc else None,
            stroke_width=d.get("stroke_width"),
        )


@dataclass
class SongSnippet:
    id: str
    name: str
    start: float
    end: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SongSnippet:
        return cls(
            id=d["id"],
            name=d["name"],
            start=float(d["start"]),
            end=d.get("end"),
        )


@dataclass
class SongRecord:
    id: str
    title: str
    path: str
    added_at: str
    lyrics: list[LyricLine] = field(default_factory=list)
    bpm: float = 0.0
    snippet_start: float = 0.0
    snippet_end: float | None = None
    snippets: list[SongSnippet] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "path": self.path,
            "added_at": self.added_at,
            "lyrics": [_lyric_to_dict(l) for l in self.lyrics],
            "bpm": self.bpm,
            "snippet_start": self.snippet_start,
            "snippet_end": self.snippet_end,
            "snippets": [s.to_dict() for s in self.snippets],
        }

    @classmethod
    def from_dict(cls, d: dict) -> SongRecord:
        snippets = [SongSnippet.from_dict(s) for s in d.get("snippets", [])]
        snippet_start = float(d.get("snippet_start", 0))
        snippet_end = d.get("snippet_end")
        if not snippets and (snippet_start > 0 or snippet_end is not None):
            snippets = [
                SongSnippet(
                    id=_new_id(),
                    name="Saved snippet",
                    start=snippet_start,
                    end=snippet_end,
                )
            ]
        return cls(
            id=d["id"],
            title=d["title"],
            path=d["path"],
            added_at=d["added_at"],
            lyrics=[_lyric_from_dict(l) for l in d.get("lyrics", [])],
            bpm=float(d.get("bpm", 0)),
            snippet_start=snippet_start,
            snippet_end=snippet_end,
            snippets=snippets,
        )

    def get_snippet(self, snippet_id: str) -> SongSnippet | None:
        return next((s for s in self.snippets if s.id == snippet_id), None)


@dataclass
class SourceRecord:
    id: str
    title: str
    path: str
    tiktok_url: str
    added_at: str
    font_style: FontStyle = field(default_factory=FontStyle)
    thumbnail: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "path": self.path,
            "tiktok_url": self.tiktok_url,
            "added_at": self.added_at,
            "font_style": font_style_to_dict(self.font_style),
            "thumbnail": self.thumbnail,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SourceRecord:
        return cls(
            id=d["id"],
            title=d["title"],
            path=d["path"],
            tiktok_url=d.get("tiktok_url", ""),
            added_at=d["added_at"],
            font_style=font_style_from_dict(d.get("font_style", {})),
            thumbnail=d.get("thumbnail"),
        )


@dataclass
class EditRecord:
    id: str
    title: str
    output_path: str
    source_id: str
    song_id: str
    added_at: str
    with_audio_path: str
    font_style: FontStyle = field(default_factory=FontStyle)
    lyrics: list[LyricLine] = field(default_factory=list)
    tweak: OverlayTweak = field(default_factory=OverlayTweak)
    snippet_start: float = 0.0
    snippet_end: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "output_path": self.output_path,
            "source_id": self.source_id,
            "song_id": self.song_id,
            "added_at": self.added_at,
            "with_audio_path": self.with_audio_path,
            "font_style": font_style_to_dict(self.font_style),
            "lyrics": [_lyric_to_dict(l) for l in self.lyrics],
            "tweak": self.tweak.to_dict(),
            "snippet_start": self.snippet_start,
            "snippet_end": self.snippet_end,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EditRecord:
        return cls(
            id=d["id"],
            title=d["title"],
            output_path=d["output_path"],
            source_id=d["source_id"],
            song_id=d["song_id"],
            added_at=d["added_at"],
            with_audio_path=d["with_audio_path"],
            font_style=font_style_from_dict(d.get("font_style", {})),
            lyrics=[_lyric_from_dict(l) for l in d.get("lyrics", [])],
            tweak=OverlayTweak.from_dict(d.get("tweak")),
            snippet_start=float(d.get("snippet_start", 0)),
            snippet_end=d.get("snippet_end"),
        )


@dataclass
class TikTokAccount:
    id: str
    label: str
    username: str
    session_id: str
    added_at: str
    last_export_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "username": self.username,
            "session_id": self.session_id,
            "added_at": self.added_at,
            "last_export_at": self.last_export_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TikTokAccount:
        return cls(
            id=d["id"],
            label=d["label"],
            username=d.get("username", ""),
            session_id=d["session_id"],
            added_at=d["added_at"],
            last_export_at=d.get("last_export_at"),
        )

    def display_name(self) -> str:
        handle = self.username.strip().lstrip("@")
        if handle:
            return f"{self.label} (@{handle})"
        return self.label


class Library:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.songs_dir = root / "library" / "songs"
        self.sources_dir = root / "library" / "sources"
        self.edits_dir = root / "library" / "edits"
        self.accounts_dir = root / "library" / "accounts"
        self.media_dir = root / "library" / "media"
        for d in (self.songs_dir, self.sources_dir, self.edits_dir, self.accounts_dir, self.media_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _read_index(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        return json.loads(path.read_text())

    def _write_index(self, path: Path, items: list[dict]) -> None:
        path.write_text(json.dumps(items, indent=2))

    # --- Songs ---

    def list_songs(self) -> list[SongRecord]:
        return [SongRecord.from_dict(d) for d in self._read_index(self.songs_dir / "index.json")]

    def get_song(self, song_id: str) -> SongRecord | None:
        return next((s for s in self.list_songs() if s.id == song_id), None)

    def add_song(
        self,
        audio_path: Path,
        lyrics: list[LyricLine] | None = None,
        bpm: float = 0.0,
        title: str | None = None,
    ) -> SongRecord:
        existing = next(
            (s for s in self.list_songs() if Path(s.path).resolve() == audio_path.resolve()),
            None,
        )
        if existing:
            if lyrics:
                existing.lyrics = lyrics
                existing.bpm = bpm
                self.update_song(existing)
            return existing

        song_id = _new_id()
        dest = self.media_dir / f"song_{song_id}{audio_path.suffix}"
        shutil.copy2(audio_path, dest)

        record = SongRecord(
            id=song_id,
            title=title or audio_path.stem,
            path=str(dest),
            added_at=_now_iso(),
            lyrics=lyrics or [],
            bpm=bpm,
        )
        items = self._read_index(self.songs_dir / "index.json")
        items.insert(0, record.to_dict())
        self._write_index(self.songs_dir / "index.json", items)
        return record

    def update_song(self, record: SongRecord) -> None:
        items = self._read_index(self.songs_dir / "index.json")
        items = [record.to_dict() if d.get("id") == record.id else d for d in items]
        self._write_index(self.songs_dir / "index.json", items)

    def delete_song(self, song_id: str) -> None:
        items = [d for d in self._read_index(self.songs_dir / "index.json") if d.get("id") != song_id]
        self._write_index(self.songs_dir / "index.json", items)

    # --- Sources ---

    def list_sources(self) -> list[SourceRecord]:
        return [SourceRecord.from_dict(d) for d in self._read_index(self.sources_dir / "index.json")]

    def get_source(self, source_id: str) -> SourceRecord | None:
        return next((s for s in self.list_sources() if s.id == source_id), None)

    def add_source(
        self,
        video_path: Path,
        tiktok_url: str,
        font_style: FontStyle,
        title: str | None = None,
    ) -> SourceRecord:
        source_id = _new_id()
        dest = self.media_dir / f"source_{source_id}.mp4"
        shutil.copy2(video_path, dest)

        record = SourceRecord(
            id=source_id,
            title=title or f"Source {source_id}",
            path=str(dest),
            tiktok_url=tiktok_url,
            added_at=_now_iso(),
            font_style=font_style,
        )
        items = self._read_index(self.sources_dir / "index.json")
        items.insert(0, record.to_dict())
        self._write_index(self.sources_dir / "index.json", items)
        return record

    def delete_source(self, source_id: str) -> None:
        items = [d for d in self._read_index(self.sources_dir / "index.json") if d.get("id") != source_id]
        self._write_index(self.sources_dir / "index.json", items)

    # --- Edits ---

    def list_edits(self) -> list[EditRecord]:
        return [EditRecord.from_dict(d) for d in self._read_index(self.edits_dir / "index.json")]

    def get_edit(self, edit_id: str) -> EditRecord | None:
        return next((e for e in self.list_edits() if e.id == edit_id), None)

    def add_edit(
        self,
        output_path: Path,
        with_audio_path: Path,
        source_id: str,
        song_id: str,
        font_style: FontStyle,
        lyrics: list[LyricLine],
        tweak: OverlayTweak | None = None,
        snippet_start: float = 0.0,
        snippet_end: float | None = None,
        title: str | None = None,
    ) -> EditRecord:
        edit_id = _new_id()
        record = EditRecord(
            id=edit_id,
            title=title or output_path.stem,
            output_path=str(output_path),
            source_id=source_id,
            song_id=song_id,
            added_at=_now_iso(),
            with_audio_path=str(with_audio_path),
            font_style=font_style,
            lyrics=lyrics,
            tweak=tweak or OverlayTweak(),
            snippet_start=snippet_start,
            snippet_end=snippet_end,
        )
        items = self._read_index(self.edits_dir / "index.json")
        items.insert(0, record.to_dict())
        self._write_index(self.edits_dir / "index.json", items)
        return record

    def update_edit(self, record: EditRecord) -> None:
        items = self._read_index(self.edits_dir / "index.json")
        items = [record.to_dict() if d.get("id") == record.id else d for d in items]
        self._write_index(self.edits_dir / "index.json", items)

    def delete_edit(self, edit_id: str) -> None:
        items = [d for d in self._read_index(self.edits_dir / "index.json") if d.get("id") != edit_id]
        self._write_index(self.edits_dir / "index.json", items)

    # --- TikTok accounts ---

    def list_accounts(self) -> list[TikTokAccount]:
        return [TikTokAccount.from_dict(d) for d in self._read_index(self.accounts_dir / "index.json")]

    def get_account(self, account_id: str) -> TikTokAccount | None:
        return next((a for a in self.list_accounts() if a.id == account_id), None)

    def add_account(self, label: str, session_id: str, username: str = "") -> TikTokAccount:
        account_id = _new_id()
        record = TikTokAccount(
            id=account_id,
            label=label.strip() or f"Account {account_id[:6]}",
            username=username.strip().lstrip("@"),
            session_id=session_id.strip(),
            added_at=_now_iso(),
        )
        items = self._read_index(self.accounts_dir / "index.json")
        items.insert(0, record.to_dict())
        self._write_index(self.accounts_dir / "index.json", items)
        return record

    def update_account(self, record: TikTokAccount) -> None:
        items = self._read_index(self.accounts_dir / "index.json")
        items = [record.to_dict() if d.get("id") == record.id else d for d in items]
        self._write_index(self.accounts_dir / "index.json", items)

    def delete_account(self, account_id: str) -> None:
        items = [d for d in self._read_index(self.accounts_dir / "index.json") if d.get("id") != account_id]
        self._write_index(self.accounts_dir / "index.json", items)
