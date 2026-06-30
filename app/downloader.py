"""Download TikTok videos at highest available quality."""

from __future__ import annotations

from pathlib import Path

import yt_dlp

from app.utils import ProgressCallback, default_progress, ensure_dir

_MIN_YTDLP = (2024, 12, 1)


def _parse_ytdlp_version() -> tuple[int, ...]:
    try:
        from yt_dlp.version import __version__ as version
    except ImportError:
        version = getattr(yt_dlp, "version", None)
        if hasattr(version, "__version__"):
            version = version.__version__
        else:
            version = "0"
    parts: list[int] = []
    for piece in str(version).replace("_", ".").split("."):
        if piece.isdigit():
            parts.append(int(piece))
        else:
            break
    return tuple(parts) or (0,)


def _version_ok() -> bool:
    current = _parse_ytdlp_version()
    return current >= _MIN_YTDLP


def _progress_hook_factory(progress: ProgressCallback, last_pct: list[float]):
    def hook(status: dict) -> None:
        if status.get("status") == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
            downloaded = status.get("downloaded_bytes", 0)
            if total:
                pct = downloaded / total
                if pct - last_pct[0] >= 0.02:
                    last_pct[0] = pct
                    progress("Downloading…", 0.05 + pct * 0.15)

    return hook


def _base_opts(output_dir: Path, progress: ProgressCallback, last_pct: list[float]) -> dict:
    opts: dict = {
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_progress_hook_factory(progress, last_pct)],
        "postprocessors": [
            {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},
        ],
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.tiktok.com/",
        },
    }
    return opts


def _resolve_downloaded_path(output_dir: Path, info: dict) -> Path:
    video_id = info.get("id", "video")
    ext = info.get("ext", "mp4")
    candidate = output_dir / f"{video_id}.{ext}"
    if candidate.exists():
        return candidate
    for path in output_dir.glob(f"{video_id}.*"):
        if path.suffix.lower() in {".mp4", ".mkv", ".webm"}:
            return path
    raise FileNotFoundError(f"Downloaded file not found for id {video_id}")


def _attempt_download(url: str, ydl_opts: dict) -> dict:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise RuntimeError("Could not extract video info from URL")
        return info


def download_tiktok(url: str, output_dir: Path, progress: ProgressCallback = default_progress) -> Path:
    ensure_dir(output_dir)
    progress("Downloading TikTok video…", 0.05)

    if not _version_ok():
        ver = ".".join(str(p) for p in _parse_ytdlp_version())
        raise RuntimeError(
            f"yt-dlp {ver} is too old for TikTok. Run: pip install -U yt-dlp"
        )

    last_pct = [0.0]
    strategies: list[tuple[str, dict]] = [
        ("default", _base_opts(output_dir, progress, last_pct)),
    ]

    impersonate_opts = _base_opts(output_dir, progress, last_pct)
    impersonate_opts["impersonate"] = "chrome"
    strategies.append(("chrome impersonation", impersonate_opts))

    cookie_opts = _base_opts(output_dir, progress, last_pct)
    cookie_opts["cookiesfrombrowser"] = ("chrome",)
    strategies.append(("browser cookies", cookie_opts))

    errors: list[str] = []
    for label, opts in strategies:
        try:
            progress(f"Downloading ({label})…", 0.08)
            info = _attempt_download(url, opts)
            path = _resolve_downloaded_path(output_dir, info)
            progress("Download complete", 0.2)
            return path
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    hint = "pip install -U yt-dlp"
    if "sigi" in " ".join(errors).lower():
        hint = (
            "TikTok blocked the download. Update yt-dlp first:\n"
            "  pip install -U yt-dlp\n"
            "Then retry. If it still fails, open the TikTok link in Chrome "
            "while logged in (cookies help)."
        )
    raise RuntimeError(f"TikTok download failed.\n{hint}\n\nDetails:\n" + "\n".join(errors))
