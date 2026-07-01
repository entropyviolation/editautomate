"""TikTok account login helpers and library edit uploads."""

from __future__ import annotations

import time
from pathlib import Path

from app.utils import ProgressCallback, default_progress


def build_post_description(caption: str, hashtags: str | list[str]) -> str:
    """Combine caption text and hashtags into a TikTok post description."""
    cap = caption.strip()
    if isinstance(hashtags, str):
        raw = hashtags.replace(",", " ").split()
    else:
        raw = hashtags
    tags = [t.strip().lstrip("#") for t in raw if t.strip()]
    tag_str = " ".join(f"#{tag}" for tag in tags if tag)
    if cap and tag_str:
        return f"{cap}\n\n{tag_str}"
    return cap or tag_str


def _session_cookies(session_id: str) -> list[dict]:
    return [
        {
            "name": "sessionid",
            "value": session_id.strip(),
            "domain": ".tiktok.com",
            "path": "/",
        }
    ]


def _require_playwright() -> None:
    try:
        import playwright  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for TikTok login and uploads.\n\n"
            "Run:\n"
            "  pip install playwright tiktok-uploader\n"
            "  playwright install chromium"
        ) from exc


def _require_uploader() -> None:
    try:
        import tiktok_uploader  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "tiktok-uploader is required to post to TikTok.\n\n"
            "Run:\n"
            "  pip install tiktok-uploader\n"
            "  playwright install chromium"
        ) from exc


def capture_session_via_browser(timeout_sec: float = 300.0) -> str:
    """Open TikTok login in Chromium and return sessionid once the user signs in."""
    _require_playwright()
    from playwright.sync_api import sync_playwright

    session_id: str | None = None
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.tiktok.com/login", wait_until="domcontentloaded")

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            for cookie in context.cookies("https://www.tiktok.com"):
                if cookie.get("name") == "sessionid" and cookie.get("value"):
                    session_id = cookie["value"]
                    break
            if session_id:
                break
            time.sleep(0.5)

        browser.close()

    if not session_id:
        raise RuntimeError(
            "Login timed out — finish signing in to TikTok in the browser window, then try again."
        )
    return session_id


def upload_video(
    video_path: Path,
    session_id: str,
    caption: str,
    hashtags: str | list[str] = "",
    *,
    progress: ProgressCallback = default_progress,
) -> None:
    """Upload a finished edit to TikTok using a saved account session."""
    _require_uploader()
    from tiktok_uploader.upload import TikTokUploader

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    description = build_post_description(caption, hashtags)
    progress("Opening TikTok uploader…", 0.08)

    uploader = TikTokUploader(cookies_list=_session_cookies(session_id))
    progress("Uploading video to TikTok…", 0.25)

    success = uploader.upload_video(str(video_path), description=description)
    if not success:
        raise RuntimeError(
            "TikTok upload failed. Your session may have expired — log in again from the Accounts tab."
        )

    progress("Posted to TikTok", 1.0)
