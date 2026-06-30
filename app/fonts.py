"""System font discovery and TikTok-style fallback constants."""

from __future__ import annotations

import platform
from pathlib import Path

# Fonts commonly used in TikTok captions / lyrics overlays
TIKTOK_FONT_CANDIDATES = [
    "Arial Narrow",
    "Arial Black",
    "Arial Bold",
    "Arial",
    "Helvetica Neue",
    "Helvetica",
    "Impact",
    "Futura",
    "Proxima Nova",
    "Montserrat Bold",
    "Montserrat",
    "SF Pro Display",
    "Segoe UI Bold",
    "Segoe UI",
    "Verdana Bold",
    "Verdana",
    "Trebuchet MS Bold",
    "Trebuchet MS",
]

FALLBACK_FONT = "Arial Narrow"
FALLBACK_HORIZONTAL_STRETCH = 0.90
FALLBACK_BLEED_BLUR_RADIUS = 1.6
FALLBACK_BLEED_OPACITY = 0.55
FONT_MATCH_MIN_CONFIDENCE = 0.48
FONT_MATCH_MIN_AGREEMENT = 0.38


def system_font_dirs() -> list[Path]:
    system = platform.system()
    if system == "Darwin":
        return [
            Path("/System/Library/Fonts"),
            Path("/System/Library/Fonts/Supplemental"),
            Path("/Library/Fonts"),
            Path.home() / "Library/Fonts",
        ]
    if system == "Windows":
        return [Path(r"C:\Windows\Fonts")]
    return [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
    ]


def find_font_file(name: str) -> Path | None:
    patterns = [
        f"{name}.ttf",
        f"{name}.ttc",
        f"{name} Bold.ttf",
        f"{name}-Bold.ttf",
        f"{name.replace(' ', '')}.ttf",
        f"{name.replace(' ', '')}.ttc",
    ]
    if name == "Arial Narrow":
        patterns.extend(
            [
                "Arial Narrow.ttf",
                "ArialNarrow.ttf",
                "arialn.ttf",
                "Arial Narrow Bold.ttf",
            ]
        )

    for directory in system_font_dirs():
        if not directory.exists():
            continue
        for pattern in patterns:
            candidate = directory / pattern
            if candidate.exists():
                return candidate
        normalized = name.lower().replace(" ", "")
        for path in directory.rglob("*.ttf"):
            if normalized in path.name.lower().replace(" ", "").replace("-", ""):
                return path
        for path in directory.rglob("*.ttc"):
            if normalized in path.name.lower().replace(" ", "").replace("-", ""):
                return path
    return None


def font_available(name: str) -> bool:
    return find_font_file(name) is not None
