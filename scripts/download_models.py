#!/usr/bin/env python3
"""Pre-download ML model weights after pip install (avoids SSL errors in the GUI)."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root: python scripts/download_models.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils import configure_https_certs, torch_acceleration_label


def main() -> None:
    configure_https_certs()
    print(f"Backend: {torch_acceleration_label()}")
    print("Downloading Whisper (lyrics transcription)…")
    from app.audio import _get_whisper

    _get_whisper()
    print("Whisper OK")

    print("Downloading EasyOCR (text detection)…")
    from app.text_detection import get_reader

    get_reader()
    print("EasyOCR OK")

    print("Downloading LaMa inpainting model…")
    from app.inpainting import _get_lama

    _get_lama()
    print("LaMa OK")

    print("\nAll models ready. You can run: python main.py")


if __name__ == "__main__":
    main()
