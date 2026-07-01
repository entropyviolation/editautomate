# EditAutomate

Desktop GUI app to remix TikTok videos:

1. **Download** — highest-quality TikTok video from a link (`yt-dlp`)
2. **Remove text & audio** — OCR detects on-screen text, **LaMa generative inpainting** fills every frame
3. **Replace song** — mux in your replacement track at 320 kbps AAC
4. **Lyrics overlay** — Whisper transcribes the new song; lyrics render in the **same position, size, color, and stroke** as the original captions

## Requirements

- Python 3.10+
- **ffmpeg** installed and on your `PATH`

```bash
brew install ffmpeg   # macOS
```

## Install

```bash
cd editautomate
pip install -r requirements.txt
```

If TikTok downloads fail, update yt-dlp:

```bash
pip install -U yt-dlp
```

First run downloads EasyOCR, Whisper, and LaMa model weights (~1–2 GB total).

## Run

```bash
python main.py
```

Or:

```bash
python -m app.gui
```

## Usage

### Create tab
1. Paste a TikTok URL (or pick an inpainted **source** from the Sources tab)
2. Choose a replacement song (or pick from **Songs** library)
3. Pick an output path (defaults to `~/Movies/EditAutomate/`)
4. Click **Start Processing**
5. Use **Open Output** or **Reveal in Finder** when done

### Songs tab
- Upload songs and view Whisper-transcribed lyrics
- Click any lyric line to edit timestamps or text, then **Save Lyrics**
- Set **snippet start/end** sliders to pick the section of the song for your edit
- **Use in Create** sends the song + snippet to the Create tab

### Sources tab
- Paste a TikTok URL and click **Download & Remove Text** to add an inpainted clip to your library
- Browse saved sources (text removed) from standalone imports or full Create runs
- **Use in Create** skips download & inpainting — remix instantly with a new song

### Studio tab
- Select any finished edit and tweak overlay settings (position, font, size, stroke)
- **Re-render Edit** applies changes without re-running inpainting

### Options

- **Beat-sync** — maps video scene cuts to song BPM; loops video when song is longer
- **Per-frame text detection** — enable when captions move or change; slower but more accurate masks for generative fill.

## Pipeline

```
TikTok URL → download (best quality)
          → OCR text detection + font style capture
          → LaMa inpainting on all frames (OpenCV fallback)
          → beat-sync: map scene cuts to song BPM, loop if needed
          → replace audio track
          → Whisper lyrics + matched font overlay
          → H.264 CRF 17 export
          → saved to Songs / Sources / Studio libraries
```

## Tips

- On Apple Silicon or NVIDIA GPUs, LaMa inpainting and OCR automatically use your GPU (MPS/CUDA). Text is inpainted on a tight crop around captions when possible for extra speed without changing quality.
- For best lyric font match, use TikToks with clear bold on-screen text.
- Output is saved as high-quality MP4 (H.264, slow preset, CRF 17–18).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ffmpeg not found` | Install ffmpeg and restart terminal |
| Slow inpainting | Ensure PyTorch is installed; Apple Silicon and NVIDIA GPUs are used automatically. Shorter clips still process faster. |
| Wrong font | Install Arial Black / Impact; app tries common TikTok fonts |
| Download fails | Check URL is public; update yt-dlp: `pip install -U yt-dlp` |
