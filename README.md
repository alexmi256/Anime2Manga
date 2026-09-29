# Anime2Manga

Turn an anime video (usually `.mkv`) into a manga/storyboard **markdown**
document.  Subtitles are the backbone: every cue in the chosen track is attached
to a scene, and each scene gets a clear representative frame (or a stitched
panorama when the camera pans).

This first milestone implements **steps 1–6** of the pipeline and emits
`output/report.md` plus `output/scenes.json` for review.  Later stages (audio
focus, face detection, crop planning, text placement, subtitle translation) are
documented stubs.

## Requirements

* Python 3.12+
* `ffmpeg` and `ffprobe` on `PATH`
* Linux/macOS (bash scripts provided)

`tesseract`/ImageMagick are **not** required for this milestone.  Bitmap
subtitles (PGS/VobSub) are explicitly rejected because they would need OCR.

## Install

```bash
just install          # or: python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

## Usage

```bash
# Convert the bundled sample; writes output/report.md
just run

# Or with the bash wrapper:
scripts/anime2manga.sh input.mkv -o output

# Skip an opening and stop before the credits:
scripts/anime2manga.sh input.mkv -o output --start-at 02:00 --end-at 20:00

# Pick a different subtitle language or an explicit track id:
scripts/anime2manga.sh input.mkv --subtitle-language por
scripts/anime2manga.sh input.mkv --subtitle-track 5
```

Invalid language/track selections fail with a list of the available tracks, and
a video with no subtitles fails with a clear message.

## Pipeline

| Step | Module | Status |
| ---- | ------ | ------ |
| 1. Container metadata, intro/credits chapters, clip window | `metadata.py` | implemented |
| 2. Subtitle track selection, extraction and parsing (SRT/ASS) | `subtitles.py` | implemented |
| 3. Scene detection (`select` / `scdet`) | `scene_detect.py` | implemented |
| 4. Pan detection and panoramic stitching, cross-scene retiming | `panorama.py` | implemented |
| 5. Timeline validation (no gaps/overlaps) | `timeline.py` | implemented |
| 6. Frame sampling and clearest-frame selection | `frames.py` | implemented |
| 7. Left/right audio focus | `audio.py` | stub |
| 8. Face detection | `faces.py` | stub |
| 9. Face-aware cropping | `cropping.py` | stub |
| 10. Text/bubble placement | `text_layout.py` | stub |
| — Subtitle translation | `translation.py` | stub |
| — Motion-vector pan fast-path | `motion_vectors.py` | research stub |

## How frames and panoramas are chosen

* **Scene detection** uses ffmpeg's `select='gt(scene,T)'` by default (score
  0–1) or `scdet` (score 0–100).  Tune with `--scene-method` /
  `--scene-threshold`.
* **Pan detection** samples the scene at 4 fps and runs `cv2.phaseCorrelate`
  (Hanning window) between consecutive grayscale frames.  A pan is declared only
  when the cumulative shift, directional consistency and median correlation
  response all clear their thresholds (`--pan-min-shift`, `--pan-consistency`,
  `--pan-response`).  If the motion continues into the next scene it peeks ahead
  (`--pan-peek`) and retimes the boundary before stitching.  Frames are
  translated onto a canvas and blended; a panoramic panel is never square.
* **Frame selection** targets the median subtitle midpoint (or the scene
  midpoint) and picks the sharpest candidate within `--selection-window` seconds
  using the variance of the Laplacian.
* **Overloaded scenes** carrying more than `--max-subtitles-per-scene` cues are
  re-detected at a lower threshold (`--subdivide-factor`) to yield more panels.

## Debug output

Running the pipeline prints, and logs to `output/scenes.json`:

* subtitle line count (raw and within the clip),
* number of scenes identified for the current settings,
* per-scene pan direction, shift, consistency, response and panorama size,
* chosen frame time, sharpness and subtitle count.

`--keep-analysis` keeps the sampled analysis frames under `output/work/`.

## Development

```bash
just lint        # ruff
just typecheck   # pyrefly
just test        # pytest
just check       # all of the above
```

## Report format

`output/report.md` contains one section per scene:

```markdown
# Anime2Manga
Input File: input.mkv
Subtitle Tracks:
- 3: English
Duration: 00:22:47.533
...

# Scene Number 1
Start Time: 00:02:02.740
End Time: 00:02:05.750
Start Frame: 2946
End Frame: 3020

## Chosen Frame
![Frame Image](frames/scene_0001.jpg)
Frame Size: 1920x1080
Is Panoramic: No
Frame Time: 00:02:04.245

## Text
- What? Two death's heads again?
```
