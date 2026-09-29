# Anime2Manga

Turn an anime video (usually `.mkv`) into a manga/storyboard **markdown**
document.  Subtitles are the backbone: every cue in the chosen track is attached
to a scene, and each scene gets a clear representative frame (or a stitched
panorama when the camera pans).

This milestone implements **steps 1–8** of the pipeline - including **left/right
audio focus (step 7)** and **face detection (step 8)** - and emits
`output/report.md` plus `output/scenes.json` for review.  Later stages (crop
planning, text placement, subtitle translation) are documented stubs.

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

# Clear generated output so the next `just run` leaves no stale files behind
just clean

# Or with the bash wrapper:
scripts/anime2manga.sh input.mkv -o output

# Skip an opening and stop before the credits:
scripts/anime2manga.sh input.mkv -o output --start-at 02:00 --end-at 20:00

# Pick a different subtitle language or an explicit track id:
scripts/anime2manga.sh input.mkv --subtitle-language por
scripts/anime2manga.sh input.mkv --subtitle-track 5

# Face boxes are drawn on the chosen frames by default; turn drawing off (boxes
# are still detected and listed in the report):
scripts/anime2manga.sh input.mkv --no-face-boxes

# Skip left/right audio focus detection entirely:
scripts/anime2manga.sh input.mkv --no-audio-direction

# Make the left/right call more or less sensitive (default: 1.5 dB imbalance):
scripts/anime2manga.sh input.mkv --audio-balance-db 5

# Narrow or widen the speech band used for the balance comparison:
scripts/anime2manga.sh input.mkv --audio-band-low 200 --audio-band-high 4000
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
| 7. Left/right audio focus | `audio.py` | implemented |
| 8. Face detection, bounding boxes on chosen frames | `faces.py` | implemented |
| 9. Face-aware cropping | `cropping.py` | stub |
| 10. Text/bubble placement | `text_layout.py` | stub |
| — Subtitle translation | `translation.py` | stub |
| — Motion-vector pan fast-path | `motion_vectors.py` | research stub |

## How frames and panoramas are chosen

* **Scene detection** uses ffmpeg's `select='gt(scene,T)'` by default (score
  0–1) or `scdet` (score 0–100).  Tune with `--scene-method` /
  `--scene-threshold`.
* **Pan detection** samples the scene at 4 fps and runs `cv2.phaseCorrelate`
  (Hanning window) between consecutive grayscale frames.  Pairs whose response
  is too low to trust (cuts, repeats, motion blur) are treated as breaks, so a
  cut is never stitched across.  Within each reliable run the strongest
  *contiguous* motion segment is chosen and a pan is declared only when the
  cumulative shift, directional consistency and median response over the moving
  pairs all clear their thresholds (`--pan-min-shift`, `--pan-consistency`,
  `--pan-response`, `--pan-pair-response`, `--pan-diagonal-ratio`).  Diagonal
  moves are named `up-left`, `down-right`, …  If the motion continues into the next scene it
  peeks ahead (`--pan-peek`) and retimes the boundary before stitching.  When a
  scene holds several shots, the scene is split so only the panning span becomes
  a panoramic panel; the surrounding shots keep their own frames.  The pan keeps
  its last sampled frame, so the next span starts one sample later instead of
  repeating the panorama's edge.
* **Panoramas** are stitched by translating each frame onto a canvas and
  blending overlaps; a panoramic panel is never square.  Colour panoramas are
  written as **PNG with an alpha channel**: transparent pixels are areas no
  sampled frame covered, ready for a later pass to fill in.  Non-panning frames
  stay JPEG.
* **Frame selection** targets the median subtitle midpoint (or the scene
  midpoint) and picks the sharpest candidate within `--selection-window` seconds
  using the variance of the Laplacian.  Sampling is half-open (`[start, end)`),
  so a frame sitting exactly on a scene boundary belongs to the next scene and
  is never stitched into the preceding panorama or reused as its own panel.
* **Post-panorama slivers**: a scene this short (default `--pan-merge-max-len
  1.0`, `0` disables) right after a panorama is folded into the panorama.  It
  usually shows the tail of the same shot, so on its own it would appear as a
  near-duplicate panel.
* **Overloaded scenes** carrying more than `--max-subtitles-per-scene` cues are
  re-detected at a lower threshold (`--subdivide-factor`) to yield more panels.

## How audio direction is found

Each scene is labelled `Left`, `Center` or `Right` (stored as lower-case
`audio_focus` in `scenes.json`, with the measured `audio_balance_db`) so step 10
can place a text bubble on the side the voice is coming from.

The feature is deliberately small and dependency-free: a compact `audio.py`
module using numpy on top of the ffmpeg the pipeline already requires.

* **No dedicated library.** There is no turnkey "stereo direction" package:
  `librosa`, `pydub` and `soundfile` can all read channels, but they are heavy
  new dependencies and their RMS helpers do not answer a left/right question.
  ffmpeg's `astats` filter does report per-channel RMS (`RMS level dB`) and
  would work, but parsing its human-readable log is brittle, so we decode raw
  PCM instead.  The dB-balance metric below is the standard one used for
  speaker/headphone channel tests ([AudioCheck](https://www.audiocheck.net/audiotests_stereo.php),
  [dsp.stackexchange.com/questions/27221](https://dsp.stackexchange.com/questions/27221)).
* **Mono is skipped.** `ffprobe` reports the chosen stream's channel count; a
  genuine single-channel (mono) source has no left/right pair, so the scene is
  reported `Center` without decoding anything.  An unknown count (`0`) is still
  decoded, in case the stream is really stereo.
* **Any codec is supported.** The span is decoded to raw `f32le` by ffmpeg, so
  E-AC-3, AAC, Vorbis/OGG, FLAC, Opus, ... all work.  `-ac 2` downmixes 5.1/7.1
  to stereo with ffmpeg's standard coefficients.  Analysis runs at 16 kHz
  (`AudioFocusConfig.analysis_sample_rate`) because only the energy balance
  matters.
* **Speech band, not full band.** The left/right comparison is made inside the
  speech band (`--audio-band-low`/`--audio-band-high`, default 300–3400 Hz)
  rather than over the whole spectrum.  Dialogue lives in that band, while
  centered music and ambience mostly sit outside it and would otherwise mask a
  panned voice.  On the bundled `input.mkv` this labels 23 scenes at the default
  threshold (vs 15 for a full-band comparison), with the strongest reaching
  ±5.6 dB.
* **Metric.** For each channel we take the power inside the speech band over the
  scene and compare in dB:
  `balance_db = 10*log10(power_left / power_right)`.  Positive favours the left
  channel, negative the right.  The same band power gates silence: a scene below
  `--audio-silence-db` (default `-60 dBFS`) is `Center`; otherwise a
  `|balance_db|` of at least `--audio-balance-db` (default `1.5 dB`) names a
  side, and anything smaller is `Center`.  No full-band measurement feeds the
  direction decision.  The default was checked against the
  `audiocheck.net_L.ogg` / `audiocheck.net_R.ogg` samples, which score far
  beyond the threshold in the correct direction.

## How faces are found

Anime faces are stylised (flat cel shading, oversized eyes, exaggerated
geometry), so photographic detectors such as the Haar frontal-face cascade or
YuNet miss many of them.  Face detection uses the
[DeepGHS anime face detector](https://huggingface.co/deepghs/anime_face_detection)
(`face_detect_v1.4_n`, MIT): a YOLOv8n single-class model trained on anime
faces (F1 ≈ 0.94).  It is bundled under `src/anime2manga/data/` and runs through
`cv2.dnn`, so there is no ML framework or model download.

1. Letterbox the frame to the model's square input (default 960, preserving
   aspect ratio) at several content scales. A scale of `1.0` sees faces at their
   normal size, while `0.5` shrinks the frame inside the canvas so very large
   close-up faces fall back into the model's training scale; a single 960 pass
   misses them.
2. Run the network at each scale, decode the `(cx, cy, w, h, score)` rows and
   keep detections above `score_threshold` (default 0.2, below the model card's
   F1 optimum because missed faces were the priority).
3. Non-maximum-suppress overlaps across all scales and return full-resolution
   frame-pixel boxes.

The model can be overridden with `FaceDetectionConfig.model_path` or the
`ANIME2MANGA_ANIME_FACE_MODEL` environment variable; `content_scales` and
`score_threshold` are tunable on `FaceDetectionConfig`.

By default boxes are also drawn onto the saved chosen frame (`--no-face-boxes`
disables the drawing; the boxes are still detected and listed in the report).

Detection is not perfect: very stylised or masked/non-human faces can still be
missed, and decorative patterns that resemble a face (e.g. skull ornaments) can
occasionally produce a false positive.  Raise `score_threshold` to trade recall
for precision, or adjust `content_scales`.

## Debug output

Running the pipeline prints, and logs to `output/scenes.json`:

* subtitle line count (raw and within the clip),
* number of scenes identified for the current settings,
* per-scene pan direction, shift, consistency, response and panorama size,
* the absolute start/end of each detected pan segment,
* chosen frame time, sharpness, subtitle count and detected face count,
* per-scene audio direction and channel balance in dB, plus a
  `left=… center=… right=…` summary.

`--keep-analysis` keeps the sampled analysis frames under `output/work/`.

## Development

```bash
just lint        # ruff
just typecheck   # pyrefly
just test        # pytest
just check       # all of the above
just clean       # remove output/
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

Audio Direction: Left
Start Time: 00:02:02.740
End Time: 00:02:05.750
Start Frame: 2946
End Frame: 3020

## Chosen Frame
![Frame Image](frames/scene_0001.jpg)
Frame Size: 1920x1080
Is Panoramic: No
Frame Time: 00:02:04.245
Frame Number: 2979
Face Bounding Boxes:
- x=812, y=356, width=180, height=196, confidence=0.74

## Text
- What? Two death's heads again?

# Scene Number 2

Audio Direction: Center
Start Time: 00:02:05.750
End Time: 00:02:09.300
Start Frame: 3020
End Frame: 3105

## Chosen Frame
![Frame Image](panoramas/scene_00302000.png)
Frame Size: 2400x1080
Is Panoramic: Yes
Pan Direction: up-left
Pan Start Time: 00:02:05.750
Pan End Time: 00:02:09.000
Pan Start Frame: 3020
Pan End Frame: 3094
Frame Time: 00:02:07.525
Face Bounding Boxes:
- (no faces detected)

## Text
- (no subtitles)
```
