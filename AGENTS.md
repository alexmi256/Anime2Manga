# AGENTS.md

Guidance for agents working in this repository. Keep it up to date when you
change architecture, commands, or conventions.

## What this project is

`anime2manga` turns an anime video (usually `.mkv`) into a manga/storyboard
**markdown** document. Subtitles are the backbone: each subtitle cue is attached
to a scene, and each scene gets a representative frame (or a stitched panorama
when the camera pans). Pipeline steps 1–8 are implemented; cropping (9) and text
placement (10) are stubs.

## Environment and commands

Python **3.12+** (this checkout runs 3.14). `ffmpeg`/`ffprobe` are required at
runtime. Prefer `just`:

```bash
just install     # create .venv and install requirements-dev.txt + `-e .`
just check       # ruff check + pyrefly + pytest  (what CI runs)
just lint        # ruff check src tests
just typecheck   # pyrefly check src tests
just test        # pytest
just run         # pipeline on the bundled input.mkv -> output/report.md
just retarget    # seam-carving experiment over output/frames -> output/retarget
```

If `just` is unavailable, use `.venv/bin/python -m pytest`,
`.venv/bin/ruff check src tests`, `.venv/bin/pyrefly check src tests`.

Runtime dependencies are deliberately small: `numpy`, `opencv-python`,
`scikit-image`. Do not add heavy frameworks.

## Repository layout

- `src/anime2manga/` — the package.
  - `pipeline.py` — orchestration (steps 1–8) and `PipelineConfig`; the seam-carve step lives here (`_step8b_seam_carve`).
  - `cli.py` — argparse CLI and `config_from_args`; one argument group per feature.
  - `models.py` — shared dataclasses (`Scene`, `FaceBox`, `PipelineResult`, …).
  - `report.py` — `report.md` and `scenes.json` writers.
  - `metadata.py`, `subtitles.py`, `scene_detect.py`, `panorama.py`, `inpaint.py`, `timeline.py`, `frames.py`, `audio.py`, `faces.py` — pipeline stages.
  - `cropping.py`, `text_layout.py`, `translation.py`, `motion_vectors.py` — stubs.
  - `seam_carving.py`, `retarget.py` — content-aware retargeting engine + metrics (step 8b / standalone).
  - `retarget_report.py` — rendering for the standalone experiment report.
  - `data/` — bundled anime face ONNX model.
- `scripts/` — `retarget_frames.py` (experiment CLI), `anime2manga.sh`, `fetch_face_model.py`.
- `tests/` — pytest suite; synthetic data only, small and fast.
- `docs/retarget_metrics.md` — full seam-carving parameter/metric reference.
- `README.md` — the user-facing document; keep it in sync with behaviour.

## Code conventions

- Every module opens with a substantial docstring explaining the method and
  references; dataclasses and `StrEnum` for configs/enums.
- Lint is `ruff check` (E, F, I, UP, B, SIM, C4, RUF; E501 ignored) and
  typecheck is `pyrefly`. **`just check` does not run `ruff format`.** Do **not**
  run `ruff format` across existing files: the older tree was not format-clean,
  so it produces large cosmetic diffs that obscure functional changes. Format
  only new files if you want, and prefer targeted edits to existing ones.
- Tests use synthetic images and monkeypatch external calls (ffmpeg, the face
  model) so they never touch the real `input.mkv`. Match the existing style:
  small factories, `tmp_path`, `monkeypatch`.
- Never run `ruff format src tests` as a cleanup step.

## Pipeline ordering and contracts

`Pipeline.run()`: metadata → subtitles → scene detection → panoramas + infill →
timeline validation → frame selection → audio → faces. `_step8_faces` **detects
faces, then seam-carves regular frames, then draws boxes** (`--face-boxes`). The
carve must run before annotation so it sees clean pixels, and it reuses the
detected `FaceBox`es so the model never runs twice. Panoramic scenes are never
seam-carved.

`scenes.json` is machine-readable and must stay in sync with `report.md`.

## Seam carving (step 8b) — the important details

- **Engine** (`seam_carving.py`): L1 gradient energy; seams chosen with
  **forward energy**; faces get a large additive energy inside a dilated mask so
  seams route around them; 50% width cap; frames are optionally downscaled to
  `working_width` before carving. `record_seams`/`reconstruct` allow rebuilding
  any intermediate width.
- **Metrics** (`retarget.py`): guards `energy`, `forward`, `detail`, `ssim`.
  Composite `badness = max(normalised enabled guard signals)`, so the composite
  limit is the **minimum of the enabled guards' stop ratios**; default enabled
  guards are `energy` + `detail`. `COMP` is the recommended stop ratio.
- **Adaptive energy threshold**: `energy_ratio + energy_baseline_multiple *
  max(0, early - energy_reference)`. This is behaviour-identical to the old
  `max(..., multiple*early)` at the defaults (`0.35`/`1.4`, ref `0.25`) but keeps
  `energy_ratio` effective at any value. Do not reintroduce a bare `max()`.
- **Defaults differ intentionally**: the `RetargetConfig`/experiment default is
  `energy_ratio=0.35`; the `anime2manga` pipeline uses `0.25` (user-tuned).
- **Carve resolution**: both the pipeline and the experiment default to
  `working_width=768`, which is ~13x faster than carving 1080p native (measured
  over 261 frames). The pipeline sets it explicitly in its default
  `RetargetConfig`. `working_width=None` (or `--seam-carve-working-width 0`)
  carves at source resolution instead (vertical seams only remove columns, so the
  height is unchanged).
- **Two entry points**: the pipeline step (`--seam-carve*` flags, on by default;
  `--seam-carve-energy-ratio 0` or `--no-seam-carve` disables) writes
  `output/seam_frames/scene_NNNN.jpg` and adds `Seam Carve Shrink Percent:` /
  `Seam Carved Frame Size:` to `report.md`; the experiment script
  (`scripts/retarget_frames.py`) tunes thresholds over a frames folder into
  carved stills, SVG plots, contact sheets, CSV/JSON/Markdown/HTML.
- **Output resolution**: the pipeline carves and emits at the 768px working width
  by default (source resolution if `working_width=None`/`0`). It is surfaced in
  `report.md` via `Seam Carved Frame Size:`. Don't make it silent.
- **Parallelism**: workers use an explicit `multiprocessing.get_context("spawn")`
  context. **Never force `fork`** — `cv2.dnn` face detection runs in the parent
  first and a forked child can deadlock. Parallel entry points need an
  `if __name__ == "__main__"` guard (the CLI and scripts have one); library use
  should set `seam_carve_jobs=1`.
- **Overlay stripping**: the experiment strips the pipeline's green face-box
  overlay by default (`strip_overlays=True`); the pipeline sets it `False`
  because its freshly extracted frames have no boxes. Keep this distinction.
- Full parameter guide: `docs/retarget_metrics.md`.

## Validation expectations

- Run `just check` (or ruff + pyrefly + pytest) before considering work done.
- The suite is currently ~200 passing, 2 skipped; keep it green and add tests
  for new behaviour (especially branches the CLI actually uses, e.g. `jobs > 1`).
- For risky changes, do a bounded real run against `input.mkv`, e.g.
  `python -m anime2manga input.mkv --start-at 02:00 --end-at 02:40 -o /tmp/smoke`.
- `just clean` removes `output/`; `--clean` removes generated experiment
  artefacts under the retarget output dir.
