# Seam-carving retarget metrics — parameter guide

This documents the seam-carving retarget engine and metrics
(`src/anime2manga/seam_carving.py`, `src/anime2manga/retarget.py`).  The engine
also has a compiled C++ backend (`src/anime2manga/_seamcarve.cpp`, built into
`anime2manga._seamcarve` by `just install`) that `carve_width` uses when
available; it is behaviourally identical to the pure-Python loop.  The goal is
to answer one question per frame: **how far can this frame be seam-carved
before it looks subjectively bad?**

There are two entry points:

* **The pipeline (step 8b).**  `anime2manga` carves every regular frame as part
  of a run, using the knobs below via CLI flags (`--seam-carve`,
  `--seam-carve-energy-ratio`, `--seam-carve-max-shrink`,
  `--seam-carve-working-width`, `--seam-carve-detail-budget`,
  `--seam-carve-jobs`, `--seam-carve-quality`).  It reuses step 8's detected
  boxes (every enabled category), runs
  before boxes are drawn, and writes `seam_frames/<scene>.jpg` plus a
  `Seam Carve Shrink Percent:` line in `report.md`.  Frames are downscaled to a
  768px working width by default (~13x faster than 1080p native;
  `--seam-carve-working-width 0` carves at source resolution); default energy
  ratio is `0.25`; `0` (or `--no-seam-carve`) disables it.
* **The experiment script** (`scripts/retarget_frames.py`) tunes thresholds over
  a folder of frames: it writes carved stills at every shrink step, signal
  plots, contact sheets and a browsable HTML report.

The sections below apply to both; the CLI flag names are those of the
experiment script unless prefixed with `--seam-carve-`.

## Pipeline in one paragraph

Each frame is scaled to `working_width` (default 768 px; `None` or `0` = carve
at the source resolution).  A gradient-magnitude energy map `E = |dI/dx| +
|dI/dy|` is computed on the grayscale image.  Vertical
seams are removed one at a time; the seam is whichever connected one-pixel-per-row
path minimises **forward energy** (the energy *introduced* by the removal),
plus a large extra cost inside the dilated protected mask (face/head/person
boxes).  Every removal is logged as
a `SeamStep`, and the log is turned into four "guard" signals plus a composite.
A guard crossing its threshold means "stop".  The recommended shrink is where the
composite first trips, capped at `max_shrink`.  Carved images are sampled at
`sample_step` intervals and at the exact recommendation.

All signals are stored in `summary.csv` / `summary.json`, and plotted per frame as
`plots/<name>.svg`.  The plot's dashed line is the trip threshold (`1.0`) and the
orange vertical line is the recommended stop.

## The four guards, precisely

Let `E0` be the original working-resolution energy map, `mean(E0)` its mean, and
`seam_k` the pixels removed by the `k`-th seam.

| Guard | Normalised signal (trips at 1.0) | Meaning |
| --- | --- | --- |
| `energy` | `removed_norm(k) / energy_threshold` | Is the removed seam now as detailed as the image average? |
| `forward` | `added_cum_norm(k) / (forward_knee_factor · forward_reference)` | Is removal introducing structure-bending energy? |
| `detail` | `max(cum_detail(k)/detail_budget, cum_protected(k)/subject_budget)` | Have we eaten too many important pixels? |
| `ssim` | `(1 − ssim(k)) / (1 − ssim_floor)` | Has global structure drifted from the original? |

with:

* `removed_norm(k)` = moving average (window `smoothing_window · width`) of the
  removed pixels' mean `E`, divided by `mean(E0)`.  **This is in units of "image
  mean energy"** — `0.5` means the removed pixels are half as detailed as an
  average pixel; `1.0` means the seam is cutting through average content.
* `cum_detail(k)` = cumulative removed pixels that were in the original top
  `(1 − detail_quantile)` of energy, divided by how many such pixels exist.
* `cum_protected(k)` = cumulative removed pixels inside the protected mask
  (face/head/person boxes), divided by the total mask area.
* `ssim(k)` = structural similarity between the carved frame and the original
  uniformly scaled to the same size.
* `badness(k) = max` of the normalised signals of the guards in `enabled_methods`;
  the composite limit is where `badness ≥ badness_threshold`.

## Energy-guard parameters (the fiddly ones)

The energy guard is the primary stop.  Two separate behaviours fight here:

1. **A fixed threshold** (`energy_ratio`) works well for typical frames, but a
   frame that is detailed *everywhere* (a full-frame face, heavy cross-hatching)
   starts with no low-energy seams, so a fixed low threshold trips instantly.
2. **An adaptive term** raises the threshold for exactly those frames so they are
   not stopped at 0%.

The effective threshold is:

```
energy_threshold = energy_ratio
                 + energy_baseline_multiple · max(0, early − energy_reference)
```

where `early` is `removed_norm` at the end of the warm-up window (default
`energy_warmup = 0.10`, i.e. after 10% of the width has been removed).

* **`--energy-ratio`** (default `0.35` in the experiment library/script, **`0.25`
  in the `anime2manga` pipeline** — the user-tuned value) — the base threshold.
  **Lower = stricter = stops earlier = less distortion.**  `1.0` would mean "keep
  going until seams are as detailed as the average pixel".  This is the main knob.
* **`--energy-baseline-multiple`** (default `1.4`) — how aggressively the adaptive
  term rises above `energy_reference`.  `0` disables adaptivity entirely, so the
  threshold is exactly `energy_ratio`; raise it for busy frames that deserve more
  room.  It only has an effect when `early > energy_reference`.
* **`--energy-reference`** (default `0.25`) — the `early` value above which the
  adaptive margin starts.  Raise it to make adaptivity apply to more frames;
  lower it to apply to fewer.  Think of it as "the easy-seam energy a normal
  frame is expected to have".
* **`--energy-warmup`** (default `0.10`; config `energy_warmup`) — how far into
  the carve `early` is measured.  Too small and `early` is noisy; too large and
  the "easy" baseline already includes hard seams.

> **Why your `--energy-ratio 0.2` seemed to do nothing.**  The previous formula was
> `max(energy_ratio, multiple · early)`.  For every frame whose `early` was above
> `0.35 / 1.4 = 0.25`, the second term won and `energy_ratio` was ignored, so
> lowering it changed nothing.  The formula is now **additive in the excess above
> `energy_reference`**, which is exactly equivalent at the defaults
> (`0.35 + 1.4·max(0, early − 0.25) == max(0.35, 1.4·early)` for all `early`) but
> makes `energy_ratio` authoritative at any value.  You can confirm which term is
> governing per frame via the **`E thr`** column of `summary.csv` / `report.md`:
> if `E thr > energy_ratio`, the adaptive term is raising it.

## Detail-budget parameters

* **`--detail-budget`** (default `0.10`) — fraction of the original high-gradient
  pixels the guard may remove.  **Lower = stricter.**  This is what protects
  text-heavy credit frames and line art, where "energy" stays moderate but losing
  detail is very visible.  When this guard binds, the `E thr` column is irrelevant.
* **`--subject-budget`** (default `0.02`) — fraction of protected-mask pixels (faces,
  heads and persons share one mask) that may be removed.  Subjects are
  energetically protected in the engine, so this is a safety
  net; it rarely binds, but it catches cases where protection is overrun.
* **`--detail-quantile`** (config only, default `0.8`) — the energy quantile above
  which a pixel counts as "detail" (top 20%).  Lower the quantile to count more
  pixels as detail and make the budget bite sooner.

## Forward-energy guard (optional)

Forward energy measures the energy the removal *introduces* by pulling separated
pixels together; a rising curve means visible bending/ghosting.

* **`--forward-knee-factor`** (default `2.5`) — stop when the cumulative added
  energy per pixel exceeds this multiple of its warm-up baseline.  **Lower =
  stricter.**  It is off by default because it is noisy and usually less
  discriminating than energy/detail; enable with `--enabled energy,detail,forward`.
* **`--warmup`** (config `warmup`, default `0.05`) — the early window used for the
  forward baseline.

## Structural guard (optional)

* **`--ssim-floor`** (default `0.5`) — stop when SSIM against the uniformly scaled
  original falls to this value.  SSIM's absolute value depends on how busy the
  frame is, so it is informational and off by default.  **Raise** for stricter
  structure preservation on frames with large smooth regions.

## Selecting and combining guards

* **`--enabled`** (default `energy,detail`) — comma-separated guards that feed the
  composite.  Only these can stop a frame.
* **`--primary-method`** (default `composite`) — which limit to report as the
  headline recommendation: `composite`, or a single guard (`energy`, `forward`,
  `detail`, `ssim`) if you have decided one works best for your taste.

## Geometry and engine parameters

* **`--max-shrink`** (default `0.5`) — hard cap on the fraction of width removed.
* **`--working-width`** (default `768` on both the experiment script and the
  pipeline flag `--seam-carve-working-width`; `0` = native) — downscale the frame
  to this width before carving.  The frame is carved **and saved** at the
  resulting resolution, so this also sets the resolution of the emitted seams.
  Both entry points downscale to `768px` by default (a 1920px source becomes
  `768px` wide, then narrows by the shrink), which is about **13x faster** than
  carving 1080p native.  Pass `0` to carve at the source resolution (a 1920px
  source stays `1920px` wide).  Higher = finer judgement and
  nicer output, roughly quadratic cost; lower = fast preview.  The image and the
  face boxes are scaled to this width.  The pipeline reports the carved size as
  `Seam Carved Frame Size:` so the two images in the report never silently differ.
* **`--sample-step`** (default `0.1`) — carve a saved snapshot every this fraction
  of width (plus the exact recommendation).  Purely for inspection; it does not
  change the metric.
* **`--no-subject-protection`** — disable the energetic protected mask (the metric
  still counts protected pixels removed, so you can see the difference).
* **`--keep-face-boxes`** — do not strip the thin green face boxes the pipeline
  draws onto frames (by default they are inpainted out, since they are annotation,
  not content; genuinely green scenes are left alone).
  **Caveat:** only the *green* (face) overlay is stripped. If you run the
  experiment directly on pipeline output with `--boxes` on, the blue head and red
  person boxes are **not** removed and will be treated as content; export clean
  frames (e.g. `--no-boxes`) before running the experiment.
* **`--clean`** — delete previously generated `carved/contacts/plots/thumbs/overview`
  and summary files before running.  Without it, per-frame outputs are still
  overwritten and stale `_recNN` files for re-processed frames are removed.
* **`-j/--jobs`** — worker processes; frames are independent.
* **`--limit N` / `--sample N` / `--names a.jpg b.jpg`** — process a subset.  This
  is the recommended way to iterate quickly on a few representative frames before
  a full run.

## Config-only parameters (not exposed on the CLI)

* `smoothing_window` (`0.02`) — moving-average window for `removed_norm`, as a
  fraction of width.  Larger = smoother signals; the stop ratio moves slightly.
* `ssim_stride` (`4`) — compute SSIM every N seams and interpolate.
* `protect_subjects`, `subject_energy_factor` (`50.0`), `subject_dilation`
  (`0.01`) — strength/size of the energy penalty inside the protected mask
  (faces, heads and persons alike).
* `strip_overlays` (`true`) — green-box removal.
* `panel_width` (`320`), `thumb_width` (`240`), `overview_width` (`300`),
  `jpeg_quality` (`88`) — report image sizes; cosmetic.  Thumbnails always keep
  the carved frame's true aspect ratio, and overview cells are uniform with each
  still pillar-boxed at its real shrunk width, so the empty space shows how much
  was removed.

## Reading the outputs

`summary.csv` one row per frame.  The four "guard" columns are **stop ratios**
(fraction of width removed, 0–1), each computed independently, *not* signal
values; the guard columns are always reported even when that guard is not
enabled, so you can compare candidates.

| Column | Meaning |
| --- | --- |
| `name` | Frame name (e.g. `scene_0120`), i.e. the source still. |
| `faces` | Subjects detected in the frame (faces, heads and persons all merge into one protective mask).  Drives the energetic mask and the `subject` budget. `0` means none. |
| `body_percent` | Percent of the frame covered by the union of **person (body)** boxes.  Observational only - it never feeds the carve.  `""` (blank) when the caller passed pre-merged `boxes` so categories are unknown.  Most reliable for single-subject frames. |
| `head_percent` | Percent of the frame covered by the union of **head** boxes.  Same caveats as `body_percent`. |
| `overlap_percent` | Percent of the frame covered by both a body and a head box (intersection of the two unions). |
| `heads_in_body` | Whether every detected head lies fully inside the body region; blank when unknown, `false` when there are heads but no body. |
| `body_lean` / `head_lean` | `left`/`middle`/`right` horizontal third the body / head region sits in; blank when that category was not detected. |
| `working_size` | `[width, height]` the frame was analysed/carved at (source size when `working_width` is `None`, otherwise after scaling to `working_width`). |
| `base_energy` | Mean gradient energy of the working image (arbitrary units); the denominator for `energy`. |
| `energy_early` | The frame's own easy-seam energy at the end of the warm-up window, in units of `base_energy`.  High = detailed everywhere.  Feeds the adaptive margin. |
| `energy_threshold` (`E thr`) | The effective threshold the **energy** guard actually used, in units of `base_energy`: `energy_ratio + multiple·max(0, early − reference)`.  When `> energy_ratio`, the adaptive term governs that frame. |
| `recommended` | The headline stop, equal to the `primary_method` limit (`composite` by default), i.e. `COMP`. |
| `recommended_triggered` | `false` when the guard never tripped and the fraction is just the cap/end. |
| `energy_ratio` | Stop ratio where the **energy** guard trips: as low-energy seams run out, removed-seam energy reaches `E thr`. |
| `forward_ratio` | Stop ratio where the **forward** guard trips: introduced (structure-bending) energy reaches `forward_knee_factor ×` its warm-up baseline.  Informational unless `forward` is enabled. |
| `detail_ratio` | Stop ratio where the **detail** guard trips: cumulative removed high-gradient pixels reach `detail_budget`, or removed protected pixels reach `subject_budget`. |
| `ssim_ratio` | Stop ratio where the **structural** guard trips: SSIM to the uniformly rescaled original falls to `ssim_floor`.  Informational unless `ssim` is enabled. |
| `composite_ratio` (`COMP`) | The recommended stop: the first *enabled* guard to trip = the minimum of the enabled guard ratios (`energy` and `detail` by default).  It is the number to act on. |

In the HTML report the first table lists frames in numeric scene order; the
"Stop preview" column is the frame carved to `COMP`.

### Why `Energy` and `COMP` are usually the same

This is expected: `COMP = min(enabled guards)`, and `energy` is enabled by
default, so `COMP == energy` whenever the energy guard trips first.  On the
bundled Death Note set that is ~90% of frames (`energy` binds on 217, `detail`
on 27, and on 17 the two coincide).  Two effects drive it:

1. **The energy guard is the tightest default.**  Its adaptive threshold is
   generally reached after removing a modest number of seams, while the detail
   budget tolerates removing 10% of all high-gradient pixels — on energy-bound
   frames the detail guard would have allowed on average ~13 percentage points
   *more* shrink.
2. **The two signals correlate.**  Both rise as seams get harder; they are not
   independent evidence.

`COMP` diverges from `Energy` exactly when `detail_ratio < energy_ratio` — 27
frames here, generally busy/text/illustration frames where detail pixels are
dense (e.g. `scene_0275` energy never trips, detail stops it at 10%;
`scene_0150` detail stops at 13% while `E thr` is 0.92).  If you want `COMP` to
track another guard instead, set `--primary-method energy` (or `detail`), or
enable more guards and the minimum naturally changes.


## Recipes

* **"Defaults distort too much; shrink less."** Lower `--energy-ratio` (e.g.
  `0.25`) and/or `--detail-budget` (e.g. `0.07`).  Check `E thr` to confirm the
  adaptive term is not masking your change.
* **"A few busy frames are stopped far too early."** Raise
  `--energy-reference` so the adaptive margin reaches them, or raise
  `--energy-baseline-multiple`.
* **"Text/credit frames must not be touched."** Lower `--detail-budget`
  (`0.03–0.05`) and keep `energy` enabled.
* **"Faces/heads/persons get squished."** Lower `--max-shrink`, lower
  `--energy-ratio`, or add `ssim` / lower `--ssim-floor`.
* **"I want one guard only."** `--primary-method energy --enabled energy` (etc.),
  then compare against `composite`.
