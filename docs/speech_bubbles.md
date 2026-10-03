# Speech bubbles (step 10)

Lettering turns a finished panel into a manga panel: the subtitles that land in
a scene become balloons drawn over the artwork.

## The ordering contract

A bubble is planned **after** the panel is finished:

1. Step 6 picks a frame; step 8 seam-carves it; step 9 crops/carves it again to
   fit its row and places it on the page.
2. Only then does step 10 letter it, in **final panel pixel coordinates**.

This matters because cropping and seam carving move the pixels.  Lettering a
source frame and then cropping would drag text off the speaker.  `plan_bubbles`
therefore takes the *final* panel size and the subject boxes mapped into that
same space, never a source frame.

`map_boxes_to_panel` converts a source-frame box into panel pixels.  The boxes
are detected on the **full-resolution source frame** (e.g. 1920x1080) while the
panel was rendered from a downscaled, cropped frame, so the helper works in
**width/height fractions**: it normalises by the source size, shifts into the
kept crop window and rescales to the panel.  Fractions are scale-free, which
also makes the result **carve-invariant** - horizontal seam carving only removes
columns, so a box's fractional x-span is unchanged and no seam-carve term is
needed.  It takes the crop values the renderer actually used
(`PanelPlan.crop_frac` / `crop_x_frac`), not the step-8b recommendation.

## Two layers

`anime2manga.speech_bubbles` is split so the placement logic can grow without
touching pixels:

| Layer | What it does |
| --- | --- |
| `plan_bubbles` | **Pure planner.** Panel size + subtitle texts + panel-space boxes + audio focus → `BubbleSpec` list. No I/O, deterministic, testable like `layout.py`. |
| `build_svg` | Turns specs into an SVG document (balloons + `<text>`), built with `svgwrite`. |
| `render_overlay_png` | Rasterises the SVG to a transparent PNG (cairosvg). |
| `write_panel_overlay` | Writes `<scene>.svg` and `<scene>.png` beside the panel. |
| `flatten_panel` | Optional export: bakes the overlay into the panel JPEG. |

The **`.svg` and `.png` are both emitted per subtitled panel**.  The pipeline's
`layout/index.html` stacks the transparent PNG over the panel, so the panel image
stays clean while the bubble sits on top.  The SVG keeps the text live and
selectable.

## Shapes

Every balloon is currently a **rounded rectangle**.  Earlier drafts picked a
shape per line from its punctuation (ellipse default, cloud for `...`, spiky for
shouts), but ellipse/cloud/spiky balloons waste far more panel area than the text
needs and the heuristic did not earn that cost.  `BubbleShape`, `classify_shape`,
and the ellipse/cloud/spiky path builders are kept so a later experiment can
reintroduce per-line styles (the idea on record: a fat **white halo** around the
text plus a thin black outline, which looks bubble-ish without the footprint).
White fill, `#111` outline sized from the font (`STROKE_FRAC`), centred text.

**No tails.**  Balloon tails were drawn for a while and looked wrong far more
often than not (they attached at odd angles and crossed the art), so they were
removed.  Their intended target and a curvature model are worth revisiting only
once placement can actually decide which subject a balloon belongs to.

## Metrics and the one gotcha

Wrapping and rendering must agree **exactly**, or text spills out of its
balloon.  Pillow and cairo disagree on text width (different bearings, and
`font-size` unit handling), so measurement is done with **cairo itself**:
`measure_line` renders one line to an alpha PNG and returns its ink bounding
box; results are cached.  Line height is the measured ink height times
`LINE_SPACING`, used identically by the planner (box sizing) and `build_svg`
(text placement).  Do not swap this for Pillow `getlength`.

The measurement canvas must be wide enough that a line never clips, or the ink
box under-reports and the font-shrink loop overshoots to the 16px floor.

## Budget

A single balloon may cover `MAX_BUBBLE_AREA` (42%) of the panel; all balloons
together `TEXT_BUDGET` (50%, the step-10 plan).  The planner shrinks the font
(2pt at a time, floor 16pt) until a spec fits, and drops a line that never fits,
recording a note.

## What is deliberately not done yet

Placement is naive and labelled as such:

* bubbles alternate left/right in reading order and stack without overlapping;
* a balloon is nudged above/below a subject only if its centre lands on one.

**Bubble-placement optimisation is the next feature.**  It belongs in
`plan_bubbles`/`_place_bubble` and may use `audio_focus`, the head/person boxes,
composition leans and reading order.  The SVG/raster layer does not need to
change.

## CLI

Lettering rides with the layout step (both on by default):

```bash
# emit *.svg + *.png overlays and overlay them in layout/index.html
anime2manga input.mkv

# also bake the bubbles into the panel JPEG
anime2manga input.mkv --flatten-bubbles

# no bubbles
anime2manga input.mkv --no-speech-bubbles
```

Set `ANIME2MANGA_BUBBLE_FONT` to a TTF path to override the font search (Comic
Neue / Comic Sans MS are preferred, then DejaVu Sans).
