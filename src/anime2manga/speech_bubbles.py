"""Step 10 - speech-bubble geometry, SVG generation and rasterisation.

A panel is finished before it is lettered
-----------------------------------------
A bubble is only ever planned **after** the panel image is final: the frame has
already been seam-carved and/or cropped by :mod:`anime2manga.layout_report`, and
its position in the page is known.  This module therefore works in *final panel
pixel coordinates* - it never sees a source frame and never has to reason about
a crop that might move underneath it.  That ordering is a hard contract, not an
implementation detail: lettering a frame before cropping would leave the text
covering the wrong pixels.

Two layers, one source of truth
-------------------------------
The lettering is deliberately split so that the *placement* logic (the next
feature) can be developed and tested without touching any pixels:

* :func:`plan_bubbles` is the **pure planner**.  It takes the final panel size,
  the scene's subtitle texts, the detected subject boxes *in panel coordinates*
  and the scene's audio focus, and returns a list of :class:`BubbleSpec`
  geometries (anchor box, shape, font size, wrapped lines).  It is deterministic,
  side-effect free and easy to score - the same shape as
  :mod:`anime2manga.layout`.
* :func:`build_svg` / :func:`render_overlay_png` are the **renderer**.  They turn
  a :class:`BubbleSpec` into an SVG document (built with ``svgwrite``) and (via
  ``cairosvg``) into an RGBA PNG overlay drawn over the panel.

Three outputs from the same specs
---------------------------------
:func:`write_panel_overlay` writes an interactive ``.svg`` beside the panel and
:func:`render_overlay_png` rasterises it to a transparent ``.png``.  The two
assets let a viewer either keep the vector text (selectable, re-styleable) or
composite the raster.  :func:`flatten_panel` is the optional export path that
bakes the overlay into the panel JPEG.

Manga lettering conventions
---------------------------
White balloon, a heavy black outline and centred text.  For now every balloon is
a **rounded rectangle** (:func:`classify_shape` always returns
``BubbleShape.ROUNDED``) and there are **no tails** - earlier ellipse/cloud/spiky
shapes and drawn tails wasted panel area and looked wrong.  The shape enum and
its path builders are kept so a later experiment (a white halo plus a thin
outline around the text) can reintroduce per-line styles.  Full method and
parameter reference: ``docs/speech_bubbles.md``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .models import DetectionBox

# --- defaults ----------------------------------------------------------------

#: Font candidates, tried in order.  A comic-style face is preferred when the
#: host has one; ``ANIME2MANGA_BUBBLE_FONT`` overrides the list.
FONT_CANDIDATES: tuple[str, ...] = (
    "Comic Neue",
    "Comic Sans MS",
    "DejaVu Sans",
    "Liberation Sans",
)
#: Point size of the bubble text (SVG user units = final panel pixels).  44 was
#: the first cut; 30 keeps short lines from ballooning (a 3-word line at 44pt
#: filled a whole panel).  This is the *base* before the per-bubble shrink loop.
FONT_SIZE = 30
#: Balloon fill/stroke.
FILL = "#ffffff"
STROKE = "#111111"
TEXT_FILL = "#111111"
#: Padding between the wrapped text and the balloon edge, and the line height.
PADDING = 18
LINE_SPACING = 1.18
#: Outline width, as a fraction of the font size (min 2px).
STROKE_FRAC = 0.09
#: Fraction of the panel a *single* balloon may occupy before the font shrinks.
MAX_BUBBLE_AREA = 0.42
#: Fraction of the panel all balloons together may occupy (the 50% text budget
#: from the step-10 plan, with headroom for tails and outlines).
TEXT_BUDGET = 0.50
#: Smallest font the planner will shrink to before dropping a line.
MIN_FONT = 16.0


class BubbleShape(StrEnum):
    """Balloon outline styles, chosen per subtitle line."""

    ELLIPSE = "ellipse"
    ROUNDED = "rounded"
    CLOUD = "cloud"
    SPIKY = "spiky"


@dataclass(frozen=True)
class BubbleSpec:
    """One balloon, in final panel pixel coordinates.

    ``lines`` are already wrapped.  ``box`` is the *text box* the balloon must
    contain (grown by :data:`PADDING` around the text); ``shape`` decides how
    that box is outlined.
    """

    text: str
    lines: tuple[str, ...]
    shape: BubbleShape
    #: Text box that the balloon must contain, in panel pixels.
    box: tuple[float, float, float, float]  # x, y, width, height
    font_size: float

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2.0, y + h / 2.0


@dataclass(frozen=True)
class BubbleLayout:
    """The result of planning one panel's dialogue."""

    panel_size: tuple[int, int]
    specs: tuple[BubbleSpec, ...]
    font_path: str | None = None
    font_size: float = FONT_SIZE
    #: Set when the planner had to shrink text, cap or drop lines.
    notes: tuple[str, ...] = ()

    @property
    def has_bubbles(self) -> bool:
        return bool(self.specs)

    @property
    def area_frac(self) -> float:
        """Total balloon area as a fraction of the panel."""
        pw, ph = self.panel_size
        total = float(pw * ph) or 1.0
        return sum(_outline_area(s) for s in self.specs) / total


# --- text classification -----------------------------------------------------


def classify_shape(text: str) -> BubbleShape:
    """Pick a balloon style from a subtitle line.

    For now **every** line gets a rounded rectangle: ellipse/cloud/spiky
    balloons waste far more panel area than they need, and the earlier
    punctuation heuristic (shout -> burst, ``...`` -> cloud) was not earning that
    cost.  The shape enum and this function are kept so a later experiment (a
    white halo + thin outline around the text, or a shape chosen from the audio)
    can reintroduce per-line styles without touching the renderer.
    """
    _ = text
    return BubbleShape.ROUNDED


# --- font handling -----------------------------------------------------------


def _cairo_font_path(name: str) -> str | None:
    try:
        from cairosvg import helpers
    except Exception:
        return None
    find_font = getattr(helpers, "find_font", None)
    if find_font is None:
        return None
    for weight in ("bold", "normal"):
        try:
            path = find_font(name, weight)
        except Exception:
            path = None
        if path:
            return str(path)
    return None


def find_font(preferred: str | None = None) -> str | None:
    """Return a usable TTF/OTF path for the bubble text (``None`` -> SVG default)."""
    override = os.environ.get("ANIME2MANGA_BUBBLE_FONT")
    names = [override] if override else [preferred] if preferred else list(FONT_CANDIDATES)
    for name in names:
        if not name:
            continue
        path = _cairo_font_path(name)
        if path:
            return path
    for root in ("/usr/share/fonts", "/usr/local/share/fonts", str(Path.home() / ".fonts")):
        for name in names:
            if not name:
                continue
            stem = name.replace(" ", "")
            for pattern in (f"*{stem}*.ttf", f"*{stem}*.otf"):
                for candidate in sorted(Path(root).rglob(pattern)):
                    return str(candidate)
    return None


def load_font(font_path: str | None, size: float):
    """Load a Pillow font for the wrap *guess*, falling back to the default.

    Widths are still verified with cairo before a bubble is finalised, so this
    only needs to be close enough to choose sensible line breaks.
    """
    from PIL import ImageFont

    if font_path:
        try:
            return ImageFont.truetype(font_path, max(1, round(size)))
        except OSError:
            pass
    return ImageFont.load_default()


def wrap_text(text: str, font, max_width: float) -> list[str]:
    """Greedy word wrap of ``text`` to ``max_width`` (an in-memory Pillow font)."""
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if not current or font.getlength(trial) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


#: Cache of measured line ink widths, keyed by (family, size, text).
_MEASURE_CACHE: dict[tuple[str, float, str], tuple[float, float]] = {}


def measure_line(text: str, font_path: str | None, size: float) -> tuple[float, float]:
    """Measure a line's rendered ink size ``(width, height)`` with cairo.

    Wrapping and rendering must agree pixel-for-pixel, so the *renderer's own*
    engine measures the text.  A line is drawn once in isolation and its alpha
    bounding box is returned; results are cached because the same probe runs for
    every shrink step.
    """
    family = _font_family(font_path)
    key = (family, round(size, 2), text)
    cached = _MEASURE_CACHE.get(key)
    if cached is not None:
        return cached
    import cairosvg
    import cv2
    import numpy as np

    pad = 40.0
    # Generous canvas so the line never clips (clipping would under-measure and
    # make the shrink loop overshoot to a tiny font).
    est = max(1.4 * size * len(text), size * 6.0)
    width = min(max(200.0, est + 2 * pad), 20000.0)
    height = size * 3.0
    probe = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}">'
        f'<text x="{pad:.0f}" y="{height - pad:.0f}" font-family="{family}" '
        f'font-weight="700" font-size="{size:.2f}">{_svg_escape(text)}</text></svg>'
    )
    png = cairosvg.svg2png(bytestring=probe.encode("utf-8"))
    assert isinstance(png, bytes)
    img = cv2.imdecode(np.frombuffer(bytearray(png), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] < 4:
        result = (size * 0.55 * len(text), size)
    else:
        alpha = img[:, :, 3]
        cols = np.where(alpha.max(axis=0) > 0)[0]
        rows = np.where(alpha.max(axis=1) > 0)[0]
        if len(cols) == 0 or len(rows) == 0:
            result = (0.0, size)
        else:
            result = (float(cols[-1] - cols[0] + 1), float(rows[-1] - rows[0] + 1))
    _MEASURE_CACHE[key] = result
    return result


def _text_size(lines: list[str], font_path: str | None, size: float) -> tuple[float, float]:
    """Rendered width and stacked height of ``lines`` (cairo-measured)."""
    widths = [measure_line(line, font_path, size)[0] for line in lines]
    line_h = _line_height(tuple(lines), font_path, size)
    return max(widths, default=0.0), line_h * len(lines)


def _font_family(font_path: str | None) -> str:
    """The family name cairo should ask fontconfig for."""
    if font_path:
        return Path(font_path).stem
    return FONT_CANDIDATES[0]


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# --- geometry ----------------------------------------------------------------


def _outline_size(spec: BubbleSpec) -> tuple[float, float]:
    """Full balloon width/height for a spec's text box (shape-dependent)."""
    _x, _y, w, h = spec.box
    if spec.shape is BubbleShape.ELLIPSE:
        # An ellipse inscribing a text box needs its axes scaled by sqrt(2).
        return w * math.sqrt(2.0), h * math.sqrt(2.0)
    if spec.shape is BubbleShape.CLOUD:
        return w * 1.12, h * 1.22
    if spec.shape is BubbleShape.SPIKY:
        return w * 1.18, h * 1.28
    return w, h


def _outline_area(spec: BubbleSpec) -> float:
    w, h = _outline_size(spec)
    if spec.shape is BubbleShape.ELLIPSE:
        return math.pi * (w / 2.0) * (h / 2.0)
    return w * h


def _subject_boxes(
    boxes: list[DetectionBox], size: tuple[int, int]
) -> list[tuple[float, float, float, float]]:
    pw, ph = size
    out: list[tuple[float, float, float, float]] = []
    for box in boxes:
        if box.width <= 0 or box.height <= 0:
            continue
        # A near full-frame box would veto every zone; ignore those.
        if box.width > 0.94 * pw and box.height > 0.94 * ph:
            continue
        out.append((float(box.x), float(box.y), float(box.width), float(box.height)))
    return out


def _place_bubble(
    index: int,
    count: int,
    bw: float,
    bh: float,
    out_w: float,
    out_h: float,
    audio: str,
    subjects: list[tuple[float, float, float, float]],
    occupied: list[tuple[float, float, float, float]],
) -> tuple[float, float]:
    """Stack a balloon in reading order without overlapping its neighbours.

    This is deliberately **not** the placement optimiser.  It does the minimum a
    first cut needs: bubbles alternate left/right down the panel, each is nudged
    off a subject, and no two balloons overlap (``occupied`` tracks what is
    already placed).  The next feature replaces the zone choice here with a
    metadata-scored search, but it can keep this stacking as the fallback.
    """
    if count == 1:
        side = audio if audio in ("left", "right") else "left"
    else:
        side = "left" if index % 2 == 0 else "right"
    margin_x = 0.045 * out_w
    margin_y = 0.045 * out_h
    gap = 0.02 * out_h
    x = margin_x if side == "left" else out_w - margin_x - bw

    # Walk down the chosen column until the balloon fits without overlapping.
    y = margin_y
    for _ in range(40):
        x_clamped = max(0.0, min(x, out_w - bw))
        nx, ny = _nudge_off_subjects(x_clamped, y, bw, bh, subjects, out_w, out_h)
        if not _overlaps(nx, ny, bw, bh, occupied):
            return nx, ny
        # Advance below the lowest obstacle this balloon collides with.
        bottoms = [
            oy + oh
            for (ox, oy, ow, oh) in occupied
            if nx < ox + ow and ox < nx + bw and ny < oy + oh and oy < ny + bh
        ]
        if not bottoms:
            break
        y = max(bottoms) + gap
        if y + bh > out_h:
            break
    # No free slot down this column: fall back to a clamped position.
    return (
        max(0.0, min(x, out_w - bw)),
        max(0.0, min(y, out_h - bh)),
    )


def _overlaps(
    x: float,
    y: float,
    bw: float,
    bh: float,
    occupied: list[tuple[float, float, float, float]],
    pad: float = 4.0,
) -> bool:
    return any(
        x < ox + ow + pad and ox < x + bw + pad and y < oy + oh + pad and oy < y + bh + pad
        for (ox, oy, ow, oh) in occupied
    )


def _nudge_off_subjects(
    x: float,
    y: float,
    bw: float,
    bh: float,
    subjects: list[tuple[float, float, float, float]],
    out_w: float,
    out_h: float,
) -> tuple[float, float]:
    """If the balloon sits over a subject, move it above (or below) that subject.

    A cheap first pass so even the unoptimised placement does not sit on a face.
    Conservative: only a clear overlap triggers a move.
    """
    cx, cy = x + bw / 2.0, y + bh / 2.0
    for sx, sy, sw, sh in subjects:
        if sx <= cx <= sx + sw and sy <= cy <= sy + sh:
            if sy - bh - 4.0 >= 0.0:
                y = sy - bh - 4.0
            elif sy + sh + 4.0 + bh <= out_h:
                y = sy + sh + 4.0
            break
    return max(0.0, min(x, out_w - bw)), max(0.0, min(y, out_h - bh))


def plan_bubbles(
    panel_size: tuple[int, int],
    subtitles: list[str],
    *,
    boxes: list[DetectionBox] | None = None,
    audio_focus: str = "center",
    font_path: str | None = None,
    font_size: float = FONT_SIZE,
    max_bubbles: int = 8,
    max_area: float = MAX_BUBBLE_AREA,
    text_budget: float = TEXT_BUDGET,
) -> BubbleLayout:
    """Plan the balloons for one finished panel (pure, no pixels touched).

    Parameters
    ----------
    panel_size:
        ``(width, height)`` of the **final** (carved/cropped) panel image.
    subtitles:
        The scene's subtitle texts, in reading order.
    boxes:
        Subject boxes in **panel coordinates** (see :func:`map_boxes_to_panel`);
        used only to steer balloons off a face for now.  ``None`` means no
        subject information.
    audio_focus:
        ``"left"``/``"center"``/``"right"`` - the dialogue direction hint.
    font_path / font_size:
        Bubble font (a TTF path) and its point size.

    The ``max_bubbles`` cap is generous: how many bubbles a panel *should* get
    (merging short lines, splitting a long one) is the next feature, and this
    function is where that decision will live.  Lines beyond the cap are dropped
    with a note.
    """
    out_w, out_h = float(panel_size[0]), float(panel_size[1])
    # Resolve the font once so measurement, wrapping and rendering all agree on
    # the same face (``font_path=None`` means "find one", not "use the default
    # family", which may not be installed).
    if font_path is None:
        font_path = find_font()
    texts = [t.strip() for t in subtitles if t and t.strip()]
    notes: list[str] = []
    if len(texts) > max_bubbles:
        notes.append(f"capped at {max_bubbles} bubbles (from {len(texts)})")
        texts = texts[:max_bubbles]
    if not texts or out_w <= 0 or out_h <= 0:
        return BubbleLayout(panel_size=panel_size, specs=(), font_path=font_path,
                            font_size=font_size, notes=tuple(notes))

    subjects = _subject_boxes(boxes or [], panel_size)
    budget = text_budget * out_w * out_h
    specs: list[BubbleSpec] = []
    occupied: list[tuple[float, float, float, float]] = []
    used = 0.0

    for index, text in enumerate(texts):
        shape = classify_shape(text)
        # Long lines get a smaller base size so a multi-line panel stays legible
        # without any one balloon dominating it.
        base = font_size * (0.85 if len(text) > 90 else 1.0)
        size = base
        chosen: tuple[list[str], float, float, float, float] | None = None
        while size >= MIN_FONT:
            font = load_font(font_path, size)
            max_line_w = out_w * (0.64 if len(texts) == 1 else 0.46)
            lines = wrap_text(text, font, max_line_w)
            tw, th = _text_size(lines, font_path, size)
            bw = tw + 2 * PADDING
            bh = th + 2 * PADDING
            probe = BubbleSpec(text=text, lines=tuple(lines), shape=shape,
                               box=(0.0, 0.0, bw, bh), font_size=size)
            ow, oh = _outline_size(probe)
            area = _outline_area(probe)
            fits = (
                ow <= out_w * 0.94
                and oh <= out_h * 0.92
                and area <= max_area * out_w * out_h
                and used + area <= budget
            )
            if fits:
                chosen = (lines, bw, bh, ow, oh)
                break
            size -= 2.0
        if chosen is None:
            notes.append("dropped an over-long line (does not fit the panel)")
            continue
        lines, bw, bh, ow, oh = chosen
        # Place using the balloon outline, then convert the outline corner back
        # to the text-box corner the spec stores.
        pad_x, pad_y = (ow - bw) / 2.0, (oh - bh) / 2.0
        ox, oy = _place_bubble(
            index, len(texts), ow, oh, out_w, out_h, audio_focus, subjects, occupied
        )
        occupied.append((ox, oy, ow, oh))
        x, y = ox + pad_x, oy + pad_y
        spec = BubbleSpec(text=text, lines=tuple(lines), shape=shape,
                          box=(x, y, bw, bh), font_size=size)
        specs.append(spec)
        used += _outline_area(spec)

    return BubbleLayout(panel_size=panel_size, specs=tuple(specs), font_path=font_path,
                        font_size=font_size, notes=tuple(notes))


def map_boxes_to_panel(
    boxes: list[DetectionBox],
    panel_size: tuple[int, int],
    source_size: tuple[int, int],
    crop_frac: float,
    crop_x_frac: float,
) -> list[DetectionBox]:
    """Map source-frame boxes into final panel coordinates.

    The pipeline detects boxes on the **full-resolution source frame** (e.g.
    1920x1080), while the panel was rendered from a downscaled frame that was
    then cropped.  So this works in **width/height fractions**, not pixels: the
    box is normalised by ``source_size``, shifted into the kept crop window and
    rescaled to the panel.  Fractions are scale-free, which also makes the result
    **carve-invariant** - horizontal seam carving only removes columns, so a
    box's fractional x-span is (to the uniform-compression approximation) the
    same before and after, and no ``seam_carve_shrink`` term is needed.

    ``crop_frac``/``crop_x_frac`` are the values the renderer actually used
    (``PanelPlan.crop_frac`` / ``crop_x_frac``), *not* the step-8b seam-carve
    recommendation - those differ whenever the layout reduction is smaller.
    """
    pw, ph = panel_size
    src_w, src_h = source_size
    kept = 1.0 - crop_frac
    if kept <= 1e-6 or pw <= 0 or ph <= 0 or src_w <= 0 or src_h <= 0:
        return []
    post_w = pw / kept  # panel width before the crop
    out: list[DetectionBox] = []
    for box in boxes:
        if box.width <= 0 or box.height <= 0:
            continue
        fx0 = box.x / src_w
        fx1 = (box.x + box.width) / src_w
        # Shift into the kept crop window (a fraction), then scale to panel px.
        px0 = (fx0 - crop_x_frac) * post_w
        px1 = (fx1 - crop_x_frac) * post_w
        if px1 <= 0 or px0 >= pw:
            continue  # fully cropped away
        x0 = max(0.0, min(px0, px1))
        x1 = min(float(pw), max(px0, px1))
        y0 = max(0.0, box.y / src_h * ph)
        y1 = min(float(ph), (box.y + box.height) / src_h * ph)
        if y1 <= y0:
            continue
        out.append(
            DetectionBox(
                x=round(x0),
                y=round(y0),
                width=max(1, round(x1 - x0)),
                height=max(1, round(y1 - y0)),
                confidence=box.confidence,
            )
        )
    return out


# --- SVG rendering -----------------------------------------------------------


def _ellipse_path(cx: float, cy: float, rx: float, ry: float) -> str:
    return (
        f"M {cx - rx:.2f} {cy:.2f} "
        f"a {rx:.2f} {ry:.2f} 0 1 0 {2 * rx:.2f} 0 "
        f"a {rx:.2f} {ry:.2f} 0 1 0 {-2 * rx:.2f} 0 Z"
    )


def _cloud_path(cx: float, cy: float, rx: float, ry: float) -> str:
    """A ring of overlapping arcs, the standard manga thought cloud."""
    bumps = 11
    points = []
    for i in range(bumps):
        angle = (i / bumps) * math.tau
        jitter = 1.0 + 0.10 * math.sin(i * 2.3)
        points.append((cx + math.cos(angle) * rx * jitter, cy + math.sin(angle) * ry * jitter))
    path = []
    for i, (px, py) in enumerate(points):
        qx, qy = points[(i + 1) % len(points)]
        mx, my = (px + qx) / 2.0, (py + qy) / 2.0
        r = math.hypot(qx - px, qy - py) * 0.62
        path.append(f"A {r:.2f} {r:.2f} 0 0 1 {qx:.2f} {qy:.2f}" if i else
                    f"M {px:.2f} {py:.2f} A {r:.2f} {r:.2f} 0 0 1 {qx:.2f} {qy:.2f}")
        _ = (mx, my)
    return " ".join(path) + " Z"


def _spiky_path(cx: float, cy: float, rx: float, ry: float) -> str:
    """A star-burst outline for shouting."""
    spikes = 14
    inner = 0.82
    pts = []
    for i in range(spikes * 2):
        angle = (i / (spikes * 2)) * math.tau - math.pi / 2
        fx = 1.0 if i % 2 == 0 else inner
        pts.append((cx + math.cos(angle) * rx * fx, cy + math.sin(angle) * ry * fx))
    return "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts) + " Z"


def _outline_path(spec: BubbleSpec, ow: float, oh: float) -> str:
    cx, cy = spec.center
    if spec.shape is BubbleShape.ELLIPSE:
        return _ellipse_path(cx, cy, ow / 2.0, oh / 2.0)
    if spec.shape is BubbleShape.CLOUD:
        return _cloud_path(cx, cy, ow / 2.0, oh / 2.0)
    if spec.shape is BubbleShape.SPIKY:
        return _spiky_path(cx, cy, ow / 2.0, oh / 2.0)
    r = min(ow, oh) * 0.30
    x, y = cx - ow / 2.0, cy - oh / 2.0
    return (
        f"M {x + r:.2f} {y:.2f} H {x + ow - r:.2f} A {r:.2f} {r:.2f} 0 0 1 {x + ow:.2f} {y + r:.2f} "
        f"V {y + oh - r:.2f} A {r:.2f} {r:.2f} 0 0 1 {x + ow - r:.2f} {y + oh:.2f} "
        f"H {x + r:.2f} A {r:.2f} {r:.2f} 0 0 1 {x:.2f} {y + oh - r:.2f} "
        f"V {y + r:.2f} A {r:.2f} {r:.2f} 0 0 1 {x + r:.2f} {y:.2f} Z"
    )


def build_svg(layout: BubbleLayout, *, background: bool = False) -> str:
    """Render a :class:`BubbleLayout` to a standalone SVG document.

    Built with ``svgwrite`` so the balloons are a real SVG tree rather than
    string concatenation.  ``background`` controls the canvas: ``False`` (the
    default) leaves the overlay transparent so the panel shows through; ``True``
    fills white, for a self-contained preview.
    """
    import svgwrite

    pw, ph = layout.panel_size
    family = _font_family(layout.font_path)
    drawing = svgwrite.Drawing(size=(pw, ph), profile="full")
    if background:
        drawing.add(drawing.rect(insert=(0, 0), size=(pw, ph), fill="#ffffff"))
    for spec in layout.specs:
        ow, oh = _outline_size(spec)
        stroke = max(2.0, spec.font_size * STROKE_FRAC)
        group = drawing.g()
        group.add(
            drawing.path(
                d=_outline_path(spec, ow, oh),
                fill=FILL,
                stroke=STROKE,
                stroke_width=f"{stroke:.2f}",
                stroke_linejoin="round",
            )
        )
        # Text: one <text> per line, centred.  The line height is measured with
        # the same cairo probe the planner used, so the rendered block matches
        # the box the balloon was sized for.
        line_h = _line_height(spec.lines, layout.font_path, spec.font_size)
        block_h = line_h * len(spec.lines)
        cx, cy = spec.center
        start_y = cy - block_h / 2.0 + line_h / 2.0
        for i, line in enumerate(spec.lines):
            group.add(
                drawing.text(
                    line,
                    insert=(f"{cx:.2f}", f"{start_y + i * line_h:.2f}"),
                    text_anchor="middle",
                    dominant_baseline="central",
                    font_family=family,
                    font_weight="700",
                    font_size=f"{spec.font_size:.2f}",
                    fill=TEXT_FILL,
                )
            )
        drawing.add(group)
    return drawing.tostring()


def _line_height(lines: tuple[str, ...], font_path: str | None, size: float) -> float:
    """The line height the planner used (max measured ink height * spacing)."""
    heights = [measure_line(line, font_path, size)[1] for line in lines] or [size]
    return max(heights) * LINE_SPACING


def render_overlay_png(layout: BubbleLayout, out_path: Path) -> Path:
    """Rasterise a layout's overlay to a transparent PNG next to the panel."""
    import cairosvg

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(
        bytestring=build_svg(layout).encode("utf-8"),
        write_to=str(out_path),
        output_width=layout.panel_size[0],
        output_height=layout.panel_size[1],
    )
    return out_path


def flatten_panel(panel_path: Path, layout: BubbleLayout, out_path: Path | None = None) -> Path:
    """Bake the overlay into the panel image (the export path).

    The overlay is rasterised to a temporary file and removed afterwards, so a
    flattened panel leaves no stray asset next to it.
    """
    import cv2
    import numpy as np

    tmp = panel_path.with_suffix(".overlay.png")
    render_overlay_png(layout, tmp)
    try:
        overlay = cv2.imread(str(tmp), cv2.IMREAD_UNCHANGED)
        base = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
        if overlay is None or base is None:
            raise FileNotFoundError(f"cannot flatten {panel_path}")
        if overlay.shape[2] != 4:
            # Without an alpha channel there is nothing to composite; keep the
            # baked bubble white rather than silently writing the base unchanged.
            raise ValueError(f"overlay for {panel_path} is not RGBA")
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        rgb = overlay[:, :, :3].astype(np.float32)
        base = (rgb * alpha + base.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        target = out_path or panel_path
        cv2.imwrite(str(target), base)
        return target
    finally:
        tmp.unlink(missing_ok=True)


def write_panel_overlay(layout: BubbleLayout, out_dir: Path, stem: str) -> tuple[Path, Path] | None:
    """Write ``<stem>.svg`` and ``<stem>.png`` overlay assets; return their paths.

    Returns ``None`` when the layout has no bubbles.
    """
    if not layout.has_bubbles:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{stem}.svg"
    png_path = out_dir / f"{stem}.png"
    svg_path.write_text(build_svg(layout), encoding="utf-8")
    render_overlay_png(layout, png_path)
    return svg_path, png_path


__all__ = [
    "BubbleLayout",
    "BubbleShape",
    "BubbleSpec",
    "build_svg",
    "classify_shape",
    "find_font",
    "flatten_panel",
    "map_boxes_to_panel",
    "plan_bubbles",
    "render_overlay_png",
    "wrap_text",
    "write_panel_overlay",
]
