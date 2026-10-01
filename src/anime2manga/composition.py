"""Per-frame subject composition metrics from detected boxes (step 8).

For each chosen frame we summarise how the detected *bodies* (person boxes) and
*heads* occupy the frame:

* **Body / Head Percent of Frame** - the share of the frame covered by the
  union of that category's boxes.  Using the union means overlapping detections
  are counted once and the number never exceeds 100%.
* **Body and Head Overlap Percent** - the share of the frame covered by both a
  body and a head box (the intersection of the two unions).
* **Heads Contained in Body** - whether every head box lies fully inside the
  union of body boxes (``None`` when no head was detected).
* **Body / Head Leans Towards** - which horizontal third of the frame the
  union of boxes sits in (``None`` when the category was not detected): the
  ``left``/``middle``/``right`` third containing the combined bounding-box
  centre.

The computation is pure and box-only, so it is cheap to run and trivial to
test; it never touches image pixels.

Caveat
------
These metrics are **observational**.  They are deliberately not wired into the
cropping (step 9) or the seam-carving (step 8b) decisions: a body/head fraction
is a poor saliency signal on its own (a large flat body needs far less
protection than a small detailed face) and a "lean" is a description, not a
composition rule.  They are also only well-defined for a frame with a single
body/head; with several detections the percentages cover the union of the
boxes and the lean uses their combined bounding box, so two subjects on
opposite sides read as ``middle``.  Treat them as report diagnostics.
"""

from __future__ import annotations

from itertools import pairwise

from .models import DetectionBox, FrameComposition, FrameLean

#: An axis-aligned rectangle ``(x1, y1, x2, y2)`` in frame pixels.
Rect = tuple[int, int, int, int]

#: Human-facing labels for :class:`FrameLean`.
_LEAN_LABELS = {
    FrameLean.LEFT: "Left",
    FrameLean.MIDDLE: "Middle",
    FrameLean.RIGHT: "Right",
}


def lean_label(lean: FrameLean | None) -> str:
    """Title-case a stored lean for the report (``middle`` -> ``Middle``).

    ``None`` (no body/head detected) renders as ``"None"`` so the report line is
    always present.
    """
    if lean is None:
        return "None"
    return _LEAN_LABELS[lean]


def _clip(box: DetectionBox, width: int, height: int) -> Rect | None:
    """Clip ``box`` to the frame; ``None`` when it falls entirely outside."""
    x1 = max(0, int(box.x))
    y1 = max(0, int(box.y))
    x2 = min(width, int(box.x) + int(box.width))
    y2 = min(height, int(box.y) + int(box.height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def clip_rects(boxes: list[DetectionBox], frame_size: tuple[int, int]) -> list[Rect]:
    """Clip every box to ``frame_size``, dropping degenerate/off-frame boxes."""
    width, height = frame_size
    rects: list[Rect] = []
    for box in boxes:
        rect = _clip(box, width, height)
        if rect is not None:
            rects.append(rect)
    return rects


def _union_length(intervals: list[tuple[int, int]]) -> int:
    """Total length covered by a set of 1-D intervals (overlaps merged)."""
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + (end - start)


def union_area(rects: list[Rect]) -> int:
    """Area covered by the union of axis-aligned rectangles.

    A sweep over the distinct x edges multiplies each vertical slab by the
    1-D union length of the rectangles active in it, so overlapping boxes are
    never double-counted.
    """
    if not rects:
        return 0
    edges = sorted({r[0] for r in rects} | {r[2] for r in rects})
    area = 0
    for xa, xb in pairwise(edges):
        if xb <= xa:
            continue
        active = [(r[1], r[3]) for r in rects if r[0] <= xa < r[2]]
        area += (xb - xa) * _union_length(active)
    return area


def intersection_area(a_rects: list[Rect], b_rects: list[Rect]) -> int:
    """Area shared by two unions of rectangles.

    The pairwise overlaps are unioned (a head may straddle two adjacent bodies),
    so the result is the true intersection of the two regions.
    """
    pieces: list[Rect] = []
    for ax1, ay1, ax2, ay2 in a_rects:
        for bx1, by1, bx2, by2 in b_rects:
            x1, y1 = max(ax1, bx1), max(ay1, by1)
            x2, y2 = min(ax2, bx2), min(ay2, by2)
            if x2 > x1 and y2 > y1:
                pieces.append((x1, y1, x2, y2))
    return union_area(pieces)


def lean(rects: list[Rect], frame_width: int) -> FrameLean | None:
    """Which horizontal third the rectangles' combined bounding box sits in."""
    if not rects:
        return None
    centre = (min(r[0] for r in rects) + max(r[2] for r in rects)) / 2.0
    if centre < frame_width / 3.0:
        return FrameLean.LEFT
    if centre > frame_width * 2.0 / 3.0:
        return FrameLean.RIGHT
    return FrameLean.MIDDLE


def heads_contained_in_body(
    head_rects: list[Rect], body_rects: list[Rect]
) -> bool | None:
    """Whether every head rectangle lies fully inside the body region.

    ``None`` when there is no head to check; ``False`` when heads exist but no
    body does (so no head can be contained).  A head that is only *partly*
    covered by the body union fails the check.
    """
    if not head_rects:
        return None
    if not body_rects:
        return False
    for head in head_rects:
        head_area = (head[2] - head[0]) * (head[3] - head[1])
        if intersection_area(body_rects, [head]) < head_area:
            return False
    return True


def analyze_composition(
    frame_size: tuple[int, int],
    bodies: list[DetectionBox],
    heads: list[DetectionBox],
) -> FrameComposition:
    """Compute the composition metrics for one frame.

    ``bodies`` are person (whole/upper body) boxes and ``heads`` the head boxes,
    both in ``frame_size`` pixels.  Empty categories yield ``0.0`` percentages,
    ``None`` leans and ``None``/``False`` containment rather than raising.
    """
    width, height = frame_size
    frame_area = width * height
    body_rects = clip_rects(bodies, frame_size)
    head_rects = clip_rects(heads, frame_size)
    scale = 100.0 / frame_area if frame_area > 0 else 0.0
    return FrameComposition(
        frame_size=(width, height),
        body_percent=round(union_area(body_rects) * scale, 2),
        head_percent=round(union_area(head_rects) * scale, 2),
        body_head_overlap_percent=round(
            intersection_area(body_rects, head_rects) * scale, 2
        ),
        heads_in_body=heads_contained_in_body(head_rects, body_rects),
        body_leans=lean(body_rects, width),
        head_leans=lean(head_rects, width),
    )


__all__ = [
    "Rect",
    "analyze_composition",
    "clip_rects",
    "heads_contained_in_body",
    "intersection_area",
    "lean",
    "lean_label",
    "union_area",
]
