"""Step 4 - pan detection and panoramic stitching.

Approach
--------
Anime frames are flat and low-texture, so keypoint stitchers (``cv2.Stitcher``)
fail on them.  Instead we use **phase correlation** between consecutive frames,
which measures global translation robustly:

1. Sample the scene at a fixed rate (default 4 fps) as grayscale.
2. Correlate consecutive frames with a Hanning window to get a sub-pixel
   ``(dx, dy)`` content displacement and a confidence response.
3. Split the scene into **contiguous motion segments** at pairs whose response
   is too low to trust (cuts, repeats, heavy motion blur).  Within each segment
   search for the sub-run with the largest coherent translation and declare a
   pan only when *all* of these hold:
   * cumulative shift >= ``min_shift_fraction`` of the frame dimension,
   * >= ``consistency`` of the per-pair shifts point the same way,
   * median response over the *moving* pairs >= ``min_response``.
4. Stitch by translating each frame onto a canvas, blending overlaps with a
   Hanning weight to hide seams.  Translation is all a flat pan needs.

Segmenting matters because scene detection can merge several shots into one
scene (soft cuts, dark scenes).  Aggregating motion over such a scene mixes
incompatible camera moves, which both hides genuine pans and invents
non-existent ones, so we look for the best coherent pan *inside* the scene
instead.

Note on motion vectors
----------------------
H.264/H.265 motion vectors would make detection much faster, but stock
``ffprobe`` only reports the *presence* of the ``Motion vectors`` side-data
block, not the vector values (they are binary and must be parsed through the
libav API or a custom build).  :mod:`anime2manga.motion_vectors` documents this
and the intended hybrid (MVs for candidate detection, phase correlation for
precise stitching).  Phase correlation is the reliable default for now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import cv2
import numpy as np

from .models import PanResult

#: Camera direction is reported, which is the opposite of content motion.
_DIRECTION_SIGNS: dict[str, tuple[int, int]] = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
    "up-left": (-1, -1),
    "up-right": (1, -1),
    "down-left": (-1, 1),
    "down-right": (1, 1),
}

#: Camera steps smaller than this (pixels, at analysis resolution) count as static.
_STATIC_EPSILON = 0.5


@dataclass
class PanConfig:
    """Tunable pan-detection parameters (all exposed on the CLI)."""

    sample_fps: float = 4.0
    min_shift_fraction: float = 0.25
    consistency: float = 0.8
    min_response: float = 0.6
    peek_seconds: float = 3.0
    #: Refuse to build absurdly large canvases from spurious shifts.
    max_canvas_factor: float = 4.0
    #: Phase-correlation response below which a pair is treated as a cut/blur break.
    min_pair_response: float = 0.2
    #: How much of the dominant shift a cross-axis component needs to name a diagonal.
    diagonal_ratio: float = 0.35


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32)
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _hanning(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return cv2.createHanningWindow((width, height), cv2.CV_32F)


def estimate_shift(
    previous: np.ndarray, current: np.ndarray
) -> tuple[tuple[float, float], float]:
    """Return the content displacement and correlation response between frames.

    The displacement is how far the *content* moved from ``previous`` to
    ``current`` in pixels: positive ``dx`` means content moved right (the camera
    panned left).
    """
    prev_gray = _to_gray(previous)
    cur_gray = _to_gray(current)
    window = _hanning(prev_gray.shape[:2])
    (dx, dy), response = cv2.phaseCorrelate(prev_gray, cur_gray, window)
    return (float(dx), float(dy)), float(response)


def _displacement_series(
    frames: list[np.ndarray],
) -> tuple[list[tuple[float, float]], list[float]]:
    displacements: list[tuple[float, float]] = []
    responses: list[float] = []
    for previous, current in pairwise(frames):
        (dx, dy), response = estimate_shift(previous, current)
        displacements.append((dx, dy))
        responses.append(response)
    return displacements, responses


def camera_direction(
    camera_dx: float, camera_dy: float, *, diagonal_ratio: float = 0.35
) -> str:
    """Map a cumulative camera displacement to a human-readable direction.

    Diagonal motion (both axes carry a meaningful share of the shift) is named
    ``"up-left"``, ``"down-right"`` and so on; otherwise the dominant axis wins.
    """
    ax, ay = abs(camera_dx), abs(camera_dy)
    horizontal = "left" if camera_dx < 0 else "right"
    vertical = "up" if camera_dy < 0 else "down"
    larger = max(ax, ay)
    if larger <= 1e-9:
        return "right"
    if min(ax, ay) >= diagonal_ratio * larger:
        return f"{vertical}-{horizontal}"
    return horizontal if ax >= ay else vertical


def _direction_matches(cum_x: float, cum_y: float, direction: str) -> bool:
    """True when the displacement agrees with ``direction`` on its moving axes."""
    sign_x, sign_y = _DIRECTION_SIGNS.get(direction, (0, 0))
    if not (sign_x or sign_y):
        return False
    if sign_x and cum_x * sign_x <= 0:
        return False
    return not (sign_y and cum_y * sign_y <= 0)


@dataclass
class _Segment:
    """A contiguous, reliable run of frame pairs and its translation statistics."""

    first_pair: int
    last_pair: int
    camera_steps: list[tuple[float, float]]
    cumulative_shift: tuple[float, float]
    consistency: float
    response: float
    magnitude: float
    shift_fraction: float
    offsets: list[tuple[float, float]] = field(default_factory=list)
    canvas_width: int = 0
    canvas_height: int = 0

    @property
    def frame_span(self) -> int:
        return self.last_pair - self.first_pair + 1

    @property
    def first_frame(self) -> int:
        return self.first_pair

    @property
    def last_frame(self) -> int:
        return self.last_pair + 1


def _segment_stats(
    camera_steps: list[tuple[float, float]],
    responses: list[float],
    first_pair: int,
    last_pair: int,
    width: int,
    height: int,
) -> _Segment | None:
    steps = camera_steps[first_pair : last_pair + 1]
    cum_x = float(sum(step[0] for step in steps))
    cum_y = float(sum(step[1] for step in steps))
    magnitude = float(np.hypot(cum_x, cum_y))
    if abs(cum_x) >= abs(cum_y):
        axis_dim = width
        dominant = 1.0 if cum_x >= 0 else -1.0
        projected = [step[0] * dominant for step in steps]
    else:
        axis_dim = height
        dominant = 1.0 if cum_y >= 0 else -1.0
        projected = [step[1] * dominant for step in steps]

    moving = [
        index
        for index, step in enumerate(steps)
        if abs(step[0]) > _STATIC_EPSILON or abs(step[1]) > _STATIC_EPSILON
    ]
    if not moving:
        return None
    moving_values = [projected[index] for index in moving]
    agreeing = [value for value in moving_values if value > 0]
    consistency = len(agreeing) / len(moving_values)
    response = float(np.median([responses[first_pair + index] for index in moving]))
    offsets = _canvas_origins(steps)
    canvas_w, canvas_h = _canvas_size(offsets, width, height)
    return _Segment(
        first_pair=first_pair,
        last_pair=last_pair,
        camera_steps=steps,
        cumulative_shift=(cum_x, cum_y),
        consistency=consistency,
        response=response,
        magnitude=magnitude,
        shift_fraction=magnitude / float(axis_dim) if axis_dim else 0.0,
        offsets=offsets,
        canvas_width=canvas_w,
        canvas_height=canvas_h,
    )


def _segment_passes(segment: _Segment, width: int, height: int, cfg: PanConfig) -> bool:
    if segment.shift_fraction < cfg.min_shift_fraction:
        return False
    if segment.consistency < cfg.consistency:
        return False
    if segment.response < cfg.min_response:
        return False
    return not (
        segment.canvas_width > cfg.max_canvas_factor * width
        or segment.canvas_height > cfg.max_canvas_factor * height
    )


def _best_segment(
    camera_steps: list[tuple[float, float]],
    responses: list[float],
    width: int,
    height: int,
    cfg: PanConfig,
) -> tuple[_Segment | None, _Segment | None]:
    """Return ``(best_passing, best_overall)`` motion segments for a scene.

    Segments never bridge a pair whose response is below
    ``cfg.min_pair_response``, so a cut cannot be stitched across.  Within each
    reliable run we search every sub-run and keep the largest coherent
    translation that satisfies the pan thresholds (ties go to the tighter span).

    Complexity is O(n^3) in the longest reliable run (every sub-run reslices and
    resums); fine for the short per-scene runs this runs on, but prefix sums
    over ``cum_x``/``cum_y`` and directional counts would make it O(n^2).
    """
    pair_count = len(camera_steps)
    best_overall: _Segment | None = None
    best_passing: _Segment | None = None
    key = lambda seg: (seg.magnitude, -seg.frame_span)  # noqa: E731

    start = 0
    while start < pair_count:
        if responses[start] < cfg.min_pair_response:
            start += 1
            continue
        end = start
        while end + 1 < pair_count and responses[end + 1] >= cfg.min_pair_response:
            end += 1
        for first in range(start, end + 1):
            for last in range(first + 1, end + 1):
                segment = _segment_stats(
                    camera_steps, responses, first, last, width, height
                )
                if segment is None:
                    continue
                if best_overall is None or key(segment) > key(best_overall):
                    best_overall = segment
                if _segment_passes(segment, width, height, cfg) and (
                    best_passing is None or key(segment) > key(best_passing)
                ):
                    best_passing = segment
        start = end + 1
    return best_passing, best_overall


def detect_pan(
    scene_index: int,
    times: list[float],
    frames: list[np.ndarray],
    *,
    config: PanConfig | None = None,
) -> PanResult:
    """Analyse sampled frames for a dominant pan and return a :class:`PanResult`.

    The returned result describes the strongest coherent motion segment, not
    necessarily the whole scene: ``sample_times``/``frames`` cover only the pan.
    Callers can use ``start_time``/``end_time`` to isolate that span.
    """
    cfg = config or PanConfig()
    empty = PanResult(
        scene_index=scene_index,
        detected=False,
        direction=None,
        cumulative_shift=(0.0, 0.0),
        consistency=0.0,
        mean_response=0.0,
    )
    if len(frames) < 3:
        return empty

    displacements, responses = _displacement_series(frames)
    if not displacements:
        return empty

    # Camera displacement is the negation of content displacement.
    camera_steps = [(-dx, -dy) for dx, dy in displacements]
    height, width = frames[0].shape[:2]
    if all(
        abs(step_x) <= _STATIC_EPSILON and abs(step_y) <= _STATIC_EPSILON
        for step_x, step_y in camera_steps
    ):
        return empty

    best_passing, best_overall = _best_segment(
        camera_steps, responses, width, height, cfg
    )
    chosen = best_passing or best_overall
    if chosen is None:
        return empty

    cum_x, cum_y = chosen.cumulative_shift
    detected = best_passing is not None
    first, last = chosen.first_frame, chosen.last_frame
    return PanResult(
        scene_index=scene_index,
        detected=detected,
        direction=(
            camera_direction(cum_x, cum_y, diagonal_ratio=cfg.diagonal_ratio)
            if detected
            else None
        ),
        cumulative_shift=(cum_x, cum_y),
        consistency=chosen.consistency,
        mean_response=chosen.response,
        canvas_width=chosen.canvas_width,
        canvas_height=chosen.canvas_height,
        offsets=chosen.offsets,
        sample_times=list(times[first : last + 1]),
        frames=list(frames[first : last + 1]),
        start_time=float(times[first]),
        end_time=float(times[last]),
    )


def _canvas_origins(
    camera_steps: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Cumulative canvas origin for each frame, normalised so the min is 0."""
    origins: list[tuple[float, float]] = [(0.0, 0.0)]
    x = y = 0.0
    for step_x, step_y in camera_steps:
        x += step_x
        y += step_y
        origins.append((x, y))
    min_x = min(origin[0] for origin in origins)
    min_y = min(origin[1] for origin in origins)
    return [(origin[0] - min_x, origin[1] - min_y) for origin in origins]


def _canvas_size(
    origins: list[tuple[float, float]], frame_w: int, frame_h: int
) -> tuple[int, int]:
    max_x = max(origin[0] for origin in origins)
    max_y = max(origin[1] for origin in origins)
    return round(max_x + frame_w), round(max_y + frame_h)


def stitch(result: PanResult, *, blend: bool = True) -> np.ndarray:
    """Composite the sampled frames of a detected pan into one wide/tall image.

    Colour panoramas are returned as **BGRA** where the alpha channel marks
    coverage: fully transparent pixels are areas no sampled frame reached, which
    a later stage can fill in.  Grayscale inputs stay grayscale (no alpha).

    The blend window is floored so every covered pixel receives some weight;
    otherwise a frame border could blend to black while its alpha still said
    "covered".
    """
    frames = result.frames
    if not frames:
        raise ValueError("PanResult has no frames to stitch")
    base = frames[0]
    channels = 1 if base.ndim == 2 else base.shape[2]
    canvas = np.zeros((result.canvas_height, result.canvas_width, channels), dtype=np.float32)
    weights = np.zeros((result.canvas_height, result.canvas_width, 1), dtype=np.float32)
    coverage = np.zeros((result.canvas_height, result.canvas_width, 1), dtype=bool)
    weight_mask = (
        (0.02 + 0.98 * _hanning(base.shape[:2]))[..., None] if blend else None
    )

    for frame, (origin_x, origin_y) in zip(frames, result.offsets, strict=False):
        x0 = round(origin_x)
        y0 = round(origin_y)
        height, width = frame.shape[:2]
        patch = frame.astype(np.float32)
        if patch.ndim == 2:
            patch = patch[..., None]
        x1 = min(x0 + width, result.canvas_width)
        y1 = min(y0 + height, result.canvas_height)
        x0c, y0c = max(x0, 0), max(y0, 0)
        patch = patch[y0c - y0 : y1 - y0, x0c - x0 : x1 - x0]
        if weight_mask is not None:
            mask = weight_mask[y0c - y0 : y1 - y0, x0c - x0 : x1 - x0]
        else:
            mask = np.ones((y1 - y0c, x1 - x0c, 1), dtype=np.float32)
        canvas[y0c:y1, x0c:x1] += patch * mask
        weights[y0c:y1, x0c:x1] += mask
        coverage[y0c:y1, x0c:x1] = True

    safe_weights = weights.copy()
    safe_weights[safe_weights == 0] = 1.0
    blended = canvas / safe_weights
    out = np.clip(blended, 0, 255).astype(np.uint8)
    if channels == 1:
        return out[..., 0]
    alpha = coverage.astype(np.uint8) * 255
    return np.concatenate([out, alpha], axis=2)


def pan_continues(
    peek_times: list[float],
    peek_frames: list[np.ndarray],
    direction: str,
    *,
    config: PanConfig | None = None,
    min_fraction: float = 0.1,
) -> bool:
    """Return True if motion in a peek window continues ``direction``.

    Used to decide whether a pan spills over a scene boundary.  We only require a
    modest additional shift (``min_fraction`` of the dimension) in the same
    direction, because the peek window is short.
    """
    cfg = config or PanConfig()
    if len(peek_frames) < 2:
        return False
    displacements, responses = _displacement_series(peek_frames)
    if not displacements:
        return False
    camera_steps = [(-dx, -dy) for dx, dy in displacements]
    cum_x = sum(step[0] for step in camera_steps)
    cum_y = sum(step[1] for step in camera_steps)
    height, width = peek_frames[0].shape[:2]
    if not _direction_matches(cum_x, cum_y, direction):
        return False
    magnitude = float(np.hypot(cum_x, cum_y))
    if direction in _DIRECTION_SIGNS and len(direction.split("-")) == 2:
        axis_dim = max(width, height)
    elif direction in {"left", "right"}:
        axis_dim = width
    else:
        axis_dim = height
    response = float(np.median(responses)) if responses else 0.0
    return magnitude / axis_dim >= min_fraction and response >= cfg.min_response
