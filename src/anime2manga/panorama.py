"""Step 4 - pan detection and panoramic stitching.

Approach
--------
Anime frames are flat and low-texture, so keypoint stitchers (``cv2.Stitcher``)
fail on them.  Instead we use **phase correlation** between consecutive frames,
which measures global translation robustly:

1. Sample the scene at a fixed rate (default 4 fps) as grayscale.
2. Correlate consecutive frames with a Hanning window to get a sub-pixel
   ``(dx, dy)`` content displacement and a confidence response.
3. Declare a pan only when *all* of these hold:
   * cumulative shift >= ``min_shift_fraction`` of the frame dimension,
   * >= ``consistency`` of the per-pair shifts point the same way,
   * median correlation response >= ``min_response``.
4. Stitch by translating each frame onto a canvas, blending overlaps with a
   Hanning weight to hide seams.  Translation is all a flat pan needs.

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

from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np

from .models import PanResult

#: Camera direction is reported, which is the opposite of content motion.
_DIRECTION_NAMES = {(-1, 0): "left", (1, 0): "right", (0, -1): "up", (0, 1): "down"}


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


def camera_direction(camera_dx: float, camera_dy: float) -> str:
    """Map a cumulative camera displacement to a human-readable direction."""
    if abs(camera_dx) >= abs(camera_dy):
        return _DIRECTION_NAMES[(1 if camera_dx > 0 else -1, 0)]
    return _DIRECTION_NAMES[(0, 1 if camera_dy > 0 else -1)]


def detect_pan(
    scene_index: int,
    times: list[float],
    frames: list[np.ndarray],
    *,
    config: PanConfig | None = None,
) -> PanResult:
    """Analyse sampled frames for a dominant pan and return a :class:`PanResult`."""
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
    cum_x = float(sum(step[0] for step in camera_steps))
    cum_y = float(sum(step[1] for step in camera_steps))
    height, width = frames[0].shape[:2]

    if abs(cum_x) >= abs(cum_y):
        axis_dim = width
        dominant = np.array([1.0 if cum_x > 0 else -1.0, 0.0])
        projected = [step[0] * dominant[0] for step in camera_steps]
    else:
        axis_dim = height
        dominant = np.array([0.0, 1.0 if cum_y > 0 else -1.0])
        projected = [step[1] * dominant[1] for step in camera_steps]

    magnitude = float(np.hypot(cum_x, cum_y))
    shift_fraction = magnitude / float(axis_dim)
    moving = [value for value in projected if abs(value) > 1e-3]
    agreeing = [value for value in moving if value > 0]
    consistency = len(agreeing) / len(moving) if moving else 0.0
    mean_response = float(np.median(responses)) if responses else 0.0

    detected = (
        shift_fraction >= cfg.min_shift_fraction
        and consistency >= cfg.consistency
        and mean_response >= cfg.min_response
    )
    if not detected:
        return PanResult(
            scene_index=scene_index,
            detected=False,
            direction=None,
            cumulative_shift=(cum_x, cum_y),
            consistency=consistency,
            mean_response=mean_response,
        )

    offsets = _canvas_origins(camera_steps)
    canvas_w, canvas_h = _canvas_size(offsets, width, height)
    # Guard against spurious enormous shifts masquerading as a pan.
    if (
        canvas_w > cfg.max_canvas_factor * width
        or canvas_h > cfg.max_canvas_factor * height
    ):
        return PanResult(
            scene_index=scene_index,
            detected=False,
            direction=None,
            cumulative_shift=(cum_x, cum_y),
            consistency=consistency,
            mean_response=mean_response,
        )
    return PanResult(
        scene_index=scene_index,
        detected=True,
        direction=camera_direction(cum_x, cum_y),
        cumulative_shift=(cum_x, cum_y),
        consistency=consistency,
        mean_response=mean_response,
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        offsets=offsets,
        sample_times=list(times),
        frames=list(frames),
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
    """Composite the sampled frames of a detected pan into one wide/tall image."""
    frames = result.frames
    if not frames:
        raise ValueError("PanResult has no frames to stitch")
    base = frames[0]
    channels = 1 if base.ndim == 2 else base.shape[2]
    canvas = np.zeros((result.canvas_height, result.canvas_width, channels), dtype=np.float32)
    weights = np.zeros((result.canvas_height, result.canvas_width, 1), dtype=np.float32)
    weight_mask = _hanning(base.shape[:2])[..., None] if blend else None

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

    weights[weights == 0] = 1.0
    blended = canvas / weights
    out = np.clip(blended, 0, 255).astype(np.uint8)
    if channels == 1:
        return out[..., 0]
    return out


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
    continued = camera_direction(cum_x, cum_y) == direction
    magnitude = float(np.hypot(cum_x, cum_y))
    axis_dim = width if direction in {"left", "right"} else height
    response = float(np.median(responses)) if responses else 0.0
    return (
        continued
        and magnitude / axis_dim >= min_fraction
        and response >= cfg.min_response
    )
