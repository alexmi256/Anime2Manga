"""Shared single-class YOLOv8 detector engine (step 8).

Faces, heads and persons are all detected with the DeepGHS anime detectors
(``deepghs/anime_face_detection``, ``deepghs/anime_head_detection`` and
``deepghs/anime_person_detection``).  They share the same architecture (a
single-class YOLOv8 head) and the same OpenCV-only inference path, so this
module holds the engine once and :mod:`anime2manga.faces`, :mod:`.heads` and
:mod:`.persons` only supply their model, thresholds and drawing colour.

Method
------
The published ONNX exports have dynamic shapes that OpenCV's importer cannot
parse, so the bundled copies under ``anime2manga/data/`` are the same
checkpoints simplified to a fixed square input (see ``data/ATTRIBUTION.md`` and
``scripts/fetch_face_model.py``).  They run through ``cv2.dnn`` with no extra
runtime dependency.  A custom model can be supplied through
:attr:`DetectorConfig.model_path` or the detector's environment variable.

Pipeline
--------
1. Letterbox the frame to the model's square input (preserving aspect ratio) at
   one or more content scales: ``1.0`` sees targets at their normal size, while
   ``0.5`` shrinks the frame so very large close-ups fall back into the model's
   training scale.
2. Run the network, decode the ``(cx, cy, w, h, score)`` rows and keep
   detections above ``score_threshold``.
3. Non-maximum-suppress overlaps (including duplicates found at several scales)
   and map the boxes back to full-resolution frame pixels as
   :class:`~anime2manga.models.DetectionBox` values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .errors import Anime2MangaError
from .models import DetectionBox

#: Grey used to pad the letterbox, matching YOLO's training convention.
_LETTERBOX_FILL = 114

#: BGR drawing colour per detection category.  Faces keep the historical green.
CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "face": (0, 255, 0),
    "head": (255, 0, 0),
    "person": (0, 0, 255),
}


@dataclass(frozen=True)
class DetectorConfig:
    """Tunables shared by the single-class anime detectors.

    Subclasses (:class:`~anime2manga.faces.FaceDetectionConfig`, …) pin the
    model file, score threshold and content scales for one category.
    """

    #: Bundled ONNX file name, resolved under ``anime2manga/data/``.
    model_file: str = ""
    #: Environment variable pointing at an alternative ONNX model.
    env_var: str = ""
    #: Detections below this score are dropped.
    score_threshold: float = 0.25
    #: IoU above which overlapping detections are merged.
    nms_threshold: float = 0.45
    #: Square network input.  Must match the bundled/custom ONNX model; a larger
    #: value finds smaller targets at the cost of speed.
    input_size: int = 960
    #: Multi-scale content factors applied before detection.  The network input
    #: is always ``input_size``, so a factor ``< 1`` shrinks the frame within
    #: the canvas and lets very large close-ups fall back into the model's
    #: training scale.
    content_scales: tuple[float, ...] = (1.0,)
    #: Explicit ONNX model; ``None`` uses the env var or the bundled file.
    model_path: Path | None = None
    #: Fraction of the frame area above which a detection is flagged as
    #: suspiciously large (see :func:`oversized_boxes`).  Whole-frame boxes are
    #: usually detector artefacts on extreme close-ups; they are reported, never
    #: dropped, because they can also be genuine.
    max_box_area_fraction: float = 0.9


def model_path(config: DetectorConfig) -> Path:
    """Resolve which ONNX model to load.

    Precedence: explicit ``config.model_path``, then the config's environment
    variable, then the file bundled with the package.
    """
    if config.model_path is not None:
        return Path(config.model_path)
    override = os.environ.get(config.env_var) if config.env_var else None
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "data" / config.model_file


@lru_cache(maxsize=8)
def _load_net(path_str: str) -> Any:
    """Load (and cache) the ONNX network, failing clearly if it is missing."""
    try:
        return cv2.dnn.readNetFromONNX(path_str)
    except cv2.error as exc:
        raise Anime2MangaError(
            f"Could not load the anime detection model at {path_str!r}. "
            "Set the detector's environment variable or its config's model_path "
            "to a valid ONNX detection model."
        ) from exc


def _letterbox(
    image: np.ndarray, size: int, content_scale: float = 1.0
) -> tuple[np.ndarray, float, int, int]:
    """Fit ``image`` into ``size`` x ``size``, padding to keep the aspect ratio.

    ``content_scale`` < 1 shrinks the frame inside the canvas (the network input
    stays ``size``), which brings very large targets back to a detectable size.
    Returns the padded image plus the scale and top-left padding needed to map
    coordinates back to the original frame.
    """
    height, width = image.shape[:2]
    target = size * content_scale
    scale = min(target / height, target / width)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), _LETTERBOX_FILL, np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def _decode(output: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    """Turn raw single-class YOLO output into ``(cx, cy, w, h, score)`` rows."""
    if output.ndim != 3 or output.shape[0] != 1:
        raise Anime2MangaError(
            f"Unexpected detection model output shape {tuple(output.shape)}; "
            "expected a single-image (1, 4+classes, anchors) tensor."
        )
    rows = output[0]
    # Bundled models emit ``(4+classes, anchors)``; accept the transposed
    # ``(anchors, 4+classes)`` layout too so a custom export still works.  A
    # single-class detector has 5 attribute columns.
    columns = 5
    if rows.shape[0] == columns and rows.shape[1] != columns:
        predictions = rows.T
    elif rows.shape[1] == columns:
        predictions = rows
    else:
        predictions = rows.T if rows.shape[0] <= rows.shape[1] else rows
    if predictions.ndim != 2 or predictions.shape[1] < columns:
        raise Anime2MangaError(
            f"Unexpected detection model output shape {tuple(output.shape)}; "
            "expected at least 5 columns (cx, cy, w, h, score)."
        )
    return [(float(a), float(b), float(c), float(d), float(e)) for a, b, c, d, e in predictions]


def _run_net(
    net: Any, image: np.ndarray, config: DetectorConfig, content_scale: float
) -> list[tuple[int, int, int, int, float]]:
    """Run one detection pass at ``content_scale`` and map boxes to the frame."""
    size = config.input_size
    canvas, scale, pad_x, pad_y = _letterbox(image, size, content_scale)
    blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (size, size), swapRB=True)
    try:
        net.setInput(blob)
        output = net.forward()
    except cv2.error as exc:
        raise Anime2MangaError(
            f"Detection model inference failed (input_size={size}, "
            f"content_scale={content_scale}). Make sure "
            "DetectorConfig.input_size matches the loaded ONNX model."
        ) from exc

    detections: list[tuple[int, int, int, int, float]] = []
    for cx, cy, box_w, box_h, score in _decode(output):
        if score < config.score_threshold:
            continue
        detections.append(
            (
                round((cx - box_w / 2 - pad_x) / scale),
                round((cy - box_h / 2 - pad_y) / scale),
                round(box_w / scale),
                round(box_h / scale),
                round(score, 3),
            )
        )
    return detections


def _detect_raw(
    image: np.ndarray, config: DetectorConfig
) -> list[tuple[int, int, int, int, float]]:
    """Run the network at every content scale and return candidate boxes.

    Cross-scale and within-scale duplicates are removed later by
    :func:`_suppress_overlaps`.
    """
    net = _load_net(str(model_path(config)))
    detections: list[tuple[int, int, int, int, float]] = []
    for content_scale in config.content_scales:
        detections.extend(_run_net(net, image, config, content_scale))
    return detections


def _iou(a: DetectionBox, b: DetectionBox) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0.0
    union = a.area + b.area - intersection
    return intersection / union if union else 0.0


def _suppress_overlaps(boxes: list[DetectionBox], iou_threshold: float) -> list[DetectionBox]:
    """Keep the strongest of each overlapping cluster, in reading order."""
    kept: list[DetectionBox] = []
    # Strongest first so it absorbs the weaker duplicates it overlaps.
    for box in sorted(boxes, key=lambda b: (-b.confidence, -b.area)):
        if all(_iou(box, other) < iou_threshold for other in kept):
            kept.append(box)
    kept.sort(key=lambda b: (b.y, b.x))
    return kept


def detect_in_image(image: np.ndarray, config: DetectorConfig) -> list[DetectionBox]:
    """Detect ``config``'s category in an in-memory BGR (or BGRA/gray) image."""
    if image is None or image.size == 0:
        return []

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]  # drop panorama alpha; transparent areas are black
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        return []

    height, width = image.shape[:2]
    boxes: list[DetectionBox] = []
    for x, y, box_w, box_h, confidence in _detect_raw(image, config):
        x, y = max(0, x), max(0, y)
        box_w = min(box_w, width - x)
        box_h = min(box_h, height - y)
        if box_w <= 0 or box_h <= 0:
            continue
        boxes.append(DetectionBox(x, y, box_w, box_h, confidence))
    return _suppress_overlaps(boxes, config.nms_threshold)


def oversized_boxes(
    boxes: list[DetectionBox],
    frame_size: tuple[int, int],
    *,
    max_area_fraction: float = 0.9,
) -> list[DetectionBox]:
    """Return the boxes whose area covers more than ``max_area_fraction`` of the frame.

    A whole-frame hit is usually a detector artefact on an extreme close-up (the
    head detector readily returns a ~98% box there), but it can also be genuine.
    Callers surface these for inspection rather than dropping them; the
    seam-carving engine already neutralises their saliency (see
    ``seam_carving._subject_energy``), so the warning is purely diagnostic.
    """
    width, height = frame_size
    if width <= 0 or height <= 0:
        return []
    limit = max_area_fraction * width * height
    return [box for box in boxes if box.area > limit]


def detect(frame_path: Path, config: DetectorConfig) -> list[DetectionBox]:
    """Return boxes in the image at ``frame_path``.

    A missing/unreadable file yields an empty list so the pipeline can carry on.
    """
    path = Path(frame_path)
    if not path.exists():
        return []
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return []
    return detect_in_image(image, config)


def draw_boxes(
    image: np.ndarray,
    boxes: list[DetectionBox],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int | None = None,
) -> np.ndarray:
    """Return a copy of ``image`` with ``boxes`` drawn on it."""
    annotated = image.copy()
    if not boxes:
        return annotated

    height, width = annotated.shape[:2]
    if thickness is None:
        thickness = max(1, round(min(width, height) / 300))

    if annotated.ndim == 2:
        line_color = int(sum(color) // 3)
    elif annotated.shape[2] == 4:
        line_color = (color[0], color[1], color[2], 255)
    else:
        line_color = color

    for box in boxes:
        top_left = (box.x, box.y)
        bottom_right = (box.x + box.width, box.y + box.height)
        cv2.rectangle(annotated, top_left, bottom_right, line_color, thickness)
    return annotated


def draw_categories(
    image: np.ndarray,
    groups: list[tuple[str, list[DetectionBox]]],
    *,
    thickness: int | None = None,
) -> np.ndarray:
    """Draw several ``(category, boxes)`` groups, one colour per category."""
    annotated = image
    for category, boxes in groups:
        if boxes:
            annotated = draw_boxes(
                annotated,
                boxes,
                color=CATEGORY_COLORS.get(category, (0, 255, 0)),
                thickness=thickness,
            )
    return annotated


def annotate(
    frame_path: Path,
    boxes: list[DetectionBox],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int | None = None,
) -> Path:
    """Draw ``boxes`` onto the image at ``frame_path``, in place."""
    path = Path(frame_path)
    if not boxes or not path.exists():
        return path
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return path
    annotated = draw_boxes(image, boxes, color=color, thickness=thickness)
    cv2.imwrite(str(path), annotated)
    return path


def annotate_categories(
    frame_path: Path,
    groups: list[tuple[str, list[DetectionBox]]],
    *,
    thickness: int | None = None,
) -> Path:
    """Draw every category's boxes onto ``frame_path`` in a single pass."""
    path = Path(frame_path)
    if not path.exists() or not any(boxes for _, boxes in groups):
        return path
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return path
    annotated = draw_categories(image, groups, thickness=thickness)
    cv2.imwrite(str(path), annotated)
    return path
