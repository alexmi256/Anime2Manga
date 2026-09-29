"""Step 8 - anime face detection on chosen frames.

Face boxes feed two later decisions: step 9 (cropping) keeps faces inside a
square crop, and step 10 (text placement) keeps speech bubbles off faces.

Method
------
Anime faces are stylised (flat cel shading, oversized eyes, exaggerated
geometry), so photographic detectors such as the Haar frontal-face cascade or
YuNet miss many of them.  This module uses the **YOLOv8n anime face detector**
from DeepGHS (``face_detect_v1.4_n``, MIT): a modern single-class object
detector trained on anime faces, with an F1 of ~0.94 on its validation set.

It runs through ``cv2.dnn`` with no extra runtime dependency.  The published
ONNX export has dynamic shapes that OpenCV's importer cannot parse, so the
bundled copy under ``anime2manga/data/`` is the same checkpoint simplified to a
fixed ``1x3x960x960`` input (see ``data/ATTRIBUTION.md``).  The model can be
overridden with ``FaceDetectionConfig.model_path`` or the
``ANIME2MANGA_ANIME_FACE_MODEL`` environment variable.

Pipeline
--------
1. Letterbox the frame to the model's square input (preserving aspect ratio) at
   several content scales: pass ``1.0`` sees faces at normal size, while ``0.5``
   shrinks the frame so very large close-up faces fall back into the model's
   training scale (a single 960 pass misses them).
2. Run the network, decode the ``(cx, cy, w, h, score)`` rows and keep
   detections above ``score_threshold``.
3. Non-maximum-suppress overlaps (including duplicates found at several scales)
   and map the boxes back to full-resolution frame pixels as
   :class:`~anime2manga.models.FaceBox` values.

:func:`detect_faces` is the disk entry point and :func:`annotate_faces` draws
the boxes onto the saved frame.
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
from .models import FaceBox

#: Environment variable pointing at an alternative face-detection ONNX model.
MODEL_ENV_VAR = "ANIME2MANGA_ANIME_FACE_MODEL"
#: File name of the ONNX model bundled with the package.
BUNDLED_MODEL = "anime_face_detection_yolov8n.onnx"
#: Grey used to pad the letterbox, matching YOLO's training convention.
_LETTERBOX_FILL = 114


@dataclass(frozen=True)
class FaceDetectionConfig:
    """Tunables for the YOLOv8 anime face detector."""

    #: Detections below this score are dropped.  Lower than the model card's
    #: F1 optimum (0.278) because missed faces were the priority and the extra
    #: low-score detections proved to be true faces.
    score_threshold: float = 0.2
    #: IoU above which overlapping detections are merged.
    nms_threshold: float = 0.45
    #: Square network input.  Must match the bundled/custom ONNX model; a larger
    #: value finds smaller faces at the cost of speed.
    input_size: int = 960
    #: Multi-scale factors applied before detection.  The network input is
    #: always ``input_size``, so a factor ``< 1`` shrinks the frame within the
    #: canvas and lets very large close-up faces fall back into the model's
    #: training scale.  ``0.5`` catches the large close-ups; smaller factors
    #: added false boxes without helping, so they are not on by default.
    content_scales: tuple[float, ...] = (1.0, 0.5)
    #: Explicit ONNX model; ``None`` uses the env var or the bundled file.
    model_path: Path | None = None


def model_path(config: FaceDetectionConfig | None = None) -> Path:
    """Resolve which ONNX model to load.

    Precedence: explicit ``config.model_path``, then ``ANIME2MANGA_ANIME_FACE_MODEL``,
    then the file bundled with the package.
    """
    cfg = config or FaceDetectionConfig()
    if cfg.model_path is not None:
        return Path(cfg.model_path)
    override = os.environ.get(MODEL_ENV_VAR)
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "data" / BUNDLED_MODEL


@lru_cache(maxsize=4)
def _load_net(path_str: str) -> Any:
    """Load (and cache) the ONNX network, failing clearly if it is missing."""
    try:
        return cv2.dnn.readNetFromONNX(path_str)
    except cv2.error as exc:
        raise Anime2MangaError(
            f"Could not load the anime face model at {path_str!r}. "
            f"Set {MODEL_ENV_VAR} or FaceDetectionConfig.model_path to a valid "
            "ONNX face-detection model."
        ) from exc


def _letterbox(
    image: np.ndarray, size: int, content_scale: float = 1.0
) -> tuple[np.ndarray, float, int, int]:
    """Fit ``image`` into ``size`` x ``size``, padding to keep the aspect ratio.

    ``content_scale`` < 1 shrinks the frame inside the canvas (the network input
    stays ``size``), which brings very large faces back to a detectable size.
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


def _run_net(
    net: Any, image: np.ndarray, config: FaceDetectionConfig, content_scale: float
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
            f"Face model inference failed (input_size={size}, content_scale={content_scale}). "
            "Make sure FaceDetectionConfig.input_size matches the loaded ONNX model."
        ) from exc

    # YOLOv8 emits ``(1, 4 + classes, anchors)``; one class here, so 5 rows.
    if output.ndim != 3 or output.shape[1] < 5:
        raise Anime2MangaError(
            f"Unexpected face model output shape {tuple(output.shape)}; "
            "expected (1, 5, N) for a single-class YOLOv8 detector."
        )

    predictions = output[0].T
    predictions = predictions[predictions[:, 4] >= config.score_threshold]
    detections: list[tuple[int, int, int, int, float]] = []
    for cx, cy, box_w, box_h, score in predictions:
        detections.append(
            (
                round((float(cx) - float(box_w) / 2 - pad_x) / scale),
                round((float(cy) - float(box_h) / 2 - pad_y) / scale),
                round(float(box_w) / scale),
                round(float(box_h) / scale),
                round(float(score), 3),
            )
        )
    return detections


def _detect_raw(
    image: np.ndarray, config: FaceDetectionConfig
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


def _iou(a: FaceBox, b: FaceBox) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0.0
    union = a.area + b.area - intersection
    return intersection / union if union else 0.0


def _suppress_overlaps(boxes: list[FaceBox], iou_threshold: float) -> list[FaceBox]:
    """Keep the strongest of each overlapping cluster, in reading order."""
    kept: list[FaceBox] = []
    # Strongest first so it absorbs the weaker duplicates it overlaps.
    for box in sorted(boxes, key=lambda b: (-b.confidence, -b.area)):
        if all(_iou(box, other) < iou_threshold for other in kept):
            kept.append(box)
    kept.sort(key=lambda b: (b.y, b.x))
    return kept


def detect_faces_in_image(
    image: np.ndarray, *, config: FaceDetectionConfig | None = None
) -> list[FaceBox]:
    """Detect anime faces in an in-memory BGR (or BGRA/gray) image."""
    cfg = config or FaceDetectionConfig()
    if image is None or image.size == 0:
        return []

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]  # drop panorama alpha; transparent areas are black
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        return []

    height, width = image.shape[:2]
    boxes: list[FaceBox] = []
    for x, y, box_w, box_h, confidence in _detect_raw(image, cfg):
        x, y = max(0, x), max(0, y)
        box_w = min(box_w, width - x)
        box_h = min(box_h, height - y)
        if box_w <= 0 or box_h <= 0:
            continue
        boxes.append(FaceBox(x, y, box_w, box_h, confidence))
    return _suppress_overlaps(boxes, cfg.nms_threshold)


def detect_faces(frame_path: Path, *, config: FaceDetectionConfig | None = None) -> list[FaceBox]:
    """Return face bounding boxes in the image at ``frame_path``.

    A missing/unreadable file yields an empty list so the pipeline can carry on.
    """
    path = Path(frame_path)
    if not path.exists():
        return []
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return []
    return detect_faces_in_image(image, config=config)


def draw_face_boxes(
    image: np.ndarray,
    boxes: list[FaceBox],
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


def annotate_faces(
    frame_path: Path,
    boxes: list[FaceBox],
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
    annotated = draw_face_boxes(image, boxes, color=color, thickness=thickness)
    cv2.imwrite(str(path), annotated)
    return path
