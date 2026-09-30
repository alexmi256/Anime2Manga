"""Step 8 - anime head detection on chosen frames.

Head boxes are broader than face boxes: they cover the whole head including
hair, and keep working when the face is turned away or partly out of frame.
They are stored separately from faces (``Scene.heads``) so later stages can
choose which signal they need.

Method
------
Uses the DeepGHS **YOLOv8s anime head detector** (``head_detect_v2.0_s``, MIT)
from ``deepghs/anime_head_detection``: a single-class object detector with an
F1 of ~0.92 on its validation set.  Inference (letterboxing, multi-scale,
decode and NMS) is shared with the face and person detectors in
:mod:`anime2manga.detection`.

The published ONNX export has dynamic shapes OpenCV cannot parse, so the
bundled copy under ``anime2manga/data/`` is the same checkpoint simplified to a
fixed ``1x3x960x960`` input (see ``data/ATTRIBUTION.md`` and
``scripts/fetch_face_model.py``).

The model can be overridden with ``HeadDetectionConfig.model_path`` or the
``ANIME2MANGA_ANIME_HEAD_MODEL`` environment variable.

:func:`detect_heads` is the disk entry point and :func:`annotate_heads` draws
the boxes onto the saved frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import detection
from .detection import CATEGORY_COLORS, DetectorConfig, detect_in_image, draw_boxes
from .models import DetectionBox

#: Environment variable pointing at an alternative head-detection ONNX model.
MODEL_ENV_VAR = "ANIME2MANGA_ANIME_HEAD_MODEL"
#: File name of the ONNX model bundled with the package.
BUNDLED_MODEL = "anime_head_detection_yolov8s.onnx"
#: Drawing colour for head boxes.
HEAD_COLOR = CATEGORY_COLORS["head"]


@dataclass(frozen=True)
class HeadDetectionConfig(DetectorConfig):
    """Tunables for the YOLOv8 anime head detector."""

    model_file: str = BUNDLED_MODEL
    env_var: str = MODEL_ENV_VAR
    #: The model card's F1-optimal threshold (0.413).
    score_threshold: float = 0.413
    nms_threshold: float = 0.45
    input_size: int = 960
    #: ``0.5`` catches very large close-up heads that a single pass misses.
    content_scales: tuple[float, ...] = (1.0, 0.5)


def model_path(config: HeadDetectionConfig | None = None) -> Path:
    """Resolve which ONNX model to load (config path, env var, then bundled)."""
    return detection.model_path(config or HeadDetectionConfig())


def detect_heads_in_image(
    image, *, config: HeadDetectionConfig | None = None
) -> list[DetectionBox]:
    """Detect anime heads in an in-memory BGR (or BGRA/gray) image."""
    return detect_in_image(image, config or HeadDetectionConfig())


def detect_heads(
    frame_path: Path, *, config: HeadDetectionConfig | None = None
) -> list[DetectionBox]:
    """Return head bounding boxes in the image at ``frame_path``."""
    return detection.detect(frame_path, config or HeadDetectionConfig())


def draw_head_boxes(
    image,
    boxes: list[DetectionBox],
    *,
    color: tuple[int, int, int] = HEAD_COLOR,
    thickness: int | None = None,
):
    """Return a copy of ``image`` with ``boxes`` drawn on it."""
    return draw_boxes(image, boxes, color=color, thickness=thickness)


def annotate_heads(
    frame_path: Path,
    boxes: list[DetectionBox],
    *,
    color: tuple[int, int, int] = HEAD_COLOR,
    thickness: int | None = None,
) -> Path:
    """Draw ``boxes`` onto the image at ``frame_path``, in place."""
    return detection.annotate(frame_path, boxes, color=color, thickness=thickness)


__all__ = [
    "BUNDLED_MODEL",
    "HEAD_COLOR",
    "MODEL_ENV_VAR",
    "HeadDetectionConfig",
    "annotate_heads",
    "detect_heads",
    "detect_heads_in_image",
    "draw_head_boxes",
    "model_path",
]
