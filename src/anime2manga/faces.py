"""Step 8 - anime face detection on chosen frames.

Face boxes feed two later decisions: step 9 (cropping) keeps faces inside a
square crop, and step 10 (text placement) keeps speech bubbles off faces; the
seam-carving step (8b) also protects them.

Method
------
Anime faces are stylised (flat cel shading, oversized eyes, exaggerated
geometry), so photographic detectors such as the Haar frontal-face cascade or
YuNet miss many of them.  This module uses the **YOLOv8n anime face detector**
from DeepGHS (``face_detect_v1.4_n``, MIT): a modern single-class object
detector trained on anime faces, with an F1 of ~0.94 on its validation set.

Inference (letterboxing, multi-scale, decode and NMS) lives in
:mod:`anime2manga.detection`, shared with the head (:mod:`.heads`) and person
(:mod:`.persons`) detectors.  Only the face-specific model, thresholds and
colour are defined here.

The model can be overridden with ``FaceDetectionConfig.model_path`` or the
``ANIME2MANGA_ANIME_FACE_MODEL`` environment variable.

:func:`detect_faces` is the disk entry point and :func:`annotate_faces` draws
the boxes onto the saved frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import detection
from .detection import (
    CATEGORY_COLORS,
    DetectorConfig,
    detect_in_image,
    draw_boxes,
)
from .models import DetectionBox

#: Environment variable pointing at an alternative face-detection ONNX model.
MODEL_ENV_VAR = "ANIME2MANGA_ANIME_FACE_MODEL"
#: File name of the ONNX model bundled with the package.
BUNDLED_MODEL = "anime_face_detection_yolov8n.onnx"
#: Drawing colour for face boxes.
FACE_COLOR = CATEGORY_COLORS["face"]


@dataclass(frozen=True)
class FaceDetectionConfig(DetectorConfig):
    """Tunables for the YOLOv8 anime face detector."""

    model_file: str = BUNDLED_MODEL
    env_var: str = MODEL_ENV_VAR
    #: Lower than the model card's F1 optimum (0.278) because missed faces were
    #: the priority and the extra low-score detections proved to be true faces.
    score_threshold: float = 0.2
    nms_threshold: float = 0.45
    input_size: int = 960
    #: ``0.5`` catches very large close-up faces that a single pass misses.
    content_scales: tuple[float, ...] = (1.0, 0.5)


def model_path(config: FaceDetectionConfig | None = None) -> Path:
    """Resolve which ONNX model to load.

    Precedence: explicit ``config.model_path``, then ``ANIME2MANGA_ANIME_FACE_MODEL``,
    then the file bundled with the package.
    """
    return detection.model_path(config or FaceDetectionConfig())


def detect_faces_in_image(
    image, *, config: FaceDetectionConfig | None = None
) -> list[DetectionBox]:
    """Detect anime faces in an in-memory BGR (or BGRA/gray) image."""
    return detect_in_image(image, config or FaceDetectionConfig())


def detect_faces(
    frame_path: Path, *, config: FaceDetectionConfig | None = None
) -> list[DetectionBox]:
    """Return face bounding boxes in the image at ``frame_path``."""
    return detection.detect(frame_path, config or FaceDetectionConfig())


def draw_face_boxes(
    image,
    boxes: list[DetectionBox],
    *,
    color: tuple[int, int, int] = FACE_COLOR,
    thickness: int | None = None,
):
    """Return a copy of ``image`` with ``boxes`` drawn on it."""
    return draw_boxes(image, boxes, color=color, thickness=thickness)


def annotate_faces(
    frame_path: Path,
    boxes: list[DetectionBox],
    *,
    color: tuple[int, int, int] = FACE_COLOR,
    thickness: int | None = None,
) -> Path:
    """Draw ``boxes`` onto the image at ``frame_path``, in place."""
    return detection.annotate(frame_path, boxes, color=color, thickness=thickness)


__all__ = [
    "BUNDLED_MODEL",
    "FACE_COLOR",
    "MODEL_ENV_VAR",
    "FaceDetectionConfig",
    "annotate_faces",
    "detect_faces",
    "detect_faces_in_image",
    "draw_face_boxes",
    "model_path",
]
