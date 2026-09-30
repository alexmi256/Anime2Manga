"""Step 8 - anime person / full-body detection on chosen frames.

Person boxes cover a whole character (or an upper body when the frame crops
them), which later stages can use to keep a subject inside a crop or to place
text away from them.  They are stored separately from faces and heads
(``Scene.persons``).

Method
------
Uses the DeepGHS **YOLOv8s anime person detector** (``person_detect_v1.3_s``,
MIT) from ``deepghs/anime_person_detection``: a single-class object detector
with an F1 of ~0.86 on its validation set.  Inference (letterboxing, decode and
NMS) is shared with the face and head detectors in
:mod:`anime2manga.detection`.

The published ONNX export has dynamic shapes OpenCV cannot parse, so the
bundled copy under ``anime2manga/data/`` is the same checkpoint simplified to a
fixed ``1x3x960x960`` input (see ``data/ATTRIBUTION.md`` and
``scripts/fetch_face_model.py``).

The model can be overridden with ``PersonDetectionConfig.model_path`` or the
``ANIME2MANGA_ANIME_PERSON_MODEL`` environment variable.

:func:`detect_persons` is the disk entry point and :func:`annotate_persons`
draws the boxes onto the saved frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import detection
from .detection import CATEGORY_COLORS, DetectorConfig, detect_in_image, draw_boxes
from .models import DetectionBox

#: Environment variable pointing at an alternative person-detection ONNX model.
MODEL_ENV_VAR = "ANIME2MANGA_ANIME_PERSON_MODEL"
#: File name of the ONNX model bundled with the package.
BUNDLED_MODEL = "anime_person_detection_yolov8s.onnx"
#: Drawing colour for person boxes.
PERSON_COLOR = CATEGORY_COLORS["person"]


@dataclass(frozen=True)
class PersonDetectionConfig(DetectorConfig):
    """Tunables for the YOLOv8 anime person detector."""

    model_file: str = BUNDLED_MODEL
    env_var: str = MODEL_ENV_VAR
    #: The model card's F1-optimal threshold (0.324).
    score_threshold: float = 0.324
    nms_threshold: float = 0.45
    input_size: int = 960
    #: A single pass is enough for whole-body targets; multi-scale added
    #: duplicate boxes without helping, so it is not on by default.
    content_scales: tuple[float, ...] = (1.0,)


def model_path(config: PersonDetectionConfig | None = None) -> Path:
    """Resolve which ONNX model to load (config path, env var, then bundled)."""
    return detection.model_path(config or PersonDetectionConfig())


def detect_persons_in_image(
    image, *, config: PersonDetectionConfig | None = None
) -> list[DetectionBox]:
    """Detect anime persons in an in-memory BGR (or BGRA/gray) image."""
    return detect_in_image(image, config or PersonDetectionConfig())


def detect_persons(
    frame_path: Path, *, config: PersonDetectionConfig | None = None
) -> list[DetectionBox]:
    """Return person bounding boxes in the image at ``frame_path``."""
    return detection.detect(frame_path, config or PersonDetectionConfig())


def draw_person_boxes(
    image,
    boxes: list[DetectionBox],
    *,
    color: tuple[int, int, int] = PERSON_COLOR,
    thickness: int | None = None,
):
    """Return a copy of ``image`` with ``boxes`` drawn on it."""
    return draw_boxes(image, boxes, color=color, thickness=thickness)


def annotate_persons(
    frame_path: Path,
    boxes: list[DetectionBox],
    *,
    color: tuple[int, int, int] = PERSON_COLOR,
    thickness: int | None = None,
) -> Path:
    """Draw ``boxes`` onto the image at ``frame_path``, in place."""
    return detection.annotate(frame_path, boxes, color=color, thickness=thickness)


__all__ = [
    "BUNDLED_MODEL",
    "MODEL_ENV_VAR",
    "PERSON_COLOR",
    "PersonDetectionConfig",
    "annotate_persons",
    "detect_persons",
    "detect_persons_in_image",
    "draw_person_boxes",
    "model_path",
]
