"""Tests for the head and person detectors and the shared detection engine.

The decode/letterbox/NMS engine itself is exercised in ``test_faces.py``; here
we cover the head/person configuration, model resolution, the category colours
and the report wiring.  Real-model tests only assert that boxes are valid
(the bundled fixture has no full head/body, so they may legitimately be empty).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from anime2manga import detection, faces, heads, persons
from anime2manga.detection import CATEGORY_COLORS
from anime2manga.models import DetectionBox

_FIXTURE = Path(__file__).parent / "data" / "anime_face.png"


def _bgr(width: int = 400, height: int = 300) -> np.ndarray:
    return np.full((height, width, 3), 30, np.uint8)


def _patch_raw(monkeypatch, detections):
    monkeypatch.setattr(detection, "_detect_raw", lambda image, config: list(detections))


def test_head_config_defaults():
    config = heads.HeadDetectionConfig()
    assert config.model_file == heads.BUNDLED_MODEL
    assert config.score_threshold == pytest.approx(0.413)
    assert config.input_size == 960
    assert config.content_scales == (1.0, 0.5)


def test_person_config_defaults():
    config = persons.PersonDetectionConfig()
    assert config.model_file == persons.BUNDLED_MODEL
    assert config.score_threshold == pytest.approx(0.324)
    assert config.input_size == 960
    assert config.content_scales == (1.0,)


def test_head_model_path_env_override(tmp_path, monkeypatch):
    explicit = tmp_path / "head.onnx"
    assert heads.model_path(heads.HeadDetectionConfig(model_path=explicit)) == explicit

    monkeypatch.setenv(heads.MODEL_ENV_VAR, str(tmp_path / "env.onnx"))
    assert heads.model_path() == tmp_path / "env.onnx"
    assert heads.model_path(heads.HeadDetectionConfig(model_path=explicit)) == explicit


def test_person_model_path_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(persons.MODEL_ENV_VAR, str(tmp_path / "env.onnx"))
    assert persons.model_path() == tmp_path / "env.onnx"


def test_oversized_boxes_flags_near_full_frame_only():
    frame = (200, 100)  # 20000 px
    small = DetectionBox(0, 0, 40, 30)  # 1200 px = 6%
    big = DetectionBox(0, 0, 199, 99)  # 19701 px = 98.5%
    flagged = detection.oversized_boxes([small, big], frame, max_area_fraction=0.9)
    assert flagged == [big]


def test_oversized_boxes_respects_threshold_and_empty_frame():
    frame = (100, 100)
    half = DetectionBox(0, 0, 100, 50)  # exactly 50%
    assert detection.oversized_boxes([half], frame, max_area_fraction=0.5) == []
    assert detection.oversized_boxes([half], frame, max_area_fraction=0.49) == [half]
    assert detection.oversized_boxes([half], (0, 0)) == []


def test_detector_config_oversized_threshold_default():
    assert faces.FaceDetectionConfig().max_box_area_fraction == pytest.approx(0.9)
    assert heads.HeadDetectionConfig().max_box_area_fraction == pytest.approx(0.9)
    assert persons.PersonDetectionConfig().max_box_area_fraction == pytest.approx(0.9)


def test_detect_heads_maps_raw_detections(monkeypatch):
    _patch_raw(monkeypatch, [(10, 20, 30, 40, 0.9)])
    assert heads.detect_heads_in_image(_bgr()) == [DetectionBox(10, 20, 30, 40, 0.9)]


def test_detect_persons_maps_raw_detections(monkeypatch):
    _patch_raw(monkeypatch, [(1, 2, 300, 250, 0.8)])
    assert persons.detect_persons_in_image(_bgr()) == [DetectionBox(1, 2, 300, 250, 0.8)]


def test_detect_heads_reads_from_disk(tmp_path, monkeypatch):
    _patch_raw(monkeypatch, [(5, 6, 7, 8, 0.7)])
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), _bgr())
    assert heads.detect_heads(path) == [DetectionBox(5, 6, 7, 8, 0.7)]


def test_detect_persons_missing_file_returns_empty(tmp_path):
    assert persons.detect_persons(tmp_path / "nope.jpg") == []


def test_bundled_head_and_person_models_load():
    assert detection._load_net(str(heads.model_path())) is not None
    assert detection._load_net(str(persons.model_path())) is not None


def test_head_and_person_models_run_on_an_image():
    image = _bgr(width=320, height=320)
    height, width = image.shape[:2]
    for boxes in (heads.detect_heads_in_image(image), persons.detect_persons_in_image(image)):
        for box in boxes:
            assert box.x >= 0 and box.y >= 0
            assert box.x + box.width <= width
            assert box.y + box.height <= height


def test_category_colors_are_distinct():
    colors = {tuple(CATEGORY_COLORS[name]) for name in ("face", "head", "person")}
    assert len(colors) == 3
    assert CATEGORY_COLORS["face"] == faces.FACE_COLOR
    assert CATEGORY_COLORS["head"] == heads.HEAD_COLOR
    assert CATEGORY_COLORS["person"] == persons.PERSON_COLOR


def test_draw_categories_uses_one_color_per_category():
    image = _bgr()
    boxes = {
        "face": [DetectionBox(10, 10, 20, 20)],
        "head": [DetectionBox(60, 10, 20, 20)],
        "person": [DetectionBox(110, 10, 20, 20)],
    }
    annotated = detection.draw_categories(image, list(boxes.items()), thickness=1)
    assert np.array_equal(image, _bgr())  # original untouched
    for name, box in ((name, box) for name, items in boxes.items() for box in items):
        pixel = tuple(int(v) for v in annotated[box.y, box.x])
        assert pixel == CATEGORY_COLORS[name]


def test_draw_categories_is_noop_without_boxes():
    image = _bgr()
    assert np.array_equal(detection.draw_categories(image, [("face", [])]), image)


def test_annotate_categories_writes_all_and_preserves_alpha(tmp_path):
    path = tmp_path / "panorama.png"
    cv2.imwrite(str(path), cv2.cvtColor(_bgr(), cv2.COLOR_BGR2BGRA))
    groups = [
        ("face", [DetectionBox(5, 5, 20, 20)]),
        ("head", [DetectionBox(60, 5, 20, 20)]),
        ("person", [DetectionBox(115, 5, 20, 20)]),
    ]
    detection.annotate_categories(path, groups, thickness=1)
    reloaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert reloaded is not None
    assert reloaded.shape[2] == 4
    assert tuple(int(v) for v in reloaded[5, 5][:3]) == CATEGORY_COLORS["face"]
    assert tuple(int(v) for v in reloaded[5, 115][:3]) == CATEGORY_COLORS["person"]
