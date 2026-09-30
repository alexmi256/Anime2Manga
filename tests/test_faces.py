"""Tests for step 8 - YOLOv8 anime face detection, drawing and annotation.

A committed fixture and a real-model smoke test cover loading and inference.
The decode, letterbox, multi-scale and NMS logic is tested with a fake network /
stubbed ``_detect_raw`` so assertions do not depend on the model recognising a
particular synthetic face.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from anime2manga import detection
from anime2manga.detection import _letterbox
from anime2manga.errors import Anime2MangaError
from anime2manga.faces import (
    MODEL_ENV_VAR,
    FaceDetectionConfig,
    annotate_faces,
    detect_faces,
    detect_faces_in_image,
    draw_face_boxes,
    model_path,
)
from anime2manga.models import FaceBox
from helpers import synthetic_scene_image

#: A committed 320x320 cel-shaded face with its centre at (160, 160).
_FIXTURE = Path(__file__).parent / "data" / "anime_face.png"


class _FakeNet:
    """Minimal stand-in for cv2.dnn Net returning fixed YOLO output."""

    def __init__(self, predictions: np.ndarray) -> None:
        self.predictions = predictions
        self.input: np.ndarray | None = None

    def setInput(self, blob: np.ndarray) -> None:
        self.input = blob

    def forward(self) -> np.ndarray:
        return self.predictions


def _patch_detections(monkeypatch, detections):
    monkeypatch.setattr(detection, "_detect_raw", lambda image, config: list(detections))


def _image() -> np.ndarray:
    """A deterministic 3-channel image (the shared helper is grayscale)."""
    gray = synthetic_scene_image(width=400, height=300)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_bundled_model_loads():
    assert detection._load_net(str(model_path())) is not None


def test_model_path_precedence(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.onnx"
    assert model_path(FaceDetectionConfig(model_path=explicit)) == explicit

    monkeypatch.setenv(MODEL_ENV_VAR, str(tmp_path / "env.onnx"))
    assert model_path() == tmp_path / "env.onnx"
    # An explicit path still wins over the environment.
    assert model_path(FaceDetectionConfig(model_path=explicit)) == explicit


def test_missing_model_raises(tmp_path):
    with pytest.raises(Anime2MangaError):
        detection._load_net(str(tmp_path / "nope.onnx"))


def test_letterbox_preserves_aspect_ratio():
    canvas, scale, pad_x, pad_y = _letterbox(np.zeros((300, 400, 3), np.uint8), 960)
    assert canvas.shape == (960, 960, 3)
    assert scale == pytest.approx(2.4)
    assert (pad_x, pad_y) == (0, 120)
    # The padding uses YOLO's grey fill.
    assert tuple(int(v) for v in canvas[0, 0]) == (114, 114, 114)


def test_letterbox_content_scale_shrinks_frame():
    canvas, scale, pad_x, pad_y = _letterbox(np.zeros((300, 400, 3), np.uint8), 960, 0.5)
    assert canvas.shape == (960, 960, 3)
    assert scale == pytest.approx(1.2)
    assert (pad_x, pad_y) == (240, 300)


def test_decode_maps_predictions_back_to_frame(monkeypatch):
    # A face centred at (125, 110) with size 50x60 in a 400x300 frame maps to
    # (300, 384, 120, 144) after a 2.4x letterbox with 120px top padding.
    predictions = np.array([[[300.0], [384.0], [120.0], [144.0], [0.9]]])
    monkeypatch.setattr(detection, "_load_net", lambda path: _FakeNet(predictions))
    config = FaceDetectionConfig(content_scales=(1.0,))
    assert detection._detect_raw(_image(), config) == [(100, 80, 50, 60, 0.9)]


def test_detect_runs_every_content_scale(monkeypatch):
    seen: list[float] = []

    def fake_run(net, image, config, content_scale):
        seen.append(content_scale)
        return [(1, 2, 3, 4, 0.9)]

    monkeypatch.setattr(detection, "_load_net", lambda path: object())
    monkeypatch.setattr(detection, "_run_net", fake_run)
    detection._detect_raw(_image(), FaceDetectionConfig(content_scales=(1.0, 0.5, 0.25)))
    assert seen == [1.0, 0.5, 0.25]


def test_detect_maps_raw_detections_to_boxes(monkeypatch):
    _patch_detections(monkeypatch, [(10, 20, 30, 40, 0.9)])
    assert detect_faces_in_image(_image()) == [FaceBox(10, 20, 30, 40, 0.9)]


def test_detect_clamps_boxes_to_frame(monkeypatch):
    _patch_detections(monkeypatch, [(390, 290, 40, 40, 0.9)])
    assert detect_faces_in_image(_image()) == [FaceBox(390, 290, 10, 10, 0.9)]


def test_detect_drops_degenerate_boxes(monkeypatch):
    _patch_detections(monkeypatch, [(10, 10, 0, 40, 0.9), (10, 60, 30, 0, 0.9)])
    assert detect_faces_in_image(_image()) == []


def test_detect_suppresses_overlapping_detections(monkeypatch):
    # Duplicates from several content scales plus a separate face: only the
    # stronger duplicate survives, in top-to-bottom / left-to-right order.
    _patch_detections(
        monkeypatch,
        [(5, 5, 100, 100, 0.5), (0, 0, 100, 100, 0.9), (300, 200, 50, 50, 0.6)],
    )
    boxes = detect_faces_in_image(_image())
    assert boxes == [FaceBox(0, 0, 100, 100, 0.9), FaceBox(300, 200, 50, 50, 0.6)]


def test_detect_keeps_distinct_faces_in_reading_order(monkeypatch):
    _patch_detections(monkeypatch, [(200, 50, 40, 40, 0.5), (10, 10, 40, 40, 0.7)])
    boxes = detect_faces_in_image(_image())
    assert [box.x for box in boxes] == [10, 200]


def test_input_size_mismatch_raises():
    previous = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    try:
        with pytest.raises(Anime2MangaError):
            detect_faces_in_image(_image(), config=FaceDetectionConfig(input_size=640))
    finally:
        cv2.utils.logging.setLogLevel(previous)


def test_detect_faces_reads_from_disk(tmp_path, monkeypatch):
    _patch_detections(monkeypatch, [(10, 20, 30, 40, 0.9)])
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), _image())
    assert detect_faces(path) == [FaceBox(10, 20, 30, 40, 0.9)]


def test_detect_faces_missing_file_returns_empty(tmp_path):
    assert detect_faces(tmp_path / "does_not_exist.jpg") == []


def test_detect_faces_handles_gray_and_alpha_images(monkeypatch):
    _patch_detections(monkeypatch, [(10, 20, 30, 40, 0.9)])
    gray = cv2.cvtColor(_image(), cv2.COLOR_BGR2GRAY)
    assert detect_faces_in_image(gray) == [FaceBox(10, 20, 30, 40, 0.9)]

    bgra = cv2.cvtColor(_image(), cv2.COLOR_BGR2BGRA)
    assert detect_faces_in_image(bgra) == [FaceBox(10, 20, 30, 40, 0.9)]


def test_real_model_detects_committed_fixture():
    boxes = detect_faces(_FIXTURE)
    assert boxes, "expected the anime face fixture to be detected"
    # The fixture's face is centred at (160, 160); a loose position check.
    assert any(
        box.x <= 160 <= box.x + box.width and box.y <= 160 <= box.y + box.height for box in boxes
    )


def test_real_model_runs_on_an_image():
    # Exercises the bundled model end to end; the synthetic image may contain
    # zero faces, but every returned box must be valid.
    image = _image()
    boxes = detect_faces_in_image(image)
    height, width = image.shape[:2]
    for box in boxes:
        assert box.x >= 0 and box.y >= 0
        assert box.x + box.width <= width
        assert box.y + box.height <= height


def test_draw_face_boxes_paints_the_border():
    image = _image()
    box = FaceBox(100, 80, 60, 70)
    annotated = draw_face_boxes(image, [box], color=(0, 255, 0), thickness=1)
    # Original image is untouched.
    assert np.array_equal(image, _image())
    # The top-left corner of the box is painted with the requested colour.
    assert tuple(int(v) for v in annotated[box.y, box.x]) == (0, 255, 0)
    assert not np.array_equal(image, annotated)


def test_draw_face_boxes_is_noop_without_boxes():
    image = _image()
    assert np.array_equal(draw_face_boxes(image, []), image)


def test_annotate_faces_rewrites_file(tmp_path):
    path = tmp_path / "frame.png"
    original = _image()
    cv2.imwrite(str(path), original)
    annotate_faces(path, [FaceBox(120, 90, 80, 100)], thickness=2)
    reloaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert reloaded is not None
    assert not np.array_equal(original, reloaded)


def test_annotate_faces_preserves_alpha_channel(tmp_path):
    path = tmp_path / "panorama.png"
    bgra = cv2.cvtColor(_image(), cv2.COLOR_BGR2BGRA)
    cv2.imwrite(str(path), bgra)
    annotate_faces(path, [FaceBox(120, 90, 80, 100)], thickness=2)
    reloaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert reloaded is not None
    assert reloaded.shape[2] == 4


def test_annotate_faces_ignores_missing_file(tmp_path):
    missing = tmp_path / "nope.png"
    assert annotate_faces(missing, [FaceBox(0, 0, 10, 10)]) == missing


def test_annotate_faces_noop_without_boxes(tmp_path):
    path = tmp_path / "frame.png"
    original = _image()
    cv2.imwrite(str(path), original)
    annotate_faces(path, [])
    reloaded = cv2.imread(str(path))
    assert reloaded is not None
    assert np.array_equal(np.asarray(reloaded), original)
