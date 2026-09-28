"""Tests for step 4 - pan detection, direction and stitching."""

from __future__ import annotations

import pytest

from anime2manga.panorama import (
    PanConfig,
    camera_direction,
    detect_pan,
    estimate_shift,
    pan_continues,
    stitch,
)
from helpers import pan_frames, synthetic_scene_image


def test_estimate_shift_reports_content_displacement():
    big = synthetic_scene_image(width=600, height=300)
    a = big[50:250, 100:300]
    b = big[50:250, 140:340]  # content moved left by 40 because the crop moved right
    (dx, dy), response = estimate_shift(a, b)
    assert dx == pytest.approx(-40.0, abs=1.0)
    assert dy == pytest.approx(0.0, abs=1.0)
    assert response > 0.5


def test_camera_direction_mapping():
    assert camera_direction(10.0, 0.0) == "right"
    assert camera_direction(-10.0, 0.0) == "left"
    assert camera_direction(0.0, 10.0) == "down"
    assert camera_direction(0.0, -10.0) == "up"


def test_detect_pan_on_synthetic_pan():
    frames = pan_frames(direction="right")
    times = [i / 4.0 for i in range(len(frames))]
    result = detect_pan(1, times, frames)
    assert result.detected
    assert result.direction == "right"
    assert result.canvas_width > frames[0].shape[1]
    assert result.canvas_height >= frames[0].shape[0]
    assert result.consistency > 0.9


def test_detect_pan_on_static_frames_is_negative():
    frame = synthetic_scene_image(width=200, height=220)
    frames = [frame.copy() for _ in range(6)]
    result = detect_pan(1, [i / 4.0 for i in range(6)], frames)
    assert not result.detected


def test_detect_pan_rejects_absurd_canvas():
    frames = pan_frames(direction="right", step=400, frame_w=200, frames=6)
    result = detect_pan(1, [i / 4.0 for i in range(6)], frames)
    assert not result.detected


def test_stitch_places_frames_side_by_side():
    frames = pan_frames(direction="right")
    result = detect_pan(1, [i / 4.0 for i in range(len(frames))], frames)
    panorama = stitch(result)
    assert panorama.shape[0] >= frames[0].shape[0]
    assert panorama.shape[1] == result.canvas_width
    # The panorama should contain more than one frame's worth of columns.
    assert panorama.shape[1] > frames[0].shape[1] * 1.5


def test_pan_continues_matches_direction():
    frames = pan_frames(direction="right", step=20, frames=6)
    times = [i / 4.0 for i in range(len(frames))]
    assert pan_continues(times, frames, "right")
    assert not pan_continues(times, frames, "left")


def test_pan_config_accepts_custom_thresholds():
    cfg = PanConfig(min_shift_fraction=0.9, consistency=1.0, min_response=0.99)
    frames = pan_frames(direction="right")
    result = detect_pan(1, [0.0, 0.25, 0.5, 0.75, 1.0, 1.25], frames, config=cfg)
    assert not result.detected  # impossible response threshold
