"""Tests for step 4 - pan detection, direction and stitching."""

from __future__ import annotations

import cv2
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


def test_camera_direction_names_diagonals():
    assert camera_direction(-10.0, -10.0) == "up-left"
    assert camera_direction(10.0, -10.0) == "up-right"
    assert camera_direction(10.0, 10.0) == "down-right"
    assert camera_direction(-10.0, 10.0) == "down-left"
    # A small cross-axis component keeps the dominant axis name.
    assert camera_direction(-10.0, -1.0) == "left"


def test_detect_pan_finds_segment_within_longer_scene():
    big = synthetic_scene_image(width=1400, height=300, seed=5)

    def crop(x: int):
        return big[20:220, x : x + 300].copy()

    frames = (
        [crop(100) for _ in range(3)]
        + [crop(100 + 40 * i) for i in range(6)]
        + [crop(340) for _ in range(3)]
    )
    times = [i / 4.0 for i in range(len(frames))]
    result = detect_pan(1, times, frames)
    assert result.detected
    assert result.direction == "right"
    # The pan sits inside the scene, not over the whole sampled span.
    assert result.start_time > times[0]
    assert result.end_time < times[-1]
    assert result.canvas_width > frames[0].shape[1]


def test_detect_pan_ignores_low_response_cut():
    first = synthetic_scene_image(width=300, height=200, seed=1)
    second = synthetic_scene_image(width=300, height=200, seed=999)
    frames = [first.copy() for _ in range(4)] + [second.copy() for _ in range(4)]
    times = [i / 4.0 for i in range(len(frames))]
    # The unrelated shots correlate poorly, so their jump must not become a pan.
    assert not detect_pan(1, times, frames).detected


def test_stitch_colour_panorama_has_alpha_coverage():
    big = cv2.cvtColor(synthetic_scene_image(width=600, height=400, seed=8), cv2.COLOR_GRAY2BGR)

    def crop(x: int, y: int):
        return big[y : y + 200, x : x + 250].copy()

    frames = [crop(0, 0), crop(40, 30), crop(80, 60)]
    result = detect_pan(1, [0.0, 0.25, 0.5], frames)
    assert result.detected
    panorama = stitch(result)
    assert panorama.shape[2] == 4
    # Diagonal offsets leave opposing corners uncovered by any frame.
    assert panorama[0, -1, 3] == 0
    assert panorama[-1, 0, 3] == 0
    assert panorama[panorama.shape[0] // 2, panorama.shape[1] // 2, 3] == 255


def test_pan_continues_matches_diagonal_direction():
    big = synthetic_scene_image(width=400, height=400, seed=11)

    def crop(i: int):
        return big[20 + i * 10 : 220 + i * 10, 20 + i * 10 : 220 + i * 10].copy()

    frames = [crop(i) for i in range(5)]
    times = [i / 4.0 for i in range(5)]
    # The crop window moves down-right, so the camera reads as down-right.
    assert pan_continues(times, frames, "down-right")
    assert not pan_continues(times, frames, "up-left")


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
