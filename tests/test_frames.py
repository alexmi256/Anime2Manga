"""Tests for step 6 - sharpness, target timing and frame ranking."""

from __future__ import annotations

import cv2
import numpy as np

from anime2manga.frames import (
    rank_candidates,
    save_image,
    select_frame,
    sharpness,
    target_time,
)
from anime2manga.models import SubtitleLine
from helpers import synthetic_scene_image


def _progressively_blurrier() -> list[np.ndarray]:
    base = synthetic_scene_image(width=200, height=200, seed=3)
    images = [base]
    for ksize in (7, 15, 31):
        images.append(cv2.GaussianBlur(base, (ksize, ksize), 0))
    return images


def test_sharpness_ranks_sharp_above_blurry():
    images = _progressively_blurrier()
    scores = [sharpness(image) for image in images]
    assert scores == sorted(scores, reverse=True)


def test_target_time_uses_median_subtitle_midpoint(make_scene):
    scene = make_scene(start=0.0, end=10.0)
    subs = [SubtitleLine(1, 2.0, 4.0, "a"), SubtitleLine(2, 6.0, 8.0, "b")]
    assert target_time(scene, subs) == 5.0


def test_target_time_falls_back_to_midpoint(make_scene):
    scene = make_scene(start=4.0, end=10.0)
    assert target_time(scene, []) == 7.0


def test_rank_candidates_prefers_sharpest_near_target(make_scene):
    images = _progressively_blurrier()
    times = [0.0, 1.0, 2.0, 3.0]
    # Shuffle sharpness so order is not simply time order.
    frames = [images[2], images[0], images[1], images[3]]
    ranked = rank_candidates(times, frames, target=2.0, window=1.5)
    # images[0] is the sharpest and sits at time 1.0, within the window.
    assert ranked[0].time == 1.0


def test_rank_candidates_falls_back_when_window_empty():
    images = _progressively_blurrier()
    ranked = rank_candidates([0.0, 1.0], images[:2], target=50.0, window=0.1)
    assert len(ranked) == 2


def test_select_frame_returns_none_without_frames(make_scene):
    scene = make_scene()
    assert select_frame(scene, [], [], []) is None


def test_save_image_writes_file(tmp_path):
    image = synthetic_scene_image(width=64, height=64)
    out = tmp_path / "img.jpg"
    save_image(image, out)
    assert out.exists()
    reloaded = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert reloaded is not None
    assert reloaded.shape == (64, 64)
