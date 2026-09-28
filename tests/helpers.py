"""Synthetic image helpers shared by tests."""

from __future__ import annotations

import cv2
import numpy as np


def synthetic_scene_image(width: int = 400, height: int = 300, seed: int = 0) -> np.ndarray:
    """A low-frequency but textured image that phase correlation handles well."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (height, width), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 0)


def pan_frames(
    *,
    frames: int = 6,
    frame_w: int = 200,
    frame_h: int = 220,
    step: int = 40,
    direction: str = "right",
) -> list[np.ndarray]:
    """Build consecutive crops simulating a camera pan over a big scene."""
    big_w = frame_w + step * (frames - 1) + 60
    big_h = frame_h + 20
    rng = np.random.default_rng(7)
    big = rng.integers(0, 255, (big_h, big_w), dtype=np.uint8)
    big = cv2.GaussianBlur(big, (5, 5), 0)
    crops: list[np.ndarray] = []
    for i in range(frames):
        x0 = 20 + i * step if direction == "right" else 20 + (frames - 1 - i) * step
        crops.append(big[10 : 10 + frame_h, x0 : x0 + frame_w].copy())
    return crops
