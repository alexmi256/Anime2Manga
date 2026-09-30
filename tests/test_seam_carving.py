"""Tests for content-aware seam carving (the retarget engine)."""

from __future__ import annotations

import cv2
import numpy as np

from anime2manga.models import FaceBox
from anime2manga.seam_carving import (
    SeamCarvingConfig,
    carve_height,
    carve_width,
    gradient_energy,
    reconstruct,
)


def _image(width: int = 80, height: int = 60, seed: int = 0) -> np.ndarray:
    """A deterministic textured BGR image."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 0)


def test_gradient_energy_matches_shape_and_dtype():
    gray = np.zeros((12, 16), np.uint8)
    energy = gradient_energy(gray)
    assert energy.shape == (12, 16)
    assert energy.dtype == np.float32


def test_carve_width_hits_target_and_records_steps():
    image = _image()
    result = carve_width(image, 40)
    assert result.image.shape == (60, 40, 3)
    assert result.width0 == 80
    assert result.height0 == 60
    assert len(result.steps) == 40
    assert result.steps[-1].width_after == 40
    assert [round(s.ratio, 4) for s in result.steps[:3]] == [0.0125, 0.025, 0.0375]


def test_carve_width_is_noop_when_target_reached():
    image = _image(40, 30)
    result = carve_width(image, 80)
    assert result.image.shape == image.shape
    assert result.steps == []


def test_seams_are_connected_paths():
    result = carve_width(_image(), 60)
    assert result.seams
    for seam in result.seams:
        assert np.all(np.abs(np.diff(seam.astype(int))) <= 1)


def test_reconstruct_matches_carved_result():
    image = _image()
    result = carve_width(image, 40)
    assert np.array_equal(reconstruct(result, image, len(result.seams)), result.image)
    assert reconstruct(result, image, 10).shape == (60, 70, 3)
    assert reconstruct(result, image, 0).shape == image.shape


def test_face_protection_reduces_face_pixels_removed():
    # Background is noisy and the "face" is a flat low-energy patch, so without
    # protection the cheapest seams happily cut straight through it.
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)
    image[25:55, 70:100] = 128
    face = [FaceBox(x=70, y=25, width=30, height=30)]
    protected = carve_width(image, 60, faces=face)
    unprotected = carve_width(image, 60, faces=face, config=SeamCarvingConfig(protect_faces=False))
    protected_removed = sum(step.protected_pixels for step in protected.steps)
    unprotected_removed = sum(step.protected_pixels for step in unprotected.steps)
    assert protected.protected_total > 0
    assert protected_removed < unprotected_removed


def test_carve_height_reduces_height():
    image = _image(80, 60)
    result = carve_height(image, 40)
    assert result.image.shape == (40, 80, 3)
    assert result.height0 == 60
    assert result.width0 == 80


def test_snapshots_and_seams_can_be_disabled():
    image = _image()
    result = carve_width(
        image, 60, snapshot_ratios=(0.0, 0.125), config=SeamCarvingConfig(record_seams=False)
    )
    assert set(result.snapshots) == {0.0, 0.125}
    assert result.snapshots[0.125].shape[1] < 80
    assert result.seams == []
