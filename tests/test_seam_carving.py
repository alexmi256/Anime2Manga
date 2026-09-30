"""Tests for content-aware seam carving (the retarget engine)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from anime2manga import seam_carving as sc
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
    protected = carve_width(image, 60, boxes=face)
    unprotected = carve_width(image, 60, boxes=face, config=SeamCarvingConfig(protect_subjects=False))
    protected_removed = sum(step.protected_pixels for step in protected.steps)
    unprotected_removed = sum(step.protected_pixels for step in unprotected.steps)
    assert protected.protected_total > 0
    assert protected_removed < unprotected_removed


def test_carve_width_rejects_removed_faces_keyword():
    """The old ``faces`` alias was removed, so passing it is a TypeError."""
    kwargs: dict = {"faces": [FaceBox(x=1, y=1, width=2, height=2)]}
    with pytest.raises(TypeError):
        carve_width(_image(), 40, **kwargs)


def test_head_and_person_boxes_are_protected_like_faces():
    """Every category inherits the same protection: boxes are category-agnostic."""
    rng = np.random.default_rng(11)
    image = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)
    image[25:55, 70:100] = 128  # a flat, otherwise "free" region
    heads = [FaceBox(x=70, y=25, width=30, height=30)]
    protected = carve_width(image, 60, boxes=heads)
    unprotected = carve_width(
        image, 60, boxes=heads, config=SeamCarvingConfig(protect_subjects=False)
    )
    assert protected.protected_total > 0
    assert sum(s.protected_pixels for s in protected.steps) < sum(
        s.protected_pixels for s in unprotected.steps
    )


def test_full_frame_near_degenerate_box_gets_no_protection():
    """A box covering nearly the whole frame must not clamp the energy factor.

    The subject penalty is ``subject_energy_factor * base_energy``.  When the
    protected mask spans the frame, that factor is rescaled by the mask's share
    of the image, so an over-large detector box (a near full-frame head on an
    extreme close-up) cannot swamp the 50x saliency and make every seam cost the
    same, which would freeze carving just below the whole-frame box's grid.
    """
    image = np.full((80, 120, 3), 128, np.uint8)  # flat: every seam is "free"
    degenerate = [FaceBox(x=2, y=2, width=116, height=76)]
    result = carve_width(image, 60, boxes=degenerate)
    assert result.protected_total > 0
    # The carve must reach the target width unhindered...
    assert result.image.shape == (80, 60, 3)
    assert len(result.steps) == 60
    # ...and remove no more protected pixels than carving the same boxes with
    # protection disabled.  The frozen-mask artefact removes every leftover
    # seam from inside the box, so it would sit far above this baseline.
    baseline = carve_width(
        image, 60, boxes=degenerate, config=SeamCarvingConfig(protect_subjects=False)
    )
    assert sum(step.protected_pixels for step in result.steps) <= sum(
        step.protected_pixels for step in baseline.steps
    )


def test_carve_height_protects_rotated_boxes():
    rng = np.random.default_rng(6)
    image = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    image[50:70, 30:50] = 128
    boxes = [FaceBox(x=30, y=50, width=20, height=20)]
    result = carve_height(image, 50, boxes=boxes)
    assert result.protected_total > 0
    assert result.image.shape == (50, 80, 3)


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


def test_python_fallback_used_when_native_absent(monkeypatch):
    # ``carve_width`` must keep working when the compiled engine is missing.
    monkeypatch.setattr(sc, "_NATIVE_CARVE", None)
    result = carve_width(_image(), 40)
    assert result.image.shape == (60, 40, 3)
    assert len(result.steps) == 40


@pytest.mark.skipif(not sc.HAVE_NATIVE, reason="native backend not built")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_native_matches_python(seed):
    image = _image(seed=seed)
    native = carve_width(image, 40)
    python = sc._carve_width_python(image, 40)
    assert np.array_equal(native.image, python.image)
    # base_energy/energy_sum0 are float32 reductions in Python and double sums
    # in C++, so they only agree to float rounding (they differ on large frames).
    assert native.base_energy == pytest.approx(python.base_energy, rel=1e-6)
    assert native.energy_sum0 == pytest.approx(python.energy_sum0, rel=1e-6)
    assert native.detail_total == python.detail_total
    assert native.protected_total == python.protected_total
    assert len(native.seams) == len(python.seams)
    for a, b in zip(native.seams, python.seams, strict=True):
        assert np.array_equal(a, b)


@pytest.mark.skipif(not sc.HAVE_NATIVE, reason="native backend not built")
@pytest.mark.parametrize("seed", range(5))
def test_native_matches_python_quantile_boundary(seed):
    # 24x49 => N - 1 == 1175 and 0.8 * 1175 == 940 exactly, so the detail
    # threshold lands on an order statistic.  Passing the quantile as a float32
    # (0.8000000119...) shifted it just past the boundary and dropped tied
    # pixels from detail_total.
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (24, 49, 3), dtype=np.uint8)
    native = carve_width(image, 30)
    python = sc._carve_width_python(image, 30)
    assert native.detail_total == python.detail_total
    assert np.array_equal(native.image, python.image)


@pytest.mark.skipif(not sc.HAVE_NATIVE, reason="native backend not built")
def test_native_matches_python_dilation_rounding_boundary():
    # 0.1 * min(25, 37) == 2.5: Python round() and the C++ must both round
    # half-to-even (radius 2), or the protected mask and chosen seams diverge.
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (25, 37, 3), dtype=np.uint8)
    face = [FaceBox(x=8, y=4, width=9, height=9)]
    cfg = SeamCarvingConfig(subject_dilation=0.1)
    native = carve_width(image, 25, boxes=face, config=cfg)
    python = sc._carve_width_python(image, 25, boxes=face, config=cfg)
    assert native.protected_total == python.protected_total
    assert np.array_equal(native.image, python.image)


@pytest.mark.skipif(not sc.HAVE_NATIVE, reason="native backend not built")
def test_native_matches_python_with_mixed_category_boxes():
    # Head/person boxes go through the same native path as faces; parity must hold.
    rng = np.random.default_rng(17)
    image = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)
    boxes = [
        FaceBox(x=70, y=25, width=30, height=30),
        FaceBox(x=5, y=40, width=25, height=35),
    ]
    native = carve_width(image, 60, boxes=boxes)
    python = sc._carve_width_python(image, 60, boxes=boxes)
    assert np.array_equal(native.image, python.image)
    assert native.protected_total == python.protected_total


@pytest.mark.skipif(not sc.HAVE_NATIVE, reason="native backend not built")
def test_native_matches_python_single_row_image():
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, (1, 20, 3), dtype=np.uint8)
    native = carve_width(image, 12)
    python = sc._carve_width_python(image, 12)
    assert native.image.shape == (1, 12, 3)
    assert np.array_equal(native.image, python.image)


@pytest.mark.skipif(not sc.HAVE_NATIVE, reason="native backend not built")
def test_native_callback_exception_propagates():
    def boom(_k, _image, _step):
        raise ValueError("callback boom")

    with pytest.raises(ValueError, match="callback boom"):
        carve_width(_image(), 40, on_seam=boom)


@pytest.mark.skipif(not sc.HAVE_NATIVE, reason="native backend not built")
def test_native_matches_python_with_faces_snapshots_and_callback():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)
    face = [FaceBox(x=70, y=25, width=30, height=30)]
    seen: list[int] = []
    native = carve_width(
        image,
        60,
        boxes=face,
        config=SeamCarvingConfig(record_seams=True),
        snapshot_ratios=(0.0, 0.25),
        on_seam=lambda k, _image, _step: seen.append(k),
    )
    python = sc._carve_width_python(
        image, 60, boxes=face, config=SeamCarvingConfig(record_seams=True),
        snapshot_ratios=(0.0, 0.25),
    )
    assert np.array_equal(native.image, python.image)
    assert set(native.snapshots) == set(python.snapshots)
    for ratio in native.snapshots:
        assert np.array_equal(native.snapshots[ratio], python.snapshots[ratio])
    assert seen == list(range(1, len(native.steps) + 1))
