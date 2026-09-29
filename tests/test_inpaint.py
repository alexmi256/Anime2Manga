"""Tests for content-aware panorama infill and its method registry."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from anime2manga.errors import Anime2MangaError
from anime2manga.inpaint import (
    BiharmonicInpaint,
    InpaintConfig,
    InpaintMethod,
    available_inpaint_methods,
    color_and_mask,
    create_inpainter,
    fill_panorama,
    register_inpaint_method,
)


def _gradient_bgra(height: int = 48, width: int = 64) -> np.ndarray:
    """A smooth BGRA panorama with a rectangular transparent hole."""
    image = np.zeros((height, width, 4), dtype=np.uint8)
    ramp = np.linspace(0, 255, width, dtype=np.float32)
    image[..., 0] = ramp.astype(np.uint8)
    image[..., 1] = 128
    image[..., 2] = (255 - ramp).astype(np.uint8)
    image[..., 3] = 255
    image[10 : height - 10, 20 : width - 20, 3] = 0
    return image


def test_registry_lists_biharmonic():
    assert "biharmonic" in available_inpaint_methods()


def test_create_inpainter_returns_registered_instance():
    method = create_inpainter("biharmonic")
    assert isinstance(method, BiharmonicInpaint)
    assert method.name == "biharmonic"


def test_create_inpainter_rejects_unknown_name():
    with pytest.raises(Anime2MangaError, match="unknown inpaint method"):
        create_inpainter("does-not-exist")


def test_register_inpaint_method_rejects_duplicate_name():
    class Duplicate(InpaintMethod):
        name = "biharmonic"

        def inpaint(self, image, mask):  # pragma: no cover - never called
            return image

    with pytest.raises(ValueError, match="duplicate"):
        register_inpaint_method(Duplicate)


def test_register_inpaint_method_requires_name():
    class Unnamed(InpaintMethod):
        def inpaint(self, image, mask):  # pragma: no cover - never called
            return image

    with pytest.raises(ValueError, match="name"):
        register_inpaint_method(Unnamed)


def test_color_and_mask_splits_bgra_and_finds_holes():
    panorama = _gradient_bgra()
    colour, mask = color_and_mask(panorama)
    assert colour.shape == (48, 64, 3)
    assert mask is not None
    assert mask[20, 30]
    assert not mask[0, 0]
    assert mask.sum() == 28 * 24


def test_color_and_mask_ignores_fully_covered_panorama():
    opaque = np.zeros((10, 12, 4), dtype=np.uint8)
    opaque[..., 3] = 255
    colour, mask = color_and_mask(opaque)
    assert colour.shape == (10, 12, 3)
    assert mask is None


def test_fill_panorama_reconstructs_holes_on_smooth_gradient():
    panorama = _gradient_bgra()
    outcome = fill_panorama(panorama, create_inpainter("biharmonic"))
    assert outcome.filled
    assert outcome.method == "biharmonic"
    assert outcome.image.shape == (48, 64, 3)
    assert outcome.image.dtype == np.uint8

    mask = panorama[..., 3] == 0
    # Biharmonic inpainting should reproduce a linear ramp almost exactly.
    error = np.abs(
        outcome.image[mask].astype(np.float32)
        - panorama[..., :3][mask].astype(np.float32)
    )
    assert error.mean() < 8.0


def test_fill_panorama_does_nothing_when_fully_covered():
    opaque = np.zeros((10, 12, 4), dtype=np.uint8)
    opaque[..., 0] = 40
    opaque[..., 1] = 80
    opaque[..., 2] = 120
    opaque[..., 3] = 255
    outcome = fill_panorama(opaque, create_inpainter("biharmonic"))
    assert not outcome.filled
    assert outcome.method is None
    assert tuple(outcome.image[0, 0]) == (40, 80, 120)


def test_fill_panorama_promotes_grayscale_to_bgr():
    gray = np.full((8, 10), 90, dtype=np.uint8)
    outcome = fill_panorama(gray, create_inpainter("biharmonic"))
    assert not outcome.filled
    assert outcome.image.shape == (8, 10, 3)
    assert tuple(outcome.image[0, 0]) == (90, 90, 90)


def test_color_and_mask_handles_single_channel_axis():
    gray = np.full((4, 6, 1), 50, dtype=np.uint8)
    colour, mask = color_and_mask(gray)
    assert colour.shape == (4, 6, 3)
    assert mask is None


def test_color_and_mask_copies_bgr_without_mask():
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    colour, mask = color_and_mask(bgr)
    assert colour.shape == (4, 6, 3)
    assert mask is None
    bgr[0, 0, 0] = 9  # the returned image must be a copy, not a view
    assert colour[0, 0, 0] == 0


def test_color_and_mask_rejects_unsupported_channel_count():
    with pytest.raises(Anime2MangaError, match="channel count"):
        color_and_mask(np.zeros((4, 6, 2), dtype=np.uint8))


def test_fill_panorama_promotes_grayscale_method_result():
    class GrayResult(InpaintMethod):
        name = "gray-test"

        def inpaint(self, image, mask):  # pragma: no cover - trivial
            return np.zeros(image.shape[:2], dtype=np.uint8)

    outcome = fill_panorama(_gradient_bgra(), GrayResult())
    assert outcome.filled
    assert outcome.method == "gray-test"
    assert outcome.image.shape == (48, 64, 3)


def test_fill_panorama_rejects_fully_transparent_canvas():
    transparent = np.zeros((12, 16, 4), dtype=np.uint8)  # alpha all zero
    with pytest.raises(Anime2MangaError, match="no known pixels"):
        fill_panorama(transparent, create_inpainter("biharmonic"))


def test_fill_panorama_wraps_method_errors():
    class Exploding(InpaintMethod):
        name = "exploding"

        def inpaint(self, image, mask):
            raise ValueError("boom")

    with pytest.raises(Anime2MangaError, match="failed: boom"):
        fill_panorama(_gradient_bgra(), Exploding())


def test_fill_panorama_propagates_typed_method_errors():
    class Typed(InpaintMethod):
        name = "typed"

        def inpaint(self, image, mask):
            raise Anime2MangaError("already typed")

    with pytest.raises(Anime2MangaError, match="already typed"):
        fill_panorama(_gradient_bgra(), Typed())


def test_fill_panorama_falls_back_when_downscale_erases_known_pixels():
    # One known pixel and a max_pixels of 1 forces a downscale that samples the
    # only known pixel away, so the full-resolution path must take over.
    panorama = np.zeros((40, 40, 4), dtype=np.uint8)
    panorama[..., :3] = 200
    panorama[..., 3] = 0
    panorama[0, 0, 3] = 255
    outcome = fill_panorama(
        panorama, create_inpainter("biharmonic"), max_pixels=1
    )
    assert outcome.filled
    assert outcome.image.shape == (40, 40, 3)


def test_fill_panorama_rejects_bad_method_shape():
    class WrongShape(InpaintMethod):
        name = "wrong-shape"

        def inpaint(self, image, mask):
            return np.zeros((3, 3, 3), dtype=np.uint8)

    with pytest.raises(Anime2MangaError, match="unexpected shape"):
        fill_panorama(_gradient_bgra(), WrongShape())


def test_fill_panorama_downscales_and_keeps_known_pixels_full_res():
    panorama = _gradient_bgra()  # 672 masked pixels
    outcome = fill_panorama(
        panorama, create_inpainter("biharmonic"), max_pixels=100
    )
    assert outcome.filled
    # The output matches the original size even though the solve was smaller.
    assert outcome.image.shape == (48, 64, 3)
    mask = panorama[..., 3] == 0
    assert np.array_equal(outcome.image[~mask], panorama[..., :3][~mask])


def test_fill_panorama_without_cap_still_fills():
    outcome = fill_panorama(
        _gradient_bgra(), create_inpainter("biharmonic"), max_pixels=0
    )
    assert outcome.filled


def test_inpaint_config_defaults():
    cfg = InpaintConfig()
    assert cfg.enabled is True
    assert cfg.method == "biharmonic"
    assert cfg.quality == 92
    assert cfg.max_pixels == 500_000


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def _panorama_pipeline(tmp_path, **overrides):
    from anime2manga.models import Scene
    from anime2manga.pipeline import Pipeline, PipelineConfig

    config = PipelineConfig(
        input_path=tmp_path / "input.mkv",
        output_dir=tmp_path / "out",
        verbose=False,
        **overrides,
    )
    pipeline = Pipeline(config)
    return pipeline, Scene


def test_pipeline_inpaints_panorama_and_writes_jpg(tmp_path):
    pipeline, Scene = _panorama_pipeline(tmp_path)
    scene = Scene(index=1, start=0.0, end=4.0, fps=24.0)
    scene.is_panoramic = True
    scene.pan_direction = "up-left"
    png = pipeline.output_dir / "panoramas" / "scene_00000000.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png), _gradient_bgra())
    scene.panorama_path = png
    pipeline.scenes = [scene]

    pipeline._inpaint_panoramas()

    assert scene.panorama_inpainted_path is not None
    assert scene.panorama_inpainted_path.name == "scene_00000000_inpainted.jpg"
    assert scene.panorama_inpainted_path.exists()
    assert scene.inpaint_method == "biharmonic"
    loaded = cv2.imread(str(scene.panorama_inpainted_path), cv2.IMREAD_UNCHANGED)
    assert loaded is not None
    assert loaded.shape[2] == 3


def test_pipeline_inpaint_can_be_disabled(tmp_path):
    pipeline, Scene = _panorama_pipeline(tmp_path, inpaint=InpaintConfig(enabled=False))
    scene = Scene(index=1, start=0.0, end=4.0, fps=24.0)
    scene.is_panoramic = True
    png = pipeline.output_dir / "panoramas" / "scene_00000000.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png), _gradient_bgra())
    scene.panorama_path = png
    pipeline.scenes = [scene]

    pipeline._inpaint_panoramas()

    assert scene.panorama_inpainted_path is None
    assert not (png.parent / "scene_00000000_inpainted.jpg").exists()


def test_pipeline_skips_fully_covered_panorama(tmp_path):
    pipeline, Scene = _panorama_pipeline(tmp_path)
    scene = Scene(index=1, start=0.0, end=4.0, fps=24.0)
    scene.is_panoramic = True
    opaque = np.zeros((20, 30, 4), dtype=np.uint8)
    opaque[..., 3] = 255
    png = pipeline.output_dir / "panoramas" / "scene_00000000.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png), opaque)
    scene.panorama_path = png
    pipeline.scenes = [scene]

    pipeline._inpaint_panoramas()

    assert scene.panorama_inpainted_path is None


def test_pipeline_skips_unreconstructable_panorama_without_aborting(tmp_path):
    pipeline, Scene = _panorama_pipeline(tmp_path)
    scene = Scene(index=1, start=0.0, end=4.0, fps=24.0)
    scene.is_panoramic = True
    transparent = np.zeros((20, 30, 4), dtype=np.uint8)  # no known pixels
    png = pipeline.output_dir / "panoramas" / "scene_00000000.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png), transparent)
    scene.panorama_path = png
    pipeline.scenes = [scene]

    pipeline._inpaint_panoramas()  # must not raise

    assert scene.panorama_inpainted_path is None
