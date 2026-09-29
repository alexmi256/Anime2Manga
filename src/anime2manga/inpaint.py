"""Content-aware infill for stitched panoramas.

When a camera pans diagonally, the stitched panorama is not a rectangle: the
corners and edges that no sampled frame covered stay transparent in the BGRA
PNG written by :mod:`anime2manga.panorama`.  Those transparent pixels are
exactly a mask of what is missing, so this module turns the alpha channel into a
mask and asks an *inpainting* method to reconstruct the missing content.  The
filled result is flattened to BGR and written alongside the original PNG as a
JPEG.

Pluggable methods
-----------------
Inpainting is deliberately pluggable because different algorithms trade quality
against compute time.  Each algorithm subclasses :class:`InpaintMethod` and
decorates itself with :func:`register_inpaint_method`::

    @register_inpaint_method
    class MyInpaint(InpaintMethod):
        name = "mine"

        def inpaint(self, image, mask):
            ...

The registry is the single source of truth, so the CLI choices, the report and
:func:`create_inpainter` all pick up a new method automatically.  The default is
``biharmonic``, which earlier research picked as the best quality/compute
trade-off for this pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import cv2
import numpy as np

from .errors import Anime2MangaError


class InpaintMethod(ABC):
    """Base class for a content-aware infill algorithm.

    Implementations receive an already-flattened BGR image plus a boolean mask of
    the pixels that need reconstructing, and return a BGR image of the same
    size.  Keeping the alpha handling in this module means every method only has
    to care about the actual reconstruction.
    """

    #: Short identifier used in config, on the CLI and in the report.
    name: ClassVar[str]

    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Reconstruct the ``mask`` pixels of ``image``.

        Parameters
        ----------
        image:
            ``H x W x 3`` BGR ``uint8`` image; pixels outside ``mask`` are valid.
        mask:
            ``H x W`` boolean array; ``True`` marks pixels to reconstruct.

        Returns
        -------
        numpy.ndarray
            ``H x W x 3`` BGR ``uint8`` image with the masked pixels filled in.
        """


_METHODS: dict[str, type[InpaintMethod]] = {}


def register_inpaint_method(cls: type[InpaintMethod]) -> type[InpaintMethod]:
    """Class decorator that adds ``cls`` to the inpainting registry."""
    name = getattr(cls, "name", "")
    if not name:
        raise ValueError("InpaintMethod subclasses must define a non-empty 'name'")
    if name in _METHODS:
        raise ValueError(f"duplicate inpaint method name: {name!r}")
    _METHODS[name] = cls
    return cls


def available_inpaint_methods() -> tuple[str, ...]:
    """Return the registered method names, sorted for stable CLI choices."""
    return tuple(sorted(_METHODS))


def create_inpainter(name: str) -> InpaintMethod:
    """Instantiate a registered inpainting method by name."""
    try:
        cls = _METHODS[name]
    except KeyError:
        choices = ", ".join(available_inpaint_methods()) or "(none)"
        raise Anime2MangaError(
            f"unknown inpaint method {name!r}; available: {choices}"
        ) from None
    return cls()


@register_inpaint_method
class BiharmonicInpaint(InpaintMethod):
    """Biharmonic (smooth) inpainting from :mod:`skimage.restoration`.

    This solves the biharmonic equation over the masked region, producing a
    smooth extension of the surrounding colours.  It is the best result for the
    computation time seen in the earlier evaluation, so it is the default.
    """

    name = "biharmonic"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Imported lazily so the rest of the pipeline still works, and fails
        # with a clear message, if scikit-image is not installed.
        try:
            from skimage.restoration import inpaint as sk_inpaint
        except ImportError as error:  # pragma: no cover - depends on environment
            raise Anime2MangaError(
                "the 'biharmonic' inpaint method needs scikit-image; "
                "install it with 'pip install scikit-image'"
            ) from error

        # skimage expects a float image; the equation is linear so the 0..1
        # scaling is arbitrary but keeps the sparse solve well conditioned.
        colour = np.asarray(image, dtype=np.float32) / 255.0
        filled = sk_inpaint.inpaint_biharmonic(colour, mask, channel_axis=-1)
        return np.clip(filled * 255.0, 0, 255).astype(np.uint8)


#: Default mask-size cap: how many masked pixels a method solves at once.
#: Chosen to keep a biharmonic solve to a few seconds on ordinary hardware.
DEFAULT_MAX_PIXELS = 500_000


@dataclass
class InpaintConfig:
    """Tunable panorama-infill settings (all exposed on the CLI)."""

    #: Fill transparent panorama holes at all.
    enabled: bool = True
    #: Registered method name used to reconstruct the holes.
    method: str = "biharmonic"
    #: JPEG quality for the filled panorama written next to the PNG.
    quality: int = 92
    #: At most this many masked pixels are handed to the method at once.
    #: Reconstructing a huge masked region is what makes biharmonic slow and
    #: memory-hungry, so a larger hole is solved on a downscaled copy and the
    #: synthetic fill is composited back over full-resolution known pixels.
    #: ``None`` or ``0`` disables the cap.
    max_pixels: int | None = DEFAULT_MAX_PIXELS


@dataclass(frozen=True)
class InpaintOutcome:
    """The result of infilling one panorama."""

    #: BGR ``uint8`` image with any holes reconstructed.
    image: np.ndarray
    #: Number of pixels the method reconstructed (``0`` when nothing was missing).
    filled_pixels: int
    #: Name of the method that ran, or ``None`` when there was nothing to fill.
    method: str | None

    @property
    def filled(self) -> bool:
        """True when at least one transparent pixel was reconstructed."""
        return self.filled_pixels > 0


def color_and_mask(panorama: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Split a stitched panorama into a BGR image and an optional hole mask.

    Panoramas are BGRA when colour and inpainting is needed; a fully covered
    panorama, or a grayscale one, has no mask.  ``None`` is returned instead of
    an all-False mask so callers can skip the work entirely.
    """
    if panorama.ndim == 2:
        return cv2.cvtColor(panorama, cv2.COLOR_GRAY2BGR), None
    channels = panorama.shape[2]
    if channels == 1:
        return cv2.cvtColor(panorama[..., 0], cv2.COLOR_GRAY2BGR), None
    if channels == 3:
        return panorama.copy(), None
    if channels == 4:
        mask = panorama[..., 3] == 0
        return panorama[..., :3].copy(), (mask if mask.any() else None)
    raise Anime2MangaError(f"unsupported panorama channel count: {channels}")


def _downscale_factor(masked: int, max_pixels: int | None) -> float:
    """Factor that brings ``masked`` pixels within ``max_pixels`` (1.0 = none).

    The solve cost tracks the number of masked pixels, not the canvas size, so
    the factor is derived from the hole area: ``masked * factor**2`` is roughly
    the number of unknowns after resizing.
    """
    if not max_pixels or max_pixels <= 0 or masked <= max_pixels:
        return 1.0
    return (max_pixels / float(masked)) ** 0.5


def _resize_for_inpaint(
    colour: np.ndarray, mask: np.ndarray, factor: float
) -> tuple[np.ndarray, np.ndarray]:
    """Downscale an image and its mask together for a cheaper solve."""
    height, width = colour.shape[:2]
    new_size = (max(1, round(width * factor)), max(1, round(height * factor)))
    small_colour = cv2.resize(colour, new_size, interpolation=cv2.INTER_AREA)
    # Resizing may shift the mask boundary by a pixel, which is harmless for a
    # smooth fill.  ``fill_panorama`` falls back to full resolution if the
    # downscale leaves no known pixel at all.
    small_mask = cv2.resize(
        mask.astype(np.uint8), new_size, interpolation=cv2.INTER_AREA
    ).astype(bool)
    return small_colour, small_mask


def _call_method(
    method: InpaintMethod, image: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Run ``method`` and normalise/validate its result.

    Third-party errors and malformed results are wrapped so a broken method
    cannot abort a whole pipeline run with an opaque traceback.
    """
    try:
        result = np.asarray(method.inpaint(image, mask))
    except Anime2MangaError:
        raise
    except Exception as error:  # surface any method failure with a clear message
        raise Anime2MangaError(
            f"inpaint method {method.name!r} failed: {error}"
        ) from error
    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    if result.ndim != 3 or result.shape[2] != 3 or result.shape[:2] != image.shape[:2]:
        raise Anime2MangaError(
            f"inpaint method {method.name!r} returned an unexpected shape "
            f"{result.shape} (expected {image.shape})"
        )
    return np.clip(result, 0, 255).astype(np.uint8)


def fill_panorama(
    panorama: np.ndarray,
    method: InpaintMethod,
    *,
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
) -> InpaintOutcome:
    """Fill the transparent holes of ``panorama`` using ``method``.

    Accepts the BGRA panorama returned by
    :func:`anime2manga.panorama.stitch` and always returns a BGR image suitable
    for JPEG output.  When the panorama has no transparent pixels the image is
    merely converted to BGR and ``filled_pixels`` is ``0``.

    ``max_pixels`` caps how many masked pixels the method has to solve at once:
    a larger hole is downscaled, inpainted and the synthetic fill is composited
    back over the full-resolution known pixels.  This keeps a large diagonal
    canvas from turning into a minutes-long, memory-hungry sparse solve.
    """
    colour, mask = color_and_mask(panorama)
    if mask is None:
        return InpaintOutcome(image=colour, filled_pixels=0, method=None)
    if bool(mask.all()):
        raise Anime2MangaError(
            "panorama has no known pixels to reconstruct from"
        )

    masked = int(np.count_nonzero(mask))
    factor = _downscale_factor(masked, max_pixels)
    if factor < 1.0:
        small_colour, small_mask = _resize_for_inpaint(colour, mask, factor)
        if not bool(small_mask.all()):
            filled = _call_method(method, small_colour, small_mask)
            filled = cv2.resize(
                filled,
                (colour.shape[1], colour.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            result = colour.copy()
            result[mask] = filled[mask]
            return InpaintOutcome(
                image=result,
                filled_pixels=masked,
                method=method.name,
            )
        # The downscale swallowed the last known pixel; fall back to full res.

    return InpaintOutcome(
        image=_call_method(method, colour, mask),
        filled_pixels=masked,
        method=method.name,
    )
