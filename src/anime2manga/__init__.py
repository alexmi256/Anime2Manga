"""Anime2Manga: turn an anime video into a manga / storyboard document.

The pipeline is intentionally staged.  Steps 1-8 are implemented here: metadata,
subtitles, scene detection, panorama stitching, frame selection, left/right
audio focus and face detection.  Cropping, text placement and subtitle
translation remain well-documented stubs for a future developer to fill in.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
