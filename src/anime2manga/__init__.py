"""Anime2Manga: turn an anime video into a manga / storyboard document.

The pipeline is intentionally staged.  Steps 1-6 are implemented here; later
stages (audio focus, face detection, cropping decisions and text placement) are
present as well-documented stubs so a future developer can fill them in.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
