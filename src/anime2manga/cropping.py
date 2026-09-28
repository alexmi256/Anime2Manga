"""Step 9 (planned) - cropping a frame into a square 1x1 panel.

Not implemented yet.  Input 16:9 frames must be cropped to a square for a
single-cell panel.  The crop should be centred on characters/faces (from step 8)
rather than the geometric centre.

Planned approach
----------------
* Work out the largest square that fits the frame height (e.g. 1080x1080 for a
  1920x1080 source).
* Slide the square horizontally (and vertically if ever needed) to maximise face
  coverage: weight each candidate by the fraction of face area it contains,
  penalise crops that cut a face in half, and keep text-free margins.
* Return the best :class:`CropOption` with a score; when no faces are found,
  fall back to a centred crop.
* A panoramic panel is *never* cropped to 1x1: it keeps its wide/tall aspect and
  spans 2-3 grid columns (or rows).

Until implemented, :func:`plan_crop` returns a centred square crop as a sensible
default.
"""

from __future__ import annotations

from .models import CropOption, FaceBox


def plan_crop(frame_width: int, frame_height: int, faces: list[FaceBox]) -> CropOption:
    """Choose a square crop for ``frame_width`` x ``frame_height``.

    TODO(step 9): implement face-aware crop selection per the module docstring.
    """
    side = min(frame_width, frame_height)
    x = (frame_width - side) // 2
    y = (frame_height - side) // 2
    return CropOption(x=x, y=y, width=side, height=side, keeps_faces=bool(faces))
