"""Step 10 (planned) - subtitle/text placement inside a panel.

Not implemented yet.  This stage decides where each panel's subtitle text goes so
that it never covers a face and never occupies more than 50% of the panel.

Planned approach
----------------
* Start from the scene's ``audio_focus`` (step 7): if dialogue is left- or
  right-heavy, try that side first.
* Load ``Scene.faces`` (step 8) and score candidate text boxes by face overlap
  (must be zero) and by text-area budget (<= 50% of the panel).
* If the scene carries many subtitle lines, prefer a wider panel (2-3 grid
  columns) instead of shrinking text, or request a scene split upstream.
* Emit the chosen regions as :class:`TextPlacement`; a later renderer draws the
  bubbles.  Nothing is rendered yet - the markdown report just lists the text.

Until implemented, :func:`plan_text_placement` returns an ``"auto"`` placement
with no regions.
"""

from __future__ import annotations

from .models import FaceBox, Scene, SubtitleLine, TextPlacement


def plan_text_placement(
    scene: Scene,
    subtitles: list[SubtitleLine],
    faces: list[FaceBox],
    panel_size: tuple[int, int],
) -> TextPlacement:
    """Plan where the scene's subtitles should be rendered.

    TODO(step 10): implement the audio/face-aware placement and the 50% text
    budget described in the module docstring.
    """
    _ = (subtitles, faces, panel_size)
    return TextPlacement(side=scene.audio_focus)
