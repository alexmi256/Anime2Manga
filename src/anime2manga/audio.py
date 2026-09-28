"""Step 7 (planned) - left/right audio focus detection.

Not implemented yet.  This stage will decide which side of a panel subtitles
should favour, based on which channel is louder during the scene.

Planned approach
----------------
* Decode only the selected scene's audio (stereo E-AC-3/AAC) with ffmpeg to raw
  f32le, e.g. ``ffmpeg -ss S -t D -i in.mkv -map 0:a:0 -ac 2 -f f32le -``.
* Compute short-time RMS per channel (or cross-correlation of L/R) on a windowed
  FFT to get a left/right dominance ratio over the *dialogue* portion.
* Return ``"left"``, ``"right"`` or ``"middle"``; store on ``Scene.audio_focus``.
* The text-placement stage (step 10) then tries that side first, subject to face
  avoidance and the 50% text-area budget.

Until this lands, :func:`detect_audio_focus` returns the neutral ``"middle"`` so
the rest of the pipeline can run.
"""

from __future__ import annotations

from .models import MediaInfo, Scene


def detect_audio_focus(media: MediaInfo, scene: Scene) -> str:
    """Return ``"left"``, ``"middle"`` or ``"right"`` for the scene.

    TODO(step 7): implement RMS balance across the stereo channels as described
    in the module docstring.  ``media`` gives access to the path and duration;
    scenes carry their own time range.
    """
    _ = (media, scene)
    return "middle"
