"""Step 5 - timeline integrity.

Once pan detection has retimed scenes (a pan can spill over a boundary, so the
earlier scene is extended and the next one starts later), we must guarantee the
scenes still tile the selected clip exactly: no gaps, no overlaps, monotonic
times, all inside the clip.  Anything else would mean frames are missing or
duplicated in the manga.
"""

from __future__ import annotations

from itertools import pairwise

from .models import ClipWindow, Scene

#: Scenes shorter than this (seconds) are considered noise after retiming.
MIN_SCENE_LEN = 0.25


def coverage(scenes: list[Scene], clip: ClipWindow) -> float:
    total = sum(scene.duration for scene in scenes)
    return total / clip.duration if clip.duration > 0 else 0.0


def validate_scenes(
    scenes: list[Scene],
    clip: ClipWindow,
    *,
    tolerance: float = 0.05,
) -> list[str]:
    """Return a list of human-readable problems (empty means healthy)."""
    problems: list[str] = []
    if not scenes:
        return ["No scenes were produced."]
    ordered = sorted(scenes, key=lambda s: s.start)
    if ordered[0].start > clip.start + tolerance:
        problems.append(
            f"Gap at start: first scene begins at {ordered[0].start:.3f}s "
            f"but clip starts at {clip.start:.3f}s."
        )
    if ordered[-1].end < clip.end - tolerance:
        problems.append(
            f"Gap at end: last scene ends at {ordered[-1].end:.3f}s "
            f"but clip ends at {clip.end:.3f}s."
        )
    for previous, current in pairwise(ordered):
        delta = current.start - previous.end
        if delta > tolerance:
            problems.append(
                f"Gap between scene {previous.index} and {current.index}: "
                f"{delta:.3f}s."
            )
        elif delta < -tolerance:
            problems.append(
                f"Overlap between scene {previous.index} and {current.index}: "
                f"{-delta:.3f}s."
            )
    for scene in ordered:
        if scene.end - scene.start <= 0:
            problems.append(f"Scene {scene.index} has non-positive duration.")
    if coverage(scenes, clip) > 1.0 + tolerance / max(clip.duration, 1e-9):
        problems.append("Scenes cover more than the clip (overlaps).")
    return problems


def extend_scene_for_pan(
    scenes: list[Scene],
    scene: Scene,
    extension: float,
    *,
    min_scene_len: float = MIN_SCENE_LEN,
) -> float:
    """Push ``scene``'s end forward by ``extension``, pulling the next scene's
    start with it.  Returns the extension actually applied.

    This keeps the timeline gapless: the earlier (panoramic) scene swallows the
    first part of the next scene.  A next scene reduced to nothing is dropped.
    """
    ordered = sorted(scenes, key=lambda s: s.start)
    position = ordered.index(scene)
    if position >= len(ordered) - 1:
        return 0.0
    following = ordered[position + 1]
    applied = max(0.0, min(extension, following.duration))
    if applied <= 0:
        return 0.0
    scene.end += applied
    following.start = scene.end
    if following.duration <= min_scene_len:
        scenes.remove(following)
    reindex(scenes)
    return applied


def reindex(scenes: list[Scene]) -> None:
    scenes.sort(key=lambda s: s.start)
    for index, scene in enumerate(scenes, start=1):
        scene.index = index
