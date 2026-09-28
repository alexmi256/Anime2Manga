"""Step 3 - scene detection.

ffmpeg offers two practical in-process detectors:

* ``select='gt(scene,T)'`` -- a 0..1 score based on frame differences.  This is
  the classic "scene" metadata and the default here.
* ``scdet=threshold=T`` -- a mean-absolute-difference score in a 0..100 range
  (default 10).  It tends to fire on fast motion as well as hard cuts, so it
  usually needs a higher threshold.

Both are invoked as a filter and parsed from ffmpeg's diagnostic output, so we
never need to write the video to disk.  See
https://ffmpeg.org/ffmpeg-filters.html#select_002c-aselect and
https://ffmpeg.org/ffmpeg-filters.html#scdet-1.
"""

from __future__ import annotations

import re

from .errors import Anime2MangaError
from .ffmpeg_utils import run
from .models import ClipWindow, MediaInfo, Scene

DEFAULT_SELECT_THRESHOLD = 0.4
DEFAULT_SCDET_THRESHOLD = 10.0

_PTS_RE = re.compile(r"pts_time:(?P<time>-?\d+(?:\.\d+)?)")
_SCD_RE = re.compile(
    r"lavfi\.scd\.score:\s*(?P<score>-?\d+(?:\.\d+)?),\s*"
    r"lavfi\.scd\.time:\s*(?P<time>-?\d+(?:\.\d+)?)"
)


def validate_threshold(method: str, threshold: float) -> None:
    if threshold <= 0:
        raise Anime2MangaError("--scene-threshold must be greater than 0.")
    if method == "select" and threshold >= 1:
        raise Anime2MangaError(
            "--scene-threshold for the 'select' method must be between 0 and 1 "
            f"(got {threshold})."
        )


def detect_scene_times(
    media: MediaInfo,
    *,
    method: str = "select",
    threshold: float = DEFAULT_SELECT_THRESHOLD,
    start: float = 0.0,
    end: float | None = None,
    timeout: float = 3600.0,
) -> list[float]:
    """Return absolute timestamps (seconds) of detected cuts.

    ``start``/``end`` restrict the decoded range; returned times are always
    absolute because we add the seek offset back (ffmpeg resets timestamps to 0
    after ``-ss``).
    """
    validate_threshold(method, threshold)
    stop = media.duration if end is None else end
    duration = max(stop - start, 0.0)
    if duration <= 0:
        return []
    base = ["ffmpeg", "-hide_banner", "-v", "error", "-ss", f"{max(start, 0.0):.3f}"]
    base += ["-i", str(media.path), "-t", f"{duration:.3f}", "-an", "-sn"]

    if method == "scdet":
        cmd = [
            *base,
            "-vf",
            f"scdet=threshold={threshold}",
            "-f",
            "null",
            "-",
        ]
        proc = run(cmd, timeout=timeout)
        return parse_scdet_output(proc.stderr or "", offset=max(start, 0.0))
    if method == "select":
        cmd = [
            *base,
            "-filter:v",
            f"select='gt(scene,{threshold})',metadata=mode=print:file=-",
            "-f",
            "null",
            "-",
        ]
        proc = run(cmd, timeout=timeout)
        return parse_select_output(proc.stdout or "", offset=max(start, 0.0))
    raise Anime2MangaError(f"Unknown scene detection method: {method!r}")


def parse_select_output(text: str, offset: float = 0.0) -> list[float]:
    """Parse ``metadata=mode=print`` output from the ``select`` filter."""
    times: list[float] = []
    for line in text.splitlines():
        match = _PTS_RE.search(line)
        if match:
            times.append(float(match.group("time")) + offset)
    return sorted(times)


def parse_scdet_output(text: str, offset: float = 0.0) -> list[float]:
    """Parse ``scdet`` filter log lines for their scene timestamps."""
    times: list[float] = []
    for match in _SCD_RE.finditer(text):
        times.append(float(match.group("time")) + offset)
    return sorted(times)


def build_scenes(
    times: list[float],
    clip: ClipWindow,
    fps: float,
    *,
    min_scene_len: float = 0.5,
) -> list[Scene]:
    """Turn cut timestamps into contiguous scenes within ``clip``.

    Cuts outside the clip are dropped, cuts too close to the previous boundary
    are merged, and the clip's own edges become the first/last boundaries.  The
    result is guaranteed to have no gaps or overlaps.
    """
    boundaries = [clip.start]
    for time in sorted(times):
        if time <= clip.start or time >= clip.end:
            continue
        if time - boundaries[-1] < min_scene_len:
            continue
        boundaries.append(time)
    boundaries.append(clip.end)

    scenes: list[Scene] = []
    for index in range(len(boundaries) - 1):
        start, end = boundaries[index], boundaries[index + 1]
        if end - start <= 0:
            continue
        scenes.append(Scene(index=len(scenes) + 1, start=start, end=end, fps=fps))
    return scenes


def subdivide_scene(
    scene: Scene,
    media: MediaInfo,
    *,
    method: str,
    threshold: float,
    min_scene_len: float = 0.5,
) -> list[Scene]:
    """Re-detect cuts *inside* one scene at a lower threshold.

    Used when a single scene carries more subtitle text than one panel can
    reasonably hold; splitting it produces more panels (and more frames) so the
    text can be spread out.  Returns the original scene when no extra cuts are
    found.
    """
    times = detect_scene_times(
        media, method=method, threshold=threshold, start=scene.start, end=scene.end
    )
    clip = ClipWindow(start=scene.start, end=scene.end, source="subdivide")
    pieces = build_scenes(times, clip, scene.fps, min_scene_len=min_scene_len)
    for piece in pieces:
        piece.notes.append(f"subdivided from scene {scene.index}")
    return pieces


def retime_scenes_to_clip(scenes: list[Scene], clip: ClipWindow) -> list[Scene]:
    """Re-index scenes and force them to tile ``clip`` with no gaps."""
    ordered = sorted(scenes, key=lambda s: s.start)
    if not ordered:
        return []
    for index, scene in enumerate(ordered, start=1):
        scene.index = index
        if index > 1:
            scene.start = ordered[index - 2].end
    ordered[-1].end = clip.end
    return ordered
