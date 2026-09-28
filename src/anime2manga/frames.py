"""Step 6 - frame sampling and selection.

For each scene we sample frames on a uniform grid, then choose the *clearest*
frame near the subtitle timing:

* If the scene has subtitles, the target is the median of their midpoints (so a
  scene full of dialogue is represented by one well-timed image).
* Otherwise the target is the scene midpoint.
* Blur is measured with the variance of the Laplacian; the sharpest candidate
  within a small window around the target wins.  Ties are broken by proximity to
  the target time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .ffmpeg_utils import extract_frame, extract_frames_at_fps
from .models import MediaInfo, Scene, SubtitleLine

#: Half-width (seconds) of the window searched around the target time.
DEFAULT_SELECTION_WINDOW = 0.75


@dataclass
class FrameCandidate:
    time: float
    image: np.ndarray
    sharpness: float


@dataclass
class SceneSampler:
    """Extracts analysis-resolution frames for a time range."""

    media: MediaInfo
    work_dir: Path
    analysis_width: int = 960
    quality: int = 3

    def sample(
        self, start: float, end: float, fps: float, tag: str
    ) -> tuple[list[float], list[np.ndarray]]:
        out_dir = self.work_dir / "analysis" / tag
        sampled = extract_frames_at_fps(
            self.media.path,
            start,
            end,
            fps,
            out_dir,
            scale_width=self.analysis_width,
            quality=self.quality,
        )
        times: list[float] = []
        frames: list[np.ndarray] = []
        for item in sampled:
            image = cv2.imread(str(item.path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            times.append(item.time)
            frames.append(image)
        return times, frames


def sharpness(image: np.ndarray) -> float:
    """Variance-of-Laplacian focus measure (higher is sharper)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def target_time(scene: Scene, subtitles: list[SubtitleLine]) -> float:
    """Where in the scene we would like the representative frame to sit."""
    if subtitles:
        midpoints = sorted(line.midpoint for line in subtitles)
        count = len(midpoints)
        median = (
            midpoints[count // 2]
            if count % 2
            else (midpoints[count // 2 - 1] + midpoints[count // 2]) / 2.0
        )
        return min(max(median, scene.start), scene.end)
    return scene.range.midpoint


def rank_candidates(
    times: list[float],
    frames: list[np.ndarray],
    *,
    target: float,
    window: float = DEFAULT_SELECTION_WINDOW,
) -> list[FrameCandidate]:
    """Score candidates near ``target`` and return them best-first."""
    candidates: list[FrameCandidate] = []
    for time, image in zip(times, frames, strict=False):
        if abs(time - target) > window:
            continue
        candidates.append(FrameCandidate(time=time, image=image, sharpness=sharpness(image)))
    if not candidates:
        candidates = [
            FrameCandidate(time=time, image=image, sharpness=sharpness(image))
            for time, image in zip(times, frames, strict=False)
        ]
    # Sharpest first; proximity to target breaks ties.
    candidates.sort(key=lambda c: (-c.sharpness, abs(c.time - target)))
    return candidates


def select_frame(
    scene: Scene,
    subtitles: list[SubtitleLine],
    times: list[float],
    frames: list[np.ndarray],
    *,
    window: float = DEFAULT_SELECTION_WINDOW,
) -> FrameCandidate | None:
    target = target_time(scene, subtitles)
    ranked = rank_candidates(times, frames, target=target, window=window)
    return ranked[0] if ranked else None


def save_frame(
    media: MediaInfo,
    time: float,
    out_path: Path,
    *,
    scale_width: int | None = None,
    quality: int = 2,
) -> Path:
    """Extract and save the full-resolution chosen frame."""
    return extract_frame(media.path, time, out_path, scale_width=scale_width, quality=quality)


def save_image(image: np.ndarray, out_path: Path, *, quality: int = 92) -> Path:
    """Write an in-memory image (e.g. a stitched panorama) to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(out_path), image)
    return out_path
