"""Core data structures shared by the pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class SceneMethod(StrEnum):
    """Which ffmpeg scene-detection strategy to use."""

    #: ``select='gt(scene,T)'`` - score is in the 0..1 range.
    SELECT = "select"
    #: ``scdet=threshold=T`` - score is a 0..100 mean absolute difference.
    SCDET = "scdet"


@dataclass(frozen=True)
class TimeRange:
    """A half-open time span ``[start, end)`` in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end

    def overlaps(self, other: TimeRange) -> bool:
        return self.start < other.end and other.start < self.end

    def intersection(self, other: TimeRange) -> TimeRange | None:
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return TimeRange(start, end) if start < end else None

    def clamp(self, lo: float, hi: float) -> TimeRange:
        return TimeRange(min(max(self.start, lo), hi), min(max(self.end, lo), hi))


#: Subtitle codecs that are images, not text, and therefore need OCR.
BITMAP_SUBTITLE_CODECS = frozenset(
    {
        "hdmv_pgs_subtitle",
        "dvd_subtitle",
        "dvb_subtitle",
        "xsub",
        "pgssub",
    }
)


@dataclass(frozen=True)
class Chapter:
    """A chapter/marker embedded in the container (Matroska chapters)."""

    index: int
    title: str
    start: float
    end: float

    @property
    def range(self) -> TimeRange:
        return TimeRange(self.start, self.end)


#: Chapter titles that commonly mark an opening/intro.
INTRO_KEYWORDS: tuple[str, ...] = (
    "intro",
    "opening",
    "op ",
    "op1",
    "op2",
    "opening theme",
    "op",
    "creditless opening",
    "ncop",
)
#: Chapter titles that commonly mark ending/credits/preview.
CREDITS_KEYWORDS: tuple[str, ...] = (
    "ending",
    "credits",
    "outro",
    "ed ",
    "ed1",
    "ed2",
    "ending theme",
    "ed",
    "preview",
    "next episode",
    "next ep",
    "nced",
)


@dataclass(frozen=True)
class SubtitleTrack:
    """A subtitle stream discovered in the input."""

    index: int
    codec: str
    language: str
    title: str
    is_bitmap: bool = False
    is_default: bool = False

    @property
    def label(self) -> str:
        name = self.title.strip() or self.language.strip() or "unknown"
        return f"#{self.index} {self.language or 'und'} - {name}"


@dataclass(frozen=True)
class MediaInfo:
    """Everything we probe up front about the input file."""

    path: Path
    duration: float
    fps: float
    width: int
    height: int
    subtitle_tracks: list[SubtitleTrack]
    chapters: list[Chapter]
    intro: TimeRange | None = None
    credits: TimeRange | None = None


@dataclass
class SubtitleLine:
    """A single subtitle cue with markup stripped."""

    index: int
    start: float
    end: float
    text: str
    track_index: int = -1
    language: str = ""

    @property
    def range(self) -> TimeRange:
        return TimeRange(self.start, self.end)

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass
class ClipWindow:
    """The region of the source video that should actually become a manga.

    ``excluded`` records intro/credits spans that were dropped so later stages
    can report *why* a chunk of video is missing.  ``source`` is a short human
    string such as ``"cli"`` or ``"chapters"``.
    """

    start: float
    end: float
    source: str
    excluded: list[TimeRange] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def range(self) -> TimeRange:
        return TimeRange(self.start, self.end)


@dataclass
class Scene:
    """A detected scene and everything the pipeline learns about it.

    Later stages attach their results here.  Fields whose values are only
    produced by planned (stubbed) stages are marked below.
    """

    index: int
    start: float
    end: float
    fps: float

    # --- Step 4: panorama -------------------------------------------------
    is_panoramic: bool = False
    pan_direction: str | None = None
    panorama_path: Path | None = None
    panorama_size: tuple[int, int] | None = None
    pan_shift: tuple[float, float] | None = None
    #: Absolute time span of the panning segment (not the whole scene).
    pan_start: float | None = None
    pan_end: float | None = None

    # --- Step 6: frame selection -----------------------------------------
    frame_path: Path | None = None
    frame_time: float | None = None
    frame_size: tuple[int, int] | None = None
    blur_score: float | None = None
    subtitles: list[SubtitleLine] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # --- Planned stages (stubs); kept typed so downstream code can rely on
    #     them without a schema migration. -----------------------------------
    audio_focus: str = "middle"  # step 7: "left" | "middle" | "right"
    faces: list[FaceBox] = field(default_factory=list)  # step 8
    crop: CropOption | None = None  # step 9
    text_placement: TextPlacement | None = None  # step 10

    @property
    def range(self) -> TimeRange:
        return TimeRange(self.start, self.end)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def frame_at(self, t: float) -> int:
        return round(t * self.fps)

    @property
    def start_frame(self) -> int:
        return self.frame_at(self.start)

    @property
    def end_frame(self) -> int:
        return self.frame_at(self.end)


@dataclass(frozen=True)
class FaceBox:
    """A face bounding box in full-resolution frame pixels (step 8).

    ``confidence`` is the detector's score for the box (``0..1``); it is a
    relative strength, not a calibrated probability.
    """

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class CropOption:
    """A candidate crop.  Produced by step 9 (not yet implemented)."""

    x: int
    y: int
    width: int
    height: int
    keeps_faces: bool = False
    score: float = 0.0


@dataclass(frozen=True)
class TextPlacement:
    """Where a scene's subtitles should sit.  Step 10 (not implemented)."""

    side: str = "auto"
    regions: tuple[tuple[int, int, int, int], ...] = ()


@dataclass
class PanResult:
    """Output of pan detection for one scene."""

    scene_index: int
    detected: bool
    direction: str | None
    cumulative_shift: tuple[float, float]
    consistency: float
    mean_response: float
    canvas_width: int = 0
    canvas_height: int = 0
    #: Per-sampled-frame cumulative content shift, used for stitching.
    offsets: list[tuple[float, float]] = field(default_factory=list)
    sample_times: list[float] = field(default_factory=list)
    frames: list[Any] = field(default_factory=list)  # numpy arrays
    #: Absolute start/end time of the detected pan segment.
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class PipelineResult:
    """Final artifact handed to the report writer."""

    media: MediaInfo
    clip: ClipWindow
    subtitle_track: SubtitleTrack
    subtitle_lines: list[SubtitleLine]
    scenes: list[Scene]
    output_dir: Path
