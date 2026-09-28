"""Step 1 - container metadata: duration, tracks and intro/credits chapters.

Matroska (``.mkv``) stores named chapters, and releases routinely name the
opening and ending chapters ("Opening", "Ending", "Preview", ...).  ffmpeg
exposes these through ``ffprobe -show_chapters``.  Some releases instead use
tags or segment info that ffmpeg does not surface, in which case we fall back to
the user's ``--start-at`` / ``--end-at`` options.
"""

from __future__ import annotations

import re
from pathlib import Path

from .errors import Anime2MangaError
from .ffmpeg_utils import ffprobe_json, parse_rate
from .models import (
    BITMAP_SUBTITLE_CODECS,
    CREDITS_KEYWORDS,
    INTRO_KEYWORDS,
    Chapter,
    ClipWindow,
    MediaInfo,
    SubtitleTrack,
    TimeRange,
)


def parse_timestamp(value: str) -> float:
    """Parse ``SS``, ``MM:SS``, ``HH:MM:SS`` or ``...:SS.mmm`` into seconds."""
    text = value.strip()
    if not text:
        raise Anime2MangaError(f"Empty timestamp: {value!r}")
    parts = text.split(":")
    if len(parts) > 3:
        raise Anime2MangaError(f"Invalid timestamp: {value!r}")
    total = 0.0
    for part in parts:
        try:
            total = total * 60.0 + float(part)
        except ValueError as exc:
            raise Anime2MangaError(f"Invalid timestamp: {value!r}") from exc
    return total


def probe_media(path: Path) -> MediaInfo:
    """Probe streams, chapters and duration for ``path``."""
    stream_data = ffprobe_json(
        [
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate"
            ":stream_tags=language,title"
            ":stream_disposition=default",
        ],
        path,
    )
    format_data = ffprobe_json(["format=duration,format_name"], path)
    chapter_data = ffprobe_json(["chapter=id,time_base,start,start_time,end,end_time"], path)

    streams = stream_data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise Anime2MangaError(f"No video stream found in {path}")

    fps = parse_rate(video.get("avg_frame_rate")) or parse_rate(video.get("r_frame_rate")) or 24.0
    subtitle_tracks: list[SubtitleTrack] = []
    for stream in streams:
        if stream.get("codec_type") != "subtitle":
            continue
        tags = stream.get("tags", {}) or {}
        disposition = stream.get("disposition", {}) or {}
        codec = stream.get("codec_name", "")
        subtitle_tracks.append(
            SubtitleTrack(
                index=int(stream.get("index", -1)),
                codec=codec,
                language=tags.get("language", ""),
                title=tags.get("title", ""),
                is_bitmap=codec in BITMAP_SUBTITLE_CODECS,
                is_default=bool(disposition.get("default", 0)),
            )
        )

    duration = float(format_data.get("format", {}).get("duration") or 0.0)
    chapters = _parse_chapters(chapter_data.get("chapters", []))
    intro, credits = classify_chapters(chapters, duration)

    return MediaInfo(
        path=path,
        duration=duration,
        fps=fps,
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        subtitle_tracks=subtitle_tracks,
        chapters=chapters,
        intro=intro,
        credits=credits,
    )


def _parse_chapters(raw_chapters: list[dict]) -> list[Chapter]:
    chapters: list[Chapter] = []
    for raw in raw_chapters:
        tags = raw.get("tags", {}) or {}
        title = tags.get("title", "") or ""
        try:
            start = float(raw.get("start_time") or raw.get("start") or 0.0)
            end = float(raw.get("end_time") or raw.get("end") or start)
        except (TypeError, ValueError):
            continue
        chapters.append(Chapter(index=len(chapters), title=title, start=start, end=end))
    return chapters


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _matches(title: str, keywords: tuple[str, ...]) -> bool:
    normalized = _normalize(title)
    if not normalized:
        return False
    words = set(normalized.split())
    for keyword in keywords:
        if keyword in words:
            return True
        # Multi-word or distinctive keywords may appear as a substring.
        if len(keyword) >= 4 and keyword in normalized:
            return True
    return False


def classify_chapters(
    chapters: list[Chapter], duration: float
) -> tuple[TimeRange | None, TimeRange | None]:
    """Split chapters into an intro range and a credits range.

    Opening chapters live in the first half of the episode and credit chapters
    in the second half; anything else (a "Part A" chapter, for example) is
    ignored.  The spans are merged so "Opening" + "Creditless Opening" becomes a
    single intro range.
    """
    if not chapters or duration <= 0:
        return None, None
    half = duration / 2.0

    intro_spans: list[TimeRange] = []
    credit_spans: list[TimeRange] = []
    for chapter in chapters:
        if _matches(chapter.title, INTRO_KEYWORDS) and chapter.start < half:
            intro_spans.append(chapter.range)
        if _matches(chapter.title, CREDITS_KEYWORDS) and chapter.start >= half:
            credit_spans.append(chapter.range)

    def merge(spans: list[TimeRange]) -> TimeRange | None:
        if not spans:
            return None
        return TimeRange(
            start=min(span.start for span in spans),
            end=max(span.end for span in spans),
        )

    return merge(intro_spans), merge(credit_spans)


def resolve_clip_window(
    media: MediaInfo,
    *,
    start_at: str | None = None,
    end_at: str | None = None,
) -> ClipWindow:
    """Combine chapter metadata and CLI overrides into the region to convert.

    Precedence: explicit CLI values win; otherwise detected intro/credits
    chapters trim the head/tail; otherwise the whole file is used.
    """
    duration = media.duration
    start = 0.0
    end = duration
    excluded: list[TimeRange] = []
    sources: list[str] = []

    if media.intro is not None:
        start = max(start, media.intro.end)
        excluded.append(media.intro)
        sources.append("intro-chapter")
    if media.credits is not None:
        end = min(end, media.credits.start)
        excluded.append(media.credits)
        sources.append("credits-chapter")

    if start_at is not None:
        start = parse_timestamp(start_at)
        sources.append("--start-at")
    if end_at is not None:
        end = parse_timestamp(end_at)
        sources.append("--end-at")

    start = max(0.0, min(start, duration))
    end = max(0.0, min(end, duration))
    if end - start <= 0:
        raise Anime2MangaError(
            f"Clip window is empty (start={start:.2f}s, end={end:.2f}s). "
            "Check --start-at / --end-at."
        )
    source = ", ".join(sources) if sources else "full video"
    return ClipWindow(start=start, end=end, source=source, excluded=excluded)
