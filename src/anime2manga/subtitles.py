"""Step 2 - subtitle track selection, extraction and parsing.

Subtitles are the backbone of the manga: every cue in the selected track must
end up attached to a panel.  We support text formats (SRT/ASS/SSA and the many
text codecs ffmpeg can transcode from) and explicitly reject bitmap formats such
as PGS/VobSub, which would require OCR.

Translation is a planned feature: all language handling goes through
:func:`select_subtitle_track`, and :mod:`anime2manga.translation` defines the
seam where a translation step will later be inserted.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from .errors import (
    NoSubtitlesError,
    SubtitleTrackNotFoundError,
    UnsupportedSubtitleError,
)
from .ffmpeg_utils import run
from .models import (
    BITMAP_SUBTITLE_CODECS,
    MediaInfo,
    Scene,
    SubtitleLine,
    SubtitleTrack,
)

#: Map common ISO 639-1 codes onto the 639-2 codes ffmpeg/Matroska usually use.
LANGUAGE_ALIASES: dict[str, str] = {
    "en": "eng",
    "ja": "jpn",
    "jp": "jpn",
    "pt": "por",
    "pt-br": "por",
    "es": "spa",
    "fr": "fre",
    "de": "ger",
    "zh": "chi",
    "ko": "kor",
    "it": "ita",
}

_ASS_TAG_RE = re.compile(r"\{[^}]*\}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SRT_TIME_RE = re.compile(
    r"(?P<sh>\d+):(?P<sm>\d{1,2}):(?P<ss>\d{1,2})[,.](?P<sms>\d{1,3})"
)
_ASS_TIME_RE = re.compile(r"(?P<sh>\d+):(?P<sm>\d{1,2}):(?P<ss>\d{1,2})[.](?P<sms>\d{1,2})")


def _normalize_language(language: str) -> str:
    key = language.strip().lower()
    return LANGUAGE_ALIASES.get(key, key)


def list_tracks(media: MediaInfo) -> list[SubtitleTrack]:
    return list(media.subtitle_tracks)


def select_subtitle_track(
    media: MediaInfo,
    *,
    language: str = "eng",
    track_id: int | None = None,
) -> SubtitleTrack:
    """Pick the subtitle track to use, with actionable errors.

    If ``track_id`` is given it wins outright.  Otherwise we match ``language``
    (English by default), preferring the stream flagged ``default`` and
    de-prioritising closed-caption/SDH tracks.
    """
    tracks = list(media.subtitle_tracks)
    if track_id is not None:
        for track in tracks:
            if track.index == track_id:
                return track
        available = _format_available(tracks)
        raise SubtitleTrackNotFoundError(
            f"No subtitle track with id {track_id}. Available tracks:\n{available}",
            available=[t.label for t in tracks],
        )
    if not tracks:
        raise NoSubtitlesError(
            f"{media.path} contains no subtitle tracks. Provide a video with "
            "subtitles (SRT/ASS), or supply one separately."
        )

    wanted = _normalize_language(language)
    matches = [t for t in tracks if _normalize_language(t.language) == wanted]
    if not matches:
        available = _format_available(tracks)
        raise SubtitleTrackNotFoundError(
            f"No subtitle track for language {language!r}. "
            f"Use --subtitle-language with one of the available languages, or "
            f"--subtitle-track with a specific id.\nAvailable tracks:\n{available}",
            available=[t.label for t in tracks],
        )

    def rank(track: SubtitleTrack) -> tuple[int, int, int]:
        title = track.title.lower()
        is_cc = int("cc" in title or "sdh" in title or "hearing" in title)
        return (is_cc, 0 if track.is_default else 1, track.index)

    return sorted(matches, key=rank)[0]


def _format_available(tracks: list[SubtitleTrack]) -> str:
    if not tracks:
        return "  (none)"
    return "\n".join(f"  {track.label}" for track in tracks)


def ensure_supported(track: SubtitleTrack) -> None:
    """Reject bitmap subtitle formats that would require OCR."""
    if track.is_bitmap or track.codec in BITMAP_SUBTITLE_CODECS:
        raise UnsupportedSubtitleError(
            f"Subtitle track {track.label} uses bitmap format {track.codec!r} "
            "(PGS/VobSub). OCR extraction is not supported. Remux a text-based "
            "subtitle track or supply an SRT/ASS file."
        )


def extract_subtitles(
    media: MediaInfo,
    track: SubtitleTrack,
    out_dir: Path,
    *,
    extension: str = "srt",
) -> Path:
    """Extract ``track`` to ``out_dir`` as a text subtitle file."""
    ensure_supported(track)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_normalize_language(track.language) or 'und'}_{track.index}"
    out_path = out_dir / f"{stem}.{extension}"
    codec = "srt" if extension.lower() == "srt" else "ass"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-i",
        str(media.path),
        "-map",
        f"0:{track.index}",
        "-c:s",
        codec,
        str(out_path),
    ]
    run(cmd)
    return out_path


def clean_text(text: str) -> str:
    """Strip HTML/ASS markup and normalise whitespace in a cue."""
    text = _ASS_TAG_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = text.replace("\\N", " ").replace("\\n", " ")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_srt_time(value: str) -> float:
    match = _SRT_TIME_RE.search(value.strip())
    if not match:
        raise ValueError(f"Bad SRT timestamp: {value!r}")
    ms = match.group("sms").ljust(3, "0")
    return (
        int(match.group("sh")) * 3600
        + int(match.group("sm")) * 60
        + int(match.group("ss"))
        + int(ms) / 1000.0
    )


def parse_srt(content: str, *, track_index: int = -1, language: str = "") -> list[SubtitleLine]:
    """Parse SRT text into cues."""
    content = content.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = content.split("\n")
    cues: list[SubtitleLine] = []
    i = 0
    index = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if "-->" in line:
            time_line = line
        elif i + 1 < len(lines) and "-->" in lines[i + 1]:
            # A numeric index line, then the timestamp line.
            i += 1
            time_line = lines[i].strip()
        else:
            i += 1
            continue
        try:
            start_raw, _, end_raw = time_line.partition("-->")
            start = _parse_srt_time(start_raw)
            end = _parse_srt_time(end_raw)
        except ValueError:
            i += 1
            continue
        i += 1
        body: list[str] = []
        while i < len(lines) and lines[i].strip():
            body.append(lines[i].strip())
            i += 1
        text = clean_text(" ".join(body))
        if text:
            index += 1
            cues.append(
                SubtitleLine(
                    index=index,
                    start=start,
                    end=end,
                    text=text,
                    track_index=track_index,
                    language=language,
                )
            )
    return cues


def _parse_ass_time(value: str) -> float:
    match = _ASS_TIME_RE.search(value.strip())
    if not match:
        raise ValueError(f"Bad ASS timestamp: {value!r}")
    cs = match.group("sms").ljust(2, "0")
    return (
        int(match.group("sh")) * 3600
        + int(match.group("sm")) * 60
        + int(match.group("ss"))
        + int(cs) / 100.0
    )


def parse_ass(content: str, *, track_index: int = -1, language: str = "") -> list[SubtitleLine]:
    """Parse ASS/SSA ``[Events]`` dialogue lines into cues."""
    content = content.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    in_events = False
    fields: list[str] = []
    cues: list[SubtitleLine] = []
    for raw in content.split("\n"):
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            in_events = line.lower() == "[events]"
            continue
        if not in_events:
            continue
        if line.lower().startswith("format:"):
            fields = [part.strip().lower() for part in line.split(":", 1)[1].split(",")]
            continue
        if not line.lower().startswith("dialogue:"):
            continue
        payload = line.split(":", 1)[1]
        if fields:
            values = payload.split(",", len(fields) - 1)
        else:
            values = payload.split(",", 9)
            fields = [
                "layer",
                "start",
                "end",
                "style",
                "name",
                "marginl",
                "marginr",
                "marginv",
                "effect",
                "text",
            ]
        record = dict(zip(fields, values, strict=False))
        try:
            start = _parse_ass_time(record.get("start", ""))
            end = _parse_ass_time(record.get("end", ""))
        except ValueError:
            continue
        text = clean_text(record.get("text", ""))
        if text:
            cues.append(
                SubtitleLine(
                    index=len(cues) + 1,
                    start=start,
                    end=end,
                    text=text,
                    track_index=track_index,
                    language=language,
                )
            )
    return cues


def load_subtitles(path: Path, track: SubtitleTrack) -> list[SubtitleLine]:
    """Load a subtitle file, choosing the parser from its extension."""
    content = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()
    if suffix in {".ass", ".ssa"}:
        return parse_ass(content, track_index=track.index, language=track.language)
    return parse_srt(content, track_index=track.index, language=track.language)


def _overlap_duration(scene: Scene, line: SubtitleLine) -> float:
    """Seconds of ``line`` that fall inside ``scene`` (0 when they do not meet)."""
    return max(0.0, min(scene.end, line.end) - max(scene.start, line.start))


def _cue_owner(scenes: list[Scene], line: SubtitleLine) -> Scene | None:
    """The single scene that should own ``line`` (``scenes`` sorted by start).

    Midpoint containment is the primary rule: scenes normally tile the clip, so
    the cue's midpoint lands in exactly one scene.  The pipeline can, however,
    leave a small gap between scenes (``timeline.extend_scene_for_pan`` drops a
    following scene whose remainder is at most ``MIN_SCENE_LEN`` without
    extending the pan), so a midpoint can fall into no scene at all.  When that
    happens the cue falls back to the scene it overlaps most, and finally to the
    nearest scene; a cue is therefore never dropped.
    """
    if not scenes:
        return None
    midpoint = line.midpoint
    for scene in scenes:
        if scene.start <= midpoint < scene.end:
            return scene
    best = max(scenes, key=lambda scene: _overlap_duration(scene, line))
    if _overlap_duration(best, line) > 0.0:
        return best
    return min(scenes, key=lambda scene: abs(scene.range.midpoint - midpoint))


def assign_subtitles(scenes: list[Scene], lines: list[SubtitleLine]) -> None:
    """Attach every cue to exactly one scene.

    Ownership is by cue midpoint, which gives a cue displayed across a scene cut
    a single owner instead of listing it in both neighbours (the bug that made
    repeated lines appear in ``report.md``/``scenes.json`` and be lettered
    twice).  A cue whose midpoint falls in a gap between scenes is still claimed
    by its highest-overlap scene, so no line is silently dropped.  The midpoint
    rule matches how :meth:`Pipeline._filter_subtitles` chooses cues within the
    clip and how :func:`frames.target_time` places the representative frame.
    """
    ordered = sorted(scenes, key=lambda scene: scene.start)
    for scene in ordered:
        scene.subtitles = []
    for line in lines:
        owner = _cue_owner(ordered, line)
        if owner is not None:
            owner.subtitles.append(line)
