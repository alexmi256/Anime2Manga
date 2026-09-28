"""Typed exceptions used across the pipeline.

Keeping errors explicit means the CLI can print actionable messages (for
example listing the subtitle languages that *are* available) instead of dumping
a traceback at the user.
"""

from __future__ import annotations


class Anime2MangaError(Exception):
    """Base class for all recoverable pipeline errors."""


class FFmpegError(Anime2MangaError):
    """ffmpeg/ffprobe failed or is not installed."""

    def __init__(self, message: str, command: list[str] | None = None) -> None:
        self.command = command
        super().__init__(message)


class NoSubtitlesError(Anime2MangaError):
    """The input has no subtitle tracks at all."""


class SubtitleTrackNotFoundError(Anime2MangaError):
    """The requested subtitle language/track is not present."""

    def __init__(self, message: str, available: list[str] | None = None) -> None:
        self.available = available or []
        super().__init__(message)


class UnsupportedSubtitleError(Anime2MangaError):
    """The subtitle track is bitmap based (PGS/VobSub) and would need OCR."""


class TranslationNotImplementedError(Anime2MangaError):
    """The user asked for subtitle translation, which is a planned feature."""
