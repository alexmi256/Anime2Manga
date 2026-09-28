"""Shared pytest fixtures and factories.

Tests never touch the real ``input.mkv``; they build small synthetic data so the
suite is fast and deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anime2manga.models import Chapter, ClipWindow, MediaInfo, Scene, SubtitleTrack


@pytest.fixture
def make_media(tmp_path: Path):
    def factory(
        *,
        duration: float = 1400.0,
        fps: float = 24.0,
        width: int = 1920,
        height: int = 1080,
        subtitle_tracks: list[SubtitleTrack] | None = None,
        chapters: list[Chapter] | None = None,
    ) -> MediaInfo:
        return MediaInfo(
            path=tmp_path / "input.mkv",
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            subtitle_tracks=subtitle_tracks or [],
            chapters=chapters or [],
        )

    return factory


@pytest.fixture
def eng_tracks() -> list[SubtitleTrack]:
    return [
        SubtitleTrack(index=3, codec="ass", language="eng", title="English", is_default=True),
        SubtitleTrack(index=4, codec="ass", language="eng", title="English(CC)"),
        SubtitleTrack(index=5, codec="ass", language="por", title="Portuguese(Brazil)"),
        SubtitleTrack(index=6, codec="ass", language="spa", title="Spanish(Latin_America)"),
    ]


@pytest.fixture
def clip() -> ClipWindow:
    return ClipWindow(start=0.0, end=100.0, source="test")


@pytest.fixture
def make_scene():
    def factory(index: int = 1, start: float = 0.0, end: float = 10.0, fps: float = 24.0) -> Scene:
        return Scene(index=index, start=start, end=end, fps=fps)

    return factory
