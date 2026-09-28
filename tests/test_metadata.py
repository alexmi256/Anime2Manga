"""Tests for step 1 - metadata, timestamps and clip-window resolution."""

from __future__ import annotations

import pytest

from anime2manga.errors import Anime2MangaError
from anime2manga.metadata import (
    classify_chapters,
    parse_timestamp,
    resolve_clip_window,
)
from anime2manga.models import Chapter


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0", 0.0),
        ("90", 90.0),
        ("02:00", 120.0),
        ("1:02:03", 3723.0),
        ("00:22:47.533", 1367.533),
    ],
)
def test_parse_timestamp(text, expected):
    assert parse_timestamp(text) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("text", ["", "abc", "1:2:3:4"])
def test_parse_timestamp_rejects_bad_input(text):
    with pytest.raises(Anime2MangaError):
        parse_timestamp(text)


def test_classify_chapters_finds_intro_and_credits():
    chapters = [
        Chapter(0, "Opening", 10.0, 100.0),
        Chapter(1, "Part A", 100.0, 1200.0),
        Chapter(2, "Ending", 1200.0, 1350.0),
        Chapter(3, "Preview", 1350.0, 1368.0),
    ]
    intro, credits = classify_chapters(chapters, duration=1368.0)
    assert intro is not None and (intro.start, intro.end) == (10.0, 100.0)
    assert credits is not None and (credits.start, credits.end) == (1200.0, 1368.0)


def test_classify_chapters_ignores_unrelated_titles():
    chapters = [Chapter(0, "Cold Open", 0.0, 60.0), Chapter(1, "Part A", 60.0, 1300.0)]
    assert classify_chapters(chapters, duration=1368.0) == (None, None)


def test_classify_chapters_does_not_match_ed_inside_words():
    chapters = [Chapter(0, "Edited Highlights", 700.0, 900.0)]
    _, credits = classify_chapters(chapters, duration=1368.0)
    assert credits is None


def test_resolve_clip_window_without_metadata(make_media):
    media = make_media(duration=1000.0)
    window = resolve_clip_window(media)
    assert (window.start, window.end) == (0.0, 1000.0)
    assert window.source == "full video"


def test_resolve_clip_window_uses_chapters(make_media):
    import dataclasses

    media = make_media(
        duration=1368.0,
        chapters=[
            Chapter(0, "Opening", 10.0, 100.0),
            Chapter(1, "Ending", 1200.0, 1368.0),
        ],
    )
    media = dataclasses.replace(
        media, intro=media.chapters[0].range, credits=media.chapters[1].range
    )
    window = resolve_clip_window(media)
    assert (window.start, window.end) == (100.0, 1200.0)
    assert "intro-chapter" in window.source
    assert len(window.excluded) == 2


def test_resolve_clip_window_cli_overrides(make_media):
    media = make_media(duration=1368.0)
    window = resolve_clip_window(media, start_at="02:00", end_at="20:00")
    assert (window.start, window.end) == (120.0, 1200.0)
    assert "--start-at" in window.source and "--end-at" in window.source


def test_resolve_clip_window_rejects_empty(make_media):
    media = make_media(duration=1000.0)
    with pytest.raises(Anime2MangaError):
        resolve_clip_window(media, start_at="20:00", end_at="10:00")
