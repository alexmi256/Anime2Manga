"""Tests for step 2 - subtitle track selection, extraction parsing and text."""

from __future__ import annotations

import pytest

from anime2manga.errors import (
    NoSubtitlesError,
    SubtitleTrackNotFoundError,
    UnsupportedSubtitleError,
)
from anime2manga.models import SubtitleLine, SubtitleTrack, TimeRange
from anime2manga.subtitles import (
    clean_text,
    ensure_supported,
    parse_ass,
    parse_srt,
    select_subtitle_track,
    subtitles_in_range,
)

SRT = """1
00:00:01,000 --> 00:00:03,500
<font face="Arial"><b>Hello</b> there</font>

2
00:00:04,000 --> 00:00:06,000
Line one
Line two
"""

ASS = """[Script Info]
Title: test

[V4+ Styles]
Format: Name, Fontname
Style: Default,Arial

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.50,Default,,0,0,0,,{\\i1}Hello{\\i0} there
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,Line one\\NLine two
"""


def test_parse_srt_strips_markup_and_multiline():
    cues = parse_srt(SRT)
    assert len(cues) == 2
    assert cues[0].text == "Hello there"
    assert cues[1].text == "Line one Line two"
    assert cues[0].start == pytest.approx(1.0)
    assert cues[0].end == pytest.approx(3.5)


def test_parse_ass_strips_override_tags():
    cues = parse_ass(ASS)
    assert len(cues) == 2
    assert cues[0].text == "Hello there"
    assert cues[1].text == "Line one Line two"


def test_clean_text_unescapes_entities():
    assert clean_text("Tom &amp; Jerry") == "Tom & Jerry"


def test_select_english_prefers_default_non_cc(make_media, eng_tracks):
    media = make_media(subtitle_tracks=eng_tracks)
    track = select_subtitle_track(media, language="eng")
    assert track.index == 3


def test_select_language_alias_en(make_media, eng_tracks):
    media = make_media(subtitle_tracks=eng_tracks)
    assert select_subtitle_track(media, language="en").index == 3


def test_select_explicit_track_id_wins(make_media, eng_tracks):
    media = make_media(subtitle_tracks=eng_tracks)
    assert select_subtitle_track(media, track_id=5).language == "por"


def test_select_missing_language_lists_available(make_media, eng_tracks):
    media = make_media(subtitle_tracks=eng_tracks)
    with pytest.raises(SubtitleTrackNotFoundError) as exc:
        select_subtitle_track(media, language="fra")
    assert "#5" in str(exc.value) or "Portuguese" in str(exc.value)


def test_select_missing_track_id_lists_available(make_media, eng_tracks):
    media = make_media(subtitle_tracks=eng_tracks)
    with pytest.raises(SubtitleTrackNotFoundError):
        select_subtitle_track(media, track_id=99)


def test_select_no_tracks_raises(make_media):
    media = make_media(subtitle_tracks=[])
    with pytest.raises(NoSubtitlesError):
        select_subtitle_track(media)


def test_bitmap_track_is_rejected():
    track = SubtitleTrack(index=0, codec="hdmv_pgs_subtitle", language="eng", title="PGS")
    with pytest.raises(UnsupportedSubtitleError):
        ensure_supported(track)


def test_subtitles_in_range_uses_overlap():
    lines = [
        SubtitleLine(1, 1.0, 2.0, "a"),
        SubtitleLine(2, 5.0, 6.0, "b"),
        SubtitleLine(3, 9.0, 10.0, "c"),
    ]
    hits = subtitles_in_range(lines, TimeRange(4.5, 8.0))
    assert [line.text for line in hits] == ["b"]
