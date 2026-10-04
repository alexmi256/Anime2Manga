"""Tests for step 2 - subtitle track selection, extraction parsing and text."""

from __future__ import annotations

import pytest

from anime2manga.errors import (
    NoSubtitlesError,
    SubtitleTrackNotFoundError,
    UnsupportedSubtitleError,
)
from anime2manga.models import Scene, SubtitleLine, SubtitleTrack
from anime2manga.subtitles import (
    assign_subtitles,
    clean_text,
    ensure_supported,
    parse_ass,
    parse_srt,
    select_subtitle_track,
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


def _scene(index: int, start: float, end: float) -> Scene:
    return Scene(index=index, start=start, end=end, fps=24.0)


def _owned(scenes: list[Scene]) -> list[str]:
    return [line.text for scene in scenes for line in scene.subtitles]


def test_assign_subtitles_by_midpoint_across_a_cut():
    """A cue displayed across a scene cut belongs to exactly one scene.

    Midpoint ownership gives it a single owner (the later scene here) instead of
    listing it in both neighbours.
    """
    before = _scene(1, 0.0, 1.0)
    after = _scene(2, 1.0, 2.0)
    cue = SubtitleLine(1, 0.8, 1.5, "straddler")  # midpoint 1.15 -> after
    assign_subtitles([before, after], [cue])
    assert before.subtitles == []
    assert _owned([after]) == ["straddler"]


def test_assign_subtitles_never_duplicates_a_cue():
    scenes = [_scene(1, 0.0, 1.0), _scene(2, 1.0, 2.0), _scene(3, 2.0, 3.0)]
    lines = [
        SubtitleLine(1, 0.2, 0.5, "one"),
        SubtitleLine(2, 0.8, 1.5, "two"),
        SubtitleLine(3, 2.2, 2.8, "three"),
    ]
    assign_subtitles(scenes, lines)
    owned = _owned(scenes)
    assert len(owned) == len(lines)
    assert sorted(owned) == ["one", "three", "two"]


def test_assign_subtitles_claims_cue_whose_midpoint_falls_in_a_gap():
    """A pan can leave a small gap; the cue must still be assigned, not dropped.

    This mirrors ``timeline.extend_scene_for_pan`` dropping a following scene
    whose remainder is at most ``MIN_SCENE_LEN`` without extending the pan.
    """
    a = _scene(1, 0.0, 8.0)
    c = _scene(2, 8.2, 20.0)
    gap_cue = SubtitleLine(1, 7.9, 8.15, "in the gap")  # midpoint 8.025
    assign_subtitles([a, c], [gap_cue])
    assert _owned([a]) == ["in the gap"]
    assert c.subtitles == []


def test_assign_subtitles_falls_back_to_nearest_scene_without_overlap():
    """A cue shorter than the gap meets neither scene; nearest wins, not dropped."""
    a = _scene(1, 0.0, 8.0)
    c = _scene(2, 8.2, 20.0)
    cue = SubtitleLine(1, 8.05, 8.15, "wholly in the gap")
    assign_subtitles([a, c], [cue])
    assert _owned([a]) == ["wholly in the gap"]
    assert c.subtitles == []


def test_assign_subtitles_boundary_belongs_to_later_scene():
    """The half-open midpoint containment puts a boundary on the later scene."""
    before = _scene(1, 0.0, 1.0)
    after = _scene(2, 1.0, 2.0)
    cue = SubtitleLine(1, 0.0, 2.0, "on the boundary")  # midpoint == 1.0
    assign_subtitles([before, after], [cue])
    assert before.subtitles == []
    assert _owned([after]) == ["on the boundary"]
