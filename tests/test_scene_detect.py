"""Tests for step 3 - scene detection parsing, tiling and subdivision."""

from __future__ import annotations

import pytest

from anime2manga import scene_detect
from anime2manga.errors import Anime2MangaError
from anime2manga.models import ClipWindow
from anime2manga.scene_detect import (
    build_scenes,
    parse_scdet_output,
    parse_select_output,
    subdivide_scene,
    validate_threshold,
)


def test_parse_select_output():
    text = (
        "frame:0    pts:2336    pts_time:2.336\n"
        "lavfi.scene_score=0.766465\n"
        "frame:1    pts:2544    pts_time:2.544\n"
        "lavfi.scene_score=0.714149\n"
    )
    assert parse_select_output(text) == [pytest.approx(2.336), pytest.approx(2.544)]


def test_parse_select_output_applies_offset():
    text = "frame:0 pts:1 pts_time:1.5\n"
    assert parse_select_output(text, offset=600.0) == [pytest.approx(601.5)]


def test_parse_scdet_output():
    text = (
        "[Parsed_scdet_0 @ 0x1] lavfi.scd.score: 29.940, lavfi.scd.time: 2.336\n"
        "[Parsed_scdet_0 @ 0x1] lavfi.scd.score: 27.896, lavfi.scd.time: 2.544\n"
    )
    assert parse_scdet_output(text) == [pytest.approx(2.336), pytest.approx(2.544)]


def test_validate_threshold():
    validate_threshold("select", 0.4)
    validate_threshold("scdet", 10.0)
    with pytest.raises(Anime2MangaError):
        validate_threshold("select", 1.5)
    with pytest.raises(Anime2MangaError):
        validate_threshold("scdet", 0)


def test_build_scenes_tiles_clip_without_gaps(clip):
    scenes = build_scenes([1.0, 2.0, 3.0], clip, fps=24.0, min_scene_len=0.1)
    assert [s.start for s in scenes] == [0.0, 1.0, 2.0, 3.0]
    assert [s.end for s in scenes] == [1.0, 2.0, 3.0, 100.0]
    assert [s.index for s in scenes] == [1, 2, 3, 4]


def test_build_scenes_merges_close_cuts_and_filters_outside():
    clip = ClipWindow(start=0.0, end=10.0, source="test")
    scenes = build_scenes([-5.0, 2.0, 2.1, 20.0], clip, fps=24.0, min_scene_len=0.5)
    # 2.0 and 2.1 merge; outside cuts dropped.
    assert [(s.start, s.end) for s in scenes] == [(0.0, 2.0), (2.0, 10.0)]


def test_subdivide_scene_uses_lower_threshold(make_media, make_scene, monkeypatch):
    media = make_media()
    scene = make_scene(index=1, start=0.0, end=10.0)

    def fake_detect(*args, **kwargs):
        assert kwargs["start"] == 0.0 and kwargs["end"] == 10.0
        return [5.0]

    monkeypatch.setattr(scene_detect, "detect_scene_times", fake_detect)
    pieces = subdivide_scene(scene, media, method="select", threshold=0.2)
    assert len(pieces) == 2
    assert [(p.start, p.end) for p in pieces] == [(0.0, 5.0), (5.0, 10.0)]
    assert "subdivided from scene 1" in pieces[0].notes


def test_subdivide_scene_returns_original_when_no_cuts(make_media, make_scene, monkeypatch):
    media = make_media()
    scene = make_scene(start=0.0, end=10.0)
    monkeypatch.setattr(scene_detect, "detect_scene_times", lambda *a, **k: [])
    pieces = subdivide_scene(scene, media, method="select", threshold=0.2)
    assert len(pieces) == 1
    assert pieces[0].start == 0.0 and pieces[0].end == 10.0
