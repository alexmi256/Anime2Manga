"""Tests for step 5 - timeline validation and pan retiming."""

from __future__ import annotations

import pytest

from anime2manga.models import ClipWindow
from anime2manga.timeline import (
    coverage,
    extend_scene_for_pan,
    reindex,
    validate_scenes,
)


def _scenes(*ranges):
    from anime2manga.models import Scene

    return [Scene(index=i + 1, start=s, end=e, fps=24.0) for i, (s, e) in enumerate(ranges)]


def test_validate_scenes_clean(clip):
    scenes = _scenes((0.0, 50.0), (50.0, 100.0))
    assert validate_scenes(scenes, clip) == []
    assert coverage(scenes, clip) == 1.0


def test_validate_scenes_detects_gap_and_overlap():
    clip = ClipWindow(start=0.0, end=100.0, source="test")
    gapped = _scenes((0.0, 40.0), (50.0, 100.0))
    assert any("Gap" in p for p in validate_scenes(gapped, clip))
    overlapping = _scenes((0.0, 60.0), (50.0, 100.0))
    assert any("Overlap" in p for p in validate_scenes(overlapping, clip))


def test_validate_scenes_empty():
    clip = ClipWindow(start=0.0, end=100.0, source="test")
    assert validate_scenes([], clip)


def test_extend_scene_for_pan_keeps_gapless(clip):
    scenes = _scenes((0.0, 10.0), (10.0, 20.0), (20.0, 30.0))
    applied = extend_scene_for_pan(scenes, scenes[0], 3.0)
    assert applied == 3.0
    assert (scenes[0].start, scenes[0].end) == (0.0, 13.0)
    assert scenes[1].start == 13.0
    assert scenes[1].end == 20.0
    assert validate_scenes(scenes, ClipWindow(0.0, 30.0, "test")) == []


def test_extend_scene_for_pan_drops_swallowed_scene():
    scenes = _scenes((0.0, 10.0), (10.0, 10.2), (10.2, 20.0))
    applied = extend_scene_for_pan(scenes, scenes[0], 5.0)
    assert applied == pytest.approx(0.2)  # capped at the tiny scene's duration
    assert len(scenes) == 2
    assert [s.index for s in scenes] == [1, 2]


def test_extend_scene_for_pan_last_scene_is_noop():
    scenes = _scenes((0.0, 10.0))
    assert extend_scene_for_pan(scenes, scenes[0], 5.0) == 0.0


def test_reindex_sorts_and_numbers():
    scenes = _scenes((10.0, 20.0), (0.0, 10.0))
    scenes[0].index = 99
    reindex(scenes)
    assert [s.index for s in scenes] == [1, 2]
    assert scenes[0].start == 0.0
