"""Tests for the per-frame subject composition metrics."""

from __future__ import annotations

import json

from anime2manga.composition import (
    analyze_composition,
    heads_contained_in_body,
    intersection_area,
    lean,
    lean_label,
    union_area,
)
from anime2manga.models import (
    ClipWindow,
    FaceBox,
    FrameLean,
    MediaInfo,
    PipelineResult,
    Scene,
    SubtitleTrack,
)
from anime2manga.report import build_markdown, scene_to_dict


def test_union_area_merges_overlaps():
    assert union_area([(0, 0, 10, 10), (5, 5, 15, 15)]) == 175
    assert union_area([]) == 0
    # Disjoint rectangles simply add up.
    assert union_area([(0, 0, 10, 10), (20, 0, 30, 10)]) == 200


def test_intersection_area_is_the_shared_region():
    assert intersection_area([(0, 0, 10, 10)], [(5, 5, 15, 15)]) == 25
    assert intersection_area([(0, 0, 10, 10)], [(20, 20, 25, 25)]) == 0
    # A head straddling two adjacent bodies stays one intersection.
    bodies = [(0, 0, 10, 10), (10, 0, 20, 10)]
    head = [(5, 0, 15, 10)]
    assert intersection_area(bodies, head) == 100


def test_lean_labels():
    assert lean([(0, 0, 10, 10)], 100) is FrameLean.LEFT
    assert lean([(45, 0, 55, 10)], 100) is FrameLean.MIDDLE
    assert lean([(90, 0, 100, 10)], 100) is FrameLean.RIGHT
    assert lean([], 100) is None
    assert lean_label(FrameLean.MIDDLE) == "Middle"
    assert lean_label(None) == "None"


def test_heads_contained_in_body():
    body = [(0, 0, 100, 100)]
    assert heads_contained_in_body([(10, 10, 30, 30)], body) is True
    assert heads_contained_in_body([(50, 50, 150, 150)], body) is False
    assert heads_contained_in_body([], body) is None
    assert heads_contained_in_body([(10, 10, 15, 15)], []) is False


def test_analyze_composition_single_subject():
    composition = analyze_composition(
        (1000, 1000),
        bodies=[FaceBox(0, 0, 300, 1000)],
        heads=[FaceBox(50, 50, 100, 100)],
    )
    assert composition.body_percent == 30.0
    assert composition.head_percent == 1.0
    assert composition.body_head_overlap_percent == 1.0
    assert composition.heads_in_body is True
    assert composition.body_leans is FrameLean.LEFT
    assert composition.head_leans is FrameLean.LEFT


def test_analyze_composition_leans_middle_and_right():
    middle = analyze_composition((1000, 1000), [FaceBox(300, 0, 400, 1000)], [])
    assert middle.body_leans is FrameLean.MIDDLE
    right = analyze_composition((1000, 1000), [FaceBox(700, 0, 300, 1000)], [])
    assert right.body_leans is FrameLean.RIGHT


def test_analyze_composition_empty_categories():
    nobody = analyze_composition((100, 100), [], [FaceBox(10, 10, 10, 10)])
    assert nobody.body_percent == 0.0
    assert nobody.head_percent == 1.0
    assert nobody.body_head_overlap_percent == 0.0
    assert nobody.heads_in_body is False
    assert nobody.body_leans is None
    assert nobody.head_leans is FrameLean.LEFT

    nothing = analyze_composition((100, 100), [], [])
    assert nothing.body_percent == 0.0
    assert nothing.head_percent == 0.0
    assert nothing.body_head_overlap_percent == 0.0
    assert nothing.heads_in_body is None
    assert nothing.body_leans is None
    assert nothing.head_leans is None


def test_analyze_composition_clips_boxes_to_the_frame():
    composition = analyze_composition(
        (1000, 1000), bodies=[FaceBox(900, 900, 300, 300)], heads=[]
    )
    assert composition.body_percent == 1.0


def test_analyze_composition_multiple_bodies_use_union():
    # Two overlapping bodies are counted once, not twice.
    composition = analyze_composition(
        (100, 100),
        bodies=[FaceBox(0, 0, 60, 100), FaceBox(40, 0, 60, 100)],
        heads=[],
    )
    assert composition.body_percent == 100.0


def _composition_result(tmp_path) -> PipelineResult:
    media = MediaInfo(
        path=tmp_path / "input.mkv",
        duration=10.0,
        fps=24.0,
        width=1000,
        height=1000,
        subtitle_tracks=[SubtitleTrack(index=1, codec="ass", language="eng", title="E")],
        chapters=[],
    )
    scene = Scene(index=1, start=0.0, end=1.0, fps=24.0)
    scene.frame_path = tmp_path / "frames" / "scene_0001.jpg"
    scene.frame_size = (1000, 1000)
    scene.persons = [FaceBox(0, 0, 300, 1000)]
    scene.heads = [FaceBox(50, 50, 100, 100)]
    scene.composition = analyze_composition(
        scene.frame_size, scene.persons, scene.heads
    )
    return PipelineResult(
        media=media,
        clip=ClipWindow(start=0.0, end=10.0, source="test"),
        subtitle_track=media.subtitle_tracks[0],
        subtitle_lines=[],
        scenes=[scene],
        output_dir=tmp_path,
    )


def test_report_renders_composition(tmp_path):
    text = build_markdown(_composition_result(tmp_path))
    assert "## Subject Composition" in text
    assert "Body Percent of Frame: 30.0%" in text
    assert "Head Percent of Frame: 1.0%" in text
    assert "Body and Head Overlap Percent: 1.0%" in text
    assert "Heads Contained in Body: Yes" in text
    assert "Body Leans Towards: Left" in text
    assert "Head Leans Towards: Left" in text


def test_report_marks_missing_composition(tmp_path):
    result = _composition_result(tmp_path)
    result.scenes[0].composition = None
    text = build_markdown(result)
    assert "## Subject Composition" in text
    assert "- (not computed)" in text


def test_scene_to_dict_serialises_composition(make_scene):
    scene = make_scene()
    scene.frame_size = (1000, 1000)
    scene.persons = [FaceBox(0, 0, 300, 1000)]
    scene.heads = [FaceBox(50, 50, 100, 100)]
    scene.composition = analyze_composition(scene.frame_size, scene.persons, scene.heads)
    payload = scene_to_dict(scene)
    json.dumps(payload)  # must not raise
    assert payload["composition"]["body_percent"] == 30.0
    assert payload["composition"]["head_percent"] == 1.0
    assert payload["composition"]["body_head_overlap_percent"] == 1.0
    assert payload["composition"]["heads_in_body"] is True
    assert payload["composition"]["body_leans"] == "left"
    assert payload["composition"]["head_leans"] == "left"


def test_scene_to_dict_composition_none_by_default(make_scene):
    payload = scene_to_dict(make_scene())
    assert payload["composition"] is None
