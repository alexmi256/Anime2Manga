"""Tests for the markdown report and JSON diagnostics."""

from __future__ import annotations

import json

from anime2manga.models import (
    ClipWindow,
    MediaInfo,
    PipelineResult,
    Scene,
    SubtitleLine,
    SubtitleTrack,
)
from anime2manga.report import (
    build_markdown,
    format_time,
    scene_to_dict,
    write_report,
    write_scene_json,
)


def _result(tmp_path) -> PipelineResult:
    media = MediaInfo(
        path=tmp_path / "input.mkv",
        duration=1367.533,
        fps=23.976,
        width=1920,
        height=1080,
        subtitle_tracks=[
            SubtitleTrack(index=3, codec="ass", language="eng", title="English"),
            SubtitleTrack(index=5, codec="ass", language="por", title="Portuguese"),
        ],
        chapters=[],
    )
    scene1 = Scene(index=1, start=100.0, end=105.0, fps=23.976)
    scene1.frame_path = tmp_path / "frames" / "scene_0001.jpg"
    scene1.frame_size = (1920, 1080)
    scene1.frame_time = 102.5
    scene1.blur_score = 123.4
    scene1.subtitles = [SubtitleLine(1, 101.0, 103.0, "Hello there")]

    scene2 = Scene(index=2, start=105.0, end=108.0, fps=23.976)
    scene2.is_panoramic = True
    scene2.pan_direction = "right"
    scene2.frame_path = tmp_path / "panoramas" / "scene_0002.jpg"
    scene2.frame_size = (3840, 1080)
    scene2.panorama_size = (3840, 1080)
    scene2.subtitles = []

    return PipelineResult(
        media=media,
        clip=ClipWindow(start=90.0, end=1200.0, source="--start-at, --end-at"),
        subtitle_track=media.subtitle_tracks[0],
        subtitle_lines=[],
        scenes=[scene1, scene2],
        output_dir=tmp_path,
    )


def test_format_time():
    assert format_time(0.0) == "00:00:00.000"
    assert format_time(3723.5) == "01:02:03.500"


def test_build_markdown_contains_expected_sections(tmp_path):
    text = build_markdown(_result(tmp_path))
    assert text.startswith("# Anime2Manga")
    assert "Input File: input.mkv" in text
    assert "Subtitle Tracks:" in text
    assert "- 3: English" in text
    assert "Duration: 00:22:47.533" in text
    assert "# Scene Number 1" in text
    assert "Start Time: 00:01:40.000" in text
    assert "Start Frame: 2398" in text
    assert "![Frame Image](frames/scene_0001.jpg)" in text
    assert "Is Panoramic: No" in text
    assert "- Hello there" in text
    assert "# Scene Number 2" in text
    assert "Is Panoramic: Yes" in text
    assert "Pan Direction: right" in text
    assert "- (no subtitles)" in text


def test_scene_to_dict_is_json_serialisable(make_scene):
    scene = make_scene(start=1.0, end=2.0)
    scene.pan_shift = (3.5, 0.0)
    payload = scene_to_dict(scene)
    json.dumps(payload)  # must not raise
    assert payload["start"] == 1.0
    assert payload["is_panoramic"] is False


def test_write_report_and_json(tmp_path):
    result = _result(tmp_path)
    report_path = write_report(result)
    json_path = write_scene_json(result)
    assert report_path.exists() and report_path.name == "report.md"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["input"] == "input.mkv"
    assert len(data["scenes"]) == 2
