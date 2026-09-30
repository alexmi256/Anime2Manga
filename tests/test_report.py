"""Tests for the markdown report and JSON diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from anime2manga.models import (
    ClipWindow,
    FaceBox,
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
    scene1.faces = [FaceBox(100, 200, 80, 90, 0.87)]
    scene1.seam_carved_path = tmp_path / "seam_frames" / "scene_0001.jpg"
    scene1.seam_carve_shrink = 0.25
    scene1.seam_carve_size = (1440, 1080)
    scene1.subtitles = [SubtitleLine(1, 101.0, 103.0, "Hello there")]
    scene1.audio_focus = "left"
    scene1.audio_balance_db = 8.25

    scene2 = Scene(index=2, start=105.0, end=108.0, fps=23.976)
    scene2.is_panoramic = True
    scene2.pan_direction = "right"
    scene2.pan_start = 105.5
    scene2.pan_end = 107.25
    scene2.frame_path = tmp_path / "panoramas" / "scene_0002.png"
    scene2.panorama_inpainted_path = tmp_path / "panoramas" / "scene_0002_inpainted.jpg"
    scene2.inpaint_method = "biharmonic"
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
    assert "Audio Direction: Left" in text
    assert "Start Time: 00:01:40.000" in text
    assert "Start Frame: 2398" in text
    assert "![Frame Image](frames/scene_0001.jpg)" in text
    assert "![Seam Carved Frame](seam_frames/scene_0001.jpg)" in text
    # The seam-carved frame sits directly under the regular frame.
    assert text.index("![Seam Carved Frame]") > text.index("![Frame Image](frames/scene_0001.jpg)")
    assert "Seam Carve Shrink Percent: 25%" in text
    assert "Seam Carved Frame Size: 1440x1080" in text
    assert "Is Panoramic: No" in text
    assert "Frame Number: 2458" in text
    assert "Face Bounding Boxes:" in text
    assert "- x=100, y=200, width=80, height=90, confidence=0.87" in text
    assert "- Hello there" in text
    assert "![Frame Image](panoramas/scene_0002.png)" in text
    assert "![Inpainted Panorama](panoramas/scene_0002_inpainted.jpg)" in text
    # The filled panorama must sit directly under the regular panorama image.
    assert text.index("![Inpainted Panorama]") > text.index(
        "![Frame Image](panoramas/scene_0002.png)"
    )
    assert "# Scene Number 2" in text
    assert "Audio Direction: Center" in text
    assert "Is Panoramic: Yes" in text
    assert "Pan Direction: right" in text
    assert "Pan Start Time: 00:01:45.500" in text
    assert "Pan End Time: 00:01:47.250" in text
    assert "Pan Start Frame: 2529" in text
    assert "Pan End Frame: 2571" in text
    assert "- (no subtitles)" in text
    # Panoramic scenes are never seam carved.
    assert "Seam Carve Shrink Percent" not in text.split("# Scene Number 2", 1)[1]


def test_frame_number_is_only_reported_for_regular_frames(tmp_path):
    result = _result(tmp_path)
    panorama = result.scenes[1]
    panorama.frame_time = 106.0
    text = build_markdown(result)
    panorama_section = text.split("# Scene Number 2", 1)[1]
    assert "Frame Number:" not in panorama_section
    assert "Frame Number: 2458" in text


def test_scene_to_dict_is_json_serialisable(make_scene):
    scene = make_scene(start=1.0, end=2.0)
    scene.pan_shift = (3.5, 0.0)
    scene.pan_start = 1.25
    scene.pan_end = 1.75
    scene.faces = [FaceBox(10, 20, 30, 40, 0.5)]
    scene.seam_carved_path = Path("seam_frames/scene_0001.jpg")
    scene.seam_carve_shrink = 0.31
    scene.seam_carve_size = (1325, 1080)
    payload = scene_to_dict(scene)
    json.dumps(payload)  # must not raise
    assert payload["start"] == 1.0
    assert payload["seam_carved_path"] == "seam_frames/scene_0001.jpg"
    assert payload["seam_carve_shrink"] == 0.31
    assert payload["seam_carve_size"] == [1325, 1080] or payload["seam_carve_size"] == (1325, 1080)
    assert payload["is_panoramic"] is False
    assert payload["pan_start"] == 1.25
    assert payload["pan_start_frame"] == 30
    assert payload["pan_end_frame"] == 42
    assert payload["audio_focus"] == "center"
    assert payload["audio_direction"] == "Center"
    assert payload["audio_balance_db"] is None
    assert payload["panorama_inpainted_path"] is None
    assert payload["inpaint_method"] is None
    assert payload["faces"] == [{"x": 10, "y": 20, "width": 30, "height": 40, "confidence": 0.5}]


def test_report_hides_inpainted_image_when_absent(tmp_path):
    result = _result(tmp_path)
    result.scenes[1].panorama_inpainted_path = None
    text = build_markdown(result)
    assert "Inpainted Panorama" not in text


def test_report_lists_no_faces_when_none_detected(tmp_path):
    result = _result(tmp_path)
    result.scenes[0].faces = []
    text = build_markdown(result)
    scene_section = text.split("# Scene Number 1", 1)[1].split("# Scene Number 2", 1)[0]
    assert "Face Bounding Boxes:" in scene_section
    assert "- (no faces detected)" in scene_section


def test_report_distinguishes_no_chosen_frame(tmp_path):
    result = _result(tmp_path)
    result.scenes[0].frame_path = None
    result.scenes[0].faces = []
    text = build_markdown(result)
    scene_section = text.split("# Scene Number 1", 1)[1].split("# Scene Number 2", 1)[0]
    assert "Face Bounding Boxes:" in scene_section
    assert "- (no chosen frame)" in scene_section
    assert "- (no faces detected)" not in scene_section


def test_write_report_and_json(tmp_path):
    result = _result(tmp_path)
    report_path = write_report(result)
    json_path = write_scene_json(result)
    assert report_path.exists() and report_path.name == "report.md"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["input"] == "input.mkv"
    assert len(data["scenes"]) == 2
