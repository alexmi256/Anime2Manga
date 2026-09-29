"""Tests for pipeline helpers, the translation stub and future-stage stubs."""

from __future__ import annotations

import pytest

from anime2manga.cropping import plan_crop
from anime2manga.errors import TranslationNotImplementedError
from anime2manga.faces import detect_faces
from anime2manga.models import (
    ClipWindow,
    FaceBox,
    PanResult,
    Scene,
    SubtitleLine,
    TimeRange,
)
from anime2manga.pipeline import Pipeline, PipelineConfig, run_pipeline
from anime2manga.timeline import coverage, validate_scenes
from anime2manga.translation import require_translation


def _line(start: float, end: float, text: str = "x") -> SubtitleLine:
    return SubtitleLine(index=1, start=start, end=end, text=text)


def test_filter_subtitles_drops_outside_and_excluded():
    clip = ClipWindow(
        start=100.0,
        end=200.0,
        source="test",
        excluded=[TimeRange(110.0, 120.0)],
    )
    lines = [
        _line(50.0, 60.0),  # before clip
        _line(101.0, 103.0),  # kept
        _line(114.0, 116.0),  # inside excluded intro/credits
        _line(150.0, 160.0),  # kept
        _line(250.0, 260.0),  # after clip
    ]
    kept = Pipeline._filter_subtitles(lines, clip)
    assert [line.text for line in kept] == ["x", "x"]
    assert kept[0].start == 101.0 and kept[1].start == 150.0


def test_isolate_pan_keeps_timeline_gapless(tmp_path):
    config = PipelineConfig(
        input_path=tmp_path / "input.mkv", output_dir=tmp_path / "out", verbose=False
    )
    pipeline = Pipeline(config)
    scene = Scene(index=1, start=100.0, end=110.0, fps=24.0)
    pipeline.scenes = [scene]
    result = PanResult(
        scene_index=1,
        detected=True,
        direction="up-left",
        cumulative_shift=(-100.0, -300.0),
        consistency=1.0,
        mean_response=0.9,
        start_time=103.1,
        end_time=109.9,
    )
    pan_scene = pipeline._isolate_pan(scene, result)
    assert pan_scene is not None
    clip = ClipWindow(start=100.0, end=110.0, source="test")
    assert validate_scenes(pipeline.scenes, clip) == []
    assert coverage(pipeline.scenes, clip) == 1.0
    # The tiny trailing sliver is absorbed rather than dropped.
    assert pipeline.scenes[-1].end == 110.0


def test_isolate_pan_snaps_whole_scene_when_segment_is_whole(tmp_path):
    config = PipelineConfig(
        input_path=tmp_path / "input.mkv", output_dir=tmp_path / "out", verbose=False
    )
    pipeline = Pipeline(config)
    scene = Scene(index=1, start=100.0, end=110.0, fps=24.0)
    pipeline.scenes = [scene]
    result = PanResult(
        scene_index=1,
        detected=True,
        direction="left",
        cumulative_shift=(-500.0, 0.0),
        consistency=1.0,
        mean_response=0.9,
        start_time=100.05,
        end_time=109.95,
    )
    pan_scene = pipeline._isolate_pan(scene, result)
    assert pan_scene is scene
    assert pipeline.scenes == [scene]


def test_run_pipeline_rejects_missing_input(tmp_path):
    from anime2manga.errors import Anime2MangaError

    config = PipelineConfig(input_path=tmp_path / "nope.mkv", output_dir=tmp_path / "out")
    with pytest.raises(Anime2MangaError):
        run_pipeline(config)


def test_translation_raises_until_implemented():
    with pytest.raises(TranslationNotImplementedError):
        require_translation(source="eng", target="spa")


def test_face_and_crop_stubs_are_safe(tmp_path):
    assert detect_faces(tmp_path / "frame.jpg") == []
    crop = plan_crop(1920, 1080, [])
    assert (crop.width, crop.height) == (1080, 1080)
    assert crop.x == (1920 - 1080) // 2
    with_face = plan_crop(1920, 1080, [FaceBox(900, 400, 100, 100)])
    assert with_face.keeps_faces is True
