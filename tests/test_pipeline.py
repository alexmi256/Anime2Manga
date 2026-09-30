"""Tests for pipeline helpers, the translation stub and future-stage stubs."""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest

from anime2manga.cropping import plan_crop
from anime2manga.errors import TranslationNotImplementedError
from anime2manga.faces import detect_faces
from anime2manga.models import (
    AudioInfo,
    ClipWindow,
    FaceBox,
    PanResult,
    Scene,
    SubtitleLine,
    TimeRange,
)
from anime2manga.pipeline import Pipeline, PipelineConfig, run_pipeline
from anime2manga.retarget import RetargetConfig
from anime2manga.text_layout import plan_text_placement
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


def test_isolate_pan_reserves_last_sampled_frame(tmp_path):
    config = PipelineConfig(
        input_path=tmp_path / "input.mkv", output_dir=tmp_path / "out", verbose=False
    )
    pipeline = Pipeline(config)
    scene = Scene(index=1, start=100.0, end=110.0, fps=24.0)
    pipeline.scenes = [scene]
    result = PanResult(
        scene_index=1,
        detected=True,
        direction="right",
        cumulative_shift=(500.0, 0.0),
        consistency=1.0,
        mean_response=0.9,
        start_time=100.0,
        end_time=104.0,
    )
    pan_scene = pipeline._isolate_pan(scene, result)
    assert pan_scene is not None
    # The pan owns the frame sampled at 104.0, so the next span starts a sample
    # later instead of reusing that boundary frame.
    assert pan_scene.start == 100.0
    assert pan_scene.end == pytest.approx(104.25)
    assert pipeline.scenes[-1].start == pytest.approx(104.25)
    clip = ClipWindow(start=100.0, end=110.0, source="test")
    assert validate_scenes(pipeline.scenes, clip) == []


def test_isolate_pan_absorbs_tail_that_starts_on_last_frame(tmp_path):
    config = PipelineConfig(
        input_path=tmp_path / "input.mkv", output_dir=tmp_path / "out", verbose=False
    )
    pipeline = Pipeline(config)
    scene = Scene(index=1, start=100.0, end=104.1, fps=24.0)
    pipeline.scenes = [scene]
    result = PanResult(
        scene_index=1,
        detected=True,
        direction="up-left",
        cumulative_shift=(-100.0, -300.0),
        consistency=1.0,
        mean_response=0.9,
        start_time=100.0,
        end_time=104.0,
    )
    pan_scene = pipeline._isolate_pan(scene, result)
    assert pan_scene is not None
    # The trailing 0.1s would have repeated the panorama's boundary frame, so it
    # is folded into the pan and no separate scene survives.
    assert len(pipeline.scenes) == 1
    assert pan_scene.end == pytest.approx(104.1)


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


def test_merge_pan_slivers_absorbs_short_scene_after_panorama(tmp_path):
    config = PipelineConfig(
        input_path=tmp_path / "input.mkv",
        output_dir=tmp_path / "out",
        pan_merge_max_len=1.0,
        verbose=False,
    )
    pipeline = Pipeline(config)
    pan = Scene(index=1, start=100.0, end=104.0, fps=24.0)
    pan.is_panoramic = True
    sliver = Scene(index=2, start=104.0, end=104.277, fps=24.0)
    following = Scene(index=3, start=104.277, end=110.0, fps=24.0)
    pipeline.scenes = [pan, sliver, following]

    pipeline._merge_pan_slivers()

    assert len(pipeline.scenes) == 2
    assert pipeline.scenes[0] is pan
    assert pan.end == 104.277
    assert [scene.index for scene in pipeline.scenes] == [1, 2]
    assert any("absorbed" in note for note in pan.notes)
    clip = ClipWindow(start=100.0, end=110.0, source="test")
    assert validate_scenes(pipeline.scenes, clip) == []


def test_merge_pan_slivers_ignores_long_scene_and_non_panorama(tmp_path):
    config = PipelineConfig(
        input_path=tmp_path / "input.mkv",
        output_dir=tmp_path / "out",
        pan_merge_max_len=1.0,
        verbose=False,
    )
    pipeline = Pipeline(config)
    pan = Scene(index=1, start=0.0, end=4.0, fps=24.0)
    pan.is_panoramic = True
    long_scene = Scene(index=2, start=4.0, end=6.0, fps=24.0)
    plain = Scene(index=3, start=6.0, end=10.0, fps=24.0)
    plain_sliver = Scene(index=4, start=10.0, end=10.5, fps=24.0)
    pipeline.scenes = [pan, long_scene, plain, plain_sliver]

    pipeline._merge_pan_slivers()

    assert len(pipeline.scenes) == 4
    assert pipeline.scenes[1] is long_scene
    assert pipeline.scenes[3] is plain_sliver


def test_merge_pan_slivers_rekeys_pan_results(tmp_path):
    config = PipelineConfig(
        input_path=tmp_path / "input.mkv",
        output_dir=tmp_path / "out",
        pan_merge_max_len=1.0,
        verbose=False,
    )
    pipeline = Pipeline(config)
    pan1 = Scene(index=1, start=0.0, end=4.0, fps=24.0)
    pan1.is_panoramic = True
    sliver = Scene(index=2, start=4.0, end=4.277, fps=24.0)
    pan2 = Scene(index=3, start=4.277, end=8.0, fps=24.0)
    pan2.is_panoramic = True
    pipeline.scenes = [pan1, sliver, pan2]
    first = PanResult(1, True, "right", (1.0, 0.0), 1.0, 0.9)
    second = PanResult(3, True, "left", (-1.0, 0.0), 1.0, 0.9)
    pipeline.pan_results = {1: first, 3: second}

    pipeline._merge_pan_slivers()

    # Removing the sliver renumbers pan2 from 3 to 2; the mapping must follow.
    assert [scene.index for scene in pipeline.scenes] == [1, 2]
    assert pipeline.pan_results[1] is first
    assert pipeline.pan_results[2] is second
    assert first.scene_index == 1
    assert second.scene_index == 2
    assert sorted(pipeline.pan_results) == [
        scene.index for scene in pipeline.scenes if scene.is_panoramic
    ]


def test_merge_pan_slivers_disabled_when_zero(tmp_path):
    config = PipelineConfig(
        input_path=tmp_path / "input.mkv",
        output_dir=tmp_path / "out",
        pan_merge_max_len=0.0,
        verbose=False,
    )
    pipeline = Pipeline(config)
    pan = Scene(index=1, start=0.0, end=4.0, fps=24.0)
    pan.is_panoramic = True
    sliver = Scene(index=2, start=4.0, end=4.277, fps=24.0)
    pipeline.scenes = [pan, sliver]

    pipeline._merge_pan_slivers()

    assert pipeline.scenes == [pan, sliver]


def test_run_pipeline_rejects_missing_input(tmp_path):
    from anime2manga.errors import Anime2MangaError

    config = PipelineConfig(input_path=tmp_path / "nope.mkv", output_dir=tmp_path / "out")
    with pytest.raises(Anime2MangaError):
        run_pipeline(config)


def test_translation_raises_until_implemented():
    with pytest.raises(TranslationNotImplementedError):
        require_translation(source="eng", target="spa")


def test_crop_stub_and_missing_face_detection_are_safe(tmp_path):
    assert detect_faces(tmp_path / "frame.jpg") == []
    crop = plan_crop(1920, 1080, [])
    assert (crop.width, crop.height) == (1080, 1080)
    assert crop.x == (1920 - 1080) // 2
    with_face = plan_crop(1920, 1080, [FaceBox(900, 400, 100, 100)])
    assert with_face.keeps_faces is True


def test_plan_text_placement_maps_center_to_auto(make_scene):
    scene = make_scene()
    panel = (100, 100)
    scene.audio_focus = "left"
    assert plan_text_placement(scene, [], [], panel).side == "left"
    scene.audio_focus = "right"
    assert plan_text_placement(scene, [], [], panel).side == "right"
    scene.audio_focus = "center"
    assert plan_text_placement(scene, [], [], panel).side == "auto"


def _face_pipeline(tmp_path, **overrides) -> Pipeline:
    config = PipelineConfig(
        input_path=tmp_path / "input.mkv",
        output_dir=tmp_path / "out",
        verbose=False,
        **overrides,
    )
    return Pipeline(config)


def test_step8_faces_attaches_boxes_and_annotates(tmp_path, monkeypatch):
    frame = tmp_path / "frame.png"
    cv2.imwrite(str(frame), np.zeros((40, 40, 3), np.uint8))
    scene = Scene(index=1, start=0.0, end=1.0, fps=24.0)
    scene.frame_path = frame
    pipeline = _face_pipeline(tmp_path)
    pipeline.scenes = [scene]

    box = FaceBox(1, 2, 3, 4, 0.5)
    monkeypatch.setattr("anime2manga.faces.detect_faces", lambda path, config=None: [box])
    drawn: list[list[FaceBox]] = []
    monkeypatch.setattr(
        "anime2manga.faces.annotate_faces",
        lambda path, boxes, **kwargs: drawn.append(list(boxes)),
    )

    pipeline._step8_faces()

    assert scene.faces == [box]
    assert drawn == [[box]]


def test_step8_faces_skips_drawing_when_disabled(tmp_path, monkeypatch):
    frame = tmp_path / "frame.png"
    cv2.imwrite(str(frame), np.zeros((40, 40, 3), np.uint8))
    scene = Scene(index=1, start=0.0, end=1.0, fps=24.0)
    scene.frame_path = frame
    pipeline = _face_pipeline(tmp_path, draw_face_boxes=False)
    pipeline.scenes = [scene]

    box = FaceBox(1, 2, 3, 4, 0.5)
    monkeypatch.setattr("anime2manga.faces.detect_faces", lambda path, config=None: [box])

    def fail(*args, **kwargs):
        raise AssertionError("annotate_faces should not be called")

    monkeypatch.setattr("anime2manga.faces.annotate_faces", fail)

    pipeline._step8_faces()

    assert scene.faces == [box]


def test_step8_faces_skips_scenes_without_frames(tmp_path, monkeypatch):
    scene = Scene(index=1, start=0.0, end=1.0, fps=24.0)
    pipeline = _face_pipeline(tmp_path)
    pipeline.scenes = [scene]

    def fail(*args, **kwargs):
        raise AssertionError("detect_faces should not be called without a frame")

    monkeypatch.setattr("anime2manga.faces.detect_faces", fail)

    pipeline._step8_faces()

    assert scene.faces == []


def _audio_media(make_media, channels: int = 2):
    return dataclasses.replace(
        make_media(),
        audio=AudioInfo(
            index=1,
            codec="eac3",
            channels=channels,
            channel_layout="stereo" if channels == 2 else "mono",
            sample_rate=48000,
        ),
    )


def test_step7_audio_attaches_focus_and_balance(tmp_path, make_media, make_scene, monkeypatch):
    from anime2manga.audio import AudioFocus

    pipeline = _face_pipeline(tmp_path)
    pipeline.media = _audio_media(make_media)
    pipeline.scenes = [make_scene(index=1), make_scene(index=2, start=10.0, end=20.0)]

    monkeypatch.setattr(
        "anime2manga.audio.detect_audio_focus",
        lambda media, scene, config=None: AudioFocus("left", 9.0, "left channel louder"),
    )

    pipeline._step7_audio()

    assert [scene.audio_focus for scene in pipeline.scenes] == ["left", "left"]
    assert pipeline.scenes[0].audio_balance_db == 9.0


def test_step7_audio_disabled_leaves_scenes_centered(tmp_path, make_media, make_scene, monkeypatch):
    pipeline = _face_pipeline(tmp_path, detect_audio=False)
    pipeline.media = _audio_media(make_media)
    pipeline.scenes = [make_scene()]

    def fail(*args, **kwargs):
        raise AssertionError("audio detection is disabled")

    monkeypatch.setattr("anime2manga.audio.detect_audio_focus", fail)

    pipeline._step7_audio()

    assert pipeline.scenes[0].audio_focus == "center"


def test_step7_audio_skips_mono_source(tmp_path, make_media, make_scene, monkeypatch):
    pipeline = _face_pipeline(tmp_path)
    pipeline.media = _audio_media(make_media, channels=1)
    pipeline.scenes = [make_scene()]

    def fail(*args, **kwargs):
        raise AssertionError("mono sources must not be decoded")

    monkeypatch.setattr("anime2manga.audio.detect_audio_focus", fail)

    pipeline._step7_audio()

    assert pipeline.scenes[0].audio_focus == "center"


# --- seam carving integration (step 8b) -----------------------------------


def _gradient_frame(path, width=160, height=90) -> str:
    ramp = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    image = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(image, (100, 20), (150, 70), (255, 255, 255), -1)
    cv2.imwrite(str(path), image)
    return str(path)


def test_step8b_writes_seam_frame_and_shrink(tmp_path, monkeypatch):
    _gradient_frame(tmp_path / "frame.png")
    scene = Scene(index=1, start=0.0, end=1.0, fps=24.0)
    scene.frame_path = tmp_path / "frame.png"
    pipeline = _face_pipeline(tmp_path)
    pipeline.scenes = [scene]
    monkeypatch.setattr("anime2manga.faces.detect_faces", lambda path, config=None: [])

    pipeline._step8_faces()

    assert scene.seam_carved_path is not None
    assert scene.seam_carved_path.exists()
    assert scene.seam_carve_shrink is not None
    assert 0.0 < scene.seam_carve_shrink <= 0.5
    assert scene.seam_carve_size is not None
    assert scene.seam_carve_size[0] <= 160


def test_seam_carve_disabled_when_energy_ratio_zero(tmp_path, monkeypatch):
    _gradient_frame(tmp_path / "frame.png")
    scene = Scene(index=1, start=0.0, end=1.0, fps=24.0)
    scene.frame_path = tmp_path / "frame.png"
    pipeline = _face_pipeline(
        tmp_path,
        seam_carve=True,
        retarget=RetargetConfig(energy_ratio=0.0, strip_overlays=False),
    )
    pipeline.scenes = [scene]
    monkeypatch.setattr("anime2manga.faces.detect_faces", lambda path, config=None: [])

    pipeline._step8_faces()

    assert scene.seam_carved_path is None
    assert not (tmp_path / "out" / "seam_frames").exists()


def test_seam_carve_skips_panorama(tmp_path, monkeypatch):
    _gradient_frame(tmp_path / "pano.png")
    scene = Scene(index=1, start=0.0, end=1.0, fps=24.0)
    scene.is_panoramic = True
    scene.frame_path = tmp_path / "pano.png"
    pipeline = _face_pipeline(tmp_path)
    pipeline.scenes = [scene]
    monkeypatch.setattr("anime2manga.faces.detect_faces", lambda path, config=None: [])

    pipeline._step8_faces()

    assert scene.seam_carved_path is None


def test_seam_carve_runs_before_face_boxes_are_drawn(tmp_path, monkeypatch):
    """The carver must see clean pixels: carve first, annotate second."""
    import types

    frame = tmp_path / "frame.png"
    cv2.imwrite(str(frame), np.zeros((60, 90, 3), np.uint8))
    scene = Scene(index=1, start=0.0, end=1.0, fps=24.0)
    scene.frame_path = frame
    pipeline = _face_pipeline(tmp_path)
    pipeline.scenes = [scene]

    order: list[str] = []

    def fake_retarget(image, **kwargs):
        order.append("carve")
        assert int(image.sum()) == 0  # clean frame, no box pixels yet
        return types.SimpleNamespace(
            recommended_image=image,
            recommended=types.SimpleNamespace(ratio=0.2),
        )

    monkeypatch.setattr(
        "anime2manga.faces.detect_faces", lambda path, config=None: [FaceBox(1, 2, 3, 4)]
    )
    monkeypatch.setattr("anime2manga.pipeline.retarget_image", fake_retarget)

    def fake_annotate(path, boxes, **kwargs):
        order.append("annotate")

    monkeypatch.setattr("anime2manga.faces.annotate_faces", fake_annotate)

    pipeline._step8_faces()

    assert order == ["carve", "annotate"]
    assert scene.seam_carve_shrink == 0.2


def test_pipeline_seam_carve_defaults_to_768_working_width(tmp_path):
    """The pipeline downscales to 768px by default (~13x faster than native)."""
    pipeline = _face_pipeline(tmp_path)
    assert pipeline.config.retarget.working_width == 768


def test_carve_frame_writes_and_reports():
    import tempfile
    from pathlib import Path

    from anime2manga.pipeline import _carve_frame

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        src = root / "frame.png"
        _gradient_frame(src)
        out = root / "carved.jpg"
        result = _carve_frame(
            (7, str(src), [], str(out), RetargetConfig(strip_overlays=False), 90)
        )
        assert result is not None
        assert result["index"] == 7
        assert out.exists()
        assert 0.0 < result["shrink"] <= 0.5
        assert result["size"][0] <= 160


def test_carve_frame_keeps_source_height_at_native_resolution():
    """The written seam frame keeps the source pixels (only columns removed)."""
    import tempfile
    from pathlib import Path

    from anime2manga.pipeline import _carve_frame

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        src = root / "frame.png"
        _gradient_frame(src, width=320, height=180)
        out = root / "carved.jpg"
        result = _carve_frame(
            (
                1,
                str(src),
                [],
                str(out),
                RetargetConfig(strip_overlays=False, working_width=None),
                90,
            )
        )
        assert result is not None
        # Vertical seams never change the height; the width shrinks by the carve.
        assert result["size"][1] == 180
        assert 160 <= result["size"][0] <= 320


def test_step8b_parallel_branch_uses_spawn_context(tmp_path, monkeypatch):
    """The CLI default (jobs>1) must use an explicit spawn context."""
    import anime2manga.pipeline as pipeline_module

    frames = []
    for index in (1, 2):
        path = tmp_path / f"frame{index}.png"
        _gradient_frame(path)
        scene = Scene(index=index, start=float(index), end=float(index) + 1, fps=24.0)
        scene.frame_path = path
        frames.append(scene)

    captured: dict = {}

    class _FakePool:
        def __init__(self, max_workers=None, mp_context=None):
            captured["context"] = mp_context

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, tasks):
            return [fn(task) for task in tasks]

    monkeypatch.setattr(pipeline_module, "ProcessPoolExecutor", _FakePool)
    pipeline = _face_pipeline(tmp_path, seam_carve_jobs=2)
    pipeline.scenes = frames

    pipeline._step8b_seam_carve()

    assert captured["context"] is not None
    assert captured["context"].get_start_method() == "spawn"
    assert all(scene.seam_carved_path is not None for scene in frames)


def test_seam_carve_panorama_skipped_and_ratio_zero_together(tmp_path, monkeypatch):
    """One pipeline: a regular frame is carved, its panorama neighbour is not;
    an energy ratio of 0 then disables carving for both."""
    regular = Scene(index=1, start=0.0, end=1.0, fps=24.0)
    regular.frame_path = tmp_path / "regular.png"
    _gradient_frame(regular.frame_path)
    panorama = Scene(index=2, start=1.0, end=2.0, fps=24.0)
    panorama.is_panoramic = True
    panorama.frame_path = tmp_path / "pano.png"
    _gradient_frame(panorama.frame_path, width=200)

    monkeypatch.setattr("anime2manga.faces.detect_faces", lambda path, config=None: [])

    enabled = _face_pipeline(tmp_path)
    enabled.scenes = [regular, panorama]
    enabled._step8_faces()
    assert regular.seam_carved_path is not None
    assert panorama.seam_carved_path is None

    regular.seam_carved_path = None
    disabled = _face_pipeline(
        tmp_path, retarget=RetargetConfig(energy_ratio=0.0, strip_overlays=False)
    )
    disabled.scenes = [regular, panorama]
    disabled._step8_faces()
    assert regular.seam_carved_path is None
    assert panorama.seam_carved_path is None
