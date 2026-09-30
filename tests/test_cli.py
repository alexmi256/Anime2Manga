"""Tests for the CLI argument handling and error path."""

from __future__ import annotations

from pathlib import Path

import pytest

from anime2manga import cli
from anime2manga.errors import Anime2MangaError
from anime2manga.models import ClipWindow, MediaInfo, PipelineResult, SubtitleTrack


def test_parser_defaults():
    args = cli.build_parser().parse_args(["video.mkv"])
    config = cli.config_from_args(args)
    assert config.input_path == Path("video.mkv")
    assert config.subtitle_language == "eng"
    assert config.scene_method == "select"
    assert config.scene_threshold is None
    assert config.detect_pans is True
    assert config.pan.sample_fps == 4.0
    assert config.pan.min_pair_response == 0.2
    assert config.pan.diagonal_ratio == 0.35
    assert config.pan_merge_max_len == 1.0
    assert config.inpaint.enabled is True
    assert config.inpaint.method == "biharmonic"
    assert config.inpaint.quality == 92
    assert config.inpaint.max_pixels == 500_000
    assert config.draw_boxes is True
    # Faces are off by default; heads and persons are on.
    assert config.detect_face is False
    assert config.detect_head is True
    assert config.detect_person is True
    assert config.draw_face_boxes is False
    assert config.draw_head_boxes is True
    assert config.draw_person_boxes is True
    assert config.detect_audio is True
    assert config.audio.balance_threshold_db == 1.5
    assert config.audio.silence_floor_db == -60.0
    assert config.audio.band_low_hz == 300.0
    assert config.audio.band_high_hz == 3400.0
    # Seam carving is on by default with the tuned energy ratio.
    assert config.seam_carve is True
    assert config.retarget.energy_ratio == 0.25
    assert config.retarget.max_shrink == 0.5
    assert config.retarget.working_width == 768
    assert config.seam_carve_quality == 92


def test_seam_carve_working_width_zero_means_native():
    """``0`` opts out of downscaling; the engine then carves at source size."""
    parser = cli.build_parser()
    native = cli.config_from_args(
        parser.parse_args(["video.mkv", "--seam-carve-working-width", "0"])
    )
    assert native.retarget.working_width is None


def test_detection_and_box_toggles():
    """Per-category toggles work, and ``--face-boxes`` stays a valid alias."""
    parser = cli.build_parser()

    all_off = cli.config_from_args(
        parser.parse_args(
            [
                "video.mkv",
                "--no-detect-face",
                "--no-detect-head",
                "--no-detect-person",
                "--no-boxes",
                "--no-head-bbox",
                "--no-person-bbox",
                "--no-face-bbox",
            ]
        )
    )
    assert all_off.detect_face is False
    assert all_off.detect_head is False
    assert all_off.detect_person is False
    assert all_off.draw_boxes is False
    assert all_off.draw_face_boxes is False
    assert all_off.draw_head_boxes is False
    assert all_off.draw_person_boxes is False

    face_on = cli.config_from_args(parser.parse_args(["video.mkv", "--detect-face"]))
    assert face_on.detect_face is True

    # ``--face-boxes`` is still accepted as an alias for ``--boxes``.
    assert (
        cli.config_from_args(parser.parse_args(["video.mkv", "--no-face-boxes"])).draw_boxes
        is False
    )


def test_parser_overrides():
    args = cli.build_parser().parse_args(
        [
            "video.mkv",
            "--subtitle-language",
            "por",
            "--subtitle-track",
            "5",
            "--scene-method",
            "scdet",
            "--scene-threshold",
            "12",
            "--start-at",
            "02:00",
            "--end-at",
            "20:00",
            "--no-pan",
            "--pan-min-shift",
            "0.3",
            "--pan-pair-response",
            "0.35",
            "--pan-diagonal-ratio",
            "0.5",
            "--pan-merge-max-len",
            "0.5",
            "--no-inpaint",
            "--inpaint-quality",
            "80",
            "--inpaint-max-pixels",
            "200000",
            "--no-face-boxes",
            "--no-audio-direction",
            "--audio-balance-db",
            "2.5",
            "--audio-silence-db",
            "-70",
            "--audio-band-low",
            "200",
            "--audio-band-high",
            "4000",
            "--quiet",
        ]
    )
    config = cli.config_from_args(args)
    assert config.subtitle_language == "por"
    assert config.subtitle_track == 5
    assert config.scene_method == "scdet"
    assert config.scene_threshold == 12.0
    assert config.start_at == "02:00"
    assert config.end_at == "20:00"
    assert config.detect_pans is False
    assert config.pan.min_shift_fraction == 0.3
    assert config.pan.min_pair_response == 0.35
    assert config.pan.diagonal_ratio == 0.5
    assert config.pan_merge_max_len == 0.5
    assert config.inpaint.enabled is False
    assert config.inpaint.quality == 80
    assert config.inpaint.max_pixels == 200000
    assert config.draw_boxes is False
    assert config.detect_audio is False
    assert config.audio.balance_threshold_db == 2.5
    assert config.audio.silence_floor_db == -70.0
    assert config.audio.band_low_hz == 200.0
    assert config.audio.band_high_hz == 4000.0
    assert config.verbose is False


def test_main_reports_errors_gracefully(monkeypatch, capsys):
    def boom(config):
        raise Anime2MangaError("no subtitles here")

    monkeypatch.setattr(cli, "run_pipeline", boom)
    code = cli.main(["video.mkv"])
    assert code == 2
    assert "no subtitles here" in capsys.readouterr().err


def test_main_success_writes_report(monkeypatch, tmp_path):
    media = MediaInfo(
        path=tmp_path / "input.mkv",
        duration=10.0,
        fps=24.0,
        width=1920,
        height=1080,
        subtitle_tracks=[SubtitleTrack(3, "ass", "eng", "English")],
        chapters=[],
    )
    result = PipelineResult(
        media=media,
        clip=ClipWindow(start=0.0, end=10.0, source="full video"),
        subtitle_track=media.subtitle_tracks[0],
        subtitle_lines=[],
        scenes=[],
        output_dir=tmp_path,
    )
    monkeypatch.setattr(cli, "run_pipeline", lambda config: result)
    code = cli.main(["video.mkv", "-o", str(tmp_path), "--quiet"])
    assert code == 0
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "scenes.json").exists()


def test_main_invalid_args_exits(capsys):
    with pytest.raises(SystemExit):
        cli.main([])


def test_seam_carve_override_and_disable():
    parser = cli.build_parser()
    tuned = cli.config_from_args(
        parser.parse_args(
            [
                "video.mkv",
                "--seam-carve-energy-ratio",
                "0.15",
                "--seam-carve-max-shrink",
                "0.4",
                "--seam-carve-working-width",
                "512",
                "--seam-carve-detail-budget",
                "0.07",
                "--seam-carve-jobs",
                "3",
                "--seam-carve-quality",
                "80",
            ]
        )
    )
    assert tuned.seam_carve is True
    assert tuned.retarget.energy_ratio == 0.15
    assert tuned.retarget.max_shrink == 0.4
    assert tuned.retarget.working_width == 512
    assert tuned.retarget.detail_budget == 0.07
    assert tuned.seam_carve_jobs == 3
    assert tuned.seam_carve_quality == 80

    # A zero energy ratio disables carving, as does --no-seam-carve.
    zero = cli.config_from_args(parser.parse_args(["video.mkv", "--seam-carve-energy-ratio", "0"]))
    assert zero.seam_carve is False
    off = cli.config_from_args(parser.parse_args(["video.mkv", "--no-seam-carve"]))
    assert off.seam_carve is False
