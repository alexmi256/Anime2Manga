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
    assert config.draw_face_boxes is True


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
            "--no-face-boxes",
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
    assert config.draw_face_boxes is False
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
