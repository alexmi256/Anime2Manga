"""Tests for the retarget metrics and the reporting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from anime2manga.models import FaceBox
from anime2manga.retarget import METHODS, RetargetConfig, analyze_frame
from anime2manga.retarget_report import (
    build_overview_sheets,
    contact_sheet,
    render_html,
    render_markdown,
    signal_svg,
    summary_record,
    write_frame_outputs,
    write_summary,
)


def _write(path: Path, image: np.ndarray) -> Path:
    cv2.imwrite(str(path), image)
    return path


def _gradient(width: int = 160, height: int = 90) -> np.ndarray:
    ramp = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    image = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(image, (100, 20), (150, 70), (255, 255, 255), -1)  # a detail block
    return image


def _noise(width: int = 160, height: int = 90, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def _config(**kwargs) -> RetargetConfig:
    base = {"working_width": 96, "max_shrink": 0.5, "ssim_stride": 2}
    base.update(kwargs)
    return RetargetConfig(**base)


def test_analyze_frame_reports_every_method_within_cap(tmp_path: Path):
    path = _write(tmp_path / "frame.png", _gradient())
    analysis = analyze_frame(path, config=_config(), faces=[])
    assert analysis is not None
    assert set(analysis.limits) == {*METHODS, "composite"}
    for limit in analysis.limits.values():
        assert 0.0 <= limit.ratio <= 0.5
    assert analysis.recommended is analysis.limits["composite"]
    assert len(analysis.badness) == len(analysis.trace.steps)
    assert analysis.recommended_image is not None


def test_max_shrink_cap_is_respected(tmp_path: Path):
    path = _write(tmp_path / "frame.png", _gradient())
    analysis = analyze_frame(path, config=_config(max_shrink=0.3), faces=[])
    assert analysis is not None
    assert all(limit.ratio <= 0.3 + 1e-9 for limit in analysis.limits.values())


def test_primary_method_selects_that_limit(tmp_path: Path):
    path = _write(tmp_path / "frame.png", _gradient())
    analysis = analyze_frame(path, config=_config(primary_method="energy"), faces=[])
    assert analysis is not None
    assert analysis.recommended is analysis.limits["energy"]


def test_flat_frame_can_shrink_far(tmp_path: Path):
    flat = np.full((90, 160, 3), 128, np.uint8)
    path = _write(tmp_path / "flat.png", flat)
    analysis = analyze_frame(path, config=_config(), faces=[])
    assert analysis is not None
    # Nothing but the frame edges is detailed, so the energy guard should allow
    # almost the whole 50%.
    assert analysis.limits["energy"].ratio >= 0.4


def test_unreadable_frame_returns_none(tmp_path: Path):
    assert analyze_frame(tmp_path / "missing.png", config=_config(), faces=[]) is None


def test_strip_green_overlay_removes_box():
    from anime2manga.retarget import _strip_green_overlay

    image = np.full((200, 300, 3), 60, np.uint8)
    cv2.rectangle(image, (60, 30), (100, 70), (0, 255, 0), 2)
    before = _green_fraction(image)
    stripped = _strip_green_overlay(image)
    assert before > 0
    assert _green_fraction(stripped) < before


def test_green_scene_is_left_alone():
    from anime2manga.retarget import _strip_green_overlay

    image = np.full((90, 160, 3), 40, np.uint8)
    image[:, :80] = (0, 220, 0)  # a genuinely green half-frame
    assert np.array_equal(_strip_green_overlay(image), image)


def _green_fraction(image: np.ndarray) -> float:
    blue, green, red = (
        image[:, :, 0].astype(int),
        image[:, :, 1].astype(int),
        image[:, :, 2].astype(int),
    )
    mask = (green - red > 90) & (green - blue > 90) & (green > 130)
    return float(mask.mean())


def test_report_artifacts_are_written(tmp_path: Path):
    path = _write(tmp_path / "frame.png", _gradient())
    analysis = analyze_frame(path, config=_config(), faces=[])
    assert analysis is not None
    paths = write_frame_outputs(
        analysis,
        carved_dir=tmp_path / "carved",
        contacts_dir=tmp_path / "contacts",
        plots_dir=tmp_path / "plots",
        thumbs_dir=tmp_path / "thumbs",
        panel_width=200,
    )
    assert Path(paths["contact"]).exists()
    assert Path(paths["plot"]).read_text(encoding="utf-8").startswith("<svg")
    assert Path(paths["thumb"]).exists()

    record = summary_record(analysis)
    assert record["composite_ratio"] == round(analysis.limits["composite"].ratio, 4)
    summary = write_summary(tmp_path / "out", [record], _config())
    assert Path(summary["csv"]).exists()
    assert Path(summary["html"]).exists()
    assert Path(summary["markdown"]).exists()

    sheets = build_overview_sheets(
        [record], thumbs_dir=tmp_path / "thumbs", sheet_dir=tmp_path / "overview", cell_width=120
    )
    assert len(sheets) == 1 and Path(sheets[0]).exists()


def test_thumbnail_preserves_carved_aspect(tmp_path: Path):
    path = _write(tmp_path / "frame.png", _gradient())
    analysis = analyze_frame(path, config=_config(), faces=[])
    assert analysis is not None
    write_frame_outputs(
        analysis,
        carved_dir=tmp_path / "carved",
        contacts_dir=tmp_path / "contacts",
        plots_dir=tmp_path / "plots",
        thumbs_dir=tmp_path / "thumbs",
        panel_width=200,
        thumb_width=200,
    )
    thumb = cv2.imread(str(tmp_path / "thumbs" / f"{analysis.name}.jpg"))
    recommended = analysis.recommended_image
    assert recommended is not None
    assert thumb is not None
    assert thumb.shape[1] == 200
    native = recommended.shape[1] / recommended.shape[0]
    rendered = thumb.shape[1] / thumb.shape[0]
    assert abs(native - rendered) < 0.02


def test_overview_pillar_boxes_to_uniform_cells(tmp_path: Path):
    records: list[dict[str, Any]] = []
    for index, width in enumerate((160, 120)):
        name = f"scene_{index:04d}"
        image = _gradient(width=width, height=90)
        path = _write(tmp_path / f"{name}.png", image)
        analysis = analyze_frame(path, config=_config(), faces=[])
        assert analysis is not None
        write_frame_outputs(
            analysis,
            carved_dir=tmp_path / "carved",
            contacts_dir=tmp_path / "contacts",
            plots_dir=tmp_path / "plots",
            thumbs_dir=tmp_path / "thumbs",
            panel_width=120,
            thumb_width=80,
        )
        records.append(summary_record(analysis))
    sheets = build_overview_sheets(
        records,
        thumbs_dir=tmp_path / "thumbs",
        sheet_dir=tmp_path / "overview",
        columns=2,
        rows=1,
        cell_width=100,
    )
    sheet = cv2.imread(sheets[0])
    assert sheet is not None
    # Two equal cells side by side, each 26 label + working-aspect image box.
    assert sheet.shape[1] == 200
    assert sheet.shape[0] == 26 + round(100 * 90 / 160)


def test_html_has_all_markdown_columns_and_numeric_order(tmp_path: Path):
    records: list[dict[str, Any]] = []
    for index in (2, 1, 10):
        name = f"scene_{index:04d}"
        records.append(
            {
                "name": name,
                "faces": 0,
                "energy_ratio": 0.2,
                "energy_threshold": 0.31,
                "forward_ratio": 0.5,
                "detail_ratio": 0.4,
                "ssim_ratio": 0.1,
                "composite_ratio": 0.2,
            }
        )
    html = render_html(records, _config(), stamp="x")
    for header in ("E thr", "Energy", "Forward", "Detail", "Structural", "COMP"):
        assert header in html
    assert "0.310" in html
    # scene_0001, scene_0002, scene_0010 - numeric, not insertion or lexicographic.
    assert html.index("scene_0001") < html.index("scene_0002") < html.index("scene_0010")


def test_retarget_image_scales_full_resolution_faces():
    """Faces are detected at source resolution and must be scaled to the carve."""
    from anime2manga.retarget import _scale_faces, retarget_image

    face = FaceBox(100, 50, 200, 100)
    scaled = _scale_faces([face], 0.5)
    assert (scaled[0].x, scaled[0].y, scaled[0].width, scaled[0].height) == (50, 25, 100, 50)

    # A 400x200 frame carved at 200px working width halves the box.  If the box
    # were used unscaled it would be clamped to the whole working image.
    image = np.full((200, 400, 3), 40, np.uint8)
    analysis = retarget_image(image, config=_config(working_width=200), faces=[face])
    assert analysis.working_size == (200, 100)
    assert 0 < analysis.trace.protected_total < 200 * 100 * 0.5


def test_energy_threshold_matches_old_floor_at_defaults(tmp_path: Path):
    path = _write(tmp_path / "frame.png", _noise())
    analysis = analyze_frame(path, config=_config(), faces=[])
    assert analysis is not None
    early = analysis.energy_early
    expected = max(0.35, 1.4 * early)
    assert abs(analysis.energy_threshold - expected) < 1e-9


def test_lowering_energy_ratio_always_tightens(tmp_path: Path):
    # A busy frame whose early-seam energy is well above the reference, so the
    # adaptive term is active and the old max() formula would have ignored
    # energy_ratio entirely.
    path = _write(tmp_path / "busy.png", _noise(seed=5))
    base = analyze_frame(
        path, config=_config(energy_ratio=0.35, energy_baseline_multiple=1.4), faces=[]
    )
    lower = analyze_frame(
        path, config=_config(energy_ratio=0.2, energy_baseline_multiple=1.4), faces=[]
    )
    assert base is not None and lower is not None
    assert base.energy_early > 0.25  # adaptive term is in play
    assert lower.energy_threshold < base.energy_threshold
    assert lower.limits["energy"].ratio <= base.limits["energy"].ratio + 1e-9


def test_summary_html_busts_browser_cache(tmp_path: Path):
    path = _write(tmp_path / "frame.png", _gradient())
    analysis = analyze_frame(path, config=_config(), faces=[])
    assert analysis is not None
    record = summary_record(analysis)
    summary = write_summary(tmp_path / "out", [record], _config(), stamp="run-42")
    html = Path(summary["html"]).read_text(encoding="utf-8")
    assert "?v=run-42" in html
    assert record["energy_threshold"] > 0


def test_contact_sheet_and_plot_shapes(tmp_path: Path):
    path = _write(tmp_path / "frame.png", _noise())
    analysis = analyze_frame(path, config=_config(), faces=[])
    assert analysis is not None
    sheet = contact_sheet(analysis, panel_width=200)
    assert sheet.shape[1] == 200
    assert sheet.shape[0] > 200
    svg = signal_svg(analysis)
    assert "composite" in svg and svg.startswith("<svg")


def test_markdown_lists_every_frame():
    records: list[dict[str, Any]] = [{"name": "scene_0001"}, {"name": "scene_0002"}]
    for record in records:
        record.update(
            {
                "faces": 0,
                "energy_ratio": 0.2,
                "forward_ratio": 0.5,
                "detail_ratio": 0.4,
                "ssim_ratio": 0.1,
                "composite_ratio": 0.2,
            }
        )
    markdown = render_markdown(records, _config())
    assert "scene_0001" in markdown and "scene_0002" in markdown
