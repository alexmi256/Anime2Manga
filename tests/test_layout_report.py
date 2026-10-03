"""Tests for the layout trial renderer/report (``anime2manga.layout_report``)."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from anime2manga.layout import (
    FrameMeta,
    LayoutConfig,
    PanelPlan,
    Row,
    paginate,
)
from anime2manga.layout_report import (
    PanelRenderer,
    evaluate_set,
    render_clean_set_html,
    render_index_html,
    render_set_html,
    rows_to_dict,
)
from anime2manga.models import DetectionBox

LAYOUT = LayoutConfig()


def _meta(index: int = 1, path: Path | None = None) -> FrameMeta:
    return FrameMeta(
        index=index,
        source_size=(160, 90),
        heads=(DetectionBox(60, 30, 20, 20, 0.9),),
        seam_carve_shrink=0.0,
        frame_path=str(path) if path else None,
    )


def _crop_plan(index: int = 1) -> PanelPlan:
    return PanelPlan(
        scene_index=index,
        side="left",
        action="crop 25%",
        carve_frac=0.0,
        crop_frac=0.25,
        crop_x_frac=0.375,
        output_units=1.0,
        score=1.0,
        rule_ids=("C1",),
        method_ids=("M2",),
        reason="",
        is_panorama=False,
        solo=False,
    )


def test_panel_renderer_crops_to_the_requested_window(tmp_path: Path):
    image = np.zeros((90, 160, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "scene_0001.jpg"), image)
    renderer = PanelRenderer(frames_dir=tmp_path, work_dir=tmp_path, working_width=160)
    meta = _meta(path=tmp_path / "scene_0001.jpg")
    panel = renderer.panel(meta, _crop_plan())
    assert panel.shape[1] == 120  # 160 * (1 - 0.25)


def test_write_panel_creates_a_file(tmp_path: Path):
    cv2.imwrite(str(tmp_path / "scene_0001.jpg"), np.zeros((90, 160, 3), np.uint8))
    renderer = PanelRenderer(frames_dir=tmp_path, work_dir=tmp_path, working_width=160)
    out = tmp_path / "out" / "panel.jpg"
    width, height = renderer.write_panel(_meta(path=tmp_path / "scene_0001.jpg"), _crop_plan(), out)
    assert out.exists() and width == 120 and height == 90


def test_render_set_html_captions_panels_with_set_and_rules():
    frame = _meta()
    rows = [Row((_crop_plan(1),), "first row has no subjects", "K2")]
    metrics = evaluate_set([frame], rows, LAYOUT)
    html = render_set_html(
        "K-A",
        "Count-K x Place-A",
        "Count-K",
        "Place-A",
        paginate(rows, 3),
        metrics=metrics,
        panel_urls={1: "panels/k_a/scene_0001.jpg"},
        layout=LAYOUT,
    )
    assert "K-A" in html
    assert "scene 1" in html
    assert "C1" in html  # the rule id is shown
    assert "panels/k_a/scene_0001.jpg" in html
    assert html.count("<table") >= 1


def test_render_clean_set_html_has_no_rules_or_row_info():
    rows = [
        Row((_crop_plan(1), _crop_plan(2)), "two frames", "default"),
        Row((_crop_plan(3),), "one frame", "K_odd"),
    ]
    info = {
        1: ("panels/x/scene_0001.jpg", 4 / 3),   # 0.75 units, as a real pair would be
        2: ("panels/x/scene_0002.jpg", 4 / 3),
        3: ("panels/x/scene_0003.jpg", 16 / 9),  # solo frame at natural width
    }
    html = render_clean_set_html("K-A", paginate(rows, 3), info, page_width=600)
    assert "cleanpage" in html
    assert "panels/x/scene_0001.jpg" in html
    assert "gap: 8px" in html  # white gutter between panels
    assert "figcaption" not in html  # no captions
    assert "rules" not in html  # no rule text
    assert "K_odd" not in html and "K2" not in html  # no row/count info
    # Every image in a row shares one height.
    heights = re.findall(r"height:([0-9.]+)px", html)
    assert len(heights) == 3
    assert heights[0] == heights[1]  # paired row
    assert heights[1] == heights[2]  # solo row is the same row height


def test_render_layout_index_html_lists_options():
    from anime2manga.layout_report import render_layout_index_html

    rows = [Row((_crop_plan(1),), "one", "K_odd")]
    info = {1: ("panels/k_a/scene_0001.jpg", 16 / 9)}
    html = render_layout_index_html(
        "K-B",
        "Default set.",
        [("K-B", "Default set."), ("K-G", "No seam carving.")],
        paginate(rows, 3),
        info,
        page_width=600,
    )
    assert "K-B" in html
    assert "Other layout options" in html
    assert "K-G" in html and "No seam carving." in html
    assert "cleanpage" in html
    assert "figcaption" not in html


def test_render_layout_index_html_stacks_bubble_overlay():
    from anime2manga.layout_report import render_layout_index_html

    rows = [Row((_crop_plan(1),), "one", "K_odd")]
    info = {1: ("panels/k_a/scene_0001.jpg", 16 / 9)}
    html = render_layout_index_html(
        "K-B",
        "Default set.",
        [],
        paginate(rows, 3),
        info,
        page_width=600,
        bubble_urls={1: "panels/k_a/scene_0001.png"},
    )
    # The overlay PNG is stacked over the panel inside a positioned cell.
    assert 'class="cell"' in html
    assert 'class="bubble"' in html
    assert "panels/k_a/scene_0001.png" in html


def test_render_layout_index_html_without_bubbles_has_no_overlay():
    from anime2manga.layout_report import render_layout_index_html

    rows = [Row((_crop_plan(1),), "one", "K_odd")]
    info = {1: ("panels/k_a/scene_0001.jpg", 16 / 9)}
    html = render_layout_index_html("K-B", "Default.", [], paginate(rows, 3), info)
    assert 'class="bubble"' not in html


def test_evaluate_set_reports_counts():
    frame = _meta()
    rows = [
        Row((_crop_plan(1), _crop_plan(2)), "two frames", "default"),
        Row((_crop_plan(3),), "last frame", "K_odd"),
    ]
    metrics = evaluate_set([frame], rows, LAYOUT)
    assert metrics["panels"] == 3
    assert metrics["single_rows"] == 1
    assert metrics["single_reasons"] == {"K_odd": 1}
    assert metrics["head_cuts"] == 0


def test_render_index_links_each_set():
    frame = _meta()
    rows = [Row((_crop_plan(1),), "one", "K2")]
    metrics = evaluate_set([frame], rows, LAYOUT)
    html = render_index_html(
        [{"name": "K-A", "file": "k_a.html", "count": "Count-K", "place": "Place-A", "metrics": metrics}]
    )
    assert "k_a.html" in html
    assert "K-A" in html


def test_rows_to_dict_round_trips():
    rows = [Row((_crop_plan(1),), "one", "K2")]
    payload = rows_to_dict(rows)
    assert payload[0]["count"] == 1
    assert payload[0]["panels"][0]["scene_index"] == 1
    assert payload[0]["panels"][0]["crop_frac"] == 0.25
