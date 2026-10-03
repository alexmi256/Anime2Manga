"""Tests for step-10 speech-bubble lettering (``anime2manga.speech_bubbles``)."""

from __future__ import annotations

import numpy as np

from anime2manga import speech_bubbles as sb
from anime2manga.models import DetectionBox


def test_classify_shape_is_rounded_for_every_line():
    # Shapes are intentionally uniform for now; the enum is kept for later.
    for text in ("WHAT?!", "Is this a thought...", "A short line.", "long line without stop"):
        assert sb.classify_shape(text) is sb.BubbleShape.ROUNDED


def test_plan_bubbles_is_deterministic_and_fits_the_panel():
    args = ((800, 450), ["First line.", "Second line here."])
    a = sb.plan_bubbles(*args, boxes=[], audio_focus="left")
    b = sb.plan_bubbles(*args, boxes=[], audio_focus="left")
    assert a.specs == b.specs
    assert len(a.specs) == 2
    for spec in a.specs:
        x, y, w, h = spec.box
        assert x >= 0 and y >= 0
        assert x + w <= 800 + 1e-6
        assert y + h <= 450 + 1e-6


def test_plan_bubbles_never_overlaps_two_bubbles():
    layout = sb.plan_bubbles(
        (800, 450), ["A first bubble.", "A second bubble."], boxes=[], audio_focus="center"
    )
    (x0, y0, w0, h0), (x1, y1, w1, h1) = (s.box for s in layout.specs)
    # Text boxes must not sit on top of one another.
    assert x0 + w0 <= x1 or x1 + w1 <= x0 or y0 + h0 <= y1 or y1 + h1 <= y0


def test_plan_bubbles_keeps_area_within_budget():
    layout = sb.plan_bubbles(
        (800, 450),
        ["One.", "Two.", "Three.", "Four.", "Five."],
        boxes=[],
        audio_focus="center",
        max_bubbles=3,
    )
    assert len(layout.specs) <= 3
    assert layout.area_frac <= sb.TEXT_BUDGET + 1e-6
    assert any("capped" in note for note in layout.notes)


def test_plan_bubbles_empty_returns_no_specs():
    layout = sb.plan_bubbles((400, 300), [], boxes=[], audio_focus="center")
    assert layout.has_bubbles is False
    assert layout.specs == ()


def test_map_boxes_to_panel_normalises_source_resolution():
    # 1920-wide source, no crop, panel is half-width: a box at x=480..960 maps
    # to x=120..240 and y scales too.
    box = DetectionBox(x=480, y=270, width=480, height=540, confidence=0.9)
    mapped = sb.map_boxes_to_panel([box], (960, 540), (1920, 1080), 0.0, 0.0)
    assert len(mapped) == 1
    out = mapped[0]
    assert out.x == 240 and out.width == 240
    assert out.y == 135 and out.height == 270


def test_map_boxes_to_panel_crops_and_shifts():
    # 1920-wide source, keep the middle 50% (crop_x_frac=0.25), panel 960 wide.
    # A box at source x=768..1152 (fractions 0.4..0.6) lands at panel
    # ((0.4-0.25)*1920, (0.6-0.25)*1920) = x=288, width=384.
    box = DetectionBox(x=768, y=108, width=384, height=216, confidence=0.9)
    mapped = sb.map_boxes_to_panel([box], (960, 540), (1920, 1080), 0.5, 0.25)
    assert len(mapped) == 1
    out = mapped[0]
    assert out.x == 288 and out.width == 384
    assert out.y == 54 and out.height == 108


def test_map_boxes_to_panel_preserves_box_fractions():
    # Carve-invariance is structural: the helper takes no carve term, so the
    # mapped box must preserve the source fractions.  Check that explicitly for
    # a box at fractions 0.25..0.5 with no crop.
    box = DetectionBox(x=480, y=270, width=480, height=540)
    mapped = sb.map_boxes_to_panel([box], (960, 540), (1920, 1080), 0.0, 0.0)
    assert len(mapped) == 1
    out = mapped[0]
    # x/width fractions 0.25/0.25 and y/height 0.25/0.5 of the panel.
    assert out.x == 240 and out.width == 240
    assert out.y == 135 and out.height == 270


def test_map_boxes_to_panel_drops_fully_cropped_box():
    box = DetectionBox(x=0, y=0, width=100, height=100)
    # Keep only the right half: a box pinned to the left edge is cropped away.
    assert sb.map_boxes_to_panel([box], (600, 400), (1920, 1080), 0.5, 0.5) == []


def test_build_svg_is_transparent_and_contains_text():
    layout = sb.plan_bubbles((500, 300), ["Hello there."], boxes=[], audio_focus="left")
    svg = sb.build_svg(layout)
    assert "<svg" in svg and "</svg>" in svg
    assert "Hello there." in svg
    # Transparent overlay by default: no opaque background rect.
    assert "<rect" not in svg
    opaque = sb.build_svg(layout, background=True)
    assert "<rect" in opaque


def test_build_svg_escapes_markup():
    layout = sb.plan_bubbles((500, 300), ["a < b & c"], boxes=[], audio_focus="left")
    svg = sb.build_svg(layout)
    assert "&lt;" in svg and "&amp;" in svg


def test_write_panel_overlay_writes_svg_and_png(tmp_path):
    layout = sb.plan_bubbles((320, 180), ["Hi."], boxes=[], audio_focus="left")
    written = sb.write_panel_overlay(layout, tmp_path, "scene_0001")
    assert written is not None
    svg_path, png_path = written
    assert svg_path.exists() and png_path.exists()
    assert svg_path.suffix == ".svg" and png_path.suffix == ".png"


def test_write_panel_overlay_is_none_without_bubbles(tmp_path):
    layout = sb.plan_bubbles((320, 180), [], boxes=[])
    assert sb.write_panel_overlay(layout, tmp_path, "scene_0001") is None


def test_flatten_panel_composites_the_overlay(tmp_path):
    import cv2

    panel = tmp_path / "scene_0001.jpg"
    cv2.imwrite(str(panel), np.zeros((200, 360, 3), np.uint8))
    layout = sb.plan_bubbles((360, 200), ["Visible text."], boxes=[], audio_focus="left")
    sb.flatten_panel(panel, layout)
    out = cv2.imread(str(panel))
    # The white balloon must have changed at least some pixels.
    assert out is not None and out.max() > 200
    # No stray temporary overlay is left behind.
    assert not (tmp_path / "scene_0001.overlay.png").exists()
