"""Tests for the pure panel-layout planner (``anime2manga.layout``)."""

from __future__ import annotations

from anime2manga.layout import (
    COUNT_POLICIES,
    DEFAULT_SETS,
    PLACE_POLICIES,
    CountContext,
    FrameMeta,
    LayoutConfig,
    PanelPlan,
    frames_per_row,
    geometry,
    head_cut,
    meta_from_scene,
    paginate,
    plan_rows,
    plan_solo,
    plan_two,
    removable_units,
    score_frame,
    subject_center_error,
)
from anime2manga.models import DetectionBox

LAYOUT = LayoutConfig()


def _head(x: int = 860, y: int = 300, w: int = 200, h: int = 200) -> DetectionBox:
    return DetectionBox(x=x, y=y, width=w, height=h, confidence=0.9)


def _frame(
    index: int = 1,
    *,
    size: tuple[int, int] = (1920, 1080),
    heads: tuple[DetectionBox, ...] = (),
    persons: tuple[DetectionBox, ...] = (),
    body_percent: float = 0.0,
    head_percent: float = 0.0,
    overlap: float = 0.0,
    sigma: float = 0.0,
    pano: bool = False,
    known: bool = True,
    time: float = 1.0,
) -> FrameMeta:
    return FrameMeta(
        index=index,
        source_size=size,
        heads=heads,
        persons=persons,
        body_percent=body_percent,
        head_percent=head_percent,
        overlap_percent=overlap,
        seam_carve_shrink=sigma,
        seam_carve_known=known,
        is_panorama=pano,
        frame_time=time,
    )


def test_unit_and_geometry_for_a_centred_head():
    frame = _frame(heads=(_head(),), body_percent=10.0, head_percent=4.0, overlap=4.0)
    assert frame.unit == 1.0
    assert frame.cover == 10.0  # 10 + 4 - 4
    geo = geometry(frame, LAYOUT)
    assert geo.has_subject
    assert geo.k_min == 0.40  # clamped to keep_floor
    assert geo.kappa == 0.60
    assert abs(geo.center_frac - 0.5) < 1e-9


def test_no_subject_geometry_uses_keep_floor():
    geo = geometry(_frame(), LAYOUT)
    assert not geo.has_subject
    assert geo.k_min == LAYOUT.keep_floor
    assert geo.center_frac == 0.5


def test_score_rules_fire_as_expected():
    policy = PLACE_POLICIES["Place-A"]
    frame = _frame(heads=(_head(),), body_percent=10.0, head_percent=4.0, overlap=4.0, sigma=0.30)
    geo = geometry(frame, LAYOUT)
    points, fired = score_frame(frame, geo, LAYOUT, policy)
    assert set(fired) == {"C1", "C2", "C8", "S1"}  # C8: single subject fits a tight crop
    assert points == 5.0

    scenery = _frame(index=2, sigma=0.05)
    points, fired = score_frame(scenery, geometry(scenery, LAYOUT), LAYOUT, policy)
    assert fired == ("C7", "S3")  # zero-point rules are still reported
    assert points == 2.0


def test_c8_rewards_a_croppable_single_subject():
    policy = PLACE_POLICIES["Place-A"]
    # A single large-but-contained close-up is a *good* crop target, not a hero.
    frame = _frame(heads=(_head(x=760, y=300, w=400, h=400),), head_percent=8.0, sigma=0.05)
    _points, fired = score_frame(frame, geometry(frame, LAYOUT), LAYOUT, policy)
    assert "C8" in fired

    # A single subject whose union already spans the frame still fits (capped).
    wide = _frame(heads=(_head(x=60, y=200, w=1600, h=500),), head_percent=30.0, sigma=0.05)
    assert geometry(wide, LAYOUT).k_min <= LAYOUT.max_keep
    assert "C8" not in score_frame(wide, geometry(wide, LAYOUT), LAYOUT, policy)[1]


def test_default_pair_counts_two():
    a = _frame(1, heads=(_head(),), sigma=0.30)
    b = _frame(2, heads=(_head(),), sigma=0.20)
    decision = frames_per_row(a, b, CountContext(), LAYOUT, COUNT_POLICIES["Count-K"])
    assert decision.count == 2
    assert decision.rule_id == "default"


def test_k2_first_row_with_scenery_is_single():
    a = _frame(1, sigma=0.30)  # no detections
    b = _frame(2, heads=(_head(),), sigma=0.30)
    decision = frames_per_row(a, b, CountContext(page_row=0), LAYOUT, COUNT_POLICIES["Count-K"])
    assert decision.count == 1
    assert decision.rule_id == "K2"
    # Not the first row -> default two.
    decision = frames_per_row(a, b, CountContext(page_row=1), LAYOUT, COUNT_POLICIES["Count-K"])
    assert decision.count == 2


def test_k3_low_combined_carve_budget_is_single():
    a = _frame(1, heads=(_head(),), sigma=0.02)
    b = _frame(2, heads=(_head(),), sigma=0.03)
    decision = frames_per_row(a, b, CountContext(page_row=1), LAYOUT, COUNT_POLICIES["Count-K"])
    assert decision.count == 1
    assert decision.rule_id == "K3"
    # Just above the 0.07 sum threshold -> the pair is kept.
    c = _frame(3, heads=(_head(),), sigma=0.05)
    d = _frame(4, heads=(_head(),), sigma=0.05)
    assert frames_per_row(c, d, CountContext(page_row=1), LAYOUT, COUNT_POLICIES["Count-K"]).count == 2


def test_wide_panorama_forces_single_row():
    pano = _frame(1, size=(3840, 1080), pano=True)
    regular = _frame(2, heads=(_head(),), sigma=0.30)
    decision = frames_per_row(pano, regular, CountContext(page_row=1), LAYOUT, COUNT_POLICIES["Count-Always2"])
    assert decision.count == 1
    assert decision.rule_id == "K1"


def test_wide_panorama_is_only_above_16_9():
    from anime2manga.layout import is_wide_panorama

    assert not is_wide_panorama(_frame(1, size=(1920, 1080), pano=True))  # exactly 16:9
    assert not is_wide_panorama(_frame(2, size=(1600, 1080), pano=True))  # 1.48
    assert is_wide_panorama(_frame(3, size=(1930, 1080), pano=True))  # just over 16:9
    assert not is_wide_panorama(_frame(4, size=(1920, 1080), pano=False))


def test_non_wide_panorama_that_does_not_fit_is_infeasible_not_wide():
    pano = _frame(1, size=(1600, 1000), pano=True)  # aspect 1.6 (< 16:9)
    huge = DetectionBox(x=0, y=0, width=1900, height=1000, confidence=0.9)
    regular = _frame(2, persons=(huge,), body_percent=80.0, sigma=0.0)
    decision = frames_per_row(pano, regular, CountContext(page_row=1), LAYOUT, COUNT_POLICIES["Count-Always2"])
    assert decision.count == 1
    assert decision.rule_id == "K_infeasible"


def test_two_wide_subjects_hard_crop_into_one_row():
    # Two subjects whose unions span their frames are no longer "infeasible";
    # the floor cap lets them share a row with a hard crop.
    wide = DetectionBox(x=0, y=0, width=1900, height=1000, confidence=0.9)
    a = _frame(1, persons=(wide,), body_percent=80.0, sigma=0.0)
    b = _frame(2, persons=(wide,), body_percent=80.0, sigma=0.0)
    assert geometry(a, LAYOUT).k_min <= LAYOUT.max_keep
    decision = frames_per_row(a, b, CountContext(page_row=1), LAYOUT, COUNT_POLICIES["Count-Always2"])
    assert decision.count == 2
    plan_a, plan_b = plan_two(a, b, LAYOUT, PLACE_POLICIES["Place-A"])
    assert plan_a.crop_frac > 0 and plan_b.crop_frac > 0
    assert abs(plan_a.output_units + plan_b.output_units - LAYOUT.row_width_units) < 1e-6


def test_square_panorama_can_share_a_row():
    pano = _frame(1, size=(1080, 1080), pano=True)
    regular = _frame(2, heads=(_head(),), sigma=0.30)
    decision = frames_per_row(pano, regular, CountContext(page_row=1), LAYOUT, COUNT_POLICIES["Count-K"])
    assert decision.count == 2

    plan_a, plan_b = plan_two(pano, regular, LAYOUT, PLACE_POLICIES["Place-A"])
    assert plan_a.carve_frac == 0.0 and plan_a.crop_frac == 0.0
    assert plan_a.output_units == pano.unit
    assert plan_b.output_units < regular.unit  # regular absorbs the whole deficit
    assert abs(plan_a.output_units + plan_b.output_units - LAYOUT.row_width_units) < 1e-6


def test_plan_two_shrinks_both_and_respects_budget():
    a = _frame(1, heads=(_head(),), body_percent=10.0, head_percent=4.0, overlap=4.0, sigma=0.30)
    b = _frame(2, sigma=0.0)  # scenery
    plan_a, plan_b = plan_two(a, b, LAYOUT, PLACE_POLICIES["Place-A"])
    assert plan_a.crop_frac > 0 and plan_b.crop_frac > 0  # both shrink
    assert plan_a.output_units <= a.unit + 1e-9
    assert plan_b.output_units <= b.unit + 1e-9
    assert abs(plan_a.output_units + plan_b.output_units - LAYOUT.row_width_units) < 1e-6
    assert not head_cut(a, plan_a)
    error = subject_center_error(a, plan_a, LAYOUT)
    assert error is not None
    assert error < 0.05


def test_plan_two_does_nothing_when_it_already_fits():
    # Two 4:3 frames already fill the 1.5-unit row, so nothing is cropped/carved.
    a = _frame(1, size=(1440, 1080), heads=(_head(),), sigma=0.0)
    b = _frame(2, size=(1440, 1080), heads=(_head(),), sigma=0.0)
    plan_a, plan_b = plan_two(a, b, LAYOUT, PLACE_POLICIES["Place-A"])
    assert plan_a.carve_frac == 0.0 and plan_a.crop_frac == 0.0
    assert plan_b.carve_frac == 0.0 and plan_b.crop_frac == 0.0
    assert plan_a.output_units + plan_b.output_units <= LAYOUT.row_width_units + 1e-9


def test_plan_solo_regular_is_natural_and_panorama_scales():
    regular = _frame(1, heads=(_head(),), sigma=0.0)
    solo = plan_solo(regular, LAYOUT)
    assert solo.solo and solo.output_units == regular.unit
    assert solo.carve_frac == 0.0 and solo.crop_frac == 0.0

    pano = _frame(2, size=(3840, 1080), pano=True)
    solo_pano = plan_solo(pano, LAYOUT)
    assert solo_pano.is_panorama
    assert solo_pano.output_units == LAYOUT.row_width_units  # scaled to the row
    assert solo_pano.crop_frac == 0.0


def test_head_cut_detects_an_off_subject_crop():
    frame = _frame(1, heads=(_head(),))
    good = PanelPlan(1, "left", "crop", 0.0, 0.25, 0.375, 0.75, 0.0, ("C1",), ("M2",), "", False, False)
    assert not head_cut(frame, good)
    bad = PanelPlan(1, "left", "crop", 0.0, 0.25, 1.0, 0.75, 0.0, ("C1",), ("M2",), "", False, False)
    assert head_cut(frame, bad)


def test_removable_units_matches_the_applied_cap():
    scenery = _frame(1, sigma=0.0)
    geo = geometry(scenery, LAYOUT)
    # Scenery crop is capped, so the removable width is the empty_crop_cap share.
    assert abs(removable_units(scenery, geo, LAYOUT) - LAYOUT.empty_crop_cap) < 1e-9


def test_max_keep_caps_a_wide_subject_floor():
    wide = _frame(1, persons=(DetectionBox(0, 0, 1900, 1000, 0.9),), body_percent=80.0)
    geo = geometry(wide, LAYOUT)
    assert geo.k_min == LAYOUT.max_keep  # not ~1.0, so two frames still fit


def test_carve_gate_skips_a_low_budget():
    a = _frame(1, heads=(_head(),), sigma=0.05)  # below Place-A's 0.10 gate
    b = _frame(2, sigma=0.0)
    plan_a, _plan_b = plan_two(a, b, LAYOUT, PLACE_POLICIES["Place-A"])
    assert plan_a.carve_frac == 0.0  # gated off -> crop only
    assert plan_a.crop_frac > 0


def test_crop_only_policy_never_carves():
    a = _frame(1, heads=(_head(),), body_percent=10.0, head_percent=4.0, overlap=4.0, sigma=0.30)
    b = _frame(2, sigma=0.0)
    plan_a, plan_b = plan_two(a, b, LAYOUT, PLACE_POLICIES["Place-G"])
    assert plan_a.carve_frac == 0.0 and plan_b.carve_frac == 0.0
    assert plan_a.crop_frac > 0 and plan_b.crop_frac > 0
    assert "M1" not in plan_a.method_ids


def test_plan_rows_walks_a_sequence_and_paginates():
    frames = [
        _frame(1, sigma=0.04),  # scenery, first row
        _frame(2, heads=(_head(),), head_percent=9.0, overlap=4.0, body_percent=10.0, sigma=0.30),
        _frame(3, persons=(DetectionBox(0, 0, 1900, 1000, 0.9),), body_percent=80.0, sigma=0.0),
        _frame(4, size=(3840, 1080), pano=True),
        _frame(5, sigma=0.05),
    ]
    rows = plan_rows(frames, LAYOUT, COUNT_POLICIES["Count-K"], PLACE_POLICIES["Place-A"])
    assert len(rows) == 4
    assert sum(len(row.panels) for row in rows) == 5
    assert rows[0].count_id == "K2"  # first row scenery
    assert rows[2].count_id == "K1"  # wide panorama
    pages = paginate(rows, 3)
    assert [len(page.rows) for page in pages] == [3, 1]


def test_tall_panorama_spans_two_rows():
    tall = _frame(1, size=(1080, 1920), pano=True)  # portrait panorama
    regular = _frame(2, heads=(_head(),), sigma=0.20)
    rows = plan_rows([tall, regular], LAYOUT, COUNT_POLICIES["Count-K"], PLACE_POLICIES["Place-A"])
    assert rows[0].count_id == "K_tall"
    assert rows[0].row_span == 2
    assert rows[0].panels[0].is_panorama
    pages = paginate(rows, 3)
    # The panorama (2 slots) plus the regular frame (1 slot) make one full page.
    assert len(pages) == 1
    assert sum(row.row_span for row in pages[0].rows) == 3


def test_unknown_seam_carve_skips_carve_budget_rules():
    # Carving disabled: shrink is unknown, not zero, so K3 must not force a
    # single row for every pair.
    a = _frame(1, heads=(_head(),), sigma=0.0, known=False)
    b = _frame(2, heads=(_head(),), sigma=0.0, known=False)
    decision = frames_per_row(a, b, CountContext(page_row=1), LAYOUT, COUNT_POLICIES["Count-K"])
    assert decision.count == 2


def test_meta_from_scene_marks_unknown_shrink_and_falls_back_to_faces():
    from anime2manga.models import DetectionBox, Scene

    scene = Scene(index=5, start=0.0, end=1.0, fps=24.0)
    scene.frame_size = (1920, 1080)
    scene.seam_carve_shrink = None  # carving disabled
    scene.faces = [DetectionBox(100, 100, 50, 50, 0.9)]
    meta = meta_from_scene(scene)
    assert meta.seam_carve_known is False
    assert meta.heads and meta.heads[0].x == 100  # faces used when no head

    scene.seam_carve_shrink = 0.2
    assert meta_from_scene(scene).seam_carve_known is True


def test_hard_crop_is_reported_when_the_cap_binds():
    # A near-full-width subject union caps k_min at max_keep; fitting two such
    # frames must surface the hard crop (action/reason), not silently drop it.
    wide = DetectionBox(0, 0, 1900, 1000, 0.9)
    a = _frame(1, persons=(wide,), body_percent=80.0, sigma=0.0)
    b = _frame(2, persons=(wide,), body_percent=80.0, sigma=0.0)
    assert geometry(a, LAYOUT).capped is True
    plan_a, _plan_b = plan_two(a, b, LAYOUT, PLACE_POLICIES["Place-B"])
    assert "hard crop" in plan_a.action
    assert "hard crop" in plan_a.reason


def test_default_sets_resolve_to_known_policies():
    for name, count_name, place_name in DEFAULT_SETS:
        assert name
        assert count_name in COUNT_POLICIES
        assert place_name in PLACE_POLICIES
