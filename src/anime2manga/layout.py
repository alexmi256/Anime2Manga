"""Panel layout planning for step 9: fitting frames into a 3x2 page grid.

The page is three rows tall and holds up to two frames per row, giving at most
six panels per page.  Row height is constant; per-panel widths vary and must sum
to a row budget measured in **16:9-at-row-height units** (one full 16:9 frame is
``1.0``, the row is ``R = 1.5``).

The design is deliberately split into two functions, because a frame count
decision is much simpler than the coupled width allocation:

* :func:`frames_per_row` answers only "does this row hold one frame or two?".
  It is a catalog of exception rules (``K*``) that force one panel; the default
  is two.  A wide panorama, an odd tail and an infeasible pair are hard,
  always-on exceptions.
* :func:`plan_two` places exactly two frames, solving ``w_a + w_b <= R`` with
  the score rules (``C*``/``S*``/``P*``/``H*``) and method rules (``M*``).
* :func:`plan_solo` places a single frame (centred; a panorama is uniformly
  downscaled to the row width, never cropped).
* :func:`plan_rows` walks the sequence, calls :func:`frames_per_row`, dispatches,
  and chunks rows into pages of three.

A rule set is therefore a **pair**: a :class:`CountPolicy` and a
:class:`PlacePolicy`.  The same placement policy can be trialled under several
frame-count policies, which is the point of the split.

Everything here is pure and box-only: no image pixels are touched and the
module imports only :mod:`anime2manga.models`.  Rendering lives in
:mod:`anime2manga.layout_report`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .models import DetectionBox, Scene

#: A 16:9 frame's width:height ratio; the unit of the row budget.
UNIT_ASPECT = 16.0 / 9.0


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class LayoutConfig:
    """Geometry and thresholds shared by every rule set.

    ``keep_floor``/``pad_frac`` etc. are intentionally here rather than on a
    placement policy so the count and placement functions always agree on what
    "fits" means.  A denser rule set uses a different :class:`LayoutConfig`
    (e.g. a lower ``keep_floor``), which the harness supplies.
    """

    #: Row width budget: 1.5 = one and a half 16:9 frames.
    row_width_units: float = 1.5
    row_height: int = 1080
    #: Subject crop floor: never keep less than this fraction of the width.
    keep_floor: float = 0.40
    #: Hard cap on the subject-protecting floor.  A wide subject union (or an
    #: oversized box) otherwise demands keeping nearly the whole width, which
    #: makes a pair look "infeasible"; capping it means the frame is *hard
    #: cropped* instead and two regular 16:9 frames always fit one row
    #: (2 x 0.75 = 1.5).  0.75 is exactly the equal-share width for the row.
    max_keep: float = 0.75
    #: Padding kept around a subject, as a fraction of frame height / head width.
    pad_frac: float = 0.06
    pad_frac_head: float = 0.35
    #: A frame may take at most this share of the required reduction.
    max_share: float = 0.70
    #: The other frame's width below which this one would be a sliver (units).
    sliver_units: float = 0.35
    #: Maximum crop applied to a frame with no detected subject.
    empty_crop_cap: float = 0.35
    #: Bounding boxes below this detection confidence never anchor a crop.
    confidence_floor: float = 0.35
    #: Carve-capacity tiers (fractions of width).
    t_carve_high: float = 0.25
    t_carve_mid: float = 0.15
    #: Maximum gaze bias when cropping (fraction of width).
    delta_max: float = 0.08


@dataclass(frozen=True)
class FrameMeta:
    """Everything a layout rule is allowed to look at for one frame."""

    index: int
    source_size: tuple[int, int]
    heads: tuple[DetectionBox, ...] = ()
    persons: tuple[DetectionBox, ...] = ()
    body_percent: float = 0.0
    head_percent: float = 0.0
    overlap_percent: float = 0.0
    heads_in_body: bool | None = None
    body_lean: str | None = None
    head_lean: str | None = None
    seam_carve_shrink: float = 0.0
    is_panorama: bool = False
    panorama_size: tuple[int, int] | None = None
    #: False when the frame was never seam-carved (carving disabled, or a
    #: panorama), so its zero shrink must not be read as "cannot be carved".
    seam_carve_known: bool = True
    subtitle_count: int = 0
    frame_time: float | None = None
    frame_path: str | None = None
    panorama_path: str | None = None

    @property
    def width(self) -> int:
        return self.source_size[0]

    @property
    def height(self) -> int:
        return self.source_size[1]

    @property
    def unit(self) -> float:
        """Natural width in 16:9-at-row-height units (1920x1080 -> 1.0)."""
        return (self.width / self.height) / UNIT_ASPECT

    @property
    def cover(self) -> float:
        """Union subject coverage (body + head - overlap), clamped to 0..100."""
        return max(0.0, min(100.0, self.body_percent + self.head_percent - self.overlap_percent))

    @property
    def boxes(self) -> tuple[DetectionBox, ...]:
        return (*self.heads, *self.persons)

    @property
    def n_heads(self) -> int:
        return len(self.heads)

    @property
    def n_persons(self) -> int:
        return len(self.persons)

    @property
    def has_subject(self) -> bool:
        return bool(self.heads or self.persons)

    @property
    def single_subject(self) -> bool:
        """One head-or-body subject; a head plus its body still counts as one."""
        return (self.n_heads + self.n_persons) > 0 and max(self.n_heads, self.n_persons) <= 1

    @property
    def multiple_subjects(self) -> bool:
        return self.n_heads >= 2 or self.n_persons >= 2


@dataclass(frozen=True)
class Geometry:
    """Derived, resolution-independent geometry for one frame."""

    unit: float
    cover: float
    span: float
    center_frac: float
    k_min: float
    kappa: float
    has_subject: bool
    #: True when the subject union is so wide that the ``max_keep`` cap binds,
    #: i.e. reaching the row budget requires a hard crop through the subject.
    capped: bool = False

    @property
    def min_units(self) -> float:
        """Smallest output width the subject margin allows before carving."""
        return self.unit * self.k_min


def geometry(frame: FrameMeta, layout: LayoutConfig) -> Geometry:
    """Subject-protecting geometry for ``frame`` (see ``docs/layout_rules.md``)."""
    width = frame.width
    height = frame.height
    boxes = [b for b in frame.boxes if b.confidence >= layout.confidence_floor]
    if not boxes:
        return Geometry(
            unit=frame.unit,
            cover=frame.cover,
            span=0.0,
            center_frac=0.5,
            k_min=layout.keep_floor,
            kappa=1.0 - layout.keep_floor,
            has_subject=False,
        )
    x0 = min(b.x for b in boxes)
    x1 = max(b.x + b.width for b in boxes)
    span = (x1 - x0) / width
    center_frac = ((x0 + x1) / 2.0) / width
    head_width = max((b.width for b in frame.heads), default=0)
    pad = max(layout.pad_frac * height, layout.pad_frac_head * head_width)
    raw_keep = (span * width + 2.0 * pad) / width
    capped = raw_keep > layout.max_keep + 1e-9
    k_min = min(max(_clip01(raw_keep), layout.keep_floor), layout.max_keep)
    return Geometry(
        unit=frame.unit,
        cover=frame.cover,
        span=span,
        center_frac=center_frac,
        k_min=k_min,
        kappa=1.0 - k_min,
        has_subject=True,
        capped=capped,
    )


# --- placement policies ------------------------------------------------------

#: Default points per score rule.  A policy may override individual entries.
_DEFAULT_POINTS: dict[str, float] = {
    "C1": 1.0,
    "C2": 1.0,
    "C3": 0.0,
    "C4": -1.0,
    "C5": -1.0,
    "C6": -1.0,
    "C7": 2.0,
    "C8": 1.0,
    "T1": -1.0,
    "S1": 2.0,
    "S2": 1.0,
    "S3": 0.0,
    "P1": 1.0,
    "P2": -1.0,
    "H1": -2.0,
    "H2": -1.0,
    "H3": 3.0,
}

#: Human-readable one-liners used in the trial HTML captions.
RULE_NOTES: dict[str, str] = {
    "PANO": "panorama: never cropped",
    "C1": "single subject <50% (good crop)",
    "C2": "single subject <25% (tiny)",
    "C3": "multiple subjects >=50% (neutral)",
    "C4": "multiple subjects span >=60% (avoid crop)",
    "C5": "subject fills >=70% (avoid crop)",
    "C6": "contained foreground head (protect) [deprecated]",
    "C7": "no detections (scenery)",
    "C8": "single subject fits a tight crop (good crop)",
    "T1": "many subtitles (reserve width)",
    "S1": "carve >=25%",
    "S2": "carve 15-25%",
    "S3": "carve <15% (rigid)",
    "P1": "subject centred",
    "P2": "subject at edge",
    "H1": "large head [deprecated]",
    "H2": "large body [deprecated]",
    "H3": "scenery (scene-setter)",
    "M1": "carve first",
    "M2": "crop centred on subject",
    "M3": "crop keeps all boxes",
    "M4": "crop from outer edge",
    "M5": "gaze bias",
    "M6": "no upscale",
}


@dataclass(frozen=True)
class PlacePolicy:
    """A placement policy: which score and method rules are active."""

    name: str
    score_rules: frozenset[str]
    methods: frozenset[str]
    base_weight: float = 0.5
    max_share: float = 0.70
    #: A frame only seam-carves when its budget reaches this fraction; below it
    #: the frame is hard-cropped instead.  ``1.0`` disables carving entirely.
    carve_min: float = 0.10
    use_gaze: bool = False
    use_balance: bool = False
    points: dict[str, float] = field(default_factory=dict)

    def value(self, rule_id: str) -> float:
        return self.points.get(rule_id, _DEFAULT_POINTS.get(rule_id, 0.0))


def score_frame(
    frame: FrameMeta, geo: Geometry, layout: LayoutConfig, policy: PlacePolicy
) -> tuple[float, tuple[str, ...]]:
    """Sum the active score rules; return ``(points, fired_rule_ids)``.

    Fired ids include zero-point rules (e.g. ``C3``) so the report can show why a
    frame scored nothing.
    """
    points = 0.0
    fired: list[str] = []
    rules = policy.score_rules

    def add(rule_id: str, condition: bool) -> None:
        nonlocal points
        if condition:
            fired.append(rule_id)
            points += policy.value(rule_id)

    if geo.has_subject:
        add("C1", "C1" in rules and geo.cover < 50.0)
        add("C2", "C2" in rules and geo.cover < 25.0)
        add("C3", "C3" in rules and frame.multiple_subjects and geo.cover >= 50.0)
        add("C4", "C4" in rules and frame.multiple_subjects and geo.span >= 0.60)
        add("C5", "C5" in rules and geo.cover >= 70.0)
        add("C6", "C6" in rules and bool(frame.heads_in_body) and frame.head_percent >= 8.0)
        # A single subject that already fits a tight crop is easy to crop around,
        # so it is a *good* shrink target (the opposite of protecting a "hero").
        add("C8", "C8" in rules and frame.single_subject and geo.k_min <= 0.60)
        add("P1", "P1" in rules and abs(geo.center_frac - 0.5) <= 0.12)
        add("P2", "P2" in rules and not 0.15 <= geo.center_frac <= 0.85)
        add("H1", "H1" in rules and frame.head_percent >= 8.0)
        add("H2", "H2" in rules and frame.body_percent >= 50.0)
    else:
        # H3 supersedes C7 when both are enabled (do not double count scenery).
        add("H3", "H3" in rules)
        add("C7", "C7" in rules and "H3" not in rules)
    add("T1", "T1" in rules and frame.subtitle_count >= 4)

    sigma = frame.seam_carve_shrink
    if frame.seam_carve_known:
        add("S1", "S1" in rules and sigma >= layout.t_carve_high)
        add("S2", "S2" in rules and layout.t_carve_mid <= sigma < layout.t_carve_high)
        add("S3", "S3" in rules and sigma < layout.t_carve_mid)
    return points, tuple(fired)


# --- count policies ----------------------------------------------------------


@dataclass(frozen=True)
class CountPolicy:
    """A frame-count policy: which exception rules may force a single-panel row."""

    name: str
    rules: frozenset[str]
    t_carve: float = 0.07
    t_crop: float = 0.20
    t_rigid: float = 0.05


@dataclass(frozen=True)
class CountContext:
    """Row position within its page (0-based), for the first-row rule."""

    page_row: int = 0


@dataclass(frozen=True)
class CountDecision:
    count: int
    rule_id: str
    reason: str


def _min_units(frame: FrameMeta, geo: Geometry, layout: LayoutConfig) -> float:
    """Smallest output width the frame can reach with a **crop only**.

    Carving is ignored here so feasibility is conservative with respect to a
    placement policy that gates carving off.  A panorama is untouchable, so its
    minimum is its full natural width; every other frame can always be hard
    cropped down to its (capped) subject floor, which is why two regular 16:9
    frames are always feasible.
    """
    if frame.is_panorama:
        return geo.unit
    return geo.unit * _min_kept(geo, layout)


def _min_kept(geo: Geometry, layout: LayoutConfig) -> float:
    """Smallest kept width fraction: subject floor, or the scenery crop cap."""
    if geo.has_subject:
        return geo.k_min
    return max(geo.k_min, 1.0 - layout.empty_crop_cap)


def removable_units(
    frame: FrameMeta, geo: Geometry, layout: LayoutConfig, carve_min: float = 0.0
) -> float:
    """Width the frame can lose: carve ``sigma`` (if allowed) then crop to floor.

    ``carve_min`` is the policy's carve gate: a frame whose ``sigma`` is below it
    does not carve, so its only capacity is crop.  A panorama can lose nothing.
    """
    if frame.is_panorama:
        return 0.0
    sigma = max(0.0, frame.seam_carve_shrink)
    if sigma < carve_min:
        sigma = 0.0
    return max(0.0, geo.unit * (1.0 - (1.0 - sigma) * _min_kept(geo, layout)))


def is_wide_panorama(frame: FrameMeta) -> bool:
    """True when a panorama's aspect ratio exceeds 16:9, so it takes its own row.

    A square or narrower panorama (aspect <= 16:9) is *not* "wide" and may share
    a row when it fits beside its partner; if it cannot fit, the normal
    infeasibility rule still sends it to a row of its own.
    """
    return frame.is_panorama and frame.unit > 1.0 + 1e-9


def frames_per_row(
    a: FrameMeta,
    b: FrameMeta,
    ctx: CountContext,
    layout: LayoutConfig,
    policy: CountPolicy,
) -> CountDecision:
    """Decide whether the row beginning with ``a`` and ``b`` holds 1 or 2 frames.

    Hard exceptions (a wide panorama, an infeasible pair) always apply; the
    policy's rules add optional exceptions.  The caller handles the odd tail.
    """
    geo_a = geometry(a, layout)
    geo_b = geometry(b, layout)
    budget = layout.row_width_units

    if is_wide_panorama(a) or is_wide_panorama(b):
        return CountDecision(1, "K1", "wide panorama (aspect > 16:9) cannot share a row")

    min_a = _min_units(a, geo_a, layout)
    min_b = _min_units(b, geo_b, layout)
    if min_a + min_b > budget + 1e-9:
        return CountDecision(1, "K_infeasible", "pair cannot fit within subject margins")

    rules = policy.rules

    # A shareable (square/narrow) panorama disables the carve-budget rules, since
    # the regular partner absorbs the whole deficit and stays at 2 frames.
    shareable_pano = (a.is_panorama and not is_wide_panorama(a)) or (
        b.is_panorama and not is_wide_panorama(b)
    )
    # Carve-budget rules need a real shrink value.  When carving was disabled the
    # shrink is unknown (not "zero"), so treating it as zero would force every
    # pair onto its own row; skip them instead.
    carve_known = a.seam_carve_known and b.seam_carve_known

    if "K_establish" in rules and ctx.page_row == 0 and not geo_a.has_subject:
        return CountDecision(1, "K2", "first row of the page; frame A has no subjects")

    if not shareable_pano and carve_known:
        sum_carve = a.seam_carve_shrink + b.seam_carve_shrink
        if "K_carve_sum" in rules and sum_carve < policy.t_carve:
            return CountDecision(1, "K3", "low combined carve budget")
        if "K_carve_min" in rules and min(a.seam_carve_shrink, b.seam_carve_shrink) < policy.t_carve:
            return CountDecision(1, "K3", "a frame has a low carve budget")
        if "K_carve_max" in rules and max(a.seam_carve_shrink, b.seam_carve_shrink) < policy.t_carve:
            return CountDecision(1, "K3", "both frames have low carve budgets")
        if "K_carve_crop" in rules:
            total = a.unit + b.unit
            predicted_crop = (total - budget) / total if total > 0 else 0.0
            if predicted_crop > policy.t_crop and (
                a.seam_carve_shrink < layout.t_carve_high
                and b.seam_carve_shrink < layout.t_carve_high
            ):
                return CountDecision(1, "K3c", "pair would need heavy cropping")

    if "K_both_rigid" in rules and carve_known and (
        a.seam_carve_shrink < policy.t_rigid
        and b.seam_carve_shrink < policy.t_rigid
        and geo_a.cover >= 50.0
        and geo_b.cover >= 50.0
    ):
        return CountDecision(1, "K_both_rigid", "both frames are rigid and dense")
    if "K_sliver" in rules and (
        budget - min_a < layout.sliver_units or budget - min_b < layout.sliver_units
    ):
        return CountDecision(1, "K_sliver", "partner would be forced below the sliver width")

    return CountDecision(2, "default", "two frames")


# --- plans -------------------------------------------------------------------


@dataclass(frozen=True)
class PanelPlan:
    """How one frame is placed in its row (and why)."""

    scene_index: int
    side: str
    action: str
    carve_frac: float
    crop_frac: float
    crop_x_frac: float
    output_units: float
    score: float
    rule_ids: tuple[str, ...]
    method_ids: tuple[str, ...]
    reason: str
    is_panorama: bool
    solo: bool


@dataclass(frozen=True)
class Row:
    """A page row: one or two panels plus the frame-count reason."""

    panels: tuple[PanelPlan, ...]
    count_reason: str
    count_id: str
    #: How many of the page's three row slots this row occupies (a tall
    #: panorama spans two).
    row_span: int = 1

    @property
    def units(self) -> float:
        return sum(p.output_units for p in self.panels)


@dataclass(frozen=True)
class Page:
    index: int
    rows: tuple[Row, ...]


def _action_label(carve: float, crop: float) -> str:
    parts = []
    if carve > 1e-9:
        parts.append(f"carve {carve * 100:.0f}%")
    if crop > 1e-9:
        parts.append(f"crop {crop * 100:.0f}%")
    return " + ".join(parts) if parts else "no resize"


def plan_solo(frame: FrameMeta, layout: LayoutConfig) -> PanelPlan:
    """Place a single frame on its row.

    A regular frame is shown at natural width (never cropped or upscaled); a
    panorama wider than the row is uniformly downscaled to the row width.
    """
    if frame.is_panorama:
        output = min(frame.unit, layout.row_width_units)
        action = "panorama scaled to row" if frame.unit > output + 1e-9 else "panorama"
        return PanelPlan(
            scene_index=frame.index,
            side="solo",
            action=action,
            carve_frac=0.0,
            crop_frac=0.0,
            crop_x_frac=0.0,
            output_units=output,
            score=0.0,
            rule_ids=("PANO",),
            method_ids=(),
            reason="panorama: never cropped; uniform downscale to fit" if frame.unit > output
            else "panorama: fits at natural width",
            is_panorama=True,
            solo=True,
        )
    return PanelPlan(
        scene_index=frame.index,
        side="solo",
        action="natural width, centred",
        carve_frac=0.0,
        crop_frac=0.0,
        crop_x_frac=0.0,
        output_units=frame.unit,
        score=0.0,
        rule_ids=(),
        method_ids=("M6",),
        reason="single frame: centred at natural width, no upscale",
        is_panorama=False,
        solo=True,
    )


def _crop_x(
    frame: FrameMeta,
    geo: Geometry,
    side: str,
    kept: float,
    methods: frozenset[str],
    policy: PlacePolicy,
    layout: LayoutConfig,
) -> tuple[float, tuple[str, ...]]:
    """Left edge of the kept window as a fraction, plus the method ids fired."""
    fired: list[str] = []
    if geo.has_subject:
        if "M2" in methods:
            fired.append("M2")
        if "M3" in methods:
            fired.append("M3")
        x = geo.center_frac - kept / 2.0
        if policy.use_gaze and frame.single_subject:
            lean = frame.head_lean or frame.body_lean
            gutter = "right" if side == "left" else "left"
            # Shifting the window this way opens space on the gutter side.
            gutter_dir = -1.0 if side == "left" else 1.0
            if lean == gutter:
                x += gutter_dir * layout.delta_max
                fired.append("M5")
            elif lean in ("left", "right"):
                x -= gutter_dir * layout.delta_max
                fired.append("M5")
    elif "M4" in methods and side in ("left", "right"):
        fired.append("M4")
        # Remove the outer edge (left panel -> left, right panel -> right) so the
        # inner/gutter side stays open.
        x = 1.0 - kept if side == "left" else 0.0
    else:
        x = 0.5 - kept / 2.0
    return _clip01(x), tuple(fired)


def _apply_reduction(
    frame: FrameMeta,
    geo: Geometry,
    side: str,
    reduction: float,
    layout: LayoutConfig,
    policy: PlacePolicy,
) -> PanelPlan:
    """Apply ``reduction`` units of width to ``frame``: carve (if worth it) then crop."""
    unit = geo.unit
    frac = reduction / unit if unit > 0 else 0.0
    frac = _clip01(frac)
    sigma = max(0.0, frame.seam_carve_shrink)
    carve = min(sigma, frac) if sigma >= policy.carve_min else 0.0
    crop = (
        0.0
        if carve >= frac - 1e-12
        else 1.0 - (1.0 - frac) / max(1e-9, 1.0 - carve)
    )
    crop = min(crop, layout.empty_crop_cap) if not geo.has_subject else min(crop, geo.kappa)
    crop = max(0.0, crop)
    kept = 1.0 - crop
    crop_x, method_ids = _crop_x(frame, geo, side, kept, policy.methods, policy, layout)
    methods: tuple[str, ...] = (("M1",) if carve > 1e-9 else ()) + method_ids + ("M6",)
    # The hard-crop event is when the subject-union cap binds and the crop
    # actually reaches the floor (the union is too wide to protect fully).
    hard = geo.capped and crop > 1e-9 and kept <= geo.k_min + 1e-6
    action = _action_label(carve, crop) + (" (hard crop)" if hard else "")
    return PanelPlan(
        scene_index=frame.index,
        side=side,
        action=action,
        carve_frac=carve,
        crop_frac=crop,
        crop_x_frac=crop_x,
        output_units=unit * (1.0 - carve) * kept,
        score=0.0,
        rule_ids=(),
        method_ids=methods,
        reason="hard crop below subject floor" if hard else "",
        is_panorama=frame.is_panorama,
        solo=False,
    )


def _panel_with_score(
    plan: PanelPlan, score: float, rule_ids: tuple[str, ...]
) -> PanelPlan:
    fired = " ".join(rule_ids) if rule_ids else "-"
    base = plan.reason or plan.action or "no resize"
    return replace(
        plan,
        score=score,
        rule_ids=rule_ids,
        reason=f"{base}; rules {fired}",
    )


def plan_two(
    a: FrameMeta,
    b: FrameMeta,
    layout: LayoutConfig,
    policy: PlacePolicy,
) -> tuple[PanelPlan, PanelPlan]:
    """Place exactly two frames into one row.

    Assumes the count step admitted the pair (so a feasible pair, and any
    panorama is shareable).  Splits the width deficit by each frame's score
    weight, carves first and crops the residue, then optionally balances widths.
    """
    geo_a = geometry(a, layout)
    geo_b = geometry(b, layout)
    unit_a, unit_b = geo_a.unit, geo_b.unit
    budget = layout.row_width_units
    deficit = max(0.0, unit_a + unit_b - budget)

    score_a, rules_a = score_frame(a, geo_a, layout, policy)
    score_b, rules_b = score_frame(b, geo_b, layout, policy)

    if deficit <= 1e-9:
        plan_a = _panel_with_score(_apply_reduction(a, geo_a, "left", 0.0, layout, policy), score_a, rules_a)
        plan_b = _panel_with_score(_apply_reduction(b, geo_b, "right", 0.0, layout, policy), score_b, rules_b)
        return plan_a, plan_b

    weight_a = max(0.0, score_a) + policy.base_weight
    weight_b = max(0.0, score_b) + policy.base_weight
    total_weight = weight_a + weight_b
    share_a = deficit * weight_a / total_weight
    share_b = deficit - share_a

    cap_a = removable_units(a, geo_a, layout, policy.carve_min)
    cap_b = removable_units(b, geo_b, layout, policy.carve_min)
    if a.is_panorama:
        cap_a = 0.0
    if b.is_panorama:
        cap_b = 0.0

    if cap_a > 0.0 and cap_b > 0.0:
        share_a = min(share_a, deficit * policy.max_share)
        share_b = deficit - share_a

    share_a = min(max(share_a, 0.0), cap_a)
    share_b = deficit - share_a
    if share_b > cap_b:
        share_b = cap_b
        share_a = min(deficit - share_b, cap_a)

    plan_a = _apply_reduction(a, geo_a, "left", share_a, layout, policy)
    plan_b = _apply_reduction(b, geo_b, "right", share_b, layout, policy)

    if policy.use_balance and not a.is_panorama and not b.is_panorama:
        plan_a, plan_b = _balance(plan_a, plan_b, a, b, geo_a, geo_b, cap_a, cap_b, layout, policy)

    return (
        _panel_with_score(plan_a, score_a, rules_a),
        _panel_with_score(plan_b, score_b, rules_b),
    )


def _balance(
    plan_a: PanelPlan,
    plan_b: PanelPlan,
    a: FrameMeta,
    b: FrameMeta,
    geo_a: Geometry,
    geo_b: Geometry,
    cap_a: float,
    cap_b: float,
    layout: LayoutConfig,
    policy: PlacePolicy,
) -> tuple[PanelPlan, PanelPlan]:
    """Nudge the split toward equal panel widths when the caps allow it."""
    target = layout.row_width_units / 2.0
    if target > geo_a.unit or target > geo_b.unit:
        return plan_a, plan_b
    reduce_a = geo_a.unit - target
    reduce_b = geo_b.unit - target
    if reduce_a < -1e-9 or reduce_b < -1e-9:
        return plan_a, plan_b
    if reduce_a > cap_a + 1e-9 or reduce_b > cap_b + 1e-9:
        return plan_a, plan_b
    old_gap = abs(plan_a.output_units - plan_b.output_units)
    new_gap = abs((geo_a.unit - reduce_a) - (geo_b.unit - reduce_b))
    if new_gap >= old_gap - 1e-9:
        return plan_a, plan_b
    rebound_a = _apply_reduction(a, geo_a, "left", reduce_a, layout, policy)
    rebound_b = _apply_reduction(b, geo_b, "right", reduce_b, layout, policy)
    return (
        replace(
            rebound_a,
            score=plan_a.score,
            rule_ids=plan_a.rule_ids,
            reason=f"{rebound_a.action or 'no resize'}; rebalanced",
        ),
        replace(
            rebound_b,
            score=plan_b.score,
            rule_ids=plan_b.rule_ids,
            reason=f"{rebound_b.action or 'no resize'}; rebalanced",
        ),
    )


def plan_rows(
    frames: list[FrameMeta],
    layout: LayoutConfig,
    count_policy: CountPolicy,
    place_policy: PlacePolicy,
) -> list[Row]:
    """Walk the sequence, decide frame counts, and lay out every row."""
    rows: list[Row] = []
    index = 0
    page_row = 0
    while index < len(frames):
        frame = frames[index]
        # A portrait ("tall") panorama is never cropped or width-resized, so it is
        # given two page rows of height instead of being squashed into one.
        if frame.is_panorama and frame.height > frame.width:
            rows.append(
                Row(
                    (plan_solo(frame, layout),),
                    "tall panorama: spans two rows",
                    "K_tall",
                    row_span=2,
                )
            )
            index += 1
            page_row = (page_row + 2) % 3
            continue
        if index == len(frames) - 1:
            rows.append(Row((plan_solo(frames[index], layout),), "last frame", "K_odd"))
            break
        a = frame
        b = frames[index + 1]
        decision = frames_per_row(a, b, CountContext(page_row=page_row), layout, count_policy)
        if decision.count == 1:
            rows.append(Row((plan_solo(a, layout),), decision.reason, decision.rule_id))
            index += 1
        else:
            plan_a, plan_b = plan_two(a, b, layout, place_policy)
            rows.append(Row((plan_a, plan_b), decision.reason, decision.rule_id))
            index += 2
        page_row = (page_row + 1) % 3
    return rows


def paginate(rows: list[Row], rows_per_page: int = 3) -> list[Page]:
    """Chunk rows into pages, counting each row's ``row_span`` slots."""
    pages: list[Page] = []
    current: list[Row] = []
    used = 0
    for row in rows:
        span = max(1, row.row_span)
        if current and used + span > rows_per_page:
            pages.append(Page(index=len(pages), rows=tuple(current)))
            current = []
            used = 0
        current.append(row)
        used += span
        if used >= rows_per_page:
            pages.append(Page(index=len(pages), rows=tuple(current)))
            current = []
            used = 0
    if current:
        pages.append(Page(index=len(pages), rows=tuple(current)))
    return pages


# --- evaluation metrics (pure, approximate where the carve shifts pixels) ----


def head_cut(frame: FrameMeta, plan: PanelPlan, *, tol_frac: float = 0.02) -> bool:
    """Whether a detected head falls outside the crop window.

    The horizontal shift carved into box coordinates is approximated by the
    carve fraction (seams are near-uniform for protected subjects), which is
    accurate enough for ranking rule sets.
    """
    if not frame.heads:
        return False
    kept = 1.0 - plan.crop_frac
    if kept >= 1.0 - 1e-9:
        return False
    post_width = frame.width * (1.0 - plan.carve_frac)
    x0 = plan.crop_x_frac * post_width
    x1 = x0 + kept * post_width
    tol = tol_frac * frame.width
    for box in frame.heads:
        bx0 = box.x * (1.0 - plan.carve_frac)
        bx1 = (box.x + box.width) * (1.0 - plan.carve_frac)
        if bx0 < x0 - tol or bx1 > x1 + tol:
            return True
    return False


def subject_center_error(frame: FrameMeta, plan: PanelPlan, layout: LayoutConfig) -> float | None:
    """Distance from the subject centre to the panel centre, as a fraction of panel width."""
    kept = 1.0 - plan.crop_frac
    if kept <= 1e-9:
        return None
    geo = geometry(frame, layout)
    if not geo.has_subject:
        return None
    post_width = frame.width * (1.0 - plan.carve_frac)
    subject_x = geo.center_frac * post_width
    panel_center = plan.crop_x_frac * post_width + kept * post_width / 2.0
    return abs(subject_x - panel_center) / (kept * post_width)


# --- rule-set registry -------------------------------------------------------


COUNT_POLICIES: dict[str, CountPolicy] = {
    "Count-K": CountPolicy("Count-K", frozenset({"K_establish", "K_carve_sum"})),
    "Count-Kp": CountPolicy("Count-Kp", frozenset({"K_establish", "K_carve_crop"})),
    "Count-Establish": CountPolicy("Count-Establish", frozenset({"K_establish"})),
    "Count-Pano": CountPolicy("Count-Pano", frozenset({"K_both_rigid", "K_sliver"})),
    "Count-Always2": CountPolicy("Count-Always2", frozenset()),
}

PLACE_POLICIES: dict[str, PlacePolicy] = {
    # Your additive score; carve only when the budget is worth it (>=10%).
    "Place-A": PlacePolicy(
        "Place-A", frozenset({"C1", "C2", "C8", "C3", "C7", "S1", "S2", "S3"}),
        frozenset({"M1", "M2", "M3", "M4", "M6"}),
        carve_min=0.10,
    ),
    # Density-first: let both frames shrink, carve whenever there is any budget.
    "Place-B": PlacePolicy(
        "Place-B", frozenset({"C1", "C2", "C8", "C3", "C7", "S1", "S2", "S3"}),
        frozenset({"M1", "M2", "M3", "M4", "M6"}),
        base_weight=1.0, max_share=0.60, carve_min=0.0,
    ),
    # Carve-heavy: the seam budget dominates and crop is the residue.
    "Place-C": PlacePolicy(
        "Place-C", frozenset({"S1", "S2", "S3", "C4", "C8", "C7"}),
        frozenset({"M1", "M2", "M3", "M6"}), base_weight=0.25, carve_min=0.0,
    ),
    # Crop-only: never seam-carve; rely entirely on hard cropping.  ``1.0`` as the
    # carve gate disables carving, and M1 is left out of the methods.
    "Place-G": PlacePolicy(
        "Place-G", frozenset({"C1", "C2", "C8", "C3", "C7", "P1", "P2"}),
        frozenset({"M2", "M3", "M4", "M6"}),
        carve_min=1.0,
    ),
    "Place-E": PlacePolicy(
        "Place-E", frozenset({"C1", "C2", "C8", "C7", "S1", "S2", "P1", "P2", "T1"}),
        frozenset({"M1", "M2", "M3", "M4", "M6"}),
        use_gaze=True, use_balance=True, carve_min=0.10,
    ),
    "Place-F": PlacePolicy(
        "Place-F", frozenset({"S2", "C7"}),
        frozenset({"M1", "M2", "M6"}), carve_min=0.0,
    ),
}

#: Named count x placement combinations to evaluate first.
DEFAULT_SETS: tuple[tuple[str, str, str], ...] = (
    ("K-B", "Count-K", "Place-B"),
    ("K-G", "Count-K", "Place-G"),
    ("2-G", "Count-Always2", "Place-G"),
    ("F-F", "Count-Always2", "Place-F"),
)

#: Every named set the tooling knows about, as ``name -> (count, place)``.
#: The pipeline default (``PIPELINE_DEFAULT_SET``) and the options shown in its
#: output come from here.
SETS: dict[str, tuple[str, str]] = {
    "K-B": ("Count-K", "Place-B"),
    "K-G": ("Count-K", "Place-G"),
    "2-G": ("Count-Always2", "Place-G"),
    "F-F": ("Count-Always2", "Place-F"),
}

#: The set the pipeline generates by default when a user runs ``anime2manga``.
PIPELINE_DEFAULT_SET = "K-B"

#: Plain-language description of each set, shown in the generated page.
SET_DESCRIPTIONS: dict[str, str] = {
    "K-B": (
        "Default. Lets both frames shrink to fit and uses any available seam "
        "budget (it carves whenever a frame can safely lose width), so pages "
        "fill up well."
    ),
    "K-G": (
        "No seam carving. Every frame is fitted by cropping only, which keeps "
        "the artwork undistorted but crops more."
    ),
    "2-G": (
        "Always two frames per row, cropping only. The most consistent, "
        "grid-like pages."
    ),
    "F-F": (
        "Minimal. A control set with very few rules; mostly places frames "
        "as-is."
    ),
}


def set_policies(name: str) -> tuple[CountPolicy, PlacePolicy]:
    """Return the ``(count, placement)`` policies for a named set."""
    if name not in SETS:
        raise KeyError(f"unknown layout set {name!r}; choose from {', '.join(SETS)}")
    count_name, place_name = SETS[name]
    return COUNT_POLICIES[count_name], PLACE_POLICIES[place_name]


def meta_from_scene(scene: Scene) -> FrameMeta:
    """Build a :class:`FrameMeta` from a completed pipeline :class:`Scene`.

    Faces are used as subject anchors only when no head was detected (the head
    detector normally supersedes the face detector).
    """
    composition = scene.composition
    heads = tuple(scene.heads) or tuple(scene.faces)
    shrink = scene.seam_carve_shrink
    return FrameMeta(
        index=scene.index,
        source_size=scene.frame_size or (1920, 1080),
        heads=heads,
        persons=tuple(scene.persons),
        body_percent=composition.body_percent if composition else 0.0,
        head_percent=composition.head_percent if composition else 0.0,
        overlap_percent=composition.body_head_overlap_percent if composition else 0.0,
        heads_in_body=composition.heads_in_body if composition else None,
        body_lean=(composition.body_leans.value if composition and composition.body_leans else None),
        head_lean=(composition.head_leans.value if composition and composition.head_leans else None),
        seam_carve_shrink=shrink or 0.0,
        seam_carve_known=shrink is not None,
        is_panorama=scene.is_panoramic,
        panorama_size=scene.panorama_size,
        subtitle_count=len(scene.subtitles),
        frame_time=scene.frame_time,
        frame_path=str(scene.frame_path) if scene.frame_path else None,
        panorama_path=(
            str(scene.panorama_inpainted_path or scene.frame_path)
            if (scene.panorama_inpainted_path or scene.frame_path)
            else None
        ),
    )


__all__ = [
    "COUNT_POLICIES",
    "DEFAULT_SETS",
    "PIPELINE_DEFAULT_SET",
    "PLACE_POLICIES",
    "RULE_NOTES",
    "SETS",
    "SET_DESCRIPTIONS",
    "CountContext",
    "CountDecision",
    "CountPolicy",
    "FrameMeta",
    "Geometry",
    "LayoutConfig",
    "Page",
    "PanelPlan",
    "PlacePolicy",
    "Row",
    "frames_per_row",
    "geometry",
    "head_cut",
    "is_wide_panorama",
    "meta_from_scene",
    "paginate",
    "plan_rows",
    "plan_solo",
    "plan_two",
    "removable_units",
    "score_frame",
    "set_policies",
    "subject_center_error",
]
