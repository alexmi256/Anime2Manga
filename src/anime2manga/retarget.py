"""Retarget quality metrics: how far can a frame be seam-carved before it looks bad?

Seam carving always *can* run to half width, but quality falls off a cliff once
the low-energy seams run out and the algorithm starts cutting through detail.
This module turns the raw per-seam measurements from
:mod:`anime2manga.seam_carving` into several independent "stop here" signals,
each with its own adjustable threshold, plus a weighted composite:

* ``energy``    - the removed seam's mean gradient energy, normalised by the
  image's mean energy.  This is the classic stopping criterion (Kapadia,
  *Improvements to Seam Carving*, TU/e): a seam is "free" while it stays well
  below the image average, and starts destroying content once it reaches it.
* ``forward``   - cumulative *forward* energy per removed pixel, normalised by
  the value measured over the first few percent.  Forward energy is the energy
  the removal *introduces* (Rubinstein et al. 2008); when it climbs above the
  early, easy-shrink baseline, seams have begun to bend structure.
* ``detail``    - budget on removed high-gradient ("detail") pixels and on
  protected (face/head/person) pixels.  Those subjects are also energetically
  protected in the engine, so the protected budget is a safety net, not the
  primary trigger.
* ``ssim``      - structural similarity between the carved image and the
  original uniformly scaled to the same size.  A cheap, principled stand-in for
  the bidirectional-similarity distortion of Simakov et al. (CVPR 2008); it
  drops as salient content is squeezed or shifted by seam removal.

The composite ``badness`` in ``[0, 1]`` is a weighted mean of the normalised
signals; the recommended shrink is where it first reaches
``badness_threshold`` (or, optionally, the most conservative single method).
Every threshold, weight and the 50% hard cap are configurable, so the metric
can be re-tuned per image without touching the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .faces import FaceDetectionConfig, detect_faces_in_image
from .heads import HeadDetectionConfig, detect_heads_in_image
from .models import DetectionBox, FaceBox
from .persons import PersonDetectionConfig, detect_persons_in_image
from .seam_carving import CarveResult, SeamCarvingConfig, SeamStep, carve_width, reconstruct

#: Names of the single-signal stopping rules, in report order.
METHODS: tuple[str, ...] = ("energy", "forward", "detail", "ssim")


@dataclass(frozen=True)
class RetargetConfig:
    """All tunables for one retarget analysis run."""

    #: Hard cap on the fraction of width removed (the brief says half).
    max_shrink: float = 0.5
    #: Working resolution: the image is downscaled to this width before carving.
    #: ``768`` is fast (~13x quicker than native on 1080p) and is the default
    #: for both the pipeline and the experiment.  ``None`` (or any non-positive
    #: value) carves the source frame at its native resolution instead.
    working_width: int | None = 768

    # --- single-signal thresholds ----------------------------------------
    #: ``energy``: stop when removed-seam energy reaches this multiple of the
    #: image's mean gradient energy.
    energy_ratio: float = 0.35
    #: ``energy``: some frames are detailed everywhere, so even their easiest
    #: seams score near the mean.  Raise the effective threshold to at least
    #: this multiple of the frame's own early-shrink baseline (see
    #: :attr:`energy_warmup`) so uniformly busy frames are not stopped instantly.
    energy_baseline_multiple: float = 1.4
    #: ``energy``: fraction of width used to measure that early baseline.
    energy_warmup: float = 0.10
    #: ``energy``: easy-seam energy (in units of the image mean) above which the
    #: adaptive term starts adding margin.  The effective threshold is
    #: ``energy_ratio + energy_baseline_multiple * max(0, early - energy_reference)``,
    #: so at the default ``energy_ratio=0.35`` / ``energy_baseline_multiple=1.4``
    #: this is exactly ``max(0.35, 1.4 * early)`` (since ``0.35 / 1.4 == 0.25``),
    #: but lowering ``energy_ratio`` now always tightens the guard instead of
    #: being silently overridden by the adaptive floor.
    energy_reference: float = 0.25
    #: ``forward``: stop when cumulative added energy per pixel reaches this
    #: multiple of its warm-up baseline.
    forward_knee_factor: float = 2.5
    #: ``detail``: fraction of original detail pixels that may be removed.
    detail_budget: float = 0.10
    #: ``detail``: fraction of original subject (face/head/person) pixels that
    #: may be removed.
    subject_budget: float = 0.02
    #: ``ssim``: stop when structural similarity falls to this floor.  SSIM is
    #: informational by default because its absolute value depends on how busy
    #: the frame is; enable it for frames with large smooth regions.
    ssim_floor: float = 0.5

    # --- composite --------------------------------------------------------
    #: Guards that feed the composite badness.  ``energy`` and ``detail`` are
    #: the reliable, well-separated signals; ``forward`` and ``ssim`` are
    #: optional and off by default.
    enabled_methods: tuple[str, ...] = ("energy", "detail")
    #: Composite badness is the worst (largest) normalised guard signal, so it
    #: reaches ``1.0`` exactly when the first enabled guard trips.
    badness_threshold: float = 1.0
    #: Which limit is reported as the recommendation: ``"composite"`` or any
    #: value from :data:`METHODS`.
    primary_method: str = "composite"

    # --- sampling / engine -------------------------------------------------
    #: Output a carved snapshot every this fraction of width (plus originals).
    sample_step: float = 0.1
    #: Compute SSIM every N seams (cheap interpolation between samples).
    ssim_stride: int = 4
    #: Moving-average window used to smooth the per-seam signals, as a fraction
    #: of the original width.
    smoothing_window: float = 0.02
    #: Fraction of width used to establish the forward-energy warm-up baseline.
    warmup: float = 0.05
    protect_subjects: bool = True
    subject_energy_factor: float = 50.0
    #: Remove the thin green face boxes the pipeline draws onto frames before
    #: analysing (they are annotation, not content).  Skipped when green covers
    #: too much of the frame to be an overlay.
    strip_overlays: bool = True


@dataclass
class MethodLimit:
    """Where one stopping rule would call it quits."""

    name: str
    ratio: float
    #: True when the threshold was actually crossed before the cap/end.
    triggered: bool
    detail: str = ""

    @property
    def label(self) -> str:
        return f"{self.name}: {self.ratio * 100:.1f}%"


@dataclass
class RetargetTrace:
    """Per-seam signals for one frame (arrays all have ``len(steps)`` entries)."""

    steps: list[SeamStep]
    base_energy: float
    energy_sum0: float
    detail_total: int
    protected_total: int
    width0: int
    height: int
    working_scale: float
    ratios: np.ndarray
    removed_norm: np.ndarray
    added_norm: np.ndarray
    added_cum_norm: np.ndarray
    cum_detail: np.ndarray
    cum_protected: np.ndarray
    energy_retained: np.ndarray
    ssim: np.ndarray

    def index_at_ratio(self, ratio: float) -> int:
        """Index of the last seam whose ratio is ``<= ratio`` (0 when none)."""
        if not len(self.ratios):
            return 0
        return int(np.searchsorted(self.ratios, ratio, side="right"))


@dataclass
class FrameAnalysis:
    """Everything one frame's retarget experiment produced."""

    path: Path
    original_size: tuple[int, int]
    working_size: tuple[int, int]
    faces: list[FaceBox]
    carve: CarveResult
    trace: RetargetTrace
    limits: dict[str, MethodLimit] = field(default_factory=dict)
    recommended: MethodLimit | None = None
    snapshots: dict[float, np.ndarray] = field(default_factory=dict)
    #: Normalised guard signals at every seam (``signal / threshold``).
    signals: dict[str, np.ndarray] = field(default_factory=dict)
    #: Composite badness at every seam (``max`` of the enabled signals).
    badness: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: The working image carved exactly to :attr:`recommended`.
    recommended_image: np.ndarray | None = None
    #: The frame's own easy-seam energy (units of the image mean).
    energy_early: float = 0.0
    #: Effective ``removed_norm`` threshold the energy guard actually used.
    energy_threshold: float = 0.0

    @property
    def name(self) -> str:
        return self.path.stem


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    window = min(window, values.size)
    kernel = np.ones(window, np.float64) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _strip_green_overlay(image: np.ndarray) -> np.ndarray:
    """Erase the pipeline's thin green face-box overlay, if present.

    Frames written by the pipeline have the detected face boxes drawn in pure
    green.  They are annotation, not scene content, so detecting and carving
    them would both pollute the energy map and make the results look wrong.
    Pixels where green dominates both red and blue are inpainted; if that covers
    a large area the frame is assumed to contain genuine green content and is
    returned untouched.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        return image
    blue, green, red = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    g = green.astype(np.int16)
    mask = (g - red.astype(np.int16) > 90) & (g - blue.astype(np.int16) > 90) & (green > 130)
    if not mask.any() or mask.mean() > 0.03:
        return image
    kernel = np.ones((3, 3), np.uint8)
    grown = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return cv2.inpaint(image, grown, 3, cv2.INPAINT_TELEA)


def _scale_boxes(boxes: list[DetectionBox], scale: float) -> list[DetectionBox]:
    return [
        type(box)(
            x=round(box.x * scale),
            y=round(box.y * scale),
            width=round(box.width * scale),
            height=round(box.height * scale),
            confidence=box.confidence,
        )
        for box in boxes
    ]


def _first_crossing(ratios: np.ndarray, values: np.ndarray, threshold: float) -> tuple[float, bool]:
    """First ratio whose signal reaches ``threshold`` (``False`` if never)."""
    if ratios.size == 0:
        return 0.0, False
    hits = np.nonzero(values >= threshold)[0]
    if hits.size == 0:
        return float(ratios[-1]), False
    return float(ratios[hits[0]]), True


def _first_fall(ratios: np.ndarray, values: np.ndarray, threshold: float) -> tuple[float, bool]:
    """First ratio whose signal falls to ``threshold`` (``False`` if never)."""
    if ratios.size == 0:
        return 0.0, False
    hits = np.nonzero(values <= threshold)[0]
    if hits.size == 0:
        return float(ratios[-1]), False
    return float(ratios[hits[0]]), True


def _build_trace(
    result: CarveResult,
    *,
    width0: int,
    height: int,
    working_scale: float,
    ssim_by_k: dict[int, float],
    config: RetargetConfig,
) -> RetargetTrace:
    steps = result.steps
    n = len(steps)
    ratios = np.array([s.ratio for s in steps], np.float64)
    removed = np.array([s.removed_energy for s in steps], np.float64)
    added = np.array([s.added_energy for s in steps], np.float64)
    detail = np.array([s.detail_pixels for s in steps], np.float64)
    protected = np.array([s.protected_pixels for s in steps], np.float64)
    energy_after = np.array([s.energy_sum_after for s in steps], np.float64)

    window = max(1, round(config.smoothing_window * width0))
    removed_norm = _moving_average(removed, window) / max(result.base_energy, 1e-9)
    added_norm = _moving_average(added, window) / max(result.base_energy, 1e-9)
    cum_detail = np.cumsum(detail) / max(result.detail_total, 1)
    cum_protected = np.cumsum(protected) / max(result.protected_total, 1)
    energy_retained = energy_after / max(result.energy_sum0, 1e-9)

    # Cumulative added energy per removed pixel, normalised - smooth and monotone.
    with np.errstate(invalid="ignore", divide="ignore"):
        added_cum = np.cumsum(added) / np.arange(1, n + 1)
    added_cum_norm = added_cum / max(result.base_energy, 1e-9)

    if n:
        ks = np.array(sorted(ssim_by_k), np.float64)
        vs = np.array([ssim_by_k[int(k)] for k in ks], np.float64)
        ssim = np.interp(np.arange(1, n + 1), ks, vs, left=1.0, right=vs[-1])
    else:
        ssim = np.array([], np.float64)

    return RetargetTrace(
        steps=steps,
        base_energy=result.base_energy,
        energy_sum0=result.energy_sum0,
        detail_total=result.detail_total,
        protected_total=result.protected_total,
        width0=width0,
        height=height,
        working_scale=working_scale,
        ratios=ratios,
        removed_norm=removed_norm,
        added_norm=added_norm,
        added_cum_norm=added_cum_norm,
        cum_detail=cum_detail,
        cum_protected=cum_protected,
        energy_retained=energy_retained,
        ssim=ssim,
    )


def energy_operating_point(trace: RetargetTrace, config: RetargetConfig) -> tuple[float, float]:
    """Return ``(early, effective_threshold)`` for the energy guard.

    ``early`` is the frame's own removed-seam energy at the end of the warm-up
    window, in units of the image mean.  ``effective_threshold`` is the value
    ``removed_norm`` must reach before the energy guard trips; reporting it makes
    it obvious when the adaptive term (rather than ``energy_ratio``) governs.
    """
    if not trace.removed_norm.size:
        return 0.0, config.energy_ratio
    j = int(np.searchsorted(trace.ratios, config.energy_warmup, side="right")) - 1
    early = float(trace.removed_norm[max(0, min(j, trace.removed_norm.size - 1))])
    threshold = config.energy_ratio + config.energy_baseline_multiple * max(
        0.0, early - config.energy_reference
    )
    return early, threshold


def _normalised_signals(trace: RetargetTrace, config: RetargetConfig) -> dict[str, np.ndarray]:
    """Per-seam signals mapped to ``[0, 1]`` multiples of their threshold."""
    # Energy guard: a fixed threshold would stop uniformly detailed frames
    # immediately, so once a frame's easiest seams score above
    # ``energy_reference`` the guard adds margin proportional to that excess.
    _early, energy_threshold = energy_operating_point(trace, config)

    forward_reference = 1.0
    if trace.added_cum_norm.size:
        warm = trace.ratios <= config.warmup
        warm_vals = trace.added_cum_norm[warm]
        forward_reference = (
            float(np.median(warm_vals)) if warm_vals.size else float(trace.added_cum_norm[0])
        )
    forward_reference = max(forward_reference, 1e-6)
    with np.errstate(invalid="ignore", divide="ignore"):
        signals = {
            "energy": trace.removed_norm / max(energy_threshold, 1e-9),
            "forward": trace.added_cum_norm / (config.forward_knee_factor * forward_reference),
            "detail": np.maximum(
                trace.cum_detail / max(config.detail_budget, 1e-9),
                trace.cum_protected / max(config.subject_budget, 1e-9),
            ),
            "ssim": (1.0 - trace.ssim) / max(1.0 - config.ssim_floor, 1e-9),
        }
    return {k: np.clip(v, 0.0, None) for k, v in signals.items()}


def composite_badness(signals: dict[str, np.ndarray], enabled: tuple[str, ...]) -> np.ndarray:
    """Worst normalised guard signal; reaches ``1.0`` when a guard trips."""
    active = [signals[name] for name in enabled if name in signals and signals[name].size]
    if not active:
        return np.zeros_like(next(iter(signals.values())))
    return np.max(np.stack(active), axis=0)


def compute_limits(
    trace: RetargetTrace,
    config: RetargetConfig,
    signals: dict[str, np.ndarray] | None = None,
) -> dict[str, MethodLimit]:
    """Evaluate every stopping rule against a trace.

    ``signals`` may be supplied to avoid recomputing
    :func:`_normalised_signals` when the caller already has it.
    """
    if signals is None:
        signals = _normalised_signals(trace, config)
    ratios = trace.ratios
    cap = config.max_shrink

    limits: dict[str, MethodLimit] = {}
    ratio, hit = _first_crossing(ratios, signals["energy"], 1.0)
    limits["energy"] = MethodLimit(
        "energy", min(ratio, cap), hit, f"removed energy / mean >= {config.energy_ratio:g}"
    )
    ratio, hit = _first_crossing(ratios, signals["forward"], 1.0)
    limits["forward"] = MethodLimit(
        "forward",
        min(ratio, cap),
        hit,
        f"added energy / warm-up >= {config.forward_knee_factor:g}x",
    )
    ratio, hit = _first_crossing(ratios, signals["detail"], 1.0)
    limits["detail"] = MethodLimit(
        "detail",
        min(ratio, cap),
        hit,
        f"detail removed >= {config.detail_budget:.0%} or subjects >= {config.subject_budget:.0%}",
    )
    ratio, hit = _first_fall(ratios, trace.ssim, config.ssim_floor)
    limits["ssim"] = MethodLimit("ssim", min(ratio, cap), hit, f"SSIM <= {config.ssim_floor:g}")

    badness = composite_badness(signals, config.enabled_methods)
    ratio, hit = _first_crossing(ratios, badness, config.badness_threshold)
    enabled = ", ".join(config.enabled_methods)
    limits["composite"] = MethodLimit(
        "composite",
        min(ratio, cap),
        hit,
        f"badness >= {config.badness_threshold:g} ({enabled})",
    )
    return limits


def detect_all_boxes(
    image: np.ndarray,
    *,
    face: FaceDetectionConfig | None = None,
    head: HeadDetectionConfig | None = None,
    person: PersonDetectionConfig | None = None,
) -> list[DetectionBox]:
    """Detect faces, heads and persons and return the combined box list.

    The seam carver protects every detected subject the same way, so the three
    categories are merged into one list (the engine does not care which category
    a box came from).
    """
    boxes = detect_faces_in_image(image, config=face)
    boxes.extend(detect_heads_in_image(image, config=head))
    boxes.extend(detect_persons_in_image(image, config=person))
    return boxes


def retarget_image(
    image: np.ndarray,
    *,
    config: RetargetConfig | None = None,
    boxes: list[DetectionBox] | None = None,
    face_config: FaceDetectionConfig | None = None,
    protect_heads: bool = True,
    protect_persons: bool = True,
    head_config: HeadDetectionConfig | None = None,
    person_config: PersonDetectionConfig | None = None,
) -> FrameAnalysis:
    """Carve an in-memory BGR frame down to the cap, recording quality signals.

    The engine protects *every* detected subject, so when ``boxes`` is not given
    the face, head and person models all run on the image.  A caller that already
    has the pipeline's boxes passes them through ``boxes`` (the pipeline merges
    all three categories); pass ``protect_heads=False`` / ``protect_persons=False``
    to skip detecting those categories.  The returned analysis carries a
    placeholder ``path`` - use :func:`analyze_frame` for the disk entry point.
    Set ``config.working_width=None`` to carve at the source resolution instead
    of downscaling first.
    """
    cfg = config or RetargetConfig()
    original = image
    if cfg.strip_overlays:
        original = _strip_green_overlay(original)
    height0, width0 = original.shape[:2]

    if boxes is None:
        boxes = detect_faces_in_image(original, config=face_config)
        if protect_heads:
            boxes.extend(detect_heads_in_image(original, config=head_config))
        if protect_persons:
            boxes.extend(detect_persons_in_image(original, config=person_config))

    if cfg.working_width is None or cfg.working_width <= 0:
        scale = 1.0
    else:
        scale = min(1.0, cfg.working_width / width0)
    work_w = max(1, round(width0 * scale))
    work_h = max(1, round(height0 * scale))
    if scale < 1.0:
        working = cv2.resize(original, (work_w, work_h), interpolation=cv2.INTER_AREA)
    else:
        # No downscale requested (``working_width=None`` or a width at least as
        # large as the source): carve the original pixels untouched.
        working = original
        work_w, work_h = width0, height0
    work_boxes = _scale_boxes(boxes, scale)

    target = max(1, round(work_w * (1.0 - cfg.max_shrink)))
    sample_ratios = tuple(
        round(r, 4) for r in np.arange(cfg.sample_step, cfg.max_shrink + 1e-9, cfg.sample_step)
    )
    snapshot_ratios = (0.0, *sample_ratios)

    gray_ref = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    ssim_by_k: dict[int, float] = {0: 1.0}

    def on_seam(k: int, image: np.ndarray, _step: SeamStep) -> None:
        if cfg.ssim_stride > 0 and k % cfg.ssim_stride == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            reference = cv2.resize(
                gray_ref, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA
            )
            ssim_by_k[k] = float(structural_similarity(reference, gray, data_range=255))

    carve_result = carve_width(
        working,
        target,
        boxes=work_boxes,
        config=SeamCarvingConfig(
            protect_subjects=cfg.protect_subjects,
            subject_energy_factor=cfg.subject_energy_factor,
        ),
        on_seam=on_seam,
        snapshot_ratios=snapshot_ratios,
    )
    if len(ssim_by_k) == 1:
        ssim_by_k = {0: 1.0}

    trace = _build_trace(
        carve_result,
        width0=work_w,
        height=work_h,
        working_scale=scale,
        ssim_by_k=ssim_by_k,
        config=cfg,
    )
    signals = _normalised_signals(trace, cfg)
    limits = compute_limits(trace, cfg, signals)
    primary = limits.get(cfg.primary_method, limits["composite"])
    badness = composite_badness(signals, cfg.enabled_methods)
    recommended_image = reconstruct(carve_result, working, trace.index_at_ratio(primary.ratio))
    energy_early, energy_threshold = energy_operating_point(trace, cfg)
    return FrameAnalysis(
        path=Path("<memory>"),
        original_size=(width0, height0),
        working_size=(work_w, work_h),
        faces=list(boxes),
        carve=carve_result,
        trace=trace,
        limits=limits,
        recommended=primary,
        snapshots=carve_result.snapshots,
        signals=signals,
        badness=badness,
        recommended_image=recommended_image,
        energy_early=energy_early,
        energy_threshold=energy_threshold,
    )


def analyze_frame(
    path: Path,
    *,
    config: RetargetConfig | None = None,
    boxes: list[DetectionBox] | None = None,
    face_config: FaceDetectionConfig | None = None,
    protect_heads: bool = True,
    protect_persons: bool = True,
    head_config: HeadDetectionConfig | None = None,
    person_config: PersonDetectionConfig | None = None,
) -> FrameAnalysis | None:
    """Carve the frame at ``path`` down to the cap; ``None`` if unreadable.

    ``boxes`` overrides detection (useful for tests); otherwise the face, head
    and person models run on the full-resolution image and the boxes are scaled
    into the working image.  Thin wrapper around :func:`retarget_image`.
    """
    path = Path(path)
    original = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if original is None or original.size == 0:
        return None
    analysis = retarget_image(
        original,
        config=config,
        boxes=boxes,
        face_config=face_config,
        protect_heads=protect_heads,
        protect_persons=protect_persons,
        head_config=head_config,
        person_config=person_config,
    )
    analysis.path = path
    return analysis
