"""Content-aware seam carving (a.k.a. liquid rescaling) for frame retargeting.

This is the engine behind step 8b's content-aware retargeting: instead of cutting
a rectangle out of a 16:9 frame, remove ``1``-pixel-wide connected vertical
seams (one pixel per row, following a continuous path) until the frame reaches
the target width.  Because each seam takes the lowest-energy path through the
image, removed pixels concentrate in low-detail regions (sky, walls, blurred
background) while salient content keeps its proportions - no cropping, no
stretching.

Method
------
The gradient-magnitude energy map ``E = |dI/dx| + |dI/dy|`` (Sobel, L1) marks
"least important" pixels.  Seams are chosen with **forward energy**
(Rubinstein, Shamir & Avidan, *Improved Seam Carving for Video Retargeting*,
TOG 2008) rather than the classic backward energy of Avidan & Shamir
(SIGGRAPH 2007): forward energy minimises the energy *introduced* by joining
the two pixels a seam pixel used to separate, which removes far fewer visible
artefacts (bent straight lines, ghosting) at aggressive shrink factors.

Faces (step 8) are protected by adding a large, configurable amount of energy
inside a dilated face mask, so the dynamic program steers seams around them.
The engine also reports, per removed seam, how much high-detail and protected
(face) content it touched - the raw signals the retarget metrics consume.

Only vertical seams are carved here; :func:`carve_height` reuses the same code
on a transposed image.

Implementation
--------------
The hot loop is available in two interchangeable forms: the pure-Python
reference in this module and a compiled C++ engine
(``src/anime2manga/_seamcarve.cpp``, built into ``anime2manga._seamcarve``).  The
two are behaviourally identical (the compiled one is bit-identical in the carved
image and seam trace, with the per-seam metrics agreeing to floating-point
rounding).  :func:`carve_width` uses the compiled engine when it is importable
and falls back to the Python loop otherwise (:data:`HAVE_NATIVE` reports which).
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

# The compiled engine is optional: when it cannot be imported the pure-Python
# implementation below is used unchanged.
try:  # pragma: no cover - presence depends on whether the build ran
    from . import _seamcarve as _native_module
except ImportError:  # pragma: no cover
    _native_module = None


class _SeamStepRaw(ctypes.Structure):
    """Mirror of the C++ ``SeamStepRaw`` (``_seamcarve.cpp``)."""

    _fields_ = [
        ("removed_energy", ctypes.c_double),
        ("added_energy", ctypes.c_double),
        ("detail_pixels", ctypes.c_longlong),
        ("protected_pixels", ctypes.c_longlong),
        ("energy_sum_after", ctypes.c_double),
        ("width_after", ctypes.c_int),
        ("reserved", ctypes.c_int),
    ]


# void (*)(int k, const uint8_t* image, int h, int w, step fields..., void* user)
_SeamCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_longlong,
    ctypes.c_longlong,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_void_p,
)

# int carve_width_full(...) -- see _seamcarve.cpp for the full signature.
_CarveProto = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8),  # src
    ctypes.c_int,  # height
    ctypes.c_int,  # width
    ctypes.c_int,  # target_width
    ctypes.POINTER(ctypes.c_int),  # faces
    ctypes.c_int,  # n_faces
    ctypes.c_int,  # protect_faces
    ctypes.c_double,  # face_dilation
    ctypes.c_double,  # face_energy_factor
    ctypes.c_double,  # detail_quantile
    ctypes.c_int,  # record_seams
    ctypes.c_int,  # n_snapshots
    ctypes.POINTER(ctypes.c_double),  # snapshot_ratios
    ctypes.POINTER(ctypes.c_uint8),  # snapshot buffers
    ctypes.POINTER(ctypes.c_int),  # snapshot widths
    ctypes.POINTER(ctypes.c_uint8),  # dst
    ctypes.POINTER(ctypes.c_int),  # out_width
    ctypes.POINTER(ctypes.c_int),  # out_seams
    ctypes.POINTER(ctypes.c_int),  # seams_trace
    ctypes.POINTER(_SeamStepRaw),  # steps
    ctypes.POINTER(ctypes.c_double),  # base_energy
    ctypes.POINTER(ctypes.c_double),  # energy_sum0
    ctypes.POINTER(ctypes.c_longlong),  # detail_total
    ctypes.POINTER(ctypes.c_longlong),  # protected_total
    _SeamCallback,  # on_seam
    ctypes.c_void_p,  # user
)

_NATIVE_CARVE = _CarveProto(_native_module.carve_addr()) if _native_module is not None else None
#: True when the compiled engine is available and :func:`carve_width` uses it.
HAVE_NATIVE = _NATIVE_CARVE is not None

#: Energy added to pixels inside the face mask, as a multiple of the image's
#: mean gradient energy.  Large enough that a seam only crosses a face when no
#: alternative path exists.
DEFAULT_FACE_ENERGY_FACTOR = 50.0
#: Fraction of the highest-energy pixels treated as "detail" for the budget
#: metric (top 20%).
DEFAULT_DETAIL_QUANTILE = 0.8


@dataclass(frozen=True)
class SeamCarvingConfig:
    """Tunables for the seam-carving engine."""

    #: Add a large energy term inside ``face_mask`` so seams avoid faces.
    protect_faces: bool = True
    #: Face-mask dilation as a fraction of the smaller image side, so the
    #: protected halo scales with resolution.  ``0`` protects the box exactly.
    face_dilation: float = 0.01
    #: Multiplier (of the mean gradient energy) applied inside the face mask.
    face_energy_factor: float = DEFAULT_FACE_ENERGY_FACTOR
    #: Energy quantile (top ``1 - q``) counted as "detail" in the budget metric.
    detail_quantile: float = DEFAULT_DETAIL_QUANTILE
    #: OpenCV threading; seam carving is single-threaded per image and the
    #: driver parallelises across frames, so this is disabled by default.
    threads: int = 1
    #: Keep the per-row column of every removed seam so the image can be rebuilt
    #: at any intermediate width (:func:`reconstruct`).
    record_seams: bool = True


@dataclass
class SeamStep:
    """Raw measurements for one removed seam.

    Energies are per-pixel means over the seam; counts are pixel counts.  All
    fields are arbitrary units except :attr:`ratio`.
    """

    #: 1-based seam index.
    k: int
    #: Fraction of the original width removed so far (``k / width0``).
    ratio: float
    #: Mean gradient energy of the removed pixels.
    removed_energy: float
    #: Mean forward energy introduced by removing this seam (``0`` in backward
    #: mode).
    added_energy: float
    #: Number of removed pixels that were in the original "detail" set.
    detail_pixels: int
    #: Number of removed pixels that were inside the protected (face) mask.
    protected_pixels: int
    #: Sum of gradient energy over the whole image after the removal.
    energy_sum_after: float
    #: Width of the image after the removal.
    width_after: int


@dataclass
class CarveResult:
    """The carved image plus the per-seam trace needed by the metrics."""

    image: np.ndarray
    steps: list[SeamStep]
    width0: int
    height0: int
    #: Mean gradient energy of the original (working-resolution) image.
    base_energy: float
    #: Total gradient energy of the original image.
    energy_sum0: float
    #: Number of original pixels at or above the detail quantile.
    detail_total: int
    #: Number of original pixels inside the protected mask.
    protected_total: int
    #: Images captured by the ``on_seam`` callback, keyed by requested ratio.
    snapshots: dict[float, np.ndarray] = field(default_factory=dict)
    #: Column index per row for every removed seam, in removal order.  Lets a
    #: caller rebuild the image at any intermediate width (see
    #: :func:`reconstruct`); disabled with ``record_seams=False``.
    seams: list[np.ndarray] = field(default_factory=list)


def reconstruct(result: CarveResult, original: np.ndarray, count: int) -> np.ndarray:
    """Rebuild the image after exactly ``count`` of ``result``'s seams.

    ``original`` must be the same image passed to :func:`carve_width`.
    """
    if not result.seams:
        return _as_bgr(original)
    image = _as_bgr(original).copy()
    for seam in result.seams[: max(0, min(count, len(result.seams)))]:
        image = _remove_vertical_seam(image, seam)
    return image


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def gradient_energy(gray: np.ndarray) -> np.ndarray:
    """Return the L1 gradient magnitude of ``gray`` as float32."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(gx) + np.abs(gy)


def _forward_costs(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-energy costs ``(cL, cU, cR)`` for every pixel.

    Removing pixel ``(i, j)`` makes its left/right neighbours adjacent and
    reconnects them to the surviving pixel on row ``i-1``:

    * ``cL`` - arrival from ``(i-1, j-1)``,
    * ``cU`` - arrival from ``(i-1, j)``,
    * ``cR`` - arrival from ``(i-1, j+1)``.

    Border pixels are handled by replicating the edge, so the costs are finite.
    """
    g = gray.astype(np.float32)
    padded = np.pad(g, 1, mode="edge")
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    up = padded[:-2, 1:-1]
    new_horizontal = np.abs(left - right)
    c_l = new_horizontal + np.abs(up - left)
    c_u = new_horizontal
    c_r = new_horizontal + np.abs(up - right)
    return c_l, c_u, c_r


def _backtrack(acc: np.ndarray, back: np.ndarray) -> np.ndarray:
    """Follow ``back`` pointers from the cheapest pixel of the last row."""
    height = acc.shape[0]
    seam = np.zeros(height, np.int32)
    seam[-1] = int(np.argmin(acc[-1]))
    for i in range(height - 1, 0, -1):
        seam[i - 1] = seam[i] + back[i, seam[i]]
    return seam


def find_vertical_seam(
    energy: np.ndarray,
    *,
    forward: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    saliency: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the lowest-cost vertical seam through ``energy``.

    With ``forward`` costs the accumulation minimises the energy *added* by the
    removal; otherwise it minimises the classic backward energy sum.  An
    optional ``saliency`` array adds a fixed per-pixel cost (used to keep seams
    off faces).

    Returns the seam's column index per row and the forward cost contributed
    per row (``0`` when carving backward).
    """
    height, width = energy.shape
    acc = np.empty((height, width), np.float64)
    back = np.zeros((height, width), np.int8)
    acc[0] = energy[0] + (saliency[0] if saliency is not None else 0.0)

    for i in range(1, height):
        prev = acc[i - 1]
        left = np.empty(width, np.float64)
        left[0] = np.inf
        left[1:] = prev[:-1]
        right = np.empty(width, np.float64)
        right[-1] = np.inf
        right[:-1] = prev[1:]
        if forward is None:
            options = np.stack([left, prev, right])
        else:
            c_l, c_u, c_r = forward
            options = np.stack([left + c_l[i], prev + c_u[i], right + c_r[i]])
        choice = np.argmin(options, axis=0)
        row_extra = saliency[i] if saliency is not None else 0.0
        acc[i] = energy[i] + row_extra + options[choice, np.arange(width)]
        back[i] = choice - 1

    seam = _backtrack(acc, back)
    added = np.zeros(height, np.float64)
    if forward is not None:
        c_l, c_u, c_r = forward
        rows = np.arange(1, height)
        moves = back[rows, seam[rows]]
        added[rows] = np.where(
            moves == -1,
            c_l[rows, seam[rows]],
            np.where(moves == 0, c_u[rows, seam[rows]], c_r[rows, seam[rows]]),
        )
    return seam, added


def _remove_vertical_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    keep = np.ones((height, width), bool)
    keep[np.arange(height), seam] = False
    return image[keep].reshape(height, width - 1, *image.shape[2:])


def _face_mask(shape: tuple[int, int], faces: list, config: SeamCarvingConfig) -> np.ndarray | None:
    """Build a boolean mask covering the faces, scaled to the working image.

    The mask is built whenever faces are known, even if protection is disabled,
    so the engine can still report how many face pixels each seam removed.
    """
    if not faces:
        return None
    height, width = shape
    mask = np.zeros((height, width), np.uint8)
    for box in faces:
        x0 = max(0, int(box.x))
        y0 = max(0, int(box.y))
        x1 = min(width, int(box.x + box.width))
        y1 = min(height, int(box.y + box.height))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
    if config.face_dilation > 0 and mask.any():
        radius = max(1, round(config.face_dilation * min(height, width)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)


def _carve_width_native(
    image: np.ndarray,
    target_width: int,
    *,
    faces: list | None,
    config: SeamCarvingConfig,
    on_seam: Callable[[int, np.ndarray, SeamStep], None] | None,
    snapshot_ratios: tuple[float, ...],
) -> CarveResult:
    """Carve ``image`` with the compiled engine (returns the same result as the loop)."""
    assert _NATIVE_CARVE is not None
    height, width = image.shape[:2]
    image = np.ascontiguousarray(image)

    face_rows = [(int(b.x), int(b.y), int(b.width), int(b.height)) for b in (faces or [])]
    face_arr = np.asarray(face_rows, np.int32) if face_rows else None
    face_ptr = (
        face_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        if face_arr is not None
        else ctypes.POINTER(ctypes.c_int)()
    )

    n_seams = width - target_width
    dst = np.empty_like(image)
    steps_raw = (_SeamStepRaw * n_seams)()
    if config.record_seams:
        trace = np.empty((n_seams, height), np.int32)
        trace_ptr = trace.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    else:
        trace = None
        trace_ptr = ctypes.POINTER(ctypes.c_int)()

    pending = sorted(r for r in snapshot_ratios if r > 0)
    n_snap = len(pending)
    if n_snap:
        snap_buf = np.empty((n_snap, height, width, 3), np.uint8)
        snap_widths = (ctypes.c_int * n_snap)()
        snap_buf_ptr = snap_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        ratios_ptr = np.asarray(pending, np.float64).ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )
        widths_ptr = snap_widths
    else:
        snap_buf = None
        snap_widths = None
        snap_buf_ptr = ctypes.POINTER(ctypes.c_uint8)()
        ratios_ptr = ctypes.POINTER(ctypes.c_double)()
        widths_ptr = ctypes.POINTER(ctypes.c_int)()

    error: list[BaseException] = []

    if on_seam is not None:

        def _callback(
            k, image_ptr, h, w, removed, added, detail, protected, energy_after, width_after, _user
        ):
            try:
                arr = np.ctypeslib.as_array(
                    (ctypes.c_uint8 * (h * w * 3)).from_address(image_ptr)
                ).reshape(h, w, 3)
                step = SeamStep(
                    k=int(k),
                    ratio=k / width,
                    removed_energy=float(removed),
                    added_energy=float(added),
                    detail_pixels=int(detail),
                    protected_pixels=int(protected),
                    energy_sum_after=float(energy_after),
                    width_after=int(width_after),
                )
                on_seam(int(k), arr, step)
            except BaseException as exc:
                # ctypes swallows callback exceptions, so stash it and re-raise
                # after the native call returns.
                error.append(exc)

        callback = _SeamCallback(_callback)
    else:
        callback = _SeamCallback()

    out_width = ctypes.c_int(0)
    out_seams = ctypes.c_int(0)
    base_energy = ctypes.c_double(0.0)
    energy_sum0 = ctypes.c_double(0.0)
    detail_total = ctypes.c_longlong(0)
    protected_total = ctypes.c_longlong(0)

    rc = _NATIVE_CARVE(
        image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        height,
        width,
        target_width,
        face_ptr,
        len(face_rows),
        int(config.protect_faces),
        config.face_dilation,
        config.face_energy_factor,
        config.detail_quantile,
        int(config.record_seams),
        n_snap,
        ratios_ptr,
        snap_buf_ptr,
        widths_ptr,
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.byref(out_width),
        ctypes.byref(out_seams),
        trace_ptr,
        steps_raw,
        ctypes.byref(base_energy),
        ctypes.byref(energy_sum0),
        ctypes.byref(detail_total),
        ctypes.byref(protected_total),
        callback,
        None,
    )
    if rc < 0:
        raise RuntimeError("native seam carving failed")
    if error:
        raise error[0]

    steps = [
        SeamStep(
            k=i + 1,
            ratio=(i + 1) / width,
            removed_energy=steps_raw[i].removed_energy,
            added_energy=steps_raw[i].added_energy,
            detail_pixels=int(steps_raw[i].detail_pixels),
            protected_pixels=int(steps_raw[i].protected_pixels),
            energy_sum_after=steps_raw[i].energy_sum_after,
            width_after=int(steps_raw[i].width_after),
        )
        for i in range(out_seams.value)
    ]
    snapshots = {r: image.copy() for r in snapshot_ratios if r <= 0}
    if snap_buf is not None and snap_widths is not None:
        for i, ratio in enumerate(pending):
            if snap_widths[i] >= 0:
                snapshots[ratio] = snap_buf[i][:, : snap_widths[i]].copy()

    return CarveResult(
        image=dst[:, : out_width.value].copy(),
        steps=steps,
        width0=width,
        height0=height,
        base_energy=float(base_energy.value),
        energy_sum0=float(energy_sum0.value),
        detail_total=int(detail_total.value),
        protected_total=int(protected_total.value),
        snapshots=snapshots,
        seams=list(trace) if trace is not None else [],
    )


def carve_width(
    image: np.ndarray,
    target_width: int,
    *,
    faces: list | None = None,
    config: SeamCarvingConfig | None = None,
    on_seam: Callable[[int, np.ndarray, SeamStep], None] | None = None,
    snapshot_ratios: tuple[float, ...] = (),
) -> CarveResult:
    """Remove vertical seams from ``image`` until it is ``target_width`` wide.

    Dispatches to the compiled engine when available (see :data:`HAVE_NATIVE`)
    and to the pure-Python implementation otherwise.  The two are behaviourally
    identical; ``faces``, ``on_seam`` and ``snapshot_ratios`` work with both.
    """
    cfg = config or SeamCarvingConfig()
    if cfg.threads > 0:
        cv2.setNumThreads(cfg.threads)
    original = _as_bgr(image)
    if original.dtype != np.uint8:
        original = np.clip(original, 0, 255).astype(np.uint8)
    width0 = original.shape[1]
    target = max(1, min(int(target_width), width0))
    if _NATIVE_CARVE is None or target >= width0:
        return _carve_width_python(
            original,
            target,
            faces=faces,
            config=cfg,
            on_seam=on_seam,
            snapshot_ratios=snapshot_ratios,
        )
    return _carve_width_native(
        original,
        target,
        faces=faces,
        config=cfg,
        on_seam=on_seam,
        snapshot_ratios=snapshot_ratios,
    )


def _carve_width_python(
    image: np.ndarray,
    target_width: int,
    *,
    faces: list | None = None,
    config: SeamCarvingConfig | None = None,
    on_seam: Callable[[int, np.ndarray, SeamStep], None] | None = None,
    snapshot_ratios: tuple[float, ...] = (),
) -> CarveResult:
    """Remove vertical seams from ``image`` until it is ``target_width`` wide.

    ``faces`` are :class:`~anime2manga.models.FaceBox` values in the *working
    image's* pixel space.  ``on_seam`` is called with the 1-based seam index,
    the image *after* the removal and the :class:`SeamStep`, which lets callers
    measure quality without the engine depending on the metrics.  Images whose
    ``ratio`` crosses a value in ``snapshot_ratios`` are kept in
    :attr:`CarveResult.snapshots` under the smallest crossing ratio.
    """
    cfg = config or SeamCarvingConfig()
    if cfg.threads > 0:
        cv2.setNumThreads(cfg.threads)

    original = _as_bgr(image)
    if original.dtype != np.uint8:
        original = np.clip(original, 0, 255).astype(np.uint8)
    width0 = original.shape[1]
    height0 = original.shape[0]
    target_width = max(1, min(int(target_width), width0))
    if target_width >= width0:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        energy0 = gradient_energy(gray)
        mean0 = float(energy0.mean())
        detail_threshold = float(np.quantile(energy0, cfg.detail_quantile))
        mask = _face_mask((height0, width0), list(faces or []), cfg)
        return CarveResult(
            image=original,
            steps=[],
            width0=width0,
            height0=height0,
            base_energy=mean0,
            energy_sum0=float(energy0.sum()),
            detail_total=int((energy0 >= detail_threshold).sum()),
            protected_total=int(mask.sum()) if mask is not None else 0,
            snapshots={r: original.copy() for r in snapshot_ratios if r <= 0},
        )

    mask = _face_mask((height0, width0), list(faces or []), cfg)
    # Detail threshold is fixed from the original image so the budget counts a
    # stable set of "important" pixels.
    gray0 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    energy0 = gradient_energy(gray0)
    base_energy = float(energy0.mean())
    detail_threshold = float(np.quantile(energy0, cfg.detail_quantile))
    detail_total = int((energy0 >= detail_threshold).sum())
    protected_total = int(mask.sum()) if mask is not None else 0
    energy_sum0 = float(energy0.sum())

    current = original
    steps: list[SeamStep] = []
    seams: list[np.ndarray] = []
    snapshots: dict[float, np.ndarray] = {r: original.copy() for r in snapshot_ratios if r <= 0}
    pending = sorted(r for r in snapshot_ratios if r > 0)
    face_energy = (
        cfg.face_energy_factor * base_energy if (cfg.protect_faces and mask is not None) else 0.0
    )
    mask_f = mask.astype(np.float32) if mask is not None else None

    while current.shape[1] > target_width:
        gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        energy = gradient_energy(gray)
        forward = _forward_costs(gray)
        saliency = mask_f * face_energy if (mask_f is not None and face_energy > 0) else None
        seam, added = find_vertical_seam(energy, forward=forward, saliency=saliency)
        if cfg.record_seams:
            seams.append(seam.copy())

        rows = np.arange(current.shape[0])
        removed = float(energy[rows, seam].mean())
        detail_pixels = int((energy[rows, seam] >= detail_threshold).sum())
        protected_pixels = int(mask_f[rows, seam].sum()) if mask_f is not None else 0
        current = _remove_vertical_seam(current, seam)
        if mask_f is not None:
            mask_f = _remove_vertical_seam(mask_f[..., None], seam)[..., 0]

        gray_after = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        energy_sum_after = float(gradient_energy(gray_after).sum())
        k = len(steps) + 1
        ratio = k / width0
        step = SeamStep(
            k=k,
            ratio=ratio,
            removed_energy=removed,
            added_energy=float(added.mean()),
            detail_pixels=detail_pixels,
            protected_pixels=protected_pixels,
            energy_sum_after=energy_sum_after,
            width_after=int(current.shape[1]),
        )
        steps.append(step)
        while pending and ratio >= pending[0]:
            snapshots[pending.pop(0)] = current.copy()
        if on_seam is not None:
            on_seam(k, current, step)

    return CarveResult(
        image=current,
        steps=steps,
        width0=width0,
        height0=height0,
        base_energy=base_energy,
        energy_sum0=energy_sum0,
        detail_total=detail_total,
        protected_total=protected_total,
        snapshots=snapshots,
        seams=seams,
    )


def carve_height(
    image: np.ndarray,
    target_height: int,
    *,
    faces: list | None = None,
    config: SeamCarvingConfig | None = None,
    on_seam: Callable[[int, np.ndarray, SeamStep], None] | None = None,
    snapshot_ratios: tuple[float, ...] = (),
) -> CarveResult:
    """Remove horizontal seams by carving a transposed copy of the image."""
    rotated = cv2.rotate(_as_bgr(image), cv2.ROTATE_90_CLOCKWISE)
    rotated_faces = None
    if faces:
        height = image.shape[0]
        # A box (x, y, w, h) becomes (x', y', w', h') after a clockwise turn.
        rotated_faces = [
            type(box)(
                x=height - (box.y + box.height),
                y=box.x,
                width=box.height,
                height=box.width,
                confidence=getattr(box, "confidence", 1.0),
            )
            for box in faces
        ]
    result = carve_width(
        rotated,
        target_height,
        faces=rotated_faces,
        config=config,
        on_seam=on_seam,
        snapshot_ratios=snapshot_ratios,
    )
    result.image = cv2.rotate(result.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    for key, value in result.snapshots.items():
        result.snapshots[key] = cv2.rotate(value, cv2.ROTATE_90_COUNTERCLOCKWISE)
    result.width0, result.height0 = result.height0, result.width0
    return result
