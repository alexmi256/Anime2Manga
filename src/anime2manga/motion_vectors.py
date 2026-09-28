"""Motion-vector research notes and a future fast-path for pan detection.

Why this is a stub
------------------
H.264/H.265 encoders already compute block motion vectors.  ``ffprobe`` with
``-flags2 +export_mvs`` reports that a frame *has* a ``Motion vectors`` side-data
block, but it does **not** serialise the vector values - they are a binary
``AVMotionVector`` array that must be read through the libav C API (or a custom
ffmpeg build such as the one behind ``mv-extractor``).  There is therefore no
stock-command-line way to read them today.

Intended hybrid design (once a reader exists)
---------------------------------------------
1. **Detect** candidate pans cheaply with a global motion vote: take the median
   block vector of each frame, require a consistent dominant direction over the
   scene.  This avoids decoding/correlating every frame.
2. **Refine** only the candidate scenes with phase correlation
   (:func:`anime2manga.panorama.detect_pan`) to get the sub-pixel shifts needed
   for seamless stitching, since encoder vectors are block-quantised and noisy on
   flat anime frames.
3. Codec-specific parsing (H.264 vs H.265 block structure) is required, see
   https://www.volcengine.com/article/643091.

Motion vectors may also help *scene* detection: a near-total change in the
vector field between frames is a strong cut signal.
"""

from __future__ import annotations

from pathlib import Path


def motion_vectors_available() -> bool:
    """Whether a motion-vector reader is wired up (always False for now)."""
    return False


def extract_global_motion(path: Path, start: float, end: float) -> list[tuple[float, float]]:
    """Return per-frame global motion from motion vectors.

    TODO(pan fast-path): implement a reader (libav API / custom build) and return
    ``(dx, dy)`` per sampled frame.  Until then this raises ``NotImplementedError``.
    """
    _ = (path, start, end)
    raise NotImplementedError(
        "Motion-vector extraction needs a libav-side reader; stock ffprobe does "
        "not expose vector values. Use the phase-correlation pan detector."
    )
