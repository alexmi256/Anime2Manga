"""Type stub for the compiled seam-carving extension.

The real module is ``_seamcarve.cpp`` built by ``setup.py``; this stub keeps
static type checkers happy and documents the one symbol the package uses.

``carve_width_full`` (reached through :func:`carve_addr`) protects every box in
its ``boxes``/``n_boxes`` array identically, so callers pass the combined
face/head/person detections there.
"""

def carve_addr() -> int:
    """Return the address of the native ``carve_width_full`` symbol."""
    ...
