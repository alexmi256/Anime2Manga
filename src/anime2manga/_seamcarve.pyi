"""Type stub for the compiled seam-carving extension.

The real module is ``_seamcarve.cpp`` built by ``setup.py``; this stub keeps
static type checkers happy and documents the one symbol the package uses.
"""

def carve_addr() -> int:
    """Return the address of the native ``carve_width_full`` symbol."""
    ...
