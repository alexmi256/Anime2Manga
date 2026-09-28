"""Step 8 (planned) - face detection on chosen frames.

Not implemented yet.  Face boxes are needed for two later decisions:

* step 9 (cropping) keeps faces inside a square crop, and
* step 10 (text placement) keeps speech bubbles off faces.

Planned approach
----------------
* Load the chosen frame as BGR.
* Run a face detector.  OpenCV ships Haar cascades (``cv2.CascadeClassifier``)
  and, in recent versions, YuNet (``cv2.FaceDetectorYN``); either avoids a heavy
  ML dependency.  A DNN detector can be swapped in behind this function.
* For anime, faces are stylised, so tune parameters (``scaleFactor``,
  ``minNeighbors``, ``minSize``) and consider running on a downscaled image.
* Return boxes in full-resolution frame pixel coordinates as ``FaceBox`` values.

Until implemented, :func:`detect_faces` returns an empty list so cropping and
text placement fall back to centre-based behaviour.
"""

from __future__ import annotations

from pathlib import Path

from .models import FaceBox


def detect_faces(frame_path: Path) -> list[FaceBox]:
    """Return face bounding boxes in ``frame_path``.

    TODO(step 8): implement as described in the module docstring and populate
    ``Scene.faces``.
    """
    _ = frame_path
    return []
