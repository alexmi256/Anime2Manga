#!/usr/bin/env python3
"""Regenerate the bundled anime face detection model.

Downloads ``face_detect_v1.4_n/model.onnx`` from the DeepGHS model repo at a
pinned revision, simplifies it to a fixed ``1x3x960x960`` input with ``onnxsim``
(OpenCV's DNN importer cannot parse the dynamic-shape export) and writes the
result to ``src/anime2manga/data/anime_face_detection_yolov8n.onnx``.

Requires the optional dev tools::

    pip install onnx onnxsim

Usage::

    python scripts/fetch_face_model.py          # download, simplify, write
    python scripts/fetch_face_model.py --check   # verify the vendored hash only
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

REPO = "deepghs/anime_face_detection"
#: Pinned upstream commit so the download is reproducible.
REVISION = "784dc4c0bb692351ddcdbe6131a050b17d3025d5"
SUBDIR = "face_detect_v1.4_n"
INPUT_NAME = "images"
INPUT_SIZE = 960
#: SHA-256 of the currently vendored, simplified file.
EXPECTED_SHA256 = "1f038cc523b129716ac6e77d90c09be28113b5dab0ceb4fa9b033a04960926de"

DEST = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "anime2manga"
    / "data"
    / "anime_face_detection_yolov8n.onnx"
)
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{SUBDIR}/model.onnx"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check() -> int:
    if not DEST.exists():
        print(f"missing: {DEST}")
        return 1
    digest = sha256(DEST)
    print(f"sha256: {digest}")
    if digest != EXPECTED_SHA256:
        print(f"warning: expected {EXPECTED_SHA256}")
        return 1
    print("ok: matches the vendored hash")
    return 0


def fetch() -> int:
    import onnx  # optional dev tool
    from onnxsim import simplify  # optional dev tool

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "model.onnx"
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, raw)
        model = onnx.load(str(raw))

    simplified, ok = simplify(
        model, overwrite_input_shapes={INPUT_NAME: [1, 3, INPUT_SIZE, INPUT_SIZE]}
    )
    if not ok:
        print("error: onnxsim could not simplify the model")
        return 1

    DEST.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(simplified, str(DEST))
    digest = sha256(DEST)
    print(f"wrote {DEST} ({DEST.stat().st_size} bytes)")
    print(f"sha256: {digest}")
    if digest != EXPECTED_SHA256:
        print(
            "warning: hash differs from EXPECTED_SHA256; update ATTRIBUTION.md and "
            "EXPECTED_SHA256 if the upstream model changed."
        )
        return 1
    print("ok: matches the recorded hash")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the vendored file hash only")
    args = parser.parse_args()
    return check() if args.check else fetch()


if __name__ == "__main__":
    raise SystemExit(main())
