#!/usr/bin/env python3
"""Regenerate the bundled DeepGHS anime detection models.

Downloads the requested ``model.onnx`` from the DeepGHS detection repositories at
pinned revisions, simplifies each to a fixed ``1x3x<size>x<size>`` input with
``onnxsim`` (OpenCV's DNN importer cannot parse the dynamic-shape exports) and
writes the result into ``src/anime2manga/data/``.

Models the script has already produced (their vendored SHA-256 matches the
recorded value) are skipped, so re-running never rewrites a good file.

Requires the optional dev tools::

    pip install onnx onnxsim

Usage::

    python scripts/fetch_face_model.py                 # fetch every missing model
    python scripts/fetch_face_model.py --only head     # just one model
    python scripts/fetch_face_model.py --check         # verify vendored hashes only
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEST_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "anime2manga" / "data"
)


@dataclass(frozen=True)
class ModelSpec:
    """One bundled model: where it comes from and where it lands."""

    name: str
    repo: str
    #: Pinned upstream commit so the download is reproducible.
    revision: str
    #: Directory inside the repository that holds ``model.onnx``.
    subdir: str
    #: File name written into ``src/anime2manga/data/``.
    dest: str
    #: SHA-256 of the currently vendored, simplified file.
    expected_sha256: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.subdir}/model.onnx"

    @property
    def path(self) -> Path:
        return DEST_DIR / self.dest


INPUT_NAME = "images"
INPUT_SIZE = 960

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="face",
        repo="deepghs/anime_face_detection",
        revision="784dc4c0bb692351ddcdbe6131a050b17d3025d5",
        subdir="face_detect_v1.4_n",
        dest="anime_face_detection_yolov8n.onnx",
        expected_sha256="1f038cc523b129716ac6e77d90c09be28113b5dab0ceb4fa9b033a04960926de",
    ),
    ModelSpec(
        name="head",
        repo="deepghs/anime_head_detection",
        revision="06604feee81983792a57c21081e539c0ae229833",
        subdir="head_detect_v2.0_s",
        dest="anime_head_detection_yolov8s.onnx",
        expected_sha256="713e2d4dbd72460a0310c3932c8616b6ba0f1298f9634ad2c27fba354d28dde2",
    ),
    ModelSpec(
        name="person",
        repo="deepghs/anime_person_detection",
        revision="e39c744c22432ad01f91dd254fe2b02c8d878b8c",
        subdir="person_detect_v1.3_s",
        dest="anime_person_detection_yolov8s.onnx",
        expected_sha256="3448f4a60f7bccb7fab57904c63674788181babe668b050c2e76551a9397f6fc",
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(specs: list[ModelSpec]) -> int:
    status = 0
    for spec in specs:
        if not spec.path.exists():
            print(f"{spec.name}: missing {spec.path}")
            status = 1
            continue
        digest = sha256(spec.path)
        if spec.expected_sha256 and digest != spec.expected_sha256:
            print(f"{spec.name}: sha256 {digest} (expected {spec.expected_sha256})")
            status = 1
        else:
            print(f"{spec.name}: ok sha256 {digest}")
    return status


def fetch_one(spec: ModelSpec) -> int:
    import onnx  # optional dev tool
    from onnxsim import simplify  # optional dev tool

    if spec.path.exists() and spec.expected_sha256 and sha256(spec.path) == spec.expected_sha256:
        print(f"{spec.name}: already up to date")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "model.onnx"
        print(f"{spec.name}: downloading {spec.url}")
        urllib.request.urlretrieve(spec.url, raw)
        model = onnx.load(str(raw))

    simplified, ok = simplify(
        model, overwrite_input_shapes={INPUT_NAME: [1, 3, INPUT_SIZE, INPUT_SIZE]}
    )
    if not ok:
        print(f"{spec.name}: error: onnxsim could not simplify the model")
        return 1

    spec.path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(simplified, str(spec.path))
    digest = sha256(spec.path)
    print(f"{spec.name}: wrote {spec.path} ({spec.path.stat().st_size} bytes)")
    print(f"{spec.name}: sha256 {digest}")
    if spec.expected_sha256 and digest != spec.expected_sha256:
        print(f"{spec.name}: warning: differs from the recorded hash; update ATTRIBUTION.md")
        return 1
    return 0


def _select(only: str | None) -> list[ModelSpec]:
    if only is None:
        return list(MODELS)
    selected = [spec for spec in MODELS if spec.name == only]
    if not selected:
        names = ", ".join(spec.name for spec in MODELS)
        raise SystemExit(f"unknown model {only!r}; choose one of: {names}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify vendored hashes only")
    parser.add_argument("--only", choices=[spec.name for spec in MODELS], default=None)
    args = parser.parse_args()

    specs = _select(args.only)
    if args.check:
        return check(specs)
    status = 0
    for spec in specs:
        status |= fetch_one(spec)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
