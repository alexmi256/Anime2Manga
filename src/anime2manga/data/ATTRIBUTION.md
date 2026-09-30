# Bundled model data

Three DeepGHS anime detection models are bundled here.  Each is the published
ONNX checkpoint simplified to a fixed `1x3x960x960` input (see below).

| Field | Value |
| --- | --- |
| File | `anime_face_detection_yolov8n.onnx` |
| Source | <https://huggingface.co/deepghs/anime_face_detection> |
| Model | `face_detect_v1.4_n` (YOLOv8n, single `face` class, 3.01M params, imgsz 640, F1 ≈ 0.94) |
| Upstream revision | `784dc4c0bb692351ddcdbe6131a050b17d3025d5` |
| Vendored file SHA-256 | `1f038cc523b129716ac6e77d90c09be28113b5dab0ceb4fa9b033a04960926de` |

| Field | Value |
| --- | --- |
| File | `anime_head_detection_yolov8s.onnx` |
| Source | <https://huggingface.co/deepghs/anime_head_detection> |
| Model | `head_detect_v2.0_s` (YOLOv8s, single `head` class, imgsz 640, F1 ≈ 0.92) |
| Upstream revision | `06604feee81983792a57c21081e539c0ae229833` |
| Vendored file SHA-256 | `713e2d4dbd72460a0310c3932c8616b6ba0f1298f9634ad2c27fba354d28dde2` |

| Field | Value |
| --- | --- |
| File | `anime_person_detection_yolov8s.onnx` |
| Source | <https://huggingface.co/deepghs/anime_person_detection> |
| Model | `person_detect_v1.3_s` (YOLOv8s, single `person` class, imgsz 640, F1 ≈ 0.86) |
| Upstream revision | `e39c744c22432ad01f91dd254fe2b02c8d878b8c` |
| Vendored file SHA-256 | `3448f4a60f7bccb7fab57904c63674788181babe668b050c2e76551a9397f6fc` |

The Hugging Face model cards declare **MIT**.

## What was changed

The published ONNX exports use dynamic input shapes, which OpenCV's DNN
importer cannot parse (`Concat` shape inference failure).  The vendored files
are the same checkpoints after `onnxsim` simplification with a fixed
`1x3x960x960` input, so they run through `cv2.dnn` with no extra runtime (no
`onnxruntime`, no ML framework).  See `scripts/fetch_face_model.py` for the
regeneration steps (`--only head`, `--only person`, `--check`).

## Licensing note (accepted risk)

The Hugging Face model cards declare **MIT**.  However, the weights were trained
with the Ultralytics YOLOv8 framework, whose license is **AGPL-3.0**, and the
project itself declares no license.  Distributing these weights in the wheel is
therefore a potential licensing risk that the maintainers should confirm before
publishing.  Shipping a differently-licensed detector (or making the models a
user-provided download) would remove the risk; this was accepted for now because
these are the only lightweight, offline, permissively-declared anime detectors
that met the recall bar.
