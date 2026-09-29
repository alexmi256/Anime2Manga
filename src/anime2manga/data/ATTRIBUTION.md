# Bundled model data

`anime_face_detection_yolov8n.onnx` is derived from the DeepGHS
`face_detect_v1.4_n` anime face detection model:

| Field | Value |
| --- | --- |
| Source | <https://huggingface.co/deepghs/anime_face_detection> |
| Architecture | YOLOv8n, single `face` class, 3.01M params, imgsz 640 |
| Upstream revision | `784dc4c0bb692351ddcdbe6131a050b17d3025d5` |
| Upstream file | `face_detect_v1.4_n/model.onnx` |
| Upstream license | MIT (declared in the model card) |
| Vendored file SHA-256 | `1f038cc523b129716ac6e77d90c09be28113b5dab0ceb4fa9b033a04960926de` |

## What was changed

The published ONNX export uses dynamic input shapes, which OpenCV's DNN importer
cannot parse (`Concat` shape inference failure).  The vendored file is the same
checkpoint after `onnxsim` simplification with a fixed `1x3x960x960` input, so it
runs through `cv2.dnn` with no extra runtime (no `onnxruntime`, no ML
framework).  See `scripts/fetch_face_model.py` for the regeneration steps.

## Licensing note (accepted risk)

The Hugging Face model card declares **MIT**.  However, the weights were trained
with the Ultralytics YOLOv8 framework, whose license is **AGPL-3.0**, and the
project itself declares no license.  Distributing these weights in the wheel is
therefore a potential licensing risk that the maintainers should confirm before
publishing.  Shipping a differently-licensed detector (or making the model a
user-provided download) would remove the risk; this was accepted for now because
the model is the only lightweight, offline, permissively-declared anime detector
that met the recall bar.
