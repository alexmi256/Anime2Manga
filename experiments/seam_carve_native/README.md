# Native seam-carving benchmark

The compiled seam-carving engine now lives in the package
(`src/anime2manga/_seamcarve.cpp`, built into `anime2manga._seamcarve` by
`just install` / `pip install -e .`) and `seam_carving.carve_width` dispatches to
it by default, falling back to the pure-Python loop when it is not built.

This folder is just the benchmark that compares the two backends.  It contains
no engine code.

## Run

```bash
cd experiments/seam_carve_native
../../.venv/bin/python benchmark.py /home/alex/PycharmProjects/Anime2Manga/output/frames \
    --frames 12 --shrink 0.7 --out benchmark_results.json
```

`--shrink 0.7` carves 1920 -> 1344 (576 seams) per frame.  The harness times the
pure-Python backend (`seam_carving._carve_width_python`) against the compiled
one (`seam_carving.carve_width`) and asserts the carved images are identical.
`--width N` downscales each frame to `N` first (`--width 0`, the default, carves
at the frame's native resolution; `--width 768` matches the pipeline default).

## Correctness

The compiled backend is bit-identical to the pure-Python one in the carved image
and seam trace, both without faces and with face protection, and the per-seam
metrics agree to float rounding (`tests/test_seam_carving.py` checks parity and
the fallback).  Getting there required matching three subtleties of the
NumPy/OpenCV path:

1. OpenCV's `cvtColor(BGR2GRAY)` uses the higher-precision 15-bit coefficients
   (`B2Y=3735, G2Y=19235, R2Y=9798`, shift 15); the 14-bit set is off by +-1 on
   a few pixels and eventually flips seams.
2. `cv2.getStructuringElement(MORPH_ELLIPSE)` rounds the per-row half-width
   (`cvRound`), so it is not exactly the mathematical disk.
3. `mask_f * face_energy` rounds the Python-float scalar to float32 once, which
   the C++ reproduces by multiplying in double then casting.

## Results

12 frames from `output/frames`, each carved to 70% of its width, best of one
timed run per frame, `cv2` threads pinned to 1.  All outputs were bit-identical.

Native frame resolution (1920x1080 -> 1344, 576 seams):

| backend | total | per frame | speedup |
| --- | ---: | ---: | ---: |
| pure Python | 491.6 s | 41.0 s | 1.0x |
| compiled | 86.2 s | 7.2 s | **5.7x** |

Pipeline working width (768 -> 537, 231 seams), i.e. with the pre-carve
downscale the pipeline does by default:

| backend | total | per frame | speedup |
| --- | ---: | ---: | ---: |
| pure Python | 36.3 s | 3.02 s | 1.0x |
| compiled | 5.3 s | 0.44 s | **6.9x** |

Raw data: `benchmark_native.json`, `benchmark_768.json` (+ the `.log` files).
The compiled build uses `-march=native`; without it the native-resolution
speedup is ~3.4x.
