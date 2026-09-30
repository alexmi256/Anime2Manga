"""Benchmark the compiled seam-carving backend against the pure-Python one.

`anime2manga.seam_carving.carve_width` now dispatches to the compiled engine
(`_seamcarve.cpp`) by default, with the pure-Python loop as fallback.  This
harness times both over a folder of frames and checks that they agree.

Usage:
    python benchmark.py /path/to/frames --frames 12 --shrink 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
# Make `anime2manga` importable straight from the checkout without PYTHONPATH.
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from anime2manga import seam_carving as sc  # noqa: E402


def _timed(fn, repeats: int):
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best, result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames_dir", help="folder of frame images")
    ap.add_argument("--frames", type=int, default=12, help="number of frames to sample")
    ap.add_argument("--shrink", type=float, default=0.7, help="target width as a fraction")
    ap.add_argument(
        "--width",
        type=int,
        default=0,
        help="downscale frames to this width first; 0 = carve at native resolution",
    )
    ap.add_argument("--repeats", type=int, default=1, help="timed repetitions per frame")
    ap.add_argument("--out", type=Path, default=_HERE / "benchmark_results.json")
    args = ap.parse_args()

    if not sc.HAVE_NATIVE:
        print("native backend not built; run `just install` (or `pip install -e .`)")
        return 1
    cv2.setNumThreads(1)  # match the engines' single-threaded default

    files = sorted(
        p
        for p in Path(args.frames_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not files:
        print(f"no frames found in {args.frames_dir}")
        return 1
    sample = files[:: max(1, len(files) // args.frames)][: args.frames]

    cfg = sc.SeamCarvingConfig(record_seams=False, threads=1)

    # Warm up both code paths so caches/allocators don't skew the first frame.
    warm = cv2.imread(str(sample[0]))
    sc.carve_width(warm, int(warm.shape[1] * 0.98), config=cfg)
    sc._carve_width_python(warm, int(warm.shape[1] * 0.98), config=cfg)

    rows = []
    print(
        f"sampling {len(sample)} of {len(files)} frames, shrink={args.shrink}, "
        f"width={'native' if args.width == 0 else args.width}"
    )
    print(f"{'frame':<14}{'in_w':>6}{'seams':>6}{'python(s)':>11}{'native(s)':>11}{'speedup':>9}{'exact':>7}")
    for path in sample:
        img = cv2.imread(str(path))
        if img is None:
            continue
        if args.width and args.width < img.shape[1]:
            scale = args.width / img.shape[1]
            img = cv2.resize(
                img,
                (args.width, max(1, round(img.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        target = int(img.shape[1] * args.shrink)

        t_py, py_img = _timed(
            lambda: sc._carve_width_python(img, target, config=cfg).image, args.repeats
        )
        t_native, native_img = _timed(
            lambda: sc.carve_width(img, target, config=cfg).image, args.repeats
        )
        exact = bool(np.array_equal(py_img, native_img))

        seams = img.shape[1] - target
        rows.append(
            {
                "frame": path.name,
                "width": img.shape[1],
                "seams": seams,
                "python_s": t_py,
                "native_s": t_native,
                "speedup": t_py / t_native,
                "exact": exact,
            }
        )
        print(
            f"{path.name:<14}{img.shape[1]:>6}{seams:>6}{t_py:>11.2f}{t_native:>11.2f}"
            f"{t_py / t_native:>8.1f}x{str(exact):>7}",
            flush=True,
        )

    py_total = sum(r["python_s"] for r in rows)
    native_total = sum(r["native_s"] for r in rows)
    summary = {
        "frames": len(rows),
        "input_width": args.width if args.width else "native",
        "seams_total": sum(r["seams"] for r in rows),
        "python_total_s": py_total,
        "native_total_s": native_total,
        "speedup_overall": py_total / native_total,
        "all_exact": all(r["exact"] for r in rows),
        "shrink": args.shrink,
        "cv2_version": cv2.__version__,
        "numpy_version": np.__version__,
    }
    print("-" * 70)
    print(
        f"TOTAL  python={py_total:.1f}s  native={native_total:.1f}s  "
        f"speedup={summary['speedup_overall']:.1f}x"
    )
    print(f"all outputs bit-exact: {summary['all_exact']}")
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
