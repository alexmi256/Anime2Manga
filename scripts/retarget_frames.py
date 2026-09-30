#!/usr/bin/env python3
"""Run the seam-carving retarget experiment over a folder of frames.

For every frame it carves vertical seams down to the 50% cap, records the
quality signals, and writes carved stills, contact strips, signal plots and a
combined report.  See ``src/anime2manga/retarget.py`` for the metric itself.

Example
-------
    PYTHONPATH=src python scripts/retarget_frames.py \
        ~/PycharmProjects/Anime2Manga/output/frames -o output/retarget -j 8
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

from anime2manga.retarget import METHODS, RetargetConfig, analyze_frame
from anime2manga.retarget_report import (
    build_overview_sheets,
    summary_record,
    write_frame_outputs,
    write_summary,
)

#: Image extensions the driver will pick up in the frames folder.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _gather_frames(
    frames_dir: Path, names: list[str] | None, limit: int | None, sample: int | None
) -> list[Path]:
    if names:
        resolved: list[Path] = []
        for name in names:
            candidate = frames_dir / name
            if not candidate.exists():
                for suffix in IMAGE_SUFFIXES:
                    if (frames_dir / f"{name}{suffix}").exists():
                        candidate = frames_dir / f"{name}{suffix}"
                        break
            resolved.append(candidate)
        missing = [str(p) for p in resolved if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Frames not found: {', '.join(missing)}")
        return resolved
    paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if sample and sample < len(paths):
        step = len(paths) / sample
        paths = [paths[min(len(paths) - 1, int(i * step))] for i in range(sample)]
    if limit:
        paths = paths[:limit]
    return paths


def _clean_output(out_dir: Path) -> None:
    """Remove previously generated artefacts, keeping any foreign files."""
    import shutil

    for name in ("carved", "contacts", "plots", "thumbs", "overview"):
        target = out_dir / name
        if target.is_dir():
            shutil.rmtree(target)
    for name in ("summary.csv", "summary.json", "report.md", "report.html"):
        target = out_dir / name
        if target.is_file():
            target.unlink()


def _process_one(
    path_str: str,
    config: RetargetConfig,
    dirs: dict[str, str],
    panel_width: int,
    thumb_width: int,
    jpeg_quality: int,
) -> dict:
    path = Path(path_str)
    analysis = analyze_frame(path, config=config)
    if analysis is None:
        return {"name": path.stem, "error": "unreadable"}
    write_frame_outputs(
        analysis,
        carved_dir=Path(dirs["carved"]),
        contacts_dir=Path(dirs["contacts"]),
        plots_dir=Path(dirs["plots"]),
        thumbs_dir=Path(dirs["thumbs"]),
        panel_width=panel_width,
        thumb_width=thumb_width,
        jpeg_quality=jpeg_quality,
    )
    return summary_record(analysis)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Parameter details and tuning recipes: docs/retarget_metrics.md\n"
            "Lower --energy-ratio and --detail-budget stop earlier (less distortion)."
        ),
    )
    parser.add_argument("frames_dir", type=Path, help="Folder of frames to analyse")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/retarget"))
    parser.add_argument("-j", "--jobs", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None, help="Only the first N frames")
    parser.add_argument("--sample", type=int, default=None, help="Evenly spaced N frames")
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Explicit frame names (with or without extension)",
    )
    parser.add_argument(
        "--max-shrink", type=float, default=0.5, help="Hard cap on width removed (0.5 = half)"
    )
    parser.add_argument(
        "--working-width", type=int, default=768, help="Carving/analysis resolution"
    )
    parser.add_argument(
        "--sample-step", type=float, default=0.1, help="Save a still every this fraction"
    )
    parser.add_argument(
        "--energy-ratio",
        type=float,
        default=0.35,
        help=(
            "Energy guard base threshold; lower = stricter = stops earlier. "
            "The library default is 0.35; the anime2manga pipeline uses 0.25 "
            "(the tuned value) - pass 0.25 to compare like for like."
        ),
    )
    parser.add_argument(
        "--energy-baseline-multiple",
        type=float,
        default=1.4,
        help="Adaptive margin per unit of early-seam energy above --energy-reference (0 disables)",
    )
    parser.add_argument(
        "--energy-reference",
        type=float,
        default=0.25,
        help="Early-seam energy above which the adaptive margin applies",
    )
    parser.add_argument(
        "--detail-budget",
        type=float,
        default=0.10,
        help="Fraction of high-gradient pixels the guard may remove; lower = stricter",
    )
    parser.add_argument(
        "--face-budget",
        type=float,
        default=0.02,
        help="Fraction of face pixels that may be removed",
    )
    parser.add_argument(
        "--forward-knee-factor",
        type=float,
        default=2.5,
        help="Forward-energy guard multiple of its warm-up baseline (optional guard)",
    )
    parser.add_argument(
        "--ssim-floor",
        type=float,
        default=0.5,
        help="Structural guard SSIM floor (optional guard)",
    )
    parser.add_argument(
        "--enabled",
        default="energy,detail",
        help=f"Comma-separated guards feeding the composite ({','.join(METHODS)})",
    )
    parser.add_argument("--primary-method", default="composite", choices=("composite", *METHODS))
    parser.add_argument(
        "--no-face-protection", action="store_true", help="Disable the energetic face mask"
    )
    parser.add_argument(
        "--keep-face-boxes", action="store_true", help="Do not strip green overlays"
    )
    parser.add_argument("--panel-width", type=int, default=320)
    parser.add_argument("--thumb-width", type=int, default=240)
    parser.add_argument("--overview-width", type=int, default=300, help="Overview cell width")
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--no-overview", action="store_true")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete previously generated carved/contacts/plots/thumbs/overview first",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cv2.setNumThreads(1)

    config = RetargetConfig(
        max_shrink=args.max_shrink,
        working_width=args.working_width,
        sample_step=args.sample_step,
        energy_ratio=args.energy_ratio,
        energy_baseline_multiple=args.energy_baseline_multiple,
        energy_reference=args.energy_reference,
        detail_budget=args.detail_budget,
        face_budget=args.face_budget,
        forward_knee_factor=args.forward_knee_factor,
        ssim_floor=args.ssim_floor,
        enabled_methods=tuple(m.strip() for m in args.enabled.split(",") if m.strip()),
        primary_method=args.primary_method,
        protect_faces=not args.no_face_protection,
        strip_overlays=not args.keep_face_boxes,
    )

    frames = _gather_frames(args.frames_dir, args.names, args.limit, args.sample)
    if not frames:
        print(f"No frames found in {args.frames_dir}", file=sys.stderr)
        return 1

    out_dir: Path = args.output
    if args.clean:
        _clean_output(out_dir)
    dirs = {
        "carved": str(out_dir / "carved"),
        "contacts": str(out_dir / "contacts"),
        "plots": str(out_dir / "plots"),
        "thumbs": str(out_dir / "thumbs"),
    }
    for directory in dirs.values():
        Path(directory).mkdir(parents=True, exist_ok=True)

    print(f"Analysing {len(frames)} frames with {args.jobs} workers -> {out_dir}")
    records: list[dict] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as pool:
        futures = {
            pool.submit(
                _process_one,
                str(path),
                config,
                dirs,
                args.panel_width,
                args.thumb_width,
                args.jpeg_quality,
            ): path
            for path in frames
        }
        for done, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            if "error" in record:
                print(f"  [{done}/{len(frames)}] {record['name']}: {record['error']}")
                continue
            records.append(record)
            print(
                f"  [{done}/{len(frames)}] {record['name']}: "
                f"stop {record['composite_ratio'] * 100:.0f}% "
                f"(energy {record['energy_ratio'] * 100:.0f}%, "
                f"detail {record['detail_ratio'] * 100:.0f}%)",
                flush=True,
            )

    records.sort(key=lambda r: r["name"])
    paths = write_summary(out_dir, records, config)
    print("Wrote:")
    for label, path in paths.items():
        print(f"  {label:9s} {path}")
    if not args.no_overview:
        sheets = build_overview_sheets(
            [r for r in records if "error" not in r],
            thumbs_dir=Path(dirs["thumbs"]),
            sheet_dir=out_dir / "overview",
            cell_width=args.overview_width,
        )
        print(f"  overview  {len(sheets)} sheet(s) in {out_dir / 'overview'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
