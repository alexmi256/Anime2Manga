#!/usr/bin/env python3
"""Render panel-layout trials for the rule sets in ``anime2manga.layout``.

For a pipeline output folder (containing ``scenes.json``) this plans every row
under each selected count x placement policy, renders the resulting panels and
writes one HTML page per rule set plus an index with the ranking metrics.

The page layout is a stack of 3x2 tables (one table per page): each of the three
rows holds up to two panels, and every panel is captioned with the rule ids it
fired, the method ids used, and the rule set it came from.

Example
-------
    # clean frames re-extracted from the source video
    PYTHONPATH=src python scripts/layout_trials.py output --source input.mkv \\
        -o output/layout_trials

    # or reuse the pipeline's frames/ folder
    PYTHONPATH=src python scripts/layout_trials.py output -o output/layout_trials
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2

from anime2manga.layout import (
    COUNT_POLICIES,
    DEFAULT_SETS,
    PLACE_POLICIES,
    FrameMeta,
    LayoutConfig,
    paginate,
    plan_rows,
)
from anime2manga.layout_report import (
    PanelRenderer,
    evaluate_set,
    render_clean_set_html,
    render_index_html,
    render_set_html,
    rows_to_dict,
    rules_legend,
    write_plans_json,
)
from anime2manga.models import DetectionBox


def _boxes(raw: list[dict[str, Any]] | None) -> tuple[DetectionBox, ...]:
    if not raw:
        return ()
    return tuple(
        DetectionBox(
            x=int(b.get("x", 0)),
            y=int(b.get("y", 0)),
            width=int(b.get("width", 0)),
            height=int(b.get("height", 0)),
            confidence=float(b.get("confidence", 1.0)),
        )
        for b in raw
    )


def load_frames(scenes_path: Path) -> tuple[list[FrameMeta], dict[str, Any]]:
    """Load ``scenes.json`` into :class:`FrameMeta` values."""
    payload = json.loads(scenes_path.read_text(encoding="utf-8"))
    frames: list[FrameMeta] = []
    for scene in payload.get("scenes", []):
        frame_size = scene.get("frame_size") or scene.get("panorama_size")
        if not frame_size:
            continue
        composition = scene.get("composition") or {}
        frames.append(
            FrameMeta(
                index=int(scene["index"]),
                source_size=(int(frame_size[0]), int(frame_size[1])),
                heads=_boxes(scene.get("heads")),
                persons=_boxes(scene.get("persons")),
                body_percent=float(composition.get("body_percent") or 0.0),
                head_percent=float(composition.get("head_percent") or 0.0),
                overlap_percent=float(composition.get("body_head_overlap_percent") or 0.0),
                heads_in_body=composition.get("heads_in_body"),
                body_lean=composition.get("body_leans"),
                head_lean=composition.get("head_leans"),
                seam_carve_shrink=float(scene.get("seam_carve_shrink") or 0.0),
                seam_carve_known=scene.get("seam_carve_shrink") is not None,
                is_panorama=bool(scene.get("is_panoramic")),
                panorama_size=(
                    tuple(scene["panorama_size"]) if scene.get("panorama_size") else None
                ),
                subtitle_count=int(scene.get("subtitle_count") or 0),
                frame_time=scene.get("frame_time"),
                frame_path=scene.get("frame_path"),
                panorama_path=scene.get("panorama_inpainted_path") or scene.get("frame_path"),
            )
        )
    return frames, payload


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output", type=Path, help="Pipeline output dir (contains scenes.json)")
    parser.add_argument("-o", "--trials", type=Path, default=None,
                        help="Where to write the trial HTML/panels "
                             "(default <output>/layout_trials)")
    parser.add_argument("--source", type=Path, default=None,
                        help="Source video to re-extract clean frames from (recommended)")
    parser.add_argument("--frames-dir", type=Path, default=None,
                        help="Frame folder instead of --source (default <output>/frames)")
    parser.add_argument("--sets", nargs="*", default=None,
                        help="Rule-set names to run (default: all in DEFAULT_SETS)")
    parser.add_argument("--count", nargs="*", default=None,
                        help="Count policies to run (cross product with --place); "
                             "reaches policies not in DEFAULT_SETS")
    parser.add_argument("--place", nargs="*", default=None,
                        help="Placement policies to run (cross product with --count)")
    parser.add_argument("--working-width", type=int, default=768,
                        help="Carve/crop at this width for speed (default 768)")
    parser.add_argument("--unit-px", type=float, default=320.0,
                        help="Pixels per row unit in the captioned HTML (default 320)")
    parser.add_argument("--page-width", type=int, default=1000,
                        help="Page width in px for the rules-free preview (default 1000)")
    parser.add_argument("--gutter", type=int, default=8,
                        help="White gutter between panels in the preview (default 8)")
    parser.add_argument("--keep-floor", type=float, default=0.40)
    parser.add_argument("--max-keep", type=float, default=0.75,
                        help="Cap on the subject crop floor; 0.75 makes any two 16:9 frames fit")
    parser.add_argument("--pad-frac", type=float, default=0.06)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--clean", action="store_true", help="Remove panels before rendering")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cv2.setNumThreads(1)

    scenes_path = args.output / "scenes.json"
    if not scenes_path.exists():
        print(f"No scenes.json in {args.output}; run the pipeline first.", file=sys.stderr)
        return 1
    frames, _payload = load_frames(scenes_path)
    if not frames:
        print("scenes.json has no frames with a known size.", file=sys.stderr)
        return 1
    print(f"Loaded {len(frames)} frames from {scenes_path}")

    trials_dir: Path = args.trials or (args.output / "layout_trials")
    panels_dir = trials_dir / "panels"
    if args.clean:
        import shutil

        if panels_dir.exists():
            shutil.rmtree(panels_dir)
        for pattern in ("*.html", "*.json", "*.csv"):
            for stale in trials_dir.glob(pattern):
                stale.unlink()

    frames_dir = args.frames_dir
    if frames_dir is None and args.source is None:
        candidate = args.output / "frames"
        frames_dir = candidate if candidate.exists() else None

    layout = LayoutConfig(
        keep_floor=args.keep_floor, max_keep=args.max_keep, pad_frac=args.pad_frac
    )
    renderer = PanelRenderer(
        source=args.source,
        frames_dir=frames_dir,
        work_dir=trials_dir / "_frames",
        working_width=args.working_width,
        jpeg_quality=args.jpeg_quality,
    )

    if args.count or args.place:
        counts = args.count or sorted(COUNT_POLICIES)
        places = args.place or sorted(PLACE_POLICIES)
        unknown = [c for c in counts if c not in COUNT_POLICIES] + [
            p for p in places if p not in PLACE_POLICIES
        ]
        if unknown:
            print(f"Unknown policy names: {', '.join(unknown)}", file=sys.stderr)
            return 1
        sets = [(f"{c} x {p}", c, p) for c in counts for p in places]
    elif args.sets:
        wanted = set(args.sets)
        sets = [s for s in DEFAULT_SETS if s[0] in wanted]
    else:
        sets = list(DEFAULT_SETS)
    if not sets:
        print(f"No known sets matched {args.sets}.", file=sys.stderr)
        return 1

    stamp = str(int(time.time()))
    summaries: list[dict[str, Any]] = []
    for set_name, count_name, place_name in sets:
        count_policy = COUNT_POLICIES[count_name]
        place_policy = PLACE_POLICIES[place_name]
        slug = _slug(set_name)
        out_dir = panels_dir / slug
        print(f"[{set_name}] {count_name} x {place_name}: planning...")
        rows = plan_rows(frames, layout, count_policy, place_policy)
        pages = paginate(rows, 3)

        urls: dict[int, str] = {}
        panel_info: dict[int, tuple[str, float]] = {}
        by_index = {f.index: f for f in frames}
        for row in rows:
            for plan in row.panels:
                meta = by_index[plan.scene_index]
                out_path = out_dir / f"scene_{plan.scene_index:04d}.jpg"
                if out_path.exists():
                    existing = cv2.imread(str(out_path))
                    height, width = existing.shape[:2] if existing is not None else (1, 1)
                else:
                    width, height = renderer.write_panel(meta, plan, out_path)
                url = f"panels/{slug}/{out_path.name}"
                urls[plan.scene_index] = url
                panel_info[plan.scene_index] = (url, width / height if height else 1.0)
        metrics = evaluate_set(frames, rows, layout)
        set_html = render_set_html(
            set_name,
            f"{count_name} x {place_name}",
            count_name,
            place_name,
            pages,
            metrics=metrics,
            panel_urls=urls,
            layout=layout,
            unit_px=args.unit_px,
        )
        (trials_dir / f"{slug}.html").write_text(set_html, encoding="utf-8")
        clean_html = render_clean_set_html(
            set_name, pages, panel_info, page_width=args.page_width, gutter=args.gutter, stamp=stamp
        )
        (trials_dir / f"{slug}.clean.html").write_text(clean_html, encoding="utf-8")
        write_plans_json(
            trials_dir / f"{slug}.json",
            {"set": set_name, "count": count_name, "place": place_name,
             "layout": layout.__dict__, "metrics": metrics, "rows": rows_to_dict(rows)},
        )
        summaries.append(
            {"name": set_name, "file": f"{slug}.html", "clean": f"{slug}.clean.html",
             "count": count_name, "place": place_name, "metrics": metrics}
        )
        print(
            f"[{set_name}] {metrics['panels']} panels, {metrics['panels_per_page']}/page, "
            f"{metrics['single_rows']} single rows, {metrics['head_cuts']} head cuts"
        )

    index = render_index_html(summaries, stamp=stamp)
    index = index.replace("</h1>", "</h1>" + rules_legend(), 1)
    (trials_dir / "index.html").write_text(index, encoding="utf-8")
    write_plans_json(trials_dir / "summary.json", {"run": stamp, "sets": summaries})
    with (trials_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["name", "count", "place", "panels", "pages", "panels_per_page",
                  "single_rows", "mean_crop", "mean_carve", "head_cuts",
                  "mean_center_error", "whitespace"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in summaries:
            writer.writerow({"name": item["name"], "count": item["count"], "place": item["place"],
                             **item["metrics"]})
    print(f"Wrote {trials_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
