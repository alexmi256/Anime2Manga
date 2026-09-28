"""Command line interface for the Anime2Manga pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .errors import Anime2MangaError
from .panorama import PanConfig
from .pipeline import PipelineConfig, run_pipeline
from .report import write_report, write_scene_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anime2manga",
        description=(
            "Convert an anime video into a manga/storyboard markdown document. "
            "Steps 1-6: metadata, subtitles, scene detection, pan stitching and "
            "frame selection."
        ),
    )
    parser.add_argument("input", type=Path, help="Input video (usually .mkv).")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for the report and extracted images (default: ./output).",
    )
    parser.add_argument("--version", action="version", version=f"anime2manga {__version__}")

    sub = parser.add_argument_group("subtitles")
    sub.add_argument(
        "--subtitle-language",
        default="eng",
        help="Subtitle language to use (ISO code, default: eng).",
    )
    sub.add_argument(
        "--subtitle-track",
        type=int,
        default=None,
        help="Explicit subtitle stream id; overrides --subtitle-language.",
    )
    sub.add_argument(
        "--translate-to",
        default=None,
        help="Reserved: translate subtitles to this language (not implemented).",
    )

    scene = parser.add_argument_group("scene detection")
    scene.add_argument(
        "--scene-method",
        choices=["select", "scdet"],
        default="select",
        help="ffmpeg detector: 'select' (0..1 score) or 'scdet' (0..100 score).",
    )
    scene.add_argument(
        "--scene-threshold",
        type=float,
        default=None,
        help="Cut threshold; default 0.4 for select and 10 for scdet.",
    )
    scene.add_argument(
        "--scene-min-len",
        type=float,
        default=0.5,
        help="Merge cuts closer than this many seconds (default: 0.5).",
    )
    scene.add_argument(
        "--start-at",
        default=None,
        help="Skip ahead to this timestamp (SS, MM:SS or HH:MM:SS[.mmm]).",
    )
    scene.add_argument(
        "--end-at",
        default=None,
        help="Stop at this timestamp (SS, MM:SS or HH:MM:SS[.mmm]).",
    )
    scene.add_argument(
        "--max-subtitles-per-scene",
        type=int,
        default=6,
        help="Split scenes carrying more cues than this at a lower threshold.",
    )
    scene.add_argument(
        "--subdivide-factor",
        type=float,
        default=0.5,
        help="Threshold multiplier used when splitting overloaded scenes.",
    )

    frame = parser.add_argument_group("frame selection")
    frame.add_argument(
        "--selection-window",
        type=float,
        default=0.75,
        help="Seconds around the target time searched for a sharp frame.",
    )
    frame.add_argument(
        "--analysis-fps",
        type=float,
        default=4.0,
        help="Frame sampling rate used for frame selection (default: 4).",
    )
    frame.add_argument(
        "--analysis-width",
        type=int,
        default=960,
        help="Width of analysis-resolution sampled frames (default: 960).",
    )

    pan = parser.add_argument_group("panorama detection")
    pan.add_argument("--no-pan", action="store_true", help="Disable pan detection.")
    pan.add_argument("--pan-fps", type=float, default=4.0, help="Pan sampling fps.")
    pan.add_argument(
        "--pan-min-shift",
        type=float,
        default=0.25,
        help="Minimum cumulative shift as a fraction of frame size (default: 0.25).",
    )
    pan.add_argument(
        "--pan-consistency",
        type=float,
        default=0.8,
        help="Minimum fraction of pairs agreeing on direction (default: 0.8).",
    )
    pan.add_argument(
        "--pan-response",
        type=float,
        default=0.6,
        help="Minimum median phase-correlation response (default: 0.6).",
    )
    pan.add_argument(
        "--pan-peek",
        type=float,
        default=3.0,
        help="Seconds to peek into the next scene for a continuing pan.",
    )
    pan.add_argument(
        "--pan-max-canvas",
        type=float,
        default=4.0,
        help="Reject pans whose canvas exceeds this multiple of a frame dimension.",
    )

    debug = parser.add_argument_group("debug")
    debug.add_argument(
        "--keep-analysis",
        action="store_true",
        help="Keep the sampled analysis frames under work/analysis/.",
    )
    debug.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output.")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    pan = PanConfig(
        sample_fps=args.pan_fps,
        min_shift_fraction=args.pan_min_shift,
        consistency=args.pan_consistency,
        min_response=args.pan_response,
        peek_seconds=args.pan_peek,
        max_canvas_factor=args.pan_max_canvas,
    )
    return PipelineConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        subtitle_language=args.subtitle_language,
        subtitle_track=args.subtitle_track,
        translate_to=args.translate_to,
        scene_method=args.scene_method,
        scene_threshold=args.scene_threshold,
        scene_min_len=args.scene_min_len,
        start_at=args.start_at,
        end_at=args.end_at,
        max_subtitles_per_scene=args.max_subtitles_per_scene,
        subdivide_factor=args.subdivide_factor,
        selection_window=args.selection_window,
        analysis_width=args.analysis_width,
        analysis_fps=args.analysis_fps,
        detect_pans=not args.no_pan,
        pan=pan,
        keep_analysis=args.keep_analysis,
        verbose=not args.quiet,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_pipeline(config_from_args(args))
    except Anime2MangaError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    report_path = write_report(result)
    json_path = write_scene_json(result)
    if not args.quiet:
        print(f"[anime2manga] report: {report_path}")
        print(f"[anime2manga] scene data: {json_path}")
        print(f"[anime2manga] scenes: {len(result.scenes)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
