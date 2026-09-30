"""Command line interface for the Anime2Manga pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .audio import AudioFocusConfig
from .errors import Anime2MangaError
from .inpaint import InpaintConfig, available_inpaint_methods
from .panorama import PanConfig
from .pipeline import PipelineConfig, run_pipeline
from .report import write_report, write_scene_json
from .retarget import RetargetConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anime2manga",
        description=(
            "Convert an anime video into a manga/storyboard markdown document. "
            "Steps 1-8: metadata, subtitles, scene detection, pan stitching and "
            "infill, frame selection, audio direction and face detection."
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
        "--pan-pair-response",
        type=float,
        default=0.2,
        help="Pairs below this response are treated as cuts/blur breaks (default: 0.2).",
    )
    pan.add_argument(
        "--pan-diagonal-ratio",
        type=float,
        default=0.35,
        help="Cross-axis share needed to name a diagonal direction (default: 0.35).",
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
    pan.add_argument(
        "--pan-merge-max-len",
        type=float,
        default=1.0,
        help=(
            "Fold a scene this short (seconds) that trails a panorama into the "
            "panorama (default: 1.0; 0 disables)."
        ),
    )

    inpaint = parser.add_argument_group("panorama inpainting")
    inpaint.add_argument(
        "--inpaint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fill the transparent holes of stitched panoramas and write the "
            "result as a JPEG next to the PNG (default: on)."
        ),
    )
    inpaint.add_argument(
        "--inpaint-method",
        choices=available_inpaint_methods(),
        default="biharmonic",
        help="Infill algorithm for panorama holes (default: biharmonic).",
    )
    inpaint.add_argument(
        "--inpaint-quality",
        type=int,
        default=92,
        help="JPEG quality for filled panoramas (default: 92).",
    )
    inpaint.add_argument(
        "--inpaint-max-pixels",
        type=int,
        default=500_000,
        help=(
            "Downscale a panorama so at most this many masked pixels are solved "
            "at once (known pixels stay full resolution; 0 disables)."
        ),
    )

    face = parser.add_argument_group("face detection")
    face.add_argument(
        "--face-boxes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw bounding boxes around detected faces on chosen frames (default: on).",
    )

    seam = parser.add_argument_group("seam carving")
    seam.add_argument(
        "--seam-carve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write a seam-carved (content-aware) version of each regular frame and "
            "report its shrink percent (default: on; panoramas are never carved)."
        ),
    )
    seam.add_argument(
        "--seam-carve-energy-ratio",
        type=float,
        default=0.25,
        help=(
            "Energy guard threshold: lower stops earlier (less distortion). "
            "0 disables seam carving entirely (default: 0.25)."
        ),
    )
    seam.add_argument(
        "--seam-carve-max-shrink",
        type=float,
        default=0.5,
        help="Hard cap on width removed by seam carving (default: 0.5).",
    )
    seam.add_argument(
        "--seam-carve-working-width",
        type=int,
        default=768,
        help=(
            "Resolution the frame is carved and saved at (default: 768; the "
            "source frame is scaled to this width, so this also sets the output "
            "resolution; larger = finer/slower)."
        ),
    )
    seam.add_argument(
        "--seam-carve-detail-budget",
        type=float,
        default=0.10,
        help="Fraction of high-gradient pixels the detail guard may remove (default: 0.10).",
    )
    seam.add_argument(
        "--seam-carve-jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Worker processes for seam carving (default: min(8, CPUs)).",
    )
    seam.add_argument(
        "--seam-carve-quality",
        type=int,
        default=92,
        help="JPEG quality for seam-carved frames (default: 92).",
    )

    audio = parser.add_argument_group("audio direction")
    audio.add_argument(
        "--no-audio-direction",
        action="store_true",
        help="Disable left/right audio focus detection for the scenes.",
    )
    audio.add_argument(
        "--audio-balance-db",
        type=float,
        default=1.5,
        help="Channel imbalance (dB) required to call a side (default: 1.5).",
    )
    audio.add_argument(
        "--audio-silence-db",
        type=float,
        default=-60.0,
        help="Scenes quieter than this in the speech band (dBFS) are centered (default: -60).",
    )
    audio.add_argument(
        "--audio-band-low",
        type=float,
        default=300.0,
        help="Lower edge of the speech band used for the balance (default: 300 Hz).",
    )
    audio.add_argument(
        "--audio-band-high",
        type=float,
        default=3400.0,
        help="Upper edge of the speech band used for the balance (default: 3400 Hz).",
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
        min_pair_response=args.pan_pair_response,
        diagonal_ratio=args.pan_diagonal_ratio,
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
        pan_merge_max_len=args.pan_merge_max_len,
        inpaint=InpaintConfig(
            enabled=args.inpaint,
            method=args.inpaint_method,
            quality=args.inpaint_quality,
            max_pixels=args.inpaint_max_pixels or None,
        ),
        detect_audio=not args.no_audio_direction,
        audio=AudioFocusConfig(
            balance_threshold_db=args.audio_balance_db,
            silence_floor_db=args.audio_silence_db,
            band_low_hz=args.audio_band_low,
            band_high_hz=args.audio_band_high,
        ),
        draw_face_boxes=args.face_boxes,
        seam_carve=args.seam_carve and args.seam_carve_energy_ratio > 0,
        retarget=RetargetConfig(
            energy_ratio=args.seam_carve_energy_ratio,
            max_shrink=args.seam_carve_max_shrink,
            working_width=args.seam_carve_working_width,
            detail_budget=args.seam_carve_detail_budget,
            strip_overlays=False,
            sample_step=1.0,
            ssim_stride=0,
        ),
        seam_carve_jobs=args.seam_carve_jobs,
        seam_carve_quality=args.seam_carve_quality,
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
