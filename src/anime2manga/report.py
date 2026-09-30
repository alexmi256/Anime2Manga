"""Markdown report generation (the deliverable for steps 1-8).

The report is the review surface: it shows which subtitle track was used, how the
video was split into scenes, which frame was chosen for each scene, whether that
frame is a stitched panorama, and which subtitle lines land in the scene.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import DetectionBox, PipelineResult, Scene


def format_time(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.mmm``."""
    total = max(seconds, 0.0)
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _relative(path: Path | None, base: Path) -> str:
    if path is None:
        return "(none)"
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


#: Human-facing labels for :attr:`Scene.audio_focus`.
_AUDIO_DIRECTION_LABELS = {"left": "Left", "center": "Center", "right": "Right"}


def audio_direction_label(value: str) -> str:
    """Title-case a stored audio focus for the report (``center`` -> ``Center``)."""
    return _AUDIO_DIRECTION_LABELS.get(value, value.title())


def build_markdown(result: PipelineResult) -> str:
    """Render the whole report as a markdown string."""
    media = result.media
    base = result.output_dir
    lines: list[str] = []

    lines.append("# Anime2Manga")
    lines.append("")
    lines.append(f"Input File: {media.path.name}")
    lines.append("Subtitle Tracks:")
    for track in media.subtitle_tracks:
        lines.append(f"- {track.index}: {track.title or track.language or 'unknown'}")
    lines.append(f"Duration: {format_time(media.duration)}")
    lines.append("")
    lines.append(f"Selected Subtitle Track: {result.subtitle_track.label}")
    lines.append(
        f"Clip Window: {format_time(result.clip.start)} - "
        f"{format_time(result.clip.end)} (source: {result.clip.source})"
    )
    lines.append(f"Scene Count: {len(result.scenes)}")
    lines.append("")

    for scene in result.scenes:
        lines.extend(_render_scene(scene, base))

    return "\n".join(lines).rstrip() + "\n"


def _detection_lines(label: str, boxes, *, has_frame: bool, empty: str) -> list[str]:
    """Render one category's ``X Bounding Boxes:`` block."""
    out = [f"{label} Bounding Boxes:"]
    if not has_frame:
        out.append("- (no chosen frame)")
    elif boxes:
        for box in boxes:
            out.append(
                f"- x={box.x}, y={box.y}, width={box.width}, "
                f"height={box.height}, confidence={box.confidence:.2f}"
            )
    else:
        out.append(f"- (no {empty} detected)")
    return out


def _render_scene(scene: Scene, base: Path) -> list[str]:
    lines: list[str] = []
    lines.append(f"# Scene Number {scene.index}")
    lines.append("")
    lines.append(f"Audio Direction: {audio_direction_label(scene.audio_focus)}")
    lines.append(f"Start Time: {format_time(scene.start)}")
    lines.append(f"End Time: {format_time(scene.end)}")
    lines.append(f"Start Frame: {scene.start_frame}")
    lines.append(f"End Frame: {scene.end_frame}")
    lines.append("")
    lines.append("## Chosen Frame")
    frame_ref = _relative(scene.frame_path, base)
    if scene.frame_path is not None:
        lines.append(f"![Frame Image]({frame_ref})")
        if scene.seam_carved_path is not None:
            seam_ref = _relative(scene.seam_carved_path, base)
            lines.append(f"![Seam Carved Frame]({seam_ref})")
        if scene.panorama_inpainted_path is not None:
            filled_ref = _relative(scene.panorama_inpainted_path, base)
            lines.append(f"![Inpainted Panorama]({filled_ref})")
    else:
        lines.append(f"Frame Image: {frame_ref}")
    size = f"{scene.frame_size[0]}x{scene.frame_size[1]}" if scene.frame_size else "unknown"
    lines.append(f"Frame Size: {size}")
    lines.append(f"Is Panoramic: {'Yes' if scene.is_panoramic else 'No'}")
    if scene.seam_carved_path is not None and scene.seam_carve_shrink is not None:
        lines.append(f"Seam Carve Shrink Percent: {round(scene.seam_carve_shrink * 100)}%")
        if scene.seam_carve_size is not None:
            lines.append(
                f"Seam Carved Frame Size: "
                f"{scene.seam_carve_size[0]}x{scene.seam_carve_size[1]}"
            )
    if scene.is_panoramic and scene.pan_direction:
        lines.append(f"Pan Direction: {scene.pan_direction}")
    if scene.is_panoramic and scene.pan_start is not None and scene.pan_end is not None:
        lines.append(f"Pan Start Time: {format_time(scene.pan_start)}")
        lines.append(f"Pan End Time: {format_time(scene.pan_end)}")
        lines.append(f"Pan Start Frame: {scene.frame_at(scene.pan_start)}")
        lines.append(f"Pan End Frame: {scene.frame_at(scene.pan_end)}")
    if scene.frame_time is not None:
        lines.append(f"Frame Time: {format_time(scene.frame_time)}")
    if scene.frame_time is not None and not scene.is_panoramic:
        lines.append(f"Frame Number: {scene.frame_at(scene.frame_time)}")
    has_frame = scene.frame_path is not None
    lines.extend(_detection_lines("Face", scene.faces, has_frame=has_frame, empty="faces"))
    lines.extend(_detection_lines("Head", scene.heads, has_frame=has_frame, empty="heads"))
    lines.extend(
        _detection_lines("Person", scene.persons, has_frame=has_frame, empty="persons")
    )
    lines.append("")
    lines.append("## Text")
    if scene.subtitles:
        for subtitle in scene.subtitles:
            lines.append(f"- {subtitle.text}")
    else:
        lines.append("- (no subtitles)")
    lines.append("")
    return lines


def write_report(result: PipelineResult) -> Path:
    """Write ``report.md`` into the output directory and return its path."""
    path = result.output_dir / "report.md"
    path.write_text(build_markdown(result), encoding="utf-8")
    return path


def _boxes(boxes: list[DetectionBox]) -> list[dict]:
    """Serialise detection boxes for ``scenes.json``."""
    return [
        {
            "x": box.x,
            "y": box.y,
            "width": box.width,
            "height": box.height,
            "confidence": box.confidence,
        }
        for box in boxes
    ]


def scene_to_dict(scene: Scene) -> dict:
    """Serialise a scene for the machine-readable ``scenes.json`` debug dump."""
    return {
        "index": scene.index,
        "start": round(scene.start, 3),
        "end": round(scene.end, 3),
        "start_frame": scene.start_frame,
        "end_frame": scene.end_frame,
        "duration": round(scene.duration, 3),
        "is_panoramic": scene.is_panoramic,
        "pan_direction": scene.pan_direction,
        "pan_shift": scene.pan_shift,
        "pan_start": round(scene.pan_start, 3) if scene.pan_start is not None else None,
        "pan_end": round(scene.pan_end, 3) if scene.pan_end is not None else None,
        "pan_start_frame": (
            scene.frame_at(scene.pan_start) if scene.pan_start is not None else None
        ),
        "pan_end_frame": (
            scene.frame_at(scene.pan_end) if scene.pan_end is not None else None
        ),
        "panorama_size": scene.panorama_size,
        "panorama_inpainted_path": (
            scene.panorama_inpainted_path.as_posix()
            if scene.panorama_inpainted_path
            else None
        ),
        "inpaint_method": scene.inpaint_method,
        "frame_path": scene.frame_path.as_posix() if scene.frame_path else None,
        "frame_time": round(scene.frame_time, 3) if scene.frame_time is not None else None,
        "frame_number": (
            scene.frame_at(scene.frame_time)
            if scene.frame_time is not None and not scene.is_panoramic
            else None
        ),
        "frame_size": scene.frame_size,
        "blur_score": round(scene.blur_score, 3) if scene.blur_score is not None else None,
        "seam_carved_path": (
            scene.seam_carved_path.as_posix() if scene.seam_carved_path else None
        ),
        "seam_carve_shrink": (
            round(scene.seam_carve_shrink, 4)
            if scene.seam_carve_shrink is not None
            else None
        ),
        "seam_carve_size": scene.seam_carve_size,
        "faces": _boxes(scene.faces),
        "heads": _boxes(scene.heads),
        "persons": _boxes(scene.persons),
        "subtitle_count": len(scene.subtitles),
        "subtitles": [
            {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
            for s in scene.subtitles
        ],
        "audio_focus": scene.audio_focus,
        "audio_direction": audio_direction_label(scene.audio_focus),
        "audio_balance_db": (
            round(scene.audio_balance_db, 2)
            if scene.audio_balance_db is not None
            else None
        ),
        "notes": scene.notes,
    }


def write_scene_json(result: PipelineResult) -> Path:
    """Write ``scenes.json`` with per-scene diagnostics."""
    path = result.output_dir / "scenes.json"
    payload = {
        "input": result.media.path.name,
        "duration": round(result.media.duration, 3),
        "clip": {
            "start": round(result.clip.start, 3),
            "end": round(result.clip.end, 3),
            "source": result.clip.source,
        },
        "scenes": [scene_to_dict(scene) for scene in result.scenes],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
