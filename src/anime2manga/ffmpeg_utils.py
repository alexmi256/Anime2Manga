"""Thin, well-tested wrappers around the ``ffmpeg`` and ``ffprobe`` binaries.

We deliberately call the command line tools with :mod:`subprocess` rather than
using a Python binding, because the pipeline depends on parsing ffmpeg's own
diagnostic output (scene scores, ``scdet`` logs) and wants to be transparent
about the exact commands it runs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .errors import FFmpegError


def require_tools() -> None:
    """Fail fast with a friendly message when ffmpeg/ffprobe are missing."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise FFmpegError(
            "Required tool(s) not found on PATH: "
            + ", ".join(missing)
            + ". Install ffmpeg (which provides ffprobe) and try again."
        )


def run(cmd: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command, raising :class:`FFmpegError` on a non-zero exit code."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - require_tools covers this
        raise FFmpegError(f"Executable not found: {cmd[0]}", cmd) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = "\n".join(tail[-8:]) if tail else "no output"
        raise FFmpegError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{detail}", cmd)
    return proc


def run_bytes(cmd: list[str], *, timeout: float | None = None) -> bytes:
    """Run a command that streams binary stdout (raw PCM) and return those bytes.

    Unlike :func:`run` this must not decode stdout as text, so it is kept
    separate instead of overloading the text wrapper.
    """
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - require_tools covers this
        raise FFmpegError(f"Executable not found: {cmd[0]}", cmd) from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or b"").decode("utf-8", "replace")
        tail = stderr.strip().splitlines()
        detail = "\n".join(tail[-8:]) if tail else "no output"
        raise FFmpegError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{detail}", cmd)
    return proc.stdout


def ffprobe_json(
    entries: list[str],
    path: Path,
    *,
    select_streams: str | None = None,
) -> dict:
    """Return parsed JSON from ``ffprobe -show_entries``."""
    cmd = ["ffprobe", "-v", "error", "-of", "json", "-show_entries", ",".join(entries)]
    if select_streams:
        cmd += ["-select_streams", select_streams]
    cmd.append(str(path))
    proc = run(cmd)
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise FFmpegError(f"Could not parse ffprobe JSON: {exc}", cmd) from exc


def parse_rate(rate: str | None) -> float:
    """Parse an ffprobe rational rate such as ``"24000/1001"`` to a float."""
    if not rate or rate in {"0/0", "N/A"}:
        return 0.0
    if "/" in rate:
        num, _, den = rate.partition("/")
        try:
            numerator, denominator = float(num), float(den)
        except ValueError:
            return 0.0
        return numerator / denominator if denominator else 0.0
    try:
        return float(rate)
    except ValueError:
        return 0.0


def _scale_filter(scale_width: int | None) -> str | None:
    if not scale_width:
        return None
    # Preserve aspect ratio; -2 keeps an even height for yuv420p encoders.
    return f"scale={int(scale_width)}:-2"


def extract_frame(
    path: Path,
    timestamp: float,
    out_path: Path,
    *,
    scale_width: int | None = None,
    quality: int = 2,
) -> Path:
    """Extract a single frame at ``timestamp`` seconds.

    Seeks with ``-ss`` *before* ``-i`` for speed; modern ffmpeg still returns
    the exact frame.  Quality maps to the mjpeg/png encoder (lower is better;
    2 is visually lossless for mjpeg).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vf = _scale_filter(scale_width)
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y"]
    cmd += ["-ss", f"{max(timestamp, 0.0):.3f}", "-i", str(path)]
    cmd += ["-frames:v", "1", "-an", "-sn"]
    if vf:
        cmd += ["-vf", vf]
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        cmd += ["-q:v", str(quality)]
    cmd.append(str(out_path))
    run(cmd)
    return out_path


@dataclass(frozen=True)
class SampledFrame:
    """A frame written to disk while sampling a scene at a fixed rate."""

    time: float
    path: Path


def uniform_sample_times(start: float, end: float, fps: float, count: int) -> list[float]:
    """Return the ``[start, end)`` subset of a uniform ``fps`` sample grid.

    The first ``count`` grid points are ``start + i / fps``.  ffmpeg's ``fps``
    filter can emit one extra frame sitting exactly on ``end``; that frame
    belongs to the *next* scene, so it is dropped here.  Keeping the range
    half-open avoids stitching the boundary frame into a panorama and reusing it
    as the following scene's representative frame.
    """
    if fps <= 0:
        return []
    return [t for i in range(count) if (t := start + i / fps) < end]


def extract_frames_at_fps(
    path: Path,
    start: float,
    end: float,
    fps: float,
    out_dir: Path,
    *,
    scale_width: int | None = None,
    quality: int = 3,
) -> list[SampledFrame]:
    """Extract frames on a uniform grid across ``[start, end)``.

    Times are assigned as ``start + i / fps`` which matches the ``fps`` filter's
    constant-rate output closely enough for ranking frames by sharpness and for
    phase-correlation strides.  Grid points on or past ``end`` are dropped so the
    range stays half-open (the boundary frame belongs to the next scene).  The
    single *chosen* frame is always re-extracted at its exact timestamp before
    being written to the report.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()
    duration = max(end - start, 0.0)
    if duration <= 0 or fps <= 0:
        return []
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y"]
    cmd += ["-ss", f"{max(start, 0.0):.3f}", "-i", str(path)]
    cmd += ["-t", f"{duration:.3f}", "-an", "-sn"]
    vf = f"fps={fps}"
    scaled = _scale_filter(scale_width)
    if scaled:
        vf = f"{scaled},{vf}"
    cmd += ["-vf", vf, "-q:v", str(quality), "-fps_mode", "cfr", str(out_dir / "%06d.jpg")]
    run(cmd)
    frames = sorted(out_dir.glob("*.jpg"))
    times = uniform_sample_times(start, end, fps, len(frames))
    return [SampledFrame(time, p) for time, p in zip(times, frames, strict=False)]
