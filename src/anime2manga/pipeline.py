"""Pipeline orchestration for steps 1-6.

Order of operations
-------------------
1. Probe the container for duration/tracks/chapters and resolve the clip window
   (``--start-at`` / ``--end-at`` or intro/credits chapters).
2. Select, extract and parse the subtitle track; drop cues in excluded ranges.
3. Detect scenes with ffmpeg (``select`` or ``scdet``).
4. Detect pans per scene and, when a pan spills over a boundary, retime the
   affected scenes and stitch a panorama.
5. Validate that scenes still tile the clip with no gaps/overlaps.
6. Re-detect text-overloaded scenes at a lower threshold, then choose the
   clearest frame near each scene's subtitle timing (or use the panorama).

Steps 7-10 (audio focus, faces, cropping, text placement) are stubbed in their
own modules and are intentionally *not* invoked yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import audio, faces
from .errors import Anime2MangaError
from .ffmpeg_utils import require_tools
from .frames import SceneSampler, save_frame, save_image, select_frame
from .metadata import probe_media, resolve_clip_window
from .models import (
    ClipWindow,
    MediaInfo,
    PanResult,
    PipelineResult,
    Scene,
    SubtitleLine,
)
from .panorama import PanConfig, detect_pan, pan_continues, stitch
from .scene_detect import (
    DEFAULT_SCDET_THRESHOLD,
    DEFAULT_SELECT_THRESHOLD,
    build_scenes,
    detect_scene_times,
    subdivide_scene,
    validate_threshold,
)
from .subtitles import (
    extract_subtitles,
    load_subtitles,
    select_subtitle_track,
    subtitles_in_range,
)
from .timeline import reindex, validate_scenes


@dataclass
class PipelineConfig:
    input_path: Path
    output_dir: Path
    subtitle_language: str = "eng"
    subtitle_track: int | None = None
    translate_to: str | None = None
    scene_method: str = "select"
    scene_threshold: float | None = None
    scene_min_len: float = 0.5
    start_at: str | None = None
    end_at: str | None = None
    max_subtitles_per_scene: int = 6
    subdivide_factor: float = 0.5
    selection_window: float = 0.75
    analysis_width: int = 960
    analysis_fps: float = 4.0
    detect_pans: bool = True
    pan: PanConfig = field(default_factory=PanConfig)
    keep_analysis: bool = False
    verbose: bool = True


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.work_dir = self.output_dir / "work"
        self.scenes: list[Scene] = []
        self.subtitle_lines: list[SubtitleLine] = []
        self.media: MediaInfo | None = None
        self.clip: ClipWindow | None = None
        self.pan_results: dict[int, PanResult] = {}

    # -- logging -----------------------------------------------------------
    def log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[anime2manga] {message}", flush=True)

    def debug(self, message: str) -> None:
        if self.config.verbose:
            print(f"[anime2manga:debug] {message}", flush=True)

    # -- steps -------------------------------------------------------------
    def run(self) -> PipelineResult:
        require_tools()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.media = probe_media(self.config.input_path)
        self.clip = resolve_clip_window(
            self.media, start_at=self.config.start_at, end_at=self.config.end_at
        )
        self._report_metadata()

        track, lines = self._step2_subtitles()
        self._step3_scenes()
        if self.config.detect_pans and self.clip is not None:
            self._step4_panoramas()
        self._step5_validate()
        self._step6_frames()

        assert self.media is not None and self.clip is not None
        reindex(self.scenes)
        for scene in self.scenes:
            scene.subtitles = subtitles_in_range(lines, scene.range)

        return PipelineResult(
            media=self.media,
            clip=self.clip,
            subtitle_track=track,
            subtitle_lines=lines,
            scenes=self.scenes,
            output_dir=self.output_dir,
        )

    def _report_metadata(self) -> None:
        assert self.media is not None and self.clip is not None
        media = self.media
        self.log(f"input: {media.path}")
        self.log(
            f"video: {media.width}x{media.height} @ {media.fps:.3f} fps, "
            f"duration {media.duration:.3f}s"
        )
        for track in media.subtitle_tracks:
            kind = "bitmap/OCR-unsupported" if track.is_bitmap else "text"
            self.log(f"subtitle track {track.label} [{track.codec}, {kind}]")
        if media.chapters:
            self.log(f"chapters: {len(media.chapters)} found")
            for chapter in media.chapters:
                self.debug(
                    f"chapter {chapter.index}: {chapter.start:.2f}-{chapter.end:.2f}s "
                    f"{chapter.title!r}"
                )
        else:
            self.log("chapters: none found (intro/credits cannot be auto-skipped)")
        if media.intro or media.credits:
            intro = f"{media.intro.start:.2f}-{media.intro.end:.2f}s" if media.intro else "n/a"
            creds = (
                f"{media.credits.start:.2f}-{media.credits.end:.2f}s"
                if media.credits
                else "n/a"
            )
            self.log(f"intro chapter: {intro}; credits chapter: {creds}")
        self.log(
            f"clip window: {self.clip.start:.3f}s..{self.clip.end:.3f}s "
            f"({self.clip.duration:.3f}s, source: {self.clip.source})"
        )

    def _step2_subtitles(self):
        assert self.media is not None and self.clip is not None
        track = select_subtitle_track(
            self.media,
            language=self.config.subtitle_language,
            track_id=self.config.subtitle_track,
        )
        if self.config.translate_to:
            from .translation import require_translation

            require_translation(source=track.language, target=self.config.translate_to)

        subtitle_dir = self.output_dir / "subtitles"
        path = extract_subtitles(self.media, track, subtitle_dir)
        all_lines = load_subtitles(path, track)
        self.log(f"subtitle track: {track.label}")
        self.log(f"subtitle lines extracted (raw): {len(all_lines)}")
        self.subtitle_lines = self._filter_subtitles(all_lines, self.clip)
        dropped = len(all_lines) - len(self.subtitle_lines)
        self.log(
            f"subtitle lines within clip: {len(self.subtitle_lines)} "
            f"({dropped} skipped outside clip/intro/credits)"
        )
        return track, self.subtitle_lines

    @staticmethod
    def _filter_subtitles(
        lines: list[SubtitleLine], clip: ClipWindow
    ) -> list[SubtitleLine]:
        kept: list[SubtitleLine] = []
        for line in lines:
            midpoint = line.midpoint
            if not (clip.start <= midpoint < clip.end):
                continue
            if any(window.contains(midpoint) for window in clip.excluded):
                continue
            kept.append(line)
        return kept

    def _resolve_threshold(self) -> float:
        if self.config.scene_threshold is not None:
            return self.config.scene_threshold
        if self.config.scene_method == "scdet":
            return DEFAULT_SCDET_THRESHOLD
        return DEFAULT_SELECT_THRESHOLD

    def _step3_scenes(self) -> None:
        assert self.media is not None and self.clip is not None
        threshold = self._resolve_threshold()
        validate_threshold(self.config.scene_method, threshold)
        self.log(
            f"scene detection: method={self.config.scene_method} threshold={threshold}"
        )
        times = detect_scene_times(
            self.media,
            method=self.config.scene_method,
            threshold=threshold,
            start=self.clip.start,
            end=self.clip.end,
        )
        self.scenes = build_scenes(
            times, self.clip, self.media.fps, min_scene_len=self.config.scene_min_len
        )
        self.log(f"scenes identified: {len(self.scenes)} (raw cuts: {len(times)})")

    def _step4_panoramas(self) -> None:
        assert self.media is not None
        cfg = self.config.pan
        sampler = SceneSampler(
            self.media, self.work_dir, analysis_width=self.config.analysis_width
        )
        index = 0
        while index < len(self.scenes):
            scene = self.scenes[index]
            if scene.duration < max(0.5, 2.0 / cfg.sample_fps):
                index += 1
                continue
            tag = f"scene_{int(scene.start * 1000):08d}_pan"
            times, frames = sampler.sample(
                scene.start, scene.end, cfg.sample_fps, tag
            )
            result = detect_pan(scene.index, times, frames, config=cfg)
            if result.detected:
                self._maybe_extend_for_pan(scene, index, sampler, result)
                if result.detected:
                    self._store_panorama(scene, result)
            if result.detected:
                self.pan_results[scene.index] = result
            else:
                self.debug(
                    f"scene {scene.index}: shift={result.cumulative_shift[0]:.1f},"
                    f"{result.cumulative_shift[1]:.1f} consistency={result.consistency:.2f}"
                    f" response={result.mean_response:.2f} -> no pan"
                )
            index += 1
        self.log(
            f"panoramic scenes: {sum(1 for s in self.scenes if s.is_panoramic)}"
        )

    def _maybe_extend_for_pan(
        self,
        scene: Scene,
        index: int,
        sampler: SceneSampler,
        result: PanResult,
    ) -> None:
        """Peek into the next scene; if the pan continues, extend and re-detect."""
        cfg = self.config.pan
        if index + 1 >= len(self.scenes):
            return
        following = self.scenes[index + 1]
        peek_end = min(scene.end + cfg.peek_seconds, following.end)
        if peek_end - scene.end < 1.0 / cfg.sample_fps:
            return
        peek_times, peek_frames = sampler.sample(
            scene.end, peek_end, cfg.sample_fps, f"scene_{int(scene.start*1000):08d}_peek"
        )
        if not pan_continues(peek_times, peek_frames, result.direction or "", config=cfg):
            return
        applied = self._extend(scene, peek_end - scene.end)
        if applied <= 0:
            return
        self.debug(
            f"scene {scene.index}: pan continues into next scene, extended by "
            f"{applied:.2f}s"
        )
        # Re-detect over the extended range so the panorama spans the full pan.
        times, frames = sampler.sample(
            scene.start, scene.end, cfg.sample_fps, f"scene_{int(scene.start*1000):08d}_pan"
        )
        redetected = detect_pan(scene.index, times, frames, config=cfg)
        result.detected = redetected.detected
        result.direction = redetected.direction
        result.cumulative_shift = redetected.cumulative_shift
        result.consistency = redetected.consistency
        result.mean_response = redetected.mean_response
        result.canvas_width = redetected.canvas_width
        result.canvas_height = redetected.canvas_height
        result.offsets = redetected.offsets
        result.sample_times = redetected.sample_times
        result.frames = redetected.frames

    def _extend(self, scene: Scene, extension: float) -> float:
        from .timeline import extend_scene_for_pan

        return extend_scene_for_pan(self.scenes, scene, extension)

    def _store_panorama(self, scene: Scene, result: PanResult) -> None:
        image = stitch(result)
        path = self.output_dir / "panoramas" / f"scene_{int(scene.start*1000):08d}.jpg"
        save_image(image, path)
        scene.is_panoramic = True
        scene.pan_direction = result.direction
        scene.panorama_path = path
        scene.panorama_size = (image.shape[1], image.shape[0])
        scene.pan_shift = result.cumulative_shift
        scene.notes.append(
            f"pan {result.direction}, canvas {image.shape[1]}x{image.shape[0]}"
        )
        self.debug(
            f"scene {scene.index}: PAN {result.direction} shift="
            f"({result.cumulative_shift[0]:.1f},{result.cumulative_shift[1]:.1f}) "
            f"consistency={result.consistency:.2f} response={result.mean_response:.2f} "
            f"panorama={image.shape[1]}x{image.shape[0]}"
        )

    def _step5_validate(self) -> None:
        assert self.clip is not None
        problems = validate_scenes(self.scenes, self.clip)
        for problem in problems:
            self.log(f"timeline WARNING: {problem}")
        if not problems:
            self.debug("timeline OK: scenes tile the clip with no gaps/overlaps")
        self._subdivide_overloaded()

    def _subdivide_overloaded(self) -> None:
        assert self.media is not None
        threshold = self._resolve_threshold() * self.config.subdivide_factor
        rebuilt: list[Scene] = []
        split_count = 0
        for scene in self.scenes:
            subs = subtitles_in_range(self.subtitle_lines, scene.range)
            if (
                scene.is_panoramic
                or len(subs) <= self.config.max_subtitles_per_scene
                or scene.duration < 2.0
            ):
                rebuilt.append(scene)
                continue
            pieces = subdivide_scene(
                scene,
                self.media,
                method=self.config.scene_method,
                threshold=threshold,
                min_scene_len=self.config.scene_min_len,
            )
            if len(pieces) <= 1:
                rebuilt.append(scene)
                continue
            split_count += 1
            rebuilt.extend(pieces)
            self.debug(
                f"scene {scene.index}: {len(subs)} subtitle lines > "
                f"{self.config.max_subtitles_per_scene}, split into {len(pieces)} scenes "
                f"at threshold {threshold}"
            )
        self.scenes = rebuilt
        reindex(self.scenes)
        if split_count:
            self.log(f"overloaded scenes split: {split_count}")

    def _step6_frames(self) -> None:
        assert self.media is not None
        sampler = SceneSampler(
            self.media, self.work_dir, analysis_width=self.config.analysis_width
        )
        frames_dir = self.output_dir / "frames"
        for scene in self.scenes:
            scene.subtitles = subtitles_in_range(self.subtitle_lines, scene.range)
            if scene.is_panoramic:
                scene.frame_path = scene.panorama_path
                scene.frame_time = scene.range.midpoint
                scene.frame_size = scene.panorama_size
                continue
            tag = f"scene_{scene.index:04d}_select"
            times, frames = sampler.sample(
                scene.start, scene.end, self.config.analysis_fps, tag
            )
            candidate = select_frame(
                scene,
                scene.subtitles,
                times,
                frames,
                window=self.config.selection_window,
            )
            if candidate is None:
                scene.notes.append("no frame candidate found")
                continue
            out_path = frames_dir / f"scene_{scene.index:04d}.jpg"
            save_frame(self.media, candidate.time, out_path)
            scene.frame_path = out_path
            scene.frame_time = candidate.time
            scene.frame_size = (self.media.width, self.media.height)
            scene.blur_score = candidate.sharpness
            self.debug(
                f"scene {scene.index}: frame @ {candidate.time:.3f}s "
                f"sharpness={candidate.sharpness:.1f} subtitles={len(scene.subtitles)}"
            )
        self._maybe_cleanup()

    def _maybe_cleanup(self) -> None:
        if self.config.keep_analysis:
            return
        analysis_dir = self.work_dir / "analysis"
        if analysis_dir.exists():
            import shutil

            shutil.rmtree(analysis_dir, ignore_errors=True)

    # -- future stages (not wired in) -------------------------------------
    def stub_audio_focus(self) -> None:
        """Step 7 hook: fill ``Scene.audio_focus`` (see :mod:`anime2manga.audio`)."""
        assert self.media is not None
        for scene in self.scenes:
            scene.audio_focus = audio.detect_audio_focus(self.media, scene)

    def stub_faces(self) -> None:
        """Step 8 hook: fill ``Scene.faces`` (see :mod:`anime2manga.faces`)."""
        for scene in self.scenes:
            if scene.frame_path is not None:
                scene.faces = faces.detect_faces(scene.frame_path)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Convenience wrapper around :class:`Pipeline`."""
    if not config.input_path.exists():
        raise Anime2MangaError(f"Input file not found: {config.input_path}")
    return Pipeline(config).run()
