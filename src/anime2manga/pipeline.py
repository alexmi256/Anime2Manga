"""Pipeline orchestration for steps 1-8.

Order of operations
-------------------
1. Probe the container for duration/tracks/chapters and resolve the clip window
   (``--start-at`` / ``--end-at`` or intro/credits chapters).
2. Select, extract and parse the subtitle track; drop cues in excluded ranges.
3. Detect scenes with ffmpeg (``select`` or ``scdet``).
4. Detect pans per scene and, when a pan spills over a boundary, retime the
   affected scenes and stitch a panorama.  Diagonal pan canvases leave
   transparent holes; a pluggable inpainting method fills them and writes a
   JPEG beside the PNG.
5. Validate that scenes still tile the clip with no gaps/overlaps.
6. Re-detect text-overloaded scenes at a lower threshold, then choose the
   clearest frame near each scene's subtitle timing (or use the panorama).
7. Measure the left/right audio balance of each scene so text placement can
   favour the side the dialogue comes from.
8. Detect faces, heads and persons on each chosen frame, seam-carve a copy of
   every regular frame (reusing the face boxes, before any boxes are drawn),
   then optionally draw every category's bounding boxes onto the saved image.

Steps 9-10 (cropping, text placement) are stubbed in their own modules and are
intentionally *not* invoked yet.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from . import audio, detection, faces, heads, persons
from .audio import AudioFocusConfig
from .composition import analyze_composition
from .errors import Anime2MangaError
from .faces import FaceDetectionConfig
from .ffmpeg_utils import require_tools
from .frames import SceneSampler, save_frame, save_image, select_frame
from .heads import HeadDetectionConfig
from .inpaint import InpaintConfig, create_inpainter, fill_panorama
from .metadata import probe_media, resolve_clip_window
from .models import (
    ClipWindow,
    MediaInfo,
    PanResult,
    PipelineResult,
    Scene,
    SubtitleLine,
    TextPlacement,
)
from .panorama import PanConfig, detect_pan, pan_continues, stitch
from .persons import PersonDetectionConfig
from .retarget import RetargetConfig, retarget_image
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
    #: Scenes this short (seconds) trailing a panorama are folded into it.
    pan_merge_max_len: float = 1.0
    #: Content-aware infill for transparent panorama holes.
    inpaint: InpaintConfig = field(default_factory=InpaintConfig)
    #: Left/right audio focus for each scene (step 7).
    detect_audio: bool = True
    audio: AudioFocusConfig = field(default_factory=AudioFocusConfig)
    #: Detect faces, heads and persons on the chosen frames (step 8).
    detect_face: bool = False
    detect_head: bool = True
    detect_person: bool = True
    #: Master switch for drawing bounding boxes on the saved frames.
    draw_boxes: bool = True
    #: Per-category drawing toggles (all default to the category's ``detect_*``).
    draw_face_boxes: bool = False
    draw_head_boxes: bool = True
    draw_person_boxes: bool = True
    face: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    head: HeadDetectionConfig = field(default_factory=HeadDetectionConfig)
    person: PersonDetectionConfig = field(default_factory=PersonDetectionConfig)
    #: Produce a seam-carved version of every regular (non-panoramic) frame.
    #: The carver reuses step 8's face boxes and runs before the boxes are
    #: drawn, so the face model is never run twice and the carve sees clean
    #: pixels.  ``retarget.energy_ratio <= 0`` also disables it.
    seam_carve: bool = True
    #: Retarget metric settings.  Defaults are tuned for the report (768px
    #: working width, energy ratio 0.25, no snapshots/SSIM since only the
    #: composite ratio is needed).  The 768px downscale is ~13x faster than
    #: carving at source resolution; set ``working_width=None`` for
    #: native-resolution output.
    retarget: RetargetConfig = field(
        default_factory=lambda: RetargetConfig(
            working_width=768,
            energy_ratio=0.25,
            strip_overlays=False,
            sample_step=1.0,
            ssim_stride=0,
        )
    )
    #: Worker processes for seam carving (1 = sequential; the CLI defaults higher).
    seam_carve_jobs: int = 1
    #: JPEG quality for seam-carved frames.
    seam_carve_quality: int = 92
    #: Generate a panel-layout preview (``output/layout/index.html``) after step 8.
    #: Off for programmatic use; the CLI turns it on by default.
    layout: bool = False
    #: Which rule set to lay the page out with (see ``layout.SETS``).
    layout_set: str = "K-B"
    #: Downscale each frame to this width before cropping for the layout panels.
    layout_working_width: int = 768
    layout_page_width: int = 1000
    layout_gutter: int = 8
    layout_quality: int = 88
    #: Letter the panels with subtitle speech bubbles (step 10).  Bubbles are
    #: planned on the **finished** panel image, after crop/carve and once the
    #: panel's place in the layout is known (see ``speech_bubbles``).  Off for
    #: programmatic use; the CLI turns it on with the layout.
    speech_bubbles: bool = False
    #: Bake the bubble overlay into the panel JPEG as well as emitting the
    #: separate ``.svg``/``.png`` overlay assets.
    flatten_bubbles: bool = False
    keep_analysis: bool = False
    verbose: bool = True


def _carve_frame(
    task: tuple[int, str, list, str, RetargetConfig, int],
) -> dict | None:
    """Seam-carve one frame in a worker process (module-level for pickling)."""
    index, frame_path, box_list, out_path, config, quality = task
    image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    # The pipeline already detected every subject; hand the boxes straight in so
    # no model runs again (``protect_heads``/``protect_persons`` are irrelevant
    # here because ``boxes`` overrides detection).
    analysis = retarget_image(image, boxes=list(box_list), config=config)
    carved = analysis.recommended_image if analysis.recommended_image is not None else image
    cv2.imwrite(out_path, carved, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    ratio = analysis.recommended.ratio if analysis.recommended else 0.0
    return {
        "index": index,
        "path": out_path,
        "shrink": float(ratio),
        "size": (int(carved.shape[1]), int(carved.shape[0])),
    }


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
            self._inpaint_panoramas()
        self._step5_validate()
        self._step6_frames()
        self._step7_audio()
        self._step8_detections()

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
        panorama_dir = self.output_dir / "panoramas"
        if panorama_dir.exists():
            # Formats and scene indices change between runs, so drop stale files.
            for stale in panorama_dir.iterdir():
                if stale.is_file():
                    stale.unlink()
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
            if not result.detected:
                self.debug(
                    f"scene {scene.index}: shift={result.cumulative_shift[0]:.1f},"
                    f"{result.cumulative_shift[1]:.1f} consistency={result.consistency:.2f}"
                    f" response={result.mean_response:.2f} -> no pan"
                )
                index += 1
                continue

            # Only a pan touching the scene's end can spill into the next scene.
            if self._pan_reaches_scene_end(scene, result):
                self._maybe_extend_for_pan(scene, index, sampler, result)
            if not result.detected:
                index += 1
                continue

            pan_scene = self._isolate_pan(scene, result)
            if pan_scene is None:  # pragma: no cover - defensive
                index += 1
                continue
            self._store_panorama(pan_scene, result)
            self.pan_results[pan_scene.index] = result
            # The non-pan pieces around the segment cannot hide a stronger pan
            # (we picked the strongest segment), so skip past them.
            index = self.scenes.index(pan_scene) + 1
        self._merge_pan_slivers()
        self.log(
            f"panoramic scenes: {sum(1 for s in self.scenes if s.is_panoramic)}"
        )

    def _merge_pan_slivers(self) -> None:
        """Fold a very short scene that trails a panorama into that panorama.

        The scene right after a pan usually shows the tail of the same shot, so
        on its own it becomes a near-duplicate panel beside the panorama.
        Absorbing slivers keeps the report readable; the panoramic scene simply
        grows to cover them and the timeline stays gapless.
        """
        limit = self.config.pan_merge_max_len
        if limit <= 0:
            return
        merged: list[Scene] = []
        absorbed = 0
        for scene in self.scenes:
            previous = merged[-1] if merged else None
            if (
                previous is not None
                and previous.is_panoramic
                and not scene.is_panoramic
                and scene.duration <= limit
            ):
                previous.end = scene.end
                previous.notes.append(
                    f"absorbed trailing scene {scene.index} ({scene.duration:.3f}s)"
                )
                absorbed += 1
                continue
            merged.append(scene)
        # Absorbing a sliver can renumber panoramas that come after it, so carry
        # the scene -> PanResult association across reindexing by object identity.
        results_by_scene = {
            id(scene): self.pan_results[scene.index]
            for scene in self.scenes
            if scene.index in self.pan_results
        }
        self.scenes = merged
        reindex(self.scenes)
        self.pan_results = {}
        for scene in self.scenes:
            result = results_by_scene.get(id(scene))
            if result is not None:
                self.pan_results[scene.index] = result
                result.scene_index = scene.index
        if absorbed:
            self.log(
                f"short scenes merged into preceding panorama: {absorbed} "
                f"(<= {limit:.2f}s)"
            )

    def _pan_reaches_scene_end(self, scene: Scene, result: PanResult) -> bool:
        tolerance = 1.5 / self.config.pan.sample_fps
        return scene.end - result.end_time <= tolerance

    def _isolate_pan(self, scene: Scene, result: PanResult) -> Scene | None:
        """Split ``scene`` so only the panning span becomes panoramic.

        Scene detection can merge several shots; a panorama may therefore cover
        just part of a scene.  The surrounding spans are kept as their own
        (non-panoramic) scenes so their content still gets a representative
        frame.  Returns the scene that now owns the panorama.
        """
        start = min(max(result.start_time, scene.start), scene.end)
        end = min(max(result.end_time, scene.start), scene.end)
        if end - start <= 0:
            return None
        min_len = 0.25
        # ``end`` is the timestamp of the pan's last sampled frame.  End the pan
        # one sample later so that frame stays owned by the pan; otherwise the
        # following span would start on the very same frame and repeat the
        # panorama's edge as its own representative frame.
        step = 1.0 / self.config.pan.sample_fps if self.config.pan.sample_fps > 0 else 0.0
        resume = min(end + step, scene.end)
        pieces: list[Scene] = []
        # Absorb slivers into the pan scene so the timeline stays gapless.  This
        # can leave a non-panoramic piece shorter than ``scene_min_len`` (or a
        # pan scene shorter than it); that is deliberate, because dropping the
        # sliver would open a timeline gap.
        pan_start = scene.start if start - scene.start <= min_len else start
        pan_end = scene.end if scene.end - resume <= min_len else resume
        if pan_start != scene.start:
            pieces.append(Scene(index=0, start=scene.start, end=start, fps=scene.fps))
        pan_scene = Scene(index=scene.index, start=pan_start, end=pan_end, fps=scene.fps)
        pieces.append(pan_scene)
        if pan_end != scene.end:
            pieces.append(Scene(index=0, start=pan_end, end=scene.end, fps=scene.fps))
        if len(pieces) == 1:
            return scene
        position = self.scenes.index(scene)
        self.scenes[position : position + 1] = pieces
        reindex(self.scenes)
        result.scene_index = pan_scene.index
        self.debug(
            f"scene {scene.index}: pan covers {start:.3f}-{end:.3f}s, "
            f"split into {len(pieces)} scenes"
        )
        return pan_scene

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
        result.start_time = redetected.start_time
        result.end_time = redetected.end_time

    def _extend(self, scene: Scene, extension: float) -> float:
        from .timeline import extend_scene_for_pan

        return extend_scene_for_pan(self.scenes, scene, extension)

    def _store_panorama(self, scene: Scene, result: PanResult) -> None:
        image = stitch(result)
        path = self.output_dir / "panoramas" / f"scene_{int(scene.start*1000):08d}.png"
        save_image(image, path)
        scene.is_panoramic = True
        scene.pan_direction = result.direction
        scene.panorama_path = path
        scene.panorama_size = (image.shape[1], image.shape[0])
        scene.pan_shift = result.cumulative_shift
        scene.pan_start = result.start_time
        scene.pan_end = result.end_time
        scene.notes.append(
            f"pan {result.direction}, canvas {image.shape[1]}x{image.shape[0]}"
        )
        self.debug(
            f"scene {scene.index}: PAN {result.direction} shift="
            f"({result.cumulative_shift[0]:.1f},{result.cumulative_shift[1]:.1f}) "
            f"consistency={result.consistency:.2f} response={result.mean_response:.2f} "
            f"panorama={image.shape[1]}x{image.shape[0]} "
            f"span={result.start_time:.3f}-{result.end_time:.3f}s"
        )

    def _inpaint_panoramas(self) -> None:
        """Fill the transparent holes of each stitched panorama (step 4b).

        The BGRA panorama keeps the raw stitch (holes and all); here we read it
        back, hand its alpha channel to the configured :class:`InpaintMethod`
        and write the filled, flattened JPEG alongside the PNG.  The method is a
        config knob, so swapping in a different algorithm needs no changes here.
        """
        cfg = self.config.inpaint
        if not cfg.enabled:
            return
        panoramas = [
            scene
            for scene in self.scenes
            if scene.is_panoramic and scene.panorama_path is not None
        ]
        if not panoramas:
            return

        method = create_inpainter(cfg.method)
        filled = 0
        for scene in panoramas:
            assert scene.panorama_path is not None
            image = cv2.imread(str(scene.panorama_path), cv2.IMREAD_UNCHANGED)
            if image is None:  # pragma: no cover - defensive
                self.log(f"inpaint: could not read {scene.panorama_path}")
                continue
            try:
                outcome = fill_panorama(image, method, max_pixels=cfg.max_pixels)
            except Anime2MangaError as error:
                # One unreconstructable panorama must not abort the whole run.
                self.log(f"inpaint: scene {scene.index} skipped: {error}")
                continue
            if not outcome.filled:
                # A fully covered canvas (e.g. a purely horizontal pan) has no
                # holes, so there is nothing to fill and a JPEG would be a copy.
                continue
            out_path = scene.panorama_path.with_name(
                f"{scene.panorama_path.stem}_inpainted.jpg"
            )
            save_image(outcome.image, out_path, quality=cfg.quality)
            scene.panorama_inpainted_path = out_path
            scene.inpaint_method = outcome.method
            filled += 1
            self.debug(
                f"scene {scene.index}: inpainted {outcome.filled_pixels} px "
                f"({outcome.method}) -> {out_path.name}"
            )
        self.log(f"panoramas inpainted: {filled} (method: {method.name})")

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

    def _step8_detections(self) -> None:
        """Detect faces, heads and persons, seam-carve, then optionally draw.

        Detection runs first so the carver can reuse the boxes, and the boxes
        are painted last so the carve works from clean pixels and the models are
        never run twice.
        """
        cfg = self.config
        face_count = head_count = person_count = 0
        if cfg.detect_face or cfg.detect_head or cfg.detect_person:
            for scene in self.scenes:
                if scene.frame_path is None:
                    continue
                if cfg.detect_face:
                    scene.faces = faces.detect_faces(scene.frame_path, config=cfg.face)
                if cfg.detect_head:
                    scene.heads = heads.detect_heads(scene.frame_path, config=cfg.head)
                if cfg.detect_person:
                    scene.persons = persons.detect_persons(
                        scene.frame_path, config=cfg.person
                    )
                face_count += len(scene.faces)
                head_count += len(scene.heads)
                person_count += len(scene.persons)
                if scene.faces or scene.heads or scene.persons:
                    self.debug(
                        f"scene {scene.index}: {len(scene.faces)} face(s), "
                        f"{len(scene.heads)} head(s), {len(scene.persons)} person(s)"
                    )
                self._warn_oversized_detections(scene)
                if scene.frame_size is not None:
                    # Pure, box-only report diagnostics; never fed back into the
                    # carve or a crop (see ``composition``'s caveat).
                    scene.composition = analyze_composition(
                        scene.frame_size, scene.persons, scene.heads
                    )
            self.log(
                f"detections: faces={face_count} heads={head_count} persons={person_count}"
            )

        self._step8b_seam_carve()

        # Layout must see clean pixels, so run it before boxes are drawn on the
        # frames.  It is best-effort: a scene whose image is missing is skipped.
        if cfg.layout:
            try:
                self._step9_layout()
            except Exception as exc:  # pragma: no cover - defensive
                self.debug(f"layout step skipped: {exc}")

        if cfg.draw_boxes:
            for scene in self.scenes:
                if scene.frame_path is None:
                    continue
                groups = self._drawing_groups(scene)
                if any(boxes for _, boxes in groups):
                    detection.annotate_categories(scene.frame_path, groups)

    def _warn_oversized_detections(self, scene: Scene) -> None:
        """Flag detections that cover almost the whole frame (likely artefacts).

        The detectors can return a near full-frame box on an extreme close-up;
        the seam carver already neutralises such masks, so this is only a nudge
        to tune thresholds.  Nothing is dropped.
        """
        if scene.frame_size is None:
            return
        configs = {
            "face": self.config.face,
            "head": self.config.head,
            "person": self.config.person,
        }
        for category, boxes in self._box_groups(scene):
            limit = configs[category].max_box_area_fraction
            oversized = detection.oversized_boxes(
                boxes, scene.frame_size, max_area_fraction=limit
            )
            for box in oversized:
                share = box.area / (scene.frame_size[0] * scene.frame_size[1])
                self.debug(
                    f"scene {scene.index}: {category} box covers {share:.0%} of the "
                    f"frame (area {box.area}); detections this large are usually "
                    "artefacts - consider raising the score threshold"
                )

    def _box_groups(self, scene: Scene) -> list[tuple[str, list]]:
        """The ``(category, boxes)`` pairs to detect/draw/protect for ``scene``.

        A category is included only when its detector is enabled, so the
        seam-carving protection and the drawn boxes follow exactly what was
        detected (faces default off, heads and persons on).
        """
        cfg = self.config
        return [
            ("face", list(scene.faces) if cfg.detect_face else []),
            ("head", list(scene.heads) if cfg.detect_head else []),
            ("person", list(scene.persons) if cfg.detect_person else []),
        ]

    def _drawing_groups(self, scene: Scene) -> list[tuple[str, list]]:
        """The ``(category, boxes)`` pairs to actually paint for ``scene``."""
        cfg = self.config
        enabled = {
            "face": cfg.draw_face_boxes,
            "head": cfg.draw_head_boxes,
            "person": cfg.draw_person_boxes,
        }
        return [
            (category, boxes)
            for category, boxes in self._box_groups(scene)
            if enabled[category]
        ]

    def _protected_boxes(self, scene: Scene) -> list:
        """All detected boxes (every enabled category) for seam-carving.

        Faces, heads and persons all inherit the same seam-carving protection,
        so the carver keeps every detected subject intact.
        """
        return [box for _category, boxes in self._box_groups(scene) for box in boxes]

    def _step8b_seam_carve(self) -> None:
        """Write a seam-carved copy of each regular frame and record its shrink.

        Panoramic scenes keep their stitched canvas untouched; every other scene
        with a chosen frame is carved down from the frame's own detected boxes.
        The recommendation is the composite limit (``energy`` + ``detail``).
        The carve runs at the config's working width (768px by default, ~13x
        faster than native; pass ``working_width=None`` /
        ``--seam-carve-working-width 0`` for full-resolution output).
        """
        cfg = self.config
        if not cfg.seam_carve or cfg.retarget.energy_ratio <= 0:
            return
        seam_dir = self.output_dir / "seam_frames"
        tasks: list[tuple] = []
        for scene in self.scenes:
            if scene.is_panoramic or scene.frame_path is None:
                continue
            out_path = seam_dir / f"scene_{scene.index:04d}.jpg"
            tasks.append(
                (
                    scene.index,
                    str(scene.frame_path),
                    self._protected_boxes(scene),
                    str(out_path),
                    cfg.retarget,
                    cfg.seam_carve_quality,
                )
            )
        if not tasks:
            return
        seam_dir.mkdir(parents=True, exist_ok=True)
        for stale in seam_dir.iterdir():
            if stale.is_file():
                stale.unlink()

        jobs = max(1, cfg.seam_carve_jobs)
        if jobs > 1 and len(tasks) > 1:
            # Explicit ``spawn``: face detection (cv2.dnn) has already run in
            # this process, and a forked child inheriting that state can
            # deadlock.  ``spawn`` starts clean interpreters.  Both entry points
            # (CLI, scripts) run behind an ``if __name__ == "__main__"`` guard.
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
                results = list(pool.map(_carve_frame, tasks))
        else:
            results = [_carve_frame(task) for task in tasks]

        by_index = {scene.index: scene for scene in self.scenes}
        carved = 0
        for result in results:
            if result is None:
                continue
            scene = by_index.get(result["index"])
            if scene is None:
                continue
            scene.seam_carved_path = Path(result["path"])
            scene.seam_carve_shrink = result["shrink"]
            scene.seam_carve_size = result["size"]
            carved += 1
            self.debug(
                f"scene {scene.index}: seam carved to {result['shrink'] * 100:.0f}% "
                f"({result['size'][0]}x{result['size'][1]})"
            )
        self.log(f"seam carved frames: {carved}")

    def _step9_layout(self) -> None:
        """Render the panel-layout preview for the chosen rule set.

        Reads the chosen frames *before* bounding boxes are drawn, lays them out
        with the configured rule set, crops/carves each panel and writes
        ``output/layout/index.html`` (a rules-free, comic-style page stack) plus
        the panel images.  The page also lists the other available sets so a
        user can see the options.
        """
        from .layout import (
            SET_DESCRIPTIONS,
            SETS,
            LayoutConfig,
            meta_from_scene,
            paginate,
            plan_rows,
            set_policies,
        )
        from .layout_report import PanelRenderer, render_layout_index_html

        cfg = self.config
        wanted = [s for s in self.scenes if s.frame_path is not None and s.frame_size is not None]
        if not wanted:
            return
        frames = [meta_from_scene(scene) for scene in wanted]
        count_policy, place_policy = set_policies(cfg.layout_set)
        rows = plan_rows(frames, LayoutConfig(), count_policy, place_policy)
        pages = paginate(rows, 3)

        out_dir = self.output_dir / "layout"
        slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cfg.layout_set)
        panels_dir = out_dir / "panels" / slug
        # Drop panels from a previous run so a changed set cannot leave stale
        # images next to the new ones.
        if panels_dir.parent.exists():
            import shutil

            shutil.rmtree(panels_dir.parent, ignore_errors=True)
        renderer = PanelRenderer(
            frames_dir=None,
            source=None,
            work_dir=out_dir / "_frames",
            working_width=cfg.layout_working_width,
            jpeg_quality=cfg.layout_quality,
        )
        by_index = {frame.index: frame for frame in frames}
        scene_by_index = {scene.index: scene for scene in wanted}
        panel_info: dict[int, tuple[str, float]] = {}
        bubbles_by_scene: dict[int, str] = {}
        for row in rows:
            for plan in row.panels:
                meta = by_index.get(plan.scene_index)
                if meta is None:
                    continue
                path = panels_dir / f"scene_{plan.scene_index:04d}.jpg"
                try:
                    width, height = renderer.write_panel(meta, plan, path)
                except Exception as exc:
                    self.debug(f"layout: scene {plan.scene_index} skipped ({exc})")
                    continue
                panel_info[plan.scene_index] = (
                    f"panels/{slug}/{path.name}",
                    width / height if height else 1.0,
                )
                if cfg.speech_bubbles:
                    # Lettering is best-effort like the panel write: a bad font or
                    # a cairo/fontconfig failure must not abort the whole page.
                    try:
                        self._letter_panel(
                            scene_by_index.get(plan.scene_index), plan, path, (width, height),
                            panels_dir, bubbles_by_scene,
                        )
                    except Exception as exc:
                        self.debug(f"layout: bubbles for scene {plan.scene_index} skipped ({exc})")

        options = [(name, SET_DESCRIPTIONS.get(name, "")) for name in SETS]
        html = render_layout_index_html(
            cfg.layout_set,
            SET_DESCRIPTIONS.get(cfg.layout_set, ""),
            options,
            pages,
            panel_info,
            page_width=cfg.layout_page_width,
            gutter=cfg.layout_gutter,
            bubble_urls=bubbles_by_scene,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        # Drop stale single-layout-set files from a previous run so only the
        # current index.html remains (the experiment uses layout_trials/).
        for pattern in ("*.html", "*.json"):
            for stale in out_dir.glob(pattern):
                stale.unlink()
        (out_dir / "index.html").write_text(html, encoding="utf-8")
        self.log(f"layout: {len(panel_info)} panels ({cfg.layout_set}) -> {out_dir / 'index.html'}")

    def _letter_panel(
        self,
        scene: Scene | None,
        plan,
        panel_path: Path,
        panel_size: tuple[int, int],
        panels_dir: Path,
        bubble_urls: dict[int, str],
    ) -> None:
        """Plan and render a panel's speech bubbles (step 10).

        Runs **after** the panel image is finished and its size is known, which
        is the whole point of the ordering: the bubble geometry is expressed in
        final panel pixels, so neither seam carving nor cropping can move it.
        """
        from . import speech_bubbles

        if scene is None or not scene.subtitles:
            return
        width, height = panel_size
        source_size = scene.frame_size or (width, height)
        # Subject boxes in panel coordinates, so a bubble can avoid a face.
        boxes = speech_bubbles.map_boxes_to_panel(
            [*scene.heads, *scene.persons, *scene.faces],
            panel_size,
            source_size,
            plan.crop_frac,
            plan.crop_x_frac,
        )
        layout = speech_bubbles.plan_bubbles(
            (width, height),
            [line.text for line in scene.subtitles],
            boxes=boxes,
            audio_focus=scene.audio_focus,
        )
        written = speech_bubbles.write_panel_overlay(layout, panels_dir, panel_path.stem)
        if written is None:
            return
        svg_path, png_path = written
        if self.config.flatten_bubbles:
            # Bake the overlay into the panel; there is then no separate PNG to
            # stack in the page (and the stale file must not be referenced).
            speech_bubbles.flatten_panel(panel_path, layout)
            png_path.unlink(missing_ok=True)
        else:
            bubble_urls[scene.index] = f"panels/{panels_dir.name}/{png_path.name}"
        scene.text_placement = TextPlacement(
            side=scene.audio_focus,
            regions=tuple(
                (round(s.box[0]), round(s.box[1]), round(s.box[2]), round(s.box[3]))
                for s in layout.specs
            ),
            source=str(svg_path.relative_to(self.output_dir)),
            bubble_count=len(layout.specs),
            area_frac=round(layout.area_frac, 4),
            notes=layout.notes,
        )
        self.debug(
            f"scene {scene.index}: {len(layout.specs)} bubble(s), "
            f"{layout.area_frac * 100:.0f}% of panel"
        )

    def _step7_audio(self) -> None:
        """Measure each scene's left/right audio balance (see :mod:`anime2manga.audio`)."""
        assert self.media is not None
        if not self.config.detect_audio:
            return
        info = self.media.audio
        if info is None:
            self.log("audio direction: no audio stream, all scenes centered")
            return
        if info.is_mono:
            self.log(f"audio direction: mono source ({info.label}), all scenes centered")
            return

        counts = {"left": 0, "center": 0, "right": 0}
        for scene in self.scenes:
            focus = audio.detect_audio_focus(self.media, scene, config=self.config.audio)
            scene.audio_focus = focus.direction
            scene.audio_balance_db = focus.balance_db
            counts[focus.direction] = counts.get(focus.direction, 0) + 1
            if focus.balance_db is None:
                self.debug(
                    f"scene {scene.index}: audio {focus.direction} ({focus.reason})"
                )
            else:
                self.debug(
                    f"scene {scene.index}: audio {focus.direction} "
                    f"balance={focus.balance_db:+.1f} dB ({focus.reason})"
                )
        self.log(
            f"audio direction: left={counts['left']} center={counts['center']} "
            f"right={counts['right']}"
        )


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Convenience wrapper around :class:`Pipeline`."""
    if not config.input_path.exists():
        raise Anime2MangaError(f"Input file not found: {config.input_path}")
    return Pipeline(config).run()
