"""Rendering and reporting for the panel-layout trials.

Two jobs:

* :class:`PanelRenderer` turns a :class:`~anime2manga.layout.PanelPlan` into an
  actual image: carve first (protecting the detected subject boxes), then crop,
  reading the source frame either from a pipeline ``frames/`` folder or by
  re-extracting a clean frame from the video.
* :func:`render_set_html` / :func:`render_index_html` write the trial pages: each
  page is a table of three rows, each row holds up to two panels, and every panel
  is captioned with the rule ids and the rule set it came from.

Nothing here plans layout; it only consumes plans from
:mod:`anime2manga.layout`.
"""

from __future__ import annotations

import html
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .ffmpeg_utils import extract_frame
from .layout import (
    RULE_NOTES,
    UNIT_ASPECT,
    LayoutConfig,
    Page,
    PanelPlan,
    Row,
    head_cut,
    subject_center_error,
)
from .models import DetectionBox
from .seam_carving import SeamCarvingConfig, carve_width


def _to_bgr_read(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        # Composite a BGRA panorama over white so holes read as paper, not black.
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        colour = image[:, :, :3].astype(np.float32)
        out = colour * alpha + 255.0 * (1.0 - alpha)
        return out.astype(np.uint8)
    return image


def _scale_boxes(boxes: tuple[DetectionBox, ...], scale: float) -> list[DetectionBox]:
    return [
        DetectionBox(
            x=round(b.x * scale),
            y=round(b.y * scale),
            width=round(b.width * scale),
            height=round(b.height * scale),
            confidence=b.confidence,
        )
        for b in boxes
    ]


@dataclass
class PanelRenderer:
    """Renders panel plans to images, caching frames and carve results."""

    source: Path | None = None
    frames_dir: Path | None = None
    work_dir: Path = Path("output/layout/_frames")
    working_width: int = 768
    jpeg_quality: int = 88

    def __post_init__(self) -> None:
        self._frames: dict[int, np.ndarray] = {}
        self._carves: dict[tuple[int, int], np.ndarray] = {}

    def frame(self, meta) -> np.ndarray:
        index = meta.index
        cached = self._frames.get(index)
        if cached is not None:
            return cached
        image: np.ndarray | None = None
        if meta.is_panorama:
            for candidate in (meta.panorama_path, meta.frame_path):
                if candidate and Path(candidate).exists():
                    image = _to_bgr_read(Path(candidate))
                    if image is not None:
                        break
        elif self.source is not None and meta.frame_time is not None:
            cache_path = self.work_dir / f"scene_{index:04d}.jpg"
            if not cache_path.exists():
                extract_frame(self.source, meta.frame_time, cache_path, scale_width=self.working_width)
            image = _to_bgr_read(cache_path)
        if image is None:
            path = None
            if self.frames_dir is not None:
                candidate = self.frames_dir / f"scene_{index:04d}.jpg"
                path = candidate if candidate.exists() else None
            if path is None and meta.frame_path:
                path = Path(meta.frame_path)
            if path is not None and path.exists():
                image = _to_bgr_read(path)
        if image is None:
            raise FileNotFoundError(f"no image for scene {index} (frame_path={meta.frame_path})")
        if not meta.is_panorama and image.shape[1] > self.working_width:
            scale = self.working_width / image.shape[1]
            image = cv2.resize(
                image,
                (self.working_width, max(1, round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        self._frames[index] = image
        return image

    def carve(self, meta, image: np.ndarray, target_width: int) -> np.ndarray:
        key = (meta.index, target_width)
        cached = self._carves.get(key)
        if cached is not None:
            return cached
        scale = image.shape[1] / meta.width
        boxes = _scale_boxes(meta.boxes, scale)
        result = carve_width(
            image,
            target_width,
            boxes=boxes,
            config=SeamCarvingConfig(protect_subjects=True),
        )
        self._carves[key] = result.image
        return result.image

    def panel(self, meta, plan: PanelPlan) -> np.ndarray:
        image = self.frame(meta)
        if not meta.is_panorama and plan.carve_frac > 1e-9:
            target = max(1, round(image.shape[1] * (1.0 - plan.carve_frac)))
            image = self.carve(meta, image, target)
        if not meta.is_panorama and plan.crop_frac > 1e-9:
            width = image.shape[1]
            crop_w = max(1, round(width * (1.0 - plan.crop_frac)))
            x0 = max(0, min(round(plan.crop_x_frac * width), width - crop_w))
            image = image[:, x0 : x0 + crop_w]
        return image

    def write_panel(self, meta, plan: PanelPlan, out_path: Path) -> tuple[int, int]:
        image = self.panel(meta, plan)
        max_width = 1600
        if image.shape[1] > max_width:
            scale = max_width / image.shape[1]
            image = cv2.resize(
                image,
                (max_width, max(1, round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        return int(image.shape[1]), int(image.shape[0])


# --- serialisation -----------------------------------------------------------


def plan_to_dict(plan: PanelPlan) -> dict[str, Any]:
    return asdict(plan)


def rows_to_dict(rows: list[Row]) -> list[dict[str, Any]]:
    return [
        {
            "count": len(row.panels),
            "count_id": row.count_id,
            "count_reason": row.count_reason,
            "row_span": row.row_span,
            "units": round(row.units, 4),
            "panels": [plan_to_dict(p) for p in row.panels],
        }
        for row in rows
    ]


# --- evaluation --------------------------------------------------------------


def evaluate_set(frames: list, rows: list[Row], layout: LayoutConfig) -> dict[str, Any]:
    """Aggregate the per-set metrics used to rank rule sets."""
    by_index = {f.index: f for f in frames}
    panels = [p for row in rows for p in row.panels]
    paired = [row for row in rows if len(row.panels) == 2]
    reasons = Counter(row.count_id for row in rows if len(row.panels) == 1)
    total_span = sum(max(1, row.row_span) for row in rows)
    pages = max(1, (total_span + 2) // 3)

    crops = [p.crop_frac for p in panels]
    carves = [p.carve_frac for p in panels]
    cuts = 0
    centre_errors: list[float] = []
    for plan in panels:
        frame = by_index.get(plan.scene_index)
        if frame is None:
            continue
        if head_cut(frame, plan):
            cuts += 1
        error = subject_center_error(frame, plan, layout)
        if error is not None:
            centre_errors.append(error)

    used_units = sum(row.units for row in rows)
    capacity = layout.row_width_units * total_span
    return {
        "panels": len(panels),
        "rows": len(rows),
        "pages": pages,
        "panels_per_page": round(len(panels) / pages, 2),
        "single_rows": len(rows) - len(paired),
        "single_reasons": dict(reasons),
        "mean_crop": round(float(np.mean(crops)) if crops else 0.0, 4),
        "mean_carve": round(float(np.mean(carves)) if carves else 0.0, 4),
        "head_cuts": cuts,
        "mean_center_error": round(float(np.mean(centre_errors)) if centre_errors else 0.0, 4),
        "whitespace": round(max(0.0, 1.0 - used_units / capacity) if capacity else 0.0, 4),
    }


# --- HTML --------------------------------------------------------------------

_CSS = """
body { font-family: system-ui, sans-serif; margin: 20px; background: #f6f6f4; color: #222; }
h1 { margin: 0 0 6px; }
h2 { margin: 26px 0 4px; }
.sub { color: #666; font-size: 13px; }
.page { background: #fff; border: 2px solid #333; border-radius: 4px; padding: 8px;
        margin: 16px 0; max-width: 1000px; }
.page-no { font-size: 12px; color: #888; margin-bottom: 4px; }
table.grid { border-collapse: collapse; width: 100%; table-layout: fixed; }
table.grid td.rowcell { border: 1px dashed #ccc; padding: 4px; vertical-align: middle; }
.row { display: flex; justify-content: center; align-items: flex-end; gap: 8px; }
.row.center { justify-content: center; }
.panel { margin: 0; }
.panel img { display: block; width: 100%; height: auto; border: 1px solid #888; }
.cap { font-size: 11px; line-height: 1.25; margin-top: 3px; text-align: center; color: #333; }
.cap .rules { color: #0a6; }
.cap .action { color: #b36b00; font-weight: 600; }
.cap ul.rulelist { margin: 3px 0 0; padding-left: 14px; text-align: left; color: #444; }
.cap ul.rulelist li { margin: 0; }
.skip { color: #bbb; font-size: 12px; display: flex; align-items: center; justify-content: center; }
.count { font-size: 11px; color: #666; margin-bottom: 2px; }
table.metrics { border-collapse: collapse; font-size: 13px; margin: 8px 0 24px; }
table.metrics th, table.metrics td { border: 1px solid #ddd; padding: 3px 8px; text-align: right; }
table.metrics th:first-child, table.metrics td:first-child { text-align: left; }
table.metrics th { background: #efefef; }
"""


def _caption(plan: PanelPlan, set_name: str) -> str:
    action = plan.action if plan.action else "no resize"
    items = []
    for rule_id in (*plan.rule_ids, *plan.method_ids):
        note = RULE_NOTES.get(rule_id)
        label = f"<b>{html.escape(rule_id)}</b>"
        items.append(f"<li>{label} {html.escape(note)}</li>" if note else f"<li>{label}</li>")
    rule_list = f'<ul class="rulelist">{"".join(items)}</ul>' if items else ""
    return (
        '<figcaption class="cap">'
        f'<b>{html.escape(set_name)}</b> &middot; scene {plan.scene_index}'
        f'<div class="action">{html.escape(action)}</div>'
        f"{rule_list}"
        "</figcaption>"
    )


def _panel_figure(
    plan: PanelPlan, set_name: str, url: str, unit_px: float, row_span: int = 1
) -> str:
    width = max(1.0, plan.output_units * unit_px * max(1, row_span))
    body = (
        f'<img loading="lazy" src="{html.escape(url)}" style="width:100%">' + _caption(plan, set_name)
    )
    return f'<figure class="panel" style="flex:0 0 {width:.1f}px; max-width:{width:.1f}px">{body}</figure>'


def _row_html(
    row: Row,
    set_name: str,
    urls: dict[int, str],
    unit_px: float,
) -> str:
    span_label = " &middot; spans 2 rows" if row.row_span > 1 else ""
    count = (
        f'<div class="count">row: {len(row.panels)} frame(s) &middot; '
        f"{html.escape(row.count_id)}{span_label} &mdash; {html.escape(row.count_reason)}</div>"
    )
    if len(row.panels) == 1:
        plan = row.panels[0]
        figure = _panel_figure(
            plan, set_name, urls[plan.scene_index], unit_px, row.row_span
        )
        return '<div class="cellrow">' + count + f'<div class="row center">{figure}</div></div>'
    figures = "".join(
        _panel_figure(p, set_name, urls[p.scene_index], unit_px, row.row_span)
        for p in row.panels
    )
    return '<div class="cellrow">' + count + f'<div class="row">{figures}</div></div>'


def render_set_html(
    set_name: str,
    display_name: str,
    count_name: str,
    place_name: str,
    pages: list[Page],
    *,
    metrics: dict[str, Any],
    panel_urls: dict[int, str],
    layout: LayoutConfig,
    unit_px: float = 320.0,
) -> str:
    """A trial page: each page is a table of three rows, two panels per row."""
    blocks: list[str] = []
    for page in pages:
        rows_html = []
        for row in page.rows:
            rows_html.append(
                '<tr><td class="rowcell" colspan="2">'
                + _row_html(row, display_name, panel_urls, unit_px)
                + "</td></tr>"
            )
        blocks.append(
            f'<div class="page"><div class="page-no">Page {page.index + 1}</div>'
            f'<table class="grid">{"".join(rows_html)}</table></div>'
        )
    reasons = ", ".join(f"{k}={v}" for k, v in sorted(metrics["single_reasons"].items())) or "none"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Layout trial {html.escape(set_name)}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"<h1>{html.escape(set_name)} &middot; {html.escape(display_name)}</h1>"
        f"<p class='sub'>count <b>{html.escape(count_name)}</b> &times; place "
        f"<b>{html.escape(place_name)}</b> &middot; row budget {layout.row_width_units:g} "
        f"units &middot; keep_floor {layout.keep_floor:g} &middot; max_keep {layout.max_keep:g} "
        f"&middot; pad {layout.pad_frac:g}</p>"
        f"<p class='sub'>{metrics['panels']} panels over {metrics['rows']} rows "
        f"({metrics['panels_per_page']}/page, {metrics['single_rows']} single rows) &middot; "
        f"mean crop {metrics['mean_crop'] * 100:.1f}% &middot; mean carve "
        f"{metrics['mean_carve'] * 100:.1f}% &middot; head cuts {metrics['head_cuts']} &middot; "
        f"centre error {metrics['mean_center_error']:.3f} &middot; single reasons: {html.escape(reasons)}</p>"
        + "".join(blocks)
        + "</body></html>"
    )


def _clean_css(page_width: int, gutter: int) -> str:
    css = """
    body { background: #e8e8e6; margin: 0; padding: 16px;
           font-family: system-ui, sans-serif; }
    .layout-head { width: __PAGEW__px; margin: 0 auto 6px; color: #333; }
    .layout-head h1 { font-size: 18px; margin: 0 0 2px; }
    .layout-head p { font-size: 13px; margin: 2px 0; color: #555; }
    .layout-head ul { font-size: 13px; color: #555; margin: 6px 0 0 0; padding-left: 18px; }
    .cleanpage { width: __PAGEW__px; margin: 14px auto; background: #fff;
                 border: 1px solid #b9b9b9; box-shadow: 0 1px 3px rgba(0,0,0,.15);
                 padding: __GUTTER__px; box-sizing: border-box; line-height: 0; }
    .cleanrow { display: flex; justify-content: center; align-items: flex-start;
                gap: __GUTTER__px; margin: 0; padding: 0; }
    .cleanrow + .cleanrow { margin-top: __GUTTER__px; }
    .cleanrow img { display: block; }
    .cell { position: relative; line-height: 0; }
    .cell > img { width: 100%; height: 100%; }
    .cell > img.bubble { position: absolute; left: 0; top: 0; pointer-events: none; }
    """
    return css.replace("__PAGEW__", str(page_width)).replace("__GUTTER__", str(gutter))


def _clean_pages_html(
    pages: list[Page],
    panel_info: dict[int, tuple[str, float]],
    page_width: int,
    gutter: int,
    stamp: str,
    bubble_urls: dict[int, str] | None = None,
) -> str:
    """The rules-free page bodies shared by the trial and pipeline previews.

    ``bubble_urls`` maps a scene index to its transparent overlay PNG; when
    present the overlay is absolutely positioned over the panel so the bubbles
    sit on the artwork without being baked into it.
    """
    query = f"?v={html.escape(stamp)}" if stamp else ""
    bubbles = bubble_urls or {}
    full_row_aspect = LayoutConfig().row_width_units * UNIT_ASPECT
    # Two-panel full row: page width minus both margins minus the single gutter.
    base_height = (page_width - 3 * gutter) / full_row_aspect
    blocks: list[str] = []
    for page in pages:
        rows_html: list[str] = []
        for row in page.rows:
            panels = [p.scene_index for p in row.panels if p.scene_index in panel_info]
            if not panels:
                continue
            total_aspect = sum(panel_info[i][1] for i in panels)
            available = page_width - 2 * gutter - (len(panels) - 1) * gutter
            target = base_height * max(1, row.row_span)
            height = target if total_aspect <= 0 else min(target, available / total_aspect)
            cells = []
            for scene_index in panels:
                url, aspect = panel_info[scene_index]
                width = max(1, round(height * aspect))
                overlay = ""
                bubble = bubbles.get(scene_index)
                if bubble:
                    overlay = (
                        f'<img class="bubble" src="{html.escape(bubble)}{query}" '
                        f'style="height:{height:.1f}px;width:{width}px">'
                    )
                cells.append(
                    f'<div class="cell" style="height:{height:.1f}px;width:{width}px">'
                    f'<img src="{html.escape(url)}{query}">'
                    f"{overlay}</div>"
                )
            rows_html.append(f'<div class="cleanrow">{"".join(cells)}</div>')
        blocks.append(f'<div class="cleanpage">{"".join(rows_html)}</div>')
    return "".join(blocks)


def render_clean_set_html(
    set_name: str,
    pages: list[Page],
    panel_info: dict[int, tuple[str, float]],
    *,
    page_width: int = 1000,
    gutter: int = 8,
    stamp: str = "",
    bubble_urls: dict[int, str] | None = None,
) -> str:
    """A rules-free preview: pages of panels only, no captions or row labels.

    Every panel in a row shares one height, and rows are stacked with a small
    white gutter (comic-style) rather than a visible table border.  Panels wider
    than the page (a very wide panorama) shrink the whole row to fit.
    """
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Layout preview {html.escape(set_name)}</title>"
        f"<style>{_clean_css(page_width, gutter)}</style></head><body>"
        + _clean_pages_html(pages, panel_info, page_width, gutter, stamp, bubble_urls)
        + "</body></html>"
    )


def render_layout_index_html(
    set_name: str,
    description: str,
    options: list[tuple[str, str]],
    pages: list[Page],
    panel_info: dict[int, tuple[str, float]],
    *,
    page_width: int = 1000,
    gutter: int = 8,
    stamp: str = "",
    bubble_urls: dict[int, str] | None = None,
) -> str:
    """The pipeline's ``layout/index.html``: the clean pages plus set options.

    ``options`` are ``(name, description)`` pairs for the other available sets;
    they are listed so a user can see what else exists and how to switch.
    """
    option_items = "".join(
        f"<li><b>{html.escape(name)}</b> &mdash; {html.escape(text)}</li>"
        for name, text in options
        if name != set_name
    )
    header = (
        '<div class="layout-head">'
        "<h1>Anime2Manga &mdash; panel layout</h1>"
        f"<p>Generated with set <b>{html.escape(set_name)}</b> &mdash; "
        f"{html.escape(description)}</p>"
        "<details><summary>Other layout options (switch with "
        f"<code>--layout-set NAME</code>)</summary><ul>{option_items}</ul></details>"
        "</div>"
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Anime2Manga layout</title>"
        f"<style>{_clean_css(page_width, gutter)}</style></head><body>"
        + header
        + _clean_pages_html(pages, panel_info, page_width, gutter, stamp, bubble_urls)
        + "</body></html>"
    )


def render_index_html(summaries: list[dict[str, Any]], *, stamp: str = "") -> str:
    """The trial index: a metrics table plus links to each set's pages."""
    query = f"?v={html.escape(stamp)}" if stamp else ""
    rows = []
    for item in summaries:
        metrics = item["metrics"]
        reasons = ", ".join(f"{k}:{v}" for k, v in sorted(metrics["single_reasons"].items())) or "-"
        clean = item.get("clean")
        clean_cell = (
            f'<a href="{html.escape(clean)}{query}">preview</a>' if clean else "-"
        )
        rows.append(
            "<tr>"
            f'<td><a href="{html.escape(item["file"])}{query}">{html.escape(item["name"])}</a></td>'
            f"<td>{clean_cell}</td>"
            f'<td>{html.escape(item["count"])}</td>'
            f'<td>{html.escape(item["place"])}</td>'
            f"<td>{metrics['panels']}</td>"
            f"<td>{metrics['panels_per_page']}</td>"
            f"<td>{metrics['single_rows']}</td>"
            f"<td>{html.escape(reasons)}</td>"
            f"<td>{metrics['mean_crop'] * 100:.1f}%</td>"
            f"<td>{metrics['mean_carve'] * 100:.1f}%</td>"
            f"<td>{metrics['head_cuts']}</td>"
            f"<td>{metrics['mean_center_error']:.3f}</td>"
            f"<td>{metrics['whitespace'] * 100:.1f}%</td>"
            "</tr>"
        )
    header = (
        "<tr><th>Set</th><th>Clean</th><th>Count</th><th>Place</th><th>Panels</th>"
        "<th>Panels/page</th><th>Single rows</th><th>Single reasons</th><th>Mean crop</th>"
        "<th>Mean carve</th><th>Head cuts</th><th>Centre error</th><th>Whitespace</th></tr>"
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Layout rule-set trials</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>Layout rule-set trials</h1>"
        "<p class='sub'>Each set is a count policy &times; placement policy. Click a set name for "
        "its captioned 3&times;2 trial pages, or <b>preview</b> for the rules-free finished-look page.</p>"
        f"<table class='metrics'>{header}{''.join(rows)}</table>"
        "</body></html>"
    )


def write_plans_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rules_legend() -> str:
    parts = [f"<code>{key}</code> {html.escape(note)}" for key, note in RULE_NOTES.items()]
    return "<p class='sub'>" + " &middot; ".join(parts) + "</p>"


__all__ = [
    "PanelRenderer",
    "evaluate_set",
    "plan_to_dict",
    "render_clean_set_html",
    "render_index_html",
    "render_layout_index_html",
    "render_set_html",
    "rows_to_dict",
    "rules_legend",
    "write_plans_json",
]
