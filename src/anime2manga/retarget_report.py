"""Rendering and reporting for the seam-carving retarget experiment.

Turns :class:`~anime2manga.retarget.FrameAnalysis` results into the artefacts a
human can actually review: per-frame contact strips (original, each 10% shrink,
and the recommended stop), per-frame signal plots as standalone SVG, and a
summary in CSV/JSON/Markdown plus a browsable HTML report.  Nothing here talks
to the seam-carving engine directly, so it stays cheap and easy to test.
"""

from __future__ import annotations

import csv
import html
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .retarget import METHODS, FrameAnalysis, RetargetConfig

#: Line colour per signal in the SVG plots.
SIGNAL_COLORS: dict[str, str] = {
    "energy": "#e4572e",
    "forward": "#17bebb",
    "detail": "#8e44ad",
    "ssim": "#2c7fb8",
    "composite": "#111111",
}
#: Short label per signal.
SIGNAL_LABELS: dict[str, str] = {
    "energy": "energy",
    "forward": "forward",
    "detail": "detail+face",
    "ssim": "structural",
    "composite": "composite",
}
PLOT_SIGNALS: tuple[str, ...] = ("energy", "forward", "detail", "ssim")


def _percent(ratio: float) -> str:
    return f"{ratio * 100:.0f}%"


def _label_panel(image: np.ndarray, text: str, width: int) -> np.ndarray:
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    panel = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    bar = np.full((26, width, 3), 25, np.uint8)
    cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, panel])


def contact_sheet(analysis: FrameAnalysis, *, panel_width: int = 320) -> np.ndarray:
    """Stack the original, every sampled shrink and the recommended stop."""
    entries: dict[float, tuple[np.ndarray, bool]] = {}
    for ratio, image in analysis.snapshots.items():
        entries[round(ratio, 4)] = (image, False)
    if analysis.recommended is not None and analysis.recommended_image is not None:
        rec = round(analysis.recommended.ratio, 4)
        entries[rec] = (analysis.recommended_image, True)
    if not entries:
        return np.zeros((1, panel_width, 3), np.uint8)

    panels: list[np.ndarray] = []
    for ratio in sorted(entries):
        image, is_recommended = entries[ratio]
        tag = f"-{_percent(ratio)}" if ratio > 0 else "original 0%"
        if is_recommended:
            tag += f"   <- recommended ({_percent(ratio)})"
        panels.append(_label_panel(image, tag, panel_width))
    separator = np.full((4, panel_width, 3), 0, np.uint8)
    stacked: list[np.ndarray] = []
    for index, panel in enumerate(panels):
        if index:
            stacked.append(separator)
        stacked.append(panel)
    return np.vstack(stacked)


def signal_svg(analysis: FrameAnalysis, *, width: int = 720, height: int = 300) -> str:
    """A small line chart of the normalised signals against shrink ratio."""
    trace = analysis.trace
    ratios = trace.ratios
    max_shrink = max(0.05, float(ratios[-1]) if ratios.size else 0.5)
    left, right, top, bottom = 46, 12, 18, 34
    plot_w = width - left - right
    plot_h = height - top - bottom

    curves: list[tuple[str, np.ndarray, str]] = []
    for name in PLOT_SIGNALS:
        values = analysis.signals.get(name)
        if values is not None and values.size:
            curves.append((name, values, SIGNAL_COLORS[name]))
    if analysis.badness.size:
        curves.append(("composite", analysis.badness, SIGNAL_COLORS["composite"]))

    y_max = 1.25
    for _name, values, _color in curves:
        if values.size:
            y_max = max(y_max, float(np.max(values)) * 1.05)

    def sx(ratio: float) -> float:
        return left + plot_w * min(1.0, max(0.0, ratio / max_shrink))

    def sy(value: float) -> float:
        return top + plot_h * (1.0 - min(1.0, max(0.0, value / y_max)))

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#fbfbfb" stroke="#dddddd"/>',
    ]
    # y gridlines and labels
    for value in np.arange(0.0, y_max + 1e-9, 0.5):
        y = sy(float(value))
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eeeeee"/>'
        )
        parts.append(
            f'<text x="{left - 5}" y="{y + 3:.1f}" font-size="9" fill="#888888" '
            f'text-anchor="end">{value:g}</text>'
        )
    # threshold at 1.0
    yt = sy(1.0)
    parts.append(
        f'<line x1="{left}" y1="{yt:.1f}" x2="{left + plot_w}" y2="{yt:.1f}" '
        'stroke="#999999" stroke-dasharray="4 3"/>'
    )
    parts.append(
        f'<text x="{left + 4}" y="{yt - 3:.1f}" font-size="9" fill="#777777">threshold 1.0</text>'
    )
    # x gridlines
    for pct in range(0, round(max_shrink * 100) + 1, 10):
        x = sx(pct / 100)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#eeeeee"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - 10}" font-size="10" fill="#666666" '
            f'text-anchor="middle">{pct}%</text>'
        )
    # recommendation marker
    if analysis.recommended is not None:
        xr = sx(analysis.recommended.ratio)
        parts.append(
            f'<line x1="{xr:.1f}" y1="{top}" x2="{xr:.1f}" y2="{top + plot_h}" '
            'stroke="#ff9f1c" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{xr:.1f}" y="{top + 10}" font-size="10" fill="#b36b00" '
            f'text-anchor="middle">stop {_percent(analysis.recommended.ratio)}</text>'
        )
    # curves
    for name, values, color in curves:
        n = min(values.size, ratios.size)
        if n == 0:
            continue
        points = " ".join(
            f"{sx(float(ratios[i])):.1f},{sy(float(values[i])):.1f}" for i in range(n)
        )
        stroke_width = 2.2 if name == "composite" else 1.4
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke_width}"/>'
        )
    # legend, one fixed slot per curve so labels never overlap
    slot = plot_w / max(1, len(curves))
    for index, (name, _values, color) in enumerate(curves):
        lx = left + 4 + index * slot
        parts.append(f'<rect x="{lx:.1f}" y="{top + 1}" width="10" height="4" fill="{color}"/>')
        parts.append(
            f'<text x="{lx + 13:.1f}" y="{top + 8}" font-size="10" fill="#444444">'
            f"{html.escape(SIGNAL_LABELS.get(name, name))}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def summary_record(analysis: FrameAnalysis) -> dict[str, Any]:
    """A flat, JSON-serialisable row for the summary tables."""
    limits = analysis.limits
    record: dict[str, Any] = {
        "name": analysis.name,
        "faces": len(analysis.faces),
        "working_size": list(analysis.working_size),
        "base_energy": round(analysis.trace.base_energy, 2),
        "energy_early": round(analysis.energy_early, 3),
        "energy_threshold": round(analysis.energy_threshold, 3),
        "recommended": round(analysis.recommended.ratio, 4) if analysis.recommended else 0.0,
        "recommended_triggered": bool(analysis.recommended.triggered)
        if analysis.recommended
        else False,
    }
    for method in METHODS:
        limit = limits.get(method)
        record[f"{method}_ratio"] = round(limit.ratio, 4) if limit else 0.0
    composite = limits.get("composite")
    record["composite_ratio"] = round(composite.ratio, 4) if composite else 0.0
    return record


def limit_rows(analysis: FrameAnalysis) -> list[dict[str, Any]]:
    """One row per stopping rule, for the Markdown/HTML detail tables."""
    rows = []
    for name, limit in analysis.limits.items():
        rows.append(
            {
                "method": name,
                "ratio": limit.ratio,
                "triggered": limit.triggered,
                "detail": limit.detail,
            }
        )
    return rows


def write_frame_outputs(
    analysis: FrameAnalysis,
    *,
    carved_dir: Path,
    contacts_dir: Path,
    plots_dir: Path,
    thumbs_dir: Path,
    panel_width: int = 320,
    thumb_width: int = 240,
    jpeg_quality: int = 88,
) -> dict[str, str]:
    """Write the carved images, contact strip, plot and thumbnail for a frame."""
    carved_dir.mkdir(parents=True, exist_ok=True)
    contacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    name = analysis.name
    paths: dict[str, str] = {}

    # Remove this frame's outputs from a previous run so a changed stop ratio or
    # sample step cannot leave stale ``_recNN`` / ``_pNN`` files behind.
    for pattern in (f"{name}_p*.jpg", f"{name}_rec*.jpg", f"{name}_orig*.jpg"):
        for stale in carved_dir.glob(pattern):
            stale.unlink()

    params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    for ratio, image in sorted(analysis.snapshots.items()):
        path = carved_dir / f"{name}_p{round(ratio * 100):02d}.jpg"
        cv2.imwrite(str(path), image, params)
        paths[f"p{round(ratio * 100):02d}"] = str(path)
    if analysis.recommended_image is not None:
        rec = analysis.recommended.ratio if analysis.recommended else 0.0
        path = carved_dir / f"{name}_rec{round(rec * 100):02d}.jpg"
        cv2.imwrite(str(path), analysis.recommended_image, params)
        paths["rec"] = str(path)
        thumb_height = max(
            1,
            round(
                analysis.recommended_image.shape[0]
                * thumb_width
                / analysis.recommended_image.shape[1]
            ),
        )
        thumb = cv2.resize(
            analysis.recommended_image,
            (thumb_width, thumb_height),
            interpolation=cv2.INTER_AREA,
        )
        thumb_path = thumbs_dir / f"{name}.jpg"
        cv2.imwrite(str(thumb_path), thumb, params)
        paths["thumb"] = str(thumb_path)

    sheet_path = contacts_dir / f"{name}.jpg"
    cv2.imwrite(str(sheet_path), contact_sheet(analysis, panel_width=panel_width), params)
    paths["contact"] = str(sheet_path)

    plot_path = plots_dir / f"{name}.svg"
    plot_path.write_text(signal_svg(analysis), encoding="utf-8")
    paths["plot"] = str(plot_path)
    return paths


def _fmt(value: float) -> str:
    return f"{value * 100:.0f}%"


def _natural_key(name: str) -> list[tuple[int, object]]:
    """Sort ``scene_2`` before ``scene_10`` (numeric, not lexicographic)."""
    return [(0, int(part)) if part.isdigit() else (1, part) for part in re.split(r"(\d+)", name)]


def render_markdown(records: list[dict[str, Any]], config: RetargetConfig) -> str:
    """A concise human-readable summary of the run."""
    lines = [
        "# Seam-carving retarget limits",
        "",
        f"- Frames analysed: **{len(records)}**",
        f"- Max shrink cap: **{_fmt(config.max_shrink)}**",
        f"- Working width: **{config.working_width}px**",
        f"- Enabled guards: **{', '.join(config.enabled_methods)}**",
        f"- Thresholds: energy ratio {config.energy_ratio:g} "
        f"(adaptive +{config.energy_baseline_multiple:g} above early {config.energy_reference:g}), "
        f"detail budget {_fmt(config.detail_budget)}, face budget {_fmt(config.face_budget)}",
        "",
        "`COMP` is the composite recommendation: the first enabled guard to trip.  "
        "`E thr` is the effective energy threshold the frame actually used; when it "
        "is above the energy ratio, the adaptive floor is governing that frame.",
        "",
        "| Frame | Faces | Energy | E thr | Forward | Detail | Structural | **COMP** |",
        "| ----- | ----: | -----: | ----: | ------: | -----: | ---------: | -------: |",
    ]
    for row in records:
        lines.append(
            f"| {row['name']} | {row['faces']} | {_fmt(row['energy_ratio'])} | "
            f"{row.get('energy_threshold', 0.0):.3f} | "
            f"{_fmt(row['forward_ratio'])} | {_fmt(row['detail_ratio'])} | "
            f"{_fmt(row['ssim_ratio'])} | **{_fmt(row['composite_ratio'])}** |"
        )
    return "\n".join(lines) + "\n"


def render_html(
    records: list[dict[str, Any]],
    config: RetargetConfig,
    *,
    contacts_dir_name: str = "contacts",
    plots_dir_name: str = "plots",
    stamp: str = "",
) -> str:
    """A browsable report: summary table then a section per frame.

    ``stamp`` is appended to every asset URL so a browser refetches the images
    after a re-run instead of serving a cached copy of the previous run.
    """
    query = f"?v={html.escape(stamp)}" if stamp else ""
    rows = []
    for row in sorted(records, key=lambda r: _natural_key(r["name"])):
        name = html.escape(row["name"])
        rows.append(
            "<tr>"
            f'<td><a href="#{name}">{name}</a></td>'
            f'<td><img loading="lazy" src="thumbs/{name}.jpg{query}" height="54"></td>'
            f"<td>{row['faces']}</td>"
            f"<td>{_fmt(row['energy_ratio'])}</td>"
            f"<td>{row.get('energy_threshold', 0.0):.3f}</td>"
            f"<td>{_fmt(row['forward_ratio'])}</td>"
            f"<td>{_fmt(row['detail_ratio'])}</td>"
            f"<td>{_fmt(row['ssim_ratio'])}</td>"
            f"<td><b>{_fmt(row['composite_ratio'])}</b></td>"
            "</tr>"
        )

    sections = []
    for row in records:
        name = html.escape(row["name"])
        threshold = row.get("energy_threshold", 0.0)
        sections.append(
            f'<details id="{name}"><summary>{name} '
            f'<span class="rec">stop at {_fmt(row["composite_ratio"])} '
            f"&middot; energy threshold {threshold:.3f}</span></summary>"
            f'<img loading="lazy" class="sheet" src="{contacts_dir_name}/{name}.jpg{query}">'
            f'<img loading="lazy" class="plot" src="{plots_dir_name}/{name}.svg{query}">'
            "</details>"
        )

    css = """
    body { font-family: system-ui, sans-serif; margin: 24px; color: #222; background: #fff; }
    table { border-collapse: collapse; font-size: 13px; }
    th, td { border: 1px solid #ddd; padding: 3px 7px; text-align: right; }
    td:nth-child(1), td:nth-child(2) { text-align: left; }
    th { background: #f4f4f4; position: sticky; top: 0; }
    details { margin: 10px 0; border-top: 1px solid #eee; padding-top: 8px; }
    summary { cursor: pointer; font-weight: 600; }
    .sheet { display: block; max-width: 420px; border: 1px solid #ccc; margin: 8px 0; }
    .plot { display: block; max-width: 720px; border: 1px solid #eee; }
    .rec { font-size: 13px; color: #b36b00; font-weight: normal; }
    .guide { border: 1px solid #e2e2e2; padding: 8px 12px; margin: 12px 0; max-width: 900px; }
    .guide table { font-size: 12px; }
    .guide td { text-align: left; vertical-align: top; }
    .guide td:first-child { white-space: nowrap; font-weight: 600; }
    """
    guide = (
        '<details class="guide" open><summary>Column guide</summary>'
        "<p><small>Guard columns (<b>Energy</b>, <b>Forward</b>, <b>Detail</b>, "
        "<b>Structural</b>) are <b>stop ratios</b> &mdash; the fraction of width "
        "removed when that guard trips (0&ndash;50%) &mdash; not signal values. "
        "They are computed even when the guard is disabled, for comparison.</small></p>"
        "<table>"
        "<tr><td>Frame</td><td>Source still (<code>scene_NNNN</code>). "
        "Click to jump to its contact strip and signal plot.</td></tr>"
        "<tr><td>Stop preview</td><td>The frame carved to <b>COMP</b> (the recommended shrink).</td></tr>"
        "<tr><td>Faces</td><td>Anime faces detected. Drives the energetic face mask and the face budget.</td></tr>"
        "<tr><td>Energy</td><td>Where the energy guard trips: the removed seam's energy reaches "
        "<b>E thr</b> as low-detail seams run out.</td></tr>"
        "<tr><td>E thr</td><td>Effective energy threshold actually used, in units of the image's "
        "mean gradient energy. Above the config energy ratio means the adaptive margin governs that frame.</td></tr>"
        "<tr><td>Forward</td><td>Where the forward-energy guard trips (energy the removal "
        "<i>introduces</i>, i.e. structure bending). Optional; informational when off.</td></tr>"
        "<tr><td>Detail</td><td>Where the detail budget trips: too many high-gradient pixels "
        "(or face pixels) have been removed.</td></tr>"
        "<tr><td>Structural</td><td>Where SSIM to the uniformly rescaled original hits the floor. "
        "Optional; informational when off.</td></tr>"
        "<tr><td>COMP</td><td><b>Recommended stop</b>: the first enabled guard to trip = the "
        "minimum of the enabled guard ratios (<code>energy</code> and <code>detail</code> by default).</td></tr>"
        "</table></details>"
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Seam-carving retarget report</title>"
        f"<style>{css}</style></head><body>"
        "<h1>Seam-carving retarget report</h1>"
        f"<p>{len(records)} frames &middot; cap {_fmt(config.max_shrink)} &middot; "
        f"working width {config.working_width}px &middot; guards "
        f"{', '.join(config.enabled_methods)} &middot; energy {config.energy_ratio:g} "
        f"(adaptive +{config.energy_baseline_multiple:g} above early {config.energy_reference:g}) "
        f"&middot; detail {_fmt(config.detail_budget)} &middot; face {_fmt(config.face_budget)}</p>"
        + guide
        + "<table><thead><tr><th>Frame</th><th>Stop preview</th><th>Faces</th>"
        "<th>Energy</th><th>E thr</th><th>Forward</th><th>Detail</th>"
        "<th>Structural</th><th>COMP</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + "".join(sections)
        + "</body></html>"
    )


def write_summary(
    out_dir: Path,
    records: list[dict[str, Any]],
    config: RetargetConfig,
    *,
    stamp: str | None = None,
) -> dict[str, str]:
    """Write summary.csv, summary.json, report.md and report.html.

    ``stamp`` busts browser caches; it defaults to the current time so a re-run
    into the same folder is never mistaken for the previous run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or str(int(time.time()))
    csv_path = out_dir / "summary.csv"
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    json_path = out_dir / "summary.json"
    json_path.write_text(
        json.dumps({"run": stamp, "config": asdict(config), "frames": records}, indent=2),
        encoding="utf-8",
    )
    md_path = out_dir / "report.md"
    md_path.write_text(render_markdown(records, config), encoding="utf-8")
    html_path = out_dir / "report.html"
    html_path.write_text(render_html(records, config, stamp=stamp), encoding="utf-8")
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
        "html": str(html_path),
    }


def _fit_into(
    image: np.ndarray | None, box_width: int, box_height: int, *, background: int = 18
) -> np.ndarray:
    """Scale ``image`` to fit inside a fixed box, pillar/letter-boxing the rest.

    The aspect ratio is always preserved, so a frame that was seam-carved
    narrower occupies a narrower region and the freed space is left as the
    background colour - which is exactly what shows how much was removed.
    """
    canvas = np.full((box_height, box_width, 3), background, np.uint8)
    if not isinstance(image, np.ndarray) or image.size == 0:
        return canvas
    scale = min(box_width / image.shape[1], box_height / image.shape[0])
    width = max(1, round(image.shape[1] * scale))
    height = max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    x = (box_width - width) // 2
    y = (box_height - height) // 2
    canvas[y : y + height, x : x + width] = resized
    cv2.rectangle(canvas, (x - 1, y - 1), (x + width, y + height), (80, 80, 80), 1)
    return canvas


def build_overview_sheets(
    records: list[dict[str, Any]],
    *,
    thumbs_dir: Path,
    sheet_dir: Path,
    columns: int = 8,
    rows: int = 5,
    cell_width: int = 300,
    label_height: int = 26,
) -> list[str]:
    """Tile the recommended (max-shrink) stills into browsable contact sheets.

    Every cell is the same size and holds the frame's full un-shrunk footprint
    (``working_size`` aspect); the carved image sits at its true, narrower size
    inside it, pillar-boxed, with its stop percentage on the label.  Comparing
    the filled width across cells shows how much each frame was shrunk.
    """
    sheet_dir.mkdir(parents=True, exist_ok=True)
    valid = [rec for rec in records if (thumbs_dir / f"{rec['name']}.jpg").exists()]
    if not valid:
        return []
    working = valid[0].get("working_size") or [16, 9]
    box_height = max(1, round(cell_width * working[1] / working[0]))
    tile_height = label_height + box_height
    per_sheet = columns * rows
    paths: list[str] = []

    for start in range(0, len(valid), per_sheet):
        chunk = valid[start : start + per_sheet]
        canvas = np.full((rows * tile_height, columns * cell_width, 3), 12, np.uint8)
        for index, rec in enumerate(chunk):
            thumb = cv2.imread(str(thumbs_dir / f"{rec['name']}.jpg"))
            cell = np.full((tile_height, cell_width, 3), 12, np.uint8)
            bar = np.full((label_height, cell_width, 3), 25, np.uint8)
            stop = rec.get("composite_ratio", 0.0)
            cv2.putText(
                bar,
                f"{rec['name']}  -{stop * 100:.0f}%",
                (6, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (60, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cell[:label_height] = bar
            cell[label_height:] = _fit_into(thumb, cell_width, box_height)
            row, col = divmod(index, columns)
            canvas[
                row * tile_height : (row + 1) * tile_height,
                col * cell_width : (col + 1) * cell_width,
            ] = cell
        path = sheet_dir / f"overview_{start // per_sheet:02d}.png"
        cv2.imwrite(str(path), canvas)
        paths.append(str(path))
    return paths
