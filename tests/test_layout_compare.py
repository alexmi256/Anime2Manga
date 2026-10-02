"""Tests for the MD5-based layout rule-set comparison (``layout_compare``)."""

from __future__ import annotations

import json
from pathlib import Path

from anime2manga.layout_compare import (
    compare_sets,
    comparison_to_dict,
    discover_sets,
    panel_hashes,
    render_markdown,
)


def _make_set(trials: Path, slug: str, name: str, images: dict[int, bytes]) -> None:
    panels = trials / "panels" / slug
    panels.mkdir(parents=True, exist_ok=True)
    for index, data in images.items():
        (panels / f"scene_{index:04d}.jpg").write_bytes(data)
    (trials / f"{slug}.json").write_text(json.dumps({"set": name}), encoding="utf-8")


def test_panel_hashes_maps_scene_index(tmp_path: Path):
    _make_set(tmp_path, "K-A", "K-A", {1: b"one", 2: b"two"})
    hashes = panel_hashes(tmp_path / "panels" / "K-A")
    assert set(hashes) == {1, 2}


def test_compare_identical_and_different_scenes(tmp_path: Path):
    _make_set(tmp_path, "K-A", "K-A", {1: b"same", 2: b"alpha"})
    _make_set(tmp_path, "K-G", "K-G", {1: b"same", 2: b"beta"})
    sets = discover_sets(tmp_path)
    assert [s.name for s in sets] == ["K-A", "K-G"]

    comparison = compare_sets(sets)
    # Scene 1 is identical, scene 2 differs -> 50% per-scene agreement.
    assert comparison.agreement[0][1] == 0.5
    assert comparison.shared[0][1] == 1
    assert comparison.shared[1][0] == 1
    assert comparison.agreement[0][0] == 1.0


def test_compare_fully_identical_sets(tmp_path: Path):
    _make_set(tmp_path, "K-A", "K-A", {1: b"x", 2: b"y", 3: b"z"})
    _make_set(tmp_path, "K-B", "K-B", {1: b"x", 2: b"y", 3: b"z"})
    comparison = compare_sets(discover_sets(tmp_path))
    assert comparison.agreement[0][1] == 1.0
    assert comparison.identical[0][1] is True
    assert comparison.jaccard[0][1] == 1.0


def test_discover_sets_skips_comparison_and_orphan_json(tmp_path: Path):
    _make_set(tmp_path, "K-B", "K-B", {1: b"a"})
    (tmp_path / "comparison.json").write_text("{}", encoding="utf-8")
    (tmp_path / "orphan.json").write_text('{"set": "orphan"}', encoding="utf-8")
    names = [s.name for s in discover_sets(tmp_path)]
    assert names == ["K-B"]  # comparison.json and a json without panels/ are ignored


def test_render_markdown_and_dict(tmp_path: Path):
    _make_set(tmp_path, "K-A", "K-A", {1: b"a"})
    _make_set(tmp_path, "K-G", "K-G", {1: b"b"})
    comparison = compare_sets(discover_sets(tmp_path))
    markdown = render_markdown(comparison)
    assert "Per-scene agreement" in markdown
    assert "K-A" in markdown and "K-G" in markdown
    payload = comparison_to_dict(comparison)
    assert payload["sets"] == ["K-A", "K-G"]
    assert payload["agreement"][0][1] == 0.0
