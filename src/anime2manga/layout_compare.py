"""Compare panel-layout rule sets by hashing the panel images they produced.

The trial harness writes one image per scene per set under
``<trials>/panels/<set-slug>/scene_NNNN.jpg``.  Two sets that make the *same*
crop/carve decision for a scene produce byte-identical JPEGs, so comparing the
image hashes tells you how similar two rule sets behave without re-planning.

This module only reads hashes; it never re-renders.  :func:`discover_sets` finds
the sets and :func:`compare_sets` builds the agreement matrices.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

_SCENE_RE = re.compile(r"scene_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


@dataclass(frozen=True)
class SetHashes:
    """One rule set's per-scene image hashes."""

    name: str
    slug: str
    hashes: dict[int, str]

    @property
    def unique_hashes(self) -> set[str]:
        return set(self.hashes.values())


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def panel_hashes(panels_dir: Path) -> dict[int, str]:
    """Map ``scene index -> md5`` for every ``scene_NNNN.jpg`` in a folder."""
    hashes: dict[int, str] = {}
    if not panels_dir.is_dir():
        return hashes
    for path in sorted(panels_dir.iterdir()):
        match = _SCENE_RE.match(path.name)
        if match:
            hashes[int(match.group(1))] = md5_file(path)
    return hashes


def discover_sets(trials_dir: Path) -> list[SetHashes]:
    """Find every rule set that has a ``panels/<stem>/`` folder and hash it.

    Skips the summary and comparison JSON files so the tool can be run twice in
    the same directory without inventing a bogus set.
    """
    sets: list[SetHashes] = []
    for meta_path in sorted(trials_dir.glob("*.json")):
        if meta_path.name in {"summary.json", "comparison.json"}:
            continue
        slug = meta_path.stem
        if not (trials_dir / "panels" / slug).is_dir():
            continue
        try:
            name = json.loads(meta_path.read_text(encoding="utf-8")).get("set") or slug
        except (OSError, ValueError):
            name = slug
        sets.append(SetHashes(name=name, slug=slug, hashes=panel_hashes(trials_dir / "panels" / slug)))
    return sets


@dataclass
class Comparison:
    """Pairwise agreement between rule sets."""

    names: list[str]
    #: ``agreement[i][j]`` = fraction of shared scenes whose image hash matches.
    agreement: list[list[float]]
    #: ``shared[i][j]`` = number of distinct hashes the two sets have in common.
    shared: list[list[int]]
    #: ``jaccard[i][j]`` = shared hashes / union of hashes.
    jaccard: list[list[float]]
    #: ``identical[i][j]`` = True when every shared scene hashes the same.
    identical: list[list[bool]]

    def closest_pairs(self, limit: int = 10) -> list[tuple[str, str, float, int]]:
        """Pairs with the highest per-scene agreement (excluding self)."""
        pairs: list[tuple[str, str, float, int]] = []
        for i, name_i in enumerate(self.names):
            for j in range(i + 1, len(self.names)):
                pairs.append((name_i, self.names[j], self.agreement[i][j], self.shared[i][j]))
        pairs.sort(key=lambda item: (-item[2], -item[3]))
        return pairs[:limit]


def compare_sets(sets: list[SetHashes]) -> Comparison:
    """Build the per-scene agreement, shared-hash and Jaccard matrices."""
    count = len(sets)
    agreement = [[0.0] * count for _ in range(count)]
    shared = [[0] * count for _ in range(count)]
    jaccard = [[0.0] * count for _ in range(count)]
    identical = [[False] * count for _ in range(count)]
    for i in range(count):
        for j in range(count):
            if i == j:
                agreement[i][j] = 1.0
                jaccard[i][j] = 1.0
                identical[i][j] = True
                shared[i][j] = len(sets[i].unique_hashes)
                continue
            common = set(sets[i].hashes) & set(sets[j].hashes)
            if common:
                same = sum(1 for scene in common if sets[i].hashes[scene] == sets[j].hashes[scene])
                agreement[i][j] = same / len(common)
                identical[i][j] = same == len(common)
            shared[i][j] = len(sets[i].unique_hashes & sets[j].unique_hashes)
            union = sets[i].unique_hashes | sets[j].unique_hashes
            jaccard[i][j] = len(sets[i].unique_hashes & sets[j].unique_hashes) / len(union) if union else 0.0
    return Comparison(
        [sets[i].name for i in range(count)], agreement, shared, jaccard, identical
    )


def render_markdown(comparison: Comparison) -> str:
    """A readable report: per-scene agreement matrix, then closest pairs."""
    names = comparison.names
    lines = ["# Layout rule-set comparison", "", "## Per-scene agreement (higher = more similar)", ""]
    lines.append("| set | " + " | ".join(names) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in names) + " |")
    for i, name in enumerate(names):
        cells = " | ".join(f"{comparison.agreement[i][j] * 100:.0f}%" for j in range(len(names)))
        lines.append(f"| **{name}** | {cells} |")
    lines.append("")
    lines.append("## Shared image hashes")
    lines.append("| set | " + " | ".join(names) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in names) + " |")
    for i, name in enumerate(names):
        cells = " | ".join(str(comparison.shared[i][j]) for j in range(len(names)))
        lines.append(f"| **{name}** | {cells} |")
    lines.append("")
    lines.append("## Closest pairs")
    lines.append("")
    lines.append("| A | B | scene agreement | shared hashes |")
    lines.append("| --- | --- | ---: | ---: |")
    for a, b, agree, shared in comparison.closest_pairs(limit=20):
        lines.append(f"| {a} | {b} | {agree * 100:.0f}% | {shared} |")
    return "\n".join(lines) + "\n"


def comparison_to_dict(comparison: Comparison) -> dict:
    return {
        "sets": comparison.names,
        "agreement": comparison.agreement,
        "shared_hashes": comparison.shared,
        "jaccard": comparison.jaccard,
        "identical": comparison.identical,
    }


__all__ = [
    "Comparison",
    "SetHashes",
    "compare_sets",
    "comparison_to_dict",
    "discover_sets",
    "md5_file",
    "panel_hashes",
    "render_markdown",
]
