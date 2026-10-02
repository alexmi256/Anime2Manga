#!/usr/bin/env python3
"""Compare layout rule sets by the MD5 hashes of the panels they produced.

Point it at a layout trials folder (the one containing ``index.html`` and the
``panels/`` directory) and it reports, for every pair of sets, how many scenes
produced byte-identical images and how much their hash sets overlap.

Example
-------
    PYTHONPATH=src python scripts/layout_compare.py output/layout_trials
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from anime2manga.layout_compare import (
    compare_sets,
    comparison_to_dict,
    discover_sets,
    render_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("trials", type=Path, nargs="?", default=Path("output/layout_trials"),
                        help="Layout trials dir (contains panels/ and <set>.json)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Write comparison.md/.json here (default: the trials dir)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sets = discover_sets(args.trials)
    if len(sets) < 2:
        print(f"Need at least two sets in {args.trials}; found {len(sets)}.", file=sys.stderr)
        return 1

    comparison = compare_sets(sets)
    names = comparison.names
    width = max(len(name) for name in names) + 2

    print(f"Compared {len(sets)} sets over {len(sets[0].hashes)} scenes\n")
    print("Per-scene agreement (identical image hash):")
    print(" " * width + " ".join(f"{name:>6}" for name in names))
    for i, name in enumerate(names):
        cells = " ".join(f"{comparison.agreement[i][j] * 100:5.0f}%" for j in range(len(names)))
        print(f"{name:<{width}}{cells}")

    print("\nClosest pairs (by per-scene agreement):")
    for a, b, agree, shared in comparison.closest_pairs(limit=20):
        print(f"  {a:>6} vs {b:<6}  {agree * 100:5.1f}% scenes identical   ({shared} shared hashes)")

    out_dir = args.output or args.trials
    (out_dir / "comparison.md").write_text(render_markdown(comparison), encoding="utf-8")
    (out_dir / "comparison.json").write_text(
        json.dumps(comparison_to_dict(comparison), indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out_dir / 'comparison.md'} and comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
