"""Merge multiple extraction JSON files into a single graph.

Deduplicates declarations by name; unions all edges where both
endpoints exist in the merged declaration set.

Usage:
    python scripts/merge_extractions.py <output_path> <input1> <input2> ...

Example:
    python scripts/merge_extractions.py data/merged.json data/nat_basic.json data/algebra_group_basic.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def merge(input_paths: list[str], output_path: str) -> None:
    declarations: dict[str, dict] = {}
    edges: list[dict] = []
    source_modules: list[str] = []

    for path in input_paths:
        with open(path) as f:
            data = json.load(f)

        source_modules.extend(data.get("metadata", {}).get("sourceModules", []))

        for decl in data["declarations"]:
            name = decl["name"]
            if name not in declarations:
                declarations[name] = decl

        for edge in data["edges"]:
            edges.append(edge)

    # Deduplicate edges and filter to known declarations
    decl_names = set(declarations.keys())
    seen_edges: set[tuple[str, str]] = set()
    filtered_edges = []
    for edge in edges:
        key = (edge["source"], edge["target"])
        if key not in seen_edges and key[0] in decl_names and key[1] in decl_names:
            seen_edges.add(key)
            filtered_edges.append(edge)

    merged = {
        "metadata": {
            "sourceModules": sorted(set(source_modules)),
            "declarationCount": len(declarations),
            "edgeCount": len(filtered_edges),
        },
        "declarations": list(declarations.values()),
        "edges": filtered_edges,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f)

    print(f"Merged {len(input_paths)} files: {len(declarations)} declarations, {len(filtered_edges)} edges")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__.strip())
        sys.exit(1)
    merge(sys.argv[2:], sys.argv[1])
