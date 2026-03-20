"""Tests for scripts/merge_extractions.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_extraction(path: Path, declarations: list[dict], edges: list[dict],
                      modules: list[str] | None = None) -> None:
    data = {
        "metadata": {"sourceModules": modules or []},
        "declarations": declarations,
        "edges": edges,
    }
    with open(path, "w") as f:
        json.dump(data, f)


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# Import the merge function directly.
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    "merge_extractions",
    str(Path(__file__).resolve().parent.parent / "scripts" / "merge_extractions.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
merge = _mod.merge


class TestMerge:
    def test_basic_merge(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        _write_extraction(f1,
            [{"name": "A", "kind": "def"}, {"name": "B", "kind": "thm"}],
            [{"source": "A", "target": "B", "kind": "depends_on"}],
            modules=["M1"],
        )
        _write_extraction(f2,
            [{"name": "C", "kind": "def"}],
            [{"source": "C", "target": "A", "kind": "depends_on"}],
            modules=["M2"],
        )
        out = tmp_path / "merged.json"
        merge([str(f1), str(f2)], str(out))

        result = _read_json(out)
        assert result["metadata"]["declarationCount"] == 3
        assert result["metadata"]["edgeCount"] == 2
        assert set(result["metadata"]["sourceModules"]) == {"M1", "M2"}

    def test_deduplicates_declarations(self, tmp_path: Path) -> None:
        """Same declaration in two files should appear once (first wins)."""
        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        _write_extraction(f1,
            [{"name": "A", "kind": "def", "module": "M1"}],
            [],
        )
        _write_extraction(f2,
            [{"name": "A", "kind": "thm", "module": "M2"}],
            [],
        )
        out = tmp_path / "merged.json"
        merge([str(f1), str(f2)], str(out))

        result = _read_json(out)
        assert result["metadata"]["declarationCount"] == 1
        # First-seen wins.
        assert result["declarations"][0]["kind"] == "def"
        assert result["declarations"][0]["module"] == "M1"

    def test_deduplicates_edges(self, tmp_path: Path) -> None:
        """Same edge in two files should appear once."""
        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        decls = [{"name": "A", "kind": "def"}, {"name": "B", "kind": "def"}]
        edge = {"source": "A", "target": "B", "kind": "depends_on"}
        _write_extraction(f1, decls, [edge])
        _write_extraction(f2, decls, [edge])
        out = tmp_path / "merged.json"
        merge([str(f1), str(f2)], str(out))

        result = _read_json(out)
        assert result["metadata"]["edgeCount"] == 1

    def test_drops_dangling_edges(self, tmp_path: Path) -> None:
        """Edges referencing declarations not in any file are dropped."""
        f1 = tmp_path / "a.json"
        _write_extraction(f1,
            [{"name": "A", "kind": "def"}],
            [{"source": "A", "target": "MISSING", "kind": "depends_on"}],
        )
        out = tmp_path / "merged.json"
        merge([str(f1)], str(out))

        result = _read_json(out)
        assert result["metadata"]["edgeCount"] == 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.json"
        _write_extraction(f1, [{"name": "A", "kind": "def"}], [])
        out = tmp_path / "sub" / "dir" / "merged.json"
        merge([str(f1)], str(out))
        assert out.exists()

    def test_empty_inputs(self, tmp_path: Path) -> None:
        f1 = tmp_path / "empty.json"
        _write_extraction(f1, [], [])
        out = tmp_path / "merged.json"
        merge([str(f1)], str(out))

        result = _read_json(out)
        assert result["metadata"]["declarationCount"] == 0
        assert result["metadata"]["edgeCount"] == 0
