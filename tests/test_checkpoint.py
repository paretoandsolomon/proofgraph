"""Tests for proofgraph.checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np

from proofgraph.checkpoint import (
    load_graph,
    load_spectral,
    load_tree,
    save_graph,
    save_metadata,
    save_spectral,
    save_tree,
    validate_checkpoint,
)


def _sample_graph() -> nx.Graph:
    G = nx.barbell_graph(5, 1)
    for n in G.nodes():
        G.nodes[n]["kind"] = "thm"
        G.nodes[n]["module"] = f"Mod.{n}"
    return G


def _write_source_json(tmp_path: Path) -> Path:
    """Write a dummy source JSON file for fingerprinting."""
    p = tmp_path / "source.json"
    p.write_text('{"declarations": [], "edges": []}')
    return p


class TestSaveLoadGraph:
    def test_roundtrip(self, tmp_path: Path) -> None:
        G = _sample_graph()
        save_graph(G, tmp_path)
        loaded = load_graph(tmp_path)
        assert loaded is not None
        assert loaded.number_of_nodes() == G.number_of_nodes()
        assert loaded.number_of_edges() == G.number_of_edges()

    def test_preserves_attributes(self, tmp_path: Path) -> None:
        G = _sample_graph()
        save_graph(G, tmp_path)
        loaded = load_graph(tmp_path)
        for n in G.nodes():
            assert loaded.nodes[n]["kind"] == G.nodes[n]["kind"]

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        assert load_graph(tmp_path) is None

    def test_creates_directory(self, tmp_path: Path) -> None:
        G = _sample_graph()
        nested = tmp_path / "sub" / "dir"
        save_graph(G, nested)
        assert (nested / "graph.pkl").exists()


class TestSaveLoadSpectral:
    def test_roundtrip(self, tmp_path: Path) -> None:
        fiedler = np.array([0.1, -0.2, 0.3, -0.4])
        ac = 0.001234
        coords = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        save_spectral(fiedler, ac, coords, tmp_path)
        result = load_spectral(tmp_path)
        assert result is not None
        loaded_f, loaded_ac, loaded_c = result
        np.testing.assert_array_almost_equal(loaded_f, fiedler)
        assert abs(loaded_ac - ac) < 1e-10
        np.testing.assert_array_almost_equal(loaded_c, coords)

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        assert load_spectral(tmp_path) is None


class TestValidateCheckpoint:
    def test_valid_checkpoint(self, tmp_path: Path) -> None:
        source = _write_source_json(tmp_path)
        save_metadata(source, tmp_path / "ckpt", 10, 20, True)
        assert validate_checkpoint(source, tmp_path / "ckpt", True)

    def test_invalid_when_no_metadata(self, tmp_path: Path) -> None:
        source = _write_source_json(tmp_path)
        assert not validate_checkpoint(source, tmp_path / "ckpt", True)

    def test_invalid_when_file_size_changes(self, tmp_path: Path) -> None:
        source = _write_source_json(tmp_path)
        save_metadata(source, tmp_path / "ckpt", 10, 20, True)
        # Modify the source file to change its size.
        source.write_text('{"declarations": [1,2,3], "edges": []}')
        assert not validate_checkpoint(source, tmp_path / "ckpt", True)

    def test_invalid_when_light_changes(self, tmp_path: Path) -> None:
        source = _write_source_json(tmp_path)
        save_metadata(source, tmp_path / "ckpt", 10, 20, True)
        assert not validate_checkpoint(source, tmp_path / "ckpt", False)

    def test_invalid_when_corrupted_metadata(self, tmp_path: Path) -> None:
        source = _write_source_json(tmp_path)
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "metadata.json").write_text("not json")
        assert not validate_checkpoint(source, ckpt, True)


class TestSaveLoadTree:
    def test_roundtrip(self, tmp_path: Path) -> None:
        tree = {
            "label": "root",
            "depth": 0,
            "node_count": 21,
            "edge_count": 50,
            "algebraic_connectivity": 0.001,
            "stopped_reason": None,
            "elapsed_seconds": 1.5,
            "analysis": None,
            "children": [],
            "semantic_label": "TestLabel",
        }
        save_tree(tree, tmp_path)
        loaded = load_tree(tmp_path)
        assert loaded is not None
        assert loaded["label"] == "root"
        assert loaded["semantic_label"] == "TestLabel"
        assert loaded["node_count"] == 21

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        assert load_tree(tmp_path) is None

    def test_roundtrip_with_children(self, tmp_path: Path) -> None:
        tree = {
            "label": "root", "depth": 0, "node_count": 10,
            "edge_count": 15, "algebraic_connectivity": 0.01,
            "stopped_reason": None, "elapsed_seconds": 0.5,
            "analysis": {0: {"count": 5, "declarations": ["a", "b"]}},
            "children": [{
                "label": "0", "depth": 1, "node_count": 5,
                "edge_count": 5, "algebraic_connectivity": 0.1,
                "stopped_reason": "max_depth", "elapsed_seconds": 0.1,
                "analysis": None, "children": [],
            }],
        }
        save_tree(tree, tmp_path)
        loaded = load_tree(tmp_path)
        assert len(loaded["children"]) == 1
        assert loaded["children"][0]["label"] == "0"
