"""Tests for proofgraph.loader."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import networkx as nx
import pytest

from proofgraph.loader import largest_connected_component, load_extraction


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


@pytest.fixture()
def simple_extraction(tmp_path: Path) -> Path:
    """Extraction JSON with 4 declarations and 3 edges."""
    data = {
        "metadata": {"source": "test"},
        "declarations": [
            {"name": "A", "kind": "theorem", "module": "Test"},
            {"name": "B", "kind": "def", "module": "Test"},
            {"name": "C", "kind": "axiom", "module": "Test"},
            {"name": "D", "kind": "def", "module": "Other"},
        ],
        "edges": [
            {"source": "A", "target": "B", "kind": "depends_on"},
            {"source": "B", "target": "C", "kind": "depends_on"},
            {"source": "A", "target": "C", "kind": "depends_on"},
        ],
    }
    p = tmp_path / "test.json"
    _write_json(p, data)
    return p


@pytest.fixture()
def disconnected_extraction(tmp_path: Path) -> Path:
    """Two disconnected components: {A, B} and {C, D, E}."""
    data = {
        "metadata": {},
        "declarations": [
            {"name": "A", "kind": "def", "module": "M1"},
            {"name": "B", "kind": "def", "module": "M1"},
            {"name": "C", "kind": "def", "module": "M2"},
            {"name": "D", "kind": "def", "module": "M2"},
            {"name": "E", "kind": "def", "module": "M2"},
        ],
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "C", "target": "D", "kind": "depends_on"},
            {"source": "D", "target": "E", "kind": "depends_on"},
        ],
    }
    p = tmp_path / "disconnected.json"
    _write_json(p, data)
    return p


class TestLoadExtraction:
    def test_returns_digraph(self, simple_extraction: Path) -> None:
        G = load_extraction(simple_extraction)
        assert isinstance(G, nx.DiGraph)

    def test_node_count(self, simple_extraction: Path) -> None:
        G = load_extraction(simple_extraction)
        assert G.number_of_nodes() == 4

    def test_edge_count(self, simple_extraction: Path) -> None:
        G = load_extraction(simple_extraction)
        assert G.number_of_edges() == 3

    def test_node_attributes(self, simple_extraction: Path) -> None:
        G = load_extraction(simple_extraction)
        assert G.nodes["A"]["kind"] == "theorem"
        assert G.nodes["B"]["module"] == "Test"

    def test_edge_kind(self, simple_extraction: Path) -> None:
        G = load_extraction(simple_extraction)
        assert G.edges["A", "B"]["kind"] == "depends_on"

    def test_edge_kind_defaults_to_depends_on(self, disconnected_extraction: Path) -> None:
        G = load_extraction(disconnected_extraction)
        assert G.edges["A", "B"]["kind"] == "depends_on"

    def test_metadata_stored(self, simple_extraction: Path) -> None:
        G = load_extraction(simple_extraction)
        assert G.graph["metadata"] == {"source": "test"}

    def test_dangling_edges_ignored(self, tmp_path: Path) -> None:
        """Edges referencing non-existent nodes are silently dropped."""
        data = {
            "metadata": {},
            "declarations": [{"name": "A", "kind": "def", "module": "M"}],
            "edges": [{"source": "A", "target": "MISSING", "kind": "depends_on"}],
        }
        p = tmp_path / "dangling.json"
        _write_json(p, data)
        G = load_extraction(p)
        assert G.number_of_edges() == 0


class TestLargestConnectedComponent:
    def test_returns_undirected(self, simple_extraction: Path) -> None:
        G = load_extraction(simple_extraction)
        H = largest_connected_component(G)
        assert isinstance(H, nx.Graph)
        assert not isinstance(H, nx.DiGraph)

    def test_selects_largest(self, disconnected_extraction: Path) -> None:
        G = load_extraction(disconnected_extraction)
        H = largest_connected_component(G)
        assert H.number_of_nodes() == 3
        assert set(H.nodes()) == {"C", "D", "E"}

    def test_preserves_attributes(self, disconnected_extraction: Path) -> None:
        G = load_extraction(disconnected_extraction)
        H = largest_connected_component(G)
        assert H.nodes["C"]["kind"] == "def"
        assert H.nodes["C"]["module"] == "M2"

    def test_single_component_returns_all(self, simple_extraction: Path) -> None:
        G = load_extraction(simple_extraction)
        H = largest_connected_component(G)
        # D is isolated (no edges to A/B/C), so the largest component is {A, B, C}
        assert H.number_of_nodes() == 3
