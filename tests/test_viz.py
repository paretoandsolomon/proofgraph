"""Tests for proofgraph.viz."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from proofgraph.viz import (
    CLUSTER_PALETTE,
    SLATE_BLUE,
    TEAL,
    cluster_color,
    plot_cluster_map,
    plot_fiedler_bipartition,
)


def _small_graph_with_fiedler() -> tuple[nx.Graph, np.ndarray]:
    """Return a small graph and a mock Fiedler vector."""
    G = nx.path_graph(6)
    fiedler = np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])
    return G, fiedler


class TestPlotFiedlerBipartition:
    def test_creates_file(self, tmp_path: Path) -> None:
        G, fiedler = _small_graph_with_fiedler()
        out = tmp_path / "test.png"
        plot_fiedler_bipartition(G, fiedler, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        G, fiedler = _small_graph_with_fiedler()
        out = tmp_path / "sub" / "dir" / "test.png"
        plot_fiedler_bipartition(G, fiedler, out)
        assert out.exists()

    def test_with_spectral_coords(self, tmp_path: Path) -> None:
        G, fiedler = _small_graph_with_fiedler()
        coords = np.column_stack([fiedler, np.arange(6, dtype=float)])
        out = tmp_path / "spectral.png"
        plot_fiedler_bipartition(G, fiedler, out, coords=coords)
        assert out.exists()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        G, fiedler = _small_graph_with_fiedler()
        out = str(tmp_path / "string_path.png")
        plot_fiedler_bipartition(G, fiedler, out)
        assert Path(out).exists()


    def test_with_title_and_connectivity(self, tmp_path: Path) -> None:
        G, fiedler = _small_graph_with_fiedler()
        out = tmp_path / "titled.png"
        plot_fiedler_bipartition(
            G, fiedler, out,
            title="Test Title",
            algebraic_connectivity=0.1234,
        )
        assert out.exists()

    def test_degree_scaled_nodes_no_error(self, tmp_path: Path) -> None:
        """Graph with heterogeneous degrees should render without error."""
        G = nx.star_graph(10)  # Center node has degree 10, leaves have degree 1.
        fiedler = np.zeros(11)
        fiedler[:6] = 1.0
        fiedler[6:] = -1.0
        coords = np.column_stack([fiedler, np.arange(11, dtype=float)])
        out = tmp_path / "star.png"
        plot_fiedler_bipartition(G, fiedler, out, coords=coords)
        assert out.exists()

    def test_single_node_graph(self, tmp_path: Path) -> None:
        """Degenerate case: single-node graph should not crash."""
        G = nx.Graph()
        G.add_node("A")
        fiedler = np.array([0.0])
        out = tmp_path / "single.png"
        plot_fiedler_bipartition(G, fiedler, out)
        assert out.exists()


class TestPlotClusterMap:
    def test_creates_file(self, tmp_path: Path) -> None:
        G = nx.path_graph(6)
        assignments = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
        coords = np.column_stack([np.arange(6, dtype=float), np.zeros(6)])
        out = tmp_path / "clusters.png"
        plot_cluster_map(G, assignments, out, coords=coords)
        assert out.exists()

    def test_many_clusters(self, tmp_path: Path) -> None:
        G = nx.path_graph(20)
        assignments = {i: i for i in range(20)}
        coords = np.column_stack([np.arange(20, dtype=float), np.zeros(20)])
        out = tmp_path / "many.png"
        plot_cluster_map(G, assignments, out, coords=coords)
        assert out.exists()

    def test_with_title_and_caption(self, tmp_path: Path) -> None:
        G = nx.path_graph(4)
        assignments = {0: 0, 1: 0, 2: 1, 3: 1}
        coords = np.column_stack([np.arange(4, dtype=float), np.zeros(4)])
        out = tmp_path / "titled.png"
        plot_cluster_map(
            G, assignments, out, coords=coords,
            title="Test", caption="A caption",
        )
        assert out.exists()

    def test_without_coords(self, tmp_path: Path) -> None:
        G = nx.path_graph(4)
        assignments = {0: 0, 1: 0, 2: 1, 3: 1}
        out = tmp_path / "no_coords.png"
        plot_cluster_map(G, assignments, out)
        assert out.exists()


class TestClusterColor:
    def test_returns_hex(self) -> None:
        c = cluster_color(0, 4)
        assert c.startswith("#")

    def test_palette_colors_distinct(self) -> None:
        colors = [cluster_color(i, 8) for i in range(8)]
        assert len(set(colors)) == 8

    def test_fallback_for_many_clusters(self) -> None:
        c = cluster_color(15, 20)
        assert c.startswith("#")


class TestColorConstants:
    def test_teal_is_hex(self) -> None:
        assert TEAL.startswith("#") and len(TEAL) == 7

    def test_slate_blue_is_hex(self) -> None:
        assert SLATE_BLUE.startswith("#") and len(SLATE_BLUE) == 7

    def test_palette_has_at_least_12(self) -> None:
        assert len(CLUSTER_PALETTE) >= 12
