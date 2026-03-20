"""Tests for proofgraph.viz."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from proofgraph.viz import SLATE_BLUE, TEAL, plot_fiedler_bipartition


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


class TestColorConstants:
    def test_teal_is_hex(self) -> None:
        assert TEAL.startswith("#") and len(TEAL) == 7

    def test_slate_blue_is_hex(self) -> None:
        assert SLATE_BLUE.startswith("#") and len(SLATE_BLUE) == 7
