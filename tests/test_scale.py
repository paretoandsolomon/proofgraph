"""Scale tests: verify pipeline handles 50K+ node graphs.

These tests use synthetic Barabasi-Albert graphs to validate that spectral
analysis and visualization complete without error at scale.

Run with: pytest tests/test_scale.py -v
Skip in fast CI: pytest tests/ -v -m "not slow"
"""

from __future__ import annotations

import time
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from proofgraph.spectral import fiedler_vector, spectral_embedding
from proofgraph.viz import plot_fiedler_bipartition


@pytest.fixture(scope="module")
def large_graph() -> nx.Graph:
    """Create a 50K-node Barabasi-Albert graph (deterministic seed)."""
    return nx.barabasi_albert_graph(50_000, 3, seed=42)


@pytest.mark.slow
class TestScaleFiedler:
    def test_fiedler_completes_under_5_minutes(self, large_graph: nx.Graph) -> None:
        """Fiedler vector computation should complete in under 5 minutes."""
        start = time.monotonic()
        fv, ac = fiedler_vector(large_graph)
        elapsed = time.monotonic() - start

        assert elapsed < 300, f"Fiedler took {elapsed:.1f}s, expected < 300s"
        assert len(fv) == 50_000
        assert ac > 0

    def test_fiedler_normalized_completes(self, large_graph: nx.Graph) -> None:
        """Normalized Fiedler vector should also complete at scale."""
        fv, ac = fiedler_vector(large_graph, normalized=True)
        assert len(fv) == 50_000
        assert 0 < ac <= 2.0


@pytest.mark.slow
class TestScaleSpectralEmbedding:
    def test_embedding_shape(self, large_graph: nx.Graph) -> None:
        """Spectral embedding should produce correct shape for 50K nodes."""
        coords = spectral_embedding(large_graph, k=2)
        assert coords.shape == (50_000, 2)

    def test_embedding_normalized_shape(self, large_graph: nx.Graph) -> None:
        coords = spectral_embedding(large_graph, k=2, normalized=True)
        assert coords.shape == (50_000, 2)


@pytest.mark.slow
class TestScaleVisualization:
    def test_plot_renders_without_error(
        self, large_graph: nx.Graph, tmp_path: Path,
    ) -> None:
        """Visualization should render a 50K-node graph without error."""
        fv, ac = fiedler_vector(large_graph, normalized=True)
        coords = spectral_embedding(large_graph, k=2, normalized=True)
        output_path = tmp_path / "scale_test.png"

        plot_fiedler_bipartition(
            large_graph,
            fv,
            output_path,
            coords=coords,
            title="Scale Test (50K nodes)",
            algebraic_connectivity=ac,
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0
