"""Tests for proofgraph.spectral."""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse

from proofgraph.spectral import fiedler_vector, graph_laplacian, spectral_embedding


def _path_graph(n: int = 6) -> nx.Graph:
    """Simple connected path graph: 0-1-2-..-(n-1)."""
    return nx.path_graph(n)


def _barbell_graph() -> nx.Graph:
    """Two cliques of 5 connected by a single edge (clear bipartition)."""
    return nx.barbell_graph(5, 0)


class TestGraphLaplacian:
    def test_returns_sparse(self) -> None:
        L = graph_laplacian(_path_graph())
        assert scipy.sparse.issparse(L)

    def test_shape(self) -> None:
        G = _path_graph(8)
        L = graph_laplacian(G)
        assert L.shape == (8, 8)

    def test_row_sums_zero(self) -> None:
        L = graph_laplacian(_path_graph())
        row_sums = np.array(L.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 0, atol=1e-12)

    def test_symmetric(self) -> None:
        L = graph_laplacian(_path_graph())
        diff = L - L.T
        assert abs(diff).max() < 1e-12

    def test_diagonal_equals_degree(self) -> None:
        G = _path_graph(5)
        L = graph_laplacian(G)
        diag = np.array(L.diagonal()).flatten()
        degrees = np.array([G.degree(n) for n in range(5)])
        np.testing.assert_array_equal(diag, degrees)


class TestFiedlerVector:
    def test_returns_tuple(self) -> None:
        result = fiedler_vector(_path_graph())
        assert isinstance(result, tuple) and len(result) == 2

    def test_vector_length_matches_nodes(self) -> None:
        G = _path_graph(10)
        fv, _ = fiedler_vector(G)
        assert len(fv) == 10

    def test_algebraic_connectivity_positive(self) -> None:
        _, ac = fiedler_vector(_path_graph())
        assert ac > 0

    def test_barbell_clear_bipartition(self) -> None:
        """Barbell graph should split cleanly into its two cliques."""
        G = _barbell_graph()
        fv, _ = fiedler_vector(G)
        signs = fv >= 0
        # One clique should be positive, the other negative
        # (or vice versa, but they should be separated)
        n_pos = signs.sum()
        n_neg = len(signs) - n_pos
        assert {n_pos, n_neg} == {5, 5}

    def test_algebraic_connectivity_complete_graph(self) -> None:
        """Complete graph on n nodes has algebraic connectivity n."""
        G = nx.complete_graph(6)
        _, ac = fiedler_vector(G)
        np.testing.assert_allclose(ac, 6.0, atol=0.1)

    def test_path_graph_connectivity_decreases_with_length(self) -> None:
        """Longer path graphs have lower algebraic connectivity."""
        _, ac_short = fiedler_vector(_path_graph(5))
        _, ac_long = fiedler_vector(_path_graph(20))
        assert ac_long < ac_short


class TestNormalizedLaplacian:
    def test_returns_sparse(self) -> None:
        L = graph_laplacian(_path_graph(), normalized=True)
        assert scipy.sparse.issparse(L)

    def test_shape(self) -> None:
        L = graph_laplacian(_path_graph(8), normalized=True)
        assert L.shape == (8, 8)

    def test_symmetric(self) -> None:
        L = graph_laplacian(_path_graph(), normalized=True)
        diff = L - L.T
        assert abs(diff).max() < 1e-12

    def test_diagonal_is_one(self) -> None:
        """Normalized Laplacian has 1 on the diagonal for non-isolated nodes."""
        G = _path_graph(5)
        L = graph_laplacian(G, normalized=True)
        diag = np.array(L.diagonal()).flatten()
        np.testing.assert_allclose(diag, 1.0, atol=1e-12)

    def test_eigenvalues_bounded_0_to_2(self) -> None:
        """All eigenvalues of the normalized Laplacian lie in [0, 2]."""
        G = _barbell_graph()
        L = graph_laplacian(G, normalized=True)
        eigenvalues = np.linalg.eigvalsh(L.toarray())
        assert eigenvalues.min() >= -1e-10
        assert eigenvalues.max() <= 2.0 + 1e-10


class TestNormalizedFiedlerVector:
    def test_returns_tuple(self) -> None:
        result = fiedler_vector(_path_graph(), normalized=True)
        assert isinstance(result, tuple) and len(result) == 2

    def test_algebraic_connectivity_positive(self) -> None:
        _, ac = fiedler_vector(_path_graph(), normalized=True)
        assert ac > 0

    def test_algebraic_connectivity_bounded_by_2(self) -> None:
        _, ac = fiedler_vector(_path_graph(), normalized=True)
        assert ac <= 2.0

    def test_complete_graph_normalized_connectivity(self) -> None:
        """Complete graph on n nodes has normalized algebraic connectivity n/(n-1)."""
        G = nx.complete_graph(6)
        _, ac = fiedler_vector(G, normalized=True)
        np.testing.assert_allclose(ac, 6.0 / 5.0, atol=0.1)

    def test_barbell_clear_bipartition(self) -> None:
        """Barbell graph should still split cleanly with normalized Laplacian."""
        G = _barbell_graph()
        fv, _ = fiedler_vector(G, normalized=True)
        signs = fv >= 0
        n_pos = signs.sum()
        n_neg = len(signs) - n_pos
        assert {n_pos, n_neg} == {5, 5}


class TestNormalizedSpectralEmbedding:
    def test_shape(self) -> None:
        G = _path_graph(10)
        coords = spectral_embedding(G, k=2, normalized=True)
        assert coords.shape == (10, 2)

    def test_first_column_correlates_with_fiedler(self) -> None:
        G = _path_graph(10)
        coords = spectral_embedding(G, k=2, normalized=True)
        fv, _ = fiedler_vector(G, normalized=True)
        corr = abs(np.corrcoef(coords[:, 0], fv)[0, 1])
        assert corr > 0.99


class TestSpectralEmbedding:
    def test_shape_default_k(self) -> None:
        G = _path_graph(10)
        coords = spectral_embedding(G)
        assert coords.shape == (10, 2)

    def test_shape_custom_k(self) -> None:
        G = _path_graph(10)
        coords = spectral_embedding(G, k=3)
        assert coords.shape == (10, 3)

    def test_first_column_is_fiedler(self) -> None:
        """First embedding dimension should be the Fiedler vector (up to sign)."""
        G = _path_graph(10)
        coords = spectral_embedding(G, k=2)
        fv, _ = fiedler_vector(G)
        # They may differ by sign; check correlation
        corr = abs(np.corrcoef(coords[:, 0], fv)[0, 1])
        assert corr > 0.99

    def test_columns_orthogonal(self) -> None:
        G = _path_graph(20)
        coords = spectral_embedding(G, k=3)
        gram = coords.T @ coords
        off_diag = gram - np.diag(np.diag(gram))
        np.testing.assert_allclose(off_diag, 0, atol=1e-10)
