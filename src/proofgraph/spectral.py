"""Spectral graph analysis: Laplacian, Fiedler vector, spectral embedding."""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def graph_laplacian(G: nx.Graph, normalized: bool = False) -> scipy.sparse.csr_matrix:
    """Compute the graph Laplacian as a sparse matrix.

    Parameters
    ----------
    G : nx.Graph
        An undirected, connected graph.
    normalized : bool
        If True, compute the symmetric normalized Laplacian
        L_sym = D^{-1/2} L D^{-1/2}. This produces more balanced embeddings
        for graphs with heterogeneous degree distributions.

    Returns
    -------
    scipy.sparse.csr_matrix
        The graph Laplacian L = D - A (or normalized variant).
    """
    A = nx.adjacency_matrix(G)
    return scipy.sparse.csgraph.laplacian(A, normed=normalized)


def fiedler_vector(
    G: nx.Graph, normalized: bool = False,
) -> tuple[np.ndarray, float]:
    """Compute the Fiedler vector and algebraic connectivity.

    The Fiedler vector is the eigenvector corresponding to the second smallest
    eigenvalue of the graph Laplacian. Its sign induces a bipartition that
    approximates the sparsest graph cut.

    Parameters
    ----------
    G : nx.Graph
        An undirected, connected graph with at least 3 nodes.
    normalized : bool
        If True, use the symmetric normalized Laplacian.

    Returns
    -------
    fiedler : np.ndarray
        The Fiedler vector (one component per node, ordered as ``list(G.nodes())``).
    algebraic_connectivity : float
        The second smallest eigenvalue of the Laplacian.
    """
    L = graph_laplacian(G, normalized=normalized)
    # Request the 2 smallest eigenvalues; the smallest is 0 (connected graph).
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(L, k=2, which="SM")

    # eigsh returns eigenvalues in ascending order.
    algebraic_connectivity = float(eigenvalues[1])
    fiedler = eigenvectors[:, 1]
    return fiedler, algebraic_connectivity


def spectral_embedding(
    G: nx.Graph, k: int = 2, normalized: bool = False,
) -> np.ndarray:
    """Compute a k-dimensional spectral embedding of the graph.

    Uses the first k non-trivial eigenvectors of the graph Laplacian as
    node coordinates.

    Parameters
    ----------
    G : nx.Graph
        An undirected, connected graph.
    k : int
        Number of embedding dimensions. Must satisfy k+1 <= number of nodes.
    normalized : bool
        If True, use the symmetric normalized Laplacian.

    Returns
    -------
    np.ndarray
        Array of shape (n_nodes, k) with spectral coordinates.
    """
    L = graph_laplacian(G, normalized=normalized)
    # Request k+1 smallest eigenvalues; skip the trivial zero eigenvalue.
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(L, k=k + 1, which="SM")
    # Columns 1..k are the non-trivial eigenvectors.
    return eigenvectors[:, 1 : k + 1]
