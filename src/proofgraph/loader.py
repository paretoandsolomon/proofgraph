"""Load ProofGraph extraction JSON into NetworkX graphs."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


def load_extraction(path: str | Path) -> nx.DiGraph:
    """Load an extraction JSON file and return a NetworkX directed graph.

    Each node carries all declaration attributes from the JSON (name, kind,
    type, module, isConstructive, etc.). Each edge carries a ``kind`` attribute.

    Parameters
    ----------
    path : str or Path
        Path to the extraction JSON file produced by proofgraph-extract.

    Returns
    -------
    nx.DiGraph
        Directed graph with declaration nodes and dependency edges.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    G = nx.DiGraph()
    G.graph["metadata"] = data.get("metadata", {})

    for decl in data["declarations"]:
        name = decl["name"]
        G.add_node(name, **decl)

    for edge in data["edges"]:
        src, tgt = edge["source"], edge["target"]
        if src in G and tgt in G:
            G.add_edge(src, tgt, kind=edge.get("kind", "depends_on"))

    return G


def largest_connected_component(G: nx.DiGraph) -> nx.Graph:
    """Extract the largest connected component as an undirected graph.

    Spectral analysis (Laplacian eigenvectors) requires a connected graph.
    This converts the digraph to undirected and returns the largest component.

    Parameters
    ----------
    G : nx.DiGraph
        The full directed extraction graph.

    Returns
    -------
    nx.Graph
        Undirected subgraph of the largest connected component, with all
        node attributes preserved.
    """
    undirected = G.to_undirected()
    components = sorted(nx.connected_components(undirected), key=len, reverse=True)
    largest = components[0]
    return undirected.subgraph(largest).copy()
