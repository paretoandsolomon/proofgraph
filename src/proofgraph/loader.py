"""Load ProofGraph extraction JSON into NetworkX graphs."""

from __future__ import annotations

import json
from pathlib import Path

import ijson
import networkx as nx

# Lightweight attribute set for spectral analysis. Drops the heavy ``type``
# field (~11 KB avg per declaration) while keeping structural and
# proof-theoretic properties.
LIGHT_ATTRS = frozenset({
    "name", "kind", "module",
    "isConstructive", "isNoncomputable",
    "usesChoice", "usesPropext", "usesQuot", "hasSorry",
})


def _filter_decl_attrs(
    decl: dict, keep_attrs: frozenset[str] | None,
) -> dict:
    """Filter declaration attributes if keep_attrs is specified."""
    if keep_attrs is None:
        return decl
    return {k: v for k, v in decl.items() if k in keep_attrs}


def _estimate_large_file(path: Path, threshold_mb: int = 500) -> bool:
    """Check if a file is larger than the threshold (in MB)."""
    return path.stat().st_size > threshold_mb * 1024 * 1024


def _load_streaming(
    path: Path, keep_attrs: frozenset[str] | None = None,
) -> nx.DiGraph:
    """Load extraction JSON using streaming parser (ijson).

    Processes declarations and edges one at a time, avoiding loading the
    entire file into memory. Suitable for multi-GB extraction files.
    """
    G = nx.DiGraph()

    # First pass: read metadata and declarations.
    with open(path, "rb") as f:
        # Stream metadata object.
        metadata = {}
        for prefix, event, value in ijson.parse(f):
            if prefix.startswith("metadata.") and event in (
                "string", "number", "boolean",
            ):
                key = prefix.split(".", 1)[1]
                metadata[key] = value
            # Stop once we've left the metadata object and entered declarations.
            if prefix == "declarations.item" and event == "start_map":
                break
        G.graph["metadata"] = metadata

    # Second pass: stream declarations.
    with open(path, "rb") as f:
        for decl in ijson.items(f, "declarations.item"):
            name = decl["name"]
            attrs = _filter_decl_attrs(decl, keep_attrs)
            G.add_node(name, **attrs)

    # Third pass: stream edges.
    with open(path, "rb") as f:
        for edge in ijson.items(f, "edges.item"):
            src, tgt = edge["source"], edge["target"]
            if src in G and tgt in G:
                G.add_edge(src, tgt, kind=edge.get("kind", "depends_on"))

    return G


def _load_standard(
    path: Path, keep_attrs: frozenset[str] | None = None,
) -> nx.DiGraph:
    """Load extraction JSON using standard json.load (for small files)."""
    with open(path) as f:
        data = json.load(f)

    G = nx.DiGraph()
    G.graph["metadata"] = data.get("metadata", {})

    for decl in data["declarations"]:
        name = decl["name"]
        attrs = _filter_decl_attrs(decl, keep_attrs)
        G.add_node(name, **attrs)

    for edge in data["edges"]:
        src, tgt = edge["source"], edge["target"]
        if src in G and tgt in G:
            G.add_edge(src, tgt, kind=edge.get("kind", "depends_on"))

    return G


def load_extraction(
    path: str | Path,
    keep_attrs: frozenset[str] | None = None,
    streaming: bool | None = None,
) -> nx.DiGraph:
    """Load an extraction JSON file and return a NetworkX directed graph.

    Each node carries declaration attributes from the JSON. Each edge carries
    a ``kind`` attribute.

    Parameters
    ----------
    path : str or Path
        Path to the extraction JSON file produced by proofgraph-extract.
    keep_attrs : frozenset[str] or None
        If provided, only these declaration attributes are stored on nodes.
        Use ``LIGHT_ATTRS`` to drop the heavy ``type`` field and reduce memory
        usage by ~2 GB at 180K declarations. If None (default), all attributes
        are kept.
    streaming : bool or None
        If True, use ijson streaming parser (processes one object at a time,
        suitable for multi-GB files). If False, use standard json.load. If
        None (default), automatically choose based on file size (streaming
        for files > 500 MB).

    Returns
    -------
    nx.DiGraph
        Directed graph with declaration nodes and dependency edges.
    """
    path = Path(path)
    use_streaming = streaming if streaming is not None else _estimate_large_file(path)
    if use_streaming:
        return _load_streaming(path, keep_attrs=keep_attrs)
    return _load_standard(path, keep_attrs=keep_attrs)


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
