"""Save and load intermediate pipeline results to avoid recomputation."""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import networkx as nx
import numpy as np


def _source_fingerprint(json_path: str | Path) -> dict:
    """Compute a fingerprint for the source JSON file."""
    p = Path(json_path)
    stat = p.stat()
    return {
        "path": str(p.resolve()),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
    }


def save_graph(H: nx.Graph, checkpoint_dir: str | Path) -> None:
    """Save the largest connected component graph to a checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with open(checkpoint_dir / "graph.pkl", "wb") as f:
        pickle.dump(H, f, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.monotonic() - t0
    print(f"  Saved graph checkpoint ({elapsed:.1f}s)")


def load_graph(checkpoint_dir: str | Path) -> nx.Graph | None:
    """Load the graph from a checkpoint directory, or return None."""
    graph_path = Path(checkpoint_dir) / "graph.pkl"
    if not graph_path.exists():
        return None
    t0 = time.monotonic()
    with open(graph_path, "rb") as f:
        H = pickle.load(f)
    elapsed = time.monotonic() - t0
    print(f"  Loaded graph checkpoint ({H.number_of_nodes():,} nodes, {elapsed:.1f}s)")
    return H


def save_spectral(
    fiedler: np.ndarray,
    algebraic_connectivity: float,
    coords: np.ndarray,
    checkpoint_dir: str | Path,
) -> None:
    """Save spectral computation results to a checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        checkpoint_dir / "spectral.npz",
        fiedler=fiedler,
        coords=coords,
        algebraic_connectivity=np.array([algebraic_connectivity]),
    )
    print("  Saved spectral checkpoint")


def load_spectral(
    checkpoint_dir: str | Path,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Load spectral results from a checkpoint directory, or return None.

    Returns (fiedler, algebraic_connectivity, coords) or None.
    """
    spectral_path = Path(checkpoint_dir) / "spectral.npz"
    if not spectral_path.exists():
        return None
    data = np.load(spectral_path)
    fiedler = data["fiedler"]
    coords = data["coords"]
    algebraic_connectivity = float(data["algebraic_connectivity"][0])
    print(
        f"  Loaded spectral checkpoint "
        f"(ac={algebraic_connectivity:.6f}, {len(fiedler):,} components)"
    )
    return fiedler, algebraic_connectivity, coords


def save_metadata(
    json_path: str | Path,
    checkpoint_dir: str | Path,
    node_count: int,
    edge_count: int,
    light: bool,
) -> None:
    """Save checkpoint metadata for cache validation."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": _source_fingerprint(json_path),
        "node_count": node_count,
        "edge_count": edge_count,
        "light": light,
    }
    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def validate_checkpoint(
    json_path: str | Path,
    checkpoint_dir: str | Path,
    light: bool,
) -> bool:
    """Check if a checkpoint is valid for the given source file and parameters.

    Returns True if the checkpoint can be used, False if it should be
    recomputed.
    """
    meta_path = Path(checkpoint_dir) / "metadata.json"
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        current = _source_fingerprint(json_path)
        stored = meta["source"]
        if current["path"] != stored["path"]:
            print(f"  Checkpoint source mismatch: {stored['path']} != {current['path']}")
            return False
        if current["size_bytes"] != stored["size_bytes"]:
            print(f"  Checkpoint stale: source file size changed")
            return False
        if current["mtime"] != stored["mtime"]:
            print(f"  Checkpoint stale: source file modified")
            return False
        if meta.get("light") != light:
            print(f"  Checkpoint stale: light mode changed")
            return False
        return True
    except (json.JSONDecodeError, KeyError):
        return False


def save_tree(tree: dict, checkpoint_dir: str | Path) -> None:
    """Save a recursive bisection tree to the checkpoint directory.

    The tree is serialized as a pickle (it contains numpy arrays in the
    analysis data that are not JSON-serializable without conversion).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "tree.pkl", "wb") as f:
        pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("  Saved bisection tree checkpoint")


def load_tree(checkpoint_dir: str | Path) -> dict | None:
    """Load a recursive bisection tree from the checkpoint directory.

    Returns None if no tree checkpoint exists.
    """
    tree_path = Path(checkpoint_dir) / "tree.pkl"
    if not tree_path.exists():
        return None
    with open(tree_path, "rb") as f:
        tree = pickle.load(f)
    print(f"  Loaded bisection tree checkpoint (label={tree.get('label', '?')})")
    return tree
