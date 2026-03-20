"""Generate the Fiedler bipartition figure from extraction JSON.

Usage:
    python scripts/generate_figure.py <json_path> <output_dir>

Example:
    python scripts/generate_figure.py data/nat_basic.json figures/
"""

from __future__ import annotations

import sys
from pathlib import Path

from proofgraph.loader import largest_connected_component, load_extraction
from proofgraph.spectral import fiedler_vector, spectral_embedding
from proofgraph.viz import plot_fiedler_bipartition


def main(json_path: str, output_dir: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading extraction from {json_path}...")
    G = load_extraction(json_path)
    print(f"  Full graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Extracting largest connected component...")
    H = largest_connected_component(G)
    print(f"  Largest component: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    if H.number_of_nodes() < 3:
        print("Error: largest connected component has fewer than 3 nodes.")
        sys.exit(1)

    print("Computing Fiedler vector (normalized Laplacian)...")
    fiedler, algebraic_connectivity = fiedler_vector(H, normalized=True)
    print(f"  Algebraic connectivity: {algebraic_connectivity:.6f}")

    n_positive = sum(1 for v in fiedler if v >= 0)
    n_negative = len(fiedler) - n_positive
    print(f"  Bipartition: {n_positive} positive, {n_negative} negative")

    print("Computing spectral embedding (2D, normalized Laplacian)...")
    coords = spectral_embedding(H, k=2, normalized=True)

    # Derive a readable module name from the JSON filename (e.g., "nat_basic" -> title).
    stem = Path(json_path).stem
    title = f"Spectral Bipartition: {stem} ({H.number_of_nodes()} declarations)"

    output_path = output_dir / "fiedler_bipartition.png"
    print(f"Generating figure at {output_path}...")
    plot_fiedler_bipartition(
        H, fiedler, output_path,
        coords=coords,
        title=title,
        algebraic_connectivity=algebraic_connectivity,
    )
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__.strip())
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
