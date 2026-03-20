"""Generate the Fiedler bipartition figure from extraction JSON.

Usage:
    python scripts/generate_figure.py <json_path> <output_dir> [--profile]

Example:
    python scripts/generate_figure.py data/nat_basic.json figures/
    python scripts/generate_figure.py data/large.json figures/ --profile
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from proofgraph.loader import largest_connected_component, load_extraction
from proofgraph.spectral import fiedler_vector, spectral_embedding
from proofgraph.viz import plot_fiedler_bipartition


def main(json_path: str, output_dir: str, profile: bool = False) -> None:
    if profile:
        import tracemalloc
        tracemalloc.start()

    timings: list[tuple[str, float]] = []
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()

    print(f"Loading extraction from {json_path}...")
    t_start = time.monotonic()
    G = load_extraction(json_path)
    timings.append(("Load extraction", time.monotonic() - t_start))
    print(f"  Full graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Extracting largest connected component...")
    t_start = time.monotonic()
    H = largest_connected_component(G)
    timings.append(("Largest component", time.monotonic() - t_start))
    print(f"  Largest component: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    if H.number_of_nodes() < 3:
        print("Error: largest connected component has fewer than 3 nodes.")
        sys.exit(1)

    print("Computing Fiedler vector (normalized Laplacian)...")
    t_start = time.monotonic()
    fiedler, algebraic_connectivity = fiedler_vector(H, normalized=True)
    timings.append(("Fiedler vector", time.monotonic() - t_start))
    print(f"  Algebraic connectivity: {algebraic_connectivity:.6f}")

    n_positive = sum(1 for v in fiedler if v >= 0)
    n_negative = len(fiedler) - n_positive
    print(f"  Bipartition: {n_positive} positive, {n_negative} negative")

    print("Computing spectral embedding (2D, normalized Laplacian)...")
    t_start = time.monotonic()
    coords = spectral_embedding(H, k=2, normalized=True)
    timings.append(("Spectral embedding", time.monotonic() - t_start))

    # Derive a readable module name from the JSON filename (e.g., "nat_basic" -> title).
    stem = Path(json_path).stem
    title = f"Spectral Bipartition: {stem} ({H.number_of_nodes()} declarations)"

    output_path = output_dir_path / "fiedler_bipartition.png"
    print(f"Generating figure at {output_path}...")
    t_start = time.monotonic()
    plot_fiedler_bipartition(
        H, fiedler, output_path,
        coords=coords,
        title=title,
        algebraic_connectivity=algebraic_connectivity,
    )
    timings.append(("Plot figure", time.monotonic() - t_start))

    total = time.monotonic() - t0
    timings.append(("Total", total))

    print("\n--- Timing Summary ---")
    for label, elapsed in timings:
        print(f"  {label:.<30s} {elapsed:>8.2f}s")

    if profile:
        import tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"\n--- Memory ---")
        print(f"  Current: {current / 1024 / 1024:.1f} MB")
        print(f"  Peak:    {peak / 1024 / 1024:.1f} MB")

    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(__doc__.strip())
        sys.exit(1)
    profile_flag = "--profile" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--profile"]
    main(args[0], args[1], profile=profile_flag)
