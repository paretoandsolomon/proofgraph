"""Generate the Fiedler bipartition figure and cluster analysis from extraction JSON.

Usage:
    python scripts/generate_figure.py <json_path> <output_dir> [--streaming] [--light] [--profile] [--top-n N]

Flags:
    --streaming  Use ijson streaming parser (low memory, three-pass).
    --light      Drop heavy attributes (type expressions) from nodes.
    --profile    Report peak memory usage via tracemalloc.
    --top-n N    Number of top modules per cluster in analysis (default 20).

Example:
    python scripts/generate_figure.py data/nat_basic.json figures/
    python scripts/generate_figure.py data/large.json figures/ --streaming --light --profile
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from proofgraph.clusters import (
    analyze_clusters,
    assign_clusters,
    print_cluster_summary,
    write_cluster_outputs,
)
from proofgraph.loader import LIGHT_ATTRS, largest_connected_component, load_extraction
from proofgraph.spectral import fiedler_vector, spectral_embedding
from proofgraph.viz import plot_fiedler_bipartition


def main(
    json_path: str, output_dir: str,
    streaming: bool = False, light: bool = False, profile: bool = False,
    top_n: int = 20,
) -> None:
    if profile:
        import tracemalloc
        tracemalloc.start()

    timings: list[tuple[str, float]] = []
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()

    use_streaming = streaming or None  # None lets load_extraction auto-detect
    mode_parts = []
    if streaming:
        mode_parts.append("streaming")
    if light:
        mode_parts.append("light attrs")
    mode_str = f" ({', '.join(mode_parts)})" if mode_parts else ""
    print(f"Loading extraction from {json_path}{mode_str}...")
    t_start = time.monotonic()
    G = load_extraction(
        json_path,
        keep_attrs=LIGHT_ATTRS if light else None,
        streaming=use_streaming,
    )
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

    # Cluster content analysis.
    print("Analyzing cluster contents...")
    t_start = time.monotonic()
    assignments = assign_clusters(H, fiedler)
    cluster_results = analyze_clusters(H, assignments, top_n=top_n)
    total_decls = H.number_of_nodes()
    md_path = write_cluster_outputs(cluster_results, output_dir_path, total_decls)
    print_cluster_summary(cluster_results, total_decls, top_n=top_n)
    timings.append(("Cluster analysis", time.monotonic() - t_start))
    print(f"  Cluster report: {md_path}")
    for label in sorted(cluster_results):
        decl_file = output_dir_path / f"cluster_{label}_declarations.txt"
        print(f"  Cluster {label} declarations: {decl_file}")

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
    flags = {"--profile", "--streaming", "--light"}
    # Parse --top-n <value> separately since it takes an argument.
    argv = sys.argv[1:]
    top_n = 20
    if "--top-n" in argv:
        idx = argv.index("--top-n")
        if idx + 1 < len(argv):
            top_n = int(argv[idx + 1])
            argv = argv[:idx] + argv[idx + 2:]
        else:
            print("Error: --top-n requires a value")
            sys.exit(1)
    positional = [a for a in argv if a not in flags]
    if len(positional) != 2:
        print(__doc__.strip())
        sys.exit(1)
    main(
        positional[0], positional[1],
        streaming="--streaming" in argv,
        light="--light" in argv,
        profile="--profile" in argv,
        top_n=top_n,
    )
