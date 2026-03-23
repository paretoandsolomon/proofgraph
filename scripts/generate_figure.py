"""Generate the Fiedler bipartition figure and cluster analysis from extraction JSON.

Usage:
    python scripts/generate_figure.py <json_path> <output_dir> [flags]

Flags:
    --streaming      Use ijson streaming parser (low memory, three-pass).
    --light          Drop heavy attributes (type expressions) from nodes.
    --profile        Report peak memory usage via tracemalloc.
    --top-n N        Number of top modules per cluster in analysis (default 20).
    --recursive      Run recursive spectral bisection after initial analysis.
    --max-depth N    Maximum recursion depth for bisection (default 4).
    --min-size N     Minimum cluster size to continue bisecting (default 200).
    --connectivity-ratio F  Stop recursing when a subcluster's algebraic
                     connectivity exceeds F times its parent's (default 10.0).
                     Higher values allow deeper splitting into well-connected regions.
    --checkpoint DIR Save/load intermediate results (graph, spectral data) to
                     skip expensive recomputation on subsequent runs.

Example:
    python scripts/generate_figure.py data/nat_basic.json figures/
    python scripts/generate_figure.py data/large.json figures/ --streaming --light --profile
    python scripts/generate_figure.py data/large.json figures/ --recursive --max-depth 3
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from proofgraph.checkpoint import (
    load_graph,
    load_spectral,
    save_graph,
    save_metadata,
    save_spectral,
    validate_checkpoint,
)
from proofgraph.clusters import (
    analyze_clusters,
    assign_clusters,
    collect_connectivity_profile,
    collect_leaf_assignments,
    print_cluster_summary,
    recursive_bisect,
    write_cluster_outputs,
    write_recursive_outputs,
)
from proofgraph.viz import plot_cluster_map
from proofgraph.loader import LIGHT_ATTRS, largest_connected_component, load_extraction
from proofgraph.spectral import fiedler_vector, spectral_embedding
from proofgraph.viz import plot_fiedler_bipartition


def main(
    json_path: str, output_dir: str,
    streaming: bool = False, light: bool = False, profile: bool = False,
    top_n: int = 20,
    recursive: bool = False, max_depth: int = 4, min_size: int = 200,
    connectivity_ratio: float = 10.0,
    checkpoint: str | None = None,
) -> None:
    if profile:
        import tracemalloc
        tracemalloc.start()

    timings: list[tuple[str, float]] = []
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()

    # --- Load graph (from checkpoint or source JSON) ---
    use_checkpoint = checkpoint is not None
    checkpoint_valid = (
        use_checkpoint and validate_checkpoint(json_path, checkpoint, light)
    )

    H = None
    if checkpoint_valid:
        print(f"Loading graph from checkpoint {checkpoint}...")
        t_start = time.monotonic()
        H = load_graph(checkpoint)
        if H is not None:
            timings.append(("Load graph (checkpoint)", time.monotonic() - t_start))

    if H is None:
        use_streaming = streaming or None
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
        del G  # Free the full digraph before saving checkpoint.

        if use_checkpoint:
            save_graph(H, checkpoint)
            save_metadata(json_path, checkpoint, H.number_of_nodes(), H.number_of_edges(), light)

    print(f"  Largest component: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    if H.number_of_nodes() < 3:
        print("Error: largest connected component has fewer than 3 nodes.")
        sys.exit(1)

    # --- Spectral computation (from checkpoint or fresh) ---
    fiedler = None
    if checkpoint_valid:
        print(f"Loading spectral results from checkpoint {checkpoint}...")
        t_start = time.monotonic()
        spectral_data = load_spectral(checkpoint)
        if spectral_data is not None:
            fiedler, algebraic_connectivity, coords = spectral_data
            timings.append(("Load spectral (checkpoint)", time.monotonic() - t_start))

    if fiedler is None:
        print("Computing Fiedler vector (normalized Laplacian)...")
        t_start = time.monotonic()
        fiedler, algebraic_connectivity = fiedler_vector(H, normalized=True)
        timings.append(("Fiedler vector", time.monotonic() - t_start))

        print("Computing spectral embedding (2D, normalized Laplacian)...")
        t_start = time.monotonic()
        coords = spectral_embedding(H, k=2, normalized=True)
        timings.append(("Spectral embedding", time.monotonic() - t_start))

        if use_checkpoint:
            save_spectral(fiedler, algebraic_connectivity, coords, checkpoint)

    print(f"  Algebraic connectivity: {algebraic_connectivity:.6f}")
    n_positive = sum(1 for v in fiedler if v >= 0)
    n_negative = len(fiedler) - n_positive
    print(f"  Bipartition: {n_positive} positive, {n_negative} negative")

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

    # Recursive spectral bisection.
    if recursive:
        bisection_dir = output_dir_path / "bisections"
        print(
            f"\nStarting recursive bisection "
            f"(max_depth={max_depth}, min_size={min_size}, connectivity_ratio={connectivity_ratio})..."
        )
        t_start = time.monotonic()
        tree = recursive_bisect(
            H,
            max_depth=max_depth,
            min_size=min_size,
            connectivity_ratio=connectivity_ratio,
            top_n=top_n,
            normalized=True,
            precomputed_fiedler=(fiedler, algebraic_connectivity),
            plot_dir=bisection_dir,
            precomputed_coords=coords,
        )
        timings.append(("Recursive bisection", time.monotonic() - t_start))

        # Generate leaf-cluster overlay on the full graph.
        print("Generating leaf-cluster overlay figure...")
        t_start = time.monotonic()
        leaf_assignments = collect_leaf_assignments(tree)
        # Convert string labels to sequential integers for coloring.
        unique_labels = sorted(set(leaf_assignments.values()))
        label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
        int_assignments = {n: label_to_int[lbl] for n, lbl in leaf_assignments.items()}
        n_leaves = len(unique_labels)
        overlay_title = (
            f"Recursive Bisection: {stem} "
            f"({H.number_of_nodes()} declarations, {n_leaves} leaf clusters)"
        )
        overlay_caption = (
            f"Colors indicate {n_leaves} leaf clusters from recursive spectral bisection "
            f"(max depth {max_depth}).\n"
            "Node positions are the same spectral embedding as the bipartition figure."
        )
        plot_cluster_map(
            H, int_assignments,
            output_dir_path / "recursive_clusters.png",
            coords=coords,
            title=overlay_title,
            caption=overlay_caption,
        )
        timings.append(("Overlay figure", time.monotonic() - t_start))

        # Write outputs.
        rb_path = write_recursive_outputs(tree, output_dir_path)
        print(f"  Recursive report: {rb_path}")

        # Print connectivity profile.
        profile = collect_connectivity_profile(tree)
        print("\n--- Spectral Connectivity Profile ---")
        print(f"  {'Label':<20s} {'Depth':>5s} {'Nodes':>10s} {'Alg. Conn.':>12s}  Status")
        for r in profile:
            ac_str = f"{r['algebraic_connectivity']:.6f}" if r["algebraic_connectivity"] is not None else "n/a"
            status = r["stopped_reason"] or "split"
            print(f"  {r['label']:<20s} {r['depth']:>5d} {r['node_count']:>10,} {ac_str:>12s}  {status}")
        print(f"\n  Leaf-cluster overlay: {output_dir_path / 'recursive_clusters.png'}")
        print(f"  Per-split figures: {bisection_dir}/")

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


def _parse_flag(argv: list[str], flag: str, default: float, cast: type = int) -> tuple[list[str], float]:
    """Extract a --flag VALUE pair from argv, returning (remaining_argv, value)."""
    if flag not in argv:
        return argv, default
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        print(f"Error: {flag} requires a value")
        sys.exit(1)
    value = cast(argv[idx + 1])
    return argv[:idx] + argv[idx + 2:], value


if __name__ == "__main__":
    flags = {"--profile", "--streaming", "--light", "--recursive"}
    argv = sys.argv[1:]
    argv, top_n = _parse_flag(argv, "--top-n", 20, int)
    argv, max_depth = _parse_flag(argv, "--max-depth", 4, int)
    argv, min_size = _parse_flag(argv, "--min-size", 200, int)
    argv, connectivity_ratio = _parse_flag(argv, "--connectivity-ratio", 10.0, float)
    # Parse --checkpoint (string flag, not numeric).
    checkpoint = None
    if "--checkpoint" in argv:
        idx = argv.index("--checkpoint")
        if idx + 1 >= len(argv):
            print("Error: --checkpoint requires a directory path")
            sys.exit(1)
        checkpoint = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2:]
    positional = [a for a in argv if a not in flags]
    if len(positional) != 2:
        print(__doc__.strip())
        sys.exit(1)
    main(
        positional[0], positional[1],
        streaming="--streaming" in argv,
        light="--light" in argv,
        profile="--profile" in argv,
        top_n=int(top_n),
        recursive="--recursive" in argv,
        max_depth=int(max_depth),
        min_size=int(min_size),
        connectivity_ratio=float(connectivity_ratio),
        checkpoint=checkpoint,
    )
