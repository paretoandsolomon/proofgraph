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
    --checkpoint DIR Save/load intermediate results (graph, spectral data,
                     bisection tree) to skip expensive recomputation.
    --rerender       Re-render all bisection figures from a saved checkpoint
                     without re-running the recursive bisection. Requires
                     --checkpoint with a previously saved tree. Use this to
                     iterate on label parameters or visualization style.

Figure Labeling:
    Per-split bisection figures are labeled semantically by default (Option B).
    Each figure title shows the parent cluster's semantic name, and the legend
    shows the semantic names of the two child clusters. Labels are computed
    on-the-fly during recursion from the module breakdown data, so they are
    available immediately without a second pass.

    The --rerender flag (Option C) reloads the graph and bisection tree from
    checkpoint, recomputes spectral embeddings for each split, and generates
    all figures with semantic labels. This lets you change label parameters
    or visualization settings without re-running the expensive recursive
    bisection (~75 min at full Mathlib scale). The spectral re-embedding adds
    ~12 min for the largest subgraph; smaller subgraphs are fast.

Example:
    # Basic analysis
    python scripts/generate_figure.py data/nat_basic.json figures/

    # Full pipeline with checkpoint
    python scripts/generate_figure.py data/mathlib_full.json figures/ \\
        --streaming --light --recursive --checkpoint data/checkpoints

    # Re-render figures from checkpoint (no re-bisection)
    python scripts/generate_figure.py data/mathlib_full.json figures/ \\
        --checkpoint data/checkpoints --rerender

    # Re-render with different recursion depth
    python scripts/generate_figure.py data/mathlib_full.json figures/ \\
        --checkpoint data/checkpoints --rerender --connectivity-ratio 50.0
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from proofgraph.checkpoint import (
    load_graph,
    load_spectral,
    load_tree,
    save_graph,
    save_metadata,
    save_spectral,
    save_tree,
    validate_checkpoint,
)
from proofgraph.clusters import (
    analyze_clusters,
    assign_clusters,
    collect_connectivity_profile,
    collect_leaf_assignments,
    label_tree,
    print_cluster_summary,
    recursive_bisect,
    rerender_bisection_figures,
    write_cluster_outputs,
    write_recursive_outputs,
)
from proofgraph.viz import plot_cluster_map, render_dendrogram
from proofgraph.loader import LIGHT_ATTRS, largest_connected_component, load_extraction
from proofgraph.spectral import fiedler_vector, spectral_embedding
from proofgraph.viz import plot_fiedler_bipartition


def _load_or_compute_graph(
    json_path: str, checkpoint: str | None, checkpoint_valid: bool,
    streaming: bool, light: bool, timings: list,
):
    """Load the graph from checkpoint or source JSON. Returns H."""
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
        del G

        if checkpoint is not None:
            save_graph(H, checkpoint)
            save_metadata(json_path, checkpoint, H.number_of_nodes(), H.number_of_edges(), light)

    print(f"  Largest component: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
    return H


def _load_or_compute_spectral(
    H, checkpoint: str | None, checkpoint_valid: bool, timings: list,
):
    """Load spectral data from checkpoint or compute fresh. Returns (fiedler, ac, coords)."""
    fiedler = None
    if checkpoint_valid:
        print(f"Loading spectral results from checkpoint {checkpoint}...")
        t_start = time.monotonic()
        spectral_data = load_spectral(checkpoint)
        if spectral_data is not None:
            fiedler, ac, coords = spectral_data
            timings.append(("Load spectral (checkpoint)", time.monotonic() - t_start))

    if fiedler is None:
        print("Computing Fiedler vector (normalized Laplacian)...")
        t_start = time.monotonic()
        fiedler, ac = fiedler_vector(H, normalized=True)
        timings.append(("Fiedler vector", time.monotonic() - t_start))

        print("Computing spectral embedding (2D, normalized Laplacian)...")
        t_start = time.monotonic()
        coords = spectral_embedding(H, k=2, normalized=True)
        timings.append(("Spectral embedding", time.monotonic() - t_start))

        if checkpoint is not None:
            save_spectral(fiedler, ac, coords, checkpoint)

    return fiedler, ac, coords


def _generate_recursive_outputs(
    tree, H, coords, stem, output_dir_path, max_depth,
    semantic_labels, timings,
):
    """Generate overlay figure, dendrogram, markdown report, and print profile."""
    # Leaf-cluster overlay on the full graph.
    print("Generating leaf-cluster overlay figure...")
    t_start = time.monotonic()
    leaf_assignments = collect_leaf_assignments(tree)
    unique_labels = sorted(set(leaf_assignments.values()))
    label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
    int_assignments = {n: label_to_int[lbl] for n, lbl in leaf_assignments.items()}
    int_label_names = {
        label_to_int[lbl]: semantic_labels.get(lbl, lbl)
        for lbl in unique_labels
    }
    n_leaves = len(unique_labels)
    overlay_title = (
        f"Recursive Spectral Bisection: {stem} "
        f"({H.number_of_nodes():,} declarations, {n_leaves} leaf clusters)"
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
        label_names=int_label_names,
        annotate=True,
    )
    timings.append(("Overlay figure", time.monotonic() - t_start))

    # Dendrogram.
    print("Generating hierarchy dendrogram...")
    t_start = time.monotonic()
    for fmt in ("png", "svg"):
        render_dendrogram(
            tree,
            output_dir_path / "dendrogram",
            semantic_labels=semantic_labels,
            fmt=fmt,
        )
    timings.append(("Dendrogram", time.monotonic() - t_start))

    # Markdown report.
    rb_path = write_recursive_outputs(
        tree, output_dir_path, semantic_labels=semantic_labels,
    )
    print(f"  Recursive report: {rb_path}")

    # Connectivity profile.
    profile = collect_connectivity_profile(tree)
    print("\n--- Spectral Connectivity Profile ---")
    max_sem_len = max(
        len(semantic_labels.get(r["label"], r["label"])) for r in profile
    )
    col_w = max(max_sem_len, 20)
    print(f"  {'Label':<{col_w}s} {'Path':<15s} {'Nodes':>10s} {'Alg. Conn.':>12s}  Status")
    for r in profile:
        ac_str = (
            f"{r['algebraic_connectivity']:.6f}"
            if r["algebraic_connectivity"] is not None else "n/a"
        )
        status = r["stopped_reason"] or "split"
        sem = semantic_labels.get(r["label"], r["label"])
        print(
            f"  {sem:<{col_w}s} {r['label']:<15s} "
            f"{r['node_count']:>10,} {ac_str:>12s}  {status}"
        )
    print(f"\n  Leaf-cluster overlay: {output_dir_path / 'recursive_clusters.png'}")
    print(f"  Dendrogram: {output_dir_path / 'dendrogram.png'}")


def main(
    json_path: str, output_dir: str,
    streaming: bool = False, light: bool = False, profile: bool = False,
    top_n: int = 20,
    recursive: bool = False, max_depth: int = 4, min_size: int = 200,
    connectivity_ratio: float = 10.0,
    checkpoint: str | None = None,
    rerender: bool = False,
) -> None:
    if profile:
        import tracemalloc
        tracemalloc.start()

    timings: list[tuple[str, float]] = []
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()

    use_checkpoint = checkpoint is not None
    checkpoint_valid = (
        use_checkpoint and validate_checkpoint(json_path, checkpoint, light)
    )

    # ------------------------------------------------------------------
    # Option C: --rerender mode
    # Loads graph + tree from checkpoint, recomputes embeddings, and
    # generates all figures with semantic labels. Skips the expensive
    # recursive bisection.
    # ------------------------------------------------------------------
    if rerender:
        if not use_checkpoint:
            print("Error: --rerender requires --checkpoint <dir>")
            sys.exit(1)

        # Load graph.
        H = _load_or_compute_graph(
            json_path, checkpoint, checkpoint_valid,
            streaming, light, timings,
        )
        if H.number_of_nodes() < 3:
            print("Error: graph too small.")
            sys.exit(1)

        # Load spectral data.
        fiedler, algebraic_connectivity, coords = _load_or_compute_spectral(
            H, checkpoint, checkpoint_valid, timings,
        )

        # Load bisection tree.
        print(f"Loading bisection tree from checkpoint {checkpoint}...")
        tree = load_tree(checkpoint)
        if tree is None:
            print(
                "Error: no bisection tree found in checkpoint. "
                "Run with --recursive first to generate the tree."
            )
            sys.exit(1)

        # Compute semantic labels.
        semantic_labels = label_tree(tree)
        print(f"  Semantic labels: {len(semantic_labels)} nodes labeled")

        # Re-render per-split figures.
        bisection_dir = output_dir_path / "bisections"
        print(f"\nRe-rendering bisection figures...")
        t_start = time.monotonic()
        rerender_bisection_figures(
            tree, H, bisection_dir,
            semantic_labels=semantic_labels,
        )
        timings.append(("Rerender figures", time.monotonic() - t_start))

        # Generate all other outputs (overlay, dendrogram, report).
        stem = Path(json_path).stem
        _generate_recursive_outputs(
            tree, H, coords, stem, output_dir_path, max_depth,
            semantic_labels, timings,
        )
        print(f"  Per-split figures: {bisection_dir}/")

        total = time.monotonic() - t0
        timings.append(("Total", total))
        _print_timing_and_memory(timings, profile)
        return

    # ------------------------------------------------------------------
    # Normal mode (with optional Option B on-the-fly labeling)
    # ------------------------------------------------------------------

    H = _load_or_compute_graph(
        json_path, checkpoint, checkpoint_valid,
        streaming, light, timings,
    )
    if H.number_of_nodes() < 3:
        print("Error: largest connected component has fewer than 3 nodes.")
        sys.exit(1)

    fiedler, algebraic_connectivity, coords = _load_or_compute_spectral(
        H, checkpoint, checkpoint_valid, timings,
    )

    print(f"  Algebraic connectivity: {algebraic_connectivity:.6f}")
    n_positive = sum(1 for v in fiedler if v >= 0)
    n_negative = len(fiedler) - n_positive
    print(f"  Bipartition: {n_positive} positive, {n_negative} negative")

    stem = Path(json_path).stem
    title = f"Spectral Bipartition: {stem} ({H.number_of_nodes():,} declarations)"

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

    # Recursive spectral bisection (Option B: on-the-fly semantic labels).
    if recursive:
        bisection_dir = output_dir_path / "bisections"
        print(
            f"\nStarting recursive bisection "
            f"(max_depth={max_depth}, min_size={min_size}, "
            f"connectivity_ratio={connectivity_ratio})..."
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

        # Save tree to checkpoint for later --rerender use.
        if use_checkpoint:
            save_tree(tree, checkpoint)

        # Compute semantic labels from the tree.
        semantic_labels = label_tree(tree)
        print(f"  Semantic labels: {len(semantic_labels)} nodes labeled")

        # Generate all other outputs.
        _generate_recursive_outputs(
            tree, H, coords, stem, output_dir_path, max_depth,
            semantic_labels, timings,
        )
        print(f"  Per-split figures: {bisection_dir}/")

    total = time.monotonic() - t0
    timings.append(("Total", total))
    _print_timing_and_memory(timings, profile)


def _print_timing_and_memory(timings: list[tuple[str, float]], profile: bool) -> None:
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


def _parse_flag(
    argv: list[str], flag: str, default: float, cast: type = int,
) -> tuple[list[str], float]:
    """Extract a --flag VALUE pair from argv, returning (remaining_argv, value)."""
    if flag not in argv:
        return argv, default
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        print(f"Error: {flag} requires a value")
        sys.exit(1)
    value = cast(argv[idx + 1])
    return argv[:idx] + argv[idx + 2:], value


def _parse_str_flag(
    argv: list[str], flag: str,
) -> tuple[list[str], str | None]:
    """Extract a --flag VALUE string pair from argv."""
    if flag not in argv:
        return argv, None
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        print(f"Error: {flag} requires a value")
        sys.exit(1)
    value = argv[idx + 1]
    return argv[:idx] + argv[idx + 2:], value


if __name__ == "__main__":
    flags = {"--profile", "--streaming", "--light", "--recursive", "--rerender"}
    argv = sys.argv[1:]
    argv, top_n = _parse_flag(argv, "--top-n", 20, int)
    argv, max_depth = _parse_flag(argv, "--max-depth", 4, int)
    argv, min_size = _parse_flag(argv, "--min-size", 200, int)
    argv, connectivity_ratio = _parse_flag(argv, "--connectivity-ratio", 10.0, float)
    argv, checkpoint = _parse_str_flag(argv, "--checkpoint")
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
        rerender="--rerender" in argv,
    )
