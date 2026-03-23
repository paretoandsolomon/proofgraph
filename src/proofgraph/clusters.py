"""Cluster content analysis for spectral partitions."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

from proofgraph.spectral import fiedler_vector, spectral_embedding
from proofgraph.viz import plot_cluster_map, plot_fiedler_bipartition


def assign_clusters(
    G: nx.Graph, fiedler: np.ndarray,
) -> dict[str, int]:
    """Assign nodes to clusters based on the sign of the Fiedler vector.

    Parameters
    ----------
    G : nx.Graph
        Graph whose nodes are ordered consistently with ``fiedler``.
    fiedler : np.ndarray
        Fiedler vector (one entry per node in ``list(G.nodes())``).

    Returns
    -------
    dict[str, int]
        Mapping from node name to cluster label (0 for non-negative, 1 for negative).
    """
    nodes = list(G.nodes())
    return {
        nodes[i]: 0 if fiedler[i] >= 0 else 1
        for i in range(len(nodes))
    }


def _module_prefix(name: str, depth: int) -> str:
    """Extract a module prefix at the given depth from a declaration name.

    For ``Mathlib.Order.Filter.Basic.foo``, depth=2 returns ``Mathlib.Order``.
    """
    parts = str(name).split(".")
    return ".".join(parts[:depth]) if len(parts) >= depth else name


def _group_by_module(
    names: list[str], depth: int,
) -> Counter[str]:
    """Count declarations by module prefix at the given depth."""
    return Counter(_module_prefix(n, depth) for n in names)


def analyze_clusters(
    G: nx.Graph,
    assignments: dict[str, int],
    top_n: int = 20,
    module_depths: tuple[int, ...] = (2, 3),
) -> dict[int, dict]:
    """Analyze the content of each cluster.

    Parameters
    ----------
    G : nx.Graph
        The graph with node attributes (``module``, ``kind``, etc.).
    assignments : dict[str, int]
        Mapping from node name to cluster label.
    top_n : int
        Number of top modules to include in the summary.
    module_depths : tuple[int, ...]
        Module path depths at which to group declarations. For example,
        depth 2 groups by ``Mathlib.Order``, depth 3 by ``Mathlib.Order.Filter``.

    Returns
    -------
    dict[int, dict]
        Per-cluster analysis with keys: ``label``, ``count``, ``fraction``,
        ``module_counts`` (keyed by depth), ``kind_counts``, ``declarations``.
    """
    total = len(assignments)
    clusters: dict[int, list[str]] = {}
    for node, label in assignments.items():
        clusters.setdefault(label, []).append(node)

    results: dict[int, dict] = {}
    for label in sorted(clusters):
        members = sorted(clusters[label])
        count = len(members)

        # Module breakdown at each depth.
        module_counts: dict[int, list[tuple[str, int]]] = {}
        for depth in module_depths:
            counter = _group_by_module(members, depth)
            module_counts[depth] = counter.most_common(top_n)

        # Declaration kind breakdown.
        kind_counter: Counter[str] = Counter()
        for name in members:
            kind = G.nodes[name].get("kind", "unknown")
            kind_counter[kind] += 1

        results[label] = {
            "label": label,
            "count": count,
            "fraction": count / total if total > 0 else 0.0,
            "module_counts": module_counts,
            "kind_counts": kind_counter.most_common(),
            "declarations": members,
        }

    return results


def format_cluster_markdown(
    results: dict[int, dict],
    total: int,
    module_depths: tuple[int, ...] = (2, 3),
) -> str:
    """Format cluster analysis results as a Markdown report.

    Parameters
    ----------
    results : dict[int, dict]
        Output of :func:`analyze_clusters`.
    total : int
        Total number of declarations across all clusters.
    module_depths : tuple[int, ...]
        Module depths included in the analysis.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines: list[str] = []
    lines.append("# Cluster Analysis")
    lines.append("")
    lines.append(f"Total declarations: {total}")
    lines.append(f"Number of clusters: {len(results)}")
    lines.append("")

    # Overview table.
    lines.append("## Overview")
    lines.append("")
    lines.append("| Cluster | Declarations | Percentage |")
    lines.append("|---------|-------------|------------|")
    for label in sorted(results):
        r = results[label]
        lines.append(f"| {label} | {r['count']:,} | {r['fraction']:.1%} |")
    lines.append("")

    # Per-cluster detail.
    for label in sorted(results):
        r = results[label]
        lines.append(f"## Cluster {label}")
        lines.append("")
        lines.append(f"**{r['count']:,} declarations** ({r['fraction']:.1%} of total)")
        lines.append("")

        # Kind breakdown.
        if r["kind_counts"]:
            lines.append("### Declaration kinds")
            lines.append("")
            lines.append("| Kind | Count | Percentage |")
            lines.append("|------|-------|------------|")
            for kind, count in r["kind_counts"]:
                pct = count / r["count"] if r["count"] > 0 else 0.0
                lines.append(f"| {kind} | {count:,} | {pct:.1%} |")
            lines.append("")

        # Module breakdowns at each depth.
        for depth in module_depths:
            if depth not in r["module_counts"]:
                continue
            entries = r["module_counts"][depth]
            if not entries:
                continue
            lines.append(f"### Top modules (depth {depth})")
            lines.append("")
            lines.append("| Module | Count | Percentage |")
            lines.append("|--------|-------|------------|")
            for module, count in entries:
                pct = count / r["count"] if r["count"] > 0 else 0.0
                lines.append(f"| {module} | {count:,} | {pct:.1%} |")
            lines.append("")

    return "\n".join(lines)


def write_cluster_outputs(
    results: dict[int, dict],
    output_dir: str | Path,
    total: int,
    module_depths: tuple[int, ...] = (2, 3),
) -> Path:
    """Write cluster analysis outputs to disk.

    Writes:
    - ``cluster_analysis.md``: summary report.
    - ``cluster_<label>_declarations.txt``: one declaration name per line, per cluster.

    Parameters
    ----------
    results : dict[int, dict]
        Output of :func:`analyze_clusters`.
    output_dir : str or Path
        Directory for output files.
    total : int
        Total number of declarations across all clusters.
    module_depths : tuple[int, ...]
        Module depths included in the analysis.

    Returns
    -------
    Path
        Path to the written markdown report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Markdown report.
    md = format_cluster_markdown(results, total, module_depths=module_depths)
    md_path = output_dir / "cluster_analysis.md"
    md_path.write_text(md)

    # Declaration listings.
    for label in sorted(results):
        decl_path = output_dir / f"cluster_{label}_declarations.txt"
        decl_path.write_text("\n".join(str(d) for d in results[label]["declarations"]) + "\n")

    return md_path


def print_cluster_summary(
    results: dict[int, dict],
    total: int,
    top_n: int = 20,
    module_depths: tuple[int, ...] = (2, 3),
) -> None:
    """Print a concise cluster summary to stdout.

    Parameters
    ----------
    results : dict[int, dict]
        Output of :func:`analyze_clusters`.
    total : int
        Total number of declarations across all clusters.
    top_n : int
        Number of top modules to display.
    module_depths : tuple[int, ...]
        Module depths included in the analysis.
    """
    print(f"\n{'='*60}")
    print(f"  Cluster Analysis ({total:,} declarations, {len(results)} clusters)")
    print(f"{'='*60}")

    for label in sorted(results):
        r = results[label]
        print(f"\n--- Cluster {label}: {r['count']:,} declarations ({r['fraction']:.1%}) ---")

        if r["kind_counts"]:
            print("\n  Declaration kinds:")
            for kind, count in r["kind_counts"]:
                pct = count / r["count"] if r["count"] > 0 else 0.0
                print(f"    {kind:<20s} {count:>7,}  ({pct:.1%})")

        for depth in module_depths:
            if depth not in r["module_counts"]:
                continue
            entries = r["module_counts"][depth][:top_n]
            if not entries:
                continue
            print(f"\n  Top modules (depth {depth}):")
            for module, count in entries:
                pct = count / r["count"] if r["count"] > 0 else 0.0
                print(f"    {module:<40s} {count:>7,}  ({pct:.1%})")

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Recursive spectral bisection
# ---------------------------------------------------------------------------


def _should_stop(
    node_count: int,
    algebraic_connectivity: float,
    parent_connectivity: float | None,
    depth: int,
    max_depth: int,
    min_size: int,
    connectivity_ratio: float,
) -> str | None:
    """Return a reason string if recursion should stop, else None."""
    if depth >= max_depth:
        return "max_depth"
    if node_count < min_size:
        return "min_size"
    if (
        parent_connectivity is not None
        and parent_connectivity > 0
        and algebraic_connectivity / parent_connectivity >= connectivity_ratio
    ):
        return "connectivity_ratio"
    return None


def recursive_bisect(
    G: nx.Graph,
    max_depth: int = 4,
    min_size: int = 200,
    connectivity_ratio: float = 10.0,
    top_n: int = 20,
    module_depths: tuple[int, ...] = (2, 3),
    normalized: bool = True,
    label: str = "root",
    depth: int = 0,
    parent_connectivity: float | None = None,
    precomputed_fiedler: tuple[np.ndarray, float] | None = None,
    plot_dir: str | Path | None = None,
    precomputed_coords: np.ndarray | None = None,
) -> dict:
    """Recursively bisect a graph using spectral partitioning.

    At each level, computes the Fiedler vector, splits by sign, analyzes
    cluster contents, and recurses into each child subgraph. Recursion
    stops when any stopping criterion is met.

    Parameters
    ----------
    G : nx.Graph
        Connected undirected graph to bisect.
    max_depth : int
        Maximum recursion depth (root is depth 0).
    min_size : int
        Minimum cluster size below which recursion stops.
    connectivity_ratio : float
        If a subgraph's algebraic connectivity divided by its parent's
        exceeds this ratio, the subgraph is considered well-connected
        and recursion stops. Higher values are more permissive.
    top_n : int
        Number of top modules per cluster in the analysis.
    module_depths : tuple[int, ...]
        Module path depths for grouping.
    normalized : bool
        Whether to use the normalized Laplacian.
    label : str
        Label for this node in the bisection tree.
    depth : int
        Current recursion depth.
    parent_connectivity : float or None
        Algebraic connectivity of the parent graph (for ratio stopping).
    precomputed_fiedler : tuple[np.ndarray, float] or None
        If provided, skip Fiedler computation and use this (fiedler_vec,
        algebraic_connectivity) pair. Useful for the root level when
        the Fiedler vector was already computed by the main pipeline.
    plot_dir : str, Path, or None
        If provided, generate a bipartition figure for each split in this
        directory. Figures are named by their tree label.
    precomputed_coords : np.ndarray or None
        If provided, use these spectral embedding coordinates for the
        figure at this level instead of recomputing. Shape (n_nodes, 2).

    Returns
    -------
    dict
        Bisection tree node with keys: ``label``, ``depth``,
        ``node_count``, ``edge_count``, ``algebraic_connectivity``,
        ``stopped_reason``, ``elapsed_seconds``, ``analysis``, ``children``.
    """
    t_start = time.monotonic()
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()

    result: dict = {
        "label": label,
        "depth": depth,
        "node_count": node_count,
        "edge_count": edge_count,
        "algebraic_connectivity": None,
        "stopped_reason": None,
        "elapsed_seconds": 0.0,
        "analysis": None,
        "children": [],
    }

    # Leaf: too small to bisect.
    if node_count < 3:
        result["stopped_reason"] = "min_size"
        result["elapsed_seconds"] = time.monotonic() - t_start
        return result

    # Compute or reuse Fiedler vector.
    if precomputed_fiedler is not None:
        fiedler_vec, ac = precomputed_fiedler
    else:
        print(f"  {'  ' * depth}[{label}] Computing Fiedler ({node_count:,} nodes)...")
        fiedler_vec, ac = fiedler_vector(G, normalized=normalized)

    result["algebraic_connectivity"] = ac

    # Check stopping criteria.
    reason = _should_stop(
        node_count, ac, parent_connectivity,
        depth, max_depth, min_size, connectivity_ratio,
    )
    if reason is not None:
        result["stopped_reason"] = reason
        result["elapsed_seconds"] = time.monotonic() - t_start
        print(
            f"  {'  ' * depth}[{label}] Stopped: {reason} "
            f"(n={node_count:,}, ac={ac:.6f})"
        )
        return result

    # Bisect.
    assignments = assign_clusters(G, fiedler_vec)
    analysis = analyze_clusters(G, assignments, top_n=top_n, module_depths=module_depths)
    result["analysis"] = analysis

    # Partition nodes.
    cluster_nodes: dict[int, list] = {}
    for node, cl in assignments.items():
        cluster_nodes.setdefault(cl, []).append(node)

    n_clusters = len(cluster_nodes)
    print(
        f"  {'  ' * depth}[{label}] Split into {n_clusters} clusters: "
        + ", ".join(f"{len(v):,}" for v in cluster_nodes.values())
        + f" (ac={ac:.6f})"
    )

    # Generate bipartition figure for this split.
    if plot_dir is not None:
        _plot_dir = Path(plot_dir)
        _plot_dir.mkdir(parents=True, exist_ok=True)
        safe_label = label.replace(".", "_")
        fig_path = _plot_dir / f"bisection_{safe_label}.png"

        if precomputed_coords is not None:
            coords = precomputed_coords
        elif node_count >= 3:
            print(f"  {'  ' * depth}[{label}] Computing embedding for figure...")
            coords = spectral_embedding(G, k=2, normalized=normalized)
        else:
            coords = None

        fig_title = (
            f"Bisection: {label} ({node_count:,} declarations, ac={ac:.6f})"
        )
        plot_fiedler_bipartition(
            G, fiedler_vec, fig_path,
            coords=coords,
            title=fig_title,
            algebraic_connectivity=ac,
        )
        print(f"  {'  ' * depth}[{label}] Figure: {fig_path}")

    # Recurse into each child.
    for cl_label in sorted(cluster_nodes):
        members = cluster_nodes[cl_label]
        child_label = f"{label}.{cl_label}" if label != "root" else str(cl_label)
        sub = G.subgraph(members).copy()
        child = recursive_bisect(
            sub,
            max_depth=max_depth,
            min_size=min_size,
            connectivity_ratio=connectivity_ratio,
            top_n=top_n,
            module_depths=module_depths,
            normalized=normalized,
            label=child_label,
            depth=depth + 1,
            parent_connectivity=ac,
            plot_dir=plot_dir,
        )
        result["children"].append(child)
        del sub

    result["elapsed_seconds"] = time.monotonic() - t_start
    return result


def collect_connectivity_profile(tree: dict) -> list[dict]:
    """Extract the algebraic connectivity at every split point in the tree.

    Returns a flat list of records sorted by depth then label, suitable
    for tabular display or plotting.

    Parameters
    ----------
    tree : dict
        Output of :func:`recursive_bisect`.

    Returns
    -------
    list[dict]
        Each dict has keys: ``label``, ``depth``, ``node_count``,
        ``algebraic_connectivity``, ``stopped_reason``.
    """
    records: list[dict] = []
    _collect_profile_recursive(tree, records)
    records.sort(key=lambda r: (r["depth"], r["label"]))
    return records


def _collect_profile_recursive(node: dict, records: list[dict]) -> None:
    records.append({
        "label": node["label"],
        "depth": node["depth"],
        "node_count": node["node_count"],
        "algebraic_connectivity": node["algebraic_connectivity"],
        "stopped_reason": node["stopped_reason"],
    })
    for child in node.get("children", []):
        _collect_profile_recursive(child, records)


def collect_leaf_assignments(tree: dict) -> dict[str, str]:
    """Map every declaration to its leaf cluster label in the bisection tree.

    Walks the tree top-down. At each internal node, the ``analysis`` dict
    records which declarations belong to each child partition. Declarations
    are assigned the label of the deepest (leaf) node they reach.

    Parameters
    ----------
    tree : dict
        Output of :func:`recursive_bisect`.

    Returns
    -------
    dict[str, str]
        Mapping from declaration name to leaf cluster label.
    """
    assignments: dict[str, str] = {}
    if tree["analysis"] is None and not tree["children"]:
        # Single unsplit node: nothing to assign.
        return assignments
    _assign_from_node(tree, assignments)
    return assignments


def _assign_from_node(node: dict, assignments: dict[str, str]) -> None:
    """Walk the tree, passing each child's declaration list down."""
    if node["analysis"] is None:
        return

    # Build a map from child label to child tree node.
    child_by_label: dict[str, dict] = {c["label"]: c for c in node["children"]}

    for cl_int, cl_data in node["analysis"].items():
        child_label = (
            f"{node['label']}.{cl_int}" if node["label"] != "root" else str(cl_int)
        )
        child_node = child_by_label.get(child_label)

        if child_node is not None and child_node["children"]:
            # Child was split further; recurse into it.
            _assign_from_node(child_node, assignments)
        else:
            # Child is a leaf; assign all its declarations this label.
            for decl in cl_data["declarations"]:
                assignments[str(decl)] = child_label


def format_recursive_markdown(tree: dict, module_depths: tuple[int, ...] = (2, 3)) -> str:
    """Format a recursive bisection tree as a Markdown report.

    Parameters
    ----------
    tree : dict
        Output of :func:`recursive_bisect`.
    module_depths : tuple[int, ...]
        Module depths included in the analysis.

    Returns
    -------
    str
        Markdown-formatted hierarchical report.
    """
    lines: list[str] = []
    lines.append("# Recursive Spectral Bisection")
    lines.append("")

    # Connectivity profile table.
    profile = collect_connectivity_profile(tree)
    lines.append("## Spectral Connectivity Profile")
    lines.append("")
    lines.append("| Label | Depth | Nodes | Algebraic Connectivity | Status |")
    lines.append("|-------|-------|-------|----------------------|--------|")
    for r in profile:
        ac_str = f"{r['algebraic_connectivity']:.6f}" if r["algebraic_connectivity"] is not None else "n/a"
        status = r["stopped_reason"] or "split"
        lines.append(f"| {r['label']} | {r['depth']} | {r['node_count']:,} | {ac_str} | {status} |")
    lines.append("")

    # Per-node detail.
    _format_tree_node(tree, lines, module_depths)

    return "\n".join(lines)


def _format_tree_node(
    node: dict, lines: list[str], module_depths: tuple[int, ...],
) -> None:
    heading_depth = min(node["depth"] + 2, 6)  # ## for root, ### for depth 1, etc.
    prefix = "#" * heading_depth
    lines.append(f"{prefix} {node['label']}")
    lines.append("")
    ac_str = f"{node['algebraic_connectivity']:.6f}" if node["algebraic_connectivity"] is not None else "n/a"
    lines.append(f"**{node['node_count']:,} nodes**, {node['edge_count']:,} edges")
    lines.append(f"Algebraic connectivity: {ac_str}")
    if node["stopped_reason"]:
        lines.append(f"Status: stopped ({node['stopped_reason']})")
    else:
        lines.append(f"Status: split into {len(node['children'])} children")
    lines.append(f"Time: {node['elapsed_seconds']:.1f}s")
    lines.append("")

    # Show analysis if this node was split (has cluster analysis).
    if node["analysis"]:
        for cl_label in sorted(node["analysis"]):
            r = node["analysis"][cl_label]
            lines.append(f"**Child {cl_label}**: {r['count']:,} declarations ({r['fraction']:.1%})")
            lines.append("")
            if r["kind_counts"]:
                lines.append("| Kind | Count | % |")
                lines.append("|------|-------|---|")
                for kind, count in r["kind_counts"]:
                    pct = count / r["count"] if r["count"] > 0 else 0.0
                    lines.append(f"| {kind} | {count:,} | {pct:.1%} |")
                lines.append("")
            for d in module_depths:
                if d not in r["module_counts"]:
                    continue
                entries = r["module_counts"][d]
                if not entries:
                    continue
                lines.append(f"Top modules (depth {d}):")
                lines.append("")
                lines.append("| Module | Count | % |")
                lines.append("|--------|-------|---|")
                for module, count in entries:
                    pct = count / r["count"] if r["count"] > 0 else 0.0
                    lines.append(f"| {module} | {count:,} | {pct:.1%} |")
                lines.append("")

    for child in node.get("children", []):
        _format_tree_node(child, lines, module_depths)


def write_recursive_outputs(
    tree: dict,
    output_dir: str | Path,
    module_depths: tuple[int, ...] = (2, 3),
) -> Path:
    """Write recursive bisection outputs to disk.

    Writes:
    - ``recursive_bisection.md``: hierarchical report.
    - ``cluster_<label>_declarations.txt``: per leaf and split node.

    Parameters
    ----------
    tree : dict
        Output of :func:`recursive_bisect`.
    output_dir : str or Path
        Directory for output files.
    module_depths : tuple[int, ...]
        Module depths included in the analysis.

    Returns
    -------
    Path
        Path to the written markdown report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md = format_recursive_markdown(tree, module_depths=module_depths)
    md_path = output_dir / "recursive_bisection.md"
    md_path.write_text(md)

    _write_declaration_files(tree, output_dir)

    return md_path


def _collect_leaf_nodes(node: dict, leaves: list[dict]) -> None:
    if not node["children"]:
        leaves.append(node)
    else:
        for child in node["children"]:
            _collect_leaf_nodes(child, leaves)


def _write_declaration_files(node: dict, output_dir: Path) -> None:
    """Write declaration listing files for nodes that have analysis data."""
    if node["analysis"]:
        for cl_label in sorted(node["analysis"]):
            r = node["analysis"][cl_label]
            # Use the child label for the filename.
            child_label = f"{node['label']}.{cl_label}" if node["label"] != "root" else str(cl_label)
            safe_label = child_label.replace(".", "_")
            decl_path = output_dir / f"cluster_{safe_label}_declarations.txt"
            decl_path.write_text(
                "\n".join(str(d) for d in r["declarations"]) + "\n"
            )
    for child in node.get("children", []):
        _write_declaration_files(child, output_dir)
