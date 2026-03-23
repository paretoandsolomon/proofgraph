"""Cluster content analysis for spectral partitions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np


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
