"""Visualization: Fiedler bipartition with spectral embedding layout."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np

# Teal/slate blue palette per project conventions.
TEAL = "#14b8a6"
SLATE_BLUE = "#475569"

# Extended palette for multi-cluster visualizations. Starts with the project
# teal/slate pair, then adds distinguishable cool-tone colors.
CLUSTER_PALETTE = [
    "#14b8a6",  # teal
    "#475569",  # slate blue
    "#6366f1",  # indigo
    "#f59e0b",  # amber
    "#ef4444",  # red
    "#8b5cf6",  # violet
    "#06b6d4",  # cyan
    "#84cc16",  # lime
    "#ec4899",  # pink
    "#f97316",  # orange
    "#10b981",  # emerald
    "#3b82f6",  # blue
]


def cluster_color(label: int | str, n_clusters: int) -> str:
    """Return a color for a cluster label.

    Uses the fixed palette for up to 12 clusters, then falls back to
    a continuous colormap for more.
    """
    if isinstance(label, int) and n_clusters <= len(CLUSTER_PALETTE):
        return CLUSTER_PALETTE[label % len(CLUSTER_PALETTE)]
    # Fall back to a perceptually uniform colormap.
    cmap = plt.get_cmap("tab20")
    idx = label if isinstance(label, int) else hash(label)
    return to_hex(cmap(idx % 20 / 20))


def log_scale_coords(v: np.ndarray, compression: float = 1000) -> np.ndarray:
    """Apply monotonic log scaling to spread dense spectral coordinates.

    Parameters
    ----------
    v : np.ndarray
        Raw spectral coordinate values.
    compression : float
        Controls the compression curve. Higher values spread small values more.

    Returns
    -------
    np.ndarray
        Log-scaled coordinates preserving sign and relative ordering.
    """
    return np.sign(v) * np.log1p(np.abs(v) * compression) / np.log1p(compression)


def plot_fiedler_bipartition(
    G: nx.Graph,
    fiedler: np.ndarray,
    output_path: str | Path,
    coords: np.ndarray | None = None,
    title: str | None = None,
    algebraic_connectivity: float | None = None,
    max_edge_artists: int = 20_000,
    cluster_labels: tuple[str, str] | None = None,
    annotate: bool = False,
) -> None:
    """Plot the Fiedler bipartition using spectral embedding coordinates.

    Nodes are colored by the sign of their Fiedler vector component:
    teal for Cluster A, slate blue for Cluster B. Positions come from spectral
    embedding (never force-directed layout).

    Parameters
    ----------
    G : nx.Graph
        The undirected connected graph.
    fiedler : np.ndarray
        Fiedler vector with one component per node in ``list(G.nodes())``.
    output_path : str or Path
        Where to save the figure (e.g., ``figures/fiedler_bipartition.png``).
    coords : np.ndarray or None
        Spectral embedding coordinates, shape (n_nodes, 2). If None, uses
        the Fiedler vector as x-axis and node index as y-axis.
    title : str or None
        Custom title. If None, uses a default describing the bipartition.
    algebraic_connectivity : float or None
        If provided, displayed in the caption.
    max_edge_artists : int
        When the graph has more edges than this threshold, edges are rendered
        using a single ``LineCollection`` instead of individual matplotlib
        artists. This avoids performance issues on large graphs. Default 20,000.
    cluster_labels : tuple[str, str] or None
        Semantic labels for (positive, negative) clusters. If None, uses
        "Cluster A" and "Cluster B".
    annotate : bool
        If True, add text annotations near each cluster's centroid.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = list(G.nodes())
    colors = [TEAL if fiedler[i] >= 0 else SLATE_BLUE for i in range(len(nodes))]

    if coords is not None:
        scaled = np.column_stack(
            [log_scale_coords(coords[:, j]) for j in range(coords.shape[1])]
        )
        pos = {nodes[i]: (scaled[i, 0], scaled[i, 1]) for i in range(len(nodes))}
    else:
        pos = {nodes[i]: (fiedler[i], float(i)) for i in range(len(nodes))}

    # Scale node size by degree so hub structure is visible.
    degrees = np.array([G.degree(n) for n in nodes], dtype=float)
    node_sizes = 5 + 40 * (degrees / max(degrees.max(), 1))

    fig, ax = plt.subplots(figsize=(12, 8))

    if G.number_of_edges() > max_edge_artists:
        segments = [(pos[u], pos[v]) for u, v in G.edges()]
        lc = LineCollection(
            segments, colors="#555e6b", linewidths=0.3, alpha=0.15,
        )
        ax.add_collection(lc)
    else:
        nx.draw_networkx_edges(
            G, pos, ax=ax, alpha=0.15, width=0.3, edge_color="#555e6b",
        )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=colors,
        node_size=node_sizes,
        linewidths=0,
    )

    n_positive = sum(1 for v in fiedler if v >= 0)
    n_negative = len(fiedler) - n_positive

    if title is None:
        title = f"Spectral Bipartition of Dependency Graph ({G.number_of_nodes()} declarations)"
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.set_xlabel("Fiedler vector (primary spectral axis)", fontsize=10)
    ax.set_ylabel("Third Laplacian eigenvector (secondary spectral axis)", fontsize=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Caption
    caption_lines = [
        "Declarations positioned by the two smallest non-trivial eigenvectors of the graph Laplacian.",
        "Color indicates the spectral bipartition: the sign of the Fiedler vector splits the graph",
        "into two clusters that approximate the sparsest cut of the dependency structure.",
    ]
    if algebraic_connectivity is not None:
        caption_lines.append(f"Algebraic connectivity: {algebraic_connectivity:.4f}")
    caption = "\n".join(caption_lines)
    ax.annotate(
        caption,
        xy=(0.5, 0),
        xycoords="axes fraction",
        xytext=(0, -28),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8,
        color="#64748b",
        style="italic",
    )

    pos_label = cluster_labels[0] if cluster_labels else "Cluster A"
    neg_label = cluster_labels[1] if cluster_labels else "Cluster B"
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TEAL,
               markersize=8, label=f"{pos_label} ({n_positive:,})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SLATE_BLUE,
               markersize=8, label=f"{neg_label} ({n_negative:,})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # In-situ annotations near each cluster's centroid.
    if annotate and coords is not None:
        for sign_val, label_text, count, color in [
            (1, pos_label, n_positive, TEAL),
            (-1, neg_label, n_negative, SLATE_BLUE),
        ]:
            member_indices = [
                i for i in range(len(nodes))
                if (fiedler[i] >= 0) == (sign_val == 1)
            ]
            if not member_indices:
                continue
            cx = np.mean([pos[nodes[i]][0] for i in member_indices])
            cy = np.mean([pos[nodes[i]][1] for i in member_indices])
            ax.annotate(
                f"{label_text}\n({count:,})",
                xy=(cx, cy), fontsize=8, fontweight="bold",
                ha="center", va="center",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white",
                    edgecolor="#cccccc", alpha=0.85,
                ),
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_map(
    G: nx.Graph,
    assignments: dict[str, int | str],
    output_path: str | Path,
    coords: np.ndarray | None = None,
    title: str | None = None,
    caption: str | None = None,
    max_edge_artists: int = 20_000,
    label_names: dict[int | str, str] | None = None,
    annotate: bool = False,
) -> None:
    """Plot a graph colored by arbitrary cluster assignments.

    Works for any number of clusters: 2 (bipartition), 4 (depth-2 bisection),
    or more. Each cluster gets a distinct color from the project palette.

    Parameters
    ----------
    G : nx.Graph
        The undirected graph.
    assignments : dict[str, int | str]
        Mapping from node name to cluster label.
    output_path : str or Path
        Where to save the figure.
    coords : np.ndarray or None
        Spectral embedding coordinates, shape (n_nodes, 2). If None, uses
        node index as position (not recommended).
    title : str or None
        Figure title.
    caption : str or None
        Caption text below the figure.
    max_edge_artists : int
        Threshold for switching to LineCollection for edges.
    label_names : dict or None
        Mapping from cluster label to semantic display name for legends
        and annotations. If None, uses the numeric label.
    annotate : bool
        If True, add text annotations near each cluster's centroid.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = list(G.nodes())

    # Determine unique labels and assign colors.
    unique_labels = sorted(set(assignments.get(n, -1) for n in nodes), key=str)
    n_clusters = len(unique_labels)
    label_to_color = {
        lbl: cluster_color(i, n_clusters) for i, lbl in enumerate(unique_labels)
    }

    # Count per cluster.
    label_counts: dict[int | str, int] = {}
    for n in nodes:
        lbl = assignments.get(n, -1)
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    colors = [label_to_color[assignments.get(n, -1)] for n in nodes]

    if coords is not None:
        scaled = np.column_stack(
            [log_scale_coords(coords[:, j]) for j in range(coords.shape[1])]
        )
        pos = {nodes[i]: (scaled[i, 0], scaled[i, 1]) for i in range(len(nodes))}
    else:
        pos = {nodes[i]: (float(i), 0.0) for i in range(len(nodes))}

    degrees = np.array([G.degree(n) for n in nodes], dtype=float)
    node_sizes = 5 + 40 * (degrees / max(degrees.max(), 1))

    fig, ax = plt.subplots(figsize=(12, 8))

    if G.number_of_edges() > max_edge_artists:
        segments = [(pos[u], pos[v]) for u, v in G.edges()]
        lc = LineCollection(
            segments, colors="#555e6b", linewidths=0.3, alpha=0.10,
        )
        ax.add_collection(lc)
    else:
        nx.draw_networkx_edges(
            G, pos, ax=ax, alpha=0.15, width=0.3, edge_color="#555e6b",
        )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors, node_size=node_sizes, linewidths=0,
    )

    if title is None:
        title = f"Spectral Cluster Map ({G.number_of_nodes()} declarations, {n_clusters} clusters)"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Fiedler vector (primary spectral axis)", fontsize=10)
    ax.set_ylabel("Third Laplacian eigenvector (secondary spectral axis)", fontsize=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    if caption:
        ax.annotate(
            caption,
            xy=(0.5, 0), xycoords="axes fraction",
            xytext=(0, -28), textcoords="offset points",
            ha="center", va="top", fontsize=8, color="#64748b", style="italic",
        )

    # Legend: show up to 16 clusters, then truncate.
    _names = label_names or {}
    max_legend = 16
    legend_elements = []
    for lbl in unique_labels[:max_legend]:
        count = label_counts.get(lbl, 0)
        display = _names.get(lbl, str(lbl))
        legend_elements.append(
            Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=label_to_color[lbl], markersize=8,
                label=f"{display} ({count:,})",
            )
        )
    if n_clusters > max_legend:
        legend_elements.append(
            Line2D([0], [0], marker="", color="w", label=f"... +{n_clusters - max_legend} more")
        )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # In-situ annotations near each cluster's centroid.
    if annotate and coords is not None:
        for lbl in unique_labels:
            member_indices = [i for i, n in enumerate(nodes) if assignments.get(n, -1) == lbl]
            if not member_indices:
                continue
            cx = np.mean([pos[nodes[i]][0] for i in member_indices])
            cy = np.mean([pos[nodes[i]][1] for i in member_indices])
            display = _names.get(lbl, str(lbl))
            count = label_counts.get(lbl, 0)
            ax.annotate(
                f"{display}\n({count:,})",
                xy=(cx, cy), fontsize=7, fontweight="bold",
                ha="center", va="center",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white",
                    edgecolor="#cccccc", alpha=0.85,
                ),
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dendrogram (hierarchy overview)
# ---------------------------------------------------------------------------

_LEAF_COLORS = CLUSTER_PALETTE
_INTERNAL_COLOR = "#f0f0f0"
_COHESIVE_COLOR = "#c8e6c9"


def _stop_icon(reason: str | None) -> str:
    if reason is None:
        return "split"
    return {
        "connectivity_ratio": "cohesive",
        "min_size": "small",
        "max_depth": "depth limit",
    }.get(reason, reason)


def render_dendrogram(
    tree: dict,
    output_path: str | Path,
    semantic_labels: dict[str, str] | None = None,
    fmt: str = "png",
) -> Path:
    """Render a hierarchy dendrogram from a recursive bisection tree.

    Produces a Graphviz DOT file and renders it to PNG and/or SVG.

    Parameters
    ----------
    tree : dict
        Output of ``recursive_bisect``.
    output_path : str or Path
        Base output path (without extension). The DOT source, PNG, and SVG
        are written alongside it.
    semantic_labels : dict[str, str] or None
        Mapping from tree node label to semantic name.
    fmt : str
        Render format: "png", "svg", or "pdf".

    Returns
    -------
    Path
        Path to the rendered file.
    """
    import graphviz

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sem = semantic_labels or {}
    dot = graphviz.Digraph(
        "recursive_bisection",
        format=fmt,
        graph_attr={
            "rankdir": "TB",
            "nodesep": "0.4",
            "ranksep": "0.6",
            "bgcolor": "white",
        },
        node_attr={
            "shape": "record",
            "style": "rounded,filled",
            "fontname": "Helvetica",
            "fontsize": "10",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "8",
        },
    )

    leaf_counter = [0]

    def _add_node(node: dict) -> str:
        node_id = node["label"].replace(".", "_")
        display = sem.get(node["label"], node["label"])
        ac_str = (
            f"ac={node['algebraic_connectivity']:.6f}"
            if node["algebraic_connectivity"] is not None
            else ""
        )
        status = _stop_icon(node["stopped_reason"])
        label_parts = [display, f"{node['node_count']:,} declarations"]
        if ac_str:
            label_parts.append(ac_str)
        label_parts.append(status)
        record_label = "{" + "|".join(label_parts) + "}"

        # Color: leaf nodes get palette colors, cohesive nodes get green,
        # internal nodes get gray.
        if not node["children"]:
            if node.get("stopped_reason") == "connectivity_ratio":
                fill = _COHESIVE_COLOR
            else:
                fill = _LEAF_COLORS[leaf_counter[0] % len(_LEAF_COLORS)]
                leaf_counter[0] += 1
        else:
            fill = _INTERNAL_COLOR

        dot.node(node_id, label=record_label, fillcolor=fill)

        for child in node.get("children", []):
            child_id = _add_node(child)
            if node["node_count"] > 0:
                pct = child["node_count"] / node["node_count"]
                dot.edge(node_id, child_id, label=f"{pct:.1%}")
            else:
                dot.edge(node_id, child_id)

        return node_id

    _add_node(tree)

    rendered = dot.render(
        filename=output_path.stem,
        directory=str(output_path.parent),
        cleanup=False,
    )
    return Path(rendered)
