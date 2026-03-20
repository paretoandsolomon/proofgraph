"""Visualization: Fiedler bipartition with spectral embedding layout."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np

# Teal/slate blue palette per project conventions.
TEAL = "#14b8a6"
SLATE_BLUE = "#475569"


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

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.3, edge_color="#555e6b")
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

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TEAL,
               markersize=8, label=f"Cluster A ({n_positive} declarations)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SLATE_BLUE,
               markersize=8, label=f"Cluster B ({n_negative} declarations)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
