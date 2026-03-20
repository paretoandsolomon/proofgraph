"""Visualization: Fiedler bipartition with spectral embedding layout."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Teal/slate blue palette per project conventions.
TEAL = "#14b8a6"
SLATE_BLUE = "#475569"


def plot_fiedler_bipartition(
    G: nx.Graph,
    fiedler: np.ndarray,
    output_path: str | Path,
    coords: np.ndarray | None = None,
) -> None:
    """Plot the Fiedler bipartition using spectral embedding coordinates.

    Nodes are colored by the sign of their Fiedler vector component:
    teal for positive, slate blue for negative. Positions come from spectral
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
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = list(G.nodes())
    colors = [TEAL if fiedler[i] >= 0 else SLATE_BLUE for i in range(len(nodes))]

    if coords is not None:
        pos = {nodes[i]: (coords[i, 0], coords[i, 1]) for i in range(len(nodes))}
    else:
        pos = {nodes[i]: (fiedler[i], float(i)) for i in range(len(nodes))}

    fig, ax = plt.subplots(figsize=(12, 8))

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.3, edge_color="#94a3b8")
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=colors,
        node_size=15,
        linewidths=0,
    )

    n_positive = sum(1 for v in fiedler if v >= 0)
    n_negative = len(fiedler) - n_positive
    ax.set_title(
        f"Fiedler Bipartition ({n_positive} / {n_negative} nodes)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Spectral coordinate 1", fontsize=10)
    ax.set_ylabel("Spectral coordinate 2", fontsize=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TEAL,
               markersize=8, label=f"Positive ({n_positive})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SLATE_BLUE,
               markersize=8, label=f"Negative ({n_negative})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
