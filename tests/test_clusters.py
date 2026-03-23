"""Tests for proofgraph.clusters."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from proofgraph.clusters import (
    analyze_clusters,
    assign_clusters,
    collect_connectivity_profile,
    collect_leaf_assignments,
    format_cluster_markdown,
    format_recursive_markdown,
    print_cluster_summary,
    recursive_bisect,
    write_cluster_outputs,
    write_recursive_outputs,
)


def _make_graph() -> nx.Graph:
    """Build a small graph with module and kind attributes."""
    G = nx.Graph()
    decls = [
        ("Mathlib.Order.Basic.le_refl", "theorem", "Mathlib.Order.Basic"),
        ("Mathlib.Order.Basic.le_trans", "theorem", "Mathlib.Order.Basic"),
        ("Mathlib.Order.Basic.PartialOrder", "inductive", "Mathlib.Order.Basic"),
        ("Mathlib.Order.Filter.map", "def", "Mathlib.Order.Filter"),
        ("Mathlib.Algebra.Group.basic_mul", "def", "Mathlib.Algebra.Group"),
        ("Mathlib.Algebra.Group.one_mul", "theorem", "Mathlib.Algebra.Group"),
    ]
    for name, kind, module in decls:
        G.add_node(name, kind=kind, module=module)
    G.add_edge("Mathlib.Order.Basic.le_refl", "Mathlib.Order.Basic.le_trans")
    G.add_edge("Mathlib.Order.Basic.le_trans", "Mathlib.Order.Basic.PartialOrder")
    G.add_edge("Mathlib.Order.Filter.map", "Mathlib.Order.Basic.le_refl")
    G.add_edge("Mathlib.Algebra.Group.basic_mul", "Mathlib.Algebra.Group.one_mul")
    return G


class TestAssignClusters:
    def test_basic_assignment(self) -> None:
        G = nx.path_graph(4)
        fiedler = np.array([0.5, 0.1, -0.1, -0.5])
        assignments = assign_clusters(G, fiedler)
        assert assignments[0] == 0  # positive -> cluster 0
        assert assignments[3] == 1  # negative -> cluster 1

    def test_zero_goes_to_cluster_0(self) -> None:
        G = nx.path_graph(3)
        fiedler = np.array([1.0, 0.0, -1.0])
        assignments = assign_clusters(G, fiedler)
        assert assignments[1] == 0  # zero is non-negative

    def test_all_same_sign(self) -> None:
        G = nx.path_graph(3)
        fiedler = np.array([0.1, 0.5, 0.9])
        assignments = assign_clusters(G, fiedler)
        assert all(v == 0 for v in assignments.values())

    def test_returns_all_nodes(self) -> None:
        G = _make_graph()
        fiedler = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        assignments = assign_clusters(G, fiedler)
        assert set(assignments.keys()) == set(G.nodes())


class TestAnalyzeClusters:
    def test_cluster_counts(self) -> None:
        G = _make_graph()
        nodes = list(G.nodes())
        assignments = {n: 0 if i < 3 else 1 for i, n in enumerate(nodes)}
        results = analyze_clusters(G, assignments)
        assert results[0]["count"] == 3
        assert results[1]["count"] == 3

    def test_fraction_sums_to_one(self) -> None:
        G = _make_graph()
        fiedler = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        assignments = assign_clusters(G, fiedler)
        results = analyze_clusters(G, assignments)
        total_frac = sum(r["fraction"] for r in results.values())
        assert abs(total_frac - 1.0) < 1e-10

    def test_kind_counts_present(self) -> None:
        G = _make_graph()
        fiedler = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        assignments = assign_clusters(G, fiedler)
        results = analyze_clusters(G, assignments)
        for r in results.values():
            assert len(r["kind_counts"]) > 0

    def test_module_counts_at_depths(self) -> None:
        G = _make_graph()
        fiedler = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        assignments = assign_clusters(G, fiedler)
        results = analyze_clusters(G, assignments, module_depths=(2, 3))
        for r in results.values():
            assert 2 in r["module_counts"]
            assert 3 in r["module_counts"]

    def test_declarations_sorted(self) -> None:
        G = _make_graph()
        fiedler = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        assignments = assign_clusters(G, fiedler)
        results = analyze_clusters(G, assignments)
        for r in results.values():
            assert r["declarations"] == sorted(r["declarations"])

    def test_top_n_limits_output(self) -> None:
        G = _make_graph()
        fiedler = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        assignments = assign_clusters(G, fiedler)
        results = analyze_clusters(G, assignments, top_n=1)
        for r in results.values():
            for depth_entries in r["module_counts"].values():
                assert len(depth_entries) <= 1

    def test_handles_missing_kind(self) -> None:
        G = nx.Graph()
        G.add_node("a")
        G.add_node("b")
        G.add_edge("a", "b")
        assignments = {"a": 0, "b": 1}
        results = analyze_clusters(G, assignments)
        assert results[0]["kind_counts"] == [("unknown", 1)]

    def test_arbitrary_cluster_labels(self) -> None:
        """Handles non-sequential integer labels (e.g., from hierarchical bisection)."""
        G = nx.path_graph(4)
        assignments = {0: 5, 1: 5, 2: 10, 3: 10}
        results = analyze_clusters(G, assignments)
        assert 5 in results
        assert 10 in results
        assert results[5]["count"] == 2
        assert results[10]["count"] == 2


class TestFormatClusterMarkdown:
    def test_contains_header(self) -> None:
        G = _make_graph()
        fiedler = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        assignments = assign_clusters(G, fiedler)
        results = analyze_clusters(G, assignments)
        md = format_cluster_markdown(results, 6)
        assert "# Cluster Analysis" in md

    def test_contains_overview_table(self) -> None:
        G = _make_graph()
        fiedler = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        assignments = assign_clusters(G, fiedler)
        results = analyze_clusters(G, assignments)
        md = format_cluster_markdown(results, 6)
        assert "| Cluster | Declarations | Percentage |" in md

    def test_contains_kind_table(self) -> None:
        G = _make_graph()
        fiedler = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        assignments = assign_clusters(G, fiedler)
        results = analyze_clusters(G, assignments)
        md = format_cluster_markdown(results, 6)
        assert "### Declaration kinds" in md

    def test_contains_module_tables(self) -> None:
        G = _make_graph()
        fiedler = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        assignments = assign_clusters(G, fiedler)
        results = analyze_clusters(G, assignments)
        md = format_cluster_markdown(results, 6)
        assert "### Top modules (depth 2)" in md
        assert "### Top modules (depth 3)" in md


class TestWriteClusterOutputs:
    def test_creates_markdown(self, tmp_path: Path) -> None:
        G = _make_graph()
        fiedler = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        assignments = assign_clusters(G, fiedler)
        results = analyze_clusters(G, assignments)
        md_path = write_cluster_outputs(results, tmp_path, 6)
        assert md_path.exists()
        assert md_path.name == "cluster_analysis.md"

    def test_creates_declaration_files(self, tmp_path: Path) -> None:
        G = _make_graph()
        fiedler = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        assignments = assign_clusters(G, fiedler)
        results = analyze_clusters(G, assignments)
        write_cluster_outputs(results, tmp_path, 6)
        assert (tmp_path / "cluster_0_declarations.txt").exists()
        assert (tmp_path / "cluster_1_declarations.txt").exists()

    def test_declaration_file_contents(self, tmp_path: Path) -> None:
        G = _make_graph()
        fiedler = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        assignments = assign_clusters(G, fiedler)
        results = analyze_clusters(G, assignments)
        write_cluster_outputs(results, tmp_path, 6)
        for label in results:
            content = (tmp_path / f"cluster_{label}_declarations.txt").read_text()
            names = [l for l in content.strip().split("\n") if l]
            assert len(names) == results[label]["count"]

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        G = _make_graph()
        fiedler = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        assignments = assign_clusters(G, fiedler)
        results = analyze_clusters(G, assignments)
        nested = tmp_path / "sub" / "dir"
        write_cluster_outputs(results, nested, 6)
        assert (nested / "cluster_analysis.md").exists()

    def test_arbitrary_labels_in_filenames(self, tmp_path: Path) -> None:
        G = nx.path_graph(4)
        assignments = {0: 7, 1: 7, 2: 42, 3: 42}
        results = analyze_clusters(G, assignments)
        write_cluster_outputs(results, tmp_path, 4)
        assert (tmp_path / "cluster_7_declarations.txt").exists()
        assert (tmp_path / "cluster_42_declarations.txt").exists()


class TestPrintClusterSummary:
    def test_does_not_raise(self, capsys) -> None:
        G = _make_graph()
        fiedler = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        assignments = assign_clusters(G, fiedler)
        results = analyze_clusters(G, assignments)
        print_cluster_summary(results, 6)
        captured = capsys.readouterr()
        assert "Cluster Analysis" in captured.out
        assert "Cluster 0" in captured.out
        assert "Cluster 1" in captured.out


# ---------------------------------------------------------------------------
# Recursive bisection tests
# ---------------------------------------------------------------------------


def _barbell_graph() -> nx.Graph:
    """A barbell graph with clear bisection structure and node attributes."""
    G = nx.barbell_graph(10, 1)
    for n in G.nodes():
        module = "Mathlib.Left" if n < 10 else "Mathlib.Right"
        kind = "thm" if n % 2 == 0 else "def"
        G.nodes[n]["module"] = module
        G.nodes[n]["kind"] = kind
    return G


class TestRecursiveBisect:
    def test_returns_tree_structure(self) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        assert "label" in tree
        assert "children" in tree
        assert "algebraic_connectivity" in tree
        assert "node_count" in tree

    def test_root_label(self) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        assert tree["label"] == "root"
        assert tree["depth"] == 0

    def test_stops_at_max_depth(self) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        # Children exist at depth 1, but they should not recurse further.
        for child in tree["children"]:
            assert child["stopped_reason"] == "max_depth"
            assert child["children"] == []

    def test_stops_at_min_size(self) -> None:
        G = nx.path_graph(5)
        tree = recursive_bisect(G, max_depth=10, min_size=100)
        # Root itself is too small to split (5 < 100 at depth>0),
        # but root always tries. After first split, children are < 100.
        for child in tree["children"]:
            assert child["stopped_reason"] == "min_size"

    def test_children_node_counts_sum(self) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        if tree["children"]:
            child_total = sum(c["node_count"] for c in tree["children"])
            assert child_total == tree["node_count"]

    def test_algebraic_connectivity_recorded(self) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        assert tree["algebraic_connectivity"] is not None
        assert tree["algebraic_connectivity"] > 0

    def test_precomputed_fiedler(self) -> None:
        G = _barbell_graph()
        from proofgraph.spectral import fiedler_vector
        f, ac = fiedler_vector(G, normalized=True)
        tree = recursive_bisect(
            G, max_depth=1, min_size=3,
            precomputed_fiedler=(f, ac),
        )
        assert tree["algebraic_connectivity"] == ac
        assert len(tree["children"]) == 2

    def test_depth_2_produces_grandchildren(self) -> None:
        # Chain of barbells: four cliques connected in a line, giving
        # hierarchical bisection structure.
        G = nx.Graph()
        cliques = [range(i * 10, (i + 1) * 10) for i in range(4)]
        for clique in cliques:
            for u in clique:
                for v in clique:
                    if u < v:
                        G.add_edge(u, v)
                G.nodes[u]["kind"] = "thm"
        # Connect adjacent cliques with a single bridge edge.
        for i in range(3):
            G.add_edge(cliques[i][-1], cliques[i + 1][0])
        tree = recursive_bisect(G, max_depth=2, min_size=3, connectivity_ratio=100.0)
        has_grandchildren = any(
            len(child["children"]) > 0 for child in tree["children"]
        )
        assert has_grandchildren

    def test_child_labels_are_hierarchical(self) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        child_labels = {c["label"] for c in tree["children"]}
        assert child_labels == {"0", "1"}

    def test_elapsed_seconds_populated(self) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        assert tree["elapsed_seconds"] > 0

    def test_tiny_graph_stops_immediately(self) -> None:
        G = nx.path_graph(2)
        tree = recursive_bisect(G, max_depth=4, min_size=3)
        assert tree["stopped_reason"] == "min_size"
        assert tree["children"] == []


class TestCollectConnectivityProfile:
    def test_returns_all_nodes(self) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        profile = collect_connectivity_profile(tree)
        # root + 2 children = 3 entries.
        assert len(profile) == 3

    def test_sorted_by_depth_then_label(self) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        profile = collect_connectivity_profile(tree)
        depths = [r["depth"] for r in profile]
        assert depths == sorted(depths)

    def test_root_is_first(self) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        profile = collect_connectivity_profile(tree)
        assert profile[0]["label"] == "root"


class TestFormatRecursiveMarkdown:
    def test_contains_header(self) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        md = format_recursive_markdown(tree)
        assert "# Recursive Spectral Bisection" in md

    def test_contains_connectivity_profile(self) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        md = format_recursive_markdown(tree)
        assert "Spectral Connectivity Profile" in md
        assert "Algebraic Connectivity" in md

    def test_contains_node_details(self) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        md = format_recursive_markdown(tree)
        assert "root" in md
        assert "nodes" in md


class TestWriteRecursiveOutputs:
    def test_creates_markdown(self, tmp_path: Path) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        md_path = write_recursive_outputs(tree, tmp_path)
        assert md_path.exists()
        assert md_path.name == "recursive_bisection.md"

    def test_creates_declaration_files(self, tmp_path: Path) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        write_recursive_outputs(tree, tmp_path)
        assert (tmp_path / "cluster_0_declarations.txt").exists()
        assert (tmp_path / "cluster_1_declarations.txt").exists()

    def test_hierarchical_declaration_files(self, tmp_path: Path) -> None:
        # Chain of four cliques for hierarchical structure.
        G = nx.Graph()
        cliques = [range(i * 10, (i + 1) * 10) for i in range(4)]
        for clique in cliques:
            for u in clique:
                for v in clique:
                    if u < v:
                        G.add_edge(u, v)
                G.nodes[u]["kind"] = "thm"
        for i in range(3):
            G.add_edge(cliques[i][-1], cliques[i + 1][0])
        tree = recursive_bisect(G, max_depth=2, min_size=3, connectivity_ratio=100.0)
        write_recursive_outputs(tree, tmp_path)
        # Root-level files should exist.
        assert (tmp_path / "cluster_0_declarations.txt").exists()
        assert (tmp_path / "cluster_1_declarations.txt").exists()


class TestCollectLeafAssignments:
    def test_depth_1_assigns_all_nodes(self) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        leaf = collect_leaf_assignments(tree)
        assert len(leaf) == G.number_of_nodes()

    def test_depth_1_uses_child_labels(self) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3)
        leaf = collect_leaf_assignments(tree)
        labels = set(leaf.values())
        assert labels == {"0", "1"}

    def test_depth_2_uses_deeper_labels(self) -> None:
        # Chain of four cliques for hierarchical structure.
        G = nx.Graph()
        cliques = [range(i * 10, (i + 1) * 10) for i in range(4)]
        for clique in cliques:
            for u in clique:
                for v in clique:
                    if u < v:
                        G.add_edge(u, v)
                G.nodes[u]["kind"] = "thm"
        for i in range(3):
            G.add_edge(cliques[i][-1], cliques[i + 1][0])
        tree = recursive_bisect(G, max_depth=2, min_size=3, connectivity_ratio=100.0)
        leaf = collect_leaf_assignments(tree)
        assert len(leaf) == G.number_of_nodes()
        # Should have labels deeper than just "0"/"1".
        labels = set(leaf.values())
        assert len(labels) > 2

    def test_unsplit_tree_returns_empty(self) -> None:
        G = nx.path_graph(2)
        tree = recursive_bisect(G, max_depth=4, min_size=3)
        leaf = collect_leaf_assignments(tree)
        assert leaf == {}

    def test_generates_per_split_figures(self, tmp_path: Path) -> None:
        G = _barbell_graph()
        tree = recursive_bisect(G, max_depth=1, min_size=3, plot_dir=tmp_path)
        assert (tmp_path / "bisection_root.png").exists()
