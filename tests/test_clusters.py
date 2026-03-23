"""Tests for proofgraph.clusters."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from proofgraph.clusters import (
    analyze_clusters,
    assign_clusters,
    format_cluster_markdown,
    print_cluster_summary,
    write_cluster_outputs,
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
