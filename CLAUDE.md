# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProofGraph applies network science and spectral graph theory to the dependency graph of formalized mathematics (Lean 4 / Mathlib). It extracts declaration-level dependency graphs, computes proof-theoretic properties, runs structural and spectral analysis, and exposes results through APIs and an MCP server. MIT licensed.

## Project Structure

```
proofgraph/
  ProofGraph/           # Lean 4 project (extraction, formalization)
  proofgraph/           # Python package (analysis, API, MCP server)
  data/                 # Generated data artifacts (git-ignored)
  docs/                 # Documentation and schema
```

## Technology Stack

- **Lean 4** for extraction (Environment API, LeanDojo v2 methodology)
- **Python**: FastAPI (REST + GraphQL), NetworkX/igraph, scipy.sparse.linalg, sentence-transformers
- **Graph database**: Memgraph (Cypher queries, vector search)
- **Visualization**: Spectral embedding (never force-directed layout)

## Build and Development

### Lean 4

```bash
cd ProofGraph
lake build           # Build the Lean project
lake env printPaths  # Show Lean environment paths
```

### Python

```bash
cd proofgraph
pip install -e ".[dev]"    # Install in editable mode with dev dependencies
pytest                     # Run all tests
pytest tests/test_foo.py   # Run a single test file
pytest -k "test_name"      # Run a specific test
```

### Memgraph

```bash
docker compose up memgraph  # Start Memgraph instance
```

## Architecture

The system follows a pipeline: **Extraction -> Storage -> Analysis -> API/MCP**.

**Graph schema** - Nodes are `Declaration` objects with properties: name, kind, type_expr, module, slogan, embedding, pagerank, cluster_id, centrality, fiedler_component, spectral_coords, heat_kernel_signature, proof-theoretic flags (is_constructive, is_computable, uses_choice, uses_propext, uses_quot). Edges: DEPENDS_ON, USES_DEF, EXTENDS, INSTANCE_OF, DEFINED_IN, IMPORTS.

**Proof-theoretic properties** extracted per declaration:
1. Noncomputable status (direct flag from ConstantInfo)
2. Transitive axiom usage (Classical.choice, propext, Quot.sound, funext)
3. Constructive status (derived from axiom usage)

**MCP server** exposes: search, get_relevant_premises_for_goal, get_neighborhood, get_graph_features, get_proof_properties, log_proof_attempt.

## Style and Conventions

- **No em-dashes.** Use commas, colons, parentheses, periods, or semicolons instead. This applies to all writing, including code comments, docstrings, and documentation.
- Academic tone in documentation. Technical precision over marketing language.
- Teal/slate blue color palette for any visual output.
- Do not reference private files' contents in committed files.
