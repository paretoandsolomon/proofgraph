# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProofGraph applies network science and spectral graph theory to the dependency graph of formalized mathematics (Lean 4 / Mathlib). It extracts declaration-level dependency graphs, computes proof-theoretic properties, runs structural and spectral analysis, and exposes results through a Go API server (REST + GraphQL), with a CLI for human and agent interaction. MIT licensed.

## Project Structure

```
proofgraph/
  ProofGraph/           # Lean 4 project (extraction, formalization)
  proofgraph/           # Python package (analysis)
  data/                 # Generated data artifacts (git-ignored)
  docs/                 # Documentation and schema
```

## Technology Stack

- **Lean 4** for extraction (Environment API, LeanDojo v2 methodology)
- **Go** API server (REST + GraphQL)
- **CLI**: Wraps REST/GraphQL APIs for human and agent interaction
- **Python**: NetworkX/igraph, scipy.sparse.linalg, matplotlib, sentence-transformers (analysis)
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

### Memgraph (not required for prototype)

```bash
docker compose up memgraph  # Start Memgraph instance
```

## Architecture

The system follows a pipeline: **Extraction -> Storage -> Analysis -> API/CLI**.

**Graph schema** - Nodes are `Declaration` objects with properties: name, kind, type_expr, module, slogan, embedding, pagerank, cluster_id, centrality, fiedler_component, spectral_coords, heat_kernel_signature, proof-theoretic flags (is_constructive, is_computable, uses_choice, uses_propext, uses_quot). Edges: DEPENDS_ON, USES_DEF, EXTENDS, INSTANCE_OF, DEFINED_IN, IMPORTS.

**Proof-theoretic properties** extracted per declaration:
1. Noncomputable status (direct flag from ConstantInfo)
2. Transitive axiom usage (Classical.choice, propext, Quot.sound, funext)
3. Constructive status (derived from axiom usage)

**Go API server** exposes REST and GraphQL endpoints. The **CLI** wraps these APIs for human and agent use, providing commands for: search, premise retrieval, neighborhood exploration, graph features, proof properties, and proof attempt logging.

## Lean 4 API Reference (Extraction Development)

Key APIs for building and extending the extraction pipeline:

- `Environment.constants.map₁`: HashMap of all declarations in the environment
- `ConstantInfo` variants: `defnInfo`, `thmInfo`, `axiomInfo`, `inductInfo`, `ctorInfo`, `recInfo`, `opaqueInfo`, `quotInfo`
- `Expr.getUsedConstants`: Extract referenced constants from an expression
- `Lean.collectAxioms`: Transitive axiom dependency collection
- `env.getModuleIdxFor?`: Map a declaration to its source module
- `initSearchPath` / `importModules`: Load an environment from built oleans

## Reference Repositories

- [go-server-template](https://github.com/jllovet/go-server-template): Go API server pattern
- [linear-cli](https://github.com/jllovet/linear-cli): CLI design reference
- [LeanDepViz](https://github.com/cameronfreer/LeanDepViz): Declaration-level extraction, filtering logic
- [lean-graph](https://github.com/patrik-cihal/lean-graph): DependencyExtractor.lean metaprogram pattern
- [ImportGraph](https://github.com/leanprover-community/import-graph): initSearchPath/importModules environment loading

## Style and Conventions

- **No em-dashes.** Use commas, colons, parentheses, periods, or semicolons instead. This applies to all writing, including code comments, docstrings, and documentation.
- Academic tone in documentation. Technical precision over marketing language.
- Teal/slate blue color palette for any visual output.
- Do not reference private files' contents in committed files.
