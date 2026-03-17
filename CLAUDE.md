# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProofGraph applies network science and spectral graph theory to the dependency graph of formalized mathematics (Lean 4 / Mathlib). It extracts declaration-level dependency graphs, computes proof-theoretic properties, runs structural and spectral analysis, and exposes results through a Go API server (REST + GraphQL), with a CLI for human and agent interaction. MIT licensed.

## Project Structure

```
proofgraph/
  ProofGraph/                  # Lean 4 extraction (adapts LeanDepViz pattern)
    lakefile.lean
    lean-toolchain
    ProofGraph/Main.lean       # Extraction: env loading, dependency walking, JSON output
  cmd/
    proofgraph/main.go         # Go composition root (CLI + API in one binary)
  internal/
    graph/                     # Domain: Declaration, Edge, Service interface
    storage/
      json/                    # Load extraction JSON from disk
      memory/                  # In-memory graph (initial backend)
    server/                    # HTTP server (REST + GraphQL)
    analysis/                  # Network analysis, spectral, taint
  scripts/
    analyze.py                 # Python bridge for scipy/matplotlib
  data/                        # Generated JSON artifacts (git-ignored)
  docs/                        # Documentation and schema
```

## Technology Stack

- **Lean 4** for extraction (adapting LeanDepViz pattern; Environment API)
- **Go** API server (REST + GraphQL) and CLI in one binary (hexagonal architecture)
- **Python**: NetworkX/igraph, scipy.sparse.linalg, matplotlib, sentence-transformers (analysis, visualization)
- **Graph database**: Memgraph (Cypher queries, vector search; not required for prototype)
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

The system follows a pipeline: **Extraction (Lean) -> JSON -> Storage/Analysis (Go + Python) -> API/CLI (Go)**.

**Data flow:** Lean extraction writes JSON to `data/`. The Go service reads JSON, builds an in-memory graph, and serves queries via CLI and REST/GraphQL. Python scripts handle scipy/matplotlib analysis that Go delegates to. The CLI and API server are subcommands of the same Go binary (`proofgraph serve`, `proofgraph search`, etc.).

**Prototype strategy:** For initial figures and analysis, load extraction JSON directly in Python with NetworkX. Build the Go service as the long-term architecture.

**Graph schema** - Nodes are `Declaration` objects with properties: name, kind, type_expr, module, slogan, embedding, pagerank, cluster_id, centrality, fiedler_component, spectral_coords, heat_kernel_signature, msc_code, ccs_code, has_docstring, proof_assistant, source_commit, proof-theoretic flags (is_constructive, is_computable, uses_choice, uses_propext, uses_quot). Edges: DEPENDS_ON, USES_DEF, EXTENDS, INSTANCE_OF, DEFINED_IN, IMPORTS.

**Proof-theoretic properties** extracted per declaration:
1. Noncomputable status (direct flag from ConstantInfo)
2. Transitive axiom usage (Classical.choice, propext, Quot.sound, funext)
3. Constructive status (derived from axiom usage)

**Go API server** exposes REST and GraphQL endpoints. The **CLI** wraps these APIs for human and agent use:

```
proofgraph search <query> [--mode structural|semantic|combined] [--limit N] [--format json|text]
proofgraph neighborhood <declaration> [--depth N] [--edge-types deps,uses,extends]
proofgraph features <declaration>          # centrality, PageRank, cluster, proof properties
proofgraph taint <declaration> [--transitive]  # taint chain, classical barriers, propagation impact
proofgraph premises <goal_state> [--context file]  # ranked premises with graph features
proofgraph log-attempt --goal <goal> --premises-used <list> --success <bool> [--error <msg>]
```

## Lean 4 API Reference (Extraction Development)

Key APIs for building and extending the extraction pipeline:

- `Environment.constants.map₁`: HashMap of all declarations in the environment
- `ConstantInfo` variants: `defnInfo`, `thmInfo`, `axiomInfo`, `inductInfo`, `ctorInfo`, `recInfo`, `opaqueInfo`, `quotInfo`
- `Expr.getUsedConstants`: Extract referenced constants from an expression
- `Lean.collectAxioms`: Transitive axiom dependency collection
- `env.getModuleIdxFor?`: Map a declaration to its source module
- `initSearchPath` / `importModules`: Load an environment from built oleans

## Reference Repositories

### Lean extraction (primary reference: LeanDepViz)

- [LeanDepViz](https://github.com/cameronfreer/LeanDepViz): **Primary extraction reference.** `Main.lean` is a near-complete working prototype: loads environment via `initSearchPath` + `importModules`, iterates `env.constants.map₁`, extracts declaration kind via `ConstantInfo` pattern matching, gets dependencies via `Expr.getUsedConstants` on type and value expressions, checks axiom usage (direct), detects `sorry`, checks noncomputability, outputs DOT and JSON. ProofGraph extends this with: (1) transitive axiom collection via `Lean.collectAxioms`, (2) `is_constructive` derived field, (3) module attribution via `env.getModuleIdxFor?` with ImportGraph's `map₂` fallback.
- [lean-graph](https://github.com/patrik-cihal/lean-graph): DependencyExtractor.lean metaprogram pattern
- [ImportGraph](https://github.com/leanprover-community/import-graph): `initSearchPath`/`importModules` environment loading; `Environment.getModuleFor?` fallback for `map₂` declarations

### Go API and CLI

- [go-server-template](https://github.com/jllovet/go-server-template): Hexagonal architecture template. Mapping: `todo.Todo` -> `declaration.Declaration`/`graph.Edge`; `todo.Service` -> `graph.Service` (search, features, taint); `todo.Repository` -> `graph.Repository` (JSON initially, Memgraph later); `server.Server` -> `server.Server` (REST API).
- [linear-cli](https://github.com/jllovet/linear-cli): CLI design reference

## Linear (Project Management)

Issue tracking uses [Linear](https://linear.app) via the `linear` CLI. The `LINEAR_API_TOKEN` is loaded by `direnv`; run `direnv allow` if the token is not set.

The team key is **PG** (ProofGraph). Issue identifiers follow the pattern `PG-<number>`.

### Common Commands

```bash
# Read
linear issue get PG-1                          # View an issue
linear issue list                               # List all issues
linear issue comments PG-1                      # List comments on an issue
linear issue children PG-1                      # List sub-issues
linear states list --team PG                    # List workflow states and their IDs

# Write
linear issue comment PG-1 --body "comment text" # Add a comment
linear issue update PG-1 --state-id <id>        # Change workflow state
linear issue update PG-1 --priority 2           # Set priority (1=urgent, 2=high, 3=medium, 4=low)
linear issue create --title "..." --description "..." # Create an issue (needs team context)
```

### Workflow States

The PG team uses: Backlog, Icebox, Todo, In Progress, In Review, Done, Canceled, Duplicate.

State changes require `--state-id` (a UUID), not a state name. To find IDs:

```bash
linear states list --team PG                    # Lists all states with their IDs
linear issue update PG-1 --state-id "<uuid>"    # Update state
```

### Conventions

- When starting work on a Linear issue, move it to **In Progress**.
- Post implementation plans and status updates as comments on the issue.
- When work is complete and verified, move to **In Review** (or **Done** if no review is needed).
- Commit messages should reference the issue identifier (e.g., `PG-1`) in the body or title.

## Style and Conventions

- **No em-dashes.** Use commas, colons, parentheses, periods, or semicolons instead. This applies to all writing, including code comments, docstrings, and documentation.
- Academic tone in documentation. Technical precision over marketing language.
- Teal/slate blue color palette for any visual output.
- Do not reference private files' contents in committed files.
