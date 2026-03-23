# ProofGraph

**Spectral analysis of formalized mathematical knowledge.**

ProofGraph applies network science and spectral graph theory to the dependency graph of formalized mathematics. Proofs are programs, propositions are types, and a formalized mathematics library like Lean's [Mathlib](https://github.com/leanprover-community/mathlib4) (250,000+ theorems, 120,000+ definitions) is a complex network whose structure can be studied empirically.

## Research Questions

- Does Mathlib exhibit small-world properties?
- Is the degree distribution scale-free?
- Does community structure correspond to MSC classification?
- Where are the structural holes?
- What is the core-periphery structure?
- How does the network evolve over time?
- How do proof-theoretic properties propagate through the network?

**Working hypothesis:** Mathlib exhibits small-world properties and partial scale-free degree distribution, with community structure that partially but not fully corresponds to MSC classification. Spectral analysis will reveal a small algebraic connectivity (structural bottlenecks) and a Fiedler vector bipartition reflecting a fundamental division in mathematical practice.

## Current Capabilities

### Lean 4 Extraction Pipeline

Extracts declaration-level dependency graphs from Lean 4 projects with proof-theoretic property annotation:

- Declaration metadata: name, kind (theorem/def/axiom/...), module, unsafe status, sorry detection
- Dependency edges via `Expr.getUsedConstants` on type and value expressions
- Transitive axiom collection: identifies Classical.choice, propext, Quot.sound usage
- Constructive status derivation, noncomputability detection
- JSON output with metadata block (module list, extraction date, declaration/edge counts)
- Streaming JSON writer for large graphs (50K+ declarations without stack overflow)
- Tested at scale: 316K+ declarations (full Mathlib)

### Python Analysis Package

Spectral graph theory and visualization tools:

- **Graph loading**: JSON to NetworkX, largest connected component extraction, attribute preservation
- **Spectral analysis**: Graph Laplacian (combinatorial and normalized), Fiedler vector and algebraic connectivity, k-dimensional spectral embedding
- **Visualization**: Fiedler bipartition plots with spectral embedding layout, log-scaled coordinates for dense clusters, degree-scaled node sizes for hub visibility, multi-cluster overlay maps
- **Cluster analysis**: Per-cluster module breakdown, declaration kind breakdown, declaration listings; data-driven cluster labels (not hardcoded)
- **Recursive spectral bisection**: Hierarchical graph partitioning with configurable stopping criteria (max depth, min cluster size, algebraic connectivity ratio); per-split figures and leaf-cluster overlay
- **Multi-module merging**: Deduplicated merge of extraction JSONs across modules

### Analysis Methods (implemented)

- Fiedler vector bipartition (sparsest cut approximation)
- Algebraic connectivity measurement
- Spectral embedding (principled, non-force-directed node positioning)
- Normalized Laplacian for degree-heterogeneous graphs
- Recursive spectral bisection with spectral connectivity profiling
- Cluster content analysis (module distribution, declaration kinds)

## Coming Soon

### Analysis Methods (planned)

- Community detection (Louvain, label propagation)
- Degree distribution analysis, PageRank, betweenness centrality
- Structural holes (Burt), core-periphery decomposition (Newman)
- Heat kernel diffusion (continuous taint analysis)
- Persistent homology (TDA)
- Temporal evolution across Mathlib commits

### Go API and CLI

REST and GraphQL API server with CLI for human and agent interaction:

- `proofgraph search`: Structural and semantic search over declarations
- `proofgraph features`: Centrality, PageRank, cluster, proof properties per declaration
- `proofgraph taint`: Classical/constructive taint chain analysis
- `proofgraph premises`: Graph-distance-aware premise ranking

### Additional Planned Features

- Proof-theoretic taint analysis (binary and continuous via heat kernel)
- Proof mining prioritization (graph-theoretic identification of classical barriers)
- Ecosystem governance (redundancy detection, structural balance monitoring)
- Graph-distance-aware premise ranking with spectral features for GNN inputs
- Open dataset and benchmark on HuggingFace

## Technology Stack

**Current:**
- **Extraction:** Lean 4 Environment API (adapting [LeanDepViz](https://github.com/cameronfreer/LeanDepViz))
- **Analysis:** Python (NetworkX, scipy.sparse.linalg, matplotlib, numpy)
- **Infrastructure:** Docker Compose for Python analysis

**Planned:**
- **API + CLI:** Go (REST + GraphQL + CLI in one binary, hexagonal architecture)
- **Graph database:** Memgraph (Cypher queries, vector search)
- **Embeddings:** sentence-transformers (local)

## Project Structure

```
proofgraph/
  ProofGraph/               # Lean 4 extraction pipeline
    ProofGraph/
      Extract.lean          # Declaration and edge extraction
      Properties.lean       # Proof-theoretic property computation
      Filters.lean          # Module-based filtering
      Json.lean             # JSON serialization
    Main.lean               # CLI entry point
  src/proofgraph/           # Python analysis package
    loader.py               # JSON to NetworkX graph
    spectral.py             # Laplacian, Fiedler vector, spectral embedding
    viz.py                  # Bipartition and multi-cluster visualization
    clusters.py             # Cluster analysis and recursive bisection
  scripts/
    generate_figure.py      # Figure generation and analysis pipeline
    merge_extractions.py    # Multi-module extraction merger
  tests/                    # Python test suite (114 tests)
  data/                     # Generated JSON artifacts (git-ignored)
  figures/                  # Generated visualizations (git-ignored)
  docs/                     # Documentation
```

## Quick Start

### Lean Extraction

```bash
cd ProofGraph
lake build
lake exe proofgraph-extract Mathlib.Data.Nat.Basic ../data/nat_basic.json
```

### Python Analysis (via Docker)

```bash
# Generate a spectral bipartition figure with cluster analysis
docker compose run --build --rm python scripts/generate_figure.py \
  data/nat_basic.json figures/

# Large-scale analysis (streaming loader, light attributes, memory profiling)
docker compose run --build --rm python scripts/generate_figure.py \
  data/mathlib_full.json figures/ --streaming --light --profile

# Recursive spectral bisection with per-split figures
docker compose run --build --rm python scripts/generate_figure.py \
  data/mathlib_full.json figures/ --streaming --light --recursive --max-depth 4

# Allow deeper splitting into well-connected regions
docker compose run --build --rm python scripts/generate_figure.py \
  data/mathlib_full.json figures/ --streaming --light --recursive \
  --max-depth 4 --connectivity-ratio 50.0

# Run tests
docker compose run --build --rm python -m pytest tests/ -v

# Merge multiple extractions
docker compose run --build --rm python scripts/merge_extractions.py \
  data/nat_basic.json data/algebra_group_basic.json -o data/merged.json
```

### Pipeline Flags

| Flag | Description |
|------|-------------|
| `--streaming` | Use ijson streaming parser (low memory, three-pass). Required for multi-GB files. |
| `--light` | Drop heavy attributes (type expressions) to reduce memory by ~2 GB at 180K declarations. |
| `--profile` | Report peak memory usage via tracemalloc. |
| `--top-n N` | Number of top modules per cluster in analysis tables (default 20). |
| `--recursive` | Run recursive spectral bisection after initial analysis. |
| `--max-depth N` | Maximum recursion depth for bisection (default 4). |
| `--min-size N` | Minimum cluster size to continue bisecting (default 200). |
| `--connectivity-ratio F` | Stop recursing when a subcluster's algebraic connectivity exceeds F times its parent's (default 10.0). Higher values allow deeper splitting. |

### Output Files

The pipeline produces the following in the output directory:

| File | Description |
|------|-------------|
| `fiedler_bipartition.png` | Two-color spectral bipartition (teal/slate blue). |
| `cluster_analysis.md` | Per-cluster module breakdown, kind breakdown, summary tables. |
| `cluster_<N>_declarations.txt` | Full declaration listing per cluster, one name per line. |
| `recursive_clusters.png` | Full graph colored by leaf clusters from recursive bisection. |
| `recursive_bisection.md` | Hierarchical report with spectral connectivity profile. |
| `bisections/bisection_<label>.png` | Per-split bipartition figure with its own spectral embedding. |

## License

[MIT](LICENSE)

## Acknowledgments

ProofGraph builds on the work of the Lean community, [Mathlib](https://github.com/leanprover-community/mathlib4), and [LeanDojo](https://leandojo.org/). The theoretical foundation draws on Newman (Networks), Chung (Spectral Graph Theory), Burt (Structural Holes), Watts and Strogatz (small-world networks), and Doignon and Falmagne (Knowledge Space Theory).

## Project Homepage

For research publications, documentation, and project updates, visit [proofgraph.org](https://proofgraph.org).

## Contact

Jonathan Llovet, jllovet@paretoandsolomon.com
