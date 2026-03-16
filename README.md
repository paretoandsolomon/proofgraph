# ProofGraph

**Spectral analysis of formalized mathematical knowledge.**

ProofGraph applies network science and spectral graph theory to the dependency graph of formalized mathematics. Proofs are programs, propositions are types, and a formalized mathematics library like Lean's [Mathlib](https://github.com/leanprover-community/mathlib4) (250,000+ theorems, 120,000+ definitions) is a complex network whose structure can be studied empirically.

ProofGraph extracts the declaration-level dependency graph from Lean 4 projects, computes proof-theoretic properties, runs structural and spectral analysis, and exposes results through REST and GraphQL APIs, with a CLI for human and agent interaction.

## Research Questions

- Does Mathlib exhibit small-world properties?
- Is the degree distribution scale-free?
- Does community structure correspond to MSC classification?
- Where are the structural holes?
- What is the core-periphery structure?
- How does the network evolve over time?
- How do proof-theoretic properties propagate through the network?

**Working hypothesis:** Mathlib exhibits small-world properties and partial scale-free degree distribution, with community structure that partially but not fully corresponds to MSC classification. Spectral analysis will reveal a small algebraic connectivity (structural bottlenecks) and a Fiedler vector bipartition reflecting a fundamental division in mathematical practice.

## What ProofGraph Provides

1. Network science analysis of formalized mathematical knowledge
2. Spectral analysis: Laplacian spectrum, Fiedler decomposition, algebraic connectivity, spectral clustering, heat kernel diffusion
3. Proof-theoretic taint analysis (binary and continuous via heat kernel)
4. Structural analysis: communities, structural holes, core-periphery decomposition
5. Graph-distance-aware premise ranking (with spectral features for GNN inputs)
6. Proof mining prioritization (using graph structure to identify high-value proof targets)
7. Ecosystem governance (structural health metrics, dependency risk analysis)
8. Open datasets and benchmarks

## Analysis Methods

**Phase 1: Standard Network Science.**
Community detection (Louvain, label propagation), degree distribution analysis, PageRank, betweenness centrality, structural holes (Burt), core-periphery decomposition (Newman), temporal evolution.

**Phase 2: Spectral Graph Theory.**
Spectral clustering (Laplacian eigenvectors), Fiedler vector analysis (natural bipartition), algebraic connectivity, hierarchical spectral bisection, Cheeger inequality bounds, spectral features as GNN inputs.

**Phase 3: Advanced Spectral and Topology.**
Heat kernel diffusion (continuous taint analysis), spectral embedding (principled visualization), persistent homology (TDA).

## Technology Stack

- **Extraction:** Lean 4 Environment API, LeanDojo v2 methodology
- **API server:** Go (REST + GraphQL)
- **CLI:** Wraps REST/GraphQL APIs for human and agent interaction
- **Graph database:** Memgraph (Cypher, vector search)
- **Analysis:** Python (NetworkX, igraph, scipy.sparse.linalg, matplotlib)
- **Embeddings:** sentence-transformers (local)
- **Visualization:** Spectral embedding

## Project Structure

```
proofgraph/
  ProofGraph/           # Lean 4 project (extraction, formalization)
  proofgraph/           # Python package (analysis)
  data/                 # Generated data artifacts (git-ignored)
  docs/                 # Documentation and schema
```

## Getting Started

*Coming soon.* ProofGraph is under active development. See the roadmap below.

## License

[MIT](LICENSE)

## Acknowledgments

ProofGraph builds on the work of the Lean community, [Mathlib](https://github.com/leanprover-community/mathlib4), and [LeanDojo](https://leandojo.org/). The theoretical foundation draws on Newman (Networks), Chung (Spectral Graph Theory), Burt (Structural Holes), Watts and Strogatz (small-world networks), and Doignon and Falmagne (Knowledge Space Theory).

## Project Homepage

For research publications, documentation, and project updates, visit [proofgraph.org](https://proofgraph.org).

## Contact

Jonathan Llovet, jllovet@paretoandsolomon.com
