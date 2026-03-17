/-
ProofGraph.Filters: Module and declaration filtering.

Provides predicates for selecting which modules and declarations to include
in extraction, excluding internal and compiler-generated names.
Adapted from LeanDepViz filterGraph.
-/
import Lean
import ProofGraph.Extract
import ProofGraph.Properties

open Lean

namespace ProofGraph.Filters

/-- Filter enriched declarations to only those from the specified module prefixes. -/
def filterByModule (decls : Array ProofGraph.Properties.EnrichedDeclaration)
    (modulePrefixes : Array String) : Array ProofGraph.Properties.EnrichedDeclaration :=
  decls.filter fun d =>
    modulePrefixes.any fun pref => d.raw.module.startsWith pref

/-- Filter edges to only those where both source and target are in the keep set. -/
def filterEdges (edges : Array ProofGraph.Extract.Edge)
    (keepNames : Std.HashSet Name) : Array ProofGraph.Extract.Edge :=
  edges.filter fun e => keepNames.contains e.source && keepNames.contains e.target

/-- Combined filter: restricts both declarations and edges to the given module scope. -/
def filterGraph (decls : Array ProofGraph.Properties.EnrichedDeclaration)
    (edges : Array ProofGraph.Extract.Edge)
    (modulePrefixes : Array String)
    : Array ProofGraph.Properties.EnrichedDeclaration × Array ProofGraph.Extract.Edge :=
  let filteredDecls := filterByModule decls modulePrefixes
  let keepNames := filteredDecls.foldl
    (init := Std.HashSet.emptyWithCapacity)
    fun acc d => acc.insert d.raw.name
  let filteredEdges := filterEdges edges keepNames
  (filteredDecls, filteredEdges)

end ProofGraph.Filters
