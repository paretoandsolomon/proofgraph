import ProofGraph
import Lean.Util.Path

open Lean

namespace ProofGraph.CLI

structure Options where
  moduleName     : Name
  outputPath     : String
  skipTransitive : Bool := false
  deriving Inhabited

/-- Convert a dotted string like "Mathlib.Data.Nat.Basic" to a Lean Name. -/
private def nameFromString (s : String) : Name :=
  (s.splitOn ".").foldl (init := Name.anonymous) fun acc part =>
    let part := part.trimAscii.toString
    if part.isEmpty then acc else Name.str acc part

def parseArgs (args : List String) : IO Options := do
  let rec go (opts : Options) (skipT : Bool) : List String → IO Options
    | [] =>
      if opts.moduleName.isAnonymous then
        throw <| IO.userError
          "Usage: lake exe proofgraph-extract <module> <output_path> [--skip-transitive]"
      else if opts.outputPath.isEmpty then
        throw <| IO.userError "missing output path"
      else
        pure { opts with skipTransitive := skipT }
    | "--skip-transitive" :: rest => go opts true rest
    | "--help" :: _ =>
      throw <| IO.userError
        "Usage: lake exe proofgraph-extract <module> <output_path> [--skip-transitive]"
    | arg :: rest =>
      if opts.moduleName.isAnonymous then
        go { opts with moduleName := nameFromString arg } skipT rest
      else if opts.outputPath.isEmpty then
        go { opts with outputPath := arg } skipT rest
      else
        throw <| IO.userError s!"unexpected argument: {arg}"
  go default false args

def run (opts : Options) : IO Unit := do
  -- Step 1: Load environment
  Lean.initSearchPath (← Lean.findSysroot)
  let imports := #[{ module := opts.moduleName : Import }]
  let env ← Lean.importModules imports {} (trustLevel := 1024)

  -- Step 2: Build raw graph
  IO.println s!"Extracting declarations from {opts.moduleName}..."
  let rawGraph := ProofGraph.Extract.buildRawGraph env
  IO.println s!"  Found {rawGraph.declarations.size} declarations, {rawGraph.edges.size} edges"

  -- Step 3: Enrich with proof-theoretic properties
  if opts.skipTransitive then
    IO.println "Computing properties (skipping transitive axiom collection)..."
  else
    IO.println "Computing proof-theoretic properties..."
  let enrichedDecls := ProofGraph.Properties.enrichGraph env rawGraph
    (skipTransitive := opts.skipTransitive)

  -- Step 4: Filter to module scope
  let modulePrefix := opts.moduleName.getRoot.toString
  let (filteredDecls, filteredEdges) :=
    ProofGraph.Filters.filterGraph enrichedDecls rawGraph.edges #[modulePrefix]
  IO.println s!"  After filtering: {filteredDecls.size} declarations, {filteredEdges.size} edges"

  -- Step 5: Serialize to JSON
  let extractionDate ← do
    let out ← IO.Process.output { cmd := "date", args := #["-u", "+%Y-%m-%dT%H:%M:%SZ"] }
    pure out.stdout.trimAscii.toString
  let json := ProofGraph.Json.buildExtractionJson
    filteredDecls filteredEdges
    #[opts.moduleName.toString]
    Lean.versionString
    extractionDate

  -- Step 6: Write to file
  let jsonStr := ProofGraph.Json.renderJson json
  IO.FS.writeFile opts.outputPath jsonStr
  IO.println s!"Wrote {filteredDecls.size} declarations and {filteredEdges.size} edges to {opts.outputPath}"

end ProofGraph.CLI

def main (args : List String) : IO UInt32 := do
  try
    let opts ← ProofGraph.CLI.parseArgs args
    ProofGraph.CLI.run opts
    return 0
  catch e =>
    IO.eprintln s!"Error: {e.toString}"
    return 1
