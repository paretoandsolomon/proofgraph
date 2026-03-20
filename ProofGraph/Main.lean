import ProofGraph
import Lean.Util.Path

open Lean

namespace ProofGraph.CLI

structure Options where
  moduleNames    : Array Name := #[]
  outputPath     : String := ""
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
      if opts.moduleNames.isEmpty then
        throw <| IO.userError
          "Usage: lake exe proofgraph-extract [--output <path>] [--skip-transitive] <module> [<module> ...]"
      else if opts.outputPath.isEmpty then
        throw <| IO.userError "missing --output <path>"
      else
        pure { opts with skipTransitive := skipT }
    | "--skip-transitive" :: rest => go opts true rest
    | "--output" :: path :: rest =>
      go { opts with outputPath := path } skipT rest
    | "--help" :: _ =>
      throw <| IO.userError
        "Usage: lake exe proofgraph-extract [--output <path>] [--skip-transitive] <module> [<module> ...]"
    | arg :: rest =>
      if arg.startsWith "--" then
        throw <| IO.userError s!"unknown flag: {arg}"
      else
        go { opts with moduleNames := opts.moduleNames.push (nameFromString arg) } skipT rest
  go default false args

def run (opts : Options) : IO Unit := do
  -- Step 1: Load environment
  Lean.initSearchPath (← Lean.findSysroot)
  let imports := opts.moduleNames.map fun m => { module := m : Import }
  let env ← Lean.importModules imports {} (trustLevel := 1024)

  -- Step 2: Build raw graph
  let moduleList := opts.moduleNames.map Name.toString
  IO.println s!"Extracting declarations from {moduleList.size} module(s)..."
  let rawGraph := ProofGraph.Extract.buildRawGraph env
  IO.println s!"  Found {rawGraph.declarations.size} declarations, {rawGraph.edges.size} edges"

  -- Step 3: Enrich with proof-theoretic properties
  if opts.skipTransitive then
    IO.println "Computing properties (skipping transitive axiom collection)..."
  else
    IO.println "Computing proof-theoretic properties..."
  let enrichedDecls := ProofGraph.Properties.enrichGraph env rawGraph
    (skipTransitive := opts.skipTransitive)
  IO.println s!"  Enriched {enrichedDecls.size} declarations"

  -- Step 4: Filter to module scope
  let filterPrefixes := opts.moduleNames.map fun n => n.getRoot.toString
  let uniquePrefixes := filterPrefixes.foldl (init := #[]) fun acc p =>
    if acc.contains p then acc else acc.push p
  let (filteredDecls, filteredEdges) :=
    ProofGraph.Filters.filterGraph enrichedDecls rawGraph.edges uniquePrefixes
  IO.println s!"  After filtering: {filteredDecls.size} declarations, {filteredEdges.size} edges"

  -- Step 5: Serialize to JSON
  let extractionDate ← do
    let out ← IO.Process.output { cmd := "date", args := #["-u", "+%Y-%m-%dT%H:%M:%SZ"] }
    pure out.stdout.trimAscii.toString
  let sourceModules := opts.moduleNames.map Name.toString
  let json := ProofGraph.Json.buildExtractionJson
    filteredDecls filteredEdges
    sourceModules.toList.toArray
    Lean.versionString
    extractionDate

  -- Step 6: Write to file (streaming to avoid stack overflow on large graphs)
  let handle ← IO.FS.Handle.mk opts.outputPath IO.FS.Mode.write
  ProofGraph.Json.writeJsonStreaming handle json
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
