/-
ProofGraph.Json: JSON serialization types and instances for extraction output.

Defines structures for declarations, edges, and metadata, along with
ToJson instances for producing the output schema consumed by the Python pipeline.
-/
import Lean
import Lean.Data.Json
import ProofGraph.Extract
import ProofGraph.Properties

open Lean

namespace ProofGraph.Json

structure MetadataJson where
  sourceModules    : Array String
  extractionDate   : String
  leanVersion      : String
  declarationCount : Nat
  edgeCount        : Nat
  deriving ToJson, FromJson

structure DeclarationJson where
  name           : String
  kind           : String
  type           : String
  module         : String
  isNoncomputable : Bool
  axioms         : Array String
  isConstructive : Bool
  usesChoice     : Bool
  usesPropext    : Bool
  usesQuot       : Bool
  hasSorry       : Bool
  deriving ToJson, FromJson

structure EdgeJson where
  source : String
  target : String
  kind   : String
  deriving ToJson, FromJson

structure ExtractionJson where
  metadata     : MetadataJson
  declarations : Array DeclarationJson
  edges        : Array EdgeJson
  deriving ToJson, FromJson

/-- Convert an EnrichedDeclaration to its JSON-serializable form. -/
def enrichedToJson (d : ProofGraph.Properties.EnrichedDeclaration) : DeclarationJson :=
  { name           := d.raw.name.toString
    kind           := d.raw.kind
    type           := toString d.raw.typeExpr
    module         := d.raw.module
    isNoncomputable := d.properties.isNoncomputable
    axioms         := d.properties.axioms.map (·.toString)
    isConstructive := d.properties.isConstructive
    usesChoice     := d.properties.usesChoice
    usesPropext    := d.properties.usesPropext
    usesQuot       := d.properties.usesQuot
    hasSorry       := d.raw.hasSorry }

/-- Convert an Edge to its JSON-serializable form. -/
def edgeToJson (e : ProofGraph.Extract.Edge) : EdgeJson :=
  { source := e.source.toString
    target := e.target.toString
    kind   := e.kind }

/-- Build the complete JSON output from filtered declarations, edges, and metadata. -/
def buildExtractionJson
    (decls : Array ProofGraph.Properties.EnrichedDeclaration)
    (edges : Array ProofGraph.Extract.Edge)
    (sourceModules : Array String)
    (leanVersion : String)
    (extractionDate : String)
    : ExtractionJson :=
  { metadata := {
      sourceModules    := sourceModules
      extractionDate   := extractionDate
      leanVersion      := leanVersion
      declarationCount := decls.size
      edgeCount        := edges.size }
    declarations := decls.map enrichedToJson
    edges        := edges.map edgeToJson }

/-- Render the ExtractionJson to a pretty-printed JSON string.

For small extractions only; large graphs will overflow the stack.
Use ``writeJsonStreaming`` for large outputs.
-/
def renderJson (extraction : ExtractionJson) : String :=
  (toJson extraction).pretty

/-- Write the extraction JSON to a file handle in a streaming fashion.

This avoids building the entire JSON tree in memory and prevents stack
overflows on large graphs (50K+ declarations, 400K+ edges).
-/
def writeJsonStreaming (h : IO.FS.Handle) (extraction : ExtractionJson) : IO Unit := do
  h.putStrLn "{"
  -- metadata
  h.putStrLn s!"  \"metadata\": {(toJson extraction.metadata).compress},"
  -- declarations
  h.putStrLn "  \"declarations\": ["
  let mut firstDecl := true
  for d in extraction.declarations do
    if firstDecl then
      firstDecl := false
    else
      h.putStr ","
      h.putStrLn ""
    h.putStr s!"    {(toJson d).compress}"
  h.putStrLn ""
  h.putStrLn "  ],"
  -- edges
  h.putStrLn "  \"edges\": ["
  let mut firstEdge := true
  for e in extraction.edges do
    if firstEdge then
      firstEdge := false
    else
      h.putStr ","
      h.putStrLn ""
    h.putStr s!"    {(toJson e).compress}"
  h.putStrLn ""
  h.putStrLn "  ]"
  h.putStrLn "}"

end ProofGraph.Json
