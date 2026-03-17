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

/-- Render the ExtractionJson to a pretty-printed JSON string. -/
def renderJson (extraction : ExtractionJson) : String :=
  (toJson extraction).pretty

end ProofGraph.Json
