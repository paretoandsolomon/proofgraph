/-
ProofGraph.Extract: Declaration and dependency edge extraction from Lean environments.

Responsible for iterating over environment constants, extracting metadata
(name, kind, type, module), and computing dependency edges via expression traversal.
Adapted from LeanDepViz Main.lean with modifications for ProofGraph's schema.
-/
import Lean

open Lean

namespace ProofGraph.Extract

/-- Classify a ConstantInfo into a human-readable kind string. -/
def constKind : ConstantInfo → String
  | .axiomInfo _   => "axiom"
  | .defnInfo _    => "def"
  | .thmInfo _     => "thm"
  | .opaqueInfo _  => "opaque"
  | .quotInfo _    => "quot"
  | .ctorInfo _    => "ctor"
  | .recInfo _     => "rec"
  | .inductInfo _  => "inductive"

/-- Resolve the module name for a declaration.
    Uses `getModuleIdxFor?` with ImportGraph's map₂ fallback for
    declarations defined in the current module. -/
def moduleOf (env : Environment) (decl : Name) : String :=
  match env.getModuleIdxFor? decl with
  | some idx =>
    match env.header.moduleNames[idx.toNat]? with
    | some m => m.toString
    | none   => "_unknown_"
  | none =>
    if env.constants.map₂.contains decl then
      env.header.mainModule.toString
    else
      "_unknown_"

/-- Raw declaration extracted from the environment, before property enrichment. -/
structure RawDeclaration where
  name     : Name
  kind     : String
  typeExpr : Expr
  module   : String
  isUnsafe : Bool
  hasSorry : Bool
  deriving Inhabited

/-- A dependency edge between two declarations. -/
structure Edge where
  source : Name
  target : Name
  kind   : String := "depends_on"
  deriving Inhabited

/-- Raw extraction result before property enrichment. -/
structure RawGraph where
  declarations : Array RawDeclaration
  edges        : Array Edge
  deriving Inhabited

/-- Extract a single declaration and its dependency edges.
    Gathers type and value dependencies, detects sorry usage.
    All dependency edges are unified as "depends_on". -/
def gatherDeclaration (env : Environment) (name : Name) (ci : ConstantInfo)
    : RawDeclaration × Array Edge :=
  let typeUses := ci.type.getUsedConstants
  let valueUses :=
    match ci.value? (allowOpaque := true) with
    | some v => v.getUsedConstants
    | none   => #[]
  let allDeps := (typeUses ++ valueUses).foldl
    (init := Std.HashSet.emptyWithCapacity)
    fun acc n => acc.insert n
  let hasSorry :=
    ci.type.hasSorry ||
    match ci.value? (allowOpaque := true) with
    | some v => v.hasSorry
    | none   => false
  let decl : RawDeclaration := {
    name     := name
    kind     := constKind ci
    typeExpr := ci.type
    module   := moduleOf env name
    isUnsafe := ci.isUnsafe
    hasSorry := hasSorry
  }
  let edges := allDeps.toArray.map fun dep =>
    { source := name, target := dep, kind := "depends_on" : Edge }
  (decl, edges)

/-- Build a raw graph from all non-internal declarations in the environment. -/
def buildRawGraph (env : Environment) : RawGraph := Id.run do
  let mut decls := Array.mkEmpty 1024
  let mut edges := Array.mkEmpty 4096
  for (name, info) in env.constants.map₁ do
    if !name.isInternalDetail then
      let (decl, es) := gatherDeclaration env name info
      decls := decls.push decl
      edges := edges ++ es
  pure { declarations := decls, edges := edges }

end ProofGraph.Extract
