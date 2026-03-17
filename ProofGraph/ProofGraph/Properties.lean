/-
ProofGraph.Properties: Proof-theoretic property computation.

Computes per-declaration properties: noncomputable status, transitive axiom usage
(Classical.choice, propext, Quot.sound, funext), and derived constructive status.
Uses Lean.CollectAxioms for transitive axiom collection.
-/
import Lean
import Lean.Util.CollectAxioms
import ProofGraph.Extract

open Lean

namespace ProofGraph.Properties

/-- Standard Lean/Mathlib axioms. -/
def standardAxiomNames : List Name :=
  [`propext, `Classical.choice, `Quot.sound]

/-- Proof-theoretic properties computed for a declaration. -/
structure ProofProperties where
  isNoncomputable : Bool
  axioms          : Array Name
  isConstructive  : Bool
  usesChoice      : Bool
  usesPropext     : Bool
  usesQuot        : Bool
  deriving Inhabited

/-- A declaration enriched with both extraction data and proof-theoretic properties. -/
structure EnrichedDeclaration where
  raw        : ProofGraph.Extract.RawDeclaration
  properties : ProofProperties
  deriving Inhabited

/-- Collect all axioms transitively used by a declaration.
    Calls CollectAxioms.collect directly with the environment. -/
def collectTransitiveAxioms (env : Environment) (name : Name) : Array Name :=
  let (_, state) := ((CollectAxioms.collect name).run env).run {}
  state.axioms

/-- Check if a constant is an axiom in the environment. -/
private def isAxiomConst (env : Environment) (n : Name) : Bool :=
  match env.find? n with
  | some (.axiomInfo _) => true
  | _ => false

/-- Check noncomputable status: whether the value expression directly references axioms.
    Adapted from LeanDepViz noncompLike. -/
def isNoncomputable (env : Environment) (ci : ConstantInfo) : Bool :=
  match ci.value? (allowOpaque := true) with
  | some v =>
    let axioms := v.getUsedConstants.filter (isAxiomConst env)
    !axioms.isEmpty
  | none => false

/-- Derive proof-theoretic property flags from a transitive axiom set. -/
def deriveProperties (transitiveAxioms : Array Name) (isNoncomp : Bool) : ProofProperties :=
  let usesChoice := transitiveAxioms.any (· == `Classical.choice)
  let usesPropext := transitiveAxioms.any (· == `propext)
  let usesQuot := transitiveAxioms.any (· == `Quot.sound)
  { isNoncomputable := isNoncomp
    axioms           := transitiveAxioms
    isConstructive   := !usesChoice
    usesChoice       := usesChoice
    usesPropext      := usesPropext
    usesQuot         := usesQuot }

/-- Enrich all declarations in a RawGraph with proof-theoretic properties.
    When skipTransitive is true, uses direct axiom detection only (performance fallback). -/
def enrichGraph (env : Environment) (raw : ProofGraph.Extract.RawGraph)
    (skipTransitive : Bool := false) : Array EnrichedDeclaration := Id.run do
  let mut result := Array.mkEmpty raw.declarations.size
  for decl in raw.declarations do
    let ci := env.find? decl.name
    let isNoncomp := match ci with
      | some ci => isNoncomputable env ci
      | none    => false
    let props := if skipTransitive then
      deriveProperties #[] isNoncomp
    else
      let transitiveAxioms := collectTransitiveAxioms env decl.name
      deriveProperties transitiveAxioms isNoncomp
    result := result.push { raw := decl, properties := props }
  pure result

end ProofGraph.Properties
