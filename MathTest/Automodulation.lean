import Mathlib.Data.Real.Basic

namespace ACFTopos

/-- The space of all computable functions (simplified to ℝ → ℝ) -/
def Comp := ℝ → ℝ

-- Axiomas topológicos: Usamos `opaque` en lugar de `variable` para definir 
-- identidades estrictas sin romper el parser de Lean 4 con docstrings.

axiom IsFMA : Comp → Prop
axiom Phi : Comp → Comp

axiom phi_yields_fma (f : Comp) : IsFMA (Phi f)
axiom phi_fixed_point (f : Comp) : IsFMA f → Phi f = f

/-- THEOREM 1: THE MONAD AND IDEMPOTENCE -/
theorem phi_idempotent (f : Comp) : Phi (Phi f) = Phi f := by
  have h_fma : IsFMA (Phi f) := phi_yields_fma f
  exact phi_fixed_point (Phi f) h_fma

/-- THEOREM 2: CO-ALGEBRA AND THE ADJUNCTION (BASE) -/
def CoPhi (g : {func : Comp // IsFMA func}) : Comp := g.val

theorem coalgebra_generates_comp (g : {func : Comp // IsFMA func}) : IsFMA (CoPhi g) := by
  exact g.property

end ACFTopos
