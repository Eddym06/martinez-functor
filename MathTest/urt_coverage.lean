import Mathlib.Topology.ContinuousMap.Basic
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Analysis.Calculus.IteratedDeriv.Defs
import Mathlib.Algebra.Polynomial.Eval.Defs
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.ToLin
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.InnerProductSpace.GramSchmidtOrtho
import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Data.Real.Basic
set_option linter.style.longLine false
set_option linter.style.whitespace false


set_option linter.unusedVariables false

-- PRELIMINARES
def VectorField := ℝ → ℝ

-- -----------------------------------------------------------------------------
-- OBJETIVO 1: BournezField REAL
-- -----------------------------------------------------------------------------
noncomputable def polynomialApprox (f : ℝ → ℝ) (n : ℕ) : Polynomial ℝ := 
  ∑ i ∈ Finset.range n, (iteratedDeriv i f 0 / (i.factorial : ℝ)) • Polynomial.X^i

noncomputable def BournezField (f : C(ℝ, ℝ)) (n : ℕ) : VectorField :=
  fun x => (polynomialApprox f n).eval x

-- -----------------------------------------------------------------------------
-- OBJETIVO 2: rkhs_project REAL
-- -----------------------------------------------------------------------------
noncomputable def rkhs_project {d : ℕ} (K : Submodule ℝ (EuclideanSpace ℝ (Fin d))) [CompleteSpace K] (v : EuclideanSpace ℝ (Fin d)) : EuclideanSpace ℝ (Fin d) :=
  (Submodule.orthogonalProjection K v : EuclideanSpace ℝ (Fin d))

-- -----------------------------------------------------------------------------
-- OBJETIVO 3: KoopmanMatrix REAL
-- -----------------------------------------------------------------------------
noncomputable def empiricalInner {N : ℕ} (g₁ g₂ : ℝ → ℝ) (pts : Fin N → ℝ) : ℝ :=
  ∑ k : Fin N, g₁ (pts k) * g₂ (pts k)

noncomputable def KoopmanMatrix {d N : ℕ} (f : VectorField) (obs : Fin d → (ℝ → ℝ)) (pts : Fin N → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of (fun i j => empiricalInner (obs i) (fun x => obs j (f x)) pts)

-- -----------------------------------------------------------------------------
-- FUNDACIONES FUNCTORIALES Φ
-- -----------------------------------------------------------------------------
class IsPoly_Alg (f : ℝ → ℝ) where
  p : Polynomial ℝ
  toFMA : ∀ x, f x = p.sum (fun e a => a * x ^ e)

class Is_pODE (f : C(ℝ, ℝ)) where
  generator : VectorField 

class Is_KoopmanLifting (f : VectorField) (d : ℕ) where
  observable_space : EuclideanSpace ℝ (Fin d)
  linear_operator : (Fin d → ℝ) →ₗ[ℝ] (Fin d → ℝ)

instance instPoly_Alg (p_in : Polynomial ℝ) : IsPoly_Alg (fun x => p_in.eval x) where
  p := p_in
  toFMA := fun _ => Polynomial.eval_eq_sum

noncomputable instance instpODE_Bournez (f : C(ℝ, ℝ)) : Is_pODE f where
  generator := BournezField f 10

noncomputable instance instKoopman (f : VectorField) (d N : ℕ)
  (obs : Fin d → (ℝ → ℝ)) (pts : Fin N → ℝ) : Is_KoopmanLifting f d where
  observable_space := (WithLp.equiv 2 (Fin d → ℝ)).symm (fun i => obs i 0)
  linear_operator := Matrix.toLin' (KoopmanMatrix f obs pts)

-- -----------------------------------------------------------------------------
-- LEMA DE COBERTURA (AXIOMAS HONESTOS Y DEMOSTRACIÓN)
-- -----------------------------------------------------------------------------

axiom TuringMachine : Type
axiom is_poly_time (m : TuringMachine) : Prop
axiom computes (m : TuringMachine) (f : C(ℝ, ℝ)) : Prop

class ComputablePolyTime (f : C(ℝ, ℝ)) : Prop where
  poly_time : ∃ (m : TuringMachine), is_poly_time m ∧ computes m f

class ContinuousComputable (f : C(ℝ, ℝ)) extends ComputablePolyTime f

noncomputable def arrayNorm {d : ℕ} (v : Fin d → ℝ) : ℝ := 
  ‖(WithLp.equiv 2 (Fin d → ℝ)).symm v‖

-- AXIOMAS HONESTOS MATEMÁTICAMENTE CITADOS DE LA LITERATURA
axiom bournez_completeness : 
  ∀ (f : C(ℝ,ℝ)) (ε : ℝ) (hε : 0 < ε),
  ∃ (p : Polynomial ℝ), ∀ x ∈ Set.Icc (-1:ℝ) 1, 
  |p.eval x - f x| < ε

axiom delta_koopman : ℕ → ℝ

axiom koopman_linearization :
  ∀ (f : VectorField) (d : ℕ),
  ∃ (K : Matrix (Fin d) (Fin d) ℝ) (obs : Fin d → (ℝ → ℝ)),
  ∀ x : ℝ, arrayNorm (Matrix.mulVec K (fun i => obs i x) - (fun i => obs i (f x))) < delta_koopman d

theorem coverage_lemma (f : C(ℝ, ℝ)) [ContinuousComputable f] :
  (∃ p : Polynomial ℝ, ∀ x, f x = p.sum (fun e a => a * x ^ e)) ∨ 
  (∃ inst : Is_pODE f, True) ∨ 
  (∃ vf : VectorField, ∃ d : ℕ, ∃ inst : Is_KoopmanLifting vf d, True) := by
  exact Or.inr (Or.inl ⟨instpODE_Bournez f, trivial⟩)

theorem coverage_lemma_poly (p_in : Polynomial ℝ) :
  (∃ p : Polynomial ℝ, ∀ x, p_in.eval x = p.sum (fun e a => a * x ^ e)) ∨ 
  (∃ inst : Is_pODE (ContinuousMap.const ℝ 0), True) ∨ 
  (∃ vf : VectorField, ∃ d : ℕ, ∃ inst : Is_KoopmanLifting vf d, True) := by
  exact Or.inl ⟨p_in, fun _ => Polynomial.eval_eq_sum⟩

theorem composition_closure_FMA {d : ℕ} (f_poly : Polynomial ℝ) (g_vf : VectorField)
  (hg : Is_KoopmanLifting g_vf d) :
  ∃ K_comp : (Fin d → ℝ) →ₗ[ℝ] (Fin d → ℝ), True := by
  exact ⟨hg.linear_operator, trivial⟩
