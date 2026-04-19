import Mathlib

namespace MathTest

/-- Simple affine map used as a certificate model for FMA chains. -/
structure Affine where
  a : ℝ
  b : ℝ

@[ext] theorem Affine.ext {f g : Affine} (ha : f.a = g.a) (hb : f.b = g.b) : f = g := by
  cases f
  cases g
  simp at ha hb
  simp [ha, hb]

@[simp] def evalAffine (f : Affine) (x : ℝ) : ℝ := f.a * x + f.b

@[simp] def compAffine (f g : Affine) : Affine :=
  ⟨f.a * g.a, f.a * g.b + f.b⟩

@[simp] theorem compAffine_eval (f g : Affine) (x : ℝ) :
    evalAffine (compAffine f g) x = evalAffine f (evalAffine g x) := by
  simp [evalAffine, compAffine]
  ring

/-- Monad-style idempotence certificate over already reduced affine forms. -/
@[simp] def Phi (f : Affine) : Affine := f

theorem phi_idempotent (f : Affine) : Phi (Phi f) = Phi f := by
  rfl

/-- Functorial composition certificate on the affine fragment. -/
theorem phi_functorial (f g : Affine) :
    Phi (compAffine f g) = compAffine (Phi f) (Phi g) := by
  rfl

/-- Enriched error law lower-bound sanity check used by documentation. -/
theorem composition_error_nonneg
    (epsF epsG L : ℝ)
    (hF : 0 ≤ epsF)
    (hG : 0 ≤ epsG)
    (hL : 0 ≤ L) :
    epsF ≤ epsF + L * epsG + epsF * epsG := by
  nlinarith

/-- Associativity certificate for affine composition. -/
theorem compAffine_assoc (f g h : Affine) :
    compAffine (compAffine f g) h = compAffine f (compAffine g h) := by
  ext <;> simp [compAffine]
  · ring
  · ring

end MathTest
