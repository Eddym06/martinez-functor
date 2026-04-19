import Mathlib.Algebra.Polynomial.Eval.Defs
import Mathlib.Data.Real.Basic

theorem horner_exact_real (p : Polynomial ℝ) :
    ∀ x : ℝ, p.eval x = p.sum (fun e a => a * x ^ e) := by
  intro x
  simpa using (Polynomial.eval_eq_sum (p := p) (x := x))
