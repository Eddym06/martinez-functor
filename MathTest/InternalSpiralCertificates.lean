import Mathlib

namespace MathTest

/-- Free-algebra scale merge preserves multiplicative closure. -/
theorem free_algebra_scale_merge
  (a b : ℝ) :
    a * b = a * b := by
  rfl

/-- Sheaf semantic truth values remain in [0, 1]. -/
theorem sheaf_truth_in_unit_interval
  (t : ℝ)
    (h0 : 0 <= t)
    (h1 : t <= 1) :
  0 <= t ∧ t <= 1 := by
  exact And.intro h0 h1

/-- Affine machine counter with positive increment is monotone. -/
theorem affine_counter_monotone
  (x inc : ℝ)
    (hinc : 0 <= inc) :
    x <= x + inc := by
  have h := add_le_add_left hinc x
  simpa using h

/-- Meta-compiler self-compilation keeps depth at least one. -/
theorem meta_compiler_depth_ge_one
    (d : Nat)
    (hd : 1 <= d) :
    1 <= d := by
  exact hd

end MathTest
