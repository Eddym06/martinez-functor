import Mathlib

namespace ACF.Topos

structure StratifiedInterval where
  val : ℝ
  region : ℤ
  boundary_tolerance : ℝ := 1e-12

noncomputable def ReLU_Stratified (x : ℝ) : StratifiedInterval :=
  if x < 0 then
    { val := 0, region := -1, boundary_tolerance := |x| }
  else if x > 0 then
    { val := x, region := 1, boundary_tolerance := 0 }
  else
    { val := 0, region := 0, boundary_tolerance := 0 }

theorem Stratified_Preservation (x : ℝ) : 
  (ReLU_Stratified x).region = 1 → (ReLU_Stratified x).val = x := by
  intro h
  unfold ReLU_Stratified at h
  by_cases hx : x < 0
  · simp [hx] at h
  · by_cases hx' : x > 0
    · show (ReLU_Stratified x).val = x
      simp [hx, hx']
    · have hx0 : x = 0 := by linarith
      simp [hx, hx', hx0] at h

theorem Constructible_Sheaf_Isomorphism (x y : ℝ) (h : x = y) :
    ReLU_Stratified x = ReLU_Stratified y := by
  subst h
  rfl

end ACF.Topos