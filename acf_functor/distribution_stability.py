"""
Distribution Stability Analysis — Perturbation Theory for LF-Spaces
====================================================================

THE DOCTORAL CORE PROBLEM:
  For a tempered distribution T of finite order k with compact support,
  its dual representation R(T) = (c, S) — where c are spectral coefficients
  and S are singularities — must be STABLE under perturbations of
  singularity positions: x₀ → x₀ + η.

  The stability must hold in the weak-* topology of Φ' (the inductive-limit
  dual of Schwartz space), which is NOT a Banach topology.

WHAT WE CAN DO NOW (Computational Certificate):
  1. Empirical stability analysis with controlled perturbations
  2. Lipschitz constant estimation via test-function sampling
  3. Adaptive spectral re-projection to absorb perturbation error
  4. Counterexample search using Genesis-style exploration
  5. Numerical certificates of stability (not formal proofs yet)

WHAT THE DOCTORAL THESIS REQUIRES (Formal Proof):
  Prove: ∃ L_k > 0 such that ∀ η with |η| < δ:
    ‖R(T_η) - R(T)‖_{Φ'} ≤ L_k · |η|

  Where ‖·‖_{Φ'} is the inductive-limit topology defined by the
  seminorms p_m(T) = sup{|⟨T, φ⟩| : φ ∈ B_m} for B_m the unit ball
  in the C^m norm.

  KEY INSIGHT FOR THE PROOF:
  The action of a distribution with singularities at perturbed positions
  differs from the original by terms of the form:
    ⟨T_η - T, φ⟩ = Σ_{singularities} a_j · (φ^{(k_j)}(x_j+η) - φ^{(k_j)}(x_j))
                 = Σ a_j · η · φ^{(k_j+1)}(x_j) + O(η²)

  By the Mean Value Theorem applied to φ^{(k_j)}, each term is bounded
  by |η| · ‖φ‖_{C^{k_j+1}}. Since T has finite order k = max k_j,
  we get |⟨T_η - T, φ⟩| ≤ C · |η| · ‖φ‖_{C^{k+1}}.

  In the Φ' topology, this means ‖T_η - T‖_{-(k+1)} ≤ C|η|,
  which is Lipschitz continuity in the H^{-(k+1)} Sobolev norm.

  The challenge: this gives stability in H^{-(k+1)}, which is a Hilbert
  (Banach) topology. But Φ' requires stability in ALL C^m seminorms
  simultaneously. The inductive-limit topology is strictly finer.
  Bridging this gap requires the Schwartz structure theorem.

PROOF STRATEGY (Documented for Doctoral Work):
  Phase 1 (Sobolev): Prove Lipschitz in H^{-(k+1)} — achievable with
    standard Sobolev embedding and MVT arguments.
  Phase 2 (Structure): Use Schwartz structure theorem to write T as
    finite sum of derivatives of continuous functions. This localizes
    the perturbation effect.
  Phase 3 (Inductive Limit): Use the universal property of LF-spaces:
    a linear map from Φ' to any locally convex space is continuous
    iff its restriction to each step of the inductive limit is continuous.
    Since our representation lives in a fixed finite-order subspace,
    the inductive-limit topology coincides with H^{-(k+1)} on this subspace.

MATHEMATICAL REFERENCES:
  - Schwartz (1950-51): Théorie des distributions, Ch. III
  - Trèves (1967): Topological Vector Spaces, Distributions and Kernels, Ch. 13
  - Gel'fand & Shilov (1968): Generalized Functions Vol. 2, Ch. II
  - Hörmander (1990): The Analysis of Linear PDEs I, Ch. 8
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import special, optimize

from acf_functor.distribution_theory import (
    DualDistribution,
    DirectionalSingularity,
    SpectralTensor,
    SingularityType,
    dirac,
    heaviside,
    dirac_derivative,
    dirac_comb,
)


# ============================================================================
# STABILITY DATA TYPES
# ============================================================================

@dataclass
class PerturbationTrial:
    """Single perturbation experiment."""
    eta: float                     # Perturbation magnitude
    position_shift: np.ndarray     # Actual position shift applied
    delta_action: float            # ‖T_η - T‖ on test functions
    delta_spectral: float          # ‖c_η - c‖ on spectral coefficients
    delta_singularity_pos: float   # Euclidean distance in singularity positions
    consistency_before: float      # Consistency residual before perturbation
    consistency_after: float       # Consistency residual after perturbation
    lipschitz_estimate: float      # delta_action / |η|


@dataclass
class StabilityCertificate:
    """
    Computational certificate of stability (NOT a formal proof).

    Provides empirical evidence that the dual representation is
    Lipschitz-continuous under perturbations, with estimated
    constants and confidence bounds.
    """
    distribution_name: str
    max_order: int
    n_trials: int
    eta_range: Tuple[float, float]
    empirical_lipschitz: float          # max_i delta_action_i / |η_i|
    mean_lipschitz: float               # mean_i delta_action_i / |η_i|
    lipschitz_std: float                # standard deviation
    consistency_maintained: bool         # All trials pass consistency check
    worst_case_eta: float               # Eta causing largest deviation
    worst_case_delta: float             # Largest action deviation
    is_stable_empirically: bool          # Empirical verdict
    confidence: str                     # "high", "medium", "low"
    sobolev_bound: float                # Theoretical H^{-(k+1)} bound
    proof_sketch_progress: str          # Which phase of the formal proof is addressed
    trials: List[PerturbationTrial] = field(default_factory=list)
    counterexamples_found: List[Dict] = field(default_factory=list)


@dataclass
class PerturbationAnalysisResult:
    """Complete stability analysis with adaptive re-projection."""
    original: DualDistribution
    stability: StabilityCertificate
    adaptive_lipschitz_map: Dict[float, float]  # eta → Lipschitz constant
    re_projection_cost: float                    # FMA ops for re-projection
    recommended_perturbation_budget: float       # Max |η| before re-projection needed


# ============================================================================
# STABILITY ANALYZER
# ============================================================================

class StabilityAnalyzer:
    """
    Computational stability analysis for distribution representations.

    Implements the EMPIRICAL side of the stability problem:
    - Systematic perturbation of singularity positions
    - Measurement of action deviation on test functions
    - Estimation of Lipschitz constants
    - Detection of stability breakdown
    - Adaptive re-projection to restore consistency

    Also contains the MATHEMATICAL PROOF SKETCH as structured
    documentation for the formal proof development.
    """

    def __init__(
        self,
        n_test_functions: int = 20,
        n_quad_points: int = 200,
        perturbation_factors: Optional[List[float]] = None,
    ):
        self.n_test_functions = n_test_functions
        self.n_quad_points = n_quad_points
        self.perturbation_factors = perturbation_factors or [
            1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1
        ]
        self._test_funcs = self._build_test_function_library()

    def _build_test_function_library(self) -> List[Callable]:
        """Build a diverse library of test functions in Φ (Schwartz space)."""
        funcs = []

        # Gaussians at various scales
        for sigma in [0.05, 0.1, 0.2, 0.5]:
            s = sigma
            funcs.append(lambda x, s=s: np.exp(-0.5 * (x[0] / s) ** 2))

        # Bump functions (C^∞ with compact support)
        def bump(x, a=-0.5, b=0.5):
            t = np.clip(x[0], a, b)
            t_scaled = (t - a) / (b - a)
            if t_scaled <= 0 or t_scaled >= 1:
                return 0.0
            return float(np.exp(-1.0 / (t_scaled * (1.0 - t_scaled))))

        funcs.append(lambda x: bump(x, -0.8, 0.8))
        funcs.append(lambda x: bump(x, -0.3, 0.3))
        funcs.append(lambda x: bump(x, 0.0, 0.6))

        # Oscillatory functions
        for freq in [1, 2, 3, 5, 8]:
            f = freq
            funcs.append(lambda x, f=f: np.cos(2 * np.pi * f * x[0]))
            funcs.append(lambda x, f=f: np.sin(2 * np.pi * f * x[0]))

        # Products of Gaussian × oscillatory
        for sigma in [0.1, 0.3]:
            for freq in [2, 5]:
                s, f = sigma, freq
                funcs.append(
                    lambda x, s=s, f=f: np.exp(-0.5 * (x[0] / s) ** 2) * np.cos(2 * np.pi * f * x[0])
                )

        return funcs[:self.n_test_functions]

    def perturb_distribution(
        self,
        T: DualDistribution,
        eta: float,
        perturbation_mode: str = "random",
    ) -> DualDistribution:
        """
        Create a perturbed copy of T with singularity positions shifted by η.

        Args:
            T: Original distribution
            eta: Perturbation magnitude
            perturbation_mode:
              - "random": each singularity shifted by random direction × eta
              - "worst_case": all singularities shifted in same direction
              - "alternating": singularities shifted in alternating directions
        """
        import copy

        perturbed_singularities = []
        for i, sing in enumerate(T.singularities):
            if perturbation_mode == "random":
                direction = np.random.randn(len(sing.position))
                direction = direction / (np.linalg.norm(direction) + 1e-15)
            elif perturbation_mode == "worst_case":
                direction = np.ones(len(sing.position))
                direction = direction / np.linalg.norm(direction)
            elif perturbation_mode == "alternating":
                direction = np.ones(len(sing.position)) * ((-1) ** i)
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.ones(len(sing.position))
                direction = direction / np.linalg.norm(direction)

            new_pos = sing.position + eta * direction
            # Clamp to domain
            a, b = T.domain
            new_pos = np.clip(new_pos, a + 1e-10, b - 1e-10)

            perturbed_singularities.append(
                DirectionalSingularity(
                    position=new_pos,
                    codirection=sing.codirection.copy(),
                    order=sing.order,
                    mass=sing.mass,
                    singularity_type=sing.singularity_type,
                )
            )

        return DualDistribution(
            spectral=SpectralTensor(
                coefficients=T.spectral.coefficients.copy(),
                basis_type=T.spectral.basis_type,
                domain=T.domain,
                n_modes=len(T.spectral.coefficients),
                dimension=T.spectral.dimension,
            ),
            singularities=perturbed_singularities,
            domain=T.domain,
            distribution_type=f"{T.distribution_type}_perturbed_η={eta:.1e}",
        )

    def measure_action_deviation(
        self,
        T_original: DualDistribution,
        T_perturbed: DualDistribution,
    ) -> float:
        """
        Measure ‖T_η - T‖ empirically via maximum deviation on test functions.

        ‖T_η - T‖_{emp} = max_{φ ∈ test_library} |⟨T_η, φ⟩ - ⟨T, φ⟩|
        """
        max_deviation = 0.0
        for phi in self._test_funcs[:10]:  # Use subset for speed
            orig_action = T_original.act_on_test_function(
                lambda x: complex(float(phi(x))),
                n_quad=self.n_quad_points,
            )
            pert_action = T_perturbed.act_on_test_function(
                lambda x: complex(float(phi(x))),
                n_quad=self.n_quad_points,
            )
            deviation = abs(orig_action - pert_action)
            max_deviation = max(max_deviation, float(deviation.real))

        return max_deviation

    def compute_sobolev_bound(
        self,
        T: DualDistribution,
        eta: float,
    ) -> float:
        """
        Compute the theoretical H^{-(k+1)} stability bound.

        For a distribution of max order k:
          ‖T_η - T‖_{H^{-(k+1)}} ≤ C · |η| · Σ |a_j| · (k_j + 1)

        where a_j are singularity masses and k_j their orders.
        """
        if not T.singularities:
            return 0.0

        max_k = max(s.order for s in T.singularities)
        total_mass = sum(
            abs(float(np.asarray(s.mass).ravel()[0]))
            for s in T.singularities
        )

        # C depends on Sobolev embedding constant for the domain
        domain_width = T.domain[1] - T.domain[0]
        C_sob = 1.0 / (domain_width ** (max_k + 0.5) + 1e-15)

        return C_sob * total_mass * (max_k + 1) * abs(eta)

    def run_stability_analysis(
        self,
        T: DualDistribution,
        perturbation_mode: str = "random",
        n_repeats: int = 3,
    ) -> StabilityCertificate:
        """
        Full stability analysis with controlled perturbations.

        Returns empirical Lipschitz constants and stability certificate.
        """
        trials = []
        max_lip = 0.0
        lip_values = []
        worst_eta = 0.0
        worst_delta = 0.0
        counterexamples = []

        for eta in self.perturbation_factors:
            for _ in range(n_repeats):
                T_pert = self.perturb_distribution(T, eta, perturbation_mode)

                delta_action = self.measure_action_deviation(T, T_pert)
                delta_spectral = np.linalg.norm(
                    T.spectral.coefficients - T_pert.spectral.coefficients
                )
                delta_pos = 0.0
                for s1, s2 in zip(T.singularities, T_pert.singularities):
                    delta_pos += np.linalg.norm(s1.position - s2.position)
                if T.singularities:
                    delta_pos /= len(T.singularities)

                consistency_before = T.check_consistency()
                consistency_after = T_pert.check_consistency()

                lip = delta_action / max(abs(eta), 1e-20)

                trial = PerturbationTrial(
                    eta=eta,
                    position_shift=np.array([eta]),
                    delta_action=delta_action,
                    delta_spectral=delta_spectral,
                    delta_singularity_pos=delta_pos,
                    consistency_before=consistency_before,
                    consistency_after=consistency_after,
                    lipschitz_estimate=lip,
                )
                trials.append(trial)
                lip_values.append(lip)

                if lip > max_lip:
                    max_lip = lip
                    worst_eta = eta
                    worst_delta = delta_action

                # Detect potential counterexamples (non-Lipschitz behavior)
                if eta > 1e-6 and lip > 10.0 * max(1.0, np.mean(lip_values) if lip_values else 1.0):
                    counterexamples.append({
                        "eta": eta,
                        "lipschitz": lip,
                        "delta_action": delta_action,
                        "suspected_breakdown": True,
                        "note": "Lipschitz constant spike detected — possible stability boundary",
                    })

        mean_lip = float(np.mean(lip_values)) if lip_values else 0.0
        std_lip = float(np.std(lip_values)) if lip_values else 0.0

        # Determine confidence
        if std_lip < 0.1 * mean_lip and mean_lip < 1e3:
            confidence = "high"
            is_stable = True
        elif std_lip < 0.5 * mean_lip:
            confidence = "medium"
            is_stable = mean_lip < 1e4
        else:
            confidence = "low"
            is_stable = mean_lip < 1e3

        sobolev_bound = self.compute_sobolev_bound(T, worst_eta)

        max_order = max((s.order for s in T.singularities), default=0)

        return StabilityCertificate(
            distribution_name=T.distribution_type,
            max_order=max_order,
            n_trials=len(trials),
            eta_range=(self.perturbation_factors[0], self.perturbation_factors[-1]),
            empirical_lipschitz=max_lip,
            mean_lipschitz=mean_lip,
            lipschitz_std=std_lip,
            consistency_maintained=all(
                t.consistency_after < 10.0 * max(t.consistency_before, 0.01)
                for t in trials
            ),
            worst_case_eta=worst_eta,
            worst_case_delta=worst_delta,
            is_stable_empirically=is_stable,
            confidence=confidence,
            sobolev_bound=sobolev_bound,
            proof_sketch_progress="Phase 1 (Sobolev): empirical validation; Phase 2-3: pending formal proof",
            trials=trials,
            counterexamples_found=counterexamples,
        )

    def adaptive_reproject(
        self,
        T: DualDistribution,
        eta: float,
        n_modes: int = 64,
    ) -> DualDistribution:
        """
        Adaptively re-project spectral coefficients after perturbation.

        When singularities move by η, the spectral part must be adjusted
        to maintain the consistency condition. This uses Leibniz rule:
          D(T_η) = D(T_reg) + Σ (shifted and order-elevated singularities)

        The re-projection distributes the perturbation error between
        spectral and singularity tensors optimally.
        """
        # Create perturbed version
        T_pert = self.perturb_distribution(T, eta, "random")

        # Re-fit spectral part to absorb perturbation
        domain = T.domain
        x_grid = np.linspace(domain[0], domain[1], n_modes * 2)
        orig_vals = T.evaluate_approximate(x_grid)
        pert_vals = T_pert.evaluate_approximate(x_grid)

        # The difference must be absorbed by the spectral tensor
        diff_vals = pert_vals - orig_vals

        # Project difference onto Chebyshev
        diff_coeffs = np.zeros(n_modes)
        N = len(x_grid)
        for k in range(n_modes):
            cos_vals = np.cos(np.pi * k * np.arange(N) / (N - 1))
            ck = 2.0 / (N - 1)
            diff_coeffs[k] = ck * np.sum(diff_vals * cos_vals)
        diff_coeffs[0] *= 0.5

        # Adjust spectral coefficients
        orig_len = len(T.spectral.coefficients)
        adjusted_coeffs = T.spectral.coefficients.copy()
        min_len = min(len(adjusted_coeffs), len(diff_coeffs))
        adjusted_coeffs[:min_len] += diff_coeffs[:min_len]

        return DualDistribution(
            spectral=SpectralTensor(
                coefficients=adjusted_coeffs,
                basis_type=T.spectral.basis_type,
                domain=T.domain,
                n_modes=max(orig_len, n_modes),
            ),
            singularities=T_pert.singularities,
            domain=T.domain,
            distribution_type=f"{T.distribution_type}_reprojected",
        )

    def search_counterexamples(
        self,
        T: DualDistribution,
        eta_range: Tuple[float, float] = (1e-6, 1.0),
        n_search: int = 50,
    ) -> List[Dict]:
        """
        Genesis-style search for perturbation magnitudes that break stability.

        Uses random sampling and local refinement to find η values where
        the Lipschitz estimate spikes, indicating potential stability boundaries.
        """
        eta_min, eta_max = eta_range
        findings = []

        # Coarse grid search
        for _ in range(n_search):
            eta = 10 ** np.random.uniform(np.log10(eta_min), np.log10(eta_max))
            T_pert = self.perturb_distribution(T, eta, "random")
            delta = self.measure_action_deviation(T, T_pert)
            lip = delta / max(eta, 1e-20)

            findings.append({
                "eta": float(eta),
                "delta_action": float(delta),
                "lipschitz": float(lip),
            })

        # Sort by Lipschitz (largest first = potential problems)
        findings.sort(key=lambda x: x["lipschitz"], reverse=True)

        # Refine around top candidates
        refined = []
        for finding in findings[:5]:
            eta0 = finding["eta"]
            for frac in [0.5, 0.9, 1.0, 1.1, 1.5]:
                eta_test = eta0 * frac
                if eta_min <= eta_test <= eta_max:
                    T_pert = self.perturb_distribution(T, eta_test, "random")
                    delta = self.measure_action_deviation(T, T_pert)
                    lip = delta / max(eta_test, 1e-20)
                    refined.append({
                        "eta": float(eta_test),
                        "delta_action": float(delta),
                        "lipschitz": float(lip),
                        "refined_from": float(eta0),
                    })

        return refined[:20]

    def full_analysis(
        self,
        T: DualDistribution,
        name: str = "",
    ) -> PerturbationAnalysisResult:
        """
        Complete stability analysis pipeline:
        1. Run perturbation trials
        2. Estimate Lipschitz constants
        3. Search for counterexamples
        4. Test adaptive re-projection
        """
        if not name:
            name = T.distribution_type

        # Phase 1: Stability analysis
        stability = self.run_stability_analysis(T)

        # Phase 2: Build Lipschitz map
        lip_map = {}
        for trial in stability.trials:
            eta_key = float(trial.eta)
            if eta_key not in lip_map:
                lip_map[eta_key] = []
            lip_map[eta_key].append(trial.lipschitz_estimate)

        # Average per eta
        lip_map_avg = {
            eta: float(np.mean(vals)) for eta, vals in lip_map.items()
        }

        # Phase 3: Counterexample search
        counterexamples = self.search_counterexamples(T)

        # Phase 4: Adaptive re-projection test
        T_reproj = self.adaptive_reproject(T, 1e-4)
        reproj_cost = len(T_reproj.spectral.coefficients)  # ~FMA ops

        # Recommended perturbation budget
        # The largest η where Lipschitz estimate is still well-behaved
        sorted_lip = sorted(lip_map_avg.items(), key=lambda x: x[1])
        rec_budget = 1e-4  # Conservative default
        for eta, lip in sorted_lip:
            if lip < stability.empirical_lipschitz * 0.5:
                rec_budget = max(rec_budget, eta)

        return PerturbationAnalysisResult(
            original=T,
            stability=stability,
            adaptive_lipschitz_map=lip_map_avg,
            re_projection_cost=reproj_cost,
            recommended_perturbation_budget=rec_budget,
        )


# ============================================================================
# PROOF SKETCH — Documented Structure for the Formal Proof
# ============================================================================

PROOF_SKETCH_STABILITY = """
PROOF SKETCH: Lipschitz Stability of Dual Representation in Φ'
===============================================================

THEOREM (to be proved):
  Let T be a tempered distribution of finite order k with compact support
  K ⊂ ℝ^d. Let R(T) = (c, S) be its dual representation where:
    - c = {c_α} are Chebyshev/wavelet spectral coefficients (the regular part)
    - S = {(x_j, ω_j, k_j, a_j)} are directional singularities

  Then ∃ L_k, δ > 0 such that ∀ η with |η| < δ:
    ‖R(T_η) - R(T)‖_{Φ'} ≤ L_k · |η|

  where T_η is T with each singularity position perturbed by at most η,
  and ‖·‖_{Φ'} is the strong topology of the distribution space.

PROOF STRUCTURE:

LEMMA 1 (Sobolev Stability):
  ‖T_η - T‖_{H^{-(k+1)}(K)} ≤ C_1(k, K) · |η| · Σ_j |a_j| · (k_j + 1)
  
  Proof: For any test function φ ∈ H^{k+1}(K),
    |⟨T_η - T, φ⟩| = |Σ_j a_j [φ^{(k_j)}(x_j+η_j) - φ^{(k_j)}(x_j)]|
                   ≤ Σ_j |a_j| · |η_j| · ‖φ‖_{C^{k_j+1}}   (by MVT)
                   ≤ |η| · Σ_j |a_j| · C_Sob(k_j) · ‖φ‖_{H^{k+1}}
  where the Sobolev embedding H^{k+1} ↪ C^{k_j+1} provides C_Sob(k_j).
  Hence ‖T_η - T‖_{H^{-(k+1)}} ≤ C_1 · |η|. QED.

LEMMA 2 (Structure Localization):
  By the Schwartz structure theorem (Schwartz 1950, Thm. XXI),
  T = Σ_{|α|≤k} ∂^α f_α where each f_α is a continuous function with
  support in an arbitrarily small neighborhood of K.
  
  The dual representation R(T) localizes each f_α to the patch containing
  its support, and the singularities S correspond to the jump set of f_α.

LEMMA 3 (Inductive Limit Reduction):
  The space of distributions of order ≤ k with support in K,
  denoted Φ'_k(K), is a BANACH space when equipped with the norm
    ‖T‖_{Φ'_k(K)} = sup{|⟨T, φ⟩| : ‖φ‖_{C^k} ≤ 1, supp(φ) ⊂ K_ε}
  
  On this subspace, the Φ' inductive-limit topology COINCIDES with
  the Banach topology. (Trèves 1967, Prop. 13.1)
  
  Since our representation R(T) only involves singularities of order ≤ k,
  the entire perturbation analysis lives in Φ'_k(K), which is Banach.
  
  Therefore: the Lipschitz estimate from Lemma 1 in H^{-(k+1)} suffices,
  because on Φ'_k(K), the H^{-(k+1)} norm is equivalent to the Φ' topology.

PROOF OF MAIN THEOREM:
  By Lemma 3, it suffices to prove stability in the Banach space Φ'_k(K).
  By Lemma 1, the action difference is O(|η|) in H^{-(k+1)}.
  By the equivalence of norms on Φ'_k(K) (Lemma 3), this implies O(|η|)
  in the Φ' topology with Lipschitz constant L_k = C_1(k, K) · C_equiv.

  The spectral part (coefficients c) is Lipschitz because the Chebyshev
  projection operator Π_N: Φ'_k(K) → ℝ^N is continuous.
  The singularity tensor S is Lipschitz because positions depend
  continuously on η by construction.

GAPS TO FORMAL PROOF:
  1. Lemma 3 requires constructing the Banach space Φ'_k(K) explicitly
     and proving norm equivalence. This needs Schwartz's structure theorem
     in a constructive form (not just existential).
  2. The Sobolev embedding constants C_Sob(k_j) must be computed explicitly
     for the given domain geometry (not just known to exist).
  3. The Chebyshev projection continuity constant must be bounded in terms
     of the mesh size and polynomial degree (Lebesgue constant estimates).

STATUS (May 2026):
  ✓ Computational evidence: empirical Lipschitz constants measured
  ✓ Lemma 1: MVT argument is standard, Sobolev embedding is classical
  ○ Lemma 2: Schwartz structure theorem exists but not in Mathlib
  ○ Lemma 3: Banach topology on Φ'_k(K) needs formal construction
  ○ Main proof: Depends on Lemmas 2-3
"""


# ============================================================================
# CONVENIENCE
# ============================================================================

def analyze_stability(
    T: DualDistribution,
    name: str = "",
    n_test_functions: int = 10,
    perturbation_mode: str = "random",
) -> PerturbationAnalysisResult:
    """Convenience wrapper for full stability analysis."""
    analyzer = StabilityAnalyzer(n_test_functions=n_test_functions)
    return analyzer.full_analysis(T, name)
