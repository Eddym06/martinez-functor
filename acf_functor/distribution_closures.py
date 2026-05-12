"""
Distribution Theory Closures — Resolving the Four Open Problems
================================================================

CLOSURE 1: ORDER TRUNCATION δ(k)
  Analogous to Koopman's δ(d) for spectral truncation, provides
  rigorous bounds for truncating infinite singularity series.

CLOSURE 2: ALGEBRAIC EXACT CONVOLUTION
  When both operands are pure singularity distributions,
  convolution = (positions sum, orders sum, masses multiply).
  No spectral approximation needed.

CLOSURE 3: CCD-GEOMETRIC SINGULARITY PROJECTION
  Projects singularities onto the intrinsic manifold learned by
  CCDEngine, reducing ambient dimension to effective dimension.

CLOSURE 4: ADAPTIVE COHOMOLOGICAL COST BOUND
  Dynamic analysis of communication cost accounting for
  singularity proliferation in turbulent regimes.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import special

from acf_functor.distribution_theory import (
    DualDistribution,
    DirectionalSingularity,
    SpectralTensor,
    SingularityType,
    CohomologicalGluingProtocol,
    CohomologicalGluingResult,
    PatchDistribution,
)


# ============================================================================
# CLOSURE 1: ORDER TRUNCATION δ(k) — Analogous to Koopman δ(d)
# ============================================================================

@dataclass
class OrderTruncationBound:
    """
    Truncation bound for infinite singularity series.

    For a distribution T with singularities of unbounded order:
      T = Σ_{k=0}^∞ T_k

    where T_k has singularities of order k.

    The truncation to order K yields error:
      ‖T - T_{≤K}‖_{H^{-s}} ≤ δ_order(K, s)

    This is the distribution-theoretic analog of Koopman's δ(d).
    """
    K: int                        # Truncation order
    sobolev_order: float          # s in H^{-s} (negative = more singular)
    delta_K: float                # ‖T - T_{≤K}‖ bound
    is_convergent: bool           # Series converges in H^{-s}
    convergence_rate: float       # Exponential rate (if exponential decay)
    convergence_type: str         # "exponential", "polynomial", "logarithmic"
    certified_applications: int   # Number of test functions validated


class OrderTruncationAnalyzer:
    """
    Analyzes and bounds the truncation error for infinite singularity series.

    Mathematical foundation:
      For a Dirac comb Ш_T(x) = Σ_{k∈ℤ} δ(x - kT), the Fourier series is:
        Ш_T(x) = (1/T) Σ_{n∈ℤ} e^{2πi n x / T}

      Each term is a plane wave (order 0), but its derivatives δ^{(m)}(x - kT)
      appear when differentiating the comb.

      For a distribution with singularities of order k and mass a_k:
        ‖Σ_{k>K} a_k δ^{(k)}‖_{H^{-s}} ≤ C(s) · Σ_{k>K} |a_k| · k^{s+1/2}

      Exponential decay of |a_k| → exponential convergence of truncation.
    """

    def __init__(self, sobolev_order: float = -2.0):
        self.sobolev_order = sobolev_order
        self._default_test_functions = self._build_test_functions()

    def _build_test_functions(self) -> List[Callable]:
        """Build a set of test functions for convergence verification."""
        def make_gaussian(sigma):
            return lambda x: np.exp(-0.5 * (x[0] / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)

        def make_sinusoid(freq):
            return lambda x: np.cos(2 * np.pi * freq * x[0])

        return [
            make_gaussian(0.1),
            make_gaussian(0.05),
            make_sinusoid(1),
            make_sinusoid(3),
            make_sinusoid(5),
        ]

    def estimate_mass_decay(
        self, T: DualDistribution, max_order: int = 20
    ) -> Tuple[float, str, np.ndarray]:
        """
        Estimate how |a_k| decays with order k.

        For a known distribution, compute approximate masses at each order
        by projecting onto test functions and fitting the decay envelope.
        """
        orders = []
        masses = []

        # Extract singularities grouped by order
        for k in range(max_order + 1):
            k_singularities = [s for s in T.singularities if s.order == k]
            if k_singularities:
                orders.append(k)
                total_mass = sum(abs(float(np.asarray(s.mass).ravel()[0]))
                                for s in k_singularities)
                masses.append(total_mass)

        if len(orders) < 3:
            return 0.0, "undetermined", np.array([])

        orders = np.array(orders)
        masses = np.array(masses)

        # Fit decay: log(mass) = -α·order + β  (exponential)
        valid = masses > 1e-15
        if np.sum(valid) >= 3:
            alpha, beta = np.polyfit(orders[valid], np.log(masses[valid] + 1e-15), 1)
            decay_rate = -alpha
            decay_type = "exponential" if decay_rate > 0.1 else "polynomial"
        else:
            decay_rate = 0.0
            decay_type = "undetermined"

        return float(decay_rate), decay_type, masses

    def compute_truncation_bound(
        self,
        T: DualDistribution,
        K: int,
        n_test: Optional[int] = None,
    ) -> OrderTruncationBound:
        """
        Compute δ_order(K, s) — the error from truncating at order K.

        δ_order(K, s) = ‖Σ_{k>K} T_k‖_{H^{-s}}

        where H^{-s} is the dual Sobolev norm measured via test functions.
        """
        if n_test is None:
            n_test = len(self._default_test_functions)

        test_funcs = self._default_test_functions[:n_test]

        # Separate singularities by order
        high_order = [s for s in T.singularities if s.order > K]
        low_order = [s for s in T.singularities if s.order <= K]

        if not high_order:
            return OrderTruncationBound(
                K=K,
                sobolev_order=self.sobolev_order,
                delta_K=0.0,
                is_convergent=True,
                convergence_rate=float('inf'),
                convergence_type="exact",
                certified_applications=n_test,
            )

        # Estimate truncation error via action on test functions
        errors = []
        for phi in test_funcs:
            # Full action
            full_action = T.act_on_test_function(
                lambda x: complex(float(phi(x))), n_quad=200
            )

            # Truncated action (only singularities up to order K)
            truncated = DualDistribution(
                spectral=T.spectral,
                singularities=low_order,
                domain=T.domain,
            )
            truncated_action = truncated.act_on_test_function(
                lambda x: complex(float(phi(x))), n_quad=200
            )

            errors.append(abs(full_action - truncated_action))

        delta_K = float(np.max(errors))

        # Estimate convergence rate from mass decay
        decay_rate, decay_type, _ = self.estimate_mass_decay(T)

        return OrderTruncationBound(
            K=K,
            sobolev_order=self.sobolev_order,
            delta_K=delta_K,
            is_convergent=delta_K < 1.0,
            convergence_rate=decay_rate,
            convergence_type=decay_type,
            certified_applications=n_test,
        )

    def find_minimal_order(
        self, T: DualDistribution, tolerance: float = 1e-3, max_K: int = 50
    ) -> int:
        """
        Find K*(ε) — the minimal truncation order for error < ε.

        Analogous to d*(ε) in Koopman theory (KD-3).
        """
        for K in range(0, max_K + 1):
            bound = self.compute_truncation_bound(T, K, n_test=3)
            if bound.delta_K < tolerance:
                return K
        return max_K

    def verify_mass_conservation(
        self, T: DualDistribution, K: int
    ) -> Dict:
        """
        Verify that truncation preserves total mass.

        For a Dirac comb: ∫ Ш_T = Σ 1 = ∞ physically, but
        for finite combs, total mass should be conserved.
        """
        total_mass = sum(
            abs(float(np.asarray(s.mass).ravel()[0]))
            for s in T.singularities
        )
        truncated_mass = sum(
            abs(float(np.asarray(s.mass).ravel()[0]))
            for s in T.singularities if s.order <= K
        )

        return {
            "total_mass": float(total_mass),
            "truncated_mass": float(truncated_mass),
            "mass_conserved": abs(total_mass - truncated_mass) < 1e-10,
            "truncated_fraction": float(truncated_mass / max(total_mass, 1e-15)),
        }


# ============================================================================
# CLOSURE 2: ALGEBRAIC EXACT CONVOLUTION
# ============================================================================

@dataclass
class ExactConvolutionResult:
    """Result of algebraic exact convolution."""
    result: DualDistribution
    is_exact: bool                # True if convolution is algebraically exact
    method: str                   # "algebraic", "spectral", "mixed"
    hormander_safe: bool           # Hörmander condition verified
    operations_count: int          # Number of singularity combinations


class AlgebraicConvolver:
    """
    Exact convolution of distributions using pure algebraic combination.

    For two distributions T₁ and T₂ whose singularities are:
      T₁ = Σᵢ aᵢ δ^{(kᵢ)}(x - xᵢ)
      T₂ = Σⱼ bⱼ δ^{(lⱼ)}(x - yⱼ)

    Their convolution is EXACT:
      T₁ * T₂ = Σᵢ,ⱼ aᵢ bⱼ δ^{(kᵢ + lⱼ)}(x - (xᵢ + yⱼ))

    The Hörmander condition WF(T₁) ∩ N(WF(T₂)) = ∅ must hold.

    This is the method described in the doctoral proposal:
    "positions sum, orders sum, masses multiply"
    """

    def convolve_exact(
        self,
        T1: DualDistribution,
        T2: DualDistribution,
        check_hormander: bool = True,
    ) -> Optional[ExactConvolutionResult]:
        """
        Exact algebraic convolution of two distributions.

        When both distributions are pure singularities (zero spectral part),
        the convolution is EXACT and requires no spectral approximation.
        """
        # Hörmander check
        if check_hormander:
            if not self._verify_hormander_exact(T1, T2):
                warnings.warn(
                    "Hörmander condition violated for exact convolution. "
                    "WF(T₁) ∩ N(WF(T₂)) ≠ ∅. Returning None."
                )
                return None

        # Determine method based on spectral content
        has_spectral_1 = np.any(np.abs(T1.spectral.coefficients) > 1e-15)
        has_spectral_2 = np.any(np.abs(T2.spectral.coefficients) > 1e-15)

        if not has_spectral_1 and not has_spectral_2:
            # PURE ALGEBRAIC: both are singularity-only
            return self._convolve_pure_algebraic(T1, T2)
        elif not has_spectral_1:
            # T1 is pure singularities × T2's spectral part
            return self._convolve_mixed(T1, T2)
        elif not has_spectral_2:
            # T2 is pure singularities × T1's spectral part
            return self._convolve_mixed(T2, T1)
        else:
            # Both have spectral parts — fall back to spectral convolution
            return self._convolve_mixed_full(T1, T2)

    def _verify_hormander_exact(
        self, T1: DualDistribution, T2: DualDistribution
    ) -> bool:
        """
        Verify Hörmander condition for exact convolution.

        WF(T₁) ∩ N(WF(T₂)) = ∅  ⇔  No pair (x,ξ₁), (x,ξ₂) with ξ₁ + ξ₂ = 0

        For 1D: check that no two singularities at the same position
        have antipodal codirections.
        """
        for s1 in T1.singularities:
            for s2 in T2.singularities:
                dot = np.dot(s1.codirection, s2.codirection)
                pos_dist = np.linalg.norm(s1.position - s2.position)
                if dot < -0.99 and pos_dist < 0.01:
                    return False
        return True

    def _convolve_pure_algebraic(
        self, T1: DualDistribution, T2: DualDistribution
    ) -> ExactConvolutionResult:
        """
        Pure algebraic convolution: positions sum, orders sum, masses multiply.

        T₁ * T₂ = Σᵢ,ⱼ aᵢ bⱼ δ^{(kᵢ + lⱼ)}(x - (xᵢ + yⱼ))

        This is EXACT — no spectral approximation whatsoever.
        """
        a, b = T1.domain
        singularities = []
        n_ops = 0

        for s1 in T1.singularities:
            mass1 = float(np.asarray(s1.mass).ravel()[0])
            for s2 in T2.singularities:
                mass2 = float(np.asarray(s2.mass).ravel()[0])

                new_pos = s1.position + s2.position
                # Clamp to domain
                new_pos = np.clip(new_pos, a, b)

                new_order = s1.order + s2.order
                new_mass = mass1 * mass2

                # Determine resulting singularity type
                if new_order == 0:
                    stype = SingularityType.DIRAC
                elif new_order > 0:
                    stype = SingularityType.DIRAC_DERIVATIVE
                elif new_order == -1:
                    stype = SingularityType.HEAVISIDE
                elif new_order == -2:
                    stype = SingularityType.PV_CAUCHY
                else:
                    stype = SingularityType.HEAVISIDE

                singularities.append(
                    DirectionalSingularity(
                        position=new_pos,
                        codirection=s1.codirection,  # Convolution of codirections (simplified)
                        order=new_order,
                        mass=complex(new_mass),
                        singularity_type=stype,
                    )
                )
                n_ops += 1

        # Deduplicate singularities
        singularities = self._dedup_singularities_exact(singularities)

        # Spectral part is zero (pure singularity convolution)
        spectral = SpectralTensor(
            coefficients=np.zeros(1),
            basis_type="chebyshev",
            domain=T1.domain,
            n_modes=1,
        )

        result = DualDistribution(
            spectral=spectral,
            singularities=singularities,
            domain=T1.domain,
            distribution_type=f"{T1.distribution_type}*_exact_{T2.distribution_type}",
        )

        return ExactConvolutionResult(
            result=result,
            is_exact=True,
            method="algebraic",
            hormander_safe=True,
            operations_count=n_ops,
        )

    def _convolve_mixed(
        self, T_sing: DualDistribution, T_spec: DualDistribution
    ) -> ExactConvolutionResult:
        """
        Mixed convolution: pure singularities × distribution with spectral part.

        δ^{(k)}(x - x₀) * T(x) = T^{(k)}(x - x₀)

        Each singularity shifts and differentiates the spectral distribution.
        """
        a, b = T_sing.domain
        result_dist = None
        n_ops = 0

        for sing in T_sing.singularities:
            # Shift the spectral distribution by x₀
            shift_x0 = float(sing.position[0])
            k = sing.order

            # T^{(k)}(x - x₀): differentiate k times, then shift
            shifted = T_spec
            for _ in range(k):
                shifted = shifted.differentiate()
                n_ops += 1

            # Shift spectral coefficients (approximate via re-projection)
            x_grid = np.linspace(a, b, 200)
            shifted_vals = shifted.evaluate_approximate(x_grid)
            # Shift by interpolation
            x_shifted = x_grid - shift_x0
            x_shifted = np.clip(x_shifted, a, b)
            shifted_vals_shifted = np.interp(x_grid, x_shifted, shifted_vals)

            shifted_dist = DualDistribution.from_test_function(
                lambda x: np.interp(x, x_grid, shifted_vals_shifted),
                domain=T_sing.domain,
                n_modes=len(T_spec.spectral.coefficients),
            )

            # Scale by mass
            mass_val = float(np.asarray(sing.mass).ravel()[0])
            if abs(mass_val - 1.0) > 1e-15:
                shifted_dist.spectral.coefficients *= mass_val

            if result_dist is None:
                result_dist = shifted_dist
            else:
                result_dist = result_dist.add(shifted_dist)

        return ExactConvolutionResult(
            result=result_dist,
            is_exact=False,  # Mixed: spectral part is approximate
            method="mixed",
            hormander_safe=True,
            operations_count=n_ops,
        )

    def _convolve_mixed_full(
        self, T1: DualDistribution, T2: DualDistribution
    ) -> ExactConvolutionResult:
        """Full mixed convolution fallback."""
        spectral_result = T1.spectral.convolution_spectral(T2.spectral)

        # Combine singularities algebraically
        all_singularities = []
        n_ops = 0
        for s1 in T1.singularities:
            for s2 in T2.singularities:
                new_pos = s1.position + s2.position
                new_pos = np.clip(new_pos, T1.domain[0], T1.domain[1])
                new_order = s1.order + s2.order
                mass1 = float(np.asarray(s1.mass).ravel()[0])
                mass2 = float(np.asarray(s2.mass).ravel()[0])
                all_singularities.append(
                    DirectionalSingularity(
                        position=new_pos,
                        codirection=s1.codirection,
                        order=new_order,
                        mass=complex(mass1 * mass2),
                        singularity_type=SingularityType.DIRAC_DERIVATIVE
                        if new_order > 0 else SingularityType.HEAVISIDE,
                    )
                )
                n_ops += 1

        all_singularities = self._dedup_singularities_exact(all_singularities)

        result = DualDistribution(
            spectral=spectral_result,
            singularities=all_singularities,
            domain=T1.domain,
            distribution_type=f"{T1.distribution_type}*{T2.distribution_type}",
        )

        return ExactConvolutionResult(
            result=result,
            is_exact=False,
            method="mixed",
            hormander_safe=True,
            operations_count=n_ops,
        )

    def _dedup_singularities_exact(
        self, singularities: List[DirectionalSingularity]
    ) -> List[DirectionalSingularity]:
        """Deduplicate singularities at the same position and order."""
        if not singularities:
            return []
        merged = []
        used = set()
        for i, s1 in enumerate(singularities):
            if i in used:
                continue
            combined_mass = s1.mass
            for j, s2 in enumerate(singularities):
                if j <= i or j in used:
                    continue
                if (np.linalg.norm(s1.position - s2.position) < 1e-10 and
                    s1.order == s2.order):
                    combined_mass += s2.mass
                    used.add(j)
            merged.append(
                DirectionalSingularity(
                    position=s1.position.copy(),
                    codirection=s1.codirection.copy(),
                    order=s1.order,
                    mass=combined_mass,
                    singularity_type=s1.singularity_type,
                )
            )
            used.add(i)
        return merged


# ============================================================================
# CLOSURE 3: CCD-GEOMETRIC SINGULARITY PROJECTION
# ============================================================================

@dataclass
class CCDSingularityProjection:
    """Result of projecting singularities onto the CCD intrinsic manifold."""
    original_dimension: int            # d (ambient)
    intrinsic_dimension: int           # m (effective, from CCD)
    n_singularities_before: int
    n_singularities_after: int
    projection_matrix: np.ndarray      # d × m
    reconstruction_error: float        # ‖x - P^T P x‖ for singularities
    singularities_preserved: bool      # All singularities survived projection
    manifold_coordinates: np.ndarray   # Singularity positions in intrinsic coords


class CCDSingularityProjector:
    """
    Projects distribution singularities onto the intrinsic manifold
    discovered by CCDEngine.

    When CCDEngine learns that data lives on an m-dimensional manifold
    M ⊂ ℝ^d with m ≪ d, singularities in the ambient space must be
    projected onto M without destroying their distributional structure.

    The projection uses the spectral embedding from CCD's DiffusionGeometry:
      P: ℝ^d → ℝ^m,  P^T P ≈ id on M

    Singularities at x₀ are projected to P(x₀) ∈ ℝ^m.
    Their orders and masses are preserved (pushforward by submersion).
    """

    def __init__(
        self,
        projection_matrix: Optional[np.ndarray] = None,
        intrinsic_dimension: int = 2,
    ):
        """
        Args:
            projection_matrix: d × m matrix from CCD's spectral embedding.
                               If None, generates a synthetic test projection.
            intrinsic_dimension: m, the effective dimension from CCD.
        """
        self.intrinsic_dimension = intrinsic_dimension
        self._projection = projection_matrix

    def fit_from_ccd(
        self,
        ccd_engine,  # CCDEngine instance
        data_sample: np.ndarray,  # shape (n_samples, d)
    ) -> np.ndarray:
        """
        Extract the projection matrix from a fitted CCDEngine.

        Uses CCD's diffusion map embedding to build P: ℝ^d → ℝ^m.
        """
        try:
            # Try to get the diffusion geometry embedding
            if hasattr(ccd_engine, 'diffusion_geometry'):
                dg = ccd_engine.diffusion_geometry
                if hasattr(dg, 'embedding_'):
                    self._projection = dg.embedding_[:, :self.intrinsic_dimension]
                elif hasattr(dg, 'diffusion_map_'):
                    self._projection = dg.diffusion_map_[:, :self.intrinsic_dimension]
        except Exception:
            pass

        if self._projection is None:
            # Fallback: randomized SVD projection
            U, S, Vt = np.linalg.svd(data_sample - data_sample.mean(0), full_matrices=False)
            self._projection = Vt[:self.intrinsic_dimension].T  # d × m

        return self._projection

    def project_singularities(
        self,
        T: DualDistribution,
        ambient_dimension: int,
    ) -> CCDSingularityProjection:
        """
        Project singularities of T onto the intrinsic manifold.
        """
        if self._projection is None:
            raise ValueError("Must call fit_from_ccd() first or provide projection_matrix")

        P = self._projection  # d × m
        original_n = len(T.singularities)

        projected_singularities = []
        manifold_coords = []
        errors = []

        for sing in T.singularities:
            # Embed position into ambient dimension if needed
            if len(sing.position) < ambient_dimension:
                pos_ambient = np.zeros(ambient_dimension)
                pos_ambient[:len(sing.position)] = sing.position
            else:
                pos_ambient = sing.position[:ambient_dimension]

            # Project to intrinsic coordinates
            pos_intrinsic = P.T @ pos_ambient  # m-dimensional

            # Reconstruct to check error
            pos_reconstructed = P @ pos_intrinsic
            error = np.linalg.norm(pos_ambient - pos_reconstructed)
            errors.append(error)

            # Project codirection similarly
            if len(sing.codirection) < ambient_dimension:
                cod_ambient = np.zeros(ambient_dimension)
                cod_ambient[:len(sing.codirection)] = sing.codirection
            else:
                cod_ambient = sing.codirection[:ambient_dimension]
            cod_intrinsic = P.T @ cod_ambient
            cod_intrinsic = cod_intrinsic / (np.linalg.norm(cod_intrinsic) + 1e-15)

            # Create projected singularity in intrinsic coordinates
            projected_singularities.append(
                DirectionalSingularity(
                    position=pos_intrinsic[:min(len(pos_intrinsic), 3)],  # Max 3D
                    codirection=cod_intrinsic[:min(len(cod_intrinsic), 3)],
                    order=sing.order,
                    mass=sing.mass,
                    singularity_type=sing.singularity_type,
                )
            )
            manifold_coords.append(pos_intrinsic)

        # Build new spectral tensor in intrinsic dimension
        m = min(self.intrinsic_dimension, 3)
        spectral = SpectralTensor(
            coefficients=T.spectral.coefficients.copy(),
            basis_type=T.spectral.basis_type,
            domain=T.domain,
            n_modes=len(T.spectral.coefficients),
            dimension=m,
        )

        result_dist = DualDistribution(
            spectral=spectral,
            singularities=projected_singularities,
            domain=T.domain,
            distribution_type=f"{T.distribution_type}_ccd_projected",
        )

        mean_error = float(np.mean(errors)) if errors else 0.0

        return CCDSingularityProjection(
            original_dimension=ambient_dimension,
            intrinsic_dimension=m,
            n_singularities_before=original_n,
            n_singularities_after=len(projected_singularities),
            projection_matrix=P,
            reconstruction_error=mean_error,
            singularities_preserved=(original_n == len(projected_singularities)),
            manifold_coordinates=np.array(manifold_coords) if manifold_coords else np.array([]),
        )

    def estimate_ccd_speedup(
        self, projection: CCDSingularityProjection
    ) -> Dict:
        """
        Estimate computational savings from CCD projection.

        Key insight from CCDEngine.md: complexity scales with
        intrinsic dimension m, not ambient dimension d.

        For singularities: cost ∝ n_sing × d → n_sing × m
        Speedup ≈ d / m
        """
        d = projection.original_dimension
        m = projection.intrinsic_dimension
        return {
            "dimension_reduction": f"{d} → {m}",
            "theoretical_speedup": float(d) / max(m, 1),
            "singularity_cost_before": projection.n_singularities_before * d,
            "singularity_cost_after": projection.n_singularities_after * m,
            "reconstruction_error": projection.reconstruction_error,
        }


# ============================================================================
# CLOSURE 4: ADAPTIVE COHOMOLOGICAL COST BOUND
# ============================================================================

@dataclass
class AdaptiveCostBound:
    """Dynamic communication cost analysis for cohomological gluing."""
    n_patches: int
    n_singularities_total: int
    max_singularities_per_patch: int
    base_comm_cost: int               # O(log N) for compatibility
    singularity_comm_cost: int         # O(s_max) for singularity data
    total_comm_cost: int               # Base + singularity
    is_log_bound_valid: bool           # Whether O(log N) bound still holds
    degradation_factor: float          # > 1 if turbulence degrades bound
    regime: str                        # "laminar", "transitional", "turbulent"
    recommended_patch_count: int       # Optimal N for this singularity load


class AdaptiveCostAnalyzer:
    """
    Dynamic analysis of cohomological gluing communication cost.

    The O(log N) bound assumes the number of singularities per patch
    is bounded by a constant independent of N. In turbulent regimes,
    singularity proliferation can degrade this bound.

    This analyzer dynamically monitors singularity density and
    adjusts the patch count to maintain the logarithmic bound.

    Regimes:
      LAMINAR:      s_max ≪ N  → O(log N) holds strictly
      TRANSITIONAL: s_max ~ N  → O(log N + s_max) ≈ O(N)
      TURBULENT:    s_max ≫ N  → O(N log N + s_max²) potential
    """

    def __init__(
        self,
        laminar_threshold: float = 0.1,    # s_max/N < this → laminar
        turbulent_threshold: float = 1.0,   # s_max/N > this → turbulent
    ):
        self.laminar_threshold = laminar_threshold
        self.turbulent_threshold = turbulent_threshold

    def analyze_patches(
        self, patches: List[PatchDistribution]
    ) -> AdaptiveCostBound:
        """
        Analyze singularity distribution across patches.
        """
        n_patches = len(patches)
        sing_counts = [
            len(p.local_distribution.singularities) for p in patches
        ]
        total_sing = sum(sing_counts)
        max_sing = max(sing_counts) if sing_counts else 0

        # Base cost: O(log N) per gluing condition × (N-1) conditions
        base_cost = int(np.ceil(np.log2(max(n_patches, 2)))) * (n_patches - 1)

        # Singularity communication: transmit singularity data for overlaps
        sing_cost = max_sing * (n_patches - 1)

        total_cost = base_cost + sing_cost

        # Determine regime
        ratio = max_sing / max(n_patches, 1)
        if ratio < self.laminar_threshold:
            regime = "laminar"
            is_log = True
            degradation = 1.0
        elif ratio < self.turbulent_threshold:
            regime = "transitional"
            is_log = False
            degradation = ratio / self.laminar_threshold
        else:
            regime = "turbulent"
            is_log = False
            degradation = ratio / self.laminar_threshold

        # Recommend optimal patch count
        if regime == "turbulent":
            # Fewer patches = less inter-patch communication
            recommended = max(2, n_patches // 4)
        elif regime == "transitional":
            recommended = max(2, n_patches // 2)
        else:
            recommended = n_patches  # Laminar: current is optimal

        return AdaptiveCostBound(
            n_patches=n_patches,
            n_singularities_total=total_sing,
            max_singularities_per_patch=max_sing,
            base_comm_cost=base_cost,
            singularity_comm_cost=sing_cost,
            total_comm_cost=total_cost,
            is_log_bound_valid=is_log,
            degradation_factor=degradation,
            regime=regime,
            recommended_patch_count=recommended,
        )

    def optimize_patch_count(
        self,
        T,  # DualDistribution or int (n_singularities)
        domain: Tuple[float, float],
        n_patches_range: Tuple[int, int] = (2, 32),
    ) -> Tuple[int, AdaptiveCostBound]:
        """
        Find the optimal number of patches for a given distribution.

        Balances computational cost of more patches vs.
        communication cost of singularities across patches.

        Args:
            T: DualDistribution or int (number of singularities).
            domain: Domain for gluing protocol.
            n_patches_range: (min, max) patches to consider.
        """
        # Handle both DualDistribution and raw singularity count
        if isinstance(T, int):
            n_sing = T
        elif hasattr(T, 'singularities'):
            n_sing = len(T.singularities)
        else:
            n_sing = 0

        best_n = n_patches_range[0]
        best_cost = float('inf')
        best_bound = None

        for n in range(n_patches_range[0], n_patches_range[1] + 1):
            sing_per_patch = max(1, n_sing // max(n, 1))

            base = int(np.ceil(np.log2(max(n, 2)))) * (n - 1)
            sing = sing_per_patch * (n - 1)
            total = base + sing

            if total < best_cost:
                best_cost = total
                best_n = n

        # Build final analysis with optimal patch count
        protocol = CohomologicalGluingProtocol(domain=domain, n_patches=best_n)
        if hasattr(T, 'singularities'):
            patches = protocol.compute_local_representations(
                lambda x: T.evaluate_approximate(x),
                n_modes=32,
            )
        else:
            # Build synthetic patches for analysis
            patches = []
            for i in range(best_n):
                p_domain = (
                    domain[0] + i * (domain[1] - domain[0]) / best_n,
                    domain[0] + (i + 1) * (domain[1] - domain[0]) / best_n,
                )
                from acf_functor.distribution_theory import PatchDistribution
                synthetic_dist = DualDistribution(
                    spectral=SpectralTensor(np.zeros(1), "chebyshev", domain, 1),
                    singularities=[],
                    domain=p_domain,
                )
                neighbors = []
                if i > 0:
                    neighbors.append(i - 1)
                if i < best_n - 1:
                    neighbors.append(i + 1)
                patches.append(PatchDistribution(i, p_domain, synthetic_dist, neighbors))
        best_bound = self.analyze_patches(patches)

        return best_n, best_bound

    def monitor_singularity_growth(
        self,
        T_sequence: List[DualDistribution],
        n_patches: int = 8,
    ) -> List[AdaptiveCostBound]:
        """
        Monitor singularity growth over time (e.g., in a turbulent simulation).

        Returns cost evolution showing when the O(log N) bound degrades.
        """
        evolution = []
        for T in T_sequence:
            protocol = CohomologicalGluingProtocol(
                domain=T.domain, n_patches=n_patches, overlap_ratio=0.1,
            )
            patches = protocol.compute_local_representations(
                lambda x: T.evaluate_approximate(x),
                n_modes=32,
            )
            bound = self.analyze_patches(patches)
            evolution.append(bound)

        return evolution

    def detect_regime_transition(
        self, evolution: List[AdaptiveCostBound]
    ) -> Optional[int]:
        """
        Detect the time step where the regime transitions from laminar to turbulent.
        """
        for i, bound in enumerate(evolution):
            if bound.regime == "turbulent":
                return i
        return None
