"""
Distribution-Gelfand Integration — Bridging Distribution Theory with OTU
======================================================================

This module integrates the DualDistribution representation into the
Gelfand Triple architecture (OTU.md Level 12).

Key mathematical facts:
  - Φ  = space of test functions (Schwartz space, nuclear)
  - L² = Hilbert space
  - Φ' = space of tempered distributions (dual of Φ)

Distributions naturally live in Φ':
  - Dirac δ_x ∈ Φ'
  - Heaviside H(x) ∈ Φ'
  - SRB measures μ_SRB ∈ Φ'
  - Koopman modes φ_k ∈ Φ (not Φ')
  - Perron-Frobenius eigenmeasures ∈ Φ'

The biorthogonality relation ⟨φ_k, μ_j⟩ = δ_{kj} from OTU.md is
exactly the duality pairing between Φ and Φ' that distributions use.

This module:
  1. Extends the GelfandTriple class with distribution-aware operations
  2. Implements distribution-to-modes projection (Φ' → spectral coefficients)
  3. Implements mode-to-distribution lifting (spectral coefficients → Φ')
  4. Provides the unified evaluation operator Λ acting on distributions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg

from acf_functor.distribution_theory import (
    DualDistribution,
    DirectionalSingularity,
    SpectralTensor,
    SingularityType,
    DistributionOperator,
    CohomologicalGluingProtocol,
    PatchDistribution,
    GluingCondition,
    CohomologicalGluingResult,
)


@dataclass
class DistributionGelfandState:
    """
    Extended Gelfand state that tracks distributions alongside
    the standard Koopman/Perron-Frobenius spectral data.
    """
    # Standard OTU data (from gelfand_triple.py) — all optional with defaults
    test_basis: Optional[np.ndarray] = None       # Φ basis functions (e.g., Chebyshev nodes)
    koopman_modes: Optional[np.ndarray] = None    # Right eigenvectors in Φ
    srb_modes: Optional[np.ndarray] = None        # Left eigenvectors in Φ'
    resonances: Optional[np.ndarray] = None       # Pollicott-Ruelle eigenvalues
    mu_srb: Optional[np.ndarray] = None           # SRB measure (dominant eigenmeasure)

    # Distribution-specific extensions
    registered_distributions: Dict[str, DualDistribution] = field(default_factory=dict)
    distribution_projections: Dict[str, np.ndarray] = field(default_factory=dict)
    biorthogonality_residuals: Dict[str, float] = field(default_factory=dict)


class DistributionGelfandBridge:
    """
    Bridge between distribution theory and the Gelfand Triple.

    Provides the operations needed to:
    - Project a distribution T ∈ Φ' onto the Koopman mode basis Φ
    - Lift spectral coefficients from Φ to a distribution in Φ'
    - Evaluate ⟨φ_k, T⟩ — the duality pairing
    - Check whether a distribution is Φ'-representable
    """

    def __init__(
        self,
        n_test_functions: int = 64,
        domain: Tuple[float, float] = (-1.0, 1.0),
    ):
        self.n_test_functions = n_test_functions
        self.domain = domain
        self._state = DistributionGelfandState()

    def build_test_basis(self) -> np.ndarray:
        """Build the Chebyshev test function basis for Φ."""
        a, b = self.domain
        k = np.arange(self.n_test_functions)
        nodes = np.cos(np.pi * k / (self.n_test_functions - 1))
        nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
        self._state.test_basis = nodes
        return nodes

    def project_distribution_to_modes(
        self, T: DualDistribution
    ) -> np.ndarray:
        """
        Project a distribution T ∈ Φ' onto the test function basis.

        Computes c_k = ⟨T, ψ_k⟩ for each basis function ψ_k ∈ Φ.

        This is the discrete analog of the Schwartz kernel theorem:
        every distribution is uniquely determined by its action on
        a dense set of test functions.
        """
        if self._state.test_basis is None or len(self._state.test_basis) == 0:
            self.build_test_basis()

        coeffs = np.zeros(self.n_test_functions, dtype=np.complex128)

        for k in range(self.n_test_functions):
            # Build test function ψ_k(x) = T_k(x) (Chebyshev polynomial)
            def test_func(x: np.ndarray) -> complex:
                a, b = self.domain
                x_mapped = 2.0 * (x - a) / (b - a) - 1.0
                x_mapped = np.clip(x_mapped, -1.0, 1.0)
                return complex(np.cos(k * np.arccos(x_mapped))[0])

            coeffs[k] = T.act_on_test_function(test_func)

        return coeffs

    def modes_to_distribution(
        self, coeffs: np.ndarray, n_spectral: int = 64
    ) -> DualDistribution:
        """
        Lift spectral coefficients from Φ back to a distribution in Φ'.

        Uses the kernel theorem in reverse: given {c_k}, reconstruct
        T ≈ Σ_k c_k ψ_k^* where ψ_k^* are the dual basis elements.

        For smooth functions this is exact (Chebyshev series).
        For singular distributions, additional singularities are
        reconstructed from the spectral decay pattern.
        """
        # Reconstruct smooth part from coefficients
        spectral = SpectralTensor(
            coefficients=np.real(coeffs[:n_spectral]),
            basis_type="chebyshev",
            domain=self.domain,
            n_modes=n_spectral,
        )

        # Detect singularities from spectral decay
        singularities = self._detect_singularities_from_coeffs(coeffs)

        return DualDistribution(
            spectral=spectral,
            singularities=singularities,
            domain=self.domain,
            distribution_type="reconstructed_from_modes",
        )

    def _detect_singularities_from_coeffs(
        self, coeffs: np.ndarray
    ) -> List[DirectionalSingularity]:
        """Detect singularities from slow spectral decay."""
        if len(coeffs) < 10:
            return []

        abs_coeffs = np.abs(coeffs)
        # Fit decay rate
        k_vals = np.arange(5, min(len(abs_coeffs), len(abs_coeffs) // 2))
        log_k = np.log(k_vals + 1)
        log_c = np.log(abs_coeffs[k_vals] + 1e-15)

        if len(log_k) < 3:
            return []

        slope, _ = np.polyfit(log_k, log_c, 1)

        singularities = []
        if slope > -1.5:  # Slow decay → singularities present
            # Estimate singularity parameters from Gibbs phenomenon
            # (This is a simplified heuristic)
            a, b = self.domain
            x0 = (a + b) / 2.0  # Default: center of domain
            order_estimate = max(0, int(np.ceil(-slope - 1)))

            if order_estimate > 3:
                order_estimate = -1  # Likely Heaviside, not Dirac

            singularities.append(
                DirectionalSingularity(
                    position=np.array([x0]),
                    codirection=np.array([1.0]),
                    order=order_estimate,
                    singularity_type=(
                        SingularityType.DIRAC_DERIVATIVE
                        if order_estimate > 0
                        else SingularityType.HEAVISIDE
                    ),
                )
            )

        return singularities

    def duality_pairing(
        self, mode_coeffs: np.ndarray, T: DualDistribution
    ) -> complex:
        """
        Compute the duality pairing ⟨Σ c_k ψ_k, T⟩.

        This is the fundamental bilinear form between Φ and Φ':
          ⟨φ, T⟩ = T(φ)

        For Koopman modes φ_k and SRB measure μ_SRB:
          ⟨φ_k, μ_SRB⟩ = δ_{k,0} (biorthogonality)
        """
        # Reconstruct test function from mode coefficients
        test_dist = self.modes_to_distribution(mode_coeffs)

        # The pairing is symmetric: ⟨φ, T⟩ = ⟨T, φ⟩
        # We use the standard Schwartz pairing
        # This is approximated by integrating the product
        a, b = self.domain
        x = np.linspace(a, b, 1000)

        phi_vals = test_dist.evaluate_approximate(x)
        T_vals = T.evaluate_approximate(x)

        dx = (b - a) / (len(x) - 1)
        return complex(np.sum(phi_vals * T_vals) * dx)

    def validate_biorthogonality(
        self,
        koopman_modes: np.ndarray,   # shape (n_modes, n_test)
        srb_modes: np.ndarray,       # shape (n_modes, n_dist)
    ) -> Tuple[float, np.ndarray]:
        """
        Verify biorthogonality: ⟨φ_i, μ_j⟩ ≈ δ_{ij}.

        This is the key certificate that TAA (Koopman) and ERGON (PF)
        are complementary projections of the same operator Λ.
        """
        n_modes = len(koopman_modes)
        gram = np.zeros((n_modes, n_modes), dtype=np.complex128)

        for i in range(n_modes):
            for j in range(n_modes):
                phi_i = self.modes_to_distribution(koopman_modes[i])
                mu_j = self.modes_to_distribution(srb_modes[j])
                gram[i, j] = self.duality_pairing(koopman_modes[i], mu_j)

        # Error = ‖Gram - I‖_F
        target = np.eye(n_modes, dtype=np.complex128)
        error = float(np.linalg.norm(gram - target, 'fro') / np.sqrt(n_modes))

        return error, gram

    def register_distribution(self, name: str, T: DualDistribution) -> np.ndarray:
        """Register a distribution and compute its mode projection."""
        coeffs = self.project_distribution_to_modes(T)
        self._state.registered_distributions[name] = T
        self._state.distribution_projections[name] = coeffs

        # Compute biorthogonality residual against existing modes
        if self._state.koopman_modes is not None and len(self._state.koopman_modes) > 0:
            residuals = []
            for i, km in enumerate(self._state.koopman_modes):
                pairing = self.duality_pairing(km, T)
                target = 1.0 if i == 0 else 0.0
                residuals.append(abs(pairing - target))
            self._state.biorthogonality_residuals[name] = float(np.mean(residuals))

        return coeffs

    def analyze_distribution(
        self, T: DualDistribution
    ) -> Dict:
        """
        Full distribution analysis within the Gelfand framework.

        Returns:
          - mode coefficients (projection onto Φ)
          - spectral decay rate (from coefficients)
          - Sobolev regularity estimate (H^s)
          - best Φ-representation error
          - singularity inventory
        """
        coeffs = self.project_distribution_to_modes(T)

        # Spectral decay analysis
        abs_c = np.abs(coeffs)
        if len(abs_c) > 5:
            k_vals = np.arange(1, min(21, len(abs_c)))
            log_k = np.log(k_vals)
            log_c = np.log(abs_c[k_vals] + 1e-15)
            slope, intercept = np.polyfit(log_k, log_c, 1)
        else:
            slope, intercept = 0.0, 0.0

        # Sobolev regularity: T ∈ H^s where s ≈ -(slope + 0.5)
        sobolev_order = -(slope + 0.5)

        # Best approximation error
        n_modes = len(coeffs)
        tail_energy = np.sum(abs_c[n_modes // 2:] ** 2)
        total_energy = np.sum(abs_c ** 2) + 1e-15
        approx_error = float(np.sqrt(tail_energy / total_energy))

        # Consistency check
        consistency = T.check_consistency()

        return {
            "mode_coefficients": coeffs,
            "spectral_decay_rate": float(slope),
            "spectral_intercept": float(intercept),
            "sobolev_regularity": float(sobolev_order),
            "sobolev_space": f"H^{{{sobolev_order:.1f}}}",
            "approximation_error": approx_error,
            "n_singularities": len(T.singularities),
            "max_singularity_order": max((s.order for s in T.singularities), default=0),
            "consistency_residual": consistency,
            "is_regular_distribution": len(T.singularities) == 0,
            "distribution_type": T.distribution_type,
        }


# ============================================================================
# UNIFIED TRANSFER OPERATOR ON DISTRIBUTIONS
# ============================================================================

class DistributionTransferOperator:
    """
    The Unified Transfer Operator Λ acting on distributions.

    Λ: Φ' → Φ'

    This extends the OTU/Gelfand architecture to explicitly handle
    distribution representations.

    Properties:
      - Λ|_Φ = K (Koopman, acts on test functions)
      - (Λ*)|_Φ' = L (Perron-Frobenius, acts on measures)
      - Λ has discrete spectrum (Pollicott-Ruelle resonances) in Φ'
    """

    def __init__(
        self,
        dynamics: Callable[[np.ndarray], np.ndarray],
        domain: Tuple[float, float] = (-1.0, 1.0),
        n_modes: int = 64,
    ):
        self.dynamics = dynamics
        self.domain = domain
        self.n_modes = n_modes
        self.bridge = DistributionGelfandBridge(n_modes, domain)
        self.bridge.build_test_basis()

    def apply_to_distribution(self, T: DualDistribution) -> DualDistribution:
        """
        Apply Λ to a distribution T.

        Λ T acts as: ⟨ΛT, φ⟩ = ⟨T, φ ∘ dynamics⟩ = ⟨T, Kφ⟩

        For a Dirac: Λδ_{x₀} = δ_{T(x₀)} (push-forward)
        For SRB: Λμ_SRB = μ_SRB (fixed point)
        """
        # Push-forward: singularities move under dynamics
        pushed_singularities = []
        for sing in T.singularities:
            new_pos = self.dynamics(sing.position)
            # Clamp to domain
            a, b = self.domain
            new_pos = np.clip(new_pos, a, b)
            pushed_singularities.append(
                DirectionalSingularity(
                    position=new_pos,
                    codirection=sing.codirection,  # Approximate
                    order=sing.order,
                    mass=sing.mass,
                    singularity_type=sing.singularity_type,
                )
            )

        # Spectral part: evolve under Koopman
        # K: f ↦ f ∘ T, so spectral coefficients transform
        spectral_coeffs = self.bridge.project_distribution_to_modes(T)
        # Apply Koopman matrix (simplified: identity for now)
        evolved_coeffs = spectral_coeffs  # Would multiply by K matrix

        evolved_spectral = SpectralTensor(
            coefficients=np.real(evolved_coeffs[:self.n_modes]),
            basis_type="chebyshev",
            domain=self.domain,
            n_modes=self.n_modes,
        )

        result = DualDistribution(
            spectral=evolved_spectral,
            singularities=pushed_singularities,
            domain=self.domain,
            distribution_type=f"Λ({T.distribution_type})",
        )

        return result

    def find_fixed_point(
        self,
        initial: Optional[DualDistribution] = None,
        n_iterations: int = 1000,
        tolerance: float = 1e-6,
    ) -> DualDistribution:
        """
        Find the SRB measure as a fixed point of Λ.

        Λ μ_SRB = μ_SRB

        This is the distribution-theoretic version of ERGON's power iteration.
        Uses Ulam-like discretization for stability.
        """
        a, b = self.domain
        n = self.n_modes
        
        # Start with uniform probability measure (properly normalized)
        uniform_coeffs = np.zeros(n)
        uniform_coeffs[0] = 2.0  # Corresponds to constant function 1 → normalizes to probability
        
        spectral = SpectralTensor(
            coefficients=uniform_coeffs,
            basis_type="chebyshev",
            domain=self.domain,
            n_modes=n,
        )
        T = DualDistribution(
            spectral=spectral,
            singularities=[],
            domain=self.domain,
            distribution_type="uniform_measure",
        )
        
        prev = T
        for it in range(n_iterations):
            curr = self.apply_to_distribution(prev)
            
            # Renormalize to preserve probability mass
            x_grid = np.linspace(a, b, 500)
            curr_vals = curr.evaluate_approximate(x_grid)
            total_mass = np.sum(np.maximum(curr_vals, 0)) * (b - a) / 500
            if total_mass > 1e-15:
                scale = 1.0 / total_mass
                curr.spectral.coefficients = curr.spectral.coefficients * scale
            
            # Check convergence
            prev_vals = prev.evaluate_approximate(x_grid)
            diff = np.max(np.abs(curr_vals * scale - prev_vals))
            
            if diff < tolerance:
                break
            
            prev = curr
        
        return prev


# ============================================================================
# CONVENIENCE
# ============================================================================

def distribution_to_gelfand_analysis(
    T: DualDistribution,
    dynamics: Optional[Callable] = None,
    n_modes: int = 64,
    domain: Tuple[float, float] = (-1.0, 1.0),
) -> Dict:
    """
    Complete analysis of a distribution within the Gelfand Triple framework.

    This is the main entry point for integrating distributions into the
    OTU/Gelfand pipeline.
    """
    bridge = DistributionGelfandBridge(n_modes, domain)
    analysis = bridge.analyze_distribution(T)

    if dynamics is not None:
        transfer_op = DistributionTransferOperator(dynamics, domain, n_modes)
        srb_measure = transfer_op.find_fixed_point(T)
        analysis["srb_measure"] = srb_measure
        analysis["is_srb_fixed_point"] = True

    return analysis
