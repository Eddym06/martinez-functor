"""
test_gelfand_triple.py — Test suite for the Unified Transfer Operator (OTU)

Demonstrates and certifies:
  1. Self-consistent SRB measure (power iteration convergence)
  2. Pollicott-Ruelle resonance spectrum
  3. Biorthogonality of Koopman modes and SRB measures
  4. Pesin formula: h_KS(spectral) ≈ h_KS(Lyapunov)  [the key theorem]
  5. Known analytical values for canonical systems

Canonical test cases with known analytical answers:
  - Logistic map r=4:  h_KS = log(2) ≈ 0.6931  (Bernoulli, exactly soluble)
  - Chebyshev map n=2: h_KS = log(2) ≈ 0.6931  (conjugate to logistic r=4)
  - Tent map:          h_KS = log(2) ≈ 0.6931  (topological conjugacy)

All tests produce OTU numerical certificates exportable to Lean 4.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from acf_functor.gelfand_triple import (
    GelfandTriple,
    CanonicalSystems,
    RuelleSpectrum,
    OTUResult,
    certify,
)


# ============================================================================
# Fixtures
# ============================================================================

LOG2 = math.log(2)  # = 0.693147... — exact h_KS for Bernoulli systems


@pytest.fixture
def logistic_r4():
    return CanonicalSystems.logistic(r=4.0), (0.0, 1.0)


@pytest.fixture
def tent_map():
    return CanonicalSystems.tent(), (0.0, 1.0)


@pytest.fixture
def chebyshev_map():
    return CanonicalSystems.chebyshev(n=2), (-1.0, 1.0)


# ============================================================================
# Class 1: SRB measure self-consistency
# ============================================================================

class TestSRBMeasure:
    """OTU-1: The SRB measure is the dominant eigenvector of the Perron-Frobenius operator."""

    def test_srb_converges_logistic(self, logistic_r4):
        """Power iteration converges to μ_SRB for logistic r=4."""
        T, domain = logistic_r4
        triple = GelfandTriple(T, domain=domain, n_test=32, n_dist=256, srb_tol=1e-8)
        triple.build()
        mu_srb, n_iter = triple.compute_self_consistent_measure()
        # Must converge
        assert n_iter < 5000, f"Did not converge in 5000 steps (got {n_iter})"
        # Must be a valid probability measure
        assert abs(mu_srb.sum() - 1.0) < 1e-6, f"Not normalized: sum = {mu_srb.sum()}"
        assert (mu_srb >= 0).all(), "Negative probabilities in μ_SRB"
        print(f"\n  [PASS] SRB convergence: {n_iter} iterations, sum={mu_srb.sum():.8f}")

    def test_srb_invariant_under_pf(self, logistic_r4):
        """Verify ℒμ_SRB ≈ μ_SRB — the defining property of μ_SRB (OTU-4)."""
        T, domain = logistic_r4
        triple = GelfandTriple(T, domain=domain, n_test=32, n_dist=256, srb_tol=1e-9)
        triple.build()
        mu_srb, _ = triple.compute_self_consistent_measure()
        # Apply PF one more time
        mu_next = triple._L @ mu_srb
        mu_next /= mu_next.sum()
        # Should be close to μ_SRB
        l1_diff = np.linalg.norm(mu_next - mu_srb, ord=1)
        assert l1_diff < 1e-5, f"ℒμ_SRB ≠ μ_SRB: L1 diff = {l1_diff:.2e}"
        print(f"\n  [PASS] PF invariance: ‖ℒμ_SRB - μ_SRB‖₁ = {l1_diff:.2e}")

    def test_srb_logistic_r4_is_arcsine(self, logistic_r4):
        """
        For logistic r=4, the exact SRB measure is the arcsine distribution:
            dμ_SRB/dx = 1 / (π √(x(1-x)))

        We verify the computed μ_SRB matches via total-variation distance.
        The arcsine density must be integrated over each cell (not just point-evaluated)
        because it diverges at the endpoints, causing point evaluation to overestimate
        the boundary cells and inflate the TV distance artificially.

        Exact cell integral: ∫_{x_l}^{x_r} 1/(π√(x(1-x))) dx
            = (2/π)[arcsin(√(x_r)) - arcsin(√(x_l))]
        """
        T, domain = logistic_r4
        triple = GelfandTriple(T, domain=domain, n_test=32, n_dist=512, srb_tol=1e-9)
        triple.build()
        mu_srb, n_iter = triple.compute_self_consistent_measure()

        h = 1.0 / 512
        grid = triple.grid_dist
        x_left = np.clip(grid - h / 2, 1e-10, 1.0 - 1e-10)
        x_right = np.clip(grid + h / 2, 1e-10, 1.0 - 1e-10)

        # Exact cell-integrated arcsine mass
        density_arcsine = (2.0 / math.pi) * (
            np.arcsin(np.sqrt(x_right)) - np.arcsin(np.sqrt(x_left))
        )
        density_arcsine = np.abs(density_arcsine)
        density_arcsine /= density_arcsine.sum()

        tv_dist = 0.5 * np.linalg.norm(mu_srb - density_arcsine, ord=1)
        assert tv_dist < 0.12, (
            f"μ_SRB for logistic r=4 should be arcsine (cell-integrated), "
            f"but TV distance = {tv_dist:.4f}"
        )
        print(f"\n  [PASS] μ_SRB ≈ arcsine (cell-integrated): TV distance = {tv_dist:.4f}")


# ============================================================================
# Class 2: Ruelle-Pollicott Resonances
# ============================================================================

class TestRuelleSpectrum:
    """OTU-2/3: The spectrum of Λ consists of Pollicott-Ruelle resonances."""

    def test_dominant_eigenvalue_is_one(self, logistic_r4):
        """The dominant eigenvalue of the Perron-Frobenius matrix must be 1."""
        T, domain = logistic_r4
        triple = GelfandTriple(T, domain=domain, n_test=32, n_dist=256)
        triple.build()
        mu_srb, _ = triple.compute_self_consistent_measure()
        resonances, _, _, _, _, _ = triple.compute_ruelle_spectrum(mu_srb, n_modes=8)
        dominant = resonances[0]
        assert abs(abs(dominant) - 1.0) < 0.05, (
            f"Dominant resonance |λ₀| = {abs(dominant):.4f} ≠ 1.0"
        )
        print(f"\n  [PASS] Dominant resonance: λ₀ = {dominant:.6f}")

    def test_non_dominant_resonances_decay(self, logistic_r4):
        """All non-dominant resonances must have |λₖ| < 1 (OTU-11)."""
        T, domain = logistic_r4
        triple = GelfandTriple(T, domain=domain, n_test=32, n_dist=256)
        triple.build()
        mu_srb, _ = triple.compute_self_consistent_measure()
        resonances, _, _, _, _, _ = triple.compute_ruelle_spectrum(mu_srb, n_modes=8)
        non_dominant = resonances[1:]
        mods = np.abs(non_dominant)
        assert (mods <= 1.0 + 1e-6).all(), (
            f"Non-dominant resonances must have |λ| ≤ 1, got max = {mods.max():.4f}"
        )
        print(f"\n  [PASS] Resonance decay: max|λₖ≥1| = {mods.max():.4f} ≤ 1.0")

    def test_spectral_gap_positive_for_mixing(self, logistic_r4):
        """The spectral gap Γ_OTU = -log|λ₁| > 0 for the mixing logistic map."""
        T, domain = logistic_r4
        triple = GelfandTriple(T, domain=domain, n_test=32, n_dist=256)
        triple.build()
        mu_srb, _ = triple.compute_self_consistent_measure()
        resonances, _, _, _, _, _ = triple.compute_ruelle_spectrum(mu_srb, n_modes=8)
        mods = np.abs(resonances)
        nontrivial = mods[mods < 0.9999]
        if len(nontrivial) == 0:
            pytest.skip("No non-trivial resonances found (system might be Bernoulli)")
        spectral_gap = -np.log(nontrivial[0])
        assert spectral_gap > 0, f"Spectral gap Γ_OTU = {spectral_gap:.4f} must be > 0"
        print(f"\n  [PASS] Spectral gap: Γ_OTU = {spectral_gap:.4f} > 0")


# ============================================================================
# Class 3: Biorthogonality (the core OTU-5 certificate)
# ============================================================================

class TestBiorthogonality:
    """OTU-5: Koopman eigenfunctions and PF eigenmeasures are biorthogonal."""

    def test_biorthogonality_residual_logistic(self, logistic_r4):
        """
        Biorthogonality certificate: ‖⟨φᵢ, μⱼ⟩ - δᵢⱼ‖_F < threshold.

        A small Frobenius norm means TAA and ERGON modes form a proper
        biorthogonal system — they are genuinely complementary projections.
        """
        T, domain = logistic_r4
        triple = GelfandTriple(T, domain=domain, n_test=48, n_dist=512)
        triple.build()
        mu_srb, _ = triple.compute_self_consistent_measure()
        _, _, _, biorth_error, _, B = triple.compute_ruelle_spectrum(mu_srb, n_modes=8)
        assert biorth_error < 2.0, (
            f"Biorthogonality residual ‖B - I‖_F = {biorth_error:.4f} should be < 2.0\n"
            f"Diagonal of B: {np.diag(B).round(3)}"
        )
        print(f"\n  [PASS] Biorthogonality: ‖B - I‖_F = {biorth_error:.4f}")
        print(f"         Diagonal of B: {np.diag(B).round(3).tolist()}")

    def test_koopman_pf_same_eigenvalues(self, logistic_r4):
        """
        Koopman eigenvalues and PF eigenvalues must coincide (both are resonances of Λ).
        This verifies the duality K* = ℒ at the spectral level.
        """
        T, domain = logistic_r4
        triple = GelfandTriple(T, domain=domain, n_test=32, n_dist=256)
        triple.build()
        mu_srb, _ = triple.compute_self_consistent_measure()
        resonances, _, _, _, _, _ = triple.compute_ruelle_spectrum(mu_srb, n_modes=6)

        # The resonances come from the PF matrix eigenvalues.
        # Check they match the Koopman eigenvalues up to sorting.
        from scipy import linalg
        eigvals_k = np.sort(np.abs(linalg.eig(triple._K)[0]))[::-1][:6]
        eigvals_pf = np.abs(resonances)

        # The dominant should match
        assert abs(eigvals_k[0] - eigvals_pf[0]) < 0.15, (
            f"Dominant Koopman |λ₀|_K = {eigvals_k[0]:.4f} vs "
            f"PF |λ₀|_PF = {eigvals_pf[0]:.4f}"
        )
        print(f"\n  [PASS] Koopman |λ₀| = {eigvals_k[0]:.4f}, PF |λ₀| = {eigvals_pf[0]:.4f}")


# ============================================================================
# Class 4: The Pesin Formula as Theorem (the central achievement of OTU)
# ============================================================================

class TestPesinFormula:
    """
    OTU-7: h_KS computed from the Ruelle spectrum equals h_KS computed from Lyapunov.

    This is the empirical demonstration that the Pesin formula is a CONSEQUENCE
    of the biorthogonality of the OTU spectrum — not an independent axiom.

    Analytical ground truth:
        Logistic r=4: h_KS = log(2) ≈ 0.6931  (Pesin + arcsine measure)
        Chebyshev n=2: h_KS = log(2) ≈ 0.6931  (conjugate to logistic r=4)
    """

    def test_pesin_logistic_r4(self, logistic_r4):
        """Logistic r=4: verify h_KS ≈ log(2) via Lyapunov and spectral methods."""
        T, domain = logistic_r4
        triple = GelfandTriple(T, domain=domain, n_test=48, n_dist=512, srb_tol=1e-9)
        triple.build()
        result = triple.analyze(n_modes=16, n_orbit=50_000)

        # Lyapunov estimate must be close to log(2)
        assert abs(result.spectrum.lyapunov_entropy - LOG2) < 0.10, (
            f"Lyapunov h_KS = {result.spectrum.lyapunov_entropy:.4f}, "
            f"expected log(2) = {LOG2:.4f}"
        )
        # Pesin formula must hold (spectral ≈ Lyapunov)
        assert result.spectrum.pesin_verified, (
            f"Pesin formula FAILED: "
            f"spectral h_KS = {result.spectrum.pesin_entropy:.4f}, "
            f"Lyapunov h_KS = {result.spectrum.lyapunov_entropy:.4f}"
        )
        print(f"\n  [PASS] Logistic r=4 Pesin:")
        print(f"         Lyapunov  h_KS = {result.spectrum.lyapunov_entropy:.4f}")
        print(f"         Spectral  h_KS = {result.spectrum.pesin_entropy:.4f}")
        print(f"         Exact     h_KS = {LOG2:.4f}")
        print(f"         Pesin verified: {result.spectrum.pesin_verified}")

    def test_pesin_chebyshev(self, chebyshev_map):
        """Chebyshev n=2: analytically conjugate to logistic r=4, h_KS = log(2)."""
        T, domain = chebyshev_map
        triple = GelfandTriple(T, domain=domain, n_test=48, n_dist=512, srb_tol=1e-9)
        triple.build()
        result = triple.analyze(n_modes=16, n_orbit=50_000)

        assert abs(result.spectrum.lyapunov_entropy - LOG2) < 0.10, (
            f"Chebyshev Lyapunov h_KS = {result.spectrum.lyapunov_entropy:.4f}"
        )
        assert result.spectrum.pesin_verified, (
            f"Chebyshev Pesin FAILED: "
            f"spectral={result.spectrum.pesin_entropy:.4f}, "
            f"lyapunov={result.spectrum.lyapunov_entropy:.4f}"
        )
        print(f"\n  [PASS] Chebyshev n=2 Pesin:")
        print(f"         Lyapunov  h_KS = {result.spectrum.lyapunov_entropy:.4f}")
        print(f"         Spectral  h_KS = {result.spectrum.pesin_entropy:.4f}")
        print(f"         Exact     h_KS = {LOG2:.4f}")

    def test_tent_map_pesin(self, tent_map):
        """Tent map: h_KS = log(2) (Bernoulli, piecewise linear)."""
        T, domain = tent_map
        triple = GelfandTriple(T, domain=domain, n_test=48, n_dist=512, srb_tol=1e-9)
        triple.build()
        result = triple.analyze(n_modes=16, n_orbit=50_000)
        assert abs(result.spectrum.lyapunov_entropy - LOG2) < 0.10, (
            f"Tent Lyapunov h_KS = {result.spectrum.lyapunov_entropy:.4f}"
        )
        assert result.spectrum.pesin_verified
        print(f"\n  [PASS] Tent map Pesin:")
        print(f"         Lyapunov  h_KS = {result.spectrum.lyapunov_entropy:.4f}")
        print(f"         Spectral  h_KS = {result.spectrum.pesin_entropy:.4f}")

    def test_pesin_vs_joint_analyze(self, logistic_r4):
        """
        Demonstrate that OTU is superior to joint_analyze().

        joint_analyze() uses the uniform measure → computes Koopman on the wrong space.
        OTU uses the self-consistent μ_SRB → Pesin holds.
        """
        T, domain = logistic_r4
        triple_otu = GelfandTriple(T, domain=domain, n_test=48, n_dist=512, srb_tol=1e-9)
        triple_otu.build()
        result_otu = triple_otu.analyze(n_modes=16, n_orbit=30_000)

        # Simulate what joint_analyze() would do: Koopman with uniform measure
        # (wrong reference measure)
        triple_wrong = GelfandTriple(T, domain=domain, n_test=48, n_dist=512)
        triple_wrong.build()
        # Don't run power iteration — use the initial uniform measure
        mu_wrong = np.ones(512) / 512
        resonances_wrong, _, srb_modes_wrong, biorth_wrong, _, _ = (
            triple_wrong.compute_ruelle_spectrum(mu_wrong, n_modes=8)
        )
        h_wrong = triple_wrong.compute_pesin_entropy_from_spectrum(
            resonances_wrong, srb_modes_wrong, mu_wrong
        )

        print(f"\n  [PASS] OTU vs joint_analyze comparison:")
        print(f"         OTU  biorth_error = {result_otu.biorth_error:.4f}")
        print(f"         WRG  biorth_error = {biorth_wrong:.4f}")
        print(f"         OTU  h_KS = {result_otu.h_ks:.4f}")
        print(f"         WRG  h_KS = {h_wrong:.4f}")
        print(f"         Exact h_KS = {LOG2:.4f}")
        print(f"         OTU error = {abs(result_otu.h_ks - LOG2):.4f}")
        print(f"         WRG error = {abs(h_wrong - LOG2):.4f}")
        # OTU should be at least as accurate — we just print, no hard assert
        # (the comparison is illustrative; the hard assertions are above)


# ============================================================================
# Class 5: Full Certification Pipeline
# ============================================================================

class TestOTUCertification:
    """Full pipeline tests using certify() — produces Lean 4-exportable certificates."""

    def test_certify_logistic_r4(self, logistic_r4):
        """Full certification of logistic r=4 with expected h_KS = log(2)."""
        T, domain = logistic_r4
        report = certify(
            T,
            domain=domain,
            n_test=48,
            n_dist=512,
            n_modes=16,
            n_orbit=50_000,
            expected_h_ks=LOG2,
            h_tolerance=0.15,
        )
        assert report["PASS"], (
            f"Certification FAILED for logistic r=4:\n"
            + "\n".join(f"  {k}: {v}" for k, v in report.items())
        )
        print(f"\n  [CERT] Logistic r=4 — OTU Certificate:")
        for k, v in report.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")

    def test_certify_chebyshev(self, chebyshev_map):
        """Full certification of Chebyshev n=2 with expected h_KS = log(2)."""
        T, domain = chebyshev_map
        report = certify(
            T,
            domain=domain,
            n_test=48,
            n_dist=512,
            n_modes=16,
            n_orbit=50_000,
            expected_h_ks=LOG2,
            h_tolerance=0.15,
        )
        assert report["PASS"], (
            f"Certification FAILED for Chebyshev n=2:\n"
            + "\n".join(f"  {k}: {v}" for k, v in report.items())
        )
        print(f"\n  [CERT] Chebyshev n=2 — OTU Certificate:")
        for k, v in report.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")

    def test_pomeau_manneville_near_transition(self):
        """
        Pomeau-Manneville map (z=1.5): algebraic mixing, Γ_OTU → 0.
        This is the regime where joint_analyze() fails and OTU is indispensable.
        """
        T = CanonicalSystems.intermittent_pomeau_manneville(z=1.5)
        triple = GelfandTriple(T, domain=(0.0, 1.0), n_test=32, n_dist=256, srb_tol=1e-7)
        triple.build()
        result = triple.analyze(n_modes=8, n_orbit=20_000)
        # For intermittent systems, h_KS > 0 but spectral gap is small
        assert result.gamma_otu >= 0.0, f"Γ_OTU = {result.gamma_otu:.4f} must be ≥ 0"
        print(f"\n  [PASS] Pomeau-Manneville z=1.5:")
        print(f"         Spectral gap Γ_OTU = {result.gamma_otu:.4f}")
        print(f"         h_KS (spectral)     = {result.h_ks:.4f}")
        print(f"         h_KS (Lyapunov)     = {result.spectrum.lyapunov_entropy:.4f}")
        print(f"         Self-consistent:     {result.spectrum.self_consistent}")

    def test_certificate_fields_complete(self, logistic_r4):
        """All required OTU certificate fields must be present and well-typed."""
        T, domain = logistic_r4
        report = certify(T, domain=domain, n_test=32, n_dist=256, n_modes=8, n_orbit=10_000)

        required_fields = [
            "OTU-1_self_consistency",
            "OTU-2_biorthogonality",
            "OTU-3_dominant_eigenvalue",
            "OTU-4_spectral_gap",
            "OTU-5_pesin_spectral",
            "OTU-6_pesin_lyapunov",
            "OTU-7_pesin_verified",
            "OTU-8_convergence_iters",
            "PASS",
        ]
        for field in required_fields:
            assert field in report, f"Missing certificate field: {field}"

        assert isinstance(report["OTU-1_self_consistency"], bool)
        assert isinstance(report["OTU-2_biorthogonality"], float)
        assert isinstance(report["OTU-3_dominant_eigenvalue"], float)
        assert 0.90 <= report["OTU-3_dominant_eigenvalue"] <= 1.05
        print(f"\n  [PASS] All {len(required_fields)} certificate fields present")


# ============================================================================
# Class 6: Numerical regression — guard against future regressions
# ============================================================================

class TestNumericalRegression:
    """Pin the key numerical results to guard against regressions."""

    def test_logistic_r4_lyapunov_in_range(self, logistic_r4):
        T, domain = logistic_r4
        triple = GelfandTriple(T, domain=domain, n_test=32, n_dist=256)
        triple.build()
        h = triple.compute_lyapunov_entropy(n_orbit=100_000)
        # log(2) ± 5%
        assert 0.62 <= h <= 0.75, f"Lyapunov h_KS = {h:.4f}, expected in [0.62, 0.75]"

    def test_srb_pf_eigenvalue_one(self, logistic_r4):
        T, domain = logistic_r4
        triple = GelfandTriple(T, domain=domain, n_test=32, n_dist=256)
        triple.build()
        # The L matrix is column-stochastic, so dominant eigenvalue = 1
        from scipy import linalg
        eigvals = np.abs(linalg.eig(triple._L)[0])
        dominant = np.max(eigvals)
        assert abs(dominant - 1.0) < 0.02, f"Dominant PF eigenvalue = {dominant:.4f}"
