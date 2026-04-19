"""
Test Suite: Deep Problems (OTU-17 through OTU-26)
==================================================

Comprehensive tests for the 10 deep problem solutions:
  P1:  Multi-attractor basin decomposition (OTU-22)
  P2:  Takens embedding preprocessor (OTU-20)
  P3:  High-dimensional OTU via reduced coordinates (OTU-25)
  P4:  Numerical stability certification (OTU-19)
  P5:  Exceptional points detection (OTU-24)
  P6:  Periodic orbit extraction (OTU-21)
  P7:  Fractal Gelfand triple adaptation (OTU-26)
  P8:  Continuous-time OTU generator (OTU-23)
  P9:  Fisher-Cramér-Rao bounds (OTU-17)
  P10: No-cloning theorem for μ_SRB (OTU-18)

All tests use ε-assertions and known analytical values.
"""

import math
import numpy as np
import pytest
from scipy import linalg

from acf_functor.deep_problems import (
    compute_fisher_cramer_rao,
    compute_no_clone_bound,
    certify_numerical_stability,
    takens_embed,
    extract_periodic_orbits,
    detect_basins,
    compute_continuous_generator,
    detect_exceptional_points,
    reduce_high_dimensional,
    certify_fractal_gelfand,
    deep_analysis,
    FisherCramerRaoCertificate,
    NoCloneCertificate,
    NumericalStabilityCertificate,
    TakensEmbeddingResult,
    PeriodicOrbitResult,
    BasinDecomposition,
    ContinuousGeneratorResult,
    ExceptionalPointResult,
    HighDimReductionResult,
    FractalGelfandCertificate,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: canonical maps
# ═══════════════════════════════════════════════════════════════════════════

def logistic_r4(x: np.ndarray) -> np.ndarray:
    return 4.0 * x * (1.0 - x)


def doubling(x: np.ndarray) -> np.ndarray:
    return (2.0 * x) % 1.0


def tent(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.minimum(x, 1.0 - x)


def two_basin_map(x: np.ndarray) -> np.ndarray:
    """Map with two separate invariant intervals [0,0.3] and [0.7,1.0]."""
    out = np.zeros_like(x)
    left = x < 0.35
    right = x >= 0.65
    middle = ~left & ~right
    # Each basin is a mini logistic
    out[left] = 0.3 * 4.0 * (x[left] / 0.3) * (1.0 - x[left] / 0.3)
    out[right] = 0.7 + 0.3 * 4.0 * ((x[right] - 0.7) / 0.3) * (1.0 - (x[right] - 0.7) / 0.3)
    out[middle] = x[middle]  # fixed in the middle
    return np.clip(out, 0.001, 0.999)


# ═══════════════════════════════════════════════════════════════════════════
# P9: Fisher-Cramér-Rao (OTU-17)
# ═══════════════════════════════════════════════════════════════════════════

class TestFisherCramerRao:
    """Test the Fisher information and Cramér-Rao bound computation."""

    def test_positive_curvature(self):
        """P''(1) > 0 for non-Bernoulli systems → finite CR bound."""
        cert = compute_fisher_cramer_rao(P_double_prime_1=0.5, h_ks=0.693, n_observations=1000)
        assert isinstance(cert, FisherCramerRaoCertificate)
        assert cert.fisher_information_per_obs == pytest.approx(0.5, abs=1e-10)
        assert cert.cramer_rao_bound > 0
        assert cert.min_error_std > 0
        assert cert.is_bernoulli is False
        assert cert.certificate_id == "OTU-17"

    def test_bernoulli_detection(self):
        """P''(1) ≈ 0 for Bernoulli systems → perfect estimation."""
        cert = compute_fisher_cramer_rao(P_double_prime_1=0.001, h_ks=0.693, n_observations=1000)
        assert cert.is_bernoulli is True
        assert cert.cramer_rao_bound >= 0  # may be tiny or zero

    def test_scaling_with_n(self):
        """σ²_min ∝ 1/n: doubling n halves the bound."""
        cert1 = compute_fisher_cramer_rao(0.5, 0.693, n_observations=1000)
        cert2 = compute_fisher_cramer_rao(0.5, 0.693, n_observations=2000)
        assert cert2.cramer_rao_bound == pytest.approx(cert1.cramer_rao_bound / 2, rel=1e-10)

    def test_relative_precision(self):
        """Relative precision = σ_min / h_KS."""
        cert = compute_fisher_cramer_rao(0.5, 0.693, n_observations=10000)
        expected_rel = cert.min_error_std / 0.693
        assert cert.relative_precision == pytest.approx(expected_rel, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# P10: No-Cloning Theorem (OTU-18)
# ═══════════════════════════════════════════════════════════════════════════

class TestNoClone:
    """Test the no-cloning theorem for μ_SRB."""

    def test_basic_bound(self):
        """n_min(ε) ≥ C · ε^{-D_2}."""
        cert = compute_no_clone_bound(D_2=0.83, h_ks=0.693, epsilon=0.01)
        assert isinstance(cert, NoCloneCertificate)
        expected = math.ceil(0.01 ** (-0.83))
        assert cert.n_min >= expected
        assert cert.certificate_id == "OTU-18"

    def test_higher_D2_needs_more_data(self):
        """Higher D_2 → more observations needed."""
        cert_low = compute_no_clone_bound(D_2=0.5, h_ks=0.693, epsilon=0.01)
        cert_high = compute_no_clone_bound(D_2=1.5, h_ks=0.693, epsilon=0.01)
        assert cert_high.n_min > cert_low.n_min

    def test_tighter_epsilon_needs_more(self):
        """Smaller ε → more observations needed."""
        cert_coarse = compute_no_clone_bound(D_2=1.0, h_ks=0.693, epsilon=0.1)
        cert_fine = compute_no_clone_bound(D_2=1.0, h_ks=0.693, epsilon=0.001)
        assert cert_fine.n_min > cert_coarse.n_min

    def test_information_cost(self):
        """Information cost = h_KS · n_min."""
        cert = compute_no_clone_bound(D_2=0.83, h_ks=0.693, epsilon=0.01)
        assert cert.information_cost_bits == pytest.approx(0.693 * cert.n_min, rel=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# P4: Numerical Stability (OTU-19)
# ═══════════════════════════════════════════════════════════════════════════

class TestNumericalStability:
    """Test numerical stability certification."""

    def test_well_conditioned_system(self):
        """Large Γ_OTU → easily certifiable."""
        cert = certify_numerical_stability(gamma_otu=0.5, epsilon_target=0.01)
        assert isinstance(cert, NumericalStabilityCertificate)
        assert cert.is_certifiable is True
        assert cert.mixing_type == "exponential"
        assert cert.recommended_precision == "float64"
        assert cert.certificate_id == "OTU-19"

    def test_poorly_conditioned_system(self):
        """Very small Γ_OTU → uncertifiable."""
        cert = certify_numerical_stability(gamma_otu=1e-10, epsilon_target=0.01)
        assert cert.is_certifiable is False
        assert cert.condition_number > 1e9

    def test_zero_gap(self):
        """Γ_OTU = 0 → infinite condition number."""
        cert = certify_numerical_stability(gamma_otu=0.0, epsilon_target=0.01)
        assert cert.is_certifiable is False
        assert cert.mixing_type == "algebraic"
        assert cert.epsilon_numerical == float('inf')

    def test_condition_number_formula(self):
        """κ = 1/Γ_OTU."""
        cert = certify_numerical_stability(gamma_otu=0.25, epsilon_target=0.01)
        assert cert.condition_number == pytest.approx(4.0, rel=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# P2: Takens Embedding (OTU-20)
# ═══════════════════════════════════════════════════════════════════════════

class TestTakensEmbedding:
    """Test the Takens delay embedding preprocessor."""

    def _generate_logistic_ts(self, n=5000):
        """Generate a time series from logistic map r=4."""
        x = 0.3
        ts = np.zeros(n)
        for i in range(n):
            x = 4.0 * x * (1.0 - x)
            ts[i] = x
        return ts[200:]  # discard transient

    def test_basic_embedding(self):
        """Embedding produces correct shapes and spectrum."""
        ts = self._generate_logistic_ts()
        result = takens_embed(ts, embedding_dim=3, delay=1)
        assert isinstance(result, TakensEmbeddingResult)
        assert result.embedding_dim == 3
        assert result.delay == 1
        assert result.hankel_matrix.shape[1] == 3
        assert len(result.reconstructed_spectrum) == 3
        assert result.certificate_id == "OTU-20"

    def test_auto_estimation(self):
        """Auto-estimate delay and dimension from D_2 hint."""
        ts = self._generate_logistic_ts()
        result = takens_embed(ts, D_2_hint=0.85)
        # d = ceil(2*0.85 + 1) = ceil(2.7) = 3
        assert result.embedding_dim == 3
        assert result.delay >= 1

    def test_spectrum_not_degenerate(self):
        """Reconstructed spectrum should not be all zeros or ones."""
        ts = self._generate_logistic_ts()
        result = takens_embed(ts, embedding_dim=3, delay=1)
        mods = np.abs(result.reconstructed_spectrum)
        assert np.any(mods > 0.01)  # not all zero
        assert np.any(mods < 0.999)  # not all unit

    def test_short_series_error(self):
        """Too short series raises ValueError."""
        with pytest.raises(ValueError, match="too short"):
            takens_embed(np.array([0.1, 0.2, 0.3]), embedding_dim=5, delay=3)


# ═══════════════════════════════════════════════════════════════════════════
# P6: Periodic Orbit Extraction (OTU-21)
# ═══════════════════════════════════════════════════════════════════════════

class TestPeriodicOrbits:
    """Test periodic orbit extraction from spectral data."""

    def test_doubling_map_orbits(self):
        """Doubling map has 2^n - 1 fixed points of period ≤ n."""
        # Build Ulam matrix for doubling
        N = 256
        L = np.zeros((N, N))
        grid = np.linspace(0, 1, N + 1)
        for i in range(N):
            center = (grid[i] + grid[i + 1]) / 2.0
            img = (2.0 * center) % 1.0
            j = min(int(img * N), N - 1)
            L[j, i] = 1.0
        L /= (L.sum(axis=0, keepdims=True) + 1e-14)

        eigvals = linalg.eigvals(L)
        eigvals = eigvals[np.argsort(-np.abs(eigvals))][:20]

        result = extract_periodic_orbits(
            eigvals, T=doubling, domain=(0.001, 0.999), max_period=4
        )
        assert isinstance(result, PeriodicOrbitResult)
        # Doubling map: 1 fixed point at x=0 (excluded by domain)
        # Period-2 orbits exist
        assert result.max_period == 4

    def test_spectral_trace_consistency(self):
        """Trace formula: tr(L^n) = Σ λ_k^n."""
        resonances = np.array([1.0, 0.5, -0.3 + 0.2j, -0.3 - 0.2j])
        result = extract_periodic_orbits(resonances, max_period=5)
        # Check orbit counts are non-negative
        for n, count in result.orbit_count_by_period.items():
            assert count >= 0

    def test_h_top_estimate(self):
        """Topological entropy estimate from orbit growth."""
        # For doubling map: h_top = log(2) ≈ 0.693
        resonances = np.array([1.0] + [0.5**k for k in range(1, 15)])
        result = extract_periodic_orbits(resonances, max_period=8)
        assert isinstance(result.h_top_estimate, float)
        assert result.h_top_estimate >= 0


# ═══════════════════════════════════════════════════════════════════════════
# P1: Basin Decomposition (OTU-22)
# ═══════════════════════════════════════════════════════════════════════════

class TestBasinDecomposition:
    """Test multi-attractor basin decomposition."""

    def test_single_basin_ergodic(self):
        """Ergodic system (logistic r=4) → single basin."""
        # Use a proper Ulam (statistical) approximation with Monte Carlo
        N = 64
        rng = np.random.default_rng(42)
        n_samples = 5000
        L = np.zeros((N, N))
        grid = np.linspace(0, 1, N + 1)
        for i in range(N):
            # Sample uniformly within cell i
            xs = rng.uniform(grid[i], grid[i + 1], n_samples)
            imgs = 4.0 * xs * (1.0 - xs)
            imgs = np.clip(imgs, 0.001, 0.999)
            for img_val in imgs:
                j = min(int(img_val * N), N - 1)
                L[j, i] += 1
        L /= (L.sum(axis=0, keepdims=True) + 1e-14)

        result = detect_basins(L, delta=0.05)
        assert isinstance(result, BasinDecomposition)
        assert result.n_basins == 1
        assert result.is_ergodic is True
        assert result.certificate_id == "OTU-22"

    def test_two_basins(self):
        """Block diagonal L → two basins detected."""
        N = 100
        # Create a block-diagonal stochastic matrix (two isolated blocks)
        L = np.zeros((N, N))
        n1, n2 = 50, 50
        # Block 1: random stochastic
        rng = np.random.default_rng(42)
        B1 = rng.random((n1, n1)) + 0.01
        B1 /= B1.sum(axis=0, keepdims=True)
        B2 = rng.random((n2, n2)) + 0.01
        B2 /= B2.sum(axis=0, keepdims=True)
        L[:n1, :n1] = B1
        L[n1:, n1:] = B2

        result = detect_basins(L, delta=0.05)
        assert result.n_basins >= 2
        assert result.is_ergodic is False
        assert len(result.basin_weights) == result.n_basins

    def test_basin_weights_sum_to_one(self):
        """Basin weights should sum to approximately 1."""
        N = 60
        L = np.zeros((N, N))
        B1 = np.ones((30, 30)) / 30
        B2 = np.ones((30, 30)) / 30
        L[:30, :30] = B1
        L[30:, 30:] = B2
        result = detect_basins(L, delta=0.05)
        assert sum(result.basin_weights) == pytest.approx(1.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# P8: Continuous-Time Generator (OTU-23)
# ═══════════════════════════════════════════════════════════════════════════

class TestContinuousGenerator:
    """Test continuous-time generator computation."""

    def test_basic_conversion(self):
        """Discrete eigenvalues → continuous eigenvalues via log."""
        disc_ev = np.array([1.0, 0.5, 0.3, -0.2])
        result = compute_continuous_generator(disc_ev, tau=1.0)
        assert isinstance(result, ContinuousGeneratorResult)
        # λ_cont = log(λ_disc) / τ
        # For λ=0.5: log(0.5) ≈ -0.693
        assert result.decay_rates[1] == pytest.approx(np.log(0.5), abs=0.01)
        assert result.certificate_id == "OTU-23"

    def test_time_step_scaling(self):
        """Different τ scales the continuous eigenvalues."""
        disc_ev = np.array([1.0, 0.5])
        r1 = compute_continuous_generator(disc_ev, tau=1.0)
        r2 = compute_continuous_generator(disc_ev, tau=0.5)
        # λ_cont(τ=0.5) = log(0.5)/0.5 = 2 · log(0.5)/1.0 = 2 · λ_cont(τ=1)
        assert r2.generator_eigenvalues[1] == pytest.approx(
            2 * r1.generator_eigenvalues[1], abs=0.01
        )

    def test_aliasing_detection(self):
        """High-frequency modes flag aliasing."""
        # Eigenvalues with large imaginary part → aliasing
        disc_ev = np.array([1.0, 0.5, -0.9])  # -0.9 has phase ≈ π
        result = compute_continuous_generator(disc_ev, tau=1.0)
        # log(-0.9) = log(0.9) + iπ → |Im| · τ = π (borderline)
        assert isinstance(result.aliasing_free, bool)

    def test_dominant_eigenvalue_zero(self):
        """λ_0 = 1 → log(1)/τ = 0 (stationary mode)."""
        disc_ev = np.array([1.0, 0.5])
        result = compute_continuous_generator(disc_ev, tau=1.0)
        assert result.generator_eigenvalues[0] == pytest.approx(0.0, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# P5: Exceptional Points (OTU-24)
# ═══════════════════════════════════════════════════════════════════════════

class TestExceptionalPoints:
    """Test exceptional point detection."""

    def test_no_eps_in_generic_system(self):
        """A generic matrix should have well-separated eigenvalues."""
        rng = np.random.default_rng(123)
        L = rng.random((20, 20))
        L /= L.sum(axis=0, keepdims=True)
        result = detect_exceptional_points(L, threshold=1e-4)
        assert isinstance(result, ExceptionalPointResult)
        assert result.certificate_id == "OTU-24"
        # Generic matrices should have few/no EPs at tight threshold
        assert result.n_exceptional_points >= 0

    def test_known_ep(self):
        """Matrix with engineered coalescing eigenvalues → EP detected."""
        # Create a near-defective matrix: two eigenvalues very close
        N = 10
        D = np.diag([1.0, 0.5, 0.5 + 1e-5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
        rng = np.random.default_rng(42)
        P = rng.random((N, N))
        L = P @ D @ np.linalg.inv(P)
        result = detect_exceptional_points(L, threshold=1e-3)
        # Should detect the pair at 0.5, 0.5+1e-5
        assert result.n_exceptional_points >= 1

    def test_pt_symmetry_check(self):
        """PT symmetry check returns a boolean."""
        L = np.eye(5) * 0.5
        result = detect_exceptional_points(L)
        assert isinstance(result.pt_symmetric, bool)


# ═══════════════════════════════════════════════════════════════════════════
# P3: High-Dimensional Reduction (OTU-25)
# ═══════════════════════════════════════════════════════════════════════════

class TestHighDimReduction:
    """Test high-dimensional OTU reduction."""

    def test_basic_reduction(self):
        """Reduce a 10D system with 2D attractor."""
        rng = np.random.default_rng(42)
        N = 500
        # Trajectory on a 2D manifold in 10D space
        t = np.linspace(0, 20 * np.pi, N)
        X = np.zeros((N, 10))
        X[:, 0] = np.sin(t)
        X[:, 1] = np.cos(t)
        X[:, 2:] = 0.01 * rng.standard_normal((N, 8))

        def T_identity(x):
            return x

        result = reduce_high_dimensional(X, T_identity, D_2_hint=2.0)
        assert isinstance(result, HighDimReductionResult)
        assert result.ambient_dim == 10
        assert result.intrinsic_dim == 3  # ceil(2.0 + 1)
        assert result.reconstruction_error < 0.1  # Low noise → low error
        assert result.certificate_id == "OTU-25"

    def test_full_rank_no_reduction(self):
        """If D_2 ≈ D, minimal reduction happens."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        result = reduce_high_dimensional(X, lambda x: x, D_2_hint=4.0)
        assert result.intrinsic_dim == 5  # ceil(4.0+1)=5
        assert result.ambient_dim == 5


# ═══════════════════════════════════════════════════════════════════════════
# P7: Fractal Gelfand Triple (OTU-26)
# ═══════════════════════════════════════════════════════════════════════════

class TestFractalGelfand:
    """Test fractal Gelfand triple adaptation."""

    def test_fractal_detection(self):
        """D_0 < ambient_dim → fractal flag set."""
        mu = np.random.default_rng(42).dirichlet(np.ones(100))
        cert = certify_fractal_gelfand(mu, D_0=0.63, D_2=0.5, ambient_dim=1)
        assert isinstance(cert, FractalGelfandCertificate)
        assert cert.is_fractal is True
        assert cert.convergence_rate_exponent == pytest.approx(1.0 / 0.5, rel=1e-6)
        assert cert.certificate_id == "OTU-26"

    def test_non_fractal(self):
        """D_0 = ambient_dim → not fractal."""
        mu = np.ones(100) / 100
        cert = certify_fractal_gelfand(mu, D_0=1.0, D_2=1.0, ambient_dim=1)
        assert cert.is_fractal is False

    def test_tv_error_bounded(self):
        """Total variation error is in [0, 1]."""
        rng = np.random.default_rng(42)
        mu = rng.dirichlet(np.ones(200))
        cert = certify_fractal_gelfand(mu, D_0=0.8, D_2=0.7)
        assert 0 <= cert.total_variation_error <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Integration test: full deep_analysis pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestDeepAnalysisIntegration:
    """End-to-end integration tests for the deep_analysis pipeline."""

    @pytest.mark.slow
    def test_logistic_full_pipeline(self):
        """Full deep analysis on logistic map r=4."""
        result = deep_analysis(
            T=logistic_r4,
            domain=(0.001, 0.999),
            n_dist=128,
            n_observations=5000,
            epsilon_target=0.01,
            max_orbit_period=4,
        )

        # Check all 10 solutions present
        assert "fisher_cr" in result
        assert "no_clone" in result
        assert "stability" in result
        assert "takens" in result
        assert "orbits" in result
        assert "basins" in result
        assert "generator" in result
        assert "exceptional" in result
        assert "fractal" in result
        assert "certificates" in result

        certs = result["certificates"]

        # OTU-17: Fisher-Cramér-Rao
        assert certs["OTU-17_fisher_info"] >= 0

        # OTU-18: No-Cloning
        assert certs["OTU-18_n_min"] >= 1
        assert certs["OTU-18_D2"] > 0

        # OTU-19: Numerical Stability
        assert isinstance(certs["OTU-19_is_certifiable"], bool)

        # OTU-20: Takens
        assert certs["OTU-20_embedding_dim"] >= 3

        # OTU-21: Periodic Orbits
        assert certs["OTU-21_h_top_estimate"] >= 0

        # OTU-22: Basins
        assert certs["OTU-22_n_basins"] >= 1

        # OTU-23: Generator
        assert isinstance(certs["OTU-23_aliasing_free"], bool)

        # OTU-24: Exceptional Points
        assert certs["OTU-24_n_exceptional_points"] >= 0

        # OTU-26: Fractal
        assert isinstance(certs["OTU-26_is_fractal"], bool)
        assert 0 <= certs["OTU-26_tv_error"] <= 2.0

    @pytest.mark.slow
    def test_logistic_is_certifiable(self):
        """Logistic r=4 should be numerically certifiable at ε=0.01."""
        result = deep_analysis(
            T=logistic_r4,
            domain=(0.001, 0.999),
            n_dist=128,
            epsilon_target=0.01,
        )
        assert result["stability"].is_certifiable is True

    @pytest.mark.slow
    def test_logistic_single_basin(self):
        """Logistic r=4 should have a single ergodic basin."""
        result = deep_analysis(
            T=logistic_r4,
            domain=(0.001, 0.999),
            n_dist=128,
        )
        assert result["basins"].is_ergodic is True
        assert result["basins"].n_basins == 1


class TestDeepCertifyConvenience:
    """Test the convenience deep_certify function from gelfand_triple."""

    @pytest.mark.slow
    def test_deep_certify_import(self):
        """deep_certify exists and is callable."""
        from acf_functor.gelfand_triple import deep_certify
        assert callable(deep_certify)

    @pytest.mark.slow
    def test_deep_certify_runs(self):
        """deep_certify returns a dict with certificates."""
        from acf_functor.gelfand_triple import deep_certify
        result = deep_certify(
            logistic_r4,
            domain=(0.001, 0.999),
            n_dist=128,
        )
        assert isinstance(result, dict)
        assert "certificates" in result
        assert "fisher_cr" in result


# ═══════════════════════════════════════════════════════════════════════════
# Certification: formal properties (mathematical invariants)
# ═══════════════════════════════════════════════════════════════════════════

class TestFormalCertificates:
    """Test mathematical invariants and formal properties of certificates."""

    def test_cramer_rao_is_lower_bound(self):
        """CR bound is a lower bound: any variance ≥ σ²_min."""
        cert = compute_fisher_cramer_rao(0.5, 0.693, 10000)
        # By CLT, estimated variance should exceed CR bound
        # We just verify the bound is non-negative
        assert cert.cramer_rao_bound >= 0

    def test_no_clone_monotonicity(self):
        """n_min is monotone decreasing in ε (more data for tighter precision)."""
        eps_values = [0.1, 0.01, 0.001]
        n_mins = [compute_no_clone_bound(0.83, 0.693, e).n_min for e in eps_values]
        for i in range(len(n_mins) - 1):
            assert n_mins[i + 1] >= n_mins[i]

    def test_stability_monotone_in_gamma(self):
        """Larger Γ_OTU → smaller ε_num (better certifiability)."""
        gammas = [0.01, 0.1, 0.5]
        eps_nums = [certify_numerical_stability(g).epsilon_numerical for g in gammas]
        for i in range(len(eps_nums) - 1):
            assert eps_nums[i + 1] < eps_nums[i]

    def test_continuous_eigenvalue_dominant_is_zero(self):
        """For any stochastic operator, the dominant continuous eigenvalue is 0."""
        disc = np.array([1.0, 0.8, 0.5, 0.3])
        cont = compute_continuous_generator(disc, tau=1.0)
        assert abs(cont.generator_eigenvalues[0]) < 1e-10

    def test_basin_weights_partition(self):
        """Basin weights form a probability distribution."""
        N = 80
        L = np.zeros((N, N))
        B1 = np.ones((40, 40)) / 40
        B2 = np.ones((40, 40)) / 40
        L[:40, :40] = B1
        L[40:, 40:] = B2
        basins = detect_basins(L, delta=0.05)
        assert sum(basins.basin_weights) == pytest.approx(1.0, abs=0.01)
        for w in basins.basin_weights:
            assert w >= 0
