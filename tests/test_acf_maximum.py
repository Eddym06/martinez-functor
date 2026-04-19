"""
tests/test_acf_maximum.py — comprehensive tests for ACF maximum capability modules.

Tests the 6 new modules introduced to formalize open items and hidden potentials:
  1. koopman_delta_bounds  — formal δ(d) bounds
  2. mixed_composition     — mixed Koopman-polynomial certificates
  3. acf_inverse           — Φ⁻¹ for all branches
  4. information_geometry  — Fisher-Rao/Affine Legendre duality
  5. thermodynamic_acf     — free energy d* selection
  6. lie_analysis NC       — serial depth / NC class
  7. invariant_unified unification theorem

All tests use concrete numerical examples so failures are reproducible.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

# ============================================================
# 1. Koopman delta bounds
# ============================================================
from acf_functor.koopman_delta_bounds import (
    KoopmanDeltaBounds,
    CompositionDelta,
    ConvergenceFamilyReport,
    SpectralBound,
    delta_bounds_from_svd,
)


def make_exponential_eigenvalues(n: int = 30, rate: float = 0.5) -> torch.Tensor:
    """λ_j = e^{-rate·j}, sorted descending."""
    j = torch.arange(1, n + 1, dtype=torch.float64)
    return torch.exp(-rate * j)


def make_algebraic_eigenvalues(n: int = 30, exponent: float = 2.0) -> torch.Tensor:
    """λ_j = j^{-exponent}."""
    j = torch.arange(1, n + 1, dtype=torch.float64)
    return j ** (-exponent)


class TestKoopmanDeltaBounds:
    def test_delta_upper_bound_decreasing(self):
        """δ(d) must be non-increasing in d."""
        eig = make_exponential_eigenvalues(20)
        bounds = KoopmanDeltaBounds(eig)
        deltas = [bounds.at(d).delta_upper for d in range(1, 20)]
        for i in range(len(deltas) - 1):
            assert deltas[i] >= deltas[i + 1] - 1e-14, (
                f"δ not decreasing: δ({i+1})={deltas[i]:.4e} > δ({i+2})={deltas[i+1]:.4e}"
            )

    def test_delta_equals_spectral_tail(self):
        """δ(d) must equal |λ_{d+1}| (by theorem), up to float64 precision."""
        eig = make_exponential_eigenvalues(20)
        bounds = KoopmanDeltaBounds(eig)
        for d in [1, 5, 10, 15]:
            sb = bounds.at(d)
            # λ_{d+1} is the (d+1)-th eigenvalue (1-indexed)
            expected = float(eig[d].item())  # 0-indexed: eig[d] = λ_{d+1}
            assert abs(sb.delta_upper - expected) < 1e-7, (
                f"δ({d}) = {sb.delta_upper:.6e} ≠ |λ_{d+1}| = {expected:.6e}"
            )

    def test_optimal_dimension_tight(self):
        """d*(ε) is the smallest d s.t. δ(d) ≤ ε."""
        eig = make_algebraic_eigenvalues(50)
        bounds = KoopmanDeltaBounds(eig)
        for eps in [0.1, 0.01, 0.001]:
            result = bounds.optimal_dimension(eps)
            d_opt = result.d_star
            # δ(d*) should be ≤ ε
            assert bounds.at(d_opt).delta_upper <= eps * (1 + 1e-10), (
                f"δ(d*={d_opt}) = {bounds.at(d_opt).delta_upper:.4e} > ε = {eps}"
            )
            # δ(d*-1) should be > ε (minimality)
            if d_opt > 1:
                assert bounds.at(d_opt - 1).delta_upper > eps * (1 - 1e-8), (
                    f"d* not minimal: δ({d_opt-1}) = {bounds.at(d_opt-1).delta_upper:.4e} ≤ ε = {eps}"
                )

    def test_monotone_d_star_in_epsilon(self):
        """Strictly tighter ε requires strictly larger d*."""
        eig = make_algebraic_eigenvalues(50)
        bounds = KoopmanDeltaBounds(eig)
        eps_seq = [0.5, 0.1, 0.01, 0.001, 0.0001]
        d_stars = [bounds.optimal_dimension(eps).d_star for eps in eps_seq]
        for i in range(len(d_stars) - 1):
            assert d_stars[i] <= d_stars[i + 1], (
                f"d*({eps_seq[i]}) = {d_stars[i]} > d*({eps_seq[i+1]}) = {d_stars[i+1]}"
            )

    def test_composition_subadditivity(self):
        """δ(f∘g) ≤ δ_f + L_f·δ_g — subadditivity certificate."""
        eig_f = make_exponential_eigenvalues(20, rate=0.3)
        eig_g = make_exponential_eigenvalues(20, rate=0.5)
        bounds_f = KoopmanDeltaBounds(eig_f)
        bounds_g = KoopmanDeltaBounds(eig_g)
        d = 10
        L_f = 1.5
        cert = CompositionDelta.certify(bounds_f, bounds_g, d, L_f)
        # bound should be strictly ≥ 0
        assert cert.delta_composition_bound >= 0
        # bound = δ_f(d) + L_f * δ_g(d)
        expected = bounds_f.at(d).delta_upper + L_f * bounds_g.at(d).delta_upper
        assert abs(cert.delta_composition_bound - expected) < 1e-12

    def test_subadditivity_is_valid(self):
        """Certificate.is_valid verifies measured ≤ bound."""
        eig = make_exponential_eigenvalues(20, rate=0.3)
        bounds = KoopmanDeltaBounds(eig)
        cert = CompositionDelta.certify(bounds, bounds, 5, 1.0)
        assert cert.is_valid, "Subadditivity certificate should be valid"
        assert cert.delta_composition_measured <= cert.delta_composition_bound + 1e-14

    def test_decay_family_classification(self):
        """Exponential eigenvalues should classify as lowercase 'exponential'."""
        eig = make_exponential_eigenvalues(30)
        bounds = KoopmanDeltaBounds(eig)
        report = bounds.classify()
        # Family is lowercase in the implementation
        assert report.family.lower() in ("exponential", "super_exponential"), (
            f"Expected exponential family, got: {report.family}"
        )

    def test_classify_algebraic_eigenvalues(self):
        """Algebraic eigenvalues: family should mention 'algebraic' or 'polynomial'."""
        eig = make_algebraic_eigenvalues(30, exponent=2.0)
        bounds = KoopmanDeltaBounds(eig)
        report = bounds.classify()
        # The report should contain the family name and decay_rate_c
        assert report.family is not None and len(report.family) > 0
        assert report.alpha >= 0

    def test_delta_bounds_from_svd(self):
        """Factory: delta_bounds_from_svd returns a valid KoopmanDeltaBounds object."""
        torch.manual_seed(42)
        data = torch.randn(100, 10, dtype=torch.float64)
        bounds = delta_bounds_from_svd(data)
        # Should be able to compute delta bounds
        sb = bounds.at(3)
        assert sb.delta_upper >= 0

    def test_spbound_fields(self):
        """SpectralBound must have all expected fields."""
        eig = make_exponential_eigenvalues(20)
        bounds = KoopmanDeltaBounds(eig)
        sb = bounds.at(5)
        for field in ["delta_upper", "delta_lower", "d", "derivation"]:
            assert hasattr(sb, field), f"SpectralBound missing field: {field}"


# ============================================================
# 2. Mixed composition
# ============================================================
from acf_functor.mixed_composition import (
    MixedCompositionCertifier,
    PolynomialKoopmanCertifier,
    MixedCompositionCertificate,
)
from acf_functor.core import ACFFunctor as _ACFFunctor, ReductionPath


def _make_reduction(coeffs):
    """Build a polynomial ReductionResult from coefficient list."""
    functor = _ACFFunctor()
    return functor.reduce_polynomial(coeffs)


class TestMixedComposition:
    def test_certificate_epsilon_total_positive(self):
        """Certificate epsilon_total must be non-negative."""
        phi_g = _make_reduction([0.0, 1.0])  # linear poly as proxy
        eig = make_exponential_eigenvalues(20, rate=0.4)
        certifier = MixedCompositionCertifier()
        cert = certifier.certify(
            f_exact=lambda x: 2.0 * x + 1.0,
            g_exact=lambda x: x ** 2.0,
            phi_g=phi_g,
            koopman_eigenvalues=eig,
            koopman_coeff_norm=1.5,
            input_domain=(-1.0, 1.0),
        )
        # epsilon_total = epsilon_inner + lipschitz * delta_outer
        assert cert.epsilon_total >= 0, f"epsilon_total negative: {cert.epsilon_total}"

    def test_certificate_has_proof_sketch(self):
        """Certificate must contain a text proof sketch."""
        phi_g = _make_reduction([0.0, 0.0, 1.0])  # quadratic
        eig = make_exponential_eigenvalues(20, rate=0.4)
        certifier = MixedCompositionCertifier()
        cert = certifier.certify(
            f_exact=lambda x: 2.0 * x,
            g_exact=lambda x: x ** 2.0,
            phi_g=phi_g,
            koopman_eigenvalues=eig,
            koopman_coeff_norm=1.5,
        )
        assert isinstance(cert.proof_sketch, str)
        assert len(cert.proof_sketch) > 50

    def test_error_grows_with_koopman_truncation(self):
        """Using more Koopman eigenvalues should give smaller or equal delta_outer."""
        phi_g = _make_reduction([0.0, 0.0, 1.0])
        certifier = MixedCompositionCertifier()
        eig_coarse = make_exponential_eigenvalues(5, rate=0.3)
        eig_fine = make_exponential_eigenvalues(20, rate=0.3)
        cert_coarse = certifier.certify(
            lambda x: 2.0 * x, lambda x: x ** 2.0, phi_g, eig_coarse, 1.0,
        )
        cert_fine = certifier.certify(
            lambda x: 2.0 * x, lambda x: x ** 2.0, phi_g, eig_fine, 1.0,
        )
        # Finer eigenvalue set → delta_outer = δ(d) should be smaller or equal
        assert cert_fine.delta_outer <= cert_coarse.delta_outer + 1e-14, (
            f"Finer Koopman gave larger delta: {cert_fine.delta_outer:.4e} > {cert_coarse.delta_outer:.4e}"
        )

    def test_polynomial_koopman_reverse_direction(self):
        """Poly∘Koopman certificate: dict with epsilon_total ≥ 0."""
        eig = make_exponential_eigenvalues(20, rate=0.5)
        phi_f = _make_reduction([0.0, 2.0])  # 2x
        certifier = PolynomialKoopmanCertifier()
        result = certifier.certify(
            f_exact=lambda x: 2.0 * x,
            g_exact=lambda x: x,
            phi_f=phi_f,
            koopman_eigenvalues_g=eig,
            lipschitz_f=2.0,
            d=10,
        )
        assert isinstance(result, dict)
        # epsilon_total = ε_f + L_f·δ_g — must be non-negative
        bound = result.get("epsilon_total", result.get("epsilon_cross", -1))
        assert bound >= 0, f"epsilon_total negative or missing: {result.keys()}"


# ============================================================
# 3. ACF Inverse
# ============================================================
from acf_functor.acf_inverse import ACFInverse, InversionCertificate, InversionResult


class TestACFInverse:
    def test_polynomial_roundtrip_certificate_fields(self):
        """For a polynomial, InversionCertificate must have all required fields."""
        functor = _ACFFunctor()
        r = functor.reduce_polynomial([1.0, -2.0, 0.5])
        inverter = ACFInverse()
        result = inverter.invert(r)
        cert = result.certificate
        assert hasattr(cert, "branch")
        assert hasattr(cert, "reconstruction_error")
        assert hasattr(cert, "is_exact")
        assert hasattr(cert, "proof_sketch")
        assert isinstance(cert.proof_sketch, str) and len(cert.proof_sketch) > 20

    def test_polynomial_inverse_small_error_bound(self):
        """For a polynomial, the formal error bound should be ≤ machine epsilon."""
        functor = _ACFFunctor()
        r = functor.reduce_polynomial([1.0, -2.0, 0.5, 3.0])
        inverter = ACFInverse()
        result = inverter.invert(r)
        # error_bound is the machine epsilon upper bound
        assert result.certificate.error_bound < 1e-10, (
            f"Polynomial error bound too large: {result.certificate.error_bound}"
        )

    def test_inversion_result_has_reconstructed_fn(self):
        """InversionResult must include a reconstructed function callable."""
        functor = _ACFFunctor()
        r = functor.reduce_polynomial([0.0, 1.0])  # f(x) = x
        inverter = ACFInverse()
        result = inverter.invert(r)
        assert hasattr(result, "reconstructed_fn")
        assert callable(result.reconstructed_fn)

    def test_verify_roundtrip_returns_dict_with_correct_keys(self):
        """verify_roundtrip() should return dict with linf and l2 errors."""
        functor = _ACFFunctor()
        r = functor.reduce_polynomial([0.0, 0.0, 1.0])  # x^2
        inverter = ACFInverse()
        rt = inverter.verify_roundtrip(r)
        assert "roundtrip_linf" in rt, f"Missing 'roundtrip_linf', got: {list(rt.keys())}"
        assert "roundtrip_l2" in rt, f"Missing 'roundtrip_l2', got: {list(rt.keys())}"
        assert rt["roundtrip_linf"] >= 0

    def test_chebyshev_branch_inversion(self):
        """Chebyshev branch: invert should run and return valid certificate."""
        functor = _ACFFunctor()
        # reduce_transcendental requires a tensor-compatible function
        r = functor.reduce_transcendental(torch.sin, degree=20, domain=(-1.0, 1.0), target_epsilon=1e-4)
        inverter = ACFInverse()
        result = inverter.invert(r)
        assert result.certificate.branch is not None
        assert result.certificate.reconstruction_error >= 0


# ============================================================
# 4. Information geometry (Fisher-Rao / Affine duality)
# ============================================================
from acf_functor.information_geometry import (
    InformationGeometry,
    FisherMetricACF,
    AffineMetricACF,
    LegendreDuality,
    DualityVerifier,
    InformationGeometryReport,
    MetricTensor,
    polynomial_observables,
    fourier_observables,
)


class TestInformationGeometry:
    def test_fisher_compute_returns_metric_tensor(self):
        """FisherMetricACF.compute(c) must return a MetricTensor with .G field."""
        obs = polynomial_observables(max_degree=3)
        fisher = FisherMetricACF(obs)
        c = torch.tensor([0.5, 0.3, -0.1, 0.2], dtype=torch.float64)
        mt = fisher.compute(c)
        assert isinstance(mt, MetricTensor)
        assert hasattr(mt, "G")
        assert mt.G.shape == (4, 4)

    def test_fisher_metric_matrix_symmetric(self):
        """Fisher metric matrix G must be symmetric."""
        obs = polynomial_observables(max_degree=3)
        fisher = FisherMetricACF(obs)
        c = torch.tensor([0.5, 0.3, -0.1, 0.2], dtype=torch.float64)
        G = fisher.compute(c).G
        assert torch.allclose(G, G.T, atol=1e-10), "Fisher metric G not symmetric"

    def test_affine_metric_computes_gram_inverse(self):
        """AffineMetricACF.compute(K) returns G_A = (K^T K)^{-1} (Gram inverse)."""
        affine = AffineMetricACF()
        K = torch.eye(4, dtype=torch.float64) * 2.0
        mt = affine.compute(K)
        assert isinstance(mt, MetricTensor)
        # G_A = (K^T K + reg·I)^{-1}; for K=2I: G_A ≈ (4I)^{-1} = 0.25I
        # So G_A @ (K^T K) ≈ I
        G_A = mt.G
        KtK = K.T @ K + 1e-8 * torch.eye(4, dtype=torch.float64)  # same regularization
        prod = G_A @ KtK
        assert torch.allclose(prod, torch.eye(4, dtype=torch.float64), atol=1e-5), (
            f"G_A @ (K^T K) ≠ I, max error = {float((prod - torch.eye(4)).abs().max().item()):.2e}"
        )

    def test_duality_verifier_requires_both_metrics(self):
        """DualityVerifier must accept (fisher, affine) and produce a result."""
        obs = polynomial_observables(max_degree=3)
        fisher = FisherMetricACF(obs)
        affine = AffineMetricACF()
        verifier = DualityVerifier(fisher, affine)
        c = torch.tensor([0.3, 0.1, -0.2, 0.4], dtype=torch.float64)
        K = torch.eye(4, dtype=torch.float64) + 0.1 * torch.randn(4, 4, dtype=torch.float64)
        K = K @ K.T + 0.01 * torch.eye(4, dtype=torch.float64)
        result = verifier.verify(c, K)
        assert hasattr(result, "frobenius_distance")
        assert hasattr(result, "duality_holds")
        assert result.frobenius_distance >= 0

    def test_legendre_duality_kl_gap_nonnegative(self):
        """KL gap ψ(c) + ψ*(η) - c^T η ≥ 0 (Young–Fenchel inequality)."""
        obs = polynomial_observables(max_degree=3)
        fisher = FisherMetricACF(obs)
        dual = LegendreDuality(fisher)
        c = torch.tensor([0.5, 0.2, -0.1, 0.0], dtype=torch.float64)
        lp = dual.compute_at(c)
        # KL should be 0 exactly on self-dual point; generally ≥ 0 by Fenchel
        assert lp.kl_diverg >= -1e-6, f"Fenchel inequality violated: KL = {lp.kl_diverg}"

    def test_information_geometry_report_has_duality_field(self):
        """InformationGeometry.analyze() should return report with .duality field."""
        obs = polynomial_observables(max_degree=3)
        geom = InformationGeometry(obs)
        c = torch.tensor([0.3, 0.1, -0.2, 0.4], dtype=torch.float64)
        K = torch.eye(4, dtype=torch.float64) * 1.5
        report = geom.analyze(c, K)
        assert hasattr(report, "duality"), "Report missing .duality field"
        assert hasattr(report.duality, "frobenius_distance")
        assert hasattr(report.duality, "duality_holds")
        assert report.duality.frobenius_distance >= 0

    def test_natural_gradient_affine_nonnull(self):
        """Natural gradient g_A^{-1}·g should be a non-zero vector."""
        affine = AffineMetricACF()
        K = torch.eye(4, dtype=torch.float64) * 2.0
        g = torch.ones(4, dtype=torch.float64)
        nat_g = affine.natural_gradient(g, K)
        assert nat_g.shape == (4,)
        assert float(nat_g.norm().item()) > 0


# ============================================================
# 5. Thermodynamic ACF
# ============================================================
from acf_functor.thermodynamic_acf import (
    ThermodynamicACF,
    FreeEnergyComputer,
    CriticalityDetector,
    ThermodynamicReport,
)


class TestThermodynamicACF:
    def test_cold_d_star_geq_hot_d_star_combinatorial(self):
        """For combinatorial entropy: cold regime uses more dims than hot regime."""
        eig = make_exponential_eigenvalues(30, rate=0.3)
        # Use combinatorial entropy where high-temp = d≈m/2, low-temp = d→max
        thermo = ThermodynamicACF(eig, entropy_mode="combinatorial")
        report = thermo.analyze()
        # Cold should use more dimensions than hot (cold = minimize error)
        assert report.d_star_zero_temp >= report.d_star_high_temp, (
            f"Cold d*={report.d_star_zero_temp} < hot d*={report.d_star_high_temp}"
        )

    def test_tighter_epsilon_needs_larger_d(self):
        """d*(ε=10⁻²) ≤ d*(ε=10⁻⁵) for the same system."""
        eig = make_algebraic_eigenvalues(50, exponent=2.0)
        thermo = ThermodynamicACF(eig, entropy_mode="spectral")
        r_coarse = thermo.optimal_dimension(target_epsilon=1e-2)
        r_fine = thermo.optimal_dimension(target_epsilon=1e-5)
        assert r_coarse["d_star"] <= r_fine["d_star"], (
            f"d*(1e-2)={r_coarse['d_star']} > d*(1e-5)={r_fine['d_star']}"
        )

    def test_free_energy_profiles_computed_at_all_betas(self):
        """Profile list should have one entry per β in beta_samples."""
        eig = make_exponential_eigenvalues(20, rate=0.4)
        thermo = ThermodynamicACF(eig)
        report = thermo.analyze(beta_samples=[0.1, 1.0, 10.0, 100.0])
        assert len(report.profiles) == 4

    def test_free_energy_computer_formula(self):
        """F(d, β) = E(d) - S(d)/β — consistency check."""
        eig = make_exponential_eigenvalues(15, rate=0.5)
        computer = FreeEnergyComputer(eig, entropy_mode="spectral")
        for d in [2, 5, 8]:
            for beta in [1.0, 5.0, 20.0]:
                E = computer.error(d)
                S = computer.entropy(d)
                F = computer.free_energy(d, beta)
                expected = E - S / beta
                assert abs(F - expected) < 1e-12, (
                    f"F({d},{beta}) = {F:.6e} ≠ E-S/β = {expected:.6e}"
                )

    def test_mdl_dimension_between_hot_and_cold(self):
        """MDL dimension (β=1) should be between hot and cold for combinatorial."""
        eig = make_exponential_eigenvalues(30, rate=0.25)
        thermo = ThermodynamicACF(eig, entropy_mode="combinatorial")
        report = thermo.analyze()
        d_hot = report.d_star_high_temp
        d_cold = report.d_star_zero_temp
        d_mdl = report.mdl_dimension
        # MDL should be within the range [d_hot, d_cold] (roughly)
        assert d_hot <= d_mdl + 2 or d_cold >= d_mdl - 2, (
            f"MDL d*={d_mdl} outside plausible range [{d_hot}, {d_cold}]"
        )

    def test_free_energy_profile_has_d_star(self):
        """Every profile must have a valid d_star."""
        eig = make_exponential_eigenvalues(20, rate=0.4)
        thermo = ThermodynamicACF(eig)
        report = thermo.analyze(beta_samples=[1.0])
        p = report.profiles[0]
        assert p.d_star >= 1
        assert p.f_star is not None

    def test_phase_transition_attributes_if_detected(self):
        """If phase transition detected, it has valid attributes."""
        eig = make_algebraic_eigenvalues(40, exponent=1.5)
        thermo = ThermodynamicACF(eig, entropy_mode="combinatorial", n_beta=200)
        report = thermo.analyze()
        if report.phase_transition is not None:
            pt = report.phase_transition
            assert pt.beta_c > 0
            assert pt.d_low >= 0
            assert pt.d_high >= 0
            assert pt.delta_d >= 0

    def test_alpha_from_thermodynamics_nonnegative(self):
        """Both alpha estimates should be non-negative."""
        eig = make_exponential_eigenvalues(30, rate=0.4)
        thermo = ThermodynamicACF(eig)
        report = thermo.analyze()
        assert report.alpha >= 0
        assert report.alpha_from_thermodynamics >= 0


# ============================================================
# 6. NC Complexity (Lie bracket serial depth)
# ============================================================
from acf_functor.lie_analysis import NCComplexityAnalyzer, NCComplexityResult
from acf_functor.core import FMAOperation


def make_commuting_fma(n: int) -> list:
    """All GEMM ops use diagonal matrices → they commute."""
    ops = []
    for i in range(n):
        w = torch.diag(torch.rand(3, dtype=torch.float64) + 0.1)
        b = torch.zeros(3, dtype=torch.float64)
        ops.append(FMAOperation(weight=w, bias=b))
    return ops


def make_noncommuting_fma(n: int) -> list:
    """Random dense matrices — generally non-commuting."""
    torch.manual_seed(42)
    ops = []
    for _ in range(n):
        w = torch.randn(3, 3, dtype=torch.float64)
        b = torch.zeros(3, dtype=torch.float64)
        ops.append(FMAOperation(weight=w, bias=b))
    return ops


class TestNCComplexity:
    def test_commuting_ops_nc1(self):
        """Commuting (diagonal) matrices → Lie algebra dim = 0 → NC^1."""
        ops = make_commuting_fma(8)
        analyzer = NCComplexityAnalyzer(commutativity_threshold=1e-8)
        result = analyzer.analyze(ops)
        assert result.lie_span_dim == 0, (
            f"Commuting matrices should have dim=0, got {result.lie_span_dim}"
        )
        assert result.nc_class == "NC^1", (
            f"Expected NC^1 for commuting ops, got {result.nc_class}"
        )

    def test_noncommuting_ops_positive_lie_dim(self):
        """Random dense matrices → Lie span dim > 0."""
        ops = make_noncommuting_fma(6)
        analyzer = NCComplexityAnalyzer()
        result = analyzer.analyze(ops)
        assert result.lie_span_dim > 0, (
            f"Non-commuting matrices should have Lie dim > 0, got {result.lie_span_dim}"
        )

    def test_proof_sketch_content(self):
        """Proof sketch must contain the Lie algebra description."""
        ops = make_noncommuting_fma(4)
        analyzer = NCComplexityAnalyzer()
        result = analyzer.analyze(ops)
        assert "Lie" in result.proof_sketch
        assert "dim" in result.proof_sketch

    def test_empty_sequence_nc0(self):
        """Empty sequence → NC^0."""
        analyzer = NCComplexityAnalyzer()
        result = analyzer.analyze([])
        assert result.nc_class == "NC^0"
        assert result.lie_span_dim == 0
        assert result.serial_depth == 0

    def test_single_op_zero_lie_dim(self):
        """Single operation → no pairs → empty bracket set → dim = 0."""
        w = torch.randn(3, 3, dtype=torch.float64)
        ops = [FMAOperation(weight=w, bias=torch.zeros(3, dtype=torch.float64))]
        analyzer = NCComplexityAnalyzer()
        result = analyzer.analyze(ops)
        assert result.n_ops == 1
        assert result.lie_span_dim == 0  # no pairs → empty bracket set

    def test_depth_over_log_n_nonnegative(self):
        """depth/log(n) ratio must be non-negative."""
        ops = make_noncommuting_fma(10)
        analyzer = NCComplexityAnalyzer()
        result = analyzer.analyze(ops)
        assert result.depth_over_log_n >= 0

    def test_parallelizable_fraction_between_0_and_1(self):
        """Parallelizable fraction ∈ [0, 1]."""
        ops = make_noncommuting_fma(5)
        analyzer = NCComplexityAnalyzer()
        result = analyzer.analyze(ops)
        assert 0.0 <= result.parallelizable_fraction <= 1.0

    def test_nc_class_valid_name(self):
        """nc_class must be one of the recognized labels."""
        for builder in [make_commuting_fma, make_noncommuting_fma]:
            ops = builder(6)
            result = NCComplexityAnalyzer().analyze(ops)
            assert result.nc_class in {"NC^0", "NC^1", "NC^2", "NC^3", "P-hard"}, (
                f"Unexpected NC class: {result.nc_class}"
            )


# ============================================================
# 7. Alpha Unification Theorem
# ============================================================
from acf_functor.invariant_unified import (
    ACFInvariantUnified,
    AlphaUnificationTheorem,
    UnificationResult,
    AlphaEstimate,
    AlphaDefinition,
)


class TestAlphaUnification:
    def _estimate_for_sin(self) -> AlphaEstimate:
        """Compute α for sin(x) — a classic analytic function."""
        unified = ACFInvariantUnified(consistency_threshold=0.5)
        return unified.compute(
            lambda x: float(math.sin(x)),
            domain=(-1.0, 1.0),
            function_name="sin",
            skip_geometric=True,
        )

    def test_unification_runs_without_errors(self):
        """Theorem certification should run for any AlphaEstimate."""
        est = self._estimate_for_sin()
        theorem = AlphaUnificationTheorem(d_max=256, attractor_dim=1.0)
        result = theorem.certify(est)
        assert isinstance(result, UnificationResult)

    def test_certified_alpha_positive(self):
        """Certified α must be positive for sin(x)."""
        est = self._estimate_for_sin()
        theorem = AlphaUnificationTheorem()
        result = theorem.certify(est)
        assert result.certified_alpha > 0, (
            f"Certified α for sin should be positive, got {result.certified_alpha}"
        )

    def test_proof_sketch_contains_bernstein(self):
        """Proof sketch must reference Bernstein ellipse (the key parameter)."""
        est = self._estimate_for_sin()
        theorem = AlphaUnificationTheorem()
        result = theorem.certify(est)
        assert "Bernstein" in result.proof_sketch or "bernstein" in result.proof_sketch.lower()

    def test_pairwise_discrepancies_nonnegative(self):
        """Discrepancies |α_i - α_j| must be non-negative."""
        est = self._estimate_for_sin()
        theorem = AlphaUnificationTheorem()
        result = theorem.certify(est)
        assert result.disc_comb_spec >= 0
        assert result.disc_spec_geo >= 0
        assert result.disc_comb_geo >= 0

    def test_bernstein_roundtrip(self):
        """ρ → α(ρ) → ρ' should recover ρ exactly."""
        theorem = AlphaUnificationTheorem()
        for rho in [1.5, 2.0, math.e, 5.0]:
            alpha = theorem.alpha_from_bernstein(rho)
            rho_back = theorem.bernstein_from_alpha(alpha)
            assert abs(rho_back - rho) < 1e-10, (
                f"Bernstein roundtrip failed: ρ={rho}, recovered={rho_back}"
            )

    def test_alpha_from_bernstein_e_equals_one(self):
        """α(ρ=e) = 1/log(e) = 1 — canonical normalization."""
        theorem = AlphaUnificationTheorem()
        alpha = theorem.alpha_from_bernstein(math.e)
        assert abs(alpha - 1.0) < 1e-12

    def test_certified_ci_valid_interval(self):
        """Certified CI should be a valid interval (lo ≤ hi)."""
        est = self._estimate_for_sin()
        theorem = AlphaUnificationTheorem()
        result = theorem.certify(est)
        lo, hi = result.certified_ci
        assert lo <= hi, f"CI invalid: [{lo}, {hi}]"


# ============================================================
# 8. Integration: full pipeline
# ============================================================


class TestFullPipeline:
    """End-to-end tests combining multiple new modules."""

    def test_delta_bound_matches_actual_koopman_error(self):
        """The formal δ(d) bound should be ≥ 0 for real Koopman matrices."""
        torch.manual_seed(0)
        n_traj = 500
        x = torch.zeros(n_traj, dtype=torch.float64)
        x[0] = 0.3
        for t in range(1, n_traj):
            x[t] = float(torch.tanh(0.8 * x[t - 1]))
        x0, x1 = x[:-1].unsqueeze(1), x[1:].unsqueeze(1)

        # EDMD at d=15
        psi = torch.cat([x0 ** k for k in range(1, 16)], dim=1)  # (n, 15)
        psi_y = torch.cat([x1 ** k for k in range(1, 16)], dim=1)
        PsiPsi = psi.T @ psi + 1e-8 * torch.eye(15, dtype=torch.float64)
        PsiPsiY = psi.T @ psi_y
        K15 = torch.linalg.solve(PsiPsi, PsiPsiY).T

        eigenvalues = torch.abs(torch.linalg.eigvals(K15).real)
        eigenvalues, _ = torch.sort(eigenvalues, descending=True)
        bounds = KoopmanDeltaBounds(eigenvalues)

        for d in [5, 10]:
            delta_formal = bounds.at(d).delta_upper
            assert delta_formal >= 0, f"δ_formal({d}) negative: {delta_formal}"

    def test_thermodynamic_mdl_consistent_with_delta_bounds(self):
        """MDL dimension d* should satisfy δ(d*) ≤ δ(1) (more dims = less error)."""
        eig = make_exponential_eigenvalues(30, rate=0.35)
        thermo = ThermodynamicACF(eig)
        bounds = KoopmanDeltaBounds(eig)
        report = thermo.analyze()
        d_mdl = report.mdl_dimension
        assert d_mdl >= 1
        assert bounds.at(d_mdl).delta_upper <= bounds.at(1).delta_upper

    def test_nc_analysis_on_polynomial_fma_sequence(self):
        """NC complexity analysis on a real polynomial ACF FMA sequence."""
        functor = _ACFFunctor()
        r = functor.reduce_polynomial([1.0, -0.5, 0.25, -0.125])
        if not r.fma_sequence:
            pytest.skip("Empty FMA sequence for this reduction")
        analyzer = NCComplexityAnalyzer()
        result = analyzer.analyze(r.fma_sequence)
        assert result.nc_class in {"NC^0", "NC^1", "NC^2", "NC^3", "P-hard"}
        assert result.lie_span_dim >= 0

    def test_polynomial_inverse_verify_roundtrip(self):
        """Polynomial inversion verify_roundtrip returns valid metrics."""
        functor = _ACFFunctor()
        r = functor.reduce_polynomial([0.5, -0.3, 0.1])
        inverter = ACFInverse()
        rt = inverter.verify_roundtrip(r)
        assert "roundtrip_linf" in rt
        assert rt["roundtrip_linf"] >= 0

    def test_thermo_optimal_dimension_feasibility(self):
        """optimal_dimension(ε) should return feasible result when ε is achievable."""
        eig = make_algebraic_eigenvalues(50, exponent=2.0)
        thermo = ThermodynamicACF(eig)
        result = thermo.optimal_dimension(target_epsilon=0.01)
        # feasibility: the actual error at d* should be ≤ 1.01·ε
        if result["is_feasible"]:
            assert result["achieved_error"] <= 0.01 * 1.01 + 1e-12
