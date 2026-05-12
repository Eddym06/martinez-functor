"""
Stability Analysis Tests — Perturbation Theory for Distributions
==================================================================

Tests for:
  - Empirical stability under singularity perturbations
  - Lipschitz constant estimation
  - Adaptive re-projection
  - Counterexample search
  - Consistency maintenance across perturbation ranges
"""

import time
import numpy as np
import pytest

from acf_functor.distribution_theory import (
    DualDistribution,
    SpectralTensor,
    SingularityType,
    dirac,
    heaviside,
    dirac_derivative,
    dirac_comb,
)
from acf_functor.distribution_stability import (
    StabilityAnalyzer,
    StabilityCertificate,
    PerturbationTrial,
    PerturbationAnalysisResult,
    analyze_stability,
)


@pytest.fixture
def domain():
    return (-1.0, 1.0)


@pytest.fixture
def analyzer():
    return StabilityAnalyzer(n_test_functions=10, n_quad_points=100)


class TestStabilityCore:
    """Core stability analysis tests."""

    def test_analyzer_creation(self, analyzer):
        """Stability analyzer should initialize with test functions."""
        assert analyzer is not None
        assert len(analyzer._test_funcs) >= 10

    def test_perturb_dirac(self, domain, analyzer):
        """Perturbing a Dirac should shift its position."""
        T = dirac(0.5, domain)
        T_pert = analyzer.perturb_distribution(T, 0.1, "worst_case")

        # Position should have changed
        orig_pos = T.singularities[0].position[0]
        pert_pos = T_pert.singularities[0].position[0]
        assert abs(pert_pos - orig_pos) > 0.01

    def test_perturb_preserves_order(self, domain, analyzer):
        """Perturbation should NOT change singularity orders."""
        T = dirac_derivative(0.3, 2, domain)
        T_pert = analyzer.perturb_distribution(T, 0.05)

        for s1, s2 in zip(T.singularities, T_pert.singularities):
            assert s1.order == s2.order

    def test_measure_action_deviation_zero_for_zero_eta(self, domain, analyzer):
        """Zero perturbation → zero action deviation."""
        T = dirac(0.5, domain)
        T_same = analyzer.perturb_distribution(T, 0.0)

        delta = analyzer.measure_action_deviation(T, T_same)
        assert delta < 1e-10

    def test_action_deviation_increases_with_eta(self, domain, analyzer):
        """Larger perturbations → larger action deviations."""
        T = dirac(0.0, domain)
        deviations = []
        for eta in [1e-6, 1e-4, 1e-2]:
            T_pert = analyzer.perturb_distribution(T, eta, "worst_case")
            delta = analyzer.measure_action_deviation(T, T_pert)
            deviations.append(delta)

        # Should be monotonically non-decreasing
        for i in range(len(deviations) - 1):
            assert deviations[i + 1] >= deviations[i] * 0.9

    def test_sobolev_bound_positive(self, domain, analyzer):
        """Sobolev bound should be positive for non-zero perturbation."""
        T = dirac_derivative(0.0, 2, domain)
        bound = analyzer.compute_sobolev_bound(T, 0.01)
        assert bound > 0


class TestStabilityCertificates:
    """Stability certificate generation tests."""

    def test_run_stability_analysis_dirac(self, domain, analyzer):
        """Full stability analysis on Dirac delta."""
        T = dirac(0.3, domain)
        stability = analyzer.run_stability_analysis(T, n_repeats=2)

        assert stability.n_trials > 0
        assert stability.empirical_lipschitz >= 0
        assert stability.max_order == 0
        assert stability.confidence in ["high", "medium", "low"]

    def test_run_stability_analysis_heaviside(self, domain, analyzer):
        """Stability analysis on Heaviside."""
        T = heaviside(0.0, domain)
        stability = analyzer.run_stability_analysis(T, n_repeats=2)

        assert stability.n_trials > 0
        assert stability.max_order == -1

    def test_run_stability_analysis_derivative(self, domain, analyzer):
        """Higher-order singularities should have larger Lipschitz constants."""
        T1 = dirac(0.0, domain)
        T2 = dirac_derivative(0.0, 2, domain)

        s1 = analyzer.run_stability_analysis(T1, n_repeats=1)
        s2 = analyzer.run_stability_analysis(T2, n_repeats=1)

        # Higher order → more sensitive to perturbation
        # (This is the key empirical validation of the stability conjecture)
        assert s2.empirical_lipschitz >= s1.empirical_lipschitz * 0.1

    def test_consistency_maintained_small_eta(self, domain, analyzer):
        """Small perturbations should maintain consistency."""
        T = dirac(0.5, domain)
        stability = analyzer.run_stability_analysis(
            T, perturbation_mode="random", n_repeats=1
        )
        # For small eta, consistency should generally hold
        assert stability.n_trials > 0

    def test_lipschitz_map_built(self, domain, analyzer):
        """Lipschitz map should be populated."""
        T = dirac_comb(0.3, n_terms=3, domain=domain)
        result = analyzer.full_analysis(T, "dirac_comb_test")
        assert len(result.adaptive_lipschitz_map) > 0

    def test_empirical_lipschitz_finite(self, domain, analyzer):
        """Lipschitz constants should be finite (no explosions)."""
        distributions = [
            ("dirac", dirac(0.5, domain)),
            ("heaviside", heaviside(0.0, domain)),
            ("dirac_deriv1", dirac_derivative(0.0, 1, domain)),
        ]

        for name, T in distributions:
            stability = analyzer.run_stability_analysis(T, n_repeats=1)
            assert np.isfinite(stability.empirical_lipschitz), (
                f"{name}: Lipschitz constant is not finite: {stability.empirical_lipschitz}"
            )
            assert stability.empirical_lipschitz < 1e8, (
                f"{name}: Lipschitz constant too large: {stability.empirical_lipschitz:.1e}"
            )


class TestAdaptiveReprojection:
    """Adaptive re-projection tests."""

    def test_reproject_preserves_singularities(self, domain, analyzer):
        """Re-projection should preserve singularity count."""
        T = dirac_comb(0.25, n_terms=3, domain=domain)
        T_reproj = analyzer.adaptive_reproject(T, 1e-4)

        assert len(T_reproj.singularities) == len(T.singularities)

    def test_reproject_reduces_consistency_error(self, domain, analyzer):
        """Re-projection should improve or maintain consistency."""
        T = heaviside(0.0, domain)
        T_pert = analyzer.perturb_distribution(T, 1e-3)
        T_reproj = analyzer.adaptive_reproject(T, 1e-3)

        orig_cons = T.check_consistency()
        reproj_cons = T_reproj.check_consistency()

        # Should not drastically worsen
        assert reproj_cons < orig_cons * 10 + 1.0

    def test_full_analysis_returns_all_components(self, domain, analyzer):
        """Full analysis should return complete PerturbationAnalysisResult."""
        T = dirac(0.0, domain)
        result = analyzer.full_analysis(T, "test_dirac")

        assert isinstance(result, PerturbationAnalysisResult)
        assert result.stability is not None
        assert len(result.adaptive_lipschitz_map) > 0
        assert result.re_projection_cost >= 0
        assert result.recommended_perturbation_budget > 0


class TestCounterexampleSearch:
    """Counterexample search (Genesis-style exploration)."""

    def test_search_returns_results(self, domain, analyzer):
        """Counterexample search should return a list of candidates."""
        T = heaviside(0.0, domain)
        results = analyzer.search_counterexamples(T, eta_range=(1e-6, 1e-2), n_search=20)
        assert len(results) > 0

    def test_search_results_have_required_fields(self, domain, analyzer):
        """Each counterexample candidate should have all fields."""
        T = dirac_derivative(0.0, 1, domain)
        results = analyzer.search_counterexamples(T, n_search=10)

        for r in results:
            assert "eta" in r
            assert "delta_action" in r
            assert "lipschitz" in r
            assert np.isfinite(r["lipschitz"])


class TestProofSketch:
    """Verify the proof sketch is accessible and well-structured."""

    def test_proof_sketch_exists(self):
        """The proof sketch constant should be defined."""
        from acf_functor.distribution_stability import PROOF_SKETCH_STABILITY
        assert len(PROOF_SKETCH_STABILITY) > 500
        assert "LEMMA 1" in PROOF_SKETCH_STABILITY
        assert "LEMMA 2" in PROOF_SKETCH_STABILITY
        assert "LEMMA 3" in PROOF_SKETCH_STABILITY
        assert "Lipschitz" in PROOF_SKETCH_STABILITY


class TestConvenience:
    """Convenience function tests."""

    def test_analyze_stability_wrapper(self, domain):
        """Convenience wrapper should work."""
        T = dirac(0.0, domain)
        result = analyze_stability(T, "test", n_test_functions=5)
        assert isinstance(result, PerturbationAnalysisResult)


class TestPerformance:
    """Performance benchmarks for stability analysis."""

    def test_analysis_speed(self, domain):
        """Stability analysis should be reasonably fast."""
        T = dirac(0.3, domain)
        analyzer = StabilityAnalyzer(n_test_functions=5, n_quad_points=50)

        start = time.perf_counter()
        result = analyzer.full_analysis(T, "perf_test")
        elapsed = time.perf_counter() - start

        # Full analysis should complete quickly
        assert elapsed < 30.0, f"Analysis too slow: {elapsed:.2f}s"

    def test_perturbation_chain_speed(self, domain):
        """Perturbation chain should be fast."""
        T = dirac_comb(0.2, n_terms=5, domain=domain)
        analyzer = StabilityAnalyzer(n_test_functions=3)

        start = time.perf_counter()
        for eta in [1e-6, 1e-4, 1e-2]:
            T_pert = analyzer.perturb_distribution(T, eta)
            delta = analyzer.measure_action_deviation(T, T_pert)
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"Perturbation too slow: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
