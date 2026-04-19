"""Tests for evolutions 8-11: geometric deepening."""

import math

import torch

from acf_functor.constructible_sheaves import ConstructibleSheaf
from acf_functor.core import ChebyshevReducer, HornerReducer
from acf_functor.galois_symmetry import GaloisAnalyzer, SymmetryType
from acf_functor.moduli_spaces import ModuliSpace
from acf_functor.persistent_homology import PersistentHomologyEngine


class TestEvolution8:
    def test_smooth_function_trivial_sheaf(self):
        sheaf = ConstructibleSheaf()
        sections, cohomology = sheaf.construct(
            torch.sin,
            domain=(-math.pi, math.pi),
            split_points=[],
            degree=20,
        )
        assert len(sections) == 1
        assert cohomology.h1_rank == 0
        assert cohomology.global_section_exists

    def test_piecewise_needs_splitting(self):
        def piecewise(x: torch.Tensor) -> torch.Tensor:
            return torch.where(x < 0, torch.sin(x), torch.exp(x) - 1.0)

        sheaf = ConstructibleSheaf(gluing_threshold=1e-3)
        sections, _ = sheaf.construct(
            piecewise,
            domain=(-3.0, 3.0),
            split_points=[0.0],
            degree=20,
        )
        assert len(sections) >= 2

    def test_global_section_evaluation(self):
        sheaf = ConstructibleSheaf()
        sections, _ = sheaf.construct(
            torch.sin,
            domain=(-math.pi, math.pi),
            split_points=[0.0],
            degree=20,
        )
        x = torch.linspace(-math.pi, math.pi, 5000, dtype=torch.float64)
        y = sheaf.evaluate_global_section(sections, x)
        y_exact = torch.sin(x)
        err = float(torch.max(torch.abs(y - y_exact)).item())
        assert err < 5e-4

    def test_euler_characteristic(self):
        sheaf = ConstructibleSheaf(gluing_threshold=1e-3)

        def step(x: torch.Tensor) -> torch.Tensor:
            return (x >= 0).to(x.dtype)

        _, cohomology = sheaf.construct(step, domain=(-2.0, 2.0), split_points=[0.0], degree=16)
        assert cohomology.euler_characteristic == cohomology.h0_rank - cohomology.h1_rank

    def test_sheaf_to_reduction_result(self):
        sheaf = ConstructibleSheaf()
        sections, cohomology = sheaf.construct(
            torch.sin,
            domain=(-math.pi, math.pi),
            split_points=[0.0],
            degree=20,
        )
        result = sheaf.to_reduction_result(sections, cohomology, (-math.pi, math.pi))
        assert result.epsilon_bound < 1.0
        assert "n_sections" in result.metadata


class TestEvolution9:
    def test_moduli_exploration_sin(self):
        moduli = ModuliSpace(degree_range=(4, 40), degree_step=4)
        analysis = moduli.explore(torch.sin, domain=(-math.pi, math.pi))
        assert analysis.optimal_point.epsilon < 1e-3
        assert len(analysis.explored_points) > 3
        assert analysis.dimension_of_moduli > 0

    def test_geodesic_monotonicity(self):
        moduli = ModuliSpace(degree_range=(4, 60), degree_step=2)
        analysis = moduli.explore(torch.exp, domain=(-2.0, 2.0))
        if analysis.geodesic_to_optimal is not None:
            path = analysis.geodesic_to_optimal
            assert path.end.epsilon <= path.start.epsilon

    def test_optimal_vs_naive(self):
        moduli = ModuliSpace(degree_range=(4, 64), degree_step=2)
        optimal_reduction = moduli.find_optimal_reduction(
            torch.sin,
            domain=(-math.pi, math.pi),
            target_epsilon=1e-8,
        )
        naive_reduction = ChebyshevReducer.reduce(torch.sin, degree=6, domain=(-math.pi, math.pi))
        assert optimal_reduction.epsilon_bound <= naive_reduction.epsilon_bound

    def test_curvature_meaningful(self):
        moduli = ModuliSpace(degree_range=(4, 30), degree_step=2)
        analysis = moduli.explore(torch.tanh, domain=(-3.0, 3.0))
        assert analysis.curvature_at_optimal >= 0


class TestEvolution10:
    def test_polynomial_converges_perfectly(self):
        engine = PersistentHomologyEngine(min_scale=2, max_scale=20, scale_step=1)

        def cubic(x: torch.Tensor) -> torch.Tensor:
            return 1.0 + 2.0 * x - 0.5 * x**2 + 0.1 * x**3

        spm = engine.compute_spm(cubic, domain=(-5.0, 5.0))
        assert spm.complexity_class in ("polynomial", "analytic")
        assert spm.optimal_epsilon < 1e-8

    def test_transcendental_classified_analytic(self):
        engine = PersistentHomologyEngine(min_scale=4, max_scale=40, scale_step=2)
        spm = engine.compute_spm(torch.sin, domain=(-math.pi, math.pi))
        assert spm.complexity_class in ("polynomial", "analytic")
        assert spm.acf_alpha >= 0.5

    def test_optimal_degree_reasonable(self):
        engine = PersistentHomologyEngine(min_scale=4, max_scale=50, scale_step=2)
        optimal_deg, eps = engine.find_optimal_degree(torch.exp, domain=(-2.0, 2.0))
        assert 4 <= optimal_deg <= 50
        assert eps < 5e-4

    def test_persistence_diagram_structure(self):
        engine = PersistentHomologyEngine(min_scale=4, max_scale=40, scale_step=2)
        spm = engine.compute_spm(torch.tanh, domain=(-3.0, 3.0))
        assert spm.diagram.total_persistence >= 0
        assert spm.diagram.optimal_scale > 0

    def test_spm_summary(self):
        engine = PersistentHomologyEngine(min_scale=4, max_scale=30, scale_step=2)
        spm = engine.compute_spm(torch.sin, domain=(-math.pi, math.pi))
        summary = spm.summary()
        assert "optimal_degree" in summary
        assert "class" in summary


class TestEvolution11:
    def test_detect_even_symmetry(self):
        analyzer = GaloisAnalyzer()
        galois = analyzer.analyze(torch.cos, domain=(-math.pi, math.pi))
        even_syms = [s for s in galois.symmetries if s.symmetry_type == SymmetryType.EVEN]
        assert len(even_syms) > 0
        assert even_syms[0].confidence > 0.9

    def test_detect_odd_symmetry(self):
        analyzer = GaloisAnalyzer()
        galois = analyzer.analyze(torch.sin, domain=(-math.pi, math.pi))
        odd_syms = [s for s in galois.symmetries if s.symmetry_type == SymmetryType.ODD]
        assert len(odd_syms) > 0

    def test_compression_even_function(self):
        analyzer = GaloisAnalyzer()
        reduction = ChebyshevReducer.reduce(torch.cos, degree=20, domain=(-math.pi, math.pi))
        galois = analyzer.analyze(torch.cos, domain=(-math.pi, math.pi), reduction=reduction)
        if galois.compressed_coefficients is not None:
            assert galois.effective_degree <= galois.original_degree

    def test_galois_compressed_reduction(self):
        analyzer = GaloisAnalyzer()
        compressed_result, _ = analyzer.compress_reduction(torch.cos, domain=(-math.pi, math.pi), degree=20)

        x = torch.linspace(-math.pi, math.pi, 5000, dtype=torch.float64)
        coeffs = compressed_result.metadata.get(
            "monomial_coefficients",
            compressed_result.metadata.get("coefficients", []),
        )
        y_exact = torch.cos(x)
        y_approx = HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=x.dtype), x)
        err = float(torch.max(torch.abs(y_exact - y_approx)).item())
        assert err < 1.0

    def test_no_symmetry_for_general(self):
        analyzer = GaloisAnalyzer()

        def asymmetric(x: torch.Tensor) -> torch.Tensor:
            return torch.sin(x) + 0.3 * torch.cos(2.0 * x) + 0.1 * x

        galois = analyzer.analyze(asymmetric, domain=(-3.0, 3.0))
        even_odd = [
            s
            for s in galois.symmetries
            if s.symmetry_type in (SymmetryType.EVEN, SymmetryType.ODD)
        ]
        assert len(even_odd) == 0

    def test_periodic_detection(self):
        analyzer = GaloisAnalyzer()
        galois = analyzer.analyze(torch.sin, domain=(-3.0 * math.pi, 3.0 * math.pi))
        periodic = [s for s in galois.symmetries if s.symmetry_type == SymmetryType.PERIODIC]
        if periodic:
            period = periodic[0].parameters.get("period", 0.0)
            assert abs(period - 2.0 * math.pi) < 0.5 or abs(period - math.pi) < 0.5


class TestGeometricIntegration:
    def test_sheaf_uses_persistence_for_degree(self):
        engine = PersistentHomologyEngine(min_scale=4, max_scale=40, scale_step=2)
        optimal_degree, _ = engine.find_optimal_degree(torch.sin, domain=(-math.pi, math.pi))

        sheaf = ConstructibleSheaf()
        _, cohomology = sheaf.construct(
            torch.sin,
            domain=(-math.pi, math.pi),
            split_points=[],
            degree=optimal_degree,
        )
        assert cohomology.global_section_exists

    def test_galois_reduces_moduli_search(self):
        analyzer = GaloisAnalyzer()
        moduli = ModuliSpace(degree_range=(4, 40), degree_step=4)
        _ = analyzer.analyze(torch.cos, domain=(-math.pi, math.pi))
        analysis = moduli.explore(torch.cos, domain=(-math.pi, math.pi))
        assert analysis.optimal_point.epsilon < 1e-3

    def test_cohomology_to_sheaf_pipeline(self):
        from acf_functor.cohomology import CohomologyAnalyzer

        def difficult(x: torch.Tensor) -> torch.Tensor:
            return torch.where(x < 0, torch.sin(3.0 * x), torch.exp(-x))

        coh_analyzer = CohomologyAnalyzer(epsilon_threshold=1e-2)
        coh_result = coh_analyzer.analyze(difficult, domain=(-3.0, 3.0), degree=20)

        split_points = [0.0]
        for interval in coh_result.h1.generators:
            mid = 0.5 * (interval[0] + interval[1])
            if -2.9 < mid < 2.9:
                split_points.append(mid)
        split_points = sorted(set(split_points))

        sheaf = ConstructibleSheaf(gluing_threshold=1e-2)
        sections, _ = sheaf.construct(
            difficult,
            domain=(-3.0, 3.0),
            split_points=split_points,
            degree=24,
        )

        x = torch.linspace(-3.0, 3.0, 10000, dtype=torch.float64)
        y = sheaf.evaluate_global_section(sections, x)
        y_exact = difficult(x)

        mask = torch.abs(x) > 0.2
        err = float(torch.max(torch.abs(y[mask] - y_exact[mask])).item())
        assert err < 1.0
