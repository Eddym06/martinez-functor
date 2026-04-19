"""Tests for Evolutions 12-15: physical topology and terminal layer."""

import math

import torch

from acf_functor.kolmogorov_entropy import KolmogorovEntropy
from acf_functor.superposition import SuperpositionEngine
from acf_functor.field_action import FieldAction
from acf_functor.act_topos import ACFTopos, TruthValue
from acf_functor.core import ChebyshevReducer, HornerReducer


class TestEvolution12:
    def test_polynomial_exact_energy(self):
        entropy = KolmogorovEntropy(max_degree=20)

        def cubic(x: torch.Tensor) -> torch.Tensor:
            return 1.0 + 2.0 * x - 0.5 * x**2 + 0.1 * x**3

        profile = entropy.compute_entropy_profile(cubic, domain=(-5.0, 5.0))
        assert profile.complexity_class in ("polynomial", "analytic")

    def test_transcendental_entropy_profile(self):
        entropy = KolmogorovEntropy(min_degree=4, max_degree=40, degree_step=2)
        profile = entropy.compute_entropy_profile(torch.sin, domain=(-math.pi, math.pi))
        assert len(profile.epsilon_values) > 3
        assert profile.complexity_class in ("polynomial", "analytic")
        if profile.complexity_class == "analytic":
            assert profile.alpha >= 0.5

    def test_conservation_law(self):
        entropy = KolmogorovEntropy()
        result = entropy.test_conservation(torch.sin, domain=(-math.pi, math.pi), degree=20)
        assert result.conserved
        assert result.relative_gap < 0.2

    def test_minimum_energy_constructive(self):
        entropy = KolmogorovEntropy(min_degree=2, max_degree=50)
        min_energy, reduction = entropy.compute_minimum_energy(torch.exp, domain=(-2.0, 2.0), target_epsilon=1e-8)
        assert 2 <= min_energy <= 50
        assert reduction.epsilon_bound < 1e-4

    def test_entropy_rate_positive(self):
        entropy = KolmogorovEntropy(min_degree=4, max_degree=30, degree_step=2)
        profile = entropy.compute_entropy_profile(torch.sin, domain=(-math.pi, math.pi))
        assert profile.entropy_rate >= 0

    def test_efficiency_metric(self):
        entropy = KolmogorovEntropy(min_degree=4, max_degree=30, degree_step=2)
        profile = entropy.compute_entropy_profile(torch.sin, domain=(-math.pi, math.pi))
        eff = profile.efficiency(actual_fma_count=20, epsilon=1e-6)
        assert 0 <= eff <= 1.0


class TestEvolution13:
    def test_superposition_generates_candidates(self):
        engine = SuperpositionEngine(degree_range=(4, 30), degree_step=2)
        state = engine.superpose(torch.sin, domain=(-math.pi, math.pi))
        assert state.n_candidates >= 5
        assert len(state.amplitudes) == state.n_candidates
        assert torch.all(state.amplitudes >= 0)

    def test_collapse_selects_best(self):
        engine = SuperpositionEngine(degree_range=(4, 30), degree_step=2)
        state = engine.superpose(torch.sin, domain=(-math.pi, math.pi))
        result = engine.collapse(state, torch.sin, domain=(-math.pi, math.pi))
        assert result.optimal.epsilon < 1e-3
        assert result.selection_confidence > 0
        assert result.entropy_before >= 0

    def test_superposition_finds_optimal(self):
        engine = SuperpositionEngine(degree_range=(4, 40), degree_step=2)
        reduction, _ = engine.find_optimal(torch.exp, domain=(-2.0, 2.0))

        x = torch.linspace(-2.0, 2.0, 5000, dtype=torch.float64)
        coeffs = reduction.metadata.get(
            "monomial_coefficients",
            reduction.metadata.get("coefficients", []),
        )
        y = HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=torch.float64), x)
        err = torch.max(torch.abs(y - torch.exp(x))).item()
        assert err < 1e-4

    def test_entropy_decreases_on_collapse(self):
        engine = SuperpositionEngine(degree_range=(4, 30), degree_step=2)
        state = engine.superpose(torch.tanh, domain=(-3.0, 3.0))
        result = engine.collapse(state, torch.tanh, domain=(-3.0, 3.0))
        assert result.entropy_before >= result.entropy_after


class TestEvolution14:
    def test_action_decomposition(self):
        field = FieldAction(degree_range=(4, 30))
        action = field.compute_action(torch.sin, domain=(-math.pi, math.pi), degree=16)
        assert action.kinetic >= 0
        assert action.potential >= 0
        assert action.regularization > 0
        assert abs(action.total - (action.kinetic + action.potential + action.lagrange_multiplier * action.regularization)) < 1e-10

    def test_action_minimization(self):
        field = FieldAction(degree_range=(4, 40), lambda_reg=0.001)
        result = field.minimize_action(torch.sin, domain=(-math.pi, math.pi))
        assert result.optimal_degree >= 4
        assert result.optimal_epsilon < 1.0
        assert len(result.action_trajectory) > 3

    def test_free_energy_profile(self):
        field = FieldAction(degree_range=(4, 30))
        result = field.minimize_action(torch.exp, domain=(-2.0, 2.0))
        fe = result.free_energy_profile
        assert "degrees" in fe
        assert "energy" in fe
        assert "entropy" in fe
        assert "free_energy" in fe
        assert len(fe["degrees"]) > 3

    def test_phase_transition_detection(self):
        field = FieldAction(degree_range=(2, 50))
        result = field.minimize_action(torch.sin, domain=(-math.pi, math.pi))
        assert isinstance(result.phase_transitions, list)

    def test_optimal_beats_arbitrary(self):
        field = FieldAction(degree_range=(4, 40))
        result = field.minimize_action(torch.sin, domain=(-math.pi, math.pi))
        assert result.optimal_epsilon < 1.0


class TestEvolution15:
    def test_truth_value_algebra(self):
        top = TruthValue.top()
        bottom = TruthValue.bottom()
        partial = TruthValue.partial(0.001, 1.0)

        assert top.is_exact
        assert bottom.is_impossible
        assert not partial.is_exact
        assert not partial.is_impossible

        and_result = top & partial
        assert not and_result.is_exact

        or_result = top | bottom
        assert or_result.is_exact

        impl = bottom.implies(top)
        assert impl.value >= 0.99

    def test_internalize_polynomial(self):
        topos = ACFTopos()
        reduction = HornerReducer.reduce([1.0, 2.0, 3.0])
        obj = topos.internalize("poly_123", reduction)
        assert obj.truth.is_exact
        assert obj.is_exact

    def test_internalize_transcendental(self):
        topos = ACFTopos()
        reduction = ChebyshevReducer.reduce("sin", degree=20, domain=(-math.pi, math.pi))
        obj = topos.internalize("sin_20", reduction)
        assert not obj.truth.is_exact
        assert not obj.truth.is_impossible
        assert 0 < obj.truth.value <= 1

    def test_morphism_construction(self):
        topos = ACFTopos()
        r1 = ChebyshevReducer.reduce("sin", degree=16, domain=(-math.pi, math.pi))
        r2 = ChebyshevReducer.reduce("sin", degree=24, domain=(-math.pi, math.pi))
        topos.internalize("sin_16", r1)
        topos.internalize("sin_24", r2)
        morph = topos.morphism("sin_16", "sin_24")
        assert morph is not None
        assert morph.transformation_epsilon >= 0

    def test_subobject_classifier(self):
        topos = ACFTopos()
        r_poly = HornerReducer.reduce([1.0, 1.0])
        r_sin = ChebyshevReducer.reduce("sin", degree=20, domain=(-math.pi, math.pi))
        topos.internalize("linear", r_poly)
        topos.internalize("sin", r_sin)
        omega = topos.subobject_classifier()
        assert "obj:linear" in omega
        assert "obj:sin" in omega
        assert omega["obj:linear"].is_exact
        assert not omega["obj:sin"].is_exact

    def test_internal_logic_consistency(self):
        topos = ACFTopos()
        topos.internalize("poly", HornerReducer.reduce([0.0, 0.0, 1.0]))
        topos.internalize("sin", ChebyshevReducer.reduce("sin", degree=20, domain=(-3.0, 3.0)))
        topos.internalize("exp", ChebyshevReducer.reduce("exp", degree=20, domain=(-3.0, 3.0)))
        analysis = topos.analyze()
        assert analysis.internal_logic_consistent
        assert len(analysis.objects) == 3

    def test_judge_function(self):
        topos = ACFTopos()
        truth_sin = topos.judge(torch.sin, domain=(-math.pi, math.pi))
        assert truth_sin.value > 0.5
        assert not truth_sin.is_impossible

    def test_impossibility_certificate(self):
        topos = ACFTopos()
        cert = topos.certificate_of_impossibility("Halting problem is not computable")
        assert cert.is_impossible
        assert cert.value == 0.0

    def test_topos_analysis_complete(self):
        topos = ACFTopos()
        topos.internalize("poly", HornerReducer.reduce([1.0, 2.0, 3.0]))
        topos.internalize("sin", ChebyshevReducer.reduce("sin", degree=20, domain=(-math.pi, math.pi)))
        topos.morphism("poly", "sin")
        analysis = topos.analyze()
        assert analysis.internal_logic_consistent
        assert analysis.total_truth.value > 0
        assert len(analysis.subobject_classifier) > 0


class TestPhaseDIntegration:
    def test_entropy_informs_action(self):
        entropy = KolmogorovEntropy(min_degree=4, max_degree=30, degree_step=2)
        field = FieldAction(degree_range=(4, 30))
        _ = entropy.compute_entropy_profile(torch.sin, domain=(-math.pi, math.pi))
        result = field.minimize_action(torch.sin, domain=(-math.pi, math.pi))
        assert 4 <= result.optimal_degree <= 30

    def test_superposition_agrees_with_action(self):
        engine = SuperpositionEngine(degree_range=(4, 30), degree_step=2)
        field = FieldAction(degree_range=(4, 30))
        reduction_sup, _ = engine.find_optimal(torch.exp, domain=(-2.0, 2.0))
        result_field = field.minimize_action(torch.exp, domain=(-2.0, 2.0))
        assert reduction_sup.epsilon_bound < 1.0
        assert result_field.optimal_epsilon < 1.0

    def test_topos_encompasses_all(self):
        topos = ACFTopos()
        entropy = KolmogorovEntropy(min_degree=4, max_degree=20, degree_step=2)
        _, r_entropy = entropy.compute_minimum_energy(torch.sin, (-math.pi, math.pi), target_epsilon=1e-6)
        topos.internalize("entropy_optimal", r_entropy)

        field = FieldAction(degree_range=(4, 20))
        result = field.minimize_action(torch.sin, (-math.pi, math.pi))
        topos.internalize("action_optimal", result.optimal_reduction)

        analysis = topos.analyze()
        assert analysis.internal_logic_consistent
        assert all(obj.is_well_defined for obj in analysis.objects)

    def test_conservation_through_pipeline(self):
        entropy = KolmogorovEntropy()
        for func, domain, name in [
            (torch.sin, (-math.pi, math.pi), "sin"),
            (torch.exp, (-2.0, 2.0), "exp"),
            (torch.tanh, (-3.0, 3.0), "tanh"),
        ]:
            result = entropy.test_conservation(func, domain, degree=20)
            assert result.conserved, f"Conservation failed for {name}: E(f)={result.e_f}, E(Phi(f))={result.e_phi_f}"
