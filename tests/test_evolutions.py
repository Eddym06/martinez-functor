"""Tests for evolutions 3-7 and Lie analysis."""

import math

import torch

from acf_functor.koopman_adaptive import AdaptiveKoopman
from acf_functor.monad import FunctorMonad
from acf_functor.adjunction import CoFunctor, AdjunctionVerifier
from acf_functor.lie_analysis import LieBracketAnalyzer
from acf_functor.cohomology import CohomologyAnalyzer
from acf_functor.core import ChebyshevReducer, FMAOperation


class TestEvolution3:
    def test_auto_rank_selection(self):
        torch.manual_seed(42)
        A = torch.zeros(3, 3, dtype=torch.float64)
        A[0, 0] = 0.9
        A[0, 1] = 0.1
        A[1, 0] = -0.1
        A[1, 1] = 0.8
        A[2, 2] = 0.01

        x = torch.zeros(3, 300, dtype=torch.float64)
        x[:, 0] = torch.randn(3, dtype=torch.float64)
        for t in range(299):
            x[:, t + 1] = A @ x[:, t]

        engine = AdaptiveKoopman(observable_families=["polynomial"], max_poly_degree=2)
        result, diag = engine.reduce(x)

        assert diag.effective_dimension <= 12
        assert diag.reconstruction_error < 0.15
        assert "acf_alpha" in result.metadata


class TestEvolution4:
    def test_nonlinear_auto_dimension(self):
        x_lin = torch.zeros(1, 300, dtype=torch.float64)
        x_lin[0, 0] = 1.0
        for t in range(299):
            x_lin[0, t + 1] = 0.9 * x_lin[0, t]

        x_nl = torch.zeros(1, 300, dtype=torch.float64)
        x_nl[0, 0] = 0.5
        for t in range(299):
            x_nl[0, t + 1] = 0.9 * x_nl[0, t] + 0.1 * x_nl[0, t] ** 2

        engine = AdaptiveKoopman()
        _, d_lin = engine.reduce(x_lin)
        _, d_nl = engine.reduce(x_nl)

        assert d_nl.optimal_rank >= d_lin.optimal_rank


class TestEvolution5:
    def test_idempotence_sin(self):
        monad = FunctorMonad(default_degree=20)
        result = monad.verify_laws(torch.sin, domain=(-math.pi, math.pi))
        assert result.idempotence_holds
        assert result.max_idempotence_error < 1e-5

    def test_unit_laws(self):
        monad = FunctorMonad(default_degree=20)
        result = monad.verify_laws(torch.exp, domain=(-2.0, 2.0))
        assert result.left_unit_holds
        assert result.right_unit_holds


class TestEvolution6:
    def test_cofunctor_synthesis(self):
        co = CoFunctor()
        x = torch.linspace(-3.0, 3.0, 120, dtype=torch.float64)
        y = torch.sin(x)

        result = co.synthesize_from_evaluations(x, y, degree=20)
        y_fit = co.evaluate(result, x)
        err = torch.max(torch.abs(y - y_fit)).item()
        assert err < 1e-3

    def test_adjunction_verification(self):
        verifier = AdjunctionVerifier()
        phi_sin = ChebyshevReducer.reduce("sin", degree=20, domain=(-math.pi, math.pi))
        result = verifier.verify(torch.sin, phi_sin, domain=(-math.pi, math.pi))
        assert result.adjunction_holds or result.adjunction_gap < 1e-5


class TestEvolution7:
    def test_smooth_no_obstructions(self):
        analyzer = CohomologyAnalyzer(epsilon_threshold=1e-4)
        result = analyzer.analyze(torch.sin, domain=(-math.pi, math.pi), degree=20)
        assert result.is_fully_reducible
        assert result.h1.is_trivial

    def test_discontinuous_has_obstruction(self):
        analyzer = CohomologyAnalyzer(epsilon_threshold=1e-2)

        def step(x):
            return (x >= 0).to(x.dtype)

        result = analyzer.analyze(step, domain=(-2.0, 2.0), degree=20)
        assert not result.is_fully_reducible
        assert result.h1.rank > 0


class TestLieBrackets:
    def test_commuting_scales(self):
        analyzer = LieBracketAnalyzer()
        out = analyzer.compute_bracket(torch.tensor(2.0), torch.tensor(3.0))
        assert out.commutes

    def test_noncommuting_matrices(self):
        analyzer = LieBracketAnalyzer()
        A = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
        B = torch.tensor([[1.0, 0.0], [3.0, 1.0]])
        out = analyzer.compute_bracket(A, B)
        assert not out.commutes

    def test_horner_serial_depth(self):
        analyzer = LieBracketAnalyzer()
        fma_seq = [FMAOperation(weight=torch.tensor(1.0), bias=torch.tensor(float(i))) for i in range(5)]
        schedule = analyzer.analyze_sequence(fma_seq)
        assert schedule.serial_depth >= 2
