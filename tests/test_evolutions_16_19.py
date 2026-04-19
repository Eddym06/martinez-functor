"""Tests for Evolutions 16-19: inward recursive spiral."""

import math

import pytest
import torch

from poema.affine_turing import AffineTuringMachine
from poema.ast_nodes import (
    AffineNode,
    ComposeNode,
    ConstantNode,
    IdentityNode,
    PolynomialNode,
    ScaleNode,
    ShiftNode,
)
from poema.free_algebra import FreeAlgebra
from poema.frontend import Poem
from poema.meta_compiler import MetaCompiler
from poema.sheaf_semantics import SheafSemantics


@pytest.fixture
def poem() -> Poem:
    return Poem()


class TestEvolution16:
    def test_rule_r2_scale_merge(self) -> None:
        algebra = FreeAlgebra()
        inner = ScaleNode(factor=torch.tensor(3.0))
        outer = ScaleNode(factor=torch.tensor(2.0), child=inner)
        normalized, trace = algebra.normalize(outer)
        assert isinstance(normalized, ScaleNode)
        assert abs(float(normalized.factor.item()) - 6.0) < 1e-10
        assert trace.total_rewrites > 0

    def test_rule_r3_shift_merge(self) -> None:
        algebra = FreeAlgebra()
        inner = ShiftNode(value=torch.tensor(3.0))
        outer = ShiftNode(value=torch.tensor(2.0), child=inner)
        normalized, _ = algebra.normalize(outer)
        assert isinstance(normalized, ShiftNode)
        assert abs(float(normalized.value.item()) - 5.0) < 1e-10

    def test_rule_r1_scale_shift_to_affine(self) -> None:
        algebra = FreeAlgebra()
        node = ComposeNode(
            outer=ScaleNode(factor=torch.tensor(2.0)),
            inner=ShiftNode(value=torch.tensor(3.0)),
        )
        normalized, _ = algebra.normalize(node)
        assert isinstance(normalized, AffineNode)
        assert abs(float(normalized.scale_factor.item()) - 2.0) < 1e-10
        assert abs(float(normalized.shift_value.item()) - 6.0) < 1e-10

    def test_word_representation(self) -> None:
        algebra = FreeAlgebra()
        node = AffineNode(scale_factor=torch.tensor(2.0), shift_value=torch.tensor(1.0))
        word = algebra.to_word(node)
        assert len(word) > 0


class TestEvolution17:
    def test_constant_is_exact(self) -> None:
        semantics = SheafSemantics()
        verdict = semantics.analyze(ConstantNode(value=torch.tensor(42.0)))
        assert verdict.is_correct
        assert verdict.truth_value == 1.0

    def test_affine_is_exact(self) -> None:
        semantics = SheafSemantics()
        node = AffineNode(scale_factor=torch.tensor(2.0), shift_value=torch.tensor(1.0))
        verdict = semantics.analyze(node, domain=(-5, 5))
        assert verdict.is_correct

    def test_polynomial_is_exact(self) -> None:
        semantics = SheafSemantics()
        node = PolynomialNode(coefficients=[1.0, 2.0, 3.0])
        verdict = semantics.analyze(node, domain=(-5, 5))
        assert verdict.is_correct
        assert verdict.truth_value >= 0.99

    def test_transcendental_has_bounded_truth(self, poem: Poem) -> None:
        semantics = SheafSemantics()
        node = poem.sin(domain=(-math.pi, math.pi))
        verdict = semantics.analyze(node, domain=(-math.pi, math.pi))
        assert verdict.is_correct
        assert 0.0 <= verdict.truth_value <= 1.0


class TestEvolution18:
    def test_counter(self) -> None:
        mta = AffineTuringMachine()
        program = AffineTuringMachine.build_counter(increment=1.0, limit=5.0)
        result = mta.execute(program, initial_tape=torch.tensor([0.0]))
        assert result.accepted
        assert float(result.final_tape.item()) >= 5.0

    def test_multiplier(self) -> None:
        mta = AffineTuringMachine()
        program = AffineTuringMachine.build_multiplier(factor=2.0, n_times=3)
        result = mta.execute(program, initial_tape=torch.tensor([1.0]))
        assert result.accepted
        assert abs(float(result.final_tape.item()) - 8.0) < 1e-10

    def test_conditional_branches(self) -> None:
        mta = AffineTuringMachine()
        program = AffineTuringMachine.build_conditional(
            threshold=0.0,
            action_above=ScaleNode(factor=torch.tensor(2.0)),
            action_below=ScaleNode(factor=torch.tensor(0.5)),
        )
        above = mta.execute(program, initial_tape=torch.tensor([5.0]))
        below = mta.execute(program, initial_tape=torch.tensor([-4.0]))
        assert above.accepted and below.accepted
        assert abs(float(above.final_tape.item()) - 10.0) < 1e-10
        assert abs(float(below.final_tape.item()) - (-2.0)) < 1e-10

    def test_horner_program_shape(self) -> None:
        coeffs = [1.0, 2.0, 3.0]
        program = AffineTuringMachine.build_horner_evaluator(coeffs)
        assert program.n_states == len(coeffs) + 1
        assert program.n_transitions == len(coeffs)


class TestEvolution19:
    def test_meta_compilation_polynomial(self, poem: Poem) -> None:
        mc = MetaCompiler()
        poly = poem.polynomial([1.0, 2.0, 3.0])
        result = mc.compile(poly, domain=(-5, 5))
        x = torch.linspace(-5, 5, 1000, dtype=torch.float64)
        y = result.compiled_function(x)
        expected = 1.0 + 2.0 * x + 3.0 * x**2
        assert torch.allclose(y, expected, atol=1e-10)

    def test_meta_compilation_affine(self) -> None:
        mc = MetaCompiler()
        node = AffineNode(scale_factor=torch.tensor(3.0), shift_value=torch.tensor(-1.0))
        result = mc.compile(node, domain=(-5, 5))
        x = torch.linspace(-5, 5, 1000, dtype=torch.float64)
        y = result.compiled_function(x)
        expected = 3.0 * x - 1.0
        assert torch.allclose(y, expected, atol=1e-10)

    def test_self_compilation(self) -> None:
        mc = MetaCompiler()
        result = mc.compile_self()
        assert result.is_self_compiling
        assert result.meta_depth == 2

    def test_meta_circularity_verification(self, poem: Poem) -> None:
        mc = MetaCompiler()
        node = poem.polynomial([1.0, -0.5, 0.1])
        x = torch.linspace(-3, 3, 1000, dtype=torch.float64)
        verification = mc.verify_meta_circularity(node, domain=(-3, 3), x_test=x)
        assert verification["outputs_agree"]
        assert verification["max_error"] < 1e-10


class TestInwardSpiral:
    def test_algebra_to_semantics_pipeline(self) -> None:
        algebra = FreeAlgebra()
        semantics = SheafSemantics()
        inner = ShiftNode(value=torch.tensor(3.0))
        outer = ShiftNode(value=torch.tensor(2.0), child=inner)
        normalized, _ = algebra.normalize(outer)
        verdict = semantics.analyze(normalized, domain=(-5, 5))
        assert verdict.is_correct

    def test_full_spiral_polynomial(self, poem: Poem) -> None:
        mc = MetaCompiler()
        poly = poem.polynomial([1.0, 0.0, -0.5])
        result = mc.compile(poly, domain=(-3, 3))
        x = torch.linspace(-3, 3, 1000, dtype=torch.float64)
        y = result.compiled_function(x)
        expected = 1.0 - 0.5 * x**2
        assert torch.allclose(y, expected, atol=1e-10)
        assert result.semantic_verdict.is_correct
        assert result.normalization_trace is not None

    def test_semantics_to_mta_pipeline(self) -> None:
        semantics = SheafSemantics()
        mta = AffineTuringMachine()
        node = AffineNode(scale_factor=torch.tensor(2.0), shift_value=torch.tensor(1.0))
        verdict = semantics.analyze(node, domain=(-5, 5))
        assert verdict.is_correct
        program = AffineTuringMachine.build_conditional(
            threshold=0.0,
            action_above=node,
            action_below=IdentityNode(),
        )
        result = mta.execute(program, torch.tensor([3.0]))
        assert result.accepted
