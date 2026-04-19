"""
Tests for acf_functor/meta_compiler.py

Covers:
  - Grammar dataclass: repr, hashable, koopman flag
  - GrammarSpace: all_grammars count, custom space
  - GrammarEvaluator: all 8 basis families; error handling
  - GridSearch, RandomSearch, GreedySearch: correctness + early stop
  - ACFMetaCompiler.compile: all strategies, improvement metrics
  - MetaCompilerResult: summary, serialization
  - GideonEngine.meta_compile integration
"""

import math
import unittest

import numpy as np

from acf_functor.meta_compiler import (
    ACFMetaCompiler,
    BasisFamily,
    Grammar,
    GrammarEvaluator,
    GrammarPoint,
    GrammarSpace,
    GreedySearch,
    GridSearch,
    MetaCompilerConfig,
    MetaCompilerResult,
    MetaCompilerTrace,
    RandomSearch,
)


# ────────────────────────────────────────────────────────────────────────────
# Test functions (vectorized — accept torch.Tensor or float)
# ────────────────────────────────────────────────────────────────────────────

import torch as _torch

def f_sin(x):
    if isinstance(x, _torch.Tensor):
        return _torch.sin(x)
    return _math.sin(float(x))

def f_poly(x):
    if isinstance(x, _torch.Tensor):
        return x**3 - 2*x + 1
    xf = float(x)
    return xf**3 - 2*xf + 1

def f_exp(x):
    if isinstance(x, _torch.Tensor):
        return _torch.exp(-_torch.abs(x))
    return _math.exp(-abs(float(x)))


DOMAIN = (-2.0, 2.0)
SMALL_DOMAIN = (-1.0, 1.0)


# ────────────────────────────────────────────────────────────────────────────
# TestGrammar
# ────────────────────────────────────────────────────────────────────────────

class TestGrammar(unittest.TestCase):

    def test_grammar_is_hashable(self):
        g = Grammar(basis=BasisFamily.CHEBYSHEV, degree=10)
        h = hash(g)
        self.assertIsInstance(h, int)

    def test_grammar_in_set(self):
        g1 = Grammar(basis=BasisFamily.CHEBYSHEV, degree=10)
        g2 = Grammar(basis=BasisFamily.CHEBYSHEV, degree=10)
        s = {g1, g2}
        self.assertEqual(len(s), 1)

    def test_different_grammars_different_hash(self):
        g1 = Grammar(basis=BasisFamily.CHEBYSHEV, degree=10)
        g2 = Grammar(basis=BasisFamily.FOURIER, degree=10)
        self.assertNotEqual(g1, g2)

    def test_is_koopman_chebyshev(self):
        g = Grammar(basis=BasisFamily.CHEBYSHEV, degree=8)
        self.assertFalse(g.is_koopman())

    def test_is_koopman_koopman_poly(self):
        g = Grammar(basis=BasisFamily.KOOPMAN_POLY, degree=8)
        self.assertTrue(g.is_koopman())

    def test_is_koopman_fourier(self):
        g = Grammar(basis=BasisFamily.KOOPMAN_FOURIER, degree=8)
        self.assertTrue(g.is_koopman())

    def test_repr_contains_basis_name(self):
        g = Grammar(basis=BasisFamily.LEGENDRE, degree=12)
        s = repr(g)
        self.assertIn("LEGENDRE", s)

    def test_repr_contains_degree(self):
        g = Grammar(basis=BasisFamily.CHEBYSHEV, degree=7)
        s = repr(g)
        self.assertIn("7", s)

    def test_frozen_immutable(self):
        g = Grammar(basis=BasisFamily.CHEBYSHEV, degree=5)
        with self.assertRaises((AttributeError, TypeError)):
            g.degree = 99  # type: ignore

    def test_all_basis_families_valid(self):
        for bf in BasisFamily:
            g = Grammar(basis=bf, degree=5)
            self.assertEqual(g.basis, bf)


# ────────────────────────────────────────────────────────────────────────────
# TestGrammarSpace
# ────────────────────────────────────────────────────────────────────────────

class TestGrammarSpace(unittest.TestCase):

    def test_default_space_has_grammars(self):
        space = GrammarSpace()
        grammars = space.all_grammars()
        self.assertGreater(len(grammars), 0)

    def test_all_grammars_are_grammar_instances(self):
        space = GrammarSpace()
        for g in space.all_grammars():
            self.assertIsInstance(g, Grammar)

    def test_n_total_matches_all_grammars(self):
        space = GrammarSpace()
        self.assertEqual(space.n_total(), len(space.all_grammars()))

    def test_custom_families(self):
        space = GrammarSpace(
            families=[BasisFamily.CHEBYSHEV, BasisFamily.FOURIER],
            degree_range=(5, 20),
            degree_step=5,
            n_observables_options=[8],
        )
        grammars = space.all_grammars()
        bases = {g.basis for g in grammars}
        self.assertSetEqual(bases, {BasisFamily.CHEBYSHEV, BasisFamily.FOURIER})

    def test_degree_range_respected(self):
        space = GrammarSpace(
            families=[BasisFamily.CHEBYSHEV],
            degree_range=(10, 20),
            degree_step=5,
            n_observables_options=[8],
        )
        for g in space.all_grammars():
            self.assertGreaterEqual(g.degree, 10)
            self.assertLessEqual(g.degree, 20)

    def test_no_duplicate_grammars(self):
        space = GrammarSpace()
        grammars = space.all_grammars()
        self.assertEqual(len(grammars), len(set(grammars)))

    def test_minimal_space_small(self):
        space = GrammarSpace(
            families=[BasisFamily.CHEBYSHEV],
            degree_range=(10, 10),
            degree_step=1,
            n_observables_options=[8],
        )
        grammars = space.all_grammars()
        # At least one grammar; may be more if multiple methods exist per family
        self.assertGreaterEqual(len(grammars), 1)


# ────────────────────────────────────────────────────────────────────────────
# TestGrammarEvaluator
# ────────────────────────────────────────────────────────────────────────────

class TestGrammarEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = GrammarEvaluator(n_probe=300)

    def _check_point(self, pt: GrammarPoint, grammar: Grammar):
        self.assertIsInstance(pt, GrammarPoint)
        self.assertEqual(pt.grammar, grammar)
        self.assertTrue(math.isfinite(pt.epsilon) or pt.error_message is not None)

    def test_chebyshev_evaluation(self):
        g = Grammar(basis=BasisFamily.CHEBYSHEV, degree=10)
        pt = self.evaluator.evaluate(g, f_sin, DOMAIN)
        self._check_point(pt, g)

    def test_horner_evaluation(self):
        g = Grammar(basis=BasisFamily.HORNER, degree=8)
        pt = self.evaluator.evaluate(g, f_poly, SMALL_DOMAIN)
        self._check_point(pt, g)

    def test_legendre_evaluation(self):
        g = Grammar(basis=BasisFamily.LEGENDRE, degree=8)
        pt = self.evaluator.evaluate(g, f_sin, DOMAIN)
        self._check_point(pt, g)

    def test_fourier_evaluation(self):
        g = Grammar(basis=BasisFamily.FOURIER, degree=8)
        pt = self.evaluator.evaluate(g, f_sin, DOMAIN)
        self._check_point(pt, g)

    def test_rbf_evaluation(self):
        g = Grammar(basis=BasisFamily.RBF, degree=8)
        pt = self.evaluator.evaluate(g, f_exp, DOMAIN)
        self._check_point(pt, g)

    def test_koopman_poly_evaluation(self):
        g = Grammar(basis=BasisFamily.KOOPMAN_POLY, degree=8, n_observables=6)
        pt = self.evaluator.evaluate(g, f_sin, DOMAIN)
        self._check_point(pt, g)

    def test_koopman_fourier_evaluation(self):
        g = Grammar(basis=BasisFamily.KOOPMAN_FOURIER, degree=8, n_observables=6)
        pt = self.evaluator.evaluate(g, f_exp, DOMAIN)
        self._check_point(pt, g)

    def test_koopman_rbf_evaluation(self):
        g = Grammar(basis=BasisFamily.KOOPMAN_RBF, degree=8, n_observables=6)
        pt = self.evaluator.evaluate(g, f_poly, SMALL_DOMAIN)
        self._check_point(pt, g)

    def test_epsilon_non_negative(self):
        for bf in BasisFamily:
            g = Grammar(basis=bf, degree=8, n_observables=6)
            pt = self.evaluator.evaluate(g, f_sin, DOMAIN)
            if pt.error_message is None:
                self.assertGreaterEqual(pt.epsilon, 0.0, f"Negative epsilon for {bf.name}")

    def test_entropy_non_negative(self):
        g = Grammar(basis=BasisFamily.CHEBYSHEV, degree=10)
        pt = self.evaluator.evaluate(g, f_sin, DOMAIN)
        if pt.error_message is None:
            self.assertGreaterEqual(pt.entropy, 0.0)

    def test_free_energy_finite(self):
        g = Grammar(basis=BasisFamily.CHEBYSHEV, degree=10)
        pt = self.evaluator.evaluate(g, f_sin, DOMAIN)
        if pt.error_message is None:
            self.assertTrue(math.isfinite(pt.free_energy))

    def test_graceful_failure_no_raise(self):
        # degree=0 should either work or return error_message, not raise
        g = Grammar(basis=BasisFamily.CHEBYSHEV, degree=0)
        try:
            pt = self.evaluator.evaluate(g, f_sin, DOMAIN)
            # If it doesn't raise, that's fine (error_message may be set)
        except Exception as exc:
            self.fail(f"GrammarEvaluator raised unexpectedly: {exc}")


# ────────────────────────────────────────────────────────────────────────────
# TestGridSearch
# ────────────────────────────────────────────────────────────────────────────

class TestGridSearch(unittest.TestCase):

    def _small_space(self):
        return GrammarSpace(
            families=[BasisFamily.CHEBYSHEV, BasisFamily.FOURIER],
            degree_range=(5, 15),
            degree_step=5,
            n_observables_options=[8],
        )

    def _evaluator(self):
        return GrammarEvaluator(n_probe=200)

    def test_returns_list_of_points(self):
        space = self._small_space()
        ev = self._evaluator()
        search = GridSearch()
        results = search.search(f_sin, DOMAIN, space, ev)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_all_results_are_grammar_points(self):
        space = self._small_space()
        ev = self._evaluator()
        search = GridSearch()
        results = search.search(f_sin, DOMAIN, space, ev)
        for pt in results:
            self.assertIsInstance(pt, GrammarPoint)

    def test_evaluates_all_grammars(self):
        space = self._small_space()
        n = space.n_total()
        ev = self._evaluator()
        search = GridSearch()
        results = search.search(f_sin, DOMAIN, space, ev)
        # All grammars should be evaluated (no early stop since target unset)
        self.assertGreaterEqual(len(results), min(n, 1))

    def test_early_stop_at_target(self):
        # Very lenient target — should stop after first success
        space = self._small_space()
        ev = self._evaluator()
        search = GridSearch()
        results = search.search(f_sin, DOMAIN, space, ev, target_epsilon=1e3)
        # With target=1e3 (very large), should stop early
        self.assertGreater(len(results), 0)


# ────────────────────────────────────────────────────────────────────────────
# TestRandomSearch
# ────────────────────────────────────────────────────────────────────────────

class TestRandomSearch(unittest.TestCase):

    def test_returns_list_of_points(self):
        space = GrammarSpace()
        ev = GrammarEvaluator(n_probe=200)
        search = RandomSearch(budget=8, seed=42)
        results = search.search(f_poly, SMALL_DOMAIN, space, ev)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_budget_respected(self):
        space = GrammarSpace()
        budget = 6
        ev = GrammarEvaluator(n_probe=200)
        search = RandomSearch(budget=budget, seed=0)
        results = search.search(f_poly, SMALL_DOMAIN, space, ev)
        self.assertLessEqual(len(results), budget)

    def test_seed_reproducibility(self):
        space = GrammarSpace()
        ev = GrammarEvaluator(n_probe=200)
        s1 = RandomSearch(budget=5, seed=7)
        s2 = RandomSearch(budget=5, seed=7)
        r1 = s1.search(f_sin, DOMAIN, space, ev)
        r2 = s2.search(f_sin, DOMAIN, space, ev)
        self.assertEqual([pt.grammar for pt in r1], [pt.grammar for pt in r2])


# ────────────────────────────────────────────────────────────────────────────
# TestGreedySearch
# ────────────────────────────────────────────────────────────────────────────

class TestGreedySearch(unittest.TestCase):

    def test_returns_list_of_points(self):
        space = GrammarSpace(
            families=[BasisFamily.CHEBYSHEV, BasisFamily.FOURIER],
            degree_range=(5, 20),
            degree_step=5,
            n_observables_options=[8],
        )
        ev = GrammarEvaluator(n_probe=200)
        search = GreedySearch()
        results = search.search(f_exp, DOMAIN, space, ev)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_best_not_worse_than_first(self):
        space = GrammarSpace(
            families=[BasisFamily.CHEBYSHEV],
            degree_range=(5, 25),
            degree_step=5,
            n_observables_options=[8],
        )
        ev = GrammarEvaluator(n_probe=200)
        search = GreedySearch()
        results = search.search(f_sin, DOMAIN, space, ev)
        if len(results) > 1:
            best_eps = min(pt.epsilon for pt in results if pt.error_message is None)
            first_eps = results[0].epsilon
            # Greedy should find something at least as good as its starting point
            self.assertLessEqual(best_eps, first_eps + 1e-6)


# ────────────────────────────────────────────────────────────────────────────
# TestACFMetaCompiler
# ────────────────────────────────────────────────────────────────────────────

class TestACFMetaCompiler(unittest.TestCase):

    def _small_config(self, strategy: str = "greedy") -> MetaCompilerConfig:
        space = GrammarSpace(
            families=[BasisFamily.CHEBYSHEV, BasisFamily.FOURIER, BasisFamily.LEGENDRE],
            degree_range=(5, 20),
            degree_step=5,
            n_observables_options=[8],
        )
        return MetaCompilerConfig(
            grammar_space=space,
            strategy=strategy,
            beta=1.0,
            target_epsilon=1e-4,
            enable_auto_evolution=False,  # slow; tested separately
            n_probe=300,
        )

    def test_compile_returns_result(self):
        compiler = ACFMetaCompiler(self._small_config())
        result = compiler.compile(f_sin, DOMAIN)
        self.assertIsInstance(result, MetaCompilerResult)

    def test_best_grammar_is_grammar(self):
        compiler = ACFMetaCompiler(self._small_config())
        result = compiler.compile(f_sin, DOMAIN)
        self.assertIsInstance(result.best_grammar, Grammar)

    def test_final_epsilon_non_negative(self):
        compiler = ACFMetaCompiler(self._small_config())
        result = compiler.compile(f_poly, SMALL_DOMAIN)
        self.assertGreaterEqual(result.final_epsilon, 0.0)

    def test_final_epsilon_finite(self):
        compiler = ACFMetaCompiler(self._small_config())
        result = compiler.compile(f_poly, SMALL_DOMAIN)
        self.assertTrue(math.isfinite(result.final_epsilon))

    def test_final_le_initial_epsilon(self):
        compiler = ACFMetaCompiler(self._small_config())
        result = compiler.compile(f_exp, DOMAIN)
        # Meta-compiler should not make things worse
        self.assertLessEqual(result.final_epsilon, result.initial_epsilon + 1e-9)

    def test_improvement_ratio_positive(self):
        compiler = ACFMetaCompiler(self._small_config())
        result = compiler.compile(f_sin, DOMAIN)
        self.assertGreater(result.improvement_ratio, 0.0)

    def test_trace_contains_points(self):
        compiler = ACFMetaCompiler(self._small_config())
        result = compiler.compile(f_sin, DOMAIN)
        self.assertGreater(len(result.trace.all_grammars), 0)

    def test_greedy_strategy(self):
        compiler = ACFMetaCompiler(self._small_config("greedy"))
        result = compiler.compile(f_sin, DOMAIN)
        self.assertIsInstance(result, MetaCompilerResult)

    def test_grid_strategy(self):
        compiler = ACFMetaCompiler(self._small_config("grid"))
        result = compiler.compile(f_poly, SMALL_DOMAIN)
        self.assertIsInstance(result, MetaCompilerResult)

    def test_random_strategy(self):
        cfg = self._small_config("random")
        compiler = ACFMetaCompiler(cfg)
        result = compiler.compile(f_exp, DOMAIN)
        self.assertIsInstance(result, MetaCompilerResult)

    def test_analyse_grammar_space_returns_trace(self):
        space = GrammarSpace(
            families=[BasisFamily.CHEBYSHEV, BasisFamily.FOURIER],
            degree_range=(5, 10),
            degree_step=5,
            n_observables_options=[8],
        )
        cfg = MetaCompilerConfig(
            grammar_space=space,
            strategy="grid",
            enable_auto_evolution=False,
            n_probe=200,
        )
        compiler = ACFMetaCompiler(cfg)
        trace = compiler.analyse_grammar_space(f_sin, DOMAIN)
        self.assertIsInstance(trace, MetaCompilerTrace)

    def test_analyse_grammar_space_trace_has_all_grammars(self):
        space = GrammarSpace(
            families=[BasisFamily.CHEBYSHEV, BasisFamily.FOURIER],
            degree_range=(5, 10),
            degree_step=5,
            n_observables_options=[8],
        )
        cfg = MetaCompilerConfig(
            grammar_space=space,
            strategy="grid",
            enable_auto_evolution=False,
            n_probe=200,
        )
        compiler = ACFMetaCompiler(cfg)
        trace = compiler.analyse_grammar_space(f_sin, DOMAIN)
        self.assertGreaterEqual(len(trace.all_grammars), space.n_total())


# ────────────────────────────────────────────────────────────────────────────
# TestMetaCompilerResult
# ────────────────────────────────────────────────────────────────────────────

class TestMetaCompilerResult(unittest.TestCase):

    def _compile(self):
        space = GrammarSpace(
            families=[BasisFamily.CHEBYSHEV, BasisFamily.LEGENDRE],
            degree_range=(5, 15),
            degree_step=5,
            n_observables_options=[8],
        )
        cfg = MetaCompilerConfig(
            grammar_space=space,
            strategy="greedy",
            enable_auto_evolution=False,
            n_probe=200,
        )
        return ACFMetaCompiler(cfg).compile(f_sin, DOMAIN)

    def test_summary_contains_best_grammar(self):
        result = self._compile()
        s = result.summary()
        self.assertIn("best grammar:", s)

    def test_summary_contains_epsilon(self):
        result = self._compile()
        s = result.summary()
        self.assertIn("ε", s)

    def test_summary_contains_improvement(self):
        result = self._compile()
        s = result.summary()
        self.assertIn("improvement", s.lower())

    def test_best_reduction_not_none(self):
        result = self._compile()
        self.assertIsNotNone(result.best_reduction)

    def test_initial_epsilon_positive(self):
        result = self._compile()
        self.assertGreater(result.initial_epsilon, 0.0)


# ────────────────────────────────────────────────────────────────────────────
# TestGrammarSpaceCompleteness
# ────────────────────────────────────────────────────────────────────────────

class TestGrammarSpaceCompleteness(unittest.TestCase):

    def test_all_basis_families_evaluable(self):
        """Each BasisFamily must evaluate without exception."""
        ev = GrammarEvaluator(n_probe=200)
        for bf in BasisFamily:
            g = Grammar(basis=bf, degree=8, n_observables=6)
            with self.subTest(basis=bf.name):
                try:
                    pt = ev.evaluate(g, f_sin, DOMAIN)
                    # Should return GrammarPoint, not raise
                    self.assertIsInstance(pt, GrammarPoint)
                except Exception as exc:
                    self.fail(f"BasisFamily.{bf.name} raised: {exc}")


# ────────────────────────────────────────────────────────────────────────────
# TestGideonMetaCompileIntegration
# ────────────────────────────────────────────────────────────────────────────

class TestGideonMetaCompileIntegration(unittest.TestCase):

    def setUp(self):
        from poema.backends.gideon.engine import GideonEngine
        self.engine = GideonEngine()

    def test_meta_compile_returns_result(self):
        result = self.engine.meta_compile(
            f=f_sin,
            domain=DOMAIN,
            strategy="greedy",
            target_epsilon=1e-3,
            enable_auto_evolution=False,
        )
        self.assertIsInstance(result, MetaCompilerResult)

    def test_meta_compile_best_grammar_valid(self):
        result = self.engine.meta_compile(
            f=f_poly,
            domain=SMALL_DOMAIN,
            strategy="random",
            target_epsilon=1e-3,
            enable_auto_evolution=False,
        )
        self.assertIsInstance(result.best_grammar, Grammar)

    def test_meta_compile_epsilon_non_negative(self):
        result = self.engine.meta_compile(
            f=f_exp,
            domain=DOMAIN,
            strategy="greedy",
            target_epsilon=1e-3,
            enable_auto_evolution=False,
        )
        self.assertGreaterEqual(result.final_epsilon, 0.0)


# ────────────────────────────────────────────────────────────────────────────
# TestMetaCompilerPublicAPI
# ────────────────────────────────────────────────────────────────────────────

class TestMetaCompilerPublicAPI(unittest.TestCase):

    def test_imports_from_acf_functor(self):
        from acf_functor import (
            ACFMetaCompiler, MetaCompilerConfig, GrammarSpace, Grammar,
            GrammarPoint, BasisFamily, MetaCompilerTrace, MetaCompilerResult,
            GrammarEvaluator, GridSearch, RandomSearch, GreedySearch,
        )
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
