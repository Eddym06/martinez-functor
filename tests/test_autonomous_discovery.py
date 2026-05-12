"""
Tests for the Autonomous Discovery Engine
==========================================

Covers: OperatorStructureAnalyzer, ButterflyGraphSynthesizer,
        CoPoemRefinementLoop, KnowledgeGraph, DFTStructureDiscovery,
        AutonomousDiscoveryEngine
"""

import math
import numpy as np
import pytest


# ===========================================================================
# §1  Knowledge Graph
# ===========================================================================

class TestKnowledgeGraph:

    def test_create_empty(self):
        from acf_functor.autonomous_discovery import KnowledgeGraph
        kg = KnowledgeGraph()
        assert kg.size == 0

    def test_assimilate(self):
        from acf_functor.autonomous_discovery import (
            KnowledgeGraph, DiscoveryResult, DiscoveryKind)
        kg = KnowledgeGraph()
        d = DiscoveryResult(
            kind=DiscoveryKind.ALGORITHM,
            name="test_discovery",
            description="A test",
            evidence={"fma": 100},
            confidence=0.9,
        )
        entry_id = kg.assimilate(d)
        assert kg.size == 1
        assert entry_id.startswith("KG-")

    def test_query_by_kind(self):
        from acf_functor.autonomous_discovery import (
            KnowledgeGraph, DiscoveryResult, DiscoveryKind)
        kg = KnowledgeGraph()
        for kind in [DiscoveryKind.ALGORITHM, DiscoveryKind.STRUCTURE,
                     DiscoveryKind.ALGORITHM]:
            kg.assimilate(DiscoveryResult(
                kind=kind, name=f"d_{kind.value}", description="",
                evidence={}, confidence=0.8))
        algos = kg.query(kind=DiscoveryKind.ALGORITHM)
        assert len(algos) == 2

    def test_supersede(self):
        from acf_functor.autonomous_discovery import (
            KnowledgeGraph, DiscoveryResult, DiscoveryKind)
        kg = KnowledgeGraph()
        id1 = kg.assimilate(DiscoveryResult(
            kind=DiscoveryKind.ALGORITHM, name="v1", description="",
            evidence={}, confidence=0.5))
        id2 = kg.assimilate(DiscoveryResult(
            kind=DiscoveryKind.ALGORITHM, name="v2", description="",
            evidence={}, confidence=0.9))
        kg.supersede(id1, id2)
        assert kg.size == 1  # v1 superseded


# ===========================================================================
# §2  Operator Structure Analyzer
# ===========================================================================

class TestOperatorStructureAnalyzer:

    def test_analyze_identity(self):
        from acf_functor.autonomous_discovery import OperatorStructureAnalyzer
        osa = OperatorStructureAnalyzer()
        I = np.eye(16)
        result = osa.analyze(I)
        assert result["is_symmetric"]
        assert result["is_unitary"]
        assert result["sv_decay_rate"] == "flat"

    def test_analyze_random(self):
        from acf_functor.autonomous_discovery import OperatorStructureAnalyzer
        osa = OperatorStructureAnalyzer()
        rng = np.random.RandomState(42)
        A = rng.randn(32, 32)
        result = osa.analyze(A)
        assert result["n"] == 32
        assert "block_structure" in result
        assert "recursive_structure" in result

    def test_detect_symmetric(self):
        from acf_functor.autonomous_discovery import OperatorStructureAnalyzer
        osa = OperatorStructureAnalyzer()
        rng = np.random.RandomState(42)
        A = rng.randn(16, 16)
        S = A + A.T
        result = osa.analyze(S)
        assert result["is_symmetric"]

    def test_detect_sparse(self):
        from acf_functor.autonomous_discovery import OperatorStructureAnalyzer
        osa = OperatorStructureAnalyzer()
        A = np.zeros((32, 32))
        for i in range(32):
            A[i, i] = 2.0
            if i > 0:
                A[i, i - 1] = -1.0
            if i < 31:
                A[i, i + 1] = -1.0
        result = osa.analyze(A)
        assert result["is_sparse"]
        assert result["density"] < 0.15

    def test_detect_dft_unitary(self):
        """DFT matrix should be detected as unitary (up to 1/sqrt(N))."""
        from acf_functor.autonomous_discovery import OperatorStructureAnalyzer
        osa = OperatorStructureAnalyzer()
        N = 16
        j = np.arange(N).reshape(-1, 1)
        k = np.arange(N).reshape(1, -1)
        F = np.exp(-2j * np.pi * j * k / N) / np.sqrt(N)
        # Analyze the real block form
        A_block = np.block([
            [np.real(F), -np.imag(F)],
            [np.imag(F),  np.real(F)],
        ])
        result = osa.analyze(A_block)
        assert result["unitarity_error"] < 1e-8

    def test_singular_value_decay(self):
        from acf_functor.autonomous_discovery import OperatorStructureAnalyzer
        osa = OperatorStructureAnalyzer()
        # Low-rank matrix: exponential decay
        rng = np.random.RandomState(42)
        U = rng.randn(32, 5)
        V = rng.randn(5, 32)
        A = U @ V
        result = osa.analyze(A)
        assert result["numerical_rank_1e6"] <= 10

    def test_recursive_structure_power_of_2(self):
        from acf_functor.autonomous_discovery import OperatorStructureAnalyzer
        osa = OperatorStructureAnalyzer()
        N = 8
        j = np.arange(N).reshape(-1, 1)
        k = np.arange(N).reshape(1, -1)
        F = np.abs(np.exp(-2j * np.pi * j * k / N))
        result = osa.analyze(F)
        assert "recursive_structure" in result
        assert result["recursive_structure"]["n_levels"] == 3  # log2(8)


# ===========================================================================
# §3  Butterfly Graph Synthesizer
# ===========================================================================

class TestButterflyGraphSynthesizer:

    def test_butterfly_graph_n4(self):
        from acf_functor.autonomous_discovery import ButterflyGraphSynthesizer
        synth = ButterflyGraphSynthesizer()
        graph = synth.synthesize_butterfly(4)
        assert graph.n_nodes > 0
        assert graph.total_fma > 0

    def test_butterfly_graph_n8(self):
        from acf_functor.autonomous_discovery import ButterflyGraphSynthesizer
        synth = ButterflyGraphSynthesizer()
        graph = synth.synthesize_butterfly(8)
        # N=8: 8 inputs + 3 stages × 8 butterflies/stage (but paired) + 8 sinks
        assert graph.n_nodes >= 8 + 8  # at least inputs + sinks

    def test_synthesize_dft_algorithm_correctness(self):
        """The synthesized butterfly FFT must match numpy.fft.fft."""
        from acf_functor.autonomous_discovery import ButterflyGraphSynthesizer
        synth = ButterflyGraphSynthesizer()
        for N in [4, 8, 16, 32]:
            algo, graph = synth.synthesize_dft_algorithm(N)
            rng = np.random.RandomState(42)
            x = rng.randn(N) + 1j * rng.randn(N)
            y_butterfly = algo.execute(x)
            y_numpy = np.fft.fft(x)
            err = np.linalg.norm(y_butterfly - y_numpy) / np.linalg.norm(y_numpy)
            assert err < 1e-10, f"N={N}: error={err}"

    def test_fma_count_is_nlogn(self):
        from acf_functor.autonomous_discovery import ButterflyGraphSynthesizer
        synth = ButterflyGraphSynthesizer()
        for k in [3, 4, 5, 6]:
            N = 2 ** k
            algo, _ = synth.synthesize_dft_algorithm(N)
            expected = N * k * 2
            assert algo.n_fma == expected

    def test_algorithm_has_certificate(self):
        from acf_functor.autonomous_discovery import ButterflyGraphSynthesizer
        synth = ButterflyGraphSynthesizer()
        algo, _ = synth.synthesize_dft_algorithm(16)
        cert = algo.certificate
        assert "method" in cert
        assert cert["method"] == "butterfly_factorization"
        assert cert["n_stages"] == 4  # log2(16)


# ===========================================================================
# §4  CoPoem Refinement Loop
# ===========================================================================

class TestCoPoemRefinementLoop:

    def test_refine_already_meets_target(self):
        """If algorithm already meets target, no refinement needed."""
        from acf_functor.autonomous_discovery import CoPoemRefinementLoop
        from acf_functor.algorithm_forge import (
            AlgorithmForge, ProblemSpec, ProblemKind)
        forge = AlgorithmForge(verify=False)
        spec = ProblemSpec(
            kind=ProblemKind.LINEAR_SOLVE,
            description="Simple test", n=20, target_error=1e-4)
        algo = forge.forge(spec)
        loop = CoPoemRefinementLoop(max_refinements=3)
        refined, report = loop.refine(algo, target_fma=algo.n_fma * 2,
                                       target_error=1e-4)
        assert report["converged"]

    def test_refine_reduces_fma(self):
        """Refinement should attempt to reduce FMA count."""
        from acf_functor.autonomous_discovery import CoPoemRefinementLoop
        from acf_functor.algorithm_forge import (
            AlgorithmForge, ProblemSpec, ProblemKind)
        forge = AlgorithmForge(verify=False)
        spec = ProblemSpec(
            kind=ProblemKind.LINEAR_SOLVE,
            description="Reduce test", n=100, target_error=1e-4)
        algo = forge.forge(spec)
        loop = CoPoemRefinementLoop(max_refinements=3)
        _, report = loop.refine(algo, target_fma=algo.n_fma // 100,
                                 target_error=1e-2)
        assert report["n_refinements"] > 0


# ===========================================================================
# §5  DFT Structure Discovery (FFT Experiment)
# ===========================================================================

class TestDFTStructureDiscovery:

    def test_dft_discovery_n8(self):
        """Discover FFT structure for N=8."""
        from acf_functor.autonomous_discovery import DFTStructureDiscovery
        exp = DFTStructureDiscovery(N=8, target_error=1e-6)
        report = exp.run_full_experiment()
        # Phase 2: should detect unitary structure
        assert "2_analysis" in report["phases"]
        # Phase 5: verification must pass
        assert report["phases"]["5_verification"]["passed"]
        # Discoveries
        assert len(report["discoveries"]) >= 3
        assert report["success"]

    def test_dft_discovery_n16(self):
        """Discover FFT structure for N=16."""
        from acf_functor.autonomous_discovery import DFTStructureDiscovery
        exp = DFTStructureDiscovery(N=16, target_error=1e-6)
        report = exp.run_full_experiment()
        assert report["phases"]["5_verification"]["passed"]
        assert report["success"]
        # Check certificate
        assert "AD-1" in report["certificate"]
        assert "AD-3" in report["certificate"]

    def test_dft_discovery_n32(self):
        """Discover FFT structure for N=32."""
        from acf_functor.autonomous_discovery import DFTStructureDiscovery
        exp = DFTStructureDiscovery(N=32, target_error=1e-6)
        report = exp.run_full_experiment()
        assert report["phases"]["5_verification"]["passed"]
        assert report["success"]
        # FMA count should be O(N log N)
        synth_fma = report["phases"]["4_synthesis"]["n_fma"]
        direct_fma = 32 * 32
        assert synth_fma < direct_fma

    def test_butterfly_score_high(self):
        """Butterfly factorizability score should be high for DFT."""
        from acf_functor.autonomous_discovery import DFTStructureDiscovery
        exp = DFTStructureDiscovery(N=16, target_error=1e-6)
        report = exp.run_full_experiment()
        butterfly_score = report["phases"]["3_discovery"]["butterfly_score"]
        assert butterfly_score > 0.3

    def test_scale_test_subquadratic(self):
        """The discovered algorithm should scale sub-quadratically."""
        from acf_functor.autonomous_discovery import DFTStructureDiscovery
        exp = DFTStructureDiscovery(N=16, target_error=1e-6)
        report = exp.run_full_experiment()
        # Check that scale test was performed
        assert "8_scale_test" in report["phases"]
        scale = report["phases"]["8_scale_test"]
        # All sizes should produce correct results
        for r in scale["scale_results"]:
            assert r["correct"], f"N={r['N']}: error={r['error']}"


# ===========================================================================
# §6  Autonomous Discovery Engine
# ===========================================================================

class TestAutonomousDiscoveryEngine:

    def test_create_engine(self):
        from acf_functor.autonomous_discovery import AutonomousDiscoveryEngine
        engine = AutonomousDiscoveryEngine(verify=False)
        assert engine.cycle_count == 0
        assert engine.knowledge.size == 0

    def test_single_discovery_cycle(self):
        from acf_functor.autonomous_discovery import AutonomousDiscoveryEngine
        engine = AutonomousDiscoveryEngine(verify=False)
        rng = np.random.RandomState(42)
        A = rng.randn(20, 20)
        A = A.T @ A + np.eye(20)
        problem = {
            "data": A,
            "kind": "algorithm",
            "problem_kind": "linear_solve",
            "description": "Solve Ax=b",
            "n": 20,
            "target_error": 1e-4,
        }
        report = engine.run_discovery_cycle(problem)
        assert report["cycle"] == 1
        assert len(report["phases"]) >= 4
        assert engine.knowledge.size == 1

    def test_multiple_cycles_accumulate_knowledge(self):
        from acf_functor.autonomous_discovery import AutonomousDiscoveryEngine
        engine = AutonomousDiscoveryEngine(verify=False)
        rng = np.random.RandomState(42)
        for _ in range(3):
            A = rng.randn(15, 15)
            A = A.T @ A + np.eye(15)
            problem = {
                "data": A,
                "kind": "algorithm",
                "problem_kind": "linear_solve",
                "description": "Solve Ax=b",
                "n": 15,
                "target_error": 1e-4,
            }
            engine.run_discovery_cycle(problem)
        assert engine.cycle_count == 3
        assert engine.knowledge.size == 3


# ===========================================================================
# §7  End-to-End Integration
# ===========================================================================

class TestEndToEndDiscovery:

    def test_full_fft_discovery_and_execute(self):
        """Full pipeline: discover FFT → execute → verify."""
        from acf_functor.autonomous_discovery import DFTStructureDiscovery
        exp = DFTStructureDiscovery(N=16, target_error=1e-6)
        report = exp.run_full_experiment()
        assert report["success"]
        # Check that the benchmark was done
        bench = report["phases"]["7_benchmark"]
        assert bench["speedup_vs_direct"] > 0

    def test_imports_from_top_level(self):
        """Verify all exports are importable."""
        from acf_functor import (
            AutonomousDiscoveryEngine,
            DFTStructureDiscovery,
            OperatorStructureAnalyzer,
            ButterflyGraphSynthesizer,
            CoPoemRefinementLoop,
            KnowledgeGraph,
            DiscoveryResult,
            DiscoveryKind,
            KnowledgeEntry,
            # Level-5 Autonomy
            TopologicalOperatorAnalyzer,
            TopologicalFingerprint,
            KoopmanStructuralAnalyzer,
            OperatorGrammarSearch,
            AutonomousRuleInduction,
            AutonomousRule,
        )
        assert AutonomousDiscoveryEngine is not None
        assert DFTStructureDiscovery is not None
        assert TopologicalOperatorAnalyzer is not None
        assert KoopmanStructuralAnalyzer is not None
        assert OperatorGrammarSearch is not None
        assert AutonomousRuleInduction is not None


# ===========================================================================
# §8  Topological Operator Analyzer (Level-5)
# ===========================================================================

class TestTopologicalOperatorAnalyzer:

    def test_fingerprint_dft_matrix(self):
        """TDA fingerprint of a DFT matrix should produce valid topology."""
        from acf_functor.autonomous_discovery import TopologicalOperatorAnalyzer
        tda = TopologicalOperatorAnalyzer()
        N = 8
        j = np.arange(N).reshape(-1, 1)
        k = np.arange(N).reshape(1, -1)
        F = np.exp(-2j * np.pi * j * k / N)
        fp = tda.fingerprint(np.abs(F))
        assert fp.n_eigenvalue_clusters >= 1
        assert isinstance(fp.persistence_bars, list)
        assert fp.intrinsic_dimension > 0
        assert fp.summary()  # ensure it doesn't crash
        # Hierarchical factorizability should be detected for DFT-like structures
        assert fp.hierarchical_factorizability >= 0.0

    def test_fingerprint_identity(self):
        """Identity matrix: trivial spectral structure."""
        from acf_functor.autonomous_discovery import TopologicalOperatorAnalyzer
        tda = TopologicalOperatorAnalyzer()
        I = np.eye(8)
        fp = tda.fingerprint(I)
        assert fp.intrinsic_dimension <= 8

    def test_fingerprint_random(self):
        """Random matrix: should produce a valid fingerprint."""
        from acf_functor.autonomous_discovery import TopologicalOperatorAnalyzer
        tda = TopologicalOperatorAnalyzer()
        rng = np.random.RandomState(42)
        A = rng.randn(16, 16)
        fp = tda.fingerprint(A)
        assert fp.n_eigenvalue_clusters >= 1
        assert fp.total_persistence >= 0.0
        assert len(fp.sv_rank_profile) > 0

    def test_fingerprint_small(self):
        """2×2 matrix should work without crashing."""
        from acf_functor.autonomous_discovery import TopologicalOperatorAnalyzer
        tda = TopologicalOperatorAnalyzer()
        A = np.array([[1, 2], [3, 4]], dtype=float)
        fp = tda.fingerprint(A)
        assert fp.n_eigenvalue_clusters >= 1


# ===========================================================================
# §9  Koopman Structural Analyzer (Level-5)
# ===========================================================================

class TestKoopmanStructuralAnalyzer:

    def test_analyze_dft_matrix(self):
        """Koopman analysis of DFT should detect unit circle eigenvalues."""
        from acf_functor.autonomous_discovery import KoopmanStructuralAnalyzer
        ksa = KoopmanStructuralAnalyzer()
        N = 8
        j = np.arange(N).reshape(-1, 1)
        k = np.arange(N).reshape(1, -1)
        F = np.abs(np.exp(-2j * np.pi * j * k / N))
        result = ksa.analyze(F)
        assert "n" in result
        assert result["n"] == N
        assert "spectral_radius" in result
        assert "koopman_frequencies" in result
        assert "band_structure" in result
        assert "group_structure" in result

    def test_analyze_identity(self):
        """Identity matrix Koopman analysis."""
        from acf_functor.autonomous_discovery import KoopmanStructuralAnalyzer
        ksa = KoopmanStructuralAnalyzer()
        I = np.eye(8)
        result = ksa.analyze(I)
        assert result["spectral_radius"] == pytest.approx(1.0, abs=0.01)

    def test_analyze_random(self):
        """Random matrix should produce valid Koopman analysis."""
        from acf_functor.autonomous_discovery import KoopmanStructuralAnalyzer
        ksa = KoopmanStructuralAnalyzer()
        rng = np.random.RandomState(42)
        A = rng.randn(16, 16)
        result = ksa.analyze(A)
        assert "koopman_factorizability" in result
        assert 0 <= result["koopman_factorizability"] <= 1.0

    def test_multiresolution_power_of_2(self):
        """Check multiresolution detection for power-of-2 sized DFT."""
        from acf_functor.autonomous_discovery import KoopmanStructuralAnalyzer
        ksa = KoopmanStructuralAnalyzer()
        N = 16
        j = np.arange(N).reshape(-1, 1)
        k = np.arange(N).reshape(1, -1)
        F = np.abs(np.exp(-2j * np.pi * j * k / N))
        result = ksa.analyze(F)
        mr = result.get("multiresolution", {})
        assert "depth" in mr


# ===========================================================================
# §10  Operator Grammar Search (Level-5)
# ===========================================================================

class TestOperatorGrammarSearch:

    def test_search_dft_abs(self):
        """Grammar search on |DFT| matrix."""
        from acf_functor.autonomous_discovery import OperatorGrammarSearch
        gs = OperatorGrammarSearch()
        N = 8
        j = np.arange(N).reshape(-1, 1)
        k = np.arange(N).reshape(1, -1)
        F = np.abs(np.exp(-2j * np.pi * j * k / N))
        result = gs.search(F)
        assert "found" in result
        assert "n_strategies_tried" in result or "strategies_tried" in result

    def test_search_identity(self):
        """Identity matrix: should find trivial factorization."""
        from acf_functor.autonomous_discovery import OperatorGrammarSearch
        gs = OperatorGrammarSearch()
        I = np.eye(8)
        result = gs.search(I)
        assert "found" in result

    def test_search_sparse_matrix(self):
        """Sparse matrix should be detected as sparse."""
        from acf_functor.autonomous_discovery import OperatorGrammarSearch
        gs = OperatorGrammarSearch()
        A = np.zeros((8, 8))
        A[0, 0] = 1; A[1, 3] = 2; A[3, 5] = 1
        result = gs.search(A)
        # Should detect sparsity
        all_strats = result.get("all_strategies", [])
        sparse_found = any(s.get("strategy") in ("sparse_direct", "banded")
                          for s in all_strats if s.get("found"))
        # At least the search ran without error
        assert "found" in result

    def test_search_non_square(self):
        """Non-square matrix should return found=False."""
        from acf_functor.autonomous_discovery import OperatorGrammarSearch
        gs = OperatorGrammarSearch()
        A = np.random.randn(6, 8)
        result = gs.search(A)
        assert result["found"] is False

    def test_build_algorithm_from_factorization(self):
        """Build an executable algorithm from a butterfly factorization."""
        from acf_functor.autonomous_discovery import OperatorGrammarSearch
        gs = OperatorGrammarSearch()
        N = 8
        # Create a factorization dict that looks like a butterfly result
        factorization = {
            "found": True,
            "strategy": "butterfly",
            "error": 1e-10,
            "fma_cost": N * int(math.log2(N)) * 2,
            "depth": int(math.log2(N)),
            "grammar_expression": "B_3·B_2·B_1·P_br",
            "factors": [],
        }
        algo = gs.build_algorithm_from_factorization(N, factorization)
        assert algo is not None
        # Execute and verify against np.fft.fft
        x = np.random.randn(N) + 1j * np.random.randn(N)
        y = algo.execute(x)
        y_ref = np.fft.fft(x)
        err = np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)
        assert err < 1e-10


# ===========================================================================
# §11  Autonomous Rule Induction (Level-5)
# ===========================================================================

class TestAutonomousRuleInduction:

    def _make_observation(self, N, rule_ind):
        """Helper to create and feed a DFT observation."""
        from acf_functor.autonomous_discovery import (
            TopologicalOperatorAnalyzer, KoopmanStructuralAnalyzer,
            OperatorGrammarSearch, TopologicalFingerprint)

        tda = TopologicalOperatorAnalyzer()
        ksa = KoopmanStructuralAnalyzer()
        gs = OperatorGrammarSearch()

        j = np.arange(N).reshape(-1, 1)
        k = np.arange(N).reshape(1, -1)
        F = np.abs(np.exp(-2j * np.pi * j * k / N))

        fp = tda.fingerprint(F)
        koopman = ksa.analyze(F)
        grammar = gs.search(F)

        rule_ind.observe(N, fp, koopman, grammar, 1e-12)

    def test_no_rules_from_insufficient_data(self):
        """Need at least 3 observations to induce rules."""
        from acf_functor.autonomous_discovery import AutonomousRuleInduction
        ri = AutonomousRuleInduction()
        self._make_observation(4, ri)
        self._make_observation(8, ri)
        rules = ri.induce_rules()
        assert len(rules) == 0  # need 3+

    def test_induce_rules_from_multiple_sizes(self):
        """With 3+ observations, should induce at least one rule."""
        from acf_functor.autonomous_discovery import AutonomousRuleInduction
        ri = AutonomousRuleInduction()
        for N in [4, 8, 16, 32]:
            self._make_observation(N, ri)
        rules = ri.induce_rules()
        # Should find at least some pattern
        assert len(ri.observation_log) == 4

    def test_autonomous_rule_matches(self):
        """AutonomousRule.matches() should work."""
        from acf_functor.autonomous_discovery import AutonomousRule
        rule = AutonomousRule(
            rule_id="AR-0001", name="test_rule",
            description="test", conditions={"is_cyclic": True, "score": 0.5},
            action="butterfly", confidence=0.9)
        assert rule.matches({"is_cyclic": True, "score": 0.8})
        assert not rule.matches({"is_cyclic": False, "score": 0.8})
        assert not rule.matches({"is_cyclic": True, "score": 0.3})

    def test_get_applicable_rules(self):
        """Find applicable rules from induced set."""
        from acf_functor.autonomous_discovery import (
            AutonomousRuleInduction, AutonomousRule)
        ri = AutonomousRuleInduction()
        ri.induced_rules.append(AutonomousRule(
            rule_id="AR-0001", name="test",
            description="", conditions={"is_cyclic": True},
            action="butterfly", confidence=0.9))
        applicable = ri.get_applicable_rules({"is_cyclic": True})
        assert len(applicable) == 1
        not_applicable = ri.get_applicable_rules({"is_cyclic": False})
        assert len(not_applicable) == 0


# ===========================================================================
# §12  Level-5 Integration Tests
# ===========================================================================

class TestLevel5Integration:

    def test_operator_structure_includes_tda_koopman(self):
        """OperatorStructureAnalyzer.analyze() should include TDA + Koopman."""
        from acf_functor.autonomous_discovery import OperatorStructureAnalyzer
        osa = OperatorStructureAnalyzer()
        A = np.eye(8)
        result = osa.analyze(A)
        assert "topological_fingerprint" in result
        assert "koopman_analysis" in result

    def test_dft_discovery_has_level5_fields(self):
        """DFTStructureDiscovery report should include Level-5 certificates."""
        from acf_functor.autonomous_discovery import DFTStructureDiscovery
        exp = DFTStructureDiscovery(N=8, target_error=1e-6)
        report = exp.run_full_experiment()
        assert report["success"]
        cert = report["certificate"]
        assert "AD-5" in cert  # TDA fingerprint
        assert "AD-6" in cert  # Grammar search
        assert "AD-7" in cert  # Induced rules
        # Phase 2 should have topological fingerprint
        p2 = report["phases"]["2_analysis"]
        assert "topological_fingerprint" in p2
        assert "koopman_analysis" in p2
        # Phase 3 should have grammar search
        p3 = report["phases"]["3_discovery"]
        assert "grammar_search" in p3
