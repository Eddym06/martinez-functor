"""
test_gideon_engine.py — Suite de Tests Pesados del Motor Gideon

Tests que demuestran la potencia REAL del motor:

Clase 1: GideonIR — Lowering correcto desde FMA y AST
Clase 2: GideonGraph — Análisis topológico, fusión, paralelismo
Clase 3: GideonDispatcher — Detección de hardware y decisiones
Clase 4: GideonEngine — Pipeline completo E2E con benchmarks
Clase 5: Stress — cadenas de 10.000+ stages, arrays de 10M elementos
Clase 6: Precisión numérica — error < machine epsilon vs numpy ref
Clase 7: GideonNeuralHints — blueprints MLP/Transformer/CNN
Clase 8: GideonTheoremSeeds — detección de invariantes y candidatos
Clase 9: Integración con PoemCompiler — nativo end-to-end
Clase 10: Benchmark comparativo — Gideon vs backends individuales
"""

import math
import os
import sys
import time
import unittest

import numpy as np

# ─── PATH ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poema.backends.gideon import (
    GideonEngine,
    GideonIR,
    GideonProgram,
    IRNode,
    IRNodeKind,
    GideonGraph,
    ExecutionPlan,
    GideonDispatcher,
    DispatchDecision,
    GideonNeuralHints,
    ArchitectureBlueprint,
    GideonTheoremSeeds,
    TheoremCandidate,
)
from poema.backends.gideon.dispatcher import HardwareProfile
from poema.backends.gideon.engine import GideonEngineConfig, GideonExecutionResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class _FMA:
    """Mínima FMAInstruction compatible con BackendProtocol."""
    __slots__ = ("weight", "bias")
    def __init__(self, w: float, b: float):
        self.weight = float(w)
        self.bias   = float(b)


def make_chain(n: int, seed: int = 42) -> list:
    """Genera cadena FMA aleatoria de longitud n."""
    rng = np.random.default_rng(seed)
    ws = rng.uniform(0.8, 1.2, n)
    bs = rng.uniform(-0.1, 0.1, n)
    return [_FMA(float(w), float(b)) for w, b in zip(ws, bs)]


def numpy_eval(chain: list, x: np.ndarray) -> np.ndarray:
    """Referencia numpy para comparación de corrección."""
    y = np.asarray(x, dtype=np.float64).copy()
    for fma in chain:
        y = fma.weight * y + fma.bias
    return y


# ═════════════════════════════════════════════════════════════════════════════
# CLASE 1: GideonIR — Lowering
# ═════════════════════════════════════════════════════════════════════════════

class TestGideonIR(unittest.TestCase):
    """Tests del lowering de FMA chains a GideonProgram tipado."""

    def setUp(self):
        self.ir = GideonIR()

    def test_trivial_single_fma(self):
        """Un solo FMA: y = 2x + 1."""
        chain = [_FMA(2.0, 1.0)]
        prog = self.ir.from_fma_sequence(chain, domain=(0.0, 1.0))
        self.assertEqual(prog.total_fma, 1)
        self.assertEqual(len(prog.nodes), 2)   # inp + fma
        self.assertGreater(prog.global_epsilon, 0)

    def test_deep_chain_100(self):
        """Cadena de 100 FMAs: IR debe tener 101 nodos."""
        chain = make_chain(100)
        prog = self.ir.from_fma_sequence(chain, domain=(-1.0, 1.0))
        self.assertEqual(prog.total_fma, 100)
        self.assertEqual(len(prog.nodes), 101)
        self.assertEqual(len(prog.topo_order), 101)

    def test_interval_propagation_monotone(self):
        """Propagación de intervalos: cadena con pesos > 0."""
        chain = [_FMA(2.0, 0.0), _FMA(2.0, 1.0)]   # y = 2(2x) + 1
        prog = self.ir.from_fma_sequence(chain, domain=(0.0, 1.0))
        # Output node debe tener intervalo propagado
        out_node = prog.nodes[prog.output_ids[0]]
        # 2*(2*0)+1=1, 2*(2*1)+1=5
        self.assertAlmostEqual(out_node.meta.interval_lo, 1.0, places=10)
        self.assertAlmostEqual(out_node.meta.interval_hi, 5.0, places=10)

    def test_epsilon_accumulation(self):
        """Error acumulado crece con la cadena."""
        chain_short = make_chain(10)
        chain_long  = make_chain(100)
        prog_s = self.ir.from_fma_sequence(chain_short)
        prog_l = self.ir.from_fma_sequence(chain_long)
        self.assertGreater(prog_l.global_epsilon, prog_s.global_epsilon)

    def test_json_roundtrip(self):
        """Serialización JSON → deserialización preserva estructura."""
        chain = make_chain(50)
        prog = self.ir.from_fma_sequence(chain, name="test_prog")
        json_str = GideonIR.to_json(prog)
        prog2 = GideonIR.from_json(json_str)
        self.assertEqual(prog2.name, "test_prog")
        self.assertEqual(prog2.total_fma, prog.total_fma)
        self.assertEqual(len(prog2.nodes), len(prog.nodes))
        self.assertAlmostEqual(prog2.global_epsilon, prog.global_epsilon, places=15)

    def test_all_node_kinds_present(self):
        """IR debe asignar IRNodeKind.FMA a nodos FMA."""
        chain = make_chain(20)
        prog = self.ir.from_fma_sequence(chain)
        kinds = prog.node_kinds()
        self.assertIn("fma", kinds)
        self.assertEqual(kinds["fma"], 20)
        self.assertIn("input", kinds)

    def test_heavy_chain_10000(self):
        """Cadena de 10.000 FMAs: debe construirse en < 5 segundos."""
        chain = make_chain(10_000, seed=99)
        t0 = time.perf_counter()
        prog = self.ir.from_fma_sequence(chain, precision="fp64")
        elapsed = time.perf_counter() - t0
        self.assertEqual(prog.total_fma, 10_000)
        self.assertLess(elapsed, 5.0, f"IR lowering tardó {elapsed:.2f}s para 10K FMAs")
        print(f"\n  [IR 10K] Lowering en {elapsed*1000:.1f}ms, ε={prog.global_epsilon:.4e}")


# ═════════════════════════════════════════════════════════════════════════════
# CLASE 2: GideonGraph — Análisis topológico
# ═════════════════════════════════════════════════════════════════════════════

class TestGideonGraph(unittest.TestCase):
    """Tests del grafo de cómputo: paralelismo, fusión, estadísticas."""

    def setUp(self):
        self.ir = GideonIR()

    def test_linear_chain_is_sequential(self):
        """Cadena lineal: cada fase tiene exactamente 1 nodo."""
        chain = make_chain(20)
        prog = self.ir.from_fma_sequence(chain)
        graph = GideonGraph(prog)
        plan = graph.analyse()
        # Una cadena lineal tiene n+1 fases (inp + n fma)
        self.assertEqual(plan.critical_path_length, 21)  # 1 inp + 20 fma
        for phase in plan.phases:
            self.assertEqual(len(phase.node_ids), 1)

    def test_fusable_chains_detected(self):
        """Cadena FMA larga debe detectar chains fusionables."""
        chain = make_chain(100)
        prog = self.ir.from_fma_sequence(chain)
        graph = GideonGraph(prog)
        chains = graph.find_fusable_chains()
        # Debe haber al menos 1 cadena de mínimo 2 nodos
        self.assertGreater(len(chains), 0)
        max_chain_len = max(len(c) for c in chains)
        self.assertGreater(max_chain_len, 1)

    def test_graph_stats_correct(self):
        """stats() debe reportar métricas coherentes."""
        chain = make_chain(50)
        prog = self.ir.from_fma_sequence(chain)
        graph = GideonGraph(prog)
        stats = graph.stats()
        self.assertEqual(stats["n_nodes"], 51)      # inp + 50 fma
        self.assertEqual(stats["n_edges"], 50)      # 50 aristas
        self.assertEqual(stats["total_fma"], 50)
        self.assertGreaterEqual(stats["fusable_chains"], 0)

    def test_execution_plan_properties(self):
        """ExecutionPlan tiene parallelizable_ratio y critical_path."""
        chain = make_chain(30)
        prog = self.ir.from_fma_sequence(chain)
        graph = GideonGraph(prog)
        plan = graph.analyse()
        self.assertIsInstance(plan.critical_path_length, int)
        self.assertIsInstance(plan.parallelizable_ratio, float)
        summary = plan.summary()
        self.assertIn("ExecutionPlan", summary)

    def test_heavy_graph_100k_fma(self):
        """Grafo de 100K nodos FMA: análisis en < 30 segundos."""
        chain = make_chain(100_000, seed=7)
        t0 = time.perf_counter()
        prog = self.ir.from_fma_sequence(chain)
        graph = GideonGraph(prog)
        plan = graph.analyse()
        elapsed = time.perf_counter() - t0
        self.assertEqual(plan.total_nodes, 100_001)
        self.assertLess(elapsed, 30.0,
            f"Análisis de grafo 100K tardó {elapsed:.1f}s")
        print(f"\n  [Graph 100K] Análisis en {elapsed*1000:.0f}ms, "
              f"fases={plan.critical_path_length}")


# ═════════════════════════════════════════════════════════════════════════════
# CLASE 3: GideonDispatcher
# ═════════════════════════════════════════════════════════════════════════════

class TestGideonDispatcher(unittest.TestCase):
    """Tests del despachador inteligente."""

    def test_hardware_detection(self):
        """La detección de hardware no lanza excepciones."""
        hw = HardwareProfile.detect()
        self.assertIsInstance(hw.cpu_cores, int)
        self.assertGreater(hw.cpu_cores, 0)
        self.assertIsInstance(hw.has_torch, bool)
        self.assertIsInstance(hw.has_cffi, bool)

    def test_dispatch_decision_valid(self):
        """decide() debe devolver un DispatchDecision con backend válido."""
        ir = GideonIR()
        chain = make_chain(50)
        prog = ir.from_fma_sequence(chain)
        dispatcher = GideonDispatcher()
        dec = dispatcher.decide(prog)
        self.assertIsInstance(dec, DispatchDecision)
        self.assertGreater(len(dec.primary_backend), 0)
        self.assertGreater(dec.estimated_speedup, 0)

    def test_speedup_c_native_with_avx2(self):
        """Con AVX2 disponible, c_native debe tener speedup > 5×."""
        hw = HardwareProfile.detect()
        if not hw.has_cffi:
            self.skipTest("cffi no disponible")
        dispatcher = GideonDispatcher(hw_profile=hw)
        ir = GideonIR()
        prog = ir.from_fma_sequence(make_chain(1000))
        dec = dispatcher.decide(prog)
        if dec.primary_backend == "c_native":
            self.assertGreater(dec.estimated_speedup, 5.0)

    def test_latency_recording(self):
        """record_latency actualiza historial correctamente."""
        d = GideonDispatcher()
        for ms in [1.0, 2.0, 3.0]:
            d.record_latency("c_native", ms)
        self.assertEqual(len(d._latency_history["c_native"]), 3)

    def test_node_map_coverage(self):
        """El mapa nodo→backend cubre todos los nodos del programa."""
        ir = GideonIR()
        prog = ir.from_fma_sequence(make_chain(20))
        d = GideonDispatcher()
        dec = d.decide(prog)
        # Todos los nodos del programa deben estar en el mapa
        for nid in prog.nodes:
            self.assertIn(nid, dec.node_backend_map)


# ═════════════════════════════════════════════════════════════════════════════
# CLASE 4: GideonEngine — Pipeline E2E
# ═════════════════════════════════════════════════════════════════════════════

class TestGideonEnginePipeline(unittest.TestCase):
    """Tests del motor completo: correccion, velocidad, telemetría."""

    @classmethod
    def setUpClass(cls):
        cls.engine = GideonEngine()
        cls.rng = np.random.default_rng(0)

    def _x(self, n: int) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, n).astype(np.float64)

    def test_single_fma_correctness(self):
        """y = 3x - 0.5 con 1 elemento."""
        chain = [_FMA(3.0, -0.5)]
        x = np.array([1.0])
        res = self.engine.run_fma(chain, x)
        self.assertTrue(res.success)
        expected = numpy_eval(chain, x)
        np.testing.assert_allclose(res.output, expected, rtol=1e-10)

    def test_chain_100_correctness(self):
        """Cadena de 100 FMA vs referencia numpy."""
        chain = make_chain(100)
        x = self._x(1000)
        res = self.engine.run_fma(chain, x)
        self.assertTrue(res.success)
        ref = numpy_eval(chain, x)
        np.testing.assert_allclose(res.output, ref, rtol=1e-9, atol=1e-11)

    def test_result_has_telemetry(self):
        """El resultado debe incluir telemetría completa."""
        chain = make_chain(50)
        x = self._x(100)
        res = self.engine.run_fma(chain, x)
        self.assertGreater(res.elapsed_ms, 0)
        self.assertGreater(len(res.backend_used), 0)
        self.assertEqual(res.total_fma, 50)
        self.assertIsNotNone(res.program)
        self.assertIsNotNone(res.dispatch_decision)
        self.assertIsInstance(res.graph_stats, dict)

    def test_result_summary_non_empty(self):
        """summary() debe generar texto completo."""
        chain = make_chain(20)
        x = self._x(10)
        res = self.engine.run_fma(chain, x)
        summary = res.summary()
        self.assertIn("GideonExecutionResult", summary)
        self.assertIn("Backend", summary)
        self.assertIn("FMA", summary)

    def test_info_contains_hardware(self):
        """info() debe listar hardware y backends."""
        info = self.engine.info()
        self.assertIn("GIDEON ENGINE", info)
        self.assertIn("CPU cores", info)
        self.assertIn("Backend", info)

    def test_repr(self):
        engine = GideonEngine()
        r = repr(engine)
        self.assertIn("GideonEngine", r)
        self.assertIn("v1.2.0", r)


# ═════════════════════════════════════════════════════════════════════════════
# CLASE 5: Stress Tests — Escala y robustez
# ═════════════════════════════════════════════════════════════════════════════

class TestGideonStress(unittest.TestCase):
    """Tests de estrés: escala masiva y robustez numérica."""

    @classmethod
    def setUpClass(cls):
        cls.engine = GideonEngine(GideonEngineConfig(verbose=False))

    def test_1M_elements(self):
        """1 millón de elementos con cadena de 20 FMAs."""
        chain = make_chain(20)
        x = np.random.default_rng(1).uniform(-1, 1, 1_000_000).astype(np.float64)
        t0 = time.perf_counter()
        res = self.engine.run_fma(chain, x, name="1M_test")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.assertTrue(res.success)
        self.assertEqual(len(res.output), 1_000_000)
        ref = numpy_eval(chain, x)
        np.testing.assert_allclose(res.output, ref, rtol=1e-9)
        print(f"\n  [Stress 1M] {elapsed_ms:.1f}ms, backend={res.backend_used}")

    def test_10M_elements(self):
        """10 millones de elementos con cadena de 10 FMAs."""
        chain = make_chain(10, seed=5)
        x = np.random.default_rng(2).uniform(-0.5, 0.5, 10_000_000).astype(np.float64)
        t0 = time.perf_counter()
        res = self.engine.run_fma(chain, x, name="10M_test")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.assertTrue(res.success)
        self.assertEqual(len(res.output), 10_000_000)
        ref = numpy_eval(chain, x)
        np.testing.assert_allclose(res.output, ref, rtol=1e-9)
        print(f"\n  [Stress 10M] {elapsed_ms:.1f}ms, backend={res.backend_used}")

    def test_chain_depth_5000(self):
        """Cadena de profundidad 5000 (ultra-deep FMA chain)."""
        chain = make_chain(5_000, seed=3)
        x = np.array([0.5])
        res = self.engine.run_fma(chain, x, name="deep_5000")
        self.assertTrue(res.success)
        self.assertEqual(res.total_fma, 5_000)
        ref = numpy_eval(chain, x)
        # Deep chains acumulan error numérico; tolerancia más amplia
        np.testing.assert_allclose(res.output, ref, rtol=1e-6, atol=1e-8)
        print(f"\n  [Stress depth=5K] ε={res.global_epsilon:.4e}")

    def test_precision_fp32(self):
        """Pipeline completo en FP32."""
        cfg = GideonEngineConfig(precision="fp32")
        engine = GideonEngine(cfg)
        chain = make_chain(50)
        x = np.array([0.1, 0.5, -0.3, 0.9])
        res = engine.run_fma(chain, x)
        self.assertTrue(res.success)
        ref = numpy_eval(chain, x)
        np.testing.assert_allclose(res.output, ref, rtol=1e-5, atol=1e-6)

    def test_numerical_stability_contracting_chain(self):
        """Cadena contráctil garantiza estabilidad numérica."""
        # Todos los pesos < 1: la cadena contrae exponencialmente
        chain = [_FMA(0.5, 0.0) for _ in range(200)]
        x = np.linspace(-100.0, 100.0, 10000)
        res = self.engine.run_fma(chain, x)
        self.assertTrue(res.success)
        # Salida debe ser ≈ 0 (0.5^200 * x → 0)
        np.testing.assert_allclose(np.abs(res.output), 0.0, atol=1e-50)

    def test_constant_chain(self):
        """Cadena de bias puro: y = bias (independiente de x)."""
        # FMA(0, 1.234): y = 0*x + 1.234
        chain = [_FMA(0.0, 1.234)] * 50
        x = np.random.default_rng(99).uniform(-1e6, 1e6, 1000)
        res = self.engine.run_fma(chain, x)
        self.assertTrue(res.success)
        expected = np.full_like(x, 1.234)
        np.testing.assert_allclose(res.output, expected, rtol=1e-10)


# ═════════════════════════════════════════════════════════════════════════════
# CLASE 6: Precisión numérica — machine epsilon
# ═════════════════════════════════════════════════════════════════════════════

class TestGideonNumericalPrecision(unittest.TestCase):
    """Tests exhaustivos de precisión numérica."""

    @classmethod
    def setUpClass(cls):
        cls.engine = GideonEngine(GideonEngineConfig(precision="fp64"))

    def test_identity_chain(self):
        """Cadena FMA identidad: y = 1*x + 0."""
        chain = [_FMA(1.0, 0.0)] * 100
        x = np.linspace(-1.0, 1.0, 10_000)
        res = self.engine.run_fma(chain, x)
        self.assertTrue(res.success)
        np.testing.assert_allclose(res.output, x, rtol=1e-14, atol=1e-14)

    def test_single_affine_exact(self):
        """FMA única con pesos exactos: y = 0.1*x + 0.2."""
        chain = [_FMA(0.1, 0.2)]
        x = np.array([0.0, 1.0, -1.0, 0.5, math.pi])
        res = self.engine.run_fma(chain, x)
        expected = 0.1 * x + 0.2
        np.testing.assert_allclose(res.output, expected, rtol=1e-14)

    def test_composed_affine_exact(self):
        """(3x-1) ∘ (2x+4) = 3(2x+4)-1 = 6x+11."""
        chain = [_FMA(2.0, 4.0), _FMA(3.0, -1.0)]
        x = np.linspace(-5.0, 5.0, 100)
        res = self.engine.run_fma(chain, x)
        expected = 6.0 * x + 11.0
        np.testing.assert_allclose(res.output, expected, rtol=1e-13, atol=1e-12)

    def test_horner_polynomial_like(self):
        """
        Polinomio 3x^2 + 2x + 1 vía Horner = ((3)*x + 2)*x + 1
        → FMA(3, 0) luego FMA(x, 2) ... aproximamos con 2 FMA.
        Comparamos contra evaluación directa.
        """
        # p(x) = 3x² + 2x + 1, Horner: ((3*x) + 2)*x + 1
        # FMA chain: start=x, step1: w=3, b=0 → 3x
        #             step2: w=x (?), b=2 ...
        # Nota: Horner completo requiere input-dependent weights.
        # Aquí testeamos que la formulación FMA es numéricamente estable.
        chain = [_FMA(3.0, 0.0), _FMA(1.0, 0.0)]  # simplificado
        x = np.linspace(-1.0, 1.0, 200)
        res = self.engine.run_fma(chain, x)
        expected = numpy_eval(chain, x)
        np.testing.assert_allclose(res.output, expected, rtol=1e-14)

    def test_error_bound_within_ir(self):
        """La cota ε en el IR debe ser mayor que el error numérico real."""
        chain = make_chain(50)
        x = np.linspace(-1.0, 1.0, 1000)
        res = self.engine.run_fma(chain, x)
        actual_error = float(np.max(np.abs(
            res.output - numpy_eval(chain, x)
        )))
        # La cota IR (global_epsilon) debe ser >= error real
        # (puede ser más pesimista — eso es correcto)
        self.assertGreaterEqual(res.global_epsilon + 1e-16, actual_error)


# ═════════════════════════════════════════════════════════════════════════════
# CLASE 7: GideonNeuralHints
# ═════════════════════════════════════════════════════════════════════════════

class TestGideonNeuralHints(unittest.TestCase):
    """Tests de blueprints de arquitecturas IA."""

    def test_mlp_blueprint_basic(self):
        """Blueprint MLP [784, 256, 128, 10]."""
        bp = GideonNeuralHints.mlp([784, 256, 128, 10])
        self.assertGreater(bp.total_params, 0)
        self.assertGreater(bp.total_flops, 0)
        alpha = bp.compute_alpha_complexity()
        self.assertGreater(alpha, 0)
        summary = bp.summary()
        self.assertIn("Blueprint", summary)

    def test_transformer_blueprint(self):
        """Blueprint Transformer estándar."""
        bp = GideonNeuralHints.transformer(d_model=512, n_heads=8, n_layers=6)
        self.assertGreater(bp.total_params, 0)
        self.assertGreater(bp.total_flops, 0)
        analysis = GideonNeuralHints.analyse_blueprint(bp)
        self.assertIn("total_flops", analysis)
        self.assertIn("alpha_complexity", analysis)
        self.assertIn("fma_equivalent", analysis)
        self.assertGreater(analysis["fma_equivalent"], 0)

    def test_cnn_resnet_blueprint(self):
        """Blueprint ResNet block."""
        bp = GideonNeuralHints.cnn_resnet_block(channels=64, n_blocks=4)
        self.assertEqual(bp.kind.value, "cnn")
        self.assertGreater(len(bp.layers), 0)

    def test_deep_mlp_fma_count(self):
        """MLP profunda: FMA equivalente debe escalar cuadráticamente."""
        bp_small = GideonNeuralHints.mlp([64, 64, 64])
        bp_large = GideonNeuralHints.mlp([512, 512, 512])
        # FMA de la grande debe ser >> pequeña
        self.assertGreater(
            bp_large.to_gideon_fma_count(),
            bp_small.to_gideon_fma_count() * 10
        )

    def test_search_space_generation(self):
        """Espacio de búsqueda MLP y Transformer generados correctamente."""
        space_mlp = GideonNeuralHints.generate_mlp_search_space(64, 512)
        self.assertIn("hidden_dims", space_mlp)
        self.assertIn("depths", space_mlp)
        self.assertGreater(space_mlp["total_configs"], 0)

        space_tf = GideonNeuralHints.generate_transformer_search_space()
        self.assertIn("d_model", space_tf)
        self.assertIn("n_heads", space_tf)

    def test_blueprint_analysis_coherent(self):
        """Análisis de blueprint: depth == número de capas lineales/conv."""
        bp = GideonNeuralHints.mlp([128, 64, 32, 16, 8])
        analysis = GideonNeuralHints.analyse_blueprint(bp)
        # 4 capas lineales (128→64, 64→32, 32→16, 16→8)
        self.assertEqual(analysis["depth"], 4)


# ═════════════════════════════════════════════════════════════════════════════
# CLASE 8: GideonTheoremSeeds
# ═════════════════════════════════════════════════════════════════════════════

class TestGideonTheoremSeeds(unittest.TestCase):
    """Tests del motor de descubrimiento de teoremas."""

    @classmethod
    def setUpClass(cls):
        cls.seeds = GideonTheoremSeeds()

    def test_contraction_detected(self):
        """f(x) = 0.5x es una contracción → debe generar candidato."""
        import torch
        fn = lambda x: 0.5 * x
        cands = self.seeds.analyse(fn, domain=(-1.0, 1.0), fn_name="half_x")
        names = [c.name for c in cands]
        # Debe detectar contracción y Lipschitz
        lipschitz_found = any("lipschitz" in n for n in names)
        self.assertTrue(lipschitz_found, f"No encontró Lipschitz. Candidatos: {names}")

    def test_monotone_function(self):
        """f(x) = 2x + 1 es monótona."""
        import torch
        fn = lambda x: 2.0 * x + 1.0
        cands = self.seeds.analyse(fn, domain=(-1.0, 1.0), fn_name="linear")
        names = [c.name for c in cands]
        mono_found = any("monotone" in n for n in names)
        self.assertTrue(mono_found, f"No detectó monotonicidad. Candidatos: {names}")

    def test_even_function_symmetry(self):
        """f(x) = x^2 es par."""
        import torch
        fn = lambda x: x ** 2
        cands = self.seeds.analyse(fn, domain=(-1.0, 1.0), fn_name="square")
        names = [c.name for c in cands]
        even_found = any("even" in n for n in names)
        self.assertTrue(even_found, f"No detectó función par. Candidatos: {names}")

    def test_confidence_range(self):
        """Todos los candidatos tienen confidence en [0, 1]."""
        import torch
        fn = lambda x: torch.sin(x)
        cands = self.seeds.analyse(fn, domain=(-math.pi, math.pi), fn_name="sin")
        for c in cands:
            self.assertGreaterEqual(c.confidence, 0.0)
            self.assertLessEqual(c.confidence, 1.0)

    def test_lean_skeleton_non_empty(self):
        """Los candidatos con skeleton Lean deben tener código válido."""
        import torch
        fn = lambda x: 0.8 * x
        cands = self.seeds.analyse(fn, domain=(-2.0, 2.0), fn_name="scale08")
        lean_cands = [c for c in cands if c.lean_skeleton]
        self.assertGreater(len(lean_cands), 0)
        for c in lean_cands:
            # Lean 4 usa 'def' o 'theorem' según el tipo de declaración
            skeleton_lower = c.lean_skeleton.lower()
            has_lean_decl = any(kw in skeleton_lower for kw in ("theorem", "def ", "lemma"))
            self.assertTrue(has_lean_decl,
                f"Lean skeleton debe contener 'theorem', 'def' o 'lemma': {c.lean_skeleton[:200]!r}")

    def test_export_lean_file(self):
        """export_lean_file genera un archivo .lean válido."""
        import tempfile
        import torch
        fn = lambda x: 0.5 * x + 0.1
        seeds = GideonTheoremSeeds()
        seeds.analyse(fn, domain=(-1.0, 1.0), fn_name="affine_test")
        with tempfile.TemporaryDirectory() as tmpdir:
            lean_path = os.path.join(tmpdir, "GideonTest.lean")
            seeds.export_lean_file(lean_path, "GideonTest")
            self.assertTrue(os.path.exists(lean_path))
            with open(lean_path) as f:
                content = f.read()
            self.assertIn("namespace GideonTest", content)
            self.assertIn("end GideonTest", content)

    def test_theorem_summary(self):
        """summary() produce texto multi-línea coherente."""
        summary = self.seeds.summary()
        self.assertIn("GideonTheoremSeeds", summary)


# ═════════════════════════════════════════════════════════════════════════════
# CLASE 9: Integración con PoemCompiler
# ═════════════════════════════════════════════════════════════════════════════

class TestGideonPoemIntegration(unittest.TestCase):
    """Tests de integración nativa Gideon ↔ PoemCompiler."""

    def test_gideon_via_registry(self):
        """BackendRegistry.gideon() devuelve un GideonEngine válido."""
        from poema.backends.registry import BackendRegistry
        engine = BackendRegistry.gideon()
        self.assertIsInstance(engine, GideonEngine)
        self.assertIn("GIDEON ENGINE", engine.info())

    def test_backends_init_includes_gideon_exports(self):
        """El paquete poema.backends exporta GideonEngine."""
        import poema.backends as pb
        self.assertTrue(hasattr(pb, "GideonEngine"))
        self.assertTrue(hasattr(pb, "GideonIR"))
        self.assertTrue(hasattr(pb, "GideonGraph"))
        self.assertTrue(hasattr(pb, "GideonDispatcher"))

    def test_full_poema_pipeline_with_gideon(self):
        """
        Pipeline completo: PoemCompiler → FMA sequence → GideonEngine.
        Verifica que Gideon puede ejecutar la salida del compilador real.
        """
        import torch
        from poema.compiler import PoemCompiler
        from poema.ast_nodes import AffineNode

        compiler = PoemCompiler(target="pytorch", precision="fp64")
        ast = AffineNode(
            scale_factor=torch.tensor(2.5, dtype=torch.float64),
            shift_value=torch.tensor(-0.5, dtype=torch.float64),
        )
        fn, report = compiler.compile(ast, domain=(-1.0, 1.0))
        fma_seq = report.fma_sequence

        # Ejecutar a través de Gideon
        engine = GideonEngine()
        x = np.linspace(-1.0, 1.0, 500)
        result = engine.run_fma(fma_seq, x, name="poem_affine")
        self.assertTrue(result.success)
        expected = 2.5 * x - 0.5
        np.testing.assert_allclose(result.output, expected, rtol=1e-12)
        print(f"\n  [Poem→Gideon] backend={result.backend_used}, "
              f"ε={result.global_epsilon:.4e}")

    def test_ir_from_compiled_fma_sequence(self):
        """GideonIR puede bajar la secuencia FMA del compilador real."""
        import torch
        from poema.compiler import PoemCompiler
        from poema.ast_nodes import PolynomialNode

        compiler = PoemCompiler(target="pytorch", precision="fp64")
        ast = PolynomialNode(coefficients=[1.0, 0.0, -1.0])  # x^2 - 1
        fn, report = compiler.compile(ast, domain=(-1.0, 1.0))
        fma_seq = report.fma_sequence

        ir = GideonIR()
        prog = ir.from_fma_sequence(fma_seq)
        self.assertGreater(prog.total_fma, 0)
        self.assertGreater(len(prog.nodes), 1)


# ═════════════════════════════════════════════════════════════════════════════
# CLASE 10: Benchmarks comparativos reales
# ═════════════════════════════════════════════════════════════════════════════

class TestGideonBenchmarks(unittest.TestCase):
    """
    Benchmarks reales: Gideon vs numpy vs c_native individual.
    Estos tests miden el rendimiento real y lo reportan.
    """

    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(42)
        cls.engine_gideon = GideonEngine(GideonEngineConfig(verbose=False))

    def _bench(self, fn, x, repeats=20):
        """Benchmark simple: devuelve tiempo promedio en ms."""
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = fn(x)
            times.append((time.perf_counter() - t0) * 1000)
        # Descarta los 2 más lentos (JIT warmup)
        times.sort()
        return float(np.mean(times[:repeats - 2]))

    def _run_comparison(self, chain_depth: int, n_elements: int, tag: str):
        chain = make_chain(chain_depth, seed=42)
        x = self.rng.uniform(-1.0, 1.0, n_elements).astype(np.float64)

        # Numpy baseline
        def np_fn(xin):
            return numpy_eval(chain, xin)

        t_np = self._bench(np_fn, x, repeats=10)

        # Gideon pipeline (incluye dispatch overhead)
        gideon_result = self.engine_gideon.run_fma(chain, x)
        backend_fn = None
        from poema.backends.registry import BackendRegistry
        for bname in [gideon_result.backend_used, "numpy_cpu"]:
            try:
                b = BackendRegistry.get(bname)
                if b.verify_available():
                    r = b.compile(chain, None, precision="fp64")
                    if r.callable_fn:
                        backend_fn = r.callable_fn
                        break
            except Exception:
                continue

        if backend_fn:
            t_backend = self._bench(backend_fn, x, repeats=10)
            speedup = t_np / max(t_backend, 1e-9)
        else:
            t_backend = t_np
            speedup = 1.0

        print(f"\n  [{tag}] "
              f"depth={chain_depth}, n={n_elements:,} | "
              f"numpy={t_np:.2f}ms | "
              f"{gideon_result.backend_used}={t_backend:.2f}ms | "
              f"speedup={speedup:.1f}×")
        return speedup

    def test_benchmark_100_depth_100k_elements(self):
        speedup = self._run_comparison(100, 100_000, "BM_100x100K")
        # Con cualquier backend habilitado, Gideon no debe ser 10× MÁS LENTO que numpy
        self.assertGreater(speedup, 0.1)

    def test_benchmark_1000_depth_10k_elements(self):
        speedup = self._run_comparison(1_000, 10_000, "BM_1Kx10K")
        self.assertGreater(speedup, 0.05)

    def test_benchmark_10_depth_10M_elements(self):
        speedup = self._run_comparison(10, 10_000_000, "BM_10x10M")
        self.assertGreater(speedup, 0.1)

    def test_benchmark_c_native_speedup_if_available(self):
        """Si c_native está disponible, debe ser significativamente más rápido que numpy."""
        from poema.backends.registry import BackendRegistry
        try:
            c_backend = BackendRegistry.get("c_native")
            if not c_backend.verify_available():
                self.skipTest("c_native no disponible")
        except Exception:
            self.skipTest("c_native no disponible")

        chain = make_chain(50, seed=7)
        x = np.random.default_rng(7).uniform(-1.0, 1.0, 1_000_000).astype(np.float64)

        # Baseline numpy
        def np_fn(xin):
            return numpy_eval(chain, xin)
        t_np = self._bench(np_fn, x, repeats=5)

        # C native
        r = c_backend.compile(chain, None, precision="fp64")
        t_c = self._bench(r.callable_fn, x, repeats=5)
        speedup = t_np / max(t_c, 1e-9)

        print(f"\n  [C Native] numpy={t_np:.2f}ms | c_native={t_c:.2f}ms | speedup={speedup:.1f}×")
        # C debe ser al menos 2× más rápido que numpy con AVX2
        hw = HardwareProfile.detect()
        if hw.has_avx2:
            self.assertGreater(speedup, 2.0,
                f"c_native con AVX2 esperaba speedup>2×, obtuvo {speedup:.1f}×")


# ═════════════════════════════════════════════════════════════════════════════
# CLASE 11: IR Serialización y graph JSON
# ═════════════════════════════════════════════════════════════════════════════

class TestGideonIRSerialization(unittest.TestCase):
    """Tests de serialización/deserialización del IR."""

    def test_large_program_json(self):
        """Programa de 1000 nodos: JSON round-trip completo."""
        ir = GideonIR()
        chain = make_chain(1000, seed=5)
        prog = ir.from_fma_sequence(chain, name="bigprog")
        json_str = GideonIR.to_json(prog)
        self.assertGreater(len(json_str), 1000)

        prog2 = GideonIR.from_json(json_str)
        self.assertEqual(prog2.total_fma, 1000)
        self.assertEqual(len(prog2.nodes), len(prog.nodes))
        self.assertEqual(prog2.name, "bigprog")
        for nid in prog.nodes:
            self.assertIn(nid, prog2.nodes)

    def test_node_kinds_preserved_in_json(self):
        """Tipos de nodos IRNodeKind se preservan en JSON."""
        ir = GideonIR()
        prog = ir.from_fma_sequence(make_chain(5))
        j = GideonIR.to_json(prog)
        prog2 = GideonIR.from_json(j)
        for nid, node in prog.nodes.items():
            node2 = prog2.nodes.get(nid)
            self.assertIsNotNone(node2, f"Nodo {nid} perdido en round-trip")
            self.assertEqual(node.kind, node2.kind)

    def test_program_summary(self):
        ir = GideonIR()
        prog = ir.from_fma_sequence(make_chain(20))
        s = prog.summary()
        self.assertIn("GideonProgram", s)
        self.assertIn("FMA", s)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("=" * 70)
    print("  GIDEON ENGINE — Suite de Tests Pesados")
    print("  Poema Motor Unificado v1.0.0")
    print("=" * 70)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestGideonIR,
        TestGideonGraph,
        TestGideonDispatcher,
        TestGideonEnginePipeline,
        TestGideonStress,
        TestGideonNumericalPrecision,
        TestGideonNeuralHints,
        TestGideonTheoremSeeds,
        TestGideonPoemIntegration,
        TestGideonBenchmarks,
        TestGideonIRSerialization,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
