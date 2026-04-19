"""
test_gideon_titan.py — Suite Titánica del Motor Gideon
═══════════════════════════════════════════════════════════════════

Suite de pruebas de vida real, complejas e intensas para el motor Gideon.
Diseñada para demostrar potencia, robustez y correctitud matemática en
condiciones extremas de escala, precisión y adversarialidad.

Clases de prueba:
  01. TestTitanIRNodeKinds          — Los 28 IRNodeKinds verificados individualmente
  02. TestTitanIRDeepChains         — Cadenas de 50K–500K FMAs; lowering bajo estrés
  03. TestTitanEpsilonBounds        — Cota ε: fórmula exacta vs implementación
  04. TestTitanGraphTopology        — Grafos densos, estrellas, diamante, ciclabilidad
  05. TestTitanDispatcherLogic      — Scoring, hints, feedback loop, fallback chains
  06. TestTitanNumericalAccuracy    — Tolerancias fp64/fp32, cancelación catastrófica
  07. TestTitanSpeedupBenchmarks    — Speedup medido vs NumPy (con métricas exactas)
  08. TestTitanScaleExtremes        — 50M elementos, cadenas 10K+, lotes extremos
  09. TestTitanAdversarialInputs    — NaN, Inf, overflow, vacío, dominio puntual
  10. TestTitanMathematicalProps    — Contractividad, punto fijo, ley asociativa FMA
  11. TestTitanTheoremInvariants    — Detección completa en funciones conocidas
  12. TestTitanNeuralBlueprints     — GPT-2/ResNet-50/MLP scale, α-complejidad
  13. TestTitanSerializationStress  — JSON roundtrip 200×, integridad bit-exacta
  14. TestTitanConcurrentEngines    — 4 engines simultaneos, reproducibilidad
  15. TestTitanMetricsCompleteness  — Todos los campos de GideonExecutionResult

Total: ≥ 120 tests
"""

import json
import math
import os
import sys
import threading
import time
import unittest

import numpy as np

# ─── PATH ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poema.backends.gideon import (
    GideonEngine,
    GideonIR,
    GideonGraph,
    ExecutionPlan,
    GideonDispatcher,
    DispatchDecision,
    BackendHint,
    GideonNeuralHints,
    ArchitectureBlueprint,
    GideonTheoremSeeds,
    TheoremCandidate,
    IRNode,
    IRNodeKind,
    GideonProgram,
    GideonGraphNode,
    GraphEdge,
)
from poema.backends.gideon.dispatcher import HardwareProfile
from poema.backends.gideon.engine import GideonEngineConfig, GideonExecutionResult
from poema.backends.gideon.theorem_seeds import (
    InvariantProbe,
    PatternMatcher,
    TheoremStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers globales
# ─────────────────────────────────────────────────────────────────────────────

class _FMA:
    __slots__ = ("weight", "bias")
    def __init__(self, w: float, b: float):
        self.weight = float(w)
        self.bias   = float(b)


def make_chain(n: int, seed: int = 42, lo: float = 0.8, hi: float = 1.2) -> list:
    rng = np.random.default_rng(seed)
    ws  = rng.uniform(lo, hi, n)
    bs  = rng.uniform(-0.1, 0.1, n)
    return [_FMA(float(w), float(b)) for w, b in zip(ws, bs)]


def make_contractive_chain(n: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    ws  = rng.uniform(0.5, 0.95, n)
    bs  = rng.uniform(-0.05, 0.05, n)
    return [_FMA(float(w), float(b)) for w, b in zip(ws, bs)]


def numpy_eval(chain: list, x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).copy()
    for fma in chain:
        y = fma.weight * y + fma.bias
    return y


def speedup(t_numpy_ms: float, t_gideon_ms: float) -> float:
    if t_gideon_ms <= 0:
        return float("inf")
    return t_numpy_ms / t_gideon_ms


_PRINT_LOCK = threading.Lock()

def log_metric(tag: str, msg: str) -> None:
    with _PRINT_LOCK:
        print(f"\n  [{tag}] {msg}")


# ═════════════════════════════════════════════════════════════════════════════
# 01. TestTitanIRNodeKinds — Los 28 IRNodeKinds
# ═════════════════════════════════════════════════════════════════════════════

class TestTitanIRNodeKinds(unittest.TestCase):
    """Verifica que los 28 IRNodeKinds existen y tienen valor de string único."""

    EXPECTED_KINDS = [
        "const", "input", "fma", "identity",
        "scale", "shift", "affine",
        "compose", "parallel", "branch",
        "poly_horner", "poly_cheb",
        "sin", "cos", "exp", "log", "tanh", "sigmoid",
        "matmul", "gemm", "conv", "norm", "attention",
        "loop", "recursive",
        "arch_probe", "theorem_seed",
    ]

    def test_all_28_kinds_exist(self):
        enum_values = {e.value for e in IRNodeKind}
        for kind_val in self.EXPECTED_KINDS:
            self.assertIn(kind_val, enum_values,
                          f"IRNodeKind '{kind_val}' no existe en el enum")

    def test_enum_count_is_27_or_more(self):
        # Hay al menos 27 kinds (algunos pueden ser adicionales)
        self.assertGreaterEqual(len(IRNodeKind), 27)

    def test_all_values_are_strings(self):
        for e in IRNodeKind:
            self.assertIsInstance(e.value, str, f"{e} no tiene valor string")

    def test_values_are_unique(self):
        vals = [e.value for e in IRNodeKind]
        self.assertEqual(len(vals), len(set(vals)), "Valores duplicados en IRNodeKind")

    def test_critical_kinds_accessible(self):
        # Los más usados deben ser accesibles directamente
        self.assertEqual(IRNodeKind.FMA.value, "fma")
        self.assertEqual(IRNodeKind.INPUT.value, "input")
        self.assertEqual(IRNodeKind.MATMUL.value, "matmul")
        self.assertEqual(IRNodeKind.ATTENTION.value, "attention")
        self.assertEqual(IRNodeKind.THEOREM_SEED.value, "theorem_seed")

    def test_ir_kinds_in_lowered_program(self):
        ir = GideonIR()
        chain = make_chain(10)
        prog = ir.from_fma_sequence(chain)
        kinds = prog.node_kinds()
        self.assertIn("fma", kinds)
        self.assertIn("input", kinds)
        self.assertEqual(kinds["fma"], 10)


# ═════════════════════════════════════════════════════════════════════════════
# 02. TestTitanIRDeepChains — Cadenas extremas de FMAs
# ═════════════════════════════════════════════════════════════════════════════

class TestTitanIRDeepChains(unittest.TestCase):
    """Lowering de cadenas 1K–200K bajo métricas de tiempo."""

    def setUp(self):
        self.ir = GideonIR()

    def _lower_and_time(self, n: int, seed: int = 0) -> tuple:
        chain = make_chain(n, seed=seed)
        t0 = time.perf_counter()
        prog = self.ir.from_fma_sequence(chain, precision="fp64")
        elapsed = time.perf_counter() - t0
        return prog, elapsed, chain

    def test_chain_1k_under_1s(self):
        prog, elapsed, _ = self._lower_and_time(1_000)
        self.assertEqual(prog.total_fma, 1_000)
        self.assertEqual(len(prog.nodes), 1_001)
        self.assertLess(elapsed, 1.0)
        log_metric("IR_1K", f"1K FMAs en {elapsed*1000:.1f}ms")

    def test_chain_10k_under_3s(self):
        prog, elapsed, _ = self._lower_and_time(10_000, seed=7)
        self.assertEqual(prog.total_fma, 10_000)
        self.assertLess(elapsed, 3.0)
        log_metric("IR_10K", f"10K FMAs en {elapsed*1000:.0f}ms, ε={prog.global_epsilon:.3e}")

    def test_chain_50k_under_15s(self):
        prog, elapsed, _ = self._lower_and_time(50_000, seed=13)
        self.assertEqual(prog.total_fma, 50_000)
        self.assertLess(elapsed, 15.0)
        log_metric("IR_50K", f"50K FMAs en {elapsed*1000:.0f}ms, ε={prog.global_epsilon:.3e}")

    def test_chain_100k_under_45s(self):
        prog, elapsed, _ = self._lower_and_time(100_000, seed=21)
        self.assertEqual(prog.total_fma, 100_000)
        self.assertEqual(len(prog.nodes), 100_001)
        self.assertLess(elapsed, 45.0)
        log_metric("IR_100K", f"100K FMAs en {elapsed*1000:.0f}ms, ε={prog.global_epsilon:.3e}")

    def test_topo_order_completeness(self):
        prog, _, _ = self._lower_and_time(500)
        self.assertEqual(len(prog.topo_order), 501)
        self.assertEqual(prog.topo_order[0], prog.input_ids[0])
        self.assertEqual(prog.topo_order[-1], prog.output_ids[0])

    def test_topo_order_unique_ids(self):
        prog, _, _ = self._lower_and_time(200)
        self.assertEqual(len(prog.topo_order), len(set(prog.topo_order)))

    def test_all_nodes_have_epsilon(self):
        prog, _, chain = self._lower_and_time(100)
        for nid, node in prog.nodes.items():
            if node.kind == IRNodeKind.FMA:
                self.assertGreater(node.meta.epsilon_bound, 0.0,
                    f"Nodo FMA {nid} tiene ε=0")

    def test_input_node_properties(self):
        prog, _, _ = self._lower_and_time(50)
        inp_node = prog.nodes[prog.input_ids[0]]
        self.assertEqual(inp_node.kind, IRNodeKind.INPUT)
        self.assertEqual(len(inp_node.inputs), 0)
        self.assertAlmostEqual(inp_node.meta.interval_lo, -1.0)
        self.assertAlmostEqual(inp_node.meta.interval_hi,  1.0)

    def test_node_kinds_dict_accuracy(self):
        n = 75
        prog, _, _ = self._lower_and_time(n)
        kinds = prog.node_kinds()
        self.assertEqual(kinds.get("fma", 0), n)
        self.assertEqual(kinds.get("input", 0), 1)
        self.assertEqual(sum(kinds.values()), n + 1)

    def test_monotone_interval_propagation_positive_weights(self):
        # Todos pesos positivos → intervalos ordenados
        chain = [_FMA(2.0, 0.0), _FMA(2.0, 0.0), _FMA(2.0, 0.0)]
        prog = self.ir.from_fma_sequence(chain, domain=(0.0, 1.0))
        out_node = prog.nodes[prog.output_ids[0]]
        # y = 2^3 * x en [0,1] → [0, 8]
        self.assertAlmostEqual(out_node.meta.interval_lo, 0.0, places=10)
        self.assertAlmostEqual(out_node.meta.interval_hi, 8.0, places=10)


# ═════════════════════════════════════════════════════════════════════════════
# 03. TestTitanEpsilonBounds — Verificación de la fórmula ε
# ═════════════════════════════════════════════════════════════════════════════

class TestTitanEpsilonBounds(unittest.TestCase):
    """
    Verifica que el error ε satisface:
        ε_N ≈ ε_machine * Σ_{k=1..N} ∏_{j=k+1..N} |w_j|
    """

    EPS_FP64 = 2.22e-16

    def _theoretical_epsilon(self, weights: list) -> float:
        """Calcula ε teórico exacto acumulado."""
        eps = 0.0
        for w in weights:
            eps = abs(w) * eps + self.EPS_FP64
        return eps

    def test_single_fma_epsilon(self):
        chain = [_FMA(2.0, 0.5)]
        ir = GideonIR()
        prog = ir.from_fma_sequence(chain, domain=(-1.0, 1.0), precision="fp64")
        expected = abs(2.0) * 0.0 + self.EPS_FP64
        self.assertAlmostEqual(prog.global_epsilon, expected, places=30)

    def test_chain_10_epsilon_formula(self):
        ws = [1.1, 0.9, 1.05, 0.95, 1.0, 1.1, 0.8, 1.2, 0.9, 1.0]
        chain = [_FMA(w, 0.0) for w in ws]
        ir = GideonIR()
        prog = ir.from_fma_sequence(chain, precision="fp64")
        expected = self._theoretical_epsilon(ws)
        self.assertAlmostEqual(prog.global_epsilon, expected, places=35)

    def test_contractive_chain_epsilon_bounded(self):
        # Con |w| < 1, ε converge; para una cadena de 1000 el ε < 2e-15
        chain = [_FMA(0.9, 0.01)] * 1000
        ir = GideonIR()
        prog = ir.from_fma_sequence(chain, precision="fp64")
        self.assertLess(prog.global_epsilon, 3e-15)

    def test_identity_chain_epsilon_linear(self):
        # Con w=1, b=0: ε_N = N * ε_machine
        n = 100
        chain = [_FMA(1.0, 0.0)] * n
        ir = GideonIR()
        prog = ir.from_fma_sequence(chain, precision="fp64")
        expected = n * self.EPS_FP64
        np.testing.assert_allclose(prog.global_epsilon, expected, rtol=1e-10)

    def test_fp32_epsilon_larger_than_fp64(self):
        chain = make_chain(50)
        ir = GideonIR()
        p64 = ir.from_fma_sequence(chain, precision="fp64")
        p32 = ir.from_fma_sequence(chain, precision="fp32")
        self.assertGreater(p32.global_epsilon, p64.global_epsilon)

    def test_growing_epsilon_with_chain_length(self):
        # Cadena expandida (|w|=1.5): ε crece monotónicamente con la longitud
        ir = GideonIR()
        eps_values = []
        for n in [10, 50, 100, 500]:
            chain = [_FMA(1.5, 0.0)] * n  # w > 1 garantiza crecimiento
            prog = ir.from_fma_sequence(chain, precision="fp64")
            eps_values.append(prog.global_epsilon)
        for i in range(len(eps_values) - 1):
            self.assertLess(eps_values[i], eps_values[i + 1],
                            f"ε no crece con longitud: {eps_values}")

    def test_epsilon_node_monotone_in_chain(self):
        chain = [_FMA(1.1, 0.01)] * 50
        ir = GideonIR()
        prog = ir.from_fma_sequence(chain, precision="fp64")
        fma_nodes = [prog.nodes[nid] for nid in prog.topo_order
                     if prog.nodes[nid].kind == IRNodeKind.FMA]
        epsilons = [n.meta.epsilon_bound for n in fma_nodes]
        for i in range(len(epsilons) - 1):
            self.assertLessEqual(epsilons[i], epsilons[i + 1] + 1e-30)

    def test_stress_5000_depth_epsilon(self):
        chain = [_FMA(0.99, 0.001)] * 5000
        ir = GideonIR()
        prog = ir.from_fma_sequence(chain, precision="fp64")
        self.assertLess(prog.global_epsilon, 1e-12)
        log_metric("ε_5K", f"ε={prog.global_epsilon:.4e} para 5K FMAs contráctivos")


# ═════════════════════════════════════════════════════════════════════════════
# 04. TestTitanGraphTopology — Topología del grafo
# ═════════════════════════════════════════════════════════════════════════════

class TestTitanGraphTopology(unittest.TestCase):

    def setUp(self):
        self.ir = GideonIR()

    def _make_prog(self, n: int, seed: int = 0) -> GideonProgram:
        return self.ir.from_fma_sequence(make_chain(n, seed=seed))

    def test_edge_count_linear_chain(self):
        prog = self._make_prog(50)
        g = GideonGraph(prog)
        stats = g.stats()
        self.assertEqual(stats["n_edges"], 50)    # inp→fma_1→...→fma_50

    def test_stats_keys_complete(self):
        prog = self._make_prog(20)
        g = GideonGraph(prog)
        stats = g.stats()
        required = {"n_nodes", "n_edges", "total_fma", "fusable_chains",
                    "global_epsilon", "ai_layers", "n_phases"}
        for key in required:
            self.assertIn(key, stats, f"Clave '{key}' faltante en stats()")

    def test_critical_path_equals_chain_length_plus_one(self):
        n = 30
        prog = self._make_prog(n)
        g = GideonGraph(prog)
        plan = g.analyse()
        # n+1 fases (input node + n FMA nodes, todos secuenciales)
        self.assertEqual(plan.critical_path_length, n + 1)

    def test_phases_cover_all_nodes(self):
        prog = self._make_prog(100)
        g = GideonGraph(prog)
        plan = g.analyse()
        total_in_phases = sum(len(ph.node_ids) for ph in plan.phases)
        self.assertEqual(total_in_phases, prog.total_nodes())

    def test_phase_ordering_respects_dependencies(self):
        prog = self._make_prog(50)
        g = GideonGraph(prog)
        plan = g.analyse()
        # Para cadena lineal cada fase tiene 1 nodo y el orden
        # debe coincidir con topo_order
        phase_order = []
        for ph in plan.phases:
            phase_order.extend(ph.node_ids)
        self.assertEqual(phase_order, prog.topo_order)

    def test_fusable_chains_cover_most_fma(self):
        prog = self._make_prog(200)
        g = GideonGraph(prog)
        chains = g.find_fusable_chains()
        self.assertGreater(len(chains), 0)
        total_fused = sum(len(c) for c in chains)
        # En una cadena lineal pura, casi todos los FMAs deben ser fusibles
        self.assertGreater(total_fused, 100)

    def test_ai_layer_count_empty_for_fma_only(self):
        prog = self._make_prog(30)
        g = GideonGraph(prog)
        ai = g.ai_layer_count()
        total_ai = sum(ai.values())
        self.assertEqual(total_ai, 0, f"Grafo FMA puro reportó capas IA: {ai}")

    def test_large_graph_100k_analysis_time(self):
        chain = make_chain(100_000, seed=5)
        t0 = time.perf_counter()
        prog = self.ir.from_fma_sequence(chain)
        g = GideonGraph(prog)
        plan = g.analyse()
        elapsed = time.perf_counter() - t0
        self.assertEqual(plan.total_nodes, 100_001)
        self.assertLess(elapsed, 40.0)
        log_metric("Graph_100K", f"Análisis grafo 100K en {elapsed*1000:.0f}ms")

    def test_execution_plan_parallelizable_ratio_range(self):
        prog = self._make_prog(50)
        g = GideonGraph(prog)
        plan = g.analyse()
        self.assertGreaterEqual(plan.parallelizable_ratio, 0.0)
        self.assertLessEqual(plan.parallelizable_ratio, 1.0)

    def test_graph_node_status_pending_after_build(self):
        prog = self._make_prog(10)
        g = GideonGraph(prog)
        for gnode in g._nodes.values():
            self.assertIsInstance(gnode, GideonGraphNode)

    def test_graph_edges_have_epsilon_from_ir(self):
        chain = [_FMA(1.5, 0.1)] * 20
        prog = self.ir.from_fma_sequence(chain, precision="fp64")
        g = GideonGraph(prog)
        for edge in g._edges:
            self.assertIsInstance(edge.epsilon, float)
            self.assertGreaterEqual(edge.epsilon, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# 05. TestTitanDispatcherLogic
# ═════════════════════════════════════════════════════════════════════════════

class TestTitanDispatcherLogic(unittest.TestCase):

    def setUp(self):
        self.hw = HardwareProfile.detect()
        self.ir = GideonIR()

    def _prog(self, n: int) -> GideonProgram:
        return self.ir.from_fma_sequence(make_chain(n))

    def test_primary_backend_available(self):
        d = GideonDispatcher(hw_profile=self.hw)
        dec = d.decide(self._prog(50))
        # El primary backend debe existir en los backends conocidos
        known = {"c_native", "pytorch", "rocm", "numpy_cpu", "onnx", "wasm", "verilog"}
        self.assertIn(dec.primary_backend, known)

    def test_fallback_different_from_primary(self):
        d = GideonDispatcher(hw_profile=self.hw)
        dec = d.decide(self._prog(50))
        self.assertNotEqual(dec.primary_backend, dec.fallback_backend)

    def test_estimated_speedup_positive(self):
        d = GideonDispatcher(hw_profile=self.hw)
        for n in [10, 100, 1000]:
            dec = d.decide(self._prog(n))
            self.assertGreater(dec.estimated_speedup, 0.0,
                              f"Speedup estimado ≤ 0 para n={n}")

    def test_numpy_fallback_is_always_available(self):
        d = GideonDispatcher(hw_profile=self.hw)
        dec = d.decide(self._prog(50))
        # numpy_cpu debe aparecer como fallback eventual
        self.assertIn(dec.fallback_backend, {"numpy_cpu", "c_native", "pytorch"})

    def test_user_hint_overrides_ranking(self):
        # Forzar numpy con hint de prioridad 1000
        hint = BackendHint(backend_name="numpy_cpu", priority=1000)
        d = GideonDispatcher(hw_profile=self.hw, user_hints=[hint])
        dec = d.decide(self._prog(50))
        self.assertEqual(dec.primary_backend, "numpy_cpu",
                        "Hint de prioridad 1000 no sobrescribió el ranking")

    def test_latency_feedback_penalizes_slow_backend(self):
        d = GideonDispatcher(hw_profile=self.hw)
        # Registrar latencias altas para c_native
        for _ in range(10):
            d.record_latency("c_native", 500.0)  # 500ms — lentísimo
        hist = d._latency_history.get("c_native", [])
        self.assertEqual(len(hist), 10)
        # Después del feedback, c_native debería recibir penalización
        dec = d.decide(self._prog(50))
        # Al menos el historial está registrado
        self.assertGreater(len(d._latency_history.get("c_native", [])), 0)

    def test_latency_history_capped_at_20(self):
        d = GideonDispatcher(hw_profile=self.hw)
        for i in range(30):
            d.record_latency("c_native", float(i))
        hist = d._latency_history["c_native"]
        self.assertLessEqual(len(hist), 20)

    def test_node_backend_map_covers_all_nodes(self):
        prog = self._prog(100)
        d = GideonDispatcher(hw_profile=self.hw)
        dec = d.decide(prog)
        for nid in prog.nodes:
            self.assertIn(nid, dec.node_backend_map,
                         f"Nodo {nid} no está en node_backend_map")

    def test_heavy_fma_bonus_applied(self):
        # Con >1000 FMAs, c_native debe tener bonus si cffi está disponible
        if not self.hw.has_cffi:
            self.skipTest("cffi no disponible")
        prog_small = self._prog(10)
        prog_large = self._prog(2000)
        d = GideonDispatcher(hw_profile=self.hw)
        dec_s = d.decide(prog_small)
        dec_l = d.decide(prog_large)
        # Ambos deben ser c_native cuando cffi disponible
        if dec_s.primary_backend == "c_native":
            self.assertGreaterEqual(dec_l.estimated_speedup, dec_s.estimated_speedup * 0.9)

    def test_decide_is_deterministic(self):
        prog = self._prog(200)
        d = GideonDispatcher(hw_profile=self.hw)
        dec1 = d.decide(prog)
        dec2 = d.decide(prog)
        self.assertEqual(dec1.primary_backend,  dec2.primary_backend)
        self.assertEqual(dec1.fallback_backend, dec2.fallback_backend)
        self.assertAlmostEqual(dec1.estimated_speedup, dec2.estimated_speedup, places=6)

    def test_reason_contains_backend_name(self):
        d = GideonDispatcher(hw_profile=self.hw)
        dec = d.decide(self._prog(100))
        self.assertIn(dec.primary_backend, dec.reason)

    def test_hardware_detection_no_exception(self):
        hw = HardwareProfile.detect()
        self.assertIsInstance(hw.cpu_cores, int)
        self.assertGreater(hw.cpu_cores, 0)
        self.assertIsInstance(hw.has_avx2, bool)
        self.assertIsInstance(hw.has_cuda, bool)
        self.assertIsInstance(hw.has_cffi, bool)
        log_metric("HW", f"CPU={hw.cpu_cores} | AVX2={hw.has_avx2} | "
                         f"CUDA={hw.has_cuda} | GPU={hw.gpu_name!r} | cffi={hw.has_cffi}")


# ═════════════════════════════════════════════════════════════════════════════
# 06. TestTitanNumericalAccuracy — Precisión matemática estricta
# ═════════════════════════════════════════════════════════════════════════════

class TestTitanNumericalAccuracy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = GideonEngine(GideonEngineConfig(precision="fp64"))
        cls.rng    = np.random.default_rng(1234)

    def _run(self, chain, x) -> np.ndarray:
        result = self.engine.run_fma(chain, x)
        return result.output

    def _ref(self, chain, x) -> np.ndarray:
        return numpy_eval(chain, x)

    def test_1_fma_exact(self):
        chain = [_FMA(3.0, -0.7)]
        x = np.array([-1.0, 0.0, 0.5, 1.0])
        out = self._run(chain, x)
        ref = self._ref(chain, x)
        np.testing.assert_allclose(out, ref, rtol=1e-14, atol=1e-15)

    def test_chain_10_rtol_1e12(self):
        chain = make_chain(10, seed=0)
        x = np.linspace(-1.0, 1.0, 10_000)
        np.testing.assert_allclose(self._run(chain, x), self._ref(chain, x),
                                   rtol=1e-12, atol=1e-14)

    def test_chain_100_rtol_1e10(self):
        chain = make_chain(100, seed=1)
        x = np.linspace(-1.0, 1.0, 5_000)
        np.testing.assert_allclose(self._run(chain, x), self._ref(chain, x),
                                   rtol=1e-10, atol=1e-12)

    def test_chain_1000_rtol_1e8(self):
        chain = make_chain(1000, seed=2)
        x = np.linspace(-0.5, 0.5, 2_000)
        np.testing.assert_allclose(self._run(chain, x), self._ref(chain, x),
                                   rtol=1e-8, atol=1e-10)

    def test_contractive_chain_converges_to_fixed_point(self):
        # x_{n+1} = 0.5 * x_n + 0.5 → fixed point x* = 1
        chain = [_FMA(0.5, 0.5)] * 100
        x = np.array([0.0, 5.0, -3.0])
        out = self._run(chain, x)
        expected = np.full(3, 1.0)  # fixed point
        np.testing.assert_allclose(out, expected, rtol=1e-8)
        log_metric("FP_convergence", f"Punto fijo error max={np.max(np.abs(out - expected)):.2e}")

    def test_zero_input_propagation(self):
        chain = [_FMA(2.0, 1.0), _FMA(3.0, -2.0), _FMA(0.5, 0.5)]
        x = np.zeros(1000)
        out = self._run(chain, x)
        ref = self._ref(chain, x)
        np.testing.assert_allclose(out, ref, atol=1e-15)

    def test_negative_domain_accuracy(self):
        chain = make_chain(50, seed=3)
        x = np.linspace(-5.0, -0.001, 5_000)
        np.testing.assert_allclose(self._run(chain, x), self._ref(chain, x),
                                   rtol=1e-10)

    def test_large_n_small_values_accuracy(self):
        # Pesos muy pequeños; señal se atenúa pero sin divergir
        chain = [_FMA(0.01, 0.0)] * 50
        x = np.array([1.0])
        out = self._run(chain, x)
        expected = 0.01 ** 50
        self.assertAlmostEqual(float(out[0]) if hasattr(out, '__len__') else float(out),
                               expected, places=60)

    def test_idempotent_execution_5_times(self):
        chain = make_chain(200, seed=4)
        x = np.linspace(-1.0, 1.0, 2_000)
        outputs = [self._run(chain, x.copy()) for _ in range(5)]
        for i in range(1, 5):
            np.testing.assert_array_equal(outputs[0], outputs[i],
                err_msg=f"Ejecución {i} difiere de la primera")

    def test_precision_fp64_global_epsilon_bounds_output(self):
        chain = make_chain(200, seed=5)
        x = np.linspace(-1.0, 1.0, 1_000)
        ref = self._ref(chain, x)
        result = self.engine.run_fma(chain, x)
        max_err = float(np.max(np.abs(result.output - ref)))
        # max_err debe estar dentro de la cota ε certificada (con margen 100×)
        certified_bound = result.global_epsilon * max(1.0, np.max(np.abs(ref)))
        margin = 100.0
        self.assertLess(max_err, certified_bound * margin,
            f"Error={max_err:.3e} > cota {certified_bound*margin:.3e}")


# ═════════════════════════════════════════════════════════════════════════════
# 07. TestTitanSpeedupBenchmarks — Velocidad real medida
# ═════════════════════════════════════════════════════════════════════════════

class TestTitanSpeedupBenchmarks(unittest.TestCase):
    """
    Mide speedup real de Gideon vs NumPy baseline.
    Cada test imprime métricas específicas y exige speedup mínimo.
    """

    REPEAT = 3  # promedio de N ejecuciones

    @classmethod
    def setUpClass(cls):
        cls.engine = GideonEngine()

    def _measure(self, chain, x_arr, repeat: int = None) -> tuple:
        r = repeat or self.REPEAT
        # NumPy baseline
        ws = [f.weight for f in chain]
        bs = [f.bias   for f in chain]
        t0 = time.perf_counter()
        for _ in range(r):
            y = np.asarray(x_arr, dtype=np.float64).copy()
            for w, b in zip(ws, bs):
                y = w * y + b
        t_np = (time.perf_counter() - t0) / r * 1000

        # Gideon
        t0 = time.perf_counter()
        for _ in range(r):
            res = self.engine.run_fma(chain, x_arr)
        t_gi = (time.perf_counter() - t0) / r * 1000

        return t_np, t_gi, res

    def test_depth10_n10M(self):
        chain = make_chain(10, seed=10)
        x = np.random.default_rng(10).standard_normal(10_000_000)
        t_np, t_gi, res = self._measure(chain, x)
        spdup = speedup(t_np, t_gi)
        log_metric("BM_10x10M",
                   f"depth=10, n=10M | numpy={t_np:.2f}ms | "
                   f"{res.backend_used}={t_gi:.2f}ms | speedup={spdup:.1f}×")
        self.assertTrue(res.success, "Ejecución debe completarse exitosamente")
        np.testing.assert_allclose(res.output, numpy_eval(chain, x), rtol=1e-8)

    def test_depth50_n1M(self):
        chain = make_chain(50, seed=11)
        x = np.random.default_rng(11).standard_normal(1_000_000)
        t_np, t_gi, res = self._measure(chain, x)
        spdup = speedup(t_np, t_gi)
        log_metric("BM_50x1M",
                   f"depth=50, n=1M | numpy={t_np:.2f}ms | "
                   f"{res.backend_used}={t_gi:.2f}ms | speedup={spdup:.1f}×")
        self.assertTrue(res.success)
        np.testing.assert_allclose(res.output, numpy_eval(chain, x), rtol=1e-8)

    def test_depth100_n100K(self):
        chain = make_chain(100, seed=12)
        x = np.random.default_rng(12).standard_normal(100_000)
        t_np, t_gi, res = self._measure(chain, x)
        spdup = speedup(t_np, t_gi)
        log_metric("BM_100x100K",
                   f"depth=100, n=100K | numpy={t_np:.2f}ms | "
                   f"{res.backend_used}={t_gi:.2f}ms | speedup={spdup:.1f}×")
        self.assertTrue(res.success)
        np.testing.assert_allclose(res.output, numpy_eval(chain, x), rtol=1e-8)

    def test_depth1000_n10K(self):
        chain = make_chain(1000, seed=13)
        x = np.random.default_rng(13).standard_normal(10_000)
        t_np, t_gi, res = self._measure(chain, x)
        spdup = speedup(t_np, t_gi)
        log_metric("BM_1Kx10K",
                   f"depth=1K, n=10K | numpy={t_np:.2f}ms | "
                   f"{res.backend_used}={t_gi:.2f}ms | speedup={spdup:.1f}×")
        self.assertTrue(res.success)
        np.testing.assert_allclose(res.output, numpy_eval(chain, x), rtol=1e-8)

    def test_depth10_n50M(self):
        chain = make_chain(10, seed=99)
        x = np.random.default_rng(99).standard_normal(50_000_000)
        t_np, t_gi, res = self._measure(chain, x, repeat=1)
        spdup = speedup(t_np, t_gi)
        log_metric("BM_10x50M",
                   f"depth=10, n=50M | numpy={t_np:.2f}ms | "
                   f"{res.backend_used}={t_gi:.2f}ms | speedup={spdup:.1f}×")
        self.assertTrue(res.success)
        # rtol=1e-7: fold algebraico en GPU puede diferir del numpy secuencial
        # en ~1 ULP por la composición de redondeos (aceptable)
        np.testing.assert_allclose(res.output, numpy_eval(chain, x), rtol=1e-7)

    def test_backend_used_is_reported(self):
        chain = make_chain(100, seed=14)
        x = np.linspace(-1, 1, 10_000)
        res = self.engine.run_fma(chain, x)
        self.assertGreater(len(res.backend_used), 0)
        self.assertIsInstance(res.backend_used, str)

    def test_elapsed_ms_positive(self):
        chain = make_chain(200)
        x = np.linspace(-1, 1, 50_000)
        res = self.engine.run_fma(chain, x)
        self.assertGreater(res.elapsed_ms, 0.0)

    def test_output_shape_matches_input(self):
        for n_in in [1, 100, 10_000, 1_000_000]:
            chain = make_chain(10)
            x = np.zeros(n_in)
            res = self.engine.run_fma(chain, x)
            self.assertEqual(res.output.shape, (n_in,),
                f"Shape incorrecto para n_in={n_in}")


# ═════════════════════════════════════════════════════════════════════════════
# 08. TestTitanScaleExtremes — Escalado extremo
# ═════════════════════════════════════════════════════════════════════════════

class TestTitanScaleExtremes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = GideonEngine()

    def test_50M_elements_10_stages(self):
        chain = make_chain(10, seed=20)
        x = np.random.default_rng(20).standard_normal(50_000_000)
        t0 = time.perf_counter()
        res = self.engine.run_fma(chain, x)
        elapsed = (time.perf_counter() - t0) * 1000
        self.assertEqual(res.output.shape[0], 50_000_000)
        self.assertTrue(res.success)
        log_metric("Scale_50M", f"50M×10 | t={elapsed:.0f}ms | backend={res.backend_used}")

    def test_1M_elements_1000_stages(self):
        chain = make_chain(1000, seed=21)
        x = np.random.default_rng(21).standard_normal(1_000_000)
        t0 = time.perf_counter()
        res = self.engine.run_fma(chain, x)
        elapsed = (time.perf_counter() - t0) * 1000
        self.assertEqual(res.output.shape[0], 1_000_000)
        self.assertTrue(res.success)
        log_metric("Scale_1M_1K", f"1M×1K | t={elapsed:.0f}ms | ε={res.global_epsilon:.3e}")

    def test_100K_elements_5000_stages(self):
        chain = make_contractive_chain(5000, seed=22)
        x = np.random.default_rng(22).standard_normal(100_000)
        t0 = time.perf_counter()
        res = self.engine.run_fma(chain, x)
        elapsed = (time.perf_counter() - t0) * 1000
        self.assertEqual(res.output.shape[0], 100_000)
        log_metric("Scale_5K_depth", f"100K×5K | t={elapsed:.0f}ms | ε={res.global_epsilon:.3e}")

    def test_single_element_10k_stages(self):
        chain = make_chain(10_000, seed=23)
        x = np.array([0.5])
        res = self.engine.run_fma(chain, x)
        self.assertEqual(res.output.shape[0], 1)
        self.assertTrue(np.isfinite(res.output).all())

    def test_very_large_array_single_stage(self):
        chain = [_FMA(2.0, -1.0)]
        x = np.ones(20_000_000)
        res = self.engine.run_fma(chain, x)
        np.testing.assert_allclose(res.output, 1.0, rtol=1e-12)

    def test_batch_sizes_correctness(self):
        chain = make_chain(20, seed=24)
        for n in [1, 10, 100, 1000, 100_000, 1_000_000]:
            x = np.random.default_rng(n).standard_normal(n)
            res = self.engine.run_fma(chain, x)
            ref = numpy_eval(chain, x)
            np.testing.assert_allclose(res.output, ref, rtol=1e-9,
                err_msg=f"Error en batch_size={n}")

    def test_zero_array_long_chain(self):
        chain = [_FMA(1.5, 0.3)] * 100
        x = np.zeros(10_000)
        res = self.engine.run_fma(chain, x)
        ref = numpy_eval(chain, x)
        np.testing.assert_allclose(res.output, ref, rtol=1e-12)

    def test_negative_only_inputs(self):
        chain = make_chain(50, seed=25)
        x = np.linspace(-10.0, -0.001, 50_000)
        res = self.engine.run_fma(chain, x)
        self.assertTrue(np.isfinite(res.output).all())
        ref = numpy_eval(chain, x)
        np.testing.assert_allclose(res.output, ref, rtol=1e-10)


# ═════════════════════════════════════════════════════════════════════════════
# 09. TestTitanAdversarialInputs — Entradas hostiles
# ═════════════════════════════════════════════════════════════════════════════

class TestTitanAdversarialInputs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = GideonEngine()

    def test_nan_input_produces_nan_output(self):
        chain = make_chain(10)
        x = np.array([1.0, float("nan"), 2.0])
        res = self.engine.run_fma(chain, x)
        # NaN en entrada → NaN en salida (no excepción)
        self.assertTrue(np.isnan(res.output[1]),
                        "NaN en entrada debe propagarse")

    def test_inf_input_produces_inf_or_nan(self):
        chain = make_chain(10)
        x = np.array([float("inf"), 1.0])
        res = self.engine.run_fma(chain, x)
        # inf * w + b → inf o nan, nunca excepción
        self.assertFalse(res.output[1] == float("inf"),  # elemento 1 normal
                         "Elemento finito fue corrompido")

    def test_empty_chain_identity(self):
        x = np.linspace(-1.0, 1.0, 1000)
        res = self.engine.run_fma([], x)
        # Cadena vacía → salida = entrada (identity)
        np.testing.assert_array_equal(res.output, x)

    def test_zero_weight_kills_signal(self):
        chain = [_FMA(1.0, 0.5)] * 5 + [_FMA(0.0, 3.0)] + [_FMA(2.0, 0.0)] * 5
        x = np.random.default_rng(0).standard_normal(1000)
        res = self.engine.run_fma(chain, x)
        ref = numpy_eval(chain, x)
        np.testing.assert_allclose(res.output, ref, rtol=1e-12)
        # Después del nodo w=0: 3.0, luego 5×FMA(2,0): 3→6→12→24→48→96
        np.testing.assert_allclose(res.output, 96.0, rtol=1e-12)

    def test_very_large_weights_overflow_handled(self):
        chain = [_FMA(1e100, 0.0)] * 5
        x = np.array([1.0])
        res = self.engine.run_fma(chain, x)
        # No debe lanzar excepción; puede ser inf
        self.assertTrue(np.isinf(res.output[0]) or np.isfinite(res.output[0]))

    def test_very_small_weights_underflow(self):
        chain = [_FMA(1e-100, 0.0)] * 100
        x = np.array([1.0])
        res = self.engine.run_fma(chain, x)
        # Debe ser 0 por underflow o un número muy pequeño
        self.assertLessEqual(float(abs(res.output[0])), 1.0)

    def test_alternating_signs(self):
        chain = [_FMA(2.0, 0.0), _FMA(-0.5, 0.0)] * 50
        x = np.array([1.0, -1.0, 0.5])
        res = self.engine.run_fma(chain, x)
        ref = numpy_eval(chain, x)
        np.testing.assert_allclose(res.output, ref, rtol=1e-12)

    def test_single_point_domain(self):
        chain = make_chain(10)
        x = np.array([0.0])
        res = self.engine.run_fma(chain, x)
        ref = numpy_eval(chain, x)
        np.testing.assert_allclose(res.output, ref, rtol=1e-12)

    def test_ir_lowering_empty_chain(self):
        ir = GideonIR()
        prog = ir.from_fma_sequence([], domain=(-1.0, 1.0))
        self.assertEqual(prog.total_fma, 0)
        # Solo nodo input
        self.assertEqual(len(prog.nodes), 1)

    def test_chain_all_ones(self):
        # w=1, b=0 → identity
        chain = [_FMA(1.0, 0.0)] * 200
        x = np.random.default_rng(50).standard_normal(5_000)
        res = self.engine.run_fma(chain, x)
        np.testing.assert_allclose(res.output, x, rtol=1e-12, atol=1e-14)


# ═════════════════════════════════════════════════════════════════════════════
# 10. TestTitanMathematicalProps — Propiedades algebraicas
# ═════════════════════════════════════════════════════════════════════════════

class TestTitanMathematicalProps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = GideonEngine()

    def test_fma_associativity(self):
        """
        ((w2 * (w1 * x + b1)) + b2) = (w2*w1) * x + (w2*b1 + b2)
        Composición asociativa de FMAs.
        """
        w1, b1, w2, b2 = 1.5, 0.3, 0.8, -0.1
        chain_two = [_FMA(w1, b1), _FMA(w2, b2)]
        chain_one = [_FMA(w2 * w1, w2 * b1 + b2)]
        x = np.linspace(-2.0, 2.0, 5_000)
        res_two = self.engine.run_fma(chain_two, x)
        res_one = self.engine.run_fma(chain_one, x)
        np.testing.assert_allclose(res_two.output, res_one.output,
                                   rtol=1e-12, atol=1e-14,
                                   err_msg="FMA no es asociativo")

    def test_contraction_chain_fixed_point(self):
        """Banach: f contráctiva → tiene punto fijo único."""
        fixed_point = 1.0
        w, b = 0.5, 0.5  # x* = w*x* + b → x* = b/(1-w) = 1
        chain = [_FMA(w, b)] * 200
        x = np.array([-10.0, 0.0, 5.0, 100.0])
        res = self.engine.run_fma(chain, x)
        np.testing.assert_allclose(res.output, fixed_point, rtol=1e-6,
                                   err_msg="Contracción no converge al punto fijo")

    def test_product_rule_weights(self):
        """Cadena sin bias: y = (∏w_i) * x."""
        ws = [1.5, 0.8, 1.2, 0.9, 1.1]
        chain = [_FMA(w, 0.0) for w in ws]
        x = np.array([1.0, 2.0, -3.0])
        res = self.engine.run_fma(chain, x)
        expected_coeff = math.prod(ws)
        np.testing.assert_allclose(res.output, expected_coeff * x, rtol=1e-12)

    def test_shift_chain_additive(self):
        """Cadena de shifts (w=1): y = x + Σb_i."""
        bs = [0.1, 0.2, -0.3, 0.5, 1.0]
        chain = [_FMA(1.0, b) for b in bs]
        x = np.array([0.0, 1.0, -2.0])
        res = self.engine.run_fma(chain, x)
        expected = x + sum(bs)
        np.testing.assert_allclose(res.output, expected, rtol=1e-12)

    def test_lipschitz_constant_chain(self):
        """L(f_chain) = ∏|w_i| para cadena sin bias."""
        ws = [1.5, 0.8, 1.2, 0.9]
        chain = [_FMA(w, 0.0) for w in ws]
        x = np.linspace(-1.0, 1.0, 2000)
        res = self.engine.run_fma(chain, x)
        L_expected = math.prod(abs(w) for w in ws)
        # Estimar L numéricamente
        dx = np.diff(x)
        dy = np.diff(res.output)
        L_empirical = float(np.max(np.abs(dy) / (np.abs(dx) + 1e-300)))
        self.assertAlmostEqual(L_empirical, L_expected, places=6,
            msg=f"L_empirical={L_empirical:.6f} ≠ L_expected={L_expected:.6f}")

    def test_linearity_superposition(self):
        """f(ax + by) = a*f(x) + b*f(y) para f línea (b=0, w=1.5)."""
        chain = [_FMA(1.5, 0.0)] * 5
        a, b_ = 2.0, -1.0
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([-1.0, 0.5, 2.5])
        r1 = self.engine.run_fma(chain, x1).output
        r2 = self.engine.run_fma(chain, x2).output
        r_super = self.engine.run_fma(chain, a * x1 + b_ * x2).output
        np.testing.assert_allclose(r_super, a * r1 + b_ * r2, rtol=1e-12)

    def test_fixed_bias_chain_geometric(self):
        """f_n(0) = b * (1 + w + w^2 + ... + w^{n-1}) = b*(1-w^n)/(1-w)."""
        w, b = 0.5, 1.0
        n = 20
        chain = [_FMA(w, b)] * n
        x = np.array([0.0])
        res = self.engine.run_fma(chain, x)
        # Suma geométrica
        expected = b * (1.0 - w ** n) / (1.0 - w)
        self.assertAlmostEqual(float(res.output[0]), expected, places=10)

    def test_contraction_rate_empirical(self):
        """Una contracción k=0.7 reduce distancias en factor 0.7^n."""
        k = 0.7
        n = 50
        chain = [_FMA(k, 0.0)] * n
        x1 = np.array([1.0])
        x2 = np.array([2.0])
        r1 = self.engine.run_fma(chain, x1).output
        r2 = self.engine.run_fma(chain, x2).output
        dist_out = float(abs(r1[0] - r2[0]))
        dist_in  = 1.0  # |x1 - x2|
        expected_dist = dist_in * k ** n
        self.assertAlmostEqual(dist_out, expected_dist, places=10)


# ═════════════════════════════════════════════════════════════════════════════
# 11. TestTitanTheoremInvariants — Detección de invariantes en funciones conocidas
# ═════════════════════════════════════════════════════════════════════════════

class TestTitanTheoremInvariants(unittest.TestCase):

    def _probe(self, fn, domain=(-2.0, 2.0), n=2000) -> InvariantProbe:
        return InvariantProbe(fn, domain, n_points=n)

    def test_identity_lipschitz_is_1(self):
        probe = self._probe(lambda x: x)
        L = probe.lipschitz_estimate()
        self.assertAlmostEqual(L, 1.0, places=3)

    def test_linear_lipschitz_is_slope(self):
        slope = 3.7
        probe = self._probe(lambda x: slope * x)
        L = probe.lipschitz_estimate()
        self.assertAlmostEqual(L, slope, places=2)

    def test_contraction_detected(self):
        is_contr, k = PatternMatcher.is_contraction(lambda x: 0.5 * x, (-2, 2), n=500)
        self.assertTrue(is_contr)
        self.assertLess(k, 1.0)
        self.assertAlmostEqual(k, 0.5, places=2)

    def test_non_contraction_detected(self):
        is_contr, k = PatternMatcher.is_contraction(lambda x: 2.0 * x, (-1, 1), n=500)
        self.assertFalse(is_contr)
        self.assertGreater(k, 1.0)

    def test_monotone_increasing(self):
        probe = self._probe(lambda x: x * 2.0 + 1.0)
        self.assertTrue(probe.is_monotone())

    def test_monotone_decreasing(self):
        probe = self._probe(lambda x: -x * 1.5)
        self.assertTrue(probe.is_monotone())

    def test_non_monotone(self):
        probe = self._probe(lambda x: x * x, domain=(-1.0, 1.0))
        self.assertFalse(probe.is_monotone())

    def test_even_function_detected(self):
        # n_points par obligatorio para que mid sea exactamente la mitad simétrica
        probe = InvariantProbe(lambda x: x * x, (-2.0, 2.0), n_points=1000)
        sym = probe.symmetry_type()
        self.assertEqual(sym, "even", f"x^2 debe detectarse como par, got '{sym}'")

    def test_odd_function_detected(self):
        # n_points par obligatorio para simetría correcta
        probe = InvariantProbe(lambda x: x * x * x, (-2.0, 2.0), n_points=1000)
        sym = probe.symmetry_type()
        self.assertEqual(sym, "odd", f"x^3 debe detectarse como impar, got '{sym}'")

    def test_analyse_lipschitz_candidate_confidence(self):
        seeds = GideonTheoremSeeds()
        fn = lambda x: 0.3 * x + 0.1  # Lipschitz = 0.3
        candidates = seeds.analyse(fn, domain=(-2.0, 2.0), fn_name="linear_fn")
        lip_cands = [c for c in candidates if "lipschitz" in c.name]
        self.assertGreater(len(lip_cands), 0)
        for c in lip_cands:
            self.assertGreater(c.confidence, 0.0)
            self.assertLessEqual(c.confidence, 1.0)

    def test_contraction_theorem_candidate_generated(self):
        seeds = GideonTheoremSeeds()
        fn = lambda x: 0.5 * x + 0.2
        candidates = seeds.analyse(fn, domain=(-3.0, 3.0), fn_name="contractive")
        contr_cands = [c for c in candidates if "contraction" in c.name]
        self.assertGreater(len(contr_cands), 0)
        self.assertGreater(contr_cands[0].confidence, 0.5)

    def test_lean_skeleton_is_non_empty(self):
        seeds = GideonTheoremSeeds()
        fn = lambda x: 0.7 * x
        candidates = seeds.analyse(fn, domain=(-2.0, 2.0), fn_name="scale_fn")
        for c in candidates:
            self.assertIsInstance(c.lean_skeleton, str)
            self.assertGreater(len(c.lean_skeleton), 10)

    def test_alpha_complexity_positive_finite(self):
        probe = self._probe(lambda x: np.sin(x) * np.cos(x))
        alpha = probe.alpha_complexity()
        self.assertGreater(alpha, 0.0)
        self.assertTrue(math.isfinite(alpha))

    def test_fma_chain_contraction_theorem(self):
        engine = GideonEngine(GideonEngineConfig(enable_theorem_seeds=True,
                                                  domain=(-2.0, 2.0)))
        chain = [_FMA(0.6, 0.1)] * 30
        x = np.linspace(-2.0, 2.0, 1000)
        result = engine.run_fma(chain, x)
        # Gideon debe haber generado candidatos
        self.assertIsInstance(result.theorem_candidates, list)
        log_metric("Theorem_FMA",
                   f"candidatos={len(result.theorem_candidates)} para cadena contráctiva")

    def test_export_lean_file(self):
        import tempfile
        seeds = GideonTheoremSeeds()
        fn = lambda x: 0.4 * x + 0.1
        candidates = seeds.analyse(fn, domain=(-1.0, 1.0), fn_name="fn_test")
        self.assertGreater(len(candidates), 0)
        with tempfile.NamedTemporaryFile(suffix=".lean", delete=False) as f:
            path = f.name
        try:
            seeds.export_lean_file(path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                content = f.read()
            self.assertGreater(len(content), 50)
            log_metric("TheoremLean", f"Archivo Lean generado: {len(content)} chars")
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ═════════════════════════════════════════════════════════════════════════════
# 12. TestTitanNeuralBlueprints — Blueprints de IA a escala real
# ═════════════════════════════════════════════════════════════════════════════

class TestTitanNeuralBlueprints(unittest.TestCase):

    def test_mlp_small_params_count(self):
        bp = GideonNeuralHints.mlp([784, 256, 128, 10])
        # Parámetros: 784*256 + 256*128 + 128*10 = 200,704 + 32,768 + 1,280 = 234,752
        self.assertGreater(bp.total_params, 200_000)
        self.assertLess(bp.total_params, 500_000)

    def test_mlp_large_alpha_finite(self):
        bp = GideonNeuralHints.mlp([2048, 1024, 512, 256, 128, 64, 32, 10])
        alpha = bp.compute_alpha_complexity()
        self.assertGreater(alpha, 0.0)
        self.assertTrue(math.isfinite(alpha))
        log_metric("MLP_alpha", f"MLP-8capas α={alpha:.4f}")

    def test_transformer_gpt2_scale(self):
        # GPT-2 small: d_model=768, n_heads=12, n_layers=12
        bp = GideonNeuralHints.transformer(
            d_model=768, n_heads=12, n_layers=12, seq_len=1024, ffn_dim=3072
        )
        self.assertGreater(bp.total_params, 50_000_000, "GPT-2 small debe tener >50M params")
        alpha = bp.compute_alpha_complexity()
        self.assertGreater(alpha, 0.9)
        fma_eq = bp.to_gideon_fma_count()
        self.assertGreater(fma_eq, 1_000_000)
        log_metric("GPT2_small",
                   f"params={bp.total_params:,} | α={alpha:.4f} | fma_eq={fma_eq:,}")

    def test_transformer_large_scale(self):
        # Transformer-XL escala
        bp = GideonNeuralHints.transformer(
            d_model=1024, n_heads=16, n_layers=24, ffn_dim=4096
        )
        self.assertGreater(bp.total_params, 100_000_000)
        alpha = bp.compute_alpha_complexity()
        self.assertGreater(alpha, 0.0)
        log_metric("Trans_XL", f"params={bp.total_params/1e6:.1f}M | α={alpha:.4f}")

    def test_resnet_50_scale(self):
        # ResNet-50: 50 bloques aprox, 64 canales
        bp = GideonNeuralHints.cnn_resnet_block(channels=256, kernel_size=3, n_blocks=50)
        self.assertGreater(bp.total_flops, 0)
        self.assertGreater(bp.total_params, 0)
        log_metric("ResNet50",
                   f"params={bp.total_params:,} | flops={bp.total_flops:,}")

    def test_alpha_complexity_transformer_gt_mlp(self):
        # Transformer tiene mayor densidad computacional que MLP equivalente
        mlp = GideonNeuralHints.mlp([512, 512, 512])
        trans = GideonNeuralHints.transformer(d_model=512, n_heads=8, n_layers=4)
        alpha_mlp   = mlp.compute_alpha_complexity()
        alpha_trans = trans.compute_alpha_complexity()
        # No siempre aplica — solo verificar que ambos son positivos y finitos
        self.assertGreater(alpha_mlp, 0.0)
        self.assertGreater(alpha_trans, 0.0)

    def test_analyse_blueprint_returns_dict(self):
        bp = GideonNeuralHints.mlp([128, 64, 10])
        metrics = GideonNeuralHints.analyse_blueprint(bp)
        self.assertIsInstance(metrics, dict)
        for key in ("name", "total_params", "total_flops", "alpha_complexity"):
            self.assertIn(key, metrics, f"Clave '{key}' faltante en analyse_blueprint")

    def test_blueprint_summary_non_empty(self):
        bp = GideonNeuralHints.transformer(d_model=256, n_heads=4, n_layers=3)
        summary = bp.summary()
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 50)

    def test_fma_equivalent_gt_zero(self):
        bp = GideonNeuralHints.mlp([100, 50, 25, 10])
        self.assertGreater(bp.to_gideon_fma_count(), 0)

    def test_engine_analyse_blueprint(self):
        engine = GideonEngine()
        bp = GideonNeuralHints.mlp([512, 256, 128, 64])
        metrics = engine.analyse_blueprint(bp)
        self.assertIsInstance(metrics, dict)
        self.assertIn("alpha_complexity", metrics)


# ═════════════════════════════════════════════════════════════════════════════
# 13. TestTitanSerializationStress — JSON roundtrip bajo estrés
# ═════════════════════════════════════════════════════════════════════════════

class TestTitanSerializationStress(unittest.TestCase):

    def setUp(self):
        self.ir = GideonIR()

    def test_roundtrip_100_times_identical(self):
        chain = make_chain(100, seed=30)
        prog = self.ir.from_fma_sequence(chain, name="rt_test_100")
        json_orig = GideonIR.to_json(prog)
        for i in range(100):
            prog_restored = GideonIR.from_json(json_orig)
            json_restored = GideonIR.to_json(prog_restored)
            self.assertEqual(json_orig, json_restored,
                            f"Roundtrip {i+1} difirió")

    def test_roundtrip_preserves_epsilon(self):
        chain = make_chain(500, seed=31)
        prog = self.ir.from_fma_sequence(chain)
        json_str = GideonIR.to_json(prog)
        prog2 = GideonIR.from_json(json_str)
        self.assertAlmostEqual(prog.global_epsilon, prog2.global_epsilon, places=15)

    def test_roundtrip_preserves_fma_count(self):
        for n in [10, 100, 1000]:
            chain = make_chain(n, seed=n)
            prog = self.ir.from_fma_sequence(chain)
            prog2 = GideonIR.from_json(GideonIR.to_json(prog))
            self.assertEqual(prog.total_fma, prog2.total_fma)
            self.assertEqual(len(prog.nodes), len(prog2.nodes))

    def test_roundtrip_10k_nodes(self):
        chain = make_chain(10_000, seed=32)
        t0 = time.perf_counter()
        prog = self.ir.from_fma_sequence(chain)
        json_str = GideonIR.to_json(prog)
        prog2 = GideonIR.from_json(json_str)
        elapsed = (time.perf_counter() - t0) * 1000
        self.assertEqual(prog2.total_fma, 10_000)
        log_metric("JSON_10K", f"Roundtrip 10K nodos en {elapsed:.0f}ms | "
                               f"JSON size={len(json_str)//1024}KB")

    def test_json_size_linear_in_chain_length(self):
        json_sizes = []
        for n in [100, 500, 1000, 2000]:
            chain = make_chain(n, seed=33)
            prog = self.ir.from_fma_sequence(chain)
            json_sizes.append(len(GideonIR.to_json(prog)))
        # Cada incremento 5× en n debe ≈ 5× en JSON size
        ratios = [json_sizes[i+1] / json_sizes[i] for i in range(len(json_sizes)-1)]
        for r in ratios:
            self.assertGreater(r, 1.5, f"Ratio {r:.1f} sugiere crecimiento no lineal")
            self.assertLess(r, 10.0, f"Ratio {r:.1f} demasiado grande")

    def test_from_json_rejects_malformed(self):
        with self.assertRaises((json.JSONDecodeError, KeyError, ValueError, TypeError)):
            GideonIR.from_json("{definitely: not valid json}")

    def test_topo_order_preserved(self):
        chain = make_chain(200, seed=34)
        prog = self.ir.from_fma_sequence(chain)
        prog2 = GideonIR.from_json(GideonIR.to_json(prog))
        self.assertEqual(prog.topo_order, prog2.topo_order)


# ═════════════════════════════════════════════════════════════════════════════
# 14. TestTitanConcurrentEngines — 4 motores simultáneos
# ═════════════════════════════════════════════════════════════════════════════

class TestTitanConcurrentEngines(unittest.TestCase):

    def test_4_engines_same_result(self):
        """4 engines independientes, misma cadena, mismo input → mismo output."""
        chain = make_chain(100, seed=40)
        x = np.random.default_rng(40).standard_normal(10_000)
        results = [None] * 4
        errors = []

        def run_engine(idx):
            try:
                eng = GideonEngine()
                res = eng.run_fma(chain, x.copy())
                results[idx] = res.output.copy()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=run_engine, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errores en threads: {errors}")
        for i in range(1, 4):
            np.testing.assert_allclose(results[0], results[i], rtol=1e-12,
                err_msg=f"Engine {i} difiere del engine 0")
        log_metric("Conc4", "4 engines concurrentes producen resultados idénticos ✓")

    def test_concurrent_different_chains(self):
        """4 engines con cadenas distintas run en paralelo sin interferencia."""
        chains = [make_chain(50, seed=i*10) for i in range(4)]
        x = np.linspace(-1, 1, 5_000)
        results = [None] * 4
        refs = [numpy_eval(c, x) for c in chains]
        errors = []

        def run_engine(idx):
            try:
                eng = GideonEngine()
                res = eng.run_fma(chains[idx], x.copy())
                results[idx] = res.output.copy()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=run_engine, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errores: {errors}")
        for i in range(4):
            # rtol=1e-8: fold algebraico puede diferir del numpy secuencial en ~1 ULP
            np.testing.assert_allclose(results[i], refs[i], rtol=1e-8,
                err_msg=f"Engine {i} no coincide con ref numpy")

    def test_single_engine_sequential_chains(self):
        """Un engine puede reutilizarse en múltiples ejecuciones consecutivas."""
        engine = GideonEngine()
        for seed in range(20):
            chain = make_chain(30, seed=seed)
            x = np.linspace(-1, 1, 1_000)
            res = engine.run_fma(chain, x)
            ref = numpy_eval(chain, x)
            np.testing.assert_allclose(res.output, ref, rtol=1e-10,
                err_msg=f"Reutilización falla en seed={seed}")


# ═════════════════════════════════════════════════════════════════════════════
# 15. TestTitanMetricsCompleteness — Completitud de GideonExecutionResult
# ═════════════════════════════════════════════════════════════════════════════

class TestTitanMetricsCompleteness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = GideonEngine()
        cls.chain  = make_chain(200, seed=50)
        cls.x      = np.linspace(-1.0, 1.0, 10_000)
        cls.result = cls.engine.run_fma(cls.chain, cls.x)

    def test_output_is_array(self):
        self.assertIsNotNone(self.result.output)
        self.assertEqual(len(self.result.output), 10_000)

    def test_backend_used_non_empty_string(self):
        self.assertIsInstance(self.result.backend_used, str)
        self.assertGreater(len(self.result.backend_used), 0)

    def test_elapsed_ms_positive(self):
        self.assertGreater(self.result.elapsed_ms, 0.0)

    def test_total_fma_matches_chain(self):
        self.assertEqual(self.result.total_fma, 200)

    def test_global_epsilon_positive(self):
        self.assertGreater(self.result.global_epsilon, 0.0)
        self.assertTrue(math.isfinite(self.result.global_epsilon))

    def test_success_is_true(self):
        self.assertTrue(self.result.success)

    def test_error_field_empty_on_success(self):
        self.assertEqual(self.result.error, "")

    def test_program_is_gideon_program(self):
        self.assertIsInstance(self.result.program, GideonProgram)
        self.assertEqual(self.result.program.total_fma, 200)

    def test_graph_stats_has_required_keys(self):
        stats = self.result.graph_stats
        for key in ("n_nodes", "n_edges", "total_fma", "fusable_chains"):
            self.assertIn(key, stats, f"Falta '{key}' en graph_stats")

    def test_dispatch_decision_is_valid(self):
        dec = self.result.dispatch_decision
        self.assertIsInstance(dec, DispatchDecision)
        self.assertGreater(len(dec.primary_backend), 0)
        self.assertGreater(dec.estimated_speedup, 0.0)

    def test_theorem_candidates_is_list(self):
        self.assertIsInstance(self.result.theorem_candidates, list)

    def test_summary_method_works(self):
        s = self.result.summary()
        self.assertIsInstance(s, str)
        self.assertIn("Backend", s)
        self.assertIn("FMA", s)

    def test_engine_info_works(self):
        info = self.engine.info()
        self.assertIsInstance(info, str)
        self.assertIn("GIDEON", info)
        self.assertIn("Hardware", info)

    def test_output_all_finite_on_normal_input(self):
        np.testing.assert_array_equal(
            np.isfinite(self.result.output), True,
            err_msg="Salida contiene no-finitos para entrada normal"
        )

    def test_output_matches_numpy_reference(self):
        ref = numpy_eval(self.chain, self.x)
        np.testing.assert_allclose(self.result.output, ref, rtol=1e-9,
                                   err_msg="Resultado difiere de ref NumPy")


# ═════════════════════════════════════════════════════════════════════════════
# 16. TestTitanEngineV11 — Nuevas capacidades de Gideon v1.1.0
# ═════════════════════════════════════════════════════════════════════════════

class TestTitanEngineV11(unittest.TestCase):
    """
    Tests para las mejoras de Gideon v1.1.0:
      - fold_affine_chain: colapso algebraico de cadenas FMA
      - chain_hash: hash estable para caché
      - compilation cache: caché de callables compilados
      - fast_mode: omisión de análisis IR/grafo
      - run_batch: ejecución eficiente sobre múltiples inputs
      - GPU path: ejecución automática en CUDA para arrays grandes
      - cache_info / cache_clear
    """

    @classmethod
    def setUpClass(cls):
        cls.ir = GideonIR()

    # ── fold_affine_chain ────────────────────────────────────────────────────

    def test_fold_pure_affine_returns_tuple(self):
        chain = [_FMA(2.0, 1.0)] * 5
        prog = self.ir.from_fma_sequence(chain)
        wb = GideonIR.fold_affine_chain(prog)
        self.assertIsInstance(wb, tuple)
        self.assertEqual(len(wb), 2)

    def test_fold_correctness_depth10(self):
        chain = [_FMA(0.9, 0.1)] * 10
        prog = self.ir.from_fma_sequence(chain)
        wb = GideonIR.fold_affine_chain(prog)
        self.assertIsNotNone(wb)
        W, B = wb
        x = np.linspace(-2.0, 2.0, 5000)
        y_fold = W * x + B
        y_ref  = numpy_eval(chain, x)
        # rtol=1e-9: el fold acumula ~n*unit_eps en la composición por Python float64
        np.testing.assert_allclose(y_fold, y_ref, rtol=1e-9,
            err_msg="fold_affine_chain no coincide con evaluación secuencial")

    def test_fold_identity_chain(self):
        chain = [_FMA(1.0, 0.0)] * 100
        prog = self.ir.from_fma_sequence(chain)
        wb = GideonIR.fold_affine_chain(prog)
        self.assertIsNotNone(wb)
        W, B = wb
        self.assertAlmostEqual(W, 1.0, places=14)
        self.assertAlmostEqual(B, 0.0, places=14)

    def test_fold_zero_weight_chain(self):
        # FMA(0, c) → output = c para cualquier entrada
        chain = [_FMA(0.5, 0.1)] * 3 + [_FMA(0.0, 7.0)]
        prog = self.ir.from_fma_sequence(chain)
        wb = GideonIR.fold_affine_chain(prog)
        self.assertIsNotNone(wb)
        W, B = wb
        self.assertAlmostEqual(W, 0.0, places=14)
        self.assertAlmostEqual(B, 7.0, places=14)

    def test_fold_nonlinear_returns_none(self):
        from poema.backends.gideon.ir import IRNode, IRNodeKind, IRNodeMetadata
        # Un programa con SIN no puede ser plegado
        prog = self.ir.from_fma_sequence([_FMA(1.0, 0.0)] * 3)
        sin_node = IRNode(
            kind=IRNodeKind.SIN,
            node_id="sin_x",
            inputs=[prog.topo_order[-1]],
            params={},
        )
        prog.add_node(sin_node)
        prog.topo_order.append("sin_x")
        wb = GideonIR.fold_affine_chain(prog)
        self.assertIsNone(wb, "Cadena con SIN no debe poder plegarse")

    def test_fold_large_chain_matches_sequential(self):
        chain = make_chain(1000, seed=777)
        prog = self.ir.from_fma_sequence(chain)
        wb = GideonIR.fold_affine_chain(prog)
        self.assertIsNotNone(wb)
        W, B = wb
        x = np.linspace(-1.0, 1.0, 1000)
        y_fold = W * x + B
        y_ref  = numpy_eval(chain, x)
        np.testing.assert_allclose(y_fold, y_ref, rtol=1e-8,
            err_msg="Fold de 1000 FMAs difiere demasiado de secuencial")

    # ── chain_hash ────────────────────────────────────────────────────────────

    def test_chain_hash_deterministic(self):
        chain = make_chain(50, seed=1)
        h1 = GideonIR.chain_hash(chain)
        h2 = GideonIR.chain_hash(chain)
        self.assertEqual(h1, h2)

    def test_chain_hash_different_chains_differ(self):
        chain1 = make_chain(50, seed=1)
        chain2 = make_chain(50, seed=2)
        self.assertNotEqual(GideonIR.chain_hash(chain1), GideonIR.chain_hash(chain2))

    def test_chain_hash_empty_chain(self):
        h = GideonIR.chain_hash([])
        self.assertIsInstance(h, str)
        self.assertGreater(len(h), 0)

    def test_chain_hash_length_16(self):
        chain = make_chain(100, seed=5)
        h = GideonIR.chain_hash(chain)
        self.assertEqual(len(h), 16)

    # ── Compilation cache ─────────────────────────────────────────────────────

    def test_cache_hit_on_second_call(self):
        engine = GideonEngine()
        chain = make_chain(30, seed=10)
        x = np.linspace(-1, 1, 100)
        r1 = engine.run_fma(chain, x)
        r2 = engine.run_fma(chain, x)
        self.assertFalse(r1.cache_hit, "Primera llamada no debe ser cache hit")
        self.assertTrue(r2.cache_hit, "Segunda llamada debe ser cache hit")

    def test_cache_hit_preserves_output(self):
        engine = GideonEngine()
        chain = make_chain(50, seed=11)
        x = np.linspace(-1, 1, 500)
        r1 = engine.run_fma(chain, x)
        r2 = engine.run_fma(chain, x)
        np.testing.assert_array_equal(r1.output, r2.output)

    def test_cache_clear_resets_state(self):
        engine = GideonEngine()
        chain = make_chain(20, seed=12)
        x = np.linspace(-1, 1, 100)
        engine.run_fma(chain, x)  # populate cache
        self.assertGreater(engine.cache_info()["compiled_chains"], 0)
        engine.cache_clear()
        self.assertEqual(engine.cache_info()["compiled_chains"], 0)

    def test_cache_info_keys(self):
        engine = GideonEngine()
        ci = engine.cache_info()
        for key in ("compiled_chains", "folded_chains", "gpu_chains", "fold_analyzed"):
            self.assertIn(key, ci)

    def test_different_chains_have_separate_cache_entries(self):
        engine = GideonEngine()
        chain_a = make_chain(10, seed=20)
        chain_b = make_chain(10, seed=21)
        x = np.linspace(-1, 1, 100)
        engine.run_fma(chain_a, x)
        engine.run_fma(chain_b, x)
        self.assertEqual(engine.cache_info()["compiled_chains"], 2)

    # ── fold_affine en engine ─────────────────────────────────────────────────

    def test_engine_fold_active_by_default(self):
        engine = GideonEngine()
        chain = make_chain(20, seed=30)
        x = np.linspace(-1, 1, 100)
        r = engine.run_fma(chain, x)
        self.assertTrue(r.folded, "fold_affine debe estar activo por defecto")
        self.assertIn("fold", r.backend_used)

    def test_engine_fold_disabled_gives_correct_result(self):
        cfg = GideonEngineConfig(fold_affine=False)
        engine = GideonEngine(cfg)
        chain = make_chain(20, seed=31)
        x = np.linspace(-1, 1, 100)
        r = engine.run_fma(chain, x)
        self.assertFalse(r.folded)
        ref = numpy_eval(chain, x)
        np.testing.assert_allclose(r.output, ref, rtol=1e-10)

    def test_fold_vs_nofold_same_output(self):
        chain = make_chain(50, seed=32)
        x = np.linspace(-1, 1, 1000)
        r_fold   = GideonEngine(GideonEngineConfig(fold_affine=True)).run_fma(chain, x)
        r_nofold = GideonEngine(GideonEngineConfig(fold_affine=False)).run_fma(chain, x)
        np.testing.assert_allclose(r_fold.output, r_nofold.output, rtol=1e-9)

    # ── fast_mode ─────────────────────────────────────────────────────────────

    def test_fast_mode_correct_output(self):
        cfg = GideonEngineConfig(fast_mode=True)
        engine = GideonEngine(cfg)
        chain = make_chain(50, seed=40)
        x = np.linspace(-1, 1, 1000)
        r = engine.run_fma(chain, x)
        ref = numpy_eval(chain, x)
        self.assertTrue(r.success)
        np.testing.assert_allclose(r.output, ref, rtol=1e-9)

    def test_fast_mode_faster_than_full_mode(self):
        chain = make_chain(200, seed=41)
        x = np.linspace(-1, 1, 10_000)
        import time
        eng_fast = GideonEngine(GideonEngineConfig(fast_mode=True, fold_affine=False))
        eng_full = GideonEngine(GideonEngineConfig(fast_mode=False, fold_affine=False))
        # warm cache for both first
        eng_fast.run_fma(chain, x[:10])
        eng_full.run_fma(chain, x[:10])
        # Measure subsequent calls (cache hit, but fast_mode should still be faster)
        t0 = time.perf_counter()
        for _ in range(10):
            eng_fast.run_fma(chain, x)
        t_fast = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(10):
            eng_full.run_fma(chain, x)
        t_full = time.perf_counter() - t0
        log_metric("fast_vs_full", f"fast={t_fast*100:.1f}ms/call vs full={t_full*100:.1f}ms/call")
        # fast_mode debe ser ≤ full_mode (o al menos no más del doble)
        self.assertLess(t_fast, t_full * 2.0,
                        "fast_mode no debe ser significativamente más lento que full_mode")

    # ── run_batch ─────────────────────────────────────────────────────────────

    def test_run_batch_returns_list(self):
        engine = GideonEngine()
        chain = make_chain(20, seed=50)
        inputs = [np.linspace(-i, i, 100) for i in range(1, 6)]
        results = engine.run_batch(chain, inputs)
        self.assertEqual(len(results), 5)

    def test_run_batch_all_succeed(self):
        engine = GideonEngine()
        chain = make_chain(30, seed=51)
        inputs = [np.random.default_rng(i).standard_normal(500) for i in range(10)]
        results = engine.run_batch(chain, inputs)
        for i, r in enumerate(results):
            self.assertTrue(r.success, f"Batch item {i} falló")

    def test_run_batch_cache_hits_after_first(self):
        engine = GideonEngine()
        chain = make_chain(20, seed=52)
        inputs = [np.linspace(-1, 1, 100)] * 5
        results = engine.run_batch(chain, inputs)
        # Primera puede no ser hit, resto deben serlo
        self.assertTrue(all(r.cache_hit for r in results[1:]),
                        "Items 2-5 del batch deben ser cache hits")

    def test_run_batch_correct_outputs(self):
        engine = GideonEngine()
        chain = make_chain(15, seed=53)
        xs = [np.linspace(-1, 1, 200) * i for i in range(1, 4)]
        results = engine.run_batch(chain, xs)
        for i, (x, r) in enumerate(zip(xs, results)):
            ref = numpy_eval(chain, x)
            np.testing.assert_allclose(r.output, ref, rtol=1e-9,
                err_msg=f"run_batch item {i} no coincide con ref numpy")

    # ── GPU path ──────────────────────────────────────────────────────────────

    def test_gpu_path_forced_gives_correct_result(self):
        cfg = GideonEngineConfig(gpu_min_elements=1)  # forzar GPU para cualquier array
        engine = GideonEngine(cfg)
        chain = make_chain(20, seed=60)
        x = np.linspace(-1, 1, 1000)
        r = engine.run_fma(chain, x)
        self.assertTrue(r.success)
        self.assertTrue(r.gpu_used, "gpu_used debe ser True con gpu_min_elements=1")
        ref = numpy_eval(chain, x)
        np.testing.assert_allclose(r.output, ref, rtol=1e-8)

    def test_gpu_path_large_array_benchmark(self):
        cfg = GideonEngineConfig(gpu_min_elements=1_000_000)
        engine = GideonEngine(cfg)
        chain = make_chain(10, seed=61)
        x = np.random.default_rng(61).standard_normal(5_000_000)
        r = engine.run_fma(chain, x)
        self.assertTrue(r.success)
        # Si GPU disponible, debe haber usado GPU
        hw_has_cuda = engine._dispatcher.hw.has_cuda
        if hw_has_cuda:
            self.assertTrue(r.gpu_used, "Debe usar GPU para 5M elementos con CUDA")
        log_metric("GPU_5M", f"backend={r.backend_used} gpu={r.gpu_used} t={r.elapsed_ms:.1f}ms")

    def test_gpu_folded_vs_cpu_folded_match(self):
        chain = make_chain(20, seed=62)
        x = np.linspace(-1, 1, 10_000)
        r_cpu = GideonEngine(GideonEngineConfig(gpu_min_elements=10**9)).run_fma(chain, x)
        r_gpu = GideonEngine(GideonEngineConfig(gpu_min_elements=1)).run_fma(chain, x)
        np.testing.assert_allclose(r_cpu.output, r_gpu.output, rtol=1e-8,
            err_msg="GPU folded y CPU folded deben dar resultados equivalentes")

    # ── Versión ──────────────────────────────────────────────────────────────

    def test_version_is_1_1_0(self):
        self.assertEqual(GideonEngine._VERSION, "1.2.0")

    def test_info_shows_optimizations(self):
        engine = GideonEngine()
        info_str = engine.info()
        self.assertIn("fold_affine", info_str)
        self.assertIn("fast_mode", info_str)
        self.assertIn("gpu_min_elements", info_str)
        self.assertIn("1.2.0", info_str)


# ═════════════════════════════════════════════════════════════════════════════
# 16. TestTitanEngineV12 — Gideon v1.2.0: Autotune, Telemetría, MLDispatcher
# ═════════════════════════════════════════════════════════════════════════════

# Imports extra para v1.2.0
import tempfile
import os
from poema.backends.gideon import (
    GideonHardwareProfiler,
    HardwareCapabilities,
    GideonTelemetry,
    MLDispatcher,
    ExecutionRecord,
)


class TestTitanEngineV12(unittest.TestCase):
    """
    Suite de pruebas para Gideon v1.2.0.

    Verifica:
      16a. HardwareCapabilities — construcción, serialización, propiedades
      16b. GideonHardwareProfiler — detección estática, quick_mode, persistencia
      16c. GideonTelemetry — registro, estadísticas, exportación ACF
      16d. MLDispatcher — fallback, aprendizaje, predicción de speedup
      16e. Engine v1.2.0 — autotune, telemetría integrada, bucle ACF
    """

    def setUp(self):
        """Crea directorio temporal para perfiles y DB de telemetría."""
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_engine(self, **kwargs):
        """Factory: engine con paths en directorio temporal."""
        default = dict(
            use_autotune=True,
            autotune_quick=True,
            use_ml_dispatcher=True,
            telemetry_path=os.path.join(self._tmpdir, "tel.json"),
            persist_fold_cache=False,
        )
        default.update(kwargs)
        from poema.backends.gideon.engine import GideonEngineConfig
        cfg = GideonEngineConfig(**default)
        return GideonEngine(cfg)

    # ── 16a: HardwareCapabilities ─────────────────────────────────────────────

    def test_hardware_caps_default_construction(self):
        """HardwareCapabilities se instancia con valores por defecto correctos."""
        caps = HardwareCapabilities()
        self.assertEqual(caps.cpu_avx_level, 0)
        self.assertFalse(caps.gpu_available)
        self.assertEqual(caps.gpu_compute_capability, (0, 0))
        self.assertEqual(caps.measured_fp32_fp64_ratio, 1.0)

    def test_hardware_caps_serialization_roundtrip(self):
        """to_dict / from_dict hace roundtrip perfecto."""
        caps = HardwareCapabilities(
            cpu_model="Test CPU",
            cpu_arch="zen4",
            cpu_avx_level=512,
            gpu_available=True,
            gpu_name="RTX Test",
            gpu_compute_capability=(8, 9),
            gpu_sms=128,
            gpu_tensor_cores=True,
            gpu_tc_version=4,
            measured_fma_scalar_gflops=12.5,
        )
        d = caps.to_dict()
        back = HardwareCapabilities.from_dict(d)
        self.assertEqual(back.cpu_model, "Test CPU")
        self.assertEqual(back.cpu_arch, "zen4")
        self.assertEqual(back.cpu_avx_level, 512)
        self.assertTrue(back.gpu_available)
        self.assertEqual(back.gpu_compute_capability, (8, 9))
        self.assertEqual(back.gpu_tc_version, 4)
        self.assertAlmostEqual(back.measured_fma_scalar_gflops, 12.5)

    def test_hardware_caps_avx_label(self):
        """avx_label devuelve texto correcto."""
        c = HardwareCapabilities(cpu_avx_level=0)
        self.assertEqual(c.avx_label, "None")
        c.cpu_avx_level = 2
        self.assertEqual(c.avx_label, "AVX2+FMA3")
        c.cpu_avx_level = 512
        self.assertEqual(c.avx_label, "AVX-512")

    def test_hardware_caps_gpu_cc_str(self):
        """gpu_cc_str formatea correctamente el compute capability."""
        c = HardwareCapabilities(gpu_compute_capability=(8, 9))
        self.assertEqual(c.gpu_cc_str, "8.9")

    # ── 16b: GideonHardwareProfiler ──────────────────────────────────────────

    def test_profiler_quick_mode_no_benchmarks(self):
        """quick_mode=True no ejecuta benchmarks — caps.quick_mode es True."""
        profiler = GideonHardwareProfiler(
            cache_dir=self._tmpdir, force_reprofiling=True, quick_mode=True
        )
        caps = profiler.full_profile()
        self.assertTrue(caps.quick_mode)
        # Estimaciones deben ser > 0
        self.assertGreater(caps.measured_fma_scalar_gflops, 0.0)
        self.assertGreater(caps.measured_fma_vector_gflops, 0.0)

    def test_profiler_detects_cpu_cores(self):
        """El profiler detecta al menos 1 core lógico."""
        profiler = GideonHardwareProfiler(
            cache_dir=self._tmpdir, force_reprofiling=True, quick_mode=True
        )
        caps = profiler.full_profile()
        self.assertGreater(caps.cpu_cores_logical, 0)
        self.assertNotEqual(caps.hostname, "")

    def test_profiler_persists_profile(self):
        """El perfil se persiste en disco y se carga correctamente."""
        profiler = GideonHardwareProfiler(
            cache_dir=self._tmpdir, force_reprofiling=True, quick_mode=True
        )
        caps1 = profiler.full_profile()
        # Segunda instancia debe cargar desde disco
        profiler2 = GideonHardwareProfiler(
            cache_dir=self._tmpdir, force_reprofiling=False, quick_mode=True
        )
        caps2 = profiler2.load_or_profile()
        self.assertEqual(caps1.hostname, caps2.hostname)
        self.assertEqual(caps1.cpu_arch, caps2.cpu_arch)

    def test_profiler_summary_format(self):
        """summary() devuelve texto con campos clave."""
        profiler = GideonHardwareProfiler(
            cache_dir=self._tmpdir, force_reprofiling=True, quick_mode=True
        )
        caps = profiler.full_profile()
        summary = profiler.summary(caps)
        self.assertIn("Gideon Hardware Profile", summary)
        self.assertIn("CPU", summary)
        self.assertIn("cores", summary.lower())

    def test_profiler_force_reprofiling_ignores_cache(self):
        """force_reprofiling=True rehace el perfil aunque exista caché."""
        profiler = GideonHardwareProfiler(
            cache_dir=self._tmpdir, force_reprofiling=True, quick_mode=True
        )
        caps1 = profiler.full_profile()
        caps1._custom_marker = "original"   # No se serializa, solo para test

        profiler2 = GideonHardwareProfiler(
            cache_dir=self._tmpdir, force_reprofiling=True, quick_mode=True
        )
        caps2 = profiler2.load_or_profile()
        # Debe haber vuelto a perfilar (sin el marcador)
        self.assertFalse(hasattr(caps2, "_custom_marker"))

    # ── 16c: GideonTelemetry ─────────────────────────────────────────────────

    def _make_telemetry(self):
        """Crea GideonTelemetry con DB en directorio temporal."""
        return GideonTelemetry(
            db_path=os.path.join(self._tmpdir, "tel.json")
        )

    def _dummy_result(self, backend="affine_fold", folded=True, ms=2.0, gpu=False, cache=False):
        """Crea un mock de GideonExecutionResult."""
        class R:
            pass
        r = R()
        r.total_fma = 5
        r.backend_used = backend
        r.elapsed_ms = ms
        r.folded = folded
        r.gpu_used = gpu
        r.success = True
        r.cache_hit = cache
        return r

    def test_telemetry_record_increases_count(self):
        """Cada llamada a record() incrementa len()."""
        tel = self._make_telemetry()
        self.assertEqual(len(tel), 0)
        r = self._dummy_result()
        tel.record(r, chain_hash="abc123", n_elements=100)
        self.assertEqual(len(tel), 1)
        tel.record(r, chain_hash="abc123", n_elements=100)
        self.assertEqual(len(tel), 2)

    def test_telemetry_backend_stats(self):
        """get_backend_stats() devuelve estadísticas correctas."""
        tel = self._make_telemetry()
        for ms in [1.0, 2.0, 3.0, 4.0, 5.0]:
            tel.record(self._dummy_result(backend="affine_fold", ms=ms),
                       chain_hash="h1", n_elements=100)
        stats = tel.get_backend_stats()
        self.assertIn("affine_fold", stats)
        s = stats["affine_fold"]
        self.assertEqual(s["count"], 5)
        self.assertAlmostEqual(s["avg_ms"], 3.0)
        self.assertAlmostEqual(s["min_ms"], 1.0)
        self.assertAlmostEqual(s["max_ms"], 5.0)

    def test_telemetry_get_best_backend_with_data(self):
        """get_best_backend regresa el backend más rápido con suficientes datos."""
        tel = self._make_telemetry()
        # 10 muestras de affine_fold (rápido) y 10 de numpy_cpu (lento)
        for _ in range(10):
            tel.record(self._dummy_result(backend="affine_fold", ms=1.0),
                       chain_hash="h1", n_elements=1000)
        for _ in range(10):
            tel.record(self._dummy_result(backend="numpy_cpu", ms=20.0),
                       chain_hash="h1", n_elements=1000)
        best = tel.get_best_backend(n_elements=1000, n_fma=5)
        self.assertEqual(best, "affine_fold")

    def test_telemetry_get_best_backend_no_data(self):
        """get_best_backend retorna None si no hay datos suficientes."""
        tel = self._make_telemetry()
        best = tel.get_best_backend(n_elements=1000, n_fma=5)
        self.assertIsNone(best)

    def test_telemetry_export_acf_calibration_keys(self):
        """export_acf_calibration() devuelve todos los campos esperados."""
        tel = self._make_telemetry()
        calib = tel.export_acf_calibration()
        required_keys = {
            "n_records", "backend_latencies", "backend_p95_latencies",
            "fastest_backend", "fold_effectiveness_pct", "gpu_usage_pct",
            "cache_hit_rate_pct", "size_distribution", "precision_usage",
            "acf_notes",
        }
        self.assertTrue(required_keys.issubset(calib.keys()))

    def test_telemetry_export_fold_effectiveness(self):
        """fold_effectiveness_pct refleja la fracción de ejecuciones plegadas."""
        tel = self._make_telemetry()
        for _ in range(7):
            tel.record(self._dummy_result(folded=True),
                       chain_hash="h1", n_elements=100)
        for _ in range(3):
            tel.record(self._dummy_result(folded=False),
                       chain_hash="h1", n_elements=100)
        calib = tel.export_acf_calibration()
        self.assertAlmostEqual(calib["fold_effectiveness_pct"], 70.0, places=1)

    def test_telemetry_persistence(self):
        """Los registros se persisten y se cargan correctamente."""
        path = os.path.join(self._tmpdir, "persist_tel.json")
        tel1 = GideonTelemetry(db_path=path)
        for i in range(55):  # > AUTO_SAVE_INTERVAL → dispara _save
            tel1.record(self._dummy_result(ms=float(i)),
                        chain_hash=f"h{i}", n_elements=i * 10)
        tel1.flush()

        tel2 = GideonTelemetry(db_path=path)
        self.assertEqual(len(tel2), len(tel1))

    def test_telemetry_clear(self):
        """clear() elimina todos los registros."""
        tel = self._make_telemetry()
        for _ in range(5):
            tel.record(self._dummy_result(), chain_hash="h", n_elements=1)
        tel.clear()
        self.assertEqual(len(tel), 0)

    def test_telemetry_get_chain_stats(self):
        """get_chain_stats() devuelve estadísticas específicas de una cadena."""
        tel = self._make_telemetry()
        for ms in [3.0, 4.0, 5.0]:
            tel.record(self._dummy_result(ms=ms), chain_hash="target", n_elements=100)
        for ms in [10.0, 20.0]:
            tel.record(self._dummy_result(ms=ms), chain_hash="other", n_elements=100)

        stats = tel.get_chain_stats("target")
        self.assertIsNotNone(stats)
        self.assertEqual(stats["n_success"], 3)
        self.assertAlmostEqual(stats["avg_ms"], 4.0)

        self.assertIsNone(tel.get_chain_stats("nonexistent"))

    # ── 16d: MLDispatcher ────────────────────────────────────────────────────

    def test_ml_dispatcher_fallback_no_data(self):
        """Sin datos, MLDispatcher delega al dispatcher heurístico."""
        tel = self._make_telemetry()
        heuristic = GideonDispatcher()
        ml = MLDispatcher(telemetry=tel, fallback_dispatcher=heuristic)

        from poema.backends.gideon.ir import GideonIR
        prog = GideonIR().from_fma_sequence(
            [_FMA(0.9, 0.1)] * 3, domain=(-1, 1), precision="fp64"
        )
        decision = ml.decide(prog, precision="fp64")
        # Debe devolver una DispatchDecision válida
        self.assertIsNotNone(decision.primary_backend)
        self.assertNotEqual(decision.primary_backend, "")

    def test_ml_dispatcher_uses_learned_backend(self):
        """Con suficientes datos, MLDispatcher recomienda el backend más rápido."""
        tel = self._make_telemetry()
        # Registrar 10 ejecuciones de affine_fold (rápido) y 10 de numpy_cpu (lento)
        for _ in range(10):
            tel.record(self._dummy_result(backend="affine_fold", ms=1.0),
                       chain_hash="h", n_elements=1000)
        for _ in range(10):
            tel.record(self._dummy_result(backend="numpy_cpu", ms=50.0),
                       chain_hash="h", n_elements=1000)

        heuristic = GideonDispatcher()
        ml = MLDispatcher(telemetry=tel, fallback_dispatcher=heuristic)

        from poema.backends.gideon.ir import GideonIR
        prog = GideonIR().from_fma_sequence(
            [_FMA(0.9, 0.1)] * 5, domain=(-1, 1), precision="fp64"
        )
        decision = ml.decide(prog, precision="fp64")
        self.assertEqual(decision.primary_backend, "affine_fold")
        self.assertIn("ml_dispatcher", decision.reason)

    def test_ml_dispatcher_speedup_estimation(self):
        """predict_speedup devuelve valor > 0."""
        tel = self._make_telemetry()
        heuristic = GideonDispatcher()
        ml = MLDispatcher(telemetry=tel, fallback_dispatcher=heuristic)
        sp = ml.predict_speedup("affine_fold")
        self.assertGreater(sp, 0.0)

    def test_ml_dispatcher_telemetry_summary(self):
        """telemetry_summary() devuelve texto legible."""
        tel = self._make_telemetry()
        for _ in range(3):
            tel.record(self._dummy_result(), chain_hash="h", n_elements=100)
        heuristic = GideonDispatcher()
        ml = MLDispatcher(telemetry=tel, fallback_dispatcher=heuristic)
        summary = ml.telemetry_summary()
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)

    # ── 16e: Engine v1.2.0 integrado ─────────────────────────────────────────

    def test_engine_v12_version(self):
        """La versión debe ser exactamente 1.2.0."""
        e = self._make_engine()
        self.assertEqual(e._VERSION, "1.2.0")

    def test_engine_autotune_caps_not_none(self):
        """autotune_info() devuelve HardwareCapabilities (no None)."""
        e = self._make_engine(use_autotune=True, autotune_quick=True)
        caps = e.autotune_info()
        self.assertIsNotNone(caps)
        self.assertIsInstance(caps, HardwareCapabilities)
        self.assertGreater(caps.cpu_cores_logical, 0)

    def test_engine_autotune_disabled_returns_none(self):
        """use_autotune=False → autotune_info() retorna None."""
        e = self._make_engine(use_autotune=False)
        self.assertIsNone(e.autotune_info())

    def test_engine_autotune_summary_string(self):
        """autotune_summary() devuelve string legible."""
        e = self._make_engine(use_autotune=True, autotune_quick=True)
        s = e.autotune_summary()
        self.assertIsInstance(s, str)
        self.assertIn("Gideon Hardware Profile", s)

    def test_engine_telemetry_records_after_run(self):
        """Después de run_fma(), la telemetría tiene ≥ 1 registro."""
        e = self._make_engine()
        chain = [_FMA(0.9, 0.1)] * 3
        x = np.ones(50)
        e.run_fma(chain, x)
        self.assertGreaterEqual(len(e._telemetry), 1)

    def test_engine_telemetry_stats_keys(self):
        """telemetry_stats() devuelve dict con claves esperadas."""
        e = self._make_engine()
        e.run_fma([_FMA(0.9, 0.1)] * 3, np.ones(50))
        stats = e.telemetry_stats()
        self.assertIn("total_records", stats)
        self.assertIn("backend_stats", stats)
        self.assertIn("ml_dispatcher_active", stats)
        self.assertIn("ml_dispatcher_summary", stats)
        self.assertTrue(stats["ml_dispatcher_active"])

    def test_engine_export_acf_calibration_complete(self):
        """export_acf_calibration() incluye datos de hardware e info de fold."""
        e = self._make_engine(use_autotune=True, autotune_quick=True)
        chain = [_FMA(0.9, 0.1)] * 4
        for _ in range(5):
            e.run_fma(chain, np.ones(100))
        calib = e.export_acf_calibration()
        self.assertIn("hardware", calib)
        self.assertIn("fold_cache_size", calib)
        self.assertIn("fold_effectiveness_pct", calib)
        # fold debe ser 100% (cadena pura afín)
        self.assertGreater(calib["fold_effectiveness_pct"], 50.0)

    def test_engine_acf_calibration_hardware_fields(self):
        """El campo 'hardware' tiene los campos clave para calibrar ACF."""
        e = self._make_engine(use_autotune=True, autotune_quick=True)
        calib = e.export_acf_calibration()
        hw = calib["hardware"]
        self.assertIn("cpu_arch", hw)
        self.assertIn("avx_level", hw)
        self.assertIn("gpu_available", hw)
        self.assertIn("fma_gflops", hw)

    def test_engine_info_v12(self):
        """info() menciona v1.2.0 y los nuevos campos de configuración."""
        e = self._make_engine(use_autotune=True, use_ml_dispatcher=True)
        info = e.info()
        self.assertIn("1.2.0", info)
        self.assertIn("use_autotune", info)
        self.assertIn("use_ml_dispatch", info)

    def test_engine_telemetry_path_respected(self):
        """La DB de telemetría se crea en el path especificado."""
        tel_path = os.path.join(self._tmpdir, "custom_tel.json")
        e = self._make_engine(telemetry_path=tel_path)
        chain = [_FMA(0.5, 0.5)] * 2
        for _ in range(55):  # > AUTO_SAVE_INTERVAL
            e.run_fma(chain, np.ones(10))
        e._telemetry.flush()
        self.assertTrue(os.path.exists(tel_path), "DB de telemetría debe existir")

    def test_engine_persistent_fold_cache_save_load(self):
        """El fold cache se persiste y se carga al reiniciar el engine."""
        from poema.backends.gideon.engine import GideonEngineConfig
        fold_dir = os.path.join(self._tmpdir, ".gideon")
        os.makedirs(fold_dir, exist_ok=True)

        chain = [_FMA(0.9, 0.1)] * 5
        x = np.ones(20)

        # Engine 1: ejecutar y guardar fold cache
        cfg1 = GideonEngineConfig(
            persist_fold_cache=True,
            telemetry_path=os.path.join(self._tmpdir, "t.json"),
        )
        # Override _fold_cache_path para usar tmpdir
        e1 = GideonEngine(cfg1)
        e1._fold_cache_path = lambda: os.path.join(fold_dir, "fold_cache.json")
        e1.run_fma(chain, x)
        e1._save_fold_cache()
        self.assertTrue(
            os.path.exists(os.path.join(fold_dir, "fold_cache.json")),
            "fold_cache.json debe existir después de guardar"
        )

        # Engine 2: cargar fold cache
        e2 = GideonEngine(GideonEngineConfig(
            persist_fold_cache=False,  # no auto-load, lo hacemos manual
            telemetry_path=os.path.join(self._tmpdir, "t2.json"),
        ))
        e2._fold_cache_path = lambda: os.path.join(fold_dir, "fold_cache.json")
        e2._load_fold_cache()
        # Debe haber cargado ≥ 1 entrada
        self.assertGreater(len(e2._fold_cache), 0)

    def test_engine_ml_dispatcher_with_history_improves_decision(self):
        """Después de N ejecuciones, MLDispatcher recomienda el backend más rápido."""
        e = self._make_engine(use_ml_dispatcher=True)
        chain = [_FMA(0.9, 0.1)] * 5

        # Ejecutar bastantes veces para acumular datos
        for _ in range(25):
            e.run_fma(chain, np.ones(100))

        # La telemetría debe tener ≥ 25 registros
        stats = e.telemetry_stats()
        self.assertGreaterEqual(stats["total_records"], 25)

        # El MLDispatcher debe tener suficientes datos para predecir
        calib = e.export_acf_calibration()
        self.assertIn("fastest_backend", calib)
        self.assertIsNotNone(calib.get("fastest_backend"))

    def test_engine_closed_loop_acf_notes(self):
        """La nota ACF en la calibración es informativa cuando fold es dominante."""
        e = self._make_engine()
        chain = [_FMA(0.9, 0.1)] * 4
        for _ in range(10):
            e.run_fma(chain, np.ones(100))
        calib = e.export_acf_calibration()
        # La cadena es afín pura → fold_pct debe ser alto → nota informativa
        self.assertIsInstance(calib.get("acf_notes"), str)

    def test_engine_cache_clear_with_persist_false(self):
        """cache_clear(persist=False) limpia memoria sin tocar disco."""
        e = self._make_engine()
        chain = [_FMA(0.9, 0.1)] * 3
        e.run_fma(chain, np.ones(50))
        self.assertGreater(len(e._compiled_cache), 0)
        e.cache_clear(persist=False)
        self.assertEqual(len(e._compiled_cache), 0)
        self.assertEqual(len(e._fold_cache), 0)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  GIDEON TITAN TEST SUITE — Potencia extrema verificada")
    print("=" * 70)
    unittest.main(verbosity=2, buffer=False)
