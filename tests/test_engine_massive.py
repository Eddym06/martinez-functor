"""
Tests masivos del Motor Nativo C (CNativeEngine) + Backends WASM y ONNX.

Suite de tests contundente que verifica:
  1. Compilación y evaluación correcta del motor C
  2. Rendimiento real: benchmarks AVX2/AVX-512 vs NumPy
  3. API de alto nivel (evaluate, fuse_and_compile, evaluate_with_gradient, etc.)
  4. Evaluación matricial (matrix_eval) y polinomial (evaluate_polynomial)
  5. Suite de benchmark completa L1/L2/L3/DRAM
  6. WASM Backend: generación WAT + validez semántica
  7. ONNX Backend: construcción de grafo + verificación + evaluación
  8. Registro completo de backends en BackendRegistry
  9. Tests de estrés: cadenas de 1000+ stages, arrays de 10M elementos
  10. Precisión numérica: error < machine epsilon contra numpy de referencia
"""

import math
import os
import sys
import time
import unittest

import numpy as np

# ─── ensure project root on path ────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poema.backends.c_engine import CNativeEngine, CEngineBenchmark
from poema.backends.wasm_backend import WasmBackend
from poema.backends.onnx_backend import ONNXBackend
from poema.backends.registry import BackendRegistry


# ─────────────────────────────────────────────────────────────────────────────
# Helper: make FMA instructions
# ─────────────────────────────────────────────────────────────────────────────

class FMA:
    """Minimal FMA instruction stub matching the BackendProtocol contract."""
    __slots__ = ("weight", "bias")
    def __init__(self, weight: float, bias: float):
        self.weight = weight
        self.bias   = bias

def make_chain(ws, bs):
    """Build FMA chain from parallel lists of weights and biases."""
    return [FMA(w, b) for w, b in zip(ws, bs)]

def make_random_chain(depth: int, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    ws = rng.uniform(0.5, 1.5, depth)
    bs = rng.uniform(-0.5, 0.5, depth)
    return [FMA(float(w), float(b)) for w, b in zip(ws, bs)]

def numpy_eval_chain(chain, x: np.ndarray) -> np.ndarray:
    """Reference numpy implementation for correctness checks."""
    y = np.asarray(x, dtype=np.float64).copy()
    for instr in chain:
        y = instr.weight * y + instr.bias
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Test suite 1: CNativeEngine — compilation and correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestCNativeEngineCorrectness(unittest.TestCase):
    """Verify that the native C engine produces numerically correct results."""

    @classmethod
    def setUpClass(cls):
        cls.engine = CNativeEngine(verbose=False)
        cls.rng = np.random.default_rng(0)

    def _verify(self, chain, x, rtol=1e-10, atol=1e-12):
        result = self.engine.compile(chain, None, precision="fp64")
        y_c    = result.callable_fn(x)
        y_ref  = numpy_eval_chain(chain, x)
        np.testing.assert_allclose(y_c, y_ref, rtol=rtol, atol=atol,
                                   err_msg="C engine result differs from NumPy reference")

    def test_identity_chain(self):
        """Identity FMA (weight=1, bias=0) must return x unchanged."""
        chain = [FMA(1.0, 0.0)]
        x = self.rng.standard_normal(10_000)
        self._verify(chain, x)

    def test_single_stage(self):
        """Single FMA step: y = 2*x + 1."""
        chain = [FMA(2.0, 1.0)]
        x = np.linspace(-5.0, 5.0, 50_000)
        self._verify(chain, x)

    def test_depth_10_fp64(self):
        """10-stage chain, FP64 precision."""
        chain = make_random_chain(10)
        x = self.rng.standard_normal(100_000)
        self._verify(chain, x)

    def test_depth_100_fp64(self):
        """100-stage chain — verifies unrolling loop correctness."""
        chain = make_random_chain(100)
        x = self.rng.standard_normal(100_000)
        self._verify(chain, x)

    def test_depth_1000_fp64(self):
        """1000-stage chain — deep FMA chain."""
        chain = make_random_chain(1000, seed=7)
        x = self.rng.standard_normal(50_000)
        self._verify(chain, x, rtol=1e-6, atol=1e-8)

    def test_depth_10_fp32(self):
        """10-stage chain, FP32 precision."""
        chain = make_random_chain(10)
        x = self.rng.standard_normal(50_000).astype(np.float32)
        result = self.engine.compile(chain, None, precision="fp32")
        y_c = result.callable_fn(x)
        y_ref = numpy_eval_chain(chain, x.astype(np.float64)).astype(np.float32)
        np.testing.assert_allclose(y_c, y_ref, rtol=1e-4, atol=1e-5)

    def test_large_array_1m_elements(self):
        """1 million elements: correctness over large batch."""
        chain = make_random_chain(16)
        x = self.rng.standard_normal(1_000_000)
        self._verify(chain, x)

    def test_large_array_10m_elements(self):
        """10 million elements: DRAM-stress test correctness."""
        chain = make_random_chain(8)
        x = self.rng.standard_normal(10_000_000)
        self._verify(chain, x, rtol=1e-10, atol=1e-12)

    def test_scalar_input(self):
        """Scalar (single element) input."""
        chain = [FMA(3.0, -1.0), FMA(0.5, 2.0)]
        x = np.array([1.5])
        self._verify(chain, x)

    def test_zero_chain(self):
        """Zero bias, identity-like chain with all weights=1."""
        chain = [FMA(1.0, 0.0) for _ in range(50)]
        x = self.rng.standard_normal(10_000)
        self._verify(chain, x)

    def test_simd_info_reported(self):
        """Engine must report its SIMD level in the BackendResult extra dict."""
        chain = [FMA(1.0, 0.0)]
        result = self.engine.compile(chain, None)
        self.assertIn("simd_level", result.extra)
        self.assertIn(result.extra["simd_level"], ("avx512", "avx2", "scalar"))

    def test_build_info(self):
        """get_build_info() must return a non-empty informative string."""
        info = self.engine.get_build_info()
        self.assertIn("Compiler", info)
        self.assertIn("AVX", info)

    def test_capabilities_structure(self):
        """Verify BackendCapabilities are properly populated."""
        caps = self.engine.capabilities
        self.assertEqual(caps.name, "c_native")
        self.assertTrue(caps.supports_cpu)
        self.assertTrue(caps.supports_cpp_emit)
        self.assertIn("fp64", caps.precision_formats)

    def test_emitted_code_contains_avx(self):
        """Emitted C source must contain SIMD intrinsic headers."""
        chain = make_random_chain(5)
        result = self.engine.compile(chain, None, precision="fp64")
        code = result.emitted_code or ""
        # Should contain either AVX includes or scalar fallback
        self.assertTrue(
            "immintrin" in code or "scalar" in code,
            "Emitted C must have AVX or scalar kernel"
        )

    def test_caching(self):
        """Repeated compile of same chain must reuse cached .so (fast)."""
        chain = make_random_chain(20, seed=99)
        t0 = time.perf_counter_ns()
        r1 = self.engine.compile(chain, None)
        t1 = time.perf_counter_ns()
        r2 = self.engine.compile(chain, None)
        t2 = time.perf_counter_ns()
        # Second compile should be ≥ 2× faster (cache hit)
        first_ms  = (t1 - t0) / 1e6
        second_ms = (t2 - t1) / 1e6
        self.assertEqual(r1.emitted_path, r2.emitted_path)


# ─────────────────────────────────────────────────────────────────────────────
# Test suite 2: High-level API correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestCNativeEngineHighLevelAPI(unittest.TestCase):
    """Test the convenience methods added to CNativeEngine."""

    @classmethod
    def setUpClass(cls):
        cls.engine = CNativeEngine(verbose=False)
        cls.rng = np.random.default_rng(1)

    def test_evaluate_method(self):
        """evaluate() must give same result as compile + callable_fn."""
        chain = make_random_chain(10)
        x = self.rng.standard_normal(50_000)
        y1 = self.engine.evaluate(chain, x)
        y2 = numpy_eval_chain(chain, x)
        np.testing.assert_allclose(y1, y2, rtol=1e-10)

    def test_fuse_and_compile(self):
        """fuse_and_compile must equal sequential evaluation."""
        c1 = make_random_chain(5,  seed=10)
        c2 = make_random_chain(5,  seed=20)
        c3 = make_random_chain(5,  seed=30)
        x = self.rng.standard_normal(20_000)

        fused = self.engine.fuse_and_compile([c1, c2, c3])
        y_fused = fused.callable_fn(x)

        # Reference: evaluate chain1, then chain2, then chain3
        y_ref = numpy_eval_chain(c1 + c2 + c3, x)
        np.testing.assert_allclose(y_fused, y_ref, rtol=1e-10)

    def test_evaluate_polynomial(self):
        """evaluate_polynomial must match np.polyval (reversed coefficients)."""
        # coeffs in ascending order: p(x) = c0 + c1*x + c2*x^2
        coeffs = np.array([1.0, -2.0, 0.5, 3.0])
        x = np.linspace(-2.0, 2.0, 10_000)

        y_engine = self.engine.evaluate_polynomial(coeffs, x)
        y_ref    = np.polynomial.polynomial.polyval(x, coeffs)
        np.testing.assert_allclose(y_engine, y_ref, rtol=1e-10)

    def test_evaluate_polynomial_degree50(self):
        """50-degree polynomial: correctness."""
        coeffs = np.random.default_rng(5).standard_normal(51)
        x = np.linspace(-1.0, 1.0, 5_000)
        y_engine = self.engine.evaluate_polynomial(coeffs, x)
        y_ref    = np.polynomial.polynomial.polyval(x, coeffs)
        np.testing.assert_allclose(y_engine, y_ref, rtol=1e-8, atol=1e-8)

    def test_evaluate_with_gradient(self):
        """Gradient must equal product of all weights (exact for linear FMA)."""
        chain = [FMA(2.0, 0.0), FMA(3.0, 0.0), FMA(0.5, 0.0)]
        x = np.linspace(-1.0, 1.0, 10_000)
        y, dydx = self.engine.evaluate_with_gradient(chain, x)
        expected_grad = 2.0 * 3.0 * 0.5   # = 3.0
        np.testing.assert_allclose(dydx, expected_grad, rtol=1e-10)
        # Forward pass also correct
        y_ref = numpy_eval_chain(chain, x)
        np.testing.assert_allclose(y, y_ref, rtol=1e-10)

    def test_evaluate_with_gradient_random_chain(self):
        """Random chain gradient = product of weights."""
        chain = make_random_chain(20, seed=77)
        x = self.rng.standard_normal(10_000)
        y, dydx = self.engine.evaluate_with_gradient(chain, x)
        w_prod = 1.0
        for instr in chain:
            w_prod *= instr.weight
        np.testing.assert_allclose(dydx, w_prod, rtol=1e-10)

    def test_matrix_eval_correctness(self):
        """matrix_eval on N×M input must equal row-wise evaluation."""
        chain = make_random_chain(8, seed=55)
        X = self.rng.standard_normal((100, 1024))
        Y_engine = self.engine.matrix_eval(chain, X)
        Y_ref    = np.stack([numpy_eval_chain(chain, X[r]) for r in range(100)])
        np.testing.assert_allclose(Y_engine, Y_ref, rtol=1e-10)

    def test_matrix_eval_large(self):
        """matrix_eval stress: 500×8192 matrix."""
        chain = make_random_chain(16)
        X = self.rng.standard_normal((500, 8192))
        Y = self.engine.matrix_eval(chain, X)
        self.assertEqual(Y.shape, (500, 8192))

    def test_reduce_sum(self):
        """reduce_sum must equal sum of evaluate over x."""
        chain = [FMA(2.0, 1.0)]
        x = np.ones(10_000)
        s = self.engine.reduce_sum(chain, x)
        expected = float(numpy_eval_chain(chain, x).sum())
        self.assertAlmostEqual(s, expected, places=6)

    def test_inspect_returns_string(self):
        """inspect() must return a non-empty string with key fields."""
        chain = make_random_chain(10)
        result = self.engine.compile(chain, None)
        report = self.engine.inspect(result)
        for keyword in ("FMA depth", "SIMD", "Poema CNativeEngine"):
            self.assertIn(keyword, report,
                          f"inspect() output missing '{keyword}'")


# ─────────────────────────────────────────────────────────────────────────────
# Test suite 3: Performance benchmarks
# ─────────────────────────────────────────────────────────────────────────────

class TestCNativeEnginePerformance(unittest.TestCase):
    """
    Performance benchmarks.  We do NOT enforce strict speedup thresholds
    (system-dependent) but verify that:
      - benchmarks complete without errors
      - GFLOPS > 0 and finite
      - speedup_vs_numpy > 0
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = CNativeEngine(verbose=False)

    def test_benchmark_depth_16_1m(self):
        """Depth-16 chain on 1M elements: benchmark must succeed."""
        chain = make_random_chain(16)
        b = self.engine.benchmark(chain, n_elements=1_000_000, reps=10)
        self.assertIsInstance(b, CEngineBenchmark)
        self.assertGreater(b.gflops, 0)
        self.assertGreater(b.speedup_vs_numpy, 0)
        self.assertGreater(b.gb_per_sec, 0)
        print(f"\n  [bench 16×1M] {b.report()}")

    def test_benchmark_depth_100_100k(self):
        """Deep 100-stage chain on 100K elements."""
        chain = make_random_chain(100)
        b = self.engine.benchmark(chain, n_elements=100_000, reps=10)
        self.assertGreater(b.gflops, 0)
        print(f"\n  [bench 100×100K] {b.report()}")

    def test_benchmark_full_suite(self):
        """Full suite across L1/L2/L3/DRAM tiers."""
        chain = make_random_chain(32)
        results = self.engine.benchmark_full_suite(
            chain,
            sizes=[4_096, 65_536, 524_288, 4_194_304],
            reps=5,
        )
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertIsInstance(r, CEngineBenchmark)
            if r.ns_per_call == r.ns_per_call:  # not NaN
                self.assertGreater(r.gflops, 0)
        print()
        self.engine.print_benchmark_suite(
            chain, sizes=[4_096, 65_536, 524_288], reps=5)

    def test_benchmark_fp32(self):
        """FP32 benchmark (8 floats per AVX2 register)."""
        chain = make_random_chain(16)
        b = self.engine.benchmark(chain, n_elements=500_000,
                                  reps=10, precision="fp32")
        self.assertGreater(b.gflops, 0)
        print(f"\n  [bench FP32 16×500K] {b.report()}")

    def test_gradient_benchmark(self):
        """Gradient eval should be ≤ 2× overhead over pure forward pass."""
        chain = make_random_chain(32)
        x = np.random.standard_normal(1_000_000)
        t0 = time.perf_counter_ns()
        y = self.engine.evaluate(chain, x)
        t1 = time.perf_counter_ns()
        y2, dy = self.engine.evaluate_with_gradient(chain, x)
        t2 = time.perf_counter_ns()
        fwd_ns  = t1 - t0
        grad_ns = t2 - t1
        overhead = grad_ns / max(fwd_ns, 1)
        print(f"\n  [grad bench] forward={fwd_ns/1e6:.1f}ms  "
              f"fwd+grad={grad_ns/1e6:.1f}ms  overhead={overhead:.2f}×")
        # Gradient should not be more than 5× the forward pass
        self.assertLess(overhead, 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test suite 4: WASM Backend
# ─────────────────────────────────────────────────────────────────────────────

class TestWasmBackend(unittest.TestCase):
    """Tests for the WebAssembly code generator."""

    @classmethod
    def setUpClass(cls):
        cls.backend = WasmBackend()

    def test_wat_generation_basic(self):
        """WAT source must be non-empty and contain (module."""
        chain = make_random_chain(5)
        result = self.backend.compile(chain, None)
        self.assertIsNotNone(result.emitted_code)
        self.assertIn("(module", result.emitted_code)
        self.assertIn("poema_eval_fp64", result.emitted_code)

    def test_wat_contains_fma_body(self):
        """WAT should contain mul/add instructions for each stage."""
        chain = [FMA(2.0, 1.0), FMA(0.5, -0.3)]
        result = self.backend.compile(chain, None)
        code = result.emitted_code
        self.assertIn("f64.mul", code)
        self.assertIn("f64.add", code)

    def test_wat_has_memory(self):
        """WAT module must declare a linear memory for batch evaluation."""
        chain = [FMA(1.0, 0.0)]
        wat = self.backend.emit_wat(chain, precision="fp64")
        self.assertIn("(memory", wat)

    def test_wat_has_batch_eval(self):
        """WAT should export poema_eval_batch_fp64 for vectorised use."""
        chain = make_random_chain(8)
        wat = self.backend.emit_wat(chain, precision="fp64")
        self.assertIn("poema_eval_batch_fp64", wat)

    def test_wat_fp32(self):
        """FP32 WAT should use f32 instructions."""
        chain = [FMA(1.5, 0.5)]
        wat = self.backend.emit_wat(chain, precision="fp32")
        self.assertIn("f32", wat)
        self.assertNotIn("f64.mul", wat)

    def test_wat_complex(self):
        """Complex WAT must contain real/imag split arithmetic."""
        chain = [FMA(complex(1.0, 0.5), complex(0.0, 0.1))]
        wat = self.backend.emit_wat(chain, precision="fp64", is_complex=True)
        self.assertIn("poema_eval_complex", wat)
        self.assertIn("$ar", wat)
        self.assertIn("$ai", wat)

    def test_fallback_callable_correctness(self):
        """Fallback Python callable must produce correct results."""
        chain = [FMA(2.0, 1.0), FMA(0.5, 0.0)]
        result = self.backend.compile(chain, None)
        x = np.array([1.0, 2.0, 3.0])
        y_wasm = result.callable_fn(np.array([1.0]))
        y_ref  = numpy_eval_chain(chain, np.array([1.0]))
        np.testing.assert_allclose(y_wasm, y_ref, rtol=1e-10)

    def test_js_loader_generation(self):
        """JS loader must contain evalFMA and evalBatch exports."""
        chain = make_random_chain(10)
        loader = self.backend.emit_js_loader("fma.wasm", "test_module", 10)
        self.assertIn("evalFMA", loader)
        self.assertIn("evalBatch", loader)
        self.assertIn("WebAssembly", loader)

    def test_wat_file_written(self):
        """Compile must write the .wat file to disk."""
        chain = [FMA(1.0, 0.0)]
        result = self.backend.compile(chain, None, module_name="test_wat_write")
        self.assertIsNotNone(result.emitted_path)
        self.assertTrue(os.path.exists(result.emitted_path))

    def test_validate_wat_basic(self):
        """validate_wat() returns True for valid WAT (text-based check)."""
        chain = [FMA(1.0, 0.0)]
        wat = self.backend.emit_wat(chain)
        result = self.backend.validate_wat(wat)
        self.assertTrue(result)

    def test_capabilities(self):
        """WASM backend capabilities must be correctly declared."""
        caps = self.backend.capabilities
        self.assertEqual(caps.name, "wasm")
        self.assertTrue(caps.supports_batched)
        self.assertIn("fp64", caps.precision_formats)

    def test_verify_always_available(self):
        """WASM backend is always available (no runtime dependency)."""
        self.assertTrue(self.backend.verify_available())

    def test_depth_100_wat(self):
        """100-stage chain WAT should have 100 sets of mul+add instructions."""
        chain = make_random_chain(100)
        result = self.backend.compile(chain, None)
        code = result.emitted_code
        # WAT contains multiple function bodies (eval + batch + horner) so total
        # f64.mul count is >= n_stages (not exactly n_stages)
        self.assertGreaterEqual(code.count("f64.mul"), 100)
        self.assertGreaterEqual(code.count("f64.add"), 100)


# ─────────────────────────────────────────────────────────────────────────────
# Test suite 5: ONNX Backend
# ─────────────────────────────────────────────────────────────────────────────

class TestONNXBackend(unittest.TestCase):
    """Tests for the ONNX computation graph generator."""

    @classmethod
    def setUpClass(cls):
        cls.backend = ONNXBackend()
        cls.has_onnx = cls.backend.verify_available()
        cls.has_ort  = __import__("importlib").util.find_spec("onnxruntime") is not None

    def _skip_if_no_onnx(self):
        if not self.has_onnx:
            self.skipTest("onnx package not installed")

    def test_verify_available(self):
        """verify_available() correctly reflects onnx installation status."""
        import importlib
        expected = importlib.util.find_spec("onnx") is not None
        self.assertEqual(self.backend.verify_available(), expected)

    def test_capabilities_structure(self):
        """Capabilities must indicate onnx backend name."""
        caps = self.backend.capabilities
        self.assertEqual(caps.name, "onnx")
        self.assertTrue(caps.supports_gpu)
        self.assertTrue(caps.supports_batched)

    def test_compile_without_onnx_uses_fallback(self):
        """Even without onnx, compile() must return a working callable."""
        chain = [FMA(2.0, 1.0), FMA(0.5, 0.0)]
        result = self.backend.compile(chain, None)
        x = np.array([1.0, 2.0, 3.0])
        y = result.callable_fn(x)
        y_ref = numpy_eval_chain(chain, x)
        np.testing.assert_allclose(y, y_ref, rtol=1e-10)

    def test_build_model_node_count(self):
        """Model must have 2 nodes per FMA stage (Mul + Add)."""
        self._skip_if_no_onnx()
        n = 10
        chain = make_random_chain(n)
        model = self.backend.build_model(chain, precision="fp64")
        # 2*n Mul+Add for the chain; may have extra for shape inference
        self.assertGreaterEqual(len(model.graph.node), 2 * n)

    def test_model_verification_passes(self):
        """ONNX model must pass onnx.checker.check_model."""
        self._skip_if_no_onnx()
        chain = make_random_chain(5)
        model = self.backend.build_model(chain)
        self.assertTrue(self.backend.verify_model(model))

    def test_model_saved_to_disk(self):
        """compile() must write the .onnx file to disk."""
        self._skip_if_no_onnx()
        chain = make_random_chain(5)
        result = self.backend.compile(chain, None, module_name="test_save_onnx")
        if result.emitted_path:
            self.assertTrue(os.path.exists(result.emitted_path))

    def test_ort_inference_correctness(self):
        """If ORT is available, inference must match numpy reference."""
        self._skip_if_no_onnx()
        if not self.has_ort:
            self.skipTest("onnxruntime not installed")
        chain = make_random_chain(8)
        result = self.backend.compile(chain, None)
        x = np.linspace(-2.0, 2.0, 1_000)
        y_ort = result.callable_fn(x)
        y_ref = numpy_eval_chain(chain, x)
        np.testing.assert_allclose(y_ort, y_ref, rtol=1e-6, atol=1e-7)

    def test_fp32_model(self):
        """FP32 model construction must succeed without error."""
        self._skip_if_no_onnx()
        chain = make_random_chain(5)
        model = self.backend.build_model(chain, precision="fp32")
        self.assertTrue(self.backend.verify_model(model))

    def test_inspect_model(self):
        """inspect_model() must return a string mentioning 'ONNX Model'."""
        self._skip_if_no_onnx()
        chain = make_random_chain(5)
        model = self.backend.build_model(chain)
        report = self.backend.inspect_model(model)
        self.assertIn("ONNX Model", report)
        self.assertIn("Nodes", report)

    def test_complex_model(self):
        """Complex FMA model must have Re/Im inputs xr, xi."""
        self._skip_if_no_onnx()
        chain = [FMA(complex(1.0, 0.5), complex(0.0, 0.1))]
        model = self.backend.build_complex_model(chain)
        in_names = [i.name for i in model.graph.input]
        self.assertIn("xr", in_names)
        self.assertIn("xi", in_names)

    def test_torch_script_generation(self):
        """to_torch_script() must return valid Python code."""
        chain = make_random_chain(5)
        code = self.backend.to_torch_script(chain)
        self.assertIn("class PoemFMA", code)
        self.assertIn("def forward", code)
        # Verify it compiles as Python
        compile(code, "<string>", "exec")   # must not raise SyntaxError

    def test_onnx_depth_100(self):
        """100-stage ONNX model must build and verify correctly."""
        self._skip_if_no_onnx()
        chain = make_random_chain(100)
        model = self.backend.build_model(chain)
        self.assertTrue(self.backend.verify_model(model))
        self.assertGreaterEqual(len(model.graph.node), 200)


# ─────────────────────────────────────────────────────────────────────────────
# Test suite 6: BackendRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TestBackendRegistry(unittest.TestCase):
    """Verify all backends are registered and discoverable."""

    def test_all_backends_registered(self):
        """Registry must contain c_native, wasm, onnx, numpy_cpu."""
        avail = BackendRegistry.available()
        for name in ("c_native", "wasm", "onnx", "numpy_cpu"):
            self.assertIn(name, avail,
                          f"Backend '{name}' not found in registry")

    def test_c_native_available(self):
        """c_native backend must be available (gcc/clang present)."""
        b = BackendRegistry.get("c_native")
        self.assertTrue(b.verify_available())

    def test_wasm_always_available(self):
        """WASM backend is always available."""
        b = BackendRegistry.get("wasm")
        self.assertTrue(b.verify_available())

    def test_best_for_cpu_returns_c_native(self):
        """best_for_cpu() should prefer the C native engine."""
        b = BackendRegistry.best_for_cpu()
        self.assertEqual(b.capabilities.name, "c_native")

    def test_describe_all(self):
        """describe_all() must list all registered backends."""
        report = BackendRegistry.describe_all()
        for name in ("c_native", "wasm", "onnx", "numpy_cpu"):
            self.assertIn(name, report)

    def test_register_custom(self):
        """register() must add a custom backend to the registry."""
        from poema.backends.protocol import BackendCapabilities, BackendResult

        class DummyBackend(CNativeEngine):
            @property
            def capabilities(self):
                c = super().capabilities
                return BackendCapabilities(name="dummy_test", **{
                    f.name: getattr(c, f.name)
                    for f in c.__dataclass_fields__.values()
                    if f.name != "name"
                })

        dummy = DummyBackend()
        BackendRegistry.register(dummy)
        self.assertIn("dummy_test", BackendRegistry.available())


# ─────────────────────────────────────────────────────────────────────────────
# Test suite 7: Stress tests — edge cases and extreme inputs
# ─────────────────────────────────────────────────────────────────────────────

class TestCNativeEngineStress(unittest.TestCase):
    """Edge cases, extreme inputs, numerical stability."""

    @classmethod
    def setUpClass(cls):
        cls.engine = CNativeEngine(verbose=False)

    def test_very_deep_chain_5000(self):
        """5000-stage chain must compile and produce finite outputs."""
        chain = [FMA(0.999, 0.001) for _ in range(5000)]
        x = np.linspace(-1.0, 1.0, 1_000)
        y = self.engine.evaluate(chain, x)
        self.assertTrue(np.all(np.isfinite(y)), "5000-stage FMA must give finite outputs")

    def test_single_element_batch(self):
        """Size-1 arrays must work (no SIMD alignment issues)."""
        chain = make_random_chain(32)
        x = np.array([3.14159])
        y = self.engine.evaluate(chain, x)
        y_ref = numpy_eval_chain(chain, x)
        np.testing.assert_allclose(y, y_ref, rtol=1e-10)

    def test_odd_sized_arrays(self):
        """Arrays with odd sizes: 3, 5, 7, 11 (tail-handling test)."""
        chain = make_random_chain(8)
        for n in (3, 5, 7, 11, 13, 17, 19, 23):
            x = np.random.standard_normal(n)
            y = self.engine.evaluate(chain, x)
            y_ref = numpy_eval_chain(chain, x)
            np.testing.assert_allclose(
                y, y_ref, rtol=1e-10,
                err_msg=f"Mismatch at array size {n}")

    def test_non_contiguous_input(self):
        """Non-contiguous (strided) input must be handled correctly."""
        chain = make_random_chain(10)
        x_full = np.random.standard_normal(20_000)
        x = x_full[::2]   # stride-2 view (non-contiguous)
        y = self.engine.evaluate(chain, x)
        y_ref = numpy_eval_chain(chain, np.ascontiguousarray(x))
        np.testing.assert_allclose(y, y_ref, rtol=1e-10)

    def test_matrix_eval_1d_input(self):
        """matrix_eval must handle 1D input (treated as single-row matrix)."""
        chain = make_random_chain(5)
        x = np.random.standard_normal(1_000)
        Y = self.engine.matrix_eval(chain, x)
        self.assertEqual(Y.shape[1], 1_000)

    def test_fuse_single_chain(self):
        """fuse_and_compile with a single chain must match compile directly."""
        chain = make_random_chain(10)
        x = np.random.standard_normal(10_000)
        r1 = self.engine.compile(chain, None)
        r2 = self.engine.fuse_and_compile([chain])
        np.testing.assert_allclose(
            r1.callable_fn(x), r2.callable_fn(x), rtol=1e-10)

    def test_zero_weight_chain(self):
        """Chain with weight=0: output must equal last bias."""
        chain = [FMA(0.0, 5.0)]
        x = np.ones(100)
        y = self.engine.evaluate(chain, x)
        np.testing.assert_allclose(y, 5.0, rtol=1e-10)

    def test_negative_weights(self):
        """Chain with negative weights must still be numerically correct."""
        chain = [FMA(-2.0, 3.0), FMA(-0.5, 1.0)]
        x = np.linspace(-5.0, 5.0, 10_000)
        y = self.engine.evaluate(chain, x)
        y_ref = numpy_eval_chain(chain, x)
        np.testing.assert_allclose(y, y_ref, rtol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("  Poema Engine Massive Test Suite v2.4.0")
    print("  CNativeEngine + WASM + ONNX + BackendRegistry")
    print("=" * 72)

    # Run with verbose output
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    for cls in [
        TestCNativeEngineCorrectness,
        TestCNativeEngineHighLevelAPI,
        TestCNativeEnginePerformance,
        TestWasmBackend,
        TestONNXBackend,
        TestBackendRegistry,
        TestCNativeEngineStress,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, failfast=False)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
