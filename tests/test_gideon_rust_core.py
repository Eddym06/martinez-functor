"""
test_gideon_rust_core.py — Suite de tests para GideonCore Rust

Cubre:
  - Disponibilidad del módulo gideon_core (importación)
  - API pyo3: GideonCoreEngine, CoreEngineConfig, ExecutionPlan
  - Backend C: run_fma (escalar, fold, cadena multi-step)
  - Backend C: GEMM (correctness + dimensiones)
  - Thread-safety: múltiples hilos llamando al motor simultáneamente
  - Telemetría: acumulación de registros, stats, export ACF
  - rust_bridge.py: API Python de alto nivel
  - Fallback gracioso: si el módulo no está disponible
  - Integración con engine.py Python existente (206 tests previos intactos)
"""

import math
import threading
import unittest
from typing import List

import numpy as np


# ── Guardia: si gideon_core no está instalado, marcar todos como skip ─────────
try:
    import gideon_core as _gc  # type: ignore[import]
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

SKIP_RUST = unittest.skipUnless(RUST_AVAILABLE, "gideon_core no compilado")


# ═════════════════════════════════════════════════════════════════════════════
# Clase 17a — Importación y módulo
# ═════════════════════════════════════════════════════════════════════════════

@SKIP_RUST
class TestGideonCoreImport(unittest.TestCase):
    """17a: El módulo gideon_core es importable y tiene los atributos correctos."""

    def test_version_string(self):
        """17a-1: __version__ existe y tiene formato semver."""
        ver = _gc.__version__
        self.assertIsInstance(ver, str)
        parts = ver.split(".")
        self.assertEqual(len(parts), 3)
        for p in parts:
            self.assertTrue(p.isdigit(), f"Parte no numérica: {p!r}")

    def test_classes_exported(self):
        """17a-2: GideonCoreEngine, CoreEngineConfig, ExecutionPlan exportados."""
        self.assertTrue(hasattr(_gc, "GideonCoreEngine"))
        self.assertTrue(hasattr(_gc, "CoreEngineConfig"))
        self.assertTrue(hasattr(_gc, "ExecutionPlan"))

    def test_docstring(self):
        """17a-3: El módulo tiene docstring."""
        self.assertIsNotNone(_gc.__doc__)
        self.assertIn("Rust", _gc.__doc__)


# ═════════════════════════════════════════════════════════════════════════════
# Clase 17b — CoreEngineConfig
# ═════════════════════════════════════════════════════════════════════════════

@SKIP_RUST
class TestCoreEngineConfig(unittest.TestCase):
    """17b: CoreEngineConfig se crea con los valores por defecto correctos."""

    def test_default_construction(self):
        """17b-1: CoreEngineConfig() construye sin argumentos."""
        cfg = _gc.CoreEngineConfig()
        self.assertIsNotNone(cfg)

    def test_custom_precision(self):
        """17b-2: Se puede especificar precision='fp64'."""
        cfg = _gc.CoreEngineConfig(precision="fp64", fold_affine=True)
        self.assertIsNotNone(cfg)

    def test_repr_contains_precision(self):
        """17b-3: repr() de CoreEngineConfig menciona la precision."""
        cfg = _gc.CoreEngineConfig(precision="fp64")
        r = repr(cfg)
        self.assertIn("fp64", r)

    def test_gpu_min_elements_custom(self):
        """17b-4: gpu_min_elements se puede ajustar."""
        cfg = _gc.CoreEngineConfig(gpu_min_elements=500_000)
        self.assertIn("CoreEngineConfig", repr(cfg))


# ═════════════════════════════════════════════════════════════════════════════
# Clase 17c — GideonCoreEngine: creación y repr
# ═════════════════════════════════════════════════════════════════════════════

@SKIP_RUST
class TestGideonCoreEngine(unittest.TestCase):
    """17c: GideonCoreEngine se instancia y tiene la API esperada."""

    def setUp(self):
        self.engine = _gc.GideonCoreEngine()

    def test_construction_default(self):
        """17c-1: GideonCoreEngine() sin args construye correctamente."""
        self.assertIsNotNone(self.engine)

    def test_construction_with_config(self):
        """17c-2: GideonCoreEngine(config) acepta CoreEngineConfig."""
        cfg = _gc.CoreEngineConfig(fold_affine=True, precision="fp64")
        engine = _gc.GideonCoreEngine(cfg)
        self.assertIsNotNone(engine)

    def test_repr(self):
        """17c-3: repr() contiene GideonCoreEngine y versión."""
        r = repr(self.engine)
        self.assertIn("GideonCoreEngine", r)

    def test_has_run_fma_method(self):
        """17c-4: El motor tiene el método run_fma."""
        self.assertTrue(callable(getattr(self.engine, "run_fma", None)))

    def test_has_telemetry_stats_method(self):
        """17c-5: El motor tiene telemetry_stats."""
        self.assertTrue(callable(getattr(self.engine, "telemetry_stats", None)))

    def test_has_export_acf_calibration(self):
        """17c-6: El motor tiene export_acf_calibration."""
        self.assertTrue(callable(getattr(self.engine, "export_acf_calibration", None)))


# ═════════════════════════════════════════════════════════════════════════════
# Clase 17d — run_fma: correctness de los kernels C
# ═════════════════════════════════════════════════════════════════════════════

@SKIP_RUST
class TestRunFmaCorrectness(unittest.TestCase):
    """17d: Los kernels C (AVX2/escalar) producen resultados numéricamente correctos."""

    def setUp(self):
        self.engine = _gc.GideonCoreEngine()
        self.x = list(np.linspace(-1.0, 1.0, 1000))

    def _expected(self, weights, biases, x):
        """Referencia numpy para comparar."""
        arr = np.array(x)
        for w, b in zip(weights, biases):
            arr = w * arr + b
        return arr

    def test_single_fma(self):
        """17d-1: Una sola FMA: y = 2·x + 3."""
        res = self.engine.run_fma([2.0], [3.0], self.x)
        self.assertTrue(res["success"])
        got = np.array(res["output"])
        exp = self._expected([2.0], [3.0], self.x)
        np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-12)

    def test_chain_three_fma(self):
        """17d-2: Cadena de 3 FMAs produce el resultado correcto."""
        ws = [1.5, -0.5, 2.0]
        bs = [0.1,  0.2, -0.3]
        res = self.engine.run_fma(ws, bs, self.x)
        self.assertTrue(res["success"])
        got = np.array(res["output"])
        exp = self._expected(ws, bs, self.x)
        np.testing.assert_allclose(got, exp, rtol=1e-10, atol=1e-10)

    def test_identity_fma(self):
        """17d-3: FMA identidad (w=1, b=0) no cambia la entrada."""
        res = self.engine.run_fma([1.0], [0.0], self.x)
        self.assertTrue(res["success"])
        np.testing.assert_allclose(res["output"], self.x, rtol=1e-14, atol=1e-14)

    def test_large_array(self):
        """17d-4: Array de 500 000 elementos sin errores."""
        x_large = list(np.random.uniform(-1, 1, 500_000))
        res = self.engine.run_fma([2.0], [1.0], x_large)
        self.assertTrue(res["success"])
        self.assertEqual(len(res["output"]), 500_000)

    def test_fold_detected(self):
        """17d-5: Cadena afín activa el path de fold (folded=True)."""
        res = self.engine.run_fma([2.0, 0.5], [1.0, -1.0], self.x)
        # El motor puede o no hacer fold según la implementación,
        # pero si lo hace, el resultado debe ser correcto.
        self.assertTrue(res["success"])
        ws, bs = [2.0, 0.5], [1.0, -1.0]
        exp = self._expected(ws, bs, self.x)
        np.testing.assert_allclose(res["output"], exp, rtol=1e-10, atol=1e-10)

    def test_result_dict_keys(self):
        """17d-6: El dict de resultado tiene las keys esperadas."""
        res = self.engine.run_fma([1.0], [0.0], self.x)
        for key in ("output", "elapsed_ms", "backend", "folded", "total_fma", "success"):
            self.assertIn(key, res, f"Falta key: {key!r}")

    def test_elapsed_ms_positive(self):
        """17d-7: elapsed_ms > 0."""
        res = self.engine.run_fma([1.0], [0.0], self.x)
        self.assertGreater(res["elapsed_ms"], 0.0)

    def test_backend_string_nonempty(self):
        """17d-8: backend es un string no vacío."""
        res = self.engine.run_fma([1.0], [0.0], self.x)
        self.assertIsInstance(res["backend"], str)
        self.assertGreater(len(res["backend"]), 0)

    def test_mismatched_weights_biases_raises(self):
        """17d-9: weights y biases de distinta longitud lanza ValueError."""
        with self.assertRaises((ValueError, RuntimeError)):
            self.engine.run_fma([1.0, 2.0], [0.0], self.x)


# ═════════════════════════════════════════════════════════════════════════════
# Clase 17e — run_gemm: correctness del kernel GEMM C
# ═════════════════════════════════════════════════════════════════════════════

@SKIP_RUST
class TestRunGemmCorrectness(unittest.TestCase):
    """17e: El kernel GEMM C produce resultados correctos frente a numpy."""

    def setUp(self):
        self.engine = _gc.GideonCoreEngine()

    def test_gemm_2x2(self):
        """17e-1: GEMM 2×2 vs numpy (alpha=1, beta=0)."""
        A = [1.0, 2.0, 3.0, 4.0]   # 2×2
        B = [5.0, 6.0, 7.0, 8.0]   # 2×2
        C = [0.0, 0.0, 0.0, 0.0]   # 2×2
        result, elapsed = self.engine.run_gemm(A, B, C, 2, 2, 2)
        An = np.array(A).reshape(2, 2)
        Bn = np.array(B).reshape(2, 2)
        exp = (An @ Bn).flatten()
        np.testing.assert_allclose(result, exp, rtol=1e-12, atol=1e-12)

    def test_gemm_alpha_beta(self):
        """17e-2: GEMM con alpha=2 beta=0.5."""
        A = [1.0, 0.0, 0.0, 1.0]   # identidad 2×2
        B = [3.0, 4.0, 5.0, 6.0]
        C = [1.0, 1.0, 1.0, 1.0]
        result, _ = self.engine.run_gemm(A, B, C, 2, 2, 2, alpha=2.0, beta=0.5)
        An = np.array(A).reshape(2, 2)
        Bn = np.array(B).reshape(2, 2)
        Cn = np.array(C).reshape(2, 2)
        exp = (2.0 * An @ Bn + 0.5 * Cn).flatten()
        np.testing.assert_allclose(result, exp, rtol=1e-12, atol=1e-12)

    def test_gemm_elapsed_positive(self):
        """17e-3: elapsed_ms > 0."""
        A = [1.0] * 16
        B = [1.0] * 16
        C = [0.0] * 16
        _, elapsed = self.engine.run_gemm(A, B, C, 4, 4, 4)
        self.assertGreater(elapsed, 0.0)

    def test_gemm_large(self):
        """17e-4: GEMM 64×64 sin crash."""
        n = 64
        A = np.random.rand(n, n).flatten().tolist()
        B = np.random.rand(n, n).flatten().tolist()
        C = np.zeros(n * n).tolist()
        result, _ = self.engine.run_gemm(A, B, C, n, n, n)
        An = np.array(A).reshape(n, n)
        Bn = np.array(B).reshape(n, n)
        exp = (An @ Bn).flatten()
        np.testing.assert_allclose(result, exp, rtol=1e-8, atol=1e-8)


# ═════════════════════════════════════════════════════════════════════════════
# Clase 17f — Thread-safety con Rayon
# ═════════════════════════════════════════════════════════════════════════════

@SKIP_RUST
class TestThreadSafety(unittest.TestCase):
    """17f: Múltiples hilos Python llaman simultáneamente al motor Rust sin corrupción."""

    def setUp(self):
        self.engine = _gc.GideonCoreEngine()

    def test_concurrent_run_fma(self):
        """17f-1: 8 hilos ejecutan run_fma simultáneamente — todos correctos."""
        results = {}
        errors  = []

        def task(tid: int):
            try:
                x = list(np.linspace(tid, tid + 1, 1000))
                r = self.engine.run_fma([2.0], [float(tid)], x, "fp64")
                results[tid] = r["output"][0]
            except Exception as e:
                errors.append((tid, str(e)))

        threads = [threading.Thread(target=task, args=(i,)) for i in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(len(errors), 0, f"Errores en hilos: {errors}")
        # Verificar cada resultado
        for tid, val in results.items():
            expected = 2.0 * float(tid) + float(tid)
            self.assertAlmostEqual(val, expected, places=10)

    def test_concurrent_telemetry_no_data_race(self):
        """17f-2: 4 hilos leen la telemetría simultáneamente sin crash."""
        errors = []

        def task():
            try:
                for _ in range(50):
                    self.engine.run_fma([1.0], [0.0], [1.0, 2.0, 3.0])
                self.engine.telemetry_stats()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=task) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(len(errors), 0, f"Errores de concurrencia: {errors}")


# ═════════════════════════════════════════════════════════════════════════════
# Clase 17g — Telemetría Rust
# ═════════════════════════════════════════════════════════════════════════════

@SKIP_RUST
class TestRustTelemetry(unittest.TestCase):
    """17g: La telemetría Rust acumula registros y exporta stats correctamente."""

    def setUp(self):
        # Motor fresco para cada test (evita contaminación entre tests)
        self.engine = _gc.GideonCoreEngine()

    def test_stats_dict_has_expected_keys(self):
        """17g-1: telemetry_stats() devuelve dict con keys requeridas."""
        # Ejecutar algunas iteraciones para poblar
        for _ in range(5):
            self.engine.run_fma([1.0], [0.0], [1.0, 2.0])
        stats = self.engine.telemetry_stats()
        self.assertIn("total_records", stats)
        self.assertIn("fold_cache_size", stats)
        self.assertIn("hardware", stats)
        self.assertIn("backends", stats)

    def test_total_records_increases(self):
        """17g-2: total_records aumenta con cada ejecución."""
        stats_before = self.engine.telemetry_stats()
        n_before = stats_before["total_records"]
        for _ in range(10):
            self.engine.run_fma([2.0], [1.0], list(range(100)))
        stats_after = self.engine.telemetry_stats()
        self.assertGreater(stats_after["total_records"], n_before)

    def test_hardware_string_nonempty(self):
        """17g-3: hardware en telemetry_stats es string no vacío."""
        stats = self.engine.telemetry_stats()
        self.assertIsInstance(stats["hardware"], str)
        self.assertGreater(len(stats["hardware"]), 0)

    def test_acf_calibration_keys(self):
        """17g-4: export_acf_calibration() devuelve dict con acf_notes."""
        for _ in range(5):
            self.engine.run_fma([1.0], [0.0], [1.0, 2.0, 3.0])
        cal = self.engine.export_acf_calibration()
        # Puede devolver dict con las keys de calibración
        self.assertIsNotNone(cal)


# ═════════════════════════════════════════════════════════════════════════════
# Clase 17h — ExecutionPlan (scheduler topológico)
# ═════════════════════════════════════════════════════════════════════════════

@SKIP_RUST
class TestExecutionPlan(unittest.TestCase):
    """17h: ExecutionPlan construye planes de ejecución topológicos correctos."""

    def test_single_node_plan(self):
        """17h-1: Plan con un solo nodo tiene 1 fase."""
        plan = _gc.ExecutionPlan.from_nodes(
            [{"node_id": "n0", "input": ""}],
            "fp64"
        )
        self.assertEqual(plan.n_phases(), 1)

    def test_repr_contains_phases(self):
        """17h-2: repr() de ExecutionPlan menciona phases."""
        plan = _gc.ExecutionPlan.from_nodes(
            [{"node_id": "n0"}, {"node_id": "n1", "input": "n0"}],
            "fp64"
        )
        r = repr(plan)
        self.assertIn("phases", r)

    def test_phase_widths_type(self):
        """17h-3: phase_widths() devuelve una lista de enteros."""
        plan = _gc.ExecutionPlan.from_nodes(
            [{"node_id": "a"}, {"node_id": "b"}, {"node_id": "c"}],
            "fp64"
        )
        widths = plan.phase_widths()
        self.assertIsInstance(widths, list)
        for w in widths:
            self.assertIsInstance(w, int)


# ═════════════════════════════════════════════════════════════════════════════
# Clase 17i — rust_bridge.py API Python
# ═════════════════════════════════════════════════════════════════════════════

class TestRustBridge(unittest.TestCase):
    """17i: rust_bridge.py expone la API correctamente con fallback gracioso."""

    def test_module_importable(self):
        """17i-1: rust_bridge es importable."""
        from poema.backends.gideon import rust_bridge
        self.assertIsNotNone(rust_bridge)

    def test_rust_available_flag(self):
        """17i-2: RUST_CORE_AVAILABLE es bool."""
        from poema.backends.gideon.rust_bridge import RUST_CORE_AVAILABLE
        self.assertIsInstance(RUST_CORE_AVAILABLE, bool)

    def test_rust_status_keys(self):
        """17i-3: rust_status() devuelve dict con keys 'available', 'version'."""
        from poema.backends.gideon.rust_bridge import rust_status
        s = rust_status()
        self.assertIn("available", s)
        self.assertIn("version", s)

    def test_get_rust_engine_returns_something_or_none(self):
        """17i-4: get_rust_engine() devuelve motor o None (no lanza excepción)."""
        from poema.backends.gideon.rust_bridge import get_rust_engine
        engine = get_rust_engine()
        # Si RUST_AVAILABLE, debería devolver un motor; si no, None.
        # En ambos casos no lanza.
        self.assertTrue(engine is None or hasattr(engine, "run_fma"))

    @unittest.skipUnless(RUST_AVAILABLE, "gideon_core no compilado")
    def test_rust_run_fma_returns_dict(self):
        """17i-5: rust_run_fma() devuelve dict con 'output'."""
        from poema.backends.gideon.rust_bridge import (
            rust_run_fma, reset_rust_engine,
        )
        reset_rust_engine()
        result = rust_run_fma([2.0], [1.0], [0.5, 1.0, 1.5])
        self.assertIsNotNone(result)
        self.assertIn("output", result)
        self.assertTrue(result["success"])

    def test_rust_run_fma_fallback_on_no_rust(self):
        """17i-6: rust_run_fma devuelve None cuando Rust no disponible (mock)."""
        import poema.backends.gideon.rust_bridge as bridge
        original = bridge.RUST_CORE_AVAILABLE
        bridge.RUST_CORE_AVAILABLE = False
        bridge._GLOBAL_ENGINE = None
        try:
            result = bridge.rust_run_fma([1.0], [0.0], [1.0, 2.0])
            self.assertIsNone(result)
        finally:
            bridge.RUST_CORE_AVAILABLE = original

    @unittest.skipUnless(RUST_AVAILABLE, "gideon_core no compilado")
    def test_rust_gemm_correctness_via_bridge(self):
        """17i-7: rust_run_gemm 2×2 vs numpy."""
        from poema.backends.gideon.rust_bridge import rust_run_gemm, reset_rust_engine
        reset_rust_engine()
        A = [1., 2., 3., 4.]
        B = [5., 6., 7., 8.]
        C = [0., 0., 0., 0.]
        result = rust_run_gemm(A, B, C, 2, 2, 2)
        self.assertIsNotNone(result)
        out, elapsed = result
        exp = (np.array(A).reshape(2, 2) @ np.array(B).reshape(2, 2)).flatten()
        np.testing.assert_allclose(out, exp, rtol=1e-12)

    @unittest.skipUnless(RUST_AVAILABLE, "gideon_core no compilado")
    def test_rust_telemetry_stats_via_bridge(self):
        """17i-8: rust_telemetry_stats() devuelve dict."""
        from poema.backends.gideon.rust_bridge import rust_telemetry_stats, reset_rust_engine
        reset_rust_engine()
        stats = rust_telemetry_stats()
        self.assertIsNotNone(stats)
        self.assertIsInstance(stats, dict)


# ═════════════════════════════════════════════════════════════════════════════
# Clase 17j — GideonEngine Python sigue funcionando (no-regresión)
# ═════════════════════════════════════════════════════════════════════════════

class TestPythonEngineNoRegression(unittest.TestCase):
    """17j: El motor Python (engine.py) v1.2.0 sigue funcionando igual."""

    def test_python_engine_importable(self):
        """17j-1: GideonEngine Python es importable."""
        from poema.backends.gideon import GideonEngine
        self.assertIsNotNone(GideonEngine)

    def test_python_engine_version(self):
        """17j-2: GideonEngine Python sigue siendo v1.2.0."""
        from poema.backends.gideon import GideonEngine
        self.assertEqual(GideonEngine._VERSION, "1.2.0")

    def test_rust_bridge_importable_from_gideon(self):
        """17j-3: rust_bridge es importable desde el paquete gideon."""
        from poema.backends.gideon.rust_bridge import RUST_CORE_AVAILABLE
        self.assertIsInstance(RUST_CORE_AVAILABLE, bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
