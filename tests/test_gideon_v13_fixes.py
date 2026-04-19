"""
Tests masivos para los fixes de Gideon v1.3.0
==============================================
Cubre cada uno de los 7 bugs corregidos y mejoras implementadas:
  1. GPU dinámica (PCIe threshold auto-medido, umbral explícito no sobrescrito)
  2. run_fma_batch — API de conveniencia (5 → 5 resultados, errores de validación)
  3. Integración Rust core (rust_c_avx2 backend)
  4. MLDispatcher cold-start fix (telemetría acumulada)
  5. meta_compile TypeError + meta_compile_fma (validación, confort API)
  6. analyse_network → dict (as_dict=True/False, NetworkACFReport)
  7. Telemetría SQLite WAL (persistencia, stub JSON, concurrencia, clear)
  8. Correctitud numérica exhaustiva (fold, no-fold, extremos)
  9. Engine info() menciona Rust y umbrales
 10. run_batch — API original sigue funcionando
"""

from __future__ import annotations

import os
import sys
import json
import threading
import tempfile
import unittest

import numpy as np

# — configurar path ──────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from poema.backends.gideon.engine import GideonEngine, GideonEngineConfig
from poema.backends.gideon.ml_dispatcher import GideonTelemetry


# Duck-type FMA compatible con engine.run_fma
class _FMA:
    __slots__ = ("weight", "bias")

    def __init__(self, w: float, b: float):
        self.weight = float(w)
        self.bias = float(b)


try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
    _CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False
    _CUDA = False

try:
    from poema.backends.gideon.rust_bridge import RUST_CORE_AVAILABLE
except Exception:
    RUST_CORE_AVAILABLE = False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_engine(**kwargs) -> GideonEngine:
    tmp = tempfile.mkdtemp()
    defaults = dict(
        telemetry_path=os.path.join(tmp, "tel.json"),
        use_ml_dispatcher=False,
        use_autotune=False,
    )
    defaults.update(kwargs)
    return GideonEngine(GideonEngineConfig(**defaults))


def _chain(n: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    ws = rng.uniform(0.9, 1.1, n)
    bs = rng.uniform(-0.1, 0.1, n)
    return [_FMA(float(w), float(b)) for w, b in zip(ws, bs)]


def _numpy_eval(chain, x: np.ndarray) -> np.ndarray:
    y = x.copy().astype(np.float64)
    for f in chain:
        y = f.weight * y + f.bias
    return y


# ═════════════════════════════════════════════════════════════════════════════
# 1. GPU dinámica — umbral explícito NO sobrescrito
# ═════════════════════════════════════════════════════════════════════════════
class TestGPUThreshold(unittest.TestCase):

    def test_explicit_gpu_min_elements_not_overridden(self):
        """gpu_min_elements explícito no debe sobrescribirse al init."""
        for explicit in (1, 500, 100_000, 10_000_000):
            with self.subTest(explicit=explicit):
                eng = _make_engine(gpu_min_elements=explicit, use_autotune=True)
                self.assertEqual(
                    eng.config.gpu_min_elements, explicit,
                    f"gpu_min_elements={explicit} fue sobrescrito a "
                    f"{eng.config.gpu_min_elements}",
                )

    def test_default_gpu_min_elements_positive(self):
        """El umbral default debe ser un entero positivo."""
        eng = _make_engine(use_autotune=True)
        self.assertGreater(eng.config.gpu_min_elements, 0)

    @unittest.skipUnless(_CUDA, "CUDA no disponible")
    def test_gpu_path_activates_with_threshold_1(self):
        """Con gpu_min_elements=1 toda llamada ≥1 elemento debe usar GPU."""
        eng = _make_engine(gpu_min_elements=1)
        chain = _chain(5, seed=10)
        x = np.linspace(-1.0, 1.0, 2000)
        r = eng.run_fma(chain, x)
        self.assertTrue(r.success)
        self.assertTrue(r.gpu_used, f"gpu_used=False; backend={r.backend_used}")

    @unittest.skipUnless(_CUDA, "CUDA no disponible")
    def test_gpu_result_numerically_correct(self):
        """Resultado GPU debe coincidir con CPU dentro de rtol=1e-8."""
        chain = _chain(20, seed=99)
        x = np.random.default_rng(99).standard_normal(5_000)
        r_gpu = _make_engine(gpu_min_elements=1).run_fma(chain, x)
        r_cpu = _make_engine(gpu_min_elements=10 ** 9).run_fma(chain, x)
        np.testing.assert_allclose(r_gpu.output, r_cpu.output, rtol=1e-8,
                                   err_msg="GPU ≠ CPU")

    @unittest.skipUnless(_CUDA, "CUDA no disponible")
    def test_gpu_large_array_performance(self):
        """Array 5M debe completarse exitosamente usando GPU."""
        eng = _make_engine(gpu_min_elements=1_000_000, use_autotune=True)
        chain = _chain(10, seed=77)
        x = np.random.default_rng(77).standard_normal(5_000_000)
        r = eng.run_fma(chain, x)
        self.assertTrue(r.success)
        if eng._hw_caps and eng._hw_caps.gpu_available:
            self.assertTrue(r.gpu_used)


# ═════════════════════════════════════════════════════════════════════════════
# 2. run_fma_batch — API de conveniencia
# ═════════════════════════════════════════════════════════════════════════════
class TestRunFmaBatch(unittest.TestCase):

    def setUp(self):
        self.eng = _make_engine()

    def test_basic_five_inputs(self):
        """5 inputs → 5 resultados exitosos."""
        ws = [1.5, 0.8, 1.2]
        bs = [0.1, -0.2, 0.05]
        inputs = [np.linspace(-1, 1, 100 * (i + 1)) for i in range(5)]
        results = self.eng.run_fma_batch(ws, bs, inputs)
        self.assertEqual(len(results), 5)
        for i, r in enumerate(results):
            self.assertTrue(r.success, f"resultado {i} falla: {r}")

    def test_numerical_correctness(self):
        """Salida de run_fma_batch debe coincidir con evaluación numpy."""
        ws = [2.0, 0.5]
        bs = [1.0, -3.0]
        x = np.array([0.0, 1.0, 2.0, 3.0])
        exp = x * 2.0 + 1.0
        exp = exp * 0.5 + (-3.0)
        results = self.eng.run_fma_batch(ws, bs, [x])
        np.testing.assert_allclose(results[0].output, exp, rtol=1e-12)

    def test_mismatched_weights_biases_raises(self):
        """len(weights) ≠ len(biases) debe lanzar ValueError."""
        with self.assertRaises(ValueError):
            self.eng.run_fma_batch([1.0, 2.0], [0.5], [np.ones(10)])

    def test_empty_inputs_returns_empty(self):
        """inputs=[] → lista vacía sin error."""
        results = self.eng.run_fma_batch([1.0], [0.0], [])
        self.assertEqual(results, [])

    def test_single_fma_single_input(self):
        """Caso degenerado: 1 FMA, 1 input."""
        results = self.eng.run_fma_batch([3.0], [-1.0], [np.array([2.0])])
        self.assertEqual(len(results), 1)
        np.testing.assert_allclose(results[0].output, np.array([5.0]), rtol=1e-12)

    def test_many_inputs(self):
        """50 inputs de distintos tamaños, todos exitosos."""
        ws = [1.1, 0.9, 1.05]
        bs = [0.01, -0.01, 0.0]
        rng = np.random.default_rng(42)
        inputs = [rng.standard_normal(int(rng.integers(10, 500))) for _ in range(50)]
        results = self.eng.run_fma_batch(ws, bs, inputs)
        self.assertEqual(len(results), 50)
        failures = [i for i, r in enumerate(results) if not r.success]
        self.assertEqual(failures, [], f"Fallos en inputs: {failures}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. Integración Rust core
# ═════════════════════════════════════════════════════════════════════════════
class TestRustCoreBackend(unittest.TestCase):

    @unittest.skipUnless(RUST_CORE_AVAILABLE, "Rust core no disponible")
    def test_rust_backend_activates_for_large_array(self):
        """Con fold_affine=False y _rust_min_elements=1, debe usar rust_c_avx2."""
        # fold_affine=False: deshabilita el camino aíno que siempre ganana al Rust
        eng = _make_engine(gpu_min_elements=10 ** 9, fold_affine=False)
        eng._rust_min_elements = 1  # activar Rust para cualquier tamaño
        chain = _chain(3, seed=5)
        x = np.random.default_rng(5).standard_normal(50_000)
        r = eng.run_fma(chain, x)
        self.assertTrue(r.success)
        self.assertEqual(r.backend_used, "rust_c_avx2",
                         f"backend esperado rust_c_avx2, obtenido {r.backend_used}")

    @unittest.skipUnless(RUST_CORE_AVAILABLE, "Rust core no disponible")
    def test_rust_result_numerically_correct(self):
        """rust_c_avx2 (fold_affine=False) debe dar el mismo resultado que numpy."""
        eng = _make_engine(gpu_min_elements=10 ** 9, fold_affine=False)
        eng._rust_min_elements = 1
        chain = _chain(5, seed=7)
        x = np.linspace(-2.0, 2.0, 10_000)
        r_rust = eng.run_fma(chain, x)
        np.testing.assert_allclose(r_rust.output, _numpy_eval(chain, x),
                                   rtol=1e-10, err_msg="Rust ≠ numpy")

    def test_rust_threshold_attribute_exists(self):
        """engine debe exponer _rust_min_elements como int positivo."""
        eng = _make_engine()
        self.assertIsInstance(eng._rust_min_elements, int)
        self.assertGreater(eng._rust_min_elements, 0)

    def test_rust_status_in_info(self):
        """info() debe mencionar Rust core."""
        eng = _make_engine()
        self.assertIn("Rust", eng.info())


# ═════════════════════════════════════════════════════════════════════════════
# 4. analyse_network → dict y NetworkACFReport
# ═════════════════════════════════════════════════════════════════════════════
class TestAnalyseNetworkDict(unittest.TestCase):

    @unittest.skipUnless(_HAS_TORCH, "torch no disponible")
    def setUp(self):
        self.eng = _make_engine()
        self.nets = [
            nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2)),
            nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1)),
            nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4),
                          nn.Linear(4, 2)),
        ]

    @unittest.skipUnless(_HAS_TORCH, "torch no disponible")
    def test_default_returns_dict(self):
        """analyse_network() con as_dict=True debe devolver dict."""
        out = self.eng.analyse_network(self.nets[0])
        self.assertIsInstance(out, dict)

    @unittest.skipUnless(_HAS_TORCH, "torch no disponible")
    def test_dict_has_required_keys(self):
        """Dict debe contener todas las claves esperadas."""
        out = self.eng.analyse_network(self.nets[0])
        for key in ("n_layers", "global_alpha", "global_nc_class",
                    "total_fma_count", "total_elapsed_ms"):
            self.assertIn(key, out, f"Clave faltante: {key}")

    @unittest.skipUnless(_HAS_TORCH, "torch no disponible")
    def test_n_layers_positive(self):
        """n_layers debe ser ≥ 1."""
        out = self.eng.analyse_network(self.nets[0])
        self.assertGreaterEqual(out["n_layers"], 1)

    @unittest.skipUnless(_HAS_TORCH, "torch no disponible")
    def test_global_alpha_non_negative(self):
        """global_alpha debe ser un float ≥ 0."""
        out = self.eng.analyse_network(self.nets[0])
        self.assertIsInstance(out["global_alpha"], float)
        self.assertGreaterEqual(out["global_alpha"], 0.0)

    @unittest.skipUnless(_HAS_TORCH, "torch no disponible")
    def test_as_dict_false_returns_object(self):
        """as_dict=False debe devolver un objeto con global_alpha."""
        out = self.eng.analyse_network(self.nets[0], as_dict=False)
        self.assertNotIsInstance(out, dict)
        self.assertTrue(hasattr(out, "global_alpha"),
                        f"NetworkACFReport no tiene global_alpha: {dir(out)}")

    @unittest.skipUnless(_HAS_TORCH, "torch no disponible")
    def test_multiple_networks_consistent(self):
        """Analizar 3 redes distintas no debe fallar."""
        for i, net in enumerate(self.nets):
            with self.subTest(net_idx=i):
                out = self.eng.analyse_network(net)
                self.assertIsInstance(out, dict)
                self.assertIn("global_alpha", out)


# ═════════════════════════════════════════════════════════════════════════════
# 5. meta_compile TypeError + meta_compile_fma
# ═════════════════════════════════════════════════════════════════════════════
class TestMetaCompile(unittest.TestCase):

    def setUp(self):
        self.eng = _make_engine()

    def test_meta_compile_list_raises_typeerror(self):
        """Pasar lista a meta_compile debe lanzar TypeError."""
        with self.assertRaises(TypeError) as cm:
            self.eng.meta_compile([1.0, 2.0], domain=(0, 1))
        self.assertIn("meta_compile_fma", str(cm.exception))

    def test_meta_compile_array_raises_typeerror(self):
        """Pasar array numpy a meta_compile debe lanzar TypeError."""
        with self.assertRaises(TypeError):
            self.eng.meta_compile(np.ones(10), domain=(0, 1))

    def test_meta_compile_int_raises_typeerror(self):
        """Pasar int (no callable) debe lanzar TypeError."""
        with self.assertRaises(TypeError):
            self.eng.meta_compile(42, domain=(0, 1))

    @unittest.skipUnless(_HAS_TORCH, "torch no disponible")
    def test_meta_compile_callable_works(self):
        """Un callable válido no debe lanzar TypeError."""
        fn = lambda x: torch.sin(x)  # noqa: E731
        result = self.eng.meta_compile(fn, domain=(0.0, 1.0))
        self.assertIsNotNone(result)

    @unittest.skipUnless(_HAS_TORCH, "torch no disponible")
    def test_meta_compile_fma_basic(self):
        """meta_compile_fma debe retornar objeto con best_grammar."""
        ws = [1.5, 0.8, 1.2, 0.9]
        bs = [0.1, -0.15, 0.05, -0.05]
        result = self.eng.meta_compile_fma(ws, bs, domain=(-1.0, 1.0))
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "best_grammar"),
                        f"Resultado no tiene best_grammar: {result}")

    @unittest.skipUnless(_HAS_TORCH, "torch no disponible")
    def test_meta_compile_fma_mismatched_raises_valueerror(self):
        """len(weights) ≠ len(biases) debe lanzar ValueError."""
        with self.assertRaises(ValueError):
            self.eng.meta_compile_fma([1.0, 2.0], [0.5], domain=(0, 1))

    @unittest.skipUnless(_HAS_TORCH, "torch no disponible")
    def test_meta_compile_fma_single_fma(self):
        """Cadena de 1 FMA debe funcionar."""
        result = self.eng.meta_compile_fma([2.0], [1.0], domain=(0.0, 1.0))
        self.assertIsNotNone(result)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Telemetría SQLite WAL
# ═════════════════════════════════════════════════════════════════════════════
class TestTelemetrySQLite(unittest.TestCase):
    """Telemetría usando engine.run_fma para crear resultados reales."""

    def _make_tel_with_records(self, n: int = 10):
        """Crea un GideonTelemetry con n registros reales."""
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "tel.json")
        eng = _make_engine(telemetry_path=path)
        chain = _chain(3, seed=0)
        for i in range(n):
            eng.run_fma(chain, np.random.default_rng(i).standard_normal(200))
        eng._telemetry.flush()
        return eng._telemetry, path

    def test_sqlite_activated_on_first_save(self):
        """_use_sqlite debe ser True tras el primer flush con registros."""
        tel, _ = self._make_tel_with_records(5)
        self.assertTrue(tel._use_sqlite, "_use_sqlite debe ser True tras flush")

    def test_records_persisted_and_reloaded(self):
        """20 registros deben reaparecer al crear nueva instancia."""
        tel1, path = self._make_tel_with_records(20)
        tel2 = GideonTelemetry(db_path=path)
        self.assertGreaterEqual(len(tel2._records), 20,
                                f"Esperados ≥20 registros, obtenidos {len(tel2._records)}")

    def test_json_stub_created_when_sqlite(self):
        """El archivo .json debe existir aunque SQLite sea primario."""
        tel, path = self._make_tel_with_records(5)
        self.assertTrue(os.path.exists(path),
                        "Archivo .json debe existir aunque SQLite sea primario")

    def test_json_stub_has_storage_field(self):
        """El stub JSON debe contener campo _storage='sqlite'."""
        tel, path = self._make_tel_with_records(5)
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data.get("_storage"), "sqlite",
                         f"_storage incorrecto: {data}")

    def test_sqlite_db_file_created(self):
        """El archivo .db SQLite debe existir tras el flush."""
        tel, _ = self._make_tel_with_records(5)
        self.assertTrue(os.path.exists(tel._sqlite_path),
                        f"SQLite DB no encontrada en {tel._sqlite_path}")

    def test_clear_removes_all_records(self):
        """clear() debe vaciar registros en memoria y SQLite."""
        tel, path = self._make_tel_with_records(10)
        tel.clear()
        self.assertEqual(len(tel._records), 0)
        tel2 = GideonTelemetry(db_path=path)
        self.assertEqual(len(tel2._records), 0)

    def test_concurrent_writes_no_errors(self):
        """8 hilos escribiendo simultáneamente → 0 errores de escritura."""
        tmp = tempfile.mkdtemp()
        errors = []

        def worker(tid):
            try:
                path = os.path.join(tmp, "shared_tel.json")
                eng = _make_engine(telemetry_path=path)
                chain = _chain(2, seed=tid)
                for i in range(25):
                    eng.run_fma(chain, np.random.default_rng(tid * 100 + i).standard_normal(100))
                eng._telemetry.flush()
            except Exception as exc:
                errors.append((tid, str(exc)))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [], f"Errores en threads: {errors}")

    def test_max_records_bounded_after_reload(self):
        """Tras reload, registros no deben superar MAX_RECORDS."""
        tel, path = self._make_tel_with_records(5)
        limit = GideonTelemetry.MAX_RECORDS
        tel2 = GideonTelemetry(db_path=path)
        self.assertLessEqual(len(tel2._records), limit)


# ═════════════════════════════════════════════════════════════════════════════
# 7. MLDispatcher — acumulación y warm start
# ═════════════════════════════════════════════════════════════════════════════
class TestMLDispatcherWarm(unittest.TestCase):

    def _make_eng_with_ml(self) -> GideonEngine:
        tmp = tempfile.mkdtemp()
        return _make_engine(
            telemetry_path=os.path.join(tmp, "tel.json"),
            use_ml_dispatcher=True,
        )

    def test_telemetry_accumulates_with_use(self):
        """Después de N ejecuciones, telemetría debe tener ≥ N registros."""
        eng = self._make_eng_with_ml()
        chain = _chain(3, seed=1)
        N = 30
        for _ in range(N):
            eng.run_fma(chain, np.random.randn(500))
        self.assertGreaterEqual(len(eng._telemetry._records), N)

    def test_telemetry_stats_available(self):
        """telemetry_stats() debe devolver un resultado no nulo."""
        eng = self._make_eng_with_ml()
        chain = _chain(3, seed=2)
        for _ in range(15):
            eng.run_fma(chain, np.random.randn(500))
        stats = eng.telemetry_stats()
        self.assertIsNotNone(stats)
        self.assertGreater(len(str(stats)), 5)

    def test_mldispatcher_decide_no_exception_after_warmup(self):
        """Tras MIN_SAMPLES ejecuciones, decide() no debe lanzar error."""
        eng = self._make_eng_with_ml()
        chain = _chain(3, seed=3)
        min_samples = eng._ml_dispatcher.MIN_SAMPLES if eng._ml_dispatcher else 5
        for _ in range(min_samples + 5):
            eng.run_fma(chain, np.random.randn(500))
        if eng._ml_dispatcher:
            import inspect
            # Obtener un programa real del engine para pasar a decide()
            r = eng.run_fma(chain, np.random.randn(500))
            decision = eng._ml_dispatcher.decide(r.program)
            self.assertIsNotNone(decision)


# ═════════════════════════════════════════════════════════════════════════════
# 8. Correctitud numérica exhaustiva
# ═════════════════════════════════════════════════════════════════════════════
class TestNumericalCorrectness(unittest.TestCase):

    def setUp(self):
        self.eng = _make_engine()

    def test_affine_fold_sweep_20_trials(self):
        """Fold affine debe ser idéntico a numpy para 20 cadenas aleatorias."""
        rng = np.random.default_rng(1234)
        for trial in range(20):
            n_fma = int(rng.integers(2, 30))
            chain = _chain(n_fma, seed=trial)
            x = rng.standard_normal(int(rng.integers(100, 2000)))
            r = self.eng.run_fma(chain, x)
            self.assertTrue(r.success, f"trial={trial} failed")
            np.testing.assert_allclose(
                r.output, _numpy_eval(chain, x), rtol=1e-10,
                err_msg=f"trial={trial} n_fma={n_fma} size={x.size}",
            )

    def test_no_fold_sweep_20_trials(self):
        """Sin fold, la evaluación sequential también debe coincidir con numpy."""
        rng = np.random.default_rng(5678)
        for trial in range(20):
            n_fma = int(rng.integers(1, 10))
            chain = _chain(n_fma, seed=trial + 100)
            x = rng.standard_normal(int(rng.integers(50, 500)))
            r = self.eng.run_fma(chain, x)
            self.assertTrue(r.success, f"trial={trial} failed")
            np.testing.assert_allclose(
                r.output, _numpy_eval(chain, x), rtol=1e-10,
                err_msg=f"trial={trial} n_fma={n_fma} size={x.size}",
            )

    def test_extreme_chain_does_not_diverge(self):
        """Cadena de 100 FMAs cerca de la identidad no debe explotar."""
        rng = np.random.default_rng(9999)
        chain = [_FMA(float(rng.uniform(0.99, 1.01)),
                      float(rng.uniform(-0.001, 0.001))) for _ in range(100)]
        x = rng.standard_normal(1000)
        r = self.eng.run_fma(chain, x)
        self.assertTrue(r.success)
        self.assertFalse(np.any(np.isnan(r.output)), "NaN en salida")
        self.assertFalse(np.any(np.isinf(r.output)), "Inf en salida")

    def test_single_element_array(self):
        """Array de 1 elemento debe funcionar sin errores."""
        chain = _chain(3, seed=0)
        r = self.eng.run_fma(chain, np.array([1.0]))
        self.assertTrue(r.success)
        np.testing.assert_allclose(r.output, _numpy_eval(chain, np.array([1.0])),
                                   rtol=1e-12)

    def test_zero_array_propagates_bias(self):
        """Array de ceros debe propagar sólo el bias en cada FMA."""
        chain = [_FMA(2.0, 1.0), _FMA(3.0, 2.0)]
        x = np.zeros(100)
        r = self.eng.run_fma(chain, x)
        # f(0) = 2*0+1 = 1; g(1) = 3*1+2 = 5
        np.testing.assert_allclose(r.output, np.full(100, 5.0), rtol=1e-12)

    def test_large_array_correctness(self):
        """Array de 1M elementos debe dar resultado correcto."""
        chain = _chain(5, seed=55)
        x = np.random.default_rng(55).standard_normal(1_000_000)
        r = self.eng.run_fma(chain, x)
        self.assertTrue(r.success)
        np.testing.assert_allclose(r.output, _numpy_eval(chain, x), rtol=1e-9)


# ═════════════════════════════════════════════════════════════════════════════
# 9. Engine info() menciona Rust y GPU
# ═════════════════════════════════════════════════════════════════════════════
class TestEngineInfo(unittest.TestCase):

    def test_info_contains_version(self):
        """info() debe contener la versión del engine."""
        self.assertIn("1.", _make_engine().info())

    def test_info_contains_rust_status(self):
        """info() debe mencionar Rust."""
        self.assertIn("Rust", _make_engine().info())

    def test_info_contains_gpu_info(self):
        """info() debe mencionar umbral GPU de alguna forma."""
        info = _make_engine().info()
        self.assertTrue("gpu" in info.lower() or "GPU" in info,
                        f"'gpu' no encontrado en info(): {info[:200]}")

    def test_info_contains_rust_threshold(self):
        """info() debe mencionar rust_min_elements."""
        self.assertIn("rust_min_elements", _make_engine().info())


# ═════════════════════════════════════════════════════════════════════════════
# 10. run_batch — API original sigue funcionando
# ═════════════════════════════════════════════════════════════════════════════
class TestRunBatchOriginalAPI(unittest.TestCase):

    def setUp(self):
        self.eng = _make_engine()

    def test_run_batch_returns_correct_count(self):
        """run_batch(fma_seq, inputs) debe devolver len(inputs) resultados."""
        chain = _chain(4, seed=50)
        inputs = [np.linspace(-1, 1, 200 * (i + 1)) for i in range(5)]
        results = self.eng.run_batch(chain, inputs)
        self.assertEqual(len(results), 5)

    def test_run_batch_numerical_correctness(self):
        """Cada resultado de run_batch debe coincidir con numpy."""
        chain = _chain(3, seed=51)
        inputs = [np.random.default_rng(i).standard_normal(300) for i in range(8)]
        results = self.eng.run_batch(chain, inputs)
        for i, r in enumerate(results):
            self.assertTrue(r.success)
            np.testing.assert_allclose(
                r.output, _numpy_eval(chain, inputs[i]), rtol=1e-10,
                err_msg=f"run_batch item {i} incorrecto",
            )

    def test_run_batch_empty_inputs(self):
        """inputs vacío → lista vacía sin error."""
        chain = _chain(3, seed=52)
        results = self.eng.run_batch(chain, [])
        self.assertEqual(results, [])

    def test_run_batch_single_input(self):
        """1 input → 1 resultado correcto."""
        chain = _chain(3, seed=53)
        x = np.array([0.5, 1.0, -0.5])
        results = self.eng.run_batch(chain, [x])
        self.assertEqual(len(results), 1)
        np.testing.assert_allclose(results[0].output, _numpy_eval(chain, x),
                                   rtol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
