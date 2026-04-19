"""
GideonEngine — Motor Principal Unificado de Poema.

GideonEngine es el corazón operativo de Gideon: orquesta el pipeline
completo desde la entrada (AST de Poema o secuencia FMA) hasta la
ejecución numérica optimizada, pasando por:

  1. Lowering a GideonIR
  2. Construcción del GideonGraph
  3. Decisión de despacho (GideonDispatcher)
  4. Compilación en el BackendRegistry de Poema
  5. Ejecución con telemetría completa
  6. Análisis de teoremas (GideonTheoremSeeds) bajo demanda
  7. Análisis de blueprints de IA (GideonNeuralHints) bajo demanda

Gideon es nativo para Poema: PoemCompiler lo instanciará automáticamente
cuando se requiera máxima potencia.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .ir import GideonIR, GideonProgram
from .graph import GideonGraph, ExecutionPlan
from .dispatcher import GideonDispatcher, DispatchDecision, HardwareProfile
from .neural_hints import GideonNeuralHints, ArchitectureBlueprint
from .theorem_seeds import GideonTheoremSeeds, TheoremCandidate
from .gideon_autotune import GideonHardwareProfiler, HardwareCapabilities
from .ml_dispatcher import GideonTelemetry, MLDispatcher
from .rust_bridge import (
    RUST_CORE_AVAILABLE, get_rust_engine, rust_run_fma as _rust_run_fma,
)

# Triton backend — import diferido para no fallar si Triton no está instalado
try:
    from .triton_kernels import GideonTritonBackend as _GideonTritonBackend
    _TRITON_BACKEND_AVAILABLE = True
except ImportError:
    _GideonTritonBackend = None  # type: ignore[misc,assignment]
    _TRITON_BACKEND_AVAILABLE = False


# Sentinel para distinquir "no calculado" de None en _fold_cache
_SENTINEL: object = object()


# ─────────────────────────────────────────────────────────────────────────────
# GideonExecutionResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GideonExecutionResult:
    """Resultado de una ejecución a través de GideonEngine."""
    output: Any = None                          # Resultado numérico
    program: Optional[GideonProgram] = None
    graph_stats: Dict[str, Any] = field(default_factory=dict)
    dispatch_decision: Optional[DispatchDecision] = None
    theorem_candidates: List[TheoremCandidate] = field(default_factory=list)
    elapsed_ms: float = 0.0
    backend_used: str = ""
    total_fma: int = 0
    global_epsilon: float = 0.0
    success: bool = True
    error: str = ""
    # — Métricas de optimización aplicadas —
    cache_hit: bool = False         # Hit en caché de compilación (pipeline omitido)
    folded: bool = False            # Cadena colapsada algebraicamente a W·x+B
    gpu_used: bool = False          # Ejecutado en GPU/CUDA

    def summary(self) -> str:
        lines = [
            "─── GideonExecutionResult ───",
            f"  Backend:      {self.backend_used}",
            f"  FMA total:    {self.total_fma}",
            f"  ε global:     {self.global_epsilon:.4e}",
            f"  Tiempo:       {self.elapsed_ms:.3f} ms",
            f"  Success:      {self.success}",
        ]
        if self.cache_hit:
            lines.append(f"  Cache:        HIT ⚡")
        if self.folded:
            lines.append(f"  Fold:         W·x+B (1 op)")
        if self.gpu_used:
            lines.append(f"  GPU:          CUDA ✓")
        if self.dispatch_decision:
            lines.append(f"  Dispatch:     {self.dispatch_decision.summary()}")
        if self.theorem_candidates:
            lines.append(f"  Teoremas:     {len(self.theorem_candidates)} candidatos")
        if self.graph_stats:
            lines.append(f"  Grafo:        n={self.graph_stats.get('n_nodes',0)}, "
                          f"phases={self.graph_stats.get('n_phases',0)}, "
                          f"chains={self.graph_stats.get('fusable_chains',0)}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# GideonEngineConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GideonEngineConfig:
    """Configuración de GideonEngine."""
    precision: str = "fp64"
    domain: Tuple[float, float] = (-1.0, 1.0)
    preferred_backend: Optional[str] = None     # Forzar backend específico
    enable_theorem_seeds: bool = False           # Analizar teoremas post-ejecución
    enable_neural_hints: bool = False            # Registrar blueprints de IA
    verbose: bool = False
    benchmark_mode: bool = False                 # Repetir N veces para benchmarks
    benchmark_repeats: int = 100
    # — Optimizaciones de rendimiento —
    fold_affine: bool = True         # Colapsa cadenas afines a W·x+B (1 op)
    fast_mode: bool = False          # Salta análisis IR/grafo (omite g_stats, dispatch)
    gpu_min_elements: int = 2_000_000  # Umbral de elementos para ruta GPU automática
    # — Autotune y aprendizaje —
    use_autotune: bool = True        # Detectar hardware y cargar perfil persistido
    autotune_quick: bool = True      # Solo detección estática (sin micro-benchmarks)
    use_ml_dispatcher: bool = True   # Usar MLDispatcher cuando haya datos suficientes
    telemetry_path: Optional[str] = None  # Ruta custom para DB de telemetría
    persist_fold_cache: bool = True  # Persistir resultados de fold en disco


class FrozenGraphError(RuntimeError):
    """
    Raised by GideonEngine.run_fma when the engine is frozen (warmup(freeze=True)
    was called) and the requested shape / backend combination was not pre-compiled
    during warmup.  Either call warmup() with the new shape, or use freeze=False.
    """


# ─────────────────────────────────────────────────────────────────────────────
# GideonEngine
# ─────────────────────────────────────────────────────────────────────────────

class GideonEngine:
    """
    Motor Unificado Gideon — el corazón computacional de Poema.

    Uso básico:
        engine = GideonEngine()
        fma_chain = [...]   # Lista de FMAInstruction de Poema
        result = engine.run_fma(fma_chain, x)
        print(result.summary())

    Uso avanzado con AST:
        result = engine.run_ast(ast_node, x, domain=(-π, π))

    Benchmarks:
        cfg = GideonEngineConfig(benchmark_mode=True, benchmark_repeats=1000)
        engine = GideonEngine(cfg)
        result = engine.run_fma(chain, x_large)
    """

    _VERSION = "1.2.0"
    _NAME = "Gideon"

    def __init__(self, config: Optional[GideonEngineConfig] = None) -> None:
        self.config = config or GideonEngineConfig()
        self._ir = GideonIR()
        self._dispatcher = GideonDispatcher(hw_profile=HardwareProfile.detect())
        self._theorem_engine = GideonTheoremSeeds()
        self._neural_hints = GideonNeuralHints()
        self._backend_registry = None   # Lazy init
        self._compiled_cache: Dict[str, Any] = {}     # full_key → callable
        self._fold_cache: Dict[str, Optional[Tuple[float, float]]] = {}  # chain_hash → (W,B)|None

        # — Triton backend (Frente 1 — bare-metal GPU FMA fusionado) ─────────
        self._triton_backend: Optional[Any] = None  # GideonTritonBackend | None

        # — Hardware autotune ─────────────────────────────────────────────────
        self._hw_caps: Optional[HardwareCapabilities] = None
        if self.config.use_autotune:
            try:
                profiler = GideonHardwareProfiler(
                    quick_mode=self.config.autotune_quick
                )
                self._hw_caps = profiler.load_or_profile()
                # Afinar umbral GPU con datos reales si están disponibles
                if (
                    self._hw_caps is not None
                    and self._hw_caps.gpu_available
                    # Solo ajustamos automáticamente si el usuario NO sobrescribió el valor por defecto
                    and self.config.gpu_min_elements == 2_000_000
                ):
                    if not self._hw_caps.quick_mode and self._hw_caps.measured_pcie_bw_gbs > 0:
                        # Ajusta gpu_min_elements según ancho de banda PCIe medido
                        # ~16 MB / (pcie_bw GB/s) → tiempo de transferencia < 1ms
                        optimal_mb = self._hw_caps.measured_pcie_bw_gbs * 1.0  # 1 ms threshold
                        self.config.gpu_min_elements = max(
                            500_000,
                            int(optimal_mb * 1024 * 1024 // 8),  # float64 elements
                        )
                    else:
                        # quick_mode: mide PCIe en línea (rápido, ~0.3s) una sola vez
                        pcie_bw = self._quick_measure_pcie()
                        if pcie_bw > 0:
                            self._hw_caps.measured_pcie_bw_gbs = pcie_bw
                            # 1 ms de umbral de transferencia → mínimo 200K elementos
                            optimal_mb = pcie_bw * 1.0
                            self.config.gpu_min_elements = max(
                                200_000,
                                int(optimal_mb * 1024 * 1024 // 8),
                            )
                            # Persistir la medición para la próxima sesión
                            try:
                                profiler._save(self._hw_caps)
                            except Exception:
                                pass
                        else:
                            # GPU disponible pero sin medición — umbral conservador
                            self.config.gpu_min_elements = 500_000
            except Exception:
                pass

        # — Telemetría persistente ────────────────────────────────────────────
        self._telemetry = GideonTelemetry(db_path=self.config.telemetry_path)

        # — Dispatcher con ML ─────────────────────────────────────────────────
        self._ml_dispatcher: Optional[MLDispatcher] = None
        if self.config.use_ml_dispatcher:
            self._ml_dispatcher = MLDispatcher(
                telemetry=self._telemetry,
                fallback_dispatcher=self._dispatcher,
            )

        # — Fold cache persistido ─────────────────────────────────────────────
        if self.config.persist_fold_cache:
            self._load_fold_cache()

        # — Rust core ─────────────────────────────────────────────────────────
        # Umbral para delegar a Rust: arrays donde SIMD C supera a numpy Python
        self._rust_min_elements: int = 10_000

        # — Triton backend (Frente 1) ─────────────────────────────────────────
        # Se inicializa aquí después de que _hw_caps ya está listo, para poder
        # pasarle los parámetros óptimos derivados del perfil de hardware.
        if _TRITON_BACKEND_AVAILABLE and _GideonTritonBackend is not None:
            try:
                self._triton_backend = _GideonTritonBackend(hw_caps=self._hw_caps)
                if self.config.verbose and self._triton_backend.available:
                    print(f"[Gideon] Triton backend activo (v{self._triton_backend.version})")
            except Exception:
                self._triton_backend = None

        if self.config.verbose:
            print(f"[Gideon] {self._NAME} v{self._VERSION} iniciado")
            print(self._dispatcher.hardware_summary())
            if self._hw_caps:
                from .gideon_autotune import GideonHardwareProfiler as _P
                print(GideonHardwareProfiler().summary(self._hw_caps))
            print(f"[Gideon] Rust core: {'activo' if RUST_CORE_AVAILABLE else 'no disponible'}")

        # — Freeze state ──────────────────────────────────────────────────────
        # Set to True by warmup(freeze=True). When frozen, run_fma raises
        # FrozenGraphError if it encounters a cache miss (i.e. a new shape or
        # backend that wasn't pre-compiled during warmup).
        self._frozen: bool = False

    # ── Warmup / static pre-compilation ──────────────────────────────────────

    def warmup(
        self,
        fma_chain: List[Any],
        input_shapes: List[tuple],
        backends: Optional[List[str]] = None,
        freeze: bool = False,
    ) -> None:
        """
        Pre-compile kernels for every (fma_chain, shape) pair so that the
        inference loop sees only O(1) cache lookups — no JIT overhead at runtime.

        This MUST be called before the temporal inference loop when O(1) dispatch
        latency is required. After warmup(freeze=True) any cache miss inside
        run_fma raises FrozenGraphError instead of recompiling silently.

        Parameters
        ----------
        fma_chain    : the FMA sequence to pre-compile.
        input_shapes : list of concrete input shapes, e.g. [(1024, 256)].
                       Shapes must be exact — symbolic shapes are not supported.
        backends     : backends to warm up (default: all available backends).
        freeze       : if True, mark the engine as frozen after warmup.
                       run_fma will raise FrozenGraphError on any cache miss.
        """
        import numpy as _np
        for shape in input_shapes:
            x_dummy = _np.zeros(shape, dtype=_np.float64)
            try:
                self.run_fma(fma_chain, x_dummy)
            except Exception:
                pass  # cache-populate; errors in actual compute are silenced here
        self._frozen = freeze
        if self.config.verbose:
            status = "frozen (O(1) dispatch guaranteed)" if freeze else "warm (no freeze)"
            print(f"[Gideon] warmup complete — {len(input_shapes)} shape(s), {status}")

    @property
    def is_frozen(self) -> bool:
        """True after warmup(freeze=True). run_fma rejects cache misses."""
        return self._frozen

    # ── Medición rápida de PCIe bandwidth ───────────────────────────────────

    @staticmethod
    def _quick_measure_pcie(size_mb: int = 64, repeats: int = 2) -> float:
        """
        Mide el ancho de banda PCIe CPU→GPU en GB/s.
        Rápido (~0.3s). Se llama UNA sola vez desde __init__ cuando GPU está
        disponible pero aún no hay medición en el perfil persistido.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return 0.0
            arr = torch.randn(size_mb * 1024 * 1024 // 4)  # float32
            # Warmup
            _ = arr.cuda()
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(repeats):
                t0 = time.perf_counter()
                gpu_t = arr.cuda()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                best = min(best, elapsed)
                del gpu_t
            return (size_mb / 1024.0) / best  # GB/s
        except Exception:
            return 0.0

    # ── Backend registry (lazy) ───────────────────────────────────────────────

    def _get_registry(self):
        if self._backend_registry is None:
            from ..registry import BackendRegistry
            self._backend_registry = BackendRegistry
        return self._backend_registry

    # ── Pipeline principal desde FMA sequence ────────────────────────────────

    def run_fma(
        self,
        fma_sequence: List[Any],
        x: Any,
        name: str = "gideon_run",
    ) -> GideonExecutionResult:
        """
        Pipeline completo desde cadena FMA hasta resultado numérico.

          fma_sequence: lista de objetos con .weight y .bias
          x: input (numpy array o torch tensor)

        Optimizaciones activas (configurables en GideonEngineConfig):
          • fold_affine   — reduce N FMAs a 1 op algebraica (W·x+B)
          • fast_mode     — omite análisis IR/grafo para máxima velocidad
          • gpu_min_elements — usa GPU/CUDA cuando len(x) ≥ umbral
          • compilación cacheada por hash de cadena (evita CFFI overhead)
        """
        from .ir import GideonIR as _GIR
        t0 = time.perf_counter()
        cfg = self.config
        cache_hit = False
        folded = False
        gpu_used = False

        # ── 1. Identificación de cadena ───────────────────────────────────────
        chain_key = _GIR.chain_hash(fma_sequence)
        full_key = f"{chain_key}:{cfg.precision}:{cfg.preferred_backend or ''}"

        # Guard: when frozen, cache misses are not allowed (no JIT at inference)
        if self._frozen and full_key not in self._compiled_cache:
            raise FrozenGraphError(
                f"GideonEngine is frozen (warmup(freeze=True) was called). "
                f"The key '{full_key}' was not pre-compiled. "
                f"Either call warmup() again with this shape / backend combination, "
                f"or set freeze=False to allow dynamic compilation at runtime."
            )

        # ── 2. Detectar si podemos usar GPU ──────────────────────────────────
        hw = self._dispatcher.hw
        x_np = np.asarray(x, dtype=np.float64)
        _GPU_BACKEND_NAMES = {"pytorch_gpu", "pytorch_gpu_folded", "cuda"}
        use_gpu = (
            (cfg.preferred_backend is None or cfg.preferred_backend in _GPU_BACKEND_NAMES)
            and hw.has_cuda
            and hw.has_torch
            and x_np.size >= cfg.gpu_min_elements
        )

        # ── 3. Búsqueda en caché de compilación ──────────────────────────────
        #    El caché almacena (callable, backend_name, global_epsilon, total_fma)
        _cached = self._compiled_cache.get(full_key)
        if _cached is not None:
            callable_fn, backend_used, global_eps, total_fma_count = _cached
            cache_hit = True
            # Fold/gpu flags guardados junto al callable
            folded = self._compiled_cache.get(f"{full_key}:folded", False)
            gpu_used = self._compiled_cache.get(f"{full_key}:gpu", False)
            prog = None
            g_stats: Dict[str, Any] = {}
            decision = None
        else:
            # ── 4. Build IR (siempre; o skip en fast_mode puro) ───────────────
            if not cfg.fast_mode:
                prog = self._ir.from_fma_sequence(
                    fma_sequence, domain=cfg.domain,
                    precision=cfg.precision, name=name,
                )
                graph = GideonGraph(prog)
                g_stats = graph.stats()
                # —— Usar MLDispatcher si está disponible, sino heurístico ─
                _dec_src = (
                    self._ml_dispatcher
                    if self._ml_dispatcher is not None
                    else self._dispatcher
                )
                decision = _dec_src.decide(prog, precision=cfg.precision)
                global_eps = prog.global_epsilon
                total_fma_count = prog.total_fma
            else:
                # fast_mode: build IR mínimo sólo para ε y total_fma
                prog = self._ir.from_fma_sequence(
                    fma_sequence, domain=cfg.domain,
                    precision=cfg.precision, name=name,
                )
                g_stats = {}
                decision = None
                global_eps = prog.global_epsilon
                total_fma_count = prog.total_fma

            # ── 5. Intento de pliegue algebraico ────────────────────────────
            wb: Optional[Tuple[float, float]] = self._fold_cache.get(chain_key, _SENTINEL)
            if wb is _SENTINEL:  # tipo: not yet computed
                wb = _GIR.fold_affine_chain(prog) if cfg.fold_affine else None
                self._fold_cache[chain_key] = wb

            # ── 6. Selección de callable ─────────────────────────────────────
            if use_gpu:
                callable_fn, backend_used = self._compile_fma_gpu(
                    fma_sequence, wb=wb
                )
                gpu_used = True
                folded = wb is not None
            elif wb is not None and cfg.fold_affine:
                W, B = wb
                _W, _B = float(W), float(B)
                def callable_fn(arr, __W=_W, __B=_B):  # noqa: E731
                    return __W * np.asarray(arr, dtype=np.float64) + __B
                backend_used = "affine_fold"
                folded = True
            elif RUST_CORE_AVAILABLE and x_np.size >= self._rust_min_elements:
                # ── Rust/C kernel: más rápido que numpy para arrays > 10K ───
                _rust_weights = [float(getattr(f, "weight", 1.0)) for f in fma_sequence]
                _rust_biases  = [float(getattr(f, "bias",   0.0)) for f in fma_sequence]
                def callable_fn(arr, _ws=_rust_weights, _bs=_rust_biases, _prec=cfg.precision):  # noqa: E731
                    res = _rust_run_fma(_ws, _bs, list(np.asarray(arr, dtype=np.float64)), _prec)
                    if res is not None and res.get("success", False):
                        return np.asarray(res["output"], dtype=np.float64)
                    # Fallback inline si Rust falla
                    y = np.asarray(arr, dtype=np.float64).copy()
                    for w, b in zip(_ws, _bs):
                        y = w * y + b
                    return y
                backend_used = "rust_c_avx2"
            else:
                primary = cfg.preferred_backend or (
                    decision.primary_backend if decision else "c_native"
                )
                fallback = decision.fallback_backend if decision else "numpy_cpu"
                callable_fn, backend_used = self._compile_fma(
                    fma_sequence, primary, fallback
                )

            # ── 7. Guardar en caché ──────────────────────────────────────────
            self._compiled_cache[full_key] = (
                callable_fn, backend_used, global_eps, total_fma_count
            )
            self._compiled_cache[f"{full_key}:folded"] = folded
            self._compiled_cache[f"{full_key}:gpu"] = gpu_used
            # Evición: limitar a 512 cadenas compiladas
            _chain_keys = [
                k for k in self._compiled_cache
                if not k.endswith((":folded", ":gpu"))
            ]
            if len(_chain_keys) > 512:
                for k in _chain_keys[: len(_chain_keys) // 4]:
                    self._compiled_cache.pop(k, None)
                    self._compiled_cache.pop(f"{k}:folded", None)
                    self._compiled_cache.pop(f"{k}:gpu", None)

        # ── 8. Ejecutar ───────────────────────────────────────────────────────
        t_exec = time.perf_counter()
        run_input = x_np if (gpu_used or folded) else x
        if cfg.benchmark_mode:
            output = self._run_benchmark(callable_fn, run_input, cfg.benchmark_repeats)
        else:
            output = callable_fn(run_input)
        elapsed_exec = (time.perf_counter() - t_exec) * 1000
        self._dispatcher.record_latency(backend_used, elapsed_exec)

        # ── 9. Análisis de teoremas (opcional, solo primera vez) ─────────────
        theorem_cands: List[TheoremCandidate] = []
        if cfg.enable_theorem_seeds and not cache_hit:
            try:
                theorem_cands = self._theorem_engine.analyse(
                    callable_fn, domain=cfg.domain, fn_name=name
                )
            except Exception:
                pass

        elapsed_total = (time.perf_counter() - t0) * 1000

        result = GideonExecutionResult(
            output=output,
            program=prog,
            graph_stats=g_stats if not cache_hit else {},
            dispatch_decision=decision,
            theorem_candidates=theorem_cands,
            elapsed_ms=elapsed_total,
            backend_used=backend_used,
            total_fma=total_fma_count if not cache_hit else (
                self._compiled_cache[full_key][3]
            ),
            global_epsilon=global_eps if not cache_hit else (
                self._compiled_cache[full_key][2]
            ),
            success=True,
            cache_hit=cache_hit,
            folded=folded,
            gpu_used=gpu_used,
        )

        # —— 10. Registrar en telemetría (bucle de retroalimentación) ─────────
        try:
            self._telemetry.record(
                result,
                chain_hash=chain_key,
                n_elements=int(x_np.size),
                precision=cfg.precision,
            )
        except Exception:
            pass

        return result

    # ── Ejecución en lote ─────────────────────────────────────────────────────

    def run_batch(
        self,
        fma_sequence: List[Any],
        inputs: List[Any],
        name: str = "batch_run",
    ) -> List[GideonExecutionResult]:
        """
        Ejecuta la misma cadena FMA sobre una lista de inputs.

        Compila la cadena UNA sola vez y reutiliza el callable para todos los
        inputs, aprovechando el caché de compilación.

        Parámetros
        ----------
        fma_sequence : lista de objetos con .weight y .bias
            La MISMA cadena FMA aplicada a todos los inputs.
        inputs : lista de np.ndarray
            Cada elemento se pasa individualmente a run_fma().
        name : str
            Prefijo de nombre para cada ejecución.

        Nota: Si quieres pasar pesos y sesgos como listas separadas usa
        run_fma_batch(weights, biases, inputs).

        Ejemplo:
            chain = build_fma_chain(weights, biases)
            results = engine.run_batch(chain, [x1, x2, x3])
            outputs = [r.output for r in results]
        """
        results = []
        for i, x_i in enumerate(inputs):
            r = self.run_fma(fma_sequence, x_i, name=f"{name}_{i}")
            results.append(r)
        return results

    def run_fma_batch(
        self,
        weights: List[float],
        biases: List[float],
        inputs: List[Any],
        name: str = "fma_batch",
    ) -> List[GideonExecutionResult]:
        """
        Conveniencia: ejecuta la misma cadena FMA (pesos y sesgos como listas)
        sobre múltiples inputs.

        Equivalente a llamar run_fma(zip(weights,biases), x) para cada x
        en inputs.

        Parámetros
        ----------
        weights : List[float] — pesos de la cadena FMA
        biases  : List[float] — sesgos de la cadena FMA
        inputs  : List[np.ndarray] — lista de vectores de entrada
        name    : str — prefijo de nombre

        Ejemplo:
            results = engine.run_fma_batch([2.0, 0.5], [1.0, -0.5], [x1, x2, x3])
            outputs = [r.output for r in results]
        """
        if len(weights) != len(biases):
            raise ValueError(
                f"run_fma_batch: weights ({len(weights)}) y biases ({len(biases)}) "
                f"deben tener la misma longitud."
            )

        class _FMA:
            __slots__ = ("weight", "bias")
            def __init__(self, w: float, b: float) -> None:
                self.weight = w
                self.bias = b

        fma_seq = [_FMA(w, b) for w, b in zip(weights, biases)]
        return self.run_batch(fma_seq, inputs, name=name)

    # ── Compilación GPU (CUDA/PyTorch) ────────────────────────────────────────

    def _compile_fma_gpu(
        self,
        fma_sequence: Optional[List[Any]],
        wb: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Callable, str]:
        """
        Compila la cadena FMA para ejecución en GPU.

        Jerarquía de backends (de más rápido a fallback):
          1. Triton fusionado — 1 kernel launch, sin round-trips DRAM intermedios.
                                Speedup ~N× respecto a PyTorch loop para N FMAs.
          2. PyTorch folded  — 1 kernel y=W·x+B cuando la cadena es afín plegada.
          3. PyTorch chain   — N kernels secuenciales (fallback si Triton falla).
          4. NumPy CPU       — último recurso.

        Devuelve (callable, backend_name).
        """
        # Extraer pesos y biases antes de decidir qué backend usar
        weights = [float(getattr(f, "weight", 1.0)) for f in (fma_sequence or [])]
        biases  = [float(getattr(f, "bias", 0.0))   for f in (fma_sequence or [])]

        # — Ruta 1: Triton fusionado (Frente 1 — bare-metal) ─────────────────
        if (
            self._triton_backend is not None
            and self._triton_backend.available
            and len(weights) > 0
        ):
            try:
                if wb is not None:
                    # Cadena plegada: usa el kernel pointwise de Triton
                    fn = self._triton_backend.get_folded_fn(wb[0], wb[1])
                    return fn, "triton_folded"
                else:
                    fn = self._triton_backend.get_fma_chain_fn(weights, biases)
                    return fn, f"triton_chain_n{len(weights)}"
            except Exception as e:
                if self.config.verbose:
                    print(f"[Gideon] Triton path falló ({e}), usando PyTorch")

        # — Ruta 2: PyTorch (fallback) ─────────────────────────────────────────
        try:
            import torch

            if wb is not None:
                _W = float(wb[0])
                _B = float(wb[1])

                def _gpu_folded(arr: np.ndarray, __W=_W, __B=_B) -> np.ndarray:
                    t = torch.as_tensor(arr, dtype=torch.float64).cuda()
                    out = __W * t + __B
                    return out.cpu().numpy()

                return _gpu_folded, "pytorch_gpu_folded"

            # PyTorch chain: N kernels secuenciales
            def _gpu_chain(arr: np.ndarray, _ws=weights, _bs=biases) -> np.ndarray:
                t = torch.as_tensor(arr, dtype=torch.float64).cuda()
                for w, b in zip(_ws, _bs):
                    t = w * t + b
                return t.cpu().numpy()

            return _gpu_chain, "pytorch_gpu"

        except Exception as e:
            if self.config.verbose:
                print(f"[Gideon] GPU path falló ({e}), usando numpy fallback")
            seq = fma_sequence or []
            return self._numpy_fallback(seq), "numpy_fallback"

    # ── Caché: información y control ─────────────────────────────────────────

    def cache_info(self) -> Dict[str, Any]:
        """Devuelve estadísticas del caché de compilación."""
        n_entries = sum(1 for k in self._compiled_cache if not k.endswith((":folded", ":gpu")))
        n_folded  = sum(1 for k, v in self._compiled_cache.items()
                        if k.endswith(":folded") and v)
        n_gpu     = sum(1 for k, v in self._compiled_cache.items()
                        if k.endswith(":gpu") and v)
        n_fold_cached = len(self._fold_cache)
        return {
            "compiled_chains": n_entries,
            "folded_chains":   n_folded,
            "gpu_chains":      n_gpu,
            "fold_analyzed":   n_fold_cached,
        }

    def cache_clear(self, persist: bool = False) -> None:
        """
        Vacía el caché de compilación y fold.

        Si persist=True (y persist_fold_cache está activo), también borra
        el fold cache en disco.
        """
        self._compiled_cache.clear()
        self._fold_cache.clear()
        if persist and self.config.persist_fold_cache:
            self._save_fold_cache()

    # ── Fold cache persistido ─────────────────────────────────────────────────

    def _fold_cache_path(self) -> str:
        import os
        return os.path.expanduser("~/.gideon/fold_cache.json")

    def _load_fold_cache(self) -> None:
        """Carga el fold cache persistido desde disco."""
        import json, os
        path = self._fold_cache_path()
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                raw = json.load(f)
            for k, v in raw.items():
                if v is None:
                    self._fold_cache[k] = None
                elif isinstance(v, list) and len(v) == 2:
                    self._fold_cache[k] = (float(v[0]), float(v[1]))
        except Exception:
            pass  # fold cache corrupto → ignorar

    def _save_fold_cache(self) -> None:
        """Persiste el fold cache a disco."""
        import json, os
        path = self._fold_cache_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            serializable = {}
            for k, v in self._fold_cache.items():
                if v is _SENTINEL:
                    continue
                serializable[k] = list(v) if v is not None else None
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(serializable, f, separators=(",", ":"))
            os.replace(tmp, path)
        except Exception:
            pass

    # ── Autotune / Hardware Profile ───────────────────────────────────────────

    def autotune_info(self) -> Optional[HardwareCapabilities]:
        """
        Retorna el perfil de hardware detectado por el autotune.

        Retorna None si use_autotune=False en la configuración.
        Los campos measured_* solo tienen valores reales si autotune_quick=False.
        """
        return self._hw_caps

    def autotune_summary(self) -> str:
        """Resumen legible del perfil de hardware."""
        if self._hw_caps is None:
            return "Autotune: desactivado (use_autotune=False)"
        from .gideon_autotune import GideonHardwareProfiler as _P
        return GideonHardwareProfiler().summary(self._hw_caps)

    # ── Telemetría ───────────────────────────────────────────────────────────

    def telemetry_stats(self) -> Dict[str, Any]:
        """
        Estadísticas de la telemetría acumulada.

        Retorna un dict con:
          total_records, backend_stats (count/avg/p50/p95 por backend),
          dispatchers en uso, y estado del MLDispatcher.
        """
        stats = self._telemetry.get_backend_stats()
        return {
            "total_records": len(self._telemetry),
            "backend_stats": stats,
            "ml_dispatcher_active": self._ml_dispatcher is not None,
            "ml_dispatcher_summary": (
                self._ml_dispatcher.telemetry_summary()
                if self._ml_dispatcher is not None
                else "N/A"
            ),
        }

    def export_acf_calibration(self) -> Dict[str, Any]:
        """
        Exporta datos de calibración para el bucle cerrado Gideon→ACF.

        Este método cierra el ciclo:
            ACF → Poema → Gideon → (ejecución medida) → export_acf_calibration()
                ↑                                               │
                └───────────────────────────────────────────────┘
                  Calibración empírica de bounds formales

        ACF puede usar estos datos para calibrar:
          - ACFInvariant.compute_alpha(): usar datos empíricos de error
          - FormalVerificationSuite: ajustar cotas epsilon con latencias reales
          - Selección de backend en Poema cuando Gideon tiene historia

        Retorna un dict con todas las estadísticas de calibración.
        """
        calibration = self._telemetry.export_acf_calibration()
        # Agregar info del hardware para que ACF pueda afinar sus bounds
        if self._hw_caps is not None:
            calibration["hardware"] = {
                "cpu_arch": self._hw_caps.cpu_arch,
                "avx_level": self._hw_caps.cpu_avx_level,
                "gpu_available": self._hw_caps.gpu_available,
                "gpu_cc": self._hw_caps.gpu_cc_str if self._hw_caps.gpu_available else "N/A",
                "fma_gflops": self._hw_caps.measured_fma_scalar_gflops,
                "memory_bw_gbs": self._hw_caps.measured_memory_bw_gbs,
            }
        calibration["fold_cache_size"] = len(
            {k: v for k, v in self._fold_cache.items() if v is not _SENTINEL}
        )
        return calibration

    # ── Pipeline desde AST de Poema ───────────────────────────────────────────

    def run_ast(
        self,
        ast_node: Any,
        x: Any,
        name: str = "ast_run",
    ) -> GideonExecutionResult:
        """
        Pipeline completo desde AST de Poema (ASTNode).
        Usa el PoemCompiler interno para obtener la secuencia FMA.
        """
        t0 = time.perf_counter()
        cfg = self.config

        # Lower AST a IR (sin compilar con Poema, usamos el lowering nativo)
        prog = self._ir.from_ast(
            ast_node,
            domain=cfg.domain,
            precision=cfg.precision,
            name=name,
        )

        # Necesitamos un callable — extraer pesos del IR
        fma_fakes = self._ir_to_fma_list(prog)

        # Reusar pipeline run_fma
        result = self.run_fma(fma_fakes, x, name=name)
        result.program = prog
        result.total_fma = prog.total_fma
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    # ── Compilación inteligente ───────────────────────────────────────────────

    def _compile_fma(
        self,
        fma_sequence: List[Any],
        primary: str,
        fallback: str,
    ) -> Tuple[Callable, str]:
        """Intenta compilar con primary, luego fallback, luego numpy puro."""
        registry = self._get_registry()

        for bname in [primary, fallback, "numpy_cpu"]:
            try:
                backend = registry.get(bname)
                if not backend.verify_available():
                    continue
                result = backend.compile(
                    fma_sequence,
                    source_ast=None,
                    domain=self.config.domain,
                    precision=self.config.precision,
                )
                if result.callable_fn is not None:
                    return result.callable_fn, bname
            except Exception as e:
                if self.config.verbose:
                    print(f"[Gideon] Backend {bname} falló: {e}")
                continue

        # Ultra-fallback: numpy puro
        return self._numpy_fallback(fma_sequence), "numpy_fallback"

    @staticmethod
    def _numpy_fallback(fma_sequence: List[Any]) -> Callable:
        weights = [float(getattr(f, "weight", 1.0)) for f in fma_sequence]
        biases  = [float(getattr(f, "bias", 0.0)) for f in fma_sequence]

        def fn(x):
            y = np.asarray(x, dtype=np.float64).copy()
            for w, b in zip(weights, biases):
                y = w * y + b
            return y

        return fn

    # ── Benchmark interno ─────────────────────────────────────────────────────

    @staticmethod
    def _run_benchmark(fn: Callable, x: Any, repeats: int) -> Any:
        """Ejecuta fn N veces sobre x, devuelve el último resultado."""
        result = None
        for _ in range(repeats):
            result = fn(x)
        return result

    # ── Conversión IR → lista FMA ─────────────────────────────────────────────

    @staticmethod
    def _ir_to_fma_list(prog: GideonProgram) -> List[Any]:
        """
        Extrae parámetros FMA del IR para pasar al backend.
        Convierte AFFINE, SCALE, SHIFT a FMAInstruction compatibles.
        """
        from .ir import IRNodeKind

        class _FMA:
            __slots__ = ("weight", "bias")
            def __init__(self, w, b):
                self.weight = w
                self.bias = b

        fma_list = []
        for nid in prog.topo_order:
            node = prog.nodes.get(nid)
            if node is None:
                continue
            kind = node.kind
            p = node.params
            if kind == IRNodeKind.FMA:
                fma_list.append(_FMA(p.get("weight", 1.0), p.get("bias", 0.0)))
            elif kind == IRNodeKind.AFFINE:
                fma_list.append(_FMA(p.get("alpha", 1.0), p.get("beta", 0.0)))
            elif kind == IRNodeKind.SCALE:
                fma_list.append(_FMA(p.get("alpha", 1.0), 0.0))
            elif kind == IRNodeKind.SHIFT:
                fma_list.append(_FMA(1.0, p.get("beta", 0.0)))
            elif kind == IRNodeKind.CONST:
                fma_list.append(_FMA(0.0, p.get("value", 0.0)))
        return fma_list

    # ── Análisis de arquitecturas IA ──────────────────────────────────────────

    def analyse_blueprint(self, bp: ArchitectureBlueprint) -> Dict[str, Any]:
        """Analiza un ArchitectureBlueprint con las métricas de Gideon."""
        return GideonNeuralHints.analyse_blueprint(bp)

    def create_mlp_blueprint(self, layer_dims: List[int]) -> ArchitectureBlueprint:
        return GideonNeuralHints.mlp(layer_dims)

    def create_transformer_blueprint(self, d_model: int = 512, n_layers: int = 6) -> ArchitectureBlueprint:
        return GideonNeuralHints.transformer(d_model=d_model, n_layers=n_layers)

    # ── Análisis de teoremas ──────────────────────────────────────────────────

    def probe_theorems(
        self,
        fn: Callable,
        domain: Tuple[float, float],
        fn_name: str = "f",
    ) -> List[TheoremCandidate]:
        """Analiza fn en busca de propiedades matemáticas/teoremas."""
        return self._theorem_engine.analyse(fn, domain, fn_name=fn_name)

    def export_lean_theorems(self, path: str) -> None:
        """Exporta esqueletos Lean 4 de todos los candidatos."""
        self._theorem_engine.export_lean_file(path)

    # ── Auto-evolución ACF ────────────────────────────────────────────────────

    def auto_evolve_fma(
        self,
        fma_sequence: List[Any],
        domain: Tuple[float, float],
        config: Optional[Any] = None,
    ):
        """
        Run the ACF auto-evolution pipeline on an FMA sequence.

        Converts the FMA chain to a callable function and runs
        ACFAutoEvolver to find the best polynomial representation.

        The four mechanisms applied are:
          1. Thermodynamic search — optimal degree via F(d, β)
          2. Fixed-point iteration — Φ² = Φ convergence
          3. Bifunctorial cycle   — Φ* ⊣ Φ alternating compression/synthesis
          4. Adaptive refinement  — residual-guided local degree increase

        Parameters
        ----------
        fma_sequence : list of FMA objects with .weight and .bias
        domain : (a, b) input domain
        config : ACFAutoEvolverConfig or None

        Returns
        -------
        AutoEvolutionResult
            Full trace. Use result.best_reduction for the evolved polynomial.
        """
        from acf_functor.auto_evolution import ACFAutoEvolver, ACFAutoEvolverConfig
        import numpy as np
        import torch

        # Build numpy-level callable from the FMA chain
        weights = [float(getattr(f, "weight", 1.0)) for f in fma_sequence]
        biases  = [float(getattr(f, "bias",   0.0)) for f in fma_sequence]

        def fma_fn(x: torch.Tensor) -> torch.Tensor:
            y = x.clone().to(torch.float64)
            for w, b in zip(weights, biases):
                y = w * y + b
            return y

        cfg = config or ACFAutoEvolverConfig()
        evolver = ACFAutoEvolver(config=cfg)
        return evolver.evolve(fma_fn, domain)

    # ── Graph ACF ─────────────────────────────────────────────────────────────

    def reduce_graph_signal(
        self,
        adjacency,
        signal_values,
        normalization: str = "symmetric",
        filter_degree: int = 10,
        config=None,
    ):
        """
        Apply ACF to a graph signal via spectral reduction.

        Reduces the graph signal s: V → ℝ to a polynomial spectral filter
        H(λ) = Σₖ cₖ Tₖ(λ̃) using Chebyshev polynomials on the Laplacian
        eigenvalues. The filtered signal is reconstructed as U·H(Λ)·Uᵀ·s.

        Parameters
        ----------
        adjacency : np.ndarray or torch.Tensor shape (n, n)
        signal_values : array-like shape (n,)
        normalization : "unnormalized" | "symmetric" | "random_walk"
        filter_degree : degree of polynomial filter
        config : optional ACFAutoEvolverConfig for auto-evolution

        Returns
        -------
        GraphReductionResult or GraphEvolutionResult (if config provided)
        """
        import numpy as np
        import torch
        from acf_functor.graph_acf import (
            GraphLaplacian, GraphReducer, GraphSignal, GraphSignalEvolver
        )
        from acf_functor.auto_evolution import ACFAutoEvolverConfig

        if isinstance(signal_values, np.ndarray):
            signal_values = torch.tensor(signal_values, dtype=torch.float64)
        elif not isinstance(signal_values, torch.Tensor):
            signal_values = torch.tensor(signal_values, dtype=torch.float64)

        spectrum = GraphLaplacian.from_adjacency(adjacency, normalization=normalization)
        signal = GraphSignal(values=signal_values, n_nodes=spectrum.n_nodes)

        if config is not None:
            evolver = GraphSignalEvolver(config=config)
            return evolver.evolve(signal, spectrum)
        else:
            reducer = GraphReducer(filter_degree=filter_degree)
            return reducer.reduce(signal, spectrum)

    def analyse_graph(self, adjacency, normalization: str = "symmetric", beta: float = 1.0):
        """
        Compute ACF invariants (α, NC-class, Fiedler value, spectral entropy)
        for a graph given its adjacency matrix.

        Returns
        -------
        GraphACFInvariants
        """
        from acf_functor.graph_acf import GraphLaplacian, GraphACFAnalyzer
        spectrum = GraphLaplacian.from_adjacency(adjacency, normalization=normalization)
        analyzer = GraphACFAnalyzer(beta=beta)
        return analyzer.analyse(spectrum)

    # ── Neural ACF ────────────────────────────────────────────────────────────

    def analyse_network(
        self,
        network,
        degree: int = 15,
        domain=(-3.0, 3.0),
        activation: str = "tanh",
        as_dict: bool = True,
    ):
        """
        Analyse a PyTorch nn.Module with ACF.

        Computes per-layer reductions (polynomial filter per layer) and
        ACF invariants (α, NC-class, singular value entropy, spectral norm).

        Parameters
        ----------
        network : nn.Sequential or nn.Module with nn.Linear / nn.Conv1d layers
        degree : polynomial degree for per-layer reduction
        domain : input domain for scalar reduction
        activation : activation name ("relu", "tanh", "gelu", "sigmoid")
        as_dict : bool (default True)
            Si True, devuelve un dict plano compatible con JSON.
            Si False, devuelve el objeto NetworkACFReport completo.

        Returns
        -------
        dict (default) o NetworkACFReport (as_dict=False)
        """
        from acf_functor.neural_acf import NetworkACFAnalyzer
        analyzer = NetworkACFAnalyzer(degree=degree, domain=domain, activation=activation)
        report = analyzer.analyse(network)
        if not as_dict:
            return report
        # Serializar a dict plano para uniformidad con el resto de la API
        return {
            "n_layers": len(report.layer_reductions),
            "global_alpha": report.global_alpha,
            "global_nc_class": report.global_nc_class,
            "total_fma_count": report.total_fma_count,
            "total_elapsed_ms": report.total_elapsed_ms,
            "layer_reductions": [
                {
                    "layer_name": lr.layer_name,
                    "layer_type": lr.layer_type,
                    "epsilon": lr.epsilon,
                    "input_dim": lr.input_dim,
                    "output_dim": lr.output_dim,
                    "filter_degree": lr.filter_degree,
                    "elapsed_ms": lr.elapsed_ms,
                    "fma_count": len(lr.fma_chain),
                }
                for lr in report.layer_reductions
            ],
            "layer_invariants": [
                {
                    "layer_name": li.layer_name,
                    "alpha": li.alpha,
                    "delta": li.delta,
                    "nc_class": li.nc_class,
                    "singular_value_entropy": li.singular_value_entropy,
                    "rank": li.rank,
                    "spectral_norm": li.spectral_norm,
                }
                for li in report.layer_invariants
            ],
            "summary": report.summary(),
            "metadata": report.metadata,
        }

    def evolve_network_function(
        self,
        network,
        domain,
        input_dim: int = 1,
        config=None,
    ):
        """
        Auto-evolve the polynomial representation of the function
        implemented by a (shallow) MLP.

        Parameters
        ----------
        network : nn.Module (best for shallow networks, e.g. 1-16-16-1)
        domain : (a, b) input domain
        input_dim : input dimensionality (treated as 1-D diagonal slice)
        config : ACFAutoEvolverConfig or None

        Returns
        -------
        NeuralEvolutionResult
        """
        from acf_functor.neural_acf import NeuralACFEvolver
        from acf_functor.auto_evolution import ACFAutoEvolverConfig
        evolver = NeuralACFEvolver(config=config)
        return evolver.evolve(network, domain=domain, input_dim=input_dim)

    def analyse_training_trajectory(self, trajectory):
        """
        Koopman analysis of a training trajectory (e.g., loss over steps).

        Parameters
        ----------
        trajectory : list or 1-D array of scalar values (loss, weight norm, etc.)

        Returns
        -------
        KoopmanNetworkResult with eigenvalues, spectral diagnostics, and
        a ReductionResult representing the dominant dynamics.
        """
        from acf_functor.neural_acf import KoopmanNetworkDynamics
        analyser = KoopmanNetworkDynamics()
        return analyser.analyse(trajectory)

    # ── Meta-Compiler ─────────────────────────────────────────────────────────

    def meta_compile(
        self,
        f,
        domain,
        strategy: str = "greedy",
        beta: float = 1.0,
        target_epsilon: float = 1e-8,
        enable_auto_evolution: bool = True,
        config=None,
    ):
        """
        Run the ACF Meta-Compiler on a function f over domain [a, b].

        The meta-compiler searches over the grammar space (Chebyshev,
        Fourier, Legendre, RBF, Koopman-Poly, Koopman-Fourier, Koopman-Mixed)
        and selects the grammar G* = argmin F(G, f, β) where
        F = ε - S/β (Helmholtz free energy over the grammar space).

        Optionally fine-tunes the winner with ACFAutoEvolver.

        Parameters
        ----------
        f : callable torch.Tensor → torch.Tensor
            La función a compilar. DEBE ser un callable, no pesos o sesgos.
            Para cadenas FMA usa meta_compile_fma(weights, biases, domain).
        domain : (a, b)
        strategy : "greedy" (default, fast) | "grid" (exhaustive) | "random"
        beta : inverse temperature (high → prioritise accuracy)
        target_epsilon : early stop if ε < target_epsilon
        enable_auto_evolution : fine-tune winner with 4-mechanism auto-evolution
        config : MetaCompilerConfig or None (overrides all above if provided)

        Returns
        -------
        MetaCompilerResult with best_grammar, best_reduction, trace, and
        improvement metrics.
        """
        if not callable(f):
            raise TypeError(
                f"meta_compile() requiere un callable (torch.Tensor → torch.Tensor) "
                f"como primer argumento, se recibió {type(f).__name__!r}.\n"
                f"Para cadenas FMA usa: engine.meta_compile_fma(weights, biases, domain)"
            )
        from acf_functor.meta_compiler import ACFMetaCompiler, MetaCompilerConfig
        if config is not None:
            mc = ACFMetaCompiler(config=config)
        else:
            mc = ACFMetaCompiler(MetaCompilerConfig(
                strategy=strategy,
                beta=beta,
                target_epsilon=target_epsilon,
                enable_auto_evolution=enable_auto_evolution,
            ))
        return mc.compile(f, domain)

    def meta_compile_fma(
        self,
        weights: List[float],
        biases: List[float],
        domain,
        strategy: str = "greedy",
        beta: float = 1.0,
        target_epsilon: float = 1e-8,
        enable_auto_evolution: bool = True,
        config=None,
    ):
        """
        Conveniencia: Meta-compila una cadena FMA (pesos y sesgos como listas).

        Construye el callable FMA internamente y llama meta_compile().

        Parámetros
        ----------
        weights : List[float]
        biases  : List[float]
        domain  : (a, b)
        (resto igual que meta_compile)

        Ejemplo
        -------
            result = engine.meta_compile_fma([2.0, 0.5], [1.0, -0.5], (-1, 1))
            print(result.best_grammar)
        """
        import torch
        _ws = [float(w) for w in weights]
        _bs = [float(b) for b in biases]
        if len(_ws) != len(_bs):
            raise ValueError(
                f"meta_compile_fma: len(weights)={len(_ws)} != len(biases)={len(_bs)}"
            )

        def _fma_fn(x: torch.Tensor) -> torch.Tensor:
            y = x.to(torch.float64)
            for w, b in zip(_ws, _bs):
                y = w * y + b
            return y

        return self.meta_compile(
            _fma_fn, domain,
            strategy=strategy, beta=beta,
            target_epsilon=target_epsilon,
            enable_auto_evolution=enable_auto_evolution,
            config=config,
        )

    # ── Tensor ACF ────────────────────────────────────────────────────────────

    def reduce_tensor(
        self,
        func: Callable,
        domains: List[Tuple[float, float]],
        degrees: Optional[List[int]] = None,
        default_degree: int = 8,
        max_rank: int = 20,
        target_epsilon: float = 1e-8,
        method: str = "tt",
    ):
        """
        Reduce a multivariate function f: ℝᵈ → ℝ to a Tensor Train (or Tucker)
        FMA chain via Chebyshev tensor decomposition.

        Parameters
        ----------
        func : callable(*floats) → float
            Multivariate function accepting d positional float arguments.
        domains : list of (a, b) tuples
            Domain per dimension.
        degrees : list of int, optional
            Chebyshev degree per dimension. If None, uses default_degree.
        default_degree : int
            Default Chebyshev degree when degrees is None.
        max_rank : int
            Maximum TT rank for truncation.
        target_epsilon : float
            Target approximation error.
        method : "tt" | "tucker"
            Decomposition method.

        Returns
        -------
        TensorReductionResult or TuckerReductionResult
        """
        from acf_functor.tensor_acf import TensorACFReducer

        reducer = TensorACFReducer(
            degrees=degrees,
            default_degree=default_degree,
            max_rank=max_rank,
            target_epsilon=target_epsilon,
            method=method,
        )
        return reducer.reduce(func, domains, degrees=degrees)

    def analyse_tensor(
        self,
        func: Callable,
        domains: List[Tuple[float, float]],
        degrees: Optional[List[int]] = None,
        default_degree: int = 8,
        max_rank: int = 20,
        target_epsilon: float = 1e-8,
    ):
        """
        Analyse a multivariate function: compute TT decomposition and return
        ACF invariants (alpha per mode, global alpha, NC class, effective dimension).

        Returns
        -------
        TensorACFInvariants
        """
        from acf_functor.tensor_acf import TensorACFReducer

        reducer = TensorACFReducer(
            degrees=degrees,
            default_degree=default_degree,
            max_rank=max_rank,
            target_epsilon=target_epsilon,
            method="tt",
        )
        result = reducer.reduce(func, domains, degrees=degrees)
        return result.invariants

    # ── Matrix ACF ────────────────────────────────────────────────────────────

    def reduce_matrix_function(
        self,
        func,
        A,
        degree: int = 30,
        target_epsilon: float = 1e-8,
        max_degree: int = 128,
        spectral_range=None,
    ):
        """
        Reduce a matrix function f(A) via Chebyshev polynomials of the matrix.

        Parameters
        ----------
        func : str or callable(float) → float
            Scalar function name ("exp", "sqrt", "log", "inv", "sign", "tanh")
            or custom callable.
        A : torch.Tensor
            Square (preferably symmetric) matrix.
        degree : int
            Initial Chebyshev degree.
        target_epsilon : float
            Target approximation error.
        max_degree : int
            Maximum allowed Chebyshev degree.
        spectral_range : tuple (λ_min, λ_max), optional
            If None, computed automatically from eigenvalues.

        Returns
        -------
        MatrixReductionResult
        """
        from acf_functor.matrix_acf import ChebyshevMatrixReducer

        return ChebyshevMatrixReducer.reduce(
            func, A,
            degree=degree,
            target_epsilon=target_epsilon,
            max_degree=max_degree,
            spectral_range=spectral_range,
        )

    def analyse_matrix(
        self,
        A,
        func="exp",
        degree: int = 40,
    ):
        """
        Compute ACF invariants for a (f, A) pair: matrix alpha, spectral range,
        condition number, NC class, effective degree.

        Returns
        -------
        MatrixACFInvariants
        """
        from acf_functor.matrix_acf import MatrixACFAnalyzer

        return MatrixACFAnalyzer.analyse(A, func=func, degree=degree)

    def matrix_exp(self, A, t: float = 1.0, degree: int = 30, target_epsilon: float = 1e-10):
        """Convenience: compute exp(tA) via Matrix ACF."""
        from acf_functor.matrix_acf import MatrixExponential
        return MatrixExponential.reduce(A, t=t, degree=degree, target_epsilon=target_epsilon)

    def matrix_sqrt(self, A, degree: int = 30, target_epsilon: float = 1e-10):
        """Convenience: compute A^{1/2} via Matrix ACF. Requires SPD matrix."""
        from acf_functor.matrix_acf import MatrixSquareRoot
        return MatrixSquareRoot.reduce(A, degree=degree, target_epsilon=target_epsilon)

    def matrix_log(self, A, degree: int = 30, target_epsilon: float = 1e-10):
        """Convenience: compute log(A) via Matrix ACF. Requires SPD matrix."""
        from acf_functor.matrix_acf import MatrixLogarithm
        return MatrixLogarithm.reduce(A, degree=degree, target_epsilon=target_epsilon)

    def matrix_resolvent(self, A, sigma: float = 1.0, degree: int = 30, target_epsilon: float = 1e-10):
        """Convenience: compute (A + σI)⁻¹ via Matrix ACF."""
        from acf_functor.matrix_acf import MatrixResolvent
        return MatrixResolvent.reduce(A, sigma=sigma, degree=degree, target_epsilon=target_epsilon)

    # ── ODE / Control ACF ─────────────────────────────────────────────────────

    def reduce_vector_field(
        self,
        f: Callable,
        dimension: int,
        order: int = 8,
        decomposition: str = "tt",
        domain=None,
        n_samples: int = 500,
    ):
        """
        Reduce a vector field f: ℝⁿ → ℝⁿ component-wise via ODE-ACF.

        Parameters
        ----------
        f : callable
            Vector field f(x) → array of shape (n,).
        dimension : int
            State space dimension n.
        order : int
            Chebyshev order per dimension.
        decomposition : str
            Tensor decomposition: 'tt' | 'tucker' | 'cp'.
        domain : ndarray (n, 2), optional
            Per-dimension domain [aᵢ, bᵢ]. Default: [-1,1]^n.
        n_samples : int
            Number of sample points for fitting.

        Returns
        -------
        VectorFieldReducer (fitted)
        """
        from acf_functor.ode_acf import VectorFieldReducer
        import numpy as _np
        reducer = VectorFieldReducer(
            dimension=dimension,
            order=order,
            decomposition=decomposition,
            domain=_np.asarray(domain) if domain is not None else None,
        )
        reducer.fit(f, n_samples=n_samples)
        return reducer

    def analyse_ode(
        self,
        f: Callable,
        dimension: int,
        T: float = 1.0,
        order: int = 8,
        eps_target: float = 1e-3,
        domain=None,
    ):
        """
        Compute ODE-ACF invariants for vector field f.

        Returns ODEACFInvariants with α per component, Gronwall bound, stability.
        """
        from acf_functor.ode_acf import VectorFieldReducer
        import numpy as _np
        reducer = VectorFieldReducer(
            dimension=dimension,
            order=order,
            domain=_np.asarray(domain) if domain is not None else None,
        )
        reducer.fit(f)
        return reducer.invariants(f, T=T, eps_target=eps_target)

    def certify_lyapunov(
        self,
        V: Callable,
        f: Callable,
        dimension: int,
        radius: float = 1.0,
        grid_res: int = 15,
    ):
        """
        Numerically certify V as a Lyapunov function for ẋ = f(x).

        Returns LyapunovCertificate with stability verdict.
        """
        from acf_functor.ode_acf import LyapunovACF
        cert = LyapunovACF(dimension=dimension, radius=radius, grid_res=grid_res)
        return cert.certify(V, f)

    def optimize_hjb(
        self,
        V: Callable,
        f: Callable,
        l: Callable,
        x: "np.ndarray",
        u_candidates: "np.ndarray",
        dimension: int,
    ):
        """
        Extract optimal HJB policy π*(x) = argmin_u [l(x,u) + ∇V(x)·f(x,u)].

        Returns the optimal control u* from u_candidates.
        """
        from acf_functor.ode_acf import HJBReducer
        hjb = HJBReducer(dimension=dimension)
        hjb.fit(V)
        return hjb.optimal_policy(x, f, l, u_candidates)

    # ── Operator / Green Function ACF ─────────────────────────────────────────

    def reduce_green_function(
        self,
        G: Callable,
        n_points: int = 64,
        order: int = 16,
        domain=(-1.0, 1.0),
    ):
        """
        Reduce a 1D Green function G(x,y) via 2D TT-ACF.

        Parameters
        ----------
        G : callable
            Kernel G(x, y) → float.
        n_points : int
            Quadrature grid resolution.
        order : int
            Chebyshev expansion order.
        domain : tuple (a, b)
            Spatial domain.

        Returns
        -------
        GreenFunctionReducer (fitted)
        """
        from acf_functor.operator_acf import GreenFunctionReducer
        reducer = GreenFunctionReducer(n_points=n_points, order=order, domain=domain)
        reducer.fit(G)
        return reducer

    def apply_integral_operator(
        self,
        G: Callable,
        u: "np.ndarray",
        n_points: int = 64,
        rank: int = 16,
        domain=(-1.0, 1.0),
    ):
        """
        Apply integral operator (Lu)(x) = ∫G(x,y)u(y)dy via rank-R decomposition.

        Parameters
        ----------
        G : callable
            Kernel G(x, y) → float.
        u : ndarray, shape (n_points,)
            Input function on the quadrature grid.
        rank : int
            Number of separable terms.

        Returns
        -------
        Lu : ndarray, shape (n_points,)
        """
        from acf_functor.operator_acf import IntegralOperatorACF
        op = IntegralOperatorACF(n_points=n_points, rank=rank, domain=domain)
        op.fit(G)
        return op.apply(u)

    def compress_attention(
        self,
        Q: "np.ndarray",
        K: "np.ndarray",
        V: "np.ndarray",
        n_features: int = 64,
        feature_type: str = "random_fourier",
    ):
        """
        Compute linearized attention output in O(n·R·d) time.

        Parameters
        ----------
        Q, K, V : ndarray, shape (n_tokens, d)
        n_features : int
            Number of random Fourier features R.
        feature_type : str
            'random_fourier' | 'relu' | 'polynomial'

        Returns
        -------
        output : ndarray, shape (n_tokens, d)
        """
        from acf_functor.operator_acf import AttentionKernelReducer
        reducer = AttentionKernelReducer(
            embed_dim=Q.shape[-1],
            n_features=n_features,
            feature_type=feature_type,
        )
        reducer.fit()
        return reducer.fast_attention(Q, K, V)

    # ── Stochastic / PCE ACF ──────────────────────────────────────────────────

    def pce_expand(
        self,
        f: Callable,
        m: int,
        p: int = 3,
        family: str = "hermite",
        n_quad: int = 8,
        method: str = "projection",
    ):
        """
        Compute Polynomial Chaos Expansion of a stochastic function f(ξ).

        Parameters
        ----------
        f : callable
            Stochastic function f(ξ) → float, ξ ∈ ℝᵐ.
        m : int
            Number of random variables.
        p : int
            Maximum PCE degree.
        family : str
            Orthogonal polynomial basis: 'hermite' | 'legendre' | 'chebyshev'.
        n_quad : int
            Quadrature points per dimension.
        method : str
            'projection' (Gauss quadrature) or 'regression' (Monte Carlo).

        Returns
        -------
        PolynomialChaosACF (fitted)
        """
        from acf_functor.stochastic_acf import PolynomialChaosACF
        pce = PolynomialChaosACF(m=m, p=p, family=family, n_quad=n_quad)
        pce.fit(f, method=method)
        return pce

    def analyse_stochastic(
        self,
        f: Callable,
        m: int,
        p: int = 3,
        family: str = "hermite",
    ):
        """
        Compute stochastic ACF invariants: α_stoch, Sobol indices,
        effective dimension, mean, variance.

        Returns StochasticACFInvariants.
        """
        from acf_functor.stochastic_acf import PolynomialChaosACF
        pce = PolynomialChaosACF(m=m, p=p, family=family)
        pce.fit(f)
        return pce.invariants()

    def uncertainty_band(
        self,
        f: Callable,
        m: int,
        p: int = 3,
        k_sigma: float = 2.0,
        family: str = "hermite",
    ):
        """
        Compute E[f] ± k·√Var[f] uncertainty band with Chebyshev probability bound.

        Returns UncertaintyBound with {truncation_error, confidence_band, confidence_level}.
        """
        from acf_functor.stochastic_acf import PolynomialChaosACF, compute_uncertainty_bound
        pce = PolynomialChaosACF(m=m, p=p, family=family)
        pce.fit(f)
        return compute_uncertainty_bound(pce, k_sigma=k_sigma)

    # ── Rational / Padé ACF ───────────────────────────────────────────────────

    def pade_reduce(
        self,
        f: Callable,
        m: int = 5,
        n: int = 5,
        x0: float = 0.0,
    ):
        """
        Reduce a scalar function f: ℝ → ℝ via [m/n] Padé approximant.

        Evaluation cost: m + 2n + 2 FMA operations (two Horner chains + division).

        Parameters
        ----------
        f : callable
            Scalar function f: float → float.
        m, n : int
            Numerator / denominator degree.
        x0 : float
            Expansion point.

        Returns
        -------
        PadeReducer (fitted)
        """
        from acf_functor.rational_acf import PadeReducer
        r = PadeReducer(m=m, n=n, x0=x0)
        r.fit(f)
        return r

    def analyse_rational(
        self,
        f: Callable,
        m: int = 5,
        n: int = 5,
        x0: float = 0.0,
    ):
        """
        Compute Padé-ACF invariants: α_rational, pole locations, residue decay, H² norm.

        Returns PadeInvariants.
        """
        from acf_functor.rational_acf import PadeReducer
        r = PadeReducer(m=m, n=n, x0=x0)
        r.fit(f)
        return r.invariants(f_exact=f)

    def hardy_reduce(
        self,
        f_circle: Callable,
        n_modes: int = 32,
        n_quad: int = 256,
    ):
        """
        Fit a Hardy space H² approximation to f defined on the unit circle.

        Parameters
        ----------
        f_circle : callable
            f(z) → complex, z on the unit circle.
        n_modes : int
            Number of Taylor/Fourier modes to keep.

        Returns
        -------
        HardySpaceACF (fitted)
        """
        from acf_functor.rational_acf import HardySpaceACF
        r = HardySpaceACF(n_modes=n_modes)
        r.fit(f_circle, n_quad=n_quad)
        return r

    # ── Info del motor ────────────────────────────────────────────────────────

    def info(self) -> str:
        registry = self._get_registry()
        avail = registry.available()
        hw = self._dispatcher.hw
        ci = self.cache_info()
        tel = self._telemetry.get_backend_stats()
        lines = [
            f"╔══════════════════════════════════════════╗",
            f"║  GIDEON ENGINE v{self._VERSION}                  ║",
            f"║  Motor Unificado de Poema                ║",
            f"╚══════════════════════════════════════════╝",
            f"",
            f"Hardware (dispatcher):",
            f"  CPU cores:   {hw.cpu_cores}",
            f"  AVX2:        {hw.has_avx2}",
            f"  AVX-512:     {hw.has_avx512}",
            f"  CUDA:        {hw.has_cuda} ({hw.gpu_name})",
            f"  ROCm:        {hw.has_rocm}",
            f"  Rust core:   {'activo' if RUST_CORE_AVAILABLE else 'no disponible'} (umbral={self._rust_min_elements:,} elem)",
            f"  GPU umbral:  {self.config.gpu_min_elements:,} elem",
        ]
        if self._hw_caps is not None:
            lines += [
                f"",
                f"Autotune Hardware Profile ({self._hw_caps.cpu_arch}):",
                f"  FMA escalar: {self._hw_caps.measured_fma_scalar_gflops:.1f} GFLOPS",
                f"  FMA vector:  {self._hw_caps.measured_fma_vector_gflops:.1f} GFLOPS",
                f"  Mem BW:      {self._hw_caps.measured_memory_bw_gbs:.1f} GB/s",
            ]
            if self._hw_caps.gpu_available:
                lines.append(
                    f"  GPU:         {self._hw_caps.gpu_name} "
                    f"(CC {self._hw_caps.gpu_cc_str}, "
                    f"{self._hw_caps.gpu_sms} SMs)"
                )
        lines += [
            f"",
            f"Optimizaciones activas:",
            f"  fold_affine:      {self.config.fold_affine}",
            f"  fast_mode:        {self.config.fast_mode}",
            f"  gpu_min_elements: {self.config.gpu_min_elements:,}",
            f"  rust_min_elements:{self._rust_min_elements:,}",
            f"  use_autotune:     {self.config.use_autotune}",
            f"  use_ml_dispatch:  {self.config.use_ml_dispatcher}",
            f"",
            f"Caché de compilación:",
            f"  Cadenas compiladas: {ci['compiled_chains']}",
            f"  Cadenas plegadas:   {ci['folded_chains']}",
            f"  Cadenas en GPU:     {ci['gpu_chains']}",
            f"  Fold cache disco:   {self.config.persist_fold_cache}",
            f"",
            f"Telemetría ({len(self._telemetry)} ejecuciones):",
        ]
        if tel:
            for bname, s in sorted(tel.items(), key=lambda kv: kv[1]["avg_ms"]):
                lines.append(
                    f"  {bname:<28} avg={s['avg_ms']:6.1f}ms  n={s['count']}"
                )
        else:
            lines.append("  (sin datos aún)")
        lines += [f"", f"Backends disponibles:"]
        for bname, is_avail in avail.items():
            mark = "✓" if is_avail else "✗"
            lines.append(f"  [{mark}] {bname}")
        lines += [
            f"",
            f"ACF subsistemas:",
            f"  [✓] Graph ACF      (reduce_graph_signal, analyse_graph)",
            f"  [✓] Neural ACF     (analyse_network, evolve_network_function)",
            f"  [✓] Meta-Compiler  (meta_compile, meta_compile_fma)",
            f"  [✓] Tensor ACF     (reduce_tensor, analyse_tensor)",
            f"  [✓] Matrix ACF     (reduce_matrix_function, analyse_matrix)",
            f"  [✓] Matrix helpers (matrix_exp, matrix_sqrt, matrix_log, matrix_resolvent)",
            f"  [✓] ODE/Control    (reduce_vector_field, analyse_ode, certify_lyapunov, optimize_hjb)",
            f"  [✓] Operator/Green (reduce_green_function, apply_integral_operator, compress_attention)",
            f"  [✓] Stochastic PCE (pce_expand, analyse_stochastic, uncertainty_band)",
            f"  [✓] Rational/Padé  (pade_reduce, analyse_rational, hardy_reduce)",
        ]
        return "\n".join(lines)

    # ── Koopman GPU (Triton GEMM Collider) ──────────────────────────────────

    def koopman_analyze(
        self,
        snapshots: "np.ndarray",
        dt: float = 0.1,
        n_modes: int = 50,
        variance_threshold: float = 0.95,
    ) -> Any:
        """
        Análisis Koopman EDMD acelerado en GPU via Triton GEMM Collider.

        Usa tl.dot para los GEMM de la covarianza dual (PCA) y la EDMD,
        reduciendo el análisis de segundos (CPU) a milisegundos (GPU).

        Parameters
        ----------
        snapshots : (N_snap, Nx, Ny) o (N_snap, D) — campo escalar
        dt : float — intervalo temporal entre snapshots
        n_modes : int — máximo número de modos Koopman
        variance_threshold : float — umbral de varianza para PCA

        Returns
        -------
        KoopmanGPUResult con eigenvalores, decay rates, frecuencias,
        modos coherentes, y métricas de rendimiento.
        """
        from .koopman_gpu import KoopmanGPU
        gpu = KoopmanGPU()
        return gpu.analyze(snapshots, dt=dt, n_modes=n_modes,
                           variance_threshold=variance_threshold)

    # ── Simulación Acoplada NS + ACF ─────────────────────────────────────────

    def run_coupled_ns(
        self,
        N: int = 256,
        Re: float = 10000.0,
        T_total: float = 40.0,
        auto_evolve: bool = True,
        backend: str = "c_native",
        **kwargs,
    ) -> Any:
        """
        Simulación acoplada Navier-Stokes 2D + ACF en tiempo real.

        Integra el solver pseudo-espectral de NS DENTRO de Gideon,
        con análisis Koopman periódico y refinamiento adaptativo.

        Parameters
        ----------
        N : int — grid size (NxN)
        Re : float — Reynolds number
        T_total : float — tiempo total de simulación
        auto_evolve : bool — activar auto-evolución del functor
        backend : str — backend de ejecución
        **kwargs : parámetros adicionales para CoupledNSACFConfig

        Returns
        -------
        dict con snapshots, energía, enstrofía, resultados Koopman, etc.
        """
        from .ns_acf_coupled import CoupledNSACFSolver, CoupledNSACFConfig
        config = CoupledNSACFConfig(
            N=N, Re=Re, T_total=T_total,
            auto_evolve=auto_evolve, backend=backend,
            **kwargs,
        )
        solver = CoupledNSACFSolver(config)
        return solver.simulate()

    def __repr__(self) -> str:
        return f"GideonEngine(v{self._VERSION}, precision={self.config.precision!r})"
