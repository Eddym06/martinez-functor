"""
GideonDispatcher — Despachador Inteligente de Backends.

El dispatcher toma un GideonProgram (o GideonGraph) y decide qué backend
ejecuta cada nodo o subgrama de forma óptima. Implementa:

  - Heurísticas de selección basadas en tipo de nodo, precisión, y hardware
  - Prioridades configurables por usuario
  - Detección automática de hardware (CUDA, ROCm, AVX2, AVX-512)
  - Profiling de latencia para re-ranking dinámico
  - Soporte para ejecución heterogénea (C para FMA, GPU para GEMM, etc.)
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .ir import GideonProgram, IRNode, IRNodeKind


# ─────────────────────────────────────────────────────────────────────────────
# HardwareProfile — detección de hardware en tiempo de ejecución
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HardwareProfile:
    """Perfil de hardware detectado en el host."""
    has_avx2: bool = False
    has_avx512: bool = False
    has_cuda: bool = False
    has_rocm: bool = False
    has_openmp: bool = False
    has_cffi: bool = False
    has_torch: bool = False
    has_onnx: bool = False
    cpu_cores: int = 1
    gpu_name: str = ""
    platform_str: str = ""

    @classmethod
    def detect(cls) -> "HardwareProfile":
        p = cls()
        p.platform_str = platform.platform()
        p.cpu_cores = os.cpu_count() or 1

        # CPU features
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo") as f:
                    flags = f.read()
                p.has_avx2   = "avx2" in flags
                p.has_avx512 = "avx512f" in flags
            elif platform.system() == "Darwin":
                out = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.features"],
                    capture_output=True, text=True
                ).stdout.upper()
                p.has_avx2   = "AVX2" in out
                p.has_avx512 = "AVX512F" in out
        except Exception:
            pass

        # Python packages
        try:
            import cffi as _cffi  # noqa: F401
            p.has_cffi = True
        except ImportError:
            pass

        try:
            import torch
            p.has_torch = True
            p.has_cuda = torch.cuda.is_available()
            if p.has_cuda:
                p.gpu_name = torch.cuda.get_device_name(0)
            # ROCm detection
            p.has_rocm = p.has_cuda and "hip" in str(torch.version.hip or "").lower()
        except ImportError:
            pass

        try:
            import onnx  # noqa: F401
            p.has_onnx = True
        except ImportError:
            pass

        # OpenMP heuristic
        p.has_openmp = p.has_cffi  # available if C backend can compile

        return p


# ─────────────────────────────────────────────────────────────────────────────
# BackendHint
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BackendHint:
    """Sugerencia de backend para un tipo de carga de trabajo."""
    backend_name: str
    priority: int        # Mayor = más prioritario
    reason: str = ""
    applicable_kinds: List[str] = field(default_factory=list)  # IRNodeKind values


# ─────────────────────────────────────────────────────────────────────────────
# DispatchDecision
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DispatchDecision:
    """Decisión de despacho para un programa o nodo."""
    primary_backend: str
    fallback_backend: str
    reason: str
    estimated_speedup: float = 1.0
    node_backend_map: Dict[str, str] = field(default_factory=dict)  # nid → backend

    def summary(self) -> str:
        return (
            f"DispatchDecision(primary={self.primary_backend!r}, "
            f"fallback={self.fallback_backend!r}, "
            f"speedup≈{self.estimated_speedup:.1f}×, "
            f"reason={self.reason!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# GideonDispatcher
# ─────────────────────────────────────────────────────────────────────────────

class GideonDispatcher:
    """
    Despachador inteligente de Gideon.

    Decide qué backend ejecuta qué parte del programa basándose en:
      1. Perfil de hardware detectado
      2. Tipo de nodos predominantes en el programa
      3. Preferencias del usuario (hints)
      4. Historial de latencias (si disponible)
    """

    # Prioridades base por backend (mayor = más preferido)
    _BASE_PRIORITY: Dict[str, int] = {
        "c_native":  100,
        "pytorch":    80,
        "rocm":       80,
        "numpy_cpu":  40,
        "onnx":       60,
        "wasm":       20,
        "verilog":     5,
    }

    # Backends con soporte nativo de IA (GEMM, CONV, ATTENTION)
    _AI_BACKENDS: List[str] = ["pytorch", "rocm", "onnx"]

    def __init__(
        self,
        hw_profile: Optional[HardwareProfile] = None,
        user_hints: Optional[List[BackendHint]] = None,
    ) -> None:
        self.hw = hw_profile or HardwareProfile.detect()
        self.user_hints = user_hints or []
        self._latency_history: Dict[str, List[float]] = {}  # backend → [ms]

    # ── Decisión principal ────────────────────────────────────────────────────

    def decide(
        self,
        program: GideonProgram,
        precision: str = "fp64",
    ) -> DispatchDecision:
        """
        Toma un GideonProgram y devuelve una DispatchDecision completa.
        """
        kinds = program.node_kinds()
        has_ai = any(
            k in kinds
            for k in ["matmul", "gemm", "conv", "norm", "attention"]
        )
        n_nodes = program.total_nodes()
        n_fma = program.total_fma

        # ── Ranking dinámico ──────────────────────────────────────────────
        scores = dict(self._BASE_PRIORITY)

        # Penalizar backends no disponibles
        if not self.hw.has_cffi:
            scores["c_native"] = 0
        if not self.hw.has_torch:
            scores["pytorch"] = 0
            scores["rocm"] = 0
        if not self.hw.has_cuda and not self.hw.has_rocm:
            scores["pytorch"] -= 30
        if not self.hw.has_onnx:
            scores["onnx"] = 0
        if not self.hw.has_rocm:
            scores["rocm"] = 0

        # Bonificar por carga de trabajo
        if has_ai:
            if self.hw.has_cuda:
                scores["pytorch"] += 50
            if self.hw.has_rocm:
                scores["rocm"] += 50
            scores["onnx"] += 20

        if n_fma > 1000 and self.hw.has_cffi:
            scores["c_native"] += 30

        if self.hw.has_avx512 and self.hw.has_cffi:
            scores["c_native"] += 20
        elif self.hw.has_avx2 and self.hw.has_cffi:
            scores["c_native"] += 10

        # Aplicar hints de usuario
        for hint in self.user_hints:
            if hint.backend_name in scores:
                scores[hint.backend_name] += hint.priority

        # Incorporar historial de latencias
        for backend, hist in self._latency_history.items():
            if hist and backend in scores:
                avg_ms = sum(hist[-5:]) / len(hist[-5:])
                # Penalizar si promedio > 10ms
                if avg_ms > 10.0:
                    scores[backend] = max(0, scores[backend] - 10)

        # Seleccionar primary y fallback
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        primary = ranked[0][0] if ranked[0][1] > 0 else "numpy_cpu"
        fallback = "numpy_cpu"
        for name, score in ranked[1:]:
            if score > 0 and name != primary:
                fallback = name
                break

        speedup = self._estimate_speedup(primary, n_fma, has_ai)
        reason = self._build_reason(primary, has_ai, n_fma)

        # Mapa nodo→backend para heterogénea
        node_map = self._build_node_map(program, primary, has_ai)

        return DispatchDecision(
            primary_backend=primary,
            fallback_backend=fallback,
            reason=reason,
            estimated_speedup=speedup,
            node_backend_map=node_map,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _estimate_speedup(
        self, backend: str, n_fma: int, has_ai: bool
    ) -> float:
        """Estimación heurística de speedup sobre numpy baseline."""
        table = {
            "c_native": 25.0 if self.hw.has_avx512 else (15.0 if self.hw.has_avx2 else 8.0),
            "pytorch":  5.0 if self.hw.has_cuda else 3.0,
            "rocm":     20.0,
            "onnx":     4.0,
            "numpy_cpu": 1.0,
            "wasm":     0.5,
            "verilog":  float("inf"),  # FPGA: fuera de escala
        }
        base = table.get(backend, 1.0)
        if has_ai and backend in self._AI_BACKENDS:
            base *= 2.0
        return base

    def _build_reason(self, backend: str, has_ai: bool, n_fma: int) -> str:
        reasons: List[str] = [f"backend_selected={backend}"]
        if has_ai:
            reasons.append("ai_workload_detected")
        if n_fma > 1000:
            reasons.append(f"heavy_fma_chain({n_fma})")
        if self.hw.has_avx512:
            reasons.append("avx512_available")
        elif self.hw.has_avx2:
            reasons.append("avx2_available")
        if self.hw.has_cuda:
            reasons.append("cuda_gpu_detected")
        return "; ".join(reasons)

    def _build_node_map(
        self,
        program: GideonProgram,
        primary: str,
        has_ai: bool,
    ) -> Dict[str, str]:
        """Asigna backends por nodo para ejecución heterogénea."""
        ai_kinds = {"matmul", "gemm", "conv", "norm", "attention"}
        gpu_backend = "pytorch" if self.hw.has_cuda else primary

        node_map: Dict[str, str] = {}
        for nid, node in program.nodes.items():
            if node.kind.value in ai_kinds and has_ai:
                node_map[nid] = gpu_backend
            else:
                node_map[nid] = primary
        return node_map

    # ── Registro de latencias para feedback loop ──────────────────────────────

    def record_latency(self, backend: str, ms: float) -> None:
        """Registra una medición de latencia para re-ranking futuro."""
        if backend not in self._latency_history:
            self._latency_history[backend] = []
        hist = self._latency_history[backend]
        hist.append(ms)
        # Mantener últimas 20
        if len(hist) > 20:
            self._latency_history[backend] = hist[-20:]

    def hardware_summary(self) -> str:
        hw = self.hw
        return (
            f"Hardware: {hw.platform_str}\n"
            f"  CPU cores: {hw.cpu_cores}\n"
            f"  AVX2: {hw.has_avx2}  AVX-512: {hw.has_avx512}\n"
            f"  CUDA: {hw.has_cuda}  GPU: {hw.gpu_name!r}\n"
            f"  ROCm: {hw.has_rocm}\n"
            f"  cffi: {hw.has_cffi}  torch: {hw.has_torch}  onnx: {hw.has_onnx}\n"
        )
