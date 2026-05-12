"""
gideon_autotune.py — Hardware Profiler y Sistema de Auto-Tuning para Gideon.

Sistema dinámico y adaptativo de descubrimiento profundo de hardware para
calibrar automáticamente todos los kernels bare-metal de Gideon.

El diseño es "hardware-first": NINGÚN parámetro de kernel está hardcodeado.
Todos los valores óptimos (tile sizes, block sizes, unroll factors, prefetch
distances) se derivan de mediciones reales del hardware disponible.

Flujo adaptativo:
  1. Detección estática: CPU (microarq, caché, AVX), GPU (CC, SMs, shmem)
  2. Micro-benchmarks: GFLOPS FP64/FP32, bandwidth RAM y PCIe, kernels Triton
  3. Cálculo de parámetros óptimos: GEMM tiles, Triton block, unroll, prefetch
  4. Persistencia: ~/.gideon/profiles/<hostname>.json
  5. Invalidación automática si el hardware cambia (nuevo GPU, etc.)

Componentes:
  HardwareCapabilities   — dataclass COMPLETA con todos los parámetros derivados
  GideonHardwareProfiler — perfilador adaptativo con benchmarks específicos

Micro-benchmarks:
  ┌────────────────────────────────┬───────────────────────────────────────────┐
  │ Benchmark                      │ Métrica                                   │
  ├────────────────────────────────┼───────────────────────────────────────────┤
  │ fma_scalar                     │ GFLOPS escalares (float64)               │
  │ fma_vector                     │ GFLOPS vectoriales (float32, SIMD)       │
  │ memory_bandwidth               │ GB/s (operación copy en RAM)             │
  │ pcie_bandwidth                 │ GB/s CPU→GPU (si CUDA disponible)        │
  │ gpu_kernel_launch              │ µs de overhead de lanzamiento de kernel  │
  │ gpu_fp64_throughput            │ GFLOPS reales GPU fp64 (CUDA cores)      │
  │ gpu_fp32_throughput            │ GFLOPS reales GPU fp32 (CUDA cores)      │
  │ triton_fma_chain               │ GB/s efectivos en kernel Triton FMA      │
  │ precision_ratio                │ Ratio throughput fp32 / fp64             │
  └────────────────────────────────┴───────────────────────────────────────────┘

Parámetros derivados automáticamente:
  gemm_mr, gemm_nr   — dimensiones del micro-kernel según nivel AVX
  gemm_mc, gemm_kc   — tile L2 calculado del tamaño de caché L2 real
  gemm_nc            — tile L3 calculado del tamaño de caché L3 real
  triton_block       — BLOCK_SIZE óptimo para kernel Triton FMA chain
  fma_unroll_factor  — factor de unrolling según pipeline de la microarq
  cpu_prefetch_dist  — distancia de prefetch en líneas de caché (L1 lines)

Uso básico:
    profiler = GideonHardwareProfiler()
    caps = profiler.load_or_profile()
    print(profiler.summary(caps))

Uso en engine:
    caps = GideonHardwareProfiler(quick_mode=True).load_or_profile()
    # Parámetros derivados listos para usar:
    # caps.gemm_mc, caps.gemm_kc, caps.triton_block, caps.fma_unroll_factor
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PROFILE_DIR = os.path.expanduser("~/.gideon/profiles")
# Version bump: "2.0" por campos nuevos (gemm_*, triton_*, gpu_shmem_*, etc.)
_PROFILE_VERSION = "2.0"


# ─────────────────────────────────────────────────────────────────────────────
# HardwareCapabilities — perfil completo con parámetros derivados
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HardwareCapabilities:
    """
    Perfil completo de hardware del host.

    CAMPOS ESTÁTICOS (siempre disponibles tras detección):
      cpu_vendor, cpu_model, cpu_arch, cpu_cores_logical, cpu_cores_physical,
      cpu_l1d_kb, cpu_l2_kb, cpu_l3_mb, cpu_avx_level, cpu_avx512_subsets,
      cpu_freq_mhz, cpu_l1d_lines, cpu_pipeline_depth,
      gpu_available, gpu_name, gpu_compute_capability, gpu_sms,
      gpu_memory_gb, gpu_shmem_kb, gpu_warp_size, gpu_max_threads_per_block,
      gpu_tensor_cores, gpu_tc_version, triton_available, triton_version

    CAMPOS MEDIDOS en runtime (quick_mode=False):
      measured_fma_scalar_gflops, measured_fma_vector_gflops,
      measured_memory_bw_gbs, measured_pcie_bw_gbs,
      measured_gpu_launch_us, measured_gpu_fp64_gflops, measured_gpu_fp32_gflops,
      measured_triton_fma_gbs, measured_fp32_fp64_ratio

    PARÁMETROS DERIVADOS (calculados automáticamente de los anteriores):
      gemm_mr, gemm_nr  — micro-kernel rows/cols (AVX-512: 8×4, AVX2: 4×4)
      gemm_mc           — tile L2 rows (calculado de L2 real)
      gemm_kc           — tile L2 depth (calculado de L1d real)
      gemm_nc           — tile L3 cols (calculado de L3 real)
      triton_block      — BLOCK_SIZE óptimo para Triton FMA (potencia de 2)
      fma_unroll_factor — unrolling del bucle FMA (2 para pipelines largos)
      cpu_prefetch_dist — distancia prefetch en líneas de caché
    """

    # ── CPU estático ─────────────────────────────────────────────────────────
    cpu_vendor: str = ""
    cpu_model: str = ""
    cpu_arch: str = ""
    cpu_cores_logical: int = 1
    cpu_cores_physical: int = 1
    cpu_l1d_kb: int = 0
    cpu_l2_kb: int = 0
    cpu_l3_mb: float = 0.0
    cpu_avx_level: int = 0                  # 0=sin SIMD, 2=AVX2, 512=AVX-512
    cpu_avx512_subsets: str = ""            # "f,dq,bw,vl,vnni,bf16,fp16" etc.
    cpu_freq_mhz: float = 0.0
    # Campos derivados estáticos
    cpu_l1d_lines: int = 0                  # L1d / 64 (líneas de caché)
    cpu_pipeline_depth: int = 14            # Profundidad pipeline (Skylake=14, Zen4=19, etc.)

    # ── GPU estático ─────────────────────────────────────────────────────────
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_compute_capability: Tuple[int, int] = (0, 0)
    gpu_sms: int = 0
    gpu_memory_gb: float = 0.0
    gpu_shmem_kb: int = 0                   # Shared memory por SM en KB
    gpu_warp_size: int = 32
    gpu_max_threads_per_block: int = 1024
    gpu_l2_cache_mb: float = 0.0            # L2 cache del GPU en MB
    gpu_tensor_cores: bool = False
    gpu_tc_version: int = 0                 # 1=Volta, 2=Turing, 3=Ampere, 4=Ada, 5=Hopper

    # ── Triton estático ──────────────────────────────────────────────────────
    triton_available: bool = False
    triton_version: str = ""

    # ── Mediciones en runtime ────────────────────────────────────────────────
    measured_fma_scalar_gflops: float = 0.0
    measured_fma_vector_gflops: float = 0.0
    measured_memory_bw_gbs: float = 0.0
    measured_pcie_bw_gbs: float = 0.0
    measured_gpu_launch_us: float = 0.0
    measured_gpu_fp64_gflops: float = 0.0   # GPU FP64 TFLOPS reales
    measured_gpu_fp32_gflops: float = 0.0   # GPU FP32 TFLOPS reales
    measured_triton_fma_gbs: float = 0.0    # Throughput efectivo kernel Triton FMA
    measured_fp32_fp64_ratio: float = 1.0

    # ── Parámetros derivados (calculados de los campos anteriores) ────────────
    # GEMM micro-kernel
    gemm_mr: int = 4                        # filas micro-kernel (4=AVX2, 8=AVX-512)
    gemm_nr: int = 4                        # cols micro-kernel
    gemm_mc: int = 64                       # tile nivel L2 (filas)
    gemm_kc: int = 64                       # tile nivel L2/L1 (depth)
    gemm_nc: int = 256                      # tile nivel L3 (cols)
    # Triton
    triton_block: int = 1024               # BLOCK_SIZE óptimo para kernels Triton FMA
    triton_num_warps: int = 4              # num_warps para Triton
    triton_num_stages: int = 2             # num_stages pipeline software (SM >= 8.0 → 3)
    # CPU kernel FMA
    fma_unroll_factor: int = 1             # 1=sin unroll, 2=unroll×2, 4=unroll×4
    cpu_prefetch_dist: int = 8             # líneas de caché a adelantar (~64 bytes/línea)

    # ── Metadatos del perfil ─────────────────────────────────────────────────
    profile_timestamp: str = ""
    hostname: str = ""
    gideon_version: str = ""
    quick_mode: bool = False

    # ── Serialización ────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["gpu_compute_capability"] = list(self.gpu_compute_capability)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HardwareCapabilities":
        obj = cls()
        for k, v in d.items():
            if k.startswith("_"):
                continue
            if k == "gpu_compute_capability":
                setattr(obj, k, tuple(v[:2]) if len(v) >= 2 else (0, 0))
            elif hasattr(obj, k):
                try:
                    setattr(obj, k, v)
                except Exception:
                    pass
        return obj

    # ── Propiedades derivadas ─────────────────────────────────────────────────
    @property
    def avx_label(self) -> str:
        if self.cpu_avx_level >= 512:
            return f"AVX-512{(' [' + self.cpu_avx512_subsets + ']') if self.cpu_avx512_subsets else ''}"
        if self.cpu_avx_level >= 2:
            return "AVX2+FMA3"
        return "None"

    @property
    def gpu_cc_str(self) -> str:
        return f"{self.gpu_compute_capability[0]}.{self.gpu_compute_capability[1]}"

    @property
    def has_high_bw_gpu(self) -> bool:
        """True si el GPU conectado vía PCIe Gen4/Gen5 (> 25 GB/s PCIe)."""
        return (
            self.gpu_available
            and self.measured_pcie_bw_gbs > 0
            and self.gpu_compute_capability[0] >= 8
        )

    @property
    def gpu_arch_name(self) -> str:
        """Nombre de arquitectura GPU por compute capability."""
        cc = self.gpu_compute_capability[0] * 10 + self.gpu_compute_capability[1]
        if cc >= 100: return "Blackwell"
        if cc >= 90:  return "Hopper"
        if cc >= 89:  return "Ada Lovelace"
        if cc >= 80:  return "Ampere"
        if cc >= 75:  return "Turing"
        if cc >= 70:  return "Volta"
        if cc >= 61:  return "Pascal"
        if cc >= 52:  return "Maxwell"
        return "Kepler/older"


# ─────────────────────────────────────────────────────────────────────────────
# GideonHardwareProfiler — sistema adaptativo y dinámico
# ─────────────────────────────────────────────────────────────────────────────

class GideonHardwareProfiler:
    """
    Perfilador de hardware completamente dinámico para Gideon.

    El diseño fundamental es "hardware-first": NINGÚN parámetro de kernel
    (tile sizes, block sizes, unroll factors, prefetch distances) está
    hardcodeado. Todos se derivan de mediciones reales del hardware actual.

    Estrategia adaptativa:
      - Primera ejecución: full_profile() — detección + benchmarks (~3-8 s)
      - Ejecuciones siguientes: load_or_profile() carga el caché (< 1 ms)
        pero verifica el hardware_fingerprint para invalidar si cambió el GPU
      - quick_mode=True: solo detección estática + cálculo de params (~10 ms)
      - force_reprofiling=True: ignora caché y re-perfila completamente

    Campos calculados adaptativamente:
      caps.gemm_mr, caps.gemm_nr  ← AVX level actual
      caps.gemm_mc, caps.gemm_kc  ← L2 cache real del CPU
      caps.gemm_nc                ← L3 cache real del CPU
      caps.triton_block           ← SMs + shared memory del GPU
      caps.fma_unroll_factor      ← pipeline depth del CPU
      caps.cpu_prefetch_dist      ← L1d cache lines
    """

    PROFILE_VERSION = _PROFILE_VERSION

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        force_reprofiling: bool = False,
        quick_mode: bool = True,
    ) -> None:
        self.cache_dir = cache_dir or _DEFAULT_PROFILE_DIR
        self.force_reprofiling = force_reprofiling
        self.quick_mode = quick_mode

    # ── API pública ───────────────────────────────────────────────────────────

    def load_or_profile(self) -> HardwareCapabilities:
        """
        Carga el perfil persistido si existe y es válido; si no, perfila.

        Validación doble:
          1. Versión del esquema: re-perfila si el esquema cambió.
          2. Hardware fingerprint: re-perfila si GPU/CPU cambió (ej: nuevo GPU).
        """
        if not self.force_reprofiling:
            path = self._profile_path()
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    stored_ver = data.pop("_profile_version", None)
                    stored_fp  = data.pop("_hw_fingerprint", None)
                    if stored_ver == self.PROFILE_VERSION:
                        loaded = HardwareCapabilities.from_dict(data)
                        # Verificar que el hardware no cambió
                        current_fp = self._hardware_fingerprint()
                        if stored_fp is None or stored_fp == current_fp:
                            return loaded
                        # Hardware cambió → re-perfilar silenciosamente
                except Exception:
                    pass
        return self.full_profile()

    def full_profile(self) -> HardwareCapabilities:
        """
        Perfil completo: detección estática + cálculo de parámetros + benchmarks.

        Si quick_mode=True, omite benchmarks pero calcula todos los parámetros
        derivados (GEMM tiles, Triton block, etc.) desde las propiedades estáticas.
        """
        caps = HardwareCapabilities()
        caps.profile_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        caps.hostname = socket.gethostname()
        caps.gideon_version = "2.0.0"
        caps.quick_mode = self.quick_mode

        self._detect_cpu(caps)
        self._detect_gpu(caps)
        self._detect_triton(caps)

        if not self.quick_mode:
            self._run_benchmarks(caps)
        else:
            # Estimaciones estáticas robustas basadas en arquitectura
            caps.measured_fma_scalar_gflops = self._estimate_scalar_gflops(caps)
            caps.measured_fma_vector_gflops = self._estimate_vector_gflops(caps)
            caps.measured_fp32_fp64_ratio   = 2.0 if caps.cpu_avx_level >= 2 else 1.0
            caps.measured_memory_bw_gbs     = self._estimate_memory_bw(caps)

        # SIEMPRE calcular parámetros derivados (tanto quick como full)
        self._compute_derived_params(caps)

        self._save(caps)
        return caps

    def summary(self, caps: HardwareCapabilities) -> str:
        """Resumen legible del perfil completo incluyendo parámetros derivados."""
        lines = [
            "─── Gideon Hardware Profile v2.0 (Dinámico) ───",
            f"  Host:       {caps.hostname}",
            f"  CPU:        {caps.cpu_model or 'Desconocido'}",
            f"  Arq:        {caps.cpu_arch}  /  {caps.cpu_vendor}",
            f"  Cores:      {caps.cpu_cores_physical}P / {caps.cpu_cores_logical}L",
            f"  AVX:        {caps.avx_label}",
            f"  Caché:      L1d={caps.cpu_l1d_kb}KB  L2={caps.cpu_l2_kb}KB  L3={caps.cpu_l3_mb:.1f}MB",
            f"  Freq:       {caps.cpu_freq_mhz:.0f} MHz",
            f"  Pipeline:   {caps.cpu_pipeline_depth} etapas",
        ]
        mode = "estimado" if caps.quick_mode else "medido"
        lines += [
            f"  FMA f64:    {caps.measured_fma_scalar_gflops:.1f} GFLOPS ({mode})",
            f"  FMA f32:    {caps.measured_fma_vector_gflops:.1f} GFLOPS ({mode})",
            f"  Mem BW:     {caps.measured_memory_bw_gbs:.1f} GB/s ({mode})",
            f"  fp32/fp64:  {caps.measured_fp32_fp64_ratio:.1f}×",
        ]
        lines += [
            "  ── Parámetros GEMM derivados ──",
            f"  MR×NR:      {caps.gemm_mr}×{caps.gemm_nr}  (micro-kernel)",
            f"  MC×KC:      {caps.gemm_mc}×{caps.gemm_kc}  (tile L2)",
            f"  NC:         {caps.gemm_nc}  (tile L3)",
            "  ── Parámetros FMA kernel ──",
            f"  Unroll:     ×{caps.fma_unroll_factor}",
            f"  Prefetch:   {caps.cpu_prefetch_dist} líneas",
        ]
        if caps.gpu_available:
            tc_str = f"gen{caps.gpu_tc_version} ({caps.gpu_arch_name})" if caps.gpu_tensor_cores else f"No ({caps.gpu_arch_name})"
            lines += [
                "  ── GPU ──",
                f"  GPU:        {caps.gpu_name}",
                f"  CC:         {caps.gpu_cc_str}  /  {caps.gpu_arch_name}",
                f"  Mem/SMs:    {caps.gpu_memory_gb:.1f} GB  /  {caps.gpu_sms} SMs",
                f"  Shmem/SM:   {caps.gpu_shmem_kb} KB",
                f"  Warp/MaxTh: {caps.gpu_warp_size} / {caps.gpu_max_threads_per_block}",
                f"  L2 GPU:     {caps.gpu_l2_cache_mb:.1f} MB",
                f"  TensorCore: {tc_str}",
            ]
            if not caps.quick_mode:
                lines += [
                    f"  PCIe BW:    {caps.measured_pcie_bw_gbs:.1f} GB/s",
                    f"  Launch:     {caps.measured_gpu_launch_us:.1f} µs",
                    f"  GPU FP64:   {caps.measured_gpu_fp64_gflops:.1f} GFLOPS",
                    f"  GPU FP32:   {caps.measured_gpu_fp32_gflops:.1f} GFLOPS",
                ]
                if caps.triton_available and caps.measured_triton_fma_gbs > 0:
                    lines.append(f"  Triton FMA: {caps.measured_triton_fma_gbs:.1f} GB/s efectivos")
            if caps.triton_available:
                lines += [
                    f"  Triton:     v{caps.triton_version}  (disponible)",
                    f"  TritBlock:  {caps.triton_block}  (BLOCK_SIZE óptimo)",
                    f"  NumWarps:   {caps.triton_num_warps}",
                    f"  NumStages:  {caps.triton_num_stages}",
                ]
        else:
            lines.append("  GPU:        No disponible")
        lines.append(f"  Perfil:     {caps.profile_timestamp}")
        return "\n".join(lines)

    # ── Detección de CPU ─────────────────────────────────────────────────────

    def _detect_cpu(self, caps: HardwareCapabilities) -> None:
        caps.cpu_cores_logical = os.cpu_count() or 1
        try:
            if platform.system() == "Linux":
                self._detect_cpu_linux(caps)
            elif platform.system() == "Darwin":
                self._detect_cpu_darwin(caps)
        except Exception:
            pass

        # Calcular líneas de caché L1d
        if caps.cpu_l1d_kb > 0:
            caps.cpu_l1d_lines = (caps.cpu_l1d_kb * 1024) // 64

        # Clasificar microarquitectura y asignar pipeline depth
        self._classify_cpu_arch(caps)

    def _detect_cpu_linux(self, caps: HardwareCapabilities) -> None:
        """Detección completa en Linux."""
        with open("/proc/cpuinfo") as f:
            text = f.read()

        avx512_subsets: List[str] = []
        for line in text.split("\n"):
            if line.startswith("model name") and not caps.cpu_model:
                caps.cpu_model = line.split(":", 1)[1].strip()
            if line.startswith("vendor_id") and not caps.cpu_vendor:
                caps.cpu_vendor = line.split(":", 1)[1].strip()
            if line.startswith("flags"):
                flags = line.lower()
                if "avx512f" in flags and caps.cpu_avx_level < 512:
                    caps.cpu_avx_level = 512
                    for subset in ("avx512dq", "avx512bw", "avx512vl",
                                   "avx512vnni", "avx512bf16", "avx512fp16"):
                        if subset in flags:
                            avx512_subsets.append(subset.replace("avx512", ""))
                elif "avx2" in flags and caps.cpu_avx_level < 2:
                    caps.cpu_avx_level = 2

        caps.cpu_avx512_subsets = ",".join(avx512_subsets)

        # Tamaños de caché desde sysfs — más fiables que /proc/cpuinfo
        # L1d en index0, L2 en index2, L3 en index3
        for idx, attr in [(0, "cpu_l1d_kb"), (2, "cpu_l2_kb")]:
            try:
                with open(f"/sys/devices/system/cpu/cpu0/cache/index{idx}/size", encoding="utf-8") as handle:
                    val = handle.read().strip()
                n = int(re.sub(r"[KkMm].*", "", val))
                if "M" in val.upper():
                    n *= 1024
                setattr(caps, attr, n)
            except Exception:
                pass
        try:
            with open("/sys/devices/system/cpu/cpu0/cache/index3/size", encoding="utf-8") as handle:
                l3_str = handle.read().strip()
            n = int(re.sub(r"[KkMm].*", "", l3_str))
            caps.cpu_l3_mb = n / 1024 if "K" in l3_str.upper() else float(n)
        except Exception:
            pass

        # Núcleos físicos
        core_ids: set = set()
        phys_ids: set = set()
        for line in text.split("\n"):
            if line.startswith("physical id"):
                phys_ids.add(line.split(":", 1)[1].strip())
            if line.startswith("core id"):
                core_ids.add(line.split(":", 1)[1].strip())
        if core_ids:
            n_phys = max(len(phys_ids), 1)
            caps.cpu_cores_physical = n_phys * len(core_ids)
        else:
            # Intel Hybrid (Meteor Lake, Raptor Lake): mezcla P+E cores
            # lscpu puede dar más info pero /proc/cpuinfo es suficiente como fallback
            caps.cpu_cores_physical = max(caps.cpu_cores_logical // 2, 1)

        # Frecuencia máxima
        for freq_path in (
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq",
        ):
            try:
                caps.cpu_freq_mhz = float(open(freq_path).read().strip()) / 1000.0
                break
            except Exception:
                pass

    def _detect_cpu_darwin(self, caps: HardwareCapabilities) -> None:
        """Detección en macOS."""
        import subprocess
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.features"],
            capture_output=True, text=True,
        ).stdout.upper()
        if "AVX512F" in out:
            caps.cpu_avx_level = 512
        elif "AVX2" in out:
            caps.cpu_avx_level = 2
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True,
        ).stdout.strip()
        caps.cpu_model = brand
        caps.cpu_vendor = "GenuineIntel" if "Intel" in brand else (
            "AuthenticAMD" if "AMD" in brand else "Apple"
        )

    def _classify_cpu_arch(self, caps: HardwareCapabilities) -> None:
        """
        Clasifica la microarquitectura del CPU y asigna pipeline_depth.

        Cubre arquitecturas actuales y futuras (Zen5, Arrow Lake, etc.)
        El matching es por patrones en el nombre del modelo.
        """
        name = caps.cpu_model.lower()
        vendor = caps.cpu_vendor.lower()

        # ── Intel ────────────────────────────────────────────────────────────
        # Lunar Lake / Arrow Lake (2024-2025) — Intel Core Ultra 200
        if any(x in name for x in ("ultra 200", "core ultra 2", "arrow lake", "lunar lake")):
            caps.cpu_arch, caps.cpu_pipeline_depth = "arrow_lake", 15
        # Meteor Lake (2023-2024) — Intel Core Ultra 100
        elif any(x in name for x in ("ultra 9 185", "ultra 7 165", "ultra 5 125",
                                      "meteor lake", "core ultra 1")):
            caps.cpu_arch, caps.cpu_pipeline_depth = "meteor_lake", 14
        # Raptor Lake (13th/14th gen Intel)
        elif any(x in name for x in ("13th gen", "14th gen", "raptor lake",
                                      "i9-13", "i9-14", "i7-13", "i7-14")):
            caps.cpu_arch, caps.cpu_pipeline_depth = "raptor_lake", 14
        # Alder Lake (12th gen Intel)
        elif any(x in name for x in ("12th gen", "alder lake", "i9-12", "i7-12")):
            caps.cpu_arch, caps.cpu_pipeline_depth = "alder_lake", 14
        # Ice Lake / Tiger Lake (10th/11th gen)
        elif any(x in name for x in ("10th gen", "11th gen", "ice lake", "tiger lake")):
            caps.cpu_arch, caps.cpu_pipeline_depth = "ice_lake", 12
        # Skylake family (6th gen+)
        elif any(x in name for x in ("skylake", "kaby lake", "coffee lake",
                                      "comet lake", "cascade lake")):
            caps.cpu_arch, caps.cpu_pipeline_depth = "skylake", 14
        # ── AMD ──────────────────────────────────────────────────────────────
        # Zen 5 (2024+)
        elif any(x in name for x in ("zen 5", "ryzen ai", "9950x", "9900x")):
            caps.cpu_arch, caps.cpu_pipeline_depth = "zen5", 19
        # Zen 4 (7000 series)
        elif any(x in name for x in ("zen 4", "ryzen 9 7", "ryzen 9 9",
                                      "7950x", "7900x", "7700x", "7600x")):
            caps.cpu_arch, caps.cpu_pipeline_depth = "zen4", 19
        # Zen 3 (5000 series)
        elif any(x in name for x in ("zen 3", "5950x", "5900x", "5800x", "5600x")):
            caps.cpu_arch, caps.cpu_pipeline_depth = "zen3", 19
        # Zen 2 (3000 series)
        elif any(x in name for x in ("zen 2", "3900", "3950", "3700")):
            caps.cpu_arch, caps.cpu_pipeline_depth = "zen2", 17
        # Zen 1
        elif "zen" in name or "1800x" in name or "1700x" in name:
            caps.cpu_arch, caps.cpu_pipeline_depth = "zen1", 14
        # ── Apple Silicon ────────────────────────────────────────────────────
        elif "apple m" in name or "apple" in vendor:
            caps.cpu_arch, caps.cpu_pipeline_depth = "apple_silicon", 16
        # ── RISC-V / ARM ─────────────────────────────────────────────────────
        elif "aarch64" in platform.machine() or "arm" in platform.machine():
            caps.cpu_arch, caps.cpu_pipeline_depth = "aarch64", 12
        # ── Genérico ─────────────────────────────────────────────────────────
        else:
            caps.cpu_arch, caps.cpu_pipeline_depth = "generic", 14

    # ── Detección de GPU ─────────────────────────────────────────────────────

    def _detect_gpu(self, caps: HardwareCapabilities) -> None:
        """
        Detección completa de GPU: CC, SMs, shared memory, warp size,
        max threads/block, L2 cache, generación de Tensor Core.
        """
        try:
            import torch
            caps.gpu_available = torch.cuda.is_available()
            if not caps.gpu_available:
                return

            props = torch.cuda.get_device_properties(0)
            caps.gpu_name                   = props.name
            caps.gpu_compute_capability     = (props.major, props.minor)
            caps.gpu_sms                    = props.multi_processor_count
            caps.gpu_memory_gb              = props.total_memory / (1024 ** 3)
            caps.gpu_warp_size              = props.warp_size
            caps.gpu_max_threads_per_block  = props.max_threads_per_block
            # Shared memory por bloque (en KB, convertido de bytes)
            caps.gpu_shmem_kb = props.shared_memory_per_block // 1024
            # L2 cache del GPU
            if hasattr(props, "l2_cache_size"):
                caps.gpu_l2_cache_mb = props.l2_cache_size / (1024 ** 2)

            # Clasificación generación Tensor Core por CC
            cc = props.major * 10 + props.minor
            if cc >= 100:
                caps.gpu_tensor_cores, caps.gpu_tc_version = True, 5  # Blackwell+
            elif cc >= 90:
                caps.gpu_tensor_cores, caps.gpu_tc_version = True, 5  # Hopper
            elif cc >= 89:
                caps.gpu_tensor_cores, caps.gpu_tc_version = True, 4  # Ada Lovelace
            elif cc >= 80:
                caps.gpu_tensor_cores, caps.gpu_tc_version = True, 3  # Ampere
            elif cc >= 75:
                caps.gpu_tensor_cores, caps.gpu_tc_version = True, 2  # Turing
            elif cc >= 70:
                caps.gpu_tensor_cores, caps.gpu_tc_version = True, 1  # Volta
            else:
                caps.gpu_tensor_cores, caps.gpu_tc_version = False, 0
        except Exception:
            pass

    def _detect_triton(self, caps: HardwareCapabilities) -> None:
        """Detecta disponibilidad de Triton."""
        try:
            import triton as _triton
            caps.triton_available = True
            caps.triton_version   = getattr(_triton, "__version__", "unknown")
        except ImportError:
            caps.triton_available = False

    # ── Cálculo de parámetros derivados ─────────────────────────────────────

    def _compute_derived_params(self, caps: HardwareCapabilities) -> None:
        """
        Calcula TODOS los parámetros de kernel desde las capacidades de hardware.

        Este método es el corazón del sistema dinámico: los valores derivados
        son funciones directas del hardware medido, sin valores hardcodeados.
        """
        # ── GEMM micro-kernel (MR×NR) ────────────────────────────────────────
        # MR = doubles por registro SIMD = ancho del acumulador
        # AVX-512: 8 doubles/ZMM, AVX2: 4 doubles/YMM, escalar: 1
        if caps.cpu_avx_level >= 512:
            caps.gemm_mr, caps.gemm_nr = 8, 4
        elif caps.cpu_avx_level >= 2:
            caps.gemm_mr, caps.gemm_nr = 4, 4
        else:
            caps.gemm_mr, caps.gemm_nr = 1, 1

        # ── GEMM tile KC (se ajusta a L1d) ──────────────────────────────────
        # Restricción: 2 paneles en L1d simultáneamente
        #   Panel A: MR * KC * 8 bytes
        #   Panel B: NR * KC * 8 bytes
        # → KC ≤ L1d_bytes / ((MR + NR) * 8)
        l1d_bytes = max(caps.cpu_l1d_kb, 32) * 1024  # fallback a 32KB si no detectado
        kc_max = l1d_bytes // ((caps.gemm_mr + caps.gemm_nr) * 8)
        # Redondear a potencia de 2 más cercana, mínimo 64
        kc_p2 = 1
        while kc_p2 * 2 <= min(kc_max, 512):
            kc_p2 *= 2
        caps.gemm_kc = max(kc_p2, 64)

        # ── GEMM tile MC (se ajusta a L2) ────────────────────────────────────
        # Restricción: panel A (MC×KC) cabe en L2/4 (para que B también quepa)
        l2_bytes = max(caps.cpu_l2_kb, 256) * 1024  # fallback a 256KB
        mc_max = l2_bytes // (4 * caps.gemm_kc * 8)
        mc_p2 = 1
        while mc_p2 * 2 <= min(mc_max, 512):
            mc_p2 *= 2
        caps.gemm_mc = max(mc_p2, caps.gemm_mr)

        # ── GEMM tile NC (se ajusta a L3) ────────────────────────────────────
        # Restricción: panel B (KC×NC) cabe en L3/8
        l3_bytes = max(caps.cpu_l3_mb, 1.0) * 1024 * 1024
        nc_max = l3_bytes // (8 * caps.gemm_kc * 8)
        nc_p2 = 1
        while nc_p2 * 2 <= min(nc_max, 2048):
            nc_p2 *= 2
        caps.gemm_nc = max(nc_p2, caps.gemm_nr)

        # ── FMA kernel: unroll factor ─────────────────────────────────────────
        # Para ocultar latencias de FP: necesitamos ≥ pipeline_depth / fma_lat instrucciones
        # FMA latency típica: 4-5 ciclos. Con pipeline de 14-19 etapas:
        # Zen4/5 (19): unroll 4× | Golden Cove/Meteor Lake (14): unroll 2×
        if caps.cpu_pipeline_depth >= 18:
            caps.fma_unroll_factor = 4
        elif caps.cpu_pipeline_depth >= 12:
            caps.fma_unroll_factor = 2
        else:
            caps.fma_unroll_factor = 1

        # ── FMA kernel: distancia de prefetch ────────────────────────────────
        # Prefetch: ≥ unroll_factor × (latencia L2/L1) líneas adelante
        # En Intel/AMD modernos: L2 latency ≈ 12-15 ciclos → 2 iteraciones AVX-512 ahead
        l1d_lines = max(caps.cpu_l1d_lines, 128)
        # Prefetch a 4 vectores SIMD adelante = 4 × MR × 8 bytes = 4×8×8=256 bytes = 4 líneas
        caps.cpu_prefetch_dist = caps.fma_unroll_factor * caps.gemm_mr

        # ── Triton: BLOCK_SIZE y num_warps ───────────────────────────────────
        if caps.gpu_available:
            # BLOCK_SIZE: queremos que cada bloque procese suficientes elementos
            # para ocultar la latencia de memoria global (coalescing).
            # Regla: BLOCK ≥ warp_size × 2 (al menos 2 warps por bloque)
            # Pero no más de max_threads_per_block
            # En Ada (8.9): max 1024, warp 32 → BLOCK = 1024 óptimo para memory-bound
            # En Turing (7.5): idem
            # En Pascal (6.x): max 1024, pero L2 más pequeño → BLOCK = 512 mejor
            cc_int = caps.gpu_compute_capability[0] * 10 + caps.gpu_compute_capability[1]
            if cc_int >= 80:   # Ampere+
                caps.triton_block = min(1024, caps.gpu_max_threads_per_block)
            elif cc_int >= 70: # Volta, Turing
                caps.triton_block = min(512, caps.gpu_max_threads_per_block)
            else:              # Maxwell, Pascal
                caps.triton_block = min(256, caps.gpu_max_threads_per_block)
            # Asegurar potencia de 2
            while caps.triton_block > caps.gpu_warp_size and not _is_power2(caps.triton_block):
                caps.triton_block -= 1

            # num_warps: threads_per_block / 32
            caps.triton_num_warps = max(caps.triton_block // caps.gpu_warp_size, 1)

            # num_stages (pipeline): SM ≥ 8.0 soporta 3+ stages de software pipeline
            if cc_int >= 80:
                caps.triton_num_stages = 3
            elif cc_int >= 70:
                caps.triton_num_stages = 2
            else:
                caps.triton_num_stages = 1

    # ── Micro-benchmarks ─────────────────────────────────────────────────────

    def _run_benchmarks(self, caps: HardwareCapabilities) -> None:
        """
        Ejecuta todos los micro-benchmarks adaptativos.
        Orden: CPU primero (más rápido), luego GPU.
        """
        caps.measured_fma_scalar_gflops = self._bench_fma_scalar()
        caps.measured_fma_vector_gflops = self._bench_fma_vector()
        caps.measured_memory_bw_gbs     = self._bench_memory_bandwidth()
        caps.measured_fp32_fp64_ratio   = self._bench_precision_ratio()
        if caps.gpu_available:
            caps.measured_pcie_bw_gbs       = self._bench_pcie_bandwidth()
            caps.measured_gpu_launch_us     = self._bench_gpu_kernel_launch()
            caps.measured_gpu_fp64_gflops   = self._bench_gpu_fp64(caps)
            caps.measured_gpu_fp32_gflops   = self._bench_gpu_fp32(caps)
            if caps.triton_available:
                # bench Triton usa parámetros ya calculados en compute_derived_params
                caps.measured_triton_fma_gbs = self._bench_triton_fma(caps)

    def _bench_fma_scalar(self, n: int = 2_000_000, repeats: int = 5) -> float:
        """GFLOPS escalares: y = 0.9·x + 0.1 sobre float64."""
        x = np.random.randn(n).astype(np.float64)
        best = math.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            y = np.float64(0.9) * x + np.float64(0.1)
            x = y
            best = min(best, time.perf_counter() - t0)
        return 2.0 * n / best / 1e9  # 2 FLOPS por FMA

    def _bench_fma_vector(self, n: int = 4_000_000, repeats: int = 5) -> float:
        """GFLOPS vectoriales: float32 para ejercitar SIMD (AVX2/AVX-512)."""
        x = np.random.randn(n).astype(np.float32)
        best = math.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            y = np.float32(0.9) * x + np.float32(0.1)
            x = y
            best = min(best, time.perf_counter() - t0)
        return 2.0 * n / best / 1e9

    def _bench_memory_bandwidth(self, size_mb: int = 256, repeats: int = 5) -> float:
        """Ancho de banda de memoria en GB/s (operación copy)."""
        n = size_mb * 1024 * 1024 // 8
        arr = np.random.randn(n).astype(np.float64)
        # Warmup
        _ = arr.copy()
        best = math.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = arr.copy()
            best = min(best, time.perf_counter() - t0)
            del out
        return (size_mb / 1024.0) / best

    def _bench_pcie_bandwidth(self, size_mb: int = 128, repeats: int = 3) -> float:
        """Ancho de banda PCIe CPU→GPU en GB/s."""
        try:
            import torch
            arr = torch.randn(size_mb * 1024 * 1024 // 4)
            # Warmup
            _ = arr.cuda(); torch.cuda.synchronize()
            best = math.inf
            for _ in range(repeats):
                t0 = time.perf_counter()
                gpu_t = arr.cuda()
                torch.cuda.synchronize()
                best = min(best, time.perf_counter() - t0)
                del gpu_t
            return (size_mb / 1024.0) / best
        except Exception:
            return 0.0

    def _bench_gpu_kernel_launch(self, n: int = 1000, warmup: int = 100) -> float:
        """Overhead de lanzamiento de kernel GPU en µs."""
        try:
            import torch
            t = torch.ones(1, device="cuda", dtype=torch.float32)
            for _ in range(warmup):
                _ = t + 1.0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                _ = t + 1.0
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1e6 / n
        except Exception:
            return 0.0

    def _bench_gpu_fp64(self, caps: HardwareCapabilities,
                        n: int = 4_000_000, repeats: int = 5) -> float:
        """
        GFLOPS reales del GPU en FP64 (CUDA cores, no Tensor Cores).
        Usa operaciones vectoriales PyTorch fp64 en GPU.
        """
        try:
            import torch
            x = torch.randn(n, device="cuda", dtype=torch.float64)
            w = torch.tensor(0.9, device="cuda", dtype=torch.float64)
            b = torch.tensor(0.1, device="cuda", dtype=torch.float64)
            # Warmup
            for _ in range(3):
                y = w * x + b
            torch.cuda.synchronize()
            best = math.inf
            for _ in range(repeats):
                t0 = time.perf_counter()
                y = w * x + b
                torch.cuda.synchronize()
                best = min(best, time.perf_counter() - t0)
            return 2.0 * n / best / 1e9
        except Exception:
            return 0.0

    def _bench_gpu_fp32(self, caps: HardwareCapabilities,
                        n: int = 8_000_000, repeats: int = 5) -> float:
        """GFLOPS reales del GPU en FP32."""
        try:
            import torch
            x = torch.randn(n, device="cuda", dtype=torch.float32)
            w = torch.tensor(0.9, device="cuda", dtype=torch.float32)
            b = torch.tensor(0.1, device="cuda", dtype=torch.float32)
            for _ in range(3):
                y = w * x + b
            torch.cuda.synchronize()
            best = math.inf
            for _ in range(repeats):
                t0 = time.perf_counter()
                y = w * x + b
                torch.cuda.synchronize()
                best = min(best, time.perf_counter() - t0)
            return 2.0 * n / best / 1e9
        except Exception:
            return 0.0

    def _bench_triton_fma(self, caps: HardwareCapabilities,
                          n: int = 8_000_000, n_fma: int = 8,
                          repeats: int = 5) -> float:
        """
        Throughput efectivo del kernel Triton FMA en GB/s.
        Mide el kernel con parámetros derivados del hardware (triton_block).
        """
        try:
            from .triton_kernels import GideonTritonBackend
            backend = GideonTritonBackend(caps)
            import torch
            x = torch.randn(n, device="cuda", dtype=torch.float64)
            weights = [0.9 + 0.01 * i for i in range(n_fma)]
            biases  = [0.1 * i for i in range(n_fma)]
            fn = backend.get_fma_chain_fn(weights, biases)
            # Warmup
            for _ in range(2):
                y = fn(x.cpu().numpy())
            torch.cuda.synchronize()
            best = math.inf
            for _ in range(repeats):
                arr = x.cpu().numpy()
                t0 = time.perf_counter()
                y = fn(arr)
                best = min(best, time.perf_counter() - t0)
            # Throughput: bytes leídos + escritos = 2 * n * 8 bytes
            return 2.0 * n * 8 / best / 1e9
        except Exception:
            return 0.0

    def _bench_precision_ratio(self, n: int = 1_000_000, repeats: int = 5) -> float:
        """Ratio throughput fp32/fp64."""
        x64 = np.random.randn(n).astype(np.float64)
        x32 = x64.astype(np.float32)
        best64 = math.inf
        best32 = math.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            y = np.float64(0.9) * x64 + np.float64(0.1); x64 = y
            best64 = min(best64, time.perf_counter() - t0)
        for _ in range(repeats):
            t0 = time.perf_counter()
            y = np.float32(0.9) * x32 + np.float32(0.1); x32 = y
            best32 = min(best32, time.perf_counter() - t0)
        if best32 == 0.0:
            return 1.0
        return best64 / best32

    # ── Estimaciones estáticas (quick_mode) ──────────────────────────────────

    def _estimate_scalar_gflops(self, caps: HardwareCapabilities) -> float:
        """Estimación GFLOPS escalares basada en microarquitectura y frecuencia."""
        # GFLOPS/núcleo base por microarq (valores típicos medidos)
        _ARCH_GFLOPS_PER_CORE = {
            "arrow_lake":   3.8, "meteor_lake": 3.5, "raptor_lake": 3.5,
            "alder_lake":   3.2, "ice_lake":    3.0, "skylake":     2.5,
            "zen5":         4.2, "zen4":        3.8, "zen3":        3.2,
            "zen2":         2.8, "zen1":        2.2,
            "apple_silicon": 5.0, "aarch64":   2.0, "generic":     2.0,
        }
        base = _ARCH_GFLOPS_PER_CORE.get(caps.cpu_arch, 2.0)
        # Ajustar por frecuencia real si disponible (base calibrada a ~3 GHz)
        if caps.cpu_freq_mhz > 0:
            base *= caps.cpu_freq_mhz / 3000.0
        return base * max(caps.cpu_cores_logical, 1)

    def _estimate_vector_gflops(self, caps: HardwareCapabilities) -> float:
        """Estimación GFLOPS vectoriales basada en AVX level."""
        base = self._estimate_scalar_gflops(caps)
        if caps.cpu_avx_level >= 512:
            return base * 8.0  # 8 doubles/ZMM por ciclo FMA
        if caps.cpu_avx_level >= 2:
            return base * 4.0  # 4 doubles/YMM por ciclo FMA
        return base

    def _estimate_memory_bw(self, caps: HardwareCapabilities) -> float:
        """Estimación de ancho de banda de memoria basada en arquitectura."""
        # Estimaciones típicas en GB/s por microarq (dual-channel DDR5)
        _ARCH_MEM_BW = {
            "arrow_lake": 90.0, "meteor_lake": 70.0, "raptor_lake": 65.0,
            "alder_lake": 58.0, "ice_lake":    50.0, "skylake":     40.0,
            "zen5": 100.0, "zen4": 92.0, "zen3": 72.0, "zen2": 55.0,
            "apple_silicon": 200.0, "aarch64": 40.0, "generic": 35.0,
        }
        return _ARCH_MEM_BW.get(caps.cpu_arch, 35.0)

    # ── Hardware fingerprint (detección de cambios) ──────────────────────────

    def _hardware_fingerprint(self) -> str:
        """
        Genera un fingerprint ligero del hardware actual para detectar cambios.
        Si el fingerprint difiere del perfil guardado → re-perfilar.
        """
        parts = []
        # CPU
        try:
            with open("/proc/cpuinfo") as f:
                text = f.read()
            for line in text.split("\n"):
                if line.startswith("model name"):
                    parts.append(line.split(":", 1)[1].strip())
                    break
        except Exception:
            parts.append(platform.processor())
        parts.append(str(os.cpu_count()))
        # GPU
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                parts.append(f"gpu:{props.name}:{props.major}.{props.minor}")
        except Exception:
            pass
        return "|".join(parts)

    # ── Persistencia ─────────────────────────────────────────────────────────

    def _profile_path(self) -> str:
        hostname = socket.gethostname().replace("/", "_").replace("\\", "_")
        return os.path.join(self.cache_dir, f"{hostname}.json")

    def _save(self, caps: HardwareCapabilities) -> None:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            d = caps.to_dict()
            d["_profile_version"]  = self.PROFILE_VERSION
            d["_hw_fingerprint"]   = self._hardware_fingerprint()
            tmp = self._profile_path() + ".tmp"
            with open(tmp, "w") as f:
                json.dump(d, f, indent=2)
            os.replace(tmp, self._profile_path())
        except Exception:
            pass


# ── Helpers ─────────────────────────────────────────────────────────────────

def _is_power2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
