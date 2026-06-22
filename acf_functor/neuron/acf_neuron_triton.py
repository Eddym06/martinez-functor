"""
acf_neuron_triton.py — Kernel GPU Nativo para Neurona ACF
==========================================================

Compila la cadena FMA de una neurona ACF a un kernel Triton que
ejecuta en Tensor Cores. Una sola neurona, un solo kernel launch,
cientos de miles de evaluaciones en paralelo.

Requisito: triton (pip install triton)
Si triton no está disponible, usa fallback NumPy.

Autor: AXIOM-1
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def compile_neuron_to_triton(weights: np.ndarray, biases: np.ndarray):
    """
    Compilar una neurona ACF a kernel Triton.

    Args:
        weights: array[k] de pesos de la cadena FMA
        biases: array[k] de biases de la cadena FMA

    Returns:
        Función kernel que acepta (x_ptr, y_ptr, n_elements) y ejecuta en GPU.
    """
    if not HAS_TRITON:
        return _make_fallback_kernel(weights, biases)

    k = len(weights)
    w_torch = triton.testing.numpy_to_torch(weights.astype(np.float32), device='cuda')
    b_torch = triton.testing.numpy_to_torch(biases.astype(np.float32), device='cuda')

    # Constantes de compilación
    @triton.jit
    def _neuron_kernel(
        x_ptr, y_ptr,
        w_ptr, b_ptr,
        n_elements,
        K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr = 1024,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Cargar entrada
        acc = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        # Evaluar cadena FMA
        for i in range(K):
            w_i = tl.load(w_ptr + i)
            b_i = tl.load(b_ptr + i)
            acc = w_i * acc + b_i

        # Guardar salida
        tl.store(y_ptr + offsets, acc, mask=mask)

    # Crear wrapper
    def kernel_fn(x: np.ndarray) -> np.ndarray:
        x_f32 = x.astype(np.float32).ravel()
        n = len(x_f32)
        y = np.zeros(n, dtype=np.float32)

        x_t = triton.testing.numpy_to_torch(x_f32, device='cuda')
        y_t = triton.testing.numpy_to_torch(y, device='cuda')

        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        _neuron_kernel[grid](x_t, y_t, w_torch, b_torch, n, K=k)

        return triton.testing.torch_to_numpy(y_t).reshape(x.shape)

    kernel_fn._triton_kernel = True
    return kernel_fn


def compile_neuron_batch_to_triton(weights: np.ndarray, biases: np.ndarray):
    """
    Compilar neurona ACF a kernel Triton para lotes 2D.

    x: [batch_size, features]
    Cada feature se evalúa con la misma cadena FMA.
    """
    if not HAS_TRITON:
        return _make_fallback_kernel(weights, biases)

    k = len(weights)
    w_torch = triton.testing.numpy_to_torch(weights.astype(np.float32), device='cuda')
    b_torch = triton.testing.numpy_to_torch(biases.astype(np.float32), device='cuda')

    @triton.jit
    def _neuron_batch_kernel(
        x_ptr, y_ptr,
        w_ptr, b_ptr,
        batch_size, features,
        K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr = 256,
    ):
        pid = tl.program_id(0)
        f_start = pid * BLOCK_SIZE
        f_offsets = f_start + tl.arange(0, BLOCK_SIZE)
        f_mask = f_offsets < features

        for b in range(batch_size):
            offs = b * features + f_offsets
            acc = tl.load(x_ptr + offs, mask=f_mask, other=0.0)

            for i in range(K):
                w_i = tl.load(w_ptr + i)
                b_i = tl.load(b_ptr + i)
                acc = w_i * acc + b_i

            tl.store(y_ptr + offs, acc, mask=f_mask)

    def kernel_fn(x: np.ndarray) -> np.ndarray:
        x_f32 = x.astype(np.float32)
        if x_f32.ndim == 1:
            x_f32 = x_f32.reshape(1, -1)
        batch, feats = x_f32.shape
        y = np.zeros_like(x_f32)

        x_t = triton.testing.numpy_to_torch(x_f32, device='cuda')
        y_t = triton.testing.numpy_to_torch(y, device='cuda')

        grid = lambda meta: (triton.cdiv(feats, meta['BLOCK_SIZE']),)
        _neuron_batch_kernel[grid](x_t, y_t, w_torch, b_torch, batch, feats, K=k)

        return triton.testing.torch_to_numpy(y_t)

    kernel_fn._triton_kernel = True
    return kernel_fn


def _make_fallback_kernel(weights: np.ndarray, biases: np.ndarray):
    """Kernel fallback en NumPy cuando Triton no está disponible."""
    w = weights.copy()
    b = biases.copy()

    def kernel_fn(x: np.ndarray) -> np.ndarray:
        result = np.asarray(x, dtype=np.float64).copy()
        for wi, bi in zip(w, b):
            result = wi * result + bi
        return result

    kernel_fn._triton_kernel = False
    return kernel_fn


def evaluate_neuron_gpu(x: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """Evaluar neurona en GPU (Triton) o CPU (fallback)."""
    kernel = compile_neuron_to_triton(weights, biases)
    return kernel(x)


def benchmark_neuron(
    weights: np.ndarray,
    biases: np.ndarray,
    n_elements: int = 1_000_000,
    n_runs: int = 100,
) -> dict:
    """
    Benchmark de rendimiento de la neurona ACF.
    Compara GPU (Triton) vs CPU (NumPy).
    """
    x = np.random.randn(n_elements).astype(np.float32)

    # CPU baseline
    import time
    cpu_kernel = _make_fallback_kernel(weights, biases)
    cpu_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        cpu_kernel(x)
        cpu_times.append(time.perf_counter() - t0)
    cpu_mean = np.mean(cpu_times)

    result = {
        "n_elements": n_elements,
        "chain_length": len(weights),
        "cpu_mean_ms": cpu_mean * 1000,
    }

    if HAS_TRITON:
        gpu_kernel = compile_neuron_to_triton(weights, biases)
        # Warmup
        for _ in range(10):
            gpu_kernel(x)
        gpu_times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            gpu_kernel(x)
            gpu_times.append(time.perf_counter() - t0)
        gpu_mean = np.mean(gpu_times)
        result["gpu_mean_ms"] = gpu_mean * 1000
        result["speedup"] = cpu_mean / gpu_mean
    else:
        result["gpu_mean_ms"] = None
        result["speedup"] = None
        result["note"] = "Triton no disponible"

    return result
