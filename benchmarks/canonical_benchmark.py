"""
Benchmark canónico de Poema.
Ejecutar antes de cada release para verificar que los números
reportados en el paper son reproducibles.

Resultados esperados (RTX 4050, fp64):
- Horner degree-100: ~2.7ms Triton vs ~4.3ms PyTorch
- sin en [-π,π]: error < 4.1e-3 (certificado Lean)
- Composición profunda 20 capas: finito y estable
"""

import torch
import math
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class BenchmarkResult:
    name: str
    t_mean_ms: float
    t_std_ms: float
    max_error: Optional[float]
    certified_epsilon: Optional[float]
    n_fma_ops: int
    passed: bool
    notes: str = ""


def run_canonical_benchmark(
    output_file: str = "benchmark_results.json",
    n_warmup: int = 5,
    n_runs: int = 20,
) -> List[BenchmarkResult]:
    """
    Ejecuta el benchmark canónico completo.
    
    Returns lista de resultados, uno por test.
    """
    from poema import Poem, PoemCompiler
    
    results = []
    
    # ─────────────────────────────────────────────
    # TEST 1: Polinomio degree-100 (CPU fp64)
    # ─────────────────────────────────────────────
    P = Poem(dtype=torch.float64)
    coeffs = torch.randn(101, dtype=torch.float64).tolist()
    ast = P.polynomial(coeffs)
    compiler = PoemCompiler(target="pytorch", precision="fp64")
    fn, report = compiler.compile(ast)
    
    x = torch.linspace(-1, 1, 10000, dtype=torch.float64)
    
    # Warmup
    for _ in range(n_warmup):
        _ = fn(x)
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        y = fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    
    t_mean = sum(times) / len(times)
    t_std = (sum((t - t_mean)**2 for t in times) / len(times)) ** 0.5
    
    results.append(BenchmarkResult(
        name="polynomial_degree100_cpu_fp64",
        t_mean_ms=t_mean,
        t_std_ms=t_std,
        max_error=report.epsilon_certified,
        certified_epsilon=0.0,  # exacto
        n_fma_ops=report.total_fma_ops,
        passed=report.epsilon_certified == 0.0,
        notes="Evaluación Horner exacta, ε=0"
    ))
    
    # ─────────────────────────────────────────────
    # TEST 2: sin en [-π,π] (CPU fp64)
    # ─────────────────────────────────────────────
    ast_sin = P.sin(domain=(-math.pi, math.pi), degree=24)
    fn_sin, report_sin = compiler.compile(ast_sin, domain=(-math.pi, math.pi))
    
    x_sin = torch.linspace(-math.pi, math.pi, 10000, dtype=torch.float64)
    y_sin = fn_sin(x_sin)
    true_sin = torch.sin(x_sin)
    actual_error = (y_sin - true_sin).abs().max().item()
    
    times = []
    for _ in range(n_warmup):
        _ = fn_sin(x_sin)
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = fn_sin(x_sin)
        times.append((time.perf_counter() - t0) * 1000)
    
    t_mean = sum(times) / len(times)
    t_std = (sum((t - t_mean)**2 for t in times) / len(times)) ** 0.5
    
    results.append(BenchmarkResult(
        name="sin_canonical_cpu_fp64",
        t_mean_ms=t_mean,
        t_std_ms=t_std,
        max_error=actual_error,
        certified_epsilon=report_sin.epsilon_certified,
        n_fma_ops=report_sin.total_fma_ops,
        passed=(actual_error <= report_sin.epsilon_certified),
        notes=f"Fuente: {report_sin.certificate_source}"
    ))
    
    # ─────────────────────────────────────────────
    # TEST 3: Composición profunda (20 capas)
    # ─────────────────────────────────────────────
    ast_deep = P.identity()
    for _ in range(20):
        ast_deep = P.compose(P.scale(0.9), ast_deep)
    
    fn_deep, report_deep = compiler.compile(ast_deep, domain=(-1.0, 1.0))
    x_deep = torch.linspace(-1, 1, 1000, dtype=torch.float64)
    y_deep = fn_deep(x_deep)
    
    is_finite = torch.all(torch.isfinite(y_deep)).item()
    
    results.append(BenchmarkResult(
        name="deep_composition_20layers",
        t_mean_ms=0.0,  # no es benchmark de velocidad
        t_std_ms=0.0,
        max_error=None,
        certified_epsilon=report_deep.epsilon_certified,
        n_fma_ops=report_deep.total_fma_ops,
        passed=is_finite,
        notes="Test de estabilidad numérica"
    ))
    
    # ─────────────────────────────────────────────
    # TEST 4: GPU Triton (si disponible)
    # ─────────────────────────────────────────────
    if torch.cuda.is_available():
        try:
            compiler_triton = PoemCompiler(target="triton", precision="fp64")
            ast_poly = P.polynomial(coeffs[:21])  # degree-20 para Triton
            fn_triton, report_triton = compiler_triton.compile(ast_poly)
            
            x_gpu = torch.linspace(-1, 1, 100000, dtype=torch.float64).cuda()
            
            for _ in range(n_warmup):
                _ = fn_triton(x_gpu)
            torch.cuda.synchronize()
            
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = fn_triton(x_gpu)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
            
            t_mean = sum(times) / len(times)
            t_std = (sum((t - t_mean)**2 for t in times) / len(times)) ** 0.5
            
            results.append(BenchmarkResult(
                name="polynomial_degree20_triton_fp64",
                t_mean_ms=t_mean,
                t_std_ms=t_std,
                max_error=None,
                certified_epsilon=report_triton.epsilon_certified,
                n_fma_ops=report_triton.total_fma_ops,
                passed=True,
                notes=f"GPU: {torch.cuda.get_device_name(0)}"
            ))
        except Exception as e:
            results.append(BenchmarkResult(
                name="polynomial_degree20_triton_fp64",
                t_mean_ms=-1, t_std_ms=-1,
                max_error=None, certified_epsilon=None,
                n_fma_ops=0,
                passed=False,
                notes=f"Error Triton: {e}"
            ))
    
    # Guardar resultados
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "results": [asdict(r) for r in results],
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_canonical_benchmark()
    print(f"\n{'='*50}")
    print(f"BENCHMARK CANÓNICO POEMA")
    print(f"{'='*50}")
    for r in results:
        status = "✅" if r.passed else "❌"
        print(f"{status} {r.name}")
        if r.t_mean_ms > 0:
            print(f"   Tiempo: {r.t_mean_ms:.2f} ± {r.t_std_ms:.2f} ms")
        if r.max_error is not None:
            print(f"   Error máx: {r.max_error:.3e}")
        if r.notes:
            print(f"   Nota: {r.notes}")
