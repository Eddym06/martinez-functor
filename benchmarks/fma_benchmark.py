"""
FMA Benchmark for Poema.

Compares Φ-FMA vs NumPy vs SciPy vs PyTorch for polynomial evaluation.
Generates reproducible benchmark results in Markdown format.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from poema import Poem, PoemCompiler


def benchmark_polynomial(degree: int, n_points: int, device: str = 'cpu') -> Dict[str, Any]:
    """
    Compare polynomial evaluation: naive vs Horner vs Φ-FMA.
    
    Returns dict with timing and error metrics for each method.
    """
    np.random.seed(42)
    coeffs = np.random.randn(degree + 1) * 0.1  # Small coeffs for stability
    x_np = np.linspace(-0.5, 0.5, n_points)
    
    results = {}
    
    # Reference: NumPy polyval (Horner implementation)
    t0 = time.perf_counter()
    y_ref = np.polyval(coeffs[::-1], x_np)  # polyval expects highest degree first
    t_numpy = time.perf_counter() - t0
    
    results['numpy_polyval'] = {
        'time_ms': t_numpy * 1000,
        'error': 0.0,  # This is our reference
    }
    
    # Φ-FMA via Poema
    P = Poem(dtype=torch.float64)
    ast = P.polynomial(coeffs.tolist())
    
    if device == 'cuda' and torch.cuda.is_available():
        target = 'triton'
        x_torch = torch.from_numpy(x_np).to('cuda', dtype=torch.float64)
    else:
        target = 'pytorch'
        x_torch = torch.from_numpy(x_np).to('cpu', dtype=torch.float64)
    
    compiler = PoemCompiler(target=target, precision='fp64')
    fn, report = compiler.compile(ast)
    
    # Warmup
    _ = fn(x_torch)
    
    # Timed run
    n_runs = 100
    t0 = time.perf_counter()
    for _ in range(n_runs):
        y_poema = fn(x_torch)
    t_poema = (time.perf_counter() - t0) / n_runs
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    results['poema_fma'] = {
        'time_ms': t_poema * 1000,
        'fma_ops': report.total_fma_ops,
        'epsilon': report.total_epsilon,
        'error': float(torch.max(torch.abs(
            y_poema.cpu() - torch.from_numpy(y_ref)
        )).item()),
    }
    
    # PyTorch direct (using torch.pow)
    x_torch_cpu = torch.from_numpy(x_np).to('cpu', dtype=torch.float64)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        y_torch = sum(c * x_torch_cpu**i for i, c in enumerate(coeffs))
    t_torch = (time.perf_counter() - t0) / n_runs
    
    results['pytorch_direct'] = {
        'time_ms': t_torch * 1000,
        'error': float(torch.max(torch.abs(
            y_torch - torch.from_numpy(y_ref)
        )).item()),
    }
    
    return results


def run_benchmark_suite() -> Dict[str, Any]:
    """Run full benchmark suite across degrees and batch sizes."""
    degrees = [10, 50, 100]
    batch_sizes = [1000, 10000, 100000]
    
    all_results = {}
    
    for degree in degrees:
        all_results[f'degree_{degree}'] = {}
        for n_points in batch_sizes:
            key = f'n={n_points}'
            all_results[f'degree_{degree}'][key] = benchmark_polynomial(
                degree, n_points, device='cpu'
            )
    
    return all_results


def generate_markdown_report(results: Dict[str, Any]) -> str:
    """Generate Markdown benchmark report."""
    lines = [
        "# Poema FMA Benchmark Results",
        "",
        "## Configuration",
        f"- Hardware: CPU (benchmark)",
        f"- PyTorch: {torch.__version__}",
        f"- NumPy: {np.__version__}",
        "",
        "## Polynomial Evaluation (Φ-FMA vs NumPy vs PyTorch)",
        "",
    ]
    
    for degree_key, batch_results in results.items():
        degree = degree_key.replace('degree_', '')
        lines.append(f"### Degree {degree}")
        lines.append("")
        lines.append("| Batch Size | NumPy (ms) | Poema Φ-FMA (ms) | PyTorch Direct (ms) | Poema Error | Speedup vs NumPy |")
        lines.append("|------------|------------|------------------|---------------------|-------------|------------------|")
        
        for batch_key, metrics in batch_results.items():
            n = batch_key.replace('n=', '')
            numpy_time = metrics['numpy_polyval']['time_ms']
            poema_time = metrics['poema_fma']['time_ms']
            torch_time = metrics['pytorch_direct']['time_ms']
            error = metrics['poema_fma']['error']
            speedup = numpy_time / poema_time if poema_time > 0 else float('inf')
            
            lines.append(
                f"| {n} | {numpy_time:.3f} | {poema_time:.3f} | {torch_time:.3f} | "
                f"{error:.2e} | {speedup:.2f}x |"
            )
        lines.append("")
    
    lines.append("## Notes")
    lines.append("- Error is measured against NumPy polyval (reference implementation)")
    lines.append("- Speedup > 1.0 means Poema is faster than NumPy")
    lines.append("- Φ-FMA uses Horner evaluation with certified error bounds")
    lines.append("")
    
    return '\n'.join(lines)


def main():
    """Run benchmarks and generate report."""
    print("Running Poema FMA Benchmark Suite...")
    print("=" * 60)
    
    results = run_benchmark_suite()
    report = generate_markdown_report(results)
    
    # Save report
    report_path = Path("benchmarks/RESULTS.md")
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    
    print(report)
    print(f"\n✓ Report saved to {report_path}")


if __name__ == "__main__":
    main()
