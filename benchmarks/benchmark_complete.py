"""Complete benchmark suite for the extended ACF roadmap."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from acf_functor.certified_koopman import CertifiedKoopman, KoopmanExactPolynomial
from acf_functor.invariant_unified import AlphaCombinatorial


class ACFBenchmarkSuite:
    def __init__(self, output_dir: str = "artifacts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    @staticmethod
    def time_function(func, *args, n_runs: int = 200, warmup: int = 20) -> Dict[str, float]:
        for _ in range(warmup):
            func(*args)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter_ns()
            func(*args)
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1e6)
        mean_ms = statistics.mean(times)
        return {
            "mean_ms": mean_ms,
            "std_ms": statistics.pstdev(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "median_ms": statistics.median(times),
            "throughput_per_sec": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        }

    def benchmark_horner_vs_naive(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for degree in [5, 20, 50, 100]:
            coeffs = np.random.randn(degree + 1)
            x = 0.7

            def horner_eval():
                v = coeffs[0]
                for c in coeffs[1:]:
                    v = v * x + c
                return v

            def naive_eval():
                return sum(c * x**i for i, c in enumerate(reversed(coeffs)))

            s_h = self.time_function(horner_eval)
            s_n = self.time_function(naive_eval)
            out[str(degree)] = {
                "horner": s_h,
                "naive": s_n,
                "speedup_vs_naive": s_n["mean_ms"] / max(s_h["mean_ms"], 1e-12),
            }
        return out

    def benchmark_alpha_estimation(self) -> Dict[str, Any]:
        calc = AlphaCombinatorial(epsilon_range=[10 ** (-k) for k in range(1, 7)])
        alpha_poly, ci_poly, _ = calc.compute(lambda x: x**3 + x + 1)
        alpha_sin, ci_sin, _ = calc.compute(np.sin, domain=(-np.pi / 2, np.pi / 2))
        return {
            "poly": {"alpha": alpha_poly, "ci": ci_poly},
            "sin": {"alpha": alpha_sin, "ci": ci_sin},
        }

    def benchmark_koopman_branches(self) -> Dict[str, Any]:
        r = 3.5
        g = lambda x: r * x * (1 - x)
        x0 = 0.3
        n_steps = 5

        x_true = x0
        for _ in range(n_steps):
            x_true = g(x_true)

        exact = KoopmanExactPolynomial().predict_trajectory(g, x0, n_steps, degree=2, domain=(0.0, 1.0))
        cert = CertifiedKoopman().predict(g, x0, n_steps, domain=(0.0, 1.0), target_error=1e-3)

        return {
            "exact_polynomial": {
                "predicted": exact.predicted_value,
                "empirical_error": abs(exact.predicted_value - x_true),
                "certified_error": exact.error_bound,
                "branch": exact.branch.value,
            },
            "certified": {
                "predicted": cert.predicted_value,
                "empirical_error": abs(cert.predicted_value - x_true),
                "certified_error": cert.error_bound,
                "branch": cert.branch.value,
            },
        }

    def run_all_and_save(self) -> Path:
        payload = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "benchmarks": {
                "horner_vs_naive": self.benchmark_horner_vs_naive(),
                "alpha_estimation": self.benchmark_alpha_estimation(),
                "koopman_branches": self.benchmark_koopman_branches(),
            },
        }
        out = self.output_dir / f"benchmark_complete_{payload['timestamp']}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out


if __name__ == "__main__":
    suite = ACFBenchmarkSuite()
    path = suite.run_all_and_save()
    print(f"Saved benchmark report to: {path}")
