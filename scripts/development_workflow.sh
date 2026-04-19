#!/bin/bash
# Progressive development workflow for the extended ACF roadmap.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH=.

echo "=== DIA 1: Alpha Combinatorial / Unified ==="
python3 -m pytest tests/test_invariant_unified.py::TestAlphaCombinatorial -v
python3 - <<'PY'
import numpy as np
from acf_functor.invariant_unified import AlphaCombinatorial

calc = AlphaCombinatorial()
a_poly, ci_poly, _ = calc.compute(lambda x: x**3, (-1.0, 1.0))
a_sin, ci_sin, _ = calc.compute(np.sin, (-1.5, 1.5))
print(f"alpha_comb(x^3)={a_poly:.4f} ci={ci_poly}")
print(f"alpha_comb(sin)={a_sin:.4f} ci={ci_sin}")
PY

echo "=== DIA 3: Polynomial Detector ==="
python3 -m pytest tests/test_certified_koopman_extended.py::TestPolynomialDetector -v

echo "=== DIA 5: Koopman Exact Polynomial ==="
python3 -m pytest tests/test_certified_koopman_extended.py::TestKoopmanExactPolynomial -v

echo "=== DIA 7: Benchmarks ==="
python3 benchmarks/benchmark_complete.py

echo "=== DIA 10: Lean Build ==="
./lean-4.29.0-rc6-linux/bin/lake build

echo "=== DIA 14: Reporte Final ==="
python3 ci/generate_validation_report.py
python3 python_analysis/dashboard.py --output plots
