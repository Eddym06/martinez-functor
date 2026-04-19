#!/bin/bash
# Setup and baseline verification for the extended ACF roadmap.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "PASO 1: Verificar Python"
python3 --version

echo "PASO 2: Verificar dependencias Python"
python3 - <<'PY'
import numpy
print(f"numpy {numpy.__version__}")
try:
    import scipy
    print(f"scipy {scipy.__version__}")
except Exception:
    print("scipy no disponible (se usaran rutas sin scipy)")
import pytest
print(f"pytest {pytest.__version__}")
PY

echo "PASO 3: Verificar Lean 4"
./lean-4.29.0-rc6-linux/bin/lean --version || true

echo "PASO 4: Tests de invariante unificado"
PYTHONPATH=. python3 -m pytest tests/test_invariant_unified.py -v --tb=short -x

echo "PASO 5: Tests de Koopman certificado"
PYTHONPATH=. python3 -m pytest tests/test_certified_koopman_extended.py -v --tb=short -x

echo "PASO 6: Reporte de validacion cientifica"
PYTHONPATH=. python3 ci/generate_validation_report.py

echo "Setup completado"
