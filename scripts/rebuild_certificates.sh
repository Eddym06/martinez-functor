#!/bin/bash
# rebuild_certificates.sh - Atomic pipeline for Lean certificate generation and validation
# 
# This script ensures that Lean certificates and Python runtime artifacts are
# always synchronized. It fails if any step fails, preventing desynchronization.
#
# Usage: ./scripts/rebuild_certificates.sh
#
# Steps:
#   1. Generate Lean certificates from Python (generate_interval_certificates.py)
#   2. Build Lean project (lake build)
#   3. Extract Python artifacts from Lean (lean --run)
#   4. Run synchronization tests
#   5. Run full test suite

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

echo "=============================================="
echo "  Poema Certificate Rebuild Pipeline"
echo "=============================================="
echo ""

# Step 1: Generate Lean certificates
echo "[1/5] Generating Lean certificates..."
python3 python_analysis/generate_interval_certificates.py
echo "  ✓ Certificates generated"
echo ""

# Step 2: Build Lean project
echo "[2/5] Building Lean project..."
./lean-4.29.0-rc6-linux/bin/lake build
echo "  ✓ Lean build successful"
echo ""

# Step 3: Extract Python artifacts from Lean
echo "[3/5] Extracting Python artifacts from Lean..."
./lean-4.29.0-rc6-linux/bin/lake env ./lean-4.29.0-rc6-linux/bin/lean --run MathTest/TranscendentalApprox.lean
echo "  ✓ Python artifacts extracted"
echo ""

# Step 4: Run synchronization tests
echo "[4/5] Running synchronization tests..."
export PYTHONPATH="$ROOT_DIR"
python3 -m pytest tests/test_poema_missing_coverage.py::TestTranscendentalCertification -v
echo "  ✓ Synchronization tests passed"
echo ""

# Step 5: Run full test suite (quick check)
echo "[5/5] Running full test suite..."
python3 -m pytest tests/ --ignore=tests/test_koopman_validation.py -q
echo "  ✓ Full test suite passed"
echo ""

echo "=============================================="
echo "  ✓ All steps completed successfully"
echo "  Certificates are synchronized and validated"
echo "=============================================="
