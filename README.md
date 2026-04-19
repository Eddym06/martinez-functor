# Índice Afín α(f): Universal Reduction Theorem

This repository contains the rigorous mathematical formalization of the **Coverage Lemma** for **Índice Afín α(f)**, also known as the **Universal Reduction Theorem (URT)**, implemented in Lean 4.

## Project Overview

The project formalizes advanced analytical and topological concepts, integrating:
- **Bournez's Completeness** and Taylor Expansions over Continuous Functions.
- **Orthogonal Projections** in Reproducing Kernel Hilbert Spaces (RKHS).
- **Koopman Operator Linearizations** using empirical inner products.
- Exact type casting and Hilbert Space coercions via $L^p$ integrations (`WithLp.equiv 2`).

The formalization successfully compiles with **0 errors and 0 warnings (no `sorry` placeholders)** against a modern Lean 4 `mathlib` environment.

## Repository Structure

- `MathTest/`: Contains the primary formal proofs in Lean 4.
  - `urt_coverage.lean`: Master formalization file asserting the Coverage Lemma mathematically.
- `python_analysis/`: Python tools and scripts for numerical modeling and analytical exploration (e.g., FMA derivations, Koopman matrices).
- `archive/`: Legacy exploratory proof attempts and theoretical drafts.
- `Paper.md`: Detailed documentation, research narrative, and theoretical background.

## Build and Verification

To verify the proofs mathematically on your local machine, ensure you have Lean 4 and `lake` installed:

```bash
# Build the entire library (downloads mathlib cache if needed)
lake build

# Or check the main coverage lemma directly
lake env lean MathTest/urt_coverage.lean
```

## End-to-End Validation (Current Baseline)

The repository currently validates a full Lean-to-Python loop for:

- Exact polynomial branch (Horner/FMA)
- Certified transcendental branch on canonical domains (`sin`, `exp`, `log`)
- Runtime equivalence and integration tests
- Native bare-metal execution via Gideon v1.4.0 (Triton GPU + AVX-512 CPU + Rust aligned buffers)

Recommended reproducible sequence in this workspace:

```bash
# 1) Regenerate constructive interval certificates
./lean-4.29.0-rc6-linux/bin/lake env .venv/bin/python python_analysis/generate_interval_certificates.py

# 2) Rebuild Lean artifacts
./lean-4.29.0-rc6-linux/bin/lake build

# 3) Re-extract synchronized Python transcendental runtime module
./lean-4.29.0-rc6-linux/bin/lake env ./lean-4.29.0-rc6-linux/bin/lean --run MathTest/TranscendentalApprox.lean

# 4) Run Python tests
export PYTHONPATH=.
.venv/bin/pytest -q
```

Expected outcome:

- Lean build completes successfully.
- PyTest completes successfully — **current baseline: 1427 passed** (including 18 bare-metal tests).

### Quick bare-metal test

```bash
export PYTHONPATH=.
.venv/bin/pytest tests/test_gideon_baremetal.py -v
# Expected: 18 passed in ~6s
```

## Key Generated/Checked Artifacts

- Lean certificates: `MathTest/TranscendentalCertificates.lean`
- Lean extraction bridge: `MathTest/TranscendentalApprox.lean`
- Generated Python runtime: `python_analysis/transcendental_generated.py`
- Generated Horner runtime: `python_analysis/horner_generated.py`
- Extraction equivalence tests: `python_analysis/test_extraction.py`
- Transcendental integration tests: `python_analysis/test_transcendental_integration.py`

## El Mapa Completo: De Teorema a Motor

```text
PAPER (arXiv)          LIBRERÍA (PyPI)         MOTOR (torch.compile)
      │                      │                          │
      ▼                      ▼                          ▼
 Fundamento            pip install               @acf_functor
 académico             acf-functor          como backend nativo
```

### Ruta 1 — La Librería PyPI (`acf-functor`)

Esta es la forma en que todo el mundo puede importar el trabajo con un solo comando:

```bash
pip install acf-functor
```

#### El API que Verá el Mundo

```python
from acf import ACFunctor

Φ = ACFunctor(target_dim_d=1024, precision="fp32")

# Cualquier función -> GEMM automáticamente
result = Φ.reduce("5*x**3 - 2*x + 1")      # Path Horner
result = Φ.reduce("sin(exp(x))")           # Path Bournez
result = Φ.reduce("my_nonlinear_ode")      # Path Koopman

# Ejecutar directo en GPU
output = Φ.execute(input_tensor, device="cuda")
```

### Ruta 2 — El Motor Real: Backend de `torch.compile`

Esta es la integración más poderosa: el Functor como **compilador nativo de PyTorch**. Cualquier modelo existente puede beneficiarse automáticamente de esta arquitectura ruteando operaciones como una multiplicación óptima de matrices tensoriales sin pasos intermedios inestables.

```python
import torch
from acf.backends import acf_backend

# Compilar CUALQUIER modelo con el Functor de Colapso Afín (ACF)
model = torch.compile(my_model, backend=acf_backend)

# Internamente, torch.compile rutea cada operación
# a través de Φ automáticamente
```

Al utilizar la API oficial de backends personalizados de `torch.compile` y acceder directamente a recursos vía Triton, se eliminan los cuellos de botella mediante hardware-aware FMA en Tensor Cores.

### Ruta 3 — El Plugin JAX (Opcional pero Poderoso)

Para la comunidad científica en Google Research que utiliza JAX, el framework se registra limpiamente como una primitiva nativa:

```python
import jax
from acf.backends import acf_primitive

# El Functor como primitiva JAX con JIT, grad, y vmap automáticos
@jax.jit
def my_computation(x):
  return acf_primitive(x)  # Φ aplicado automáticamente y factorizado
```
