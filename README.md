# Martínez's Invariant: Universal Reduction Theorem

This repository contains the rigorous mathematical formalization of the **Coverage Lemma** for **Martínez's Invariant**, also known as the **Universal Reduction Theorem (URT)**, implemented in Lean 4.

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

## El Mapa Completo: De Teorema a Motor

```text
PAPER (arXiv)          LIBRERÍA (PyPI)         MOTOR (torch.compile)
      │                      │                          │
      ▼                      ▼                          ▼
 Fundamento            pip install               @martinez_functor
 académico             martinez-functor          como backend nativo
```

### Ruta 1 — La Librería PyPI (`martinez-functor`)

Esta es la forma en que todo el mundo puede importar el trabajo con un solo comando:

```bash
pip install martinez-functor
```

#### El API que Verá el Mundo

```python
from martinez import MartinezFunctor

Φ = MartinezFunctor(target_dim_d=1024, precision="fp32")

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
from martinez.backends import martinez_backend

# Compilar CUALQUIER modelo con el Functor de Martínez
model = torch.compile(my_model, backend=martinez_backend)

# Internamente, torch.compile rutea cada operación
# a través de Φ automáticamente
```

Al utilizar la API oficial de backends personalizados de `torch.compile` y acceder directamente a recursos vía Triton, se eliminan los cuellos de botella mediante hardware-aware FMA en Tensor Cores.

### Ruta 3 — El Plugin JAX (Opcional pero Poderoso)

Para la comunidad científica en Google Research que utiliza JAX, el framework se registra limpiamente como una primitiva nativa:

```python
import jax
from martinez.backends import martinez_primitive

# El Functor como primitiva JAX con JIT, grad, y vmap automáticos
@jax.jit
def my_computation(x):
    return martinez_primitive(x)  # Φ aplicado automáticamente y factorizado
```
