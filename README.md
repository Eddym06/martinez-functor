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

---

## Campo de Curvatura Dinámica (CCD Engine) — Solución Geométrica a la Maldición de la Dimensionalidad

El **CCD Engine** (`acf_functor/ccd_engine.py`) es el motor del ecosistema para procesar problemas en alta dimensión, escapando de la **Maldición de la Dimensionalidad (CoD)** mediante geometría diferencial adaptativa.

### Fundamento matemático

Dado un sistema en $\mathbb{R}^d$ con dimensión intrínseca $m \ll d$, la complejidad de una grilla $\varepsilon$-densa escala como $O(\varepsilon^{-d})$. El CCD Engine detecta y explota la variedad de baja dimensión para reducir esto a $O(\varepsilon^{-k})$ con $k \approx m$. La reducción de CoD es:

$$\log_{10} \text{CoD} = (d - k) \cdot \log_{10}(1/\varepsilon)$$

Para $d=50$, $k=3$ (Lorenz), $\varepsilon=0.01$: reducción de $\mathbf{10^{94}}$ operaciones.

### Arquitectura de 5 capas

| Capa | Clase | Rol |
|------|-------|-----|
| 1 — Expansión espectral | `ChebyshevShell` | Expansión en $T_0 \ldots T_{m-1}$, compresión SVD al rango efectivo |
| 2 — Geometría difusiva | `DiffusionGeometry` | Mapas de difusión adaptativos (Coifman & Lafon 2006), coordenadas intrínsecas |
| 3 — Modos normales | `CoupledOscillators` | Eigendescomposición de covarianza, grupos de resonancia y coherencia adaptativa |
| 4 — Entropía local | `LocalEntropyOperator` | $H_\text{local}(x) = \text{std}(\log d_1,\ldots,\log d_k)$, temperatura adaptativa $T(x)$ |
| 5 — Purificador Langevin | `LangevinPurifier` | $dx = -\nabla U(x)\,dt + \sqrt{2T(x)}\,dW_t$, Euler-Maruyama con clipping de gradiente |

### Uso rápido

```python
from acf_functor import CCDEngine, preprocess_high_dim, estimate_intrinsic_dimension

# 1. Dimensión intrínseca
result = estimate_intrinsic_dimension(X_high_dim)
print(f"Ambient d={result['d_ambient']}, intrinsic ≈ {result['spectral_gap_dim']}")

# 2. Pipeline completo (fit + transform)
Z, engine = preprocess_high_dim(X_train, d_threshold=5)
# Z: coordenadas difusivas (n, k_eff) con k_eff << d

# 3. Motor completo
engine = CCDEngine(d_threshold=5, n_diffusion_components=10).fit(X_train)
Z_diff = engine.transform(X_train)          # coordenadas difusivas
Z_res  = engine.transform_resonance(X)     # modos normales
X_pure = engine.langevin_purify(X_noisy)   # denoising geométrico

# 4. Certificado formal
cert = engine.certificate()
print(cert)
# CCDCertificate(d_input=50, k_effective=3, curse_escaped=True,
#                cod_reduction_log10=94.0, ...)
```

### Integración en el ecosistema

- **`AutopoieticScientist._observe()`**: cuando la trayectoria tiene $d \geq$ `ccd_d_threshold`, aplica automáticamente `CCDEngine.transform()` antes de la estimación Koopman, reduciendo la dimensión al espacio difusivo donde el aprendizaje de operador es tractable.
- **Versión paquete**: `acf_functor.__version__ == "5.1.0"`
- **Tests**: `tests/test_ccd_engine.py` — 74 tests, 9 clases, **74/74 pasando**.

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
