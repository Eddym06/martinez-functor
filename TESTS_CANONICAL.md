# TESTS CANONICAL — Poema / Functor de Colapso Afín (ACF)

**Última verificación:** 2026-04-06
**Total de tests:** 356
**Regresiones:** 0

**Validación teórica reciente (resultados reales):**
- Certificados Lean rebuild: sin/cos/exp/log/tanh/sigmoid sincronizados, errores < 4e-3.
- Canonical benchmark: poly ε=0, sin error 9.99e-16, Triton speedup.
- FMA benchmark: error 1.39e-17, speedup hasta 6.87x vs NumPy en batch grande.
- Conservation test: E(f)=E(Φ(f))=7, invariant True, error 5.96e-08.

**Nuevos tests v2.2.0 (35 tests):**
- GEMM-Triton Collider: 10 tests (colapso de cadenas, condition number, fp64 promotion, memory tiling)
- Kahan Horner Kernel: 3 tests (correctitud grado 10, estabilidad, fallback PyTorch)
- Auto-Domain Repair: 8 tests (certificados expandidos sin/cos/exp/tanh/sigmoid, evaluadores reparados, caché)
- CoPoem Multiobjetivo: 10 tests (compatibilidad, incompatibilidad ortogonal/Lyapunov/Frobenius, Anderson, estancamiento)
- Integración: 3 tests (GEMM+Kahan, AutoRepair+CoPoem, pipeline completo)

## Ejecución canónica

```bash
PYTHONPATH=. python3 -m pytest tests -q
```

## Tests excluidos

Actualmente no hay exclusiones en la corrida canónica `tests -q`.

## Distribución por módulo

| Módulo | Tests |
|--------|-------|
| `test_composition_exhaustive.py` | Composición de polinomios |
| `test_evolutions*.py` | Evoluciones del functor |
| `test_functor_engine.py` | Motor del functor |
| `test_genesis.py` | Genesis certificates |
| `test_koopman_validation.py` | Validación Koopman |
| `test_poema.py` | Core Poema |
| `test_poema_hardening.py` | Hardening tests |
| `test_poema_missing_coverage.py` | Cobertura extendida (+42 tests) |
| `test_alpha_consistency_report.py` | Evidencia de consistencia empírica de alpha |
| `test_traceability_matrix_report.py` | Matriz ejecutable de trazabilidad claim -> evidencia |
| `test_self_modulation.py` | Auto-modulación |

## Nota

Este archivo es la **fuente canónica** para el número de tests.
Cualquier discrepancia con README.md, Poema.md, o Poema-manual.md debe corregirse para coincidir con este archivo.
