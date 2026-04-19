# Traceability Matrix

Generated on: 2026-04-06

## Summary

- Total entries: 5
- Certified: 2
- Validated: 3

## Entries

| ID | Claim | Rigor | Status |
| --- | --- | --- | --- |
| C-EXACT-POLY | Polynomial reduction/evaluation path is exact under Horner/FMA branch assumptions. | Formal + runtime | Certified |
| C-TRANSCENDENTAL-CERT | Canonical transcendental branches use constructive interval certificates synchronized to runtime. | Constructive + runtime | Certified |
| C-KOOPMAN-ADAPTIVE | Adaptive Koopman diagnostics are implemented and empirically validated on covered benchmark classes. | Empirical systems | Validated |
| C-ALPHA-CONSISTENCY | Alpha estimators are tracked with executable discrepancy reporting across canonical functions. | Empirical systems | Validated |
| C-SECTION18-CLOSURE | Section 18 engineering closure is implemented with reproducible closure tests and benchmarks. | Empirical systems | Validated |

## Detailed Evidence

### C-EXACT-POLY
- Claim: Polynomial reduction/evaluation path is exact under Horner/FMA branch assumptions.
- Rigor layer: Formal + runtime
- Status: Certified
- Implementation:
  - poema/compiler.py
  - MathTest/HornerExact.lean
  - MathTest/HornerExtract.lean
- Validation:
  - tests/test_poema.py
  - tests/test_functor_engine.py
- Artifacts:
  - artifacts/poema_hardening_metrics.json
- Reproducible commands:
  - PYTHONPATH=. python3 -m pytest tests/test_poema.py tests/test_functor_engine.py -q
  - ./lean-4.29.0-rc6-linux/bin/lake build
- Current limit: Certified on implemented branch assumptions; does not claim unrestricted global theorem for all runtimes.

### C-TRANSCENDENTAL-CERT
- Claim: Canonical transcendental branches use constructive interval certificates synchronized to runtime.
- Rigor layer: Constructive + runtime
- Status: Certified
- Implementation:
  - python_analysis/generate_interval_certificates.py
  - MathTest/TranscendentalCertificates.lean
- Validation:
  - python_analysis/test_transcendental_integration.py
  - tests/test_jit_onnx_export.py
- Artifacts:
  - python_analysis/certificates/transcendental_runtime.py
- Reproducible commands:
  - PYTHONPATH=. python3 -m pytest python_analysis/test_transcendental_integration.py -q
- Current limit: Coverage is explicit for canonical domains/functions; not a claim of universal transcendental closure.

### C-KOOPMAN-ADAPTIVE
- Claim: Adaptive Koopman diagnostics are implemented and empirically validated on covered benchmark classes.
- Rigor layer: Empirical systems
- Status: Validated
- Implementation:
  - acf_functor/koopman_adaptive.py
  - acf_functor/kolmogorov_entropy.py
- Validation:
  - tests/test_invariant_unified.py
  - tests/test_certified_koopman_extended.py
- Artifacts:
  - artifacts/cluster_bridge_metrics.json
  - artifacts/periodic_table.md
- Reproducible commands:
  - PYTHONPATH=. python3 -m pytest tests/test_invariant_unified.py tests/test_certified_koopman_extended.py -q
  - python3 benchmarks/periodic_table.py --output artifacts/periodic_table.md
- Current limit: General finite-dimensional truncation theorem for broad nonlinear families remains open.

### C-ALPHA-CONSISTENCY
- Claim: Alpha estimators are tracked with executable discrepancy reporting across canonical functions.
- Rigor layer: Empirical systems
- Status: Validated
- Implementation:
  - python_analysis/alpha_consistency_report.py
- Validation:
  - tests/test_alpha_consistency_report.py
- Artifacts:
  - artifacts/alpha_consistency_report.json
  - artifacts/alpha_consistency_report.md
- Reproducible commands:
  - PYTHONPATH=. python3 python_analysis/alpha_consistency_report.py --fast --skip-geometric --output-json artifacts/alpha_consistency_report.json --output-md artifacts/alpha_consistency_report.md
  - PYTHONPATH=. python3 -m pytest tests/test_alpha_consistency_report.py -q
- Current limit: Estimator agreement is measured; full theorem-level equivalence is explicitly open.

### C-SECTION18-CLOSURE
- Claim: Section 18 engineering closure is implemented with reproducible closure tests and benchmarks.
- Rigor layer: Empirical systems
- Status: Validated
- Implementation:
  - benchmarks/periodic_table.py
  - benchmarks/cluster_proxy.py
  - poema/frontend.py
- Validation:
  - tests/test_section18_closure.py
  - tests/test_symbiotic_convergence.py
- Artifacts:
  - artifacts/periodic_table.json
  - artifacts/cluster_proxy_metrics.json
- Reproducible commands:
  - PYTHONPATH=. python3 -m pytest tests/test_section18_closure.py -q
  - python3 benchmarks/cluster_proxy.py --steps 5 --output artifacts/cluster_proxy_metrics.json
- Current limit: Global theorem closure for all nonlinear classes is outside the Section 18 closure scope.
