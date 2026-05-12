#!/usr/bin/env python3
"""
FFT Discovery Experiment — Can the ACF System Rediscover the FFT?
==================================================================

THE QUESTION:
  Given the specification "Multiply a vector by the DFT matrix of size N
  with O(N log N) complexity and error < 1e-6", can the Autonomous
  Discovery Engine discover the butterfly structure of the FFT?

THE EXPERIMENT:
  For N = 8, 16, 32, 64, 128, 256, 512, 1024:
    1. Construct F_N (DFT matrix)
    2. Analyze structure via TAA (spectral, block, recursive)
    3. Discover butterfly factorizability
    4. Synthesize butterfly algorithm
    5. Verify correctness vs numpy.fft.fft
    6. Measure speedup vs direct O(N²)
    7. Verify O(N log N) scaling

Martínez's Invariant — April 2026
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from acf_functor.autonomous_discovery import DFTStructureDiscovery

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]
TARGET_ERROR = 1e-6

# ═════════════════════════════════════════════════════════════════════════════
# Run Experiment
# ═════════════════════════════════════════════════════════════════════════════

print("╔" + "═" * 70 + "╗")
print("║  EXPERIMENTO: DESCUBRIMIENTO AUTÓNOMO DE LA FFT                   ║")
print("║  ¿Puede el ACF redescubrir el algoritmo de Cooley-Tukey?          ║")
print("╚" + "═" * 70 + "╝")
print()

all_results = {}
t_experiment_start = time.time()

for N in SIZES:
    print(f"\n{'='*70}")
    print(f"  N = {N} (log₂N = {int(np.log2(N))})")
    print(f"{'='*70}")

    t0 = time.time()
    exp = DFTStructureDiscovery(N=N, target_error=TARGET_ERROR)
    report = exp.run_full_experiment()
    elapsed = time.time() - t0

    # Extract key metrics
    phase2 = report["phases"]["2_analysis"]
    phase3 = report["phases"]["3_discovery"]
    phase5 = report["phases"]["5_verification"]
    phase7 = report["phases"]["7_benchmark"]

    print(f"\n  FASE 1 — Construcción de F_{N}")
    print(f"    Matriz DFT: {N}×{N} compleja")
    print(f"    FMA directo: {N*N:,} operaciones")

    print(f"\n  FASE 2 — Análisis Estructural (TAA)")
    print(f"    Unitaria: {'SÍ' if phase2.get('is_unitary') else 'NO'} "
          f"(error: {phase2.get('unitarity_error', 'N/A'):.2e})")
    print(f"    Decaimiento SV: {phase2.get('sv_decay_rate', 'N/A')}")
    print(f"    Densidad: {phase2.get('density', 0):.3f}")

    block = phase2.get("block_structure", {})
    if block.get("detected"):
        print(f"    Estructura de bloques: DETECTADA "
              f"(bloques {block.get('block_size')}×{block.get('block_size')}, "
              f"rango off-diag medio: {block.get('mean_off_diag_rank', '?'):.1f})")
    else:
        print(f"    Estructura de bloques: factorizability_score="
              f"{block.get('factorizability_score', 0):.3f}")

    recursive = phase2.get("recursive_structure", {})
    print(f"    Estructura recursiva: "
          f"{'DETECTADA' if recursive.get('has_recursive_structure') else 'parcial'} "
          f"(butterfly_score={recursive.get('butterfly_score', 0):.3f})")

    circ = phase2.get("circulant_structure", {})
    print(f"    Circulante: {'SÍ' if circ.get('is_circulant') else 'NO'}")

    print(f"\n  FASE 3 — Descubrimientos")
    print(f"    Butterfly score combinado: {phase3['butterfly_score']:.3f}")
    print(f"    Sparse factor score: {phase3['sparse_factor_score']:.3f}")
    for d in report["discoveries"]:
        print(f"    → [{d['kind']}] {d['name']}: confianza={d['confidence']:.2f}")

    print(f"\n  FASE 4 — Síntesis del Algoritmo Mariposa")
    phase4 = report["phases"]["4_synthesis"]
    print(f"    Nombre: {phase4['algorithm_name']}")
    print(f"    FMA: {phase4['n_fma']:,} (vs directo {N*N:,})")
    print(f"    Speedup teórico: {N*N / phase4['n_fma']:.1f}×")
    print(f"    Nodos del grafo: {phase4['n_graph_nodes']}")

    print(f"\n  FASE 5 — Verificación")
    print(f"    Tests: {phase5['n_tests']}")
    print(f"    Error máximo: {phase5['max_error']:.2e}")
    print(f"    Error medio: {phase5['mean_error']:.2e}")
    print(f"    Target: {phase5['target_error']:.2e}")
    print(f"    RESULTADO: {'✅ CORRECTO' if phase5['passed'] else '❌ FALLÓ'}")

    print(f"\n  FASE 7 — Benchmark")
    print(f"    Butterfly FFT: {phase7['time_butterfly_us']:.1f} μs")
    print(f"    Directo (F·x): {phase7['time_direct_us']:.1f} μs")
    print(f"    NumPy FFT:     {phase7['time_numpy_fft_us']:.1f} μs")
    print(f"    Speedup vs directo: {phase7['speedup_vs_direct']:.2f}×")
    print(f"    Ratio vs NumPy: {phase7['ratio_vs_numpy']:.2f}×")

    if "8_scale_test" in report["phases"]:
        scale = report["phases"]["8_scale_test"]
        print(f"\n  FASE 8 — Test de Escala")
        print(f"    Exponente medido: {scale['measured_scaling_exponent']:.2f} "
              f"(esperado ≈ 1.0-1.3 para N·log₂N)")
        print(f"    Sub-cuadrático: {'SÍ' if scale['is_subquadratic'] else 'NO'}")
        for r in scale["scale_results"]:
            status = "✅" if r["correct"] else "❌"
            print(f"      N={r['N']:5d}: FMA_teórico={r['n_fma_theoretical']:8d}, "
                  f"t={r['time_us']:8.1f}μs, error={r['error']:.2e} {status}")

    print(f"\n  RESULTADO FINAL: {'✅ ÉXITO' if report['success'] else '❌ FALLO'}")
    print(f"  Tiempo total para N={N}: {elapsed:.2f}s")

    # Store for summary
    all_results[N] = {
        "success": report["success"],
        "butterfly_score": phase3["butterfly_score"],
        "max_error": phase5["max_error"],
        "passed_verification": phase5["passed"],
        "fma_butterfly": phase4["n_fma"],
        "fma_direct": N * N,
        "speedup": N * N / phase4["n_fma"],
        "time_butterfly_us": phase7["time_butterfly_us"],
        "time_direct_us": phase7["time_direct_us"],
        "time_numpy_us": phase7["time_numpy_fft_us"],
        "elapsed_s": elapsed,
        "certificate": report.get("certificate", {}),
    }

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

t_total = time.time() - t_experiment_start

print("\n\n" + "═" * 70)
print("  RESUMEN DEL EXPERIMENTO — DESCUBRIMIENTO AUTÓNOMO DE FFT")
print("═" * 70)

print(f"\n  {'N':>6} │ {'Éxito':>6} │ {'Score':>6} │ {'FMA Mariposa':>12} │ "
      f"{'FMA Directo':>11} │ {'Speedup':>8} │ {'Error Máx':>10} │ {'Tiempo':>7}")
print(f"  {'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*12}─┼─"
      f"{'─'*11}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*7}")

for N, r in all_results.items():
    status = "✅" if r["success"] else "❌"
    print(f"  {N:>6} │ {status:>6} │ {r['butterfly_score']:>6.3f} │ "
          f"{r['fma_butterfly']:>12,} │ {r['fma_direct']:>11,} │ "
          f"{r['speedup']:>7.1f}× │ {r['max_error']:>10.2e} │ {r['elapsed_s']:>6.1f}s")

all_passed = all(r["success"] for r in all_results.values())
all_correct = all(r["passed_verification"] for r in all_results.values())

print(f"\n  Total tests: {len(SIZES)}")
print(f"  Todos exitosos: {'SÍ' if all_passed else 'NO'}")
print(f"  Todos correctos (error < {TARGET_ERROR}): {'SÍ' if all_correct else 'NO'}")
print(f"  Tiempo total del experimento: {t_total:.1f}s")

print("\n" + "═" * 70)
if all_passed and all_correct:
    print("  ██████╗ ███████╗███████╗ ██████╗██╗   ██╗██████╗ ██████╗ ██╗███╗   ██╗███████╗███╗   ██╗████████╗ ██████╗ ")
    print("  ██╔══██╗██╔════╝██╔════╝██╔════╝██║   ██║██╔══██╗██╔══██╗██║████╗  ██║██╔════╝████╗  ██║╚══██╔══╝██╔═══██╗")
    print("  ██║  ██║█████╗  ███████╗██║     ██║   ██║██████╔╝██████╔╝██║██╔██╗ ██║█████╗  ██╔██╗ ██║   ██║   ██║   ██║")
    print("  ██║  ██║██╔══╝  ╚════██║██║     ██║   ██║██╔══██╗██╔══██╗██║██║╚██╗██║██╔══╝  ██║╚██╗██║   ██║   ██║   ██║")
    print("  ██████╔╝███████╗███████║╚██████╗╚██████╔╝██████╔╝██║  ██║██║██║ ╚████║███████╗██║ ╚████║   ██║   ╚██████╔╝")
    print("  ╚═════╝ ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ")
    print()
    print("  EL SISTEMA ACF HA REDESCUBIERTO AUTÓNOMAMENTE LA FFT.")
    print("  Uno de los algoritmos más importantes de la historia de la humanidad.")
print("═" * 70)

# Save results
results_path = ROOT / "fft_discovery_results.json"
with open(results_path, "w") as f:
    json.dump({
        "experiment": "FFT Autonomous Discovery",
        "date": "2026-04-19",
        "target_error": TARGET_ERROR,
        "sizes_tested": SIZES,
        "all_passed": all_passed,
        "all_correct": all_correct,
        "total_elapsed_s": t_total,
        "results": {str(k): v for k, v in all_results.items()},
    }, f, indent=2, default=str)
print(f"\n  Resultados guardados en: {results_path}")
