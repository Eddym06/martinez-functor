#!/usr/bin/env python3
"""Analizar resultados parciales del experimento"""
import json
import numpy as np
from datetime import datetime

# Resultados parciales (primeros 25 tests)
results = [
    # Swiss Roll
    {'scenario': 'swiss_roll', 'seed': 42, 'trust_pca': 0.8566, 'trust_ccd': 0.8575, 'improvement': 0.0008, 'alpha_learned': 0.200},
    {'scenario': 'swiss_roll', 'seed': 123, 'trust_pca': 0.8662, 'trust_ccd': 0.8658, 'improvement': -0.0005, 'alpha_learned': 0.450},
    {'scenario': 'swiss_roll', 'seed': 456, 'trust_pca': 0.8722, 'trust_ccd': 0.8713, 'improvement': -0.0008, 'alpha_learned': 0.300},
    {'scenario': 'swiss_roll', 'seed': 789, 'trust_pca': 0.8599, 'trust_ccd': 0.8585, 'improvement': -0.0014, 'alpha_learned': 0.350},
    {'scenario': 'swiss_roll', 'seed': 999, 'trust_pca': 0.8708, 'trust_ccd': 0.8706, 'improvement': -0.0001, 'alpha_learned': 0.500},
    
    # S-Curve
    {'scenario': 's_curve', 'seed': 42, 'trust_pca': 0.8601, 'trust_ccd': 0.8602, 'improvement': 0.0001, 'alpha_learned': 0.500},
    {'scenario': 's_curve', 'seed': 123, 'trust_pca': 0.8747, 'trust_ccd': 0.8742, 'improvement': -0.0006, 'alpha_learned': 0.500},
    {'scenario': 's_curve', 'seed': 456, 'trust_pca': 0.8548, 'trust_ccd': 0.8595, 'improvement': 0.0046, 'alpha_learned': 0.100},
    {'scenario': 's_curve', 'seed': 789, 'trust_pca': 0.8683, 'trust_ccd': 0.8681, 'improvement': -0.0002, 'alpha_learned': 0.450},
    {'scenario': 's_curve', 'seed': 999, 'trust_pca': 0.8619, 'trust_ccd': 0.8652, 'improvement': 0.0034, 'alpha_learned': 0.100},
    
    # High-dim Noise
    {'scenario': 'high_dim_noise', 'seed': 42, 'trust_pca': 0.5766, 'trust_ccd': 0.5770, 'improvement': 0.0004, 'alpha_learned': 0.513},
    {'scenario': 'high_dim_noise', 'seed': 123, 'trust_pca': 0.5738, 'trust_ccd': 0.5736, 'improvement': -0.0002, 'alpha_learned': 0.306},
    {'scenario': 'high_dim_noise', 'seed': 456, 'trust_pca': 0.5820, 'trust_ccd': 0.5806, 'improvement': -0.0013, 'alpha_learned': 0.169},
    {'scenario': 'high_dim_noise', 'seed': 789, 'trust_pca': 0.5755, 'trust_ccd': 0.5745, 'improvement': -0.0010, 'alpha_learned': 0.169},
    {'scenario': 'high_dim_noise', 'seed': 999, 'trust_pca': 0.5755, 'trust_ccd': 0.5733, 'improvement': -0.0022, 'alpha_learned': 0.100},
    
    # Outliers
    {'scenario': 'outliers', 'seed': 42, 'trust_pca': 0.5949, 'trust_ccd': 0.5941, 'improvement': -0.0008, 'alpha_learned': 0.300},
    {'scenario': 'outliers', 'seed': 123, 'trust_pca': 0.5969, 'trust_ccd': 0.5962, 'improvement': -0.0007, 'alpha_learned': 0.300},
    {'scenario': 'outliers', 'seed': 456, 'trust_pca': 0.5894, 'trust_ccd': 0.5902, 'improvement': 0.0008, 'alpha_learned': 0.300},
    {'scenario': 'outliers', 'seed': 789, 'trust_pca': 0.5934, 'trust_ccd': 0.5909, 'improvement': -0.0025, 'alpha_learned': 0.781},
    {'scenario': 'outliers', 'seed': 999, 'trust_pca': 0.5987, 'trust_ccd': 0.5993, 'improvement': 0.0006, 'alpha_learned': 0.644},
    
    # MNIST-like
    {'scenario': 'mnist_like', 'seed': 42, 'trust_pca': 0.7073, 'trust_ccd': 0.7073, 'improvement': -0.0000, 'alpha_learned': 0.100},
    {'scenario': 'mnist_like', 'seed': 123, 'trust_pca': 0.7134, 'trust_ccd': 0.7143, 'improvement': 0.0009, 'alpha_learned': 0.169},
]

print("=" * 80)
print("ANÁLISIS PARCIAL (25 tests)")
print("=" * 80)

improvements = [r['improvement'] for r in results]
alphas = [r['alpha_learned'] for r in results]

print(f"Tests analizados: {len(results)}")
print(f"Mejora promedio: {np.mean(improvements):.6f}")
print(f"Mejora std: {np.std(improvements):.6f}")
print(f"Mejora mínima: {np.min(improvements):.6f}")
print(f"Mejora máxima: {np.max(improvements):.6f}")
print(f"Alpha promedio: {np.mean(alphas):.3f}")

# Contar victorias/empates/derrotas
wins = sum(1 for r in results if r['improvement'] > 0.001)
ties = sum(1 for r in results if abs(r['improvement']) <= 0.001)
losses = sum(1 for r in results if r['improvement'] < -0.001)

print(f"\nResultados:")
print(f"  Victorias CCD (Δ > 0.001): {wins}")
print(f"  Empates (|Δ| ≤ 0.001): {ties}")
print(f"  Derrotas CCD (Δ < -0.001): {losses}")

# Análisis por escenario
print(f"\nAnálisis por escenario:")
scenarios = set(r['scenario'] for r in results)
for scenario in sorted(scenarios):
    scenario_results = [r for r in results if r['scenario'] == scenario]
    scenario_improvements = [r['improvement'] for r in scenario_results]
    scenario_alphas = [r['alpha_learned'] for r in scenario_results]
    
    print(f"\n  {scenario}:")
    print(f"    Tests: {len(scenario_results)}")
    print(f"    Mejora promedio: {np.mean(scenario_improvements):.6f}")
    print(f"    Alpha promedio: {np.mean(scenario_alphas):.3f}")
    print(f"    Victorias: {sum(1 for r in scenario_results if r['improvement'] > 0.001)}")
    print(f"    Empates: {sum(1 for r in scenario_results if abs(r['improvement']) <= 0.001)}")
    print(f"    Derrotas: {sum(1 for r in scenario_results if r['improvement'] < -0.001)}")

print(f"\n" + "=" * 80)
print("CONCLUSIÓN PARCIAL")
print("=" * 80)

if losses == 0:
    print("✅ CCD CON SKIP-CONNECTION NUNCA PIERDE SIGNIFICATIVAMENTE VS PCA")
    print(f"   ({wins} mejoras, {ties} empates, {losses} pérdidas)")
else:
    print(f"⚠️  CCD tiene {losses} pérdidas significativas vs PCA")
    
# Análisis de α aprendido
print(f"\nAnálisis de α aprendido:")
print(f"  α < 0.3 (predominio no-lineal): {sum(1 for a in alphas if a < 0.3)} tests")
print(f"  0.3 ≤ α ≤ 0.7 (mezcla balanceada): {sum(1 for a in alphas if 0.3 <= a <= 0.7)} tests")
print(f"  α > 0.7 (predominio lineal): {sum(1 for a in alphas if a > 0.7)} tests")

# Guardar análisis
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"analisis_parcial_{timestamp}.json"

with open(output_file, 'w') as f:
    json.dump({
        'metadata': {
            'total_tests': len(results),
            'timestamp': datetime.now().isoformat(),
            'scenarios': list(scenarios)
        },
        'results': results,
        'statistical_summary': {
            'mean_improvement': float(np.mean(improvements)),
            'std_improvement': float(np.std(improvements)),
            'min_improvement': float(np.min(improvements)),
            'max_improvement': float(np.max(improvements)),
            'mean_alpha': float(np.mean(alphas)),
            'wins': wins,
            'ties': ties,
            'losses': losses
        }
    }, f, indent=2)

print(f"\nAnálisis guardado en: {output_file}")